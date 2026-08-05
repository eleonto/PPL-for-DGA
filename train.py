import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import seaborn as sns
import json
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================================
# 路径统一管理（基于脚本文件位置，避免工作目录影响）
# ============================================================
# 脚本所在目录 (PLL-DGA/pll-DGA/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根目录 (PLL-DGA/) - 脚本的上级目录
ROOT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))

def _ps(*parts):
    """基于 SCRIPT_DIR (脚本目录) 构造路径 - 用于读取数据集"""
    return os.path.normpath(os.path.join(SCRIPT_DIR, *parts))

def _pr(*parts):
    """基于 ROOT_DIR (项目根) 构造路径 - 用于保存模型和输出文件"""
    return os.path.normpath(os.path.join(ROOT_DIR, *parts))

# ============================================================
# 配置参数 & 数据集文件路径（已替换为你指定路径）
# ============================================================
# WEIGHT_ORIGIN = 1.0    # 真实标签：权重最高
# WEIGHT_DPM = 0.75      # DPM 准确率 ~84%
# WEIGHT_DTM = 0.625     # DTM 准确率 ~70%
# WEIGHT_IRM = 0.5       # IRM 准确率 ~60%
WEIGHT_ORIGIN = 1    # 真实标签：权重最高
WEIGHT_DPM = 1     # DPM 准确率 ~84%
WEIGHT_DTM = 1     # DTM 准确率 ~70%
WEIGHT_IRM = 1       # IRM 准确率 ~60%

ENABLE_COUNT_BOOST = True  # 重复出现自动增加权重
W_SUP = 0.9
lam_init = 0.1
K=2

epochs = 500
batch_size = 64
patience = 50
conf_update_interval = 10
conf_smooth_alpha = 0.6

# 数据集路径（使用脚本目录下的 dataset/ 文件夹）
TRAIN_PATH = _ps("dataset", "train.csv")
VAL_PATH = _ps("dataset", "val.csv")
TEST_PATH = _ps("dataset", "test.csv")

# ============================================================
# 1. 加载三份独立数据集 + 标签预处理
# ============================================================
# 读取csv
df_train = pd.read_csv(TRAIN_PATH)
df_val = pd.read_csv(VAL_PATH)
df_test = pd.read_csv(TEST_PATH)

feature_cols = ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2']
true_label_col = 'origin'
candidate_label_cols = ['IRM', 'DTM', 'DPM']

# 标签映射（统一中文标签）
label_mapping = {
    'DT': 'Unidentified',
    'S': 'Unidentified'
}
# 批量替换三份数据标签
for df_data in [df_train, df_val, df_test]:
    for col in [true_label_col] + candidate_label_cols:
        df_data[col] = df_data[col].replace(label_mapping)

# 从全部数据提取全局标签词典（训练/验证/测试合并）
all_df = pd.concat([df_train, df_val, df_test], axis=0)
all_labels = []
for col in [true_label_col] + candidate_label_cols:
    all_labels.extend(all_df[col].tolist())
unique_labels = sorted(list(set(all_labels)))
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}
num_classes = len(unique_labels)
print(f"标签类别数量：{num_classes}")
print(f"标签映射：{label_to_idx}")

# 特征标准化：仅用训练集拟合，val/test仅转换
scaler = StandardScaler()
X_train = scaler.fit_transform(df_train[feature_cols])
X_val = scaler.transform(df_val[feature_cols])
X_test = scaler.transform(df_test[feature_cols])

print(f"训练集特征形状：{X_train.shape}")
print(f"验证集特征形状：{X_val.shape}")
print(f"测试集特征形状：{X_test.shape}")

# ============================================================
# 2. 构建候选标签矩阵函数（不变）
# ============================================================
def build_candidate_matrix(df, true_label_col, candidate_cols, 
                           weight_origin, weight_methods, 
                           enable_count_boost=True):
    """
    构建候选标签矩阵
    核心逻辑：统计每个标签的出现次数 × 来源权重
    """
    num_samples = len(df)
    y_candidate = np.zeros((num_samples, num_classes), dtype=np.float32)
    y_true = np.zeros(num_samples, dtype=np.int64)
    
    method_weights = {'origin': weight_origin}
    for method in candidate_cols:
        method_weights[method] = weight_methods.get(method, 0.5)
    
    for i, (idx, row) in enumerate(df.iterrows()):
        label_counts = {}
        label_scores = {}
        
        # 真实标签
        true_label = row[true_label_col]
        if true_label in label_to_idx:
            label_counts[true_label] = label_counts.get(true_label, 0) + 1
            label_scores[true_label] = label_scores.get(true_label, 0) + method_weights['origin']
            y_true[i] = label_to_idx[true_label]
        
        # 候选方法
        for col in candidate_cols:
            pred_label = row[col]
            if pred_label == "Unidentified" or pred_label not in label_to_idx:
                continue
            
            if enable_count_boost:
                label_counts[pred_label] = label_counts.get(pred_label, 0) + 1
            label_scores[pred_label] = label_scores.get(pred_label, 0) + method_weights[col]
        
        # 重复计数增强
        if enable_count_boost:
            for label in label_scores:
                count = label_counts.get(label, 1)
                label_scores[label] = label_scores[label] * count
        
        # 归一化
        total_score = sum(label_scores.values())
        if total_score > 0:
            for label, score in label_scores.items():
                if label in label_to_idx:
                    y_candidate[i, label_to_idx[label]] = score / total_score
        
        # 候选集为空时使用真实标签
        if total_score == 0 and true_label in label_to_idx:
            y_candidate[i, label_to_idx[true_label]] = 1.0
    
    return y_candidate, y_true

# 生成三份数据集的候选标签 & 真实标签
method_weights = {'IRM': WEIGHT_IRM, 'DTM': WEIGHT_DTM, 'DPM': WEIGHT_DPM}
y_candidate_train, y_train = build_candidate_matrix(
    df_train, true_label_col, candidate_label_cols,
    weight_origin=WEIGHT_ORIGIN,
    weight_methods=method_weights,
    enable_count_boost=ENABLE_COUNT_BOOST
)
y_candidate_val, y_val = build_candidate_matrix(
    df_val, true_label_col, candidate_label_cols,
    weight_origin=WEIGHT_ORIGIN,
    weight_methods=method_weights,
    enable_count_boost=ENABLE_COUNT_BOOST
)
y_candidate_test, y_test = build_candidate_matrix(
    df_test, true_label_col, candidate_label_cols,
    weight_origin=WEIGHT_ORIGIN,
    weight_methods=method_weights,
    enable_count_boost=ENABLE_COUNT_BOOST
)

print(f"\n候选标签矩阵形状：")
print(f"训练集: {y_candidate_train.shape}")
print(f"验证集: {y_candidate_val.shape}")
print(f"测试集: {y_candidate_test.shape}")
print(f"训练集每个样本平均候选标签数：{np.mean(np.sum(y_candidate_train > 0.01, axis=1)):.2f}")

# 保存原始索引（用于后续打印原始数据）
idx_train = np.arange(len(df_train))
idx_val = np.arange(len(df_val))
idx_test = np.arange(len(df_test))

print(f"\n数据集信息：")
print(f"  训练集: {X_train.shape[0]} 样本")
print(f"  验证集: {X_val.shape[0]} 样本")
print(f"  测试集: {X_test.shape[0]} 样本")

# ============================================================
# 3. 构建模型（完全不变）
# ============================================================
def build_model(input_dim, hidden_dim, num_classes):
    feature_input = Input(shape=(input_dim,))
    x = Dense(hidden_dim, activation='relu')(feature_input)
    x = Dropout(0.3)(x)
    x = Dense(hidden_dim, activation='relu')(x)
    x = Dropout(0.3)(x)
    y_pred = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=feature_input, outputs=y_pred)
    return model

input_dim = X_train.shape[1]
hidden_dim = 128
model = build_model(input_dim, hidden_dim, num_classes)
model.summary()

# ============================================================
# 4. 初始化置信度
# ============================================================
confidence_train = y_candidate_train.astype(np.float32)
print(f"\n初始置信度统计：")
print(f"  训练集平均候选标签数: {np.mean(np.sum(confidence_train > 0.01, axis=1)):.2f}")

# ============================================================
# 5. pll 损失函数 & 训练步（完全不变）
# ============================================================
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, weight_decay=1e-4)

def pll_loss(x, part_y, confidence):
    probs_list = []
    for k in range(K):
        probs = model(x, training=True)
        probs_list.append(probs)
    
    # L_super: 非候选标签监督学习，内部乘惩罚权重
    super_loss = -tf.reduce_mean(
        W_SUP * tf.reduce_sum(
            tf.math.log(1.0000001 - probs_list[0]) * (1.0 - part_y),
            axis=1
        )
    )

    # L_consist: 候选标签一致性正则化
    consist_loss = 0.0
    for k in range(K):
        log_probs = tf.math.log(probs_list[k] + 1e-8)
        log_conf = tf.math.log(confidence + 1e-8)
        kl_div = tf.reduce_sum(confidence * (log_conf - log_probs), axis=1)
        consist_loss += tf.reduce_mean(kl_div)
    consist_loss /= K
    
    return super_loss, consist_loss, probs_list

@tf.function
def train_step(x, part_y, confidence, lam):
    with tf.GradientTape() as tape:
        super_loss, consist_loss, probs_list = pll_loss(x, part_y, confidence)
        total_loss = lam * consist_loss + super_loss
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_loss, super_loss, consist_loss, probs_list

def confidence_update(probs_list, part_y, max_conf=0.9):
    geo_mean = tf.ones_like(probs_list[0])
    for probs in probs_list:
        geo_mean = geo_mean * tf.pow(probs, 1.0 / K)
    
    new_conf = part_y * geo_mean
    new_conf = tf.clip_by_value(new_conf, 0, max_conf)
    new_conf = new_conf / (tf.reduce_sum(new_conf, axis=1, keepdims=True) + 1e-8)
    return new_conf.numpy()

# ============================================================
# 6. 训练循环（完全不变）
# ============================================================


# 记录列表
train_losses = []
train_super_losses = []
train_consist_losses = []
val_accs = []
val_losses = []

best_val_acc = 0.0
best_model_weights = None
early_stop_counter = 0

print("\n开始训练...")
print("="*60)

for epoch in range(epochs):
    lam = min((epoch / 50.0) * lam_init, lam_init)
    
    # 训练数据集
    train_dataset = tf.data.Dataset.from_tensor_slices((
        X_train.astype(np.float32),
        y_candidate_train.astype(np.float32),
        confidence_train
    ))
    train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(batch_size)
    
    epoch_loss = 0.0
    epoch_super_loss = 0.0
    epoch_consist_loss = 0.0
    total_train = 0
    
    for x_batch, part_y_batch, conf_batch in train_dataset:
        total_loss, super_loss, consist_loss, probs_list = train_step(
            x_batch, part_y_batch, conf_batch, lam
        )
        batch_size_actual = x_batch.shape[0]
        epoch_loss += total_loss.numpy() * batch_size_actual
        epoch_super_loss += super_loss.numpy() * batch_size_actual
        epoch_consist_loss += consist_loss.numpy() * batch_size_actual
        total_train += batch_size_actual
    
    epoch_loss /= total_train
    epoch_super_loss /= total_train
    epoch_consist_loss /= total_train
    
    train_losses.append(epoch_loss)
    train_super_losses.append(epoch_super_loss)
    train_consist_losses.append(epoch_consist_loss)
    
    # 更新置信度
    if (epoch + 1) % conf_update_interval == 0:
        all_probs_list = []
        for k in range(K):
            probs = model(X_train.astype(np.float32), training=True)
            all_probs_list.append(probs)
        new_conf = confidence_update(all_probs_list, 
                                     tf.convert_to_tensor(y_candidate_train, dtype=tf.float32))
        confidence_train = conf_smooth_alpha * confidence_train + (1 - conf_smooth_alpha) * new_conf
    
    # 验证
    y_pred_val = model(X_val.astype(np.float32), training=False)
    y_pred_labels = np.argmax(y_pred_val.numpy(), axis=1)
    val_acc = accuracy_score(y_val, y_pred_labels)
    val_accs.append(val_acc)
    
    val_loss = tf.keras.losses.sparse_categorical_crossentropy(
        y_val.astype(np.int64), y_pred_val
    ).numpy().mean()
    val_losses.append(val_loss)
    
    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_weights = model.get_weights()
        early_stop_counter = 0
    else:
        early_stop_counter += 1
    
    if early_stop_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch+1}")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        break
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Total: {epoch_loss:.4f}, Super: {epoch_super_loss:.4f}, Consist: {epoch_consist_loss:.4f}")
        print(f"  Val Acc: {val_acc:.4f}, Best: {best_val_acc:.4f}")
        print("-" * 50)

# 加载最佳模型
if best_model_weights is not None:
    model.set_weights(best_model_weights)

# ============================================================
# 7. 训练曲线可视化（不变）
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# (a) 总损失
ax1 = axes[0, 0]
ax1.plot(train_losses, 'b-', label='Total Loss', linewidth=2.5)
ax1.fill_between(range(len(train_losses)), 0, train_losses, alpha=0.1, color='blue')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('(a) Training Total Loss', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# (b) 监督损失 vs 一致性损失
ax2 = axes[0, 1]
ax2.plot(train_super_losses, 'r-', label='Super Loss', linewidth=2.5)
ax2.plot(train_consist_losses, 'g-', label='Consist Loss', linewidth=2.5)
ax2.fill_between(range(len(train_super_losses)), 0, train_super_losses, alpha=0.1, color='red')
ax2.fill_between(range(len(train_consist_losses)), 0, train_consist_losses, alpha=0.1, color='green')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('(b) Super Loss vs Consist Loss', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# (c) 验证准确率
ax3 = axes[1, 0]
ax3.plot(val_accs, 'b-', label='Validation Accuracy', linewidth=2.5)
ax3.fill_between(range(len(val_accs)), 0, val_accs, alpha=0.1, color='blue')
ax3.axhline(y=best_val_acc, color='r', linestyle='--', 
            label=f'Best: {best_val_acc:.4f}', linewidth=2.5)
best_idx = val_accs.index(max(val_accs))
ax3.scatter(best_idx, best_val_acc, color='red', s=100, zorder=5, marker='*')
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Accuracy', fontsize=12)
ax3.set_title('(c) Validation Accuracy', fontsize=14, fontweight='bold')
ax3.legend(loc='lower right')
ax3.grid(True, alpha=0.3)
ax3.set_ylim([0, 1.05])

# (d) 验证损失
ax4 = axes[1, 1]
ax4.plot(val_losses, 'orange', label='Validation Loss', linewidth=2.5)
ax4.fill_between(range(len(val_losses)), 0, val_losses, alpha=0.1, color='orange')
min_val_loss = min(val_losses)
min_epoch = val_losses.index(min_val_loss)
ax4.axhline(y=min_val_loss, color='green', linestyle='--', 
            label=f'Min: {min_val_loss:.4f}', linewidth=2)
ax4.scatter(min_epoch, min_val_loss, color='green', s=100, zorder=5, marker='*')
ax4.set_xlabel('Epoch', fontsize=12)
ax4.set_ylabel('Loss', fontsize=12)
ax4.set_title('(d) Validation Loss', fontsize=14, fontweight='bold')
ax4.legend(loc='upper right')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(_pr('training_curves_pll_detailed.png'), dpi=300, bbox_inches='tight')
print(f"\n训练曲线已保存: {_pr('training_curves_pll_detailed.png')}")

# 打印训练统计
print("\n" + "="*60)
print("训练统计信息")
print("="*60)
print(f"总训练轮数: {len(train_losses)}")
print(f"最佳验证准确率: {best_val_acc:.4f} (Epoch {best_idx+1})")
print(f"最低验证损失: {min_val_loss:.4f} (Epoch {min_epoch+1})")
print(f"初始总损失: {train_losses[0]:.4f} → 最终总损失: {train_losses[-1]:.4f}")
print(f"初始验证准确率: {val_accs[0]:.4f} → 最终验证准确率: {val_accs[-1]:.4f}")

# ============================================================
# 8. 测试集评估（不变，df替换为df_test）
# ============================================================
print("\n" + "="*60)
print("测试集评估")
print("="*60)

y_pred_probs = model(X_test.astype(np.float32), training=False).numpy()
y_pred = np.argmax(y_pred_probs, axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {accuracy:.4f}")

print("\n分类报告:")
print(classification_report(y_test, y_pred, 
                          labels=list(label_to_idx.values()), 
                          target_names=list(label_to_idx.keys())))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred, labels=list(label_to_idx.values()))
plt.figure(figsize=(12, 10))
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                 xticklabels=list(label_to_idx.keys()), 
                 yticklabels=list(label_to_idx.keys()))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (pll with Count Boost)')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.savefig(_pr('confusion_matrix_pll_countboost.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"混淆矩阵已保存: {_pr('confusion_matrix_pll_countboost.png')}")

# ============================================================
# 9. 打印测试集前20个样本详情
# ============================================================
print("\n" + "="*100)
print("测试集详情（含特征 + 真实标签 + IRM/DTM/DPM原始预测 + 模型预测）")
print("="*100)

X_test_raw = scaler.inverse_transform(X_test)

print(f"{'序号':<6} {'H2':<8} {'CH4':<8} {'C2H6':<8} {'C2H4':<8} {'C2H2':<8} {'真实':<8} {'模型预测':<10} {'IRM':<8} {'DTM':<8} {'DPM':<8}  {'置信度':<8} {'正确':<4}")
print("-"*120)

num_print = min(20, len(X_test))
for i in range(num_print):
    h2, ch4, c2h6, c2h4, c2h2 = X_test_raw[i]
    true_label = idx_to_label[y_test[i]]
    pred_label = idx_to_label[y_pred[i]]
    confidence = y_pred_probs[i][y_pred[i]]
    is_correct = "✓" if y_test[i] == y_pred[i] else "✗"
    
    orig_idx = idx_test[i]
    row = df_test.iloc[orig_idx]
    irm_pred = row['IRM']
    dtm_pred = row['DTM']
    dpm_pred = row['DPM']
    
    print(f"{i:<6} {h2:<8.2f} {ch4:<8.2f} {c2h6:<8.2f} {c2h4:<8.2f} {c2h2:<8.2f} {true_label:<8} {pred_label:<10}  {irm_pred:<8} {dtm_pred:<8} {dpm_pred:<8} {confidence:<8.4f} {is_correct:<4}")

print("-"*120)
total_correct = np.sum(y_test == y_pred)
total_wrong = len(y_test) - total_correct
print(f"总测试样本数: {len(X_test)}")
print(f"正确预测数: {total_correct} ({total_correct/len(X_test)*100:.2f}%)")
print(f"错误预测数: {total_wrong} ({total_wrong/len(X_test)*100:.2f}%)")
print(f"测试准确率: {accuracy:.4f}")
print("="*100)

# ============================================================
# 10. 模型预测与DGA方法一致性分析
# ============================================================
print("\n模型预测与DGA方法一致性分析（前20个）:")
print(f"{'序号':<6} {'真实标签':<10} {'模型预测':<10} {'与IRM一致':<10} {'与DTM一致':<10} {'与DPM一致':<10} {'最多一致的方法':<15}")
print("-"*80)

for i in range(num_print):
    orig_idx = idx_test[i]
    row = df_test.iloc[orig_idx]
    true_label = idx_to_label[y_test[i]]
    pred_label = idx_to_label[y_pred[i]]
    
    irm_pred = row['IRM']
    dtm_pred = row['DTM']
    dpm_pred = row['DPM']
    
    match_irm = "✓" if pred_label == irm_pred else "✗"
    match_dtm = "✓" if pred_label == dtm_pred else "✗"
    match_dpm = "✓" if pred_label == dpm_pred else "✗"
    
    matches = []
    if pred_label == irm_pred and irm_pred != "Unidentified":
        matches.append("IRM")
    if pred_label == dtm_pred and dtm_pred != "Unidentified":
        matches.append("DTM")
    if pred_label == dpm_pred and dpm_pred != "Unidentified":
        matches.append("DPM")
    
    if len(matches) > 1:
        most_match = f"{', '.join(matches)} ({len(matches)}种)"
    elif len(matches) == 1:
        most_match = matches[0]
    else:
        most_match = "无匹配"
    
    print(f"{i:<6} {true_label:<10} {pred_label:<10} {match_irm:<10} {match_dtm:<10} {match_dpm:<10} {most_match:<15}")

print("-"*80)

# ============================================================
# 11. 预测置信度分布图
# ============================================================
plt.figure(figsize=(10, 6))
max_probs = np.max(y_pred_probs, axis=1)
correct_mask = (y_pred == y_test)
incorrect_mask = (y_pred != y_test)

plt.hist(max_probs[correct_mask], bins=30, alpha=0.7, 
         label=f'Correct Predictions (n={np.sum(correct_mask)})', 
         color='green', edgecolor='black')
plt.hist(max_probs[incorrect_mask], bins=30, alpha=0.7, 
         label=f'Incorrect Predictions (n={np.sum(incorrect_mask)})', 
         color='red', edgecolor='black')

plt.xlabel('Prediction Confidence', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Test Set Prediction Confidence Distribution', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(_pr('prediction_confidence_distribution.png'), dpi=300, bbox_inches='tight')
print(f"预测置信度分布图已保存: {_pr('prediction_confidence_distribution.png')}")

# ============================================================
# 12. 打印验证集和测试集错误样本（函数适配三份df）
# ============================================================

def print_error_samples(dataset_name, X_data, y_data, y_pred_labels, y_pred_probs, idx_data, 
                        scaler, df_source, idx_to_label, label_to_idx, num_print=None):
    """
    打印错误样本的详细信息
    """
    # 找出错误样本的索引
    error_mask = (y_data != y_pred_labels)
    error_indices = np.where(error_mask)[0]
    num_errors = len(error_indices)
    
    if num_errors == 0:
        print(f"\n{dataset_name}集: 没有错误样本！🎉")
        return
    
    print(f"\n{'='*120}")
    print(f"{dataset_name}集错误样本 (共 {num_errors} 个)")
    print(f"{'='*120}")
    
    # 如果指定了打印数量，取前 num_print 个；否则全部打印
    if num_print is not None:
        print_indices = error_indices[:num_print]
    else:
        print_indices = error_indices
    
    # 反标准化特征
    X_data_raw = scaler.inverse_transform(X_data)
    
    # 打印表头
    print(f"{'序号':<6} {'H2':<8} {'CH4':<8} {'C2H6':<8} {'C2H4':<8} {'C2H2':<8} "
          f"{'真实':<10} {'模型预测':<10} {'IRM':<8} {'DTM':<8} {'DPM':<8} "
          f"{'置信度':<8} {'正确':<4}")
    print("-"*130)
    
    for i in print_indices:
        h2, ch4, c2h6, c2h4, c2h2 = X_data_raw[i]
        true_label = idx_to_label[y_data[i]]
        pred_label = idx_to_label[y_pred_labels[i]]
        confidence = y_pred_probs[i][y_pred_labels[i]]
        is_correct = "✗"
        
        orig_idx = idx_data[i]
        row = df_source.iloc[orig_idx]
        irm_pred = row['IRM']
        dtm_pred = row['DTM']
        dpm_pred = row['DPM']
        
        print(f"{i:<6} {h2:<8.2f} {ch4:<8.2f} {c2h6:<8.2f} {c2h4:<8.2f} {c2h2:<8.2f} "
              f"{true_label:<10} {pred_label:<10} {irm_pred:<8} {dtm_pred:<8} {dpm_pred:<8} "
              f"{confidence:<8.4f} {is_correct:<4}")
    
    print("-"*130)
    
    # 统计错误类型分布
    print(f"\n错误类型统计 (真实标签 → 预测标签):")
    error_types = {}
    for i in error_indices:
        true_label = idx_to_label[y_data[i]]
        pred_label = idx_to_label[y_pred_labels[i]]
        key = f"{true_label} → {pred_label}"
        error_types[key] = error_types.get(key, 0) + 1
    
    # 按次数排序
    sorted_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)
    for key, count in sorted_errors:
        print(f"  {key}: {count} 个")
    
    print("="*120)

# ---- 测试集错误样本 ----
print_error_samples(
    dataset_name="测试集",
    X_data=X_test,
    y_data=y_test,
    y_pred_labels=y_pred,
    y_pred_probs=y_pred_probs,
    idx_data=idx_test,
    scaler=scaler,
    df_source=df_test,
    idx_to_label=idx_to_label,
    label_to_idx=label_to_idx,
    num_print=None
)

# ---- 验证集错误样本 ----
y_pred_val_probs = model(X_val.astype(np.float32), training=False).numpy()
y_pred_val_labels = np.argmax(y_pred_val_probs, axis=1)

print_error_samples(
    dataset_name="验证集",
    X_data=X_val,
    y_data=y_val,
    y_pred_labels=y_pred_val_labels,
    y_pred_probs=y_pred_val_probs,
    idx_data=idx_val,
    scaler=scaler,
    df_source=df_val,
    idx_to_label=idx_to_label,
    label_to_idx=label_to_idx,
    num_print=None
)

# ============================================================
# 13. 保存结果
# ============================================================
model.save(_pr('pll_model_countboost.h5'))
with open(_pr('label_mapping_countboost.json'), 'w', encoding='utf-8') as f:
    json.dump(label_to_idx, f, ensure_ascii=False, indent=4)

# 保存测试结果
test_result = pd.DataFrame({
    '真实标签': [idx_to_label[t] for t in y_test],
    '预测标签': [idx_to_label[p] for p in y_pred],
    '预测概率': [max(probs) for probs in y_pred_probs]
})
test_result.to_excel(_pr('test_results_pll_countboost.xlsx'), index=False)

print("\n模型和结果已保存:")
print(f"  - {_pr('pll_model_countboost.h5')}")
print(f"  - {_pr('label_mapping_countboost.json')}")
print(f"  - {_pr('test_results_pll_countboost.xlsx')}")

print("\n" + "="*60)
print("程序执行完成！")
print("="*60)
# 那你根据我的代码帮我看看我是怎么生成候选标签的