import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import json
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================================
# 路径统一管理（基于脚本文件位置，避免工作目录影响）
# ============================================================
# 脚本所在目录（作为根目录锚点）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _p(*parts):
    """基于 BASE_DIR 构造规范化的绝对路径"""
    return os.path.normpath(os.path.join(BASE_DIR, *parts))

# ============================================================
# 配置参数
# ============================================================
MODEL_PATH = _p("pll_model_countboost.h5")
LABEL_MAPPING_PATH = _p("label_mapping_countboost.json")
TRAIN_DATA_PATH = _p("pll-DGA", "dataset", "train.csv")  # 用于拟合标准化器

FEATURE_COLS = ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2']

# ============================================================
# 1. 加载模型和标签映射
# ============================================================
print("正在加载模型...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")
model = load_model(MODEL_PATH)
print(f"模型加载成功: {MODEL_PATH}")
model.summary()

print("\n正在加载标签映射...")
if not os.path.exists(LABEL_MAPPING_PATH):
    raise FileNotFoundError(f"标签映射文件不存在: {LABEL_MAPPING_PATH}")
with open(LABEL_MAPPING_PATH, 'r', encoding='utf-8') as f:
    label_to_idx = json.load(f)
idx_to_label = {v: k for k, v in label_to_idx.items()}
print(f"标签映射加载成功: {label_to_idx}")

# ============================================================
# 2. 加载训练数据并拟合标准化器
# ============================================================
print("\n正在加载训练数据用于标准化...")
if not os.path.exists(TRAIN_DATA_PATH):
    raise FileNotFoundError(f"训练数据文件不存在: {TRAIN_DATA_PATH}")

df_train = pd.read_csv(TRAIN_DATA_PATH)
scaler = StandardScaler()
scaler.fit(df_train[FEATURE_COLS])
print(f"标准化器已从训练数据拟合完成")
print(f"  均值: {dict(zip(FEATURE_COLS, scaler.mean_))}")
print(f"  标准差: {dict(zip(FEATURE_COLS, scaler.scale_))}")

# ============================================================
# 3. 单条样本预测函数
# ============================================================
def predict_single(h2, ch4, c2h6, c2h4, c2h2, verbose=True):
    """
    预测单条样本的故障类型
    
    参数:
        h2: H2 含量
        ch4: CH4 含量
        c2h6: C2H6 含量
        c2h4: C2H4 含量
        c2h2: C2H2 含量
        verbose: 是否打印详细信息
    
    返回:
        dict: 包含预测标签、置信度、各类别概率的字典
    """
    features = np.array([[h2, ch4, c2h6, c2h4, c2h2]])
    features_scaled = scaler.transform(features)
    
    probs = model.predict(features_scaled, verbose=0)[0]
    pred_idx = np.argmax(probs)
    pred_label = idx_to_label[pred_idx]
    confidence = probs[pred_idx]
    
    result = {
        'prediction': pred_label,
        'confidence': float(confidence),
        'probabilities': {idx_to_label[i]: float(probs[i]) for i in range(len(probs))}
    }
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"单条样本预测")
        print(f"{'='*50}")
        print(f"输入特征:")
        print(f"  H2: {h2}")
        print(f"  CH4: {ch4}")
        print(f"  C2H6: {c2h6}")
        print(f"  C2H4: {c2h4}")
        print(f"  C2H2: {c2h2}")
        print(f"\n预测结果:")
        print(f"  故障类型: {pred_label}")
        print(f"  置信度: {confidence:.4f} ({confidence*100:.2f}%)")
        print(f"\n各类别概率:")
        for label, prob in sorted(result['probabilities'].items(), key=lambda x: -x[1]):
            bar = '█' * int(prob * 50)
            print(f"  {label:<12}: {prob:.4f} {bar}")
        print(f"{'='*50}\n")
    
    return result

# ============================================================
# 4. 批量 CSV 文件预测函数
# ============================================================
def predict_csv(csv_path, output_path=None, verbose=True):
    """
    对 CSV 文件中的所有样本进行预测
    
    参数:
        csv_path: 输入 CSV 文件路径（需包含 H2, CH4, C2H6, C2H4, C2H2 列）
        output_path: 输出结果 Excel 文件路径（可选）
        verbose: 是否打印详细信息
    
    返回:
        pd.DataFrame: 包含原始特征和预测结果的 DataFrame
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"输入文件不存在: {csv_path}")
    
    print(f"\n正在加载数据文件: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 检查必需的特征列是否存在
    missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV 文件缺少以下特征列: {missing_cols}")
    
    print(f"样本数量: {len(df)}")
    
    # 特征标准化
    X = df[FEATURE_COLS].values
    X_scaled = scaler.transform(X)
    
    # 批量预测
    print("正在进行预测...")
    probs = model.predict(X_scaled, verbose=0)
    pred_indices = np.argmax(probs, axis=1)
    pred_labels = [idx_to_label[idx] for idx in pred_indices]
    confidences = np.max(probs, axis=1)
    
    # 构建结果 DataFrame
    result_df = df.copy()
    result_df['预测标签'] = pred_labels
    result_df['预测置信度'] = confidences
    
    # 添加各类别概率列
    for i in range(len(idx_to_label)):
        result_df[f'概率_{idx_to_label[i]}'] = probs[:, i]
    
    # 统计信息
    if verbose:
        print(f"\n{'='*60}")
        print(f"批量预测统计")
        print(f"{'='*60}")
        print(f"总样本数: {len(df)}")
        print(f"\n预测标签分布:")
        label_counts = result_df['预测标签'].value_counts()
        for label, count in label_counts.items():
            percentage = count / len(df) * 100
            print(f"  {label:<12}: {count:4d} 个 ({percentage:.1f}%)")
        
        print(f"\n置信度统计:")
        print(f"  平均置信度: {confidences.mean():.4f}")
        print(f"  最高置信度: {confidences.max():.4f}")
        print(f"  最低置信度: {confidences.min():.4f}")
        print(f"  标准差:     {confidences.std():.4f}")
        
        # 显示前10条预测结果
        print(f"\n前10条预测结果:")
        print(f"{'序号':<6} {'H2':<10} {'CH4':<10} {'C2H6':<10} {'C2H4':<10} {'C2H2':<10} {'预测标签':<12} {'置信度':<10}")
        print("-" * 90)
        for i in range(min(10, len(df))):
            row = result_df.iloc[i]
            print(f"{i:<6} {row['H2']:<10.2f} {row['CH4']:<10.2f} {row['C2H6']:<10.2f} "
                  f"{row['C2H4']:<10.2f} {row['C2H2']:<10.2f} {row['预测标签']:<12} {row['预测置信度']:<10.4f}")
        print(f"{'='*60}\n")
    
    # 保存结果
    if output_path:
        result_df.to_excel(output_path, index=False)
        print(f"预测结果已保存: {output_path}")
    
    return result_df

# ============================================================
# 5. 交互式预测模式
# ============================================================
def interactive_mode():
    """
    交互式预测模式，用户可以手动输入特征值
    """
    print("\n" + "="*60)
    print("交互式预测模式")
    print("输入变压器溶解气体特征进行故障诊断")
    print("输入 'q' 退出")
    print("="*60)
    
    while True:
        try:
            print("\n请输入特征值（用逗号或空格分隔）:")
            print("顺序: H2, CH4, C2H6, C2H4, C2H2")
            user_input = input("> ").strip()
            
            if user_input.lower() == 'q':
                print("\n退出预测。")
                break
            
            # 解析输入
            values = [float(x.strip()) for x in user_input.replace(',', ' ').split() if x.strip()]
            
            if len(values) != 5:
                print(f"错误: 需要输入5个特征值，当前输入了 {len(values)} 个")
                continue
            
            # 预测
            h2, ch4, c2h6, c2h4, c2h2 = values
            predict_single(h2, ch4, c2h6, c2h4, c2h2)
            
        except ValueError as e:
            print(f"输入错误: {e}")
        except Exception as e:
            print(f"发生错误: {e}")

# ============================================================
# 6. 主程序入口
# ============================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PLL-DGA 变压器故障诊断推理系统')
    parser.add_argument('--mode', choices=['single', 'batch', 'interactive'], 
                        default='interactive', help='预测模式: single(单条), batch(批量CSV), interactive(交互)')
    parser.add_argument('--input', type=str, help='输入CSV文件路径 (batch模式)')
    parser.add_argument('--output', type=str, help='输出结果文件路径 (batch模式)')
    parser.add_argument('--h2', type=float, help='H2含量 (single模式)')
    parser.add_argument('--ch4', type=float, help='CH4含量 (single模式)')
    parser.add_argument('--c2h6', type=float, help='C2H6含量 (single模式)')
    parser.add_argument('--c2h4', type=float, help='C2H4含量 (single模式)')
    parser.add_argument('--c2h2', type=float, help='C2H2含量 (single模式)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("PLL-DGA 变压器故障诊断推理系统")
    print("="*60)
    
    if args.mode == 'single':
        # 命令行单条预测
        if all(v is not None for v in [args.h2, args.ch4, args.c2h6, args.c2h4, args.c2h2]):
            predict_single(args.h2, args.ch4, args.c2h6, args.c2h4, args.c2h2)
        else:
            # 使用示例数据
            print("\n未提供完整参数，使用示例数据演示:")
            print("示例: H2=100, CH4=50, C2H6=30, C2H4=20, C2H2=5")
            predict_single(100.0, 50.0, 30.0, 20.0, 5.0)
    
    elif args.mode == 'batch':
        # 批量CSV预测
        if args.input:
            output_path = args.output or _p('prediction_results.xlsx')
            results = predict_csv(args.input, output_path)
        else:
            print("\n未指定输入文件，使用测试集示例:")
            test_path = _p("pll-DGA", "dataset", "test.csv")
            if os.path.exists(test_path):
                results = predict_csv(test_path, _p('test_prediction_results.xlsx'))
            else:
                print(f"测试文件不存在: {test_path}")
    
    elif args.mode == 'interactive':
        # 交互模式
        interactive_mode()