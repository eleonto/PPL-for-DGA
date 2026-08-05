# PLL-DGA 变压器油中溶解气体故障诊断系统

基于 **PLL (Partial Label Learning)** 方法的变压器 DGA (Dissolved Gas Analysis) 故障诊断系统。结合多种 DGA 诊断方法（IRM、DTM、DPM）的候选标签，通过偏标记学习训练神经网络，实现更准确的变压器故障类型识别。

---

## 项目结构

```
PLL-DGA/
├── pll-DGA/
│   ├── dataset/                # 数据集目录
│   │   ├── train.csv          # 训练集
│   │   ├── val.csv            # 验证集
│   │   └── test.csv           # 测试集
│   ├── train.py               # 训练脚本
│   ├── label_mapping.json     # 标签映射（旧版）
│   └── 数据处理标准化.py       # 数据预处理脚本
│
├── predict.py                 # 推理脚本（单条/批量/交互三种模式）
├── pll_model_countboost.h5    # 训练好的模型权重（CountBoost版）
├── label_mapping_countboost.json  # 标签映射（训练产出）
│
├── training_curves_pll_detailed.png   # 训练曲线
├── confusion_matrix_pll_countboost.png # 混淆矩阵
├── prediction_confidence_distribution.png # 置信度分布
└── test_prediction_results.xlsx        # 示例预测结果
```

> **注意**：所有路径均基于脚本文件位置自动解析，不受工作目录影响。

---

## 环境要求

- Python 3.8+
- TensorFlow / Keras 2.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- openpyxl （用于读写 Excel）

### 安装依赖

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn openpyxl
```

---

## 数据集格式

CSV 文件需包含以下列：

| 列名    | 含义                                |
|---------|-------------------------------------|
| H2      | 氢气含量 (μL/L)                    |
| CH4     | 甲烷含量                            |
| C2H6    | 乙烷含量                            |
| C2H4    | 乙烯含量                            |
| C2H2    | 乙炔含量                            |
| origin  | 真实故障标签（ground truth）        |
| IRM     | IEC Ratio Method 诊断结果           |
| DTM     | Duval Triangle Method 诊断结果      |
| DPM     | Doernenburg Percentage Method 诊断  |

### 故障类别说明

| 标签         | 含义                  |
|--------------|-----------------------|
| Normal       | 正常                  |
| PD           | 局部放电 (Partial Discharge) |
| D1           | 低能放电 (Low-energy Discharge) |
| D2           | 高能放电 (High-energy Discharge) |
| T1           | 低温过热 < 300℃       |
| T2           | 中温过热 300~700℃     |
| T3           | 高温过热 > 700℃       |
| Unidentified | 无法识别              |

---

## 模型训练

运行训练脚本：

```bash
cd pll-DGA
python train.py
```

### 训练核心参数

可在 [train.py](file:///c:/Users/zhujin/Desktop/正在做/PLL-DGA/pll-DGA/train.py#L34-L56) 顶部调整：

```python
WEIGHT_ORIGIN = 1      # 真实标签权重
WEIGHT_DPM = 1         # DPM 方法权重
WEIGHT_DTM = 1         # DTM 方法权重
WEIGHT_IRM = 1         # IRM 方法权重

ENABLE_COUNT_BOOST = True  # 多方法一致时自动增强权重
W_SUP = 0.9                # 监督损失权重
lam_init = 0.1             # 一致性损失初始权重
K = 2                      # 随机前向次数

epochs = 500               # 最大训练轮数
batch_size = 64            # 批大小
patience = 50              # EarlyStopping 耐心值
```

### 模型架构

```
Input(5) → Dense(128, relu) → Dropout(0.3)
         → Dense(128, relu) → Dropout(0.3)
         → Dense(8, softmax)
```

- 参数量：18,312
- 输入维度：5（H2, CH4, C2H6, C2H4, C2H2，经 StandardScaler 标准化）
- 输出维度：8（8 类故障概率）

### 训练产出

训练完成后，以下文件自动保存至项目根目录：

| 文件名                                | 内容                     |
|---------------------------------------|--------------------------|
| pll_model_countboost.h5               | Keras 模型权重           |
| label_mapping_countboost.json         | 标签 → 索引 映射字典     |
| training_curves_pll_detailed.png      | 训练/验证曲线 (4子图)    |
| confusion_matrix_pll_countboost.png   | 测试集混淆矩阵           |
| prediction_confidence_distribution.png| 预测置信度分布图         |
| test_results_pll_countboost.xlsx      | 测试集逐条预测结果       |

---

## 模型推理 / 预测

[predict.py](file:///c:/Users/zhujin/Desktop/正在做/PLL-DGA/predict.py) 提供三种推理模式，均会自动加载训练好的模型和标准化参数。

### 模式 1：交互式预测（默认）

直接输入 5 个气体特征值，实时获得诊断结果：

```bash
python predict.py
# 或
python predict.py --mode interactive
```

运行示例：
```
请输入特征值（用逗号或空格分隔）:
顺序: H2, CH4, C2H6, C2H4, C2H2
> 100, 50, 30, 20, 5
```

输入 `q` 退出。

---

### 模式 2：单条样本预测（命令行传参）

```bash
python predict.py --mode single \
  --h2 100 --ch4 50 --c2h6 30 --c2h4 20 --c2h2 5
```

不传参数时会使用示例数据演示。

---

### 模式 3：批量 CSV 预测

```bash
# 使用内置测试集
python predict.py --mode batch

# 指定输入 CSV 和输出 Excel
python predict.py --mode batch \
  --input your_data.csv \
  --output your_result.xlsx
```

**输入要求**：CSV 文件需包含 `H2, CH4, C2H6, C2H4, C2H2` 列。

**输出内容**：
- 原始 5 项特征
- `预测标签`、`预测置信度`
- 每类故障单独一列的概率：`概率_D1`、`概率_D2`、`概率_Normal` ...

---

## 推理输出示例

单条预测输出：

```
==================================================
单条样本预测
==================================================
输入特征:
  H2: 100.0, CH4: 50.0, C2H6: 30.0, C2H4: 20.0, C2H2: 5.0

预测结果:
  故障类型: Normal
  置信度: 0.2437 (24.37%)

各类别概率:
  Normal      : 0.2437 ████████████
  T1          : 0.2404 ████████████
  PD          : 0.1820 █████████
  ...
==================================================
```

---

## 核心算法说明

### PLL 损失函数

训练过程中使用组合损失：

```
L_total = L_super + λ × L_consist
```

1. **L_super（监督损失）**：对非候选标签施加惩罚，鼓励模型排除不可能类别
2. **L_consist（一致性损失）**：多次随机前向（K=2）的预测分布与置信度矩阵 KL 散度

### 置信度动态更新

每 10 个 epoch 更新一次置信度矩阵：

- 几何平均多次前向概率
- 仅在候选标签集合内保留
- 指数滑动平均平滑：`conf = α·conf_old + (1-α)·conf_new`

### CountBoost 机制

同一故障被多个 DGA 方法（IRM/DTM/DPM+origin）同时命中时，候选标签权重按出现次数放大，避免单一方法误判的干扰。

---

## 常见问题

### Q1: 报错 "模型文件不存在"
确认项目根目录下存在 `pll_model_countboost.h5` 和 `label_mapping_countboost.json`。如果缺失，请先运行训练脚本。

### Q2: 预测置信度普遍偏低？
1. 检查输入气体含量的数量级是否与训练集一致（参考训练集均值和标准差）
2. 部分样本本身处于类别边界，多类别概率接近属于正常现象

### Q3: 可以从任意目录运行脚本吗？
可以。所有路径均基于脚本的 `__file__` 绝对路径解析，不依赖 `os.getcwd()`。

### Q4: 更换数据集后需要重新训练吗？
是的。数据分布变化后，建议更新 `pll-DGA/dataset/` 下的 CSV 并重新运行训练。

---

## 文件清单速查

| 功能 | 文件 |
|------|------|
| 训练入口 | [pll-DGA/train.py](file:///c:/Users/zhujin/Desktop/正在做/PLL-DGA/pll-DGA/train.py) |
| 推理入口 | [predict.py](file:///c:/Users/zhujin/Desktop/正在做/PLL-DGA/predict.py) |
| 训练数据集 | [pll-DGA/dataset/train.csv](file:///c:/Users/zhujin/Desktop/正在做/PLL-DGA/pll-DGA/dataset/train.csv) |
| 模型权重 | pll_model_countboost.h5 |
| 标签映射 | label_mapping_countboost.json |
