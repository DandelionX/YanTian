# YanTian 全球中期天气预报推理模型

## 一、项目简介

**YanTian（言天）** 是一个面向全球中期天气预报（Medium-range Weather Forecasting）的 AI 大模型。该模型基于提出的 **Searth Transformer（Shifted Earth Transformer）** 架构，并结合创新的 **Relay Autoregressive (RAR)** 微调策略，实现了在较低计算资源条件下的高精度中期天气预报能力。

本项目开源内容为 **推理模型（ONNX 格式）及实时推理脚本**，可用于基于实时 GFS 数据进行全球 6 小时间隔滚动预报。

------

## 二、论文信息

### 论文标题

**Searth Transformer: A Transformer Architecture Incorporating Earth's Geospheric Physical Priors for Global Mid-Range Weather Forecasting**

### 论文链接

https://doi.org/10.48550/arXiv.2601.09467

### 论文核心贡献

本文提出：

### 1️⃣ Searth Transformer（Shifted Earth Transformer）

- 在 Transformer 的 window-based self-attention 中显式引入：
  - 经向非周期边界（南北边界）
  - 纬向周期连续性（东西方向周期）
- 通过非对称 shift-and-mask 机制：
  - 取消经向周期 mask，实现全球经向信息连续传播
  - 保留极区 mask，避免物理不合理的跨极信息混合
- 提升了大尺度环流建模能力

### 2️⃣ Relay Autoregressive (RAR) 微调策略

- 将长时间滚动预测分解为多个子阶段
- 每阶段独立反向传播
- 阶段间进行梯度 detach
- 显著降低 GPU 显存占用
- 支持学习 15 天连续演变

### 3️⃣ YanTian 模型性能

- 分辨率：1°
- 变量数：69 个气象变量
- 时间分辨率：6 小时
- Z500 技能时效达到 **10.3 天**
- 在 1° 分辨率下：
  - 超越 ECMWF HRES
  - 达到当前主流 AI 模型水平
- 训练峰值显存 < 25GB

## 三、模型基本信息

| 项目        | 内容                 |
| ----------- | -------------------- |
| 模型名称    | YanTian              |
| 架构        | Encoder-Core-Decoder |
| Transformer | Searth Transformer   |
| 参数量      | ~600M                |
| 输入时间步  | 2 个历史时刻         |
| 输出时间步  | 1 个未来 6h 预报     |
| 水平分辨率  | 1° (180 × 360)       |
| 时间间隔    | 6 小时               |
| 变量数      | 69                   |

## 四、输入数据说明

### 输入数据预处理

YanTian 推理模型对输入数据的物理变量顺序、空间排布方式以及归一化方式有严格要求。如不符合下述规范，模型预测结果将出现严重偏差。

模型输入前必须使用 `statistics.json` 文件中提供的 **69 个变量对应的均值（mean）和标准差（std）进行标准化处理**，同时需要对输入变量水平空间分辨率降尺度从0.25度降低到1度，维度从（721，1440）变为（180，360），（由于采用窗口平均风发721无法被4整除，因此舍弃最后的南极点，从721变为180）。归一化和降尺度窗口平均操作和参考GFS推理文件夹

标准化方式：
$$
X_{norm} = \frac{X - \mu}{\sigma}
$$
其中：

- $X$ 为原始气象场
- $\mu$ 为 `statistics.json` 中对应变量的均值
- $\sigma$ 为 `statistics.json` 中对应变量的标准差

### 输入

$$
x \in \mathbb{R}^{B \times 2 \times 69 \times 180 \times 360}
$$

维度含义：

| 维度 | 含义                      |
| ---- | ------------------------- |
| B    | Batch size                |
| 2    | 两个历史时间步（t-6h, t） |
| 69   | 气象变量通道数            |
| 180  | 纬度（H）                 |
| 360  | 经度（W）                 |

即：

```
(Batch, Time, Channel, Lat, Lon)
```

------

### 输出

$$
\hat{y} \in \mathbb{R}^{B \times 69 \times 180 \times 360}
$$

输出为：

- 下一个 6 小时的预报场
- 69 个变量
- 1° 全球网格

即：

```
(Batch, Channel, Lat, Lon)
```

### 69个变量

高度层顺序为[50,100,150,200,250,300,400,500,600,700,850,925,1000]

1. Z（位势而非位势高度，单位m2 s-2），索引：z50-z1000: 0-12
2. R（相对湿度，单位%），索引：r50-r1000: 13-25
3. T（开尔文温度，单位K），索引：t50-t1000: 26-38
4. U（东西风），索引：u50-u1000: 39-51
5. V（南北风），索引：v50-v1000: 52-64

#### 4个地表变量：

1. U10：（10米高度东西风），索引：65
2. V10：（10米高度南北风），索引：66
3. T2M：（两米温度），索引：67
4. MSL：（海平面气压），索引：68

## 五、项目文件结构说明

YanTian/
│
├── inference/
│   ├── run_cpu.py        # 基于 GFS 实时数据使用CPU进行推理脚本
│   ├── prepare.py         # 归一化、窗口平均降尺度的数据预处理脚本
│   ├── download.py          # GFS 数据下载脚本

├── model/
│   ├── YanTian.onnx               # ONNX 推理模型
│   ├── YanTian.data               # ONNX 权重数据文件
│
└── README.md
