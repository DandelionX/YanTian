# YanTian Global Medium-Range Weather Forecasting Inference Model

## 1. Project Overview

**YanTian** is a large-scale AI model designed for **global medium-range weather forecasting**. The model is built upon the proposed **Searth Transformer (Shifted Earth Transformer)** architecture and incorporates the novel **Relay Autoregressive (RAR)** fine-tuning strategy, enabling high-accuracy medium-range forecasting under relatively limited computational resources.

This repository releases the **inference model (ONNX format) and real-time inference scripts**, which support rolling global 6-hour forecasts driven by real-time GFS data.

------

## 2. Paper Information

### Title

**Searth Transformer: A Transformer Architecture Incorporating Earth's Geospheric Physical Priors for Global Mid-Range Weather Forecasting**

### Paper Link

https://doi.org/10.48550/arXiv.2601.09467

------

## Core Contributions

### 1️⃣ Searth Transformer (Shifted Earth Transformer)

- Explicitly incorporates physical priors of the Earth system into window-based self-attention:
  - Meridional non-periodic boundaries (North–South boundaries)
  - Zonal periodic continuity (East–West periodicity)
- Introduces an asymmetric shift-and-mask mechanism:
  - Removes zonal boundary masks to enable global longitudinal information exchange
  - Preserves polar masks to prevent physically unrealistic cross-pole mixing
- Significantly improves large-scale atmospheric circulation modeling capability

------

### 2️⃣ Relay Autoregressive (RAR) Fine-Tuning Strategy

- Decomposes long-horizon autoregressive forecasting into multiple sub-stages
- Performs independent backpropagation within each stage
- Applies gradient detachment between stages
- Substantially reduces GPU memory consumption
- Enables learning of continuous atmospheric evolution up to 15 days

------

### 3️⃣ YanTian Model Performance

- Resolution: 1°
- Number of variables: 69 atmospheric variables
- Temporal resolution: 6 hours
- Z500 skillful forecast lead time reaches **10.3 days**
- At 1° resolution:
  - Outperforms ECMWF HRES
  - Achieves performance comparable to state-of-the-art AI models
- Peak training GPU memory usage < 25 GB

------

## 3. Model Specifications

| Item                  | Description              |
| --------------------- | ------------------------ |
| Model Name            | YanTian                  |
| Architecture          | Encoder–Core–Decoder     |
| Transformer           | Searth Transformer       |
| Parameters            | ~600M                    |
| Input Time Steps      | 2 historical states      |
| Output Time Step      | 1 future 6-hour forecast |
| Horizontal Resolution | 1° (180 × 360)           |
| Time Interval         | 6 hours                  |
| Number of Variables   | 69                       |

------

# 4. Input Data Description

## Input Data Preprocessing

The YanTian inference model requires strict adherence to:

- Variable ordering
- Spatial arrangement
- Normalization procedure

Failure to follow the specifications below will result in severe forecast degradation.

------

### 4.1 Normalization

Before inference, input data must be standardized using the **mean and standard deviation of the 69 variables provided in `statistics.json`**.

Additionally:

- The input spatial resolution must be downscaled from **0.25° to 1°**
- The grid dimension must change from **(721, 1440) → (180, 360)**

Since 721 cannot be evenly divided by 4 when applying 4×4 window averaging, the last Antarctic latitude row is discarded, resulting in 180 latitudes.

The normalization and 4×4 window-averaging downscaling procedures follow the implementation in the `inference` directory (GFS processing scripts).

Standardization formula:
$$
X_{norm} = \frac{X - \mu}{\sigma}
$$
where:

- $X$ = raw meteorological field
- $\mu$ = mean from `statistics.json`
- $\sigma$ = standard deviation from `statistics.json`

------

## Model Input

$$
x \in \mathbb{R}^{B \times 2 \times 69 \times 180 \times 360}
$$

Dimension definitions:

| Dimension | Meaning                             |
| --------- | ----------------------------------- |
| B         | Batch size                          |
| 2         | Two historical time steps (t-6h, t) |
| 69        | Number of meteorological variables  |
| 180       | Latitude (H)                        |
| 360       | Longitude (W)                       |

Tensor layout:

```
(Batch, Time, Channel, Lat, Lon)
```

------

## Model Output

$$
\hat{y} \in \mathbb{R}^{B \times 69 \times 180 \times 360}
$$

The output represents:

- The next 6-hour forecast
- 69 variables
- 1° global grid

Tensor layout:

```
(Batch, Channel, Lat, Lon)
```

------

# 69 Atmospheric Variables

### Pressure Level Order

```
[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000] hPa
```

------

### Upper-Air Variables (5 × 13 = 65 channels)

1. **Z** – Geopotential (NOT geopotential height), unit: m² s⁻²
   Index: z50–z1000 → 0–12
2. **R** – Relative Humidity, unit: %
   Index: r50–r1000 → 13–25
3. **T** – Temperature, unit: Kelvin (K)
   Index: t50–t1000 → 26–38
4. **U** – Zonal wind (east–west component)
   Index: u50–u1000 → 39–51
5. **V** – Meridional wind (north–south component)
   Index: v50–v1000 → 52–64

------

### Surface Variables (4 channels)

1. **U10** – 10 m zonal wind
   Index: 65
2. **V10** – 10 m meridional wind
   Index: 66
3. **T2M** – 2 m temperature
   Index: 67
4. **MSL** – Mean sea level pressure
   Index: 68

------

# 5. Project Structure

```
YanTian/
│
├── inference/
│   ├── run_cpu.py      # CPU-based inference using real-time GFS data
│   ├── prepare.py      # Normalization and 4×4 window-averaging downscaling
│   ├── download.py     # GFS data download script
│
├── model/
│   ├── YanTian.onnx    # ONNX inference model
│   ├── YanTian.data    # ONNX weight data file
│
└── README.md
```
