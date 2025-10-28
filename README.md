Here is a **single, complete, highly detailed, academic-grade `README.md`** file — ready to be copied into your GitHub repository root. It includes everything: paper abstract, model overview, folder structure, GEE setup, code usage, cross-sensor/index generalization, citation, and reproducibility.

---

```markdown
# ConvLSTM-GCN-Transformer: Spatiotemporal Graph-Attention Model for Multisource Vegetation Index Map Forecasting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](https://your-paper-link.pdf)
[![DOI](https://img.shields.io/badge/DOI-10.1000/xyz123-blue)](https://doi.org/10.1000/xyz123)

> **Official Code Repository**  
> *IEEE Transactions on Geoscience and Remote Sensing (Under Review)*

---

## Abstract

Vegetation indices (VIs) play a key role in monitoring crop growth and ecosystem dynamics under changing climate conditions. Accurate forecasting of these indices supports sustainable resource management and environmental planning. However, most existing deep learning studies focus on predicting single scalar VI values or regional averages, overlooking the spatial continuity and fine-scale mapping required for operational forecasting. Approaches that generate VI maps often remain limited to small spatial extents and tend to lose consistency when applied to complex landscapes combining forests, water bodies, urban areas, and bare soil.

This paper introduces a **hybrid ConvLSTM–GCN–Transformer model** for predicting high-resolution VI maps from satellite image sequences.  
- **ConvLSTM** captures temporal variations  
- **Graph Convolutional Network (GCN)** models local spatial dependencies using a 4-neighborhood pixel graph  
- **Transformer** enhances long-range feature representation and scene-level coherence  

The model was trained on monthly **NDVI maps** derived from **Landsat** imagery from **1996 to 2025** over **northern Tunisia**, achieving an **RMSE of 0.034** and an **NSE of 0.866**, outperforming baseline architectures. Two generalization studies were conducted:  
1. **Cross-sensor evaluation** using **Sentinel-2** and **MODIS** imagery (RMSEs of 0.064 and 0.0901)  
2. **Cross-index adaptation** to **EVI** and **SAVI** (RMSEs of 0.059 and 0.052)  

**All codes and architectures are publicly available** to support reproducibility.

---

## Key Contributions

- A **ConvLSTM–GCN–Transformer architecture** for map-based prediction, integrating spatial structures and temporal dynamics with high precision  
- A **graph-based spatial formulation** using sparse GCN operations to efficiently model spatial interactions across large-scale, full-resolution satellite maps  
- Training on a **heterogeneous landscape** (forests, urban areas, water bodies, bare soil) over an extended temporal range (**1996–2025**)  
- Validation of **cross-sensor** (Sentinel-2, MODIS) and **cross-index** (EVI, SAVI) generalization  

---

## Model Architecture

```
Input Sequence (9 months) → ConvLSTM → Reshape → GCN (4-neighbor graph) → Transformer → Reshape → Conv2D → Output Map (t+1)
```

- **Input**: `(batch, 9, 372, 743, 1)`  
- **Output**: `(batch, 372, 743, 1)`  
- **GCN Graph**: Sparse 4-neighbor adjacency (precomputed)  
- **Transformer**: 4 heads, embed_dim=32, ff_dim=64  

![Model Diagram](CONVLSTM_GCN_TRANSFORMER.png)

---

## Repository Structure

```
ConvLSTM-GCN-Transformer-VI-Forecasting/
│
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── CONVLSTM_GCN_TRANSFORMER.png     # Model architecture diagram
├── implementation_functions.py      # Shared utilities (graph, layers)
│
├── architectures/                   # All models (ablation + final)
│   ├── ConvLSTM.py
│   ├── ConvLSTM_3DCNN.py
│   ├── ConvLSTM_GCN.py
│   ├── ConvLSTM_Transformer.py
│   ├── ConvLSTM_GCN_Transformer.py
│   └── GCN_Transformer.py
│
├── data_collection/                 # GEE data download
│   ├── landsat.py
│   ├── sentinel2.py
│   ├── modis.py
│   └── README.md                    # GEE setup guide
│
├── data_preparation/
│   ├── sequences_creation.py
│   └── utils.py
│
├── training/
│   ├── train.py
│   ├── test.py
│   └── evaluation_metrics.py
│
├── models/                          # Saved weights (.h5)
│   └── .gitignore
│
├── results/                         # Sample outputs
│   └── sample_prediction.png
│
└── notebooks/                       # Optional analysis
    └── data_exploration.ipynb
```

---

## Installation

```bash
git clone https://github.com/yourusername/ConvLSTM-GCN-Transformer-VI-Forecasting.git
cd ConvLSTM-GCN-Transformer-VI-Forecasting
pip install -r requirements.txt
```

### Required Packages (`requirements.txt`)

```txt
tensorflow==2.15.0
numpy==1.24.3
earthengine-api==0.1.383
geemap==0.32.0
matplotlib==3.7.2
scikit-learn==1.3.0
pandas==2.0.3
tqdm==4.66.1
```

---

## Step-by-Step Usage

### 1. Set Up Google Earth Engine (GEE)

1. Sign up at [https://code.earthengine.google.com](https://code.earthengine.google.com)  
2. Create a **Google Cloud Project** and enable **Earth Engine API**  
3. Authenticate:

```bash
earthengine authenticate
```

> Follow the link, log in, copy the token.

---

### 2. Download Satellite Data (GEE)

Use scripts in `data_collection/`. All accept the same CLI arguments.

#### Example: Download Landsat NDVI (1996–2025)

```bash
python data_collection/landsat.py \
  --output_dir ./data/landsat_ndvi \
  --start_date 1996-01-01 \
  --end_date 2025-12-31 \
  --region "[[-10.5, 35.0], [-8.0, 35.0], [-8.0, 37.0], [-10.5, 37.0]]" \
  --cloud_project your-gcp-project-id \
  --index NDVI
```

#### Supported Indices

| Index | Formula (modify in script) |
|------|----------------------------|
| `NDVI` | `(nir - red) / (nir + red + 1e-6)` |
| `EVI`  | `2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1)` |
| `SAVI` | `((nir - red) / (nir + red + 0.5)) * 1.5` |

> **Tip**: Change the formula in `landsat.py`, `sentinel2.py`, or `modis.py` to compute any VI.

---

### 3. Create Training Sequences

```bash
python data_preparation/sequences_creation.py \
  --input_dir ./data/landsat_ndvi \
  --output_dir ./data/sequences \
  --seq_length 9 \
  --forecast_step 1
```

- Output: `.npy` files of shape `(samples, 9, 372, 743, 1)`  
- Splits: `train/`, `val/`, `test/`

---

### 4. Train the Model

```bash
python training/train.py \
  --data_path ./data/sequences \
  --model_name ConvLSTM_GCN_Transformer \
  --epochs 100 \
  --batch_size 4 \
  --output_model models/ndvi_final.h5
```

#### Available Models (`--model_name`)

- `ConvLSTM`
- `ConvLSTM_3DCNN`
- `ConvLSTM_GCN`
- `ConvLSTM_Transformer`
- `ConvLSTM_GCN_Transformer` (final)
- `GCN_Transformer`

---

### 5. Test & Evaluate

```bash
python training/test.py \
  --model_path models/ndvi_final.h5 \
  --test_data ./data/sequences/test \
  --output_dir results/
```

Outputs:
- Predicted vs. ground truth maps
- Error heatmaps
- Metrics: RMSE, NSE, MAE

---

## Generalization Experiments

### Cross-Sensor (Zero-Shot)

Train on **Landsat**, test on **Sentinel-2** or **MODIS**:

```bash
python training/test.py \
  --model_path models/landsat_ndvi.h5 \
  --test_data ./data/sequences/sentinel2_test \
  --normalize_using landsat_stats.json
```

### Cross-Index (EVI / SAVI)

1. Download EVI/SAVI using `--index EVI` or `--index SAVI`  
2. Retrain or fine-tune  
3. Evaluate transfer performance

---

## Reproducibility

| Component | Specification |
|---------|---------------|
| Region | Northern Tunisia (372×743 px) |
| Time Range | 1996–2025 (monthly) |
| Sensors | Landsat-5/7/8, Sentinel-2, MODIS |
| Indices | NDVI, EVI, SAVI |
| Hardware | GPU (12GB+), 32GB RAM |
| Seeds | Fixed in `implementation_functions.py` |

---

## Citation

Please cite our work as:

```bibtex
@article{yourname2025convlstm,
  title={ConvLSTM-GCN-Transformer: Spatiotemporal Graph-Attention Model for Multisource Vegetation Index Map Forecasting},
  author={Your Name and Co-authors},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2025},
  note={under review},
  doi={10.1000/xyz123}
}
```

---

## License

[MIT License](LICENSE) – Free for academic and non-commercial use.

---

## Contact & Issues

- **Email**: your.email@university.edu  
- **Issues**: [GitHub Issues](https://github.com/yourusername/ConvLSTM-GCN-Transformer-VI-Forecasting/issues)

---