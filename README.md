# ConvLSTM-GCN-Transformer: Spatiotemporal Graph-Attention Model for Vegetation Index Map Forecasting

## 🌟 Overview

This repository provides the official implementation of the **ConvLSTM-GCN-Transformer** hybrid model, a state-of-the-art architecture designed for **high-resolution spatio-temporal forecasting of Vegetation Index (VI) maps** from satellite image sequences.

The model addresses key limitations in current forecasting approaches—namely, the loss of spatial continuity and the struggle to model long-range dependencies in complex landscapes. Our solution is a synergy of three mechanisms:
1.  **ConvLSTM:** Effectively captures the temporal dynamics and sequential dependencies within the VI time series.
2.  **GCN (Graph Convolutional Network):** Models local spatial dependencies between pixels using a sparse 4-neighbor graph, enforcing spatial coherence in the predicted map.
3.  **Transformer Encoder:** Utilizes the self-attention mechanism to enhance long-range feature representation and scene-level coherence across the entire map.

The architecture was validated on forecasting monthly **NDVI** maps derived from **Landsat** imagery (1996–2025) over northern Tunisia, achieving superior performance (e.g., RMSE of 0.034 and NSE of 0.866) against baseline models. The model also demonstrated strong **generalization capabilities** across different sensors (**Sentinel-2**, **MODIS**) and other vegetation indices (**EVI**, **SAVI**).

## ✨ Key Features & Contributions

* **Novel Hybrid Architecture:** Seamlessly integrates recurrent, graph, and attention mechanisms for comprehensive spatio-temporal modeling.
* **High-Resolution Map Prediction:** Generates detailed VI maps, critical for precision agriculture and environmental monitoring, rather than simple scalar regional averages.
* **Graph-based Spatial Modeling:** Employs a **sparse adjacency matrix** formulation for the GCN layer, allowing efficient processing of high-resolution image maps.
* **Extensive Generalization Study:** Code supports data collection from Landsat, Sentinel-2, and MODIS, enabling replication of cross-sensor and cross-index evaluation.

## 📁 Repository Structure

The project is organized for clarity and reproducibility, with core components grouped into dedicated directories:

```

ConvLSTM-GCN-Transformer/
├── All Architectures/             \# All models used for the ablation study, including the final one.
│   ├── ConvLstm.py
│   ├── ConvLstm_3DCNN.py
│   ├── ConvLstm_Gcn(4-neighbor).py
│   ├── ConvLstm_Gcn (8-neighbor).py
│   ├── ConvLstm_Transformer.py
│   ├── ConvLstm_Gcn_Transformer.py  \# The final architecture with 4-neighbor graph
    └── ConvLstm_Gcn_Transformer(8-neighbor).py  \# The final architecture with 8-neighbor 

├── Data_Collection/           \# Google Earth Engine (GEE) scripts for automated data fetching.
│   ├── sentinel_2_data.py
│   ├── modis.py
│   └── landsat.py
├── Data_Preparation/          \# Scripts to convert raw data into model-ready sequences.
│   ├── ALL_gap_filling_methods.py
    └── sequences_creation.py
├── Train and Validation/                  \# Main training and testing scripts.
│   ├── ConvLstm_Gcn_Transformer.py
│   └── test_with_spatial_metrics
    ├── train.py
│   └── test.py 
├── utils/                     \# Utility functions (e.g., GCN adjacency matrix, custom layers).
│   └── implementation_functions.py
├── assets/                    \# Project images and diagrams.
│   └── CONVlstm_gcn_TRANSFORMER.PNG
├── README.md                  \# This file.
└── requirements.txt           \# Required Python packages.

````

## ⚙️ Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Chahinechahine123/ConvLSTM-GCN-Transformer-An-advanced-acrhitecture-for-vegeation-indexes-map-prediction-.git
cd ConvLSTM-GCN-Transformer
````

### 2\. Python Environment

We recommend using a virtual environment (e.g., `conda` or `venv`).

```bash
# Create and activate a conda environment
conda create -n convlstm_gcn_transformer python=3.9
conda activate convlstm_gcn_transformer

# Install dependencies
pip install -r requirements.txt
```

> **Note:** Key dependencies include `tensorflow` (\>= 2.x), `numpy`, and the `earthengine-api`.

## 🌍 Data Collection Guide (Google Earth Engine - GEE)

The scripts in `data_collection/` automate the retrieval and export of satellite image collections (Landsat, Sentinel-2, MODIS) using the Google Earth Engine API.

### 1\. GEE Configuration

Ensure the GEE API is installed and authenticated:

```bash
pip install earthengine-api
earthengine authenticate
```

### 2\. Mandatory Parameter Customization

**You must modify the following variables** at the beginning of the chosen collection script (e.g., `landsat.py`) to reflect your study area and export settings:

| Variable | Description | **User Action Required** |
| :--- | :--- | :--- |
| `AOI_COORDINATES` | A list of coordinate pairs defining the polygon for your **Area of Interest (AOI)**. | **Replace with your study area coordinates.** |
| `START_DATE`, `END_DATE` | Strings specifying the desired **temporal range** (e.g., `'1996-01-01'`, `'2025-12-31'`). | **Set your desired time range.** |
| `GCLOUD_PROJECT_FOLDER` | The name of your **Google Cloud/GEE Drive folder** where the exported images will be stored. | **Specify your GEE Cloud output directory.** |
| `VI_EQUATION` | The spectral band equation for the desired Vegetation Index (default is NDVI). | **Customize for other indices (see below).** |

### 3\. Adapting the Vegetation Index (VI) Equation

The data collection scripts are designed to be flexible. By default, they calculate **NDVI**. To collect data for a different index (e.g., EVI, SAVI, NDWI), you only need to change the `VI_EQUATION` string and ensure the necessary bands are selected in the script.

  * **Default NDVI (Landsat example):**
    ```python
    VI_EQUATION = '(NIR - RED) / (NIR + RED)'
    ```
  * **Example for EVI:**
    If you wish to calculate the **Enhanced Vegetation Index (EVI)**, you must update the equation to use the appropriate bands (`BLUE`, `RED`, `NIR`) and coefficients:
    ```python
    # EVI Example: G * (NIR - RED) / (NIR + C1 * RED - C2 * BLUE + L)
    VI_EQUATION = '2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)'
    ```
    **Crucially, the band names used in the equation (`NIR`, `RED`, `BLUE`, etc.) must exactly match the band names defined in the specific GEE image collection (Landsat, Sentinel-2, or MODIS) you are using.**

### 4\. Running the Collection

Execute the desired script to start the export process to your Google Cloud repository:

```bash
# To collect Landsat data
python data_collection/landsat.py

# To collect Sentinel-2 data
python data_collection/sentinel_2_data.py
```

## 🧩 Data Preparation

After exporting the raw images from GEE, the `data_preparation/sequences_creation.py` script must be run. This script:

1.  Reads the exported VI maps.
2.  Applies normalization (scaling).
3.  Creates the spatio-temporal sequences: $\mathbf{X} \in \mathbb{R}^{T \times H \times W \times 1}$ (the input sequence of $T$ past images) and $\mathbf{Y} \in \mathbb{R}^{H \times W \times 1}$ (the target image for the next time step).

**Make sure to check the `SEQUENCE_LENGTH` variable in this script** to set the number of time steps ($T$) used for the input (default is $T=9$ based on our study).

## 🚀 Training and Evaluation

### 1\. Model Training

Use the `training/train.py` script to train the final model or any of the ablation study architectures.

```bash
# To train the main ConvLSTM-GCN-Transformer model
python training/train.py --architecture ConvLstm_Gcn_Transformer

# To train an ablation model (e.g., ConvLSTM-GCN)
python training/train.py --architecture ConvLstm_Gcn
```

### 2\. Model Evaluation (Test)

The `training/test.py` script handles the evaluation, calculating metrics (RMSE, MAE, NSE) and generating prediction maps on the test set.

```bash
# To evaluate a trained model
python training/test.py --model_path /path/to/your/saved_model.h5
```

## 🖼️ Architecture Breakdown

The **ConvLSTM-GCN-Transformer** is implemented in `architectures/ConvLstm_Gcn_Transformer.py` and `utils/implementation_functions.py` (which defines the custom GCN layer and sparse adjacency).

The core flow is:

1.  **Input Sequence**: $(Batch \times T \times H \times W \times 1)$.
2.  **ConvLSTM2D**: Extracts spatio-temporal features, outputting a tensor $(B \times H \times W \times C)$.
3.  **Reshape**: Flattens the image features into node representations $(B \times N \times C)$, where $N = H \times W$.
4.  **GraphConvLayer (GCN)**: Applies graph convolution over the grid-based sparse adjacency matrix (4-neighbors) for local spatial smoothing and context.
5.  **TransformerBlock**: Applies global self-attention across all $N$ nodes (pixels) to capture long-range dependencies.
6.  **Reshape**: Reshapes the node features back into the image structure $(B \times H \times W \times D)$.
7.  **Final Conv2D**: A standard 2D Convolutional layer with `sigmoid` activation produces the predicted VI map $\hat{Y} \in \mathbb{R}^{H\times W\times1}$.

## 📈 Results Summary

The architecture significantly outperformed baselines across various metrics and generalization tests.

| Metric | ConvLSTM-GCN-Transformer (Landsat NDVI) | 
| :--- | :--- | 
| **RMSE** | 0.034 |
| **NSE** | 0.866 |
| **MAE** | 0.015 | 
| **SSIM** |  0.863 |
| **Moran's I** | 0.094 |
| 

For a complete analysis of the ablation study and cross-sensor/cross-index generalization results (Sentinel-2, MODIS, EVI, SAVI), please refer to the accompanying paper.

## 📚 Citation

If you use this model, the code, or the methodology in your research, please cite the corresponding paper:

```
@article{votre_nom_2025_convlstm,
  title={ConvLSTM-GCN-Transformer: Spatiotemporal Graph-Attention Model for Multisource Vegetation Index Map Forecasting},
  author={},
  journal={TBD},
  year={2025},
  url={}
}
```

## ✉️ Contact

For questions, issues, or collaborations related to this project, please contact:

**[Mohamed Chahine BOUAZIZ]**

  * **Email:** [bouazizchahine7@gmail.com]
  

---
