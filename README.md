````markdown
# ConvLSTM-GCN-Transformer: Spatiotemporal Graph-Attention Model for Vegetation Index Map Forecasting

[![License](https://img.shields.io/github/license/google/gts/master)](https://github.com/your-username/your-repo/LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/your-username/your-repo?style=social)](https://github.com/your-username/your-repo)

## 🌟 Introduction and Research Context

This repository hosts the source code for the **ConvLSTM-GCN-Transformer** architecture, a novel hybrid Deep Learning model designed for **high-resolution spatial forecasting of Vegetation Index (VI) maps** from satellite image sequences.

The work aims to address the limitations of conventional deep learning methods by simultaneously modeling:
1.  The **temporal dynamics** of image time series (using ConvLSTM).
2.  **Local spatial dependencies** (pixel-to-pixel neighborhood effects via GCN).
3.  The **global scene coherence** and long-range feature representation (via Transformer's attention mechanism).

This research is detailed in the accompanying paper: **"ConvLSTM-GCN-Transformer: Spatiotemporal Graph-Attention Model for Multisource Vegetation Index Map Forecasting."**

## 🧠 Model Architecture (ConvLSTM-GCN-T)

The proposed model integrates three distinct components, specialized for spatiotemporal analysis of satellite data:

1.  **ConvLSTM (Temporal Encoding):** The input sequence of VI maps is processed by a `ConvLSTM2D` layer to extract compact spatiotemporal features, capturing historical variations while preserving spatial locality.
    * *Output Shape Example:* `(Batch, H, W, 64)`

2.  **GCN (Local Spatial Encoding):**
    * The features are reshaped from image format `(H, W, F)` to graph node format `(N, F)`, where $N=H \times W$ (total pixels/nodes).
    * A custom **Graph Convolutional Layer** is applied to explicitly model local dependencies. This implementation leverages a **Sparse Adjacency Matrix** based on a **4-neighborhood graph**, connecting each pixel to its immediate North, South, East, and West neighbors. This step is crucial for maintaining spatial consistency.

3.  **Transformer Encoder (Global Attention):**
    * A **Multi-Head Self-Attention** mechanism follows the GCN layer. It captures global, long-range relationships between all nodes (pixels) across the entire image/scene, thus enriching the feature representation and ensuring scene-level coherence, overcoming the locality constraint of ConvLSTM and GCN.

![Architectural Diagram](assets/CONVlstm_gcn_TRANSFORMER.PNG)

## 📁 Repository Structure

For maximal clarity and adherence to standard practices, the project is organized as follows:

| Folder/File | Description | Your Current Content |
| :--- | :--- | :--- |
| **`assets/`** | Contains documentation resources, figures, and diagrams. | `CONVlstm_gcn_TRANSFORMER.PNG` |
| **`data_tools/`** | Standalone scripts for raw data acquisition from Google Earth Engine (GEE). | `gee_landsat_download.py`, `gee_sentinel2_download.py`, `gee_modis_download.py` |
| **`src/`** | Core source code for models, utilities, and data processing. | |
| **`src/models/`** | Definitions for all architectures, including the final model and ablation variants. | `convlstm_gcn_transformer.py` (Final Model), `ablation_convlstm.py`, `ablation_gcn_transformer.py`, etc. |
| **`src/data_prep/`** | Code for preprocessing and formatting data into spatiotemporal sequences. | `sequences_creation.py` |
| **`src/utils.py`** | Generic utility functions for data handling, plotting, or model operations. | `implementation_function.py` (Integrated/Renamed) |
| **`scripts/`** | High-level scripts to execute the main workflows (training, testing). | `train.py`, `test.py` |
| **`requirements.txt`** | List of Python dependencies. | `requirements.txt` |
| **`README.md`** | Project documentation (this file). | |

## 🛠️ Prerequisites and Setup

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo.git](https://github.com/your-username/your-repo.git)
    cd your-repo
    ```

2.  **Install Dependencies:**
    This project requires Python 3.x and the packages listed. Key dependencies include `tensorflow`, `numpy`, and the `earthengine-api`.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Google Earth Engine (GEE) Configuration:**
    You **must** install and authenticate the GEE API to run the data collection scripts:
    ```bash
    pip install earthengine-api
    earthengine authenticate
    ```

## 🌍 Detailed Data Acquisition Guide (Google Earth Engine)

The scripts in the `data_tools/` folder are responsible for the automated downloading of satellite images (Landsat, Sentinel-2, MODIS) and the calculation of the desired Vegetation Index (VI).

### ⚠️ Essential Configuration Steps (User Customization)

Before running any script in `data_tools/`, you **MUST** update the following parameters within the individual files (e.g., `data_tools/gee_landsat_download.py`) to reflect your study area and data needs:

#### 1. Define Area of Interest (AOI) and Temporal Range

* **Coordinates/Geometry:** Update the latitude/longitude or define a new geometry object to precisely delineate your **Area of Interest (AOI)**.
* **Temporal Range:** Specify the exact start and end dates for your time series:
    ```python
    START_DATE = 'YYYY-MM-DD'  # e.g., '2000-01-01'
    END_DATE = 'YYYY-MM-DD'    # e.g., '2020-12-31'
    ```

#### 2. Specify Your Cloud Export Location

* You need to provide the path to your own Google Cloud Storage or Earth Engine Asset folder where the exported data will be saved:
    ```python
    # Example: Path to your Google Cloud Storage Bucket or GEE Asset
    EXPORT_FOLDER = 'users/your_gee_username/VI_Forecast_Data' 
    ```

#### 3. Customize the Vegetation Index (VI) Equation

* By default, the scripts are configured to calculate the **NDVI** (Normalized Difference Vegetation Index).
* If your study requires a different index (e.g., EVI, NDWI, LAI), you **must modify the equation** within the dedicated index calculation function (e.g., `calculate_index(image)`).
* **The user only needs to change the spectral band equation.**

| Index | Generic Equation | Bands (Sentinel-2 Example) |
| :--- | :--- | :--- |
| **NDVI (Default)** | $(NIR - RED) / (NIR + RED)$ | `(image.select('B8').subtract(image.select('B4'))).divide(...)` |
| **EVI (Example)** | $2.5 \times \frac{(NIR - RED)}{(NIR + 6 \times RED - 7.5 \times BLUE + 1)}$ | Requires adapting to specific bands (B8, B4, B2) and GEE syntax. |

### Execution of Data Download

Execute the scripts one by one to queue the data export tasks in GEE:

```bash
# Download Landsat data (make sure GEE tasks finish before proceeding)
python data_tools/gee_landsat_download.py

# Download Sentinel-2 data (if multi-source approach is used)
python data_tools/gee_sentinel2_download.py 
````

## 🎞️ Data Preparation and Sequencing

Once the VI maps are downloaded and stored, the `src/data_prep/sequences_creation.py` script is used to format the data into the required spatiotemporal sequences $(X, Y)$ for model training:

  * $X$: A sequence of $T$ VI map images (historical time steps).
  * $Y$: The VI map of the time step $T+1$ (the target to be predicted).

<!-- end list -->

```bash
python src/data_prep/sequences_creation.py
```

## 🚀 Training and Evaluation

### Training the Final Model

The `scripts/train.py` script loads the prepared sequences and initiates the training process for the `ConvLSTM-GCN-Transformer` model.

```bash
python scripts/train.py
```

### Evaluation and Testing

Use `scripts/test.py` to load the trained model weights and evaluate its performance against the test set, reproducing key metrics (RMSE, MAE, NSE, etc.) from the paper.

```bash
python scripts/test.py
```

## 🧪 Ablation Study Architectures

The `src/models/` directory includes the implementations of all baseline and intermediate models used in the paper's ablation study, allowing for direct comparison and verification of the architectural contributions.

| File Name | Architecture | Rationale in Ablation Study |
| :--- | :--- | :--- |
| `ablation_convlstm.py` | ConvLSTM Only | Baseline for temporal processing. |
| `ablation_convlstm_3dcnn.py` | ConvLSTM + 3D-CNN | Comparison of GCN vs. traditional 3D convolutions for spatiotemporal feature extraction. |
| `ablation_convlstm_gcn.py` | ConvLSTM + GCN | Assesses the contribution of the Graph Convolutional component. |
| `ablation_convlstm_transformer.py` | ConvLSTM + Transformer | Assesses the contribution of the global self-attention mechanism. |
| `convlstm_gcn_transformer.py` | **Final Model** | Full integrated architecture. |

## 📖 Citation

If you find this code or methodology useful in your research, please cite the corresponding paper:

```
@article{VotreNom2024ConvLSTMGCNTransformer,
  title={ConvLSTM-GCN-Transformer: Spatiotemporal Graph-Attention Model for Multisource Vegetation Index Map Forecasting},
  author={Votre Nom, Autres Co-Auteurs},
  journal={Nom du Journal Scientifique},
  volume={X},
  number={Y},
  pages={Z},
  year={2024}
}
```

```
```
