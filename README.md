# Crop Detection from Satellite Time Series

## Overview
This project explores crop type classification from satellite imagery using parcel-level time series of spectral indices.  
It combines geospatial preprocessing (Sentinel-2 extraction and zonal statistics) with machine learning and deep learning pipelines.  
The repository includes experiments with Random Forest, XGBoost, LSTM, and Transformer-based models for multivariate time series classification.  
The goal is to learn temporal vegetation patterns and map them to crop classes at parcel level.

## Objective
Build and compare classification pipelines that detect crop types from temporal spectral signatures.  
The core idea is to use satellite-derived indices across time to capture vegetation dynamics during a full season/year.

## Context
Crop classification from remote sensing is useful for:
- agricultural monitoring,
- land-use and territorial analysis,
- environmental assessment,
- scalable large-area mapping when field observations are limited.

## Data
The code expects parcel-level spectral-index time series derived from Sentinel-2, plus parcel labels.

Main data-related assets and scripts in this repository:
- `dlSentinel.py`: downloads/filters Sentinel-2 scenes (Planetary Computer STAC), computes spectral indices, and aggregates parcel statistics.
- `Parcelles-Herault.gpkg`: parcel geometries used for zonal extraction.
- `export.csv`: parcel labels (`CODE_CULTU`) used for supervised training.
- `data/s2_herault_2024_full/indices_parcelles_2024-01-01_2024-12-31_win5d.csv`: long-format parcel index time series used by ML/DL scripts (local generated artifact).
- `lstm_data/*.npy`: prepared tensor inputs for the LSTM pipeline.

The raw dataset is not fully included in this repository. The repository focuses on preprocessing, modeling and experimentation code.  
Large generated artifacts under `data/` are not versioned by default, so a clean clone may require regenerating them.

## Methodology
Pipeline implemented across the repository:

1. Data loading  
   Load parcel geometry and crop labels.
2. Satellite processing  
   Query Sentinel-2 L2A scenes, filter by cloud cover, and select observations in fixed temporal windows.
3. Spectral index computation  
   Compute indices such as NDVI, NDMI, NDWI, EVI (and additional indices in the Transformer pipeline).
4. Parcel-level aggregation  
   Convert raster data into parcel-level statistics (mean values, pixel counts) per date and index.
5. Time series preprocessing  
   Build wide tabular features or `[N, T, F]` tensors, apply interpolation/imputation, and filter rare classes.
6. Model training  
   Train ML and DL models with stratified or spatially-aware splitting (parcel/tile depending on pipeline).
7. Model evaluation  
   Export metrics, classification reports, confusion matrices, training history, and feature importance/attention artifacts.

## Models
Models currently present in code:

- Random Forest (scikit-learn)  
  - `train_randomforest.py`  
  - `RF_tempfeatures.py`
- XGBoost multiclass classification  
  - `train_xgboost.py`
- LSTM (TensorFlow / Keras)  
  - `lstm_data.py` (tensor preparation)  
  - `train_lstm.py`
- Temporal Transformer (PyTorch)  
  - `parcel_transformer/train.py`, `model.py`, `evaluate.py`  
  - Includes advanced experiments: reliability-aware encoding, SSL pretraining (`pretrain_ssl.py`), ensemble evaluation (`evaluate_ensemble.py`), and distillation (`distill_ensemble.py`).

## Repository Structure
```text
ProjetPoster/
  data/
    s2_herault_2024_full/                    # local generated long CSV time series
  parcel_transformer/
    config.py
    data.py
    model.py
    train.py
    evaluate.py
    pretrain_ssl.py
    evaluate_ensemble.py
    distill_ensemble.py
    prepare_dataset.py
    build_training_csv.py
    README.md
  outputs_random_forest/                     # RF metrics/reports/feature importance
  outputs_xgboost/                           # XGBoost metrics/reports/confusion matrix/importance
  outputs_lstm/                              # LSTM metrics/history/model
  transformer_output/                        # Transformer run artifacts (metrics, reports, plots)
  outputs_transformer/                       # multiple Transformer experiment runs
  dlSentinel.py                              # Sentinel-2 -> parcel index extraction pipeline
  train_randomforest.py
  train_xgboost.py
  lstm_data.py
  train_lstm.py
  requirements.txt
```

## How to Run
### 1) Environment setup
```bash
python -m venv .venv
```

Windows (PowerShell):
```bash
.venv\Scripts\activate
```

Linux/macOS:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
pip install scipy matplotlib tensorflow
pip install -r parcel_transformer/requirements.txt
```

Notes:
- `requirements.txt` covers geospatial + classic ML dependencies.
- Transformer dependencies are in `parcel_transformer/requirements.txt`.
- TensorFlow (for LSTM) is imported in `train_lstm.py` and is not listed in the root requirements file.

### 2) Example execution flow
```bash
# Build LSTM tensors
python lstm_data.py

# Train baseline models
python train_randomforest.py
python train_xgboost.py
python train_lstm.py

# Train Transformer (from prepared NPZ, if available)
python parcel_transformer/train.py --prepared-npz data/parcel_dataset_ext.npz --output-dir outputs_transformer
```

Because the full generated dataset under `data/` is not version-controlled, the full training pipeline may not be reproducible directly from a fresh clone without regenerating data.

## Results
Evaluation artifacts are available in:
- `outputs_random_forest/metrics_rf.json`
- `outputs_xgboost/metrics.json`
- `outputs_lstm/metrics_lstm.json`
- `transformer_output/test_metrics.json`

Snapshot of available metrics files:

| Model | Metrics file | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|
| Random Forest | `outputs_random_forest/metrics_rf.json` | 0.6458 | 0.3421 | 0.6178 |
| XGBoost | `outputs_xgboost/metrics.json` | 0.6573 | 0.4632 | 0.6663 |
| LSTM | `outputs_lstm/metrics_lstm.json` | 0.4215 | 0.2633 | 0.4763 |
| Transformer | `transformer_output/test_metrics.json` | 0.6331 | 0.3380 | 0.6368 |

Important: these runs are produced under different experiment settings (for example class filtering, feature sets, and pipeline variants), so they should not be interpreted as a strict like-for-like benchmark.

## Technical Skills Demonstrated
- Python
- geospatial data processing (raster/vector) for remote sensing
- spectral indices engineering from satellite imagery
- time series preprocessing and tensor construction
- machine learning (Random Forest, XGBoost)
- deep learning (LSTM, Transformer)
- PyTorch and TensorFlow pipelines
- model evaluation (accuracy, macro/weighted F1, confusion matrices, reports)
- data preprocessing and feature engineering
- experiment scripting and output artifact management

## Limitations and Future Work
- Consolidate dependencies into a single reproducible environment file (`requirements.txt` + optional `environment.yml`).
- Improve run reproducibility with a unified CLI and documented end-to-end pipeline.
- Add an explicit experiment comparison table (same split, same class subset) for fair model benchmarking.
- Add lightweight sample/synthetic data for quick functional testing without full satellite processing.
- Add experiment tracking (for example MLflow/W&B) and versioned configuration snapshots.
- Provide a dedicated inference script for batch prediction on new parcel time series.

## French Summary
Ce projet porte sur la detection/classification de cultures agricoles a partir de series temporelles d'indices spectraux extraits d'images satellites.  
Il explore plusieurs approches de machine learning et de deep learning, notamment des architectures de type Transformer, pour analyser l'evolution temporelle de la vegetation.
