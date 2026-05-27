# Crop Detection from Satellite Time Series

[English Version](#english-version) | [Version française](#version-francaise)

<a id="english-version"></a>
## English Version

### Overview
This project explores crop type classification from satellite imagery using parcel-level time series of spectral indices.  
It combines geospatial preprocessing (Sentinel-2 extraction and zonal statistics) with machine learning and deep learning pipelines.  
The repository includes experiments with Random Forest, XGBoost, LSTM, and Transformer-based models for multivariate time series classification.  
The goal is to learn temporal vegetation patterns and map them to crop classes at parcel level.

### Objective
Build and compare classification pipelines that detect crop types from temporal spectral signatures.  
The core idea is to use satellite-derived indices over time to capture vegetation dynamics during a season/year.

### Context
Crop classification from remote sensing is useful for:
- agricultural monitoring,
- land-use and territorial analysis,
- environmental assessment,
- scalable large-area mapping when field observations are limited.

### Data
The code expects parcel-level spectral-index time series derived from Sentinel-2, plus parcel labels.

Main data-related assets and scripts in this repository:
- `dlSentinel.py`: downloads/filters Sentinel-2 scenes (Planetary Computer STAC), computes spectral indices, and aggregates parcel statistics.
- `Parcelles-Herault.gpkg`: parcel geometries used for zonal extraction.
- `export.csv`: parcel labels (`CODE_CULTU`) used for supervised training.
- `data/s2_herault_2024_full/indices_parcelles_2024-01-01_2024-12-31_win5d.csv`: long-format parcel index time series used by ML/DL scripts (local generated artifact).
- `lstm_data/*.npy`: prepared tensor inputs for the LSTM pipeline.

The raw dataset is not fully included in this repository. The repository focuses on preprocessing, modeling and experimentation code.  
Large generated artifacts under `data/` are not versioned by default, so a clean clone may require regenerating them.

### Methodology
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

### Models
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

### Repository Structure
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

### How to Run
Environment setup:

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

Example execution flow:
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

Notes:
- `requirements.txt` covers geospatial + classic ML dependencies.
- Transformer dependencies are in `parcel_transformer/requirements.txt`.
- TensorFlow (for LSTM) is imported in `train_lstm.py` and is not listed in the root requirements file.
- Because the full generated dataset under `data/` is not version-controlled, full reproducibility from a fresh clone may require data regeneration.

### Results
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

Important: these runs were produced with different experiment settings (class filtering, feature sets, pipeline variants), so they are not a strict like-for-like benchmark.

### Technical Skills Demonstrated
- Python
- machine learning
- deep learning
- Transformer architectures
- time series classification
- satellite imagery processing
- remote sensing
- spectral indices engineering
- data preprocessing
- model evaluation
- geospatial processing (raster/vector)
- PyTorch and TensorFlow workflows

### Limitations and Future Work
- Consolidate dependencies into a single reproducible environment file (`requirements.txt` + optional `environment.yml`).
- Improve reproducibility with a unified CLI and documented end-to-end pipeline.
- Add a strict model comparison table (same split, same class subset, same feature scope).
- Add lightweight sample/synthetic data for quick functional tests.
- Add experiment tracking (for example MLflow/W&B) and versioned configuration snapshots.
- Provide a dedicated inference script for batch prediction on new parcel time series.

---

<a id="version-francaise"></a>
## Version française

### Vue d'ensemble
Ce projet explore la classification des types de cultures à partir d'images satellites, en utilisant des séries temporelles d'indices spectraux à l'échelle parcellaire.  
Il combine un prétraitement géospatial (extraction Sentinel-2 et statistiques zonales) avec des pipelines de machine learning et de deep learning.  
Le dépôt contient des expérimentations avec Random Forest, XGBoost, LSTM et des modèles de type Transformer pour la classification de séries temporelles multivariées.  
L'objectif est d'apprendre des signatures temporelles de végétation pour les relier à des classes de cultures.

### Objectif
Construire et comparer des pipelines de classification capables d'identifier les types de cultures à partir de signatures spectrales temporelles.  
L'idée centrale est d'utiliser des indices dérivés des satellites au fil du temps pour capturer la dynamique de la végétation sur une saison/année.

### Contexte
La classification des cultures par télédétection est utile pour:
- le suivi agricole,
- l'analyse de l'occupation des sols et des territoires,
- l'évaluation environnementale,
- la cartographie à grande échelle lorsque les observations de terrain sont limitées.

### Données
Le code attend des séries temporelles d'indices spectraux à l'échelle parcellaire dérivées de Sentinel-2, ainsi que des labels de parcelles.

Principaux scripts et artefacts liés aux données dans ce dépôt:
- `dlSentinel.py`: télécharge/filtre les scènes Sentinel-2 (STAC Planetary Computer), calcule les indices spectraux et agrège les statistiques par parcelle.
- `Parcelles-Herault.gpkg`: géométries de parcelles utilisées pour l'extraction zonale.
- `export.csv`: labels de parcelles (`CODE_CULTU`) utilisés pour l'entraînement supervisé.
- `data/s2_herault_2024_full/indices_parcelles_2024-01-01_2024-12-31_win5d.csv`: séries temporelles d'indices au format long utilisées par les scripts ML/DL (artefact généré localement).
- `lstm_data/*.npy`: entrées tenseur préparées pour le pipeline LSTM.

Le jeu de données brut n'est pas entièrement inclus dans ce dépôt. Le dépôt se concentre sur le code de prétraitement, de modélisation et d'expérimentation.  
Les gros artefacts générés sous `data/` ne sont pas versionnés par défaut, donc un clone propre peut nécessiter leur régénération.

### Méthodologie
Pipeline implémenté dans le dépôt:

1. Chargement des données  
   Chargement des géométries de parcelles et des labels de cultures.
2. Traitement satellite  
   Requête des scènes Sentinel-2 L2A, filtrage par couverture nuageuse et sélection d'observations par fenêtres temporelles.
3. Calcul d'indices spectraux  
   Calcul d'indices tels que NDVI, NDMI, NDWI, EVI (et d'autres indices dans le pipeline Transformer).
4. Agrégation par parcelle  
   Conversion des rasters en statistiques par parcelle (moyennes, nombre de pixels) par date et par indice.
5. Prétraitement des séries temporelles  
   Construction de features tabulaires ou de tenseurs `[N, T, F]`, interpolation/imputation, filtrage des classes rares.
6. Entraînement des modèles  
   Entraînement de modèles ML et DL avec split stratifié ou spatial (parcelle/tuile selon le pipeline).
7. Évaluation des modèles  
   Export des métriques, rapports de classification, matrices de confusion, historique d'entraînement et artefacts d'importance/attention.

### Modèles
Modèles actuellement présents dans le code:

- Random Forest (scikit-learn)  
  - `train_randomforest.py`  
  - `RF_tempfeatures.py`
- XGBoost multiclasses  
  - `train_xgboost.py`
- LSTM (TensorFlow / Keras)  
  - `lstm_data.py` (préparation des tenseurs)  
  - `train_lstm.py`
- Transformer temporel (PyTorch)  
  - `parcel_transformer/train.py`, `model.py`, `evaluate.py`  
  - Inclut des expériences avancées: encodage reliability-aware, pré-entraînement SSL (`pretrain_ssl.py`), évaluation d'ensemble (`evaluate_ensemble.py`) et distillation (`distill_ensemble.py`).

### Structure du dépôt
```text
ProjetPoster/
  data/
    s2_herault_2024_full/                    # séries temporelles CSV longues générées localement
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
  outputs_random_forest/                     # métriques/rapports/importances RF
  outputs_xgboost/                           # métriques/rapports/matrice confusion/importances XGBoost
  outputs_lstm/                              # métriques/historique/modèle LSTM
  transformer_output/                        # artefacts d'un run Transformer (métriques, rapports, graphiques)
  outputs_transformer/                       # runs Transformer multiples
  dlSentinel.py                              # pipeline Sentinel-2 -> extraction d'indices par parcelle
  train_randomforest.py
  train_xgboost.py
  lstm_data.py
  train_lstm.py
  requirements.txt
```

### Lancer le projet
Mise en place de l'environnement:

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

Installation des dépendances:
```bash
pip install -r requirements.txt
pip install scipy matplotlib tensorflow
pip install -r parcel_transformer/requirements.txt
```

Exemple de flux d'exécution:
```bash
# Construction des tenseurs LSTM
python lstm_data.py

# Entraînement des modèles de base
python train_randomforest.py
python train_xgboost.py
python train_lstm.py

# Entraînement du Transformer (depuis un NPZ préparé, si disponible)
python parcel_transformer/train.py --prepared-npz data/parcel_dataset_ext.npz --output-dir outputs_transformer
```

Notes:
- `requirements.txt` couvre les dépendances géospatiales et de ML classique.
- Les dépendances Transformer sont dans `parcel_transformer/requirements.txt`.
- TensorFlow (pour LSTM) est importé dans `train_lstm.py` mais non listé dans les dépendances racine.
- Comme le dataset généré sous `data/` n'est pas versionné, la reproductibilité complète depuis un clone propre peut nécessiter une régénération des données.

### Résultats
Les artefacts d'évaluation sont disponibles dans:
- `outputs_random_forest/metrics_rf.json`
- `outputs_xgboost/metrics.json`
- `outputs_lstm/metrics_lstm.json`
- `transformer_output/test_metrics.json`

Aperçu des métriques disponibles:

| Modèle | Fichier de métriques | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|
| Random Forest | `outputs_random_forest/metrics_rf.json` | 0.6458 | 0.3421 | 0.6178 |
| XGBoost | `outputs_xgboost/metrics.json` | 0.6573 | 0.4632 | 0.6663 |
| LSTM | `outputs_lstm/metrics_lstm.json` | 0.4215 | 0.2633 | 0.4763 |
| Transformer | `transformer_output/test_metrics.json` | 0.6331 | 0.3380 | 0.6368 |

Important: ces runs ont été produits avec des configurations expérimentales différentes (filtrage de classes, jeux de features, variantes de pipeline), donc ce n'est pas un benchmark strictement comparable.

### Compétences techniques démontrées
- Python
- machine learning
- deep learning
- architectures Transformer
- classification de séries temporelles
- traitement d'images satellites
- télédétection
- ingénierie d'indices spectraux
- prétraitement de données
- évaluation de modèles
- traitement géospatial (raster/vector)
- workflows PyTorch et TensorFlow

### Limites et améliorations futures
- Consolider les dépendances dans un environnement reproductible unique (`requirements.txt` + éventuellement `environment.yml`).
- Améliorer la reproductibilité avec une CLI unifiée et un pipeline documenté de bout en bout.
- Ajouter un tableau de comparaison strict des modèles (même split, même sous-ensemble de classes, même périmètre de features).
- Ajouter un jeu de données échantillon/synthétique pour des tests rapides.
- Ajouter un suivi d'expériences (par exemple MLflow/W&B) et des snapshots de configuration versionnés.
- Fournir un script d'inférence dédié pour la prédiction batch sur de nouvelles séries temporelles parcellaires.
