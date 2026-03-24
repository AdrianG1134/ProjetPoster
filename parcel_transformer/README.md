# Parcel Temporal Transformer

Classification multiclasses de cultures par parcelle a partir de series temporelles Sentinel-2 deja pretraitees (format CSV long).

## Fichiers
- `config.py`: configuration centralisee.
- `data.py`: chargement CSV long, pivot en tenseur `[N, T, F]`, masque temporel, split `parcel` ou `tile`.
- `model.py`: Transformer Encoder temporel avec encodage jour de l'annee.
- `train.py`: entrainement PyTorch complet (AdamW, early stopping, meilleur checkpoint).
- `evaluate.py`: metriques, matrice de confusion, rapport de classification, attention temporelle moyenne.
- `evaluate_ensemble.py`: evaluation par moyenne des logits de plusieurs checkpoints.
- `prepare_dataset.py`: conversion CSV long -> dataset `.npz` pret pour PyTorch.
- `build_training_csv.py`: fusion `indices_parcelles*.csv` + labels RPG pour produire le CSV long attendu.

## Installation
```bash
pip install -r parcel_transformer/requirements.txt
```

## Entrainement direct depuis CSV long
```bash
python parcel_transformer/train.py --csv-path data/mon_csv_long.csv --output-dir outputs_transformer
```

## Construire le CSV long d'entrainement depuis tes fichiers actuels
```bash
python parcel_transformer/build_training_csv.py --indices-csv data/s2_herault_2024_full_year_5day_cloudmask_fast/indices_parcelles_2024-01-01_2024-12-31_win5d.csv --labels-csv export.csv --label-col CODE_CULTU --output-csv data/s2_herault_2024_full_year_5day_cloudmask_fast/indices_parcelles_2024-01-01_2024-12-31_win5d_with_labels.csv
```

## Construire le CSV avec niveau hierarchique (groupe + culture)
```bash
python parcel_transformer/build_training_csv.py --indices-csv data/s2_herault_2024_full_year_5day_cloudmask_fast/indices_parcelles_2024-01-01_2024-12-31_win5d.csv --labels-csv export.csv --label-col CODE_CULTU --group-col CODE_GROUP --output-csv data/s2_herault_2024_full_year_5day_cloudmask_fast/indices_parcelles_2024-01-01_2024-12-31_win5d_with_labels_and_group.csv
```

## Entrainement avec split spatial par tuile
```bash
python parcel_transformer/train.py --csv-path data/mon_csv_long.csv --split-method tile
```

## Grille temporelle reguliere (binning)
```bash
python parcel_transformer/train.py --csv-path data/mon_csv_long.csv --split-method tile --time-grid-frequency 5D
```
`time-grid-frequency` applique un binning temporel (moyenne des observations dans chaque bin), puis construit la sequence reguliere.

## Entrainement Focal Loss + ponderation adoucie
```bash
python parcel_transformer/train.py --csv-path data/mon_csv_long.csv --split-method tile --loss-type focal --focal-gamma 2.0 --class-weighting --class-weight-power 0.5
```

## Alternatives de loss long-tail
Balanced Softmax:
```bash
python parcel_transformer/train.py --csv-path data/mon_csv_long.csv --split-method tile --loss-type balanced_softmax --class-weighting --class-weight-power 0.5
```

Logit-Adjusted CrossEntropy:
```bash
python parcel_transformer/train.py --csv-path data/mon_csv_long.csv --split-method tile --loss-type logit_adjusted --logit-adjust-tau 1.0 --class-weighting --class-weight-power 0.5
```

## Options avancees (normalisation + sampler)
```bash
python parcel_transformer/train.py --csv-path data/mon_csv_long.csv --split-method tile --loss-type focal --focal-gamma 1.5 --class-weighting --class-weight-power 0.5 --standardize-features --weighted-sampler --sampler-power 1.0
```

## Augmentations temporelles legeres (train uniquement)
```bash
python parcel_transformer/train.py --csv-path data/mon_csv_long.csv --split-method tile --loss-type focal --focal-gamma 1.5 --class-weighting --class-weight-power 0.5 --standardize-features --temporal-augmentation --time-mask-ratio 0.05 --jitter-std 0.01
```

## Fine-tuning phase 2 cible classes rares
Phase 1 classique, puis phase 2 (LR bas) avec sampler pondere pour sur-echantillonner les classes rares.
```bash
python parcel_transformer/train.py --csv-path data/mon_csv_long.csv --split-method tile --loss-type focal --focal-gamma 1.5 --class-weighting --class-weight-power 0.5 --standardize-features --no-weighted-sampler --phase2-rare-finetune --phase2-epochs 12 --phase2-lr 1e-4 --phase2-sampler-power 1.0 --phase2-rare-quantile 0.25 --phase2-rare-boost 2.0
```

## Entrainement multi-tache hierarchique (bonus)
```bash
python parcel_transformer/train.py --csv-path data/mon_csv_long.csv --split-method tile --loss-type focal --focal-gamma 1.5 --class-weighting --class-weight-power 0.5 --standardize-features --use-group-task --group-loss-weight 0.3
```
Le CSV doit contenir une colonne `label_group` (generee via `build_training_csv.py --group-col ...`).

## Preparation dataset NPZ (bonus)
```bash
python parcel_transformer/prepare_dataset.py --csv-path data/mon_csv_long.csv --output-npz data/parcel_dataset.npz
```

## Entrainement depuis NPZ prepare
```bash
python parcel_transformer/train.py --prepared-npz data/parcel_dataset.npz --output-dir outputs_transformer
```

## Reevaluation d'un checkpoint
```bash
python parcel_transformer/evaluate.py --checkpoint outputs_transformer/temporal_transformer_YYYYMMDD_HHMMSS/best_model.pt --prepared-npz data/parcel_dataset.npz
```

## Evaluation d'un ensemble de checkpoints
```bash
python parcel_transformer/evaluate_ensemble.py --checkpoint-glob "outputs_transformer/seeds_std_focal_s*/temporal_transformer_*/best_model.pt" --csv-path data/s2_herault_2024_full_year_5day_cloudmask_fast/indices_parcelles_2024-01-01_2024-12-31_win5d_with_labels_min200.csv --split-method tile --index-filter NDVI,NDMI,NDWI,EVI --output-dir outputs_transformer/ensemble_eval
```

## Distillation d'un ensemble vers un seul modele
```bash
python parcel_transformer/distill_ensemble.py --teacher-checkpoint-glob "outputs_transformer/phase2_seeds_ensemble/s*/temporal_transformer_*/best_model.pt" --csv-path data/s2_herault_2024_full_year_5day_cloudmask_fast/indices_parcelles_2024-01-01_2024-12-31_win5d_with_labels_and_group_min200.csv --split-method tile --index-filter NDVI,NDMI,NDWI,EVI --standardize-features --class-weighting --class-weight-power 0.5 --epochs 60 --batch-size 64 --lr 7e-4 --hard-label-weight 0.4 --temperature 2.0 --output-dir outputs_transformer/distill_phase2_ensemble
```

## Mini sweep automatique des losses
```bash
python parcel_transformer/sweep_loss_strategies.py --csv-path data/s2_herault_2024_full_year_5day_cloudmask_fast/indices_parcelles_2024-01-01_2024-12-31_win5d_with_labels_and_group_min200.csv --split-method tile --index-filter NDVI,NDMI,NDWI,EVI --seed 42
```

## CV spatiale robuste (GroupKFold par tile)
```bash
python parcel_transformer/spatial_cv_groupkfold.py --csv-path data/mon_csv_long.csv --index-filter NDVI,NDMI,NDWI,EVI --n-splits 5 --val-size 0.1 --seed 42 --train-extra-args "--loss-type focal --focal-gamma 1.5 --class-weighting --class-weight-power 0.5 --standardize-features --no-use-group-task"
```

## Lancement automatise (augmentation + phase 2 + CV spatiale)
```powershell
powershell -ExecutionPolicy Bypass -File parcel_transformer/run_rare_aug_cv_experiments.ps1
```

## Lancement automatise (phase 2 + seeds + ensemble)
```powershell
powershell -ExecutionPolicy Bypass -File parcel_transformer/run_phase2_seeds_ensemble.ps1
```
