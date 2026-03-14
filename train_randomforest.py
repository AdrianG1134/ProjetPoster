#!/usr/bin/env python
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


# =========================
# CONFIG
# =========================
INDICES_CSV = "data/s2_herault_2024_full/indices_parcelles_2024-01-01_2024-12-31_win5d.csv"
LABELS_CSV = "export.csv"
ID_COL = "ID_PARCEL"
TARGET_COL = "CODE_CULTU"

INDICES_KEEP = {"NDVI", "EVI", "NDMI", "NDWI"}
CLOUD_MAX = 40.0
PX_COUNT_MIN = 10
MIN_CLASS_COUNT = 50   # commence à 50, puis tu pourras tester 100 ou 200

TEST_SIZE = 0.2
RANDOM_STATE = 42

OUTPUT_DIR = Path("outputs_random_forest")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 1) Charger les labels
# =========================
def load_labels(path: str, id_col: str, target_col: str) -> pd.DataFrame:
    labels = pd.read_csv(path, usecols=[id_col, target_col])
    labels = labels.dropna(subset=[target_col]).copy()
    labels[id_col] = labels[id_col].astype(str)
    labels[target_col] = labels[target_col].astype(str)
    labels = labels.drop_duplicates(subset=[id_col])
    return labels


# =========================
# 2) Agréger les indices
# =========================
def aggregate_indices(
    path: str,
    id_set: set,
    indices_keep: set,
    cloud_max: float,
    px_count_min: int,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    cols = ["date", "ID_PARCEL", "index", "value_mean", "px_count", "cloud_scene"]
    agg_chunks = []

    for chunk in pd.read_csv(path, usecols=cols, chunksize=chunksize):
        chunk = chunk.dropna(subset=["date", "ID_PARCEL", "index", "value_mean", "px_count", "cloud_scene"])
        chunk["ID_PARCEL"] = chunk["ID_PARCEL"].astype(str)

        chunk = chunk[chunk["ID_PARCEL"].isin(id_set)]
        chunk = chunk[chunk["index"].isin(indices_keep)]
        chunk = chunk[chunk["cloud_scene"] <= cloud_max]
        chunk = chunk[chunk["px_count"] >= px_count_min]

        if chunk.empty:
            continue

        # moyenne pondérée par nombre de pixels valides
        chunk["weighted"] = chunk["value_mean"] * chunk["px_count"]

        g = (
            chunk.groupby(["ID_PARCEL", "date", "index"], as_index=False)
            .agg(weighted_sum=("weighted", "sum"),
                 px_sum=("px_count", "sum"))
        )
        agg_chunks.append(g)

    if not agg_chunks:
        raise RuntimeError("Aucune ligne conservée après filtrage.")

    agg = pd.concat(agg_chunks, ignore_index=True)

    agg = (
        agg.groupby(["ID_PARCEL", "date", "index"], as_index=False)
        .agg(weighted_sum=("weighted_sum", "sum"),
             px_sum=("px_sum", "sum"))
    )

    agg["value"] = agg["weighted_sum"] / agg["px_sum"]
    return agg[["ID_PARCEL", "date", "index", "value"]]


# =========================
# 3) Pivot wide
# =========================
def build_wide_features(agg: pd.DataFrame) -> pd.DataFrame:
    feat_name = agg["index"].astype(str) + "__" + agg["date"].astype(str)

    tmp = agg[["ID_PARCEL", "value"]].copy()
    tmp["feature"] = feat_name

    wide = tmp.pivot_table(
        index="ID_PARCEL",
        columns="feature",
        values="value",
        aggfunc="mean"
    )

    wide.columns.name = None
    wide = wide.reset_index()
    return wide


# =========================
# 4) Features phénologiques simples
# =========================
def build_phenology_features(agg: pd.DataFrame) -> pd.DataFrame:
    base = agg.copy()
    base["date_dt"] = pd.to_datetime(base["date"], errors="coerce")
    base = base.dropna(subset=["date_dt"]).copy()
    base["doy"] = base["date_dt"].dt.dayofyear.astype(float)

    gkeys = ["ID_PARCEL", "index"]

    stats = (
        base.groupby(gkeys)["value"]
        .agg(["mean", "std", "min", "max", "median"])
        .reset_index()
    )
    stats["amp"] = stats["max"] - stats["min"]

    idx_max = base.groupby(gkeys)["value"].idxmax()
    idx_min = base.groupby(gkeys)["value"].idxmin()

    peak = base.loc[idx_max, gkeys + ["doy"]].rename(columns={"doy": "doy_peak"})
    trough = base.loc[idx_min, gkeys + ["doy"]].rename(columns={"doy": "doy_trough"})

    srt = base.sort_values(gkeys + ["doy"]).copy()
    srt["doy_next"] = srt.groupby(gkeys)["doy"].shift(-1)
    srt["val_next"] = srt.groupby(gkeys)["value"].shift(-1)
    srt["auc_seg"] = 0.5 * (srt["value"] + srt["val_next"]) * (srt["doy_next"] - srt["doy"])

    auc = (
        srt.groupby(gkeys, as_index=False)["auc_seg"]
        .sum()
        .rename(columns={"auc_seg": "auc"})
    )

    pheno = (
        stats.merge(peak, on=gkeys, how="left")
             .merge(trough, on=gkeys, how="left")
             .merge(auc, on=gkeys, how="left")
    )

    pheno_wide = pheno.set_index(gkeys).unstack("index")
    pheno_wide.columns = [f"{idx}__{feat}" for feat, idx in pheno_wide.columns]
    pheno_wide = pheno_wide.reset_index()

    return pheno_wide


# =========================
# 5) Imputation
# =========================
def interpolate_and_impute(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    cols = [c for c in X.columns if "__" in c]
    temporal_cols = []

    for col in cols:
        _, suffix = col.split("__", 1)
        parsed = pd.to_datetime(suffix, errors="coerce")
        if not pd.isna(parsed):
            temporal_cols.append(col)

    grouped = {}
    for col in temporal_cols:
        idx_name, date_str = col.split("__", 1)
        grouped.setdefault(idx_name, []).append((date_str, col))

    for idx_name, items in grouped.items():
        sorted_cols = [c for _, c in sorted(items, key=lambda x: x[0])]
        X[sorted_cols] = X[sorted_cols].interpolate(axis=1, limit_direction="both")

    numeric_cols = X.select_dtypes(include=[np.number]).columns
    medians = X[numeric_cols].median(axis=0)
    X[numeric_cols] = X[numeric_cols].fillna(medians)

    return X


# =========================
# MAIN
# =========================
def main():
    print("Chargement des labels...")
    labels = load_labels(LABELS_CSV, ID_COL, TARGET_COL)
    print(f"Nb parcelles labellisées : {len(labels):,}")
    print(f"Nb classes initiales : {labels[TARGET_COL].nunique()}")

    id_set = set(labels[ID_COL].tolist())

    print("Agrégation des indices...")
    agg = aggregate_indices(
        path=INDICES_CSV,
        id_set=id_set,
        indices_keep=INDICES_KEEP,
        cloud_max=CLOUD_MAX,
        px_count_min=PX_COUNT_MIN,
    )

    print(f"Lignes agrégées : {len(agg):,}")
    print(f"Nb dates : {agg['date'].nunique()}")
    print(f"Nb indices : {agg['index'].nunique()}")

    print("Construction des features wide...")
    X_wide = build_wide_features(agg).rename(columns={"ID_PARCEL": ID_COL})

    print("Construction des features phénologiques...")
    pheno_wide = build_phenology_features(agg).rename(columns={"ID_PARCEL": ID_COL})

    print("Fusion features + labels...")
    data = labels.merge(X_wide, on=ID_COL, how="inner")
    data = data.merge(pheno_wide, on=ID_COL, how="left")

    # filtrer classes rares
    class_counts = data[TARGET_COL].value_counts()
    keep_classes = class_counts[class_counts >= MIN_CLASS_COUNT].index
    data = data[data[TARGET_COL].isin(keep_classes)].copy()

    if data.empty:
        raise RuntimeError("Aucune donnée restante après filtrage des classes rares.")

    print(f"Nb parcelles après filtrage : {len(data):,}")
    print(f"Nb classes après filtrage : {data[TARGET_COL].nunique()}")

    y_str = data[TARGET_COL].astype(str)
    X = data.drop(columns=[ID_COL, TARGET_COL]).copy()

    print("Imputation des valeurs manquantes...")
    missing_before = X.isna().mean().mean()
    X = interpolate_and_impute(X)
    missing_after = X.isna().mean().mean()

    print(f"Missing avant : {missing_before:.4f}")
    print(f"Missing après : {missing_after:.4f}")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("Entraînement Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample"
    )

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    print("\n=== Résultats Random Forest ===")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Macro-F1     : {macro_f1:.4f}")
    print(f"Weighted-F1  : {weighted_f1:.4f}")

    class_names = encoder.classes_

    report = classification_report(
        y_test, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report).T
    report_df.to_csv(OUTPUT_DIR / "classification_report_rf.csv")

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(OUTPUT_DIR / "confusion_matrix_rf.csv")

    importances = pd.DataFrame({
        "feature": X.columns,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)

    importances.to_csv(OUTPUT_DIR / "feature_importance_rf.csv", index=False)

    metrics = {
        "n_samples": int(len(data)),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(class_names)),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "missing_before_imputation": float(missing_before),
        "missing_after_imputation": float(missing_after),
    }

    with open(OUTPUT_DIR / "metrics_rf.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\nFichiers écrits dans : {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()