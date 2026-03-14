from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# =========================
# CONFIG
# =========================
INDICES_CSV = "data/s2_herault_2024_full/indices_parcelles_2024-01-01_2024-12-31_win5d.csv"
LABELS_CSV = "export.csv"

ID_COL = "ID_PARCEL"
TARGET_COL = "CODE_CULTU"

INDICES_KEEP = ["NDVI", "EVI", "NDMI", "NDWI"]
CLOUD_MAX = 40.0
PX_COUNT_MIN = 10
MIN_CLASS_COUNT = 200
MAX_DATES = 30

OUTDIR = Path("lstm_data")
OUTDIR.mkdir(exist_ok=True)


# =========================
# 1) Charger les labels
# =========================
def load_labels(path, id_col, target_col):
    labels = pd.read_csv(path, usecols=[id_col, target_col])
    labels = labels.dropna(subset=[target_col]).copy()
    labels[id_col] = labels[id_col].astype(str)
    labels[target_col] = labels[target_col].astype(str)
    labels = labels.drop_duplicates(subset=[id_col])
    return labels


# =========================
# 2) Agréger les indices
# =========================
def aggregate_indices(path, id_set, indices_keep, cloud_max, px_count_min, chunksize=500_000):
    cols = ["date", "ID_PARCEL", "index", "value_mean", "px_count", "cloud_scene"]
    parts = []

    for chunk in pd.read_csv(path, usecols=cols, chunksize=chunksize):
        chunk = chunk.dropna(subset=cols).copy()
        chunk["ID_PARCEL"] = chunk["ID_PARCEL"].astype(str)

        chunk = chunk[chunk["ID_PARCEL"].isin(id_set)]
        chunk = chunk[chunk["index"].isin(indices_keep)]
        chunk = chunk[chunk["cloud_scene"] <= cloud_max]
        chunk = chunk[chunk["px_count"] >= px_count_min]

        if chunk.empty:
            continue

        # moyenne pondérée par le nombre de pixels valides
        chunk["weighted"] = chunk["value_mean"] * chunk["px_count"]

        g = (
            chunk.groupby(["ID_PARCEL", "date", "index"], as_index=False)
            .agg(
                weighted_sum=("weighted", "sum"),
                px_sum=("px_count", "sum")
            )
        )
        parts.append(g)

    if not parts:
        raise RuntimeError("Aucune donnée après filtrage.")

    agg = pd.concat(parts, ignore_index=True)

    agg = (
        agg.groupby(["ID_PARCEL", "date", "index"], as_index=False)
        .agg(
            weighted_sum=("weighted_sum", "sum"),
            px_sum=("px_sum", "sum")
        )
    )

    agg["value"] = agg["weighted_sum"] / agg["px_sum"]
    return agg[["ID_PARCEL", "date", "index", "value"]]


# =========================
# 3) Limiter le nombre de dates
# =========================
def limit_dates(agg, max_dates):
    if max_dates <= 0:
        ordered_dates = sorted(agg["date"].unique().tolist())
        return agg.copy(), ordered_dates

    date_counts = agg.groupby("date")["ID_PARCEL"].nunique().sort_values(ascending=False)
    keep_dates = sorted(date_counts.head(max_dates).index.tolist())
    agg = agg[agg["date"].isin(keep_dates)].copy()
    return agg, keep_dates


# =========================
# 4) Construire le tenseur LSTM
# =========================
def build_tensor(agg, labels, id_col, target_col, indices_keep, ordered_dates, min_class_count):
    # Pivot : lignes = parcelles, colonnes = (date, index)
    data = agg.pivot_table(
        index=id_col,
        columns=["date", "index"],
        values="value",
        aggfunc="mean"
    )

    # Garantir toutes les combinaisons date x indice
    full_cols = pd.MultiIndex.from_product(
        [ordered_dates, indices_keep],
        names=["date", "index"]
    )
    data = data.reindex(columns=full_cols)

    # Aplatir les colonnes MultiIndex -> "2024-01-12__NDVI"
    data.columns = [f"{d}__{idx}" for d, idx in data.columns]
    data = data.reset_index()

    # Sécurité sur les types
    data[id_col] = data[id_col].astype(str)
    labels = labels.copy()
    labels[id_col] = labels[id_col].astype(str)

    # Jointure avec la cible
    merged = labels.merge(data, on=id_col, how="inner").copy()

    # Filtrer classes trop rares
    class_counts = merged[target_col].value_counts()
    keep_classes = class_counts[class_counts >= min_class_count].index
    merged = merged[merged[target_col].isin(keep_classes)].copy()

    if merged.empty:
        raise RuntimeError("Aucune donnée restante après filtrage des classes.")

    # Sauvegarder y et ids
    y_str = merged[target_col].astype(str).values
    ids = merged[id_col].values

    # Features
    X_df = merged.drop(columns=[id_col, target_col]).copy()

    # Interpolation horizontale (au fil du temps)
    X_df = X_df.interpolate(axis=1, limit_direction="both")

    # Imputation finale par médiane de colonne
    X_df = X_df.fillna(X_df.median())

    # Conversion en matrice numpy
    X = X_df.to_numpy(dtype=np.float32)

    n_samples = X.shape[0]
    n_dates = len(ordered_dates)
    n_indices = len(indices_keep)

    expected_n_features = n_dates * n_indices
    if X.shape[1] != expected_n_features:
        raise ValueError(
            f"Nombre de features inattendu : {X.shape[1]} colonnes, "
            f"alors qu'on attend {expected_n_features} = {n_dates} x {n_indices}."
        )

    # Reshape en tenseur 3D : (n_samples, n_dates, n_indices)
    X = X.reshape(n_samples, n_dates, n_indices)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_str)

    return X, y, ids, encoder, merged


# =========================
# MAIN
# =========================
def main():
    print("Chargement des labels...")
    labels = load_labels(LABELS_CSV, ID_COL, TARGET_COL)
    id_set = set(labels[ID_COL].tolist())

    print("Agrégation des indices...")
    agg = aggregate_indices(
        INDICES_CSV,
        id_set=id_set,
        indices_keep=set(INDICES_KEEP),
        cloud_max=CLOUD_MAX,
        px_count_min=PX_COUNT_MIN
    )

    agg, ordered_dates = limit_dates(agg, MAX_DATES)

    print(f"Nb dates retenues : {len(ordered_dates)}")
    print(f"Dates : {ordered_dates[:3]} ... {ordered_dates[-3:]}")

    print("Construction du tenseur LSTM...")
    X, y, ids, encoder, merged = build_tensor(
        agg=agg,
        labels=labels,
        id_col=ID_COL,
        target_col=TARGET_COL,
        indices_keep=INDICES_KEEP,
        ordered_dates=ordered_dates,
        min_class_count=MIN_CLASS_COUNT
    )

    print("Shape X :", X.shape)
    print("Shape y :", y.shape)
    print("Nb classes :", len(encoder.classes_))

    # Sauvegardes
    np.save(OUTDIR / "X_lstm.npy", X)
    np.save(OUTDIR / "y_lstm.npy", y)

    pd.Series(ids, name=ID_COL).to_csv(OUTDIR / "ids_lstm.csv", index=False)
    pd.Series(ordered_dates, name="date").to_csv(OUTDIR / "dates_lstm.csv", index=False)
    pd.Series(encoder.classes_, name="class").to_csv(OUTDIR / "classes_lstm.csv", index=False)

    # Petit fichier résumé utile
    summary = {
        "n_samples": int(X.shape[0]),
        "n_dates": int(X.shape[1]),
        "n_indices": int(X.shape[2]),
        "n_classes": int(len(encoder.classes_)),
        "indices": INDICES_KEEP,
        "target_col": TARGET_COL,
        "min_class_count": MIN_CLASS_COUNT,
        "cloud_max": CLOUD_MAX,
        "px_count_min": PX_COUNT_MIN
    }
    pd.Series(summary).to_csv(OUTDIR / "summary_lstm.csv")

    print(f"Données sauvegardées dans : {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()