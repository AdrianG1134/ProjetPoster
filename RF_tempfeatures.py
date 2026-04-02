import json
import numpy as np
import pandas as pd

from scipy.integrate import trapezoid
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

# =========================
# 1) Chargement des données
# =========================
X = np.load("lstm_data/X_lstm.npy")   # shape: (N, T, F)
y = np.load("lstm_data/y_lstm.npy")

dates_df = pd.read_csv("lstm_data/dates_lstm.csv")
ids_df = pd.read_csv("lstm_data/ids_lstm.csv")

# ordre confirmé par summary_lstm.csv
indices = ["NDVI", "EVI", "NDMI", "NDWI"]

print("Shape X:", X.shape)
print("Shape y:", y.shape)

N, T, F = X.shape

# =========================
# 2) Préparation dates
# =========================
dates = pd.to_datetime(dates_df["date"])
doy = dates.dt.dayofyear.to_numpy()

# =========================
# 3) Fonctions utilitaires
# =========================
def get_valid_rows(sample: np.ndarray) -> np.ndarray:
    """
    Détecte les lignes paddées :
    ici, une ligne paddée a les 4 colonnes identiques.
    """
    return ~(np.all(sample == sample[:, [0]], axis=1))

def summarize_series(ts: np.ndarray, doy_valid: np.ndarray, prefix: str) -> dict:
    """
    Résume une série temporelle 1D en features phénologiques simples.
    """
    out = {}

    if len(ts) == 0:
        out[f"{prefix}_mean"] = np.nan
        out[f"{prefix}_max"] = np.nan
        out[f"{prefix}_min"] = np.nan
        out[f"{prefix}_std"] = np.nan
        out[f"{prefix}_amplitude"] = np.nan
        out[f"{prefix}_peak_doy"] = np.nan
        out[f"{prefix}_peak_pos"] = np.nan
        out[f"{prefix}_auc"] = np.nan
        out[f"{prefix}_mean_diff"] = np.nan
        return out

    peak_idx = int(np.argmax(ts))

    out[f"{prefix}_mean"] = float(np.mean(ts))
    out[f"{prefix}_max"] = float(np.max(ts))
    out[f"{prefix}_min"] = float(np.min(ts))
    out[f"{prefix}_std"] = float(np.std(ts))
    out[f"{prefix}_amplitude"] = float(np.max(ts) - np.min(ts))
    out[f"{prefix}_peak_doy"] = float(doy_valid[peak_idx])
    out[f"{prefix}_peak_pos"] = float(peak_idx / max(len(ts) - 1, 1))
    out[f"{prefix}_auc"] = float(trapezoid(ts, doy_valid)) if len(ts) > 1 else 0.0
    out[f"{prefix}_mean_diff"] = float(np.mean(np.diff(ts))) if len(ts) > 1 else 0.0

    return out

# =========================
# 4) Extraction des features
# =========================
rows = []

for i in range(N):
    sample = X[i]  # shape: (30, 4)
    valid_mask = get_valid_rows(sample)
    valid_sample = sample[valid_mask]

    if len(valid_sample) == 0:
        continue

    doy_valid = doy[valid_mask]

    row = {
        "label": int(y[i]),
        "n_valid_dates": int(len(valid_sample))
    }

    if ids_df.shape[1] > 0:
        row["ID_PARCEL"] = ids_df.iloc[i, 0]

    for f_idx, name in enumerate(indices):
        ts = valid_sample[:, f_idx]
        row.update(summarize_series(ts, doy_valid, name))

    rows.append(row)

df = pd.DataFrame(rows)

print("\nAperçu features:")
print(df.head())
print("\nShape features:", df.shape)

df.to_csv("temporal_features_full_v2.csv", index=False)

# =========================
# 5) Split train / test
# =========================
drop_cols = ["label"]
if "ID_PARCEL" in df.columns:
    drop_cols.append("ID_PARCEL")

X_tab = df.drop(columns=drop_cols)
y_tab = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X_tab,
    y_tab,
    test_size=0.2,
    random_state=42,
    stratify=y_tab
)

# =========================
# 6) Modèle RF pondéré
# =========================
model = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced_subsample",
    max_depth=20,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# =========================
# 7) Évaluation
# =========================
acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
bal_acc = balanced_accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {acc:.6f}")
print(f"Macro-F1: {macro_f1:.6f}")
print(f"Weighted-F1: {weighted_f1:.6f}")
print(f"Balanced Accuracy: {bal_acc:.6f}")

report_dict = classification_report(
    y_test,
    y_pred,
    zero_division=0,
    output_dict=True
)
report_df = pd.DataFrame(report_dict).transpose()
print("\nClassification report:")
print(report_df)

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm)

# =========================
# 8) Importance des variables
# =========================
importances = pd.DataFrame({
    "feature": X_tab.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTop 15 features:")
print(importances.head(15))

# =========================
# 9) Sauvegardes
# =========================
metrics = {
    "n_samples": int(len(df)),
    "n_features": int(X_tab.shape[1]),
    "n_classes": int(y_tab.nunique()),
    "accuracy": float(acc),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "balanced_accuracy": float(bal_acc)
}

with open("rf_temporal_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

report_df.to_csv("rf_temporal_classification_report.csv", index=True)
cm_df.to_csv("rf_temporal_confusion_matrix.csv", index=False)
importances.to_csv("rf_temporal_feature_importance.csv", index=False)

print("\nFichiers générés :")
print("- temporal_features_full_v2.csv")
print("- rf_temporal_metrics.json")
print("- rf_temporal_classification_report.csv")
print("- rf_temporal_confusion_matrix.csv")
print("- rf_temporal_feature_importance.csv")