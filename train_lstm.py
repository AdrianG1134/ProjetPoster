from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.callbacks import EarlyStopping


# =========================
# CONFIG
# =========================
DATADIR = Path("lstm_data")
OUTDIR = Path("outputs_lstm")
OUTDIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
BATCH_SIZE = 128
EPOCHS = 40


def main():
    print("Chargement des données LSTM...")
    X = np.load(DATADIR / "X_lstm.npy")
    y = np.load(DATADIR / "y_lstm.npy")
    class_names = pd.read_csv(DATADIR / "classes_lstm.csv")["class"].astype(str).values

    print("X shape :", X.shape)
    print("y shape :", y.shape)
    print("Nb classes :", len(class_names))

    # Standardisation globale
    print("Standardisation...")
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True) + 1e-6
    X = (X - mean) / std

    # Split train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Poids de classes
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}

    n_dates = X.shape[1]
    n_features = X.shape[2]
    n_classes = len(np.unique(y))

    print("Construction du modèle LSTM...")
    model = Sequential([
        Masking(mask_value=0.0, input_shape=(n_dates, n_features)),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(n_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    print("Entraînement...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=[early_stop],
        verbose=1
    )

    print("Prédiction...")
    y_prob = model.predict(X_test, batch_size=BATCH_SIZE)
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    print("\n=== Résultats LSTM ===")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Macro-F1    : {macro_f1:.4f}")
    print(f"Weighted-F1 : {weighted_f1:.4f}")

    # Rapport détaillé
    report = classification_report(
        y_test, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    pd.DataFrame(report).T.to_csv(OUTDIR / "classification_report_lstm.csv")

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        OUTDIR / "confusion_matrix_lstm.csv"
    )

    # Historique d'entraînement
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(OUTDIR / "training_history_lstm.csv", index=False)

    # Metrics globales
    metrics = {
        "n_samples": int(len(y)),
        "n_dates": int(n_dates),
        "n_features_per_date": int(n_features),
        "n_classes": int(n_classes),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }

    with open(OUTDIR / "metrics_lstm.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    model.save(OUTDIR / "lstm_model.keras")

    print(f"\nFichiers écrits dans : {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()