import numpy as np
import pandas as pd

X = np.load("lstm_data/X_lstm.npy")
y = np.load("lstm_data/y_lstm.npy")

indices = ["NDVI", "EVI", "NDMI", "NDWI"]

N, T, F = X.shape

rows = []

for i in range(N):
    sample = X[i]

    # enlever padding
    valid_mask = ~(np.all(sample == sample[:, [0]], axis=1))
    valid_sample = sample[valid_mask]

    if len(valid_sample) == 0:
        continue

    row = {
        "label": y[i],
        "n_valid_dates": len(valid_sample)
    }

    # pour chaque indice
    for f, name in enumerate(indices):
        ts = valid_sample[:, f]

        row[f"{name}_mean"] = np.mean(ts)
        row[f"{name}_max"] = np.max(ts)
        row[f"{name}_min"] = np.min(ts)
        row[f"{name}_std"] = np.std(ts)
        row[f"{name}_amplitude"] = np.max(ts) - np.min(ts)

    rows.append(row)

df = pd.DataFrame(rows)

print(df.head())
print(df.shape)

df.to_csv("temporal_features_full.csv", index=False)