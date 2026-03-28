import pandas as pd

path = "data/s2_herault_2024_full/indices_parcelles_2024-01-01_2024-12-31_win5d.csv"

print("Lecture CSV...")
df = pd.read_csv(path)

print("Pivot des indices...")

df_wide = df.pivot_table(
    index=["date", "ID_PARCEL"],
    columns="index",
    values="value_mean"
).reset_index()

print(df_wide.head())

out = "data/s2_herault_2024_full/dataset_ml.parquet"

df_wide.to_parquet(out)

print("Dataset ML sauvegardé :", out)
print("Shape :", df_wide.shape)