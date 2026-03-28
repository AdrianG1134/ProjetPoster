import pandas as pd

path = "data/s2_herault_2024_full/indices_parcelles_2024-01-01_2024-12-31_win5d.csv"

df = pd.read_csv(path)

print("Shape :", df.shape)
print("Date max :", df["date"].max())
print("Date min :", df["date"].min())