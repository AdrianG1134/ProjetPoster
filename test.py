import rasterio
import numpy as np

p = "data/sentinel2_herault_2023/2023-03-16/B02.tif"  # adapte

with rasterio.open(p) as src:
    a = src.read(1)
    print("CRS:", src.crs)
    print("bounds:", src.bounds)
    print("dtype:", a.dtype, "shape:", a.shape)
    print("nodata:", src.nodata)
    print("min/max:", float(np.nanmin(a)), float(np.nanmax(a)))
    print("pct non-zero:", 100.0 * np.mean(a != 0))
