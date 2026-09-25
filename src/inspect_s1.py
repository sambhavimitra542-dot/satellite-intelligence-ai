import rasterio
import numpy as np
from pathlib import Path

s1_file = Path(
    "data/raw/sample/S1/Spain_7370579_S1Hand.tif"
)

with rasterio.open(s1_file) as src:
    vv = src.read(1)
    vh = src.read(2)

    print("Sentinel-1 analysis")
    print("-------------------")

    print("Image shape:", vv.shape)
    print("Number of bands:", src.count)

    print("\nVV band:")
    print("Minimum:", np.min(vv))
    print("Maximum:", np.max(vv))
    print("Mean:", np.mean(vv))

    print("\nVH band:")
    print("Minimum:", np.min(vh))
    print("Maximum:", np.max(vh))
    print("Mean:", np.mean(vh))