import rasterio
import numpy as np
from pathlib import Path

label_file = Path(
    "data/raw/sample/Labels/Spain_7370579_LabelHand.tif"
)

with rasterio.open(label_file) as src:
    label = src.read(1)

print("Label analysis")
print("----------------")

print("Image shape:", label.shape)

# Count each type of pixel
invalid_pixels = np.sum(label == -1)
non_water_pixels = np.sum(label == 0)
water_pixels = np.sum(label == 1)

total_valid_pixels = non_water_pixels + water_pixels

print("Invalid pixels:", invalid_pixels)
print("Non-water pixels:", non_water_pixels)
print("Water pixels:", water_pixels)

if total_valid_pixels > 0:
    flood_percentage = (
        water_pixels / total_valid_pixels
    ) * 100

    print(
        "Flooded/water area in this chip:",
        round(flood_percentage, 2),
        "%"
    )