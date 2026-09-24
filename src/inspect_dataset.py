import rasterio
from pathlib import Path

# Location of our sample satellite dataset
data_path = Path("data/raw/sample")

# Paths to the three important files
s1_file = data_path / "S1" / "Spain_7370579_S1Hand.tif"
s2_file = data_path / "S2" / "Spain_7370579_S2Hand.tif"
label_file = data_path / "Labels" / "Spain_7370579_LabelHand.tif"

print("Checking Sen1Floods11 sample dataset...\n")

# Check Sentinel-1
with rasterio.open(s1_file) as src:
    print("S1 Sentinel-1")
    print("  Shape:", (src.height, src.width))
    print("  Bands:", src.count)
    print("  Data type:", src.dtypes)
    print("  CRS:", src.crs)

# Check Sentinel-2
with rasterio.open(s2_file) as src:
    print("\nS2 Sentinel-2")
    print("  Shape:", (src.height, src.width))
    print("  Bands:", src.count)
    print("  Data type:", src.dtypes)
    print("  CRS:", src.crs)

# Check flood label
with rasterio.open(label_file) as src:
    print("\nFlood Label")
    print("  Shape:", (src.height, src.width))
    print("  Bands:", src.count)
    print("  Data type:", src.dtypes)
    print("  CRS:", src.crs)

print("\nDataset inspection completed successfully!")
# ---------------------------------------------------------
# VISUALIZE THE SATELLITE DATA
# ---------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

print("\nCreating satellite image visualization...")

# Read Sentinel-1
with rasterio.open(s1_file) as src:
    s1 = src.read()

# Read Sentinel-2
with rasterio.open(s2_file) as src:
    s2 = src.read()

# Read flood label
with rasterio.open(label_file) as src:
    label = src.read(1)


# ---------------------------------------------------------
# Sentinel-1 visualization
# We use the first radar band
# ---------------------------------------------------------

s1_image = s1[0]

# Improve the display contrast
s1_min = np.percentile(s1_image, 2)
s1_max = np.percentile(s1_image, 98)

s1_display = np.clip(
    (s1_image - s1_min) / (s1_max - s1_min),
    0,
    1
)


# ---------------------------------------------------------
# Sentinel-2 RGB visualization
# Sentinel-2 bands 4, 3 and 2 correspond to RGB
# ---------------------------------------------------------

red = s2[3]
green = s2[2]
blue = s2[1]

def normalize_band(band):
    minimum = np.percentile(band, 2)
    maximum = np.percentile(band, 98)

    return np.clip(
        (band - minimum) / (maximum - minimum),
        0,
        1
    )

red = normalize_band(red)
green = normalize_band(green)
blue = normalize_band(blue)

rgb = np.dstack((red, green, blue))


# ---------------------------------------------------------
# Create the three-panel figure
# ---------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(s1_display, cmap="gray")
axes[0].set_title("Sentinel-1 Radar")
axes[0].axis("off")

axes[1].imshow(rgb)
axes[1].set_title("Sentinel-2 RGB")
axes[1].axis("off")

axes[2].imshow(label, cmap="gray")
axes[2].set_title("Flood Label / Mask")
axes[2].axis("off")

plt.tight_layout()


# ---------------------------------------------------------
# Save the visualization
# ---------------------------------------------------------

output_file = "data/processed/sample_visualization.png"

plt.savefig(output_file, dpi=150)

print(f"\nVisualization saved to: {output_file}")

plt.show()