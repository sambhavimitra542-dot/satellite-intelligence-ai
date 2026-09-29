import rasterio
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib


# --------------------------------------------------
# 1. Define file locations
# --------------------------------------------------

s1_file = Path(
    "data/raw/sample/S1/Spain_7370579_S1Hand.tif"
)

label_file = Path(
    "data/raw/sample/Labels/Spain_7370579_LabelHand.tif"
)


# --------------------------------------------------
# 2. Load Sentinel-1 data
# --------------------------------------------------

with rasterio.open(s1_file) as src:
    vv = src.read(1)
    vh = src.read(2)


# --------------------------------------------------
# 3. Load flood labels
# --------------------------------------------------

with rasterio.open(label_file) as src:
    labels = src.read(1)


# --------------------------------------------------
# 4. Convert image into machine-learning features
# --------------------------------------------------

vv_flat = vv.flatten()
vh_flat = vh.flatten()
labels_flat = labels.flatten()


# Each pixel becomes:
#
# [VV value, VH value]
#
# Example:
# Pixel 1 -> [-12.5, -19.3]
# Pixel 2 -> [-8.2, -15.7]
#
# The model will learn whether these values
# correspond to water or non-water.


X = np.column_stack((vv_flat, vh_flat))
y = labels_flat


# --------------------------------------------------
# 5. Remove invalid pixels
# --------------------------------------------------

valid_pixels = y != -1

X = X[valid_pixels]
y = y[valid_pixels]


print("Dataset prepared")
print("----------------")
print("Total valid pixels:", len(y))
print("Feature shape:", X.shape)
print("Label shape:", y.shape)


# --------------------------------------------------
# 6. Use a manageable sample for the first model
# --------------------------------------------------

sample_size = min(50000, len(y))

rng = np.random.default_rng(42)

sample_indices = rng.choice(
    len(y),
    size=sample_size,
    replace=False
)

X = X[sample_indices]
y = y[sample_indices]


print("\nUsing pixels for training:", len(y))


# --------------------------------------------------
# 7. Split data into training and testing sets
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Training samples:", len(y_train))
print("Testing samples:", len(y_test))


# --------------------------------------------------
# 8. Create the Random Forest model
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)


# --------------------------------------------------
# 9. Train the model
# --------------------------------------------------

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Training completed!")


# --------------------------------------------------
# 10. Make predictions
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# 11. Evaluate the model
# --------------------------------------------------

print("\nClassification Report")
print("---------------------")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Non-water", "Water"]
    )
)


print("Confusion Matrix")
print("----------------")

print(confusion_matrix(y_test, y_pred))


# --------------------------------------------------
# 12. Save the trained model
# --------------------------------------------------

models_folder = Path("models")
models_folder.mkdir(exist_ok=True)

model_path = models_folder / "baseline_random_forest.joblib"

joblib.dump(model, model_path)

print("\nModel saved to:")
print(model_path)