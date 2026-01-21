from tensorflow import keras
import shutil
import os

export_path = "models/artifacts/performance_predictor/tf_model_saved"

# Clean old export if it exists
if os.path.exists(export_path):
    shutil.rmtree(export_path)

# Load existing Keras model
model = keras.models.load_model(
    "models/artifacts/performance_predictor/tf_model.keras",
    compile=False
)

# Export as TensorFlow SavedModel (Keras 3 correct API)
model.export(export_path)

print("SavedModel exported successfully")
