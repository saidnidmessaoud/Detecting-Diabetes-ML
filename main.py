import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.models import Model, Sequential
import h5py
from PIL import Image
import numpy as np
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Retina & Diabetes Image Classifier API")


# ---------------------------
# Custom DepthwiseConv2D Fix
# ---------------------------
class CompatibleDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        # Remove incompatible parameters
        kwargs.pop('groups', None)
        kwargs.pop('activity_regularizer', None)
        super().__init__(*args, **kwargs)


# ---------------------------
# Model Loading with Fallbacks
# ---------------------------
def load_model_with_fallback(model_path):
    custom_objects = {
        'DepthwiseConv2D': CompatibleDepthwiseConv2D,
        # Add other custom layers if needed
    }

    try:
        # Try standard loading first
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    except Exception as e:
        logger.warning(f"Standard load failed, attempting manual reconstruction: {str(e)}")
        try:
            # Manual model reconstruction
            with h5py.File(model_path, 'r') as f:
                # Handle different h5py versions
                model_config = f.attrs.get('model_config')
                if model_config is None:
                    raise ValueError("No model configuration found in h5 file")

                if isinstance(model_config, (bytes, bytearray)):
                    model_config = model_config.decode('utf-8')
                model_config = json.loads(model_config)

                # Rebuild model
                model = Model.from_config(model_config, custom_objects=custom_objects)

                # Load weights
                weight_values = []
                for layer in model.layers:
                    if layer.name in f:
                        weight_values.append(f[layer.name][()])

                if weight_values:
                    model.set_weights(weight_values)

                return model
        except Exception as e:
            logger.error(f"Manual model reconstruction failed: {str(e)}")
            raise


# ---------------------------
# Load Labels
# ---------------------------
def load_labels(path: str):
    try:
        with open(path, "r", encoding='utf-8') as f:
            return [line.split(" ", 1)[1].strip() for line in f.read().splitlines()]
    except Exception as e:
        logger.error(f"Error loading labels from {path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Label file error: {str(e)}")


try:
    labels_classifier = load_labels("labels1.txt")
    labels_diabetes = load_labels("labels.txt")
except Exception as e:
    logger.error(f"Failed to load labels: {str(e)}")
    raise

# ---------------------------
# Load Models
# ---------------------------
try:
    models = {
        "classifier": load_model_with_fallback("classifier.h5"),
        "diabetes": tf.keras.models.load_model("model.h5")
    }
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    raise HTTPException(status_code=500, detail="Model initialization failed")


# ---------------------------
# Preprocessing
# ---------------------------
def preprocess_image(file, target_size=(224, 224), normalize="tanh"):
    try:
        image = Image.open(file.file).convert("RGB").resize(target_size)
        image_array = np.array(image, dtype=np.float32)

        if normalize == "tanh":
            image_array = image_array / 127.5 - 1.0
        elif normalize == "sigmoid":
            image_array = image_array / 255.0

        return np.expand_dims(image_array, axis=0)
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")


# ---------------------------
# Prediction Helper
# ---------------------------
def predict_with_confidence(model, labels, image_array):
    try:
        prediction = model.predict(image_array)[0]
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        return {
            "class": labels[predicted_class],
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ---------------------------
# Routes
# ---------------------------
@app.get("/", summary="Health Check")
async def root():
    return {
        "message": "API is running. Use /docs for Swagger UI.",
        "status": "healthy",
        "models": list(models.keys())
    }


@app.post("/predict")
async def predict_diabetes_api(file: UploadFile = File(...)):
    image_array = preprocess_image(file)
    result = await run_in_threadpool(
        lambda: predict_with_confidence(models["diabetes"], labels_diabetes, image_array)
    )
    return JSONResponse(content=result)


@app.post("/classify")
async def classify_retina_api(file: UploadFile = File(...)):
    image_array = preprocess_image(file)
    result = await run_in_threadpool(
        lambda: predict_with_confidence(models["classifier"], labels_classifier, image_array)
    )
    return JSONResponse(content=result)


# ---------------------------
# Run Server
# ---------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)