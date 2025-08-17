import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import DepthwiseConv2D
import h5py
import json
from PIL import Image
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fixed Retina & Diabetes Classifier")


# 1. FIRST - Add this custom layer fix at the TOP of your file
class FixedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)  # Remove problematic parameter
        super().__init__(*args, **kwargs)


# 2. THEN - Replace your model loading code with this
def load_labels(path: str):
    with open(path, "r") as f:
        return [line.split(" ", 1)[1].strip() for line in f.read().splitlines()]


def load_model_safely(model_path):
    custom_objects = {'DepthwiseConv2D': FixedDepthwiseConv2D}

    try:
        # Try standard loading first
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        logger.info(f"Successfully loaded {model_path} normally")
        return model
    except Exception as e:
        logger.warning(f"Standard load failed, trying manual load: {str(e)}")
        try:
            # Manual loading if standard fails
            with h5py.File(model_path, 'r') as f:
                model_config = json.loads(f.attrs['model_config'].decode('utf-8'))
                model = Model.from_config(model_config, custom_objects=custom_objects)
                model.load_weights(model_path)
                logger.info(f"Manually loaded {model_path}")
                return model
        except Exception as e:
            logger.error(f"Failed to load {model_path}: {str(e)}")
            raise


# Load models and labels
try:
    labels_classifier = load_labels("labels1.txt")
    labels_diabetes = load_labels("labels.txt")
    models = {
        "classifier": load_model_safely("classifier.h5"),
        "diabetes": load_model_safely("model.h5")
    }
except Exception as e:
    logger.error(f"Initialization failed: {str(e)}")
    raise HTTPException(status_code=500, detail="Model loading failed")


# 3. KEEP your existing preprocessing and prediction functions
def preprocess_image(file, target_size=(224, 224)):
    try:
        image = Image.open(file.file).convert("RGB").resize(target_size)
        image_array = np.array(image, dtype=np.float32) / 127.5 - 1.0
        return np.expand_dims(image_array, axis=0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image error: {str(e)}")


def predict_with_confidence(model, labels, image_array):
    prediction = model.predict(image_array)[0]
    return {
        "class": labels[np.argmax(prediction)],
        "confidence": float(np.max(prediction))
    }


# 4. KEEP your existing API endpoints
@app.get("/")
async def health_check():
    return {"status": "running", "models": list(models.keys())}


@app.post("/predict")
async def predict_diabetes(file: UploadFile = File(...)):
    img = preprocess_image(file)
    result = await run_in_threadpool(
        lambda: predict_with_confidence(models["diabetes"], labels_diabetes, img)
    )
    return JSONResponse(result)


@app.post("/classify")
async def classify_retina(file: UploadFile = File(...)):
    img = preprocess_image(file)
    result = await run_in_threadpool(
        lambda: predict_with_confidence(models["classifier"], labels_classifier, img)
    )
    return JSONResponse(result)


# 5. FINALLY - Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)