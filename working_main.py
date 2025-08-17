import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, DepthwiseConv2D, Conv2D, BatchNormalization,
    ReLU, GlobalAveragePooling2D, Dense
)
import h5py
import numpy as np
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Diabetes Classifier API")


# Custom fixed layer
class FixedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        kwargs.pop('activity_regularizer', None)
        super().__init__(*args, **kwargs)


# Build model architecture based on your structure
def build_model():
    input_layer = Input(shape=(224, 224, 3))

    # Feature extraction
    x = Conv2D(32, (3, 3), padding='same', use_bias=False)(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Depthwise convolution
    x = FixedDepthwiseConv2D((3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Classification head
    x = GlobalAveragePooling2D()(x)
    x = Dense(1280, activation='relu')(x)  # Matching your 1280-dim intermediate layer
    output = Dense(2, activation='softmax')(x)

    return Model(inputs=input_layer, outputs=output)


def load_model_correctly(model_path):
    custom_objects = {
        'DepthwiseConv2D': FixedDepthwiseConv2D,
        'FixedDepthwiseConv2D': FixedDepthwiseConv2D
    }

    try:
        # First try standard loading
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    except Exception as e:
        logger.warning(f"Standard load failed, trying manual loading: {str(e)}")
        try:
            # Build model architecture
            model = build_model()

            # Load weights using the new HDF5 format
            with h5py.File(model_path, 'r') as f:
                if 'model_weights' in f:
                    # New weight loading approach
                    for layer in model.layers:
                        if layer.name in f['model_weights']:
                            layer_group = f['model_weights'][layer.name]
                            weight_values = []
                            for weight_name in layer_group.attrs['weight_names']:
                                weight_values.append(layer_group[weight_name.decode()][:])
                            try:
                                layer.set_weights(weight_values)
                            except Exception as e:
                                logger.warning(f"Couldn't set weights for {layer.name}: {str(e)}")

            return model
        except Exception as e:
            logger.error(f"Manual loading failed: {str(e)}")
            raise


# Load labels
def load_labels(path):
    with open(path, "r") as f:
        return [line.strip().split(" ", 1)[1] for line in f if line.strip()]


try:
    labels_classifier = load_labels("labels1.txt")
    labels_diabetes = load_labels("labels.txt")
    models = {
        "classifier": load_model_correctly("classifier.h5"),
        "diabetes": load_model_correctly("model.h5")
    }
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Initialization failed: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")


# Image preprocessing
def preprocess_image(file, target_size=(224, 224)):
    try:
        img = Image.open(file.file).convert("RGB").resize(target_size)
        return np.expand_dims(np.array(img, dtype=np.float32) / 127.5 - 1.0, axis=0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image error: {str(e)}")


# Prediction function
def predict(model, labels, image):
    pred = model.predict(image)[0]
    return {
        "class": labels[np.argmax(pred)],
        "confidence": float(np.max(pred))
    }


# API endpoints
@app.get("/")
async def health_check():
    return {"status": "running", "models": list(models.keys())}


@app.post("/predict")
async def predict_diabetes(file: UploadFile = File(...)):
    img = preprocess_image(file)
    result = await run_in_threadpool(lambda: predict(models["diabetes"], labels_diabetes, img))
    return JSONResponse(result)


@app.post("/classify")
async def classify_retina(file: UploadFile = File(...)):
    img = preprocess_image(file)
    result = await run_in_threadpool(lambda: predict(models["classifier"], labels_classifier, img))
    return JSONResponse(result)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)