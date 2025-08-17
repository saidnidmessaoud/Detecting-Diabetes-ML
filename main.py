import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
from PIL import Image
import numpy as np

app = FastAPI()

# Load the classifier model and labels
model_classifier = tf.keras.models.load_model('classifier.h5')
with open('labels1.txt', 'r') as file:
    labels_classifier = file.read().splitlines()


# Load the diabete model and labels
model_diabete = tf.keras.models.load_model('model.h5')
with open('labels.txt', 'r') as file:
    labels_diabetes = file.read().splitlines()

# Define image preprocessing function
def preprocess_image(file):
    image = Image.open(file.file).convert('RGB')
    image = image.resize((224, 224))  # Adjust the size as needed
    image_array = np.array(image) / 127.5 - 1.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Define classifier function
def classify_retine(image_array):
    prediction = model_classifier.predict(image_array)
    predicted_class = int(np.argmax(prediction))
    return labels_classifier[predicted_class][2:]

# Define diabete prediction function
def predict_diabetes(image_array):
    prediction = model_diabete.predict(image_array)
    predicted_class = int(np.argmax(prediction))
    return labels_diabetes[predicted_class][2:]


# Define FastAPI route
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_array = preprocess_image(file)
        prediction = predict_diabetes(image_array)
        return JSONResponse(content={"result": prediction}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/classify")
async def predict(file: UploadFile = File(...)):
    try:
        image_array = preprocess_image(file)
        prediction = classify_retine(image_array)
        return JSONResponse(content={"result": prediction}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

uvicorn.run(app, port=8000,host='0.0.0.0')
