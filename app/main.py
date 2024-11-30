from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import json
import os

# Load model and categories
model_path = os.path.join("app", "model", "best_model.h5")
model = load_model(model_path)

categories_path = os.path.join("app", "model", "categories.json")
with open(categories_path, "r") as f:
    categories = json.load(f)

# Load nutrition data
nutritions_path = os.path.join("app", "model", "nutritions.json")
with open(nutritions_path, "r") as f:
    nutritions = json.load(f)

app = FastAPI()

@app.get("/")
async def home():
    return {"message": "sehatin-cc-model-2 API is running"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Load and preprocess the image
        image = Image.open(file.file).convert("RGB")
        image = image.resize((224, 224))  # Resize to model's input size
        image = img_to_array(image) / 255.0  # Normalize
        image = np.expand_dims(image, axis=0)

        # Predict class
        predictions = model.predict(image)
        predicted_index = np.argmax(predictions)
        predicted_class = categories[predicted_index]

        # Get nutrition data for the predicted class
        nutrition_info = nutritions.get(predicted_class, "Nutrition data not available")

        # Return the predicted class and its nutrition data
        return {
            "predicted_class": predicted_class,
            "nutritions": nutrition_info
        }
    except Exception as e:
        return {"error": str(e)}

# from fastapi import FastAPI, File, UploadFile
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# from PIL import Image
# import numpy as np
# import json
# import os

# # Load model and categories
# model_path = os.path.join("app", "model", "best_model.h5")
# model = load_model(model_path)

# categories_path = os.path.join("app", "model", "categories.json")
# with open(categories_path, "r") as f:
#     categories = json.load(f)

# app = FastAPI()

# @app.get("/")
# async def home():
#     return {"message": "sehatin-cc-model-2 API is running"}

# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     try:
#         # Load image
#         image = Image.open(file.file).convert("RGB")
#         image = image.resize((224, 224))  # Resize to model's input size
#         image = img_to_array(image) / 255.0  # Normalize
#         image = np.expand_dims(image, axis=0)

#         # Predict
#         predictions = model.predict(image)
#         predicted_class = categories[np.argmax(predictions)]

#         return {"predicted_class": predicted_class, "probabilities": predictions.tolist()}
#     except Exception as e:
#         return {"error": str(e)}
