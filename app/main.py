from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from google.cloud import storage
import mysql.connector
from mysql.connector import Error
import numpy as np
import json
import os
import uuid
from datetime import datetime
from jose import jwt, JWTError
from dotenv import load_dotenv
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = os.getenv("DB_PORT")
BUCKET_NAME = os.getenv("BUCKET_NAME")
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
SECRET_KEY = os.getenv("SECRET_KEY", "1234")  # Ganti dengan kunci rahasia Anda
ALGORITHM = "HS256"

# Load model and categories
model_path = os.path.join("app", "model", "best_model.h5")
model = load_model(model_path)

categories_path = os.path.join("app", "model", "categories.json")
with open(categories_path, "r") as f:
    categories = json.load(f)

nutritions_path = os.path.join("app", "model", "nutritions.json")
with open(nutritions_path, "r") as f:
    nutritions = json.load(f)

# Initialize FastAPI
app = FastAPI()

# Configure Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH

# JWT Configuration
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="http://sehatin-cc.vercel.app/api/auth/login")

def verify_jwt_token(token: str = Depends(oauth2_scheme)):
    """
    Middleware untuk memverifikasi JWT token.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("id")  # Menggunakan "id" sesuai dengan payload token
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: id missing."
            )
        return {"user_id": user_id}
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired"
        )
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}"
        )

# Database connection function
def get_db_connection():
    """Create a database connection."""
    try:
        db_config = {
            "host": DB_HOST,
            "user": DB_USER,
            "password": DB_PASSWORD,
            "database": DB_NAME,
            "port": int(DB_PORT),
        }
        connection = mysql.connector.connect(**db_config)
        return connection
    except Error as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

# Upload file to Google Cloud Storage
def upload_to_gcs(file_path, destination_blob_name):
    """Upload a file to Google Cloud Storage."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(file_path)
        return f"https://storage.googleapis.com/{BUCKET_NAME}/{destination_blob_name}"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload to GCS: {str(e)}")

@app.get("/")
async def home():
    return {"message": "Sehatin Model 2 API is running"}

@app.post("/predict/image/")
async def predict_image(
    file: UploadFile = File(...),
    token_payload: dict = Depends(verify_jwt_token)
):
    user_id = token_payload["user_id"]  # Extract user_id from JWT payload
    temp_file_path = f"temp_{uuid.uuid4().hex}.jpg"
    try:
        # Save uploaded file temporarily
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())

        # Load and preprocess the image
        image = Image.open(temp_file_path).convert("RGB")
        image = image.resize((224, 224))  # Resize to model's input size
        image = img_to_array(image) / 255.0  # Normalize
        image = np.expand_dims(image, axis=0)

        # Predict class
        predictions = model.predict(image)
        predicted_index = np.argmax(predictions)
        predicted_class = categories[predicted_index]

        # Get nutrition data
        nutrition_info = nutritions.get(predicted_class, None)
        if not nutrition_info:
            raise HTTPException(status_code=404, detail="Nutrition data not found.")

        # Upload the image to Google Cloud Storage
        gcs_url = upload_to_gcs(temp_file_path, f"uploads/{uuid.uuid4().hex}.jpg")

        # Save data to MySQL
        connection = get_db_connection()
        cursor = connection.cursor()
        query = """
            INSERT INTO nutritions (user_id, image_url, calories, protein, fat, carbohydrates, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (
            user_id,
            gcs_url,
            nutrition_info["calories"],
            nutrition_info["protein"],
            nutrition_info["fat"],
            nutrition_info["carbohydrates"],
            datetime.now()
        ))
        connection.commit()
        cursor.close()
        connection.close()

        # Delete the temporary file
        os.remove(temp_file_path)

        return {
            "success": True,
            "message": "Nutrition data saved successfully."
        }
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/image/")
async def get_predictions(token_payload: dict = Depends(verify_jwt_token)):
    user_id = token_payload["user_id"]  # Extract user_id from JWT payload
    try:
        # Fetch data from MySQL
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        query = "SELECT * FROM nutritions WHERE user_id = %s ORDER BY created_at DESC"
        cursor.execute(query, (user_id,))
        results = cursor.fetchall()
        cursor.close()
        connection.close()

        return {
            "success": True,
            "data": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
