from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from typing import List

# Function to convert a hex color code to RGB
def hex_to_rgb(hex_value: str) -> List[int]:
    # Remove the '#' if it's there and convert hex to RGB
    hex_value = hex_value.lstrip('#')
    return [int(hex_value[i:i+2], 16) for i in (0, 2, 4)]

# Load the item data with dominant colors
item_data = pd.read_csv('item_data_with_dominant_colors.csv')

# Ensure 'dominant_color' is in the correct format (list of integers)
item_data['dominant_color'] = item_data['dominant_color'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ', dtype=int))

# Prepare the data for recommendation (extract dominant colors as feature vector)
X = np.array(item_data['dominant_color'].tolist())

# Load the saved KNN model
knn = joblib.load('knn_model.pkl')

# Prepare the item_id to use in the API (for ease of reference in FastAPI)
item_data['item_id'] = item_data['item_id'].astype(str)

# FastAPI app setup
app = FastAPI()

# Create a Pydantic model for the request body
class SkinToneRequest(BaseModel):
    skin_tone: str  # Hex format string (e.g., #f5dcb4)

# Function to get recommendations based on skin tone
def get_recommendations(skin_tone: List[int], n_neighbors: int = 5):
    # Convert skin tone into a numpy array
    skin_tone = np.array(skin_tone).reshape(1, -1)
    
    # Find the nearest neighbors
    distances, indices = knn.kneighbors(skin_tone, n_neighbors=n_neighbors)
    
    # Get recommended item IDs
    recommended_items = item_data.iloc[indices[0]]['image_url'].tolist()
    
    return recommended_items

# Define an endpoint for getting recommendations based on skin tone
@app.post("/recommend/")
def recommend(skin_tone_request: SkinToneRequest):
    # Convert hex to RGB
    skin_tone_rgb = hex_to_rgb(skin_tone_request.skin_tone)
    
    # Get recommendations
    recommendations = get_recommendations(skin_tone_rgb)
    
    return {"recommended_items": recommendations}

