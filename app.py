from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
    
# 1. Initialize FastAPI
app = FastAPI(
    title="Second-Hand Car Recommendation System", 
    description="An API to recommend top 5 second-hand cars based on user budget and preferences.", 
    version="1.0"
)

# 2. Load the trained model when the server starts
try:
    # Make sure 'car_recommendation_model.pkl' is in the exact same folder as this app.py file
    model_pipeline = joblib.load('car_recommendation_model.pkl')
    model_classes = model_pipeline.classes_
    print("Model loaded successfully!")
except Exception as e:
    model_pipeline = None
    print(f"Error loading model: {e}")

# 3. Define the expected input from the user using Pydantic
class CarRequest(BaseModel):
    price: int
    manufacturing_year: int
    km_driven: float
    fuel_type: str
    transmission_type: str
    brand: str
    city: str
    bodytype: str

# 4. Create the POST endpoint
@app.post("/recommend")
def get_recommendations(request: CarRequest):
    # Security check: Did the model actually load?
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model pipeline failed to load on the server.")

    try:
        # Convert the validated user input directly into a 1-row Pandas DataFrame
        input_df = pd.DataFrame([request.model_dump()])

        # Get the confidence percentages (probabilities) for every single car
        probs = model_pipeline.predict_proba(input_df)[0]

        # Sort the probabilities highest to lowest, and grab the top 5
        top5_indices = np.argsort(probs)[::-1][:5]
        top5_cars = model_classes[top5_indices]
        top5_probs = probs[top5_indices]

        # Format the output cleanly for the frontend to consume
        recommendations = []
        for i in range(5):
            recommendations.append({
                "rank": i + 1,
                "model": str(top5_cars[i]),
                "confidence_score": round(float(top5_probs[i]) * 100, 2)
            })

        return {
            "status": "success",
            "user_input": request.model_dump(),
            "top_5_recommendations": recommendations
        }

    except Exception as e:
        # If anything goes wrong during prediction, send the error back to the user
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")