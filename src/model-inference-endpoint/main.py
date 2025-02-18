from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import os
import sys
# Get the absolute path of the 'src' directory dynamically
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "MLE_TakeHomeAssignment-DEV", "src"))

# Add it to the Python path
sys.path.append(project_root)

# Import DataLoader correctly
from text_loader.loader import DataLoader

class InputText(BaseModel):
    input_texts: str

app = FastAPI()

@app.get("/health")
def get_health():
    return {"status": "OK"}

@app.post("/get-prediction/")
def get_prediction(input_data: InputText):
    """Loads model, vectorizer, and encoder to make a prediction."""
    with open("xgb_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    with open("encoder.pkl", "rb") as encoder_file:
        encoder = pickle.load(encoder_file)

    # Preprocess input
    cleaned_text = DataLoader.remove_characters(input_data.input_texts)
    vectorized_text = vectorizer.transform([cleaned_text]).toarray()

    # Predict
    prediction = model.predict(vectorized_text)
    predicted_label = encoder.inverse_transform(prediction)[0]

    return {"prediction": predicted_label}
