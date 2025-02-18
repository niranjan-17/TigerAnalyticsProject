import streamlit as st
import pickle
# Get the absolute path of the 'src' directory dynamically
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "MLE_TakeHomeAssignment-DEV", "src"))

# Add it to the Python path
sys.path.append(project_root)

# Import DataLoader correctly
from text_loader.loader import DataLoader


def get_prediction(input_text):
    """Loads model, vectorizer, and encoder to make a prediction."""
    with open("xgb_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    with open("encoder.pkl", "rb") as encoder_file:
        encoder = pickle.load(encoder_file)

    # Preprocess input
    cleaned_text = DataLoader.remove_characters(input_text)
    vectorized_text = vectorizer.transform([cleaned_text]).toarray()

    # Predict
    prediction = model.predict(vectorized_text)
    predicted_label = encoder.inverse_transform(prediction)[0]

    return predicted_label


# Streamlit page configuration
st.set_page_config(page_title="Tweet Classifier", layout="wide")

# Streamlit UI components
st.title("Classify your tweet")

# User inputs the tweet
tweet_input = st.text_input("Enter your tweet", "")

# Button to trigger prediction
if st.button("Classify Tweet"):
    if tweet_input.strip():
        prediction = get_prediction(tweet_input)
        st.success(f"Prediction: {prediction}")
    else:
        st.warning("Please enter a tweet before classifying.")
