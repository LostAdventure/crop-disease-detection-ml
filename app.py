"""Simple Streamlit interface for the trained crop disease classifier."""
import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

ARTIFACTS = Path("artifacts")
MODEL_PATH = ARTIFACTS / "crop_disease_model.keras"
CLASSES_PATH = ARTIFACTS / "class_names.json"
IMAGE_SIZE = (224, 224)

st.set_page_config(page_title="Crop Disease Detection", page_icon="🌿")
st.title("🌿 Crop Disease Detection System")
st.write(
    "Upload a clear image of a single crop leaf. The model will predict "
    "the closest matching PlantVillage class."
)

if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
    st.error(
        "Trained model files were not found. Run `python src/train.py` first "
        "to create the files inside the artifacts/ folder."
    )
    st.stop()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()
class_names = json.loads(CLASSES_PATH.read_text())

uploaded = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded leaf image", use_container_width=True)
    resized = image.resize(IMAGE_SIZE)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    probabilities = model.predict(np.expand_dims(array, axis=0), verbose=0)[0]
    best = int(np.argmax(probabilities))
    confidence = float(probabilities[best]) * 100

    st.subheader("Prediction")
    st.success(f"**{class_names[best].replace('___', ' — ').replace('_', ' ')}**")
    st.metric("Model confidence", f"{confidence:.2f}%")

    top_indices = np.argsort(probabilities)[::-1][:5]
    st.subheader("Top predictions")
    for index in top_indices:
        label = class_names[index].replace("___", " — ").replace("_", " ")
        st.write(f"{label}: {probabilities[index] * 100:.2f}%")

    st.warning(
        "This is an educational decision-support tool, not a substitute for "
        "diagnosis by an agricultural extension officer. PlantVillage images "
        "were largely collected under controlled conditions; confirm results "
        "in the field before applying treatment."
    )
