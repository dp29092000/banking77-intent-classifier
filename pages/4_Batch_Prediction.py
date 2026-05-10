import streamlit as st
import requests

st.set_page_config(page_title="Batch Predict", page_icon="🔮", layout="wide")
st.title("🔮 Batch Prediction")
st.divider()

input_file = st.file_uploader("Upload Batch File")

if st.button("Predict Batch Intent", type="primary"):
    response = requests.post("https://banking77-intent-classifier.onrender.com/predict-batch",files={"file": input_file})
    result = response.json()
    st.dataframe(result)
    