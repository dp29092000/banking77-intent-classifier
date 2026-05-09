import streamlit as st
import requests

st.set_page_config(page_title="Predict", page_icon="🔮", layout="wide")
st.title("🔮 NLP Prediction")
st.divider()

st.subheader("Enter Text Details")
text = st.text_area("Enter support ticket text")


if st.button("Predict Intent", type="primary"):
    response = requests.post("http://127.0.0.1:8000/predict",json={"text": text})
    result = response.json()
    confidence = result['confidence']
    if confidence > 0.6:
        st.success(f"**Intent:** {result['intent']}")
        st.info(f"**Confidence:** {result['confidence']}")
    else: 
        st.warning("Human Review needed")
        st.info(f"**Model's best guess:** {result['intent']} ({result['confidence']})")