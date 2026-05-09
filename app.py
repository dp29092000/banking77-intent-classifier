import streamlit as st

st.set_page_config(
                page_title = 'Intent Classifier for Banking Tickets',
                page_icon = "🏦",
                layout = "wide"
)

st.title(" 🏦 Intent Classifier for Banking Support Tickets")
st.markdown("### Automated Support Routing System using NLP")
st.divider()

st.markdown("## Problem Statement")
st.write("""
Customer support teams at fintech companies receive thousands of tickets daily. 
Manually routing each ticket to the right team is slow and error-prone. 
This project builds an NLP-based intent classifier that automatically routes incoming support tickets 
to the correct department based on customer intent, thus reducing manual effort and improving response time.
""")
st.divider()

st.markdown("## Dataset")
col1, col2, col3 = st.columns(3)
col1.metric("Dataset","Banking77")
col2.metric("Total Tickets","13,083")
col3.metric("Intent Classess","77")
st.divider()

st.markdown("## Models Built")
col1, col2 = st.columns(2)
with col1:
     st.success("**Baseline: TF-IDF + Logistic Regression**\n\nAccuracy: 86%")
with col2:
    st.success("**Fine-tuned DistilBERT**\n\nAccuracy: 91%")
st.divider()

st.markdown("## Tech Stack")
col1, col2, col3, col4 = st.columns(4)
col1.info("🤗 HuggingFace Transformers")
col2.info("⚡ FastAPI")
col3.info("🎈 Streamlit")
col4.info("🔥 PyTorch")
st.divider()

st.info("👈 Use the sidebar to explore EDA, Model Comparison, and LIVE Predictions.")