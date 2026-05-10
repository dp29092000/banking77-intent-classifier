# 🏦 Banking77 Intent Classifier

An end-to-end NLP system that automatically classifies customer support tickets into 77 banking intent categories - built to simulate how fintech companies route support tickets automatically.

---

## 📌 Project Description

Customer support teams at fintech companies receive thousands of tickets daily. Manually routing each ticket to the right team is slow and error-prone. This project builds an NLP-based intent classifier that automatically routes incoming support tickets to the correct department based on customer intent.

The system is built on the **Banking77** dataset — a benchmark dataset of 13,083 banking customer support queries across 77 intent categories.

---

## 🏗️ Architecture

User Input (Ticket Text) → Streamlit Frontend → FastAPI Backend (Render) → TF-IDF Pipeline → Predicted Intent + Confidence Score

---

## 📊 Model Performance

| Model | Accuracy | Macro F1 |
|---|---|---|
| TF-IDF + Logistic Regression | 86% | 0.86 |
| Fine-tuned DistilBERT | 91% | 0.91 |

> DistilBERT was fine-tuned locally on the Banking77 dataset. The deployed API uses the TF-IDF pipeline for lightweight serving - a common production tradeoff between model performance and infrastructure cost.

---

## 🛠️ Tech Stack

- **NLP**: Scikit-learn (TF-IDF), HuggingFace Transformers (DistilBERT)
- **Backend**: FastAPI, Uvicorn
- **Frontend**: Streamlit
- **Deployment**: Render (API), Streamlit Cloud (Frontend)
- **Model Registry**: HuggingFace Hub

---

## 📁 Project Structure

    banking77-intent-classifier/
    ├── pages/
    │   ├── 1_EDA.py
    │   ├── 2_Model_Comparison.py
    │   ├── 3_Single_Prediction.py
    │   └── 4_Batch_Prediction.py
    ├── app.py
    ├── main.py
    ├── tfidf_pipeline.pkl
    ├── requirements.txt
    └── README.md

---

## 🚀 How to Run Locally

1. Clone the repository
```bash
git clone https://github.com/dp29092000/Banking77-intent-classifier.git
cd Banking77-intent-classifier
```

2. Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run FastAPI backend
```bash
uvicorn main:app --reload
```

5. Run Streamlit frontend
```bash
streamlit run app.py
```

---

## 🌐 Live Deployment

| Service | URL |
|---|---|
| FastAPI Backend | https://banking77-intent-classifier.onrender.com/docs |
| Streamlit App | https://banking77-intent-classifier-gxvu9z7bbgsdffjjhx9tki.streamlit.app/ |

---

## Author
**Prasanna D**  
IIT Gandhinagar (B.Tech, 2022)  
Incoming MS Applied Machine Learning - University of Maryland, Fall 2026