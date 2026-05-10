from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pickle
import pandas as pd
import io
from datasets import load_dataset

app = FastAPI()

# Load TF-IDF pipeline
with open("tfidf_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Load label names
dataset = load_dataset("banking77")
label_names = dataset['train'].features['label'].names

class TicketRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: TicketRequest):
    predicted_class = pipeline.predict([request.text])[0]
    confidence = pipeline.predict_proba([request.text]).max()

    return {
        "intent": label_names[predicted_class],
        "confidence": round(float(confidence), 4)
    }

@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    results = []
    for text in df["text"]:
        predicted_class = pipeline.predict([text])[0]
        confidence = pipeline.predict_proba([text]).max()
        results.append({
            "text": text,
            "intent": label_names[predicted_class],
            "confidence": round(float(confidence), 4)
        })
        
    return results