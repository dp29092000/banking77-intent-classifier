from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from transformers import AutoTokenizer, DistilBertForSequenceClassification
import torch
import pandas as pd
import io

app = FastAPI()

# Load model and tokenizer
model_path = "dp29092k/banking77-intent-classifier"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Label names - same order as Banking77
from datasets import load_dataset
dataset = load_dataset("banking77")
label_names = dataset['train'].features['label'].names

class TicketRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: TicketRequest):
    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    confidence = torch.softmax(logits, dim=1).max().item()

    return {
        "intent": label_names[predicted_class],
        "confidence": round(confidence, 4)
    }

@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    results = []
    for text in df["text"]:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        confidence = torch.softmax(logits, dim=1).max().item()

        results.append({
            "text": text,
            "intent": label_names[predicted_class],
            "confidence": round(confidence, 4)
        })

    return results
