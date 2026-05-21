from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

app = FastAPI(title="FinSentinel API")

MODEL_PATH = "/Users/saumyagoyal/finsentinel/model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

LABELS = {0: "negative", 1: "neutral", 2: "positive"}

class TextInput(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: list[str]

@app.get("/health")
def health():
    return {"status": "ok", "model": "FinSentinel — Fine-tuned FinBERT"}

@app.post("/predict")
def predict(input: TextInput):
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1).squeeze().tolist()
    pred = torch.argmax(outputs.logits, dim=1).item()
    return {
        "text": input.text,
        "sentiment": LABELS[pred],
        "confidence": round(max(probs), 4),
        "scores": {
            "negative": round(probs[0], 4),
            "neutral": round(probs[1], 4),
            "positive": round(probs[2], 4)
        }
    }

@app.post("/predict_batch")
def predict_batch(input: BatchInput):
    results = []
    for text in input.texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).squeeze().tolist()
        pred = torch.argmax(outputs.logits, dim=1).item()
        results.append({
            "text": text,
            "sentiment": LABELS[pred],
            "confidence": round(max(probs), 4),
            "scores": {
                "negative": round(probs[0], 4),
                "neutral": round(probs[1], 4),
                "positive": round(probs[2], 4)
            }
        })
    return {"results": results}
