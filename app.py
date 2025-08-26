from flask import Flask, render_template, request
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# Initialize Flask app
app = Flask(__name__)

# Load trained model and tokenizer
model_path = "roberta_fake_news_model"
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)

import torch
import torch.nn.functional as F

# Make sure model and tokenizer are already loaded globally:
# tokenizer = ...
# model = ...
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict(texts):
    """
    Predicts whether texts are 'Fake' or 'Real'.

    Args:
        texts (str or list of str): Single text or list of texts.

    Returns:
        list of tuples: Each tuple is (label, confidence_score)
    """
    if isinstance(texts, str):
        texts = [texts]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)  # Get probabilities for each class
        confidences, predictions = torch.max(probs, dim=1)

    labels = ["Fake", "Real"]  # Adjust if your classes are switched

    results = []
    for pred, conf in zip(predictions.cpu(), confidences.cpu()):
        label = labels[pred.item()]
        confidence_score = conf.item()
        results.append((label, confidence_score))

    return results if len(results) > 1 else results[0]


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")  # or Home.html if you renamed it

@app.route("/check", methods=["GET", "POST"])
def check():
    result = None
    if request.method == "POST":
        user_input = request.form["news_text"]
        result = predict(user_input)
    return render_template("checknews.html", result=result)

@app.route("/about")
def about():
    return render_template("aboutus.html")

if __name__ == "__main__":
    app.run(debug=True)
