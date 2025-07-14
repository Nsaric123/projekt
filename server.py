from flask import Flask, request, jsonify
import torch
from model import NeuralNet
from nltk_utils import tokenize, bag_of_words
import json
import random
#import nltk
#nltk.download("punkt", quiet=True)

app = Flask(__name__)

# Učitaj model i podatke
with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

bot_name = "MATHOSBot"

@app.route("/chat", methods=["POST"])
def chat():
    req = request.get_json()
    sentence = req.get("message")

    if not sentence:
        return jsonify({"error": "No message provided"}), 400

    sentence_tokens = tokenize(sentence)
    X = bag_of_words(sentence_tokens, all_words)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return jsonify({
                    "bot": random.choice(intent["responses"])
                })

    return jsonify({
        "bot": "Nažalost, ne razumijem pitanje. Možeš probati drukčije formulirati?"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
