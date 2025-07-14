import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Učitaj model i podatke
with open("intents.json", "r", encoding="utf-8") as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

# Inicijalizacija modela
model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

bot_name = "MATHOSBot"
print(f"{bot_name} je spreman za razgovor! (upiši 'kraj' za izlaz)\n")

while True:
    sentence = input("Ti: ")
    if sentence.lower() in ["kraj", "izlaz", "quit", "exit"]:
        print(f"{bot_name}: Doviđenja! 👋")
        break

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
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: Nažalost, ne razumijem pitanje. Možeš probati drukčije formulirati?")
