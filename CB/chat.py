import random
import json
import torch

from model import NeuralNet
from ex import bow,tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #if we have GPU support 

with open('CB.json','r') as f:
    intents = json.load(f)

FILE="data.pth"
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

bot_name = "PK"
print("Let's Chat! Type 'quit' to exit")
while True:
    sentence = input('You: ')
    if sentence == "quit":
        break
    
    sentence = tokenize(sentence)
    ab = bow(sentence, all_words)
    ab = ab.reshape(1, ab.shape[0])
    ab = torch.from_numpy(ab) #as bow returns numpy array
    output = model(ab) #gives us the prediction
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    #softmax
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() >0.725:
        #check if tag present in the intents and print that response
        for intent in intents["intents"]:
            if tag==intent["tag"]:
                print(f"{bot_name}:{random.choice(intent['responses'])}")

    else:
        print(f"{bot_name}: I do not understand.")