import random 
import joblib
import json
import torch
from model import NeuralNet
from Sentence_Processing import Tokenize,Bag_of_words,Stem

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('data/intents.json', 'r') as f:
    intents = json.load(f)

File = "models/chat.pkl"

data = joblib.load(File)

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
model_state = data['model_state']
tags = data['tags']

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

#running the chat

bot_name = "Sam"
print("Let's chat! type 'quit' to exit")
while True:
    sentence = input('You: ')
    if sentence == "quit":
        break
    sentence = Tokenize(sentence)
    X = Bag_of_words(sentence,all_words)
    X = X.reshape(1,X.shape[0])
    X = torch.from_numpy(X)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f'{bot_name}: {random.choice(intent["answers"])}')
    else:
       print(f"{bot_name}: I’m here to assist you as a chatbot; if I can’t address your query, please contact our support team via the 'Contact Us' link for further assistance.")      
