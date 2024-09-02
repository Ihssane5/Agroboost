import json
import numpy as np
from Sentence_Processing import Tokenize,Stem,Bag_of_words
from model import NeuralNet
#from ChatDataset import ChatDataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib



with open('./data/intents.json','r') as f:
    intents = json.load(f)  #Dictionnary with one key intents

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['pattern']:
        # create an instanse
        tokenized_sentence = Tokenize(pattern)
        all_words.extend(tokenized_sentence)
        # matching the tokenized sentence with the tag 
        xy.append((tokenized_sentence, tag))

ignore_words = ['?', '.', '!',  ',']

all_words = [Stem(word) for word in all_words if word not in ignore_words]


all_words = sorted(set(all_words))
tags = sorted(set(tags))
#Train DataSet

x_train = []
y_train = []

for pattern_sentence,tag in xy:
    bag = Bag_of_words(pattern_sentence,all_words)
    x_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)
#convert xtrain and ytrain into numpy array 
x_train = np.array(x_train)
y_train = np.array(y_train)

#class 

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

    def __len__(self):
        return self.n_samples


#hyperparameter
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(all_words)
learning_rate = 0.001
num_epochs = 1000
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

model = NeuralNet(input_size, hidden_size,output_size)

device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# loss and optimizer

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

for param in model.parameters():
    assert param.requires_grad, "Parameter does not require gradient"

#traning loop

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device).long()  # Convert labels to LongTensor

        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch + 1}/{num_epochs}, loss={loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')

data = { "model_state": model.state_dict(),
        "input_size": input_size, 
        "output_size": output_size,
        "hidden_size": hidden_size,
        "all_words": all_words,
        "tags" : tags,
    }

FILE = "./models/chat.pkl"
joblib.dump(data, FILE)
print(f"training complete, file saved to {FILE}")
print(f"all words {data['all_words']}")


    












