from Model import Model
import pandas as pd
import numpy as np
import torch
import torch.nn as nn


# Naive Manual Test
winedata = pd.read_csv('edited_wine.csv')
wine_org = pd.read_csv('edited_wine.csv')
wine = winedata['Wine']
winedata.drop('Wine', axis=1, inplace=True)
wine_org.drop('Wine', axis=1, inplace=True)
means = winedata.mean()
stds = winedata.std()
winedata = (winedata - means) / stds
wine_org = (wine_org - means) / stds

winedata_t = torch.tensor(winedata.values, dtype=torch.float32)
wine_org_t = torch.tensor(wine_org.values, dtype=torch.float32)
wine_t = torch.tensor(wine.values)
wine_dataset = torch.utils.data.TensorDataset(winedata_t, wine_t)
wine_loader = torch.utils.data.DataLoader(wine_dataset, batch_size=10, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(13, 20)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(20, 20)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(20, 3)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

model = Model(model, optimizer, criterion)
model.fit(wine_loader, 100)
print("FITTING IS FINE")
predictions = model.predict(wine_org_t)
print("PREDICTING IS FINE")
preds = torch.argmax(predictions, 1)
print(preds)
print(wine)
