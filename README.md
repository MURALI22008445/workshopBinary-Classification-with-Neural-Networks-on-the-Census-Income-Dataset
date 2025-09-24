# Binary Classification with Neural Networks on the Census Income Dataset
## AIM : 
To perform Binary Classification with Neural Networks on the Census Income Dataset
## Software Required:
Anaconda - Python 3.7
## Algorithm:
### Step1:
Data Preparation

### Step2:
Train/Test Split

### Step3:
Model Building

### Step4:
Model Training

### Step5:
Model Evaluation & Visualization
```
### PROGRAM:
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time

## Load dataset
data = pd.read_csv("income.csv")

## Define columns
cat = ['sex','education','marital-status','workclass','occupation']
target = ['label']
con = ['age','hours-per-week']

print(f'cat_cols  has {len(cat)} columns')
print(f'cont_cols has {len(con)} columns')
print(f'y_col     has {len(target)} column')

## Convert categorical columns to category type
for col in cat:
    data[col] = data[col].astype('category')

## Embedding sizes
cat_szs = [len(data[col].cat.categories) for col in cat] 
emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs] 

## Convert categorical data to tensor
cats = np.stack([data[col].cat.codes.values for col in cat], 1)
cats_tensor = torch.tensor(cats, dtype=torch.int64)

## Convert continuous data to tensor
conts = np.stack([data[col].values for col in con], 1)
conts_tensor = torch.tensor(conts, dtype=torch.float32)

# Target tensor
y = torch.tensor(data['label'].values, dtype=torch.long)

# Train/Test split
split = 30000
cats_train, cats_test = cats_tensor[:split], cats_tensor[split:]
conts_train, conts_test = conts_tensor[:split], conts_tensor[split:]
y_train, y_test = y[:split], y[split:]

# Define Tabular Model
class TabularModel(nn.Module):
    def __init__(self, emb_szs, n_cont, out_sz, layers, p=0.5):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_szs])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        
        layerlist = []
        n_emb = sum((nf for ni, nf in emb_szs))
        n_in = n_emb + n_cont
        for i in layers:
            layerlist.append(nn.Linear(n_in, i)) 
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1], out_sz))
        self.layers = nn.Sequential(*layerlist)
    
    def forward(self, x_cat, x_cont):
        embeddings = [e(x_cat[:, i]) for i, e in enumerate(self.embeds)]
        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)
        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)
        return self.layers(x)

## Initialize model, loss, optimizer
model = TabularModel(emb_szs, n_cont=2, out_sz=2, layers=[50], p=0.4)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

## Training loop
start_time = time.time()
epochs = 300
losses = []

for i in range(epochs):
    y_pred = model(cats_train, conts_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss)
    
    if i % 25 == 1:
        print(f'epoch: {i:3}  loss: {loss.item():10.8f}')
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'epoch: {i:3}  loss: {loss.item():10.8f}')
print(f'\nDuration: {time.time() - start_time:.0f} seconds')

## Plot training loss
losses_np = [loss.detach().cpu().numpy() for loss in losses]
plt.plot(losses_np)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

## Evaluation
model.eval()
with torch.no_grad():
    y_pred_test = model(cats_test, conts_test)
    loss = criterion(y_pred_test, y_test)

print(f'Test CE Loss: {loss.item():.8f}')

y_pred_labels = torch.argmax(y_pred_test, dim=1)
correct = (y_pred_labels == y_test).sum().item()
accuracy = correct / y_test.size(0) * 100
print(f'Accuracy: {accuracy:.2f}%')
### PROGRAM:
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time

# Load dataset
data = pd.read_csv("income.csv")

# Define columns
cat = ['sex','education','marital-status','workclass','occupation']
target = ['label']
con = ['age','hours-per-week']

# Convert categorical columns
for col in cat:
    data[col] = data[col].astype('category')

# Embedding sizes
cat_szs = [len(data[col].cat.categories) for col in cat]
emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]

# Convert to tensors
cats = np.stack([data[col].cat.codes.values for col in cat], 1)
cats_tensor = torch.tensor(cats, dtype=torch.int64)
conts = np.stack([data[col].values for col in con], 1)
conts_tensor = torch.tensor(conts, dtype=torch.float32)
y = torch.tensor(data['label'].values, dtype=torch.long)

# Train/Test split
split = 30000
cats_train, cats_test = cats_tensor[:split], cats_tensor[split:]
conts_train, conts_test = conts_tensor[:split], conts_tensor[split:]
y_train, y_test = y[:split], y[split:]

# Model
class TabularModel(nn.Module):
    def __init__(self, emb_szs, n_cont, out_sz, layers, p=0.5):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_szs])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        layerlist = []
        n_emb = sum((nf for ni, nf in emb_szs))
        n_in = n_emb + n_cont
        for i in layers:
            layerlist.append(nn.Linear(n_in, i))
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1], out_sz))
        self.layers = nn.Sequential(*layerlist)
    def forward(self, x_cat, x_cont):
        embeddings = [e(x_cat[:, i]) for i, e in enumerate(self.embeds)]
        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)
        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)
        return self.layers(x)

# Initialize
model = TabularModel(emb_szs, n_cont=2, out_sz=2, layers=[50], p=0.4)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

## Training
epochs = 300
losses = []
start_time = time.time()
for i in range(epochs):
    y_pred = model(cats_train, conts_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss)
    if i % 25 == 1:
        print(f'epoch: {i:3}  loss: {loss.item():10.8f}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(f'epoch: {i:3}  loss: {loss.item():10.8f}')
print(f'\nDuration: {time.time() - start_time:.0f} seconds')

## Plot loss
losses_np = [loss.detach().cpu().numpy() for loss in losses]
plt.plot(losses_np)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

## Evaluation
model.eval()
with torch.no_grad():
    y_pred_test = model(cats_test, conts_test)
    loss = criterion(y_pred_test, y_test)
print(f'Test CE Loss: {loss.item():.8f}')
y_pred_labels = torch.argmax(y_pred_test, dim=1)
correct = (y_pred_labels == y_test).sum().item()
accuracy = correct / y_test.size(0) * 100
print(f'Accuracy: {accuracy:.2f}%')
```
