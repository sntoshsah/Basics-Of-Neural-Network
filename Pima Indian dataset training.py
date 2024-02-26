import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from Intro2NeuralNet import model

dataset = np.loadtxt('pima-indian-diabeties.csv', delimiter=',')
x_data = torch.Tensor(dataset[:,0:8])
y_data = torch.Tensor(dataset[:,8]).reshape(-1,1)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 100
batch_size = 32

for epoch in range(500):
    for i in range(0,len(x_data),batch_size):
        x_batch = x_data[i:i+batch_size]
        y_pred = model(x_batch)
        y_batch  = y_data[i:i+batch_size]
        loss = criterion(y_pred,y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Finished epoch {epoch}, latest loss {loss}')

# Test the model     
i = 5
X_sample = x_data[i:i+1]
y_samPred = model(X_sample)
print(f"{X_sample[0]} -> {y_samPred[0]}")

# Test the model with eval mode
model.eval()
with torch.no_grad():
    y_predwithoutgrad = model(X_sample)
y_samPred = model(X_sample)
print(f"{X_sample[0]} -> {y_samPred[0]}")


#Accuracy
model.eval()
with torch.no_grad():
    y_pred = model(X_sample)
accuracy = (y_pred.round() == y_data[i:i+1]).float().mean()
print(f"Accuracy {accuracy}")