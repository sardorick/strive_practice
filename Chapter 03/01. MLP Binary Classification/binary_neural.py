import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
np.random.seed(0)

class Classifier(nn.Module):
    def __init__(self, input_dim, numhidden1, numhidden2, numhidden3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, numhidden1)
        self.fc2 = nn.Linear(numhidden1, numhidden2)
        self.fc3 = nn.Linear(numhidden2, numhidden3)
        self.fc4 = nn.Linear(numhidden3, 1)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        layer_1 = self.fc1(x)
        act_1 = self.sigmoid(layer_1)
        layer_2 = self.fc2(act_1)
        act_2 = self.sigmoid(layer_2)
        layer_3 = self.fc3(act_2)
        act_3 = self.sigmoid(layer_3)
        layer_4 = self.fc4(act_3)
        output = self.sigmoid(layer_4)
        return output

model = Classifier(2, 10, 10, 10)

epochs = 100

criterion = nn.BCELoss()
data = pd.read_csv('Chapter 03/01. MLP Binary Classification/data.csv', header=None)
x = torch.tensor(data.drop(2, axis=1).values, dtype=torch.float)
y = torch.tensor(data[2].values, dtype=torch.float).reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)

def fit(x_train, x_test, y_train, y_test, model, criterion, lr, num_epochs):
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        #train
        optim.zero_grad()
        train_pred = model(x_train)
        loss_value = criterion(train_pred, y_train)
        train_losses.append(loss_value.detach().item())
        loss_value.backward()
        optim.step()
        
        # evaluate
        model.eval()
        with torch.no_grad():
            test_preds = model.forward(x_test)
            test_loss = criterion(test_preds, y_test)
            test_losses.append(test_loss.item())

        # check model accuracy
        pred_label = []
        for pred in test_preds:
            pred_label.append(1) if pred > 0.5 else pred_label.append(0)
        pred_label = torch.tensor(pred_label).reshape(-1, 1)
        acc = (pred_label == y_test).sum() / len(y_test)
        print(f'Accuracy of the model for the epoch {epoch} is {acc}')


        model.train()
    # print(acc)
    # plt.plot(train_losses)
    # plt.plot(test_losses)
    # plt.show()

model = fit(x_train, x_test, y_train, y_test, model, criterion, lr=0.01, num_epochs=1000)

# print(y_test.shape)