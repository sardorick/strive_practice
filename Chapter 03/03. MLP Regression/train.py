#import the needed libraries

import data_handler as dh           # yes, you can import your code. Cool!
import torch
import torch.optim as optim
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

from model import NeuralNetwork


# Remember to validate your model: model.eval .........with torch.no_grad() ......model.train
model = NeuralNetwork(8, 10, 20)

data = 'Chapter 03/03. MLP Regression/data/turkish_stocks.csv'

x_train, x_test, y_train, y_test = dh.load_data(pth=data)

optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.L1Loss()

def torch_fit(model, x_train, x_test, y_train, y_test, num_epochs):
    train_loss = 0
    test_loss = 0
    for epoch in range(num_epochs):
        
        optimizer.zero_grad()
        
