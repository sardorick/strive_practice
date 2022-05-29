#import the needed libraries

from turtle import color
from sklearn.metrics import r2_score
import data_handler as dh           # yes, you can import your code. Cool!
import torch
import torch.optim as optim
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

from model import NeuralNetwork


# Remember to validate your model: model.eval .........with torch.no_grad() ......model.train

data = 'Chapter 03/03. MLP Regression/data/turkish_stocks.csv'

x_train, x_test, y_train, y_test = dh.load_data(pth=data)
x_train_b, x_test_b, y_train_b, y_test_b = dh.to_batches(x_train, x_test, y_train, y_test, batch_size=20)
print(x_train.shape)
model = NeuralNetwork(8, 200, 100)

optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.L1Loss()

def torch_fit(model, x_train, x_test, y_train, y_test, x_train_b, x_test_b, y_train_b, y_test_b, num_epochs):
    train_loss_list = []
    test_loss_list = []
    benchmark = 0.7
    for epoch in range(num_epochs):
        print(f"Current epoch: {epoch+1}/{num_epochs}")
        current_loss = 0
        for batch, (x_train_b, y_train_b) in enumerate(zip(x_train, y_train)):
            optimizer.zero_grad()
            train_pred = model(x_train_b)

            train_loss = criterion(train_pred, y_train_b)
            current_loss += train_loss.item()

            train_loss.backward()

            optimizer.step()
            # print(f"Current loss: {train_loss.item() :.4f}")
        train_loss_list.append(current_loss/x_train.shape[0])

        # test

        model.eval()
        with torch.no_grad():
            current_loss = 0
            current_r2 = 0
            r2_scores = []
            for batch_t, (x_test_b, y_test_b) in enumerate(zip(x_test, y_test)):
                test_pred = model(x_test_b)

                # current_r2 += r2_score(y_test_b, test_pred)

                test_loss = criterion(test_pred, y_test_b)
                current_loss += test_loss.item()
            # r2_scores.append(current_r2/x_test.shape[0])
            test_loss_list.append(current_loss/x_test.shape[0])

            # check best model
            if test_loss_list[-1] > benchmark:
                torch.save(model, "model.pth")

        model.train()
    x_axis = list(range(num_epochs))
    plt.subplot(1,2,1)
    plt.plot(x_axis, train_loss_list, marker='o', label='Train loss')
    plt.plot(x_axis, test_loss_list, marker='o', label='Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')

    # # r2 score plt
    # plt.subplot(1, 2, 2)
    # plt.plot(x_axis, r2_scores, marker='o', color='green', label='r2 scores')
    # plt.xlabel('Epoch')
    # plt.ylabel('R2 error')
    # plt.axhline(benchmark, c='red', label=f'Benchmark score{benchmark}')
    # plt.legend(loc='best')

model = torch_fit(model, x_train, x_test, y_train, y_test, x_train_b, x_test_b, y_train_b, y_test_b, num_epochs=50)



    


        
