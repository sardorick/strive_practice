from matplotlib.pyplot import cla
import torch
import torch.nn.functional as F
from torch import nn
from torchsummary import summary


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_1, hidden_2):
        super(NeuralNetwork, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_1)
        self.hidden2 = nn.Linear(hidden_1, hidden_2)
        self.output = nn.Linear(hidden_2, 1)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        layer_1 = self.hidden1(x)
        act_1 = self.tanh(layer_1)
        layer_2 = self.hidden2(act_1)
        act_2 = self.tanh(layer_2)
        output = self.output(act_2)
        return output

model = NeuralNetwork(10, 20, 20)

# print(model)




        