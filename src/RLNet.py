import torch
from torch import nn


class RL_net(nn.Module):
    def __init__(self, moves_num):
        super(RL_net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(5, 5, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=(11, 11, 1)))
        self.layer2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(11, 11, 1), stride=(3, 3, 1), padding=2),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=(3, 3, 1)))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(15360, 100)
        self.fc2 = nn.Linear(100, moves_num)

    def forward(self, input_o):
        x = self.layer1(input_o)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.drop_out(x)
        x = self.fc1(x)
        out = self.fc2(x)
        return out
