from torch import nn
import torch.nn.functional as F


class Player_net(nn.Module):
    def __init__(self, moves_num):
        super(Player_net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=(5, 5, 1), stride=(2, 2, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3, 3, 2), stride=(3, 3, 1)))
        self.layer2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(5, 5, 1), stride=(2, 2, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3, 3, 1), stride=(3, 3, 1)))
        self.layer3 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(5, 5, 1), stride=(2, 2, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1)))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, moves_num)

    def forward(self, input_o):
        x = self.layer1(input_o)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(x.size(0), -1)
        x = self.drop_out(x)
        x = self.fc1(x)
        x = self.fc2(x)
        out = F.softmax(x, 1)
        return out


class State_net(nn.Module):
    def __init__(self):
        super(State_net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=(5, 5, 3), stride=(3, 3, 3)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3, 3, 1), stride=(3, 3, 1)))
        self.layer2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(5, 5, 1), stride=(3, 3, 3)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3, 3, 1), stride=(3, 3, 1)))
        self.layer3 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(5, 5, 1), stride=(3, 3, 3)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3, 3, 1), stride=(3, 3, 1)))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(768, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, input_o):
        x = self.layer1(input_o)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.drop_out(x)
        x = self.fc1(x)
        out = self.fc2(x)
        return out
