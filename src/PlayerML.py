import os
from typing import Callable

import numpy as np
import torch
from torch import nn

import RLNet

LEARN = True


class AutoPlayer:
    rand_ratio = 0.6

    def __init__(self):
        self.moves = []

        self.pred_heap = []
        self.current_score = 0

    def setMoves(self, move_lst: list):
        self.moves = move_lst

    def smartMove(self, observation, reward, done) -> None:
        # Choose action
        tensor_img = torch.from_numpy(observation).to(self.device)

        score_pred = self.model_player(tensor_img.float())
        if np.random.random() > self.rand_ratio:
            move_idx = np.random.randint(0, len(self.moves))
        else:
            move_idx = torch.argmax(score_pred)
        # Execute action
        if move_idx == 0:
            self.moves[move_idx](self.ge)
            reward -= .1
        else:
            self.moves[move_idx]()

        # Update results
        score_loss = score_pred.clone()
        score_loss[0, move_idx] = reward
        loss = self.criterion_player(score_pred, score_loss)
        loss_flt = loss.item()
        # Backprop and perform Adam optimisation
        self.optimizer_player.zero_grad()
        loss.backward()
        self.optimizer_player.step()

        print("\tLoss: {:.2f}".format(loss_flt), end='')
        self.pred_heap = []

        AutoPlayer.rand_ratio = min(AutoPlayer.rand_ratio + 5e-6, 0.8)

    def saveNet(self):
        if LEARN:
            torch.save(self.model_player, '../player.mdl')

    def setupDNN(self):
        # Setting the NN
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        if os.path.exists('../player.mdl'):
            self.model_player = torch.load('../player.mdl')
        else:
            self.model_player = RLNet.Player_net(len(self.moves)).to(self.device)
        self.model_player = self.model_player.float()
        self.criterion_player = nn.MSELoss()
        learning_rate_player = 1e-5
        self.optimizer_player = torch.optim.Adam(self.model_player.parameters(), lr=learning_rate_player)

        self.model_state = RLNet.State_net(len(self.moves)).to(self.device)
        self.model_state = self.model_state.float()
        self.criterion_state = nn.MSELoss()
        learning_rate_state = 0.01
        self.optimizer_state = torch.optim.Adam(self.model_state.parameters(), lr=learning_rate_state)
