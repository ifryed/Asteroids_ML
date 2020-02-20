import os
from typing import Callable

import numpy as np
import torch
from torch import nn
import torchsummary.torchsummary
from torch.distributions import Categorical

import RLNet
from AstroGame.StarShip import SpaceShipActions
from AstroGame.utils import STATE_NET_PATH, PLAYER_NET_PATH

LEARN = True
GAMMA = 0.9
BETA = 1


class AutoPlayer:
    rand_ratio = 10.7

    def __init__(self):
        self.moves = []

        self.pred_heap = []
        self.current_score = 0

    def setMoves(self, move_lst: list):
        self.moves = move_lst

    def sampleAction(self, action_vec) -> int:
        # return SpaceShipActions.ACCELERATE
        rand_fraction = np.random.random()
        eps = 0.000
        acc_prob = action_vec.cumsum(1) + eps
        acc_prob = acc_prob / acc_prob.max()
        for i, prob in enumerate(acc_prob[0]):
            if rand_fraction < prob + eps:
                return i

    def smartMove(self, observation) -> None:
        self.optimizer_player.zero_grad()
        # Choose action
        tensor_img_action = torch.from_numpy(observation).to(self.device).float()

        # Retrieve Action
        self.score_pred = self.model_player(tensor_img_action)
        self.m = Categorical(self.score_pred)
        self.action_idx = self.m.sample()

        # Execute action
        shoot_penalty = 0
        if self.action_idx == 0:
            self.moves[self.action_idx](self.ge)
            shoot_penalty -= .1
        else:
            self.moves[self.action_idx]()

    def updateWeights(self, reward, old_state, new_state, not_final_state: bool):
        # Get State estimation
        new_state = torch.from_numpy(new_state).to(self.device).float()
        old_state = torch.from_numpy(old_state).to(self.device).float()
        new_state_val_pred = self.model_state(new_state)

        self.optimizer_state.zero_grad()
        old_state_val_pred = self.model_state(old_state)

        state_score = reward + GAMMA * (new_state_val_pred - old_state_val_pred)

        # Update results
        loss_player = -self.m.log_prob(self.action_idx) * old_state_val_pred.item()
        # loss_player = self.loss_player(self.score_pred, score_loss)
        p_loss = loss_player.item()
        # Backprop and perform Adam optimisation
        loss_player.backward()
        self.optimizer_player.step()

        # Update state net
        loss_state = self.loss_state(old_state_val_pred, state_score)
        q_loss = loss_state.item()
        # Backprop and perform Adam optimisation
        loss_state.backward()
        self.optimizer_state.step()
        print("\rScore:", self.ge.score, "Act:", np.array(self.score_pred.tolist()[0]), self.action_idx.item(),
              " P-Loss: {:.2f} Q-Loss: {:.2f} S:{:.2f}".format(p_loss, q_loss, old_state_val_pred.item()), end='')

        if not not_final_state:
            self.optimizer_state.zero_grad()
            new_state_val_pred = self.model_state(new_state)
            loss_state = self.loss_state(new_state_val_pred, torch.tensor(reward, dtype=torch.float).to(self.device))
            # Backprop and perform Adam optimisation
            loss_state.backward()
            self.optimizer_state.step()

            print("\nFinal Q:", new_state_val_pred[0][0].item())
            print("Final Loss:", loss_state.item())

    def saveNet(self):
        if LEARN:
            torch.save(self.model_player, PLAYER_NET_PATH)
            torch.save(self.model_state, STATE_NET_PATH)

    def setupDNN(self):
        # Setting the NN
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        if os.path.exists(PLAYER_NET_PATH):
            self.model_player = torch.load(PLAYER_NET_PATH)
        else:
            self.model_player = RLNet.Player_net(len(self.moves)).to(self.device)

        if os.path.exists(STATE_NET_PATH):
            self.model_state = torch.load(STATE_NET_PATH)
        else:
            self.model_state = RLNet.State_net().to(self.device)

        print(torchsummary.summary(self.model_state, (2, 400, 300, 3)))

        self.model_player = self.model_player.float()
        self.loss_player = nn.MSELoss()
        learning_rate_player = 1e-4
        self.optimizer_player = torch.optim.Adam(self.model_player.parameters(), lr=learning_rate_player)

        self.model_state = self.model_state.float()
        self.loss_state = nn.L1Loss()
        learning_rate_state = 1e-1
        self.optimizer_state = torch.optim.Adam(self.model_state.parameters(), lr=learning_rate_state)
