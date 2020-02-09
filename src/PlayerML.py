import os
from typing import Callable

import numpy as np
import torch
from torch import nn

import RLNet

LEARNING_PATH_LEN = 2
LEARN = True


class AutoPlayer:
    rand_ratio = 0.6

    def __init__(self):
        self.moves = []
        self.last_score = 0

        self.pred_heap = []
        self.new_snap_shot = None
        self.old_snap_shot = None
        self.toggle_learning = True
        self.current_score = 0

    def setMoves(self, move_lst: list):
        self.moves = move_lst

    def smartMove(self, end=False) -> None:
        self.toggle_learning = not self.toggle_learning
        if not end and self.toggle_learning:
            return
        # Get data
        h, w, z = self.new_snap_shot.shape

        # Choose action
        input_data = np.zeros((1, 2, h, w, z))
        input_data[0, 0, :, :, :] = self.old_snap_shot
        input_data[0, 1, :, :, :] = self.new_snap_shot
        tensor_img = torch.from_numpy(input_data).to(self.device)

        moves_pred_score = self.model_player(tensor_img.float())
        if np.random.random() > self.rand_ratio:
            move_idx = np.random.randint(0, len(self.moves))
        else:
            move_idx = torch.argmax(moves_pred_score)
        # Execute action
        if move_idx == 0:
            self.moves[move_idx](self.ge)
        else:
            self.moves[move_idx]()

        # save the preditions
        if LEARN:
            self.pred_heap.append((move_idx, moves_pred_score))

        if len(self.pred_heap) > LEARNING_PATH_LEN \
                or end \
                and LEARN:
            # Update results
            score_diff = self.last_score - self.current_score
            self.last_score = self.current_score
            loss_score = 0

            for idx, t_pred in enumerate(self.pred_heap):
                move_idx = t_pred[0]
                score_pred = t_pred[1]
                score_loss = score_pred.clone()
                score_loss[0, move_idx] = score_diff / (len(self.pred_heap) - idx)
                loss = self.criterion_player(score_pred, score_loss)
                loss_score += loss.item()
                # Backprop and perform Adam optimisation
                self.optimizer_player.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer_player.step()

            print("\nLoss: {:.2f}".format(loss_score / len(self.pred_heap)))
            self.pred_heap = []

        self.old_snap_shot = self.new_snap_shot.copy()

        AutoPlayer.rand_ratio = min(AutoPlayer.rand_ratio + 1e-5, 0.8)

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

        # self.model_state = RLNet.State_net(len(self.moves)).to(self.device)
        # self.model_state = self.model_state.float()
        # self.criterion_state = nn.MSELoss()
        # learning_rate_state = 0.01
        # self.optimizer_state = torch.optim.Adam(self.model_state.parameters(), lr=learning_rate_state)
