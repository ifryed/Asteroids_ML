import os

import numpy as np
import torch
from torch import nn

import RLNet

LEARNING_PATH_LEN = 30
LEARN = True


class AutoPlayer:
    rand_ratio = .5

    def __init__(self, game_eng):
        self.ge = game_eng
        self.moves = game_eng.player.getMoves()
        self.last_score = 0

        # Setting the NN
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        if os.path.exists('../player.mdl'):
            self.model_player = torch.load('../player.mdl')
        else:
            self.model_player = RLNet.Player_net(len(self.moves)).to(self.device)
        self.model_player = self.model_player.float()
        self.criterion_player = nn.MSELoss()
        learning_rate_player = 1e-10
        self.optimizer_player = torch.optim.Adam(self.model_player.parameters(), lr=learning_rate_player)

        self.model_state = RLNet.State_net(len(self.moves)).to(self.device)
        self.model_state = self.model_state.float()
        self.criterion_state = nn.MSELoss()
        learning_rate_state = 0.01
        self.optimizer_state = torch.optim.Adam(self.model_state.parameters(), lr=learning_rate_state)

        self.loss_heap = []
        self.old_snap_shot = self.ge.getSnapShot()
        self.toggle_learning = True

    def smartMove(self, end=False) -> None:
        self.toggle_learning = not self.toggle_learning
        if end or self.toggle_learning:
            return
        # Get data
        new_snap_shot = self.ge.getSnapShot()
        h, w, z = new_snap_shot.shape

        # Choose action
        input_data = np.zeros((1, 2, h, w, z))
        input_data[0, 0, :, :, :] = self.old_snap_shot
        input_data[0, 1, :, :, :] = new_snap_shot
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
        self.loss_heap.append(moves_pred_score)

        if len(self.loss_heap) > LEARNING_PATH_LEN \
                or end:
            # Update results
            score_diff = self.last_score - self.ge.score
            self.last_score = self.ge.score
            score_loss = moves_pred_score.clone()
            score_loss[0, move_idx] = score_diff
            loss_score = 0
            for idx, t_pred in enumerate(self.loss_heap):
                loss = self.criterion_player(t_pred,
                                             score_loss / (len(self.loss_heap) - idx))
                loss_score += loss.item()
                # Backprop and perform Adam optimisation
                self.optimizer_player.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer_player.step()

            print("\nLoss: {:.2f}".format(loss_score / len(self.loss_heap)))
            self.loss_heap = []

        self.old_snap_shot = new_snap_shot.copy()

        AutoPlayer.rand_ratio += 1e-5

    def saveNet(self):
        torch.save(self.model_player, '../player.mdl')
