import numpy as np
import torch
from torch import nn

import RLNet


class AutoPlayer:
    def __init__(self, game_eng):
        self.ge = game_eng
        self.moves = game_eng.player.getMoves()
        self.last_score = 0

        # Setting the NN
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = RLNet.RL_net(len(self.moves)).to(self.device)
        self.model = self.model.float()
        self.criterion = nn.L1Loss()
        learning_rate = 0.01
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def smartMove(self) -> None:
        # Get data
        snap_shot = self.ge.getSnapShot()
        h, w, z = snap_shot.shape

        # Choose action
        # move_idx = np.random.randint(0, len(self.moves))
        tensor_img = torch.from_numpy(snap_shot.reshape((1, 1, h, w, z))).to(self.device)
        moves_prob = self.model(tensor_img.float())
        move_idx = torch.argmax(moves_prob)
        # Execute action
        if move_idx == 0:
            self.moves[move_idx](self.ge)
        else:
            self.moves[move_idx]()

        # Get results
        score_diff = self.last_score - (self.ge.score + 1)
        self.last_score = self.ge.score
        # Update results
        loss = self.criterion(torch.from_numpy(np.array([score_diff])).float(), torch.from_numpy(np.zeros(1)).float())

        # Backprop and perform Adam optimisation
        self.optimizer.zero_grad()
        # loss.backward()
        self.optimizer.step()
