import argparse
import os

import PlayerML
from AstroGame.GameEngi import GameEngi
from AstroGame.utils import STATE_NET_PATH, PLAYER_NET_PATH


def main(sys_args):
    # Delete pre-trained models
    if sys_args.reset:
        if os.path.exists(PLAYER_NET_PATH):
            os.remove(PLAYER_NET_PATH)
        if os.path.exists(STATE_NET_PATH):
            os.remove(STATE_NET_PATH)

    auto_player = PlayerML.AutoPlayer() if sys_args.mode == 'auto' else None
    game = GameEngi(sys_args.mode, auto_player)
    if sys_args.mode == 'auto':
        game.setSmartPlayer(auto_player)
        auto_player.setupDNN()
    game_on = True
    tot_won = 0

    while game_on:
        print("Games Won:", tot_won)
        game.reset()
        game.startGame()
        game_on, status = game.gameClouser()
        tot_won += status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Astroid Game")
    parser.add_argument('--mode', dest='mode', required=True, choices=["manual", "auto"])
    parser.add_argument('--reset', dest='reset', required=False, action='store_true' )

    args = parser.parse_args()
    main(args)
