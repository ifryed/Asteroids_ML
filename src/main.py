import argparse

import PlayerML
from AstroGame.GameEngi import GameEngi


def main(sys_args):
    auto_player = PlayerML.AutoPlayer() if sys_args.mode == 'auto' else None
    game = GameEngi(sys_args.mode, auto_player)
    if sys_args.mode == 'auto':
        game.setSmartPlayer(auto_player)
        auto_player.setupDNN()
    game_on = True

    while game_on:
        game.reset()
        game.startGame()
        game_on = game.gameClouser()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Astroid Game")
    parser.add_argument('--mode', dest='mode', required=True, choices=["manual", "auto"])

    args = parser.parse_args()
    main(args)
