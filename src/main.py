import argparse

import PlayerML
from AstroGame.GameEngi import GameEngi


def main(sys_args):
    auto_player = PlayerML.AutoPlayer()
    game = GameEngi(sys_args.mode, auto_player)
    game.setSmartPlayer(auto_player)
    game_on = True

    while game_on:
        game.startGame()
        game_on = game.gameClouser()
        game.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Astroid Game")
    parser.add_argument('--mode', dest='mode', required=True, choices=["manual", "auto"])

    args = parser.parse_args()
    main(args)
