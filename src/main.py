import argparse

from AstroGame.GameEngi import GameEngi


def main(args):
    game = GameEngi(args.mode)
    game.startGame()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Astroid Game")
    parser.add_argument('--mode', dest='mode', required=True, choices=["manual", "auto"])

    args = parser.parse_args()
    main(args)
