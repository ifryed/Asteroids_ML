import sys
import numpy as np
import cv2
import pygame
from pygame.time import Clock
import matplotlib.pyplot as plt

from AstroGame import Asteroids, StarShip
from AstroGame.Asteroids import handelCollision
from AstroGame.utils import TIME_DELTA, SCREEN_WIDTH, SCREEN_HEIGHT

BLACK_SCREEN = (0, 0, 0)


class GameEngi:

    def __init__(self, mode='manual'):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.gameIsRunning = True

        self.asteroids_lst = []
        self.player = StarShip.StarShip()
        self.player_group = pygame.sprite.Group()
        self.player_group.add(self.player)
        self.mode = mode
        pygame.init()

        self.bg_canvas = None
        # Create Astroid
        self.asteroids_lst = Asteroids.create(1)
        self.ast_group = pygame.sprite.Group()
        for ast in self.asteroids_lst:
            self.ast_group.add(ast)

    def paint(self):
        self.bg_canvas = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH))
        self.ast_group.draw(self.screen)
        self.player_group.draw(self.screen)

        pygame.display.update()

    def gameUpdate(self):
        self.screen.fill(BLACK_SCREEN)
        self.player.move()
        for ast in self.asteroids_lst:
            ast.move()

        self.checkCollision()

    def startGame(self):
        while self.gameIsRunning:
            event = pygame.event.poll()
            if event.type == pygame.QUIT:
                sys.exit()

            self.player.handle_keys()

            self.gameUpdate()
            self.paint()
            Clock().tick(TIME_DELTA)

    def checkCollision(self):

        # Spaceship collision
        for ast in self.asteroids_lst:
            if ast.rect.colliderect(self.player.rect):
                self.player.kill()
                print("BOOM")
                self.endgame()

        # Astroids collision
        handled_ast = []
        for idx, ast in enumerate(self.asteroids_lst):
            handled_ast.append(ast)
            for o_ast in self.asteroids_lst:
                if o_ast in handled_ast:
                    continue
                if ast.rect.colliderect(o_ast.rect):
                    handelCollision(ast, o_ast)
                    handled_ast.append(o_ast)

    def endgame(self):
        self.gameIsRunning = False
