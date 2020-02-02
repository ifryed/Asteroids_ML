import sys
import time

import numpy as np
import cv2
import pygame
from pygame.time import Clock
import matplotlib.pyplot as plt

from AstroGame import Asteroids, StarShip, utils
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
        self.asteroids_lst = Asteroids.create(3)
        self.fire_group = pygame.sprite.Group()
        self.ast_group = pygame.sprite.Group()
        for ast in Asteroids.create(3):
            self.ast_group.add(ast)

    def paint(self):
        self.bg_canvas = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH))
        self.ast_group.draw(self.screen)
        self.player_group.draw(self.screen)
        self.fire_group.draw(self.screen)

        pygame.display.update()

    def gameUpdate(self):
        self.screen.fill(BLACK_SCREEN)
        self.player.move()
        for ast in self.ast_group:
            ast.move()
        for fire in self.fire_group:
            fire.move()

        self.checkCollision()

    def startGame(self):
        t = time.time()
        while self.gameIsRunning:
            event = pygame.event.poll()
            if event.type == pygame.QUIT:
                sys.exit()

            self.player.handle_keys(self)

            self.gameUpdate()
            self.paint()

            utils.RT_FPS = 1 / (time.time() - t)
            print("\rFPS:{0:.2f}".format(utils.RT_FPS), end='')
            t = time.time()
            Clock().tick(TIME_DELTA)

    def checkCollision(self):
        # Spaceship collision
        for ast in self.ast_group:
            if self.player.collide(ast):
                self.player.kill()
                print("BOOM")
                self.endgame()

        # Fire collision
        for fire in self.fire_group:
            for ast in self.ast_group:
                if fire.collide(ast):
                    ast.gotShot(fire, self)
                    fire.kill()
                    break

        if len(self.ast_group) < 1:
            self.gameIsRunning = False
        # Astroids collision
        handled_ast = []
        for ast in self.ast_group:
            handled_ast.append(ast)
            for o_ast in self.ast_group:
                if o_ast in handled_ast:
                    continue
                if ast.collide(o_ast):
                    handelCollision(ast, o_ast)

    def endgame(self):
        self.gameIsRunning = False
