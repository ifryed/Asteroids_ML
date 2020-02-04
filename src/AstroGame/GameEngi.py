import sys
import threading
import time

import numpy as np
import cv2
import pygame
from pygame.time import Clock
import matplotlib.pyplot as plt

import PlayerML
from AstroGame import Asteroids, StarShip, utils
from AstroGame.Asteroids import handelCollision
from AstroGame.utils import TIME_DELTA, SCREEN_WIDTH, SCREEN_HEIGHT

BLACK_SCREEN = (0, 0, 0)


class GameEngi:

    def __init__(self, mode='manual'):
        self.mode = mode
        self.reset()

    def reset(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.gameIsRunning = True

        pygame.init()

        # Create Astroid
        self.asteroids_lst = Asteroids.create(3)
        self.fire_group = pygame.sprite.Group()
        self.ast_group = pygame.sprite.Group()
        for ast in Asteroids.create(3):
            self.ast_group.add(ast)

        self.player_group = pygame.sprite.Group()
        self.player = StarShip.StarShip()
        self.player.rect.center = self.setPlayerPos()
        self.player_group.add(self.player)

        self.driver = lambda: self.player.handle_keys(self)
        if self.mode == "auto":
            self.smart_player = PlayerML.AutoPlayer(self)
            self.driver = self.smart_player.smartMove

        self.score = 0

    def paint(self):
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
        start_time = time.time()
        t = time.time()
        while self.gameIsRunning:
            event = pygame.event.poll()
            if event.type == pygame.QUIT:
                sys.exit()

            # threading.Thread(target=self.driver).start()
            self.driver()

            self.gameUpdate()
            self.paint()

            utils.RT_FPS = 1 / (time.time() - t)
            print("\rFPS:{:.2f}\tScore:{}\t".format(utils.RT_FPS, self.score), end='')
            t = time.time()
            Clock().tick(TIME_DELTA)

            if self.mode == 'auto' \
                    and time.time() - start_time > 120:
                break
        self.endgame()

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
                    self.score += 100 / ast.size
                    fire.kill()
                    break

        if len(self.ast_group) < 1:
            self.endgame()
        # Astroids collision
        handled_ast = []
        for ast in self.ast_group:
            handled_ast.append(ast)
            for o_ast in self.ast_group:
                if o_ast in handled_ast:
                    continue
                if ast.collide(o_ast):
                    handelCollision(ast, o_ast)

    def getSnapShot(self) -> np.ndarray:
        small_img = pygame.surfarray.array3d(self.screen).copy()
        small_img = np.resize(small_img, (SCREEN_HEIGHT // 2, SCREEN_WIDTH // 2, 3))
        return small_img

    def endgame(self):
        self.gameIsRunning = False
        if self.mode == 'auto':
            # Won game
            if len(self.ast_group) == 0:
                self.score += 1000
                print("VICTORY")
            else:
                self.score -= 1000

            self.smart_player.smartMove(True)
            self.smart_player.saveNet()
            print("Rand_ratio:", PlayerML.AutoPlayer.rand_ratio)
            # RestartGame
            self.reset()
            self.startGame()

    def setPlayerPos(self):
        self.paint()
        canvas = (pygame.surfarray.array3d(self.screen).max(2) == 0).astype(np.uint8)
        canvas = np.swapaxes(canvas, 1, 0)
        open_sky = cv2.erode(canvas, np.ones((100, 100)))
        open_sky_xy = np.array(np.where(open_sky))
        idx = np.random.randint(0, len(open_sky_xy[0]))

        return tuple(open_sky_xy[:, idx])
