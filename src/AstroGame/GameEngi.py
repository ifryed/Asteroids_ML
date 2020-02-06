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
MAX_GAME_TIME = 60 * 5


class GameEngi:

    def __init__(self, mode='manual', smrt_player: PlayerML.AutoPlayer = None):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.mode = mode
        # Init variables
        self.driver = lambda: self.player.handle_keys(self)
        self.score = 0
        self.smart_player = None
        self.gameIsRunning = False
        self.asteroids_lst = []
        self.fire_group = None
        self.ast_group = None
        self.player_group = None
        self.player = None

        self.setSmartPlayer(smrt_player)
        self.reset()

    def setSmartPlayer(self, smrt_player: PlayerML.AutoPlayer):
        if smrt_player is None:
            return
        self.smart_player = smrt_player
        smrt_player.ge = self
        self.driver = self.smart_player.smartMove

    def reset(self):
        print("New Game")
        print("=============================")
        self.gameIsRunning = True

        # Create Asteroid
        self.asteroids_lst = Asteroids.create(np.random.randint(3, 5))
        self.fire_group = pygame.sprite.Group()
        self.ast_group = pygame.sprite.Group()
        for ast in Asteroids.create(3):
            self.ast_group.add(ast)

        self.player_group = pygame.sprite.Group()
        self.player = StarShip.StarShip()
        self.player.rect.center = self.setPlayerPos()
        self.player_group.add(self.player)
        if self.mode == 'auto':
            self.smart_player.setMoves(self.player.getMoves())
            self.smart_player.old_snap_shot = self.getSnapShot()
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
            if self.mode == 'auto':
                self.smart_player.new_snap_shot = self.getSnapShot()
            self.driver()

            self.gameUpdate()
            self.paint()

            utils.RT_FPS = 1 / (time.time() - t)
            print("\rFPS:{:.2f}\tScore:{:.2f}\t".format(utils.RT_FPS, self.score), end='')
            t = time.time()
            Clock().tick(TIME_DELTA)

            if self.mode == 'auto' \
                    and time.time() - start_time > MAX_GAME_TIME:
                break

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
                    self.score += 100 * 1 / np.sqrt(ast.size)
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
        small_img = cv2.resize(small_img, (0, 0), fx=.5, fy=.5)
        small_img = np.swapaxes(small_img, 0, 1)
        return small_img

    def endgame(self):
        self.gameIsRunning = False

    def gameClouser(self) -> bool:
        if self.mode == 'auto':
            # Won game
            if len(self.ast_group) == 0:
                self.score += 1000
                print("VICTORY")
            elif len(self.player_group) == 0:
                self.score = -100

            self.smart_player.smartMove(True)
            self.smart_player.saveNet()
            print("Rand_ratio:", PlayerML.AutoPlayer.rand_ratio)

            return True

    def setPlayerPos(self):
        self.paint()
        canvas = (pygame.surfarray.array3d(self.screen).max(2) == 0).astype(np.uint8)
        canvas = np.swapaxes(canvas, 1, 0)
        open_sky = cv2.erode(canvas, np.ones((100, 100)))
        open_sky_xy = np.array(np.where(open_sky))
        idx = np.random.randint(0, len(open_sky_xy[0]))

        return tuple(open_sky_xy[:, idx])
