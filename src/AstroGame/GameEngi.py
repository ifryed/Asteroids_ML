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
from AstroGame.utils import TIME_DELTA, SCREEN_WIDTH, SCREEN_HEIGHT, Scores

np.set_printoptions(precision=3)
BLACK_SCREEN = (0, 0, 0)
MAX_GAME_TIME = 60 * 5


class GameEngi:

    def __init__(self, mode='manual', smrt_player: PlayerML.AutoPlayer = None):
        self.status = False
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
        self.reward = 0
        self.old_snap_shot = None
        self.new_snap_shot = None
        self.thread = None
        self.setSmartPlayer(smrt_player)

        self.reset()

    def setSmartPlayer(self, smrt_player: PlayerML.AutoPlayer):
        if smrt_player is None:
            return
        self.smart_player = smrt_player
        smrt_player.ge = self
        self.driver = self.smart_player.smartMove

    def reset(self):
        print("=============================")
        print("New Game")
        print("-----------------------------")

        self.thread = None
        self.gameIsRunning = True

        # Create Asteroid
        self.asteroids_lst = Asteroids.create(np.random.randint(3, 5))
        ## TODO sandbox
        # self.asteroids_lst = Asteroids.create(1 + 0 * np.random.randint(3, 4))
        # self.asteroids_lst[0].speed = 0
        # self.asteroids_lst[0].rect.x = 400
        # self.asteroids_lst[0].rect.y = 400
        ## sandbox

        self.fire_group = pygame.sprite.Group()
        self.ast_group = pygame.sprite.Group()
        for ast in self.asteroids_lst:
            self.ast_group.add(ast)

        self.player_group = pygame.sprite.Group()
        self.player = StarShip.StarShip()
        ## TODO sandbox
        # self.player.rect.x = 410
        # self.player.rect.y = 360
        self.player.rect.center = self.setPlayerPos()
        ## sandbox

        self.player_group.add(self.player)
        if self.mode == 'auto':
            self.smart_player.setMoves(self.player.getMoves())
        self.score = 0
        self.reward = 0

    def render(self):
        self.ast_group.draw(self.screen)
        self.player_group.draw(self.screen)
        self.fire_group.draw(self.screen)

        pygame.display.update()

    def gameUpdate(self) -> float:
        state_reward = -1
        self.screen.fill(BLACK_SCREEN)
        self.player.shipStep()
        for ast in self.ast_group:
            ast.asteroidStep()
        for fire in self.fire_group:
            fire.fireStep()

        state_reward += self.checkCollision()

        return state_reward

    def startGame(self):
        start_time = time.time()
        t = time.time()
        self.old_snap_shot = self.getSnapShot()
        self.new_snap_shot = self.getSnapShot()
        self.thread = threading.Thread()
        while self.gameIsRunning:
            event = pygame.event.poll()
            if event.type == pygame.QUIT:
                sys.exit()

            if self.mode == 'auto':
                old_state = np.array([[self.old_snap_shot,
                                       self.new_snap_shot]])
                action_idx = self.getAction(old_state)
                # move_fun = lambda: self.smart_player.smartMove(
                #     old_state,
                #     self.reward
                # )
                # self.thread = threading.Thread(
                #     target=move_fun)
                # self.thread.start()
                # action_idx = move_fun()
                self.old_snap_shot = self.new_snap_shot[:]
            else:
                self.driver()

            reward = self.gameUpdate()
            self.render()

            if self.mode == 'auto':
                self.new_snap_shot = self.getSnapShot()
                new_state = np.array([[self.old_snap_shot,
                                       self.new_snap_shot]])
                self.updateNet(action_idx, reward, old_state, new_state, self.gameIsRunning)

            utils.RT_FPS = 1 / (time.time() - t)
            # print("\rFPS:{:.2f}\tScore:{:.2f}\t".format(utils.RT_FPS, self.score), end='')
            t = time.time()
            # Clock().tick(TIME_DELTA)

            if self.mode == 'auto' \
                    and time.time() - start_time > MAX_GAME_TIME:
                break

    def checkCollision(self) -> float:
        col_reward = 0
        # Spaceship collision
        for ast in self.ast_group:
            if self.player.collide(ast):
                self.player.kill()
                print("BOOM")
                col_reward += self.endgame(victory=False)

        # Fire collision
        for fire in self.fire_group:
            for ast in self.ast_group:
                if fire.collide(ast):
                    Asteroids.gotShot(ast, fire, self)
                    col_reward += Scores.AST_HIT_SCORE
                    self.score += 1e4 * 1 / ast.size
                    fire.kill()
                    break

        if len(self.ast_group) < 1:
            col_reward += self.endgame(victory=True)

        # Astroids collision
        handled_ast = []
        for ast in self.ast_group:
            handled_ast.append(ast)
            for o_ast in self.ast_group:
                if o_ast in handled_ast:
                    continue
                if ast.collide(o_ast):
                    handelCollision(ast, o_ast)
        return col_reward

    def getSnapShot(self) -> np.ndarray:
        small_img = pygame.surfarray.array3d(self.screen).copy()
        small_img = cv2.resize(small_img, (0, 0), fx=.5, fy=.5)
        small_img = np.swapaxes(small_img, 0, 1)
        return small_img

    def endgame(self, victory=True) -> float:
        self.status = victory
        self.gameIsRunning = False

        if victory:
            return Scores.GAME_WON
        return Scores.GAME_LOST

    def gameClouser(self) -> (bool, bool):
        if self.mode == 'auto':
            # Won game
            if self.status:
                print("\nVICTORY")
            elif len(self.player_group) == 0:
                print("\nGAME_OVER")

            self.smart_player.saveNet()
        return True, self.status

    def setPlayerPos(self):
        self.render()
        canvas = (pygame.surfarray.array3d(self.screen).max(2) == 0).astype(np.uint8)
        canvas = np.swapaxes(canvas, 1, 0)
        open_sky = cv2.erode(canvas, np.ones((200, 200)))
        open_sky_xy = np.array(np.where(open_sky))
        idx = np.random.randint(0, len(open_sky_xy[0]))

        return tuple(open_sky_xy[:, idx])

    def getAction(self, state) -> int:
        action = self.smart_player.smartMove(state)
        return action

    def updateNet(self, action_idx, reward, old_state, new_state, not_done):
        self.smart_player.updateWeights(reward, old_state, new_state, not_done)
