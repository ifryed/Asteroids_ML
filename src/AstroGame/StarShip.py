import time

import numpy as np

import pygame
import pygame.gfxdraw

from AstroGame.Asteroids import Asteroid
from AstroGame.Fire import Fire
from AstroGame.utils import SPACE_SHIP_GREEN, SCREEN_WIDTH, SCREEN_HEIGHT, RT_FPS, rotateObject

MAX_SPEED = 1000
MIN_SHOOT_TIME = .5
SPEED_DECAY = 0.9
ACCELERATE_SPEED = 10
TURN_ANGLE = 5


class StarShip(pygame.sprite.Sprite):
    def __init__(self, *groups):
        super().__init__(*groups)

        # Making the shape
        ship_size = 20
        poly_img = pygame.Surface((ship_size, ship_size), pygame.SRCALPHA)
        self.polygon = np.array([
            [0, 5],
            [10, 20],
            [20, 5],
            [15, 0],
            [5, 0],

        ])
        pygame.gfxdraw.filled_polygon(poly_img, self.polygon, SPACE_SHIP_GREEN)

        self.image_org = poly_img.copy()
        self.image = poly_img
        self.rect = self.image.get_rect()
        self.rect.center = (400, 400)
        self.rot_angle = 0
        self.speed = np.array([0, 0], dtype=np.float32)

        self.last_shoot_ts = 0

    def move(self):
        self.rect.x += self.speed[0] / RT_FPS
        self.rect.y += self.speed[1] / RT_FPS

        cx, cy = self.rect.center
        if cx < 0:
            self.rect.x = SCREEN_WIDTH - self.rect.w / 2
        if cx > SCREEN_WIDTH:
            self.rect.x = 0
        if cy < 0:
            self.rect.y = SCREEN_HEIGHT - self.rect.h / 2
        if cy > SCREEN_HEIGHT:
            self.rect.y = 0

    def fire(self, game_engi):
        if (time.time() - self.last_shoot_ts) >= MIN_SHOOT_TIME:
            self.last_shoot_ts = time.time()
            game_engi.fire_group.add(
                Fire(
                    self.rect.center,
                    self.rot_angle,
                    self.speed
                ))

    def brake(self):
        self.speed *= SPEED_DECAY
        mag = np.sqrt(np.power(self.speed, 2).sum())
        norm_speed = self.speed / (mag + np.finfo('float').eps)
        self.speed = norm_speed * mag if mag > 1 else np.zeros_like(self.speed)

    def handle_keys(self, game_engi):
        """ Handles Keys """
        key = pygame.key.get_pressed()
        if key[pygame.K_SPACE]:  # Shoot
            self.fire(game_engi)
        if key[pygame.K_DOWN]:  # Forward
            self.brake()
        elif key[pygame.K_UP]:  # Forward
            self.accelerate()
        if key[pygame.K_RIGHT]:  # Turn right
            self.rotateRight()
        elif key[pygame.K_LEFT]:  # Turn left
            self.rotateLeft()

    def getMoves(self) -> list:
        return [self.fire, self.brake, self.accelerate, self.rotateRight, self.rotateLeft]

    def collide(self, ast: Asteroid) -> bool:
        dist = (np.array(self.rect.center) - np.array(ast.rect.center)) ** 2
        dist = np.sqrt(dist.sum())
        return dist < (self.rect.w + ast.size)

    def accelerate(self):
        self.speed = self.speed + ACCELERATE_SPEED * np.array([np.sin(np.radians(self.rot_angle)),
                                                               np.cos(np.radians(self.rot_angle))])
        mag = np.sqrt(np.power(self.speed, 2).sum())
        norm_speed = self.speed / (mag + np.finfo('float').eps)
        mag = np.min((MAX_SPEED, mag))
        self.speed = norm_speed * mag

    def rotateRight(self):
        rotateObject(self, -TURN_ANGLE)

    def rotateLeft(self):
        rotateObject(self, TURN_ANGLE)
