import numpy as np

import pygame
import pygame.gfxdraw

from AstroGame.Asteroids import Asteroid
from AstroGame.utils import SPACE_SHIP_GREEN, TIME_DELTA, SCREEN_WIDTH, SCREEN_HEIGHT, RT_FPS, rotateObject, FIRE_RED

FIRE_SPEED = 500
MAX_DIST = 400


class Fire(pygame.sprite.Sprite):
    def __init__(self, pos, direction, speed, *groups):
        super().__init__(*groups)

        # Making the shape
        self.size = 11
        poly_img = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        pygame.gfxdraw.filled_ellipse(poly_img, 6, 6, 6, 3, FIRE_RED)

        self.image_org = poly_img.copy()
        self.image = poly_img
        self.rect = self.image.get_rect()
        self.rect.center = pos
        self.rot_angle = 90

        rotateObject(self, direction)

        self.rot_angle = direction
        self.speed = np.array([FIRE_SPEED * np.sin(np.radians(self.rot_angle)),
                               FIRE_SPEED * np.cos(np.radians(self.rot_angle))]) + speed
        self.dist_traveled = 0

    def move(self):
        self.rect.x += self.speed[0] / RT_FPS
        self.rect.y += self.speed[1] / RT_FPS

        self.dist_traveled += np.sqrt(np.power(self.speed / RT_FPS, 2).sum())
        if self.dist_traveled > MAX_DIST:
            self.kill()

        cx, cy = self.rect.center
        if cx < 0:
            self.rect.x = SCREEN_WIDTH - self.rect.w / 2
        if cx > SCREEN_WIDTH:
            self.rect.x = 0
        if cy < 0:
            self.rect.y = SCREEN_HEIGHT - self.rect.h / 2
        if cy > SCREEN_HEIGHT:
            self.rect.y = 0

    def collide(self, ast: Asteroid) -> bool:
        dist = (np.array(self.rect.center) - np.array(ast.rect.center)) ** 2
        dist = np.sqrt(dist.sum())
        return dist < (self.size + ast.size)
