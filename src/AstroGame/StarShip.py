import numpy as np

import pygame
import pygame.gfxdraw

from AstroGame.utils import SPACE_SHIP_GREEN, TIME_DELTA, SCREEN_WIDTH, SCREEN_HEIGHT

MAX_SPEED = 1000


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
        self.speed = np.array([0, 0])

    def rotateShip(self, angle: float):
        center = self.rect.center
        self.rot_angle = (self.rot_angle + angle) % 360
        self.image = pygame.transform.rotate(self.image_org, self.rot_angle)
        self.rect = self.image.get_rect(center=center)

    def move(self):
        self.rect.x += self.speed[0] / TIME_DELTA
        self.rect.y += self.speed[1] / TIME_DELTA

        cx, cy = self.rect.center
        if cx < 0:
            self.rect.x = SCREEN_WIDTH - self.rect.w / 2
        if cx > SCREEN_WIDTH:
            self.rect.x = 0
        if cy < 0:
            self.rect.y = SCREEN_HEIGHT - self.rect.h / 2
        if cy > SCREEN_HEIGHT:
            self.rect.y = 0

    def handle_keys(self):
        """ Handles Keys """
        key = pygame.key.get_pressed()
        speed_inc = 10  # distance moved in 1 frame, try changing it to 5
        speec_dec = .9
        rot_ang = 5
        if key[pygame.K_SPACE]:  # Shoot
            pass
        elif key[pygame.K_DOWN]:  # Forward
            self.speed *= speec_dec
            mag = np.sqrt(np.power(self.speed, 2).sum())
            norm_speed = self.speed / (mag + np.finfo('float').eps)
            self.speed = norm_speed * mag if mag > 1 else np.zeros_like(self.speed)

        elif key[pygame.K_UP]:  # Forward
            self.speed = self.speed + speed_inc * np.array([np.sin(np.radians(self.rot_angle)),
                                                            np.cos(np.radians(self.rot_angle))])
            mag = np.sqrt(np.power(self.speed, 2).sum())
            norm_speed = self.speed / (mag + np.finfo('float').eps)
            mag = np.min((MAX_SPEED, mag))
            self.speed = norm_speed * mag

        if key[pygame.K_RIGHT]:  # Turn right
            self.rotateShip(-rot_ang)
        elif key[pygame.K_LEFT]:  # Turn left
            self.rotateShip(rot_ang)
