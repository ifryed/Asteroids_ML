from typing import List, Optional, Any

import numpy as np
import pygame
import pygame.gfxdraw

from AstroGame.utils import TIME_DELTA, SCREEN_WIDTH, SCREEN_HEIGHT, WHITE

SPIK_RAND_FACTOR = 5
ASTERO_WIDTH = 3
MIN_SIZE = 2


class Asteroid(pygame.sprite.Sprite):
    polygon: Optional[Any]
    id_counter = 0

    @staticmethod
    def getID():
        Asteroid.id_counter += 1
        return Asteroid.id_counter

    def __init__(self, size: int = 40, *groups) -> None:
        # Setting speed and direction
        super().__init__(*groups)
        self.id = Asteroid.getID()
        self.speed = 240 + np.random.randint(-3, 3)*10
        self.rot_angle = 0
        self.rotation_speed = (2 * ((np.random.random() < .5) - .5)) * np.random.randint(30, 60) / TIME_DELTA
        self.direction = np.radians(np.random.randint(0, 360))
        self.color = WHITE

        # Making the shape
        spikes_num = np.random.randint(5, 10)
        angle_step = np.radians(360 / spikes_num)
        polygon = []
        for s in range(spikes_num):
            x = size * np.cos(s * angle_step) + (np.random.random() - .5) * SPIK_RAND_FACTOR
            y = size * np.sin(s * angle_step) + (np.random.random() - .5) * SPIK_RAND_FACTOR

            polygon.append((x, y))
        self.polygon = np.array(polygon)

        poly_width = self.polygon[:, 0].max() - self.polygon[:, 0].min()
        poly_height = self.polygon[:, 1].max() - self.polygon[:, 1].min()

        rect_size = max(poly_height, poly_width)
        self.polygon += np.array([poly_height / 2, poly_width / 2])

        poly_img = pygame.Surface((poly_height, poly_width), pygame.SRCALPHA)
        pygame.gfxdraw.filled_polygon(poly_img, self.polygon, self.color)

        self.image_org = poly_img.copy()
        self.image = poly_img
        self.rect = self.image.get_rect()

    def setPos(self, new_pos: np.ndarray) -> None:
        self.rect.center = new_pos

    def addShock(self, direction: float, speed: float):
        dirx = np.cos(direction) * speed + np.cos(self.direction) * self.speed
        diry = np.sin(direction) * speed + np.sin(self.direction) * self.speed
        self.direction = np.arctan2(diry, dirx)[0]

    def move(self) -> None:
        center = self.rect.center
        self.image = pygame.transform.rotate(self.image_org, self.rot_angle)
        self.rot_angle += self.rotation_speed % 360
        self.rect = self.image.get_rect(center=center)

        self.rect.x += np.cos(self.direction) * self.speed / TIME_DELTA
        self.rect.y += np.sin(self.direction) * self.speed / TIME_DELTA
        cx, cy = self.rect.center
        if cx < 0:
            self.rect.x = SCREEN_WIDTH - self.rect.w / 2
        if cx > SCREEN_WIDTH:
            self.rect.x = 0
        if cy < 0:
            self.rect.y = SCREEN_HEIGHT - self.rect.h / 2
        if cy > SCREEN_HEIGHT:
            self.rect.y = 0


def create(n_asteroids: int) -> List[Asteroid]:
    ret_ast = []
    for i in range(n_asteroids):
        n_ast = Asteroid()
        x = np.random.randint(0, SCREEN_WIDTH)
        y = np.random.randint(0, SCREEN_WIDTH)
        n_ast.rect.center = np.array([x, y], dtype=np.float32)
        ret_ast.append(n_ast)

    return ret_ast


def handelCollision(ast: Asteroid, o_ast: Asteroid) -> None:
    rect1 = ast.rect
    rect2 = o_ast.rect

    vec = np.array(rect1.center) - np.array(rect2.center)

    ast.addShock(vec, o_ast.speed)
    o_ast.addShock(-vec, ast.speed)
