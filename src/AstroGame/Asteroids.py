from typing import List, Optional, Any

import numpy as np
import pygame
import pygame.gfxdraw

from AstroGame.utils import TIME_DELTA, SCREEN_WIDTH, SCREEN_HEIGHT, WHITE, RT_FPS

SPIK_RAND_FACTOR = .2
ASTERO_WIDTH = 3
MIN_ASTROID_SIZE = 5


class Asteroid(pygame.sprite.Sprite):
    polygon: Optional[Any]
    id_counter = 0

    @staticmethod
    def getID():
        Asteroid.id_counter += 1
        return Asteroid.id_counter

    def __init__(self, size: int = 20, *groups) -> None:
        # Setting speed and direction
        super().__init__(*groups)
        self.id = Asteroid.getID()
        self.speed = (1 + np.random.randint(-1, 1) * .3) * 200
        self.rot_angle = 0
        self.rotation_speed = (2 * ((np.random.random() < .5) - .5)) * np.random.randint(30, 60) / RT_FPS
        self.direction = np.radians(np.random.randint(0, 360))
        self.color = WHITE
        self.size = size

        # Making the shape
        spikes_num = np.random.randint(5, 10)
        angle_step = np.radians(360 / spikes_num)
        polygon = []
        for s in range(spikes_num):
            x = size * (np.cos(s * angle_step) + (np.random.random() - .5) * SPIK_RAND_FACTOR)
            y = size * (np.sin(s * angle_step) + (np.random.random() - .5) * SPIK_RAND_FACTOR)

            polygon.append((x, y))
        self.polygon = np.array(polygon)

        poly_width = self.polygon[:, 0].max() - self.polygon[:, 0].min()
        poly_height = self.polygon[:, 1].max() - self.polygon[:, 1].min()

        self.polygon += np.array([poly_height / 2, poly_width / 2])

        poly_img = pygame.Surface((poly_height, poly_width), pygame.SRCALPHA)
        pygame.gfxdraw.filled_polygon(poly_img, self.polygon, self.color)

        self.image_org = poly_img.copy()
        self.image = poly_img
        self.rect = self.image.get_rect()

    def setPos(self, new_pos: np.ndarray) -> None:
        self.rect.center = new_pos

    def getPos(self) -> np.ndarray:
        return np.array(self.rect.center)

    def addShock(self, direction: float, speed: float):
        dirx = np.cos(direction) * speed + np.cos(self.direction) * self.speed
        diry = np.sin(direction) * speed + np.sin(self.direction) * self.speed
        self.direction = np.arctan2(diry, dirx)

    def move(self) -> None:
        center = self.rect.center
        self.image = pygame.transform.rotate(self.image_org, self.rot_angle)
        self.rot_angle += self.rotation_speed % 360
        self.rect = self.image.get_rect(center=center)

        self.rect.x += np.cos(self.direction) * self.speed / RT_FPS
        self.rect.y += np.sin(self.direction) * self.speed / RT_FPS
        cx, cy = self.rect.center
        if cx < 0:
            self.rect.x = SCREEN_WIDTH - self.rect.w / 2
        if cx > SCREEN_WIDTH:
            self.rect.x = 0
        if cy < 0:
            self.rect.y = SCREEN_HEIGHT - self.rect.h / 2
        if cy > SCREEN_HEIGHT:
            self.rect.y = 0

    def collide(self, o_obj) -> bool:
        dist = (np.array(self.rect.center) - np.array(o_obj.rect.center)) ** 2
        dist = np.sqrt(dist.sum())
        return dist < (self.size + o_obj.size)

    def gotShot(self, fire, game_engi):
        if self.size > 2 * MIN_ASTROID_SIZE:
            hit_vec = np.array(self.rect.center) - np.array(fire.rect.center)
            hit_ang = np.arctan2(hit_vec[1], hit_vec[0])
            explod_vec_norm = np.array([np.cos(hit_ang + np.pi / 2), np.sin(hit_ang + np.pi / 2)])

            tot_speed = self.speed + fire.speed
            tot_speed = np.linalg.norm(tot_speed)
            a1 = Asteroid(self.size // 2)
            a1.setPos(self.getPos() + 10 * explod_vec_norm)
            a1.speed = tot_speed * min(.8, max(.2, np.random.random()))
            a2 = Asteroid(self.size // 2)
            a2.setPos(self.getPos() - 10 * explod_vec_norm)
            a2.speed = tot_speed - a1.speed

            game_engi.ast_group.add(a1)
            game_engi.ast_group.add(a2)
        self.kill()


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
    direction = np.arctan2(vec[1], vec[0])
    tot_speed = o_ast.speed + ast.speed
    tot_size = ast.size + o_ast.size
    ast.addShock(direction, tot_speed * ast.size / tot_size)
    o_ast.addShock(-direction, tot_speed * o_ast.size / tot_size)
