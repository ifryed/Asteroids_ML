from dataclasses import dataclass

import pygame
from pygame.sprite import Sprite

TIME_DELTA = 60
RT_FPS = TIME_DELTA

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Colors
WHITE = (255, 255, 255)
SPACE_SHIP_GREEN = (0, 255, 130)
FIRE_RED = (255, 255, 130)

# Models path
PLAYER_NET_PATH = '../player.mdl'
STATE_NET_PATH = '../state.mdl'


@dataclass
class Scores:
    AST_HIT_SCORE = 100
    GAME_WON = 100
    GAME_LOST = -100


def rotateObject(obj: Sprite, angle):
    center = obj.rect.center
    obj.rot_angle = (obj.rot_angle + angle) % 360
    obj.image = pygame.transform.rotate(obj.image_org, obj.rot_angle)
    obj.rect = obj.image.get_rect(center=center)
