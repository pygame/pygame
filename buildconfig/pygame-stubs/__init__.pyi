from typing import Any

import pygame.event as event
import pygame.joystick as joystick
from pygame.rect import Rect as Rect
from pygame.surface import *  # noqa: F403

import pygame.color
import pygame.bufferproxy

Color = pygame.color.Color
BufferProxy = pygame.bufferproxy.BufferProxy

def __getattr__(name) -> Any: ...  # don't error on missing stubs
