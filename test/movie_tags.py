__tags__ = []

import pygame
try:
    pygame.movie._NOT_IMPLEMENTED_
except AttributeError:
    pass
else:
    __tags__.append('not_implemented')
