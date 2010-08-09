__tags__ = []

try:
    import pygame._movie
except ImportError:
    __tags__.extend(('ignore', 'subprocess_ignore'))

