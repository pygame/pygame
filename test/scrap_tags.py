__tags__ = []

import sys

exclude = False

if sys.platform == 'win32' or sys.platform == 'linux2':
    try:
        import pygame
        pygame.scrap._NOT_IMPLEMENTED_
    except AttributeError:
        pass
    else:
        exclude = True
else:
    exclude = True

if exclude:
    __tags__.extend(['ignore', 'subprocess_ignore'])




