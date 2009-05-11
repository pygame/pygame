__tags__ = []

import sys

exclude = False

if ((sys.platform == 'win32' or sys.platform == 'linux2') and
    sys.version_info < (3, 0, 0)):
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




