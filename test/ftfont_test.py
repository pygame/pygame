import sys
import os
import unittest
from pygame.tests import font_test

import pygame.ftfont

font_test.pygame_font = pygame.ftfont
# Disable UCS-4 specific tests as this "Font" type does accept UCS-4 codes.
font_test.UCS_4 = False

for name in dir(font_test):
    obj = getattr(font_test, name)
    if (isinstance(obj, type) and  # conditional and
        issubclass(obj, unittest.TestCase)):
        globals()[name] = obj

if __name__ == '__main__':
    unittest.main()
