import os
import sys
import unittest

import pygame.ftfont

from . import font_test

font_test.pygame_font = pygame.ftfont

for name in dir(font_test):
    obj = getattr(font_test, name)
    if isinstance(obj, type) and issubclass(obj, unittest.TestCase):  # conditional and
        new_name = f"Ft{name}"
        globals()[new_name] = type(new_name, (obj,), {})

if __name__ == "__main__":
    unittest.main()
