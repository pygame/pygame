import sys
import os
if __name__ == '__main__':
    pkg_dir = os.path.split(os.path.abspath(__file__))[0]
    parent_dir, pkg_name = os.path.split(pkg_dir)
    is_pygame_pkg = (pkg_name == 'tests' and
                     os.path.split(parent_dir)[1] == 'pygame')
    if not is_pygame_pkg:
        sys.path.insert(0, parent_dir)
else:
    is_pygame_pkg = __name__.startswith('pygame.tests.')

if is_pygame_pkg:
    from pygame.tests import font_test
    from pygame.tests.test_utils import unittest
else:
    from test import font_test
    from test.test_utils import unittest

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
