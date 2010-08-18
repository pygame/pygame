if __name__ == '__main__':
    import sys
    import os
    pkg_dir = os.path.split(os.path.abspath(__file__))[0]
    parent_dir, pkg_name = os.path.split(pkg_dir)
    is_pygame_pkg = (pkg_name == 'tests' and
                     os.path.split(parent_dir)[1] == 'pygame')
    if not is_pygame_pkg:
        sys.path.insert(0, parent_dir)
else:
    is_pygame_pkg = __name__.startswith('pygame.tests.')

if is_pygame_pkg:
    from pygame.tests import test_utils
    from pygame.tests.test_utils import unittest
else:
    from test import test_utils
    from test.test_utils import unittest
import pygame, os
from pygame.locals import *

class GL_ImageSave(unittest.TestCase):
    def test_image_save_works_with_opengl_surfaces(self):
        "|tags:display,slow|"

        pygame.display.init()
        
        
        screen = pygame.display.set_mode((640,480), OPENGL|DOUBLEBUF)

        pygame.display.flip()
        
        tmp_dir = test_utils.get_tmp_dir()
        # Try the imageext module.
        tmp_file = os.path.join(tmp_dir, "opengl_save_surface_test.png")
        
        pygame.image.save(screen, tmp_file)
        
        self.assert_(os.path.exists(tmp_file))
        
        os.remove(tmp_file)

        # Only test the image module.
        tmp_file = os.path.join(tmp_dir, "opengl_save_surface_test.bmp")
        
        pygame.image.save(screen, tmp_file)
        
        self.assert_(os.path.exists(tmp_file))
        
        os.remove(tmp_file)
        
        # stops tonnes of tmp dirs building up in trunk dir
        os.rmdir(tmp_dir)
        
        
        pygame.display.quit()
if __name__ == '__main__': 
    unittest.main()
