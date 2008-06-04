import pygame, unittest, os, test_utils
from pygame.locals import *


class GL_ImageSave(unittest.TestCase):
    def test_image_save_works_with_opengl_surfaces(self):
        screen = pygame.display.set_mode((640,480), OPENGL|DOUBLEBUF)

        pygame.display.flip()

        tmp_dir = test_utils.get_tmp_dir()
        tmp_file = os.path.join(tmp_dir, "opengl_save_surface_test.png")

        pygame.image.save(screen, tmp_file)

        self.assert_(os.path.exists(tmp_file))

        os.remove(tmp_file)


if __name__ == '__main__': 
    unittest.main()
