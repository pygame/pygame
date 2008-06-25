import pygame, unittest, os, test_utils
from pygame.locals import *


class GL_ImageSave(unittest.TestCase):
    def test_image_save_works_with_opengl_surfaces(self):
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

if __name__ == '__main__': 
    unittest.main()

    def test_frombuffer(self):

        # __doc__ (as of 2008-06-25) for pygame.image.frombuffer:

          # pygame.image.frombuffer(string, size, format): return Surface
          # create a new Surface that shares data inside a string buffer

        self.assert_(test_not_implemented()) 

    def test_fromstring(self):

        # __doc__ (as of 2008-06-25) for pygame.image.fromstring:

          # pygame.image.fromstring(string, size, format, flipped=False): return Surface
          # create new Surface from a string buffer

        self.assert_(test_not_implemented()) 

    def test_get_extended(self):

        # __doc__ (as of 2008-06-25) for pygame.image.get_extended:

          # pygame.image.get_extended(): return bool
          # test if extended image formats can be loaded

        self.assert_(test_not_implemented()) 

    def test_load_basic(self):

        # __doc__ (as of 2008-06-25) for pygame.image.load_basic:

          # pygame.image.load(filename): return Surface
          # pygame.image.load(fileobj, namehint=): return Surface
          # load new image from a file

        self.assert_(test_not_implemented()) 

    def test_load_extended(self):

        # __doc__ (as of 2008-06-25) for pygame.image.load_extended:

          # pygame module for image transfer

        self.assert_(test_not_implemented()) 

    def test_save(self):

        # __doc__ (as of 2008-06-25) for pygame.image.save:

          # pygame.image.save(Surface, filename): return None
          # save an image to disk

        self.assert_(test_not_implemented()) 

    def test_save_extended(self):

        # __doc__ (as of 2008-06-25) for pygame.image.save_extended:

          # pygame module for image transfer

        self.assert_(test_not_implemented()) 

    def test_tostring(self):

        # __doc__ (as of 2008-06-25) for pygame.image.tostring:

          # pygame.image.tostring(Surface, format, flipped=False): return string
          # transfer image to string buffer

        self.assert_(test_not_implemented()) 
