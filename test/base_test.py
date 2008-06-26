import unittest, pygame
import test_utils
from test_utils import test_not_implemented

init_called = quit_called = 0
def __PYGAMEinit__(): #called automatically by pygame.init()
    global init_called
    init_called = init_called + 1
    pygame.register_quit(pygame_quit)
def pygame_quit():
    global quit_called
    quit_called = quit_called + 1


class BaseModuleTest(unittest.TestCase):
    def testAutoInit(self):
        pygame.init()
        pygame.quit()
        self.assertEqual(init_called, 1)
        self.assertEqual(quit_called, 1)

    def test_get_error(self):

        # __doc__ (as of 2008-06-25) for pygame.base.get_error:

          # pygame.get_error(): return errorstr
          # get the current error message

        self.assert_(test_not_implemented()) 

    def test_get_sdl_byteorder(self):

        # __doc__ (as of 2008-06-25) for pygame.base.get_sdl_byteorder:

          # pygame.get_sdl_byteorder(): return int
          # get the byte order of SDL

        self.assert_(test_not_implemented()) 

    def test_get_sdl_version(self):

        # __doc__ (as of 2008-06-25) for pygame.base.get_sdl_version:

          # pygame.get_sdl_version(): return major, minor, patch
          # get the version number of SDL

        self.assert_(test_not_implemented()) 

    def test_init(self):

        # __doc__ (as of 2008-06-25) for pygame.base.init:

          # pygame.init(): return (numpass, numfail)
          # initialize all imported pygame modules

        self.assert_(test_not_implemented()) 

    def test_quit(self):

        # __doc__ (as of 2008-06-25) for pygame.base.quit:

          # pygame.quit(): return None
          # uninitialize all pygame modules

        self.assert_(test_not_implemented()) 

    def test_register_quit(self):

        # __doc__ (as of 2008-06-25) for pygame.base.register_quit:

          # register_quit(callable): return None
          # register a function to be called when pygame quits

        self.assert_(test_not_implemented()) 

    def test_segfault(self):

        # __doc__ (as of 2008-06-25) for pygame.base.segfault:

          # crash

        self.assert_(test_not_implemented()) 


if __name__ == '__main__':
    test_utils.get_fail_incomplete_tests_option()
    unittest.main()