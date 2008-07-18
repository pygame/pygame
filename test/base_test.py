import test_utils
import test.unittest as unittest
import pygame
from test_utils import test_not_implemented

init_called = quit_called = 0
def __PYGAMEinit__(): #called automatically by pygame.init()
    global init_called
    init_called = init_called + 1
    pygame.register_quit(pygame_quit)
def pygame_quit():
    global quit_called
    quit_called = quit_called + 1


quit_hook_ran = 0
def quit_hook():
    global quit_hook_ran
    quit_hook_ran = 1

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

        self.assert_(pygame.get_sdl_byteorder() + 1)

    def test_get_sdl_version(self):

        # __doc__ (as of 2008-06-25) for pygame.base.get_sdl_version:

          # pygame.get_sdl_version(): return major, minor, patch
          # get the version number of SDL

        self.assert_( len(pygame.get_sdl_version()) == 3) 

    def test_init(self):
        return

        # __doc__ (as of 2008-06-25) for pygame.base.init:

          # pygame.init(): return (numpass, numfail)
          # initialize all imported pygame modules

        self.assert_(test_not_implemented()) 

    def not_init_assertions(self):
        self.assert_(not pygame.display.get_init(),  "display shouldn't be initialized" )
        self.assert_(not pygame.mixer.get_init(),  "mixer shouldn't be initialized" )
        self.assert_(not pygame.font.get_init(),  "init shouldn't be initialized" )
        
        self.assertRaises(pygame.error, pygame.scrap.get)
        
        # pygame.cdrom
        # pygame.joystick

    def init_assertions(self):
        self.assert_(pygame.display.get_init())
        self.assert_(pygame.mixer.get_init())
        self.assert_(pygame.font.get_init())

    def test_quit__and_init(self):
        return # TODO

        # __doc__ (as of 2008-06-25) for pygame.base.quit:

          # pygame.quit(): return None
          # uninitialize all pygame modules
        
        # Make sure everything is not init
        self.not_init_assertions()
    
        # Initiate it
        pygame.init()
        
        # Check
        self.init_assertions()

        # Quit
        pygame.quit()
        
        # All modules have quit
        self.not_init_assertions()

    def test_register_quit(self):

        # __doc__ (as of 2008-06-25) for pygame.base.register_quit:

          # register_quit(callable): return None
          # register a function to be called when pygame quits
        
        self.assert_(not quit_hook_ran)

        pygame.init()
        pygame.register_quit(quit_hook)
        pygame.quit()

        self.assert_(quit_hook_ran)

    def test_segfault(self):

        # __doc__ (as of 2008-06-25) for pygame.base.segfault:

          # crash

        self.assert_(test_not_implemented()) 

if __name__ == '__main__':
    unittest.main()