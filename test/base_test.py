
import unittest, pygame


init_called = quit_called = 0
def __PYGAMEinit__(): #called automatically by pygame.init()
    global init_called
    init_called = init_called + 1
    pygame.register_quit(pygame_quit)
def pygame_quit():
    global quit_called
    quit_called = quit_called + 1



class BaseTest(unittest.TestCase):
    def testAutoInit(self):
        pygame.init()
        pygame.quit()
        self.assertEqual(init_called, 1)
        self.assertEqual(quit_called, 1)



if __name__ == '__main__':
    unittest.main()
