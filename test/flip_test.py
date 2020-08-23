#import test_utils
import unittest
import pygame

disp = pygame.display.set_mode((100,100))

class TestMain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.display.init()

    @classmethod
    def tearDownClass(cls):
        pygame.display.quit()

    def test_nochange(self):
        self.assertIsNone(pygame.display.flip())

    def test_change(self):
        pygame.Surface.fill(disp, (66,66,53))
    
    def test_novideo(self):
        pygame.display.quit()
        with self.assertRaises(pygame.error):
            (pygame.display.flip())
    
    def test_nowindow(self):
        global disp
        del disp
        with self.assertRaises(pygame.error):
            (pygame.display.flip())

if __name__ == "__main__":
    unittest.main()