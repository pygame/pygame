import test_utils
from pygame.tests.test_utils import example_path
import unittest
import pygame

display = pygame.display.set_mode((100,100))

class TestMain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.display.init()

    @classmethod
    def tearDownClass(cls):
        pygame.display.quit()

    def test_normal(self):
        gameIcon = pygame.image.load(example_path("data/alien1.jpg"))
        self.assertIsNone(pygame.display.set_icon(gameIcon))

    def test_text(self):
        with self.assertRaises(TypeError):
            pygame.display.set_icon('testing text')

    def test_integer(self):
        with self.assertRaises(TypeError):
            pygame.display.set_icon(1234)

    def test_dict(self):
        with self.assertRaises(TypeError):
            pygame.display.set_icon({'testing text':'test'})

    def test_list(self):
        with self.assertRaises(TypeError):
            pygame.display.set_icon(['testing text','fdsd'])

    def test_display_not_initialized(self):
        pygame.display.quit()
        with self.assertRaises(TypeError):
            pygame.display.set_icon('test.png')

    def test_audio_not_initialized(self):
        pygame.mixer.quit()
        with self.assertRaises(TypeError):
            pygame.display.set_icon('test.png')

if __name__ == "__main__":
    unittest.main()
