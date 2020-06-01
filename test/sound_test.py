import os
import unittest

import pygame
import pygame.sound
from pygame.tests.test_utils import example_path


class SoundModuleTest(unittest.TestCase):

    def test_load(self):
        wave_path = example_path(os.path.join("data", "house_lo.wav"))
        snd = pygame.sound.load(file=wave_path)
        self.assertTrue(snd.get_length() > 0.5)
        snd_bytes = snd.get_raw()
        self.assertTrue(len(snd_bytes) > 1000)


if __name__ == "__main__":
    unittest.main()
