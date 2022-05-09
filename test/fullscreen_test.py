import unittest
import sys

import pygame
from pygame.tests import test_utils

class VisualTests(unittest.TestCase):
    __tags__ = [ "interactive" ]

    screen = None
    font = None
    isfullscreen = False
    aborted = False

    WIDTH = 800
    HEIGHT = 600

    def setUp(self):
        if self.screen is None:
            pygame.init()
            if sys.platform == "win32":
                # known issue with windows, must have mode from pygame.display.list_modes()
                # or window created with flag pygame.SCALED
                self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT), flags=pygame.SCALED)
            else:
                self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Fullscreen Tests")
            self.screen.fill((255, 255, 255))
            pygame.display.flip()
            self.font = pygame.font.Font(None, 32)

    def quit(self):
        if self.screen is not None:
            if self.isfullscreen:
                pygame.display.toggle_fullscreen()
            pygame.quit()
        self.aborted = True
    
    def visual_test(self, fullscreen=False):
        if self.aborted:
            return False
        text = ""
        if fullscreen:
            text = "Is this in fullscreen? [y/n]"
            if not self.isfullscreen:
                pygame.display.toggle_fullscreen()
                self.isfullscreen = True
        else:
            if self.isfullscreen:
                pygame.display.toggle_fullscreen()
                self.isfullscreen = False
            text = "Is this not in fullscreen [y/n]"
        s = self.font.render(text, False, (0, 0, 0))
        self.screen.blit(s, (self.WIDTH / 2 - self.font.size(text)[0] / 2, 100))
        pygame.display.flip()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.quit()
                        return False
                    if event.key == pygame.K_y:
                        return True
                    if event.key == pygame.K_n:
                        return False
                if event.type == pygame.QUIT:
                    self.quit()
                    return False

    def test_fullscreen_true(self):
        self.assertTrue(self.visual_test(fullscreen=True))

    def test_fullscreen_false(self):
        self.assertTrue(self.visual_test(fullscreen=False))

if __name__ == "__main__":
    unittest.main()