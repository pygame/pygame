import unittest
import pygame

SDL2 = pygame.get_sdl_version()[0] >= 2

if SDL2:
    from pygame._sdl2 import video


    class VideoModuleTest(unittest.TestCase):
        default_caption = "pygame window"

        def test_renderer_set_viewport(self):
            """ works.
            """
            window = video.Window(title=self.default_caption, size=(800, 600))
            renderer = video.Renderer(window=window)
            renderer.logical_size = (1920, 1080)
            rect = pygame.Rect(0, 0, 1920, 1080)
            renderer.set_viewport(rect)
            self.assertEqual(renderer.get_viewport(), (0, 0, 1920, 1080))


    if __name__ == "__main__":
        unittest.main()
