import unittest
import sys
import pygame

from pygame._sdl2 import video


class VideoModuleTest(unittest.TestCase):
    default_caption = "pygame window"

    @unittest.skipIf(
        not (sys.maxsize > 2**32),
        "32 bit SDL 2.0.16 has an issue.",
    )
    def test_renderer_set_viewport(self):
        """works."""
        window = video.Window(title=self.default_caption, size=(800, 600))
        renderer = video.Renderer(window=window)
        renderer.logical_size = (1920, 1080)
        rect = pygame.Rect(0, 0, 1920, 1080)
        renderer.set_viewport(rect)
        self.assertEqual(renderer.get_viewport(), (0, 0, 1920, 1080))

    def test_window_default_caption(self):
        window = video.Window()
        self.assertEqual(window.title, self.default_caption)

    def test_window_software_render(self):
        window = video.Window()
        screen = window.get_surface()

        screen.fill("red")
        window.flip()

    def test_window_opacity(self):
        window = video.Window()

        self.assertEqual(window.opacity, 1.0)

        window.opacity = 0.4
        self.assertAlmostEqual(window.opacity, 0.4, 2)


if __name__ == "__main__":
    unittest.main()
