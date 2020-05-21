import unittest
import pygame

SDL2 = pygame.get_sdl_version()[0] >= 2


class OverlayTypeTest(unittest.TestCase):
    @unittest.skipIf(SDL2, "Overlay not ported to SDL2")
    def test_display(self):
        """Test we can create an overlay of all the different types"""
        pygame.display.init()

        display_surface = pygame.display.set_mode((320, 240))
        display_surface_rect = display_surface.get_rect()
        display_center = display_surface_rect.center

        supported_formats = (pygame.YV12_OVERLAY,
                             pygame.IYUV_OVERLAY,
                             pygame.YUY2_OVERLAY,
                             pygame.UYVY_OVERLAY,
                             pygame.YVYU_OVERLAY)

        sizes = ((4, 4),
                 (2, 2),
                 (2, 2),
                 (2, 2),
                 (2, 2))

        data = ((b'0xFF0xFF0xFF0xFF0xFF0xFF0xFF0xFF'
                 b'0xFF0xFF0xFF0xFF0xFF0xFF0xFF0xFF',
                 b'0xFF0xFF0xFF0xFF0xFF0xFF0xFF0xFF'
                 b'0xFF0xFF0xFF0xFF0xFF0xFF0xFF0xFF',
                 b'0xFF0xFF0xFF0xFF0xFF0xFF0xFF0xFF'
                 b'0xFF0xFF0xFF0xFF0xFF0xFF0xFF0xFF'),
                (b'0xFF0xFF0xFF0xFF', b'0xFF0xFF0xFF0xFF',
                 b'0xFF0xFF0xFF0xFF'),
                (b'0xFF0xFF0xFF0xFF', b'0xFF0xFF0xFF0xFF',
                 b'0xFF0xFF0xFF0xFF'),
                (b'0xFF0xFF0xFF0xFF', b'0xFF0xFF0xFF0xFF',
                 b'0xFF0xFF0xFF0xFF'),
                (b'0xFF0xFF0xFF0xFF', b'0xFF0xFF0xFF0xFF',
                 b'0xFF0xFF0xFF0xFF'))

        created_overlays = 0
        for overlay_format, size, raw in zip(supported_formats, sizes, data):
            overlay = pygame.Overlay(overlay_format, size)
            overlay_location = pygame.Rect((0, 0), size)
            overlay_location.center = display_center
            overlay.set_location(overlay_location)
            overlay.display(raw)
            created_overlays += 1

        self.assertEqual(created_overlays, 5)

    @unittest.skipIf(SDL2, "Overlay not ported to SDL2")
    def test_get_hardware(self):
        """Test for overlay hardware acceleration"""
        pygame.display.init()

        display_surface = pygame.display.set_mode((320, 240))
        overlay = pygame.Overlay(pygame.YV12_OVERLAY, (4, 4))
        self.assertFalse(overlay.get_hardware())

    @unittest.skipIf(SDL2, "Overlay not ported to SDL2")
    def test_set_location(self):
        """Test overlay set location"""
        pygame.display.init()

        display_surface = pygame.display.set_mode((320, 240))
        display_surface.fill((0, 0, 0))
        pygame.display.update()

        overlay = pygame.Overlay(pygame.YV12_OVERLAY, (4, 4))

        raw_data = (b'0xFF0xFF0xFF0xFF0xFF0xFF0xFF0xFF'
                    b'0xFF0xFF0xFF0xFF0xFF0xFF0xFF0xFF',
                    b'0xFF0xFF0xFF0xFF0xFF0xFF0xFF0xFF'
                    b'0xFF0xFF0xFF0xFF0xFF0xFF0xFF0xFF',
                    b'0xFF0xFF0xFF0xFF0xFF0xFF0xFF0xFF'
                    b'0xFF0xFF0xFF0xFF0xFF0xFF0xFF0xFF')
        overlay.display(raw_data)
        pygame.display.update()
        overlay_location = pygame.Rect((0, 0), (4, 4))

        positions = ((1, 1), (1, 190), (270, 1), (270, 190))
        for position in positions:
            overlay_location.topleft = position
            overlay.set_location(overlay_location)
            overlay.display()
            pygame.display.update()
            self.assertEqual(display_surface.get_at(position),
                             pygame.Color(0, 171, 0))


################################################################################

if __name__ == "__main__":
    unittest.main()
