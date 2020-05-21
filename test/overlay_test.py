import unittest
import pygame
from pygame.compat import xrange_

SDL2 = pygame.get_sdl_version()[0] >= 2


class OverlayTypeTest(unittest.TestCase):
    @unittest.skipIf(
        os.environ.get("SDL_VIDEODRIVER") == "dummy",
        'OpenGL requires a non-"dummy" SDL_VIDEODRIVER',
    )
    @unittest.skipIf(SDL2, "Overlay not ported to SDL2")
    def test_display(self):
        """Test we can create an overlay of all the different types"""

        f = open('../examples/data/yuv_1.pgm', "rb")
        fmt = f.readline().strip().decode()
        res = f.readline().strip().decode()
        unused_col = f.readline().strip()
        if fmt != "P5":
            print(
                "Unknown format: %s ( len %d ). Exiting..." % (fmt, len(fmt)))
            return

        w, h = [int(x) for x in res.split(" ")]
        h = int((h * 2) / 3)
        # Read into strings
        y = f.read(w * h)
        u = b''
        v = b''
        for _ in xrange_(0, int(h / 2)):
            u += (f.read(int(w / 2)))
            v += (f.read(int(w / 2)))

        f.close()

        pygame.display.init()

        display_surface = pygame.display.set_mode((w, h))
        display_surface_rect = display_surface.get_rect()
        display_center = display_surface_rect.center

        # data from pgm file works for these two formats
        formats_24_bit = (pygame.YV12_OVERLAY,
                          pygame.IYUV_OVERLAY)

        # still need good data for these three formats
        formats_xx_bit = (pygame.YUY2_OVERLAY,
                          pygame.UYVY_OVERLAY,
                          pygame.YVYU_OVERLAY)

        sizes = ((w, h),
                 (w, h))
        raw_data = ((y, u, v),
                    (y, u, v))

        created_overlays = 0
        for overlay_format, size, raw in zip(formats_24_bit, sizes, raw_data):
            overlay = pygame.Overlay(overlay_format, size)
            overlay_location = pygame.Rect((0, 0), size)
            overlay_location.center = display_center
            overlay.set_location(overlay_location)
            overlay.display(raw)
            created_overlays += 1

        self.assertEqual(created_overlays, 2)

    @unittest.skipIf(
        os.environ.get("SDL_VIDEODRIVER") == "dummy",
        'OpenGL requires a non-"dummy" SDL_VIDEODRIVER',
    )
    @unittest.skipIf(SDL2, "Overlay not ported to SDL2")
    def test_get_hardware(self):
        """Test for overlay hardware acceleration"""
        pygame.display.init()

        display_surface = pygame.display.set_mode((320, 240))
        overlay = pygame.Overlay(pygame.YV12_OVERLAY, (4, 4))
        self.assertFalse(overlay.get_hardware())

    @unittest.skipIf(
        os.environ.get("SDL_VIDEODRIVER") == "dummy",
        'OpenGL requires a non-"dummy" SDL_VIDEODRIVER',
    )
    @unittest.skipIf(SDL2, "Overlay not ported to SDL2")
    def test_set_location(self):
        """Test overlay set location"""

        f = open('../examples/data/yuv_1.pgm', "rb")
        fmt = f.readline().strip().decode()
        res = f.readline().strip().decode()
        unused_col = f.readline().strip()
        if fmt != "P5":
            print(
                "Unknown format: %s ( len %d ). Exiting..." % (fmt, len(fmt)))
            return

        w, h = [int(x) for x in res.split(" ")]
        h = int((h * 2) / 3)
        # Read into strings
        y = f.read(w * h)
        u = b''
        v = b''
        for _ in xrange_(0, int(h / 2)):
            u += (f.read(int(w / 2)))
            v += (f.read(int(w / 2)))

        f.close()

        pygame.display.init()

        display_surface = pygame.display.set_mode((w, h))
        display_surface.fill((0, 0, 0))
        pygame.display.update()

        overlay = pygame.Overlay(pygame.YV12_OVERLAY, (w, h))

        raw_data = (y, u, v)
        overlay.display(raw_data)
        pygame.display.update()
        overlay_location = pygame.Rect((0, 0), (w, h))

        positions = ((1, 1), (1, 190), (270, 1), (270, 190))
        for position in positions:
            overlay_location.topleft = position
            overlay.set_location(overlay_location)
            overlay.display()
            pygame.display.update()
            self.assertEqual(display_surface.get_at(position),
                             pygame.Color(0, 151, 0))


################################################################################

if __name__ == "__main__":
    unittest.main()
