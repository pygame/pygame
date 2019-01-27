# -*- coding: utf-8 -*-

import array
import os
import tempfile
import unittest
import glob

from pygame.tests.test_utils import example_path, png
import pygame, pygame.image, pygame.pkgdata
from pygame.compat import xrange_, ord_, unicode_


def test_magic(f, magic_hex):
    """ tests a given file to see if the magic hex matches.
    """
    data = f.read(len(magic_hex))

    if len(data) != len(magic_hex):
        return 0

    for i in range(len(magic_hex)):
        if magic_hex[i] != ord_(data[i]):
            return 0

    return 1


class ImageModuleTest( unittest.TestCase ):
    def testLoadIcon(self):
        """ see if we can load the pygame icon.
        """
        f = pygame.pkgdata.getResource("pygame_icon.bmp")
        self.assertEqual(f.mode, "rb")

        surf = pygame.image.load_basic(f)

        self.assertEqual(surf.get_at((0,0)),(5, 4, 5, 255))
        self.assertEqual(surf.get_height(),32)
        self.assertEqual(surf.get_width(),32)

    def testLoadPNG(self):
        """ see if we can load a png with color values in the proper channels.
        """
        # Create a PNG file with known colors
        reddish_pixel = (210, 0, 0, 255)
        greenish_pixel = (0, 220, 0, 255)
        bluish_pixel = (0, 0, 230, 255)
        greyish_pixel = (110, 120, 130, 140)
        pixel_array = [reddish_pixel + greenish_pixel,
                       bluish_pixel + greyish_pixel]

        f_descriptor, f_path = tempfile.mkstemp(suffix='.png')
        f = os.fdopen(f_descriptor, 'wb')
        w = png.Writer(2, 2, alpha=True)
        w.write(f, pixel_array)
        f.close()

        # Read the PNG file and verify that pygame interprets it correctly
        surf = pygame.image.load(f_path)

        self.assertEqual(surf.get_at((0, 0)), reddish_pixel)
        self.assertEqual(surf.get_at((1, 0)), greenish_pixel)
        self.assertEqual(surf.get_at((0, 1)), bluish_pixel)
        self.assertEqual(surf.get_at((1, 1)), greyish_pixel)

        # Read the PNG file obj. and verify that pygame interprets it correctly
        f = open(f_path, 'rb')
        surf = pygame.image.load(f)
        f.close()

        self.assertEqual(surf.get_at((0, 0)), reddish_pixel)
        self.assertEqual(surf.get_at((1, 0)), greenish_pixel)
        self.assertEqual(surf.get_at((0, 1)), bluish_pixel)
        self.assertEqual(surf.get_at((1, 1)), greyish_pixel)

        os.remove(f_path)

    def testLoadJPG(self):
        """ see if we can load a jpg.
        """

        f = example_path('data/alien1.jpg')      # normalized
        # f = os.path.join("examples", "data", "alien1.jpg")
        surf = pygame.image.load(f)

        f = open(f, "rb")

        # f = open(os.path.join("examples", "data", "alien1.jpg"), "rb")

        surf = pygame.image.load(f)

        # surf = pygame.image.load(open(os.path.join("examples", "data", "alien1.jpg"), "rb"))

    def testSaveJPG(self):
        """ JPG equivalent to issue #211 - color channel swapping

        Make sure the SDL surface color masks represent the rgb memory format
        required by the JPG library. The masks are machine endian dependent
        """

        from pygame import Color, Rect

        # The source image is a 2 by 2 square of four colors. Since JPEG is
        # lossy, there can be color bleed. Make each color square 16 by 16,
        # to avoid the significantly color value distorts found at color
        # boundaries due to the compression value set by Pygame.
        square_len = 16
        sz = 2 * square_len, 2 * square_len

        #  +---------------------------------+
        #  | red            | green          |
        #  |----------------+----------------|
        #  | blue           | (255, 128, 64) |
        #  +---------------------------------+
        #
        #  as (rect, color) pairs.
        def as_rect(square_x, square_y):
            return Rect(square_x * square_len, square_y * square_len,
                        square_len, square_len)
        squares = [(as_rect(0, 0), Color("red")),
                   (as_rect(1, 0), Color("green")),
                   (as_rect(0, 1), Color("blue")),
                   (as_rect(1, 1), Color(255, 128, 64))]

        # A surface format which is not directly usable with libjpeg.
        surf = pygame.Surface(sz, 0, 32)
        for rect, color in squares:
            surf.fill(color, rect)

        # Assume pygame.image.Load works correctly as it is handled by the
        # third party SDL_image library.
        f_path = tempfile.mktemp(suffix='.jpg')
        pygame.image.save(surf, f_path)
        jpg_surf = pygame.image.load(f_path)

        # Allow for small differences in the restored colors.
        def approx(c):
            mask = 0xFC
            return pygame.Color(c.r & mask, c.g & mask, c.b & mask)
        offset = square_len // 2
        for rect, color in squares:
            posn = rect.move((offset, offset)).topleft
            self.assertEqual(approx(jpg_surf.get_at(posn)), approx(color))

    def testSavePNG32(self):
        """ see if we can save a png with color values in the proper channels.
        """
        # Create a PNG file with known colors
        reddish_pixel = (215, 0, 0, 255)
        greenish_pixel = (0, 225, 0, 255)
        bluish_pixel = (0, 0, 235, 255)
        greyish_pixel = (115, 125, 135, 145)

        surf = pygame.Surface((1, 4), pygame.SRCALPHA, 32)
        surf.set_at((0, 0), reddish_pixel)
        surf.set_at((0, 1), greenish_pixel)
        surf.set_at((0, 2), bluish_pixel)
        surf.set_at((0, 3), greyish_pixel)

        f_path = tempfile.mktemp(suffix='.png')
        pygame.image.save(surf, f_path)

        # Read the PNG file and verify that pygame saved it correctly
        reader = png.Reader(filename=f_path)
        width, height, pixels, metadata = reader.asRGBA8()

        # pixels is a generator
        self.assertEqual(tuple(next(pixels)), reddish_pixel)
        self.assertEqual(tuple(next(pixels)), greenish_pixel)
        self.assertEqual(tuple(next(pixels)), bluish_pixel)
        self.assertEqual(tuple(next(pixels)), greyish_pixel)

        if not reader.file.closed:
            reader.file.close()
        del reader
        os.remove(f_path)

    def testSavePNG24(self):
        """ see if we can save a png with color values in the proper channels.
        """
        # Create a PNG file with known colors
        reddish_pixel = (215, 0, 0)
        greenish_pixel = (0, 225, 0)
        bluish_pixel = (0, 0, 235)
        greyish_pixel = (115, 125, 135)

        surf = pygame.Surface((1, 4), 0, 24)
        surf.set_at((0, 0), reddish_pixel)
        surf.set_at((0, 1), greenish_pixel)
        surf.set_at((0, 2), bluish_pixel)
        surf.set_at((0, 3), greyish_pixel)

        f_path = tempfile.mktemp(suffix='.png')
        pygame.image.save(surf, f_path)

        # Read the PNG file and verify that pygame saved it correctly
        reader = png.Reader(filename=f_path)
        width, height, pixels, metadata = reader.asRGB8()

        # pixels is a generator
        self.assertEqual(tuple(next(pixels)), reddish_pixel)
        self.assertEqual(tuple(next(pixels)), greenish_pixel)
        self.assertEqual(tuple(next(pixels)), bluish_pixel)
        self.assertEqual(tuple(next(pixels)), greyish_pixel)

        if not reader.file.closed:
            reader.file.close()
        del reader
        os.remove(f_path)

    def test_save(self):

        s = pygame.Surface((10,10))
        s.fill((23,23,23))
        magic_hex = {}
        magic_hex['jpg'] = [0xff, 0xd8, 0xff, 0xe0]
        magic_hex['png'] = [0x89 ,0x50 ,0x4e ,0x47]
        # magic_hex['tga'] = [0x0, 0x0, 0xa]
        magic_hex['bmp'] = [0x42, 0x4d]


        formats = ["jpg", "png", "bmp"]
        # uppercase too... JPG
        formats = formats + [x.upper() for x in formats]

        for fmt in formats:
            try:
                temp_filename = "%s.%s" % ("tmpimg", fmt)
                pygame.image.save(s, temp_filename)
                # test the magic numbers at the start of the file to ensure they are saved
                #   as the correct file type.
                handle = open(temp_filename, "rb")
                self.assertEqual((1, fmt), (test_magic(handle, magic_hex[fmt.lower()]), fmt))
                handle.close()
                # load the file to make sure it was saved correctly.
                #    Note load can load a jpg saved with a .png file name.
                s2 = pygame.image.load(temp_filename)
                #compare contents, might only work reliably for png...
                #   but because it's all one color it seems to work with jpg.
                self.assertEqual(s2.get_at((0,0)), s.get_at((0,0)))
                handle.close()
            finally:
                #clean up the temp file, comment out to leave tmp file after run.
                os.remove(temp_filename)

    def test_save_colorkey(self):
        """ make sure the color key is not changed when saving.
        """
        s = pygame.Surface((10,10), pygame.SRCALPHA, 32)
        s.fill((23,23,23))
        s.set_colorkey((0,0,0))
        colorkey1 = s.get_colorkey()
        p1 = s.get_at((0,0))

        temp_filename = "tmpimg.png"
        try:
            pygame.image.save(s, temp_filename)
            s2 = pygame.image.load(temp_filename)
        finally:
            os.remove(temp_filename)

        colorkey2 = s.get_colorkey()
        # check that the pixel and the colorkey is correct.
        self.assertEqual(colorkey1, colorkey2)
        self.assertEqual(p1, s2.get_at((0,0)))

    def test_load_unicode_path(self):
        import shutil
        orig = unicode_(example_path("data/asprite.bmp"))
        temp = os.path.join(unicode_(example_path('data')), u'你好.bmp')
        shutil.copy(orig, temp)
        try:
            im = pygame.image.load(temp)
        finally:
            os.remove(temp)

    def _unicode_save(self, temp_file):
        im = pygame.Surface((10, 10), 0, 32)
        try:
            with open(temp_file, 'w') as f:
                pass
            os.remove(temp_file)
        except IOError:
            raise unittest.SkipTest('the path cannot be opened')

        self.assertFalse(os.path.exists(temp_file))

        try:
            pygame.image.save(im, temp_file)

            self.assertGreater(os.path.getsize(temp_file), 10)
        finally:
            try:
                os.remove(temp_file)
            except EnvironmentError:
                pass

    def test_save_unicode_path(self):
        """save unicode object with non-ASCII chars"""
        self._unicode_save(u"你好.bmp")

    def assertPremultipliedAreEqual(self, string1, string2, source_string):
        self.assertEqual(len(string1), len(string2))
        block_size = 20
        if string1 != string2:
            for block_start in xrange_(0, len(string1), block_size):
                block_end = min(block_start + block_size, len(string1))
                block1 = string1[block_start:block_end]
                block2 = string2[block_start:block_end]
                if block1 != block2:
                    source_block = source_string[block_start:block_end]
                    msg = "string difference in %d to %d of %d:\n%s\n%s\nsource:\n%s" % (block_start, block_end, len(string1), block1.encode("hex"), block2.encode("hex"), source_block.encode("hex"))
                    self.fail(msg)

    def test_to_string__premultiplied(self):
        """ test to make sure we can export a surface to a premultiplied alpha string
        """

        def convertRGBAtoPremultiplied(surface_to_modify):
            for x in xrange_(surface_to_modify.get_width()):
                for y in xrange_(surface_to_modify.get_height()):
                    color = surface_to_modify.get_at((x, y))
                    premult_color = (color[0]*color[3]/255,
                                     color[1]*color[3]/255,
                                     color[2]*color[3]/255,
                                     color[3])
                    surface_to_modify.set_at((x, y), premult_color)

        test_surface = pygame.Surface((256, 256), pygame.SRCALPHA, 32)
        for x in xrange_(test_surface.get_width()):
            for y in xrange_(test_surface.get_height()):
                i = x + y*test_surface.get_width()
                test_surface.set_at((x,y), ((i*7) % 256, (i*13) % 256, (i*27) % 256, y))
        premultiplied_copy = test_surface.copy()
        convertRGBAtoPremultiplied(premultiplied_copy)
        self.assertPremultipliedAreEqual(pygame.image.tostring(test_surface, "RGBA_PREMULT"),
                                         pygame.image.tostring(premultiplied_copy, "RGBA"),
                                         pygame.image.tostring(test_surface, "RGBA"))
        self.assertPremultipliedAreEqual(pygame.image.tostring(test_surface, "ARGB_PREMULT"),
                                         pygame.image.tostring(premultiplied_copy, "ARGB"),
                                         pygame.image.tostring(test_surface, "ARGB"))

        no_alpha_surface = pygame.Surface((256, 256), 0, 24)
        self.assertRaises(ValueError, pygame.image.tostring, no_alpha_surface, "RGBA_PREMULT")

    # Custom assert method to check for identical surfaces.
    def _assertSurfaceEqual(self, surf_a, surf_b, msg=None):
        a_width, a_height = surf_a.get_width(), surf_a.get_height()

        # Check a few things to see if the surfaces are equal.
        self.assertEqual(a_width, surf_b.get_width(), msg)
        self.assertEqual(a_height, surf_b.get_height(), msg)
        self.assertEqual(surf_a.get_size(), surf_b.get_size(), msg)
        self.assertEqual(surf_a.get_rect(), surf_b.get_rect(), msg)
        self.assertEqual(surf_a.get_colorkey(), surf_b.get_colorkey(), msg)
        self.assertEqual(surf_a.get_alpha(), surf_b.get_alpha(), msg)
        self.assertEqual(surf_a.get_flags(), surf_b.get_flags(), msg)
        self.assertEqual(surf_a.get_bitsize(), surf_b.get_bitsize(), msg)
        self.assertEqual(surf_a.get_bytesize(), surf_b.get_bytesize(), msg)
        # Anything else?

        # Making the method lookups local for a possible speed up.
        surf_a_get_at = surf_a.get_at
        surf_b_get_at = surf_b.get_at
        for y in xrange_(a_height):
            for x in xrange_(a_width):
                self.assertEqual(surf_a_get_at((x, y)), surf_b_get_at((x, y)),
                                 msg)

    def test_fromstring__and_tostring(self):
        """Ensure methods tostring() and fromstring() are symmetric."""

        ####################################################################
        def RotateRGBAtoARGB(str_buf):
            byte_buf = array.array("B", str_buf)
            num_quads = len(byte_buf)//4
            for i in xrange_(num_quads):
                alpha = byte_buf[i*4 + 3]
                byte_buf[i*4 + 3] = byte_buf[i*4 + 2]
                byte_buf[i*4 + 2] = byte_buf[i*4 + 1]
                byte_buf[i*4 + 1] = byte_buf[i*4 + 0]
                byte_buf[i*4 + 0] = alpha
            return byte_buf.tostring()

        ####################################################################
        def RotateARGBtoRGBA(str_buf):
            byte_buf = array.array("B", str_buf)
            num_quads = len(byte_buf)//4
            for i in xrange_(num_quads):
                alpha = byte_buf[i*4 + 0]
                byte_buf[i*4 + 0] = byte_buf[i*4 + 1]
                byte_buf[i*4 + 1] = byte_buf[i*4 + 2]
                byte_buf[i*4 + 2] = byte_buf[i*4 + 3]
                byte_buf[i*4 + 3] = alpha
            return byte_buf.tostring()

        ####################################################################
        test_surface = pygame.Surface((64, 256), flags=pygame.SRCALPHA,
                                      depth=32)
        for i in xrange_(256):
            for j in xrange_(16):
                intensity = j*16 + 15
                test_surface.set_at((j + 0, i), (intensity, i, i, i))
                test_surface.set_at((j + 16, i), (i, intensity, i, i))
                test_surface.set_at((j + 32, i), (i, i, intensity, i))
                test_surface.set_at((j + 32, i), (i, i, i, intensity))

        self._assertSurfaceEqual(test_surface, test_surface,
                                 'failing with identical surfaces')

        rgba_buf = pygame.image.tostring(test_surface, "RGBA")
        rgba_buf = RotateARGBtoRGBA(RotateRGBAtoARGB(rgba_buf))
        test_rotate_functions = pygame.image.fromstring(
            rgba_buf, test_surface.get_size(), "RGBA")

        self._assertSurfaceEqual(test_surface, test_rotate_functions,
                                 'rotate functions are not symmetric')

        rgba_buf = pygame.image.tostring(test_surface, "RGBA")
        argb_buf = RotateRGBAtoARGB(rgba_buf)
        test_from_argb_string = pygame.image.fromstring(
            argb_buf, test_surface.get_size(), "ARGB")

        self._assertSurfaceEqual(test_surface, test_from_argb_string,
                                 '"RGBA" rotated to "ARGB" failed')

        argb_buf = pygame.image.tostring(test_surface, "ARGB")
        rgba_buf = RotateARGBtoRGBA(argb_buf)
        test_to_argb_string = pygame.image.fromstring(
            rgba_buf, test_surface.get_size(), "RGBA")

        self._assertSurfaceEqual(test_surface, test_to_argb_string,
                                 '"ARGB" rotated to "RGBA" failed')

        for fmt in ('ARGB', 'RGBA'):
            fmt_buf = pygame.image.tostring(test_surface, fmt)
            test_to_from_fmt_string = pygame.image.fromstring(
                fmt_buf, test_surface.get_size(), fmt)

            self._assertSurfaceEqual(test_surface, test_to_from_fmt_string,
                                     'tostring/fromstring functions are not '
                                     'symmetric with "{}" format'.format(fmt))

    def todo_test_frombuffer(self):

        # __doc__ (as of 2008-08-02) for pygame.image.frombuffer:

          # pygame.image.frombuffer(string, size, format): return Surface
          # create a new Surface that shares data inside a string buffer
          #
          # Create a new Surface that shares pixel data directly from the string
          # buffer. This method takes the same arguments as
          # pygame.image.fromstring(), but is unable to vertically flip the
          # source data.
          #
          # This will run much faster than pygame.image.fromstring, since no
          # pixel data must be allocated and copied.

        self.fail()

    def todo_test_get_extended(self):

        # __doc__ (as of 2008-08-02) for pygame.image.get_extended:

          # pygame.image.get_extended(): return bool
          # test if extended image formats can be loaded
          #
          # If pygame is built with extended image formats this function will
          # return True. It is still not possible to determine which formats
          # will be available, but generally you will be able to load them all.

        self.fail()

    def todo_test_load_basic(self):

        # __doc__ (as of 2008-08-02) for pygame.image.load_basic:

          # pygame.image.load(filename): return Surface
          # pygame.image.load(fileobj, namehint=): return Surface
          # load new image from a file

        self.fail()

    def todo_test_load_extended(self):

        # __doc__ (as of 2008-08-02) for pygame.image.load_extended:

          # pygame module for image transfer

        self.fail()

    def todo_test_save_extended(self):

        # __doc__ (as of 2008-08-02) for pygame.image.save_extended:

          # pygame module for image transfer

        self.fail()

    def threads_load(self, images):
        import pygame.threads
        for i in range(10):
            surfs = pygame.threads.tmap(pygame.image.load, images)
            for s in surfs:
                self.assertIsInstance(s, pygame.Surface)

    def test_load_png_threads(self):
        self.threads_load(glob.glob(example_path("data/*.png")))

    def test_load_jpg_threads(self):
        self.threads_load(glob.glob(example_path("data/*.jpg")))

    def test_load_bmp_threads(self):
        self.threads_load(glob.glob(example_path("data/*.bmp")))

    def test_load_gif_threads(self):
        self.threads_load(glob.glob(example_path("data/*.gif")))

if __name__ == '__main__':
    unittest.main()
