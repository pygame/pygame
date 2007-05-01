


import unittest
import pygame, pygame.image, pygame.pkgdata
import os

import array


class ImageTest( unittest.TestCase ):
    
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
        """ see if we can load a png.
        """
        f = os.path.join("examples", "data", "alien1.png")
        surf = pygame.image.load(f)

        f = open(os.path.join("examples", "data", "alien1.png"), "rb")
        surf = pygame.image.load(f)


    def testLoadJPG(self):
        """ see if we can load a jpg.
        """
        f = os.path.join("examples", "data", "alien1.jpg")
        surf = pygame.image.load(f)

        f = open(os.path.join("examples", "data", "alien1.jpg"), "rb")
        surf = pygame.image.load(f)


    def test_from_to_string(self):
        """ see if fromstring, and tostring methods are symetrical.
        """


        def AreSurfacesIdentical(surf_a, surf_b):
            if surf_a.get_width() != surf_b.get_width() or surf_a.get_height() != surf_b.get_height():
                return False
            for y in xrange(surf_a.get_height()):
                for x in xrange(surf_b.get_width()):
                    if surf_a.get_at((x,y)) != surf_b.get_at((x,y)):
                        return False
            return True

        ####################################################################
        def RotateRGBAtoARGB(str_buf):
            byte_buf = array.array("B", str_buf)
            num_quads = len(byte_buf)/4
            for i in xrange(num_quads):
                alpha = byte_buf[i*4 + 3]
                byte_buf[i*4 + 3] = byte_buf[i*4 + 2]
                byte_buf[i*4 + 2] = byte_buf[i*4 + 1]
                byte_buf[i*4 + 1] = byte_buf[i*4 + 0]
                byte_buf[i*4 + 0] = alpha
            return byte_buf.tostring()

        ####################################################################
        def RotateARGBtoRGBA(str_buf):
            byte_buf = array.array("B", str_buf)
            num_quads = len(byte_buf)/4
            for i in xrange(num_quads):
                alpha = byte_buf[i*4 + 0]
                byte_buf[i*4 + 0] = byte_buf[i*4 + 1]
                byte_buf[i*4 + 1] = byte_buf[i*4 + 2]
                byte_buf[i*4 + 2] = byte_buf[i*4 + 3]
                byte_buf[i*4 + 3] = alpha
            return byte_buf.tostring()
                
        ####################################################################
        test_surface = pygame.Surface((48, 256), flags=pygame.SRCALPHA, depth=32)
        for i in xrange(256):
            for j in xrange(16):
                intensity = j*16 + 15
                test_surface.set_at((j + 0, i), (intensity, i, i, 255))
                test_surface.set_at((j + 16, i), (i, intensity, i, 255))
                test_surface.set_at((j + 32, i), (i, i, intensity, 255))
            
        self.assertTrue(AreSurfacesIdentical(test_surface, test_surface))

        rgba_buf = pygame.image.tostring(test_surface, "RGBA")
        rgba_buf = RotateARGBtoRGBA(RotateRGBAtoARGB(rgba_buf))
        test_rotate_functions = pygame.image.fromstring(rgba_buf, test_surface.get_size(), "RGBA")

        self.assertTrue(AreSurfacesIdentical(test_surface, test_rotate_functions))

        rgba_buf = pygame.image.tostring(test_surface, "RGBA")
        argb_buf = RotateRGBAtoARGB(rgba_buf)
        test_from_argb_string = pygame.image.fromstring(argb_buf, test_surface.get_size(), "ARGB")

        self.assertTrue(AreSurfacesIdentical(test_surface, test_from_argb_string))
        #"ERROR: image.fromstring with ARGB failed"


        argb_buf = pygame.image.tostring(test_surface, "ARGB")
        rgba_buf = RotateARGBtoRGBA(argb_buf)
        test_to_argb_string = pygame.image.fromstring(rgba_buf, test_surface.get_size(), "RGBA")

        self.assertTrue(AreSurfacesIdentical(test_surface, test_to_argb_string))
        #"ERROR: image.tostring with ARGB failed"


        argb_buf = pygame.image.tostring(test_surface, "ARGB")
        test_to_from_argb_string = pygame.image.fromstring(argb_buf, test_surface.get_size(), "ARGB")

        self.assertTrue(AreSurfacesIdentical(test_surface, test_to_from_argb_string))
        #"ERROR: image.fromstring and image.tostring with ARGB are not symmetric"





if __name__ == '__main__':
    unittest.main()
