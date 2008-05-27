import unittest
import pygame

def _assignr (x, y):
    x.r = y

def _assigng (x, y):
    x.g = y

def _assignb (x, y):
    x.b = y

def _assigna (x, y):
    x.a = y

class ColorTest (unittest.TestCase):
    def test_color (self):
        c = pygame.Color (10, 20, 30, 40)
        self.assertEquals (c.r, 10)
        self.assertEquals (c.g, 20)
        self.assertEquals (c.b, 30)
        self.assertEquals (c.a, 40)
        
        c = pygame.Color ("indianred3")
        self.assertEquals (c.r, 205)
        self.assertEquals (c.g, 85)
        self.assertEquals (c.b, 85)
        self.assertEquals (c.a, 255)
        
        c = pygame.Color (0xAABBCCDD)
        self.assertEquals (c.r, 0xAA)
        self.assertEquals (c.g, 0xBB)
        self.assertEquals (c.b, 0xCC)
        self.assertEquals (c.a, 0xDD)

        self.assertRaises (ValueError, pygame.Color, 257, 10, 105, 44)
        self.assertRaises (ValueError, pygame.Color, 10, 257, 105, 44)
        self.assertRaises (ValueError, pygame.Color, 10, 105, 257, 44)
        self.assertRaises (ValueError, pygame.Color, 10, 105, 44, 257)

    def test_rgba (self):
        c = pygame.Color (0)
        self.assertEquals (c.r, 0)
        self.assertEquals (c.g, 0)
        self.assertEquals (c.b, 0)
        self.assertEquals (c.a, 0)

        # Test simple assignments
        c.r = 123
        self.assertEquals (c.r, 123)
        self.assertRaises (ValueError, _assignr, c, 537)
        self.assertEquals (c.r, 123)
        self.assertRaises (ValueError, _assignr, c, -3)
        self.assertEquals (c.r, 123)

        c.g = 55
        self.assertEquals (c.g, 55)
        self.assertRaises (ValueError, _assigng, c, 348)
        self.assertEquals (c.g, 55)
        self.assertRaises (ValueError, _assigng, c, -44)
        self.assertEquals (c.g, 55)

        c.b = 77
        self.assertEquals (c.b, 77)
        self.assertRaises (ValueError, _assignb, c, 256)
        self.assertEquals (c.b, 77)
        self.assertRaises (ValueError, _assignb, c, -12)
        self.assertEquals (c.b, 77)

        c.a = 255
        self.assertEquals (c.a, 255)
        self.assertRaises (ValueError, _assigna, c, 312)
        self.assertEquals (c.a, 255)
        self.assertRaises (ValueError, _assigna, c, -10)
        self.assertEquals (c.a, 255)

    def test_repr (self):
        c = pygame.Color (68, 38, 26, 69)
        t = "(68, 38, 26, 69)"
        self.assertEquals (repr (c), t)

    def test_add (self):
        c1 = pygame.Color (0)
        self.assertEquals (c1.r, 0)
        self.assertEquals (c1.g, 0)
        self.assertEquals (c1.b, 0)
        self.assertEquals (c1.a, 0)

        c2 = pygame.Color (20, 33, 82, 193)
        self.assertEquals (c2.r, 20)
        self.assertEquals (c2.g, 33)
        self.assertEquals (c2.b, 82)
        self.assertEquals (c2.a, 193)

        c3 = c1 + c2
        self.assertEquals (c3.r, 20)
        self.assertEquals (c3.g, 33)
        self.assertEquals (c3.b, 82)
        self.assertEquals (c3.a, 193)

        c3 = c3 + c2
        self.assertEquals (c3.r, 40)
        self.assertEquals (c3.g, 66)
        self.assertEquals (c3.b, 164)
        self.assertEquals (c3.a, 255)

    def test_sub (self):
        c1 = pygame.Color (0xFFFFFFFF)
        self.assertEquals (c1.r, 255)
        self.assertEquals (c1.g, 255)
        self.assertEquals (c1.b, 255)
        self.assertEquals (c1.a, 255)

        c2 = pygame.Color (20, 33, 82, 193)
        self.assertEquals (c2.r, 20)
        self.assertEquals (c2.g, 33)
        self.assertEquals (c2.b, 82)
        self.assertEquals (c2.a, 193)

        c3 = c1 - c2
        self.assertEquals (c3.r, 235)
        self.assertEquals (c3.g, 222)
        self.assertEquals (c3.b, 173)
        self.assertEquals (c3.a, 62)

        c3 = c3 - c2
        self.assertEquals (c3.r, 215)
        self.assertEquals (c3.g, 189)
        self.assertEquals (c3.b, 91)
        self.assertEquals (c3.a, 0)

if __name__ == '__main__':
    unittest.main()
