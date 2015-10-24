#################################### IMPORTS ###################################

from __future__ import generators

if __name__ == '__main__':
    import sys
    import os
    pkg_dir = os.path.split(os.path.abspath(__file__))[0]
    parent_dir, pkg_name = os.path.split(pkg_dir)
    is_pygame_pkg = (pkg_name == 'tests' and
                     os.path.split(parent_dir)[1] == 'pygame')
    if not is_pygame_pkg:
        sys.path.insert(0, parent_dir)
else:
    is_pygame_pkg = __name__.startswith('pygame.tests.')

if is_pygame_pkg:
    from pygame.tests.test_utils import test_not_implemented, unittest
else:
    from test.test_utils import test_not_implemented, unittest
import pygame
from pygame.compat import long_
import math

################################### CONSTANTS ##################################

rgba_vals = [0, 1, 62, 63, 126, 127, 255]

rgba_combinations =  [ (r,g,b,a) for r in rgba_vals
                                 for g in rgba_vals
                                 for b in rgba_vals
                                 for a in rgba_vals ]

################################################################################

def rgba_combos_Color_generator ():
    for rgba in rgba_combinations:
        yield pygame.Color(*rgba)

# Python gamma correct
def gamma_correct (rgba_0_255, gamma):
    corrected = round(255.0 * math.pow(rgba_0_255/255.0, gamma))
    return max(min( int(corrected), 255), 0)

################################################################################

# TODO: add tests for
# correct_gamma()  -- test against statically defined verified correct values
# coerce ()        --  ??

def _assignr (x, y):
    x.r = y

def _assigng (x, y):
    x.g = y

def _assignb (x, y):
    x.b = y

def _assigna (x, y):
    x.a = y

def _assign_item (x, p, y):
    x[p] = y

class ColorTypeTest (unittest.TestCase):
    def test_invalid_html_hex_codes(self):
        # This was a problem with the way 2 digit hex numbers were
        # calculated. The test_hex_digits test is related to the fix.
        Color = pygame.color.Color
        self.failUnlessRaises(ValueError, lambda: Color('# f000000'))
        self.failUnlessRaises(ValueError, lambda: Color('#f 000000'))
        self.failUnlessRaises(ValueError, lambda: Color('#-f000000'))

    def test_hex_digits(self):
        # This is an implementation specific test.
        # Two digit hex numbers are calculated using table lookups
        # for the upper and lower digits.
        Color = pygame.color.Color
        self.assertEqual(Color('#00000000').r, 0x00)
        self.assertEqual(Color('#10000000').r, 0x10)
        self.assertEqual(Color('#20000000').r, 0x20)
        self.assertEqual(Color('#30000000').r, 0x30)
        self.assertEqual(Color('#40000000').r, 0x40)
        self.assertEqual(Color('#50000000').r, 0x50)
        self.assertEqual(Color('#60000000').r, 0x60)
        self.assertEqual(Color('#70000000').r, 0x70)
        self.assertEqual(Color('#80000000').r, 0x80)
        self.assertEqual(Color('#90000000').r, 0x90)
        self.assertEqual(Color('#A0000000').r, 0xA0)
        self.assertEqual(Color('#B0000000').r, 0xB0)
        self.assertEqual(Color('#C0000000').r, 0xC0)
        self.assertEqual(Color('#D0000000').r, 0xD0)
        self.assertEqual(Color('#E0000000').r, 0xE0)
        self.assertEqual(Color('#F0000000').r, 0xF0)
        self.assertEqual(Color('#01000000').r, 0x01)
        self.assertEqual(Color('#02000000').r, 0x02)
        self.assertEqual(Color('#03000000').r, 0x03)
        self.assertEqual(Color('#04000000').r, 0x04)
        self.assertEqual(Color('#05000000').r, 0x05)
        self.assertEqual(Color('#06000000').r, 0x06)
        self.assertEqual(Color('#07000000').r, 0x07)
        self.assertEqual(Color('#08000000').r, 0x08)
        self.assertEqual(Color('#09000000').r, 0x09)
        self.assertEqual(Color('#0A000000').r, 0x0A)
        self.assertEqual(Color('#0B000000').r, 0x0B)
        self.assertEqual(Color('#0C000000').r, 0x0C)
        self.assertEqual(Color('#0D000000').r, 0x0D)
        self.assertEqual(Color('#0E000000').r, 0x0E)
        self.assertEqual(Color('#0F000000').r, 0x0F)

    def test_comparison(self):
        Color = pygame.color.Color

        # Check valid comparisons
        self.failUnless(Color(255, 0, 0, 0) == Color(255, 0, 0, 0))
        self.failUnless(Color(0, 255, 0, 0) == Color(0, 255, 0, 0))
        self.failUnless(Color(0, 0, 255, 0) == Color(0, 0, 255, 0))
        self.failUnless(Color(0, 0, 0, 255) == Color(0, 0, 0, 255))
        self.failIf(Color(0, 0, 0, 0) == Color(255, 0, 0, 0))
        self.failIf(Color(0, 0, 0, 0) == Color(0, 255, 0, 0))
        self.failIf(Color(0, 0, 0, 0) == Color(0, 0, 255, 0))
        self.failIf(Color(0, 0, 0, 0) == Color(0, 0, 0, 255))
        self.failUnless(Color(0, 0, 0, 0) != Color(255, 0, 0, 0))
        self.failUnless(Color(0, 0, 0, 0) != Color(0, 255, 0, 0))
        self.failUnless(Color(0, 0, 0, 0) != Color(0, 0, 255, 0))
        self.failUnless(Color(0, 0, 0, 0) != Color(0, 0, 0, 255))
        self.failIf(Color(255, 0, 0, 0) != Color(255, 0, 0, 0))
        self.failIf(Color(0, 255, 0, 0) != Color(0, 255, 0, 0))
        self.failIf(Color(0, 0, 255, 0) != Color(0, 0, 255, 0))
        self.failIf(Color(0, 0, 0, 255) != Color(0, 0, 0, 255))

        self.failUnless(Color(255, 0, 0, 0) == (255, 0, 0, 0))
        self.failUnless(Color(0, 255, 0, 0) == (0, 255, 0, 0))
        self.failUnless(Color(0, 0, 255, 0) == (0, 0, 255, 0))
        self.failUnless(Color(0, 0, 0, 255) == (0, 0, 0, 255))
        self.failIf(Color(0, 0, 0, 0) == (255, 0, 0, 0))
        self.failIf(Color(0, 0, 0, 0) == (0, 255, 0, 0))
        self.failIf(Color(0, 0, 0, 0) == (0, 0, 255, 0))
        self.failIf(Color(0, 0, 0, 0) == (0, 0, 0, 255))
        self.failUnless(Color(0, 0, 0, 0) != (255, 0, 0, 0))
        self.failUnless(Color(0, 0, 0, 0) != (0, 255, 0, 0))
        self.failUnless(Color(0, 0, 0, 0) != (0, 0, 255, 0))
        self.failUnless(Color(0, 0, 0, 0) != (0, 0, 0, 255))
        self.failIf(Color(255, 0, 0, 0) != (255, 0, 0, 0))
        self.failIf(Color(0, 255, 0, 0) != (0, 255, 0, 0))
        self.failIf(Color(0, 0, 255, 0) != (0, 0, 255, 0))
        self.failIf(Color(0, 0, 0, 255) != (0, 0, 0, 255))

        self.failUnless((255, 0, 0, 0) == Color(255, 0, 0, 0))
        self.failUnless((0, 255, 0, 0) == Color(0, 255, 0, 0))
        self.failUnless((0, 0, 255, 0) == Color(0, 0, 255, 0))
        self.failUnless((0, 0, 0, 255) == Color(0, 0, 0, 255))
        self.failIf((0, 0, 0, 0) == Color(255, 0, 0, 0))
        self.failIf((0, 0, 0, 0) == Color(0, 255, 0, 0))
        self.failIf((0, 0, 0, 0) == Color(0, 0, 255, 0))
        self.failIf((0, 0, 0, 0) == Color(0, 0, 0, 255))
        self.failUnless((0, 0, 0, 0) != Color(255, 0, 0, 0))
        self.failUnless((0, 0, 0, 0) != Color(0, 255, 0, 0))
        self.failUnless((0, 0, 0, 0) != Color(0, 0, 255, 0))
        self.failUnless((0, 0, 0, 0) != Color(0, 0, 0, 255))
        self.failIf((255, 0, 0, 0) != Color(255, 0, 0, 0))
        self.failIf((0, 255, 0, 0) != Color(0, 255, 0, 0))
        self.failIf((0, 0, 255, 0) != Color(0, 0, 255, 0))
        self.failIf((0, 0, 0, 255) != Color(0, 0, 0, 255))

        class TupleSubclass(tuple):
            pass
        self.failUnless(Color(255, 0, 0, 0) == TupleSubclass((255, 0, 0, 0)))
        self.failUnless(TupleSubclass((255, 0, 0, 0)) == Color(255, 0, 0, 0))
        self.failIf(Color(255, 0, 0, 0) != TupleSubclass((255, 0, 0, 0)))
        self.failIf(TupleSubclass((255, 0, 0, 0)) != Color(255, 0, 0, 0))

        # These are not supported so will be unequal.
        self.failIf(Color(255, 0, 0, 0) == "#ff000000")
        self.failUnless(Color(255, 0, 0, 0) != "#ff000000")

        self.failIf("#ff000000" == Color(255, 0, 0, 0))
        self.failUnless("#ff000000" != Color(255, 0, 0, 0))

        self.failIf(Color(255, 0, 0, 0) == 0xff000000)
        self.failUnless(Color(255, 0, 0, 0) != 0xff000000)

        self.failIf(0xff000000 == Color(255, 0, 0, 0))
        self.failUnless(0xff000000 != Color(255, 0, 0, 0))

        self.failIf(Color(255, 0, 0, 0) == [255, 0, 0, 0])
        self.failUnless(Color(255, 0, 0, 0) != [255, 0, 0, 0])

        self.failIf([255, 0, 0, 0] == Color(255, 0, 0 ,0))
        self.failUnless([255, 0, 0, 0] != Color(255, 0, 0, 0))

        # Comparison is not implemented for invalid color values.
        class Test(object):
            def __eq__(self, other):
                return -1
            def __ne__(self, other):
                return -2
        class TestTuple(tuple):
            def __eq__(self, other):
                return -1
            def __ne__(self, other):
                return -2

        t = Test()
        t_tuple = TestTuple(('a', 0, 0, 0))
        black = Color('black')
        self.assertEqual(black == t, -1)
        self.assertEqual(t == black, -1)
        self.assertEqual(black != t, -2)
        self.assertEqual(t != black, -2)
        self.assertEqual(black == t_tuple, -1)
        self.assertEqual(black != t_tuple, -2)
        self.assertEqual(t_tuple == black, -1)
        self.assertEqual(t_tuple != black, -2)

    def test_ignore_whitespace(self):
        self.assertEquals(pygame.color.Color('red'), pygame.color.Color(' r e d '))

    def test_slice(self):
        #"""|tags: python3_ignore|"""

        # slicing a color gives you back a tuple.
        # do all sorts of slice combinations.
        c = pygame.Color(1,2,3,4)

        self.assertEquals((1,2,3,4), c[:])
        self.assertEquals((1,2,3), c[:-1])

        self.assertEquals((), c[:-5])

        self.assertEquals((1,2,3,4), c[:4])
        self.assertEquals((1,2,3,4), c[:5])
        self.assertEquals((1,2), c[:2])
        self.assertEquals((1,), c[:1])
        self.assertEquals((), c[:0])


        self.assertEquals((2,), c[1:-2])
        self.assertEquals((3, 4), c[-2:])
        self.assertEquals((4,), c[-1:])


        # NOTE: assigning to a slice is currently unsupported.


    def test_unpack(self):
        # should be able to unpack to r,g,b,a and r,g,b
        c = pygame.Color(1,2,3,4)
        r,g,b,a = c
        self.assertEquals((1,2,3,4), (r,g,b,a))
        self.assertEquals(c, (r,g,b,a))

        c.set_length(3)
        r,g,b = c
        self.assertEquals((1,2,3), (r,g,b))






    def test_length(self):
        # should be able to unpack to r,g,b,a and r,g,b
        c = pygame.Color(1,2,3,4)
        self.assertEquals(len(c), 4)

        c.set_length(3)
        self.assertEquals(len(c), 3)

        # it keeps the old alpha anyway...
        self.assertEquals(c.a, 4)

        # however you can't get the alpha in this way:
        self.assertRaises (IndexError, lambda x:c[x], 4)



        c.set_length(4)
        self.assertEquals(len(c), 4)
        self.assertEquals(len(c), 4)

        self.assertRaises (ValueError, c.set_length, 5)
        self.assertRaises (ValueError, c.set_length, -1)
        self.assertRaises (ValueError, c.set_length, 0)
        self.assertRaises (ValueError, c.set_length, pow(2,long_(33)))


    def test_case_insensitivity_of_string_args(self):
        self.assertEquals(pygame.color.Color('red'), pygame.color.Color('Red'))

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

    def test_mul (self):
        c1 = pygame.Color (0x01010101)
        self.assertEquals (c1.r, 1)
        self.assertEquals (c1.g, 1)
        self.assertEquals (c1.b, 1)
        self.assertEquals (c1.a, 1)

        c2 = pygame.Color (2, 5, 3, 22)
        self.assertEquals (c2.r, 2)
        self.assertEquals (c2.g, 5)
        self.assertEquals (c2.b, 3)
        self.assertEquals (c2.a, 22)

        c3 = c1 * c2
        self.assertEquals (c3.r, 2)
        self.assertEquals (c3.g, 5)
        self.assertEquals (c3.b, 3)
        self.assertEquals (c3.a, 22)

        c3 = c3 * c2
        self.assertEquals (c3.r, 4)
        self.assertEquals (c3.g, 25)
        self.assertEquals (c3.b, 9)
        self.assertEquals (c3.a, 255)

    def test_div (self):
        c1 = pygame.Color (0x80808080)
        self.assertEquals (c1.r, 128)
        self.assertEquals (c1.g, 128)
        self.assertEquals (c1.b, 128)
        self.assertEquals (c1.a, 128)

        c2 = pygame.Color (2, 4, 8, 16)
        self.assertEquals (c2.r, 2)
        self.assertEquals (c2.g, 4)
        self.assertEquals (c2.b, 8)
        self.assertEquals (c2.a, 16)

        c3 = c1 // c2
        self.assertEquals (c3.r, 64)
        self.assertEquals (c3.g, 32)
        self.assertEquals (c3.b, 16)
        self.assertEquals (c3.a, 8)

        c3 = c3 // c2
        self.assertEquals (c3.r, 32)
        self.assertEquals (c3.g, 8)
        self.assertEquals (c3.b, 2)
        self.assertEquals (c3.a, 0)

    def test_mod (self):
        c1 = pygame.Color (0xFFFFFFFF)
        self.assertEquals (c1.r, 255)
        self.assertEquals (c1.g, 255)
        self.assertEquals (c1.b, 255)
        self.assertEquals (c1.a, 255)

        c2 = pygame.Color (2, 4, 8, 16)
        self.assertEquals (c2.r, 2)
        self.assertEquals (c2.g, 4)
        self.assertEquals (c2.b, 8)
        self.assertEquals (c2.a, 16)

        c3 = c1 % c2
        self.assertEquals (c3.r, 1)
        self.assertEquals (c3.g, 3)
        self.assertEquals (c3.b, 7)
        self.assertEquals (c3.a, 15)

    def test_float (self):
        c = pygame.Color (0xCC00CC00)
        self.assertEquals (c.r, 204)
        self.assertEquals (c.g, 0)
        self.assertEquals (c.b, 204)
        self.assertEquals (c.a, 0)
        self.assertEquals (float (c), float (0xCC00CC00))

        c = pygame.Color (0x33727592)
        self.assertEquals (c.r, 51)
        self.assertEquals (c.g, 114)
        self.assertEquals (c.b, 117)
        self.assertEquals (c.a, 146)
        self.assertEquals (float (c), float (0x33727592))

    def test_oct (self):
        c = pygame.Color (0xCC00CC00)
        self.assertEquals (c.r, 204)
        self.assertEquals (c.g, 0)
        self.assertEquals (c.b, 204)
        self.assertEquals (c.a, 0)
        self.assertEquals (oct (c), oct (0xCC00CC00))

        c = pygame.Color (0x33727592)
        self.assertEquals (c.r, 51)
        self.assertEquals (c.g, 114)
        self.assertEquals (c.b, 117)
        self.assertEquals (c.a, 146)
        self.assertEquals (oct (c), oct (0x33727592))

    def test_hex (self):
        c = pygame.Color (0xCC00CC00)
        self.assertEquals (c.r, 204)
        self.assertEquals (c.g, 0)
        self.assertEquals (c.b, 204)
        self.assertEquals (c.a, 0)
        self.assertEquals (hex (c), hex (0xCC00CC00))

        c = pygame.Color (0x33727592)
        self.assertEquals (c.r, 51)
        self.assertEquals (c.g, 114)
        self.assertEquals (c.b, 117)
        self.assertEquals (c.a, 146)
        self.assertEquals (hex (c), hex (0x33727592))


    def test_webstyle(self):
        c = pygame.Color ("#CC00CC11")
        self.assertEquals (c.r, 204)
        self.assertEquals (c.g, 0)
        self.assertEquals (c.b, 204)
        self.assertEquals (c.a, 17)
        self.assertEquals (hex (c), hex (0xCC00CC11))

        c = pygame.Color ("#CC00CC")
        self.assertEquals (c.r, 204)
        self.assertEquals (c.g, 0)
        self.assertEquals (c.b, 204)
        self.assertEquals (c.a, 255)
        self.assertEquals (hex (c), hex (0xCC00CCFF))

        c = pygame.Color ("0xCC00CC11")
        self.assertEquals (c.r, 204)
        self.assertEquals (c.g, 0)
        self.assertEquals (c.b, 204)
        self.assertEquals (c.a, 17)
        self.assertEquals (hex (c), hex (0xCC00CC11))

        c = pygame.Color ("0xCC00CC")
        self.assertEquals (c.r, 204)
        self.assertEquals (c.g, 0)
        self.assertEquals (c.b, 204)
        self.assertEquals (c.a, 255)
        self.assertEquals (hex (c), hex (0xCC00CCFF))

        self.assertRaises (ValueError, pygame.Color, "#cc00qq")
        self.assertRaises (ValueError, pygame.Color, "0xcc00qq")
        self.assertRaises (ValueError, pygame.Color, "09abcdef")
        self.assertRaises (ValueError, pygame.Color, "09abcde")
        self.assertRaises (ValueError, pygame.Color, "quarky")

    def test_int (self):
        # This will be a long
        c = pygame.Color (0xCC00CC00)
        self.assertEquals (c.r, 204)
        self.assertEquals (c.g, 0)
        self.assertEquals (c.b, 204)
        self.assertEquals (c.a, 0)
        self.assertEquals (int (c), int (0xCC00CC00))

        # This will be an int
        c = pygame.Color (0x33727592)
        self.assertEquals (c.r, 51)
        self.assertEquals (c.g, 114)
        self.assertEquals (c.b, 117)
        self.assertEquals (c.a, 146)
        self.assertEquals (int (c), int (0x33727592))

    def test_long (self):
        # This will be a long
        c = pygame.Color (0xCC00CC00)
        self.assertEquals (c.r, 204)
        self.assertEquals (c.g, 0)
        self.assertEquals (c.b, 204)
        self.assertEquals (c.a, 0)
        self.assertEquals (long_ (c), long_ (0xCC00CC00))

        # This will be an int
        c = pygame.Color (0x33727592)
        self.assertEquals (c.r, 51)
        self.assertEquals (c.g, 114)
        self.assertEquals (c.b, 117)
        self.assertEquals (c.a, 146)
        self.assertEquals (long_ (c), long_ (0x33727592))

    def test_normalize (self):
        c = pygame.Color (204, 38, 194, 55)
        self.assertEquals (c.r, 204)
        self.assertEquals (c.g, 38)
        self.assertEquals (c.b, 194)
        self.assertEquals (c.a, 55)

        t = c.normalize ()

        self.assertAlmostEquals (t[0], 0.800000, 5)
        self.assertAlmostEquals (t[1], 0.149016, 5)
        self.assertAlmostEquals (t[2], 0.760784, 5)
        self.assertAlmostEquals (t[3], 0.215686, 5)

    def test_len (self):
        c = pygame.Color (204, 38, 194, 55)
        self.assertEquals (len (c), 4)

    def test_get_item (self):
        c = pygame.Color (204, 38, 194, 55)
        self.assertEquals (c[0], 204)
        self.assertEquals (c[1], 38)
        self.assertEquals (c[2], 194)
        self.assertEquals (c[3], 55)

    def test_set_item (self):
        c = pygame.Color (204, 38, 194, 55)
        self.assertEquals (c[0], 204)
        self.assertEquals (c[1], 38)
        self.assertEquals (c[2], 194)
        self.assertEquals (c[3], 55)

        c[0] = 33
        self.assertEquals (c[0], 33)
        c[1] = 48
        self.assertEquals (c[1], 48)
        c[2] = 173
        self.assertEquals (c[2], 173)
        c[3] = 213
        self.assertEquals (c[3], 213)

        # Now try some 'invalid' ones
        self.assertRaises (ValueError, _assign_item, c, 0, 95.485)
        self.assertEquals (c[0], 33)
        self.assertRaises (ValueError, _assign_item, c, 1, -83)
        self.assertEquals (c[1], 48)
        self.assertRaises (ValueError, _assign_item, c, 2, "Hello")
        self.assertEquals (c[2], 173)

    def test_Color_type_works_for_Surface_get_and_set_colorkey(self):
        s = pygame.Surface((32, 32))

        c = pygame.Color(33, 22, 11, 255)
        s.set_colorkey(c)

        get_r, get_g, get_b, get_a = s.get_colorkey()

        self.assert_(get_r == c.r)
        self.assert_(get_g == c.g)
        self.assert_(get_b == c.b)
        self.assert_(get_a == c.a)

########## HSLA, HSVA, CMY, I1I2I3 ALL ELEMENTS WITHIN SPECIFIED RANGE #########

    def test_hsla__all_elements_within_limits (self):
        for c in rgba_combos_Color_generator():
            h, s, l, a = c.hsla
            self.assert_(0 <= h <= 360)
            self.assert_(0 <= s <= 100)
            self.assert_(0 <= l <= 100)
            self.assert_(0 <= a <= 100)

    def test_hsva__all_elements_within_limits (self):
        for c in rgba_combos_Color_generator():
            h, s, v, a = c.hsva
            self.assert_(0 <= h <= 360)
            self.assert_(0 <= s <= 100)
            self.assert_(0 <= v <= 100)
            self.assert_(0 <= a <= 100)

    def test_cmy__all_elements_within_limits (self):
        for c in rgba_combos_Color_generator():
            c, m, y = c.cmy
            self.assert_(0 <= c <= 1)
            self.assert_(0 <= m <= 1)
            self.assert_(0 <= y <= 1)

    def test_i1i2i3__all_elements_within_limits (self):
        for c in rgba_combos_Color_generator():
            i1, i2, i3 = c.i1i2i3
            self.assert_(  0   <= i1 <= 1)
            self.assert_( -0.5 <= i2 <= 0.5)
            self.assert_( -0.5 <= i3 <= 0.5)

    def test_issue_269 (self):
        """PyColor OverflowError on HSVA with hue value of 360

           >>> c = pygame.Color(0)
           >>> c.hsva = (360,0,0,0)
           Traceback (most recent call last):
             File "<stdin>", line 1, in <module>
           OverflowError: this is not allowed to happen ever
           >>> pygame.ver
           '1.9.1release'
           >>>

        """

        c = pygame.Color(0)
        c.hsva = 360, 0, 0, 0
        self.assertEqual(c.hsva, (0, 0, 0, 0))
        c.hsva = 360, 100, 100, 100
        self.assertEqual(c.hsva, (0, 100, 100, 100))
        self.assertEqual(c, (255, 0, 0, 255))

####################### COLORSPACE PROPERTY SANITY TESTS #######################

    def colorspaces_converted_should_not_raise (self, prop):
        fails = 0

        x = 0
        for c in rgba_combos_Color_generator():
            x += 1

            other = pygame.Color(0)

            try:
                setattr(other, prop, getattr(c, prop))
                #eg other.hsla = c.hsla

            except ValueError:
                fails += 1

        self.assert_(x > 0, "x is combination counter, 0 means no tests!")
        self.assert_((fails, x) == (0, x))

    def test_hsla__sanity_testing_converted_should_not_raise (self):
        self.colorspaces_converted_should_not_raise('hsla')

    def test_hsva__sanity_testing_converted_should_not_raise (self):
        self.colorspaces_converted_should_not_raise('hsva')

    def test_cmy__sanity_testing_converted_should_not_raise (self):
        self.colorspaces_converted_should_not_raise('cmy')

    def test_i1i2i3__sanity_testing_converted_should_not_raise (self):
        self.colorspaces_converted_should_not_raise('i1i2i3')

################################################################################

    def colorspaces_converted_should_equate_bar_rounding (self, prop):
        for c in rgba_combos_Color_generator():
            other = pygame.Color(0)

            try:
                setattr(other, prop, getattr(c, prop))
                #eg other.hsla = c.hsla

                self.assert_(abs(other.r - c.r) <= 1)
                self.assert_(abs(other.b - c.b) <= 1)
                self.assert_(abs(other.g - c.g) <= 1)
                # CMY and I1I2I3 do not care about the alpha
                if not prop in ("cmy", "i1i2i3"):
                    self.assert_(abs(other.a - c.a) <= 1)

            except ValueError:
                pass        # other tests will notify, this tests equation

    def test_hsla__sanity_testing_converted_should_equate_bar_rounding(self):
        self.colorspaces_converted_should_equate_bar_rounding('hsla')

    def test_hsva__sanity_testing_converted_should_equate_bar_rounding(self):
        self.colorspaces_converted_should_equate_bar_rounding('hsva')

    def test_cmy__sanity_testing_converted_should_equate_bar_rounding(self):
        self.colorspaces_converted_should_equate_bar_rounding('cmy')

    def test_i1i2i3__sanity_testing_converted_should_equate_bar_rounding(self):
        self.colorspaces_converted_should_equate_bar_rounding('i1i2i3')

################################################################################

    def test_correct_gamma__verified_against_python_implementation(self):
        "|tags:slow|"
        # gamma_correct defined at top of page

        gammas = [i / 10.0 for i in range(1, 31)]  # [0.1 ... 3.0]
        gammas_len = len(gammas)

        for i, c in enumerate(rgba_combos_Color_generator()):
            gamma = gammas[i % gammas_len]

            corrected = pygame.Color(*[gamma_correct(x, gamma)
                                                 for x in tuple(c)])
            lib_corrected = c.correct_gamma(gamma)

            self.assert_(corrected.r == lib_corrected.r)
            self.assert_(corrected.g == lib_corrected.g)
            self.assert_(corrected.b == lib_corrected.b)
            self.assert_(corrected.a == lib_corrected.a)

        # TODO: test against statically defined verified _correct_ values
        # assert corrected.r == 125 etc.


    def test_pickle(self):
        import pickle
        c1 = pygame.Color(1,2,3,4)
        #c2 = pygame.Color(255,254,253,252)
        pickle_string = pickle.dumps(c1)
        c1_frompickle = pickle.loads(pickle_string)
        self.assertEqual(c1,c1_frompickle)

################################################################################
# only available if ctypes module is also available

    def test_arraystruct(self):
        import pygame.tests.test_utils.arrinter as ai
        import ctypes as ct

        c_byte_p = ct.POINTER(ct.c_byte)
        c = pygame.Color(5, 7, 13, 23)
        flags = (ai.PAI_CONTIGUOUS | ai.PAI_FORTRAN |
                 ai.PAI_ALIGNED | ai.PAI_NOTSWAPPED)
        for i in range(1, 5):
            c.set_length(i)
            inter = ai.ArrayInterface(c)
            self.assertEqual(inter.two, 2)
            self.assertEqual(inter.nd, 1)
            self.assertEqual(inter.typekind, 'u')
            self.assertEqual(inter.itemsize, 1)
            self.assertEqual(inter.flags, flags)
            self.assertEqual(inter.shape[0], i)
            self.assertEqual(inter.strides[0], 1)
            data = ct.cast(inter.data, c_byte_p)
            for j in range(i):
                self.assertEqual(data[j], c[j])

    if pygame.HAVE_NEWBUF:
        def test_newbuf(self):
            self.NEWBUF_test_newbuf()
        if is_pygame_pkg:
            from pygame.tests.test_utils import buftools
        else:
            from test.test_utils import buftools

    def NEWBUF_test_newbuf(self):
        from ctypes import cast, POINTER, c_uint8
        buftools = self.buftools

        class ColorImporter(buftools.Importer):
            def __init__(self, color, flags):
                super(ColorImporter, self).__init__(color, flags)
                self.items = cast(self.buf, POINTER(c_uint8))
            def __getitem__(self, index):
                if 0 <= index < 4:
                    return self.items[index]
                raise IndexError("valid index values are between 0 and 3: "
                                 "got {}".format(index))
            def __setitem__(self, index, value):
                if 0 <= index < 4:
                    self.items[index] = value
                else:
                    raise IndexError("valid index values are between 0 and 3: "
                                     "got {}".format(index))

        c = pygame.Color(50, 100, 150, 200)
        imp = ColorImporter(c, buftools.PyBUF_SIMPLE)
        self.assertTrue(imp.obj is c)
        self.assertEqual(imp.ndim, 0)
        self.assertEqual(imp.itemsize, 1)
        self.assertEqual(imp.len, 4)
        self.assertTrue(imp.readonly)
        self.assertTrue(imp.format is None)
        self.assertTrue(imp.shape is None)
        self.assertTrue(imp.strides is None)
        self.assertTrue(imp.suboffsets is None)
        for i in range(4):
            self.assertEqual(c[i], imp[i])
        imp[0] = 60
        self.assertEqual(c.r, 60)
        imp[1] = 110
        self.assertEqual(c.g, 110)
        imp[2] = 160
        self.assertEqual(c.b, 160)
        imp[3] = 210
        self.assertEqual(c.a, 210)
        imp = ColorImporter(c, buftools.PyBUF_FORMAT)
        self.assertEqual(imp.ndim, 0)
        self.assertEqual(imp.itemsize, 1)
        self.assertEqual(imp.len, 4)
        self.assertEqual(imp.format, 'B')
        self.assertEqual(imp.ndim, 0)
        self.assertEqual(imp.itemsize, 1)
        self.assertEqual(imp.len, 4)
        imp = ColorImporter(c, buftools.PyBUF_ND)
        self.assertEqual(imp.ndim, 1)
        self.assertEqual(imp.itemsize, 1)
        self.assertEqual(imp.len, 4)
        self.assertTrue(imp.format is None)
        self.assertEqual(imp.shape, (4,))
        self.assertEqual(imp.strides, None)
        imp = ColorImporter(c, buftools.PyBUF_STRIDES)
        self.assertEqual(imp.ndim, 1)
        self.assertTrue(imp.format is None)
        self.assertEqual(imp.shape, (4,))
        self.assertEqual(imp.strides, (1,))
        imp = ColorImporter(c, buftools.PyBUF_C_CONTIGUOUS)
        self.assertEqual(imp.ndim, 1)
        imp = ColorImporter(c, buftools.PyBUF_F_CONTIGUOUS)
        self.assertEqual(imp.ndim, 1)
        imp = ColorImporter(c, buftools.PyBUF_ANY_CONTIGUOUS)
        self.assertEqual(imp.ndim, 1)
        for i in range(1, 5):
            c.set_length(i)
            imp = ColorImporter(c, buftools.PyBUF_ND)
            self.assertEqual(imp.ndim, 1)
            self.assertEqual(imp.len, i)
            self.assertEqual(imp.shape, (i,))
        self.assertRaises(BufferError, ColorImporter,
                          c, buftools.PyBUF_WRITABLE)

    try:
        import ctypes
    except ImportError:
        del test_arraystruct


################################################################################

if __name__ == '__main__':
    unittest.main()
