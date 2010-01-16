try:
    import pygame2.test.pgunittest as unittest
except:
    import pgunittest as unittest

import sys
import math
from pygame2.base import Color

if sys.version_info >= (3, 0, 0):
    long = int

rgba_vals = [0, 1, 62, 63, 126, 127, 255]

rgba_combinations =  [ (r,g,b,a) for r in rgba_vals
                                 for g in rgba_vals
                                 for b in rgba_vals
                                 for a in rgba_vals ]

################################################################################

def rgba_combos_Color_generator ():
    for rgba in rgba_combinations:
        yield Color(*rgba)

# Python gamma correct
def gamma_correct (rgba_0_255, gamma):
    corrected = round(255.0 * math.pow(rgba_0_255/255.0, gamma))
    return max(min( int(corrected), 255), 0)

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

class ColorTest (unittest.TestCase):
    def test_invalid_html_hex_codes (self):
        # This was a problem with the way 2 digit hex numbers were
        # calculated. The test_hex_digits test is related to the fix.
        self.failUnlessRaises(ValueError, lambda: Color('# f000000'))
        self.failUnlessRaises(ValueError, lambda: Color('#f 000000'))
        self.failUnlessRaises(ValueError, lambda: Color('#-f000000'))

    def test_hex_digits (self):
        # This is an implementation specific test.
        # Two digit hex numbers are calculated using table lookups
        # for the upper and lower digits.
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

        self.failUnless(tuple(Color(255, 0, 0, 0)) == (255, 0, 0, 0))
        self.failUnless(tuple(Color(0, 255, 0, 0)) == (0, 255, 0, 0))
        self.failUnless(tuple(Color(0, 0, 255, 0)) == (0, 0, 255, 0))
        self.failUnless(tuple(Color(0, 0, 0, 255)) == (0, 0, 0, 255))
        
        self.failIf(tuple(Color(0, 0, 0, 0)) == (255, 0, 0, 0))
        self.failIf(tuple(Color(0, 0, 0, 0)) == (0, 255, 0, 0))
        self.failIf(tuple(Color(0, 0, 0, 0)) == (0, 0, 255, 0))
        self.failIf(tuple(Color(0, 0, 0, 0)) == (0, 0, 0, 255))
        
        self.failUnless(tuple(Color(0, 0, 0, 0)) != (255, 0, 0, 0))
        self.failUnless(tuple(Color(0, 0, 0, 0)) != (0, 255, 0, 0))
        self.failUnless(tuple(Color(0, 0, 0, 0)) != (0, 0, 255, 0))
        self.failUnless(tuple(Color(0, 0, 0, 0)) != (0, 0, 0, 255))
        
        self.failIf(tuple(Color(255, 0, 0, 0)) != (255, 0, 0, 0))
        self.failIf(tuple(Color(0, 255, 0, 0))!= (0, 255, 0, 0))
        self.failIf(tuple(Color(0, 0, 255, 0)) != (0, 0, 255, 0))
        self.failIf(tuple(Color(0, 0, 0, 255)) != (0, 0, 0, 255))

        self.failUnless(int(Color(255, 0, 0, 0)) == 0x00ff0000)
        self.failUnless(int(Color(0, 255, 0, 0)) == 0x0000ff00)
        self.failUnless(int(Color(0, 0, 255, 0)) == 0x000000ff)
        self.failUnless(int(Color(0, 0, 0, 255)) == 0xff000000)
        
        self.failIf(int(Color(0, 0, 0, 0)) == 0xff000000)
        self.failIf(int(Color(0, 0, 0, 0)) == 0x00ff0000)
        self.failIf(int(Color(0, 0, 0, 0)) == 0x0000ff00)
        self.failIf(int(Color(0, 0, 0, 0)) == 0x000000ff)
        
        self.failUnless(int(Color(0, 0, 0, 0)) != 0xff000000)
        self.failUnless(int(Color(0, 0, 0, 0)) != 0x00ff0000)
        self.failUnless(int(Color(0, 0, 0, 0)) != 0x0000ff00)
        self.failUnless(int(Color(0, 0, 0, 0)) != 0x000000ff)
        
        self.failIf(int(Color(255, 0, 0, 0)) != 0x00ff0000)
        self.failIf(int(Color(0, 255, 0, 0)) != 0x0000ff00)
        self.failIf(int(Color(0, 0, 255, 0)) != 0x000000ff)
        self.failIf(int(Color(0, 0, 0, 255)) != 0xff000000)

    def test_color (self):
        c = Color (10, 20, 30, 40)
        self.assertEqual (c.r, 10)
        self.assertEqual (c.g, 20)
        self.assertEqual (c.b, 30)
        self.assertEqual (c.a, 40)

        c = Color ("indianred3")
        self.assertEqual (c.r, 205)
        self.assertEqual (c.g, 85)
        self.assertEqual (c.b, 85)
        self.assertEqual (c.a, 255)

        c = Color (0xDDAABBCC)
        self.assertEqual (c.r, 0xAA)
        self.assertEqual (c.g, 0xBB)
        self.assertEqual (c.b, 0xCC)
        self.assertEqual (c.a, 0xDD)

        self.assertRaises (ValueError, Color, 257, 10, 105, 44)
        self.assertRaises (ValueError, Color, 10, 257, 105, 44)
        self.assertRaises (ValueError, Color, 10, 105, 257, 44)
        self.assertRaises (ValueError, Color, 10, 105, 44, 257)
        
    def test_rgba (self):
        c = Color (0)
        self.assertEqual (c.r, 0)
        self.assertEqual (c.g, 0)
        self.assertEqual (c.b, 0)
        self.assertEqual (c.a, 0)

        # Test simple assignments
        c.r = 123
        self.assertEqual (c.r, 123)
        self.assertRaises (ValueError, _assignr, c, 537)
        self.assertEqual (c.r, 123)
        self.assertRaises (ValueError, _assignr, c, -3)
        self.assertEqual (c.r, 123)

        c.g = 55
        self.assertEqual (c.g, 55)
        self.assertRaises (ValueError, _assigng, c, 348)
        self.assertEqual (c.g, 55)
        self.assertRaises (ValueError, _assigng, c, -44)
        self.assertEqual (c.g, 55)

        c.b = 77
        self.assertEqual (c.b, 77)
        self.assertRaises (ValueError, _assignb, c, 256)
        self.assertEqual (c.b, 77)
        self.assertRaises (ValueError, _assignb, c, -12)
        self.assertEqual (c.b, 77)

        c.a = 255
        self.assertEqual (c.a, 255)
        self.assertRaises (ValueError, _assigna, c, 312)
        self.assertEqual (c.a, 255)
        self.assertRaises (ValueError, _assigna, c, -10)
        self.assertEqual (c.a, 255)

    def test_repr (self):
        c = Color (68, 38, 26, 69)
        t = "Color(68, 38, 26, 69)"
        self.assertEqual (repr (c), t)

    def test_add (self):
        c1 = Color (0)
        self.assertEqual (c1.r, 0)
        self.assertEqual (c1.g, 0)
        self.assertEqual (c1.b, 0)
        self.assertEqual (c1.a, 0)

        c2 = Color (20, 33, 82, 193)
        self.assertEqual (c2.r, 20)
        self.assertEqual (c2.g, 33)
        self.assertEqual (c2.b, 82)
        self.assertEqual (c2.a, 193)

        c3 = c1 + c2
        self.assertEqual (c3.r, 20)
        self.assertEqual (c3.g, 33)
        self.assertEqual (c3.b, 82)
        self.assertEqual (c3.a, 193)

        c3 = c3 + c2
        self.assertEqual (c3.r, 40)
        self.assertEqual (c3.g, 66)
        self.assertEqual (c3.b, 164)
        self.assertEqual (c3.a, 255)
    
    def test_sub (self):
        c1 = Color (0xFFFFFFFF)
        self.assertEqual (c1.r, 255)
        self.assertEqual (c1.g, 255)
        self.assertEqual (c1.b, 255)
        self.assertEqual (c1.a, 255)

        c2 = Color (20, 33, 82, 193)
        self.assertEqual (c2.r, 20)
        self.assertEqual (c2.g, 33)
        self.assertEqual (c2.b, 82)
        self.assertEqual (c2.a, 193)

        c3 = c1 - c2
        self.assertEqual (c3.r, 235)
        self.assertEqual (c3.g, 222)
        self.assertEqual (c3.b, 173)
        self.assertEqual (c3.a, 62)

        c3 = c3 - c2
        self.assertEqual (c3.r, 215)
        self.assertEqual (c3.g, 189)
        self.assertEqual (c3.b, 91)
        self.assertEqual (c3.a, 0)
    
    def test_mul (self):
        c1 = Color (0x01010101)
        self.assertEqual (c1.r, 1)
        self.assertEqual (c1.g, 1)
        self.assertEqual (c1.b, 1)
        self.assertEqual (c1.a, 1)

        c2 = Color (2, 5, 3, 22)
        self.assertEqual (c2.r, 2)
        self.assertEqual (c2.g, 5)
        self.assertEqual (c2.b, 3)
        self.assertEqual (c2.a, 22)

        c3 = c1 * c2
        self.assertEqual (c3.r, 2)
        self.assertEqual (c3.g, 5)
        self.assertEqual (c3.b, 3)
        self.assertEqual (c3.a, 22)

        c3 = c3 * c2
        self.assertEqual (c3.r, 4)
        self.assertEqual (c3.g, 25)
        self.assertEqual (c3.b, 9)
        self.assertEqual (c3.a, 255)

    def test_div (self):
        c1 = Color (0x80808080)
        self.assertEqual (c1.r, 128)
        self.assertEqual (c1.g, 128)
        self.assertEqual (c1.b, 128)
        self.assertEqual (c1.a, 128)

        c2 = Color (2, 4, 8, 16)
        self.assertEqual (c2.r, 2)
        self.assertEqual (c2.g, 4)
        self.assertEqual (c2.b, 8)
        self.assertEqual (c2.a, 16)

        c3 = c1 / c2
        self.assertEqual (c3.r, 64)
        self.assertEqual (c3.g, 32)
        self.assertEqual (c3.b, 16)
        self.assertEqual (c3.a, 8)

        c3 = c3 / c2
        self.assertEqual (c3.r, 32)
        self.assertEqual (c3.g, 8)
        self.assertEqual (c3.b, 2)
        self.assertEqual (c3.a, 0)
    
    def test_mod (self):
        c1 = Color (0xFFFFFFFF)
        self.assertEqual (c1.r, 255)
        self.assertEqual (c1.g, 255)
        self.assertEqual (c1.b, 255)
        self.assertEqual (c1.a, 255)

        c2 = Color (2, 4, 8, 16)
        self.assertEqual (c2.r, 2)
        self.assertEqual (c2.g, 4)
        self.assertEqual (c2.b, 8)
        self.assertEqual (c2.a, 16)

        c3 = c1 % c2
        self.assertEqual (c3.r, 1)
        self.assertEqual (c3.g, 3)
        self.assertEqual (c3.b, 7)
        self.assertEqual (c3.a, 15)

    def test_float (self):
        c = Color (0xCC00CC00)
        self.assertEqual (c.r, 0)
        self.assertEqual (c.g, 204)
        self.assertEqual (c.b, 0)
        self.assertEqual (c.a, 204)
        self.assertEqual (float (c), float (0xCC00CC00))

        c = Color (0x33727592)
        self.assertEqual (c.r, 114)
        self.assertEqual (c.g, 117)
        self.assertEqual (c.b, 146)
        self.assertEqual (c.a, 51)
        self.assertEqual (float (c), float (0x33727592))

    def test_oct (self):
        c = Color (0xCC00CC00)
        self.assertEqual (c.r, 0)
        self.assertEqual (c.g, 204)
        self.assertEqual (c.b, 0)
        self.assertEqual (c.a, 204)
        self.assertEqual (oct (c), oct (0xCC00CC00))

        c = Color (0x33727592)
        self.assertEqual (c.r, 114)
        self.assertEqual (c.g, 117)
        self.assertEqual (c.b, 146)
        self.assertEqual (c.a, 51)
        self.assertEqual (oct (c), oct (0x33727592))

    def test_hex (self):
        c = Color (0xCC00CC00)
        self.assertEqual (c.r, 0)
        self.assertEqual (c.g, 204)
        self.assertEqual (c.b, 0)
        self.assertEqual (c.a, 204)
        self.assertEqual (hex (c), hex (0xCC00CC00))

        c = Color (0x33727592)
        self.assertEqual (c.r, 114)
        self.assertEqual (c.g, 117)
        self.assertEqual (c.b, 146)
        self.assertEqual (c.a, 51)
        self.assertEqual (hex (c), hex (0x33727592))

    def test_webstyle(self):
        c = Color ("#CC00CC11")
        self.assertEqual (c.r, 204)
        self.assertEqual (c.g, 0)
        self.assertEqual (c.b, 204)
        self.assertEqual (c.a, 17)
        self.assertEqual (hex (c), hex (0x11CC00CC))

        c = Color ("#CC00CC")
        self.assertEqual (c.r, 204)
        self.assertEqual (c.g, 0)
        self.assertEqual (c.b, 204)
        self.assertEqual (c.a, 255)
        self.assertEqual (hex (c), hex (0xFFCC00CC))

        c = Color ("0xCC00CC11")
        self.assertEqual (c.r, 204)
        self.assertEqual (c.g, 0)
        self.assertEqual (c.b, 204)
        self.assertEqual (c.a, 17)
        self.assertEqual (hex (c), hex (0x11CC00CC))

        c = Color ("0xCC00CC")
        self.assertEqual (c.r, 204)
        self.assertEqual (c.g, 0)
        self.assertEqual (c.b, 204)
        self.assertEqual (c.a, 255)
        self.assertEqual (hex (c), hex (0xFFCC00CC))

        self.assertRaises (ValueError, Color, "#cc00qq")
        self.assertRaises (ValueError, Color, "0xcc00qq")
        self.assertRaises (ValueError, Color, "09abcdef")
        self.assertRaises (ValueError, Color, "09abcde")
        self.assertRaises (ValueError, Color, "quarky")

    def test_int (self):
        # This will be a long
        c = Color (0xCC00CC00)
        self.assertEqual (c.r, 0)
        self.assertEqual (c.g, 204)
        self.assertEqual (c.b, 0)
        self.assertEqual (c.a, 204)
        self.assertEqual (int (c), int (0xCC00CC00))

        # This will be an int
        c = Color (0x33727592)
        self.assertEqual (c.r, 114)
        self.assertEqual (c.g, 117)
        self.assertEqual (c.b, 146)
        self.assertEqual (c.a, 51)
        self.assertEqual (int (c), int (0x33727592))

    def test_long (self):
        # This will be a long
        c = Color (0xCC00CC00)
        self.assertEqual (c.r, 0)
        self.assertEqual (c.g, 204)
        self.assertEqual (c.b, 0)
        self.assertEqual (c.a, 204)
        self.assertEqual (long (c), long (0xCC00CC00))

        # This will be an int
        c = Color (0x33727592)
        self.assertEqual (c.r, 114)
        self.assertEqual (c.g, 117)
        self.assertEqual (c.b, 146)
        self.assertEqual (c.a, 51)
        self.assertEqual (long (c), long (0x33727592))

    def test_normalize (self):
        c = Color (204, 38, 194, 55)
        self.assertEqual (c.r, 204)
        self.assertEqual (c.g, 38)
        self.assertEqual (c.b, 194)
        self.assertEqual (c.a, 55)

        t = c.normalize ()

        self.assertAlmostEquals (t[0], 0.800000, places=5)
        self.assertAlmostEquals (t[1], 0.149016, places=5)
        self.assertAlmostEquals (t[2], 0.760784, places=5)
        self.assertAlmostEquals (t[3], 0.215686, places=5)

    def test_len (self):
        c = Color (204, 38, 194, 55)
        self.assertEqual (len (c), 4)

    def test_get_item (self):
        c = Color (204, 38, 194, 55)
        self.assertEqual (c[0], 204)
        self.assertEqual (c[1], 38)
        self.assertEqual (c[2], 194)
        self.assertEqual (c[3], 55)

    def test_set_item (self):
        c = Color (204, 38, 194, 55)
        self.assertEqual (c[0], 204)
        self.assertEqual (c[1], 38)
        self.assertEqual (c[2], 194)
        self.assertEqual (c[3], 55)

        c[0] = 33
        self.assertEqual (c[0], 33)
        c[1] = 48
        self.assertEqual (c[1], 48)
        c[2] = 173
        self.assertEqual (c[2], 173)
        c[3] = 213
        self.assertEqual (c[3], 213)

        # Now try some 'invalid' ones
        self.assertRaises (ValueError, _assign_item, c, 1, -83)
        self.assertEqual (c[1], 48)
        self.assertRaises (TypeError, _assign_item, c, 2, "Hello")
        self.assertEqual (c[2], 173)

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

####################### COLORSPACE PROPERTY SANITY TESTS #######################

    def colorspaces_converted_should_not_raise (self, prop):
        fails = 0

        x = 0
        for c in rgba_combos_Color_generator():
            x += 1
            
            other = Color(0)
            
            try:
                setattr(other, prop, getattr(c, prop))
                #eg other.hsla = c.hsla

            except ValueError:
                fails += 1

        self.assert_(x > 0, "x is combination counter, 0 means no tests!")
        self.assertEqual((fails, x), (0, x))

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
            other = Color(0)

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
        # gamma_correct defined at top of page

        gammas = list (map(lambda i: i / 10.0, range(1, 31))) # [0.1 .. 3.0]
        gammas_len = len(gammas)

        for i, c in enumerate(rgba_combos_Color_generator()):
            gamma = gammas[i % gammas_len]

            corrected = Color(*[gamma_correct(x, gamma) for x in tuple(c)])
            lib_corrected = c.correct_gamma(gamma)

            self.assert_(corrected.r == lib_corrected.r)
            self.assert_(corrected.g == lib_corrected.g)
            self.assert_(corrected.b == lib_corrected.b)
            self.assert_(corrected.a == lib_corrected.a)

        # TODO: test against statically defined verified _correct_ values
        # assert corrected.r == 125 etc.

################################################################################

    def test_pygame2_base_Color_a(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Color.a:

        # Gets or sets the alpha value of the Color.
        c = Color (100, 100, 100)
        self.assertEqual (c.a, 255)
        
        c = Color (100, 100, 100, 100)
        self.assertEqual (c.a, 100)
        
        c = Color (100, 100, 100)
        self.assertEqual (c.a, 255)
        c.a = 200
        self.assertEqual (c.a, 200)
        c.a += 22
        self.assertEqual (c.a, 222)
        
        c = Color (0xAABBCCDD)
        self.assertEqual (c.a, 0xAA)
        
        c = Color ("0xBBCCDD")
        self.assertEqual (c.a, 0xFF)
        c = Color ("0xAABBCCDD")
        self.assertEqual (c.a, 0xDD)
        
        c = Color ("#AABBCCDD")
        self.assertEqual (c.a, 0xDD)
        c = Color ("#BBCCDD")
        self.assertEqual (c.a, 0xFF)
        
        c = Color ("red")
        self.assertEqual (c.a, 255)

    def test_pygame2_base_Color_b(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Color.b:

        # Gets or sets the blue value of the Color.
        c = Color (100, 100, 100)
        self.assertEqual (c.b, 100)
        
        c = Color (100, 100, 100, 100)
        self.assertEqual (c.b, 100)
        
        c = Color (100, 100, 100)
        self.assertEqual (c.b, 100)
        c.b = 200
        self.assertEqual (c.b, 200)
        c.b += 22
        self.assertEqual (c.b, 222)
        
        c = Color (0xAABBCCDD)
        self.assertEqual (c.b, 0xDD)
        
        c = Color ("0xBBCCDD")
        self.assertEqual (c.b, 0xDD)
        c = Color ("0xAABBCCDD")
        self.assertEqual (c.b, 0xCC)
        
        c = Color ("#AABBCCDD")
        self.assertEqual (c.b, 0xCC)
        c = Color ("#BBCCDD")
        self.assertEqual (c.b, 0xDD)
        
        c = Color ("red")
        self.assertEqual (c.b, 0)
        c = Color ("blue")
        self.assertEqual (c.b, 255)

    def todo_test_pygame2_base_Color_cmy(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Color.cmy:

        # The CMY representation of the Color. The CMY components are in the
        # ranges C = [0, 1], M = [0, 1], Y = [0, 1]. Note that this will not
        # return the absolutely exact CMY values for the set RGB values in all
        # cases. Due to the RGB mapping from 0-255 and the CMY mapping from
        # 0-1 rounding errors may cause the CMY values to differ slightly from
        # what you might expect.

        self.fail() 

    def todo_test_pygame2_base_Color_correct_gamma(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Color.correct_gamma:

        # Color.correct_gamma (gamma) -> Color
        # 
        # Applies a certain gamma value to the Color.
        # 
        # Applies a certain gamma value to the Color and returns a new
        # Color with the adjusted RGBA values.

        self.fail() 

    def test_pygame2_base_Color_g(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Color.g:

        # Gets or sets the green value of the Color.

        c = Color (100, 100, 100)
        self.assertEqual (c.g, 100)
        
        c = Color (100, 100, 100, 100)
        self.assertEqual (c.g, 100)
        
        c = Color (100, 100, 100)
        self.assertEqual (c.g, 100)
        c.g = 200
        self.assertEqual (c.g, 200)
        c.g += 22
        self.assertEqual (c.g, 222)
        
        c = Color (0xAABBCCDD)
        self.assertEqual (c.g, 0xCC)
        
        c = Color ("0xBBCCDD")
        self.assertEqual (c.g, 0xCC)
        c = Color ("0xAABBCCDD")
        self.assertEqual (c.g, 0xBB)
        
        c = Color ("#AABBCCDD")
        self.assertEqual (c.g, 0xBB)
        c = Color ("#BBCCDD")
        self.assertEqual (c.g, 0xCC)
        
        c = Color ("red")
        self.assertEqual (c.g, 0)
        c = Color ("green")
        self.assertEqual (c.g, 255)

    def todo_test_pygame2_base_Color_hsla(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Color.hsla:

        # The HSLA representation of the Color. The HSLA components are in the
        # ranges H = [0, 360], S = [0, 100], L = [0, 100], A = [0, 100]. Note
        # that this will not return the absolutely exact HSL values for the
        # set RGB values in all cases. Due to the RGB mapping from 0-255 and
        # the HSL mapping from 0-100 and 0-360 rounding errors may cause the
        # HSL values to differ slightly from what you might expect.

        self.fail() 

    def todo_test_pygame2_base_Color_hsva(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Color.hsva:

        # The HSVA representation of the Color. The HSVA components are in the
        # ranges H = [0, 360], S = [0, 100], V = [0, 100], A = [0, 100]. Note
        # that this will not return the absolutely exact HSV values for the
        # set RGB values in all cases. Due to the RGB mapping from 0-255 and
        # the HSV mapping from 0-100 and 0-360 rounding errors may cause the
        # HSV values to differ slightly from what you might expect.

        self.fail() 

    def todo_test_pygame2_base_Color_i1i2i3(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Color.i1i2i3:

        # The I1I2I3 representation of the Color. The I1I2I3 components are in
        # the ranges I1 = [0, 1], I2 = [-0.5, 0.5], I3 = [-0.5, 0.5]. Note
        # that this will not return the absolutely exact I1I2I3 values for the
        # set RGB values in all cases. Due to the RGB mapping from 0-255 and
        # the I1I2I3 mapping from 0-1 rounding errors may cause the I1I2I3
        # values to differ slightly from what you might expect.

        self.fail() 

    def test_pygame2_base_Color_normalize(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Color.normalize:

        # Color.normalize() -> tuple
        # 
        # Returns the normalized RGBA values of the Color.
        # 
        # Returns the normalized RGBA values of the Color as floating
        # point values.
        c = Color (255, 255, 255, 255)
        self.assertEqual (c.normalize (), (1.0, 1.0, 1.0, 1.0))
        c = Color (0, 0, 0, 0)
        self.assertEqual (c.normalize (), (0.0, 0.0, 0.0, 0.0))
        c = Color (128, 128, 128, 128)
        t = c.normalize ()
        for v in t:
            self.assertAlmostEquals (v, 0.5, 2)

        c = Color (128, 255, 0, 52)
        t = c.normalize ()
        self.assertAlmostEquals (t[0], 0.5, 2)
        self.assertEqual (t[1], 1.0)
        self.assertEqual (t[2], 0.0)
        # 52 / 255 ~= .20
        self.assertAlmostEquals (t[3], 0.2, 2)

    def test_pygame2_base_Color_r(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Color.r:

        # Gets or sets the red value of the Color.
        
        c = Color (100, 100, 100)
        self.assertEqual (c.r, 100)
        
        c = Color (100, 100, 100, 100)
        self.assertEqual (c.r, 100)
        
        c = Color (100, 100, 100)
        self.assertEqual (c.r, 100)
        c.r = 200
        self.assertEqual (c.r, 200)
        c.r += 22
        self.assertEqual (c.r, 222)
        
        c = Color (0xAABBCCDD)
        self.assertEqual (c.r, 0xBB)
        
        c = Color ("0xBBCCDD")
        self.assertEqual (c.r, 0xBB)
        c = Color ("0xAABBCCDD")
        self.assertEqual (c.r, 0xAA)
        
        c = Color ("#AABBCCDD")
        self.assertEqual (c.r, 0xAA)
        c = Color ("#BBCCDD")
        self.assertEqual (c.r, 0xBB)
        
        c = Color ("red")
        self.assertEqual (c.r, 255)
        c = Color ("blue")
        self.assertEqual (c.r, 0)

if __name__ == "__main__":
    unittest.main ()
