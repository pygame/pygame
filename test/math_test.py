try:
    import pygame2.test.pgunittest as unittest
except:
    import pgunittest as unittest

import copy
import math
import pygame2
import pygame2.math as pmath
from pygame2.math import Vector, Vector2, Vector3

class MathTest (unittest.TestCase):

    def test_pygame2_math_base_Vector_copy (self):
        #
        # Test the __copy__ implementation and copy constructor
        #
        v1 = Vector ((1, 2, 3))
        self.assertEqual (v1.dimension, 3)

        v2 = copy.copy (v1)
        self.assertEqual (v1, v2)
        v3 = copy.deepcopy (v1)
        self.assertEqual (v1, v3)
        v4 = Vector (v1)
        self.assertEqual (v1, v4)

    def test_pygame2_math_base_vector_from_polar(self):

        # __doc__ (as of 2010-01-09) for pygame2.math.base.vector_from_polar:

        # vector_from_polar (p1, p2) -> Vector2
        e1 = Vector2(1, 0)
        e2 = Vector2(0, 1)
        v1 = Vector2(1.2, 3.4)

        v = Vector2()
        from_polar = pmath.vector_from_polar
        v = from_polar(*v1.as_polar())
        self.assertEqual(v1, v)
        self.assertEqual(e1.as_polar(), (1, 0))
        self.assertEqual(e2.as_polar(), (1, 90))
        self.assertEqual((2 * e2).as_polar(), (2, 90))
        self.assertRaises(TypeError, lambda : from_polar((None, None)))
        self.assertRaises(TypeError, lambda : from_polar("ab"))
        self.assertRaises(TypeError, lambda : from_polar((None, 1)))
        self.assertRaises(TypeError, lambda : from_polar((1, 2, 3)))
        self.assertRaises(TypeError, lambda : from_polar((1,)))
        v = from_polar(.5, 90)
        self.assertEqual(v, .5 * e2)

    def test_pygame2_math_base_vector_from_spherical(self):

        # __doc__ (as of 2010-01-09) for pygame2.math.base.vector_from_spherical:

        # vector_from_spherical (p1, p2, p3) -> Vector2
        e1 = Vector3(1, 0, 0)
        e2 = Vector3(0, 1, 0)
        e3 = Vector3(0, 0, 1)
        v1 = Vector3(1.2, 3.4, 9.6)
        
        v = Vector3()
        from_spherical = pmath.vector_from_spherical
        v = from_spherical(*v1.as_spherical())
        self.assertEqual(v1, v)
        self.assertEqual(e1.as_spherical(), (1, 90, 0))
        self.assertEqual(e2.as_spherical(), (1, 90, 90))
        self.assertEqual(e3.as_spherical(), (1, 0, 0))
        self.assertEqual((2 * e2).as_spherical(), (2, 90, 90))
        self.assertRaises(TypeError, lambda : from_spherical((None, None, None)))
        self.assertRaises(TypeError, lambda : from_spherical("abc"))
        self.assertRaises(TypeError, lambda : from_spherical((None, 1, 2)))
        self.assertRaises(TypeError, lambda : from_spherical((1, 2, 3, 4)))
        self.assertRaises(TypeError, lambda : from_spherical((1, 2)))
        self.assertRaises(TypeError, lambda : from_spherical((1, 2, 3)))
        v = from_spherical(.5, 90, 90)
        self.assertEqual(v, .5 * e2)

if __name__ == "__main__":
    unittest.main ()

