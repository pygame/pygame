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

import math
from pygame.math import Vector2
from time import clock

class Vector2TypeTest(unittest.TestCase):
    def testConstructionDefault(self):
        v = Vector2()
        self.assertEqual(v.x, 0.)
        self.assertEqual(v.y, 0.)

    def testConstructionXY(self):
        v = Vector2(1.2, 3.4)
        self.assertEqual(v.x, 1.2)
        self.assertEqual(v.y, 3.4)

    def testConstructionTuple(self):
        v = Vector2((1.2, 3.4))
        self.assertEqual(v.x, 1.2)
        self.assertEqual(v.y, 3.4)

    def testConstructionList(self):
        v = Vector2([1.2, 3.4])
        self.assertEqual(v.x, 1.2)
        self.assertEqual(v.y, 3.4)

    def testConstructionVector2(self):
        v = Vector2(Vector2(1.2, 3.4))
        self.assertEqual(v.x, 1.2)
        self.assertEqual(v.y, 3.4)

    def testSequence(self):
        v = Vector2(1.2, 3.4)
        self.assertEqual(len(v), 2)
        self.assertEqual(v[0], 1.2)
        self.assertEqual(v[1], 3.4)
        self.assertRaises(IndexError, lambda : v[2])
        self.assertEqual(v[-1], 3.4)
        self.assertEqual(v[-2], 1.2)
        self.assertRaises(IndexError, lambda : v[-3])
        self.assertEqual(v[:], [1.2, 3.4])
        self.assertEqual(v[1:], [3.4])
        self.assertEqual(v[:1], [1.2])
        self.assertEqual(list(v), [1.2, 3.4])
        self.assertEqual(tuple(v), (1.2, 3.4))
        v[0] = 5.6
        v[1] = 7.8
        self.assertEqual(v.x, 5.6)
        self.assertEqual(v.y, 7.8)
        v[:] = [9.1, 11.12]
        self.assertEqual(v.x, 9.1)
        self.assertEqual(v.y, 11.12)
        def overpopulate():
            v = Vector2()
            v[:] = [1, 2, 3]
        self.assertRaises(ValueError, overpopulate)
        def underpopulate():
            v = Vector2()
            v[:] = [1]
        self.assertRaises(ValueError, underpopulate)

    def testAdd(self):
        v1 = Vector2(1.2, 3.4)
        v2 = Vector2(5.6, 7.8)
        v3 = v1 + v2
        self.assert_(isinstance(v3, Vector2))
        self.assertEqual(v3.x, 6.8)
        self.assertEqual(v3.y, 11.2)
        v3 = v1 + (5.6, 7.8)
        self.assert_(isinstance(v3, Vector2))
        self.assertEqual(v3.x, 6.8)
        self.assertEqual(v3.y, 11.2)
        v3 = v1 + [5.6, 7.8]
        self.assert_(isinstance(v3, Vector2))
        self.assertEqual(v3.x, 6.8)
        self.assertEqual(v3.y, 11.2)
        v3 = (1.2, 3.4) + v2
        self.assert_(isinstance(v3, Vector2))
        self.assertEqual(v3.x, 6.8)
        self.assertEqual(v3.y, 11.2)
        v3 = [1.2, 3.4] + v2
        self.assert_(isinstance(v3, Vector2))
        self.assertEqual(v3.x, 6.8)
        self.assertEqual(v3.y, 11.2)

    def testSub(self):
        v1 = Vector2(1.2, 3.4)
        v2 = Vector2(5.6, 7.9)
        v3 = v1 - v2
        self.assert_(isinstance(v3, Vector2))
        self.assertEqual(v3.x, 1.2 - 5.6)
        self.assertEqual(v3.y, 3.4 - 7.9)
        v3 = v1 - (5.6, 7.9)
        self.assert_(isinstance(v3, Vector2))
        self.assertEqual(v3.x, 1.2 - 5.6)
        self.assertEqual(v3.y, 3.4 - 7.9)
        v3 = v1 - [5.6, 7.9]
        self.assert_(isinstance(v3, Vector2))
        self.assertEqual(v3.x, 1.2 - 5.6)
        self.assertEqual(v3.y, 3.4 - 7.9)
        v3 = (1.2, 3.4) - v2
        self.assert_(isinstance(v3, Vector2))
        self.assertEqual(v3.x, 1.2 - 5.6)
        self.assertEqual(v3.y, 3.4 - 7.9)
        v3 = [1.2, 3.4] - v2
        self.assert_(isinstance(v3, Vector2))
        self.assertEqual(v3.x, 1.2 - 5.6)
        self.assertEqual(v3.y, 3.4 - 7.9)

    def testScalarMultiplication(self):
        v1 = Vector2(1.2, 3.4)
        v2 = 5.6 * v1
        self.assertEqual(v2.x, 5.6 * 1.2)
        self.assertEqual(v2.y, 5.6 * 3.4)
        v2 = v1 * 7.8
        self.assertEqual(v2.x, 1.2 * 7.8)
        self.assertEqual(v2.y, 3.4 * 7.8)

    def testScalarDivision(self):
        v1 = Vector2(1.2, 3.4)
        v2 = v1 / 5.6
        self.assertAlmostEqual(v2.x, 1.2 / 5.6)
        self.assertAlmostEqual(v2.y, 3.4 / 5.6)
        v2 = v1 // -1.2
        self.assertEqual(v2.x, 1.2 // -1.2)
        self.assertEqual(v2.y, 3.4 // -1.2)

    def testBool(self):
        v0 = Vector2(0.0, 0.0)
        v1 = Vector2(1.2, 3.4)
        self.assertEqual(bool(v0), False)
        self.assertEqual(bool(v1), True)
        self.assert_(not v0)
        self.assert_(v1)

    def testUnary(self):
        v1 = Vector2(1.2, 3.4)
        v2 = +v1
        self.assertEqual(v2.x, 1.2)
        self.assertEqual(v2.y, 3.4)
        self.assertNotEqual(id(v1), id(v2))
        v3 = -v1
        self.assertEqual(v3.x, -1.2)
        self.assertEqual(v3.y, -3.4)
        self.assertNotEqual(id(v1), id(v3))
        
    def testCompare(self):
        int_vec = Vector2(3, -2)
        flt_vec = Vector2(3.0, -2.0)
        zero_vec = Vector2(0, 0)
        self.assertEqual(int_vec == flt_vec, True)
        self.assertEqual(int_vec != flt_vec, False)
        self.assertEqual(int_vec != zero_vec, True)
        self.assertEqual(flt_vec == zero_vec, False)
        self.assertEqual(int_vec == (3, -2), True)
        self.assertEqual(int_vec != (3, -2), False)
        self.assertEqual(int_vec != [0, 0], True)
        self.assertEqual(int_vec == [0, 0], False)
        self.assertEqual(int_vec != 5, True)
        self.assertEqual(int_vec == 5, False)
        self.assertEqual(int_vec != [3, -2, 0], True)
        self.assertEqual(int_vec == [3, -2, 0], False)

    def testStr(self):
        v = Vector2(1.2, 3.4)
        self.assertEqual(str(v), "[1.2, 3.4]")

    def testRepr(self):
        v = Vector2(1.2, 3.4)
        self.assertEqual(v.__repr__(), "<Vector2(1.2, 3.4)>")
        self.assertEqual(v, Vector2(v.__repr__()))

    def testIter(self):
        v = Vector2(1.2, 3.4)
        it = v.__iter__()
        self.assertEqual(it.next(), 1.2)
        self.assertEqual(it.next(), 3.4)
        self.assertRaises(StopIteration, lambda : it.next())
        it1 = v.__iter__()
        it2 = v.__iter__()
        self.assertNotEqual(id(it1), id(it2))
        self.assertEqual(id(it1), id(it1.__iter__()))
        self.assertEqual(list(it1), list(it2));
        self.assertEqual(list(v.__iter__()), [1.2, 3.4])
        idx = 0
        for val in v:
            self.assertEqual(val, v[idx])
            idx += 1
        
    def test_rotate(self):
        v1 = Vector2(1, 0)
        v2 = v1.rotate(90)
        v3 = v1.rotate(90 + 360)
        self.assertEqual(v1.x, 1)
        self.assertEqual(v1.y, 0)
        self.assertEqual(v2.x, 0)
        self.assertEqual(v2.y, 1)
        self.assertEqual(v3.x, v2.x)
        self.assertEqual(v3.y, v2.y)
        v1 = Vector2(-1, -1)
        v2 = v1.rotate(-90)
        self.assertEqual(v2.x, -1)
        self.assertEqual(v2.y, 1)
        v2 = v1.rotate(360)
        self.assertEqual(v1.x, v2.x)
        self.assertEqual(v1.y, v2.y)
        v2 = v1.rotate(0)
        self.assertEqual(v1.x, v2.x)
        self.assertEqual(v1.y, v2.y)

    def test_rotate_ip(self):
        v = Vector2(1, 0)
        self.assertEqual(v.rotate_ip(90), None)
        self.assertEqual(v.x, 0)
        self.assertEqual(v.y, 1)
        v = Vector2(-1, -1)
        v.rotate_ip(-90)
        self.assertEqual(v.x, -1)
        self.assertEqual(v.y, 1)

    def test_normalize(self):
        v1 = Vector2(1.2, 3.4)
        v2 = v1.normalize()
        # length is 1
        self.assertEqual(v2.x * v2.x + v2.y * v2.y, 1.)
        # v1 is unchanged
        self.assertEqual(v1.x, 1.2)
        self.assertEqual(v1.y, 3.4)
        # v2 is paralell to v1
        self.assertEqual(v1.x * v2.y - v1.y * v2.x, 0.)
        self.assertRaises(ZeroDivisionError, lambda : Vector2().normalize())
        
    def test_normalize_ip(self):
        v1 = Vector2(1.2, 3.4)
        v2 = v1
        self.assertEqual(v2.normalize_ip(), None)
        # length is 1
        self.assertEqual(v2.x * v2.x + v2.y * v2.y, 1.)
        # v2 is paralell to v1
        self.assertEqual(v1.x * v2.y - v1.y * v2.x, 0.)
        self.assertRaises(ZeroDivisionError, lambda : Vector2().normalize_ip())

    def test_is_normalized(self):
        v1 = Vector2(1.2, 3.4)
        self.assertEqual(v1.is_normalized(), False)
        v2 = v1.normalize()
        self.assertEqual(v2.is_normalized(), True)
        v3 = Vector2(1, 0)
        self.assertEqual(v3.is_normalized(), True)
        self.assertEqual(Vector2().is_normalized(), False)
        
    def test_cross(self):
        v1 = Vector2(1.2, 3.4)
        v2 = Vector2(5.6, 7.8)
        self.assertEqual(v1.cross(v2), 1.2 * 7.8 - 3.4 * 5.6)
        self.assertEqual(v1.cross(v2), -v2.cross(v1))
        self.assertEqual(v1.cross(v1), 0)

    def test_dot(self):
        v1 = Vector2(1.2, 3.4)
        v2 = Vector2(5.6, 7.8)
        self.assertEqual(v1.dot(v2), 1.2 * 5.6 + 3.4 * 7.8)
        self.assertEqual(v1.dot([5.6, 7.8]), 1.2 * 5.6 + 3.4 * 7.8)
        self.assertEqual(v1.dot(v2), v2.dot(v1))
        self.assertEqual(v1.dot(v2), v1 * v2)

    def test_angle_to(self):
        v1 = Vector2(1.2, 3.4)
        v2 = Vector2(5.6, 7.8)
        self.assertEqual(v1.rotate(v1.angle_to(v2)).normalize(), v2.normalize())
        self.assertEqual(Vector2(1, 1).angle_to((-1, 1)), 90)
        self.assertEqual(Vector2(1, 0).angle_to((0, -1)), -90)
        self.assertEqual(Vector2(1, 0).angle_to((-1, 1)), 135)
        self.assertEqual(abs(Vector2(1, 0).angle_to((-1, 0))), 180)

    def test_scale_to_length(self):
        import math
        v1 = Vector2(1, 1)
        v1.scale_to_length(2.5)
        self.assertEqual(v1, Vector2(2.5, 2.5) / math.sqrt(2))
        self.assertRaises(ZeroDivisionError, lambda : Vector2().scale_to_length(1))
        self.assertEqual(v1.scale_to_length(0), None)
        self.assertEqual(v1, Vector2())

    def test_length(self):
        self.assertEqual(Vector2(3, 4).length(), 5)
        self.assertEqual(Vector2(-3, 4).length(), 5)
        self.assertEqual(Vector2().length(), 0)
        
    def test_length_squared(self):
        self.assertEqual(Vector2(3, 4).length_squared(), 25)
        self.assertEqual(Vector2(-3, 4).length_squared(), 25)
        self.assertEqual(Vector2().length_squared(), 0)

    def test_reflect(self):
        v = Vector2(1, -1)
        n = Vector2(0, 1)
        self.assertEqual(v.reflect(n), Vector2(1, 1))
        self.assertEqual(v.reflect(3*n), v.reflect(n))
        self.assertEqual(v.reflect(-v), -v)
        self.assertRaises(ZeroDivisionError, lambda : v.reflect(Vector2()))
        
    def test_reflect_ip(self):
        v1 = Vector2(1, -1)
        v2 = Vector2(v1)
        n = Vector2(0, 1)
        self.assertEqual(v2.reflect_ip(n), None)
        self.assertEqual(v2, Vector2(1, 1))
        v2 = Vector2(v1)
        v2.reflect_ip(3*n)
        self.assertEqual(v2, v1.reflect(n))
        v2 = Vector2(v1)
        v2.reflect_ip(-v1)
        self.assertEqual(v2, -v1)
        self.assertRaises(ZeroDivisionError, lambda : v2.reflect_ip(Vector2()))

    def test_distance_to(self):
        v1 = Vector2(1.2, 3.4)
        v2 = Vector2(3.4, 4.5)
        diff = v1 - v2
        self.assertEqual(Vector2(1, 0).distance_to(Vector2(0, 1)), math.sqrt(2))
        self.assertEqual(v1.distance_to(v2),
                         math.sqrt(diff.x * diff.x + diff.y * diff.y))
        self.assertEqual(v1.distance_to(v1), 0)
        self.assertEqual(v1.distance_to(v2), v2.distance_to(v1))

    def test_distance_squared_to(self):
        v1 = Vector2(1, 3)
        v2 = Vector2(0, -1)
        diff = v1 - v2
        self.assertEqual(Vector2(1, 0).distance_squared_to(Vector2(0, 1)), 2)
        self.assertEqual(v1.distance_squared_to(v2),
                         diff.x * diff.x + diff.y * diff.y)
        self.assertEqual(v1.distance_squared_to(v1), 0)
        self.assertEqual(v1.distance_squared_to(v2), v2.distance_squared_to(v1))
                         
    def testSwizzle(self):
        v1 = Vector2(1, 2)
        self.assertEquals(hasattr(v1, "enable_swizzle"), True)
        self.assertEquals(hasattr(v1, "disable_swizzle"), True)
        # swizzling disabled by default
        self.assertRaises(AttributeError, lambda : v1.yx)
        v1.enable_swizzle()
        
        self.assertEqual(v1.yx, (v1.y, v1.x))
        self.assertEqual(v1.xxyyxy, (v1.x, v1.x, v1.y, v1.y, v1.x, v1.y))
        v1.xy = (3, -4.5)
        self.assertEqual(v1, Vector2(3, -4.5))
        v1.yx = (3, -4.5)
        self.assertEqual(v1, Vector2(-4.5, 3))
        self.assertEqual(type(v1), Vector2)
        def invalidSwizzleX():
            Vector2().xx = (1, 2)
        def invalidSwizzleY():
            Vector2().yy = (1, 2)
        self.assertRaises(AttributeError, invalidSwizzleX)
        self.assertRaises(AttributeError, invalidSwizzleY)
        def invalidAssignment():
            Vector2().xy = 3
        self.assertRaises(TypeError, invalidAssignment)

        

if __name__ == '__main__':
    unittest.main()
