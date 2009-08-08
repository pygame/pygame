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
from random import random

class Vector2TypeTest(unittest.TestCase):
    def setUp(self):
        self.zeroVec = Vector2()
        self.e1 = Vector2(1, 0)
        self.e2 = Vector2(0, 1)
#        self.t1 = (random(), random())
        self.t1 = (1.2, 3.4)
        self.l1 = list(self.t1)
        self.v1 = Vector2(self.t1)
#        self.t2 = (random(), random())
        self.t2 = (5.6, 7.8)
        self.l2 = list(self.t2)
        self.v2 = Vector2(self.t2)
#        self.s1 = random()
#        self.s2 = random()
        self.s1 = 5.6
        self.s2 = 7.8
        
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
        v3 = self.v1 + self.v2
        self.assert_(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.v1.x + self.v2.x)
        self.assertEqual(v3.y, self.v1.y + self.v2.y)
        v3 = self.v1 + self.t2
        self.assert_(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.v1.x + self.t2[0])
        self.assertEqual(v3.y, self.v1.y + self.t2[1])
        v3 = self.v1 + self.l2
        self.assert_(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.v1.x + self.l2[0])
        self.assertEqual(v3.y, self.v1.y + self.l2[1])
        v3 = self.t1 + self.v2
        self.assert_(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.t1[0] + self.v2.x)
        self.assertEqual(v3.y, self.t1[1] + self.v2.y)
        v3 = self.l1 + self.v2
        self.assert_(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.l1[0] + self.v2.x)
        self.assertEqual(v3.y, self.l1[1] + self.v2.y)

    def testSub(self):
        v3 = self.v1 - self.v2
        self.assert_(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.v1.x - self.v2.x)
        self.assertEqual(v3.y, self.v1.y - self.v2.y)
        v3 = self.v1 - self.t2
        self.assert_(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.v1.x - self.t2[0])
        self.assertEqual(v3.y, self.v1.y - self.t2[1])
        v3 = self.v1 - self.l2
        self.assert_(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.v1.x - self.l2[0])
        self.assertEqual(v3.y, self.v1.y - self.l2[1])
        v3 = self.t1 - self.v2
        self.assert_(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.t1[0] - self.v2.x)
        self.assertEqual(v3.y, self.t1[1] - self.v2.y)
        v3 = self.l1 - self.v2
        self.assert_(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.l1[0] - self.v2.x)
        self.assertEqual(v3.y, self.l1[1] - self.v2.y)

    def testScalarMultiplication(self):
        v = self.s1 * self.v1
        self.assert_(isinstance(v, type(self.v1)))
        self.assertEqual(v.x, self.s1 * self.v1.x)
        self.assertEqual(v.y, self.s1 * self.v1.y)
        v = self.v1 * self.s2
        self.assertEqual(v.x, self.v1.x * self.s2)
        self.assertEqual(v.y, self.v1.y * self.s2)

    def testScalarDivision(self):
        v = self.v1 / self.s1
        self.assert_(isinstance(v, type(self.v1)))
        self.assertAlmostEqual(v.x, self.v1.x / self.s1)
        self.assertAlmostEqual(v.y, self.v1.y / self.s1)
        v = self.v1 // self.s2
        self.assert_(isinstance(v, type(self.v1)))
        self.assertEqual(v.x, self.v1.x // self.s2)
        self.assertEqual(v.y, self.v1.y // self.s2)

    def testBool(self):
        self.assertEqual(bool(self.zeroVec), False)
        self.assertEqual(bool(self.v1), True)
        self.assert_(not self.zeroVec)
        self.assert_(self.v1)

    def testUnary(self):
        v = +self.v1
        self.assert_(isinstance(v, type(self.v1)))
        self.assertEqual(v.x, self.v1.x)
        self.assertEqual(v.y, self.v1.y)
        self.assertNotEqual(id(v), id(self.v1))
        v = -self.v1
        self.assert_(isinstance(v, type(self.v1)))
        self.assertEqual(v.x, -self.v1.x)
        self.assertEqual(v.y, -self.v1.y)
        self.assertNotEqual(id(v), id(self.v1))
        
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
        it = self.v1.__iter__()
        self.assertEqual(it.next(), self.v1[0])
        self.assertEqual(it.next(), self.v1[1])
        self.assertRaises(StopIteration, lambda : it.next())
        it1 = self.v1.__iter__()
        it2 = self.v1.__iter__()
        self.assertNotEqual(id(it1), id(it2))
        self.assertEqual(id(it1), id(it1.__iter__()))
        self.assertEqual(list(it1), list(it2));
        self.assertEqual(list(self.v1.__iter__()), self.l1)
        idx = 0
        for val in self.v1:
            self.assertEqual(val, self.v1[idx])
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
        v = self.v1.normalize()
        # length is 1
        self.assertAlmostEqual(v.x * v.x + v.y * v.y, 1.)
        # v1 is unchanged
        self.assertEqual(self.v1.x, self.l1[0])
        self.assertEqual(self.v1.y, self.l1[1])
        # v2 is paralell to v1
        self.assertAlmostEqual(self.v1.x * v.y - self.v1.y * v.x, 0.)
        self.assertRaises(ZeroDivisionError, lambda : self.zeroVec.normalize())
        
    def test_normalize_ip(self):
        v = +self.v1
        # v has length != 1 before normalizing
        self.assertNotEqual(v.x * v.x + v.y * v.y, 1.)
        # inplace operations should return None
        self.assertEqual(v.normalize_ip(), None)
        # length is 1
        self.assertAlmostEqual(v.x * v.x + v.y * v.y, 1.)
        # v2 is paralell to v1
        self.assertAlmostEqual(self.v1.x * v.y - self.v1.y * v.x, 0.)
        self.assertRaises(ZeroDivisionError,
                          lambda : self.zeroVec.normalize_ip())

    def test_is_normalized(self):
        self.assertEqual(self.v1.is_normalized(), False)
        v = self.v1.normalize()
        self.assertEqual(v.is_normalized(), True)
        self.assertEqual(self.e2.is_normalized(), True)
        self.assertEqual(self.zeroVec.is_normalized(), False)
        
    def test_cross(self):
        self.assertEqual(self.v1.cross(self.v2),
                         self.v1.x * self.v2.y - self.v1.y * self.v2.x)
        self.assertEqual(self.v1.cross(self.l2),
                         self.v1.x * self.l2[1] - self.v1.y * self.l2[0])
        self.assertEqual(self.v1.cross(self.t2),
                         self.v1.x * self.t2[1] - self.v1.y * self.t2[0])
        self.assertEqual(self.v1.cross(self.v2), -self.v2.cross(self.v1))
        self.assertEqual(self.v1.cross(self.v1), 0)

    def test_dot(self):
        self.assertEqual(self.v1.dot(self.v2),
                         self.v1.x * self.v2.x + self.v1.y * self.v2.y)
        self.assertEqual(self.v1.dot(self.l2),
                         self.v1.x * self.l2[0] + self.v1.y * self.l2[1])
        self.assertEqual(self.v1.dot(self.t2),
                         self.v1.x * self.t2[0] + self.v1.y * self.t2[1])
        self.assertEqual(self.v1.dot(self.v2), self.v2.dot(self.v1))
        self.assertEqual(self.v1.dot(self.v2), self.v1 * self.v2)

    def test_angle_to(self):
        self.assertEqual(self.v1.rotate(self.v1.angle_to(self.v2)).normalize(),
                         self.v2.normalize())
        self.assertEqual(Vector2(1, 1).angle_to((-1, 1)), 90)
        self.assertEqual(Vector2(1, 0).angle_to((0, -1)), -90)
        self.assertEqual(Vector2(1, 0).angle_to((-1, 1)), 135)
        self.assertEqual(abs(Vector2(1, 0).angle_to((-1, 0))), 180)

    def test_scale_to_length(self):
        v = Vector2(1, 1)
        v.scale_to_length(2.5)
        self.assertEqual(v, Vector2(2.5, 2.5) / math.sqrt(2))
        self.assertRaises(ZeroDivisionError,
                          lambda : self.zeroVec.scale_to_length(1))
        self.assertEqual(v.scale_to_length(0), None)
        self.assertEqual(v, self.zeroVec)

    def test_length(self):
        self.assertEqual(Vector2(3, 4).length(), 5)
        self.assertEqual(Vector2(-3, 4).length(), 5)
        self.assertEqual(self.zeroVec.length(), 0)
        
    def test_length_squared(self):
        self.assertEqual(Vector2(3, 4).length_squared(), 25)
        self.assertEqual(Vector2(-3, 4).length_squared(), 25)
        self.assertEqual(self.zeroVec.length_squared(), 0)

    def test_reflect(self):
        v = Vector2(1, -1)
        n = Vector2(0, 1)
        self.assertEqual(v.reflect(n), Vector2(1, 1))
        self.assertEqual(v.reflect(3*n), v.reflect(n))
        self.assertEqual(v.reflect(-v), -v)
        self.assertRaises(ZeroDivisionError, lambda : v.reflect(self.zeroVec))
        
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
        diff = self.v1 - self.v2
        self.assertEqual(self.e1.distance_to(self.e2), math.sqrt(2))
        self.assertEqual(self.v1.distance_to(self.v2),
                         math.sqrt(diff.x * diff.x + diff.y * diff.y))
        self.assertEqual(self.v1.distance_to(self.v1), 0)
        self.assertEqual(self.v1.distance_to(self.v2),
                         self.v2.distance_to(self.v1))

    def test_distance_squared_to(self):
        diff = self.v1 - self.v2
        self.assertEqual(self.e1.distance_squared_to(self.e2), 2)
        self.assertEqual(self.v1.distance_squared_to(self.v2),
                         diff.x * diff.x + diff.y * diff.y)
        self.assertEqual(self.v1.distance_squared_to(self.v1), 0)
        self.assertEqual(self.v1.distance_squared_to(self.v2),
                         self.v2.distance_squared_to(self.v1))
        
    def testSwizzle(self):
        self.assertEquals(hasattr(self.v1, "enable_swizzle"), True)
        self.assertEquals(hasattr(self.v1, "disable_swizzle"), True)
        # swizzling disabled by default
        self.assertRaises(AttributeError, lambda : self.v1.yx)
        self.v1.enable_swizzle()
        
        self.assertEqual(self.v1.yx, (self.v1.y, self.v1.x))
        self.assertEqual(self.v1.xxyyxy, (self.v1.x, self.v1.x, self.v1.y,
                                          self.v1.y, self.v1.x, self.v1.y))
        self.v1.xy = self.t2
        self.assertEqual(self.v1, self.t2)
        self.v1.yx = self.t2
        self.assertEqual(self.v1, (self.t2[1], self.t2[0]))
        self.assertEqual(type(self.v1), Vector2)
        def invalidSwizzleX():
            Vector2().xx = (1, 2)
        def invalidSwizzleY():
            Vector2().yy = (1, 2)
        self.assertRaises(AttributeError, invalidSwizzleX)
        self.assertRaises(AttributeError, invalidSwizzleY)
        def invalidAssignment():
            Vector2().xy = 3
        self.assertRaises(TypeError, invalidAssignment)

    def test_elementwise(self):
        # behaviour for "elementwise op scalar"
        self.assertEqual(self.v1.elementwise() + self.s1,
                         (self.v1.x + self.s1, self.v1.y + self.s1))
        self.assertEqual(self.v1.elementwise() - self.s1,
                         (self.v1.x - self.s1, self.v1.y - self.s1))
        self.assertEqual(self.v1.elementwise() * self.s2,
                         (self.v1.x * self.s2, self.v1.y * self.s2))
        self.assertEqual(self.v1.elementwise() / self.s2,
                         (self.v1.x / self.s2, self.v1.y / self.s2))
        self.assertEqual(self.v1.elementwise() // self.s1,
                         (self.v1.x // self.s1, self.v1.y // self.s1))
        self.assertEqual(self.v1.elementwise() ** self.s1,
                         (self.v1.x ** self.s1, self.v1.y ** self.s1))
        self.assertEqual(self.v1.elementwise() % self.s1,
                         (self.v1.x % self.s1, self.v1.y % self.s1))
        self.assertEqual(self.v1.elementwise() > self.s1,
                         self.v1.x > self.s1 and self.v1.y > self.s1)
        self.assertEqual(self.v1.elementwise() < self.s1,
                         self.v1.x < self.s1 and self.v1.y < self.s1)
        self.assertEqual(self.v1.elementwise() == self.s1,
                         self.v1.x == self.s1 and self.v1.y == self.s1)
        self.assertEqual(self.v1.elementwise() != self.s1,
                         self.v1.x != self.s1 and self.v1.y != self.s1)
        self.assertEqual(self.v1.elementwise() >= self.s1,
                         self.v1.x >= self.s1 and self.v1.y >= self.s1)
        self.assertEqual(self.v1.elementwise() <= self.s1,
                         self.v1.x <= self.s1 and self.v1.y <= self.s1)
        self.assertEqual(self.v1.elementwise() != self.s1,
                         self.v1.x != self.s1 and self.v1.y != self.s1)
        # behaviour for "scalar op elementwise"
        self.assertEqual(5 + self.v1.elementwise(), Vector2(5, 5) + self.v1)
        self.assertEqual(3.5 - self.v1.elementwise(), Vector2(3.5, 3.5) - self.v1)
        self.assertEqual(7.5 * self.v1.elementwise() , 7.5 * self.v1)
        self.assertEqual(-3.5 / self.v1.elementwise(), (-3.5 / self.v1.x, -3.5 / self.v1.y))
        self.assertEqual(-3.5 // self.v1.elementwise(), (-3.5 // self.v1.x, -3.5 // self.v1.y))
        self.assertEqual(-3.5 ** self.v1.elementwise(), (-3.5 ** self.v1.x, -3.5 ** self.v1.y))
        self.assertEqual(3 % self.v1.elementwise(), (3 % self.v1.x, 3 % self.v1.y))
        self.assertEqual(2 < self.v1.elementwise(), 2 < self.v1.x and 2 < self.v1.y)
        self.assertEqual(2 > self.v1.elementwise(), 2 > self.v1.x and 2 > self.v1.y)
        self.assertEqual(1 == self.v1.elementwise(), 1 == self.v1.x and 1 == self.v1.y)
        self.assertEqual(1 != self.v1.elementwise(), 1 != self.v1.x and 1 != self.v1.y)
        self.assertEqual(2 <= self.v1.elementwise(), 2 <= self.v1.x and 2 <= self.v1.y)
        self.assertEqual(-7 >= self.v1.elementwise(), -7 >= self.v1.x and -7 >= self.v1.y)
        self.assertEqual(-7 != self.v1.elementwise(), -7 != self.v1.x and -7 != self.v1.y)

        # behaviour for "elementwise op vector"
        self.assertEqual(type(self.v1.elementwise() * self.v2), type(self.v1))
        self.assertEqual(self.v1.elementwise() + self.v2, self.v1 + self.v2)
        self.assertEqual(self.v1.elementwise() + self.v2, self.v1 + self.v2)
        self.assertEqual(self.v1.elementwise() - self.v2, self.v1 - self.v2)
        self.assertEqual(self.v1.elementwise() * self.v2, (self.v1.x * self.v2.x, self.v1.y * self.v2.y))
        self.assertEqual(self.v1.elementwise() / self.v2, (self.v1.x / self.v2.x, self.v1.y / self.v2.y))
        self.assertEqual(self.v1.elementwise() // self.v2, (self.v1.x // self.v2.x, self.v1.y // self.v2.y))
        self.assertEqual(self.v1.elementwise() ** self.v2, (self.v1.x ** self.v2.x, self.v1.y ** self.v2.y))
        self.assertEqual(self.v1.elementwise() % self.v2, (self.v1.x % self.v2.x, self.v1.y % self.v2.y))
        self.assertEqual(self.v1.elementwise() > self.v2, self.v1.x > self.v2.x and self.v1.y > self.v2.y)
        self.assertEqual(self.v1.elementwise() < self.v2, self.v1.x < self.v2.x and self.v1.y < self.v2.y)
        self.assertEqual(self.v1.elementwise() >= self.v2, self.v1.x >= self.v2.x and self.v1.y >= self.v2.y)
        self.assertEqual(self.v1.elementwise() <= self.v2, self.v1.x <= self.v2.x and self.v1.y <= self.v2.y)
        self.assertEqual(self.v1.elementwise() == self.v2, self.v1.x == self.v2.x and self.v1.y == self.v2.y)
        self.assertEqual(self.v1.elementwise() != self.v2, self.v1.x != self.v2.x and self.v1.y != self.v2.y)
        # behaviour for "vector op elementwise"
        self.assertEqual(self.v2 + self.v1.elementwise(), self.v2 + self.v1)
        self.assertEqual(self.v2 - self.v1.elementwise(), self.v2 - self.v1)
        self.assertEqual(self.v2 * self.v1.elementwise(), (self.v2.x * self.v1.x, self.v2.y * self.v1.y))
        self.assertEqual(self.v2 / self.v1.elementwise(), (self.v2.x / self.v1.x, self.v2.y / self.v1.y))
        self.assertEqual(self.v2 // self.v1.elementwise(), (self.v2.x // self.v1.x, self.v2.y // self.v1.y))
        self.assertEqual(self.v2 ** self.v1.elementwise(), (self.v2.x ** self.v1.x, self.v2.y ** self.v1.y))
        self.assertEqual(self.v2 % self.v1.elementwise(), (self.v2.x % self.v1.x, self.v2.y % self.v1.y))
        self.assertEqual(self.v2 < self.v1.elementwise(), self.v2.x < self.v1.x and self.v2.y < self.v1.y)
        self.assertEqual(self.v2 > self.v1.elementwise(), self.v2.x > self.v1.x and self.v2.y > self.v1.y)
        self.assertEqual(self.v2 <= self.v1.elementwise(), self.v2.x <= self.v1.x and self.v2.y <= self.v1.y)
        self.assertEqual(self.v2 >= self.v1.elementwise(), self.v2.x >= self.v1.x and self.v2.y >= self.v1.y)
        self.assertEqual(self.v2 == self.v1.elementwise(), self.v2.x == self.v1.x and self.v2.y == self.v1.y)
        self.assertEqual(self.v2 != self.v1.elementwise(), self.v2.x != self.v1.x and self.v2.y != self.v1.y)

        # behaviour for "elementwise op elementwise"
        self.assertEqual(self.v2.elementwise() + self.v1.elementwise(), self.v2 + self.v1)
        self.assertEqual(self.v2.elementwise() - self.v1.elementwise(), self.v2 - self.v1)
        self.assertEqual(self.v2.elementwise() * self.v1.elementwise(), (self.v2.x * self.v1.x, self.v2.y * self.v1.y))
        self.assertEqual(self.v2.elementwise() / self.v1.elementwise(), (self.v2.x / self.v1.x, self.v2.y / self.v1.y))
        self.assertEqual(self.v2.elementwise() // self.v1.elementwise(), (self.v2.x // self.v1.x, self.v2.y // self.v1.y))
        self.assertEqual(self.v2.elementwise() ** self.v1.elementwise(), (self.v2.x ** self.v1.x, self.v2.y ** self.v1.y))
        self.assertEqual(self.v2.elementwise() % self.v1.elementwise(), (self.v2.x % self.v1.x, self.v2.y % self.v1.y))
        self.assertEqual(self.v2.elementwise() < self.v1.elementwise(), self.v2.x < self.v1.x and self.v2.y < self.v1.y)
        self.assertEqual(self.v2.elementwise() > self.v1.elementwise(), self.v2.x > self.v1.x and self.v2.y > self.v1.y)
        self.assertEqual(self.v2.elementwise() <= self.v1.elementwise(), self.v2.x <= self.v1.x and self.v2.y <= self.v1.y)
        self.assertEqual(self.v2.elementwise() >= self.v1.elementwise(), self.v2.x >= self.v1.x and self.v2.y >= self.v1.y)
        self.assertEqual(self.v2.elementwise() == self.v1.elementwise(), self.v2.x == self.v1.x and self.v2.y == self.v1.y)
        self.assertEqual(self.v2.elementwise() != self.v1.elementwise(), self.v2.x != self.v1.x and self.v2.y != self.v1.y)

        # other behaviour
        self.assertEqual(abs(self.v1.elementwise()), (abs(self.v1.x), abs(self.v1.y)))
        self.assertEqual(-self.v1.elementwise(), -self.v1)
        self.assertEqual(+self.v1.elementwise(), +self.v1)
        self.assertEqual(bool(self.v1.elementwise()), bool(self.v1))
        self.assertEqual(bool(Vector2().elementwise()), bool(Vector2()))
        self.assertEqual(self.zeroVec.elementwise() ** 0, (1, 1))
        self.assertRaises(ValueError, lambda : pow(Vector2(-1, 0).elementwise(), 1.2))
        self.assertRaises(ZeroDivisionError, lambda : self.zeroVec.elementwise() ** -1)

    def test_elementwise(self):
        v1 = self.v1
        v2 = self.v2
        s1 = self.s1
        s2 = self.s2
        # behaviour for "elementwise op scalar"
        self.assertEqual(v1.elementwise() + s1, (v1.x + s1, v1.y + s1))
        self.assertEqual(v1.elementwise() - s1, (v1.x - s1, v1.y - s1))
        self.assertEqual(v1.elementwise() * s2, (v1.x * s2, v1.y * s2))
        self.assertEqual(v1.elementwise() / s2, (v1.x / s2, v1.y / s2))
        self.assertEqual(v1.elementwise() // s1, (v1.x // s1, v1.y // s1))
        self.assertEqual(v1.elementwise() ** s1, (v1.x ** s1, v1.y ** s1))
        self.assertEqual(v1.elementwise() % s1, (v1.x % s1, v1.y % s1))
        self.assertEqual(v1.elementwise() > s1, v1.x > s1 and v1.y > s1)
        self.assertEqual(v1.elementwise() < s1, v1.x < s1 and v1.y < s1)
        self.assertEqual(v1.elementwise() == s1, v1.x == s1 and v1.y == s1)
        self.assertEqual(v1.elementwise() != s1, v1.x != s1 and v1.y != s1)
        self.assertEqual(v1.elementwise() >= s1, v1.x >= s1 and v1.y >= s1)
        self.assertEqual(v1.elementwise() <= s1, v1.x <= s1 and v1.y <= s1)
        self.assertEqual(v1.elementwise() != s1, v1.x != s1 and v1.y != s1)
        # behaviour for "scalar op elementwise"
        self.assertEqual(s1 + v1.elementwise(), (s1 + v1.x, s1 + v1.y))
        self.assertEqual(s1 - v1.elementwise(), (s1 - v1.x, s1 - v1.y))
        self.assertEqual(s1 * v1.elementwise(), (s1 * v1.x, s1 * v1.y))
        self.assertEqual(s1 / v1.elementwise(), (s1 / v1.x, s1 / v1.y))
        self.assertEqual(s1 // v1.elementwise(), (s1 // v1.x, s1 // v1.y))
        self.assertEqual(s1 ** v1.elementwise(), (s1 ** v1.x, s1 ** v1.y))
        self.assertEqual(s1 % v1.elementwise(), (s1 % v1.x, s1 % v1.y))
        self.assertEqual(s1 < v1.elementwise(), s1 < v1.x and s1 < v1.y)
        self.assertEqual(s1 > v1.elementwise(), s1 > v1.x and s1 > v1.y)
        self.assertEqual(s1 == v1.elementwise(), s1 == v1.x and s1 == v1.y)
        self.assertEqual(s1 != v1.elementwise(), s1 != v1.x and s1 != v1.y)
        self.assertEqual(s1 <= v1.elementwise(), s1 <= v1.x and s1 <= v1.y)
        self.assertEqual(s1 >= v1.elementwise(), s1 >= v1.x and s1 >= v1.y)
        self.assertEqual(s1 != v1.elementwise(), s1 != v1.x and s1 != v1.y)

        # behaviour for "elementwise op vector"
        self.assertEqual(type(v1.elementwise() * v2), type(v1))
        self.assertEqual(v1.elementwise() + v2, v1 + v2)
        self.assertEqual(v1.elementwise() - v2, v1 - v2)
        self.assertEqual(v1.elementwise() * v2, (v1.x * v2.x, v1.y * v2.y))
        self.assertEqual(v1.elementwise() / v2, (v1.x / v2.x, v1.y / v2.y))
        self.assertEqual(v1.elementwise() // v2, (v1.x // v2.x, v1.y // v2.y))
        self.assertEqual(v1.elementwise() ** v2, (v1.x ** v2.x, v1.y ** v2.y))
        self.assertEqual(v1.elementwise() % v2, (v1.x % v2.x, v1.y % v2.y))
        self.assertEqual(v1.elementwise() > v2, v1.x > v2.x and v1.y > v2.y)
        self.assertEqual(v1.elementwise() < v2, v1.x < v2.x and v1.y < v2.y)
        self.assertEqual(v1.elementwise() >= v2, v1.x >= v2.x and v1.y >= v2.y)
        self.assertEqual(v1.elementwise() <= v2, v1.x <= v2.x and v1.y <= v2.y)
        self.assertEqual(v1.elementwise() == v2, v1.x == v2.x and v1.y == v2.y)
        self.assertEqual(v1.elementwise() != v2, v1.x != v2.x and v1.y != v2.y)
        # behaviour for "vector op elementwise"
        self.assertEqual(v2 + v1.elementwise(), v2 + v1)
        self.assertEqual(v2 - v1.elementwise(), v2 - v1)
        self.assertEqual(v2 * v1.elementwise(), (v2.x * v1.x, v2.y * v1.y))
        self.assertEqual(v2 / v1.elementwise(), (v2.x / v1.x, v2.y / v1.y))
        self.assertEqual(v2 // v1.elementwise(), (v2.x // v1.x, v2.y // v1.y))
        self.assertEqual(v2 ** v1.elementwise(), (v2.x ** v1.x, v2.y ** v1.y))
        self.assertEqual(v2 % v1.elementwise(), (v2.x % v1.x, v2.y % v1.y))
        self.assertEqual(v2 < v1.elementwise(), v2.x < v1.x and v2.y < v1.y)
        self.assertEqual(v2 > v1.elementwise(), v2.x > v1.x and v2.y > v1.y)
        self.assertEqual(v2 <= v1.elementwise(), v2.x <= v1.x and v2.y <= v1.y)
        self.assertEqual(v2 >= v1.elementwise(), v2.x >= v1.x and v2.y >= v1.y)
        self.assertEqual(v2 == v1.elementwise(), v2.x == v1.x and v2.y == v1.y)
        self.assertEqual(v2 != v1.elementwise(), v2.x != v1.x and v2.y != v1.y)

        # behaviour for "elementwise op elementwise"
        self.assertEqual(v2.elementwise() + v1.elementwise(), v2 + v1)
        self.assertEqual(v2.elementwise() - v1.elementwise(), v2 - v1)
        self.assertEqual(v2.elementwise() * v1.elementwise(), (v2.x * v1.x, v2.y * v1.y))
        self.assertEqual(v2.elementwise() / v1.elementwise(), (v2.x / v1.x, v2.y / v1.y))
        self.assertEqual(v2.elementwise() // v1.elementwise(), (v2.x // v1.x, v2.y // v1.y))
        self.assertEqual(v2.elementwise() ** v1.elementwise(), (v2.x ** v1.x, v2.y ** v1.y))
        self.assertEqual(v2.elementwise() % v1.elementwise(), (v2.x % v1.x, v2.y % v1.y))
        self.assertEqual(v2.elementwise() < v1.elementwise(), v2.x < v1.x and v2.y < v1.y)
        self.assertEqual(v2.elementwise() > v1.elementwise(), v2.x > v1.x and v2.y > v1.y)
        self.assertEqual(v2.elementwise() <= v1.elementwise(), v2.x <= v1.x and v2.y <= v1.y)
        self.assertEqual(v2.elementwise() >= v1.elementwise(), v2.x >= v1.x and v2.y >= v1.y)
        self.assertEqual(v2.elementwise() == v1.elementwise(), v2.x == v1.x and v2.y == v1.y)
        self.assertEqual(v2.elementwise() != v1.elementwise(), v2.x != v1.x and v2.y != v1.y)

        # other behaviour
        self.assertEqual(abs(v1.elementwise()), (abs(v1.x), abs(v1.y)))
        self.assertEqual(-v1.elementwise(), -v1)
        self.assertEqual(+v1.elementwise(), +v1)
        self.assertEqual(bool(v1.elementwise()), bool(v1))
        self.assertEqual(bool(Vector2().elementwise()), bool(Vector2()))
        self.assertEqual(self.zeroVec.elementwise() ** 0, (1, 1))
        self.assertRaises(ValueError, lambda : pow(Vector2(-1, 0).elementwise(), 1.2))
        self.assertRaises(ZeroDivisionError, lambda : self.zeroVec.elementwise() ** -1)

if __name__ == '__main__':
    unittest.main()
