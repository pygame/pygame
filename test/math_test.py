# -*- coding: utf-8 -*-
import sys
import unittest
import math
from time import clock
import platform

import pygame.math
from pygame.math import Vector2, Vector3

IS_PYPY = 'PyPy' == platform.python_implementation()
PY3 = sys.version_info.major == 3


class Vector2TypeTest(unittest.TestCase):

    def setUp(self):
        pygame.math.enable_swizzling()
        self.zeroVec = Vector2()
        self.e1 = Vector2(1, 0)
        self.e2 = Vector2(0, 1)
        self.t1 = (1.2, 3.4)
        self.l1 = list(self.t1)
        self.v1 = Vector2(self.t1)
        self.t2 = (5.6, 7.8)
        self.l2 = list(self.t2)
        self.v2 = Vector2(self.t2)
        self.s1 = 5.6
        self.s2 = 7.8

    def tearDown(self):
        pygame.math.enable_swizzling()

    def testConstructionDefault(self):
        v = Vector2()
        self.assertEqual(v.x, 0.)
        self.assertEqual(v.y, 0.)

    def testConstructionScalar(self):
        v = Vector2(1)
        self.assertEqual(v.x, 1.)
        self.assertEqual(v.y, 1.)

    def testConstructionScalarKeywords(self):
        v = Vector2(x=1)
        self.assertEqual(v.x, 1.)
        self.assertEqual(v.y, 1.)

    def testConstructionKeywords(self):
        v = Vector2(x=1, y=2)
        self.assertEqual(v.x, 1.)
        self.assertEqual(v.y, 2.)

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

    def testAttributAccess(self):
        tmp = self.v1.x
        self.assertEqual(tmp, self.v1.x)
        self.assertEqual(tmp, self.v1[0])
        tmp = self.v1.y
        self.assertEqual(tmp, self.v1.y)
        self.assertEqual(tmp, self.v1[1])
        self.v1.x = 3.141
        self.assertEqual(self.v1.x, 3.141)
        self.v1.y = 3.141
        self.assertEqual(self.v1.y, 3.141)
        def assign_nonfloat():
            v = Vector2()
            v.x = "spam"
        self.assertRaises(TypeError, assign_nonfloat)

    def testSequence(self):
        v = Vector2(1.2, 3.4)
        Vector2()[:]
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
        def assign_nonfloat():
            v = Vector2()
            v[0] = "spam"
        self.assertRaises(TypeError, assign_nonfloat)

    def testExtendedSlicing(self):
        #  deletion
        def delSlice(vec, start=None, stop=None, step=None):
            if start is not None and stop is not None and step is not None:
                del vec[start:stop:step]
            elif start is not None and stop is None and step is not None:
                del vec[start::step]
            elif start is None and stop is None and step is not None:
                del vec[::step]
        v = Vector2(self.v1)
        self.assertRaises(TypeError, delSlice, v, None, None, 2)
        self.assertRaises(TypeError, delSlice, v, 1, None, 2)
        self.assertRaises(TypeError, delSlice, v, 1, 2, 1)

        #  assignment
        v = Vector2(self.v1)
        v[::2] = [-1]
        self.assertEqual(v, [-1, self.v1.y])
        v = Vector2(self.v1)
        v[::-2] = [10]
        self.assertEqual(v, [self.v1.x, 10])
        v = Vector2(self.v1)
        v[::-1] = v
        self.assertEqual(v, [self.v1.y, self.v1.x])
        a = Vector2(self.v1)
        b = Vector2(self.v1)
        c = Vector2(self.v1)
        a[1:2] = [2.2]
        b[slice(1,2)] = [2.2]
        c[1:2:] = (2.2,)
        self.assertEqual(a, b)
        self.assertEqual(a, c)
        self.assertEqual(type(a), type(self.v1))
        self.assertEqual(type(b), type(self.v1))
        self.assertEqual(type(c), type(self.v1))

    def testAdd(self):
        v3 = self.v1 + self.v2
        self.assertTrue(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.v1.x + self.v2.x)
        self.assertEqual(v3.y, self.v1.y + self.v2.y)
        v3 = self.v1 + self.t2
        self.assertTrue(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.v1.x + self.t2[0])
        self.assertEqual(v3.y, self.v1.y + self.t2[1])
        v3 = self.v1 + self.l2
        self.assertTrue(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.v1.x + self.l2[0])
        self.assertEqual(v3.y, self.v1.y + self.l2[1])
        v3 = self.t1 + self.v2
        self.assertTrue(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.t1[0] + self.v2.x)
        self.assertEqual(v3.y, self.t1[1] + self.v2.y)
        v3 = self.l1 + self.v2
        self.assertTrue(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.l1[0] + self.v2.x)
        self.assertEqual(v3.y, self.l1[1] + self.v2.y)

    def testSub(self):
        v3 = self.v1 - self.v2
        self.assertTrue(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.v1.x - self.v2.x)
        self.assertEqual(v3.y, self.v1.y - self.v2.y)
        v3 = self.v1 - self.t2
        self.assertTrue(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.v1.x - self.t2[0])
        self.assertEqual(v3.y, self.v1.y - self.t2[1])
        v3 = self.v1 - self.l2
        self.assertTrue(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.v1.x - self.l2[0])
        self.assertEqual(v3.y, self.v1.y - self.l2[1])
        v3 = self.t1 - self.v2
        self.assertTrue(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.t1[0] - self.v2.x)
        self.assertEqual(v3.y, self.t1[1] - self.v2.y)
        v3 = self.l1 - self.v2
        self.assertTrue(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.l1[0] - self.v2.x)
        self.assertEqual(v3.y, self.l1[1] - self.v2.y)

    def testScalarMultiplication(self):
        v = self.s1 * self.v1
        self.assertTrue(isinstance(v, type(self.v1)))
        self.assertEqual(v.x, self.s1 * self.v1.x)
        self.assertEqual(v.y, self.s1 * self.v1.y)
        v = self.v1 * self.s2
        self.assertEqual(v.x, self.v1.x * self.s2)
        self.assertEqual(v.y, self.v1.y * self.s2)

    def testScalarDivision(self):
        v = self.v1 / self.s1
        self.assertTrue(isinstance(v, type(self.v1)))
        self.assertAlmostEqual(v.x, self.v1.x / self.s1)
        self.assertAlmostEqual(v.y, self.v1.y / self.s1)
        v = self.v1 // self.s2
        self.assertTrue(isinstance(v, type(self.v1)))
        self.assertEqual(v.x, self.v1.x // self.s2)
        self.assertEqual(v.y, self.v1.y // self.s2)

    def testBool(self):
        self.assertEqual(bool(self.zeroVec), False)
        self.assertEqual(bool(self.v1), True)
        self.assertTrue(not self.zeroVec)
        self.assertTrue(self.v1)

    def testUnary(self):
        v = +self.v1
        self.assertTrue(isinstance(v, type(self.v1)))
        self.assertEqual(v.x, self.v1.x)
        self.assertEqual(v.y, self.v1.y)
        self.assertNotEqual(id(v), id(self.v1))
        v = -self.v1
        self.assertTrue(isinstance(v, type(self.v1)))
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
        if PY3:
            next_ = it.__next__
        else:
            next_ = it.next
        self.assertEqual(next_(), self.v1[0])
        self.assertEqual(next_(), self.v1[1])
        self.assertRaises(StopIteration, lambda : next_())
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
        # issue 214
        self.assertEqual(Vector2(0, 1).rotate(359.99999999), Vector2(0, 1))

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
        self.assertRaises(ValueError, lambda : self.zeroVec.normalize())

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
        self.assertRaises(ValueError, lambda : self.zeroVec.normalize_ip())

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
        self.assertAlmostEqual(self.v1.dot(self.v2),
                            self.v1.x * self.v2.x + self.v1.y * self.v2.y)
        self.assertAlmostEqual(self.v1.dot(self.l2),
                            self.v1.x * self.l2[0] + self.v1.y * self.l2[1])
        self.assertAlmostEqual(self.v1.dot(self.t2),
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
        self.assertRaises(ValueError, lambda : self.zeroVec.scale_to_length(1))
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
        self.assertRaises(ValueError, lambda : v.reflect(self.zeroVec))

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
        self.assertRaises(ValueError, lambda : v2.reflect_ip(Vector2()))

    def test_distance_to(self):
        diff = self.v1 - self.v2
        self.assertEqual(self.e1.distance_to(self.e2), math.sqrt(2))
        self.assertAlmostEqual(self.v1.distance_to(self.v2),
                            math.sqrt(diff.x * diff.x + diff.y * diff.y))
        self.assertEqual(self.v1.distance_to(self.v1), 0)
        self.assertEqual(self.v1.distance_to(self.v2),
                         self.v2.distance_to(self.v1))

    def test_distance_squared_to(self):
        diff = self.v1 - self.v2
        self.assertEqual(self.e1.distance_squared_to(self.e2), 2)
        self.assertAlmostEqual(self.v1.distance_squared_to(self.v2),
                            diff.x * diff.x + diff.y * diff.y)
        self.assertEqual(self.v1.distance_squared_to(self.v1), 0)
        self.assertEqual(self.v1.distance_squared_to(self.v2),
                         self.v2.distance_squared_to(self.v1))

    def test_swizzle(self):
        self.assertTrue(hasattr(pygame.math, "enable_swizzling"))
        self.assertTrue(hasattr(pygame.math, "disable_swizzling"))
        # swizzling not disabled by default
        pygame.math.disable_swizzling()
        self.assertRaises(AttributeError, lambda : self.v1.yx)
        pygame.math.enable_swizzling()

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
        def unicodeAttribute():
            getattr(Vector2(), "ä")
        self.assertRaises(AttributeError, unicodeAttribute)

    def test_swizzle_return_types(self):
        self.assertEqual(type(self.v1.x), float)
        self.assertEqual(type(self.v1.xy), Vector2)
        self.assertEqual(type(self.v1.xyx), Vector3)
        # but we don't have vector4 or above... so tuple.
        self.assertEqual(type(self.v1.xyxy), tuple)
        self.assertEqual(type(self.v1.xyxyx), tuple)

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
        self.assertRaises(ZeroDivisionError, lambda : self.zeroVec.elementwise() ** -1)
        self.assertRaises(ZeroDivisionError, lambda : Vector2(1,1).elementwise() / 0)
        self.assertRaises(ZeroDivisionError, lambda : Vector2(1,1).elementwise() // 0)
        self.assertRaises(ZeroDivisionError, lambda : Vector2(1,1).elementwise() % 0)
        self.assertRaises(ZeroDivisionError, lambda : Vector2(1,1).elementwise() / self.zeroVec)
        self.assertRaises(ZeroDivisionError, lambda : Vector2(1,1).elementwise() // self.zeroVec)
        self.assertRaises(ZeroDivisionError, lambda : Vector2(1,1).elementwise() % self.zeroVec)
        self.assertRaises(ZeroDivisionError, lambda : 2 / self.zeroVec.elementwise())
        self.assertRaises(ZeroDivisionError, lambda : 2 // self.zeroVec.elementwise())
        self.assertRaises(ZeroDivisionError, lambda : 2 % self.zeroVec.elementwise())

    def test_slerp(self):
        self.assertRaises(ValueError, lambda : self.zeroVec.slerp(self.v1, .5))
        self.assertRaises(ValueError, lambda : self.v1.slerp(self.zeroVec, .5))
        self.assertRaises(ValueError,
                          lambda : self.zeroVec.slerp(self.zeroVec, .5))
        v1 = Vector2(1, 0)
        v2 = Vector2(0, 1)
        steps = 10
        angle_step = v1.angle_to(v2) / steps
        for i, u in ((i, v1.slerp(v2, i/float(steps))) for i in range(steps+1)):
            self.assertAlmostEqual(u.length(), 1)
            self.assertAlmostEqual(v1.angle_to(u), i * angle_step)
        self.assertEqual(u, v2)

        v1 = Vector2(100, 0)
        v2 = Vector2(0, 10)
        radial_factor = v2.length() / v1.length()
        for i, u in ((i, v1.slerp(v2, -i/float(steps))) for i in range(steps+1)):
            self.assertAlmostEqual(u.length(), (v2.length() - v1.length()) * (float(i)/steps) + v1.length())
        self.assertEqual(u, v2)
        self.assertEqual(v1.slerp(v1, .5), v1)
        self.assertEqual(v2.slerp(v2, .5), v2)
        self.assertRaises(ValueError, lambda : v1.slerp(-v1, 0.5))

    def test_lerp(self):
        v1 = Vector2(0, 0)
        v2 = Vector2(10, 10)
        self.assertEqual(v1.lerp(v2, 0.5), (5, 5))
        self.assertRaises(ValueError, lambda : v1.lerp(v2, 2.5))

        v1 = Vector2(-10, -5)
        v2 = Vector2(10, 10)
        self.assertEqual(v1.lerp(v2, 0.5), (0, 2.5))

    def test_polar(self):
        v = Vector2()
        v.from_polar(self.v1.as_polar())
        self.assertEqual(self.v1, v)
        self.assertEqual(self.e1.as_polar(), (1, 0))
        self.assertEqual(self.e2.as_polar(), (1, 90))
        self.assertEqual((2 * self.e2).as_polar(), (2, 90))
        self.assertRaises(TypeError, lambda : v.from_polar((None, None)))
        self.assertRaises(TypeError, lambda : v.from_polar("ab"))
        self.assertRaises(TypeError, lambda : v.from_polar((None, 1)))
        self.assertRaises(TypeError, lambda : v.from_polar((1, 2, 3)))
        self.assertRaises(TypeError, lambda : v.from_polar((1,)))
        self.assertRaises(TypeError, lambda : v.from_polar(1, 2))
        v.from_polar((.5, 90))
        self.assertEqual(v, .5 * self.e2)
        v.from_polar((1, 0))
        self.assertEqual(v, self.e1)

    def test_subclass_operation(self):
        class Vector(pygame.math.Vector2):
            pass

        vec = Vector()

        self.assertRaises(TypeError, lambda : vec.__imul__(1.0))



class Vector3TypeTest(unittest.TestCase):

    def setUp(self):
        self.zeroVec = Vector3()
        self.e1 = Vector3(1, 0, 0)
        self.e2 = Vector3(0, 1, 0)
        self.e3 = Vector3(0, 0, 1)
        self.t1 = (1.2, 3.4, 9.6)
        self.l1 = list(self.t1)
        self.v1 = Vector3(self.t1)
        self.t2 = (5.6, 7.8, 2.1)
        self.l2 = list(self.t2)
        self.v2 = Vector3(self.t2)
        self.s1 = 5.6
        self.s2 = 7.8

    def testConstructionDefault(self):
        v = Vector3()
        self.assertEqual(v.x, 0.)
        self.assertEqual(v.y, 0.)
        self.assertEqual(v.z, 0.)

    def testConstructionXYZ(self):
        v = Vector3(1.2, 3.4, 9.6)
        self.assertEqual(v.x, 1.2)
        self.assertEqual(v.y, 3.4)
        self.assertEqual(v.z, 9.6)

    def testConstructionTuple(self):
        v = Vector3((1.2, 3.4, 9.6))
        self.assertEqual(v.x, 1.2)
        self.assertEqual(v.y, 3.4)
        self.assertEqual(v.z, 9.6)

    def testConstructionList(self):
        v = Vector3([1.2, 3.4, -9.6])
        self.assertEqual(v.x, 1.2)
        self.assertEqual(v.y, 3.4)
        self.assertEqual(v.z, -9.6)

    def testConstructionVector3(self):
        v = Vector3(Vector3(1.2, 3.4, -9.6))
        self.assertEqual(v.x, 1.2)
        self.assertEqual(v.y, 3.4)
        self.assertEqual(v.z, -9.6)

    def testConstructionScalar(self):
        v = Vector3(1)
        self.assertEqual(v.x, 1.)
        self.assertEqual(v.y, 1.)
        self.assertEqual(v.z, 1.)

    def testConstructionScalarKeywords(self):
        v = Vector3(x=1)
        self.assertEqual(v.x, 1.)
        self.assertEqual(v.y, 1.)
        self.assertEqual(v.z, 1.)

    def testConstructionKeywords(self):
        v = Vector3(x=1, y=2, z=3)
        self.assertEqual(v.x, 1.)
        self.assertEqual(v.y, 2.)
        self.assertEqual(v.z, 3.)

    def testConstructionMissing(self):
        def assign_missing_value():
            v = Vector3(1, 2)
        self.assertRaises(ValueError, assign_missing_value)

        def assign_missing_value():
            v = Vector3(x=1, y=2)
        self.assertRaises(ValueError, assign_missing_value)

    def testAttributAccess(self):
        tmp = self.v1.x
        self.assertEqual(tmp, self.v1.x)
        self.assertEqual(tmp, self.v1[0])
        tmp = self.v1.y
        self.assertEqual(tmp, self.v1.y)
        self.assertEqual(tmp, self.v1[1])
        tmp = self.v1.z
        self.assertEqual(tmp, self.v1.z)
        self.assertEqual(tmp, self.v1[2])
        self.v1.x = 3.141
        self.assertEqual(self.v1.x, 3.141)
        self.v1.y = 3.141
        self.assertEqual(self.v1.y, 3.141)
        self.v1.z = 3.141
        self.assertEqual(self.v1.z, 3.141)
        def assign_nonfloat():
            v = Vector2()
            v.x = "spam"
        self.assertRaises(TypeError, assign_nonfloat)

    def testSequence(self):
        v = Vector3(1.2, 3.4, -9.6)
        self.assertEqual(len(v), 3)
        self.assertEqual(v[0], 1.2)
        self.assertEqual(v[1], 3.4)
        self.assertEqual(v[2], -9.6)
        self.assertRaises(IndexError, lambda : v[3])
        self.assertEqual(v[-1], -9.6)
        self.assertEqual(v[-2], 3.4)
        self.assertEqual(v[-3], 1.2)
        self.assertRaises(IndexError, lambda : v[-4])
        self.assertEqual(v[:], [1.2, 3.4, -9.6])
        self.assertEqual(v[1:], [3.4, -9.6])
        self.assertEqual(v[:1], [1.2])
        self.assertEqual(v[:-1], [1.2, 3.4])
        self.assertEqual(v[1:2], [3.4])
        self.assertEqual(list(v), [1.2, 3.4, -9.6])
        self.assertEqual(tuple(v), (1.2, 3.4, -9.6))
        v[0] = 5.6
        v[1] = 7.8
        v[2] = -2.1
        self.assertEqual(v.x, 5.6)
        self.assertEqual(v.y, 7.8)
        self.assertEqual(v.z, -2.1)
        v[:] = [9.1, 11.12, -13.41]
        self.assertEqual(v.x, 9.1)
        self.assertEqual(v.y, 11.12)
        self.assertEqual(v.z, -13.41)
        def overpopulate():
            v = Vector3()
            v[:] = [1, 2, 3, 4]
        self.assertRaises(ValueError, overpopulate)
        def underpopulate():
            v = Vector3()
            v[:] = [1]
        self.assertRaises(ValueError, underpopulate)
        def assign_nonfloat():
            v = Vector2()
            v[0] = "spam"
        self.assertRaises(TypeError, assign_nonfloat)

    def testExtendedSlicing(self):
        #  deletion
        def delSlice(vec, start=None, stop=None, step=None):
            if start is not None and stop is not None and step is not None:
                del vec[start:stop:step]
            elif start is not None and stop is None and step is not None:
                del vec[start::step]
            elif start is None and stop is None and step is not None:
                del vec[::step]
        v = Vector3(self.v1)
        self.assertRaises(TypeError, delSlice, v, None, None, 2)
        self.assertRaises(TypeError, delSlice, v, 1, None, 2)
        self.assertRaises(TypeError, delSlice, v, 1, 2, 1)

        #  assignment
        v = Vector3(self.v1)
        v[::2] = [-1.1, -2.2]
        self.assertEqual(v, [-1.1, self.v1.y, -2.2])
        v = Vector3(self.v1)
        v[::-2] = [10, 20]
        self.assertEqual(v, [20, self.v1.y, 10])
        v = Vector3(self.v1)
        v[::-1] = v
        self.assertEqual(v, [self.v1.z, self.v1.y, self.v1.x])
        a = Vector3(self.v1)
        b = Vector3(self.v1)
        c = Vector3(self.v1)
        a[1:2] = [2.2]
        b[slice(1,2)] = [2.2]
        c[1:2:] = (2.2,)
        self.assertEqual(a, b)
        self.assertEqual(a, c)
        self.assertEqual(type(a), type(self.v1))
        self.assertEqual(type(b), type(self.v1))
        self.assertEqual(type(c), type(self.v1))

    def testAdd(self):
        v3 = self.v1 + self.v2
        self.assertTrue(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.v1.x + self.v2.x)
        self.assertEqual(v3.y, self.v1.y + self.v2.y)
        self.assertEqual(v3.z, self.v1.z + self.v2.z)
        v3 = self.v1 + self.t2
        self.assertTrue(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.v1.x + self.t2[0])
        self.assertEqual(v3.y, self.v1.y + self.t2[1])
        self.assertEqual(v3.z, self.v1.z + self.t2[2])
        v3 = self.v1 + self.l2
        self.assertTrue(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.v1.x + self.l2[0])
        self.assertEqual(v3.y, self.v1.y + self.l2[1])
        self.assertEqual(v3.z, self.v1.z + self.l2[2])
        v3 = self.t1 + self.v2
        self.assertTrue(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.t1[0] + self.v2.x)
        self.assertEqual(v3.y, self.t1[1] + self.v2.y)
        self.assertEqual(v3.z, self.t1[2] + self.v2.z)
        v3 = self.l1 + self.v2
        self.assertTrue(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.l1[0] + self.v2.x)
        self.assertEqual(v3.y, self.l1[1] + self.v2.y)
        self.assertEqual(v3.z, self.l1[2] + self.v2.z)

    def testSub(self):
        v3 = self.v1 - self.v2
        self.assertTrue(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.v1.x - self.v2.x)
        self.assertEqual(v3.y, self.v1.y - self.v2.y)
        self.assertEqual(v3.z, self.v1.z - self.v2.z)
        v3 = self.v1 - self.t2
        self.assertTrue(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.v1.x - self.t2[0])
        self.assertEqual(v3.y, self.v1.y - self.t2[1])
        self.assertEqual(v3.z, self.v1.z - self.t2[2])
        v3 = self.v1 - self.l2
        self.assertTrue(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.v1.x - self.l2[0])
        self.assertEqual(v3.y, self.v1.y - self.l2[1])
        self.assertEqual(v3.z, self.v1.z - self.l2[2])
        v3 = self.t1 - self.v2
        self.assertTrue(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.t1[0] - self.v2.x)
        self.assertEqual(v3.y, self.t1[1] - self.v2.y)
        self.assertEqual(v3.z, self.t1[2] - self.v2.z)
        v3 = self.l1 - self.v2
        self.assertTrue(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.l1[0] - self.v2.x)
        self.assertEqual(v3.y, self.l1[1] - self.v2.y)
        self.assertEqual(v3.z, self.l1[2] - self.v2.z)

    def testScalarMultiplication(self):
        v = self.s1 * self.v1
        self.assertTrue(isinstance(v, type(self.v1)))
        self.assertEqual(v.x, self.s1 * self.v1.x)
        self.assertEqual(v.y, self.s1 * self.v1.y)
        self.assertEqual(v.z, self.s1 * self.v1.z)
        v = self.v1 * self.s2
        self.assertEqual(v.x, self.v1.x * self.s2)
        self.assertEqual(v.y, self.v1.y * self.s2)
        self.assertEqual(v.z, self.v1.z * self.s2)

    def testScalarDivision(self):
        v = self.v1 / self.s1
        self.assertTrue(isinstance(v, type(self.v1)))
        self.assertAlmostEqual(v.x, self.v1.x / self.s1)
        self.assertAlmostEqual(v.y, self.v1.y / self.s1)
        self.assertAlmostEqual(v.z, self.v1.z / self.s1)
        v = self.v1 // self.s2
        self.assertTrue(isinstance(v, type(self.v1)))
        self.assertEqual(v.x, self.v1.x // self.s2)
        self.assertEqual(v.y, self.v1.y // self.s2)
        self.assertEqual(v.z, self.v1.z // self.s2)

    def testBool(self):
        self.assertEqual(bool(self.zeroVec), False)
        self.assertEqual(bool(self.v1), True)
        self.assertTrue(not self.zeroVec)
        self.assertTrue(self.v1)

    def testUnary(self):
        v = +self.v1
        self.assertTrue(isinstance(v, type(self.v1)))
        self.assertEqual(v.x, self.v1.x)
        self.assertEqual(v.y, self.v1.y)
        self.assertEqual(v.z, self.v1.z)
        self.assertNotEqual(id(v), id(self.v1))
        v = -self.v1
        self.assertTrue(isinstance(v, type(self.v1)))
        self.assertEqual(v.x, -self.v1.x)
        self.assertEqual(v.y, -self.v1.y)
        self.assertEqual(v.z, -self.v1.z)
        self.assertNotEqual(id(v), id(self.v1))

    def testCompare(self):
        int_vec = Vector3(3, -2, 13)
        flt_vec = Vector3(3.0, -2.0, 13.)
        zero_vec = Vector3(0, 0, 0)
        self.assertEqual(int_vec == flt_vec, True)
        self.assertEqual(int_vec != flt_vec, False)
        self.assertEqual(int_vec != zero_vec, True)
        self.assertEqual(flt_vec == zero_vec, False)
        self.assertEqual(int_vec == (3, -2, 13), True)
        self.assertEqual(int_vec != (3, -2, 13), False)
        self.assertEqual(int_vec != [0, 0], True)
        self.assertEqual(int_vec == [0, 0], False)
        self.assertEqual(int_vec != 5, True)
        self.assertEqual(int_vec == 5, False)
        self.assertEqual(int_vec != [3, -2, 0, 1], True)
        self.assertEqual(int_vec == [3, -2, 0, 1], False)

    def testStr(self):
        v = Vector3(1.2, 3.4, 5.6)
        self.assertEqual(str(v), "[1.2, 3.4, 5.6]")

    def testRepr(self):
        v = Vector3(1.2, 3.4, -9.6)
        self.assertEqual(v.__repr__(), "<Vector3(1.2, 3.4, -9.6)>")
        self.assertEqual(v, Vector3(v.__repr__()))

    def testIter(self):
        it = self.v1.__iter__()
        if PY3:
            next_ = it.__next__
        else:
            next_ = it.next
        self.assertEqual(next_(), self.v1[0])
        self.assertEqual(next_(), self.v1[1])
        self.assertEqual(next_(), self.v1[2])
        self.assertRaises(StopIteration, lambda : next_())
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
        v1 = Vector3(1, 0, 0)
        axis = Vector3(0, 1, 0)
        v2 = v1.rotate(90, axis)
        v3 = v1.rotate(90 + 360, axis)
        self.assertEqual(v1.x, 1)
        self.assertEqual(v1.y, 0)
        self.assertEqual(v1.z, 0)
        self.assertEqual(v2.x, 0)
        self.assertEqual(v2.y, 0)
        self.assertEqual(v2.z, -1)
        self.assertEqual(v3.x, v2.x)
        self.assertEqual(v3.y, v2.y)
        self.assertEqual(v3.z, v2.z)
        v1 = Vector3(-1, -1, -1)
        v2 = v1.rotate(-90, axis)
        self.assertEqual(v2.x, 1)
        self.assertEqual(v2.y, -1)
        self.assertEqual(v2.z, -1)
        v2 = v1.rotate(360, axis)
        self.assertEqual(v1.x, v2.x)
        self.assertEqual(v1.y, v2.y)
        self.assertEqual(v1.z, v2.z)
        v2 = v1.rotate(0, axis)
        self.assertEqual(v1.x, v2.x)
        self.assertEqual(v1.y, v2.y)
        self.assertEqual(v1.z, v2.z)
        # issue 214
        self.assertEqual(Vector3(0, 1, 0).rotate(359.9999999, Vector3(0, 0, 1)),
                         Vector3(0, 1, 0))

    def test_rotate_ip(self):
        v = Vector3(1, 0, 0)
        axis = Vector3(0, 1, 0)
        self.assertEqual(v.rotate_ip(90, axis), None)
        self.assertEqual(v.x, 0)
        self.assertEqual(v.y, 0)
        self.assertEqual(v.z, -1)
        v = Vector3(-1, -1, 1)
        v.rotate_ip(-90, axis)
        self.assertEqual(v.x, -1)
        self.assertEqual(v.y, -1)
        self.assertEqual(v.z, -1)

    def test_rotate_x(self):
        v1 = Vector3(1, 0, 0)
        v2 = v1.rotate_x(90)
        v3 = v1.rotate_x(90 + 360)
        self.assertEqual(v1.x, 1)
        self.assertEqual(v1.y, 0)
        self.assertEqual(v1.z, 0)
        self.assertEqual(v2.x, 1)
        self.assertEqual(v2.y, 0)
        self.assertEqual(v2.z, 0)
        self.assertEqual(v3.x, v2.x)
        self.assertEqual(v3.y, v2.y)
        self.assertEqual(v3.z, v2.z)
        v1 = Vector3(-1, -1, -1)
        v2 = v1.rotate_x(-90)
        self.assertEqual(v2.x, -1)
        self.assertAlmostEqual(v2.y, -1)
        self.assertAlmostEqual(v2.z, 1)
        v2 = v1.rotate_x(360)
        self.assertAlmostEqual(v1.x, v2.x)
        self.assertAlmostEqual(v1.y, v2.y)
        self.assertAlmostEqual(v1.z, v2.z)
        v2 = v1.rotate_x(0)
        self.assertEqual(v1.x, v2.x)
        self.assertAlmostEqual(v1.y, v2.y)
        self.assertAlmostEqual(v1.z, v2.z)

    def test_rotate_x_ip(self):
        v = Vector3(1, 0, 0)
        self.assertEqual(v.rotate_x_ip(90), None)
        self.assertEqual(v.x, 1)
        self.assertEqual(v.y, 0)
        self.assertEqual(v.z, 0)
        v = Vector3(-1, -1, 1)
        v.rotate_x_ip(-90)
        self.assertEqual(v.x, -1)
        self.assertAlmostEqual(v.y, 1)
        self.assertAlmostEqual(v.z, 1)

    def test_rotate_y(self):
        v1 = Vector3(1, 0, 0)
        v2 = v1.rotate_y(90)
        v3 = v1.rotate_y(90 + 360)
        self.assertEqual(v1.x, 1)
        self.assertEqual(v1.y, 0)
        self.assertEqual(v1.z, 0)
        self.assertAlmostEqual(v2.x, 0)
        self.assertEqual(v2.y, 0)
        self.assertAlmostEqual(v2.z, -1)
        self.assertAlmostEqual(v3.x, v2.x)
        self.assertEqual(v3.y, v2.y)
        self.assertAlmostEqual(v3.z, v2.z)
        v1 = Vector3(-1, -1, -1)
        v2 = v1.rotate_y(-90)
        self.assertAlmostEqual(v2.x, 1)
        self.assertEqual(v2.y, -1)
        self.assertAlmostEqual(v2.z, -1)
        v2 = v1.rotate_y(360)
        self.assertAlmostEqual(v1.x, v2.x)
        self.assertEqual(v1.y, v2.y)
        self.assertAlmostEqual(v1.z, v2.z)
        v2 = v1.rotate_y(0)
        self.assertEqual(v1.x, v2.x)
        self.assertEqual(v1.y, v2.y)
        self.assertEqual(v1.z, v2.z)

    def test_rotate_y_ip(self):
        v = Vector3(1, 0, 0)
        self.assertEqual(v.rotate_y_ip(90), None)
        self.assertAlmostEqual(v.x, 0)
        self.assertEqual(v.y, 0)
        self.assertAlmostEqual(v.z, -1)
        v = Vector3(-1, -1, 1)
        v.rotate_y_ip(-90)
        self.assertAlmostEqual(v.x, -1)
        self.assertEqual(v.y, -1)
        self.assertAlmostEqual(v.z, -1)

    def test_rotate_z(self):
        v1 = Vector3(1, 0, 0)
        v2 = v1.rotate_z(90)
        v3 = v1.rotate_z(90 + 360)
        self.assertEqual(v1.x, 1)
        self.assertEqual(v1.y, 0)
        self.assertEqual(v1.z, 0)
        self.assertAlmostEqual(v2.x, 0)
        self.assertAlmostEqual(v2.y, 1)
        self.assertEqual(v2.z, 0)
        self.assertAlmostEqual(v3.x, v2.x)
        self.assertAlmostEqual(v3.y, v2.y)
        self.assertEqual(v3.z, v2.z)
        v1 = Vector3(-1, -1, -1)
        v2 = v1.rotate_z(-90)
        self.assertAlmostEqual(v2.x, -1)
        self.assertAlmostEqual(v2.y, 1)
        self.assertEqual(v2.z, -1)
        v2 = v1.rotate_z(360)
        self.assertAlmostEqual(v1.x, v2.x)
        self.assertAlmostEqual(v1.y, v2.y)
        self.assertEqual(v1.z, v2.z)
        v2 = v1.rotate_z(0)
        self.assertAlmostEqual(v1.x, v2.x)
        self.assertAlmostEqual(v1.y, v2.y)
        self.assertEqual(v1.z, v2.z)

    def test_rotate_z_ip(self):
        v = Vector3(1, 0, 0)
        self.assertEqual(v.rotate_z_ip(90), None)
        self.assertAlmostEqual(v.x, 0)
        self.assertAlmostEqual(v.y, 1)
        self.assertEqual(v.z, 0)
        v = Vector3(-1, -1, 1)
        v.rotate_z_ip(-90)
        self.assertAlmostEqual(v.x, -1)
        self.assertAlmostEqual(v.y, 1)
        self.assertEqual(v.z, 1)

    def test_normalize(self):
        v = self.v1.normalize()
        # length is 1
        self.assertAlmostEqual(v.x * v.x + v.y * v.y + v.z * v.z, 1.)
        # v1 is unchanged
        self.assertEqual(self.v1.x, self.l1[0])
        self.assertEqual(self.v1.y, self.l1[1])
        self.assertEqual(self.v1.z, self.l1[2])
        # v2 is paralell to v1 (tested via cross product)
        cross = ((self.v1.y * v.z - self.v1.z * v.y) ** 2 +
                 (self.v1.z * v.x - self.v1.x * v.z) ** 2 +
                 (self.v1.x * v.y - self.v1.y * v.x) ** 2)
        self.assertAlmostEqual(cross, 0.)
        self.assertRaises(ValueError, lambda : self.zeroVec.normalize())

    def test_normalize_ip(self):
        v = +self.v1
        # v has length != 1 before normalizing
        self.assertNotEqual(v.x * v.x + v.y * v.y + v.z * v.z, 1.)
        # inplace operations should return None
        self.assertEqual(v.normalize_ip(), None)
        # length is 1
        self.assertAlmostEqual(v.x * v.x + v.y * v.y + v.z * v.z, 1.)
        # v2 is paralell to v1 (tested via cross product)
        cross = ((self.v1.y * v.z - self.v1.z * v.y) ** 2 +
                 (self.v1.z * v.x - self.v1.x * v.z) ** 2 +
                 (self.v1.x * v.y - self.v1.y * v.x) ** 2)
        self.assertAlmostEqual(cross, 0.)
        self.assertRaises(ValueError, lambda : self.zeroVec.normalize_ip())

    def test_is_normalized(self):
        self.assertEqual(self.v1.is_normalized(), False)
        v = self.v1.normalize()
        self.assertEqual(v.is_normalized(), True)
        self.assertEqual(self.e2.is_normalized(), True)
        self.assertEqual(self.zeroVec.is_normalized(), False)

    def test_cross(self):
        def cross(a, b):
            return Vector3(a[1] * b[2] - a[2] * b[1],
                           a[2] * b[0] - a[0] * b[2],
                           a[0] * b[1] - a[1] * b[0])
        self.assertEqual(self.v1.cross(self.v2), cross(self.v1, self.v2))
        self.assertEqual(self.v1.cross(self.l2), cross(self.v1, self.l2))
        self.assertEqual(self.v1.cross(self.t2), cross(self.v1, self.t2))
        self.assertEqual(self.v1.cross(self.v2), -self.v2.cross(self.v1))
        self.assertEqual(self.v1.cross(self.v1), self.zeroVec)

    def test_dot(self):
        self.assertAlmostEqual(self.v1.dot(self.v2),
                               self.v1.x * self.v2.x + self.v1.y * self.v2.y + self.v1.z * self.v2.z)
        self.assertAlmostEqual(self.v1.dot(self.l2),
                               self.v1.x * self.l2[0] + self.v1.y * self.l2[1] + self.v1.z * self.l2[2])
        self.assertAlmostEqual(self.v1.dot(self.t2),
                         self.v1.x * self.t2[0] + self.v1.y * self.t2[1] + self.v1.z * self.t2[2])
        self.assertAlmostEqual(self.v1.dot(self.v2), self.v2.dot(self.v1))
        self.assertAlmostEqual(self.v1.dot(self.v2), self.v1 * self.v2)

    def test_angle_to(self):
        self.assertEqual(Vector3(1, 1, 0).angle_to((-1, 1, 0)), 90)
        self.assertEqual(Vector3(1, 0, 0).angle_to((0, 0, -1)), 90)
        self.assertEqual(Vector3(1, 0, 0).angle_to((-1, 0, 1)), 135)
        self.assertEqual(abs(Vector3(1, 0, 1).angle_to((-1, 0, -1))), 180)
        # if we rotate v1 by the angle_to v2 around their cross product
        # we should look in the same direction
        self.assertEqual(self.v1.rotate(self.v1.angle_to(self.v2), self.v1.cross(self.v2)).normalize(),
                         self.v2.normalize())

    def test_scale_to_length(self):
        v = Vector3(1, 1, 1)
        v.scale_to_length(2.5)
        self.assertEqual(v, Vector3(2.5, 2.5, 2.5) / math.sqrt(3))
        self.assertRaises(ValueError, lambda : self.zeroVec.scale_to_length(1))
        self.assertEqual(v.scale_to_length(0), None)
        self.assertEqual(v, self.zeroVec)

    def test_length(self):
        self.assertEqual(Vector3(3, 4, 5).length(), math.sqrt(3 * 3 + 4 * 4 + 5 * 5))
        self.assertEqual(Vector3(-3, 4, 5).length(), math.sqrt(-3 * -3 + 4 * 4 + 5 * 5))
        self.assertEqual(self.zeroVec.length(), 0)

    def test_length_squared(self):
        self.assertEqual(Vector3(3, 4, 5).length_squared(), 3 * 3 + 4 * 4 + 5 * 5)
        self.assertEqual(Vector3(-3, 4, 5).length_squared(), -3 * -3 + 4 * 4 + 5 * 5)
        self.assertEqual(self.zeroVec.length_squared(), 0)

    def test_reflect(self):
        v = Vector3(1, -1, 1)
        n = Vector3(0, 1, 0)
        self.assertEqual(v.reflect(n), Vector3(1, 1, 1))
        self.assertEqual(v.reflect(3*n), v.reflect(n))
        self.assertEqual(v.reflect(-v), -v)
        self.assertRaises(ValueError, lambda : v.reflect(self.zeroVec))

    def test_reflect_ip(self):
        v1 = Vector3(1, -1, 1)
        v2 = Vector3(v1)
        n = Vector3(0, 1, 0)
        self.assertEqual(v2.reflect_ip(n), None)
        self.assertEqual(v2, Vector3(1, 1, 1))
        v2 = Vector3(v1)
        v2.reflect_ip(3*n)
        self.assertEqual(v2, v1.reflect(n))
        v2 = Vector3(v1)
        v2.reflect_ip(-v1)
        self.assertEqual(v2, -v1)
        self.assertRaises(ValueError, lambda : v2.reflect_ip(self.zeroVec))

    def test_distance_to(self):
        diff = self.v1 - self.v2
        self.assertEqual(self.e1.distance_to(self.e2), math.sqrt(2))
        self.assertEqual(self.v1.distance_to(self.v2),
                         math.sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z))
        self.assertEqual(self.v1.distance_to(self.v1), 0)
        self.assertEqual(self.v1.distance_to(self.v2),
                         self.v2.distance_to(self.v1))

    def test_distance_squared_to(self):
        diff = self.v1 - self.v2
        self.assertEqual(self.e1.distance_squared_to(self.e2), 2)
        self.assertAlmostEqual(self.v1.distance_squared_to(self.v2),
                            diff.x * diff.x + diff.y * diff.y + diff.z * diff.z)
        self.assertEqual(self.v1.distance_squared_to(self.v1), 0)
        self.assertEqual(self.v1.distance_squared_to(self.v2),
                         self.v2.distance_squared_to(self.v1))

    def test_swizzle(self):
        self.assertTrue(hasattr(pygame.math, "enable_swizzling"))
        self.assertTrue(hasattr(pygame.math, "disable_swizzling"))
        # swizzling enabled by default
        pygame.math.disable_swizzling()
        self.assertRaises(AttributeError, lambda : self.v1.yx)
        pygame.math.enable_swizzling()

        self.assertEqual(self.v1.yxz, (self.v1.y, self.v1.x, self.v1.z))
        self.assertEqual(self.v1.xxyyzzxyz, (self.v1.x, self.v1.x, self.v1.y,
                                             self.v1.y, self.v1.z, self.v1.z,
                                             self.v1.x, self.v1.y, self.v1.z))
        self.v1.xyz = self.t2
        self.assertEqual(self.v1, self.t2)
        self.v1.zxy = self.t2
        self.assertEqual(self.v1, (self.t2[1], self.t2[2], self.t2[0]))
        self.v1.yz = self.t2[:2]
        self.assertEqual(self.v1, (self.t2[1], self.t2[0], self.t2[1]))
        self.assertEqual(type(self.v1), Vector3)

    @unittest.skipIf(IS_PYPY, "known pypy failure")
    def test_invalid_swizzle(self):
        def invalidSwizzleX():
            Vector3().xx = (1, 2)
        def invalidSwizzleY():
            Vector3().yy = (1, 2)
        def invalidSwizzleZ():
            Vector3().zz = (1, 2)
        def invalidSwizzleW():
            Vector3().ww = (1, 2)
        self.assertRaises(AttributeError, invalidSwizzleX)
        self.assertRaises(AttributeError, invalidSwizzleY)
        self.assertRaises(AttributeError, invalidSwizzleZ)
        self.assertRaises(AttributeError, invalidSwizzleW)
        def invalidAssignment():
            Vector3().xy = 3
        self.assertRaises(TypeError, invalidAssignment)

    def test_swizzle_return_types(self):
        self.assertEqual(type(self.v1.x), float)
        self.assertEqual(type(self.v1.xy), Vector2)
        self.assertEqual(type(self.v1.xyz), Vector3)
        # but we don't have vector4 or above... so tuple.
        self.assertEqual(type(self.v1.xyxy), tuple)
        self.assertEqual(type(self.v1.xyxyx), tuple)

    def test_dir_works(self):
        # not every single one of the attributes...
        attributes = set(['lerp', 'normalize', 'normalize_ip', 'reflect', 'slerp', 'x', 'y'])
        # check if this selection of attributes are all there.
        self.assertTrue(attributes.issubset(set(dir(self.v1))))

    def test_elementwise(self):
        # behaviour for "elementwise op scalar"
        self.assertEqual(self.v1.elementwise() + self.s1,
                         (self.v1.x + self.s1, self.v1.y + self.s1, self.v1.z + self.s1))
        self.assertEqual(self.v1.elementwise() - self.s1,
                         (self.v1.x - self.s1, self.v1.y - self.s1, self.v1.z - self.s1))
        self.assertEqual(self.v1.elementwise() * self.s2,
                         (self.v1.x * self.s2, self.v1.y * self.s2, self.v1.z * self.s2))
        self.assertEqual(self.v1.elementwise() / self.s2,
                         (self.v1.x / self.s2, self.v1.y / self.s2, self.v1.z / self.s2))
        self.assertEqual(self.v1.elementwise() // self.s1,
                         (self.v1.x // self.s1, self.v1.y // self.s1, self.v1.z // self.s1))
        self.assertEqual(self.v1.elementwise() ** self.s1,
                         (self.v1.x ** self.s1, self.v1.y ** self.s1, self.v1.z ** self.s1))
        self.assertEqual(self.v1.elementwise() % self.s1,
                         (self.v1.x % self.s1, self.v1.y % self.s1, self.v1.z % self.s1))
        self.assertEqual(self.v1.elementwise() > self.s1,
                         self.v1.x > self.s1 and self.v1.y > self.s1 and self.v1.z > self.s1)
        self.assertEqual(self.v1.elementwise() < self.s1,
                         self.v1.x < self.s1 and self.v1.y < self.s1 and self.v1.z < self.s1)
        self.assertEqual(self.v1.elementwise() == self.s1,
                         self.v1.x == self.s1 and self.v1.y == self.s1 and self.v1.z == self.s1)
        self.assertEqual(self.v1.elementwise() != self.s1,
                         self.v1.x != self.s1 and self.v1.y != self.s1 and self.v1.z != self.s1)
        self.assertEqual(self.v1.elementwise() >= self.s1,
                         self.v1.x >= self.s1 and self.v1.y >= self.s1 and self.v1.z >= self.s1)
        self.assertEqual(self.v1.elementwise() <= self.s1,
                         self.v1.x <= self.s1 and self.v1.y <= self.s1 and self.v1.z <= self.s1)
        # behaviour for "scalar op elementwise"
        self.assertEqual(5 + self.v1.elementwise(), Vector3(5, 5, 5) + self.v1)
        self.assertEqual(3.5 - self.v1.elementwise(), Vector3(3.5, 3.5, 3.5) - self.v1)
        self.assertEqual(7.5 * self.v1.elementwise() , 7.5 * self.v1)
        self.assertEqual(-3.5 / self.v1.elementwise(), (-3.5 / self.v1.x, -3.5 / self.v1.y, -3.5 / self.v1.z))
        self.assertEqual(-3.5 // self.v1.elementwise(), (-3.5 // self.v1.x, -3.5 // self.v1.y, -3.5 // self.v1.z))
        self.assertEqual(-3.5 ** self.v1.elementwise(), (-3.5 ** self.v1.x, -3.5 ** self.v1.y, -3.5 ** self.v1.z))
        self.assertEqual(3 % self.v1.elementwise(), (3 % self.v1.x, 3 % self.v1.y, 3 % self.v1.z))
        self.assertEqual(2 < self.v1.elementwise(), 2 < self.v1.x and 2 < self.v1.y and 2 < self.v1.z)
        self.assertEqual(2 > self.v1.elementwise(), 2 > self.v1.x and 2 > self.v1.y and 2 > self.v1.z)
        self.assertEqual(1 == self.v1.elementwise(), 1 == self.v1.x and 1 == self.v1.y and 1 == self.v1.z)
        self.assertEqual(1 != self.v1.elementwise(), 1 != self.v1.x and 1 != self.v1.y and 1 != self.v1.z)
        self.assertEqual(2 <= self.v1.elementwise(), 2 <= self.v1.x and 2 <= self.v1.y and 2 <= self.v1.z)
        self.assertEqual(-7 >= self.v1.elementwise(), -7 >= self.v1.x and -7 >= self.v1.y and -7 >= self.v1.z)
        self.assertEqual(-7 != self.v1.elementwise(), -7 != self.v1.x and -7 != self.v1.y and -7 != self.v1.z)

        # behaviour for "elementwise op vector"
        self.assertEqual(type(self.v1.elementwise() * self.v2), type(self.v1))
        self.assertEqual(self.v1.elementwise() + self.v2, self.v1 + self.v2)
        self.assertEqual(self.v1.elementwise() + self.v2, self.v1 + self.v2)
        self.assertEqual(self.v1.elementwise() - self.v2, self.v1 - self.v2)
        self.assertEqual(self.v1.elementwise() * self.v2, (self.v1.x * self.v2.x, self.v1.y * self.v2.y, self.v1.z * self.v2.z))
        self.assertEqual(self.v1.elementwise() / self.v2, (self.v1.x / self.v2.x, self.v1.y / self.v2.y, self.v1.z / self.v2.z))
        self.assertEqual(self.v1.elementwise() // self.v2, (self.v1.x // self.v2.x, self.v1.y // self.v2.y, self.v1.z // self.v2.z))
        self.assertEqual(self.v1.elementwise() ** self.v2, (self.v1.x ** self.v2.x, self.v1.y ** self.v2.y, self.v1.z ** self.v2.z))
        self.assertEqual(self.v1.elementwise() % self.v2, (self.v1.x % self.v2.x, self.v1.y % self.v2.y, self.v1.z % self.v2.z))
        self.assertEqual(self.v1.elementwise() > self.v2, self.v1.x > self.v2.x and self.v1.y > self.v2.y and self.v1.z > self.v2.z)
        self.assertEqual(self.v1.elementwise() < self.v2, self.v1.x < self.v2.x and self.v1.y < self.v2.y and self.v1.z < self.v2.z)
        self.assertEqual(self.v1.elementwise() >= self.v2, self.v1.x >= self.v2.x and self.v1.y >= self.v2.y and self.v1.z >= self.v2.z)
        self.assertEqual(self.v1.elementwise() <= self.v2, self.v1.x <= self.v2.x and self.v1.y <= self.v2.y and self.v1.z <= self.v2.z)
        self.assertEqual(self.v1.elementwise() == self.v2, self.v1.x == self.v2.x and self.v1.y == self.v2.y and self.v1.z == self.v2.z)
        self.assertEqual(self.v1.elementwise() != self.v2, self.v1.x != self.v2.x and self.v1.y != self.v2.y and self.v1.z != self.v2.z)
        # behaviour for "vector op elementwise"
        self.assertEqual(self.v2 + self.v1.elementwise(), self.v2 + self.v1)
        self.assertEqual(self.v2 - self.v1.elementwise(), self.v2 - self.v1)
        self.assertEqual(self.v2 * self.v1.elementwise(), (self.v2.x * self.v1.x, self.v2.y * self.v1.y, self.v2.z * self.v1.z))
        self.assertEqual(self.v2 / self.v1.elementwise(), (self.v2.x / self.v1.x, self.v2.y / self.v1.y, self.v2.z / self.v1.z))
        self.assertEqual(self.v2 // self.v1.elementwise(), (self.v2.x // self.v1.x, self.v2.y // self.v1.y, self.v2.z // self.v1.z))
        self.assertEqual(self.v2 ** self.v1.elementwise(), (self.v2.x ** self.v1.x, self.v2.y ** self.v1.y, self.v2.z ** self.v1.z))
        self.assertEqual(self.v2 % self.v1.elementwise(), (self.v2.x % self.v1.x, self.v2.y % self.v1.y, self.v2.z % self.v1.z))
        self.assertEqual(self.v2 < self.v1.elementwise(), self.v2.x < self.v1.x and self.v2.y < self.v1.y and self.v2.z < self.v1.z)
        self.assertEqual(self.v2 > self.v1.elementwise(), self.v2.x > self.v1.x and self.v2.y > self.v1.y and self.v2.z > self.v1.z)
        self.assertEqual(self.v2 <= self.v1.elementwise(), self.v2.x <= self.v1.x and self.v2.y <= self.v1.y and self.v2.z <= self.v1.z)
        self.assertEqual(self.v2 >= self.v1.elementwise(), self.v2.x >= self.v1.x and self.v2.y >= self.v1.y and self.v2.z >= self.v1.z)
        self.assertEqual(self.v2 == self.v1.elementwise(), self.v2.x == self.v1.x and self.v2.y == self.v1.y and self.v2.z == self.v1.z)
        self.assertEqual(self.v2 != self.v1.elementwise(), self.v2.x != self.v1.x and self.v2.y != self.v1.y and self.v2.z != self.v1.z)

        # behaviour for "elementwise op elementwise"
        self.assertEqual(self.v2.elementwise() + self.v1.elementwise(), self.v2 + self.v1)
        self.assertEqual(self.v2.elementwise() - self.v1.elementwise(), self.v2 - self.v1)
        self.assertEqual(self.v2.elementwise() * self.v1.elementwise(),
                         (self.v2.x * self.v1.x, self.v2.y * self.v1.y, self.v2.z * self.v1.z))
        self.assertEqual(self.v2.elementwise() / self.v1.elementwise(),
                         (self.v2.x / self.v1.x, self.v2.y / self.v1.y, self.v2.z / self.v1.z))
        self.assertEqual(self.v2.elementwise() // self.v1.elementwise(),
                         (self.v2.x // self.v1.x, self.v2.y // self.v1.y, self.v2.z // self.v1.z))
        self.assertEqual(self.v2.elementwise() ** self.v1.elementwise(),
                         (self.v2.x ** self.v1.x, self.v2.y ** self.v1.y, self.v2.z ** self.v1.z))
        self.assertEqual(self.v2.elementwise() % self.v1.elementwise(),
                         (self.v2.x % self.v1.x, self.v2.y % self.v1.y, self.v2.z % self.v1.z))
        self.assertEqual(self.v2.elementwise() < self.v1.elementwise(),
                         self.v2.x < self.v1.x and self.v2.y < self.v1.y and self.v2.z < self.v1.z)
        self.assertEqual(self.v2.elementwise() > self.v1.elementwise(),
                         self.v2.x > self.v1.x and self.v2.y > self.v1.y and self.v2.z > self.v1.z)
        self.assertEqual(self.v2.elementwise() <= self.v1.elementwise(),
                         self.v2.x <= self.v1.x and self.v2.y <= self.v1.y and self.v2.z <= self.v1.z)
        self.assertEqual(self.v2.elementwise() >= self.v1.elementwise(),
                         self.v2.x >= self.v1.x and self.v2.y >= self.v1.y and self.v2.z >= self.v1.z)
        self.assertEqual(self.v2.elementwise() == self.v1.elementwise(),
                         self.v2.x == self.v1.x and self.v2.y == self.v1.y and self.v2.z == self.v1.z)
        self.assertEqual(self.v2.elementwise() != self.v1.elementwise(),
                         self.v2.x != self.v1.x and self.v2.y != self.v1.y and self.v2.z != self.v1.z)

        # other behaviour
        self.assertEqual(abs(self.v1.elementwise()), (abs(self.v1.x), abs(self.v1.y), abs(self.v1.z)))
        self.assertEqual(-self.v1.elementwise(), -self.v1)
        self.assertEqual(+self.v1.elementwise(), +self.v1)
        self.assertEqual(bool(self.v1.elementwise()), bool(self.v1))
        self.assertEqual(bool(Vector3().elementwise()), bool(Vector3()))
        self.assertEqual(self.zeroVec.elementwise() ** 0, (1, 1, 1))
        self.assertRaises(ValueError, lambda : pow(Vector3(-1, 0, 0).elementwise(), 1.2))
        self.assertRaises(ZeroDivisionError, lambda : self.zeroVec.elementwise() ** -1)
        self.assertRaises(ZeroDivisionError, lambda : Vector3(1,1,1).elementwise() / 0)
        self.assertRaises(ZeroDivisionError, lambda : Vector3(1,1,1).elementwise() // 0)
        self.assertRaises(ZeroDivisionError, lambda : Vector3(1,1,1).elementwise() % 0)
        self.assertRaises(ZeroDivisionError, lambda : Vector3(1,1,1).elementwise() / self.zeroVec)
        self.assertRaises(ZeroDivisionError, lambda : Vector3(1,1,1).elementwise() // self.zeroVec)
        self.assertRaises(ZeroDivisionError, lambda : Vector3(1,1,1).elementwise() % self.zeroVec)
        self.assertRaises(ZeroDivisionError, lambda : 2 / self.zeroVec.elementwise())
        self.assertRaises(ZeroDivisionError, lambda : 2 // self.zeroVec.elementwise())
        self.assertRaises(ZeroDivisionError, lambda : 2 % self.zeroVec.elementwise())


    def test_slerp(self):
        self.assertRaises(ValueError, lambda : self.zeroVec.slerp(self.v1, .5))
        self.assertRaises(ValueError, lambda : self.v1.slerp(self.zeroVec, .5))
        self.assertRaises(ValueError,
                          lambda : self.zeroVec.slerp(self.zeroVec, .5))
        steps = 10
        angle_step = self.e1.angle_to(self.e2) / steps
        for i, u in ((i, self.e1.slerp(self.e2, i/float(steps))) for i in range(steps+1)):
            self.assertAlmostEqual(u.length(), 1)
            self.assertAlmostEqual(self.e1.angle_to(u), i * angle_step)
        self.assertEqual(u, self.e2)

        v1 = Vector3(100, 0, 0)
        v2 = Vector3(0, 10, 7)
        radial_factor = v2.length() / v1.length()
        for i, u in ((i, v1.slerp(v2, -i/float(steps))) for i in range(steps+1)):
            self.assertAlmostEqual(u.length(), (v2.length() - v1.length()) * (float(i)/steps) + v1.length())
        self.assertEqual(u, v2)
        self.assertEqual(v1.slerp(v1, .5), v1)
        self.assertEqual(v2.slerp(v2, .5), v2)
        self.assertRaises(ValueError, lambda : v1.slerp(-v1, 0.5))

    def test_lerp(self):
        v1 = Vector3(0, 0, 0)
        v2 = Vector3(10, 10, 10)
        self.assertEqual(v1.lerp(v2, 0.5), (5, 5, 5))
        self.assertRaises(ValueError, lambda : v1.lerp(v2, 2.5))

        v1 = Vector3(-10, -5, -20)
        v2 = Vector3(10, 10, -20)
        self.assertEqual(v1.lerp(v2, 0.5), (0, 2.5, -20))

    def test_spherical(self):
        v = Vector3()
        v.from_spherical(self.v1.as_spherical())
        self.assertEqual(self.v1, v)
        self.assertEqual(self.e1.as_spherical(), (1, 90, 0))
        self.assertEqual(self.e2.as_spherical(), (1, 90, 90))
        self.assertEqual(self.e3.as_spherical(), (1, 0, 0))
        self.assertEqual((2 * self.e2).as_spherical(), (2, 90, 90))
        self.assertRaises(TypeError, lambda : v.from_spherical((None, None, None)))
        self.assertRaises(TypeError, lambda : v.from_spherical("abc"))
        self.assertRaises(TypeError, lambda : v.from_spherical((None, 1, 2)))
        self.assertRaises(TypeError, lambda : v.from_spherical((1, 2, 3, 4)))
        self.assertRaises(TypeError, lambda : v.from_spherical((1, 2)))
        self.assertRaises(TypeError, lambda : v.from_spherical(1, 2, 3))
        v.from_spherical((.5, 90, 90))
        self.assertEqual(v, .5 * self.e2)

    def test_inplace_operators(self):

        v = Vector3(1,1,1)
        v *= 2
        self.assertEqual(v, (2.0,2.0,2.0))

        v = Vector3(4,4,4)
        v /= 2
        self.assertEqual(v, (2.0,2.0,2.0))


        v = Vector3(3.0,3.0,3.0)
        v -= (1,1,1)
        self.assertEqual(v, (2.0,2.0,2.0))

        v = Vector3(3.0,3.0,3.0)
        v += (1,1,1)
        self.assertEqual(v, (4.0,4.0,4.0))

    def test_pickle(self):
        import pickle
        v2 = Vector2(1, 2)
        v3 = Vector3(1, 2, 3)
        self.assertEqual(pickle.loads(pickle.dumps(v2)), v2)
        self.assertEqual(pickle.loads(pickle.dumps(v3)), v3)


if __name__ == '__main__':
    unittest.main()
