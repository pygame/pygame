try:
    import pygame2.test.pgunittest as unittest
    import pygame2.test.pgunittest as unittest
except:
    import pgunittest as unittest

import sys, math
import pygame2
import pygame2.math as pmath
from pygame2.math import Vector, Vector3

class MathVector3Test (unittest.TestCase):

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
        
    def tearDown (self):
        pass
    
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
        self.assertEqual(v[:1], 1.2)
        self.assertEqual(v[:-1], [1.2, 3.4])
        self.assertEqual(v[1:2], 3.4)
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
        self.assertRaises(ValueError, delSlice, v, None, None, 2)
        self.assertRaises(ValueError, delSlice, v, 1, None, 2)
        self.assertRaises(ValueError, delSlice, v, 1, 2, 1)

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
        self.assert_(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.v1.x + self.v2.x)
        self.assertEqual(v3.y, self.v1.y + self.v2.y)
        self.assertEqual(v3.z, self.v1.z + self.v2.z)
        v3 = self.v1 + self.t2
        self.assert_(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.v1.x + self.t2[0])
        self.assertEqual(v3.y, self.v1.y + self.t2[1])
        self.assertEqual(v3.z, self.v1.z + self.t2[2])
        v3 = self.v1 + self.l2
        self.assert_(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.v1.x + self.l2[0])
        self.assertEqual(v3.y, self.v1.y + self.l2[1])
        self.assertEqual(v3.z, self.v1.z + self.l2[2])
        v3 = self.t1 + self.v2
        self.assert_(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.t1[0] + self.v2.x)
        self.assertEqual(v3.y, self.t1[1] + self.v2.y)
        self.assertEqual(v3.z, self.t1[2] + self.v2.z)
        v3 = self.l1 + self.v2
        self.assert_(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.l1[0] + self.v2.x)
        self.assertEqual(v3.y, self.l1[1] + self.v2.y)
        self.assertEqual(v3.z, self.l1[2] + self.v2.z)

    def testSub(self):
        v3 = self.v1 - self.v2
        self.assert_(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.v1.x - self.v2.x)
        self.assertEqual(v3.y, self.v1.y - self.v2.y)
        self.assertEqual(v3.z, self.v1.z - self.v2.z)
        v3 = self.v1 - self.t2
        self.assert_(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.v1.x - self.t2[0])
        self.assertEqual(v3.y, self.v1.y - self.t2[1])
        self.assertEqual(v3.z, self.v1.z - self.t2[2])
        v3 = self.v1 - self.l2
        self.assert_(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.v1.x - self.l2[0])
        self.assertEqual(v3.y, self.v1.y - self.l2[1])
        self.assertEqual(v3.z, self.v1.z - self.l2[2])
        v3 = self.t1 - self.v2
        self.assert_(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.t1[0] - self.v2.x)
        self.assertEqual(v3.y, self.t1[1] - self.v2.y)
        self.assertEqual(v3.z, self.t1[2] - self.v2.z)
        v3 = self.l1 - self.v2
        self.assert_(isinstance(v3, type(self.v1)))
        self.assertEqual(v3.x, self.l1[0] - self.v2.x)
        self.assertEqual(v3.y, self.l1[1] - self.v2.y)
        self.assertEqual(v3.z, self.l1[2] - self.v2.z)

    def testScalarMultiplication(self):
        v = self.s1 * self.v1
        self.assert_(isinstance(v, type(self.v1)))
        self.assertEqual(v.x, self.s1 * self.v1.x)
        self.assertEqual(v.y, self.s1 * self.v1.y)
        self.assertEqual(v.z, self.s1 * self.v1.z)
        v = self.v1 * self.s2
        self.assertEqual(v.x, self.v1.x * self.s2)
        self.assertEqual(v.y, self.v1.y * self.s2)
        self.assertEqual(v.z, self.v1.z * self.s2)

    def testScalarDivision(self):
        v = self.v1 / self.s1
        self.assert_(isinstance(v, type(self.v1)))
        self.assertAlmostEqual(v.x, self.v1.x / self.s1)
        self.assertAlmostEqual(v.y, self.v1.y / self.s1)
        self.assertAlmostEqual(v.z, self.v1.z / self.s1)
        v = self.v1 // self.s2
        self.assert_(isinstance(v, type(self.v1)))
        self.assertEqual(v.x, self.v1.x // self.s2)
        self.assertEqual(v.y, self.v1.y // self.s2)
        self.assertEqual(v.z, self.v1.z // self.s2)

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
        self.assertEqual(v.z, self.v1.z)
        self.assertNotEqual(id(v), id(self.v1))
        v = -self.v1
        self.assert_(isinstance(v, type(self.v1)))
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
        self.assertEqual(str(v), repr (v))
        
    def testRepr(self):
        v = Vector3(1.2, 3.4, -9.6)
        self.assertEqual (repr (v),
                          "Vector3(1.20000000, 3.40000000, -9.60000000)")
        self.assertEqual(v, eval(v.__repr__()))

    def testIter(self):
        it = self.v1.__iter__()
        # support py2.x and 3.x
        if sys.version_info < (2, 6):
            self.assertEqual(it.next(), self.v1[0])
            self.assertEqual(it.next(), self.v1[1])
            self.assertEqual(it.next(), self.v1[2])
            self.assertRaises(StopIteration, lambda : it.next())
        else:
            self.assertEqual(next(it), self.v1[0])
            self.assertEqual(next(it), self.v1[1])
            self.assertEqual(next(it), self.v1[2])
            self.assertRaises(StopIteration, lambda : next(it))
        it1 = iter (self.v1)
        it2 = iter (self.v1)
        self.assertNotEqual(id(it1), id(it2))
        self.assertEqual(id(it1), id(iter (it1)))
        self.assertEqual(list(it1), list(it2));
        self.assertEqual(list(iter(self.v1)), self.l1)
        idx = 0
        for val in self.v1:
            self.assertEqual(val, self.v1[idx])
            idx += 1

    def test_pygame2_math_base_Vector3 (self):
        self.assertRaises (TypeError, Vector3, 0, 0, 0, 0)
        self.assertRaises (TypeError, Vector3, None, None, None)
        self.assertRaises (TypeError, Vector3, None, None, 0)
        self.assertRaises (TypeError, Vector3, None, 0, None)
        self.assertRaises (TypeError, Vector3, None, 0, 0)
        self.assertRaises (TypeError, Vector3, 0, None, 0)
        self.assertRaises (TypeError, Vector3, 0, None, None)
        self.assertRaises (TypeError, Vector3, 0, 0, None)
        self.assertRaises (TypeError, Vector3, "Hello", "World", "!")
        
        v = Vector3 (0, 0, 0)
        self.assertEqual (v.dimension, 3)
        self.assertEqual (len (v.elements), 3)
        
    def test_pygame2_math_base_Vector3_x(self):

        # __doc__ (as of 2010-01-06) for pygame2.math.base.Vector3.x:

        # Gets or sets first element of the Vector3.
        
        v = Vector3 (10, 10, 10)
        self.assertEqual (v.x, 10)
        v.x = 33.44
        self.assertEqual (v.x, 33.44)
        self.assertEqual (v.elements[0], 33.44)
        self.assertEqual (v.elements[1], 10)
        self.assertEqual (v.elements[2], 10)

    def test_pygame2_math_base_Vector3_y(self):

        # __doc__ (as of 2010-01-06) for pygame2.math.base.Vector3.y:

        # Gets or sets second element of the Vector3.

        v = Vector3 (10, 10, 10)
        self.assertEqual (v.y, 10)
        v.y = 33.44
        self.assertEqual (v.y, 33.44)
        self.assertEqual (v.elements[0], 10)
        self.assertEqual (v.elements[1], 33.44)
        self.assertEqual (v.elements[2], 10)

    def test_pygame2_math_base_Vector3_z(self):

        # __doc__ (as of 2010-01-06) for pygame2.math.base.Vector3.z:

        # Gets or sets third element of the Vector3.

        v = Vector3 (10, 10, 10)
        self.assertEqual (v.z, 10)
        v.z = 33.44
        self.assertEqual (v.z, 33.44)
        self.assertEqual (v.elements[0], 10)
        self.assertEqual (v.elements[1], 10)
        self.assertEqual (v.elements[2], 33.44)

    def todo_test_pygame2_math_base_Vector3_angle_to(self):

        # __doc__ (as of 2010-01-09) for pygame2.math.base.Vector3.angle_to:

        # angle_to (v) -> float
        self.assertEqual(Vector3(1, 1, 0).angle_to((-1, 1, 0)), 90)
        self.assertEqual(Vector3(1, 0, 0).angle_to((0, 0, -1)), 90)
        self.assertEqual(Vector3(1, 0, 0).angle_to((-1, 0, 1)), 135)
        self.assertEqual(abs(Vector3(1, 0, 1).angle_to((-1, 0, -1))), 180)
        # if we rotate v1 by the angle_to v2 around their cross product
        # we should look in the same direction
        self.assertEqual(self.v1.rotate(self.v1.angle_to(self.v2), self.v1.cross(self.v2)).normalize(),
                         self.v2.normalize())


    def todo_test_pygame2_math_base_Vector3_as_spherical(self):

        # __doc__ (as of 2010-01-09) for pygame2.math.base.Vector3.as_spherical:

        # as_spherical () -> float, float, float

        self.fail() 

    def todo_test_pygame2_math_base_Vector3_cross(self):

        # __doc__ (as of 2010-01-09) for pygame2.math.base.Vector3.cross:

        # cross (v) -> Vector3
        def cross(a, b):
            return Vector3(a[1] * b[2] - a[2] * b[1],
                           a[2] * b[0] - a[0] * b[2],
                           a[0] * b[1] - a[1] * b[0])
        self.assertEqual(self.v1.cross(self.v2), cross(self.v1, self.v2))
        self.assertEqual(self.v1.cross(self.l2), cross(self.v1, self.l2))
        self.assertEqual(self.v1.cross(self.t2), cross(self.v1, self.t2))
        self.assertEqual(self.v1.cross(self.v2), -self.v2.cross(self.v1))
        self.assertEqual(self.v1.cross(self.v1), self.zeroVec)

    def test_pygame2_math_base_Vector3_rotate(self):

        # __doc__ (as of 2010-01-09) for pygame2.math.base.Vector3.rotate:

        # rotate () -> Vector3
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

    def test_pygame2_math_base_Vector3_rotate_ip(self):

        # __doc__ (as of 2010-01-09) for pygame2.math.base.Vector3.rotate_ip:

        # rotate_ip () -> Vector3
        v = Vector3(1, 0)
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

    def test_pygame2_math_base_Vector3_rotate_x(self):

        # __doc__ (as of 2010-01-09) for pygame2.math.base.Vector3.rotate_x:

        # rotate_x () -> Vector3
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

    def test_pygame2_math_base_Vector3_rotate_x_ip(self):

        # __doc__ (as of 2010-01-09) for pygame2.math.base.Vector3.rotate_x_ip:

        # rotate_x_ip () -> None
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

    def test_pygame2_math_base_Vector3_rotate_y(self):

        # __doc__ (as of 2010-01-09) for pygame2.math.base.Vector3.rotate_y:

        # rotate_y () -> Vector3
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

    def test_pygame2_math_base_Vector3_rotate_y_ip(self):

        # __doc__ (as of 2010-01-09) for pygame2.math.base.Vector3.rotate_y_ip:

        # rotate_y_ip () -> None
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

    def test_pygame2_math_base_Vector3_rotate_z(self):

        # __doc__ (as of 2010-01-09) for pygame2.math.base.Vector3.rotate_z:

        # rotate_z () -> Vector3
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

    def test_pygame2_math_base_Vector3_rotate_z_ip(self):

        # __doc__ (as of 2010-01-09) for pygame2.math.base.Vector3.rotate_z_ip:

        # rotate_z_ip () -> None
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
        self.assertRaises(ZeroDivisionError, lambda : self.zeroVec.normalize())
        
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
        self.assertRaises(ZeroDivisionError,
                          lambda : self.zeroVec.normalize_ip())

    def test_normalized(self):
        self.assertEqual(self.v1.normalized, False)
        v = self.v1.normalize()
        self.assertEqual(v.normalized, True)
        self.assertEqual(self.e2.normalized, True)
        self.assertEqual(self.zeroVec.normalized, False)

    def test_dot(self):
        self.assertAlmostEqual(self.v1.dot(self.v2),
                               self.v1.x * self.v2.x + self.v1.y * self.v2.y + self.v1.z * self.v2.z)
        self.assertAlmostEqual(self.v1.dot(self.l2),
                               self.v1.x * self.l2[0] + self.v1.y * self.l2[1] + self.v1.z * self.l2[2])
        self.assertAlmostEqual(self.v1.dot(self.t2),
                         self.v1.x * self.t2[0] + self.v1.y * self.t2[1] + self.v1.z * self.t2[2])
        self.assertAlmostEqual(self.v1.dot(self.v2), self.v2.dot(self.v1))
        self.assertAlmostEqual(self.v1.dot(self.v2), self.v1 * self.v2)
    
    def test_scale_to(self):
        v = Vector3(1, 1, 1)
        v.scale_to(2.5)
        self.assertEqual(v, Vector3(2.5, 2.5, 2.5) / math.sqrt(3))
        self.assertRaises(ZeroDivisionError, lambda : self.zeroVec.scale_to(1))
        self.assertEqual(v.scale_to(0), None)
        self.assertEqual(v, self.zeroVec)

    def test_length(self):
        self.assertEqual(Vector3(3, 4, 5).length, math.sqrt(3 * 3 + 4 * 4 + 5 * 5))
        self.assertEqual(Vector3(-3, 4, 5).length, math.sqrt(-3 * -3 + 4 * 4 + 5 * 5))
        self.assertEqual(self.zeroVec.length, 0)
        
    def test_length_squared(self):
        self.assertEqual(Vector3(3, 4, 5).length_squared, 3 * 3 + 4 * 4 + 5 * 5)
        self.assertEqual(Vector3(-3, 4, 5).length_squared, -3 * -3 + 4 * 4 + 5 * 5)
        self.assertEqual(self.zeroVec.length_squared, 0)

    def test_reflect(self):
        v = Vector3(1, -1, 1)
        n = Vector3(0, 1, 0)
        self.assertEqual(v.reflect(n), Vector3(1, 1, 1))
        self.assertEqual(v.reflect(3*n), v.reflect(n))
        self.assertEqual(v.reflect(-v), -v)
        self.assertRaises(ZeroDivisionError, lambda : v.reflect(self.zeroVec))
        
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
        self.assertRaises(ZeroDivisionError, lambda : v2.reflect_ip(self.zeroVec))

    def test_distance(self):
        diff = self.v1 - self.v2
        self.assertEqual(self.e1.distance(self.e2), math.sqrt(2))
        self.assertEqual(self.v1.distance(self.v2),
                         math.sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z))
        self.assertEqual(self.v1.distance(self.v1), 0)
        self.assertEqual(self.v1.distance(self.v2),
                         self.v2.distance(self.v1))

    def test_distance_squared(self):
        diff = self.v1 - self.v2
        self.assertAlmostEqual(self.e1.distance_squared(self.e2), 2)
        self.assertAlmostEqual(self.v1.distance_squared(self.v2),
            diff.x * diff.x + diff.y * diff.y + diff.z * diff.z)
        self.assertAlmostEqual(self.v1.distance_squared(self.v1), 0)
        self.assertAlmostEqual(self.v1.distance_squared(self.v2),
                               self.v2.distance_squared(self.v1))
    
    def test_slerp(self):
        self.assertRaises(ZeroDivisionError,
                          lambda : self.zeroVec.slerp(self.v1, .5))
        self.assertRaises(ZeroDivisionError,
                          lambda : self.v1.slerp(self.zeroVec, .5))
        self.assertRaises(ZeroDivisionError,
                          lambda : self.zeroVec.slerp(self.zeroVec, .5))
        steps = 10
        angle_step = self.e1.angle_to(self.e2) / steps
        for i, u in ((i, self.e1.slerp(self.e2, i/float(steps))) for i in range(steps+1)):
            self.assertAlmostEqual(u.length, 1)
            self.assertAlmostEqual(self.e1.angle_to(u), i * angle_step)
        self.assertEqual(u, self.e2)
        
        v1 = Vector3(100, 0, 0)
        v2 = Vector3(0, 10, 7)
        radial_factor = v2.length / v1.length
        for i, u in ((i, v1.slerp(v2, -i/float(steps))) for i in range(steps+1)):
            self.assertAlmostEqual(u.length, (v2.length - v1.length) * (float(i)/steps) + v1.length)
        self.assertEqual(u, v2)

    def test_spherical(self):
        v = Vector3()
        from_spherical = pmath.vector_from_spherical
        v = from_spherical(*self.v1.as_spherical ())
        self.assertEqual(self.v1, v)
        self.assertEqual(self.e1.as_spherical(), (1, math.pi / 2., 0))
        self.assertEqual(self.e2.as_spherical(), (1, math.pi / 2., math.pi / 2))
        self.assertEqual(self.e3.as_spherical(), (1, 0, 0))
        self.assertEqual((2 * self.e2).as_spherical(),
                         (2, math.pi / 2., math.pi / 2))
        self.assertRaises(TypeError, lambda : from_spherical((None, None, None)))
        self.assertRaises(TypeError, lambda : from_spherical("abc"))
        self.assertRaises(TypeError, lambda : from_spherical((None, 1, 2)))
        self.assertRaises(TypeError, lambda : from_spherical((1, 2, 3, 4)))
        self.assertRaises(TypeError, lambda : from_spherical((1, 2)))
        self.assertRaises(TypeError, lambda : from_spherical((1, 2, 3)))
        v = from_spherical(.5, math.pi / 2., math.pi / 2.)
        self.assertEqual(v, .5 * self.e2)

if __name__ == "__main__":
    unittest.main ()
