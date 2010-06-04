import unittest
import sys, math
import pygame2
import pygame2.math as pmath
from pygame2.math import Vector, Vector2

class MathVector2Test (unittest.TestCase):
    def setUp (self):
        self.zeroVec = Vector2 ()
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

    def tearDown (self):
        pass

    def testConstructionDefault(self):
        v = Vector2()
        self.assertEqual(v.x, 0.)
        self.assertEqual(v.y, 0.)

    def testConstructionXY(self):
        v = Vector2(1.2, 3.4)
        self.assertEqual(v.x, 1.2)
        self.assertEqual(v.y, 3.4)

    def testConstructionTuple(self):
        v = Vector((1.2, 3.4))
        self.assertEqual(v.elements[0], 1.2)
        self.assertEqual(v.elements[1], 3.4)

    def testConstructionList(self):
        v = Vector([1.2, 3.4])
        self.assertEqual(v.elements[0], 1.2)
        self.assertEqual(v.elements[1], 3.4)

    def testConstructionVector2(self):
        v = Vector(Vector2(1.2, 3.4))
        self.assertEqual(v.elements[0], 1.2)
        self.assertEqual(v.elements[1], 3.4)

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
        self.assertEqual(v[1:], 3.4)
        self.assertEqual(v[:1], 1.2)
        self.assertEqual(list(v), [1.2, 3.4])
        self.assertEqual(tuple(v), (1.2, 3.4))
        v[0] = 5.6
        v[1] = 7.8
        self.assertEqual(v.x, 5.6)
        self.assertEqual(v.y, 7.8)
        v[:] = [9.1, 11.12]
        self.assertEqual(v.x, 9.1)
        self.assertEqual(v.y, 11.12)
        def overpopulate ():
            v = Vector2()
            v[:] = [1, 2, 3]
        self.assertRaises(ValueError, overpopulate)
        def underpopulate():
            v = Vector2()
            v[:] = [1]
        self.assertRaises(ValueError, underpopulate)

    def testExtendedSlicing(self):
        #  deletion
        _v1 = Vector ((1.2, 3.4))
        
        def delSlice(vec, start=None, stop=None, step=None):
            if start is not None and stop is not None and step is not None:
                del vec[start:stop:step]
            elif start is not None and stop is None and step is not None:
                del vec[start::step]
            elif start is None and stop is None and step is not None:
                del vec[::step]
        v = Vector(_v1)
        
        self.assertRaises(ValueError, delSlice, v, None, None, 2)
        self.assertRaises(ValueError, delSlice, v, 1, None, 2)
        self.assertRaises(ValueError, delSlice, v, 1, 2, 1)

        #  assignment
        v = (_v1)
        v[::2] = [-1]
        self.assertEqual(v, Vector ((-1, _v1.elements[1])))
        v = Vector(_v1)
        v[::-2] = [10]
        self.assertEqual(v, Vector2 ((_v1.elements[0], 10)))
        v = Vector(_v1)
        v[::-1] = v
        self.assertEqual(v, Vector ((_v1.elements[1], _v1.elements[0])))
        a = Vector(_v1)
        b = Vector(_v1)
        c = Vector(_v1)
        a[1:2] = [2.2]
        b[slice(1,2)] = [2.2]
        c[1:2:] = (2.2,)
        self.assertEqual(a, b)
        self.assertEqual(a, c)
        self.assertEqual(type(a), type(_v1))
        self.assertEqual(type(b), type(_v1))
        self.assertEqual(type(c), type(_v1))

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
        self.assertEqual(str(v), repr (v))
        
    def testRepr(self):
        v = Vector2(1.2, 3.4)
        self.assertEqual(repr (v), "Vector2(1.20000000, 3.40000000)")
        self.assertEqual(v, eval(repr (v)))

    def testIter(self):
        it = iter (self.v1)
        # support py2.x and 3.x
        if sys.version_info < (2, 6):
            self.assertEqual(it.next(), self.v1[0])
            self.assertEqual(it.next(), self.v1[1])
            self.assertRaises(StopIteration, lambda : it.next())
        else:
            self.assertEqual(next(it), self.v1[0])
            self.assertEqual(next(it), self.v1[1])
            self.assertRaises(StopIteration, lambda : next(it))
        it1 = iter (self.v1)
        it2 = iter (self.v1)
        self.assertNotEqual(id(it1), id(it2))
        self.assertEqual(id(it1), id(iter (it1)))
        self.assertEqual(list(it1), list(it2));
        self.assertEqual(list(iter (self.v1)), self.l1)
        idx = 0
        for val in self.v1:
            self.assertEqual(val, self.v1[idx])
            idx += 1

    def test_pygame2_math_base_Vector_dimension(self):

        # __doc__ (as of 2010-01-06) for pygame2.math.base.Vector.dimension:

        # Gets the dimensions of the Vector.
        
        self.assertRaises (ValueError, Vector, -1)
        self.assertRaises (ValueError, Vector, 0)
        self.assertRaises (ValueError, Vector, 1)

        for i in range (2, 50):
            v = Vector (i)
            self.assertEqual (v.dimension, i)

    def test_pygame2_math_base_Vector_elements(self):

        # __doc__ (as of 2010-01-06) for pygame2.math.base.Vector.elements:

        # Gets or sets the elements of the Vector.
        # 
        # This will only set a maximum of dimension values.
        
        def setelems (v, elems):
            v.elements = elems
        
        for i in range (2, 50):
            v = Vector (i)
            elems = v.elements
            self.assertEqual (len (elems), i)
            for j in range (i):
                self.assertEqual (elems[j], 0)
        v = Vector ([1,2,3,4])
        self.assertEqual (len (v.elements), 4)
        self.assertEqual (v.elements[0], 1)
        self.assertEqual (v.elements[1], 2)
        self.assertEqual (v.elements[2], 3)
        self.assertEqual (v.elements[3], 4)
        v = Vector (10)
        self.assertEqual (len (v.elements), 10)
        v.elements = (1, 2, 3, 4)
        self.assertEqual (v.elements[0], 1)
        self.assertEqual (v.elements[1], 2)
        self.assertEqual (v.elements[2], 3)
        self.assertEqual (v.elements[3], 4)
        for i in range (4, 10):
            self.assertEqual (v.elements[i], 0)
        v.elements = range (50)
        self.assertEqual (len (v.elements), 10)
        for i in range (10):
            self.assertEqual (v.elements[i], i)

        self.assertRaises (TypeError, setelems, v, "Hello")
        self.assertRaises (TypeError, setelems, v, None)
            
    def test_pygame2_math_base_Vector_epsilon(self):

        # __doc__ (as of 2010-01-06) for pygame2.math.base.Vector.epsilon:

        # Gets or sets the exactness delta of the Vector.

        def seteps (v, eps):
            v.epsilon = eps
        
        v = Vector (2);
        self.assertAlmostEqual (v.epsilon, 0, places=5)
        v.epsilon = 0.0000004
        self.assertEqual (v.epsilon, 0.0000004)
        v.epsilon = 57293.2
        self.assertEqual (v.epsilon, 57293.2)
        
        self.assertRaises (TypeError, seteps, v, None)
        self.assertRaises (TypeError, seteps, v, "Hello")

    def test_pygame2_math_base_Vector2 (self):
        self.assertRaises (TypeError, Vector2, 0, 0, 0)
        self.assertRaises (TypeError, Vector2, None, None)
        self.assertRaises (TypeError, Vector2, None, 0)
        self.assertRaises (TypeError, Vector2, 0, None)
        self.assertRaises (TypeError, Vector2, "Hello", "World")
        
        v = Vector2 (0, 0)
        self.assertEqual (v.dimension, 2)
        self.assertEqual (len (v.elements), 2)
        
    def test_pygame2_math_base_Vector2_x(self):

        # __doc__ (as of 2010-01-06) for pygame2.math.base.Vector2.x:

        # Gets or sets first element of the Vector2.
        
        v = Vector2 (10, 10)
        self.assertEqual (v.x, 10)
        v.x = 33.44
        self.assertEqual (v.x, 33.44)
        self.assertEqual (v.elements[0], 33.44)
        self.assertEqual (v.elements[1], 10)

    def test_pygame2_math_base_Vector2_y(self):

        # __doc__ (as of 2010-01-06) for pygame2.math.base.Vector2.y:

        # Gets or sets second element of the Vector2.

        v = Vector2 (10, 10)
        self.assertEqual (v.y, 10)
        v.y = 33.44
        self.assertEqual (v.y, 33.44)
        self.assertEqual (v.elements[0], 10)
        self.assertEqual (v.elements[1], 33.44)


    def test_pygame2_math_base_Vector_length(self):

        # __doc__ (as of 2010-01-07) for pygame2.math.base.Vector.length:

        # Gets the length of the Vector.
        self.assertEqual(Vector2(3, 4).length, 5)
        self.assertEqual(Vector2(-3, 4).length, 5)
        self.assertEqual(self.zeroVec.length, 0)

    def test_pygame2_math_base_Vector_length_squared(self):

        # __doc__ (as of 2010-01-07) for pygame2.math.base.Vector.length_squared:

        # Gets the length of the Vector.
        self.assertEqual(Vector2(3, 4).length_squared, 25)
        self.assertEqual(Vector2(-3, 4).length_squared, 25)
        self.assertEqual(self.zeroVec.length_squared, 0)

    def test_pygame2_math_base_Vector_normalize(self):

        # __doc__ (as of 2010-01-07) for pygame2.math.base.Vector.normalize:

        # ormalize () -> Vector
        v = self.v1.normalize()
        # length is 1
        self.assertAlmostEqual(v.x * v.x + v.y * v.y, 1.)
        # v1 is unchanged
        self.assertEqual(self.v1.x, self.l1[0])
        self.assertEqual(self.v1.y, self.l1[1])
        # v2 is paralell to v1
        self.assertAlmostEqual(self.v1.x * v.y - self.v1.y * v.x, 0.)
        self.assertRaises(ValueError, lambda : self.zeroVec.normalize())

    def test_pygame2_math_base_Vector_normalize_ip(self):

        # __doc__ (as of 2010-01-07) for pygame2.math.base.Vector.normalize_ip:

        # ormalize_ip () -> None
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

    def test_pygame2_math_base_Vector_distance(self):

        # __doc__ (as of 2010-01-09) for pygame2.math.base.Vector.distance:

        # distance () -> None
        diff = self.v1 - self.v2
        self.assertEqual(self.e1.distance(self.e2), math.sqrt(2))
        self.assertEqual(self.v1.distance(self.v2),
                         math.sqrt(diff.x * diff.x + diff.y * diff.y))
        self.assertEqual(self.v1.distance(self.v1), 0)
        self.assertEqual(self.v1.distance(self.v2),
                         self.v2.distance(self.v1))

    def test_pygame2_math_base_Vector_distance_squared(self):

        # __doc__ (as of 2010-01-09) for pygame2.math.base.Vector.distance_squared:

        # distance_squared () -> None
        diff = self.v1 - self.v2
        self.assertEqual(self.e1.distance_squared(self.e2), 2)
        self.assertEqual(self.v1.distance_squared(self.v2),
                         diff.x * diff.x + diff.y * diff.y)
        self.assertEqual(self.v1.distance_squared(self.v1), 0)
        self.assertEqual(self.v1.distance_squared(self.v2),
                         self.v2.distance_squared(self.v1))

    def test_pygame2_math_base_Vector_dot(self):

        # __doc__ (as of 2010-01-09) for pygame2.math.base.Vector.dot:

        # dot () -> float
        self.assertAlmostEqual(self.v1.dot(self.v2),
                               self.v1.x * self.v2.x + self.v1.y * self.v2.y)
        self.assertAlmostEqual(self.v1.dot(self.l2),
                               self.v1.x * self.l2[0] + self.v1.y * self.l2[1])
        self.assertAlmostEqual(self.v1.dot(self.t2),
                               self.v1.x * self.t2[0] + self.v1.y * self.t2[1])
        self.assertAlmostEqual(self.v1.dot(self.v2), self.v2.dot(self.v1))
        self.assertAlmostEqual(self.v1.dot(self.v2), self.v1 * self.v2)

    def todo_test_pygame2_math_base_Vector_lerp(self):

        # __doc__ (as of 2010-01-09) for pygame2.math.base.Vector.lerp:

        # lerp () -> Vector

        self.fail() 

    def test_pygame2_math_base_Vector_normalized(self):

        # __doc__ (as of 2010-01-09) for pygame2.math.base.Vector.normalized:

        # Gets whether the Vector is normalized.
        self.assertEqual(self.v1.normalized, False)
        v = self.v1.normalize()
        self.assertEqual(v.normalized, True)
        self.assertEqual(self.e2.normalized, True)
        self.assertEqual(self.zeroVec.normalized, False)

    def test_pygame2_math_base_Vector_reflect(self):

        # __doc__ (as of 2010-01-09) for pygame2.math.base.Vector.reflect:

        # reflect () -> Vector
        v = Vector2(1, -1)
        n = Vector2(0, 1)
        self.assertEqual(v.reflect(n), Vector2(1, 1))
        self.assertEqual(v.reflect(3*n), v.reflect(n))
        self.assertEqual(v.reflect(-v), -v)
        self.assertRaises(ValueError, lambda : v.reflect(self.zeroVec))        

    def test_pygame2_math_base_Vector_reflect_ip(self):

        # __doc__ (as of 2010-01-09) for pygame2.math.base.Vector.reflect_ip:

        # reflect_ip () -> None
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

    def test_pygame2_math_base_Vector_scale_to(self):

        # __doc__ (as of 2010-01-09) for pygame2.math.base.Vector.scale_to:

        # scale_to (v) -> Vector
        v = Vector2(1, 1)
        v.scale_to(2.5)
        self.assertEqual(v, Vector2(2.5, 2.5) / math.sqrt(2))
        self.assertRaises(ValueError, lambda : self.zeroVec.scale_to(1))
        self.assertEqual(v.scale_to(0), None)
        self.assertEqual(v, self.zeroVec)

    def todo_test_pygame2_math_base_Vector_slerp(self):

        # __doc__ (as of 2010-01-09) for pygame2.math.base.Vector.slerp:

        # slerp () -> Vector

        self.fail() 

    def test_pygame2_math_base_Vector2_angle_to(self):

        # __doc__ (as of 2010-01-09) for pygame2.math.base.Vector2.angle_to:

        # angle_to (v) -> float
        self.assertEqual(self.v1.rotate(self.v1.angle_to(self.v2)).normalize(),
                         self.v2.normalize())
        self.assertEqual(Vector2(1, 1).angle_to((-1, 1)), 90)
        self.assertEqual(Vector2(1, 0).angle_to((0, -1)), -90)
        self.assertEqual(Vector2(1, 0).angle_to((-1, 1)), 135)
        self.assertEqual(abs(Vector2(1, 0).angle_to((-1, 0))), 180)

    def test_pygame2_math_base_Vector2_as_polar(self):

        # __doc__ (as of 2010-01-09) for pygame2.math.base.Vector2.as_polar:

        # as_polar () -> float, float
        v = Vector2()
        from_polar = pmath.vector_from_polar
        v = from_polar(*self.v1.as_polar())
        self.assertEqual(self.v1, v)
        self.assertEqual(self.e1.as_polar(), (1, 0))
        self.assertEqual(self.e2.as_polar(), (1, 90))
        self.assertEqual((2 * self.e2).as_polar(), (2, 90))
        self.assertRaises(TypeError, lambda : from_polar((None, None)))
        self.assertRaises(TypeError, lambda : from_polar("ab"))
        self.assertRaises(TypeError, lambda : from_polar((None, 1)))
        self.assertRaises(TypeError, lambda : from_polar((1, 2, 3)))
        self.assertRaises(TypeError, lambda : from_polar((1,)))
        v = from_polar(.5, 90)
        self.assertEqual(v, .5 * self.e2)

    def test_pygame2_math_base_Vector2_cross(self):

        # __doc__ (as of 2010-01-09) for pygame2.math.base.Vector2.cross:

        # cross (v) -> Vector2

        self.assertEqual(self.v1.cross(self.v2),
                         self.v1.x * self.v2.y - self.v1.y * self.v2.x)
        self.assertEqual(self.v1.cross(self.l2),
                         self.v1.x * self.l2[1] - self.v1.y * self.l2[0])
        self.assertEqual(self.v1.cross(self.t2),
                         self.v1.x * self.t2[1] - self.v1.y * self.t2[0])
        self.assertEqual(self.v1.cross(self.v2), -self.v2.cross(self.v1))
        self.assertEqual(self.v1.cross(self.v1), 0)

    def test_pygame2_math_base_Vector2_rotate(self):

        # __doc__ (as of 2010-01-09) for pygame2.math.base.Vector2.rotate:

        # rotate () -> Vector2

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

    def test_pygame2_math_base_Vector2_rotate_ip(self):

        # __doc__ (as of 2010-01-09) for pygame2.math.base.Vector2.rotate_ip:

        # rotate_ip () -> Vector2
        v = Vector2(1, 0)
        self.assertEqual(v.rotate_ip(90), None)
        self.assertEqual(v.x, 0)
        self.assertEqual(v.y, 1)
        v = Vector2(-1, -1)
        v.rotate_ip(-90)
        self.assertEqual(v.x, -1)
        self.assertEqual(v.y, 1)

if __name__ == "__main__":
    unittest.main ()
