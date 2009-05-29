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
from pygame import Math


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

    def testAdd(self):
        v1 = Vector2(1.2, 3.4)
        v2 = Vector2(5.6, 7.8)
        v3 = v1 + v2
        self.assert_(isinstance(v3, Vector2))
        self.assertEquacl(v3.x, 6.8)
        self.assertEquacl(v3.y, 11.2)
        v3 = v1 + (5.6, 7.8)
        self.assert_(isinstance(v3, Vector2))
        self.assertEquacl(v3.x, 6.8)
        self.assertEquacl(v3.y, 11.2)
        v3 = v1 + [5.6, 7.8]
        self.assert_(isinstance(v3, Vector2))
        self.assertEquacl(v3.x, 6.8)
        self.assertEquacl(v3.y, 11.2)
        v3 = (1.2, 3.4) + v2
        self.assert_(isinstance(v3, Vector2))
        self.assertEquacl(v3.x, 6.8)
        self.assertEquacl(v3.y, 11.2)
        v3 = [1.2, 3.4] + v2
        self.assert_(isinstance(v3, Vector2))
        self.assertEquacl(v3.x, 6.8)
        self.assertEquacl(v3.y, 11.2)

    def testSub(self):
        v1 = Vector2(1.2, 3.4)
        v2 = Vector2(5.6, 7.9)
        v3 = v1 - v2
        self.assert_(isinstance(v3, Vector2))
        self.assertEquacl(v3.x, -4.4)
        self.assertEquacl(v3.y, -4.5)
        v3 = v1 + (5.6, 7.9)
        self.assert_(isinstance(v3, Vector2))
        self.assertEquacl(v3.x, -4.4)
        self.assertEquacl(v3.y, -4.5)
        v3 = v1 + [5.6, 7.9]
        self.assert_(isinstance(v3, Vector2))
        self.assertEquacl(v3.x, -4.4)
        self.assertEquacl(v3.y, -4.5)
        v3 = (1.2, 3.4) + v2
        self.assert_(isinstance(v3, Vector2))
        self.assertEquacl(v3.x, -4.4)
        self.assertEquacl(v3.y, -4.5)
        v3 = [1.2, 3.4] + v2
        self.assert_(isinstance(v3, Vector2))
        self.assertEquacl(v3.x, -4.4)
        self.assertEquacl(v3.y, -4.5)

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
        self.assertEqual(v2.x, 1.2 / 5.6)
        self.assertEqual(v2.y, 3.4 / 5.6)
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
        self.assertEqual(v2.x, -1.2)
        self.assertEqual(v2.y, -3.4)
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
        self.assertEqual(v2.x, 1)
        self.assertEqual(v2.y, -1)
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
        self.assertEqual(v.x, 1)
        self.assertEqual(v.y, -1)

        
if __name__ == '__main__':
    unittest.main()
