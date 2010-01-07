try:
    import pygame2.test.pgunittest as unittest
except:
    import pgunittest as unittest

import pygame2
import pygame2.math as math

class MathTest (unittest.TestCase):

    def test_pygame2_math_base_Vector_dimension(self):

        # __doc__ (as of 2010-01-06) for pygame2.math.base.Vector.dimension:

        # Gets the dimensions of the Vector.
        
        self.assertRaises (ValueError, math.Vector, -1)
        self.assertRaises (ValueError, math.Vector, 0)
        self.assertRaises (ValueError, math.Vector, 1)

        for i in range (2, 50):
            v = math.Vector (i)
            self.assertEqual (v.dimension, i)

    def test_pygame2_math_base_Vector_elements(self):

        # __doc__ (as of 2010-01-06) for pygame2.math.base.Vector.elements:

        # Gets or sets the elements of the Vector.
        # 
        # This will only set a maximum of dimension values.
        
        def setelems (v, elems):
            v.elements = elems
        
        for i in range (2, 50):
            v = math.Vector (i)
            elems = v.elements
            self.assertEqual (len (elems), i)
            for j in range (i):
                self.assertEqual (elems[j], 0)
        v = math.Vector ([1,2,3,4])
        self.assertEqual (len (v.elements), 4)
        self.assertEqual (v.elements[0], 1)
        self.assertEqual (v.elements[1], 2)
        self.assertEqual (v.elements[2], 3)
        self.assertEqual (v.elements[3], 4)
        v = math.Vector (10)
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
        
        v = math.Vector (2);
        self.assertAlmostEqual (v.epsilon, 0)
        v.epsilon = 0.0000004
        self.assertEqual (v.epsilon, 0.0000004)
        v.epsilon = 57293.2
        self.assertEqual (v.epsilon, 57293.2)
        
        self.assertRaises (TypeError, seteps, v, None)
        self.assertRaises (TypeError, seteps, v, "Hello")

    def test_pygame2_math_base_Vector2 (self):
        self.assertRaises (TypeError, math.Vector2)
        self.assertRaises (TypeError, math.Vector2, 0)
        self.assertRaises (TypeError, math.Vector2, 0, 0, 0)
        self.assertRaises (TypeError, math.Vector2, None, None)
        self.assertRaises (TypeError, math.Vector2, None, 0)
        self.assertRaises (TypeError, math.Vector2, 0, None)
        self.assertRaises (TypeError, math.Vector2, "Hello", "World")
        
        v = math.Vector2 (0, 0)
        self.assertEqual (v.dimension, 2)
        self.assertEqual (len (v.elements), 2)
        
    def test_pygame2_math_base_Vector2_x(self):

        # __doc__ (as of 2010-01-06) for pygame2.math.base.Vector2.x:

        # Gets or sets first element of the Vector2.
        
        v = math.Vector2 (10, 10)
        self.assertEqual (v.x, 10)
        v.x = 33.44
        self.assertEqual (v.x, 33.44)
        self.assertEqual (v.elements[0], 33.44)
        self.assertEqual (v.elements[1], 10)

    def test_pygame2_math_base_Vector2_y(self):

        # __doc__ (as of 2010-01-06) for pygame2.math.base.Vector2.y:

        # Gets or sets second element of the Vector2.

        v = math.Vector2 (10, 10)
        self.assertEqual (v.y, 10)
        v.y = 33.44
        self.assertEqual (v.y, 33.44)
        self.assertEqual (v.elements[0], 10)
        self.assertEqual (v.elements[1], 33.44)

    def test_pygame2_math_base_Vector3 (self):
        self.assertRaises (TypeError, math.Vector3)
        self.assertRaises (TypeError, math.Vector3, 0)
        self.assertRaises (TypeError, math.Vector3, 0, 0)
        self.assertRaises (TypeError, math.Vector3, 0, 0, 0, 0)
        self.assertRaises (TypeError, math.Vector3, None, None, None)
        self.assertRaises (TypeError, math.Vector3, None, None, 0)
        self.assertRaises (TypeError, math.Vector3, None, 0, None)
        self.assertRaises (TypeError, math.Vector3, None, 0, 0)
        self.assertRaises (TypeError, math.Vector3, 0, None, 0)
        self.assertRaises (TypeError, math.Vector3, 0, None, None)
        self.assertRaises (TypeError, math.Vector3, 0, 0, None)
        self.assertRaises (TypeError, math.Vector3, "Hello", "World", "!")
        
        v = math.Vector3 (0, 0, 0)
        self.assertEqual (v.dimension, 3)
        self.assertEqual (len (v.elements), 3)
        
    def test_pygame2_math_base_Vector3_x(self):

        # __doc__ (as of 2010-01-06) for pygame2.math.base.Vector3.x:

        # Gets or sets first element of the Vector3.
        
        v = math.Vector3 (10, 10, 10)
        self.assertEqual (v.x, 10)
        v.x = 33.44
        self.assertEqual (v.x, 33.44)
        self.assertEqual (v.elements[0], 33.44)
        self.assertEqual (v.elements[1], 10)
        self.assertEqual (v.elements[2], 10)

    def test_pygame2_math_base_Vector3_y(self):

        # __doc__ (as of 2010-01-06) for pygame2.math.base.Vector3.y:

        # Gets or sets second element of the Vector3.

        v = math.Vector3 (10, 10, 10)
        self.assertEqual (v.y, 10)
        v.y = 33.44
        self.assertEqual (v.y, 33.44)
        self.assertEqual (v.elements[0], 10)
        self.assertEqual (v.elements[1], 33.44)
        self.assertEqual (v.elements[2], 10)

    def test_pygame2_math_base_Vector3_z(self):

        # __doc__ (as of 2010-01-06) for pygame2.math.base.Vector3.z:

        # Gets or sets third element of the Vector3.

        v = math.Vector3 (10, 10, 10)
        self.assertEqual (v.z, 10)
        v.z = 33.44
        self.assertEqual (v.z, 33.44)
        self.assertEqual (v.elements[0], 10)
        self.assertEqual (v.elements[1], 10)
        self.assertEqual (v.elements[2], 33.44)

    def todo_test_pygame2_math_base_Vector_length(self):

        # __doc__ (as of 2010-01-07) for pygame2.math.base.Vector.length:

        # Gets the length of the Vector.

        self.fail() 

    def todo_test_pygame2_math_base_Vector_length_squared(self):

        # __doc__ (as of 2010-01-07) for pygame2.math.base.Vector.length_squared:

        # Gets the length of the Vector.

        self.fail() 

    def todo_test_pygame2_math_base_Vector_normalize(self):

        # __doc__ (as of 2010-01-07) for pygame2.math.base.Vector.normalize:

        # ormalize () -> Vector

        self.fail() 

    def todo_test_pygame2_math_base_Vector_normalize_ip(self):

        # __doc__ (as of 2010-01-07) for pygame2.math.base.Vector.normalize_ip:

        # ormalize_ip () -> None

        self.fail() 

if __name__ == "__main__":
    unittest.main ()
