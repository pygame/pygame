.. include:: common.txt

:mod:`pygame.math`
==================

.. module:: pygame.math
   :synopsis: pygame module for vector classes

| :sl:`pygame module for vector classes`

!!!EXPERIMENTAL!!! Note: This Modul is still in development and the ``API``
might change. Please report bug and suggestions to pygame-users@seul.org

The pygame math module currently provides Vector classes in two and three
dimensions, Vector2 and Vector3 respectively.

They support the following numerical operations: vec+vec, vec-vec, vec*number,
number*vec, vec/number, vec//number, vec+=vec, vec-=vec, vec*=number,
vec/=number, vec//=number. All these operations will be performed elementwise.
In addition vec*vec will perform a scalar-product (a.k.a. dot-product). If you
want to multiply every element from vector v with every element from vector w
you can use the elementwise method: ``v.elementwise()`` ``\*`` w

Furthermore, element access is provided via the sequence notation
(``Vector3()[1]``) or accessor methods (``Vector3().y()``). It is
however noteworthy that Vectors are immutable. In other words they
behave more like tuple or strings than like lists.  For example ``a =
b = Vector2()`` you can safely change ``b`` without affecting ``a``.

New in Pygame 1.10

.. function:: enable_swizzling

   | :sl:`globally enables swizzling for vectors.`
   | :sg:`enable_swizzling() -> None`

   Enables swizzling for all vectors until ``disable_swizzling()`` is called.
   By default swizzling is disabled.

   .. ## pygame.math.enable_swizzling ##

.. function:: disable_swizzling

   | :sl:`globally disables swizzling for vectors.`
   | :sg:`disable_swizzling() -> None`

   Disables swizzling for all vectors until ``enable_swizzling()`` is called.
   By default swizzling is disabled.

   .. ## pygame.math.disable_swizzling ##

.. class:: Vector2

   | :sl:`a 2-Dimensional Vector`
   | :sg:`Vector2() -> Vector2`
   | :sg:`Vector2(Vector2) -> Vector2`
   | :sg:`Vector2(x, y) -> Vector2`
   | :sg:`Vector2((x, y)) -> Vector2`

   Some general information about the Vector2 class.

   .. method:: x

      | :sl:`return the first component of the vector`
      | :sg:`x() -> float`

      .. ## Vector2.x ##

   .. method:: y

      | :sl:`return the second component of the vector`
      | :sg:`y() -> float`

      .. ## Vector2.y ##

   .. method:: dot

      | :sl:`calculates the dot- or scalar-product with the other vector`
      | :sg:`dot(Vector2) -> float`

      .. ## Vector2.dot ##

   .. method:: cross

      | :sl:`calculates the cross- or vector-product`
      | :sg:`cross(Vector2) -> float`

      calculates the third component of the cross-product.

      .. ## Vector2.cross ##

   .. method:: length

      | :sl:`returns the euclidic length of the vector.`
      | :sg:`length() -> float`

      calculates the euclidic length of the vector which follows from the
      Pythagorean theorem: ``vec.length()`` ==
      ``math.sqrt(vec.x**2 + vec.y**2)``

      .. ## Vector2.length ##

   .. method:: length_squared

      | :sl:`returns the squared euclidic length of the vector.`
      | :sg:`length_squared() -> float`

      calculates the euclidic length of the vector which follows from the
      Pythagorean theorem: ``vec.length_squared()`` == vec.x**2 + vec.y**2 This
      is faster than ``vec.length()`` because it avoids the square root.

      .. ## Vector2.length_squared ##

   .. method:: normalize

      | :sl:`returns a vector with the same direction but length 1.`
      | :sg:`normalize() -> Vector2`

      Returns a new vector that has length == 1 and the same direction as self.

      .. ## Vector2.normalize ##

   .. method:: is_normalized

      | :sl:`tests if the vector is normalized i.e. has length == 1.`
      | :sg:`is_normalized() -> Bool`

      Returns True if the vector has length == 1. Otherwise it returns False.

      .. ## Vector2.is_normalized ##

   .. method:: scale_to_length

      | :sl:`returns a vector in the direction of self and magnitude length.`
      | :sg:`scale_to_length(float) -> Vector2`

      Returns a Vector that points in the direction of self and has
      the given length. You can also scale to length 0. If the vector
      is the zero vector (i.e. has length 0 thus no direction) an
      ZeroDivisionError is raised.

      .. ## Vector2.scale_to_length ##

   .. method:: reflect

      | :sl:`returns a vector reflected of a given normal.`
      | :sg:`reflect(Vector2) -> Vector2`

      Returns a new vector that points in the direction as if self would bounce
      of a surface characterized by the given surface normal. The length of the
      new vector is the same as self's.

      .. ## Vector2.reflect ##

   .. method:: distance_to

      | :sl:`calculates the euclidic distance to a given vector.`
      | :sg:`distance_to(Vector2) -> float`

      .. ## Vector2.distance_to ##

   .. method:: distance_squared_to

      | :sl:`calculates the squared euclidic distance to a given vector.`
      | :sg:`distance_squared_to(Vector2) -> float`

      .. ## Vector2.distance_squared_to ##

   .. method:: lerp

      | :sl:`returns a linear interpolation to the given vector.`
      | :sg:`lerp(Vector2, float) -> Vector2`

      Returns a Vector which is a linear interpolation between self and the
      given Vector. The second parameter determines how far between self an
      other the result is going to be. It must be a value between 0 and 1 where
      0 means self an 1 means other will be returned.

      .. ## Vector2.lerp ##

   .. method:: slerp

      | :sl:`returns a spherical interpolation to the given vector.`
      | :sg:`slerp(Vector2, float) -> Vector2`

      Calculates the spherical interpolation from self to the given Vector. The
      second argument - often called t - must be in the range [-1, 1]. It
      parametrizes where - in between the two vectors - the result should be.
      If a negative value is given the interpolation will not take the
      complement of the shortest path.

      .. ## Vector2.slerp ##

   .. method:: elementwise

      | :sl:`The next operation will be performed elementwize.`
      | :sg:`elementwise() -> VectorElementwizeProxy`

      Applies the following operation to each element of the vector.

      .. ## Vector2.elementwise ##

   .. method:: rotate

      | :sl:`rotates a vector by a given angle in degrees.`
      | :sg:`rotate(float) -> Vector2`

      Returns a vector which has the same length as self but is rotated
      counterclockwise by the given angle in degrees.

      .. ## Vector2.rotate ##

   .. method:: angle_to

      | :sl:`calculates the angle to a given vector in degrees.`
      | :sg:`angle_to(Vector2) -> float`

      Returns the angle between self and the given vector.

      .. ## Vector2.angle_to ##

   .. method:: as_polar

      | :sl:`returns a tuple with radial distance and azimuthal angle.`
      | :sg:`as_polar() -> (r, phi)`

      Returns a tuple (r, phi) where r is the radial distance, and phi is the
      azimuthal angle.

      .. ## Vector2.as_polar ##

   .. classmethod:: from_polar

      | :sl:`Creates a new Vector from a radius and an angle.`
      | :sg:`from_polar((r, phi)) -> Vector2`

      Creates a new Vector from a tuple (r, phi) where r is the radial
      distance, and phi is the azimuthal angle.

      .. ## Vector2.from_polar ##

   .. ## pygame.math.Vector2 ##

.. class:: Vector3

   | :sl:`a 3-Dimensional Vector`
   | :sg:`Vector3() -> Vector3`
   | :sg:`Vector3(Vector3) -> Vector3`
   | :sg:`Vector3(x, y, z) -> Vector3`
   | :sg:`Vector3((x, y, z)) -> Vector3`

   Some general information about the Vector3 class.

   .. method:: x

      | :sl:`return the first component of the vector`
      | :sg:`x() -> float`

      .. ## Vector3.x ##

   .. method:: y

      | :sl:`return the second component of the vector`
      | :sg:`y() -> float`

      .. ## Vector3.y ##

   .. method:: z

      | :sl:`return the third component of the vector`
      | :sg:`z() -> float`

      .. ## Vector3.z ##

   .. method:: dot

      | :sl:`calculates the dot- or scalar-product with the other vector`
      | :sg:`dot(Vector3) -> float`

      .. ## Vector3.dot ##

   .. method:: cross

      | :sl:`calculates the cross- or vector-product`
      | :sg:`cross(Vector3) -> float`

      calculates the cross-product.

      .. ## Vector3.cross ##

   .. method:: length

      | :sl:`returns the euclidic length of the vector.`
      | :sg:`length() -> float`

      calculates the euclidic length of the vector which follows from the
      Pythagorean theorem: ``vec.length()`` ==
      ``math.sqrt(vec.x**2 + vec.y**2 + vec.z**2)``

      .. ## Vector3.length ##

   .. method:: length_squared

      | :sl:`returns the squared euclidic length of the vector.`
      | :sg:`length_squared() -> float`

      calculates the euclidic length of the vector which follows from the
      Pythagorean theorem: ``vec.length_squared()`` == vec.x**2 + vec.y**2 +
      vec.z**2 This is faster than ``vec.length()`` because it avoids the
      square root.

      .. ## Vector3.length_squared ##

   .. method:: normalize

      | :sl:`returns a vector with the same direction but length 1.`
      | :sg:`normalize() -> Vector3`

      Returns a new vector that has length == 1 and the same direction as self.

      .. ## Vector3.normalize ##

   .. method:: is_normalized

      | :sl:`tests if the vector is normalized i.e. has length == 1.`
      | :sg:`is_normalized() -> Bool`

      Returns True if the vector has length == 1. Otherwise it returns False.

      .. ## Vector3.is_normalized ##

   .. method:: scale_to_length

      | :sl:`scales the vector to a given length.`
      | :sg:`scale_to_length(float) -> None`

      Scales the vector so that it has the given length. The direction of the
      vector is not changed. You can also scale to length 0. If the vector is
      the zero vector (i.e. has length 0 thus no direction) an
      ZeroDivisionError is raised.

      .. ## Vector3.scale_to_length ##

   .. method:: reflect

      | :sl:`returns a vector reflected of a given normal.`
      | :sg:`reflect(Vector3) -> Vector3`

      Returns a new vector that points in the direction as if self would bounce
      of a surface characterized by the given surface normal. The length of the
      new vector is the same as self's.

      .. ## Vector3.reflect ##

   .. method:: distance_to

      | :sl:`calculates the euclidic distance to a given vector.`
      | :sg:`distance_to(Vector3) -> float`

      .. ## Vector3.distance_to ##

   .. method:: distance_squared_to

      | :sl:`calculates the squared euclidic distance to a given vector.`
      | :sg:`distance_squared_to(Vector3) -> float`

      .. ## Vector3.distance_squared_to ##

   .. method:: lerp

      | :sl:`returns a linear interpolation to the given vector.`
      | :sg:`lerp(Vector3, float) -> Vector3`

      Returns a Vector which is a linear interpolation between self and the
      given Vector. The second parameter determines how far between self an
      other the result is going to be. It must be a value between 0 and 1 where
      0 means self an 1 means other will be returned.

      .. ## Vector3.lerp ##

   .. method:: slerp

      | :sl:`returns a spherical interpolation to the given vector.`
      | :sg:`slerp(Vector3, float) -> Vector3`

      Calculates the spherical interpolation from self to the given Vector. The
      second argument - often called t - must be in the range [-1, 1]. It
      parametrizes where - in between the two vectors - the result should be.
      If a negative value is given the interpolation will not take the
      complement of the shortest path.

      .. ## Vector3.slerp ##

   .. method:: elementwise

      | :sl:`The next operation will be performed elementwize.`
      | :sg:`elementwise() -> VectorElementwizeProxy`

      Applies the following operation to each element of the vector.

      .. ## Vector3.elementwise ##

   .. method:: rotate

      | :sl:`rotates a vector by a given angle in degrees.`
      | :sg:`rotate(Vector3, float) -> Vector3`

      Returns a vector which has the same length as self but is rotated
      counterclockwise by the given angle in degrees around the given axis.

      .. ## Vector3.rotate ##

   .. method:: rotate_x

      | :sl:`rotates a vector around the x-axis by the angle in degrees.`
      | :sg:`rotate_x(float) -> Vector3`

      Returns a vector which has the same length as self but is rotated
      counterclockwise around the x-axis by the given angle in degrees.

      .. ## Vector3.rotate_x ##

   .. method:: rotate_y

      | :sl:`rotates a vector around the y-axis by the angle in degrees.`
      | :sg:`rotate_y(float) -> Vector3`

      Returns a vector which has the same length as self but is rotated
      counterclockwise around the y-axis by the given angle in degrees.

      .. ## Vector3.rotate_y ##

   .. method:: rotate_z

      | :sl:`rotates a vector around the z-axis by the angle in degrees.`
      | :sg:`rotate_z(float) -> Vector3`

      Returns a vector which has the same length as self but is rotated
      counterclockwise around the z-axis by the given angle in degrees.

      .. ## Vector3.rotate_z ##

   .. method:: angle_to

      | :sl:`calculates the angle to a given vector in degrees.`
      | :sg:`angle_to(Vector3) -> float`

      Returns the angle between self and the given vector.

      .. ## Vector3.angle_to ##

   .. method:: as_spherical

      | :sl:`returns a tuple with radial distance, inclination and azimuthal angle.`
      | :sg:`as_spherical() -> (r, theta, phi)`

      Returns a tuple (r, theta, phi) where r is the radial distance, theta is
      the inclination angle and phi is the azimuthal angle.

      .. ## Vector3.as_spherical ##

   .. classmethod:: from_spherical

      | :sl:`Create a new Vector from a spherical coordinates 3-tuple.`
      | :sg:`from_spherical((r, theta, phi)) -> Vector3`

      Creates a new Vector3 from a tuple (r, theta, phi) where r is the radial
      distance, theta is the inclination angle and phi is the azimuthal angle.

      .. ## Vector3.from_spherical ##

   .. ##  ##

   .. ## pygame.math.Vector3 ##

.. ## pygame.math ##
