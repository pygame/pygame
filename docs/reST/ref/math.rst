.. include:: common.txt

:mod:`pygame.math`
==================

.. module:: pygame.math
   :synopsis: pygame module for vector classes

| :sl:`pygame module for vector classes`

The pygame math module currently provides Vector classes in two and three
dimensions, Vector2 and Vector3 respectively.

They support the following numerical operations: vec+vec, vec-vec, vec*number,
number*vec, vec/number, vec//number, vec+=vec, vec-=vec, vec*=number,
vec/=number, vec//=number. All these operations will be performed elementwise.
In addition vec*vec will perform a scalar-product (a.k.a. dot-product). If you
want to multiply every element from vector v with every element from vector w
you can use the elementwise method: ``v.elementwise() * w``

The coordinates of a vector can be retrieved or set using attributes or
subscripts::

   v = pygame.Vector3()

   v.x = 5
   v[1] = 2 * v.x
   print(v[1]) # 10

   v.x == v[0]
   v.y == v[1]
   v.z == v[2]

.. versionadded:: 1.9.2pre
.. versionchanged:: 1.9.4 Removed experimental notice.
.. versionchanged:: 1.9.4 Constructors require 2-3 elements rather than assigning 0 by default.
.. versionchanged:: 1.9.4 Allow scalar construction like GLSL Vector2(2) == Vector2(2.0, 2.0)
.. versionchanged:: 1.9.4 :mod:`pygame.math` required import. More convenient ``pygame.Vector2`` and ``pygame.Vector3``.

.. class:: Vector2

   | :sl:`a 2-Dimensional Vector`
   | :sg:`Vector2() -> Vector2`
   | :sg:`Vector2(int) -> Vector2`
   | :sg:`Vector2(float) -> Vector2`
   | :sg:`Vector2(Vector2) -> Vector2`
   | :sg:`Vector2(x, y) -> Vector2`
   | :sg:`Vector2((x, y)) -> Vector2`

   Some general information about the Vector2 class.

   .. method:: dot

      | :sl:`calculates the dot- or scalar-product with the other vector`
      | :sg:`dot(Vector2) -> float`

      .. ## Vector2.dot ##

   .. method:: cross

      | :sl:`calculates the cross- or vector-product`
      | :sg:`cross(Vector2) -> Vector2`

      calculates the third component of the cross-product.

      .. ## Vector2.cross ##

   .. method:: magnitude

      | :sl:`returns the Euclidean magnitude of the vector.`
      | :sg:`magnitude() -> float`

      calculates the magnitude of the vector which follows from the
      theorem: ``vec.magnitude()`` == ``math.sqrt(vec.x**2 + vec.y**2)``

      .. ## Vector2.magnitude ##

   .. method:: magnitude_squared

      | :sl:`returns the squared magnitude of the vector.`
      | :sg:`magnitude_squared() -> float`

      calculates the magnitude of the vector which follows from the
      theorem: ``vec.magnitude_squared()`` == vec.x**2 + vec.y**2 This
      is faster than ``vec.magnitude()`` because it avoids the square root.

      .. ## Vector2.magnitude_squared ##

   .. method:: length

      | :sl:`returns the Euclidean length of the vector.`
      | :sg:`length() -> float`

      calculates the Euclidean length of the vector which follows from the
      Pythagorean theorem: ``vec.length()`` ==
      ``math.sqrt(vec.x**2 + vec.y**2)``

      .. ## Vector2.length ##

   .. method:: length_squared

      | :sl:`returns the squared Euclidean length of the vector.`
      | :sg:`length_squared() -> float`

      calculates the Euclidean length of the vector which follows from the
      Pythagorean theorem: ``vec.length_squared()`` == vec.x**2 + vec.y**2 This
      is faster than ``vec.length()`` because it avoids the square root.

      .. ## Vector2.length_squared ##

   .. method:: normalize

      | :sl:`returns a vector with the same direction but length 1.`
      | :sg:`normalize() -> Vector2`

      Returns a new vector that has length == 1 and the same direction as self.

      .. ## Vector2.normalize ##

   .. method:: normalize_ip

      | :sl:`normalizes the vector in place so that its length is 1.`
      | :sg:`normalize_ip() -> Vector2`

      Normalizes the vector so that it has length == 1. The direction of the
      vector is not changed.

      .. versionchanged:: 1.9.5 The vector itself is returned instead of ``None``.

      .. ## Vector2.normalize_ip ##

   .. method:: is_normalized

      | :sl:`tests if the vector is normalized i.e. has length == 1.`
      | :sg:`is_normalized() -> Bool`

      Returns True if the vector has length == 1. Otherwise it returns False.

      .. ## Vector2.is_normalized ##

   .. method:: scale_to_length

      | :sl:`scales the vector in place to a given length.`
      | :sg:`scale_to_length(float) -> Vector2`

      Scales the vector in place so that it has the given length. The direction of
      the vector is not changed. You can also scale to length 0. If the vector is
      the zero vector (i.e. has length 0 thus no direction) an
      ZeroDivisionError is raised.

      .. versionchanged:: 1.9.5 The vector itself is returned instead of ``None``.

      .. ## Vector2.scale_to_length ##

   .. method:: reflect

      | :sl:`returns a vector reflected of a given normal.`
      | :sg:`reflect(Vector2) -> Vector2`

      Returns a new vector that points in the direction as if self would bounce
      of a surface characterized by the given surface normal. The length of the
      new vector is the same as self's.

      .. ## Vector2.reflect ##

   .. method:: reflect_ip

      | :sl:`reflect the vector of a given normal in place.`
      | :sg:`reflect_ip(Vector2) -> Vector2`

      Changes the direction of self as if it would have been reflected of a
      surface with the given surface normal.

      .. versionchanged:: 1.9.5 The vector itself is returned instead of ``None``.

      .. ## Vector2.reflect_ip ##

   .. method:: distance_to

      | :sl:`calculates the Euclidean distance to a given vector.`
      | :sg:`distance_to(Vector2) -> float`

      .. ## Vector2.distance_to ##

   .. method:: distance_squared_to

      | :sl:`calculates the squared Euclidean distance to a given vector.`
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

      | :sl:`The next operation will be performed elementwise.`
      | :sg:`elementwise() -> VectorElementwiseProxy`

      Applies the following operation to each element of the vector.

      .. ## Vector2.elementwise ##

   .. method:: rotate

      | :sl:`rotates a vector by a given angle in degrees.`
      | :sg:`rotate(float) -> Vector2`

      Returns a vector which has the same length as self but is rotated
      counterclockwise by the given angle in degrees.

      .. ## Vector2.rotate ##

   .. method:: rotate_ip

      | :sl:`rotates the vector by a given angle in degrees in place.`
      | :sg:`rotate_ip(float) -> Vector2`

      Rotates the vector counterclockwise by the given angle in degrees. The
      length of the vector is not changed.

      .. versionchanged:: 1.9.5 The vector itself is returned instead of ``None``.

      .. ## Vector2.rotate_ip ##

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

   .. method:: from_polar

      | :sl:`Sets x and y from a polar coordinates tuple.`
      | :sg:`from_polar((r, phi)) -> Vector2`

      Sets x and y from a tuple (r, phi) where r is the radial distance, and
      phi is the azimuthal angle.

      .. versionchanged:: 1.9.5 The vector itself is returned instead of ``None``.

      .. ## Vector2.from_polar ##

   .. ## pygame.math.Vector2 ##

.. class:: Vector3

   | :sl:`a 3-Dimensional Vector`
   | :sg:`Vector3() -> Vector3`
   | :sg:`Vector3(int) -> Vector2`
   | :sg:`Vector3(float) -> Vector2`
   | :sg:`Vector3(Vector3) -> Vector3`
   | :sg:`Vector3(x, y, z) -> Vector3`
   | :sg:`Vector3((x, y, z)) -> Vector3`

   Some general information about the Vector3 class.

   .. method:: dot

      | :sl:`calculates the dot- or scalar-product with the other vector`
      | :sg:`dot(Vector3) -> float`

      .. ## Vector3.dot ##

   .. method:: cross

      | :sl:`calculates the cross- or vector-product`
      | :sg:`cross(Vector3) -> Vector3`

      calculates the cross-product.

      .. ## Vector3.cross ##

   .. method:: magnitude

      | :sl:`returns the Euclidean magnitude of the vector.`
      | :sg:`magnitude() -> float`

      calculates the magnitude of the vector which follows from the
      theorem: ``vec.magnitude()`` == ``math.sqrt(vec.x**2 + vec.y**2 + vec.z**2)``

      .. ## Vector3.magnitude ##

   .. method:: magnitude_squared

      | :sl:`returns the squared Euclidean magnitude of the vector.`
      | :sg:`magnitude_squared() -> float`

      calculates the magnitude of the vector which follows from the
      theorem: ``vec.magnitude_squared()`` == vec.x**2 + vec.y**2 +
      vec.z**2 This is faster than ``vec.magnitude()`` because it avoids the
      square root.

      .. ## Vector3.magnitude_squared ##

   .. method:: length

      | :sl:`returns the Euclidean length of the vector.`
      | :sg:`length() -> float`

      calculates the Euclidean length of the vector which follows from the
      Pythagorean theorem: ``vec.length()`` ==
      ``math.sqrt(vec.x**2 + vec.y**2 + vec.z**2)``

      .. ## Vector3.length ##

   .. method:: length_squared

      | :sl:`returns the squared Euclidean length of the vector.`
      | :sg:`length_squared() -> float`

      calculates the Euclidean length of the vector which follows from the
      Pythagorean theorem: ``vec.length_squared()`` == vec.x**2 + vec.y**2 +
      vec.z**2 This is faster than ``vec.length()`` because it avoids the
      square root.

      .. ## Vector3.length_squared ##

   .. method:: normalize

      | :sl:`returns a vector with the same direction but length 1.`
      | :sg:`normalize() -> Vector3`

      Returns a new vector that has length == 1 and the same direction as self.

      .. ## Vector3.normalize ##

   .. method:: normalize_ip

      | :sl:`normalizes the vector in place so that its length is 1.`
      | :sg:`normalize_ip() -> Vector3`

      Normalizes the vector so that it has length == 1. The direction of the
      vector is not changed.

      .. versionchanged:: 1.9.5 The vector itself is returned instead of ``None``.

      .. ## Vector3.normalize_ip ##

   .. method:: is_normalized

      | :sl:`tests if the vector is normalized i.e. has length == 1.`
      | :sg:`is_normalized() -> Bool`

      Returns True if the vector has length == 1. Otherwise it returns False.

      .. ## Vector3.is_normalized ##

   .. method:: scale_to_length

      | :sl:`scales the vector in place to a given length.`
      | :sg:`scale_to_length(float) -> Vector3`

      Scales the vector in place so that it has the given length. The direction of
      the vector is not changed. You can also scale to length 0. If the vector is
      the zero vector (i.e. has length 0 thus no direction) an
      ZeroDivisionError is raised.

      .. versionchanged:: 1.9.5 The vector itself is returned instead of ``None``.

      .. ## Vector3.scale_to_length ##

   .. method:: reflect

      | :sl:`returns a vector reflected of a given normal.`
      | :sg:`reflect(Vector3) -> Vector3`

      Returns a new vector that points in the direction as if self would bounce
      of a surface characterized by the given surface normal. The length of the
      new vector is the same as self's.

      .. ## Vector3.reflect ##

   .. method:: reflect_ip

      | :sl:`reflect the vector of a given normal in place.`
      | :sg:`reflect_ip(Vector3) -> Vector3`

      Changes the direction of self as if it would have been reflected of a
      surface with the given surface normal.

      .. versionchanged:: 1.9.5 The vector itself is returned instead of ``None``.

      .. ## Vector3.reflect_ip ##

   .. method:: distance_to

      | :sl:`calculates the Euclidean distance to a given vector.`
      | :sg:`distance_to(Vector3) -> float`

      .. ## Vector3.distance_to ##

   .. method:: distance_squared_to

      | :sl:`calculates the squared Euclidean distance to a given vector.`
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

      | :sl:`The next operation will be performed elementwise.`
      | :sg:`elementwise() -> VectorElementwiseProxy`

      Applies the following operation to each element of the vector.

      .. ## Vector3.elementwise ##

   .. method:: rotate

      | :sl:`rotates a vector by a given angle in degrees.`
      | :sg:`rotate(Vector3, float) -> Vector3`

      Returns a vector which has the same length as self but is rotated
      counterclockwise by the given angle in degrees around the given axis.

      .. ## Vector3.rotate ##

   .. method:: rotate_ip

      | :sl:`rotates the vector by a given angle in degrees in place.`
      | :sg:`rotate_ip(Vector3, float) -> Vector3`

      Rotates the vector counterclockwise around the given axis by the given
      angle in degrees. The length of the vector is not changed.

      .. versionchanged:: 1.9.5 The vector itself is returned instead of ``None``.

      .. ## Vector3.rotate_ip ##

   .. method:: rotate_x

      | :sl:`rotates a vector around the x-axis by the angle in degrees.`
      | :sg:`rotate_x(float) -> Vector3`

      Returns a vector which has the same length as self but is rotated
      counterclockwise around the x-axis by the given angle in degrees.

      .. ## Vector3.rotate_x ##

   .. method:: rotate_x_ip

      | :sl:`rotates the vector around the x-axis by the angle in degrees in place.`
      | :sg:`rotate_x_ip(float) -> Vector3`

      Rotates the vector counterclockwise around the x-axis by the given angle
      in degrees. The length of the vector is not changed.

      .. versionchanged:: 1.9.5 The vector itself is returned instead of ``None``.

      .. ## Vector3.rotate_x_ip ##

   .. method:: rotate_y

      | :sl:`rotates a vector around the y-axis by the angle in degrees.`
      | :sg:`rotate_y(float) -> Vector3`

      Returns a vector which has the same length as self but is rotated
      counterclockwise around the y-axis by the given angle in degrees.

      .. ## Vector3.rotate_y ##

   .. method:: rotate_y_ip

      | :sl:`rotates the vector around the y-axis by the angle in degrees in place.`
      | :sg:`rotate_y_ip(float) -> Vector3`

      Rotates the vector counterclockwise around the y-axis by the given angle
      in degrees. The length of the vector is not changed.

      .. versionchanged:: 1.9.5 The vector itself is returned instead of ``None``.

      .. ## Vector3.rotate_y_ip ##

   .. method:: rotate_z

      | :sl:`rotates a vector around the z-axis by the angle in degrees.`
      | :sg:`rotate_z(float) -> Vector3`

      Returns a vector which has the same length as self but is rotated
      counterclockwise around the z-axis by the given angle in degrees.

      .. ## Vector3.rotate_z ##

   .. method:: rotate_z_ip

      | :sl:`rotates the vector around the z-axis by the angle in degrees in place.`
      | :sg:`rotate_z_ip(float) -> Vector3`

      Rotates the vector counterclockwise around the z-axis by the given angle
      in degrees. The length of the vector is not changed.

      .. versionchanged:: 1.9.5 The vector itself is returned instead of ``None``.

      .. ## Vector3.rotate_z_ip ##

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

   .. method:: from_spherical

      | :sl:`Sets x, y and z from a spherical coordinates 3-tuple.`
      | :sg:`from_spherical((r, theta, phi)) -> Vector3`

      Sets x, y and z from a tuple (r, theta, phi) where r is the radial
      distance, theta is the inclination angle and phi is the azimuthal angle.

      .. versionchanged:: 1.9.5 The vector itself is returned instead of ``None``.

      .. ## Vector3.from_spherical ##

   .. ##  ##

   .. ## pygame.math.Vector3 ##


.. function:: enable_swizzling

   | :sl:`globally enables swizzling for vectors.`
   | :sg:`enable_swizzling() -> None`

   DEPRECATED: Not needed anymore. Will be removed in a later version.

   Enables swizzling for all vectors until ``disable_swizzling()`` is called.
   By default swizzling is disabled.

   .. ## pygame.math.enable_swizzling ##

.. function:: disable_swizzling

   | :sl:`globally disables swizzling for vectors.`
   | :sg:`disable_swizzling() -> None`

   DEPRECATED: Not needed anymore. Will be removed in a later version.

   Disables swizzling for all vectors until ``enable_swizzling()`` is called.
   By default swizzling is disabled.

   .. ## pygame.math.disable_swizzling ##

.. ## pygame.math ##
