/* Auto generated file: with makeref.py .  Docs go in src/ *.doc . */
#define DOC_PYGAMEMATH "pygame module for vector classes"

#define DOC_PYGAMEMATHENABLESWIZZLING "enable_swizzling() -> None\nglobally enables swizzling for vectors."

#define DOC_PYGAMEMATHDISABLESWIZZLING "disable_swizzling() -> None\nglobally disables swizzling for vectors."

#define DOC_PYGAMEMATHVECTOR2 "Vector2() -> Vector2\nVector2(Vector2) -> Vector2\nVector2(x, y) -> Vector2\nVector2((x, y)) -> Vector2\na 2-Dimensional Vector"

#define DOC_VECTOR2X "x() -> float\nreturn the first component of the vector"

#define DOC_VECTOR2Y "y() -> float\nreturn the second component of the vector"

#define DOC_VECTOR2DOT "dot(Vector2) -> float\ncalculates the dot- or scalar-product with the other vector"

#define DOC_VECTOR2CROSS "cross(Vector2) -> float\ncalculates the cross- or vector-product"

#define DOC_VECTOR2LENGTH "length() -> float\nreturns the euclidic length of the vector."

#define DOC_VECTOR2LENGTHSQUARED "length_squared() -> float\nreturns the squared euclidic length of the vector."

#define DOC_VECTOR2NORMALIZE "normalize() -> Vector2\nreturns a vector with the same direction but length 1."

#define DOC_VECTOR2ISNORMALIZED "is_normalized() -> Bool\ntests if the vector is normalized i.e. has length == 1."

#define DOC_VECTOR2SCALETOLENGTH "scale_to_length(float) -> Vector2\nreturns a vector in the direction of self and magnitude length."

#define DOC_VECTOR2REFLECT "reflect(Vector2) -> Vector2\nreturns a vector reflected of a given normal."

#define DOC_VECTOR2DISTANCETO "distance_to(Vector2) -> float\ncalculates the euclidic distance to a given vector."

#define DOC_VECTOR2DISTANCESQUAREDTO "distance_squared_to(Vector2) -> float\ncalculates the squared euclidic distance to a given vector."

#define DOC_VECTOR2LERP "lerp(Vector2, float) -> Vector2\nreturns a linear interpolation to the given vector."

#define DOC_VECTOR2SLERP "slerp(Vector2, float) -> Vector2\nreturns a spherical interpolation to the given vector."

#define DOC_VECTOR2ELEMENTWISE "elementwise() -> VectorElementwizeProxy\nThe next operation will be performed elementwize."

#define DOC_VECTOR2ROTATE "rotate(float) -> Vector2\nrotates a vector by a given angle in degrees."

#define DOC_VECTOR2ANGLETO "angle_to(Vector2) -> float\ncalculates the angle to a given vector in degrees."

#define DOC_VECTOR2ASPOLAR "as_polar() -> (r, phi)\nreturns a tuple with radial distance and azimuthal angle."

#define DOC_VECTOR2FROMPOLAR "from_polar((r, phi)) -> Vector2\nCreates a new Vector from a radius and an angle."

#define DOC_PYGAMEMATHVECTOR3 "Vector3() -> Vector3\nVector3(Vector3) -> Vector3\nVector3(x, y, z) -> Vector3\nVector3((x, y, z)) -> Vector3\na 3-Dimensional Vector"

#define DOC_VECTOR3X "x() -> float\nreturn the first component of the vector"

#define DOC_VECTOR3Y "y() -> float\nreturn the second component of the vector"

#define DOC_VECTOR3Z "z() -> float\nreturn the third component of the vector"

#define DOC_VECTOR3DOT "dot(Vector3) -> float\ncalculates the dot- or scalar-product with the other vector"

#define DOC_VECTOR3CROSS "cross(Vector3) -> float\ncalculates the cross- or vector-product"

#define DOC_VECTOR3LENGTH "length() -> float\nreturns the euclidic length of the vector."

#define DOC_VECTOR3LENGTHSQUARED "length_squared() -> float\nreturns the squared euclidic length of the vector."

#define DOC_VECTOR3NORMALIZE "normalize() -> Vector3\nreturns a vector with the same direction but length 1."

#define DOC_VECTOR3ISNORMALIZED "is_normalized() -> Bool\ntests if the vector is normalized i.e. has length == 1."

#define DOC_VECTOR3SCALETOLENGTH "scale_to_length(float) -> None\nscales the vector to a given length."

#define DOC_VECTOR3REFLECT "reflect(Vector3) -> Vector3\nreturns a vector reflected of a given normal."

#define DOC_VECTOR3DISTANCETO "distance_to(Vector3) -> float\ncalculates the euclidic distance to a given vector."

#define DOC_VECTOR3DISTANCESQUAREDTO "distance_squared_to(Vector3) -> float\ncalculates the squared euclidic distance to a given vector."

#define DOC_VECTOR3LERP "lerp(Vector3, float) -> Vector3\nreturns a linear interpolation to the given vector."

#define DOC_VECTOR3SLERP "slerp(Vector3, float) -> Vector3\nreturns a spherical interpolation to the given vector."

#define DOC_VECTOR3ELEMENTWISE "elementwise() -> VectorElementwizeProxy\nThe next operation will be performed elementwize."

#define DOC_VECTOR3ROTATE "rotate(Vector3, float) -> Vector3\nrotates a vector by a given angle in degrees."

#define DOC_VECTOR3ROTATEX "rotate_x(float) -> Vector3\nrotates a vector around the x-axis by the angle in degrees."

#define DOC_VECTOR3ROTATEY "rotate_y(float) -> Vector3\nrotates a vector around the y-axis by the angle in degrees."

#define DOC_VECTOR3ROTATEZ "rotate_z(float) -> Vector3\nrotates a vector around the z-axis by the angle in degrees."

#define DOC_VECTOR3ANGLETO "angle_to(Vector3) -> float\ncalculates the angle to a given vector in degrees."

#define DOC_VECTOR3ASSPHERICAL "as_spherical() -> (r, theta, phi)\nreturns a tuple with radial distance, inclination and azimuthal angle."

#define DOC_VECTOR3FROMSPHERICAL "from_spherical((r, theta, phi)) -> Vector3\nCreate a new Vector from a spherical coordinates 3-tuple."



/* Docs in a comment... slightly easier to read. */

/*

pygame.math
pygame module for vector classes

pygame.math.enable_swizzling
 enable_swizzling() -> None
globally enables swizzling for vectors.

pygame.math.disable_swizzling
 disable_swizzling() -> None
globally disables swizzling for vectors.

pygame.math.Vector2
 Vector2() -> Vector2
 Vector2(Vector2) -> Vector2
 Vector2(x, y) -> Vector2
 Vector2((x, y)) -> Vector2
a 2-Dimensional Vector

pygame.math.Vector2.x
 x() -> float
return the first component of the vector

pygame.math.Vector2.y
 y() -> float
return the second component of the vector

pygame.math.Vector2.dot
 dot(Vector2) -> float
calculates the dot- or scalar-product with the other vector

pygame.math.Vector2.cross
 cross(Vector2) -> float
calculates the cross- or vector-product

pygame.math.Vector2.length
 length() -> float
returns the euclidic length of the vector.

pygame.math.Vector2.length_squared
 length_squared() -> float
returns the squared euclidic length of the vector.

pygame.math.Vector2.normalize
 normalize() -> Vector2
returns a vector with the same direction but length 1.

pygame.math.Vector2.is_normalized
 is_normalized() -> Bool
tests if the vector is normalized i.e. has length == 1.

pygame.math.Vector2.scale_to_length
 scale_to_length(float) -> Vector2
returns a vector in the direction of self and magnitude length.

pygame.math.Vector2.reflect
 reflect(Vector2) -> Vector2
returns a vector reflected of a given normal.

pygame.math.Vector2.distance_to
 distance_to(Vector2) -> float
calculates the euclidic distance to a given vector.

pygame.math.Vector2.distance_squared_to
 distance_squared_to(Vector2) -> float
calculates the squared euclidic distance to a given vector.

pygame.math.Vector2.lerp
 lerp(Vector2, float) -> Vector2
returns a linear interpolation to the given vector.

pygame.math.Vector2.slerp
 slerp(Vector2, float) -> Vector2
returns a spherical interpolation to the given vector.

pygame.math.Vector2.elementwise
 elementwise() -> VectorElementwizeProxy
The next operation will be performed elementwize.

pygame.math.Vector2.rotate
 rotate(float) -> Vector2
rotates a vector by a given angle in degrees.

pygame.math.Vector2.angle_to
 angle_to(Vector2) -> float
calculates the angle to a given vector in degrees.

pygame.math.Vector2.as_polar
 as_polar() -> (r, phi)
returns a tuple with radial distance and azimuthal angle.

pygame.math.Vector2.from_polar
 from_polar((r, phi)) -> Vector2
Creates a new Vector from a radius and an angle.

pygame.math.Vector3
 Vector3() -> Vector3
 Vector3(Vector3) -> Vector3
 Vector3(x, y, z) -> Vector3
 Vector3((x, y, z)) -> Vector3
a 3-Dimensional Vector

pygame.math.Vector3.x
 x() -> float
return the first component of the vector

pygame.math.Vector3.y
 y() -> float
return the second component of the vector

pygame.math.Vector3.z
 z() -> float
return the third component of the vector

pygame.math.Vector3.dot
 dot(Vector3) -> float
calculates the dot- or scalar-product with the other vector

pygame.math.Vector3.cross
 cross(Vector3) -> float
calculates the cross- or vector-product

pygame.math.Vector3.length
 length() -> float
returns the euclidic length of the vector.

pygame.math.Vector3.length_squared
 length_squared() -> float
returns the squared euclidic length of the vector.

pygame.math.Vector3.normalize
 normalize() -> Vector3
returns a vector with the same direction but length 1.

pygame.math.Vector3.is_normalized
 is_normalized() -> Bool
tests if the vector is normalized i.e. has length == 1.

pygame.math.Vector3.scale_to_length
 scale_to_length(float) -> None
scales the vector to a given length.

pygame.math.Vector3.reflect
 reflect(Vector3) -> Vector3
returns a vector reflected of a given normal.

pygame.math.Vector3.distance_to
 distance_to(Vector3) -> float
calculates the euclidic distance to a given vector.

pygame.math.Vector3.distance_squared_to
 distance_squared_to(Vector3) -> float
calculates the squared euclidic distance to a given vector.

pygame.math.Vector3.lerp
 lerp(Vector3, float) -> Vector3
returns a linear interpolation to the given vector.

pygame.math.Vector3.slerp
 slerp(Vector3, float) -> Vector3
returns a spherical interpolation to the given vector.

pygame.math.Vector3.elementwise
 elementwise() -> VectorElementwizeProxy
The next operation will be performed elementwize.

pygame.math.Vector3.rotate
 rotate(Vector3, float) -> Vector3
rotates a vector by a given angle in degrees.

pygame.math.Vector3.rotate_x
 rotate_x(float) -> Vector3
rotates a vector around the x-axis by the angle in degrees.

pygame.math.Vector3.rotate_y
 rotate_y(float) -> Vector3
rotates a vector around the y-axis by the angle in degrees.

pygame.math.Vector3.rotate_z
 rotate_z(float) -> Vector3
rotates a vector around the z-axis by the angle in degrees.

pygame.math.Vector3.angle_to
 angle_to(Vector3) -> float
calculates the angle to a given vector in degrees.

pygame.math.Vector3.as_spherical
 as_spherical() -> (r, theta, phi)
returns a tuple with radial distance, inclination and azimuthal angle.

pygame.math.Vector3.from_spherical
 from_spherical((r, theta, phi)) -> Vector3
Create a new Vector from a spherical coordinates 3-tuple.

*/