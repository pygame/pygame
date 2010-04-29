============
pygame2.math
============
The :mod:`pygame2.math` C API contains various objects and functions for
math and vector operations.

Import
------
Include headers::

  pygame2/pgmath.h

.. cfunction:: int import_pygame2_math (void)

  Imports the :mod:`pygame2.math` module. This returns 0 on success and
  -1 on failure.

Macros
------
.. data:: VEC_EPSILON

  A small fractional value to come around rounding issues on floating
  point calculations. VEC_EPSILON is usually used to soothe rounding
  differences and thus to allow the system to return an appropriate
  state, with a small tradeoff for the exactness.

  **Example:** ::

     (0.0000000000001 - 0.0000000000009) == 0.0

  Given that VEC_EPSILON is large enough, the above calculation will be
  true. To get over rounding issues on vector operations with small
  fractional parts or to make them more exact, the concrete epsilon
  value can be adjusted on a per :class:`PyVector` basis.

Functions
---------
.. cfunction:: double* VectorCoordsFromObj (PyObject *obj, Py_ssize_t *size)

  Tries to retrieve as many double values as possible from the passed *obj.
  If *obj* is a :ctype:`PyVector`, a copy of all its elements is returned.
  Otherwise the method treats *obj* as a sequence of float values. The total
  amount of doubles returned will be stored in *size*. The caller has to free
  the return value using :cfunc:`PyMem_Free`. This returns 1 on success and 0
  on failure.

PyVector
--------
.. ctype:: PyVector
.. ctype:: PyVector_Type

The PyVector object is a generic vector implementation, which can deal with
any dimension size. Specialized (and in some terms faster) implementations
for 2 and 3 dimensions can be found in the :ctype:`PyVector2` and
:ctype:`PyVector3` implementations.

Members
^^^^^^^
.. cmember:: double* PyVector.coords

  The vector coordinates. It holds up to *dim* values.
  
.. cmember:: Py_ssize_t PyVector.dim

  The number of dimensions (coordinate values), the PyVector can hold.

.. cmember:: double PyVector.epsilon

  The fractional delta to use for soothing round differences for
  floating point operations on the PyVector. see :cdata:`VEC_EPSILON`
  for more details.

Functions
^^^^^^^^^
.. cfunction:: int PyVector_Check (PyObject *obj)

  Returns true, if the argument is a :ctype:`PyVector` or a subclass of
  :ctype:`PyVector`.

.. cfunction:: PyObject* PyVector_New (Py_ssize_t dims)

  Creates a new :ctype:`PyVector` object with the given number of dimensions.
  On failure, this returns NULL.

.. cfunction:: PyObject* PyVector_NewFromSeq (PyObject *obj)

  Creates a new :ctype:`PyVector` object from the passed *obj*. *obj* is treated
  as a sequence of float values (or :ctype:`PyVector`). On failure, this
  returns NULL.

.. cfunction:: PyVector_NewSpecialized (Py_ssize_t dims)

  This behaves like :cfunc:`PyVector_New`, but creates specialized
  :ctype:`PyVector2` or :ctype:`PyVector3` instances for a *dims* value of 2 or
  3. On failure, this returns NULL.

PyVector2
---------
.. ctype:: PyVector2
.. ctype:: PyVector2_Type

A specialized :ctype:`PyVector` class limited to two dimensions.

Members
^^^^^^^
.. cmember:: PyVector PyVector2.vector

  The parent :ctype:`PyVector` class the PyVector2 inherits from.

Functions
^^^^^^^^^
.. cfunction:: int PyVector2_Check (PyObject *obj)

  Returns true, if the argument is a :ctype:`PyVector2` or a subclass of
  :ctype:`PyVector2`.

.. cfunction:: PyObject* PyVector2_New (double x, double y)

  Creates a new :ctype:`PyVector2` object with the specified *x* and *y*
  coordinates. On failure, this returns NULL.

PyVector3
---------
.. ctype:: PyVector3
.. ctype:: PyVector3_Type

A specialized :ctype:`PyVector` class limited to three dimensions.

Members
^^^^^^^
.. cmember:: PyVector PyVector3.vector

  The parent :ctype:`PyVector` class the PyVector3 inherits from.

Functions
^^^^^^^^^
.. cfunction:: int PyVector3_Check (PyObject *obj)

  Returns true, if the argument is a :ctype:`PyVector3` or a subclass of
  :ctype:`PyVector3`.

.. cfunction:: PyObject* PyVector3_New (double x, double y, double z)

  Creates a new :ctype:`PyVector3` object with the specified *x*, *y* and *z*
  coordinates. On failure, this returns NULL.
