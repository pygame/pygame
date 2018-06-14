.. include:: ../common.txt

.. highlight:: c

***********************************
  API exported by pygame.rwobject
***********************************

src/rwobject.c
==============

This extension module implements functions for wrapping a Python file like
object in a :c:type:`SDL_RWops` struct for SDL file access.

Header file: src/pygame.h


.. c:function:: SDL_RWops* pgRWopsFromObject(PyObject *obj)

   Return a SDL_RWops struct filled to access *obj*.
   If *obj* is a string then let SDL open the file it names.
   Otherwise, if *obj* is a Python file-like object then use its ``read``, ``write``,
   ``seek``, ``tell``, and ``close`` methods.
   On error raise a Python exception and return ``NULL``.

.. c:function:: int pgRWopsCheckObject(SDL_RWops *rw)

   Return true if *rw* is a Python file-like object wrapper returned by :c:func:`pgRWopsFromObject`.

.. c:function:: SDL_RWops* pgRWopsFromFileObjectThreaded(PyObject *obj)

   Return a SDL_RWops struct filled to access the Python file-like object *obj*
   in a thread-safe manner.
   The Python GIL is aquired before calling any of the *obj* methods
   ``read``, ``write``, ``seek``, ``tell``, or ``close``.
   On error raise a Python exception and return ``NULL``.

.. c:function:: int pgRWopsCheckObjectThreaded(SDL_RWops *rw)

   Return true if *rw* is a Python file-like object wrapper returned by
   :c:func:`pgRWopsFromFileObjectThreaded`.

.. c:function:: PyObject* pgRWopsEncodeFilePath(PyObject *obj, PyObject *eclass)

   Return the file path *obj* as a byte string properly encoded for the OS.
   Null bytes are forbidden in the encoded file path.
   On error raise a Python exception and return ``NULL``,
   using *eclass* as the exception type if it is not ``NULL``.
   If *obj* is ``NULL`` assume an exception was already raised and pass it on.

.. c:function:: PyObject* pgRWopsEncodeString(PyObject *obj, const char *encoding, const char *errors, PyObject *eclass)

   Return string *obj* as an encoded byte string.
   The C string arguments *encoding* and *errors* are the same as for
   :c:func:`PyUnicode_AsEncodedString`.
   On error raise a Python exception and return ``NULL``,
   using *eclass* as the exception type if it is not ``NULL``.
   If *obj* is ``NULL`` assume an exception was already raised and pass it on.

.. c:function:: SDL_RWops* pgRWopsFromFileObject(PyObject *obj)

   Return a SDL_RWops struct filled to access the Python file-like object *obj*.
   Use its ``read``, ``write``, ``seek``, ``tell``, and ``close`` methods.
   On error raise a Python exception and return ``NULL``.
