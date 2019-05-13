.. include:: ../common.txt

.. highlight:: c

***********************************
  API exported by pygame.rwobject
***********************************

src_c/rwobject.c
================

This extension module implements functions for wrapping a Python file like
object in a :c:type:`SDL_RWops` struct for SDL file access.

Header file: src_c/include/pygame.h


.. c:function:: SDL_RWops* pgRWops_FromObject(PyObject *obj)

   Return a SDL_RWops struct filled to access *obj*.
   If *obj* is a string then let SDL open the file it names.
   Otherwise, if *obj* is a Python file-like object then use its ``read``, ``write``,
   ``seek``, ``tell``, and ``close`` methods. If threads are available,
   the Python GIL is acquired before calling any of the *obj* methods.
   On error raise a Python exception and return ``NULL``.

.. c:function:: SDL_RWops* pgRWops_FromFileObject(PyObject *obj)

   Return a SDL_RWops struct filled to access the Python file-like object *obj*.
   Uses its ``read``, ``write``, ``seek``, ``tell``, and ``close`` methods.
   If threads are available, the Python GIL is acquired before calling any of the *obj* methods.
   On error raise a Python exception and return ``NULL``.

.. c:function:: int pgRWops_CheckObject(SDL_RWops *rw)

   Return true if *rw* is a Python file-like object wrapper returned by :c:func:`pgRWops_FromObject`
   or :c:func:`pgRWops_FromFileObject`.

.. c:function:: int pgRWops_ReleaseObject(SDL_RWops *context)

   Free a SDL_RWops struct. If it is attached to a Python file-like object, decrement its
   refcount. Otherwise, close the file handle.
   Return 0 on success. On error, raise a Python exception and return a negative value.

.. c:function:: PyObject* pg_EncodeFilePath(PyObject *obj, PyObject *eclass)

   Return the file path *obj* as a byte string properly encoded for the OS.
   Null bytes are forbidden in the encoded file path.
   On error raise a Python exception and return ``NULL``,
   using *eclass* as the exception type if it is not ``NULL``.
   If *obj* is ``NULL`` assume an exception was already raised and pass it on.

.. c:function:: PyObject* pg_EncodeString(PyObject *obj, const char *encoding, const char *errors, PyObject *eclass)

   Return string *obj* as an encoded byte string.
   The C string arguments *encoding* and *errors* are the same as for
   :c:func:`PyUnicode_AsEncodedString`.
   On error raise a Python exception and return ``NULL``,
   using *eclass* as the exception type if it is not ``NULL``.
   If *obj* is ``NULL`` assume an exception was already raised and pass it on.