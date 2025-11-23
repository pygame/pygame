.. include:: ../common.txt

.. highlight:: c

**********************************
  API exported by pygame.display
**********************************

src_c/display.c
===============

This is the :py:mod:`pygame.display` extension module.

Header file: src_c/include/pygame.h


.. c:type:: pgVidInfoObject

   A pygame object that wraps an SDL_VideoInfo struct.
   The object returned by :py:func:`pygame.display.Info()`.

.. c:var:: PyTypeObject *pgVidInfo_Type

   The pgVidInfoObject object Python type.

.. c:function:: SDL_VideoInfo pgVidInfo_AsVidInfo(PyObject *obj)

   Return the SDL_VideoInfo field of *obj*, a :c:data:`pgVidInfo_Type` instance.
   This macro does not check that *obj* is not ``NULL`` or an actual :c:type:`pgVidInfoObject` object.

.. c:function:: PyObject* pgVidInfo_New(SDL_VideoInfo *i)

   Return a new :c:type:`pgVidInfoObject` object for the SDL_VideoInfo *i*.
   On failure, raise a Python exception and return ``NULL``.

.. c:function:: int pgVidInfo_Check(PyObject *x)

   Return true if *x* is a :c:data:`pgVidInfo_Type` instance

   Will return false if *x* is a subclass of :c:data:`pgVidInfo_Type`.
   This macro does not check that *x* is not ``NULL``.
