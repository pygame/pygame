.. include:: ../common.txt

.. highlight:: c

******************************************
  Class Rect API exported by pygame.rect
******************************************

src_c/rect.c
============

This extension module defines Python type :py:class:`pygame.Rect` & :py:class:`pygame.FRect`.

Header file: src_c/include/pygame.h

.. c:type:: pgRectObject

   .. c:member:: SDL_Rect r

   The Pygame rectangle type instance.

.. c:type:: pgFRectObject

   .. c:member:: SDL_FRect r

   The Pygame rectangle type instance.

.. c:var:: PyTypeObject *pgRect_Type

   The Pygame rectangle object type pygame.Rect.

.. c:var:: PyTypeObject *pgRFect_Type

   The Pygame rectangle object type pygame.FRect.

.. c:function:: SDL_Rect pgRect_AsRect(PyObject *obj)

   A macro to access the SDL_Rect field of a :py:class:`pygame.Rect` instance.

.. c:function:: SDL_FRect pgFRect_AsRect(PyObject *obj)

   A macro to access the SDL_FRect field of a :py:class:`pygame.FRect` instance.

.. c:function:: PyObject* pgRect_New(SDL_Rect *r)

   Return a new :py:class:`pygame.Rect` instance from the SDL_Rect *r*.
   On failure, raise a Python exception and return *NULL*.

.. c:function:: PyObject* pgRFect_New(SDL_FRect *r)

   Return a new :py:class:`pygame.FRect` instance from the SDL_FRect *r*.
   On failure, raise a Python exception and return *NULL*.

.. c:function:: PyObject* pgRect_New4(int x, int y, int w, int h)

   Return a new pygame.Rect instance with position (*x*, *y*) and
   size (*w*, *h*).
   On failure raise a Python exception and return *NULL*.

.. c:function:: PyObject* pgFRect_New4(float x, float y, float w, float h)

   Return a new pygame.FRect instance with position (*x*, *y*) and
   size (*w*, *h*).
   On failure raise a Python exception and return *NULL*.

.. c:function:: SDL_Rect* pgRect_FromObject(PyObject *obj, SDL_Rect *temp)

   Translate a Python rectangle representation as a Pygame :c:type:`SDL_Rect`.
   A rectangle can be a length 4 sequence integers (x, y, w, h),
   or a length 2 sequence of position (x, y) and size (w, h),
   or a length 1 tuple containing a rectangle representation,
   or have a method *rect* that returns a rectangle.
   Pass a pointer to a locally declared :c:type:`SDL_Rect` as *temp*.
   Do not rely on this being filled in; use the function's return value instead.
   On success, return a pointer to a :c:type:`SDL_Rect` representation
   of the rectangle, else return *NULL*.
   No Python exceptions are raised.

.. c:function:: SDL_FRect* pgFRect_FromObject(PyObject *obj, SDL_FRect *temp)

   Translate a Python rectangle representation as a Pygame :c:type:`SDL_FRect`.
   A rectangle can be a length 4 sequence floats (x, y, w, h),
   or a length 2 sequence of position (x, y) and size (w, h),
   or a length 1 tuple containing a rectangle representation,
   or have a method *rect* that returns a rectangle.
   Pass a pointer to a locally declared :c:type:`SDL_FRect` as *temp*.
   Do not rely on this being filled in; use the function's return value instead.
   On success, return a pointer to a :c:type:`SDL_FRect` representation
   of the rectangle, else return *NULL*.
   No Python exceptions are raised.

.. c:function:: void pgRect_Normalize(SDL_Rect *rect)

   Normalize the given rect. A rect with a negative size (negative width and/or
   height) will be adjusted to have a positive size.

.. c:function:: void pgFRect_Normalize(SDL_FRect *rect)

   Normalize the given rect. A rect with a negative size (negative width and/or
   height) will be adjusted to have a positive size.
