.. include:: ../common.txt

.. highlight:: c

******************************************
  Class Rect API exported by pygame.rect
******************************************

src/rect.c
==========

This extension module defines Python type :py:class:`pygame.Rect`.

Header file: src/pygame.h

.. c:type:: GAME_Rect

   .. c:member:: int x

      Horizontal position of the top-left corner.

   .. c:member:: int y

      Vertical position of the top-left corner.

   .. c:member:: int w

      Width.

   .. c:member:: int h

      Height.

   A rectangle at (*x*, *y*) and of size (*w*, *h*).

   SDL 2: equivalent to SDL_Rect.

.. c:type:: pgRectObject

   .. c:member:: GAME_Rect r

   The Pygame rectangle type instance.

.. c:var:: PyTypeObject *pgRect_Type

   The Pygame rectangle object type pygame.Rect.

.. c:function:: GAME_Rect pgRect_AsRect(PyObject *obj)

   A macro to access the GAME_Rect field of a :py:class:`pygame.Rect` instance.

.. c:function:: PyObject* pgRect_New(SDL_Rect *r)

   Return a new :py:class:`pygame.Rect` instance from the SDL_Rect *r*.
   On failure, raise a Python exception and return *NULL*.

.. c:function:: PyObject* pgRect_New4(int x, int y, int w, int h)

   Return a new pygame.Rect instance with position (*x*, *y*) and
   size (*w*, *h*).
   On failure raise a Python exception and return *NULL*.

.. c:function:: GAME_Rect* pgRect_FromObject(PyObject *obj, GAME_Rect *temp)

   Translate a Python rectangle representation as a Pygame :c:type:`GAME_Rect`.
   A rectangle can be a length 4 sequence integers (x, y, w, h),
   or a length 2 sequence of position (x, y) and size (w, h),
   or a length 1 tuple containing a rectangle representation,
   or have a method *rect* that returns a rectangle.
   Pass a pointer to a locally declared c:type:`GAME_Rect` as *temp*.
   Do not rely on this being filled in; use the function's return value instead.
   On success return a pointer to a :c:type:`GAME_Rect` representation
   of the rectangle.
   One failure may raise a Python exception before returning *NULL*.
