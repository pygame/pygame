.. include:: ../common.txt

.. highlight:: c

************************************************
  Class Surface API exported by pygame.surface
************************************************

src_c/surface.c
===============

This extension module defines Python type :py:class:`pygame.Surface`.

Header file: src_c/include/pygame.h


.. c:type:: pgSurfaceObject

   A :py:class:`pygame.Surface` instance.

.. c:var:: PyTypeObject *pgSurface_Type

   The :py:class:`pygame.Surface` Python type.

.. c:function:: int pgSurface_Check(PyObject *x)

   Return true if *x* is a :py:class:`pygame.Surface` instance

   Will return false if *x* is a subclass of `Surface`.
   This is a macro. No check is made that *x* is not *NULL*.

.. c:function:: pgSurfaceObject* pgSurface_New(SDL_Surface *s)

   Return a new new pygame surface instance for SDL surface *s*.
   Return *NULL* on error.

.. c:function:: SDL_Surface* pgSurface_AsSurface(PyObject *x)

   Return a pointer the SDL surface represented by the pygame Surface instance
   *x*.

   This is a macro. Argument *x* is assumed to be a Surface, or subclass of
   Surface, instance.

.. c:function:: int pgSurface_Blit(PyObject *dstobj, PyObject *srcobj, SDL_Rect *dstrect, SDL_Rect *srcrect, int the_args)

   Blit the *srcrect* portion of Surface *srcobj* onto Surface *dstobj* at *srcobj*

   Argument *the_args* indicates the type of blit to perform:
   Normal blit (``0``), ``PYGAME_BLEND_ADD``, ``PYGAME_BLEND_SUB``,
   ``PYGAME_BLEND_SUB``, ``PYGAME_BLEND_MULT``, ``PYGAME_BLEND_MIN``,
   ``PYGAME_BLEND_MAX``, ``PYGAME_BLEND_RGBA_ADD``, ``PYGAME_BLEND_RGBA_SUB``,
   ``PYGAME_BLEND_RGBA_MULT``, ``PYGAME_BLEND_RGBA_MIN``,
   ``PYGAME_BLEND_RGBA_MAX``, ``PYGAME_BLEND_ALPHA_SDL2`` and ``PYGAME_BLEND_PREMULTIPLIED``.
   Argument *dstrect* is updated to the actual area on *dstobj* affected
   by the blit.

   The C version of the :py:meth:`pygame.Surface.blit` method.
   Return ``1`` on success, ``0`` on an exception.
