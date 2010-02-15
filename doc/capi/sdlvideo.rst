=================
pygame2.sdl.video
=================

The :mod:`pygame2.sdl.video` C API contains fundamental objects and functions
for accessing and manipulating the screen display, image surface objects and
overlay graphics.

Import
------
Include headers::

  pygame2/pgsdl.h

.. cfunction:: int import_pygame2_sdl_video (void)

  Imports the :mod:`pygame2.sdl.video` module. This returns 0 on success
  and -1 on failure.

Basic Types
-----------
.. ctype:: SDLSurfaceLock

  Internally used lock tracking object. It is used in conjunction with
  the :cfunc:`PySDLSurface_AcquireLockObj` function to keep track of the 
  external locking objects for a :ctype:`PySDLSurface`.
  
  .. cmember:: PyObject* SDLSurfaceLock.surface
  
  The :ctype:`PySDLSurface` object that is locked.
  
  .. cmember:: PyObject* SDLSurfaceLock.lockobj
  
  The :ctype:`PyObject` that causes the lock on the :ctype:`PySDLSurface`.

Macros
------
.. cfunction:: ASSERT_VIDEO_INIT (retval)

  Checks, whether the video subsystem was properly initialised. If not,
  this will set a :exc:`PyExc_PyGameError` and return *retval*.

.. cfunction:: ASSERT_VIDEO_SURFACE_SET (retval)

  Checks, whether a display surface was created already using
  :func:`pygame2.sdl.video.set_mode`. If not, this will set a
  :exc:`PyExc_PyGameError` and return *retval*.

Functions
---------
.. cfunction:: int SDLColorFromObj (PyObject *obj, SDL_PixelFormat *format, Uint32 *val)

  Converts the passed object to a 32-bit integer color value matching
  the passed *format* and stores the result in *val*. This returns 1 on
  success and 0 on failure.

PyPixelFormat
-------------
.. ctype:: PyPixelFormat
.. ctype:: PyPixelFormat_Type

The :ctype:`PyPixelFormat` object is a wrapper around the
:ctype:`SDL_PixelFormat` type, which contains format information about surfaces,
such as the bit depth, color mask, etc.

Members
^^^^^^^
.. cmember:: SDL_PixelFormat* PyPixelFormat.format
  
  The SDL_PixelFormat pointer to access the pixel format information.

.. cmember:: int PyPixelFormat.readonly

  A read-only flag that indicates whether the information of the underlying
  SDL_PixelFormat are allowed to be changed.

Functions
^^^^^^^^^^
.. cfunction:: int PyPixelFormat_Check (PyObject *obj)

  Returns true, if the argument is a :ctype:`PyPixelFormat` or a subclass of
  :ctype:`PyPixelFormat`.

.. cfunction:: PyObject* PyPixelFormat_New (void)

  Creates a new, empty and writable :ctype:`PyPixelFormat` object. On
  failure, this returns NULL.

.. cfunction:: PyObject* PyPixelFormat_NewFromSDLPixelFormat (SDL_PixelFormat *format)

  Creates a new, read-only :ctype:`PyPixelFormat` object from the passed
  SDL_PixelFormat. The passed *format* must not be freed during the lifetime
  of the :ctype:`PyPixelFormat` object. On failure, this returns NULL.

.. cfunction:: SDL_PixelFormat* PyPixelFormat_AsPixelFormat (PyObject *obj)

  Macro for accessing the *format* member of the :ctype:`PyPixelFormat`. This
  does not perform any type or argument checks.

PySDLSurface
------------
.. ctype:: PySDLSurface
.. ctype:: PySDLSurface_Type

The PySDLSurface is the most important object type for the
:mod:`pygame2.sdl.video` module. It is used to hold information about the 
2D pixel buffer information of any visible object for the :mod:`pygame2.sdl`
modules.

The :ctype:`PySDLSurface` class inherits from the :ctype:`PySurface` class of
the :mod:`pygame2.base` module.

Members
^^^^^^^
.. cmember:: PySurface PySDLSurface.pysurface

  The base class of the :ctype:`PySDLSurface`.

.. cmember:: SDL_Surface* PySDLSurface.surface

  The SDL_Surface pointer to access the surface information.

.. cmember:: PyObject* PySDLSurface.locklist

  A list of external objects owning a lock on the surface. Never manipulate the
  list directly. Use the :cfunc:`PySDLSurface_AddRefLock` and
  :cfunc:`PySDLSurface_RemoveRefLock` functions instad to acquire or release an
  external lock

.. cmember:: pguint16 PySDLSurface.intlocks

  Counter of internally set locks on the surface. This value is usually
  incremented and decremented by the :meth:`pygame2.sdl.video.Surface.lock` and
  :meth:`pygame2.sdl.video.Surface.unlock` methods and should not manipulated
  directly.

Functions
^^^^^^^^^^
.. cfunction:: SDL_Surface* PySDLSurface_AsSDLSurface (PyObject *obj)

  Macro for accessing the *surface* member of the :ctype:`PySDLSurface`. This
  does not perform any type checks.
  
.. cfunction:: PySurface* PySDLSurface_AsPySurface (PyObject *obj)

  Macro for accessing the *pysurface* member of the :ctype:`PySDLSurface`. This
  does not perform any type or argument checks.

.. cfunction:: int PySDLSurface_Check (PyObject *obj)
  
  Returns true, if the argument is a :ctype:`PySDLSurface` or a subclass of
  :ctype:`PySDLSurface`.

.. cfunction:: PyObject* PySDLSurface_New (int width, int height)

  Creates a new :ctype:`PySDLSurface` with the specified *width* and *height*.
  On failure, this returns NULL.

.. cfunction:: PyObject* PySDLSurface_NewFromSDLSurface (SDL_Surface *surface)

  Creates a new :ctype:`PySDLSurface` from an existing :ctype:`SDL_Surface`.
  The passed *surface* must not be freed during the lifetime of the
  :ctype:`PySDLSurface` object. On failure, this returns NULL.
  
.. cfunction:: PyObject* PySDLSurface_Copy (PyObject *obj)

  Creates an exact copy of the passed :ctype:`PySDLSurface`. This creates
  a new :ctype:`PySDLSurface` and copies the information of *obj* to it (except
  for the locks). On failure, this returns NULL.

.. cfunction:: int PySDLSurface_AddRefLock (PyObject *surface, PyObject *lockobj)

  Adds a lock to the passed :ctype:`PySDLSurface`, which will be hold by
  *lockobj*. This will not increase *lockobj*'s refcount, but use weak
  references instead. If *lockobj* is garbage-collected any time later,
  the lock on the :ctype:`PySDLSurface` will be removed automatically on the
  next invocation of :cfunc:`PySDLSurface_RemoveRefLock`. This returns 1 on
  success and 0 on failure.
  
.. cfunction:: int PySDLSurface_RemoveRefLock (PyObject *surface, PyObject *lockobj)

  Removes a lock from the passed :ctype:`PySDLSurface`. *lockobj* denotes the
  object holding the lock. It also removes any other outstanding
  garbage-collected lock references. This returns 1 on success and 0 on failure.

.. cfunction:: PyObject* PySDLSurface_AcquireLockObj (PyObject *surface, PyObject *lockobj)

  Acquires a :ctype:`PyCObject` that keeps a lock on the passed
  :ctype:`PySDLSurface`. *lockobj* denotes the object holding the lock. If
  the return value is garbage-collected, the lock on the :ctype:`PySDLSurface`
  will be removed immediately.

PyOverlay
---------
.. ctype:: PyOverlay
.. ctype:: PyOverlay_Type

PyOverlay is a low-level overlay graphics class for :ctype:`PySDLSurface`
objects. It support direct operations on the YUV overlay buffers of the
graphics objects.

Members
^^^^^^^
.. cmember:: SDL_Overlay* PyOverlay.overlay

  The SDL_Overlay pointer to access the overlay information.

.. cmember:: PyObject* PyOverlay.surface

  The :ctype:`PySDLSurface` the :ctype:`PyOverlay` was created for.
  
.. cmember:: PyObject* PyOverlay.locklist

  A list of external objects owning a lock on the overlay. Never manipulate the
  list directly. Use the :cfunc:`PyOverlay_AddRefLock` and
  :cfunc:`PyOverlay_RemoveRefLock` functions instad to acquire or release an
  external lock

Functions
^^^^^^^^^^
.. cfunction:: SDL_Overlay* PyOverlay_AsOverlay (PyObject *obj)

  Macro for accessing the *overlay* member of the :ctype:`PyOverlay`. This
  does not perform any type checks.

.. cfunction:: PyObject* PyOverlay_New (PyObject *obj, int width, int height, Uint32 format)

  Creates a new :ctype:`PyOverlay` for the passed :ctype:`PySDLSurface` *obj*.
  *width* and *height* specify the width and height of the :ctype:`PyOverlay`,
  which may or may not exceed the size of the :ctype:`PySDLSurface`.
  The *format* argument specifies the YUV overlay type to use.

  +--------------+--------------------------------+
  | YV12_OVERLAY | Planar mode: Y + V + U         |
  +--------------+--------------------------------+
  | IYUV_OVERLAY | Planar mode: Y + U + V         |
  +--------------+--------------------------------+
  | YUY2_OVERLAY | Packed mode: Y0 + U0 + Y1 + V0 |
  +--------------+--------------------------------+
  | UYVY_OVERLAY | Packed mode: U0 + Y0 + V0 + Y1 |
  +--------------+--------------------------------+
  | YVYU_OVERLAY | Packed mode: Y0 + V0 + Y1 + U0 |
  +--------------+--------------------------------+
  
  On failure, this returns NULL.

.. cfunction:: int PyOverlay_AddRefLock (PyObject *overlay, PyObject *lockobj)

  Adds a lock to the passed :ctype:`PyOverlay`, which will be hold by
  *lockobj*. This will not increase *lockobj*'s refcount, but use weak
  references instead. If *lockobj* is garbage-collected any time later, the
  lock on the :ctype:`PyOverlay` will be removed automatically on the next
  invocation of :cfunc:`PyOverlay_RemoveRefLock`. This returns 1 on success and
  0 on failure.
  
.. cfunction:: int PyOverlay_RemoveRefLock (PyObject *overlay, PyObject *lockobj)
  
  Removes a lock from the passed :ctype:`PyOverlay`. *lockobj* denotes the
  object holding the lock. It also removes any other outstanding
  garbage-collected lock references. This returns 1 on success and 0 on failure.
