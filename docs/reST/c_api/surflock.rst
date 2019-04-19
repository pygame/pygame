.. include:: ../common.txt

.. highlight:: c

***********************************
  API exported by pygame.surflock
***********************************

src_c/surflock.c
================

This extension module implements SDL surface locking for the
:py:class:`pygame.Surface` type.

Header file: src_c/include/pygame.h


.. c:type:: pgLifetimeLockObject

   .. c:member:: PyObject *surface

      An SDL locked pygame surface.

   .. c:member:: PyObject *lockobj

      The Python object which owns the lock on the surface.
      This field does not own a reference to the object.

   The lifetime lock type instance.
   A lifetime lock pairs a locked pygame surface with
   the Python object that locked the surface for modification.
   The lock is removed automatically when the lifetime lock instance
   is garbage collected.

.. c:var:: PyTypeObject *pgLifetimeLock_Type

   The pygame internal surflock lifetime lock object type.

.. c:function:: int pgLifetimeLock_Check(PyObject *x)

   Return true if Python object *x* is a :c:data:`pgLifetimeLock_Type` instance,
   false otherwise.
   This will return false on :c:data:`pgLifetimeLock_Type` subclass instances as well.

.. c:function:: void pgSurface_Prep(PyObject *surfobj)

   If *surfobj* is a subsurface, then lock the parent surface with *surfobj*
   the owner of the lock.

.. c:function:: void pgSurface_Unprep(PyObject *surfobj)

   If *surfobj* is a subsurface, then release its lock on the parent surface.

.. c:function:: int pgSurface_Lock(PyObject *surfobj)

   Lock pygame surface *surfobj*, with *surfobj* owning its own lock.

.. c:function:: int pgSurface_LockBy(PyObject *surfobj, PyObject *lockobj)

   Lock pygame surface *surfobj* with Python object *lockobj* the owning
   the lock.

   The surface will keep a weak reference to object *lockobj*,
   and eventually remove the lock on itself if *lockobj* is garbage collected.
   However, it is best if *lockobj* also keep a reference to the locked surface
   and call to :c:func:`pgSurface_UnLockBy` when finished with the surface.

.. c:function:: int pgSurface_UnLock(PyObject *surfobj)

   Remove the pygame surface *surfobj* object's lock on itself.

.. c:function:: int pgSurface_UnLockBy(PyObject *surfobj, PyObject *lockobj)

   Remove the lock on pygame surface *surfobj* owned by Python object *lockobj*.

.. c:function:: PyObject *pgSurface_LockLifetime(PyObject *surfobj, PyObject *lockobj)

   Lock pygame surface *surfobj* for Python object *lockobj* and return a
   new :c:data:`pgLifetimeLock_Type` instance for the lock.

   This function is not called anywhere within pygame.
   It and pgLifetimeLock_Type are candidates for removal.
