.. include:: ../common.txt

.. highlight:: c

********************************
  API exported by pygame.cdrom
********************************

src_c/cdrom.c
=============

The :py:mod:`pygame.cdrom` extension module. Only available for SDL 1.

Header file: src_c/pygame.h


.. c:type:: pgCDObject

   The :py:class:`pygame.cdrom.CD` instance C struct.

.. c:var:: PyTypeObject pgCD_Type

   The :py:class:`pygame.cdrom.CD` Python type.

.. c:function:: PyObject* pgCD_New(int id)

   Return a new :py:class:`pygame.cdrom.CD` instance for CD drive *id*.
   On error raise a Python exception and return ``NULL``.

.. c:function:: int pgCD_Check(PyObject *x)

   Return true if *x* is a :py:class:`pygame.cdrom.CD` instance.
   Will return false for a subclass of :py:class:`CD`.
   This is a macro. No check is made that *x* is not ``NULL``.

.. c:function:: int pgCD_AsID(PyObject *x)

   Return the CD identifier associated with the :py:class:`pygame.cdrom.CD`
   instance *x*.
   This is a macro. No check is made that *x* is a :py:class:`pygame.cdrom.CD`
   instance or is not ``NULL``.
