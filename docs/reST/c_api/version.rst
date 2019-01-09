.. include:: ../common.txt

.. highlight:: c

******************************************
  API exported by pygame.version
******************************************

src_py/version.py
=================

Header file: src_c/pygame.h

Version information can be retrieved at compile-time using these macros.

.. versionadded:: 1.9.5

.. c:macro:: PG_MAJOR_VERSION

.. c:macro:: PG_MINOR_VERSION

.. c:macro:: PG_PATCH_VERSION

.. c:function:: PG_VERSIONNUM(MAJOR, MINOR, PATCH)

   Returns an integer representing the given version.

.. c:function:: PG_VERSION_ATLEAST(MAJOR, MINOR, PATCH)

   Returns true if the current version is at least equal
   to the specified version.
