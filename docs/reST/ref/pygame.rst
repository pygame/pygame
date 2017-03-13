.. include:: common.txt

:mod:`pygame`
=============

.. module:: pygame
   :synopsis: the top level pygame package

| :sl:`the top level pygame package`

The pygame package represents the top-level package for others to use. Pygame
itself is broken into many submodules, but this does not affect programs that
use pygame.

As a convenience, most of the top-level variables in pygame have been placed
inside a module named 'pygame.locals'. This is meant to be used with 'from
:mod:`pygame.locals` import \*', in addition to 'import pygame'.

When you 'import pygame' all available pygame submodules are automatically
imported. Be aware that some of the pygame modules are considered "optional",
and may not be available. In that case, pygame will provide a placeholder
object instead of the module, which can be used to test for availability.

.. function:: init

   | :sl:`initialize all imported pygame modules`
   | :sg:`init() -> (numpass, numfail)`

   Initialize all imported pygame modules. No exceptions will be raised if a
   module fails, but the total number if successful and failed inits will be
   returned as a tuple. You can always initialize individual modules manually,
   but :func:`pygame.init` is a convenient way to get everything started. The
   ``init()`` functions for individual modules will raise exceptions when they
   fail.

   You may want to initialize the different modules separately to speed up your
   program or to not use things your game does not.

   It is safe to call this ``init()`` more than once: repeated calls will have
   no effect. This is true even if you have ``pygame.quit()`` all the modules.

   .. ## pygame.init ##

.. function:: quit

   | :sl:`uninitialize all pygame modules`
   | :sg:`quit() -> None`

   Uninitialize all pygame modules that have previously been initialized. When
   the Python interpreter shuts down, this method is called regardless, so your
   program should not need it, except when it wants to terminate its pygame
   resources and continue. It is safe to call this function more than once:
   repeated calls have no effect.

   Note, that :func:`pygame.quit` will not exit your program. Consider letting
   your program end in the same way a normal python program will end.

   .. ## pygame.quit ##

.. exception:: error

   | :sl:`standard pygame exception`
   | :sg:`raise pygame.error(message)`

   This exception is raised whenever a pygame or ``SDL`` operation fails. You
   can catch any anticipated problems and deal with the error. The exception is
   always raised with a descriptive message about the problem.

   Derived from the RuntimeError exception, which can also be used to catch
   these raised errors.

   .. ## pygame.error ##

.. function:: get_error

   | :sl:`get the current error message`
   | :sg:`get_error() -> errorstr`

   ``SDL`` maintains an internal error message. This message will usually be
   given to you when :func:`pygame.error` is raised. You will rarely need to
   call this function.

   .. ## pygame.get_error ##

.. function:: set_error

   | :sl:`set the current error message`
   | :sg:`set_error(error_msg) -> None`

   ``SDL`` maintains an internal error message. This message will usually be
   given to you when :func:`pygame.error` is raised. You will rarely need to
   call this function.

   .. ## pygame.set_error ##

.. function:: get_sdl_version

   | :sl:`get the version number of SDL`
   | :sg:`get_sdl_version() -> major, minor, patch`

   Returns the three version numbers of the ``SDL`` library. This version is
   built at compile time. It can be used to detect which features may not be
   available through pygame.

   get_sdl_version is new in pygame 1.7.0

   .. ## pygame.get_sdl_version ##

.. function:: get_sdl_byteorder

   | :sl:`get the byte order of SDL`
   | :sg:`get_sdl_byteorder() -> int`

   Returns the byte order of the ``SDL`` library. It returns ``LIL_ENDIAN`` for
   little endian byte order and ``BIG_ENDIAN`` for big endian byte order.

   get_sdl_byteorder is new in pygame 1.8

   .. ## pygame.get_sdl_byteorder ##

.. function:: register_quit

   | :sl:`register a function to be called when pygame quits`
   | :sg:`register_quit(callable) -> None`

   When :func:`pygame.quit` is called, all registered quit functions are
   called. Pygame modules do this automatically when they are initializing.
   This function is not be needed for regular pygame users.

   .. ## pygame.register_quit ##

.. function:: encode_string

   | :sl:`Encode a Unicode or bytes object`
   | :sg:`encode_string([obj [, encoding [, errors [, etype]]]]) -> bytes or None`

   obj: If Unicode, encode; if bytes, return unaltered; if anything else,
   return None; if not given, raise SyntaxError.

   encoding (string): If present, encoding to use. The default is
   'unicode_escape'.

   errors (string): If given, how to handle unencodable characters. The default
   is 'backslashreplace'.

   etype (exception type): If given, the exception type to raise for an
   encoding error. The default is UnicodeEncodeError, as returned by
   ``PyUnicode_AsEncodedString()``. For the default encoding and errors values
   there should be no encoding errors.

   This function is used in encoding file paths. Keyword arguments are
   supported.

   Added in pygame 1.9.2 (primarily for use in unit tests)

   .. ## pygame.encode_string ##

.. function:: encode_file_path

   | :sl:`Encode a Unicode or bytes object as a file system path`
   | :sg:`encode_file_path([obj [, etype]]) -> bytes or None`

   obj: If Unicode, encode; if bytes, return unaltered; if anything else,
   return None; if not given, raise SyntaxError.

   etype (exception type): If given, the exception type to raise for an
   encoding error. The default is UnicodeEncodeError, as returned by
   ``PyUnicode_AsEncodedString()``.

   This function is used to encode file paths in pygame. Encoding is to the
   codec as returned by ``sys.getfilesystemencoding()``. Keyword arguments are
   supported.

   Added in pygame 1.9.2 (primarily for use in unit tests)

   .. ## pygame.encode_file_path ##

:mod:`pygame.version`
=====================

.. module:: pygame.version
   :synopsis: small module containing version information

| :sl:`small module containing version information`

This module is automatically imported into the pygame package and offers a few
variables to check with version of pygame has been imported.

.. data:: ver

   | :sl:`version number as a string`
   | :sg:`ver = '1.2'`

   This is the version represented as a string. It can contain a micro release
   number as well, ``e.g.``, '1.5.2'

   .. ## pygame.version.ver ##

.. data:: vernum

   | :sl:`tupled integers of the version`
   | :sg:`vernum = (1, 5, 3)`

   This variable for the version can easily be compared with other version
   numbers of the same format. An example of checking pygame version numbers
   would look like this:

   ::

       if pygame.version.vernum < (1, 5):
           print 'Warning, older version of pygame (%s)' %  pygame.version.ver
           disable_advanced_features = True

   .. ## pygame.version.vernum ##

.. data:: rev

   | :sl:`repository revision of the build`
   | :sg:`rev = 'a6f89747b551+'`

   The Mercurial node identifier of the repository checkout from which this
   package was built. If the identifier ends with a plus sign '+' then the
   package contains uncommitted changes. Please include this revision number
   in bug reports, especially for non-release pygame builds.

.. ## pygame.version ##

.. ## pygame ##
