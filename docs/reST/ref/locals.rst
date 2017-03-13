.. include:: common.txt

:mod:`pygame.locals`
====================

.. module:: pygame.locals
   :synopsis: pygame constants

| :sl:`pygame constants`

This module contains various constants used by pygame. It's contents are
automatically placed in the pygame module namespace. However, an application
can use :mod:`pygame.locals` to include only the pygame constants with a 'from
:mod:`pygame.locals` import \*'.

Detailed descriptions of the various constants are found throughout the pygame
documentation. :func:`pygame.display.set_mode` flags like ``HWSURFACE`` are
found in the Display section. Event types are explained in the Event section.
Keyboard ``K_`` constants relating to the key attribute of a ``KEYDOWN`` or
``KEYUP`` event are listed in the Key section. Also found there are the various
``MOD_`` key modifiers. Finally, ``TIMER_RESOLUTION`` is defined in Time.

.. ## pygame.locals ##
