.. include:: common.txt

:mod:`pygame.mouse`
===================

.. module:: pygame.mouse
   :synopsis: pygame module to work with the mouse

| :sl:`pygame module to work with the mouse`

The mouse functions can be used to get the current state of the mouse device.
These functions can also alter the system cursor for the mouse.

When the display mode is set, the event queue will start receiving mouse
events. The mouse buttons generate ``pygame.MOUSEBUTTONDOWN`` and
``pygame.MOUSEBUTTONUP`` events when they are pressed and released. These
events contain a button attribute representing which button was pressed. The
mouse wheel will generate ``pygame.MOUSEBUTTONDOWN`` and 
``pygame.MOUSEBUTTONUP`` events when rolled. The button will be set to 4 
when the wheel is rolled up, and to button 5 when the wheel is rolled down. 
Whenever the mouse is moved it generates a ``pygame.MOUSEMOTION`` event. The 
mouse movement is broken into small and accurate motion events. As the mouse 
is moving many motion events will be placed on the queue. Mouse motion events 
that are not properly cleaned from the event queue are the primary reason the 
event queue fills up.

If the mouse cursor is hidden, and input is grabbed to the current display the
mouse will enter a virtual input mode, where the relative movements of the
mouse will never be stopped by the borders of the screen. See the functions
``pygame.mouse.set_visible()`` and ``pygame.event.set_grab()`` to get this
configured.


**Mouse Wheel Behavior in pygame2**

There is proper functionality for mouse wheel behaviour with SDL2's
``SDL_MOUSEWHEEL`` events. ``SDL_MOUSEWHEEL`` replaces the old method of treating
mouse scrolling as a type of button like, as such in SDL1. The new events support
horizontal and vertical scroll movements, with signed integer values representing
the amount scrolled, as well as "flipped" direction (the set positive and negative
values for each axis is flipped). Read more about SDL2 input-related changes `here
<https://wiki.libsdl.org/MigrationGuide#Input>`_

In pygame2, the mouse wheel functionality can be used by listening for the ``pygame.events.MOUSEWHEEL`` EventType.
When this event is triggered, a developer can access the appropriate ``Event`` object with ``pygame.event.get()``. The object can be used
to access data about the mouse scroll, such as ``which`` (it will tell you what exact mouse device trigger the event).

.. code-block:: python
   :caption: Code example of mouse scroll (tested on 2.0.0.dev7)
   :name: test.py

   # Taken from husano896's PR thread:
   import pygame
   from pygame.locals import *
   pygame.init()
   screen = pygame.display.set_mode((640,480))
   clock = pygame.time.Clock()
   
   def main():
      while True:
         for event_var in pygame.event.get():
               if event_var.type == QUIT:
                  pygame.quit()
                  return
               elif event_var.type == MOUSEWHEEL:
                  print(event_var) # can access properties with prop notation 
                                   # (ex: event_var.y)
         clock.tick(60)

   # Execute game:
   main()

.. function:: get_pressed

   | :sl:`get the state of the mouse buttons`
   | :sg:`get_pressed() -> (button1, button2, button3, button4, button5)`

   Returns a sequence of booleans representing the state of all the mouse
   buttons. A true value means the mouse is currently being pressed at the time
   of the call.

   Note, to get all of the mouse events it is better to use either

   ::

    pygame.event.wait() or pygame.event.get() and check all of those events

   to see if they are ``MOUSEBUTTONDOWN``, ``MOUSEBUTTONUP``, or
   ``MOUSEMOTION``.

   Note, that on ``X11`` some X servers use middle button emulation. When you
   click both buttons 1 and 3 at the same time a 2 button event can be emitted.

   Note, remember to call ``pygame.event.get()`` before this function.
   Otherwise it will not work.

   ``button4`` & ``button5`` were added to the returned tuple for pygame 2.
   .. versionadded:: 2.0.0

   .. ## pygame.mouse.get_pressed ##

.. function:: get_pos

   | :sl:`get the mouse cursor position`
   | :sg:`get_pos() -> (x, y)`

   Returns the ``X`` and ``Y`` position of the mouse cursor. The position is
   relative to the top-left corner of the display. The cursor position can be
   located outside of the display window, but is always constrained to the
   screen.

   .. ## pygame.mouse.get_pos ##

.. function:: get_rel

   | :sl:`get the amount of mouse movement`
   | :sg:`get_rel() -> (x, y)`

   Returns the amount of movement in ``X`` and ``Y`` since the previous call to
   this function. The relative movement of the mouse cursor is constrained to
   the edges of the screen, but see the virtual input mouse mode for a way
   around this. Virtual input mode is described at the top of the page.

   .. ## pygame.mouse.get_rel ##

.. function:: set_pos

   | :sl:`set the mouse cursor position`
   | :sg:`set_pos([x, y]) -> None`

   Set the current mouse position to arguments given. If the mouse cursor is
   visible it will jump to the new coordinates. Moving the mouse will generate
   a new ``pygame.MOUSEMOTION`` event.

   .. ## pygame.mouse.set_pos ##

.. function:: set_visible

   | :sl:`hide or show the mouse cursor`
   | :sg:`set_visible(bool) -> bool`

   If the bool argument is true, the mouse cursor will be visible. This will
   return the previous visible state of the cursor.

   .. ## pygame.mouse.set_visible ##

.. function:: get_visible

   | :sl:`get the current visibility state of the mouse cursor`
   | :sg:`get_visible() -> bool`

   Get the current visibility state of the mouse cursor.

   :returns: ``True`` if the mouse cursor is currently visible and ``False`` if
      the mouse cursor is not visible
   :rtype: bool

   .. versionadded:: 2.0.0

   .. ## pygame.mouse.get_visible ##

.. function:: get_focused

   | :sl:`check if the display is receiving mouse input`
   | :sg:`get_focused() -> bool`

   Returns true when pygame is receiving mouse input events (or, in windowing
   terminology, is "active" or has the "focus").

   This method is most useful when working in a window. By contrast, in
   full-screen mode, this method always returns true.

   Note: under ``MS`` Windows, the window that has the mouse focus also has the
   keyboard focus. But under X-Windows, one window can receive mouse events and
   another receive keyboard events. ``pygame.mouse.get_focused()`` indicates
   whether the pygame window receives mouse events.

   .. ## pygame.mouse.get_focused ##

.. function:: set_cursor

   | :sl:`set the image for the mouse cursor`
   | :sg:`set_cursor(size, hotspot, xormasks, andmasks) -> None`

   When the mouse cursor is visible, it will be displayed as a black and white
   bitmap using the given bitmask arrays. The size is a sequence containing the
   cursor width and height. Hotspot is a sequence containing the cursor hotspot
   position. xormasks is a sequence of bytes containing the cursor xor data
   masks. Lastly is andmasks, a sequence of bytes containing the cursor
   bitmask data.

   Width must be a multiple of 8, and the mask arrays must be the correct size
   for the given width and height. Otherwise an exception is raised.

   See the ``pygame.cursor`` module for help creating default and custom masks
   for the mouse cursor.

   .. ## pygame.mouse.set_cursor ##

.. function:: set_system_cursor

   | :sl:`set the mouse cursor to a system variant`
   | :sg:`set_system_cursor(constant) -> None`

   When the mouse cursor is visible, it will displayed as a operating system
   specific variant of the options below.

   ::

      Pygame Cursor Constant           Description
      --------------------------------------------
      pygame.SYSTEM_CURSOR_ARROW       arrow
      pygame.SYSTEM_CURSOR_IBEAM       i-beam
      pygame.SYSTEM_CURSOR_WAIT        wait
      pygame.SYSTEM_CURSOR_CROSSHAIR   crosshair
      pygame.SYSTEM_CURSOR_WAITARROW   small wait cursor 
                                       (or wait if not available)
      pygame.SYSTEM_CURSOR_SIZENWSE    double arrow pointing 
                                       northwest and southeast
      pygame.SYSTEM_CURSOR_SIZENESW    double arrow pointing
                                       northeast and southwest
      pygame.SYSTEM_CURSOR_SIZEWE      double arrow pointing
                                       west and east
      pygame.SYSTEM_CURSOR_SIZENS      double arrow pointing 
                                       north and south
      pygame.SYSTEM_CURSOR_SIZEALL     four pointed arrow pointing
                                       north, south, east, and west
      pygame.SYSTEM_CURSOR_NO          slashed circle or crossbones
      pygame.SYSTEM_CURSOR_HAND        hand

   .. versionadded:: 2.0.0

   .. ## pygame.mouse.set_system_cursor ##

.. function:: get_cursor

   | :sl:`get the image of the mouse cursor`
   | :sg:`get_cursor() -> (size, hotspot, xormasks, andmasks)`

   Get the information about the mouse system cursor. The return value is the
   same data as the arguments passed into ``pygame.mouse.set_cursor()``.

   .. note:: This method is unavailable with SDL2, as SDL2 does not provide
             the underlying code to implement this method.

   .. ## pygame.mouse.get_cursor ##

.. ## pygame.mouse ##
