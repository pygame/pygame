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
mouse wheel will generate ``pygame.MOUSEBUTTONDOWN`` events when rolled. The
button will be set to 4 when the wheel is rolled up, and to button 5 when the
wheel is rolled down. Anytime the mouse is moved it generates a
``pygame.MOUSEMOTION`` event. The mouse movement is broken into small and
accurate motion events. As the mouse is moving many motion events will be
placed on the queue. Mouse motion events that are not properly cleaned from the
event queue are the primary reason the event queue fills up.

If the mouse cursor is hidden, and input is grabbed to the current display the
mouse will enter a virtual input mode, where the relative movements of the
mouse will never be stopped by the borders of the screen. See the functions
``pygame.mouse.set_visible()`` and ``pygame.event.set_grab()`` to get this
configured.

.. function:: get_pressed

   | :sl:`get the state of the mouse buttons`
   | :sg:`get_pressed() -> (button1, button2, button3)`

   Returns a sequence of booleans representing the state of all the mouse
   buttons. A true value means the mouse is currently being pressed at the time
   of the call.

   Note, to get all of the mouse events it is better to use either

   ::

    pygame.event.wait() or pygame.event.get() and check all of those events

   to see if they are ``MOUSEBUTTONDOWN``, ``MOUSEBUTTONUP``, or
   ``MOUSEMOTION``.

   Note, that on ``X11`` some XServers use middle button emulation. When you
   click both buttons 1 and 3 at the same time a 2 button event can be emitted.

   Note, remember to call ``pygame.event.get()`` before this function.
   Otherwise it will not work.

   .. ## pygame.mouse.get_pressed ##

.. function:: get_pos

   | :sl:`get the mouse cursor position`
   | :sg:`get_pos() -> (x, y)`

   Returns the ``X`` and ``Y`` position of the mouse cursor. The position is
   relative the the top-left corner of the display. The cursor position can be
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
   a new ``pygaqme.MOUSEMOTION`` event.

   .. ## pygame.mouse.set_pos ##

.. function:: set_visible

   | :sl:`hide or show the mouse cursor`
   | :sg:`set_visible(bool) -> bool`

   If the bool argument is true, the mouse cursor will be visible. This will
   return the previous visible state of the cursor.

   .. ## pygame.mouse.set_visible ##

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

   | :sl:`set the image for the system mouse cursor`
   | :sg:`set_cursor(size, hotspot, xormasks, andmasks) -> None`

   When the mouse cursor is visible, it will be displayed as a black and white
   bitmap using the given bitmask arrays. The size is a sequence containing the
   cursor width and height. Hotspot is a sequence containing the cursor hotspot
   position. xormasks is a sequence of bytes containing the cursor xor data
   masks. Lastly is andmasks, a sequence of bytes containting the cursor
   bitmask data.

   Width must be a multiple of 8, and the mask arrays must be the correct size
   for the given width and height. Otherwise an exception is raised.

   See the ``pygame.cursor`` module for help creating default and custom masks
   for the system cursor.

   .. ## pygame.mouse.set_cursor ##

.. function:: get_cursor

   | :sl:`get the image for the system mouse cursor`
   | :sg:`get_cursor() -> (size, hotspot, xormasks, andmasks)`

   Get the information about the mouse system cursor. The return value is the
   same data as the arguments passed into ``pygame.mouse.set_cursor()``.

   .. ## pygame.mouse.get_cursor ##

.. ## pygame.mouse ##
