#################################### IMPORTS ###################################

if __name__ == '__main__':
    import sys
    import os
    pkg_dir = os.path.split(os.path.abspath(__file__))[0]
    parent_dir, pkg_name = os.path.split(pkg_dir)
    is_pygame_pkg = (pkg_name == 'tests' and
                     os.path.split(parent_dir)[1] == 'pygame')
    if not is_pygame_pkg:
        sys.path.insert(0, parent_dir)
else:
    is_pygame_pkg = __name__.startswith('pygame.tests.')

if is_pygame_pkg:
    from pygame.tests.test_utils import test_not_implemented, unittest
else:
    from test.test_utils import test_not_implemented, unittest

################################################################################

class MouseModuleTest(unittest.TestCase):
    def todo_test_get_cursor(self):

        # __doc__ (as of 2008-08-02) for pygame.mouse.get_cursor:

          # pygame.mouse.get_cursor(): return (size, hotspot, xormasks, andmasks)
          # get the image for the system mouse cursor
          # 
          # Get the information about the mouse system cursor. The return value
          # is the same data as the arguments passed into
          # pygame.mouse.set_cursor().
          # 

        self.fail() 

    def todo_test_get_focused(self):

        # __doc__ (as of 2008-08-02) for pygame.mouse.get_focused:

          # pygame.mouse.get_focused(): return bool
          # check if the display is receiving mouse input
          # 
          # Returns true when pygame is receiving mouse input events (or, in
          # windowing terminology, is "active" or has the "focus").
          # 
          # This method is most useful when working in a window. By contrast, in
          # full-screen mode, this method always returns true.
          # 
          # Note: under MS Windows, the window that has the mouse focus also has
          # the keyboard focus. But under X-Windows, one window can receive
          # mouse events and another receive keyboard events.
          # pygame.mouse.get_focused() indicates whether the pygame window
          # receives mouse events.
          # 

        self.fail() 

    def todo_test_get_pos(self):

        # __doc__ (as of 2008-08-02) for pygame.mouse.get_pos:

          # pygame.mouse.get_pos(): return (x, y)
          # get the mouse cursor position
          # 
          # Returns the X and Y position of the mouse cursor. The position is
          # relative the the top-left corner of the display. The cursor position
          # can be located outside of the display window, but is always
          # constrained to the screen.
          # 

        self.fail() 

    def todo_test_get_pressed(self):

        # __doc__ (as of 2008-08-02) for pygame.mouse.get_pressed:

          # pygame.moouse.get_pressed(): return (button1, button2, button3)
          # get the state of the mouse buttons
          # 
          # Returns a sequence of booleans representing the state of all the
          # mouse buttons. A true value means the mouse is currently being
          # pressed at the time of the call.
          # 
          # Note, to get all of the mouse events it is better to use either 
          #  pygame.event.wait() or pygame.event.get() and check all of those events
          # to see if they are MOUSEBUTTONDOWN, MOUSEBUTTONUP, or MOUSEMOTION. 
          # Note, that on X11 some XServers use middle button emulation.  When
          # you click both buttons 1 and 3 at the same time a 2 button event can
          # be emitted.
          # 
          # Note, remember to call pygame.event.get() before this function.
          # Otherwise it will not work.
          # 

        self.fail() 

    def todo_test_get_rel(self):

        # __doc__ (as of 2008-08-02) for pygame.mouse.get_rel:

          # pygame.mouse.get_rel(): return (x, y)
          # get the amount of mouse movement
          # 
          # Returns the amount of movement in X and Y since the previous call to
          # this function. The relative movement of the mouse cursor is
          # constrained to the edges of the screen, but see the virtual input
          # mouse mode for a way around this.  Virtual input mode is described
          # at the top of the page.
          # 

        self.fail() 

    def todo_test_set_cursor(self):

        # __doc__ (as of 2008-08-02) for pygame.mouse.set_cursor:

          # pygame.mouse.set_cursor(size, hotspot, xormasks, andmasks): return None
          # set the image for the system mouse cursor
          # 
          # When the mouse cursor is visible, it will be displayed as a black
          # and white bitmap using the given bitmask arrays. The size is a
          # sequence containing the cursor width and height. Hotspot is a
          # sequence containing the cursor hotspot position. xormasks is a
          # sequence of bytes containing the cursor xor data masks. Lastly is
          # andmasks, a sequence of bytes containting the cursor bitmask data.
          # 
          # Width must be a multiple of 8, and the mask arrays must be the
          # correct size for the given width and height. Otherwise an exception
          # is raised.
          # 
          # See the pygame.cursor module for help creating default and custom
          # masks for the system cursor.
          # 

        self.fail() 

    def todo_test_set_pos(self):

        # __doc__ (as of 2008-08-02) for pygame.mouse.set_pos:

          # pygame.mouse.set_pos([x, y]): return None
          # set the mouse cursor position
          # 
          # Set the current mouse position to arguments given. If the mouse
          # cursor is visible it will jump to the new coordinates. Moving the
          # mouse will generate a new pygaqme.MOUSEMOTION event.
          # 

        self.fail() 

    def todo_test_set_visible(self):

        # __doc__ (as of 2008-08-02) for pygame.mouse.set_visible:

          # pygame.mouse.set_visible(bool): return bool
          # hide or show the mouse cursor
          # 
          # If the bool argument is true, the mouse cursor will be visible. This
          # will return the previous visible state of the cursor.
          # 

        self.fail() 

################################################################################

if __name__ == '__main__':
    unittest.main()
