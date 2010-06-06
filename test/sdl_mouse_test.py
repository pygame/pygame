import unittest
import pygame2
import pygame2.sdl.mouse as image
import pygame2.sdl.video as video
import pygame2.sdl.constants as constants

class SDLMouseTest (unittest.TestCase):
    __tags__ = [ "sdl" ]

    def todo_test_pygame2_sdl_mouse_get_position(self):

        # __doc__ (as of 2010-01-13) for pygame2.sdl.mouse.get_position:

        # get_position () -> x, y
        # 
        # Gets the current mouse position on the main screen.

        self.fail() 

    def todo_test_pygame2_sdl_mouse_get_rel_position(self):

        # __doc__ (as of 2010-01-13) for pygame2.sdl.mouse.get_rel_position:

        # get_rel_position () -> x, y
        # 
        # Gets the relative mouse movement.
        # 
        # Gets the relative mouse movement since the last call to this function.

        self.fail() 

    def todo_test_pygame2_sdl_mouse_get_rel_state(self):

        # __doc__ (as of 2010-01-13) for pygame2.sdl.mouse.get_rel_state:

        # get_rel_state () -> buttons, x, y
        # 
        # Gets the relative mouse state.
        # 
        # Gets the relative mouse state since the last call to this function.

        self.fail() 

    def todo_test_pygame2_sdl_mouse_get_state(self):

        # __doc__ (as of 2010-01-13) for pygame2.sdl.mouse.get_state:

        # get_state () -> buttons, x, y
        # 
        # Gets the current mouse state.
        # 
        # Gets the current mouse state. The returned buttons value
        # represents a bitmask with the pressed buttons.

        self.fail() 

    def todo_test_pygame2_sdl_mouse_set_cursor(self):

        # __doc__ (as of 2010-01-13) for pygame2.sdl.mouse.set_cursor:

        # set_cursor (cursor) -> None
        # 
        # Sets the mouse cursor to be shown.
        # 
        # Sets the mouse cursor to be displayed. The change will be visible
        # immediately.

        self.fail() 

    def todo_test_pygame2_sdl_mouse_set_position(self):

        # __doc__ (as of 2010-01-13) for pygame2.sdl.mouse.set_position:

        # set_position (x, y) -> None
        # set_position (point) -> None
        # 
        # Sets the position of the mouse cursor.
        # 
        # Sets the position of the mouse cursor on the main screen. This
        # behaves exactly as the warp method.

        self.fail() 

    def todo_test_pygame2_sdl_mouse_set_visible(self):

        # __doc__ (as of 2010-01-13) for pygame2.sdl.mouse.set_visible:

        # set_visible (show) -> state
        # 
        # Toggles whether the cursor is shown on the screen.
        # 
        # Toggles whether the cursor is shown on the screen. This
        # behaves exactly as the show_cursor method.

        self.fail() 

    def todo_test_pygame2_sdl_mouse_show_cursor(self):

        # __doc__ (as of 2010-01-13) for pygame2.sdl.mouse.show_cursor:

        # show_cursor (show) -> state
        # 
        # Toggles whether the cursor is shown on the screen.
        # 
        # Toggles whether the cursor is shown on the screen. *show* can be a
        # boolean or one a value of ENABLE, DISABLE and
        # QUERY. QUERY will not change the visibility state but
        # query, if the cursor is currently shown or not.
        # 
        # The state returned is a value of ENABLE and DISABLE,
        # indicating, whether the cursor is currently shown or not.

        self.fail() 

    def todo_test_pygame2_sdl_mouse_warp(self):

        # __doc__ (as of 2010-01-13) for pygame2.sdl.mouse.warp:

        # warp (x, y) -> None
        # warp (point) -> None
        # 
        # Sets the position of the mouse cursor.
        # 
        # Sets the position of the mouse cursor on the main screen. This
        # also will generate a mouse motion event.

        self.fail() 

if __name__ == "__main__":
    unittest.main ()
