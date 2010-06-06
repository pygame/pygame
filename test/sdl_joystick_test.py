import unittest
import pygame2
import pygame2.sdl.joystick as joystick
import pygame2.sdl.video as video
import pygame2.sdl.constants as constants

class SDLJoystickTest (unittest.TestCase):
    __tags__ = [ "sdl" ]

    def todo_test_pygame2_sdl_joystick_Joystick_close(self):

        # __doc__ (as of 2010-01-13) for pygame2.sdl.joystick.Joystick.close:

        # close () -> None
        # 
        # Closes the access to the underlying joystick device.
        # 
        # Calling or accessing any other method or attribute of the
        # Joystick after closing it will cause an exception to be
        # thrown. To reaccess the Joystick, you have to open it
        # again.

        self.fail() 

    def todo_test_pygame2_sdl_joystick_Joystick_get_axis(self):

        # __doc__ (as of 2010-01-13) for pygame2.sdl.joystick.Joystick.get_axis:

        # get_axis (index) -> int
        # 
        # Gets the current position of the specified axis.
        # 
        # Gets the current position of the specified axis. The axis *index*
        # must be a valid value within the range [0, num_axes - 1].

        self.fail() 

    def todo_test_pygame2_sdl_joystick_Joystick_get_ball(self):

        # __doc__ (as of 2010-01-13) for pygame2.sdl.joystick.Joystick.get_ball:

        # get_ball (index) -> int, int
        # 
        # Gets the relative movement of a trackball.
        # 
        # Gets the relative movement of a trackball since the last call to
        # get_ball. The ball *index* must be a valid value within the
        # range [0, num_balls - 1].

        self.fail() 

    def todo_test_pygame2_sdl_joystick_Joystick_get_button(self):

        # __doc__ (as of 2010-01-13) for pygame2.sdl.joystick.Joystick.get_button:

        # get_button (index) -> bool
        # 
        # Gets the state of a button.
        # 
        # Gets the current pressed state of a button. The button *index*
        # must be a valid value within the range [0, num_buttons - 1].

        self.fail() 

    def todo_test_pygame2_sdl_joystick_Joystick_get_hat(self):

        # __doc__ (as of 2010-01-13) for pygame2.sdl.joystick.Joystick.get_hat:

        # get_hat (index) -> int
        # 
        # Gets the state of a hat.
        # 
        # Gets the current state of a hat. The return value will be a
        # bitwise OR'd combination of the hat constants as specified in the
        # :mod:pygame2.sdl.constants module. The hat *index* must be a valid
        # value within the range [0, num_hats - 1].

        self.fail() 

    def todo_test_pygame2_sdl_joystick_Joystick_index(self):

        # __doc__ (as of 2010-01-13) for pygame2.sdl.joystick.Joystick.index:

        # Gets the physical device index of the Joystick.

        self.fail() 

    def todo_test_pygame2_sdl_joystick_Joystick_name(self):

        # __doc__ (as of 2010-01-13) for pygame2.sdl.joystick.Joystick.name:

        # Gets the physical device name of the Joystick.

        self.fail() 

    def todo_test_pygame2_sdl_joystick_Joystick_num_axes(self):

        # __doc__ (as of 2010-01-13) for pygame2.sdl.joystick.Joystick.num_axes:

        # Gets the number of axes available on the Joystick.

        self.fail() 

    def todo_test_pygame2_sdl_joystick_Joystick_num_balls(self):

        # __doc__ (as of 2010-01-13) for pygame2.sdl.joystick.Joystick.num_balls:

        # Gets the number of balls available on the Joystick.

        self.fail() 

    def todo_test_pygame2_sdl_joystick_Joystick_num_buttons(self):

        # __doc__ (as of 2010-01-13) for pygame2.sdl.joystick.Joystick.num_buttons:

        # Gets the number of buttons available on the Joystick.

        self.fail() 

    def todo_test_pygame2_sdl_joystick_Joystick_num_hats(self):

        # __doc__ (as of 2010-01-13) for pygame2.sdl.joystick.Joystick.num_hats:

        # Gets the number of hats available on the Joystick.

        self.fail() 

    def todo_test_pygame2_sdl_joystick_Joystick_open(self):

        # __doc__ (as of 2010-01-13) for pygame2.sdl.joystick.Joystick.open:

        # open () -> None
        # 
        # Opens the (closed) Joystick.
        # 
        # Opens the (closed) Joystick. If the Joystick is already open, this
        # method will have no effect.

        self.fail() 

    def todo_test_pygame2_sdl_joystick_Joystick_opened(self):

        # __doc__ (as of 2010-01-13) for pygame2.sdl.joystick.Joystick.opened:

        # Gets, whether the Joystick is open or not.

        self.fail() 

    def todo_test_pygame2_sdl_joystick_event_state(self):

        # __doc__ (as of 2010-01-13) for pygame2.sdl.joystick.event_state:

        # event_state (state) -> int
        # 
        # Enables or disable joystick event polling.
        # 
        # Enables or disables joystick event processing. If the joystick
        # event processing is disabled, you will have to update the joystick
        # states manually using update and read the information
        # manually using the specific attributes and methods. Otherwise,
        # joystick events are consumed and process by the
        # :mod:pygame2.sdl.event module.
        # 
        # The *state* argument can be ENABLE or IGNORE for
        # enabling or disabling the event processing or QUERY to receive
        # the current state. The return value also will be one of those three
        # constants.

        self.fail() 

    def todo_test_pygame2_sdl_joystick_get_name(self):

        # __doc__ (as of 2010-01-13) for pygame2.sdl.joystick.get_name:

        # get_name (index) -> str
        # 
        # Gets the physical device name of a Joystick.
        # 
        # Gets the physical device name of a Joystick. The *index*
        # specifies the device to get the name for and must be in the range [0,
        # num_joysticks - 1].

        self.fail() 

    def todo_test_pygame2_sdl_joystick_init(self):

        # __doc__ (as of 2010-01-13) for pygame2.sdl.joystick.init:

        # init () -> None
        # 
        # Initializes the joystick subsystem of the SDL library.

        self.fail() 

    def todo_test_pygame2_sdl_joystick_num_joysticks(self):

        # __doc__ (as of 2010-01-13) for pygame2.sdl.joystick.num_joysticks:

        # um_joysticks () -> int
        # 
        # Gets the number of detected and available joysticks for the system.

        self.fail() 

    def todo_test_pygame2_sdl_joystick_opened(self):

        # __doc__ (as of 2010-01-13) for pygame2.sdl.joystick.opened:

        # opened (index) -> bool
        # 
        # Gets, whether the specified joystick is opened for access or not.
        # 
        # Gets, whether the specified joystick is opened for access or
        # not. The *index* specifies the joystick device to get the state for
        # and must be in the range [0, num_joysticks - 1].

        self.fail() 

    def todo_test_pygame2_sdl_joystick_quit(self):

        # __doc__ (as of 2010-01-13) for pygame2.sdl.joystick.quit:

        # quit () -> None
        # 
        # Shuts down the joystick subsystem of the SDL library.
        # 
        # Shuts down the joystick subsystem of the SDL library and closes all
        # existing Joystick objects (leaving them intact).
        # 
        # After calling this function, you should not invoke any class,
        # method or function related to the joystick subsystem as they are
        # likely to fail or might give unpredictable results.

        self.fail() 

    def todo_test_pygame2_sdl_joystick_update(self):

        # __doc__ (as of 2010-01-13) for pygame2.sdl.joystick.update:

        # update () -> None
        # 
        # Updates the joystick states (in case event processing is disabled).

        self.fail() 

    def todo_test_pygame2_sdl_joystick_was_init(self):

        # __doc__ (as of 2010-01-13) for pygame2.sdl.joystick.was_init:

        # was_init () -> bool
        # 
        # Returns, whether the joystick subsystem of the SDL library is
        # initialized.

        self.fail() 

if __name__ == "__main__":
    unittest.main ()
