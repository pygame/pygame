import unittest
import pygame


class JoystickTypeTest(unittest.TestCase):
    def todo_test_Joystick(self):

        # __doc__ (as of 2008-08-02) for pygame.joystick.Joystick:

        # pygame.joystick.Joystick(id): return Joystick
        # create a new Joystick object
        #
        # Create a new joystick to access a physical device. The id argument
        # must be a value from 0 to pygame.joystick.get_count()-1.
        #
        # To access most of the Joystick methods, you'll need to init() the
        # Joystick. This is separate from making sure the joystick module is
        # initialized. When multiple Joysticks objects are created for the
        # same physical joystick device (i.e., they have the same ID number),
        # the state and values for those Joystick objects will be shared.
        #
        # The Joystick object allows you to get information about the types of
        # controls on a joystick device. Once the device is initialized the
        # Pygame event queue will start receiving events about its input.
        #
        # You can call the Joystick.get_name() and Joystick.get_id() functions
        # without initializing the Joystick object.
        #

        self.fail()


class JoytickModuleTest(unittest.TestCase):
    def todo_test_get_count(self):

        # __doc__ (as of 2008-08-02) for pygame.joystick.get_count:

        # pygame.joystick.get_count(): return count
        # number of joysticks on the system
        #
        # Return the number of joystick devices on the system. The count will
        # be 0 if there are no joysticks on the system.
        #
        # When you create Joystick objects using Joystick(id), you pass an
        # integer that must be lower than this count.
        #

        self.fail()

    def test_get_init(self):
        # Check that get_init() matches what is actually happening
        def error_check_get_init():
            try:
                pygame.joystick.get_count()
            except pygame.error:
                return False
            return True

        # Start uninitialised
        self.assertEqual(pygame.joystick.get_init(), False)

        pygame.joystick.init()
        self.assertEqual(pygame.joystick.get_init(), error_check_get_init()) # True
        pygame.joystick.quit()
        self.assertEqual(pygame.joystick.get_init(), error_check_get_init()) # False

        pygame.joystick.init()
        pygame.joystick.init()
        self.assertEqual(pygame.joystick.get_init(), error_check_get_init()) # True
        pygame.joystick.quit()
        self.assertEqual(pygame.joystick.get_init(), error_check_get_init()) # False

        pygame.joystick.quit()
        self.assertEqual(pygame.joystick.get_init(), error_check_get_init())  # False

        for i in range(100):
            pygame.joystick.init()
        self.assertEqual(pygame.joystick.get_init(), error_check_get_init()) # True
        pygame.joystick.quit()
        self.assertEqual(pygame.joystick.get_init(), error_check_get_init()) # False

        for i in range(100):
            pygame.joystick.quit()
        self.assertEqual(pygame.joystick.get_init(), error_check_get_init()) # False



    def todo_test_init(self):

        # __doc__ (as of 2008-08-02) for pygame.joystick.init:

        # pygame.joystick.init(): return None
        # initialize the joystick module
        #
        # This function is called automatically by pygame.init().
        # It initializes the joystick module. This will scan the system for
        # all joystick devices. The module must be initialized before any
        # other functions will work.
        #
        # It is safe to call this function more than once.

        self.fail()

    def todo_test_quit(self):

        # __doc__ (as of 2008-08-02) for pygame.joystick.quit:

        # pygame.joystick.quit(): return None
        # uninitialize the joystick module
        #
        # Uninitialize the joystick module. After you call this any existing
        # joystick objects will no longer work.
        #
        # It is safe to call this function more than once.

        self.fail()


################################################################################

if __name__ == "__main__":
    unittest.main()
