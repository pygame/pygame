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
import pygame

class SndarrayTest (unittest.TestCase):
    def test_import(self):
        'does it import'
        import pygame.sndarray

    def todo_test_array(self):

        # __doc__ (as of 2008-08-02) for pygame.sndarray.array:

          # pygame.sndarray.array(Sound): return array
          # 
          # Copy Sound samples into an array.
          # 
          # Creates a new array for the sound data and copies the samples. The
          # array will always be in the format returned from
          # pygame.mixer.get_init().
          # 
          # Creates a new array for the sound data and copies the samples. The
          # array will always be in the format returned from
          # pygame.mixer.get_init().
          # 

        self.fail() 

    def todo_test_get_arraytype(self):

        # __doc__ (as of 2008-08-02) for pygame.sndarray.get_arraytype:

          # pygame.sndarray.get_arraytype (): return str
          # 
          # Gets the currently active array type.
          # 
          # Returns the currently active array type. This will be a value of the
          # get_arraytypes() tuple and indicates which type of array module is
          # used for the array creation.
          # 
          # Returns the currently active array type. This will be a value of the
          # get_arraytypes() tuple and indicates which type of array module is
          # used for the array creation.
          # 
          # New in pygame 1.8 

        self.fail() 

    def todo_test_get_arraytypes(self):

        # __doc__ (as of 2008-08-02) for pygame.sndarray.get_arraytypes:

          # pygame.sndarray.get_arraytypes (): return tuple
          # 
          # Gets the array system types currently supported.
          # 
          # Checks, which array system types are available and returns them as a
          # tuple of strings. The values of the tuple can be used directly in
          # the use_arraytype () method.
          # 
          # If no supported array system could be found, None will be returned.
          # 
          # Checks, which array systems are available and returns them as a
          # tuple of strings. The values of the tuple can be used directly in
          # the pygame.sndarray.use_arraytype () method. If no supported array
          # system could be found, None will be returned.
          # 
          # New in pygame 1.8. 

        self.fail() 

    def todo_test_make_sound(self):

        # __doc__ (as of 2008-08-02) for pygame.sndarray.make_sound:

          # pygame.sndarray.make_sound(array): return Sound
          # 
          # Convert an array into a Sound object.
          # 
          # Create a new playable Sound object from an array. The mixer module
          # must be initialized and the array format must be similar to the mixer
          # audio format.
          # 
          # Create a new playable Sound object from an array. The mixer module
          # must be initialized and the array format must be similar to the
          # mixer audio format.
          # 

        self.fail() 

    def todo_test_samples(self):

        # __doc__ (as of 2008-08-02) for pygame.sndarray.samples:

          # pygame.sndarray.samples(Sound): return array
          # 
          # Reference Sound samples into an array.
          # 
          # Creates a new array that directly references the samples in a Sound
          # object. Modifying the array will change the Sound. The array will
          # always be in the format returned from pygame.mixer.get_init().
          # 
          # Creates a new array that directly references the samples in a Sound
          # object. Modifying the array will change the Sound. The array will
          # always be in the format returned from pygame.mixer.get_init().
          # 

        self.fail() 

    def todo_test_use_arraytype(self):

        # __doc__ (as of 2008-08-02) for pygame.sndarray.use_arraytype:

          # pygame.sndarray.use_arraytype (arraytype): return None
          # 
          # Sets the array system to be used for sound arrays.
          # 
          # Uses the requested array type for the module functions.
          # Currently supported array types are:
          # 
          #   numeric 
          #   numpy
          # 
          # If the requested type is not available, a ValueError will be raised.
          # 
          # Uses the requested array type for the module functions. Currently
          # supported array types are:
          # 
          #   numeric
          #   numpy
          # If the requested type is not available, a ValueError will be raised. 
          # New in pygame 1.8. 

        self.fail()

if __name__ == '__main__':
    unittest.main()
