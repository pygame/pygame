import test_utils
import test.unittest as unittest

from test_utils import test_not_implemented

import pygame

class SndarrayTest (unittest.TestCase):
    def test_import(self):
        'does it import'
        import pygame.sndarray
    
    def test_array(self):
    
        # __doc__ (as of 2008-06-25) for pygame.sndarray.array:
    
          # pygame.sndarray.array(Sound): return array
          # 
          # Copy Sound samples into an array.
          # 
          # Creates a new array for the sound data and copies the samples. The
          # array will always be in the format returned from
          # pygame.mixer.get_init().
    
        self.assert_(test_not_implemented()) 
    
    def test_get_arraytype(self):
    
        # __doc__ (as of 2008-06-25) for pygame.sndarray.get_arraytype:
    
          # pygame.sndarray.get_arraytype (): return str
          # 
          # Gets the currently active array type.
          # 
          # Returns the currently active array type. This will be a value of the
          # get_arraytypes() tuple and indicates which type of array module is
          # used for the array creation.
    
        self.assert_(test_not_implemented()) 
    
    def test_get_arraytypes(self):
    
        # __doc__ (as of 2008-06-25) for pygame.sndarray.get_arraytypes:
    
          # pygame.sndarray.get_arraytypes (): return tuple
          # 
          # Gets the array system types currently supported.
          # 
          # Checks, which array system types are available and returns them as a
          # tuple of strings. The values of the tuple can be used directly in
          # the use_arraytype () method.
          # 
          # If no supported array system could be found, None will be returned.
    
        self.assert_(test_not_implemented()) 
    
    def test_make_sound(self):
    
        # __doc__ (as of 2008-06-25) for pygame.sndarray.make_sound:
    
          # pygame.sndarray.make_sound(array): return Sound
          # 
          # Convert an array into a Sound object.
          # 
          # Create a new playable Sound object from an array. The mixer module
          # must be initialized and the array format must be similar to the mixer
          # audio format.
    
        self.assert_(test_not_implemented()) 
    
    def test_samples(self):
    
        # __doc__ (as of 2008-06-25) for pygame.sndarray.samples:
    
          # pygame.sndarray.samples(Sound): return array
          # 
          # Reference Sound samples into an array.
          # 
          # Creates a new array that directly references the samples in a Sound
          # object. Modifying the array will change the Sound. The array will
          # always be in the format returned from pygame.mixer.get_init().
    
        self.assert_(test_not_implemented()) 
    
    def test_use_arraytype(self):
    
        # __doc__ (as of 2008-06-25) for pygame.sndarray.use_arraytype:
    
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
    
        self.assert_(test_not_implemented()) 


if __name__ == '__main__':
    unittest.main()