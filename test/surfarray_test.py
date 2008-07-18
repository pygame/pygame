import test_utils
import test.unittest as unittest
from test_utils import test_not_implemented

import pygame

class SurfarrayModuleTest (unittest.TestCase):
    def test_import(self):
        'does it import'
        import pygame.surfarray

    def test_array2d(self):

        # __doc__ (as of 2008-06-25) for pygame.surfarray.array2d:

          # pygame.surfarray.array2d (Surface): return array
          # 
          # Copy pixels into a 2d array.
          # 
          # Copy the pixels from a Surface into a 2D array. The bit depth of the
          # surface will control the size of the integer values, and will work
          # for any type of pixel format.
          # 
          # This function will temporarily lock the Surface as pixels are copied
          # (see the Surface.lock - lock the Surface memory for pixel access
          # method).

        self.assert_(test_not_implemented()) 

    def test_array3d(self):

        # __doc__ (as of 2008-06-25) for pygame.surfarray.array3d:

          # pygame.surfarray.array3d (Surface): return array
          # 
          # Copy pixels into a 3d array.
          # 
          # Copy the pixels from a Surface into a 3D array. The bit depth of the
          # surface will control the size of the integer values, and will work
          # for any type of pixel format.
          # 
          # This function will temporarily lock the Surface as pixels are copied
          # (see the Surface.lock - lock the Surface memory for pixel access
          # method).

        self.assert_(test_not_implemented()) 

    def test_array_alpha(self):

        # __doc__ (as of 2008-06-25) for pygame.surfarray.array_alpha:

          # pygame.surfarray.array_alpha (Surface): return array
          # 
          # Copy pixel alphas into a 2d array.
          # 
          # Copy the pixel alpha values (degree of transparency) from a Surface
          # into a 2D array. This will work for any type of Surface
          # format. Surfaces without a pixel alpha will return an array with all
          # opaque values.
          # 
          # This function will temporarily lock the Surface as pixels are copied
          # (see the Surface.lock - lock the Surface memory for pixel access
          # method).

        self.assert_(test_not_implemented()) 

    def test_array_colorkey(self):

        # __doc__ (as of 2008-06-25) for pygame.surfarray.array_colorkey:

          # pygame.surfarray.array_colorkey (Surface): return array
          # 
          # Copy the colorkey values into a 2d array.
          # 
          # Create a new array with the colorkey transparency value from each
          # pixel. If the pixel matches the colorkey it will be fully
          # tranparent; otherwise it will be fully opaque.
          # 
          # This will work on any type of Surface format. If the image has no
          # colorkey a solid opaque array will be returned.
          # 
          # This function will temporarily lock the Surface as pixels are
          # copied.

        self.assert_(test_not_implemented()) 

    def test_blit_array(self):

        # __doc__ (as of 2008-06-25) for pygame.surfarray.blit_array:

          # pygame.surfarray.blit_array (Surface, array): return None
          # 
          # Blit directly from a array values.
          # 
          # Directly copy values from an array into a Surface. This is faster
          # than converting the array into a Surface and blitting. The array
          # must be the same dimensions as the Surface and will completely
          # replace all pixel values.
          # 
          # This function will temporarily lock the Surface as the new values
          # are copied.

        self.assert_(test_not_implemented()) 

    def test_get_arraytype(self):

        # __doc__ (as of 2008-06-25) for pygame.surfarray.get_arraytype:

          # pygame.surfarray.get_arraytype (): return str
          # 
          # Gets the currently active array type.
          # 
          # Returns the currently active array type. This will be a value of the
          # get_arraytypes() tuple and indicates which type of array module is
          # used for the array creation.

        self.assert_(test_not_implemented()) 

    def test_get_arraytypes(self):

        # __doc__ (as of 2008-06-25) for pygame.surfarray.get_arraytypes:

          # pygame.surfarray.get_arraytypes (): return tuple
          # 
          # Gets the array system types currently supported.
          # 
          # Checks, which array system types are available and returns them as a
          # tuple of strings. The values of the tuple can be used directly in
          # the use_arraytype () method.
          # 
          # If no supported array system could be found, None will be returned.

        self.assert_(test_not_implemented()) 

    def test_make_surface(self):

        # __doc__ (as of 2008-06-25) for pygame.surfarray.make_surface:

          # pygame.surfarray.make_surface (array): return Surface
          # 
          # Copy an array to a new surface.
          # 
          # Create a new Surface that best resembles the data and format on the
          # array. The array can be 2D or 3D with any sized integer values.

        self.assert_(test_not_implemented()) 

    def test_map_array(self):

        # __doc__ (as of 2008-06-25) for pygame.surfarray.map_array:

          # pygame.surfarray.map_array (Surface, array3d): return array2d
          # 
          # Map a 3D array into a 2D array.
          # 
          # Convert a 3D array into a 2D array. This will use the given Surface
          # format to control the conversion. Palette surface formats are not
          # supported.

        self.assert_(test_not_implemented()) 

    def test_pixels2d(self):

        # __doc__ (as of 2008-06-25) for pygame.surfarray.pixels2d:

          # pygame.surfarray.pixels2d (Surface): return array
          # 
          # Reference pixels into a 2d array.
          # 
          # Create a new 2D array that directly references the pixel values in a
          # Surface. Any changes to the array will affect the pixels in the
          # Surface. This is a fast operation since no data is copied.
          # 
          # Pixels from a 24-bit Surface cannot be referenced, but all other
          # Surface bit depths can.
          # 
          # The Surface this references will remain locked for the lifetime of
          # the array (see the Surface.lock - lock the Surface memory for pixel
          # access method).

        self.assert_(test_not_implemented()) 

    def test_pixels3d(self):

        # __doc__ (as of 2008-06-25) for pygame.surfarray.pixels3d:

          # pygame.surfarray.pixels3d (Surface): return array
          # 
          # Reference pixels into a 3d array.
          # 
          # Create a new 3D array that directly references the pixel values in a
          # Surface. Any changes to the array will affect the pixels in the
          # Surface. This is a fast operation since no data is copied.
          # 
          # This will only work on Surfaces that have 24-bit or 32-bit
          # formats. Lower pixel formats cannot be referenced.
          # 
          # The Surface this references will remain locked for the lifetime of
          # the array (see the Surface.lock - lock the Surface memory for pixel
          # access method).

        self.assert_(test_not_implemented()) 

    def test_pixels_alpha(self):

        # __doc__ (as of 2008-06-25) for pygame.surfarray.pixels_alpha:

          # pygame.surfarray.pixels_alpha (Surface): return array
          # 
          # Reference pixel alphas into a 2d array.
          # 
          # Create a new 2D array that directly references the alpha values
          # (degree of transparency) in a Surface. Any changes to the array will
          # affect the pixels in the Surface. This is a fast operation since no
          # data is copied.
          # 
          # This can only work on 32-bit Surfaces with a per-pixel alpha value.
          # 
          # The Surface this array references will remain locked for the
          # lifetime of the array.

        self.assert_(test_not_implemented()) 

    def test_use_arraytype(self):

        # __doc__ (as of 2008-06-25) for pygame.surfarray.use_arraytype:

          # pygame.surfarray.use_arraytype (arraytype): return None
          # 
          # Sets the array system to be used for surface arrays.
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