try:
    import test.pgunittest as unittest
except:
    import pgunittest as unittest
from pygame2.base import Surface

class TestSurface (Surface):
    def __init__ (self):
        Surface.__init__ (self)
        self.height = 10
        self.width = 20
        self.size = (20, 10)
        self.pixels = buffer ("pixels")
    def blit (self, **kwds):
        return "blit"
    def copy (self):
        return TestSurface ()
    def blit (self, **kwds):
        return "blit"

class SurfaceTest (unittest.TestCase):

    def todo_test_pygame2_base_Surface_blit(self):

        # __doc__ (as of 2009-03-28) for pygame2.base.Surface.blit:

        # blit (**kwds) -> object
        # 
        # Performs a blit operation on the Surface.
        # 
        # blit (**kwds) -> object  Performs a blit operation on the Surface.
        # The behaviour, arguments and return value depend on the concrete
        # Surface implementation.

        sf = TestSurface ()
        self.assertEquals (sf.blit (), "blit")
        self.assertEquals (super (Surface, sf).blit (self, {}), "blit")

    def todo_test_pygame2_base_Surface_copy(self):

        # __doc__ (as of 2009-03-28) for pygame2.base.Surface.copy:

        # copy () -> Surface
        #
        # Creates a copy of this Surface.

        sf = TestSurface ()
        sf2 = sf.copy()
        self.assertEquals (sf.size, sf2.size)

        sf2 = super (Surface, sf).copy()
        self.assertEquals (sf.size, sf2.size)

    def todo_test_pygame2_base_Surface_height(self):

        # __doc__ (as of 2009-03-28) for pygame2.base.Surface.height:

        # Gets the height of the Surface.
        sf = TestSurface ()

        self.assertEquals (sf.height, 10)
        self.assertEquals (super (Surface, sf).height, 10)

    def todo_test_pygame2_base_Surface_pixels(self):

        # __doc__ (as of 2009-03-28) for pygame2.base.Surface.pixels:

        # Gets a buffer with the pixels of the Surface.
        sf = TestSurface ()

        self.assertEquals (sf.height, 10)
        self.assertEquals (super (Surface, sf).height, 10)

    def todo_test_pygame2_base_Surface_size(self):

        # __doc__ (as of 2009-03-28) for pygame2.base.Surface.size:

        # Gets the width and height of the Surface.
        sf = TestSurface ()
        
        self.assertEquals (sf.size, (20, 10))
        self.assertEquals (super (Surface, sf).size, (20, 10))

    def todo_test_pygame2_base_Surface_width(self):

        # __doc__ (as of 2009-03-28) for pygame2.base.Surface.width:

        # Gets the width of the Surface.
        sf = TestSurface ()

        self.assertEquals (sf.width, 20)
        self.assertEquals (super (Surface, sf).width, 20)

if __name__ == "__main__":
    unittest.main ()
