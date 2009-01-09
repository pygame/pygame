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

class OverlayTypeTest(unittest.TestCase):
    def todo_test_display(self):

        # __doc__ (as of 2008-08-02) for pygame.overlay.overlay.display:

          # Overlay.display((y, u, v)): return None
          # Overlay.display(): return None
          # set the overlay pixel data

        self.fail() 

    def todo_test_get_hardware(self):

        # __doc__ (as of 2008-08-02) for pygame.overlay.overlay.get_hardware:

          # Overlay.get_hardware(rect): return int
          # test if the Overlay is hardware accelerated

        self.fail() 

    def todo_test_set_location(self):

        # __doc__ (as of 2008-08-02) for pygame.overlay.overlay.set_location:

          # Overlay.set_location(rect): return None
          # control where the overlay is displayed

        self.fail() 

################################################################################

if __name__ == '__main__':
    unittest.main()
