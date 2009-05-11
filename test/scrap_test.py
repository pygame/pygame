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
    from pygame.tests.test_utils \
         import test_not_implemented, trunk_relative_path, unittest
else:
    from test.test_utils \
         import test_not_implemented, trunk_relative_path, unittest
import pygame
from pygame import scrap

class ScrapModuleTest(unittest.TestCase):
    not_initialized = True

    def setUp(self):
        if self.not_initialized:
            pygame.init ()
            pygame.display.set_mode ((1, 1))
            scrap.init ()
            self.not_initialized = False

    def todo_test_contains(self):

        # __doc__ (as of 2008-08-02) for pygame.scrap.contains:

          # scrap.contains (type) -> bool
          # Checks, whether a certain type is available in the clipboard.
          # 
          # Returns True, if data for the passed type is available in the
          # clipboard, False otherwise.
          # 
          #   if pygame.scrap.contains (SCRAP_TEXT):
          #       print "There is text in the clipboard."
          #   if pygame.scrap.contains ("own_data_type"):
          #       print "There is stuff in the clipboard."

        self.fail() 

    def todo_test_get(self):

        # __doc__ (as of 2008-08-02) for pygame.scrap.get:

          # scrap.get (type) -> string
          # Gets the data for the specified type from the clipboard.
          # 
          # Returns the data for the specified type from the clipboard. The data
          # is returned as string and might need further processing. If no data
          # for the passed type is available, None is returned.
          # 
          #   text = pygame.scrap.get (SCRAP_TEXT)
          #   if text:
          #       # Do stuff with it.
          #   else:
          #       print "There does not seem to be text in the clipboard."

        self.fail() 

    def todo_test_get_types(self):

        # __doc__ (as of 2008-08-02) for pygame.scrap.get_types:

          # scrap.get_types () -> list
          # Gets a list of the available clipboard types.
          # 
          # Gets a list of strings with the identifiers for the available
          # clipboard types. Each identifier can be used in the scrap.get()
          # method to get the clipboard content of the specific type. If there
          # is no data in the clipboard, an empty list is returned.
          # 
          #   types = pygame.scrap.get_types ()
          #   for t in types:
          #       if "text" in t:
          #           # There is some content with the word "text" in it. It's
          #           # possibly text, so print it.
          #           print pygame.scrap.get (t)

        self.fail() 

    def todo_test_init(self):

        # __doc__ (as of 2008-08-02) for pygame.scrap.init:

          # scrap.init () -> None
          # Initializes the scrap module.
          # 
          # Tries to initialize the scrap module and raises an exception, if it
          # fails. Note that this module requires a set display surface, so you
          # have to make sure, you acquired one earlier using
          # pygame.display.set_mode().
          # 

        self.fail() 

    def todo_test_lost(self):

        # __doc__ (as of 2008-08-02) for pygame.scrap.lost:

          # scrap.lost() -> bool
          # Checks whether the clipboard is currently owned by the application.
          # 
          # Returns True, if the clipboard is currently owned by the pygame
          # application, False otherwise.
          # 
          #   if pygame.scrap.lost ():
          #      print "No content from me anymore. The clipboard is used by someone else."

        self.fail() 

    def test_set_mode (self):
        scrap.set_mode (pygame.SCRAP_SELECTION)
        scrap.set_mode (pygame.SCRAP_CLIPBOARD)
        self.assertRaises (ValueError, scrap.set_mode, 1099)

    def test_scrap_put_text (self):
        scrap.put (pygame.SCRAP_TEXT, "Hello world")
        self.assertEquals (scrap.get (pygame.SCRAP_TEXT), "Hello world")

        scrap.put (pygame.SCRAP_TEXT, "Another String")
        self.assertEquals (scrap.get (pygame.SCRAP_TEXT), "Another String")

    def test_scrap_put_image (self):
        sf = pygame.image.load (
            trunk_relative_path("examples/data/asprite.bmp")
        )
        string = pygame.image.tostring (sf, "RGBA")
        scrap.put (pygame.SCRAP_BMP, string)
        self.assertEquals (scrap.get(pygame.SCRAP_BMP), string)

    def test_put (self):
        scrap.put ("arbitrary buffer", "buf")
        r = scrap.get ("arbitrary buffer")
        self.assertEquals (r, "buf")

if __name__ == '__main__':
    unittest.main()
