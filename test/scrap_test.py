import os
import sys
if os.environ.get('SDL_VIDEODRIVER') == 'dummy':
    __tags__ = ('ignore', 'subprocess_ignore')
import unittest
from pygame.tests.test_utils import trunk_relative_path

import pygame
from pygame import scrap
from pygame.compat import as_bytes

class ScrapModuleTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pygame.init()
        pygame.display.set_mode((1, 1))
        scrap.init()

    @classmethod
    def tearDownClass(cls):
        # scrap.quit()  # Does not exist!
        pygame.quit()

    def test_init(self):
        # Test if module initialized after multiple init() calls.
        scrap.init()
        scrap.init()

        self.assertTrue(scrap.get_init())

    def test_get_init(self):
        # Test if get_init() gets the init state.
        self.assertTrue(scrap.get_init())

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
        scrap.put (pygame.SCRAP_TEXT, as_bytes("Hello world"))
        self.assertEquals (scrap.get (pygame.SCRAP_TEXT),
                           as_bytes("Hello world"))

        scrap.put (pygame.SCRAP_TEXT, as_bytes("Another String"))
        self.assertEquals (scrap.get (pygame.SCRAP_TEXT),
                           as_bytes("Another String"))

    def test_scrap_put_image (self):
        if 'pygame.image' not in sys.modules:
            return
        sf = pygame.image.load (
            trunk_relative_path("examples/data/asprite.bmp")
        )
        string = pygame.image.tostring (sf, "RGBA")
        scrap.put (pygame.SCRAP_BMP, string)
        self.assertEquals (scrap.get(pygame.SCRAP_BMP), string)

    def test_put (self):
        scrap.put ("arbitrary buffer", as_bytes("buf"))
        r = scrap.get ("arbitrary buffer")
        self.assertEquals (r, as_bytes("buf"))

class X11InteractiveTest(unittest.TestCase):
    __tags__ = ['ignore', 'subprocess_ignore']
    try:
        pygame.display.init()
    except Exception:
        pass
    else:
        if pygame.display.get_driver() == 'x11':
            __tags__ = ['interactive']
        pygame.display.quit()

    def test_issue_208(self):
        """PATCH: pygame.scrap on X11, fix copying into PRIMARY selection

           Copying into theX11 PRIMARY selection (mouse copy/paste) would not
           work due to a confusion between content type and clipboard type.

        """

        from pygame import display, event, freetype
        from pygame.locals import SCRAP_SELECTION, SCRAP_TEXT
        from pygame.locals import KEYDOWN, K_y, QUIT

        success = False
        freetype.init()
        font = freetype.Font(None, 24)
        display.init()
        display.set_caption("Interactive X11 Paste Test")
        screen = display.set_mode((600, 200))
        screen.fill(pygame.Color('white'))
        text = "Scrap put() succeeded."
        msg = ('Some text has been placed into the X11 clipboard.'
               ' Please click the center mouse button in an open'
               ' text window to retrieve it.'
               '\n\nDid you get "{}"? (y/n)').format(text)
        word_wrap(screen, msg, font, 6)
        display.flip()
        event.pump()
        scrap.init()
        scrap.set_mode(SCRAP_SELECTION)
        scrap.put(SCRAP_TEXT, text.encode('UTF-8'))
        while True:
            e = event.wait()
            if e.type == QUIT:
                break
            if e.type == KEYDOWN:
                success = (e.key == K_y)
                break
        pygame.display.quit()
        self.assertTrue(success)

def word_wrap(surf, text, font, margin=0, color=(0, 0, 0)):
    font.origin = True
    surf_width, surf_height = surf.get_size()
    width = surf_width - 2 * margin
    height = surf_height - 2 * margin
    line_spacing = int(1.25 * font.get_sized_height())
    x, y = margin, margin + line_spacing
    space = font.get_rect(' ')
    for word in iwords(text):
        if word == '\n':
            x, y = margin, y + line_spacing
        else:
            bounds = font.get_rect(word)
            if x + bounds.width + bounds.x >= width:
                x, y = margin, y + line_spacing
            if x + bounds.width + bounds.x >= width:
                raise ValueError("word too wide for the surface")
            if y + bounds.height - bounds.y >= height:
                raise ValueError("text to long for the surface")
            font.render_to(surf, (x, y), None, color)
            x += bounds.width + space.width
    return x, y

def iwords(text):
    #  r"\n|[^ ]+"
    #
    head = 0
    tail = head
    end = len(text)
    while head < end:
        if text[head] == ' ':
            head += 1
            tail = head + 1
        elif text[head] == '\n':
            head += 1
            yield '\n'
            tail = head + 1
        elif tail == end:
            yield text[head:]
            head = end
        elif text[tail] == '\n':
            yield text[head:tail]
            head = tail
        elif text[tail] == ' ':
            yield text[head:tail]
            head = tail
        else:
            tail += 1

if __name__ == '__main__':
    unittest.main()
