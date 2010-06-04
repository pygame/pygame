import unittest

from pygame2.base import Font

class TestFont(Font):
    def __init__(self):
        Font.__init__(self)
        self._height = 11
        self._style = "No style"
        self._name = "Arial Black"

    def render(self, **kwds):
        return "rendered"

    def copy(self):
        return TestFont()

    def get_size(self, text):
        return len(text)

    def setstyle(self, style):
        self._style = style

    height = property(lambda self: self._height)
    style = property(lambda self: self._style, setstyle)
    name = property(lambda self: self._name)

class BrokenTestFont(Font):
    def __init__(self):
        Font.__init__(self)
        self._height = 11

    def copy(self):
        return TestFont()

    height = property(lambda self: self._height)


class FontTest(unittest.TestCase):

    def test_pygame2_base_Font_copy(self):

        # __doc__ (as of 2009-05-16) for pygame2.base.Font.copy:

        # copy () -> Font
        # 
        # Creates a copy of this Font.

        font1 = TestFont()
        font2 = font1.copy()
        self.assertEqual(font1.height, font2.height)

        font1 = BrokenTestFont()
        font2 = font1.copy()
        self.assertEqual(font1.height, font2.height)

    def test_pygame2_base_Font_get_size(self):

        # __doc__ (as of 2009-05-16) for pygame2.base.Font.get_size:

        # Gets the width and height of the Font typography.

        font = TestFont()
        self.assertEqual(font.get_size("test string"), len("test string"))

        font = BrokenTestFont()
        self.assertRaises(NotImplementedError, font.get_size, "test string")

    def test_pygame2_base_Font_height(self):

        # __doc__ (as of 2009-05-16) for pygame2.base.Font.height:

        # Gets the standard height of the Font typography.

        font = TestFont()
        self.assertEqual(font.height, 11)

        font = BrokenTestFont()
        self.assertEqual(font.height, 11)

    def test_pygame2_base_Font_name(self):

        # __doc__ (as of 2009-05-16) for pygame2.base.Font.name:

        # Gets the name of the loaded Font.

        font = TestFont()
        self.assertEqual(font.name, "Arial Black")

        font = BrokenTestFont()
        self.assertRaises(NotImplementedError, getattr, font, "name")

    def test_pygame2_base_Font_render(self):

        # __doc__ (as of 2009-05-16) for pygame2.base.Font.render:

        # render (**kwds) -> object
        # 
        # Renders the specified text using the Font object. The exact input
        # arguments and return value is dependand on each specific Font
        # implementation.

        font = TestFont()
        self.assertEqual(font.render(), "rendered")

        font = BrokenTestFont()
        self.assertRaises(NotImplementedError, font.render)

    def test_pygame2_base_Font_style(self):

        # __doc__ (as of 2009-05-16) for pygame2.base.Font.style:

        # Gets or sets the style used to render the Font.

        font = TestFont()
        self.assertEqual(font.style, "No style")
        font.style = "Bold"
        self.assertEqual(font._style, "Bold")

        font = BrokenTestFont()
        self.assertRaises(NotImplementedError, getattr, font, "style")

    def test_pygame2_base_Font___repr__(self):
        font = TestFont()
        text = "<Generic Font>"
        self.assertEqual (repr (font), text)

if __name__ == '__main__':
    unittest.main()
