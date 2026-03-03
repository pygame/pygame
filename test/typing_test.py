import unittest

# A basic unit test to test that pygame.typing can import in python code, and
# the documented attributes are accessible.
# More rigorous testing needs mypy and is therefore implemented in the stubs
# directory.
TYPING_PUBLIC_ATTRS = [
    "RectLike",
    "SequenceLike",
    "FileLike",
    "ColorLike",
    "Point",
    "IntPoint",
]


class TypingTest(unittest.TestCase):
    def test_typing_has_attrs(self):
        import pygame.typing

        for i in TYPING_PUBLIC_ATTRS:
            self.assertTrue(hasattr(pygame.typing, i))


if __name__ == "__main__":
    unittest.main()
