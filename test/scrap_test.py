import os
import platform
import sys
import unittest

import pygame
from pygame import scrap
from pygame.tests.test_utils import trunk_relative_path

if os.environ.get("SDL_VIDEODRIVER") == pygame.NULL_VIDEODRIVER:
    __tags__ = ("ignore", "subprocess_ignore")


class ScrapModuleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.display.init()
        pygame.display.set_mode((1, 1))
        scrap.init()

    @classmethod
    def tearDownClass(cls):
        # scrap.quit()  # Does not exist!
        pygame.display.quit()

    def test_init(self):
        """Ensures scrap module still initialized after multiple init calls."""
        scrap.init()
        scrap.init()

        self.assertTrue(scrap.get_init())

    def test_init__reinit(self):
        """Ensures reinitializing the scrap module doesn't clear its data."""
        data_type = pygame.SCRAP_TEXT
        expected_data = b"test_init__reinit"
        scrap.put(data_type, expected_data)

        scrap.init()

        self.assertEqual(scrap.get(data_type), expected_data)

    def test_get_init(self):
        """Ensures get_init gets the init state."""
        self.assertTrue(scrap.get_init())

    def todo_test_contains(self):
        """Ensures contains works as expected."""
        self.fail()

    def todo_test_get(self):
        """Ensures get works as expected."""
        self.fail()

    @unittest.skipIf(
        platform.system() != "Windows",
        "scrap features are broken on non windows platforms",
    )
    def test_get__owned_empty_type(self):
        """Ensures get works when there is no data of the requested type
        in the clipboard and the clipboard is owned by the pygame application.
        """
        # Use a unique data type identifier to ensure there is no preexisting
        # data.
        DATA_TYPE = "test_get__owned_empty_type"

        if scrap.lost():
            # Try to acquire the clipboard.
            scrap.put(pygame.SCRAP_TEXT, b"text to clipboard")

            if scrap.lost():
                self.skipTest("requires the pygame application to own the clipboard")

        data = scrap.get(DATA_TYPE)

        self.assertIsNone(data)

    def todo_test_get_types(self):
        """Ensures get_types works as expected."""
        self.fail()

    def todo_test_lost(self):
        """Ensures lost works as expected."""
        self.fail()

    def test_set_mode(self):
        """Ensures set_mode works as expected."""
        scrap.set_mode(pygame.SCRAP_SELECTION)
        scrap.set_mode(pygame.SCRAP_CLIPBOARD)

        self.assertRaises(ValueError, scrap.set_mode, 1099)

    def test_put__text(self):
        """Ensures put can place text into the clipboard."""
        scrap.put(pygame.SCRAP_TEXT, b"Hello world")

        self.assertEqual(scrap.get(pygame.SCRAP_TEXT), b"Hello world")

        scrap.put(pygame.SCRAP_TEXT, b"Another String")

        self.assertEqual(scrap.get(pygame.SCRAP_TEXT), b"Another String")

    @unittest.skipIf(
        platform.system() != "Windows",
        "scrap features are broken on non windows platforms",
    )
    def test_put__bmp_image(self):
        """Ensures put can place a BMP image into the clipboard."""
        sf = pygame.image.load(trunk_relative_path("examples/data/asprite.bmp"))
        expected_bytes = pygame.image.tobytes(sf, "RGBA")
        scrap.put(pygame.SCRAP_BMP, expected_bytes)

        self.assertEqual(scrap.get(pygame.SCRAP_BMP), expected_bytes)

    @unittest.skipIf(
        platform.system() != "Windows",
        "scrap features are broken on non windows platforms",
    )
    def test_put(self):
        """Ensures put can place data into the clipboard
        when using a user defined type identifier.
        """
        DATA_TYPE = "arbitrary buffer"

        scrap.put(DATA_TYPE, b"buf")
        r = scrap.get(DATA_TYPE)

        self.assertEqual(r, b"buf")

    def test_new_put(self):
        """Ensures that text can be placed into the clipboard"""
        text = "Welcome to scrap!"
        scrap.put_text(text)
        self.assertTrue(scrap.has_text())
        self.assertEqual(scrap.get_text(), text)

    def test_has_text_when_empty(self):
        """Ensures that has_text returns False when clipboard is empty"""
        scrap.put_text("")
        self.assertFalse(scrap.has_text())
        self.assertEqual(scrap.get_text(), "")


class ScrapModuleClipboardNotOwnedTest(unittest.TestCase):
    """Test the scrap module's functionality when the pygame application is
    not the current owner of the clipboard.

    A separate class is used to prevent tests that acquire the clipboard from
    interfering with these tests.
    """

    @classmethod
    def setUpClass(cls):
        pygame.display.init()
        pygame.display.set_mode((1, 1))
        scrap.init()

    @classmethod
    def tearDownClass(cls):
        # scrap.quit()  # Does not exist!
        pygame.quit()
        pygame.display.quit()

    def _skip_if_clipboard_owned(self):
        # Skip test if the pygame application owns the clipboard. Currently,
        # there is no way to give up ownership.
        if not scrap.lost():
            self.skipTest("requires the pygame application to not own the clipboard")

    def test_get__not_owned(self):
        """Ensures get works when there is no data of the requested type
        in the clipboard and the clipboard is not owned by the pygame
        application.
        """
        self._skip_if_clipboard_owned()

        # Use a unique data type identifier to ensure there is no preexisting
        # data.
        DATA_TYPE = "test_get__not_owned"

        data = scrap.get(DATA_TYPE)

        self.assertIsNone(data)

    def test_get_types__not_owned(self):
        """Ensures get_types works when the clipboard is not owned
        by the pygame application.
        """
        self._skip_if_clipboard_owned()

        data_types = scrap.get_types()

        self.assertIsInstance(data_types, list)

    def test_contains__not_owned(self):
        """Ensures contains works when the clipboard is not owned
        by the pygame application.
        """
        self._skip_if_clipboard_owned()

        # Use a unique data type identifier to ensure there is no preexisting
        # data.
        DATA_TYPE = "test_contains__not_owned"

        contains = scrap.contains(DATA_TYPE)

        self.assertFalse(contains)

    def test_lost__not_owned(self):
        """Ensures lost works when the clipboard is not owned
        by the pygame application.
        """
        self._skip_if_clipboard_owned()

        lost = scrap.lost()

        self.assertTrue(lost)


if __name__ == "__main__":
    unittest.main()
