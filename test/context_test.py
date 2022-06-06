import unittest
import os

import pygame


class ContextModuleTest(unittest.TestCase):
    def test_get_pref_path(self):
        get_pref_path = pygame.context.get_pref_path

        """ test the argument validation """
        # wrong arg count
        self.assertRaises(TypeError, get_pref_path, "one arg")
        # wrong arg types
        self.assertRaises(TypeError, get_pref_path, 0, 1)
        # wrong kwargs
        self.assertRaises(TypeError, get_pref_path, norg="test", napp="app")
        # not enough info
        self.assertRaises(TypeError, get_pref_path, "testorg", org="testorg")

        org = "pygame test organization"
        app = "the best app"

        # gets the path, creates the folder
        path = get_pref_path(org, app)
        try:  # try removing the folder, it should work fine
            os.rmdir(path)
        except FileNotFoundError:  # if the folder isn't found
            raise FileNotFoundError("pygame.context.get_pref_path folder not created")
        except OSError:  # if the dir isn't empty (shouldn't happen)
            raise OSError("pygame.context.get_pref_path folder already occupied")

        # gets the path, creates the folder, uses kwargs
        path = get_pref_path(org=org, app=app)
        try:  # try removing the folder, it should work fine
            os.rmdir(path)
        except FileNotFoundError:  # if the folder isn't found
            raise FileNotFoundError("pygame.context.get_pref_path folder not created")
        except OSError:  # if the dir isn't empty (shouldn't happen)
            raise OSError("pygame.context.get_pref_path folder already occupied")

    def test_get_pref_locales(self):
        locales = pygame.context.get_pref_locales()

        # check type of return first
        self.assertIsInstance(locales, list)
        for lc in locales:
            self.assertIsInstance(lc, dict)
            lang = lc["language"]
            self.assertIsInstance(lang, str)

            # length of language code should be greater than 1
            self.assertTrue(len(lang) > 1)

            country = lc["country"]
            if country is not None:
                # country field is optional, but when defined it should be a
                # string
                self.assertIsInstance(country, str)

                # length of country code should be greater than 1
                self.assertTrue(len(country) > 1)

        # passing args should raise error
        for arg in (None, 1, "hello"):
            self.assertRaises(TypeError, pygame.context.get_pref_locales, arg)


if __name__ == "__main__":
    unittest.main()
