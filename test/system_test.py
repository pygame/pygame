import os
import unittest

import pygame


class SystemModuleTest(unittest.TestCase):
    def test_get_cpu_instruction_sets(self):
        instruction_sets = pygame.system.get_cpu_instruction_sets()

        for name in instruction_sets:
            self.assertIsInstance(name, str)
            self.assertIsInstance(instruction_sets[name], bool)

    def test_get_total_ram(self):
        ram = pygame.system.get_total_ram()

        self.assertIsInstance(ram, int)

    def test_get_pref_path(self):
        get_pref_path = pygame.system.get_pref_path

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
            raise FileNotFoundError("pygame.system.get_pref_path folder not created")
        except OSError:  # if the dir isn't empty (shouldn't happen)
            raise OSError("pygame.system.get_pref_path folder already occupied")

        # gets the path, creates the folder, uses kwargs
        path = get_pref_path(org=org, app=app)
        try:  # try removing the folder, it should work fine
            os.rmdir(path)
        except FileNotFoundError:  # if the folder isn't found
            raise FileNotFoundError("pygame.system.get_pref_path folder not created")
        except OSError:  # if the dir isn't empty (shouldn't happen)
            raise OSError("pygame.system.get_pref_path folder already occupied")

    def test_get_pref_locales(self):
        locales = pygame.system.get_pref_locales()

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
            self.assertRaises(TypeError, pygame.system.get_pref_locales, arg)

    def test_get_power_state(self):
        power_state = pygame.system.get_power_state()
        self.assertIsInstance(power_state, (type(None), pygame.system.PowerState))

        expected_types = {
            "battery_seconds": (type(None), int),
            "battery_percent": (type(None), int),
            "on_battery": bool,
            "no_battery": bool,
            "charging": bool,
            "charged": bool,
            "plugged_in": bool,
            "has_battery": bool,
        }

        if power_state is not None:
            for attr_name in expected_types:
                self.assertTrue(hasattr(power_state, attr_name))
                self.assertIsInstance(
                    getattr(power_state, attr_name), expected_types[attr_name]
                )

            self.assertTrue(power_state.plugged_in == (not power_state.on_battery))
            self.assertTrue(
                power_state.has_battery
                == (power_state.on_battery or not power_state.no_battery)
            )

            # There should be only one `True`
            self.assertEqual(
                sum(
                    [
                        power_state.on_battery,
                        power_state.no_battery,
                        power_state.charged,
                        power_state.charging,
                    ]
                ),
                1,
            )


if __name__ == "__main__":
    unittest.main()
