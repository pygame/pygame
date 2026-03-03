import os
import unittest
from importlib.metadata import version

import pygame

pg_header = os.path.join("src_c", "include", "_pygame.h")


class VersionTest(unittest.TestCase):
    def test_vernum_obj(self):
        major, minor, patch = map(int, pygame.version.ver.split(".")[:3])
        self.assertEqual(pygame.version.vernum.major, major)
        self.assertEqual(pygame.version.vernum[0], major)
        self.assertEqual(pygame.version.vernum.minor, minor)
        self.assertEqual(pygame.version.vernum[1], minor)
        self.assertEqual(pygame.version.vernum.patch, patch)
        self.assertEqual(pygame.version.vernum[2], patch)

    def test_sdl_version(self):
        self.assertEqual(len(pygame.version.SDL), 3)
        self.assertEqual(tuple(pygame.version.SDL), pygame.get_sdl_version())

    def test_installed_version_and_dunder(self):
        self.assertEqual(pygame.__version__, pygame.version.ver)
        self.assertEqual(pygame.__version__, version("pygame"))


if __name__ == "__main__":
    unittest.main()
