import unittest
import pygame

import os


class DocsIncludedTest(unittest.TestCase):
    @unittest.skipIf("CI" not in os.environ, "Docs not required for local builds")
    def test_docs_included(self):
        from pygame import docs
        from pygame.docs import util

        self.assertTrue(util.has_local_docs())


if __name__ == "__main__":
    unittest.main()
