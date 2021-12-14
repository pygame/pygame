import unittest
import pygame

class DocsIncludedTest(unittest.TestCase):
    def test_docs_included(self):
        from pygame import docs
        from pygame.docs import util

        self.assertTrue(util.has_local_docs())

if __name__ == "__main__":
    unittest.main()
