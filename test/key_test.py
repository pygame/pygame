import unittest
import pygame


class KeyTest (unittest.TestCase):

    def test_import(self):
        'does it import'
        import pygame.key


    def test_get_repeat(self):
        delay, interval = pygame.key.get_repeat()

    def test_add_more_tests(self):
        'we need to add more tests'
        pass
        #raise NotImplementedError("TODO: key tests need improving.")



if __name__ == '__main__':
    unittest.main()
