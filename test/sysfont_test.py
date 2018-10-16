import unittest
import platform

class SysfontModuleTest(unittest.TestCase):
    def todo_test_create_aliases(self):
        self.fail()

    def todo_test_initsysfonts(self):
        self.fail()

    @unittest.skipIf('Darwin' not in platform.platform(), 'Not mac we skip.')
    def test_initsysfonts_darwin(self):
        import pygame.sysfont
        self.assertTrue(len(pygame.sysfont.get_fonts()) > 10)

    def todo_test_initsysfonts_unix(self):
        self.fail()

    def todo_test_initsysfonts_win32(self):
        self.fail()

################################################################################

if __name__ == '__main__':
    unittest.main()
