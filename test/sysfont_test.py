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

    def test_sysfont(self):
        import pygame.font
        pygame.font.init()
        arial = pygame.font.SysFont('Arial', 40)

    def todo_test_initsysfonts_unix(self):
        self.fail()

    def test_get_fonts(self):
        import pygame.sysfont
        self.assertTrue(len(pygame.sysfont.get_fonts()) > 1)

    @unittest.skipIf('Windows' not in platform.system(), 'Not win we skip.')
    def test_initsysfonts_win32(self):
        import pygame.sysfont
        fonts = pygame.sysfont.initsysfonts_win32()
        self.assertTrue(fonts)

################################################################################

if __name__ == '__main__':
    unittest.main()
