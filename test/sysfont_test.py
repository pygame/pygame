import unittest
import platform
import pygame.sysfont

class SysfontModuleTest(unittest.TestCase):
    FONT = None
    FONTSLIST = []
    PREFERED_FONT = 'Arial'

    pygame.font.init()
    FONTSLIST = pygame.font.get_fonts()
    if PREFERED_FONT in FONTSLIST:
        # Try to use arial rather than random font based on installed fonts on the system.
        FONT = PREFERED_FONT
    else:
        FONT = sorted(FONTSLIST)[0]

    def todo_test_create_aliases(self):
        self.fail()

    def todo_test_initsysfonts(self):
        self.fail()

    @unittest.skipIf('Darwin' not in platform.platform(), 'Not mac we skip.')
    def test_initsysfonts_darwin(self):
        self.assertGreater(len(pygame.sysfont.get_fonts()), 10)

    def todo_test_initsysfonts_unix(self):
        self.fail()

    @unittest.skipIf('Windows' not in platform.system(), 'Not win we skip.')
    def test_initsysfonts_win32(self):
        fonts = pygame.sysfont.initsysfonts_win32()
        self.assertTrue(fonts)
        
    def test_sysfont(self):
        import pygame.font
        pygame.font.init()
        arial = pygame.font.SysFont('Arial', 40)

    def test_get_fonts(self):
        self.assertGreater(len(pygame.sysfont.get_fonts()), 1)

    def test_match_font_known(self):
        font = pygame.sysfont.match_font(self.FONT, 1, 1)
        self.assertTrue(font)
        self.assertIn(".ttf", font)

    def test_match_font_unkown(self):
        font = pygame.sysfont.match_font('1234567890')
        self.assertIsNone(font)

    def test_match_font_none(self):
        self.assertRaises(Exception, pygame.sysfont.match_font, None)

################################################################################

if __name__ == '__main__':
    unittest.main()
