import unittest
import pygame.constants


class KmodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.constants = [
            'KMOD_NONE',
            'KMOD_LSHIFT',
            'KMOD_RSHIFT',
            'KMOD_LCTRL',
            'KMOD_RCTRL',
            'KMOD_LALT',
            'KMOD_RALT',
            'KMOD_LMETA',
            'KMOD_RMETA',
            'KMOD_NUM',
            'KMOD_CAPS',
            'KMOD_MODE',
            'KMOD_CTRL',
            'KMOD_SHIFT',
            'KMOD_ALT',
            'KMOD_META',
        ]
        if pygame.get_sdl_version()[0] >= 2:
            cls.constants.extend([
                'KMOD_LGUI',
                'KMOD_RGUI',
                'KMOD_GUI',
            ])

    def test_kmod_existence(self):
        for k in self.constants:
            self.assertTrue(hasattr(pygame.constants, k), 'missing constant {}'.format(k))

    def test_kmod_types(self):
        for k in self.constants:
            self.assertEqual(type(getattr(pygame.constants, k)), int)

class KeyConstantTests(unittest.TestCase):
    def test_letters(self):
        for c in range(ord('a'), ord('z') + 1):
            c = chr(c)
            self.assertTrue(hasattr(pygame.constants, 'K_%s' % c),
                                    'missing constant: K_%s' % c)

################################################################################

if __name__ == '__main__':
    unittest.main()
