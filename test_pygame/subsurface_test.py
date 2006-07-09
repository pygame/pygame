#!/usr/bin/env python

'''
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

import unittest

import pygame

class SubsurfaceTest(unittest.TestCase):
    def testSubsurface(self):
        s1 = pygame.Surface((100, 100))
        s2 = s1.subsurface((15, 15, 15, 15))
        col = (10, 20, 30, 255)
        s2.set_at((0, 0), col)
        self.assertEqual(s1.get_at((15, 15)), col)

if __name__ == '__main__':
    unittest.main()
