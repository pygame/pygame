import unittest
import pygame


"""
SRC: (255, 255, 255, 100)
DST: (  0,   0,   0,   0)
SDL1:  -> (255, 255, 255, 100)
SDL2:  -> ( 99,  99,  99,  99)

SRC: (255, 255, 255, 100)
DST: (  0,   0,   0, 255)
SDL1:  -> (100, 100, 100, 255)
SDL2: -> ( 99,  99,  99, 253)

SRC: (255, 255, 255, 100)
DST: (  0,   0,   0,  10)
SDL1:  -> (100, 100, 100, 107)
SDL2: -> ( 99,  99,  99, 105)

"""

class BlitIssueTest(unittest.TestCase):

    # @unittest.skip("causes failures in other tests if run, so skip")
    def test_src_alpha_issue_1289(self):
        """ blit should be white.
        """
        surf1 = pygame.Surface((1, 1), pygame.SRCALPHA, 32)
        surf1.fill((255, 255, 255, 100))

        surf2 = pygame.Surface((1, 1), pygame.SRCALPHA, 32)
        surf2.fill((0, 0, 0, 0))
        self.assertEqual(surf2.get_at((0, 0)), (0, 0, 0, 0))
        surf2.blit(surf1, (0, 0))

        self.assertEqual(surf1.get_at((0, 0)), (255, 255, 255, 100))
        self.assertEqual(surf2.get_at((0, 0)), (255, 255, 255, 100))
        # SDL2 result:                         ( 99,  99,  99,  99)

        # (255, 255, 255, 100)   (0, 0, 0, 0)
        # SDL1:  -> (255, 255, 255, 100)
        # SDL2:  -> ( 99,  99,  99,  99)



    def test_src_alpha_issue_1289_2(self):
        """ this test is similar.

        Except it sets the alpha to 255 on surf2.
        """
        surf1 = pygame.Surface((1, 1), pygame.SRCALPHA, 32)
        surf1.fill((255, 255, 255, 100))

        surf2 = pygame.Surface((1, 1), pygame.SRCALPHA, 32)
        surf2.fill((0, 0, 0, 255))
        self.assertEqual(surf2.get_at((0, 0)), (0, 0, 0, 255))
        surf2.blit(surf1, (0, 0))

        self.assertEqual(surf1.get_at((0, 0)), (255, 255, 255, 100))
        self.assertEqual(surf2.get_at((0, 0)), (100, 100, 100, 255))
        # SDL2 result:                         ( 99,  99,  99, 253)

        # (255, 255, 255, 100) (0, 0, 0, 255)
        # SDL1:  -> (100, 100, 100, 255)
        # SDL2: -> ( 99,  99,  99, 253)


    def test_src_alpha_issue_1289_3(self):
        """ this test is similar.

        Except it sets the alpha to 10 on surf2.
        """
        surf1 = pygame.Surface((1, 1), pygame.SRCALPHA, 32)
        surf1.fill((255, 255, 255, 100))

        surf2 = pygame.Surface((1, 1), pygame.SRCALPHA, 32)
        surf2.fill((0, 0, 0, 10))
        self.assertEqual(surf2.get_at((0, 0)), (0, 0, 0, 10))
        surf2.blit(surf1, (0, 0))

        self.assertEqual(surf1.get_at((0, 0)), (255, 255, 255, 100))
        self.assertEqual(surf2.get_at((0, 0)), (100, 100, 100, 107))
        # SDL2 result                          ( 99,  99,  99, 105)

        # (255, 255, 255, 100) (0, 0, 0, 10)
        # SDL1:  -> (100, 100, 100, 107)
        # SDL2: -> ( 99,  99,  99, 105)


if __name__ == "__main__":
    unittest.main()
