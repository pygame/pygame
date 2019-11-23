import unittest
import pygame


"""
SRC: (  0,   0,   0,   0)
DST: (255, 255, 255, 100)
SDL1:  -> (255, 255, 255, 100)
SDL2:  -> ( 99,  99,  99,  99)

SRC: (  0,   0,   0, 255)
DST: (255, 255, 255, 100)
SDL1:  -> (100, 100, 100, 255)
SDL2: -> ( 99,  99,  99, 253)

SRC: (  0,   0,   0,  10)
DST: (255, 255, 255, 100)
SDL1:  -> (100, 100, 100, 107)
SDL2: -> ( 99,  99,  99, 105)


SDL_BLENDMODE_BLEND
alpha blending
dstRGB = (srcRGB * srcA) + (dstRGB * (1-srcA))
dstA = srcA + (dstA * (1-srcA))
"""

class BlitIssueTest(unittest.TestCase):

    # @unittest.skip("causes failures in other tests if run, so skip")
    def test_src_alpha_issue_1289_255(self):
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

    def test_blending_formula(self):
        """

        This tries to replicate the SDL2 blending in python.

        SDL_BLENDMODE_BLEND
        alpha blending
        dstRGB = (srcRGB * srcA) + (dstRGB * (1-srcA))
        dstA = srcA + (dstA * (1-srcA))
        """
        def blend_part(srcRGB, dstRGB, src_a, dst_a):
            srcA = (1.0 / 255) * src_a
            dstA = (1.0 / 255) * dst_a
            return ((srcRGB * srcA) + (dstRGB * dstA))
            return round((srcRGB * srcA) + (dstRGB * dstA))
            return int((srcRGB * srcA) + (dstRGB * dstA))
            # return (srcRGB * srcA) + (dstRGB * (1 - srcA))
            # return (srcRGB * srcA) + (dstRGB * (1 - dstA))

        def blend_a(src_a, dst_a):
            """ dstA = srcA + (dstA * (1-srcA))
            """
            dstA = (1.0 / 255) * dst_a
            srcA = (1.0 / 255) * src_a
            return ((srcA + (dstA * (1 - srcA))) * 255)
            return round((srcA + (dstA * (1 - srcA))) * 255)
            return int((srcA + (dstA * (1 - srcA))) * 255)

        def blend(src, dst):
            src_r, src_g, src_b, src_a = src
            dst_r, dst_g, dst_b, dst_a = dst

            r = blend_part(src_r, dst_r, src_a, dst_a)
            g = blend_part(src_g, dst_g, src_a, dst_a)
            b = blend_part(src_b, dst_b, src_a, dst_a)
            a = blend_a(src_a, dst_a)
            return (r, g, b, a)

        print("")
        src = (0, 0, 0, 0)
        dst = (255, 255, 255, 100)

        print("SRC:", src)
        print("DST:", dst)
        print(blend(src, dst))
        print("SDL1:  -> (255, 255, 255, 100)")
        print("SDL2:  -> ( 99,  99,  99,  99)")
        print("")

        src = (0, 0, 0, 255)
        dst = (255, 255, 255, 100)
        print("SRC:", src)
        print("DST:", dst)
        print(blend(src, dst))
        print("SDL1:  -> (100, 100, 100, 255)")
        print("SDL2: -> ( 99,  99,  99, 253)")
        print("")

        src = (0, 0, 0, 10)
        dst = (255, 255, 255, 100)
        print("SRC:", src)
        print("DST:", dst)
        print(blend(src, dst))
        print("SDL1:  -> (100, 100, 100, 107)")
        print("SDL2: -> ( 99,  99,  99, 105)")
        print("")


if __name__ == "__main__":
    unittest.main()
