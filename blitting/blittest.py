"""Exercise the alphablit module

This runs self blit tests on alphablit to check for invalid memory
accesses and that the self blit check in SoftBlitPyGame works properly.

"""

# Requires Python 2.5

import alphablit
from alphablit import (SDL_Surface, pygame_AlphaBlit, SDL_SRCALPHA,
                       SDL_Rect, PYGAME_BLEND_RGBA_ADD)

def testcase(bpp, w, sx, sy, w2, h2, dx, dy, do_overlap):
    surf = SDL_Surface(bpp, w, 300, flags=SDL_SRCALPHA)
    pygame_AlphaBlit(surf, SDL_Rect(sx, sy, w2, h2),
                     surf, SDL_Rect(dx, dy, w2, h2),
                     PYGAME_BLEND_RGBA_ADD)
    if alphablit.was_reversed:
        if do_overlap:
            return
    else:
        if not do_overlap:
            return
    # fail
    if do_overlap:
        print "** False negative for"
    else:
        print "** False positive for"
    print "   parent width %i," % w
    print "   source rect (%i, %i, %i, %i)," % (sx, sy, w2, h2)
    print "   destination rect (%i, %i, %i, %i)" % (dx, dy, w2, h2)

def test():
    for bpp in [2, 4]:
        # dstpx < srcpx
        testcase(bpp, 51, 1, 0, 50, 50, 0, 0, False)
        # dstpx == srcpx
        testcase(bpp, 50, 0, 0, 50, 50, 0, 0, False)
        # srcpx < dstpx < srcpx + rect_width * BytesPerPixel
        testcase(bpp, 51, 0, 0, 50, 50, 1, 0, True)
        # dst rect outside src rect
        testcase(bpp, 100, 0, 0, 40, 40, 50, 0, False) #
        testcase(bpp, 100, 0, 0, 40, 40, 50, 48, False)
        # dst rect next to src rect
        testcase(bpp, 100, 0, 0, 50, 50, 50, 0, False) #
        testcase(bpp, 100, 0, 0, 50, 50, 50, 10, False) #
        # dst rect overlaps src rect on right
        testcase(bpp, 100, 0, 0, 50, 50, 49, 1, True)
        testcase(bpp, 100, 0, 0, 50, 50, 49, 49, True)
        # dst rect overlaps src rect on left
        testcase(bpp, 100, 49, 0, 50, 50, 0, 1, True)
        testcase(bpp, 100, 49, 0, 50, 50, 0, 49, True)
        # last pixel of src is first pixel of dst
        testcase(bpp, 100, 0, 0, 40, 40, 39, 39, True)
        # start of dst is just pass source
        testcase(bpp, 100, 0, 0, 40, 40, 39, 40, False)


if __name__ == '__main__':
    print "Go"
    test()
    print "End"
