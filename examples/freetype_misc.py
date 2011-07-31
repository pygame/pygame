import sys, os
import pygame
from pygame.locals import *

try:
    import pygame.freetype as freetype
except ImportError:
    print ("No FreeType support compiled")
    sys.exit ()

colors = {
    "grey_light"    :   pygame.Color(200, 200, 200),
    "grey_dark"     :   pygame.Color(100, 100, 100),
    "green"         :   pygame.Color(50, 255, 63),
    "red"           :   pygame.Color(220, 30, 30),
    "blue"          :   pygame.Color(50, 75, 245)
}

def run():
    pygame.init()

    fontdir = os.path.dirname(os.path.abspath (__file__))
    face = freetype.Face(os.path.join (fontdir, "data", "sans.ttf"))

    screen = pygame.display.set_mode((800, 600))
    screen.fill (colors["grey_light"])

    face.underline_adjustment = 0.5
    face.pad = True
    face.render((screen, 32, 32), "Hello World", colors["red"], colors['grey_dark'],
            ptsize=64, style=freetype.STYLE_UNDERLINE|freetype.STYLE_OBLIQUE)
    face.pad = False

    face.render((screen, 32, 128), "abcdefghijklm", colors["grey_dark"], colors["green"],
            ptsize=64)

    face.vertical = True
    face.render((screen, 32, 200), "Vertical?", colors["blue"], None, ptsize=32)
    face.vertical = False

    face.render((screen, 64, 190), "Let's spin!", colors["red"], None,
            ptsize=48, rotation=55)

    face.render((screen, 160, 290), "All around!", colors["green"], None,
            ptsize=48, rotation=-55)

    face.render((screen, 250, 220), "and BLEND", pygame.Color(255, 0, 0, 128), None,
            ptsize=64)

    face.render((screen, 265, 237), "or BLAND!", pygame.Color(0, 0xCC, 28, 128), None,
            ptsize=64)

    face.origin = True
    for angle in range(0, 360, 45):
        face.render((screen, 200, 500), ")", pygame.Color('black'),
                    ptsize=48, rotation=angle)
    face.vertical = True
    for angle in range(15, 375, 30):
        face.render((screen, 600, 400), "|^*", pygame.Color('orange'),
                    ptsize=48, rotation=angle)
    face.vertical = False
    face.origin = False

    utext = pygame.compat.as_unicode(r"I \u2665 Unicode")
    face.render((screen, 298, 320), utext, pygame.Color(0, 0xCC, 0xDD), None,
            ptsize=64)

    utext = pygame.compat.as_unicode(r"\u2665")
    face.render((screen, 480, 32), utext, colors["grey_light"], colors["red"],
            ptsize=148)

    face.render((screen, 380, 380), "...yes, this is an SDL surface", pygame.Color(0, 0, 0), None,
            ptsize=24, style=freetype.STYLE_STRONG)

    pygame.display.flip()

    while 1:
        if pygame.event.wait().type in (QUIT, KEYDOWN, MOUSEBUTTONDOWN):
            break

    pygame.quit()

if __name__ == "__main__":
    run ()
