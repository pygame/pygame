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
    font = freetype.Font(os.path.join (fontdir, "data", "sans.ttf"))

    screen = pygame.display.set_mode((800, 600))
    screen.fill (colors["grey_light"])

    font.underline_adjustment = 0.5
    font.pad = True
    font.render_to(screen, (32, 32), "Hello World", colors["red"],
                   colors['grey_dark'], size=64,
                   style=freetype.STYLE_UNDERLINE|freetype.STYLE_OBLIQUE)
    font.pad = False

    font.render_to(screen, (32, 128), "abcdefghijklm", colors["grey_dark"],
                   colors["green"], size=64)

    font.vertical = True
    font.render_to(screen, (32, 200), "Vertical?", colors["blue"],
                   None, size=32)
    font.vertical = False

    font.render_to(screen, (64, 190), "Let's spin!", colors["red"],
                   None, size=48, rotation=55)

    font.render_to(screen, (160, 290), "All around!", colors["green"],
                   None, size=48, rotation=-55)

    font.render_to(screen, (250, 220), "and BLEND",
                   pygame.Color(255, 0, 0, 128), None, size=64)

    font.render_to(screen, (265, 237), "or BLAND!",
                   pygame.Color(0, 0xCC, 28, 128), None, size=64)

    # Some pinwheels
    font.origin = True
    for angle in range(0, 360, 45):
        font.render_to(screen, (150, 420), ")", pygame.Color('black'),
                       size=48, rotation=angle)
    font.vertical = True
    for angle in range(15, 375, 30):
        font.render_to(screen, (600, 400), "|^*", pygame.Color('orange'),
                       size=48, rotation=angle)
    font.vertical = False
    font.origin = False

    utext = pygame.compat.as_unicode(r"I \u2665 Unicode")
    font.render_to(screen, (298, 320), utext, pygame.Color(0, 0xCC, 0xDD),
                   None, size=64)

    utext = pygame.compat.as_unicode(r"\u2665")
    font.render_to(screen, (480, 32), utext, colors["grey_light"],
                   colors["red"], size=148)

    font.render_to(screen, (380, 380), "...yes, this is an SDL surface",
                   pygame.Color(0, 0, 0),
                   None, size=24, style=freetype.STYLE_STRONG)

    font.origin = True
    r = font.render_to(screen, (100, 530), "stretch",
                   pygame.Color('red'),
                   None, size=(24, 24), style=freetype.STYLE_NORMAL)
    font.render_to(screen, (100 + r.width, 530), " VERTICAL",
                   pygame.Color('red'),
                   None, size=(24, 48), style=freetype.STYLE_NORMAL)

    r = font.render_to(screen, (100, 580), "stretch",
                   pygame.Color('blue'),
                   None, size=(24, 24), style=freetype.STYLE_NORMAL)
    font.render_to(screen, (100 + r.width, 580), " HORIZONTAL",
                   pygame.Color('blue'),
                   None, size=(48, 24), style=freetype.STYLE_NORMAL)

    pygame.display.flip()

    while 1:
        if pygame.event.wait().type in (QUIT, KEYDOWN, MOUSEBUTTONDOWN):
            break

    pygame.quit()

if __name__ == "__main__":
    run ()
