import sys, os
import pygame2
import pygame2.font
try:
    import pygame2.sdl.constants as constants
    import pygame2.sdl.event as event
    import pygame2.sdl.video as video
except ImportError:
    print ("No pygame2.sdl support")
    sys.exit ()

try:
    import pygame2.freetype as freetype
    import pygame2.freetype.constants as ftconstants
except ImportError:
    print ("No pygame2.freetype support")
    sys.exit ()

colors = {
    "grey_light"    :   pygame2.Color(200, 200, 200),
    "grey_dark"     :   pygame2.Color(100, 100, 100),
    "green"         :   pygame2.Color(50, 255, 63),
    "red"           :   pygame2.Color(220, 30, 30),
    "blue"          :   pygame2.Color(50, 75, 245)
}

def run():
    video.init ()
    freetype.init (8)

    fontdir = os.path.dirname (os.path.abspath (__file__))
    font = freetype.Font (os.path.join (fontdir, "sans.ttf"))

    screen = video.set_mode (800, 600)
    screen.fill (colors["grey_light"])

    w,h, sf = font.render(None, "Hello World", colors["red"],
                colors['grey_dark'], ptsize=64,
                style=ftconstants.STYLE_UNDERLINE|ftconstants.STYLE_ITALIC)
    print screen.format.bits_per_pixel, screen.format.masks, screen.flags
    screen.blit (sf, (32, 32))

    font.render((screen, 32, 128), "abcdefghijklm", colors["grey_dark"],
                colors["green"], ptsize=64)

    font.vertical = True
    font.render((screen, 32, 190), "Vertical?", colors["blue"], None, ptsize=32)
    font.vertical = False

    font.render((screen, 64, 190), "Let's spin!", colors["red"], None,
                ptsize=48, rotation=55)

    font.render((screen, 150, 270), "All around!", colors["green"], None,
                ptsize=48, rotation=-55)

    font.render((screen, 250, 220), "and BLEND", pygame2.Color(255, 0, 0, 128),
                None, ptsize=64)

    font.render((screen, 258, 237), "or BLAND!",
                pygame2.Color(0, 0xCC, 28, 128), None, ptsize=64)

    font.render((screen, 298, 320), "I \u2665 Unicode",
                pygame2.Color(0, 0xCC, 0xDD), None, ptsize=64)

    font.render((screen, 480, 32), "\u2665", colors["grey_light"],
                colors["red"], ptsize=148)

    font.render((screen, 380, 380), "...yes, this is a SDL surface",
                pygame2.Color(0, 0, 0), None,ptsize=24,
                style=ftconstants.STYLE_BOLD)

    screen.flip ()

    okay = True
    while okay:
        for ev in event.get ():
            if ev.type == constants.QUIT:
                okay = False
            if ev.type == constants.KEYDOWN and ev.key == constants.K_ESCAPE:
                okay = False
    video.quit ()

if __name__ == "__main__":
    run ()
