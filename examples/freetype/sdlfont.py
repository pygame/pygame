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
    freetype.init ()

    fontdir = os.path.dirname (os.path.abspath (__file__))
    font = freetype.Font (os.path.join (fontdir, "sans.ttf"))

    screen = video.set_mode (300, 200)
    screen.fill (colors["grey_light"])

    font.render("Hello World", colors["red"], None, screen, 32, 32, ptsize=64, style=ftconstants.STYLE_BOLD)
    font.render("do it now with this...", colors["grey_dark"], colors["green"], screen, 32, 128, ptsize=64)
    font.render("Vertical?", colors["blue"], None, screen, 32, 190, ptsize=32, vertical=True)
    font.render("Let's spin!", colors["red"], None, screen, 64, 190, ptsize=48, rotation=55)
    font.render("All around!", colors["green"], None, screen, 150, 270, ptsize=48, rotation=-55)
    font.render("and BLEND", pygame2.Color(255, 0, 0, 128), None, screen, 250, 220, ptsize=64)
    font.render("or BLAND!", pygame2.Color(0, 0xCC, 28, 128), None, screen, 258, 237, ptsize=64)
    font.render("I \u2665 Unicode", pygame2.Color(0, 0xCC, 0xDD), None, screen, 298, 320, ptsize=64)
    font.render("\u2665", colors["grey_light"], colors["red"], screen, 480, 32, ptsize=148)
    font.render("...yes, this is a SDL surface", pygame2.Color(0, 0, 0), None, screen, 380, 380, ptsize=24,
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
