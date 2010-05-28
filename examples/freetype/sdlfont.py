import sys, os
import pygame2
import pygame2.font
import pygame2.examples

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

    font = freetype.Font (pygame2.examples.RESOURCES.get ("sans.ttf"))

    screen = video.set_mode (800, 600)
    screen.fill (colors["grey_light"])

    sf, w, h = font.render("Hello World", colors["red"], colors['grey_dark'],
        ptsize=64, style=ftconstants.STYLE_UNDERLINE|ftconstants.STYLE_ITALIC)
    screen.blit (sf, (32, 32))

    font.render("abcdefghijklm", colors["grey_dark"], colors["green"],
                ptsize=64, dest=(screen, 32, 128))

    font.vertical = True
    font.render("Vertical?", colors["blue"], ptsize=32, dest=(screen, 32, 190))
    font.vertical = False

    font.render("Let's spin!", colors["red"], ptsize=48, rotation=55,
                dest=(screen, 64, 190))

    font.render("All around!", colors["green"], ptsize=48, rotation=-55,
                dest=(screen, 150, 270))

    font.render("and BLEND", pygame2.Color(255, 0, 0, 128), ptsize=64,
                dest=(screen, 250, 220))

    font.render("or BLAND!", pygame2.Color(0, 0xCC, 28, 128), ptsize=64,
                dest=(screen, 258, 237))

    text = "I \u2665 Unicode"
    if sys.version_info[0] < 3:
        text = "I " + unichr(0x2665) + " Unicode"
    font.render(text, pygame2.Color(0, 0xCC, 0xDD), ptsize=64,
                dest=(screen, 298, 320))
    
    text = "\u2665"
    if sys.version_info[0] < 3:
        text = unichr(0x2665)
    font.render(text, colors["grey_light"], colors["red"], ptsize=148,
                dest=(screen, 480, 32))

    font.render("...yes, this is a SDL surface", pygame2.Color(0, 0, 0),
                ptsize=24, style=ftconstants.STYLE_BOLD,
                dest=(screen, 380, 380))

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
