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
    

def run():
    video.init ()
    freetype.init ()

    fontdir = os.path.dirname (os.path.abspath (__file__))
    font = freetype.Font (os.path.join (fontdir, "sans.ttf"))

    screen = video.set_mode (640, 480)
    screen.fill (pygame2.Color (200, 200, 200))
    w, h, _ = font.render("Hello", pygame2.Color(100, 100, 100), None, screen, 100, 100, ptsize=24)

    w, h, _ = font.render("Hello qjky", pygame2.Color(100, 100, 100), None, screen, 100, 200, ptsize=48)

#    w, g, buf = font.render("Hello World, Jay",
#            pygame2.Color(100, 200, 32),
#            pygame2.Color(240, 40, 100),
#            None, ptsize = 32)

#    screen.blit (buf, (5, 100))
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
