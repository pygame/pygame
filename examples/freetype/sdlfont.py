import sys, os
import pygame2
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

    screen = video.set_mode (320, 240)
    screen.fill (pygame2.Color (255, 255, 255))
    w, h, buf = font.render ("Hello", 12, ftconstants.RENDER_NEWSURFACE)
    print (type (buf), buf)
    screen.blit (buf, (5, 5))
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
