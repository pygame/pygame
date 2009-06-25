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
    font1 = freetype.Font (os.path.join (fontdir, "sans.ttf"))
    font2 = freetype.Font (pygame2.font.find_font ("times", ftype="ttf")[0])

    screen = video.set_mode (320, 240)
    screen.fill (pygame2.Color (255, 255, 255))
    w, h, buf = font1.render ("Hello", 12, screen, xpos=10, ypos=10,
                              fgcolor=pygame2.Color(100, 100, 100))

    w, h, buf = font2.render ("Hello", 28, screen, xpos=10, ypos=40,
                              bgcolor=pygame2.Color(100, 100, 100))
    #screen.blit (buf, (5, 5))
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
