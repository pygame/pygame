import sys, os
import pygame2
import pygame2.examples
try:
    import pygame2.sdl.constants as sdlconst
    import pygame2.sdl.event as event
    import pygame2.sdl.video as video
    import pygame2.sdl.mouse as mouse
    import pygame2.sdl.image as image
    import pygame2.sdl.wm as wm
    from pygame2.sdlext.font import BitmapFont
except ImportError:
    print ("No pygame2.sdl support")
    sys.exit ()

def run ():
    white = pygame2.Color (255, 255, 255)
    black = pygame2.Color (0, 0, 0)

    fontmap = [ "0123456789",
                "ABCDEFGHIJ",
                "KLMNOPQRST",
                "UVWXYZ    ",
                "abcdefghij",
                "klmnopqrst",
                "uvwxyz    ",
                ",;.:!?-+()" ]

    video.init ()

    imgfont = image.load_bmp (pygame2.examples.RESOURCES.get ("font.bmp"))
    bmpfont = BitmapFont (imgfont, (32, 32), fontmap)
    
    screen = video.set_mode (640, 480)
    screen.fill (white)
    
    center = (320 - bmpfont.surface.w / 2, 10)
    screen.blit (bmpfont.surface, center)
    screen.flip ()

    wm.set_caption ("Keyboard demo")

    x = 0, 0
    pos = (310, 300)
    area = pygame2.Rect (300, 290, 50, 50)

    okay = True
    while okay:
        for ev in event.get ():
            if ev.type == sdlconst.QUIT:
                okay = False
            if ev.type == sdlconst.KEYDOWN:
                if ev.key == sdlconst.K_ESCAPE:
                    okay = False
                elif bmpfont.contains (ev.unicode):
                    screen.fill (white)
                    screen.fill (black, area)
                    screen.blit (bmpfont.surface, center)
                    bmpfont.render_on (screen, ev.unicode, pos)
                    screen.flip ()
    video.quit ()

if __name__ == "__main__":
    run ()
