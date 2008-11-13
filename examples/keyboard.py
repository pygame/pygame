import sys
import pygame2

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

white = pygame2.Color (255, 255, 255)
black = pygame2.Color (0, 0, 0)

fontmap = [ u"0123456789",
            u"ABCDEFGHIJ",
            u"KLMNOPQRST",
            u"UVWXYZ \xc4\xd6\xdc",
            u"abcdefghij",
            u"klmnopqrst",
            u"uvwxyz \xe4\xf6\xfc",
            u",;.:!?-+()" ]

def run ():
    
    video.init ()

    bmpfont = BitmapFont (image.load_bmp ("font.bmp"), (32, 32), fontmap)
    
    screen = video.set_mode (640, 480)
    screen.fill (white)
    
    center = (320 - bmpfont.surface.w / 2, 10)
    screen.blit (bmpfont.surface, center)
    screen.flip ()

    wm.set_caption ("Keyboard demo")

    x = 0, 0
    pos = (310, 300)
    area = pygame2.Rect (300, 290, 50, 50)
    while True:
        for ev in event.get ():
            if ev.type == sdlconst.QUIT:
                sys.exit ()
            if ev.type == sdlconst.KEYDOWN:
                if ev.key == sdlconst.K_ESCAPE:
                    sys.exit ()
                elif bmpfont.contains (ev.unicode):
                    screen.fill (white)
                    screen.fill (black, area)
                    screen.blit (bmpfont.surface, center)
                    bmpfont.render_on (screen, ev.unicode, pos)
                    screen.flip ()
                    

if __name__ == "__main__":
    run ()
