import sys
import pygame2

try:
    import pygame2.sdl.constants as sdlconst
    import pygame2.sdl.event as event
    import pygame2.sdl.video as video
    import pygame2.sdl.mouse as mouse
    import pygame2.sdl.image as image
    import pygame2.sdl.wm as wm
except ImportError:
    print ("No pygame2.sdl support")
    sys.exit ()

white = pygame2.Color (255, 255, 255)
black = pygame2.Color (0, 0, 0)

def create_mapping ():
    fontmap = [ u"0123456789",
                u"ABCDEFGHIJ",
                u"KLMNOPQRST",
                u"UVWXYZ \xc4\xd6\xdc",
                u"abcdefghij",
                u"klmnopqrst",
                u"uvwxyz \xe4\xf6\xfc",
                u",;.:!?-+()" ]
    fontimgmap = {}
    x, y = 0, 0
    for line in fontmap:
        x = 0
        for c in line:
            fontimgmap[c] = pygame2.Rect (x, y, 32, 32)
            x += 32
        y += 33
    print fontimgmap
    return fontimgmap

def run ():
    
    video.init ()
    
    dest = pygame2.Rect (0, 0, 0, 0)
    fontimgmap = create_mapping ()
    font_surface = image.load_bmp ("font.bmp")
    
    screen = video.set_mode (640, 480)
    screen.fill (white)
    
    centerx = 320 - font_surface.w / 2
    font_rect = pygame2.Rect (centerx, 10, 0, 0)
    screen.blit (font_surface, font_rect)
    screen.flip ()

    wm.set_caption ("Keyboard demo")

    x = 0, 0
    pos = pygame2.Rect (310, 300, 0, 0)
    area = pygame2.Rect (300, 290, 50, 50)
    while True:
        for ev in event.get ():
            if ev.type == sdlconst.QUIT:
                sys.exit ()
            if ev.type == sdlconst.KEYDOWN:
                if ev.key == sdlconst.K_ESCAPE:
                    sys.exit ()
                elif fontimgmap.has_key (ev.unicode):
                    screen.fill (white)
                    screen.fill (black, area)
                    screen.blit (font_surface, font_rect)
                    screen.blit (font_surface, pos, fontimgmap[ev.unicode])
                    screen.flip ()
                    

if __name__ == "__main__":
    run ()
