import sys
import pygame2
try:
    import pygame2.sdl.constants as sdlconst
    import pygame2.sdl.event as event
    import pygame2.sdl.video as video
    import pygame2.sdl.wm as wm
except ImportError:
    print ("No pygame2.sdl support")
    sys.exit ()

white = pygame2.Color (255, 255, 255)
black = pygame2.Color (0)
red = pygame2.Color (255, 0, 0)
yellow = pygame2.Color (255, 255, 0)
blue = pygame2.Color (0, 0, 255)

rect = pygame2.Rect (10, 10, 10, 10)
rect2 = pygame2.Rect (220, 220, 200, 200)

def blit (screen, sf1, sf2, args1=None, args2=None):
    screen.fill (white)
    if args1 is not None:
        screen.blit (sf1, rect, args1)
    else:
        screen.blit (sf1, rect)
    if args2 is not None:
        screen.blit (sf2, dstrect=rect2, blendargs=args2)
    else:
        screen.blit (sf2, rect2)

def blit_solid (screen):
    wm.set_caption ("Solid blit")
    surface1 = video.Surface (300, 300, 32)
    surface1.fill (red)
    surface2 = video.Surface (200, 200, 32)
    surface2.fill (yellow)
    blit (screen, surface1, surface2, None, None)

def blit_min (screen):
    wm.set_caption ("BLEND_MIN blit")
    surface1 = video.Surface (300, 300, 32)
    surface1.fill (red)
    surface2 = video.Surface (200, 200, 32)
    surface2.fill (yellow)
    blit (screen, surface1, surface2, None, sdlconst.BLEND_MIN)

def blit_max (screen):
    wm.set_caption ("BLEND_MAX blit")
    surface1 = video.Surface (300, 300, 32)
    surface1.fill (red)
    surface2 = video.Surface (200, 200, 32)
    surface2.fill (yellow)
    blit (screen, surface1, surface2, None, sdlconst.BLEND_MAX)

def blit_add (screen):
    wm.set_caption ("BLEND_ADD blit")
    surface1 = video.Surface (300, 300, 32)
    surface1.fill (red)
    surface2 = video.Surface (200, 200, 32)
    surface2.fill (yellow)
    blit (screen, surface1, surface2, None, sdlconst.BLEND_ADD)

def blit_sub (screen):
    wm.set_caption ("BLEND_SUB blit")
    surface1 = video.Surface (300, 300, 32)
    surface1.fill (red)
    surface2 = video.Surface (200, 200, 32)
    surface2.fill (yellow)
    blit (screen, surface1, surface2, None, sdlconst.BLEND_SUB)

def run ():
    blittypes = [ blit_solid, blit_min, blit_max, blit_add, blit_sub ]
    curtype = 0
    video.init ()
    screen = video.set_mode (640, 480, 32)

    blit_solid (screen)
    screen.flip ()
    while True:
        for ev in event.get ():
            if ev.type == sdlconst.QUIT:
                sys.exit ()
            if ev.type == sdlconst.KEYDOWN and ev.key == sdlconst.K_ESCAPE:
                sys.exit ()
            if ev.type == sdlconst.MOUSEBUTTONDOWN:
                curtype += 1
                if curtype >= len (blittypes):
                    curtype = 0
                blittypes[curtype] (screen)
                screen.flip ()

if __name__ == "__main__":
    run ()
