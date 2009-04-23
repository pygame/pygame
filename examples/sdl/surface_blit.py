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
black = pygame2.Color (0, 0, 0)
red = pygame2.Color (255, 0, 0)
yellow = pygame2.Color (0, 255, 0)
blue = pygame2.Color (0, 0, 255)

redt = pygame2.Color (255, 0, 0, 75)
yellowt = pygame2.Color (0, 255, 0, 75)
bluet = pygame2.Color (0, 0, 255, 75)

def blit (screen, sf1, sf2, sf3, args1=None, args2=None, args3=None):
    if args1 is not None:
        screen.blit (sf1, dstrect=(10, 10), blendargs=args1)
    else:
        screen.blit (sf1, (10, 10))
    if args2 is not None:
        screen.blit (sf2, dstrect=(220, 220), blendargs=args2)
    else:
        screen.blit (sf2, (220, 220))
    if args3 is not None:
        screen.blit (sf3, dstrect=(250, 170), blendargs=args3)
    else:
        screen.blit (sf3, (250, 170))

def blit_solid (screen):
    wm.set_caption ("Solid blit")
    surface1 = video.Surface (300, 300, 32)
    surface1.fill (red)
    surface2 = video.Surface (200, 200, 32)
    surface2.fill (yellow)
    surface3 = video.Surface (240, 100, 32)
    surface3.fill (blue)
    blit (screen, surface1, surface2, surface3)

def blit_min (screen):
    wm.set_caption ("BLEND_MIN blit")
    surface1 = video.Surface (300, 300, 32)
    surface1.fill (red)
    surface2 = video.Surface (200, 200, 32)
    surface2.fill (yellow)
    surface3 = video.Surface (240, 100, 32)
    surface3.fill (blue)
    blit (screen, surface1, surface2, surface3, sdlconst.BLEND_MIN,
        sdlconst.BLEND_MIN, sdlconst.BLEND_MIN)

def blit_max (screen):
    wm.set_caption ("BLEND_MAX blit")
    surface1 = video.Surface (300, 300, 32)
    surface1.fill (red)
    surface2 = video.Surface (200, 200, 32)
    surface2.fill (yellow)
    surface3 = video.Surface (240, 100, 32)
    surface3.fill (blue)
    blit (screen, surface1, surface2, surface3, sdlconst.BLEND_MAX,
        sdlconst.BLEND_MAX, sdlconst.BLEND_MAX)

def blit_add (screen):
    wm.set_caption ("BLEND_ADD blit")
    surface1 = video.Surface (300, 300, 32)
    surface1.fill (red)
    surface2 = video.Surface (200, 200, 32)
    surface2.fill (yellow)
    surface3 = video.Surface (240, 100, 32)
    surface3.fill (blue)
    blit (screen, surface1, surface2, surface3, sdlconst.BLEND_ADD,
        sdlconst.BLEND_ADD, sdlconst.BLEND_ADD)

def blit_sub (screen):
    wm.set_caption ("BLEND_SUB blit")
    surface1 = video.Surface (300, 300, 32)
    surface1.fill (red)
    surface2 = video.Surface (200, 200, 32)
    surface2.fill (yellow)
    surface3 = video.Surface (240, 100, 32)
    surface3.fill (blue)
    blit (screen, surface1, surface2, surface3, sdlconst.BLEND_SUB,
        sdlconst.BLEND_SUB, sdlconst.BLEND_SUB)

def blit_mult (screen):
    wm.set_caption ("BLEND_MULT blit")
    surface1 = video.Surface (300, 300, 32)
    surface1.fill (red)
    surface2 = video.Surface (200, 200, 32)
    surface2.fill (yellow)
    surface3 = video.Surface (240, 100, 32)
    surface3.fill (blue)
    blit (screen, surface1, surface2, surface3, sdlconst.BLEND_MULT,
        sdlconst.BLEND_MULT, sdlconst.BLEND_MULT)

def blit_rgba (screen):
    wm.set_caption ("Solid RGBA blit")
    surface1 = video.Surface (300, 300, 32, sdlconst.SRCALPHA)
    surface1.fill (redt)
    surface2 = video.Surface (200, 200, 32, sdlconst.SRCALPHA)
    surface2.fill (yellowt)
    surface3 = video.Surface (240, 100, 32, sdlconst.SRCALPHA)
    surface3.fill (bluet)
    blit (screen, surface1, surface2, surface3)

def blit_rgba_min (screen):
    wm.set_caption ("BLEND_RGBA_MIN blit")
    surface1 = video.Surface (300, 300, 32, sdlconst.SRCALPHA)
    surface1.fill (redt)
    surface2 = video.Surface (200, 200, 32, sdlconst.SRCALPHA)
    surface2.fill (yellowt)
    surface3 = video.Surface (240, 100, 32, sdlconst.SRCALPHA)
    surface3.fill (bluet)
    blit (screen, surface1, surface2, surface3, sdlconst.BLEND_RGBA_MIN,
        sdlconst.BLEND_RGBA_MIN, sdlconst.BLEND_RGBA_MIN)

def blit_rgba_max (screen):
    wm.set_caption ("BLEND_RGBA_MAX blit")
    surface1 = video.Surface (300, 300, 32, sdlconst.SRCALPHA)
    surface1.fill (redt)
    surface2 = video.Surface (200, 200, 32, sdlconst.SRCALPHA)
    surface2.fill (yellowt)
    surface3 = video.Surface (240, 100, 32, sdlconst.SRCALPHA)
    surface3.fill (bluet)
    blit (screen, surface1, surface2, surface3, sdlconst.BLEND_RGBA_MAX,
        sdlconst.BLEND_RGBA_MAX, sdlconst.BLEND_RGBA_MAX)

def blit_rgba_add (screen):
    wm.set_caption ("BLEND_RGBA_ADD blit")
    surface1 = video.Surface (300, 300, 32, sdlconst.SRCALPHA)
    surface1.fill (redt)
    surface2 = video.Surface (200, 200, 32, sdlconst.SRCALPHA)
    surface2.fill (yellowt)
    surface3 = video.Surface (240, 100, 32, sdlconst.SRCALPHA)
    surface3.fill (bluet)
    blit (screen, surface1, surface2, surface3, sdlconst.BLEND_RGBA_ADD,
        sdlconst.BLEND_RGBA_ADD, sdlconst.BLEND_RGBA_ADD)

def blit_rgba_sub (screen):
    wm.set_caption ("BLEND_RGBA_SUB blit")
    surface1 = video.Surface (300, 300, 32, sdlconst.SRCALPHA)
    surface1.fill (redt)
    surface2 = video.Surface (200, 200, 32, sdlconst.SRCALPHA)
    surface2.fill (yellowt)
    surface3 = video.Surface (240, 100, 32, sdlconst.SRCALPHA)
    surface3.fill (bluet)
    blit (screen, surface1, surface2, surface3, sdlconst.BLEND_RGBA_SUB,
        sdlconst.BLEND_RGBA_SUB, sdlconst.BLEND_RGBA_SUB)

def blit_rgba_mult (screen):
    wm.set_caption ("BLEND_RGBA_MULT blit")
    surface1 = video.Surface (300, 300, 32, sdlconst.SRCALPHA)
    surface1.fill (redt)
    surface2 = video.Surface (200, 200, 32, sdlconst.SRCALPHA)
    surface2.fill (yellowt)
    surface3 = video.Surface (240, 100, 32, sdlconst.SRCALPHA)
    surface3.fill (bluet)
    blit (screen, surface1, surface2, surface3, sdlconst.BLEND_RGBA_MULT,
        sdlconst.BLEND_RGBA_MULT, sdlconst.BLEND_RGBA_MULT)

def run ():
    blittypes = [ blit_solid, blit_min, blit_max, blit_add, blit_sub,
                  blit_mult, blit_rgba, blit_rgba_min, blit_rgba_max,
                  blit_rgba_add, blit_rgba_sub, blit_rgba_mult ]
    curtype = 0
    video.init ()
    screen = video.set_mode (640, 480, 32)
    color = white
    screen.fill (color)
    blit_solid (screen)
    screen.flip ()

    okay = True
    while okay:
        for ev in event.get ():
            if ev.type == sdlconst.QUIT:
                okay = False
            if ev.type == sdlconst.KEYDOWN and ev.key == sdlconst.K_ESCAPE:
                okay = False
            if ev.type == sdlconst.MOUSEBUTTONDOWN:
                curtype += 1
                if curtype >= len (blittypes):
                    curtype = 0
                    if color == black:
                        color = white
                    else:
                        color = black

                screen.fill (color)
                blittypes[curtype] (screen)
                screen.flip ()
    video.quit ()

if __name__ == "__main__":
    run ()
