import os, sys
import pygame2
import pygame2.examples
try:
    import pygame2.sdl.constants as sdlconst
    import pygame2.sdl.event as event
    import pygame2.sdl.video as video
    import pygame2.sdl.wm as wm
except ImportError:
    print ("No pygame2.sdl support")
    sys.exit ()

try:
    import pygame2.sdlimage as image
    hassdlimage = True
except ImportError:
    hassdlimage = False
    import pygame2.sdl.image as image

white = pygame2.Color (255, 255, 255)
black = pygame2.Color (0, 0, 0)
red = pygame2.Color (255, 0, 0)
yellow = pygame2.Color (0, 255, 0)
blue = pygame2.Color (0, 0, 255)

redt = pygame2.Color (255, 0, 0, 75)
yellowt = pygame2.Color (0, 255, 0, 75)
bluet = pygame2.Color (0, 0, 255, 75)

def screenblit (screen, sf1, sf2, sf3, args1=None, args2=None, args3=None):
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

def blit (screen, args1=None, args2=None, args3=None):
    surface1 = video.Surface (300, 300, 32)
    surface1.fill (red)
    surface2 = video.Surface (200, 200, 32)
    surface2.fill (yellow)
    surface3 = video.Surface (240, 100, 32)
    surface3.fill (blue)
    screenblit (screen, surface1, surface2, surface3, args1, args2, args3)

def rgbablit (screen, args1=None, args2=None, args3=None):
    surface1 = video.Surface (300, 300, 32, sdlconst.SRCALPHA)
    surface1.fill (redt)
    surface2 = video.Surface (200, 200, 32, sdlconst.SRCALPHA)
    surface2.fill (yellowt)
    surface3 = video.Surface (240, 100, 32, sdlconst.SRCALPHA)
    surface3.fill (bluet)
    screenblit (screen, surface1, surface2, surface3, args1, args2, args3)
    
def blit_solid (screen):
    wm.set_caption ("Solid blit")
    blit (screen)

def blit_min (screen):
    wm.set_caption ("BLEND_RGB_MIN blit")
    blit (screen, sdlconst.BLEND_RGB_MIN, sdlconst.BLEND_RGB_MIN,
          sdlconst.BLEND_RGB_MIN)

def blit_max (screen):
    wm.set_caption ("BLEND_RGB_MAX blit")
    blit (screen, sdlconst.BLEND_RGB_MAX, sdlconst.BLEND_RGB_MAX,
          sdlconst.BLEND_RGB_MAX)

def blit_add (screen):
    wm.set_caption ("BLEND_RGB_ADD blit")
    blit (screen, sdlconst.BLEND_RGB_ADD, sdlconst.BLEND_RGB_ADD,
          sdlconst.BLEND_RGB_ADD)

def blit_sub (screen):
    wm.set_caption ("BLEND_RGB_SUB blit")
    blit (screen, sdlconst.BLEND_RGB_SUB, sdlconst.BLEND_RGB_SUB,
          sdlconst.BLEND_RGB_SUB)

def blit_mult (screen):
    wm.set_caption ("BLEND_RGB_MULT blit")
    blit (screen, sdlconst.BLEND_RGB_MULT, sdlconst.BLEND_RGB_MULT,
          sdlconst.BLEND_RGB_MULT)

def blit_and (screen):
    wm.set_caption ("BLEND_RGB_AND blit")
    blit (screen, sdlconst.BLEND_RGB_AND, sdlconst.BLEND_RGB_AND,
          sdlconst.BLEND_RGB_AND)

def blit_or (screen):
    wm.set_caption ("BLEND_RGB_OR blit")
    blit (screen, sdlconst.BLEND_RGB_OR, sdlconst.BLEND_RGB_OR,
          sdlconst.BLEND_RGB_OR)

def blit_xor (screen):
    wm.set_caption ("BLEND_RGB_XOR blit")
    blit (screen, sdlconst.BLEND_RGB_XOR, sdlconst.BLEND_RGB_XOR,
          sdlconst.BLEND_RGB_XOR)

def blit_diff (screen):
    wm.set_caption ("BLEND_RGB_DIFF blit")
    blit (screen, sdlconst.BLEND_RGB_DIFF, sdlconst.BLEND_RGB_DIFF,
          sdlconst.BLEND_RGB_DIFF)

def blit_screen (screen):
    wm.set_caption ("BLEND_RGB_SCREEN blit")
    blit (screen, sdlconst.BLEND_RGB_SCREEN, sdlconst.BLEND_RGB_SCREEN,
          sdlconst.BLEND_RGB_SCREEN)

def blit_avg (screen):
    wm.set_caption ("BLEND_RGB_AVG blit")
    blit (screen, sdlconst.BLEND_RGB_AVG, sdlconst.BLEND_RGB_AVG,
          sdlconst.BLEND_RGB_AVG)

def blit_rgba (screen):
    wm.set_caption ("Solid RGBA blit")
    rgbablit (screen)

def blit_rgba_min (screen):
    wm.set_caption ("BLEND_RGBA_MIN blit")
    rgbablit (screen, sdlconst.BLEND_RGBA_MIN, sdlconst.BLEND_RGBA_MIN,
              sdlconst.BLEND_RGBA_MIN)

def blit_rgba_max (screen):
    wm.set_caption ("BLEND_RGBA_MAX blit")
    rgbablit (screen, sdlconst.BLEND_RGBA_MAX, sdlconst.BLEND_RGBA_MAX,
              sdlconst.BLEND_RGBA_MAX)

def blit_rgba_add (screen):
    wm.set_caption ("BLEND_RGBA_ADD blit")
    rgbablit (screen, sdlconst.BLEND_RGBA_ADD, sdlconst.BLEND_RGBA_ADD,
              sdlconst.BLEND_RGBA_ADD)

def blit_rgba_sub (screen):
    wm.set_caption ("BLEND_RGBA_SUB blit")
    rgbablit (screen, sdlconst.BLEND_RGBA_SUB, sdlconst.BLEND_RGBA_SUB,
              sdlconst.BLEND_RGBA_SUB)

def blit_rgba_mult (screen):
    wm.set_caption ("BLEND_RGBA_MULT blit")
    rgbablit (screen, sdlconst.BLEND_RGBA_MULT, sdlconst.BLEND_RGBA_MULT,
          sdlconst.BLEND_RGBA_MULT)

def blit_rgba_and (screen):
    wm.set_caption ("BLEND_RGBA_AND blit")
    rgbablit (screen, sdlconst.BLEND_RGBA_AND, sdlconst.BLEND_RGBA_AND,
          sdlconst.BLEND_RGBA_AND)

def blit_rgba_or (screen):
    wm.set_caption ("BLEND_RGBA_OR blit")
    rgbablit (screen, sdlconst.BLEND_RGBA_OR, sdlconst.BLEND_RGBA_OR,
          sdlconst.BLEND_RGBA_OR)

def blit_rgba_xor (screen):
    wm.set_caption ("BLEND_RGBA_XOR blit")
    rgbablit (screen, sdlconst.BLEND_RGBA_XOR, sdlconst.BLEND_RGBA_XOR,
          sdlconst.BLEND_RGBA_XOR)

def blit_rgba_diff (screen):
    wm.set_caption ("BLEND_RGBA_DIFF blit")
    rgbablit (screen, sdlconst.BLEND_RGBA_DIFF, sdlconst.BLEND_RGBA_DIFF,
          sdlconst.BLEND_RGBA_DIFF)

def blit_rgba_screen (screen):
    wm.set_caption ("BLEND_RGBA_SCREEN blit")
    rgbablit (screen, sdlconst.BLEND_RGBA_SCREEN, sdlconst.BLEND_RGBA_SCREEN,
          sdlconst.BLEND_RGBA_SCREEN)

def blit_rgba_avg (screen):
    wm.set_caption ("BLEND_RGBA_AVG blit")
    rgbablit (screen, sdlconst.BLEND_RGBA_AVG, sdlconst.BLEND_RGBA_AVG,
          sdlconst.BLEND_RGBA_AVG)

def run ():
    blittypes = [ blit_solid, blit_min, blit_max, blit_add, blit_sub,
                  blit_mult, blit_and, blit_or, blit_xor, blit_diff,
                  blit_screen, blit_avg,
                  blit_rgba, blit_rgba_min, blit_rgba_max, blit_rgba_add,
                  blit_rgba_sub, blit_rgba_mult, blit_rgba_and, blit_rgba_or,
                  blit_rgba_xor, blit_rgba_diff, blit_rgba_screen,
                  blit_rgba_avg ]
    curtype = 0
    video.init ()
    screen = video.set_mode (640, 480, 32)
    color = white
    imgdir = os.path.dirname (os.path.abspath (__file__))
    logo = None
    if hassdlimage:
        logo = image.load (os.path.join (pygame2.examples.IMAGEDIR, "logo.gif"))
    else:
        logo = image.load_bmp (os.path.join
                               (pygame2.examples.IMAGEDIR, "logo.bmp"))
    
    screen.fill (color)
    screen.blit (logo, (-10, 140))
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
                screen.blit (logo, (-10, 140))
                blittypes[curtype] (screen)
                screen.flip ()
    video.quit ()

if __name__ == "__main__":
    run ()
