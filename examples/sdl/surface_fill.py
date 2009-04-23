import sys, os
import pygame2
try:
    import pygame2.sdl.constants as sdlconst
    import pygame2.sdl.event as event
    import pygame2.sdl.video as video
    import pygame2.sdl.image as image
    import pygame2.sdl.wm as wm
except ImportError:
    print ("No pygame2.sdl support")
    sys.exit ()

white = pygame2.Color (255, 255, 255)
black = pygame2.Color (0, 0, 0)
red = pygame2.Color (255, 0, 0)
green = pygame2.Color (0, 255, 0)
blue = pygame2.Color (0, 0, 255)

redt = pygame2.Color (255, 0, 0, 75)
greent = pygame2.Color (0, 255, 0, 75)
bluet = pygame2.Color (0, 0, 255, 75)

rect = pygame2.Rect (10, 10, 200, 200)
rect2 = pygame2.Rect (280, 10, 200, 200)
rect3 = pygame2.Rect (550, 10, 200, 200)

def fill_solid (screen):
    wm.set_caption ("Solid RGB fill")
    screen.fill (red, rect)
    screen.fill (green, rect2)
    screen.fill (blue, rect3)

def fill_rgba (screen):
    wm.set_caption ("Solid RGBA fill")
    screen.fill (redt, rect)
    screen.fill (greent, rect2)
    screen.fill (bluet, rect3)

def fill_min (screen):
    wm.set_caption ("BLEND_MIN fill")
    screen.fill (red, rect, sdlconst.BLEND_MIN)
    screen.fill (green, rect2, sdlconst.BLEND_MIN)
    screen.fill (blue, rect3, sdlconst.BLEND_MIN)

def fill_rgba_min (screen):
    screen.fill (redt, rect, sdlconst.BLEND_RGBA_MIN)
    screen.fill (greent, rect2, sdlconst.BLEND_RGBA_MIN)
    screen.fill (bluet, rect3, sdlconst.BLEND_RGBA_MIN)
    wm.set_caption ("BLEND_RGBA_MIN fill")

def fill_max (screen):
    wm.set_caption ("BLEND_MAX fill")
    screen.fill (red, rect, sdlconst.BLEND_MAX)
    screen.fill (green, rect2, sdlconst.BLEND_MAX)
    screen.fill (blue, rect3, sdlconst.BLEND_MAX)

def fill_rgba_max (screen):
    wm.set_caption ("BLEND_RGBA_MAX fill")
    screen.fill (redt, rect, sdlconst.BLEND_RGBA_MAX)
    screen.fill (greent, rect2, sdlconst.BLEND_RGBA_MAX)
    screen.fill (bluet, rect3, sdlconst.BLEND_RGBA_MAX)

def fill_add (screen):
    wm.set_caption ("BLEND_ADD fill")
    screen.fill (red, rect, sdlconst.BLEND_ADD)
    screen.fill (green, rect2, sdlconst.BLEND_ADD)
    screen.fill (blue, rect3, sdlconst.BLEND_ADD)

def fill_rgba_add (screen):
    wm.set_caption ("BLEND_RGBA_ADD fill")
    screen.fill (redt, rect, sdlconst.BLEND_RGBA_ADD)
    screen.fill (greent, rect2, sdlconst.BLEND_RGBA_ADD)
    screen.fill (bluet, rect3, sdlconst.BLEND_RGBA_ADD)

def fill_sub (screen):
    wm.set_caption ("BLEND_SUB fill")
    screen.fill (red, rect, sdlconst.BLEND_SUB)
    screen.fill (green, rect2, sdlconst.BLEND_SUB)
    screen.fill (blue, rect3, sdlconst.BLEND_SUB)

def fill_rgba_sub (screen):
    wm.set_caption ("BLEND_RGBA_SUB fill")
    screen.fill (redt, rect, sdlconst.BLEND_RGBA_SUB)
    screen.fill (greent, rect2, sdlconst.BLEND_RGBA_SUB)
    screen.fill (bluet, rect3, sdlconst.BLEND_RGBA_SUB)

def fill_mult (screen):
    wm.set_caption ("BLEND_MULT fill")
    screen.fill (red, rect, sdlconst.BLEND_MULT)
    screen.fill (green, rect2, sdlconst.BLEND_MULT)
    screen.fill (blue, rect3, sdlconst.BLEND_MULT)

def fill_rgba_mult (screen):
    wm.set_caption ("BLEND_RGBA_MULT fill")
    screen.fill (redt, rect, sdlconst.BLEND_RGBA_MULT)
    screen.fill (greent, rect2, sdlconst.BLEND_RGBA_MULT)
    screen.fill (bluet, rect3, sdlconst.BLEND_RGBA_MULT)

def run ():
    filltypes = [ fill_solid, fill_rgba, fill_min, fill_rgba_min, fill_max,
                  fill_rgba_max, fill_add, fill_rgba_add, fill_sub,
                  fill_rgba_sub, fill_mult, fill_rgba_mult]
    curtype = 0
    video.init ()
    
    screen = video.set_mode (760, 300, 32)
    imgdir = os.path.dirname (os.path.abspath (__file__))
    surface = image.load_bmp (os.path.join (imgdir, "logo.bmp"))
    surface = surface.convert (flags=sdlconst.SRCALPHA)
    
    color = white
    screen.fill (color)
    screen.blit (surface, (40, 50))
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
                if curtype >= len (filltypes):
                    curtype = 0
                    if color == black:
                        color = white
                    else:
                        color = black

                screen.fill (color)
                screen.blit (surface, (40, 50))
                filltypes[curtype] (screen)
                screen.flip ()
    video.quit ()

if __name__ == "__main__":
    run ()
