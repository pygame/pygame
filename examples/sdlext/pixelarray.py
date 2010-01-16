import sys, os
import pygame2
import pygame2.examples
try:
    import pygame2.sdl.constants as sdlconst
    import pygame2.sdl.event as event
    import pygame2.sdl.video as video
    import pygame2.sdl.wm as wm
    import pygame2.sdl.image as image
    from pygame2.sdlext import PixelArray
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

def draw_checked (screen):
    wm.set_caption ("Manipulating every second pixel in every second row")
    pxarray = PixelArray (screen)
    pxarray[::2,::2] = white
    del pxarray

def draw_striped (screen):
    wm.set_caption ("Manipulating every third column")
    pxarray = PixelArray (screen)
    pxarray[::3] = white
    del pxarray

def draw_flipped (screen):
    wm.set_caption ("Flipping around x and y")
    pxarray = PixelArray (screen)
    pxarray[:] = pxarray[::-1, ::-1]
    del pxarray

def draw_mixed (screen):
    wm.set_caption ("Manipulating parts")
    pxarray = PixelArray (screen)
    pxarray[::2, :120:2] = pxarray[-1, -1]
    pxarray[::6, 120::] = red
    del pxarray

def draw_zoomed (screen):
    wm.set_caption ("2x zoom")
    pxarray = PixelArray (screen)
    
    # Temporary array.
    sf = video.Surface (640, 480, 32)
    tmparray = PixelArray (sf)
    tmparray[::2, ::2] = pxarray[:]
    tmparray[1::2, ::2] = pxarray[:-1]
    tmparray[:, 1::2] = tmparray[:, :-1:2]
    
    pxarray[:] = tmparray[80:400, 80:320]
    del tmparray
    del sf
    del pxarray

def draw_replaced (screen):
    wm.set_caption ("Replacing colors")
    pxarray = PixelArray (screen)
    pxarray.replace (black, white, 0.06)
    pxarray.replace (red, green, 0)
    del pxarray
    
def draw_extracted (screen):
    wm.set_caption ("Extracting colors (black, exact match)")
    pxarray = PixelArray (screen)
    pxarray[:] = pxarray.extract (black, 0)
    del pxarray

def draw_extracted2 (screen):
    wm.set_caption ("Extracting colors (black, 50% match)")
    pxarray = PixelArray (screen)
    pxarray[:] = pxarray.extract (black, 0.5)
    del pxarray

def run ():
    methods = [ draw_checked, draw_striped, draw_flipped, draw_mixed,
                draw_zoomed, draw_replaced, draw_extracted, draw_extracted2]
    curmethod = -1
    
    video.init ()
    screen = video.set_mode (320, 240, 32)
    screen.fill (black)
    
    surface = image.load_bmp (pygame2.examples.RESOURCES.get ("array.bmp"))
    surface = surface.convert (flags=sdlconst.SRCALPHA)
    screen.blit (surface)
    screen.flip ()

    okay = True
    while okay:
        for ev in event.get ():
            if ev.type == sdlconst.QUIT:
                okay = False
            if ev.type == sdlconst.KEYDOWN and ev.key == sdlconst.K_ESCAPE:
                okay = False
            if ev.type == sdlconst.MOUSEBUTTONDOWN:
                curmethod += 1
                if curmethod >= len (methods):
                    curmethod = 0
                screen.fill (black)
                screen.blit (surface)
                methods[curmethod](screen)
                screen.flip ()
    video.quit ()

if __name__ == "__main__":
    run ()
