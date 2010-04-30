import os, sys, time
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
    import pygame2.freetype as freetype
except ImportError:
    print ("No pygame2.freetype support")
    sys.exit ()

try:
    import pygame2.sdlgfx as sdlgfx
    import pygame2.sdlgfx.constants as gfxconst
except ImportError:
    print ("No pygame2.sdlgfx support")
    sys.exit ()

black = pygame2.Color (0, 0, 0)
white = pygame2.Color (255, 255, 255)

def run ():
    video.init ()
    freetype.init ()

    font = freetype.Font (pygame2.examples.RESOURCES.get ("sans.ttf"))

    fpsmanager = sdlgfx.FPSmanager (2)

    screen = video.set_mode (640, 480)
    wm.set_caption ("FPSmanager example")
    screenrect = pygame2.Rect (640, 480)
    screen.fill (black)
    screen.flip ()

    okay = True
    while okay:
        for ev in event.get ():
            if ev.type == sdlconst.QUIT:
                okay = False
            if ev.type == sdlconst.KEYDOWN:
                framerate = fpsmanager.framerate
                if ev.key == sdlconst.K_ESCAPE:
                    okay = False
                elif ev.key in (sdlconst.K_PLUS, sdlconst.K_KP_PLUS):
                    framerate = min (framerate + 1, gfxconst.FPS_UPPER_LIMIT)
                    fpsmanager.framerate = framerate
                elif ev.key in (sdlconst.K_MINUS, sdlconst.K_KP_MINUS):
                    framerate = max (framerate - 1, gfxconst.FPS_LOWER_LIMIT)
                    fpsmanager.framerate = framerate

        screen.fill (black)

        prev = time.time ()
        fpsmanager.delay ()
        last = time.time ()

        millis = ((last - prev) * 1000)
        fpstext = "FPS: %d" % fpsmanager.framerate
        timetext = "time (ms) passed since last update: %.3f" % millis
                   
        surfacef, w, h = font.render (fpstext, white, ptsize=28)
        surfacet, w2, h2 = font.render (timetext, white, ptsize=28)
        blitrect = pygame2.Rect (w, h)
        blitrect.center = screenrect.centerx, screenrect.centery - h
        screen.blit (surfacef, blitrect.topleft)
        blitrect = pygame2.Rect (w2, h2)
        blitrect.center = screenrect.centerx, screenrect.centery + h
        screen.blit (surfacet, blitrect.topleft)
        screen.flip ()
    
    video.quit ()

if __name__ == "__main__":
    run ()
