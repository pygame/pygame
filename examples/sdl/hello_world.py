import sys, os
import pygame2
import pygame2.examples
try:
    import pygame2.sdl.constants as constants
    import pygame2.sdl.event as event
    import pygame2.sdl.video as video
except ImportError:
    print ("No pygame2.sdl support")
    sys.exit ()

try:
    import pygame2.sdlimage as image
    hassdlimage = True
except ImportError:
    hassdlimage = False
    import pygame2.sdl.image as image

def run():
    video.init ()

    surface = None
    if hassdlimage:
        surface = image.load (pygame2.examples.RESOURCES.get ("logo.gif"))
    else:
        surface = image.load_bmp (pygame2.examples.RESOURCES.get ("logo.bmp"))

    screen = video.set_mode (surface.w + 10, surface.h + 10)
    screen.fill (pygame2.Color (255, 255, 255))
    screen.blit (surface, (5, 5))
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
