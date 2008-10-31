import sys
import pygame2
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

video.init ()

surface = None
if hassdlimage:
    surface = image.load ("logo.gif")
else:
    surface = image.load_bmp ("logo.bmp")

screen = video.set_mode (surface.w + 10, surface.h + 10)
screen.fill (pygame2.Color (255, 255, 255))
screen.blit (surface, pygame2.Rect (5, 5, 10, 10))
screen.flip ()

while True:
    for ev in event.get ():
        if ev.type == constants.QUIT:
            sys.exit ()
        if ev.type == constants.KEYDOWN and ev.key == constants.K_ESCAPE:
            sys.exit ()
