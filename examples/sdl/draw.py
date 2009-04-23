import sys
import pygame2
try:
    import pygame2.sdl.constants as sdlconst
    import pygame2.sdl.event as event
    import pygame2.sdl.video as video
    import pygame2.sdlext.draw as draw
except ImportError:
    print ("No pygame2.sdl support")
    sys.exit ()

white = pygame2.Color (255, 255, 255)
black = pygame2.Color (0, 0, 0)
red = pygame2.Color (255, 0, 0)
green = pygame2.Color (0, 255, 0)
blue = pygame2.Color (0, 0, 255)

def draw_rect (screen):
    draw.rect (screen, black, pygame2.Rect (20, 20, 100, 100))
    draw.rect (screen, red, pygame2.Rect (160, 100, 160, 60))
    draw.rect (screen, green, pygame2.Rect (180, 190, 180, 200))
    draw.rect (screen, blue, pygame2.Rect (390, 120, 200, 140))

def draw_circle (screen):
    draw.circle (screen, black, (100, 100), 50)
    draw.circle (screen, red, (200, 160), 80, 4)
    draw.circle (screen, green, (370, 210), 100, 12)
    draw.circle (screen, blue, (400, 400), 45, 40)

def draw_arc (screen):
    draw.arc (screen, black, pygame2.Rect (80, 100, 100, 100), 0, 30)

def run ():
    drawtypes = [ draw_rect, draw_circle, draw_arc ]
    curtype = 0
    video.init ()

    screen = video.set_mode (640, 480)
    screen.fill (white)
    draw_rect (screen)
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
                if curtype >= len (drawtypes):
                    curtype = 0
                screen.fill (white)
                drawtypes[curtype] (screen)
                screen.flip ()
    video.quit ()

if __name__ == "__main__":
    run ()
