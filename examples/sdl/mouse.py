import sys
import pygame2

try:
    import pygame2.sdl.constants as sdlconst
    import pygame2.sdl.event as event
    import pygame2.sdl.video as video
    import pygame2.sdl.mouse as mouse
    import pygame2.sdl.wm as wm
except ImportError:
    print ("No pygame2.sdl support")
    sys.exit ()

def run ():
    black = pygame2.Color (0, 0, 0)
    white = pygame2.Color (255, 255, 255)
    green = pygame2.Color (0, 255, 0)
    red = pygame2.Color (255, 0, 0)
    
    curcolor = black
    pressed = False
    lastpos = 0, 0
    
    video.init ()
    screen = video.set_mode (640, 480)
    
    screen.fill (white)
    screen.flip ()

    wm.set_caption ("Mouse demo")

    okay = True
    while okay:
        for ev in event.get ():
            if ev.type == sdlconst.QUIT:
                okay = False
            if ev.type == sdlconst.KEYDOWN and ev.key == sdlconst.K_ESCAPE:
                okay = False
            if ev.type == sdlconst.MOUSEMOTION:
                if pressed:
                    x, y = ev.pos
                    lastpos = ev.pos
                    screen.fill (curcolor, pygame2.Rect (x - 2, y - 2, 5, 5))
                    screen.flip ()
            elif ev.type == sdlconst.MOUSEBUTTONDOWN:
                if ev.button == 1:
                    pressed = True
                elif ev.button == 4:
                    x, y = lastpos[0], lastpos[1] - 2
                    screen.fill (green, pygame2.Rect (x - 2, y - 2, 5, 5))
                    lastpos = x, y
                    screen.flip ()
                elif ev.button == 5:
                    x, y = lastpos[0], lastpos[1] + 2
                    screen.fill (red, pygame2.Rect (x - 2, y - 2, 5, 5))
                    lastpos = x, y
                    screen.flip ()
            elif ev.type == sdlconst.MOUSEBUTTONUP:
                if ev.button == 1:
                    pressed = False
                elif ev.button == 3:
                    if curcolor == white:
                        curcolor = black
                        screen.fill (white)
                    else:
                        curcolor = white
                        screen.fill (black)
                    screen.flip ()
    video.quit ()

if __name__ == "__main__":
    run ()
