import sys
import pygame2
try:
    import pygame2.sdl.constants as sdlconst
    import pygame2.sdl.event as event
    import pygame2.sdl.video as video
    import pygame2.sdlext.draw as draw
    import pygame2.sdl.wm as wm
except ImportError:
    print ("No pygame2.sdl support")
    sys.exit ()

white = pygame2.Color (255, 255, 255)
black = pygame2.Color (0, 0, 0)
red = pygame2.Color (255, 0, 0)
green = pygame2.Color (0, 255, 0)
blue = pygame2.Color (0, 0, 255)

def draw_rect (screen):
    wm.set_caption ("draw.rect examples")
    draw.rect (screen, black, pygame2.Rect (20, 20, 100, 100))
    draw.rect (screen, red, pygame2.Rect (160, 100, 160, 60))
    draw.rect (screen, green, pygame2.Rect (180, 190, 180, 200))
    draw.rect (screen, blue, pygame2.Rect (390, 120, 200, 140))

def draw_circle (screen):
    wm.set_caption ("draw.circle examples")
    draw.circle (screen, black, (100, 100), 50)
    draw.circle (screen, red, (200, 160), 80, 4)
    draw.circle (screen, green, (370, 210), 100, 12)
    draw.circle (screen, blue, (400, 400), 45, 40)

def draw_arc (screen):
    wm.set_caption ("draw.arc examples")
    draw.arc (screen, black, pygame2.Rect (80, 100, 100, 100), 0, 30)
    draw.arc (screen, red, pygame2.Rect (70, 200, 300, 150), 4, 70)
    draw.arc (screen, green, pygame2.Rect (40, 20, 100, 150), 7, 230)
    draw.arc (screen, blue, pygame2.Rect (400, 80, 200, 300), 10, 360)

def draw_line (screen):
    wm.set_caption ("draw.line examples")
    draw.line (screen, black, 4, 40, 17, 320)
    draw.line (screen, red, 280, 7, 40, 220, 4)
    draw.line (screen, green, 33, 237, 580, 370, 8)
    draw.line (screen, blue, 0, 0, 640, 480, 16)

def draw_aaline (screen):
    wm.set_caption ("draw.aaline examples")
    draw.aaline (screen, black, 4, 40, 17, 320, True)
    draw.aaline (screen, red, 280, 7, 40, 220, True)
    draw.aaline (screen, green, 33, 237, 580, 370, True)
    draw.aaline (screen, blue, 0, 0, 640, 480, False)

def draw_lines (screen):
    wm.set_caption ("draw.lines examples")
    draw.lines (screen, black, ((4, 40), (280, 7), (40, 220),
                                (33, 237), (580, 370),
                                (0, 0), (640, 480)), 4)
def draw_aalines (screen):
    wm.set_caption ("draw.aalines examples")
    draw.aalines (screen, black, ((4, 40), (280, 7), (40, 220),
                                  (33, 237), (580, 370),
                                  (0, 0), (640, 480)), 4)

def draw_polygon (screen):
    wm.set_caption ("draw.polygon examples")
    draw.polygon (screen, black, ((4, 40), (280, 7), (40, 220),
                                  (33, 237), (580, 370),
                                  (0, 0), (640, 480)), 4)

def draw_aapolygon (screen):
    wm.set_caption ("draw.aapolygon examples")
    draw.aapolygon (screen, black, ((4, 40), (280, 7), (40, 220),
                                    (33, 237), (580, 370),
                                    (0, 0), (640, 480)), 4)

def draw_ellipse (screen):
    wm.set_caption ("draw.ellipse examnples")
    draw.ellipse (screen, black, pygame2.Rect (20, 20, 100, 100), 4)
    draw.ellipse (screen, red, pygame2.Rect (160, 100, 160, 60))
    draw.ellipse (screen, green, pygame2.Rect (180, 190, 180, 200))
    draw.ellipse (screen, blue, pygame2.Rect (390, 120, 200, 140), 7)
    
def run ():
    drawtypes = [ draw_rect, draw_circle, draw_arc, draw_line, draw_aaline,
                  draw_lines, draw_aalines, draw_polygon, draw_aapolygon,
                  draw_ellipse ]
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
