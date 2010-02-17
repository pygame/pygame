import sys
import pygame2
import pygame2.examples
try:
    import pygame2.sdl.constants as sdlconst
    import pygame2.sdl.event as event
    import pygame2.sdl.video as video
    import pygame2.sdl.wm as wm
    import pygame2.sdl.image as image
except ImportError:
    print ("No pygame2.sdl support")
    sys.exit ()

try:
    import pygame2.sdlgfx.primitives as primitives
except ImportError:
    print ("No pygame2.sdlgfx support")
    sys.exit ()

white = pygame2.Color (255, 255, 255)
black = pygame2.Color (0, 0, 0)
red = pygame2.Color (255, 0, 0)
green = pygame2.Color (0, 255, 0)
blue = pygame2.Color (0, 0, 255)

colors = [ black, red, green, blue ]

def draw_aacircle (screen):
    wm.set_caption ("primitives.aacircle examples")
    primitives.aacircle (screen, (100, 200), 45, black)
    primitives.aacircle (screen, (200, 160), 80, red)
    primitives.aacircle (screen, (370, 210), 100, green)
    primitives.aacircle (screen, (400, 400), 45, blue)

def draw_circle (screen):
    wm.set_caption ("primitives.circle examples")
    primitives.circle (screen, (100, 200), 45, black)
    primitives.circle (screen, (200, 160), 80, red)
    primitives.circle (screen, (370, 210), 100, green)
    primitives.circle (screen, (400, 400), 45, blue)

def draw_filledcircle (screen):
    wm.set_caption ("primitives.filled_circle examples")
    primitives.filled_circle (screen, (100, 200), 45, black)
    primitives.filled_circle (screen, (200, 160), 80, red)
    primitives.filled_circle (screen, (370, 210), 100, green)
    primitives.filled_circle (screen, (400, 400), 45, blue)

def draw_box (screen):
    wm.set_caption ("primitves.box examples")
    primitives.box (screen, pygame2.Rect (20, 20, 100, 100), black)
    primitives.box (screen, pygame2.Rect (160, 100, 160, 60), red)
    primitives.box (screen, pygame2.Rect (180, 190, 180, 200), green)
    primitives.box (screen, pygame2.Rect (390, 120, 200, 140), blue)

def draw_rectangle (screen):
    wm.set_caption ("primitves.rectangle examples")
    primitives.rectangle (screen, pygame2.Rect (20, 20, 100, 100), black)
    primitives.rectangle (screen, pygame2.Rect (160, 100, 160, 60), red)
    primitives.rectangle (screen, pygame2.Rect (180, 190, 180, 200), green)
    primitives.rectangle (screen, pygame2.Rect (390, 120, 200, 140), blue)

def draw_arc (screen):
    wm.set_caption ("primitives.arc examples")
    primitives.arc (screen, (80, 100), 75, 0, 30, black)
    primitives.arc (screen, (100, 200), 123, 4, 70, red)
    primitives.arc (screen, (300, 300), 78, 200, 30, green)
    primitives.arc (screen, (400, 80), 55, 10, 360, blue)

def draw_line (screen):
    wm.set_caption ("primitives.line examples")
    primitives.line (screen, 4, 40, 17, 320, black)
    primitives.line (screen, 280, 7, 40, 220, red)
    primitives.line (screen, 33, 237, 580, 370, green)
    primitives.line (screen, 0, 0, 640, 480, blue)

def draw_aaline (screen):
    wm.set_caption ("primitives.aaline examples")
    primitives.aaline (screen, 4, 40, 17, 320, black)
    primitives.aaline (screen, 280, 7, 40, 220, red)
    primitives.aaline (screen, 33, 237, 580, 370, green)
    primitives.aaline (screen, 0, 0, 640, 480, blue)

def draw_polygon (screen):
    wm.set_caption ("primitives.polygon examples")
    primitives.polygon (screen, ((4, 40), (280, 7), (40, 220),
                                 (33, 237), (580, 370),
                                 (0, 0), (640, 480)), black)

def draw_aapolygon (screen):
    wm.set_caption ("primitives.aapolygon examples")
    primitives.aapolygon (screen, ((4, 40), (280, 7), (40, 220),
                                   (33, 237), (580, 370),
                                   (0, 0), (640, 480)), black)

def draw_filledpolygon (screen):
    wm.set_caption ("primitives.filled_polygon examples")
    primitives.filled_polygon (screen, ((4, 40), (280, 7), (40, 220),
                                        (33, 237), (580, 370),
                                        (0, 0), (640, 480)), black)

def draw_ellipse (screen):
    wm.set_caption ("primitives.ellipse examnples")
    primitives.ellipse (screen, 210, 400, 50, 50, black)
    primitives.ellipse (screen, 160, 100, 80, 30, red)
    primitives.ellipse (screen, 180, 190, 90, 100, green)
    primitives.ellipse (screen, 390, 120, 100, 70, blue)

def draw_aaellipse (screen):
    wm.set_caption ("primitives.aaellipse examnples")
    primitives.aaellipse (screen, 210, 400, 50, 50, black)
    primitives.aaellipse (screen, 160, 100, 80, 30, red)
    primitives.aaellipse (screen, 180, 190, 90, 100, green)
    primitives.aaellipse (screen, 390, 120, 100, 70, blue)

def draw_filledellipse (screen):
    wm.set_caption ("primitives.filled_ellipse examnples")
    primitives.filled_ellipse (screen, 210, 400, 50, 50, black)
    primitives.filled_ellipse (screen, 160, 100, 80, 30, red)
    primitives.filled_ellipse (screen, 180, 190, 90, 100, green)
    primitives.filled_ellipse (screen, 390, 120, 100, 70, blue)

def draw_aatrigon (screen):
    wm.set_caption ("primitives.aatrigon examples")
    primitives.aatrigon (screen, (10, 10), (420, 40), (60, 200), black)
    primitives.aatrigon (screen, (240, 100), (370, 300), (40, 300), red)
    primitives.aatrigon (screen, (300, 400), (620, 270), (470, 200), green)
    primitives.aatrigon (screen, (33, 400), (440, 460), (312, 410), blue)

def draw_trigon (screen):
    wm.set_caption ("primitives.trigon examples")
    primitives.trigon (screen, (10, 10), (420, 40), (60, 200), black)
    primitives.trigon (screen, (240, 100), (370, 300), (40, 300), red)
    primitives.trigon (screen, (300, 400), (620, 270), (470, 200), green)
    primitives.trigon (screen, (33, 400), (440, 460), (312, 410), blue)

def draw_filledtrigon (screen):
    wm.set_caption ("primitives.filled_trigon examples")
    primitives.filled_trigon (screen, (10, 10), (420, 40), (60, 200), black)
    primitives.filled_trigon (screen, (240, 100), (370, 300), (40, 300), red)
    primitives.filled_trigon (screen, (300, 400), (620, 270), (470, 200), green)
    primitives.filled_trigon (screen, (33, 400), (440, 460), (312, 410), blue)

def draw_bezier (screen):
    wm.set_caption ("primitives.bezier examples")
    primitives.bezier (screen, ((10, 10), (420, 40), (60, 200)), 50, black)
    primitives.bezier (screen, ((240, 100), (370, 300), (40, 300)), 30, red)
    primitives.bezier (screen, ((300, 400), (620, 270), (470, 200)), 10, green)
    primitives.bezier (screen, ((33, 400), (440, 460), (312, 410)), 3, blue)

def draw_pie (screen):
    wm.set_caption ("primitives.pie examples")
    primitives.pie (screen, (100, 200), 45, 0, 90, black)
    primitives.pie (screen, (200, 160), 80, 45, 135, red)
    primitives.pie (screen, (370, 210), 100, 89, 217, green)
    primitives.pie (screen, (400, 400), 45, 10, 350, blue)

def draw_filledpie (screen):
    wm.set_caption ("primitives.filled_pie examples")
    primitives.filled_pie (screen, (100, 200), 45, 0, 90, black)
    primitives.filled_pie (screen, (200, 160), 80, 45, 135, red)
    primitives.filled_pie (screen, (370, 210), 100, 89, 217, green)
    primitives.filled_pie (screen, (400, 400), 45, 10, 350, blue)

def draw_hline (screen):
    wm.set_caption ("primitives.hline examples")
    off = 0
    for y in range (0, 480, 4):
        primitives.hline (screen, 10, 630, y, colors[off])
        off = (off + 1) % 4

def draw_vline (screen):
    wm.set_caption ("primitives.vline examples")
    off = 0
    for x in range (0, 640, 4):
        primitives.vline (screen, x, 10, 470, colors[off])
        off = (off + 1) % 4

def draw_texturedpolygon (screen):
    wm.set_caption ("primitives.textured_polygon examples")
    tex = image.load_bmp (pygame2.examples.RESOURCES.get ("logo.bmp"))
    
    primitives.textured_polygon (screen, ((4, 40), (280, 7), (40, 220),
                                          (33, 237), (580, 370), (0, 0),
                                          (640, 480)), tex, 10, 30)
    
def run ():
    drawtypes = [ draw_aacircle, draw_circle, draw_filledcircle,
                  draw_box, draw_rectangle,
                  draw_arc,
                  draw_line, draw_aaline, draw_hline, draw_vline, 
                  draw_polygon, draw_aapolygon, draw_filledpolygon,
                  draw_texturedpolygon,
                  draw_ellipse, draw_aaellipse, draw_filledellipse,
                  draw_aatrigon, draw_trigon, draw_filledtrigon,
                  draw_bezier,
                  draw_pie, draw_filledpie, 
                  ]
    curtype = 0
    video.init ()

    screen = video.set_mode (640, 480, 32)
    screen.fill (white)
    drawtypes[0] (screen)
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
