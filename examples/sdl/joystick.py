import sys
import pygame2

try:
    import pygame2.sdl.constants as sdlconst
    import pygame2.sdlext.draw as draw
    import pygame2.sdl.event as event
    import pygame2.sdl.video as video
    import pygame2.sdl.joystick as joystick
    import pygame2.sdl.wm as wm
except ImportError:
    print ("No pygame2.sdl support")
    sys.exit ()

black = pygame2.Color (0, 0, 0)
white = pygame2.Color (255, 255, 255)
green = pygame2.Color (0, 255, 0)
red = pygame2.Color (255, 0, 0)

def run ():
    video.init ()
    joystick.init ()

    if joystick.num_joysticks () == 0:
        print ("No joysticks found. Exiting.")
        return
    
    screen = video.set_mode (640, 480)
    screen.fill (white)
    screen.flip ()

    wm.set_caption ("Joystick demo")

    crect = pygame2.Rect (400, 300)
    srect = pygame2.Rect (640, 480)
    crect.center = srect.center
    joyrect = pygame2.Rect (4, 4)
    joyrect.center = srect.center

    joy = joystick.Joystick (0)

    btndict = {}
    xoff, yoff = 50, crect.top
    for i in range (joy.num_buttons):
        btndict[i] = [False, pygame2.Rect (xoff, yoff, 20, 20)]
        yoff += 25

    xpart = ((crect.width - joyrect.width) / 2) / 32768.0
    ypart = ((crect.height - joyrect.height) / 2) / 32768.0
    
    okay = True
    while okay:
        for ev in event.get ():
            print ev
            if ev.type == sdlconst.QUIT:
                okay = False
            elif ev.type == sdlconst.KEYDOWN and ev.key == sdlconst.K_ESCAPE:
                okay = False
            elif ev.type == sdlconst.JOYAXISMOTION and ev.joy == joy.index:
                if ev.axis == 0:
                    # x axis movement
                    if ev.value == 0:
                        joyrect.centerx = crect.centerx
                    else:
                        joyrect.centerx = crect.centerx + ev.value * xpart
                elif ev.axis == 1:
                    # y axis movement
                    if ev.value == 0:
                        joyrect.centery = crect.centery
                    else:
                        joyrect.centery = crect.centery + ev.value * ypart

            elif ev.type == sdlconst.JOYBUTTONDOWN and ev.joy == joy.index:
                btndict[ev.button][0] = True
            elif ev.type == sdlconst.JOYBUTTONUP and ev.joy == joy.index:
                btndict[ev.button][0] = False
            elif ev.type == sdlconst.JOYHATMOTION and ev.joy == joy.index:
                pass
            elif ev.type == sdlconst.JOYBALLMOTION and ev.joy == joy.index:
                pass

            draw.rect (screen, black, crect)
            draw.line (screen, white, crect.midtop, crect.midbottom)
            draw.line (screen, white, crect.midleft, crect.midright)
            draw.rect (screen, green, joyrect)

            for (state, rect) in btndict.values():
                if state:
                    draw.rect (screen, green, rect)
                else:
                    draw.rect (screen, black, rect)
            
            screen.flip ()
    video.quit ()

if __name__ == "__main__":
    run ()
