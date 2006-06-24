#!/usr/bin/env python

'''Simple program to test the SDL joystick routines.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

import sys

from SDL import *

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480

def WatchJoystick(joystick):
    x, y, draw = 0, 0, 0
    axis_area = [SDL_Rect(), SDL_Rect()]

    screen = SDL_SetVideoMode(SCREEN_WIDTH, SCREEN_HEIGHT, 16, 0)

    name = SDL_JoystickName(SDL_JoystickIndex(joystick))
    if not name:
        name = 'Unknown Joystick'
    print 'Watching joystick %d: (%s)' % (SDL_JoystickIndex(joystick), name)

    done = 0
    while not done:
        event = SDL_PollEventAndReturn()
        while event:
            if event.type == SDL_JOYAXISMOTION:
                print 'Joystick %d axis %d value: %d' % \
                    (event.which, event.axis, event.value)
            elif event.type == SDL_JOYHATMOTION:
                print ('Joystick %d hat %d value:' % \
                    (event.which, event.hat)),
                if event.value == SDL_HAT_CENTERED:
                    print ' centered'
                elif event.value == SDL_HAT_UP:
                    print ' up'
                elif event.value == SDL_HAT_RIGHT:
                    print ' right'
                elif event.value == SDL_HAT_DOWN:
                    print ' down'
                elif event.value == SDL_HAT_LEFT:
                    print ' left'
            elif event.type == SDL_JOYBALLMOTION:
                print 'Joystick %d ball %d delta: (%d,%d)' % \
                    (event.which, event.ball, event.xrel, event.yrel)
            elif event.type == SDL_JOYBUTTONDOWN:
                print 'Joystick %d button %d down' % \
                    (event.which, event.button)
            elif event.type == SDL_JOYBUTTONUP:
                print 'Joystick %d button %d up' % \
                    (event.which, event.button)
            elif (event.type == SDL_KEDOWN and \
                  event.keysym.sym == SDLK_ESCAPE) or event.type == SDL_QUIT:
                done = 1
            event = SDL_PollEventAndReturn()
        
        # Update visual joystick state
        for i in range(SDL_JoystickNumButtons(joystick)):
            area = SDL_Rect(i*34, SCREEN_HEIGHT-34, 32, 32)
            if SDL_JoystickGetButton(joystick, i) == SDL_PRESSED:
                SDL_FillRect(screen, area, 0xffff)
            else:
                SDL_FillRect(screen, area, 0x0)
            SDL_UpdateRects(screen, [area])

        # Erase previous axes
        SDL_FillRect(screen, axis_area[draw], 0x0000)
            
        # Draw the X/Y axis
        draw = not draw
        x = SDL_JoystickGetAxis(joystick, 0) + 32768
        x *= SCREEN_WIDTH
        x /= 65535
        x = min(max(0, x), SCREEN_WIDTH-16)
        y = SDL_JoystickGetAxis(joystick, 1) + 32768
        y *= SCREEN_HEIGHT
        y /= 65535
        y = min(max(0, y), SCREEN_HEIGHT-16)
        axis_area[draw].x = x
        axis_area[draw].y = y
        axis_area[draw].w = 16
        axis_area[draw].h = 16
        SDL_FillRect(screen, axis_area[draw], 0xffff)
        SDL_UpdateRects(screen, axis_area)

if __name__ == '__main__':
    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_JOYSTICK)
    print 'There are %d joysticks attached' % SDL_NumJoysticks()
    for i in range(SDL_NumJoysticks()):
        name = SDL_JoystickName(i)
        if not name:
            name = 'Unknown Joystick'
        print 'Joystick %d: %s' % (i, name)

    if len(sys.argv) > 1:
        joystick = SDL_JoystickOpen(int(sys.argv[1]))
        WatchJoystick(joystick)
        SDL_JoystickClose(joystick)

    SDL_QuitSubSystem(SDL_INIT_VIDEO | SDL_INIT_JOYSTICK)

