#!/usr/bin/env python

'''Simple program: Loop, watching keystrokes.

Note that you need to call `SDL_PollEvent` or `SDL_WaitEvent` to
pump the event loop and catch keystrokes.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

import sys

from SDL import *

def print_modifiers():
    print ' modifiers:',
    mod = SDL_GetModState()
    if not mod:
        print ' (none)',
        return
    if mod & KMOD_LSHIFT:
        print ' LSHIFT',
    if mod & KMOD_RSHIFT:
        print ' RSHIFT',
    if mod & KMOD_LCTRL:
        print ' LCTRL',
    if mod & KMOD_RCTRL:
        print ' RCTRL',
    if mod & KMOD_LALT:
        print ' LALT',
    if mod & KMOD_RALT:
        print ' RALT',
    if mod & KMOD_LMETA:
        print ' LMETA',
    if mod & KMOD_RMETA:
        print ' RMETA',
    if mod & KMOD_NUM:
        print ' NUM',
    if mod & KMOD_CAPS:
        print ' CAPS',
    if mod & KMOD_MODE:
        print ' MODE',

def PrintKey(sym, pressed):
    # Print the keycode, name and state
    if pressed:
        p = 'pressed'
    else:
        p = 'release'
    if sym.sym:
        print 'Key %s:  %d-%s ,' % (p, sym.sym, SDL_GetKeyName(sym.sym)),
    else:
        print 'Unknown Key (scancode = %d) %s ' % (sym.scancode, p),

    # Print the translated character, if one exists
    if sym.unicode:
        print u' (%s)' % sym.unicode
    print_modifiers()
    print

if __name__ == '__main__':
    SDL_Init(SDL_INIT_VIDEO)

    videoflags = SDL_SWSURFACE
    for arg in sys.argv[1:]:
        if arg == '-fullscreen':
            videoflags |= SDL_FULLSCREEN
        else:
            print >> sys.stderr, 'Usage: %s [-fullscreen]' % sys.argv[0]
            sys.exit(1)

    # Set 640x480 video mode
    SDL_SetVideoMode(640, 480, 0, videoflags)

    # Enable Unicode translation for keyboard input
    SDL_EnableUNICODE(True)

    # Enable auto repeat for keyboard input
    SDL_EnableKeyRepeat(SDL_DEFAULT_REPEAT_DELAY,
                        SDL_DEFAULT_REPEAT_INTERVAL)

    # Watch keystrokes
    done = False
    while not done:
        event = SDL_WaitEventAndReturn()
        if event.type == SDL_KEYDOWN:
            PrintKey(event.keysym, True)
        elif event.type == SDL_KEYUP:
            PrintKey(event.keysym, False)
        elif event.type in (SDL_MOUSEBUTTONDOWN, SDL_QUIT):
            done = True
    
    SDL_Quit()
