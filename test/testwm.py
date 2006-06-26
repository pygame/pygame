#!/usr/bin/env python

'''Test out the window manager interaction functions.

BUG: Segfaults in SDL_PrivateResize (events/SDL_resize.c) on Linux/AMD64
     when window resized; intermittent.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

import os
import sys

from SDL import *

ICON_BMP = os.path.join(os.path.dirname(sys.argv[0]), 'icon.bmp')

visible = 1
video_bpp = 0
video_flags = 0
reallyquit = 0

def SetVideoMode(w, h):
    screen = SDL_SetVideoMode(w, h, video_bpp, video_flags)
    if screen.flags & SDL_FULLSCREEN:
        print 'Running in fullscreen mode'
    else:
        print 'Running in windowed mode'

    palette = [SDL_Color(255 - i, 255 - i, 255 - i) for i in range(256)]
    SDL_SetColors(screen, palette, 0)

    SDL_LockSurface(screen)
    buffer = screen.pixels.as_bytes()
    w = screen.w*screen.format.BytesPerPixel
    for i in range(screen.h):
        buffer[i*screen.pitch:i*screen.pitch + w] = [(i * 255)/screen.h] * w
    SDL_UnlockSurface(screen)
    SDL_UpdateRect(screen, 0, 0, 0, 0)

def LoadIconSurface(file):
    icon = SDL_LoadBMP(file)

    if not icon.format.palette:
        print >> sys.stderr, 'Icon must have a palette'
        SDL_FreeSurface(icon)
        return None, None

    SDL_SetColorKey(icon, SDL_SRCCOLORKEY, icon.pixels.as_bytes()[0])
    pixels = icon.pixels.as_bytes()
    print 'Transparent pixel: (%d,%d,%d)' % \
        (icon.format.palette.colors[pixels[0]].r,
         icon.format.palette.colors[pixels[0]].g,
         icon.format.palette.colors[pixels[0]].b)
    mlen = (icon.w * icon.h + 7) / 8
    mask = [0] * mlen
    for i in range(icon.h):
        for j in range(icon.w):
            pindex = i * icon.pitch + j
            mindex = i * icon.w + j
            if pixels[pindex] != pixels[0]:
                mask[mindex>>3] |= 1 << (7 - (mindex & 7))
    
    return icon, mask

def HotKey_ToggleFullScreen():
    screen = SDL_GetVideoSurface()
    SDL_WM_ToggleFullScreen(screen)
    if screen.flags & SDL_FULLSCREEN:
        s = 'fullscreen'
    else:
        s = 'windowed'
    print 'Toggled fullscreen mode - now %s' % s

def HotKey_ToggleGrab():
    print 'Ctrl-G: toggling input grab!'
    mode = SDL_WM_GrabInput(SDL_GRAB_QUERY)
    if mode == SDL_GRAB_ON:
        print 'Grab was on'
    else:
        print 'Grab was off'
    mode = SDL_WM_GrabInput(not mode)
    if mode == SDL_GRAB_ON:
        print 'Grab is now on'
    else:
        print 'Grab is now off'

def HotKey_Iconify():
    print 'Ctrl-Z: iconifying window!'
    SDL_WM_IconifyWindow()

def HotKey_Quit():
    print 'Posting internal quit request'
    event = SDL_Event()
    event.type = SDL_USEREVENT
    SDL_PushEvent(event)

def FilterEvents(event):
    global visible
    global reallyquit

    if event.type == SDL_ACTIVEEVENT:
        s1 = 'lost'
        if event.gain:
            s1 = 'gained'
        s2 = ''
        if event.state & SDL_APPACTIVE:
            s2 = 'active'
        if event.state & SDL_APPMOUSEFOCUS:
            s2 = 'mouse'
        if event.state & SDL_APPINPUTFOCUS:
            s2 = 'input'
        print 'app %s %s focus' % (s1, s2)
        return 0
    elif event.type in (SDL_MOUSEBUTTONDOWN, SDL_MOUSEBUTTONUP):
        if event.state == SDL_PRESSED:
            visible = not visible
            SDL_ShowCursor(visible)
            statestr = 'pressed'
        else:
            statestr = 'released'
        print 'Mouse button %d has been %s' % (event.button, statestr)
        return 0
    elif event.type == SDL_KEYDOWN:
        if event.keysym.sym == SDLK_ESCAPE:
            HotKey_Quit()
        elif event.keysym.sym == SDLK_g and event.keysym.mod & KMOD_CTRL:
            HotKey_ToggleGrab()
        if event.keysym.sym == SDLK_z and event.keysym.mod & KMOD_CTRL:
            HotKey_Iconify()
        if event.keysym.sym == SDLK_RETURN and event.keysym.mod & KMOD_ALT:
            HotKey_ToggleFullScreen()
        print 'key "%s" pressed' % SDL_GetKeyName(event.keysym.sym)
        return 0
    elif event.type == SDL_VIDEORESIZE:
        return 1
    elif event.type == SDL_QUIT:
        if not reallyquit:
            reallyquit = 1
            print 'Quit requested'
            return 0
        print 'Quit demanded'
        return 1
    elif event.type == SDL_USEREVENT:
        return 1
    else:
        return 0

if __name__ == '__main__':
    SDL_Init(SDL_INIT_VIDEO)

    w = 640
    h = 480
    video_bpp = 8
    video_flags = SDL_SWSURFACE
    i = 1
    while i < len(sys.argv): 
        arg = sys.argv[i]
        if arg == '-fullscreen':
            video_flags |= SDL_FULLSCREEN
        elif arg == '-resize':
            video_flags |= SDL_RESIZABLE
        elif arg == '-noframe':
            video_flags |= SDL_NOFRAME
        elif arg == '-width':
            i += 1
            w = int(sys.argv[i])
        elif arg == '-height':
            i += 1
            h = int(sys.argv[i])
        elif arg == '-bpp':
            i += 1
            video_bpp = int(sys.argv[i])
        else:
            break
        i += 1

    icon, icon_mask = LoadIconSurface(ICON_BMP)
    if icon:
        SDL_WM_SetIcon(icon, icon_mask)

    if i >= len(sys.argv):
        title = 'Testing 1.. 2.. 3...'
    else:
        title = sys.argv[i]
    SDL_WM_SetCaption(title, 'testwm')

    title, titlemini = SDL_WM_GetCaption()
    if title:
        print 'Title was set to: %s' % title
    else:
        print 'No window title was set!'

    SetVideoMode(w, h)

    SDL_SetEventFilter(FilterEvents)
    SDL_EventState(SDL_KEYUP, SDL_IGNORE)

    while True:
        event = SDL_WaitEventAndReturn()
        if event.type == SDL_VIDEORESIZE:
            print 'Got a resize event: %dx%d' % (event.w, event.h)
            SetVideoMode(event.w, event.h)
        elif event.type == SDL_USEREVENT:
            print 'Handling internal quit request'
            print 'Bye bye..'
            break
        elif event.type == SDL_QUIT:
            print 'Bye bye..'
            break
        else:
            print "Warning: Event %d wasn't filtered" % event.type

    SDL_Quit()
