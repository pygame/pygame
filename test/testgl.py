#!/usr/bin/env python

'''
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

import sys

from SDL import *
from OpenGL.GL import *

global USE_DEPRECATED_OPENGLBLIT 
USE_DEPRECATED_OPENGLBLIT = False
global_image = None
global_texture = 0
cursor_texture = 0

SHADED_CUBE = True

def HotKey_ToggleFullScreen():
    screen = SDL_GetVideoSurface()
    SDL_WM_ToggleFullScreen(screen)
    if screen.floags & SDL_FULLSCREEN:
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

def HandleEvent(event):
    if event.type == SDL_ACTIVEEVENT:
        pass
    elif event.type == SDL_QUIT:
        return 1
    return 0

def RunGLTest(logo, logocursor, slowly, bpp, gamma, noframe, fsaa, sync, accel):
    w = 640
    h = 480
    done = 0
    color = [(1.0, 1.0, 0.0),
             (1.0, 0.0, 0.0),
             (0.0, 0.0, 0.0),
             (0.0, 1.0, 0.0),
             (0.0, 1.0, 1.0),
             (1.0, 1.0, 1.0),
             (1.0, 0.0, 1.0),
             (0.0, 0.0, 1.0)]
    cube = [( 0.5, 0.5, -0.5), 
            ( 0.5, -0.5, -0.5),
            (-0.5, -0.5, -0.5),
            (-0.5, 0.5, -0.5),
            (-0.5, 0.5, 0.5),
            ( 0.5, 0.5, 0.5),
            ( 0.5, -0.5, 0.5),
            (-0.5, -0.5, 0.5)]

    SDL_Init(SDL_INIT_VIDEO)

    if bpp == 0:
        if SDL_GetVideoInfo().vfmt.BitsPerPixel <= 8:
            bpp = 8
        else:
            bpp = 16 # More doesn't seem to work.. [TODO: huh?]

    if logo and USE_DEPRECATED_OPENGLBLIT:
        video_flags = SDL_OPENGLBLIT
    else:
        video_flags = SDL_OPENGL

    if '-fullscreen' in sys.argv:
        video_flags |= SDL_FULLSCREEN

    if noframe:
        video_flags |= SDL_NOFRAME

    if bpp == 8:
        rgb_size = (3, 3, 2)
    elif bpp in (15, 16):
        rgb_size = (5, 5, 5)
    else:
        rgb_size = (8, 8, 8)

    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, rgb_size[0])
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, rgb_size[1])
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, rgb_size[2])
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 16)
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1)
    if fsaa:
        SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1)
        SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, fsaa)
    if accel:
        SDL_GL_SetAttribute(SDL_GL_ACCELERATED_VISUAL, 1)
    if sync:
        SDL_GL_SetAttribute(SDL_GL_SWAP_CONTROL, 1)
    else:
        SDL_GL_SetAttribute(SDL_GL_SWAP_CONTROL, 0)
    SDL_SetVideoMode(w, h, bpp, video_flags)
    
    print 'Screen BPP: %d' % SDL_GetVideoSurface().format.BitsPerPixel
    print
    print 'Vendor       : %s' % glGetString(GL_VENDOR)
    print 'Renderer     : %s' % glGetString(GL_RENDERER)
    print 'Version      : %s' % glGetString(GL_VERSION)
    print 'Extensions   : %s' % glGetString(GL_EXTENSIONS)

    value = SDL_GL_GetAttribute(SDL_GL_RED_SIZE)
    print 'SDL_GL_RED_SIZE: requested %d, got %d' % (rgb_size[0], value)
    value = SDL_GL_GetAttribute(SDL_GL_GREEN_SIZE)
    print 'SDL_GL_GREEN_SIZE: requested %d, got %d' % (rgb_size[1], value)
    value = SDL_GL_GetAttribute(SDL_GL_BLUE_SIZE)
    print 'SDL_GL_BLUE_SIZE: requested %d, got %d' % (rgb_size[2], value)
    value = SDL_GL_GetAttribute(SDL_GL_DEPTH_SIZE)
    print 'SDL_GL_DEPTH_SIZE: requested %d, got %d' % (16, value)
    value = SDL_GL_GetAttribute(SDL_GL_DOUBLEBUFFER)
    print 'SDL_GL_DEPTH_SIZE: requested %d, got %d' % (1, value)
    if fsaa:
        value = SDL_GL_GetAttribute(SDL_GL_MULTISAMPLEBUFFERS)
        print 'SDL_GL_MULTISAMPLEBUFFERS: requested %d, got %d' % (1, value)
        value = SDL_GL_GetAttribute(SDL_GL_MULTISAMPLESAMPLES)
        print 'SDL_GL_MULTISAMPLESAMPLES: requested %d, got %d' % (fsaa, value)
    if accel:
        value = SDL_GL_GetAttribute(SDL_GL_ACCELERATED_VISUAL)
        print 'SDL_GL_ACCELERATED_VISUAL: requested %d, got %d' % (1, value)
    if sync:
        value = SDL_GL_GetAttribute(SDL_GL_SWAP_CONTROL)
        print 'SDL_GL_SWAP_CONTROL: requested %d, got %d' % (1, value)

    SDL_WM_SetCaption('SDL GL test', 'testgl')
    if gamma != 0.0:
        SDL_SetGamma(gamma, gamma, gamma)

    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    
    glOrtho( -2.0, 2.0, -2.0, 2.0, -20.0, 20.0 )

    glMatrixMode( GL_MODELVIEW )
    glLoadIdentity( )

    glEnable(GL_DEPTH_TEST)

    glDepthFunc(GL_LESS)

    glShadeModel(GL_SMOOTH)

    # Loop until done
    start_time = SDL_GetTicks()
    frames = 0
    while not done:
        glClearColor( 0.0, 0.0, 0.0, 1.0 )
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glBegin( GL_QUADS )
        if SHADED_CUBE:
            glColor(color[0])
            glVertex(cube[0])
            glColor(color[1])
            glVertex(cube[1])
            glColor(color[2])
            glVertex(cube[2])
            glColor(color[3])
            glVertex(cube[3])
            
            glColor(color[3])
            glVertex(cube[3])
            glColor(color[4])
            glVertex(cube[4])
            glColor(color[7])
            glVertex(cube[7])
            glColor(color[2])
            glVertex(cube[2])
            
            glColor(color[0])
            glVertex(cube[0])
            glColor(color[5])
            glVertex(cube[5])
            glColor(color[6])
            glVertex(cube[6])
            glColor(color[1])
            glVertex(cube[1])
            
            glColor(color[5])
            glVertex(cube[5])
            glColor(color[4])
            glVertex(cube[4])
            glColor(color[7])
            glVertex(cube[7])
            glColor(color[6])
            glVertex(cube[6])

            glColor(color[5])
            glVertex(cube[5])
            glColor(color[0])
            glVertex(cube[0])
            glColor(color[3])
            glVertex(cube[3])
            glColor(color[4])
            glVertex(cube[4])

            glColor(color[6])
            glVertex(cube[6])
            glColor(color[1])
            glVertex(cube[1])
            glColor(color[2])
            glVertex(cube[2])
            glColor(color[7])
            glVertex(cube[7])
        else: # flat cube
            glColor(1.0, 0.0, 0.0)
            glVertex(cube[0])
            glVertex(cube[1])
            glVertex(cube[2])
            glVertex(cube[3])
            
            glColor(0.0, 1.0, 0.0)
            glVertex(cube[3])
            glVertex(cube[4])
            glVertex(cube[7])
            glVertex(cube[2])
            
            glColor(0.0, 0.0, 1.0)
            glVertex(cube[0])
            glVertex(cube[5])
            glVertex(cube[6])
            glVertex(cube[1])
            
            glColor(0.0, 1.0, 1.0)
            glVertex(cube[5])
            glVertex(cube[4])
            glVertex(cube[7])
            glVertex(cube[6])

            glColor(1.0, 1.0, 0.0)
            glVertex(cube[5])
            glVertex(cube[0])
            glVertex(cube[3])
            glVertex(cube[4])

            glColor(1.0, 0.0, 1.0)
            glVertex(cube[6])
            glVertex(cube[1])
            glVertex(cube[2])
            glVertex(cube[7])

        glEnd( )
        
        glMatrixMode(GL_MODELVIEW)
        glRotate(5.0, 1.0, 1.0, 1.0)

        # Draw 2D logo onto the 3D display
        if logo:
            if USE_DEPRECATED_OPENGLBLIT:
                pass
            else:
                pass # TODO

        if logocursor:
            pass # TODO

        SDL_GL_SwapBuffers()

        if slowly:
            SDL_Delay(20)

        event = SDL_PollEventAndReturn()
        while event:
            done = HandleEvent(event)
            event = SDL_PollEventAndReturn()
        frames += 1

    this_time = SDL_GetTicks()
    if this_time != start_time:
        print '%2.2f FPS' % (frames / float(this_time - start_time) * 1000.0)

    if global_image:
        SDL_FreeSurface(global_image)
    if global_texture:
        glDeleteTextures(1, global_texture)
    if cursor_texture:
        glDeleteTextures(1, cursor_texture)
    
    SDL_Quit()

if __name__ == '__main__':
    logo = logocursor = 0
    bpp = 0
    slowly = 0
    gamma = 0.0
    noframe = 0
    fsaa = 0
    accel = 0
    sync = 0
    numtests = 1
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '-twice':
            numtests += 1
        elif arg == '-logo':
            logo = 1
        elif arg == '-logoblit':
            logo = 1
            USE_DEPRECATED_OPENGLBLIT = True
        elif arg == '-logocursor':
            logocursor = 1
        elif arg == '-slow':
            slowly = 1
        elif arg == '-bpp':
            i += 1
            bpp = int(sys.argv[i])
        elif arg == '-gamma':
            i += 1
            gamma = float(sys.argv[i])
        elif arg == '-noframe':
            noframe = 1
        elif arg == '-fsaa':
            fsaa += 1
        elif arg == '-accel':
            accel += 1
        elif arg == '-sync':
            sync += 1
        elif arg[:2] == '-h':
            print ('Usage: %s [-twice] [-logo] [-logocursor] [-slow] [-bpp n]'+\
                ' [-gamma n] [-noframe] [-fsaa] [-fullscreen]') % sys.argv[0]
            sys.exit(0)
        i += 1

    for i in range(numtests):
        RunGLTest(logo, logocursor, slowly, bpp, gamma, noframe, fsaa, 
                  sync, accel)
