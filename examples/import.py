def run ():
    import sys, os

    import pygame2

    try:
        import pygame2.freetype
        import pygame2.freetype.constants
    except ImportError:
        print (sys.exc_info()[1])

    import pygame2.mask

    try:
        import pygame2.midi
    except ImportError:
        print (sys.exc_info()[1])
    
    try:
        import pygame2.sdl
        import pygame2.sdl.audio
        import pygame2.sdl.cdrom
        import pygame2.sdl.constants
        import pygame2.sdl.event
        import pygame2.sdl.cursors
        import pygame2.sdl.gl
        import pygame2.sdl.image
        import pygame2.sdl.joystick
        import pygame2.sdl.keyboard
        import pygame2.sdl.mouse
        import pygame2.sdl.rwops
        import pygame2.sdl.video
        import pygame2.sdl.time
        import pygame2.sdl.wm
    except ImportError:
        print (sys.exc_info()[1])
        
    try:
        import pygame2.sdlext
        import pygame2.sdlext.constants
        import pygame2.sdlext.draw
        import pygame2.sdlext.scrap
        import pygame2.sdlext.transform
    except ImportError:
        print (sys.exc_info()[1])
        
    try:
        import pygame2.sdlext.numericsurfarray
        import pygame2.sdlext.numpysurfarray
        import pygame2.sdlext.surfarray
    except ImportError:
        print (sys.exc_info()[1])


    try:
        import pygame2.sdlgfx
        import pygame2.sdlgfx.constants
        import pygame2.sdlgfx.primitives
        import pygame2.sdlgfx.rotozoom
    except ImportError:
        print (sys.exc_info()[1])

    try:
        import pygame2.sdlmixer
        import pygame2.sdlmixer.channel
        import pygame2.sdlmixer.constants
        import pygame2.sdlmixer.music
    except ImportError:
        print (sys.exc_info()[1])

    try:
        import pygame2.sdlmixer.numericsndarray
        import pygame2.sdlmixer.numpysndarray
        import pygame2.sdlmixer.sndarray
    except ImportError:
        print (sys.exc_info()[1])

    try:
        import pygame2.sdlimage
    except ImportError:
        print (sys.exc_info()[1])

    try:
        import pygame2.sdlttf
        import pygame2.sdlttf.constants
    except ImportError:
        print (sys.exc_info()[1])

    import pygame2.sprite
    import pygame2.threads

if __name__ == "__main__":
    run ()
