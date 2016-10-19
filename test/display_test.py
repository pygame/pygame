if __name__ == '__main__':
    import sys
    import os
    pkg_dir = os.path.split(os.path.abspath(__file__))[0]
    parent_dir, pkg_name = os.path.split(pkg_dir)
    is_pygame_pkg = (pkg_name == 'tests' and
                     os.path.split(parent_dir)[1] == 'pygame')
    if not is_pygame_pkg:
        sys.path.insert(0, parent_dir)
else:
    is_pygame_pkg = __name__.startswith('pygame.tests.')

import unittest
import pygame, pygame.transform

class DisplayModuleTest( unittest.TestCase ):
    def test_update( self ):
        """ see if pygame.display.update takes rects with negative values.
            "|Tags:display|"
        """

        if 1:
            pygame.init()
            screen = pygame.display.set_mode((100,100))
            screen.fill((55,55,55))

            r1 = pygame.Rect(0,0,100,100)
            pygame.display.update(r1)

            r2 = pygame.Rect(-10,0,100,100)
            pygame.display.update(r2)

            r3 = pygame.Rect(-10,0,-100,-100)
            pygame.display.update(r3)

            # NOTE: if I don't call pygame.quit there is a segfault.  hrmm.
            pygame.quit()
            #  I think it's because unittest runs stuff in threads
            # here's a stack trace...
            
            # NOTE to author of above:
            #   unittest doesn't run tests in threads    
            #   segfault was probably caused by another tests need 
            #   for a "clean slate"
            
            """
    #0  0x08103b7c in PyFrame_New ()
    #1  0x080bd666 in PyEval_EvalCodeEx ()
    #2  0x08105202 in PyFunction_SetClosure ()
    #3  0x080595ae in PyObject_Call ()
    #4  0x080b649f in PyEval_CallObjectWithKeywords ()
    #5  0x08059585 in PyObject_CallObject ()
    #6  0xb7f7aa2d in initbase () from /usr/lib/python2.4/site-packages/pygame/base.so
    #7  0x080e09bd in Py_Finalize ()
    #8  0x08055597 in Py_Main ()
    #9  0xb7e04eb0 in __libc_start_main () from /lib/tls/libc.so.6
    #10 0x08054e31 in _start ()

            """

    def todo_test_Info(self):

        # __doc__ (as of 2008-08-02) for pygame.display.Info:

          # pygame.display.Info(): return VideoInfo
          # Create a video display information object
          # 
          # Creates a simple object containing several attributes to describe
          # the current graphics environment. If this is called before
          # pygame.display.set_mode() some platforms can provide information
          # about the default display mode. This can also be called after
          # setting the display mode to verify specific display options were
          # satisfied. The VidInfo object has several attributes:
          # 
          #   hw:         True if the display is hardware accelerated
          #   wm:         True if windowed display modes can be used
          #   video_mem:  The megabytes of video memory on the display. This is 0 if unknown
          #   bitsize:    Number of bits used to store each pixel
          #   bytesize:   Number of bytes used to store each pixel
          #   masks:      Four values used to pack RGBA values into pixels
          #   shifts:     Four values used to pack RGBA values into pixels
          #   losses:     Four values used to pack RGBA values into pixels
          #   blit_hw:    True if hardware Surface blitting is accelerated
          #   blit_hw_CC: True if hardware Surface colorkey blitting is accelerated
          #   blit_hw_A:  True if hardware Surface pixel alpha blitting is accelerated
          #   blit_sw:    True if software Surface blitting is accelerated
          #   blit_sw_CC: True if software Surface colorkey blitting is accelerated
          #   blit_sw_A:  True if software Surface pixel alpha blitting is acclerated
          #   current_h, current_h:  Width and height of the current video mode, or of the
          #     desktop mode if called before the display.set_mode is called.
          #     (current_h, current_w are available since SDL 1.2.10, and pygame 1.8.0)
          #     They are -1 on error, or if an old SDL is being used.

        self.fail()
        
        if 0:
        
            pygame.init()
            inf = pygame.display.Info()
            print ("before a display mode has been set")
            print (inf)
        
            self.assertNotEqual(inf.current_h, -1)
            self.assertNotEqual(inf.current_w, -1)
            #probably have an older SDL than 1.2.10 if -1.
        
            screen = pygame.display.set_mode((100,100))
            inf = pygame.display.Info()
        
            print (inf)
        
            self.assertNotEqual(inf.current_h, -1)
            self.assertEqual(inf.current_h, 100)
            self.assertEqual(inf.current_w, 100)
        
            #pygame.quit()

    def todo_test_flip(self):

        # __doc__ (as of 2008-08-02) for pygame.display.flip:

          # pygame.display.flip(): return None
          # update the full display Surface to the screen
          # 
          # This will update the contents of the entire display. If your display
          # mode is using the flags pygame.HWSURFACE and pygame.DOUBLEBUF, this
          # will wait for a vertical retrace and swap the surfaces. If you are
          # using a different type of display mode, it will simply update the
          # entire contents of the surface.
          # 
          # When using an pygame.OPENGL display mode this will perform a gl buffer swap. 

        self.fail() 

    def todo_test_get_active(self):

        # __doc__ (as of 2008-08-02) for pygame.display.get_active:

          # pygame.display.get_active(): return bool
          # true when the display is active on the display
          # 
          # After pygame.display.set_mode() is called the display Surface will
          # be visible on the screen. Most windowed displays can be hidden by
          # the user. If the display Surface is hidden or iconified this will
          # return False.
          # 

        self.fail() 

    def todo_test_get_caption(self):

        # __doc__ (as of 2008-08-02) for pygame.display.get_caption:

          # pygame.display.get_caption(): return (title, icontitle)
          # get the current window caption
          # 
          # Returns the title and icontitle for the display Surface. These will
          # often be the same value.
          # 

        self.fail() 

    def todo_test_get_driver(self):

        # __doc__ (as of 2008-08-02) for pygame.display.get_driver:

          # pygame.display.get_driver(): return name
          # get the name of the pygame display backend
          # 
          # Pygame chooses one of many available display backends when it is
          # initialized. This returns the internal name used for the display
          # backend. This can be used to provide limited information about what
          # display capabilities might be accelerated. See the SDL_VIDEODRIVER
          # flags in pygame.display.set_mode() to see some of the common
          # options.
          # 

        self.fail() 

    def todo_test_get_init(self):

        # __doc__ (as of 2008-08-02) for pygame.display.get_init:

          # pygame.display.get_init(): return bool
          # true if the display module is initialized
          # 
          # Returns True if the pygame.display module is currently initialized. 

        self.fail() 

    def todo_test_get_surface(self):

        # __doc__ (as of 2008-08-02) for pygame.display.get_surface:

          # pygame.display.get_surface(): return Surface
          # get a reference to the currently set display surface
          # 
          # Return a reference to the currently set display Surface. If no
          # display mode has been set this will return None.
          # 

        self.fail() 

    def todo_test_get_wm_info(self):

        # __doc__ (as of 2008-08-02) for pygame.display.get_wm_info:

          # pygame.display.get_wm_info(): return dict
          # Get information about the current windowing system
          # 
          # Creates a dictionary filled with string keys. The strings and values
          # are arbitrarily created by the system. Some systems may have no
          # information and an empty dictionary will be returned. Most platforms
          # will return a "window" key with the value set to the system id for
          # the current display.
          # 
          # New with pygame 1.7.1 

        self.fail() 

    def todo_test_gl_get_attribute(self):

        # __doc__ (as of 2008-08-02) for pygame.display.gl_get_attribute:

          # pygame.display.gl_get_attribute(flag): return value
          # get the value for an opengl flag for the current display
          # 
          # After calling pygame.display.set_mode() with the pygame.OPENGL flag,
          # it is a good idea to check the value of any requested OpenGL
          # attributes. See pygame.display.gl_set_attribute() for a list of
          # valid flags.
          # 

        self.fail() 

    def todo_test_gl_set_attribute(self):

        # __doc__ (as of 2008-08-02) for pygame.display.gl_set_attribute:

          # pygame.display.gl_set_attribute(flag, value): return None
          # request an opengl display attribute for the display mode
          # 
          # When calling pygame.display.set_mode() with the pygame.OPENGL flag,
          # Pygame automatically handles setting the OpenGL attributes like
          # color and doublebuffering. OpenGL offers several other attributes
          # you may want control over. Pass one of these attributes as the flag,
          # and its appropriate value. This must be called before
          # pygame.display.set_mode()
          # 
          # The OPENGL flags are; 
          #   GL_ALPHA_SIZE, GL_DEPTH_SIZE, GL_STENCIL_SIZE, GL_ACCUM_RED_SIZE,
          #   GL_ACCUM_GREEN_SIZE,  GL_ACCUM_BLUE_SIZE, GL_ACCUM_ALPHA_SIZE,
          #   GL_MULTISAMPLEBUFFERS, GL_MULTISAMPLESAMPLES, GL_STEREO

        self.fail() 

    def todo_test_iconify(self):

        # __doc__ (as of 2008-08-02) for pygame.display.iconify:

          # pygame.display.iconify(): return bool
          # iconify the display surface
          # 
          # Request the window for the display surface be iconified or hidden.
          # Not all systems and displays support an iconified display. The
          # function will return True if successfull.
          # 
          # When the display is iconified pygame.display.get_active() will
          # return False. The event queue should receive a ACTIVEEVENT event
          # when the window has been iconified.
          # 

        self.fail() 

    def todo_test_init(self):

        # __doc__ (as of 2008-08-02) for pygame.display.init:

          # pygame.display.init(): return None
          # initialize the display module
          # 
          # Initializes the pygame display module. The display module cannot do
          # anything until it is initialized. This is usually handled for you
          # automatically when you call the higher level pygame.init().
          # 
          # Pygame will select from one of several internal display backends
          # when it is initialized. The display mode will be chosen depending on
          # the platform and permissions of current user. Before the display
          # module is initialized the environment variable SDL_VIDEODRIVER can
          # be set to control which backend is used. The systems with multiple
          # choices are listed here.
          # 
          #    Windows : windib, directx
          #    Unix    : x11, dga, fbcon, directfb, ggi, vgl, svgalib, aalib
          # On some platforms it is possible to embed the pygame display into an
          # already existing window. To do this, the environment variable
          # SDL_WINDOWID must be set to a string containing the window id or
          # handle. The environment variable is checked when the pygame display
          # is initialized. Be aware that there can be many strange side effects
          # when running in an embedded display.
          # 
          # It is harmless to call this more than once, repeated calls have no effect. 

        self.fail() 

    def todo_test_list_modes(self):

        # __doc__ (as of 2008-08-02) for pygame.display.list_modes:

          # pygame.display.list_modes(depth=0, flags=pygame.FULLSCREEN): return list
          # get list of available fullscreen modes
          # 
          # This function returns a list of possible dimensions for a specified
          # color depth. The return value will be an empty list if no display
          # modes are available with the given arguments. A return value of -1
          # means that any requested resolution should work (this is likely the
          # case for windowed modes). Mode sizes are sorted from biggest to
          # smallest.
          # 
          # If depth is 0, SDL will choose the current/best color depth for the
          # display. The flags defaults to pygame.FULLSCREEN, but you may need
          # to add additional flags for specific fullscreen modes.
          # 

        self.fail() 

    def todo_test_mode_ok(self):

        # __doc__ (as of 2008-08-02) for pygame.display.mode_ok:

          # pygame.display.mode_ok(size, flags=0, depth=0): return depth
          # pick the best color depth for a display mode
          # 
          # This function uses the same arguments as pygame.display.set_mode().
          # It is used to depermine if a requested display mode is available. It
          # will return 0 if the display mode cannot be set. Otherwise it will
          # return a pixel depth that best matches the display asked for.
          # 
          # Usually the depth argument is not passed, but some platforms can
          # support multiple display depths. If passed it will hint to which
          # depth is a better match.
          # 
          # The most useful flags to pass will be pygame.HWSURFACE,
          # pygame.DOUBLEBUF, and maybe pygame.FULLSCREEN. The function will
          # return 0 if these display flags cannot be set.
          # 

        self.fail() 

    def todo_test_quit(self):

        # __doc__ (as of 2008-08-02) for pygame.display.quit:

          # pygame.display.quit(): return None
          # uninitialize the display module
          # 
          # This will shut down the entire display module. This means any active
          # displays will be closed. This will also be handled automatically
          # when the program exits.
          # 
          # It is harmless to call this more than once, repeated calls have no effect. 

        self.fail() 

    def todo_test_set_caption(self):

        # __doc__ (as of 2008-08-02) for pygame.display.set_caption:

          # pygame.display.set_caption(title, icontitle=None): return None
          # set the current window caption
          # 
          # If the display has a window title, this function will change the
          # name on the window. Some systems support an alternate shorter title
          # to be used for minimized displays.
          # 

        self.fail() 

    def todo_test_set_gamma(self):

        # __doc__ (as of 2008-08-02) for pygame.display.set_gamma:

          # pygame.display.set_gamma(red, green=None, blue=None): return bool
          # change the hardware gamma ramps
          # 
          # Set the red, green, and blue gamma values on the display hardware.
          # If the green and blue arguments are not passed, they will both be
          # the same as red. Not all systems and hardware support gamma ramps,
          # if the function succeeds it will return True.
          # 
          # A gamma value of 1.0 creates a linear color table. Lower values will
          # darken the display and higher values will brighten.
          # 

        self.fail() 

    def todo_test_set_gamma_ramp(self):

        # __doc__ (as of 2008-08-02) for pygame.display.set_gamma_ramp:

          # change the hardware gamma ramps with a custom lookup
          # pygame.display.set_gamma_ramp(red, green, blue): return bool
          # set_gamma_ramp(red, green, blue): return bool
          # 
          # Set the red, green, and blue gamma ramps with an explicit lookup
          # table. Each argument should be sequence of 256 integers. The
          # integers should range between 0 and 0xffff. Not all systems and
          # hardware support gamma ramps, if the function succeeds it will
          # return True.
          # 

        self.fail() 

    def todo_test_set_icon(self):

        # __doc__ (as of 2008-08-02) for pygame.display.set_icon:

          # pygame.display.set_icon(Surface): return None
          # change the system image for the display window
          # 
          # Sets the runtime icon the system will use to represent the display
          # window. All windows default to a simple pygame logo for the window
          # icon.
          # 
          # You can pass any surface, but most systems want a smaller image
          # around 32x32. The image can have colorkey transparency which will be
          # passed to the system.
          # 
          # Some systems do not allow the window icon to change after it has
          # been shown. This function can be called before
          # pygame.display.set_mode() to create the icon before the display mode
          # is set.
          # 

        self.fail() 

    def todo_test_set_mode(self):

        # __doc__ (as of 2008-08-02) for pygame.display.set_mode:

          # pygame.display.set_mode(resolution=(0,0), flags=0, depth=0): return Surface
          # initialize a window or screen for display
          # 
          # This function will create a display Surface. The arguments passed in
          # are requests for a display type. The actual created display will be
          # the best possible match supported by the system.
          # 
          # The resolution argument is a pair of numbers representing the width
          # and height. The flags argument is a collection of additional
          # options.  The depth argument represents the number of bits to use
          # for color.
          # 
          # The Surface that gets returned can be drawn to like a regular
          # Surface but changes will eventually be seen on the monitor.
          # 
          # If no resolution is passed or is set to (0, 0) and pygame uses SDL
          # version 1.2.10 or above, the created Surface will have the same size
          # as the current screen resolution. If only the width or height are
          # set to 0, the Surface will have the same width or height as the
          # screen resolution. Using a SDL version prior to 1.2.10 will raise an
          # exception.
          # 
          # It is usually best to not pass the depth argument. It will default
          # to the best and fastest color depth for the system. If your game
          # requires a specific color format you can control the depth with this
          # argument. Pygame will emulate an unavailable color depth which can
          # be slow.
          # 
          # When requesting fullscreen display modes, sometimes an exact match
          # for the requested resolution cannot be made. In these situations
          # pygame will select the closest compatable match. The returned
          # surface will still always match the requested resolution.
          # 
          # The flags argument controls which type of display you want. There
          # are several to choose from, and you can even combine multiple types
          # using the bitwise or operator, (the pipe "|" character). If you pass
          # 0 or no flags argument it will default to a software driven window.
          # Here are the display flags you will want to choose from:
          # 
          #    pygame.FULLSCREEN    create a fullscreen display
          #    pygame.DOUBLEBUF     recommended for HWSURFACE or OPENGL
          #    pygame.HWSURFACE     hardware accelerated, only in FULLSCREEN
          #    pygame.OPENGL        create an opengl renderable display
          #    pygame.RESIZABLE     display window should be sizeable
          #    pygame.NOFRAME       display window will have no border or controls

        self.fail() 

    def todo_test_set_palette(self):

        # __doc__ (as of 2008-08-02) for pygame.display.set_palette:

          # pygame.display.set_palette(palette=None): return None
          # set the display color palette for indexed displays
          # 
          # This will change the video display color palette for 8bit displays.
          # This does not change the palette for the actual display Surface,
          # only the palette that is used to display the Surface. If no palette
          # argument is passed, the system default palette will be restored. The
          # palette is a sequence of RGB triplets.
          # 

        self.fail() 

    def todo_test_toggle_fullscreen(self):

        # __doc__ (as of 2008-08-02) for pygame.display.toggle_fullscreen:

          # pygame.display.toggle_fullscreen(): return bool
          # switch between fullscreen and windowed displays
          # 
          # Switches the display window between windowed and fullscreen modes.
          # This function only works under the unix x11 video driver. For most
          # situations it is better to call pygame.display.set_mode() with new
          # display flags.
          # 

        self.fail() 

if __name__ == '__main__':
    unittest.main()
