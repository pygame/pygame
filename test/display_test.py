import test_utils
import test.unittest as unittest
import pygame, pygame.transform

from test_utils import test_not_implemented

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
    def test_Info(self):
    
        # __doc__ (as of 2008-06-25) for pygame.display.Info:
    
          # pygame.display.Info(): return VideoInfo
          # Create a video display information object
    
        self.assert_(test_not_implemented()) 
    
    def test_flip(self):
    
        # __doc__ (as of 2008-06-25) for pygame.display.flip:
    
          # pygame.display.flip(): return None
          # update the full display Surface to the screen
    
        self.assert_(test_not_implemented()) 
    
    def test_get_active(self):
    
        # __doc__ (as of 2008-06-25) for pygame.display.get_active:
    
          # pygame.display.get_active(): return bool
          # true when the display is active on the display
    
        self.assert_(test_not_implemented()) 
    
    def test_get_caption(self):
    
        # __doc__ (as of 2008-06-25) for pygame.display.get_caption:
    
          # pygame.display.get_caption(): return (title, icontitle)
          # get the current window caption
    
        self.assert_(test_not_implemented()) 
    
    def test_get_driver(self):
    
        # __doc__ (as of 2008-06-25) for pygame.display.get_driver:
    
          # pygame.display.get_driver(): return name
          # get the name of the pygame display backend
    
        self.assert_(test_not_implemented()) 
    
    def test_get_init(self):
    
        # __doc__ (as of 2008-06-25) for pygame.display.get_init:
    
          # pygame.display.get_init(): return bool
          # true if the display module is initialized
    
        self.assert_(test_not_implemented()) 
    
    def test_get_surface(self):
    
        # __doc__ (as of 2008-06-25) for pygame.display.get_surface:
    
          # pygame.display.get_surface(): return Surface
          # get a reference to the currently set display surface
    
        self.assert_(test_not_implemented()) 
    
    def test_get_wm_info(self):
    
        # __doc__ (as of 2008-06-25) for pygame.display.get_wm_info:
    
          # pygame.display.get_wm_info(): return dict
          # Get information about the current windowing system
    
        self.assert_(test_not_implemented()) 
    
    def test_gl_get_attribute(self):
    
        # __doc__ (as of 2008-06-25) for pygame.display.gl_get_attribute:
    
          # pygame.display.gl_get_attribute(flag): return value
          # get the value for an opengl flag for the current display
    
        self.assert_(test_not_implemented()) 
    
    def test_gl_set_attribute(self):
    
        # __doc__ (as of 2008-06-25) for pygame.display.gl_set_attribute:
    
          # pygame.display.gl_set_attribute(flag, value): return None
          # request an opengl display attribute for the display mode
    
        self.assert_(test_not_implemented()) 
    
    def test_iconify(self):
    
        # __doc__ (as of 2008-06-25) for pygame.display.iconify:
    
          # pygame.display.iconify(): return bool
          # iconify the display surface
    
        self.assert_(test_not_implemented()) 
    
    def test_init(self):
    
        # __doc__ (as of 2008-06-25) for pygame.display.init:
    
          # pygame.display.init(): return None
          # initialize the display module
    
        self.assert_(test_not_implemented()) 
    
    def test_list_modes(self):
    
        # __doc__ (as of 2008-06-25) for pygame.display.list_modes:
    
          # pygame.display.list_modes(depth=0, flags=pygame.FULLSCREEN): return list
          # get list of available fullscreen modes
    
        self.assert_(test_not_implemented()) 
    
    def test_mode_ok(self):
    
        # __doc__ (as of 2008-06-25) for pygame.display.mode_ok:
    
          # pygame.display.mode_ok(size, flags=0, depth=0): return depth
          # pick the best color depth for a display mode
    
        self.assert_(test_not_implemented()) 
    
    def test_quit(self):
    
        # __doc__ (as of 2008-06-25) for pygame.display.quit:
    
          # pygame.display.quit(): return None
          # uninitialize the display module
    
        self.assert_(test_not_implemented()) 
    
    def test_set_caption(self):
    
        # __doc__ (as of 2008-06-25) for pygame.display.set_caption:
    
          # pygame.display.set_caption(title, icontitle=None): return None
          # set the current window caption
    
        self.assert_(test_not_implemented()) 
    
    def test_set_gamma(self):
    
        # __doc__ (as of 2008-06-25) for pygame.display.set_gamma:
    
          # pygame.display.set_gamma(red, green=None, blue=None): return bool
          # change the hardware gamma ramps
    
        self.assert_(test_not_implemented()) 
    
    def test_set_gamma_ramp(self):
    
        # __doc__ (as of 2008-06-25) for pygame.display.set_gamma_ramp:
    
          # change the hardware gamma ramps with a custom lookup
          # pygame.display.set_gamma_ramp(red, green, blue): return bool
          # set_gamma_ramp(red, green, blue): return bool
    
        self.assert_(test_not_implemented()) 
    
    def test_set_icon(self):
    
        # __doc__ (as of 2008-06-25) for pygame.display.set_icon:
    
          # pygame.display.set_icon(Surface): return None
          # change the system image for the display window
    
        self.assert_(test_not_implemented()) 
    
    def test_set_mode(self):
    
        # __doc__ (as of 2008-06-25) for pygame.display.set_mode:
    
          # pygame.display.set_mode(resolution=(0,0), flags=0, depth=0): return Surface
          # initialize a window or screen for display
    
        self.assert_(test_not_implemented()) 
    
    def test_set_palette(self):
    
        # __doc__ (as of 2008-06-25) for pygame.display.set_palette:
    
          # pygame.display.set_palette(palette=None): return None
          # set the display color palette for indexed displays
    
        self.assert_(test_not_implemented()) 
    
    def test_toggle_fullscreen(self):
    
        # __doc__ (as of 2008-06-25) for pygame.display.toggle_fullscreen:
    
          # pygame.display.toggle_fullscreen(): return bool
          # switch between fullscreen and windowed displays
    
        self.assert_(test_not_implemented()) 

    def test_vid_info( self ):
        """ 
        """
        
        # old test, was disabled so placing reminder
        self.assert_(test_not_implemented())
        
        if 0:

            pygame.init()
            inf = pygame.display.Info()
            print "before a display mode has been set"
            print inf
            self.assertNotEqual(inf.current_h, -1)
            self.assertNotEqual(inf.current_w, -1)
            #probably have an older SDL than 1.2.10 if -1.


            screen = pygame.display.set_mode((100,100))
            inf = pygame.display.Info()
            print inf
            self.assertNotEqual(inf.current_h, -1)
            self.assertEqual(inf.current_h, 100)
            self.assertEqual(inf.current_w, 100)

            #pygame.quit()

if __name__ == '__main__':
    unittest.main()