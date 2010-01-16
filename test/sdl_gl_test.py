import os, sys
try:
    import pygame2.test.pgunittest as unittest
except:
    import pgunittest as unittest

import pygame2
import pygame2.sdl.video as video
import pygame2.sdl.gl as gl
import pygame2.sdl.constants as constants

class SDLGLTest (unittest.TestCase):

    def _get_gllib (self):
        gllib = ""
        dirs = []
        if sys.platform == "win32":
            dirs = [ "C:\\WINDOWS\\system32", "C:\\WINDOWS\\system" ]
            gllib = "opengl32.dll"
        elif sys.platform == "darwin":
            # TODO
            return None
        else:
            dirs = [ "/usr/lib", "/usr/local/lib" ]
            gllib = "libopengl.so"
        
        for d in dirs:
            path = os.path.join (d, gllib)
            if os.path.exists (path):
                return path
        return None

    def test_pygame2_sdl_gl_get_attribute(self):

        # __doc__ (as of 2009-12-14) for pygame2.sdl.gl.get_attribute:

        # get_attribute (attribute) -> int
        # 
        # Gets an OpenGL attribute value.
        # 
        # Gets the current value of the specified OpenGL attribute constant.
        gllib = self._get_gllib ()
        if not gllib:
            return
        
        # No video.
        self.assertRaises (pygame2.Error, gl.get_attribute, constants.GL_DEPTH_SIZE)
        
        video.init ()
        # No GL library
        self.assertRaises (pygame2.Error, gl.get_attribute, constants.GL_DEPTH_SIZE)

        # No screen
        self.assertEquals (gl.load_library (gllib), None)
        self.assertRaises (pygame2.Error, gl.get_attribute, constants.GL_DEPTH_SIZE)
        
        # No OpenGL screen
        screen = video.set_mode (10, 10)
        self.assertEquals (gl.get_attribute (constants.GL_DEPTH_SIZE), 0)
        
        screen = video.set_mode (10, 10, bpp=32, flags=constants.OPENGL)
        self.assertEquals (gl.get_attribute (constants.GL_DEPTH_SIZE), 24)
        video.quit ()

    def todo_test_pygame2_sdl_gl_get_proc_address(self):

        # __doc__ (as of 2009-12-14) for pygame2.sdl.gl.get_proc_address:

        # get_proc_address (procname) -> CObject
        # 
        # Gets the proc address of a function in the loaded OpenGL libraries.
        # 
        # The proc address is an encapsuled function pointer and as such only
        # useful for ctypes bindings or other C API modules.

        self.fail() 

    def test_pygame2_sdl_gl_load_library(self):

        # __doc__ (as of 2009-12-14) for pygame2.sdl.gl.load_library:

        # load_library (libraryname) -> None
        # 
        # Loads the desired OpenGL library.
        # 
        # Loads the desired OpenGL library specified by the passed full qualified
        # path. This must be called before any first call to
        # pygame2.sdl.video.set_mode to have any effect.
        self.assertRaises (pygame2.Error, gl.load_library, "invalid_opengl_lib")
        gllib = self._get_gllib ()
        if not gllib:
            return
        self.assertRaises (pygame2.Error, gl.load_library, gllib)

        video.init ()
        self.assertRaises (pygame2.Error, gl.load_library, "invalid_opengl_lib")
        self.assertEquals (gl.load_library (gllib), None)
        video.quit ()
        
    def test_pygame2_sdl_gl_set_attribute(self):

        # __doc__ (as of 2009-12-14) for pygame2.sdl.gl.set_attribute:

        # set_attribute (attribute, value) -> None
        # 
        # Sets an OpenGL attribute value.
        # 
        # Sets the value of the specified OpenGL attribute.
        gllib = self._get_gllib ()
        if not gllib:
            return
        
        # No video.
        self.assertRaises (pygame2.Error, gl.set_attribute, constants.GL_RED_SIZE, 1)
        
        video.init ()

        self.assertEquals (gl.load_library (gllib), None)
        
        # No OpenGL screen
        screen = video.set_mode (10, 10)
        self.assertEquals (gl.set_attribute (constants.GL_RED_SIZE, 1), None)
        self.assertEquals (gl.get_attribute (constants.GL_RED_SIZE), 0)

    def todo_test_pygame2_sdl_gl_swap_buffers(self):

        # __doc__ (as of 2009-12-14) for pygame2.sdl.gl.swap_buffers:

        # swap_buffers () -> None
        # 
        # Swap the OpenGL buffers, if double-buffering is supported.
        gllib = self._get_gllib ()
        if not gllib:
            return
        self.assertEquals (gl.load_library (gllib), None)

        self.fail() 

if __name__ == "__main__":
    unittest.main ()
