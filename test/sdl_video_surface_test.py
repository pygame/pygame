try:
    import pygame2.test.pgunittest as unittest
except:
    import pgunittest as unittest

import pygame2
import pygame2.sdl.video as video
import pygame2.sdl.constants as constants

class SDLVideoSurfaceTest (unittest.TestCase):

    def todo_test_pygame2_sdl_video_Surface_blit(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.blit:

        # blit (srcsurface[, destrect, srcrect, blendargs]) -> Rect
        # 
        # Draws the passed source surface onto this surface.
        # 
        # Draws the passed source surface onto this surface. The dstrect
        # and srcrect arguments are used for clipping the destionation
        # area (on this surface) and source area (on the source
        # surface). For the destination rectangle, the width and height
        # are ignored, only the position is taken into account. For the
        # source rectangle, all values, position as well as the size are
        # used for clipping.
        # 
        # The optional blending arguments for the drawing operation perform
        # certain specialised pixel manipulations. For those, pixels on the same
        # position are evaluated and the result manipulated according to the
        # argument.
        # 
        # +---------------------------+----------------------------------------+
        # | Blend type                | Effect                                 |
        # +===========================+========================================+
        # | BLEND_RGB_ADD             | The sum of both pixel values will be   |
        # |                           | set as resulting pixel.                |
        # +---------------------------+----------------------------------------+
        # | BLEND_RGB_SUB             | The difference of both pixel values    |
        # |                           | will be set as resulting pixel.        |
        # +---------------------------+----------------------------------------+
        # | BLEND_RGB_MIN             | The minimum of each R, G and B channel |
        # |                           | of the pixels will be set as result.   |
        # +---------------------------+----------------------------------------+
        # | BLEND_RGB_MAX             | The maximum of each R, G and B channel |
        # |                           | of the pixels will be set as result.   |
        # +---------------------------+----------------------------------------+
        # | BLEND_RGB_MULT            | The result of a multiplication of both |
        # |                           | pixel values will be used.             |
        # +---------------------------+----------------------------------------+
        # 
        # The BLEND_RGB_*** flags do not take the alpha channel into
        # account and thus are much faster for most blit operations
        # without alpha transparency. Whenever alpha transparency has to
        # be taken into account, the blending flags below should be
        # used.
        # 
        # +---------------------------+----------------------------------------+
        # | Blend type                | Effect                                 |
        # +===========================+========================================+
        # | BLEND_RGBA_ADD            | The sum of both pixel values will be   |
        # |                           | set as resulting pixel.                |
        # +---------------------------+----------------------------------------+
        # | BLEND_RGBA_SUB            | The difference of both pixel values    |
        # |                           | will be set as resulting pixel.        |
        # +---------------------------+----------------------------------------+
        # | BLEND_RGBA_MIN            | The minimum of each R, G, B and A      |
        # |                           | channel of the pixels will be set as   |
        # |                           | result.                                |
        # +---------------------------+----------------------------------------+
        # | BLEND_RGBA_MAX            | The maximum of each R, G, B and A      |
        # |                           | channel of the pixels will be set as   |
        # |                           | result.                                |
        # +---------------------------+----------------------------------------+
        # | BLEND_RGBA_MULT           | The result of a multiplication of both |
        # |                           | pixel values will be used.             |
        # +---------------------------+----------------------------------------+

        self.fail() 

    def todo_test_pygame2_sdl_video_Surface_clip_rect(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.clip_rect:

        # Gets or sets the current clipping rectangle for
        # operations on the Surface.

        self.fail() 

    def todo_test_pygame2_sdl_video_Surface_convert(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.convert:

        # convert ([pixelformat, flags]) -> Surface
        # 
        # Converts the Surface to the desired pixel format.
        # 
        # Converts the Surface to the desired pixel format. If no format
        # is given, the active display format is used, which will be the
        # fastest for blit operations to the screen. The flags are the
        # same as for surface creation.
        # 
        # This creates a new, converted surface and leaves the original one
        # untouched.

        self.fail() 

    def todo_test_pygame2_sdl_video_Surface_copy(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.copy:

        # copy () -> Surface
        # 
        # Creates an exact copy of the Surface and its image data.

        self.fail() 

    def todo_test_pygame2_sdl_video_Surface_fill(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.fill:

        # fill (color[, dstrect, blendargs]) -> None
        # 
        # Fills the Surface with a color.
        # 
        # Fills the Surface with the desired color. The color does not need to
        # match the Surface's format, it will be converted implicitly to the
        # nearest appropriate color for its format.
        # 
        # The optional destination rectangle limits the color fill to
        # the specified area. The blendargs are the same as for the blit
        # operation, but compare the color with the specific Surface
        # pixel value.

        self.fail() 

    def todo_test_pygame2_sdl_video_Surface_flags(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.flags:

        # The currently set flags for the Surface.

        self.fail() 

    def todo_test_pygame2_sdl_video_Surface_flip(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.flip:

        # flip () -> None
        # 
        # Swaps the screen buffers for the Surface.
        # 
        # Swaps screen buffers for the Surface, causing a full update
        # and redraw of its whole area.

        self.fail() 

    def todo_test_pygame2_sdl_video_Surface_format(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.format:

        # Gets the (read-only) pygame2.sdl.video.PixelFormat for this
        # Surface.

        self.fail() 

    def todo_test_pygame2_sdl_video_Surface_get_alpha(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.get_alpha:

        # get_alpha () -> int
        # 
        # Gets the current overall alpha value of the Surface.
        # 
        # Gets the current overall alpha value of the Surface. In case the
        # surface does not support alpha transparency (SRCALPHA flag not set),
        # None will be returned.

        self.fail() 

    def todo_test_pygame2_sdl_video_Surface_get_at(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.get_at:

        # get_at (x, y) -> Color
        # get_at (point) -> Color
        # 
        # Gets the Surface pixel value at the specified point.

        self.fail() 

    def todo_test_pygame2_sdl_video_Surface_get_colorkey(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.get_colorkey:

        # get_colorkey () -> Color
        # 
        # Gets the colorkey for the Surface.
        # 
        # Gets the colorkey for the Surface or None in case it has no colorkey
        # (SRCCOLORKEY flag not set).

        self.fail() 

    def todo_test_pygame2_sdl_video_Surface_get_palette(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.get_palette:

        # get_palette () -> (Color, Color, ...)

        self.fail() 

    def todo_test_pygame2_sdl_video_Surface_h(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.h:

        # Gets the height of the Surface.

        self.fail() 

    def todo_test_pygame2_sdl_video_Surface_height(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.height:

        # Gets the height of the Surface.

        self.fail() 

    def todo_test_pygame2_sdl_video_Surface_lock(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.lock:

        # lock () -> None
        # 
        # Locks the Surface for a direct access to its internal pixel data.

        self.fail() 

    def todo_test_pygame2_sdl_video_Surface_locked(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.locked:

        # Gets, whether the Surface is currently locked.

        self.fail() 

    def todo_test_pygame2_sdl_video_Surface_pitch(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.pitch:

        # Get the length of a surface scanline in bytes.

        self.fail() 

    def todo_test_pygame2_sdl_video_Surface_pixels(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.pixels:

        # Gets the pixel buffer of the Surface.

        self.fail() 

    def todo_test_pygame2_sdl_video_Surface_save(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.save:

        # save (file[, type]) -> None
        # 
        # Saves the Surface to a file.
        # 
        # Saves the Surface to a file. The file argument can be either a
        # file name or a file-like object to save the Surface to. The
        # optional type argument is required, if the file type cannot be
        # determined by the suffix.
        # 
        # Currently supported file types (suitable to pass as string for
        # the type argument) are:
        # 
        # * BMP
        # * TGA
        # * PNG
        # * JPEG (JPG)
        # 
        # If no type information is supplied and the file type cannot be
        # determined either, it will use TGA.
        video.init ()
        sf = video.Surface (16, 16)
        sf.fill (pygame2.Color ("red"))
        sf.save ("tmp.png")

    def todo_test_pygame2_sdl_video_Surface_set_alpha(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.set_alpha:

        # set_alpha (alpha[, flags]) -> None
        # 
        # Adjusts the alpha properties of the Surface.
        # 
        # TODO

        self.fail() 

    def todo_test_pygame2_sdl_video_Surface_set_at(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.set_at:

        # set_at (x, y, color) -> None
        # set_at (point, color) -> None
        # 
        # Sets the Surface pixel value at the specified point.

        self.fail() 

    def todo_test_pygame2_sdl_video_Surface_set_colorkey(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.set_colorkey:

        # set_colorkey (colorkey[, flags]) -> None
        # 
        # Adjusts the colorkey of the Surface.
        # 
        # TODO

        self.fail() 

    def todo_test_pygame2_sdl_video_Surface_set_colors(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.set_colors:

        # set_colors ((color1, color2, ...)[, first]) -> bool
        # 
        # Sets a portion of the colormap palette for the 8-bit Surface.
        # 
        # Sets a portion of the colormap palette for the 8-bit Surface,
        # starting at the desired first position. If the first position
        # plus length of the passed colormap exceeds the Surface palette
        # size, the palette will be unchanged and False returned.
        # 
        # If any other error occurs, False will be returned and the
        # Surface palette should be inspected for any changes.

        self.fail() 

    def todo_test_pygame2_sdl_video_Surface_set_palette(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.set_palette:

        # set_palette ((color1, color2, ...), flags[, first]) -> bool
        # 
        # Sets a portion of the palette for the given 8-bit surface.
        # 
        # Sets a portion of the color palette for the 8-bit Surface,
        # starting at the desired first position. If the first position
        # plus length of the passed colormap exceeds the Surface palette
        # size, the palette will be unchanged and False returned.
        # 
        # If any other error occurs, False will be returned and the
        # Surface palette should be inspected for any changes.

        self.fail() 

    def todo_test_pygame2_sdl_video_Surface_size(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.size:

        # Gets the size of the Surface.

        self.fail() 

    def todo_test_pygame2_sdl_video_Surface_unlock(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.unlock:

        # unlock () -> None
        # 
        # Unlocks the Surface, releasing the direct access to the pixel data.

        self.fail() 

    def todo_test_pygame2_sdl_video_Surface_update(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.update:

        # update ([rect]) -> None
        # update ([(rect1, rect2, ...)]) -> None
        # 
        # Updates the given area on the Surface.
        # 
        # Upates the given area (or areas, if a list of rects is passed)
        # on the Surface.

        self.fail() 

    def todo_test_pygame2_sdl_video_Surface_w(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.w:

        # Gets the width of the Surface.

        self.fail() 

    def todo_test_pygame2_sdl_video_Surface_width(self):

        # __doc__ (as of 2009-05-15) for pygame2.sdl.video.Surface.width:

        # Gets the width of the Surface.

        self.fail() 

if __name__ == "__main__":
    unittest.main ()
