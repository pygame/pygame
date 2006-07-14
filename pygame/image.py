#!/usr/bin/env python

'''Pygame module for image transfer.

The image module contains functions for loading and saving pictures, as
well as transferring Surfaces to formats usable by other packages.

Note that there is no Image class; an image is loaded as a
Surface object. The Surface class allows manipulation (drawing lines,
setting pixels, capturing regions, etc.).

The image module is a required dependency of Pygame, but it only optionally
supports any extended file formats.  By default it can only load uncompressed
BMP images. When built with full image support, the pygame.image.load()
function can support the following formats.

* JPG
* PNG
* GIF (non animated)
* BMP
* PCX
* TGA (uncompressed)
* TIF
* LBM (and PBM)
* PBM (and PGM, PPM)
* XPM

Saving images only supports a limited set of formats. You can save
to the following formats.

* BMP
* TGA
* PNG
* JPEG

PNG, JPEG saving new in pygame 1.8.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id$'

import os.path
import re

from SDL import *

import pygame.surface

try:
    from SDL.image import *
    _have_SDL_image = True
except ImportError:
    _have_SDL_image = False

def load_extended(file, namehint=''):
    '''Load new image from a file, using SDL.image.

    :see: `load`

    :Parameters:
        `file` : str or file-like object
            Image file or filename to load.
        `namehint` : str
            Optional file extension.

    :rtype: `Surface`
    '''
    if not _have_SDL_image:
        raise NotImplementedError, 'load_extended requires SDL.image'

    if not hasattr(file, 'read'):
        surf = IMG_Load(file)
    else:
        if not namehint and hasattr(file, 'name'):
            namehint = file.name
        namehint = os.path.splitext(namehint)[1]
        rw = SDL_RWFromObject(file)
        # XXX Differ from pygame: don't freesrc when we didn't allocate it
        surf = IMG_LoadTyped_RW(rw, 0, namehint)
    return pygame.surface.Surface(surf=surf)

def load_basic(file, namehint=''):
    '''Load BMP image from a file.

    :see: `load`

    :Parameters:
        `file` : str or file-like object
            Image file or filename to load.
        `namehint` : str
            Ignored, for compatibility.

    :rtype: `Surface`
    '''
    if not hasattr(file, 'read'):
        surf = SDL_LoadBMP(file)
    else:
        rw = SDL_RWFromObject(file)
        # XXX Differ from pygame: don't freesrc when we didn't allocate it
        surf = SDL_LoadBMP_RW(rw, 0)
    return pygame.surface.Surface(surf=surf)

def load(file, namehint=''):
    '''Load a new image from a file.

    Pygame will automatically determine the image type (e.g., GIF or bitmap)
    and create a new Surface object from the data. In some cases it will need
    to know the file extension (e.g., GIF images should end in ".gif").  If
    you pass a raw file-like object, you may also want to pass the original
    filename as the namehint argument.

    The returned Surface will contain the same color format, colorkey and
    alpha transparency as the file it came from. You will often want to call
    Surface.convert() with no arguments, to create a copy that will draw more
    quickly on the screen.

    For alpha transparency, like in .png images use the convert_alpha() method
    after loading so that the image has per pixel transparency.

    Pygame may not always be built to support all image formats. At minimum it
    will support uncompressed BMP. If pygame.image.get_extended() returns
    'True', you should be able to load most images (including png, jpg and gif).

    You should use os.path.join() for compatibility, e.g.::  
    
        asurf = pygame.image.load(os.path.join('data', 'bla.png'))

    This function calls `load_extended` if SDL.image is available, otherwise
    `load_basic`.

    :Parameters:
        `file` : str or file-like object
            Image file or filename to load.
        `namehint` : str
            Optional file extension.

    :rtype: `Surface`    
    '''
    if _have_SDL_image:
        return load_extended(file, namehint)
    else:
        return load_basic(file, namehint)

def save(surface, file):
    '''Save an image to disk.

    This will save your Surface as either a BMP, TGA, PNG, or JPEG image. If
    the filename extension is unrecognized it will default to TGA. Both TGA,
    and BMP file formats create uncompressed files.  

    :note: Only BMP is currently implemented.

    :Parameters:
        `surface` : `Surface`
            Surface containing image data to save.
        `file` : str or file-like object
            File or filename to save to.

    '''
    if surface._surf.flags & SDL_OPENGL:
        surf = _get_opengl_surface(surface._surf)
    else:
        surface._prep()
        surf = surface._surf

    if hasattr(file, 'write'):
        # TODO TGA not BMP save
        rw = SDL_RWFromObject(file)
        # XXX Differ from pygame: don't freesrc when we didn't allocate it
        SDL_SaveBMP_RW(surf, rw, 0)  
    else:
        fileext = os.path.splitext(file)[1].lower()
        if fileext == '.bmp':
            SDL_SaveBMP(surf, file)
        elif fileext in ('.jpg', '.jpeg'):
            raise pygame.base.error, 'No support for jpg compiled in.' # TODO
        elif fileext == '.png':
            raise pygame.base.error, 'No support for png compiled in.' # TODO
        else:
            raise NotImplementedError, 'TODO: TGA support'
    
    if surface._surf.flags & SDL_OPENGL:
        SDL_FreeSurface(surf)
    else:
        surface._unprep()


def get_extended():
    '''Test if extended image formats can be loaded.

    If pygame is built with extended image formats this function will return
    True.  It is still not possible to determine which formats will be
    available, but generally you will be able to load them all.

    :rtype: bool
    '''
    return _have_SDL_image

def tostring(surface, format, flipped=False):
    '''Transfer image to string buffer.

    Creates a string that can be transferred with the 'fromstring' method in
    other Python imaging packages. Some Python image packages prefer their
    images in bottom-to-top format (PyOpenGL for example). If you pass True
    for the flipped argument, the string buffer will be vertically flipped.

    The format argument is a string of one of the following values. Note that
    only 8bit Surfaces can use the "P" format. The other formats will work for
    any Surface. Also note that other Python image packages support more
    formats than Pygame.

    * P, 8bit palettized Surfaces
    * RGB, 24bit image 
    * RGBX, 32bit image with alpha channel derived from color key
    * RGBA, 32bit image with an alpha channel
    * ARGB, 32bit image with alpha channel first
    
    :Parameters:
        `surface` : `Surface`
            Surface containing data to convert.
        `format` : str
            One of 'P', 'RGB', 'RGBX', 'RGBA' or 'ARGB'
        `flipped` : bool
            If True, data is ordered from bottom row to top.

    :rtype: str
    '''
    surf = surface._surf
    if surf.flags & SDL_OPENGL:
        surf = _get_opengl_surface(surf)

    result = None
    rows = []
    pitch = surf.pitch
    w = surf.w
    h = surf.h

    if flipped:
        h_range = range(surf.h - 1, -1, -1)
    else:
        h_range = range(surf.h)

    if format == 'P':
        # The only case for creating palette data.
        if surf.format.BytesPerPixel != 1:
            raise ValueError, \
                  'Can only create "P" format data with 8bit Surfaces'

        surface.lock()
        pixels = surf.pixels.to_string()
        surface.unlock()

        if pitch == w:
            result = pixels # easy exit
        else:
            flipped = False # Flipping taken care of by h_range
            for y in h_range:
                rows.append(pixels[y*pitch:y*pitch + w])
    elif surf.format.BytesPerPixel == len(format) and format != 'RGBX':
        # No conversion required?
        # This is an optimisation; could also use the default case.
        if format == 'RGBA':
            Rmask = SDL_SwapLE32(0x000000ff)
            Gmask = SDL_SwapLE32(0x0000ff00)
            Bmask = SDL_SwapLE32(0x00ff0000)
            Amask = SDL_SwapLE32(0xff000000)
        elif format == 'ARGB':
            Amask = SDL_SwapLE32(0x000000ff)
            Rmask = SDL_SwapLE32(0x0000ff00)
            Gmask = SDL_SwapLE32(0x00ff0000)
            Bmask = SDL_SwapLE32(0xff000000)
        elif format == 'RGB':
            if SDL_BYTEORDER == SDL_LIL_ENDIAN:
                Rmask = 0x000000ff
                Gmask = 0x0000ff00
                Bmask = 0x00ff0000
            else:
                Rmask = 0x00ff0000
                Gmask = 0x0000ff00
                Bmask = 0x000000ff 
            Amask = surf.format.Amask   # ignore
        if surf.format.Rmask == Rmask and \
           surf.format.Gmask == Gmask and \
           surf.format.Bmask == Bmask and \
           surf.format.Amask == Amask and \
           pitch == w * surf.format.BytesPerPixel:
            # Pixel data is already in required format, simply memcpy will
            # work fast.
            surface.lock()
            result = surf.pixels.to_string()
            surface.unlock()
    elif surf.format.BytesPerPixel == 4 and format == 'RGB':
        # Optimised conversion from RGBA or ARGB to RGB.
        # This is an optimisation; could also use the default case.
        if surf.format.Rmask == SDL_SwapLE32(0x000000ff):
            # Internal format is RGBA
            Gmask = SDL_SwapLE32(0x0000ff00)
            Bmask = SDL_SwapLE32(0x00ff0000)
            pattern = '(...).'
        elif surf.format.Rmask == SDL_SwapLE32(0x0000ff00):
            # Internal format is ARGB
            Gmask = SDL_SwapLE32(0x00ff0000)
            Bmask = SDL_SwapLE32(0xff000000)
            pattern = '.(...)'
        else:
            # Internal format is something else, give up.
            pattern = None
        
        if pattern and \
           surf.format.Gmask == Gmask and \
           surf.format.Bmask == Bmask and \
           pitch == w * surf.format.BytesPerPixel:
            surface.lock()
            result = surf.pixels.to_string()
            surface.unlock()
            
            # Squeeze out the alpha byte
            result = ''.join(re.findall(pattern, result, re.DOTALL))

    if not result and not rows:
        # Default case, works for any conversion, but is slow.
        surface.lock()
        if surf.format.BytesPerPixel == 1:
            palette = surf.format.palette.colors
            if surf.flags & SDL_SRCCOLORKEY and not Amask and format == 'RGBX':
                colorkey = surf.format.colorkey
                pixels = [(palette[c].r, palette[c].g, palette[c].b, 
                           (c != colorkey) * 0xff) \
                          for c in surf.pixels]
            else:
                pixels = [(palette[c].r, palette[c].g, palette[c].b, 255) \
                          for c in surf.pixels]
        else:
            Rmask = surf.format.Rmask
            Gmask = surf.format.Gmask
            Bmask = surf.format.Bmask
            Amask = surf.format.Amask
            Rshift = surf.format.Rshift
            Gshift = surf.format.Gshift
            Bshift = surf.format.Bshift
            Ashift = surf.format.Ashift
            Rloss = surf.format.Rloss
            Gloss = surf.format.Gloss
            Bloss = surf.format.Bloss
            Aloss = surf.format.Aloss
            if surf.flags & SDL_SRCCOLORKEY and not Amask and format == 'RGBX':
                colorkey = surf.format.colorkey
                pixels = [( ((c & Rmask) >> Rshift) << Rloss,
                            ((c & Gmask) >> Gshift) << Gloss,
                            ((c & Bmask) >> Bshift) << Bloss,
                            (c != colorkey) * 0xff ) \
                          for c in surf.pixels]
            else:
                pixels = [( ((c & Rmask) >> Rshift) << Rloss,
                            ((c & Gmask) >> Gshift) << Gloss,
                            ((c & Bmask) >> Bshift) << Bloss,
                            ((c & Amask) >> Ashift) << Aloss ) \
                          for c in surf.pixels]
        surface.unlock()
        pitch /= surf.format.BytesPerPixel
        flipped = False  # Flipping taken care of by h_range
        if format == 'RGB':
            for y in h_range:
                rows.append(''.join([ chr(c[0]) + chr(c[1]) + chr(c[2]) \
                                      for c in pixels[y*pitch:y*pitch + w] ]))
        elif format in ('RGBA', 'RGBX'):
            for y in h_range:
                rows.append(''.join([ chr(c[0]) + chr(c[1]) + chr(c[2]) + \
                                      chr(c[3]) \
                                      for c in pixels[y*pitch:y*pitch + w] ]))
        elif format == 'ARGB':
            for y in h_range:
                rows.append(''.join([ chr(c[3]) + chr(c[1]) + chr(c[2]) + \
                                      chr(c[0]) \
                                      for c in pixels[y*pitch:y*pitch + w] ]))

    if surface._surf.flags & SDL_OPENGL:
        SDL_FreeSurface(surf)

    # Is pixel data already one big string?
    if result:
        if flipped:
            # Split it into rows so it can be flipped vertically.
            rows = re.findall('.' * w * len(format), result, re.DOTALL)
        else:
            return result

    if flipped:
        rows.reverse()
    return ''.join(rows)

def fromstring(string, size, format, flipped=False):
    '''Create new Surface from a string buffer.

    This function takes arguments similar to pygame.image.tostring(). The size
    argument is a pair of numbers representing the width and height. Once the
    new Surface is created you can destroy the string buffer.

    The size and format image must compute the exact same size as the passed
    string buffer. Otherwise an exception will be raised. 

    :Parameters:
        `string` : str
            String containing image data.
        `size` : (int, int)
            Width, height of the image.
        `format` : str
            One of 'P', 'RGB', 'RGBA' or 'ARGB'
        `flipped` : bool
            If True, data is ordered from bottom row to top.

    :rtype: `Surface`
    '''
    width, height = size
    if format == 'P':
        Rmask = 0
        Gmask = 0
        Bmask = 0
        Amask = 0
        depth = 8
        pitch = width
    elif format == 'RGB':
        if SDL_BYTEORDER == SDL_LIL_ENDIAN:
            Rmask = 0x000000ff
            Gmask = 0x0000ff00
            Bmask = 0x00ff0000
        else:
            Rmask = 0x00ff0000
            Gmask = 0x0000ff00
            Bmask = 0x000000ff
        Amask = 0x00000000
        depth = 24
        pitch = width * 3
    elif format in ('RGBA', 'RGBX'):
        if SDL_BYTEORDER == SDL_LIL_ENDIAN:
            Rmask = 0x000000ff
            Gmask = 0x0000ff00
            Bmask = 0x00ff0000
            Amask = 0xff000000
        else:
            Rmask = 0xff000000
            Gmask = 0x00ff0000
            Bmask = 0x0000ff00
            Amask = 0x000000ff
        if format == 'RGBX':
            Amask = 0x00000000
        depth = 32
        pitch = width * 4
    elif format == 'ARGB':
        if SDL_BYTEORDER == SDL_LIL_ENDIAN:
            Rmask = 0x0000ff00
            Gmask = 0x00ff0000
            Bmask = 0xff000000
            Amask = 0x000000ff
        else:
            Rmask = 0x00ff0000
            Gmask = 0x0000ff00
            Bmask = 0x000000ff
            Amask = 0xff000000
        depth = 32
        pitch = width * 4
    if len(string) != pitch * height:
        raise ValueError, \
              'String length does not equal format and resolution size'
    if flipped:
        string = ''.join([string[y*pitch:y*pitch+pitch] \
                          for y in range(height - 1, -1, -1)])
    surf = SDL_CreateRGBSurfaceFrom(string, width, height, depth, pitch,
                                    Rmask, Gmask, Bmask, Amask)

    return pygame.surface.Surface(surf=surf)

def frombuffer(string, size, format):
    '''Create a new Surface that shares data inside a string buffer.

    Create a new Surface that shares pixel data directly from the string buffer.
    This method takes the same arguments as pygame.image.fromstring(), but is
    unable to vertically flip the source data.

    :note: In pygame-ctypes, this function is identical to `fromstring`.

    :Parameters:
        `string` : str
            String containing image data.
        `size` : (int, int)
            Width, height of the image.
        `format` : str
            One of 'P', 'RGB', 'RGBA', 'RGBX' or 'ARGB'
    
    :rtype: `Surface`
    '''
    return fromstring(string, size, format)

def _get_opengl_surface(surf):
    import OpenGL.GL
    data = OpenGL.GL.glReadPixels(0, 0, surf.w, surf.h, 
                                  OpenGL.GL.GL_RGB, OpenGL.GL.GL_UNSIGNED_BYTE)
    if SDL_BYTEORDER == SDL_LIL_ENDIAN:
        Rmask = 0x000000ff
        Gmask = 0x0000ff00
        Bmask = 0x00ff0000
    else:
        Rmask = 0x00ff0000
        Gmask = 0x0000ff00
        Bmask = 0x000000ff
    # Flip vertically
    pitch = surf.w * 3
    data = ''.join([data[y*pitch:y*pitch+pitch] \
                    for y in range(surf.h - 1, -1, -1)])
    newsurf = SDL_CreateRGBSurfaceFrom(data, surf.w, surf.h, 24, pitch,
                                       Rmask, Gmask, Bmask, 0)
    return newsurf

