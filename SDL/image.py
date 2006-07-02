#!/usr/bin/env python

'''Load images of various formats as SDL surfaces.

This module supports BMP, PNM (PPM/PGM/PBM), XPM, LBM, PCX, GIF, JPEG, PNG,
TGA and TIFF formats.

Typical usage::

    from SDL.image import *
    surface = IMG_Load('image.png')

:note: Early versions of this library (pre-1.2.5) do not have versioning
    information; do not count on `IMG_Linked_Version` being available.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

from ctypes import *

import SDL.dll
import SDL.error
import SDL.rwops
import SDL.version
import SDL.video

_dll = SDL.dll.SDL_DLL('SDL_image', 'IMG_Linked_Version')

IMG_Linked_Version = _dll.function('IMG_Linked_Version',
    '''Get the version of the dynamically linked SDL_image library.

    :since: SDL_image 1.2.5
    ''',
    args=[],
    arg_types=[],
    return_type=POINTER(SDL.version.SDL_version),
    dereference_return=True,
    require_return=True,
    since=(1,2,5))

IMG_LoadTyped_RW = _dll.function('IMG_LoadTyped_RW',
    '''Load an image from an SDL data source, specifying a type.

    If the image format supports a transparent pixel, SDL will set the
    colorkey for the surface.  You can enable RLE acceleration on the
    surface afterwards by calling::
        
        SDL_SetColorKey(image, SDL_RLEACCEL, image.format.colorkey)

    :Parameters:
        `src` : `SDL_RWops`
            Source RWops to load from.
        `freesrc` : int
            If non-zero, the source will be freed after loading.
        `type` : string
            One of "BMP", "GIF", "PNG", etc.

    :rtype: `SDL_Surface`
    ''',
    args=['src', 'freesrc', 'type'],
    arg_types=[POINTER(SDL.rwops.SDL_RWops), c_int, c_char_p],
    return_type=POINTER(SDL.video.SDL_Surface),
    dereference_return=True,
    require_return=True)

IMG_Load = _dll.function('IMG_Load',
    '''Load an image from a file.

    If the image format supports a transparent pixel, SDL will set the
    colorkey for the surface.  You can enable RLE acceleration on the
    surface afterwards by calling::
        
        SDL_SetColorKey(image, SDL_RLEACCEL, image.format.colorkey)

    :Parameters:
        `file` : string
            Filename to load.

    :rtype: `SDL_Surface`
    ''',
    args=['file'],
    arg_types=[c_char_p],
    return_type=POINTER(SDL.video.SDL_Surface),
    dereference_return=True,
    require_return=True)

IMG_Load_RW = _dll.function('IMG_Load_RW',
    '''Load an image from an SDL data source.

    If the image format supports a transparent pixel, SDL will set the
    colorkey for the surface.  You can enable RLE acceleration on the
    surface afterwards by calling::
        
        SDL_SetColorKey(image, SDL_RLEACCEL, image.format.colorkey)

    :Parameters:
        `src` : `SDL_RWops`
            Source RWops to load from.
        `freesrc` : int
            If non-zero, the source will be freed after loading.

    :rtype: `SDL_Surface`
    ''',
    args=['src', 'freesrc'],
    arg_types=[POINTER(SDL.rwops.SDL_RWops), c_int],
    return_type=POINTER(SDL.video.SDL_Surface),
    dereference_return=True,
    require_return=True)

# IMG_InvertAlpha is a no-op.

IMG_isBMP = _dll.function('IMG_isBMP',
    '''Detect if a seekable source is a BMP image.

    :Parameters:
        `src` : `SDL_RWops`
            Source RWops to inspect.
    
    :rtype: int
    ''',
    args=['src'],
    arg_types=[POINTER(SDL.rwops.SDL_RWops)],
    return_type=c_int)

IMG_isGIF = _dll.function('IMG_isGIF',
    '''Detect if a seekable source is a GIF image.

    :Parameters:
        `src` : `SDL_RWops`
            Source RWops to inspect.
    
    :rtype: int
    ''',
    args=['src'],
    arg_types=[POINTER(SDL.rwops.SDL_RWops)],
    return_type=c_int)


IMG_isJPG = _dll.function('IMG_isJPG',
    '''Detect if a seekable source is a JPG image.

    :Parameters:
        `src` : `SDL_RWops`
            Source RWops to inspect.
    
    :rtype: int
    ''',
    args=['src'],
    arg_types=[POINTER(SDL.rwops.SDL_RWops)],
    return_type=c_int)


IMG_isLBM = _dll.function('IMG_isLBM',
    '''Detect if a seekable source is a LBM image.

    :Parameters:
        `src` : `SDL_RWops`
            Source RWops to inspect.
    
    :rtype: int
    ''',
    args=['src'],
    arg_types=[POINTER(SDL.rwops.SDL_RWops)],
    return_type=c_int)


IMG_isPCX = _dll.function('IMG_isPCX',
    '''Detect if a seekable source is a PCX image.

    :Parameters:
        `src` : `SDL_RWops`
            Source RWops to inspect.
    
    :rtype: int
    ''',
    args=['src'],
    arg_types=[POINTER(SDL.rwops.SDL_RWops)],
    return_type=c_int)


IMG_isPNG = _dll.function('IMG_isPNG',
    '''Detect if a seekable source is a PNG image.

    :Parameters:
        `src` : `SDL_RWops`
            Source RWops to inspect.
    
    :rtype: int
    ''',
    args=['src'],
    arg_types=[POINTER(SDL.rwops.SDL_RWops)],
    return_type=c_int)


IMG_isPNM = _dll.function('IMG_isPNM',
    '''Detect if a seekable source is a PNM image.

    :Parameters:
        `src` : `SDL_RWops`
            Source RWops to inspect.
    
    :rtype: int
    ''',
    args=['src'],
    arg_types=[POINTER(SDL.rwops.SDL_RWops)],
    return_type=c_int)


IMG_isTIF = _dll.function('IMG_isTIF',
    '''Detect if a seekable source is a TIF image.

    :Parameters:
        `src` : `SDL_RWops`
            Source RWops to inspect.
    
    :rtype: int
    ''',
    args=['src'],
    arg_types=[POINTER(SDL.rwops.SDL_RWops)],
    return_type=c_int)


IMG_isXCF = _dll.function('IMG_isXCF',
    '''Detect if a seekable source is a XCF image.

    :Parameters:
        `src` : `SDL_RWops`
            Source RWops to inspect.
    
    :rtype: int
    ''',
    args=['src'],
    arg_types=[POINTER(SDL.rwops.SDL_RWops)],
    return_type=c_int)


IMG_isXPM = _dll.function('IMG_isXPM',
    '''Detect if a seekable source is a XPM image.

    :Parameters:
        `src` : `SDL_RWops`
            Source RWops to inspect.
    
    :rtype: int
    ''',
    args=['src'],
    arg_types=[POINTER(SDL.rwops.SDL_RWops)],
    return_type=c_int)


if hasattr(_dll._dll, 'IMG_isXV'):
    IMG_isXV = _dll.function('IMG_isXV',
        '''Detect if a seekable source is a XV image.

        :Parameters:
            `src` : `SDL_RWops`
                Source RWops to inspect.
        
        :rtype: int
        :since: SDL_image 1.2.5
        ''',
        args=['src'],
        arg_types=[POINTER(SDL.rwops.SDL_RWops)],
        return_type=c_int,
        since=(1,2,5))
else:
    # Broken build of SDL_image 1.2.5 on OS X does define xv.c symbols
    def IMG_isXV(src):
        raise SDL.error.SDL_NotImplementedError, 'Linked version of ' + \
            'SDL_image does not define IMG_isXV'

IMG_LoadBMP_RW = _dll.function('IMG_LoadBMP_RW',
    '''Load a BMP image from an SDL data source.

    :Parameters:
        `src` : `SDL_RWops`
            Source RWops to load from.

    :rtype: `SDL_Surface`
    ''',
    args=['src'],
    arg_types=[POINTER(SDL.rwops.SDL_RWops)],
    return_type=POINTER(SDL.video.SDL_Surface),
    dereference_return=True,
    require_return=True)

IMG_LoadGIF_RW = _dll.function('IMG_LoadGIF_RW',
    '''Load a GIF image from an SDL data source.

    :Parameters:
        `src` : `SDL_RWops`
            Source RWops to load from.

    :rtype: `SDL_Surface`
    ''',
    args=['src'],
    arg_types=[POINTER(SDL.rwops.SDL_RWops)],
    return_type=POINTER(SDL.video.SDL_Surface),
    dereference_return=True,
    require_return=True)

IMG_LoadJPG_RW = _dll.function('IMG_LoadJPG_RW',
    '''Load a JPG image from an SDL data source.

    :Parameters:
        `src` : `SDL_RWops`
            Source RWops to load from.

    :rtype: `SDL_Surface`
    ''',
    args=['src'],
    arg_types=[POINTER(SDL.rwops.SDL_RWops)],
    return_type=POINTER(SDL.video.SDL_Surface),
    dereference_return=True,
    require_return=True)

IMG_LoadLBM_RW = _dll.function('IMG_LoadLBM_RW',
    '''Load a LBM image from an SDL data source.

    :Parameters:
        `src` : `SDL_RWops`
            Source RWops to load from.

    :rtype: `SDL_Surface`
    ''',
    args=['src'],
    arg_types=[POINTER(SDL.rwops.SDL_RWops)],
    return_type=POINTER(SDL.video.SDL_Surface),
    dereference_return=True,
    require_return=True)

IMG_LoadPCX_RW = _dll.function('IMG_LoadPCX_RW',
    '''Load a PCX image from an SDL data source.

    :Parameters:
        `src` : `SDL_RWops`
            Source RWops to load from.

    :rtype: `SDL_Surface`
    ''',
    args=['src'],
    arg_types=[POINTER(SDL.rwops.SDL_RWops)],
    return_type=POINTER(SDL.video.SDL_Surface),
    dereference_return=True,
    require_return=True)

IMG_LoadPNG_RW = _dll.function('IMG_LoadPNG_RW',
    '''Load a PNG image from an SDL data source.

    :Parameters:
        `src` : `SDL_RWops`
            Source RWops to load from.

    :rtype: `SDL_Surface`
    ''',
    args=['src'],
    arg_types=[POINTER(SDL.rwops.SDL_RWops)],
    return_type=POINTER(SDL.video.SDL_Surface),
    dereference_return=True,
    require_return=True)

IMG_LoadPNM_RW = _dll.function('IMG_LoadPNM_RW',
    '''Load a PNM image from an SDL data source.

    :Parameters:
        `src` : `SDL_RWops`
            Source RWops to load from.

    :rtype: `SDL_Surface`
    ''',
    args=['src'],
    arg_types=[POINTER(SDL.rwops.SDL_RWops)],
    return_type=POINTER(SDL.video.SDL_Surface),
    dereference_return=True,
    require_return=True)

IMG_LoadTGA_RW = _dll.function('IMG_LoadTGA_RW',
    '''Load a TGA image from an SDL data source.

    :Parameters:
        `src` : `SDL_RWops`
            Source RWops to load from.

    :rtype: `SDL_Surface`
    ''',
    args=['src'],
    arg_types=[POINTER(SDL.rwops.SDL_RWops)],
    return_type=POINTER(SDL.video.SDL_Surface),
    dereference_return=True,
    require_return=True)

IMG_LoadTIF_RW = _dll.function('IMG_LoadTIF_RW',
    '''Load a TIF image from an SDL data source.

    :Parameters:
        `src` : `SDL_RWops`
            Source RWops to load from.

    :rtype: `SDL_Surface`
    ''',
    args=['src'],
    arg_types=[POINTER(SDL.rwops.SDL_RWops)],
    return_type=POINTER(SDL.video.SDL_Surface),
    dereference_return=True,
    require_return=True)

IMG_LoadXCF_RW = _dll.function('IMG_LoadXCF_RW',
    '''Load a XCF image from an SDL data source.

    :Parameters:
        `src` : `SDL_RWops`
            Source RWops to load from.

    :rtype: `SDL_Surface`
    ''',
    args=['src'],
    arg_types=[POINTER(SDL.rwops.SDL_RWops)],
    return_type=POINTER(SDL.video.SDL_Surface),
    dereference_return=True,
    require_return=True)

IMG_LoadXPM_RW = _dll.function('IMG_LoadXPM_RW',
    '''Load a XPM image from an SDL data source.

    :Parameters:
        `src` : `SDL_RWops`
            Source RWops to load from.

    :rtype: `SDL_Surface`
    ''',
    args=['src'],
    arg_types=[POINTER(SDL.rwops.SDL_RWops)],
    return_type=POINTER(SDL.video.SDL_Surface),
    dereference_return=True,
    require_return=True)

if hasattr(_dll._dll, 'IMG_LoadXV_RW'):
    IMG_LoadXV_RW = _dll.function('IMG_LoadXV_RW',
        '''Load a XV image from an SDL data source.

        :Parameters:
            `src` : `SDL_RWops`
                Source RWops to load from.

        :rtype: `SDL_Surface`
        :since: SDL_image 1.2.5
        ''',
        args=['src'],
        arg_types=[POINTER(SDL.rwops.SDL_RWops)],
        return_type=POINTER(SDL.video.SDL_Surface),
        dereference_return=True,
        require_return=True,
        since=(1,2,5))
else:
    # Broken build of SDL_image 1.2.5 on OS X does define xv.c symbols
    def IMG_LoadXV_RW(src):
        raise SDL.error.SDL_NotImplementedError, 'Linked version of ' + \
            'SDL_image does not define IMG_LoadXV_RW'

# IMG_ReadXPMFromArray cannot be implemented.
