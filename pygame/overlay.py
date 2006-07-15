#!/usr/bin/env python

'''Pygame support for hardware overlays.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id$'

from SDL import *

import pygame.base
import pygame.rect

class Overlay(object):
    '''pygame object for video overlay graphics

    The Overlay objects provide support for accessing hardware video overlays.
    Video overlays do not use standard RGB pixel formats, and can use multiple
    resolutions of data to create a single image.

    The Overlay objects represent lower level access to the display hardware.
    To use the object you must understand the technical details of video
    overlays.

    The Overlay format determines the type of pixel data used. Not all hardware
    will support all types of overlay formats. Here is a list of available
    format types:

    * YV12_OVERLAY
    * IYUV_OVERLAY
    * YUV2_OVERLAY
    * UYVY_OVERLAY
    * YVYU_OVERLAY

    The width and height arguments control the size for the overlay image data.
    The overlay image can be displayed at any size, not just the resolution of
    the overlay.

    The overlay objects are always visible, and always show above the regular
    display contents.
    '''

    __slots__ = ['_overlay', '_rect']

    def __init__(self, format, size):
        '''Create an Overlay object.

        :Parameters:
            `format` : int
                Overlay format constant (e.g. YV12_OVERLAY)
            `size` : int, int
                Width, height of overlay data.

        '''
        if not SDL_WasInit(SDL_INIT_VIDEO):
            raise pygame.base.error, \
                  'cannot create overlay without pygame.display intialized'
        
        screen = SDL_GetVideoSurface()
        if not screen:
            raise pygame.base.error, 'Display mode not set'

        w, h = size
        self._overlay = SDL_CreateYUVOverlay(w, h, format, screen)
        self._rect = SDL_Rect(0, 0, w, h)

    def display(self, planes=None):
        '''Set the overlay pixel data.

        Display the yuv data in SDL's overlay planes.  The data must be in the
        correct format used to create the Overlay.

        If no argument is passed in, the Overlay will simply be redrawn with
        the current data. This can be useful when the Overlay is not really
        hardware accelerated.

        The strings are not validated, and improperly sized strings could
        crash the program.

        :Parameters:
            `planes` : sequence of str
                Each element is a data plane.  Expects three planes for
                YV12 or IYUV, one plane for other formats.

        '''
        # XXX Differ from Pygame: accept any overlay format, not just YV12.
        # Planes must be in native order.  Incompatibility: pygame expects
        # planes in order YUV, whereas YV12 native order is YVU.
        if planes:
            SDL_LockYUVOverlay(self._overlay)
            for i in range(len(planes)):
                memmove(self._overlay.pixels[i].ptr, planes[i], len(planes[i]))
            SDL_UnlockYUVOverlay(self._overlay)

        SDL_DisplayYUVOverlay(self._overlay, self._rect)

    def set_location(self, *rect):
        '''Control where the overlay is displayed.

        Set the location for the overlay. The overlay will always be shown
        relative to the main display Surface. This does not actually redraw
        the overlay, it will be updated on the next call to Overlay.display().

        :Parameters:
            `rect` : `Rect`
                Overlay position and size.
        '''
        rect = pygame.rect._rect_from_object(rect)._r
        self._rect.x = rect.x
        self._rect.y = rect.y
        self._rect.w = rect.w
        self._rect.h = rect.h

    def get_hardware(self):
        '''Determine if the overlay is hardware accelerated.

        Returns a True value when the Overlay is hardware accelerated. If the
        platform does not support acceleration, software rendering is used.

        :rtype: bool
        '''
        return self._overlay.hw_overlay
