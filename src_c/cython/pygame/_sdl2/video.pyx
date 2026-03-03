from cpython cimport PyObject
from cpython.ref cimport Py_DECREF
cimport cython
from pygame._sdl2.sdl2 import error
from pygame._sdl2.sdl2 import error as errorfnc
from libc.stdlib cimport free, malloc


WINDOWPOS_UNDEFINED = _SDL_WINDOWPOS_UNDEFINED
WINDOWPOS_CENTERED = _SDL_WINDOWPOS_CENTERED

SCALEQUALITY_NEAREST=SDL_ScaleMode.SDL_ScaleModeNearest
SCALEQUALITY_LINEAR=SDL_ScaleMode.SDL_ScaleModeLinear
SCALEQUALITY_BEST=SDL_ScaleMode.SDL_ScaleModeBest

import_pygame_base()
import_pygame_color()
import_pygame_surface()
import_pygame_rect()
import_pygame_window()

class RendererDriverInfo:
    def __repr__(self):
        return "<%s(name: %s, flags: 0x%02x, num_texture_formats: %d, max_texture_width: %d, max_texture_height: %d)>" % (
            self.__class__.__name__,
            self.name,
            self.flags,
            self.num_texture_formats,
            self.max_texture_width,
            self.max_texture_height,
        )

def get_drivers():
    """Yield info about the rendering drivers available for Renderer objects
    """
    cdef int num = SDL_GetNumRenderDrivers()
    cdef SDL_RendererInfo info
    cdef int ind
    for ind from 0 <= ind < num:
        SDL_GetRenderDriverInfo(ind, &info)
        ret = RendererDriverInfo()
        ret.name = info.name.decode("utf-8")
        ret.flags = info.flags
        ret.num_texture_formats = info.num_texture_formats
        ret.max_texture_width = info.max_texture_width
        ret.max_texture_height = info.max_texture_height
        yield ret


def get_grabbed_window():
    """Get the window with input grab enabled

    Gets the :class:`Window` with input grab enabled, if input is grabbed,
    otherwise ``None`` is returned.
    """
    cdef SDL_Window *win = SDL_GetGrabbedWindow()
    cdef void *ptr
    if win:
        ptr = SDL_GetWindowData(win, "pg_window")
        if not ptr:
            return None
        return <object>ptr
    return None

globals()["Window"] = Window
_Window = Window

cdef Uint32 format_from_depth(int depth):
    cdef Uint32 Rmask, Gmask, Bmask, Amask
    if depth == 16:
        Rmask = 0xF << 8
        Gmask = 0xF << 4
        Bmask = 0xF
        Amask = 0xF << 12
    elif depth in (0, 32):
        Rmask = 0xFF << 16
        Gmask = 0xFF << 8
        Bmask = 0xFF
        Amask = 0xFF << 24
    else:
        raise ValueError("no standard masks exist for given bitdepth with alpha")
    return SDL_MasksToPixelFormatEnum(depth,
                                      Rmask, Gmask, Bmask, Amask)


# disable auto_pickle since it causes stubcheck error
@cython.auto_pickle(False)
cdef class Texture:

    def __init__(self,
                 Renderer renderer,
                 size, int depth=0,
                 static=False, streaming=False,
                 target=False, scale_quality=None):
        """pygame object that represents a texture

        Creates an empty texture.

        :param Renderer renderer: The rendering context for the texture.
        :param tuple size: The width and height for the texture.
        :param int depth: The pixel format (``0`` to use the default).
        :param bool static: Initialize the texture as static
                            (changes rarely, not lockable).
        :param bool streaming: Initialize the texture as streaming
                               (changes frequently, lockable).
        :param bool target: Initialize the texture as target
                            (can be used as a rendering target).
        :param int scale_quality: Set the texture scale quality. Can be ``0``
                                  (nearest pixel sampling), ``1``
                                  (linear filtering, supported by OpenGL and Direct3D)
                                  and ``2`` (anisotropic filtering, supported by Direct3D).

        One of ``static``, ``streaming``, or ``target`` can be set
        to ``True``. If all are ``False``, then ``static`` is used.


        :class:`Texture` objects provide a platform-agnostic API for working with GPU textures.
        They are stored in GPU video memory (VRAM), and are therefore very fast to
        rotate and resize when drawn onto a :class:`Renderer`
        (an object that manages a rendering context inside a :class:`Window`) on most GPUs.

        Since textures are stored in GPU video memory, they aren't as easy to modify
        as the image data of :class:`pygame.Surface` objects, which reside in RAM.

        Textures can be modified in 2 ways:

           * By drawing other textures onto them, achieved by marking them as "target" textures and setting them as the rendering target of their Renderer object (if properly configured and supported).

           * By updating them with a Surface.

            .. note::
                A :class:`pygame.Surface`-to-:class:`Texture` update is generally
                considered a slow operation, as it requires image data to be uploaded from
                RAM to VRAM, which can have a notable overhead cost.
        """
        # https://wiki.libsdl.org/SDL_CreateTexture
        # TODO: masks
        cdef Uint32 format
        try:
            format = format_from_depth(depth)
        except ValueError as e:
            raise e

        cdef int width, height
        if len(size) != 2:
            raise ValueError('size must have two elements')
        width, height = size[0], size[1]
        if width <= 0 or height <= 0:
            raise ValueError('size must contain two positive values')

        cdef int access
        if static:
            if streaming or target:
                raise ValueError('only one of static, streaming, or target can be true')
            access = _SDL_TEXTUREACCESS_STATIC
        elif streaming:
            if static or target:
                raise ValueError('only one of static, streaming, or target can be true')
            access = _SDL_TEXTUREACCESS_STREAMING
        elif target:
            if streaming or static:
                raise ValueError('only one of static, streaming, or target can be true')
            access = _SDL_TEXTUREACCESS_TARGET
        else:
            # Create static texture by default.
            access = _SDL_TEXTUREACCESS_STATIC

        self.renderer = renderer
        cdef SDL_Renderer* _renderer = renderer._renderer
        self._tex = SDL_CreateTexture(_renderer,
                                      format,
                                      access,
                                      width, height)
        if not self._tex:
            raise error()

        if not scale_quality is None:
            if SDL_VERSION_ATLEAST(2,0,12):
                SDL_SetTextureScaleMode(self._tex,scale_quality)
            else:
                SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY,{
                    0: b'nearest',
                    1: b'linear',
                    2: b'best'
                }[scale_quality])

        self.width, self.height = width, height

    @staticmethod
    def from_surface(Renderer renderer, surface):
        """Create a texture from an existing surface

        :param Renderer renderer: Rendering context for the texture.
        :param pygame.Surface surface: The surface to create a texture from.
        """
        # https://wiki.libsdl.org/SDL_CreateTextureFromSurface
        if not pgSurface_Check(surface):
            raise TypeError('2nd argument must be a surface')
        cdef Texture self = Texture.__new__(Texture)
        self.renderer = renderer
        cdef SDL_Renderer* _renderer = renderer._renderer
        cdef SDL_Surface *surf_ptr = pgSurface_AsSurface(surface)
        self._tex = SDL_CreateTextureFromSurface(_renderer,
                                                 surf_ptr)
        if not self._tex:
            raise error()
        self.width = surface.get_width()
        self.height = surface.get_height()
        return self

    def __dealloc__(self):
        if self._tex:
            SDL_DestroyTexture(self._tex)

    @property
    def alpha(self):
        """Get or set the additional alpha value multiplied into draw operations
        """
        # https://wiki.libsdl.org/SDL_GetTextureAlphaMod
        cdef Uint8 alpha
        cdef int res = SDL_GetTextureAlphaMod(self._tex, &alpha)
        if res < 0:
            raise error()

        return alpha

    @alpha.setter
    def alpha(self, Uint8 new_value):
        # https://wiki.libsdl.org/SDL_SetTextureAlphaMod
        cdef int res = SDL_SetTextureAlphaMod(self._tex, new_value)
        if res < 0:
            raise error()

    @property
    def blend_mode(self):
        """Get or set the blend mode for texture drawing operations

        Gets or sets the blend mode for the texture's drawing operations.
        Valid blend modes are any of the ``BLENDMODE_*`` constants or a custom one.
        """
        # https://wiki.libsdl.org/SDL_GetTextureBlendMode
        cdef SDL_BlendMode blendMode
        cdef int res = SDL_GetTextureBlendMode(self._tex, &blendMode)
        if res < 0:
            raise error()

        return blendMode

    @blend_mode.setter
    def blend_mode(self, blendMode):
        # https://wiki.libsdl.org/SDL_SetTextureBlendMode
        cdef int res = SDL_SetTextureBlendMode(self._tex, blendMode)
        if res < 0:
            raise error()

    @property
    def color(self):
        """Get or set the additional color value multiplied into texture drawing operations
        """
        cdef Uint8[4] rgba

        # https://wiki.libsdl.org/SDL_GetTextureColorMod
        cdef int res = SDL_GetTextureColorMod(self._tex,
                                              &(rgba[0]),
                                              &(rgba[1]),
                                              &(rgba[2]))
        rgba[3] = 255

        if res < 0:
            raise error()

        return pgColor_NewLength(rgba, 4)

    @color.setter
    def color(self, new_value):
        cdef Uint8[4] rgba
        pg_RGBAFromObjEx(new_value, rgba, PG_COLOR_HANDLE_ALL)

        # https://wiki.libsdl.org/SDL_SetTextureColorMod
        cdef int res = SDL_SetTextureColorMod(self._tex,
                                              rgba[0],
                                              rgba[1],
                                              rgba[2])

        if res < 0:
            raise error()

    def get_rect(self, **kwargs):
        """Get the rectangular area of the texture

        Like :meth:`pygame.Surface.get_rect`, this method returns a **new**
        :class:`pygame.Rect` covering the entire texture. It will always
        have a ``topleft`` of ``(0, 0)`` with a ``width`` and ``height`` the same size as the
        texture.
        """
        rect = pgRect_New4(0, 0, self.width, self.height)
        for key in kwargs:
            setattr(rect, key, kwargs[key])

        return rect

    cpdef void draw(self, srcrect=None, dstrect=None, float angle=0, origin=None,
                    bint flip_x=False, bint flip_y=False):
        """Copy a portion of the texture to the rendering target

        :param srcrect: The source rectangle on the texture, or ``None`` for the
                        entire texture.
        :param dstrect: The destination rectangle on the rendering target, or
                        ``None`` for the entire rendering target. The texture
                        will be stretched to fill ``dstrect``.
        :param float angle: The angle (in degrees) to rotate dstrect around
                            (clockwise).
        :param origin: The point around which dstrect will be rotated.
                       If ``None``, it will equal the center:
                       ``(dstrect.w/2, dstrect.h/2)``.
        :param bool flip_x: Flip the drawn texture portion horizontally (x - axis).
        :param bool flip_y: Flip the drawn texture portion vertically (y - axis).
        """
        cdef SDL_Rect src
        cdef SDL_Rect *csrcrect = NULL
        cdef SDL_FRect dst
        cdef SDL_FRect *cdstrect = NULL
        cdef SDL_FPoint corigin
        cdef SDL_FPoint *originptr

        if srcrect is not None:
            csrcrect = pgRect_FromObject(srcrect, &src)
            if not csrcrect:
                raise TypeError("the argument is not a rectangle or None")

        if dstrect is not None:
            cdstrect = pgFRect_FromObject(dstrect, &dst)
            if cdstrect == NULL:
                if len(dstrect) == 2:
                    dst.x = dstrect[0]
                    dst.y = dstrect[1]
                    dst.w = <float> self.width
                    dst.h = <float> self.height
                    cdstrect = &dst
                else:
                    raise TypeError('dstrect must be a position, rect, or None')

        if origin:
            originptr = &corigin
            corigin.x = origin[0]
            corigin.y = origin[1]
        else:
            originptr = NULL

        cdef int flip = SDL_FLIP_NONE
        if flip_x:
            flip |= SDL_FLIP_HORIZONTAL
        if flip_y:
            flip |= SDL_FLIP_VERTICAL

        cdef int res = SDL_RenderCopyExF(self.renderer._renderer, self._tex, csrcrect, cdstrect,
                                        angle, originptr, <SDL_RendererFlip>flip)
        if res < 0:
            raise error()

    def draw_triangle(self, p1_xy, p2_xy, p3_xy,
                      p1_uv=(0.0, 0.0), p2_uv=(1.0, 1.0), p3_uv=(0.0, 1.0),
                      p1_mod=(255, 255, 255, 255), p2_mod=(255, 255, 255, 255), p3_mod=(255, 255, 255, 255)):
        """Copy a triangle portion of the texture to the rendering target using the given coordinates

        :param p1_xy: The first vertex coordinate on the rendering target.
        :param p2_xy: The second vertex coordinate on the rendering target.
        :param p3_xy: The third vertex coordinate on the rendering target.
        :param p1_uv: The first UV coordinate of the triangle portion.
        :param p2_uv: The second UV coordinate of the triangle portion.
        :param p3_uv: The third UV coordinate of the triangle portion.
        :param p1_mod: The first vertex color modulation.
        :param p2_mod: The second vertex color modulation.
        :param p3_mod: The third vertex color modulation.
        """
        if not SDL_VERSION_ATLEAST(2, 0, 18):
            raise error("draw_triangle requires SDL 2.0.18 or newer")

        cdef Uint8 _r_mod, _g_mod, _b_mod, _a_mod
        SDL_GetTextureColorMod(self._tex, &_r_mod, &_g_mod, &_b_mod)
        SDL_GetTextureAlphaMod(self._tex, &_a_mod)

        cdef float r_mod = <float>_r_mod / <float>255.0
        cdef float g_mod = <float>_g_mod / <float>255.0
        cdef float b_mod = <float>_b_mod / <float>255.0
        cdef float a_mod = <float>_a_mod / <float>255.0

        cdef SDL_Vertex vertices[3]
        for i, vert in enumerate(((p1_xy, p1_mod, p1_uv),
                                  (p2_xy, p2_mod, p2_uv),
                                  (p3_xy, p3_mod, p3_uv))):
            xy, mod, uv = vert
            vertices[i].position.x = xy[0]
            vertices[i].position.y = xy[1]
            vertices[i].color.r = r_mod * mod[0]
            vertices[i].color.g = g_mod * mod[1]
            vertices[i].color.b = b_mod * mod[2]
            vertices[i].color.a = a_mod * mod[3] if len(mod) > 3 else _a_mod
            vertices[i].tex_coord.x = uv[0]
            vertices[i].tex_coord.y = uv[1]

        cdef int res = SDL_RenderGeometry(self.renderer._renderer, self._tex, vertices, 3, NULL, 0)
        if res < 0:
            raise error()

    def draw_quad(self, p1_xy, p2_xy, p3_xy, p4_xy,
                  p1_uv=(0.0, 0.0), p2_uv=(1.0, 0.0),
                  p3_uv=(1.0, 1.0), p4_uv=(0.0, 1.0),
                  p1_mod=(255, 255, 255, 255), p2_mod=(255, 255, 255, 255),
                  p3_mod=(255, 255, 255, 255), p4_mod=(255, 255, 255, 255)):
        """Copy a quad portion of the texture to the rendering target using the given coordinates

        :param p1_xy: The first vertex coordinate on the rendering target.
        :param p2_xy: The second vertex coordinate on the rendering target.
        :param p3_xy: The third vertex coordinate on the rendering target.
        :param p4_xy: The fourth vertex coordinate on the rendering target.
        :param p1_uv: The first UV coordinate of the quad portion.
        :param p2_uv: The second UV coordinate of the quad portion.
        :param p3_uv: The third UV coordinate of the quad portion.
        :param p4_uv: The fourth UV coordinate of the quad portion.
        :param p1_mod: The first vertex color modulation.
        :param p2_mod: The second vertex color modulation.
        :param p3_mod: The third vertex color modulation.
        :param p4_mod: The fourth vertex color modulation.
        """
        if not SDL_VERSION_ATLEAST(2, 0, 18):
            raise error("draw_quad requires SDL 2.0.18 or newer")

        cdef Uint8 _r_mod, _g_mod, _b_mod, _a_mod
        SDL_GetTextureColorMod(self._tex, &_r_mod, &_g_mod, &_b_mod)
        SDL_GetTextureAlphaMod(self._tex, &_a_mod)

        cdef float r_mod = <float>_r_mod / <float>255.0
        cdef float g_mod = <float>_g_mod / <float>255.0
        cdef float b_mod = <float>_b_mod / <float>255.0
        cdef float a_mod = <float>_a_mod / <float>255.0

        cdef SDL_Vertex vertices[6]
        for i, vert in enumerate(((p1_xy, p1_mod, p1_uv),
                                  (p2_xy, p2_mod, p2_uv),
                                  (p3_xy, p3_mod, p3_uv),
                                  (p3_xy, p3_mod, p3_uv),
                                  (p4_xy, p4_mod, p4_uv),
                                  (p1_xy, p1_mod, p1_uv))):
            xy, mod, uv = vert
            vertices[i].position.x = xy[0]
            vertices[i].position.y = xy[1]
            vertices[i].color.r = r_mod * mod[0]
            vertices[i].color.g = g_mod * mod[1]
            vertices[i].color.b = b_mod * mod[2]
            vertices[i].color.a = a_mod * mod[3] if len(mod) > 3 else _a_mod
            vertices[i].tex_coord.x = uv[0]
            vertices[i].tex_coord.y = uv[1]

        cdef int res = SDL_RenderGeometry(self.renderer._renderer, self._tex, vertices, 6, NULL, 0)
        if res < 0:
            raise error()

    def update(self, surface, area=None):
        # https://wiki.libsdl.org/SDL_UpdateTexture
        # Should it accept a raw pixel data array too?
        """Update the texture with Surface (slow operation, use sparingly)

        :param Surface surface: The source surface.
        :param area: The rectangular area of the texture to update.

        .. note::
            This is a fairly slow function, intended for use with static textures
            that do not change often.

            If the texture is intended to be updated often, it is preferred to create
            the texture as streaming and use the locking functions.

            While this function will work with streaming textures, for optimization
            reasons you may not get the pixels back if you lock the texture afterward.
        """

        if not pgSurface_Check(surface):
            raise TypeError("update source should be a Surface.")

        cdef SDL_Rect rect
        rect.x = 0
        rect.y = 0
        cdef SDL_Rect *rectptr = pgRect_FromObject(area, &rect)
        cdef SDL_Surface *surf = pgSurface_AsSurface(surface)

        if rectptr == NULL and area is not None:
            raise TypeError('area must be a rectangle or None')

        cdef int dst_width, dst_height
        if rectptr == NULL:
            dst_width = self.width
            dst_height = self.height
        else:
            dst_width = rect.w
            dst_height = rect.h

        if dst_height > surf.h or dst_width > surf.w:
            # if the surface is smaller than the destination rect,
            # clip the rect to prevent segfault
            rectptr = &rect # make sure rectptr is not NULL
            rect.h = surf.h
            rect.w = surf.w

        # For converting the surface, if needed
        cdef SDL_Surface *converted_surf = NULL
        cdef SDL_PixelFormat *pixel_format = NULL
        cdef SDL_BlendMode blend

        cdef Uint32 format_
        if (SDL_QueryTexture(self._tex, &format_, NULL, NULL, NULL) != 0):
            raise error()

        cdef int res
        if format_ != surf.format.format:
            if (SDL_GetSurfaceBlendMode(surf, &blend) != 0):
                raise error()

            pixel_format = SDL_AllocFormat(format_)
            if (pixel_format == NULL):
                raise error()

            converted_surf = SDL_ConvertSurface(surf, pixel_format, 0)
            if (SDL_SetSurfaceBlendMode(converted_surf, blend) != 0):
                SDL_FreeSurface(converted_surf)
                SDL_FreeFormat(pixel_format)
                raise error()

            res = SDL_UpdateTexture(self._tex, rectptr, converted_surf.pixels, converted_surf.pitch)
            SDL_FreeSurface(converted_surf)
            SDL_FreeFormat(pixel_format)
        else:
            res = SDL_UpdateTexture(self._tex, rectptr, surf.pixels, surf.pitch)

        if res < 0:
            raise error()

# disable auto_pickle since it causes stubcheck error
@cython.auto_pickle(False)
cdef class Image:

    def __cinit__(self):
        self.angle = 0
        self._origin.x = 0
        self._origin.y = 0
        self._originptr = NULL
        self.flip_x = False
        self.flip_y = False

        cdef Uint8[4] defaultColor = [255, 255, 255, 255]
        self._color = pgColor_NewLength(defaultColor, 4)
        self.alpha = 255

    def __init__(self, texture_or_image, srcrect=None):
        """pygame object that represents a portion of a texture

        Creates an Image.

        :param Texture | Image texture_or_image: The Texture or an existing Image
                                                 object to create the Image from.
        :param srcrect: The rectangular portion of the Texture or Image object
                        passed to ``texture_or_image``.

        An :class:`Image` object represents a portion of a :class:`Texture`. Specifically,
        they can be used to store and manipulate arguments for :meth:`Texture.draw`
        in a more user-friendly way.
        """
        cdef SDL_Rect temp
        cdef SDL_Rect *rectptr

        if isinstance(texture_or_image, Image):
            self.texture = texture_or_image.texture
            self.srcrect = pgRect_New(&(<Rect>texture_or_image.srcrect).r)
        else:
            self.texture = texture_or_image
            self.srcrect = texture_or_image.get_rect()
        self.blend_mode = texture_or_image.blend_mode

        if srcrect is not None:
            rectptr = pgRect_FromObject(srcrect, &temp)
            if rectptr == NULL:
                raise TypeError('srcrect must be None or a rectangle')
            temp.x = rectptr.x
            temp.y = rectptr.y
            temp.w = rectptr.w
            temp.h = rectptr.h

            if temp.x < 0 or temp.y < 0 or \
                temp.w < 0 or temp.h < 0 or \
                temp.x + temp.w > self.srcrect.w or \
                temp.y + temp.h > self.srcrect.h:
                raise ValueError('rect values are out of range')
            temp.x += self.srcrect.x
            temp.y += self.srcrect.y
            self.srcrect = pgRect_New(&temp)

    @property
    def color(self):
        """Gets or sets the Image color modifier
        """
        return self._color

    @color.setter
    def color(self, new_color):
        cdef Uint8[4] rgba
        pg_RGBAFromObjEx(new_color, rgba, PG_COLOR_HANDLE_ALL)

        self._color[:3] = rgba[:3]

    @property
    def origin(self):
        """Get or set the Image's origin of rotation

        Gets or sets the Image's origin of rotation, a 2D coordinate relative to the topleft coordinate of the Image's rectangular area.

        An origin of ``None`` means no origin was set and the Image will be rotated around
        its center.
        """
        if self._originptr == NULL:
            return None
        else:
            return (self._origin.x, self._origin.y)

    @origin.setter
    def origin(self, new_origin):
        if new_origin:
            self._origin.x = <int>new_origin[0]
            self._origin.y = <int>new_origin[1]
            self._originptr = &self._origin
        else:
            self._originptr = NULL

    def get_rect(self):
        """Get the rectangular area of the Image

        .. note::
            The returned :class:`Rect` object might contain position information
            relative to the bounds of the :class:`Texture` or Image object it was
            created from.
        """
        return pgRect_New(&self.srcrect.r)

    cpdef void draw(self, srcrect=None, dstrect=None):
        """Copy a portion of the Image to the rendering target

        :param srcrect: Source rectangle specifying a sub-Image, or None for the
                        entire Image.
        :param dstrect: Destination rectangle or position on the render target,
                        or ``None`` for entire target. The Image is stretched to
                        fill dstrect.
        """

        self.texture.color = self._color
        self.texture.alpha = self.alpha
        self.texture.blend_mode = self.blend_mode
        self.texture.draw(
            (srcrect if not srcrect is None else self.srcrect),
            dstrect,
            self.angle,
            self.origin,
            self.flip_x,
            self.flip_y)

# disable auto_pickle since it causes stubcheck error
@cython.auto_pickle(False)
cdef class Renderer:

    @classmethod
    def from_window(cls, Window window):
        cdef Renderer self = cls.__new__(cls)
        self._win = window
        if self._win._is_borrowed:
            self._is_borrowed=1
        else:
            raise error()
        if not self._win:
            raise error()

        self._renderer =  SDL_GetRenderer(self._win._win)
        if not self._renderer:
            raise error()

        self._target = None
        return self

    def __init__(self,Window window, int index=-1,
                 int accelerated=-1, bint vsync=False,
                 bint target_texture=False):
        """pygame object wrapping a 2D rendering context for a window

        Creates a 2D rendering context for a window.

        :param Window window: The window onto which the rendering context should be
                              placed.
        :param int index: The index of rendering driver to initialize, or ``-1`` to init
                          the first supporting the requested options.
        :param int accelerated: If 1, the renderer uses hardware acceleration.
                                if 0, the renderer is a software fallback.
                                ``-1`` gives precedence to renderers using hardware
                                acceleration.
        :param bool vsync: If ``True`` :meth:`Renderer.present` is synchronized with
                           the refresh rate.
        :param bool target_texture: Whether the renderer should support setting
                                   :class:`Texture` objects as target textures, to
                                   enable drawing onto them.


        :class:`Renderer` objects provide a cross-platform API for rendering 2D
        graphics onto a :class:`Window`, by using either Metal (macOS), OpenGL
        (macOS, Windows, Linux) or Direct3D (Windows) rendering drivers, depending
        on what is set or is available on a system during their creation.

        They can be used to draw both :class:`Texture` objects and simple points,
        lines and rectangles (which are colored based on :attr:`Renderer.draw_color`).

        If configured correctly and supported by an underlying rendering driver, Renderer
        objects can have a :class:`Texture` object temporarily set as a target texture
        (the Texture object must have been created with target texture usage support),
        which allows those textures to be drawn onto.

        To present drawn content onto the window, :meth:`Renderer.present` should be
        called. :meth:`Renderer.clear` should be called to clear any drawn content
        with the set Renderer draw color.

        When things are drawn onto Renderer objects, an internal batching system is
        used by default to batch those "draw calls" together, to have all of them be
        processed in one go when :meth:`Renderer.present` is called. This is unlike
        :class:`pygame.Surface` objects, on which modifications via blitting occur
        immediately, but lends well to the behavior of GPUs, as draw calls can be
        expensive on lower-end models.
        """

        # https://wiki.libsdl.org/SDL_CreateRenderer
        # https://wiki.libsdl.org/SDL_RendererFlags
        flags = 0
        if accelerated >= 0:
            flags |= _SDL_RENDERER_ACCELERATED if accelerated else _SDL_RENDERER_SOFTWARE
        if vsync:
            flags |= _SDL_RENDERER_PRESENTVSYNC
        if target_texture:
            flags |= _SDL_RENDERER_TARGETTEXTURE

        self._renderer = SDL_CreateRenderer(window._win, index, flags)
        if not self._renderer:
            raise error()

        cdef Uint8[4] defaultColor = [255, 255, 255, 255]
        self._target = None
        self._win = window
        self._is_borrowed=0

    def __dealloc__(self):
        if self._is_borrowed:
            return
        if self._renderer:
            SDL_DestroyRenderer(self._renderer)

    @property
    def draw_blend_mode(self):
        """Get or set the blend mode used for primitive drawing operations
        """
        # https://wiki.libsdl.org/SDL_GetRenderDrawBlendMode
        cdef SDL_BlendMode blendMode
        cdef int res = SDL_GetRenderDrawBlendMode(self._renderer, &blendMode)
        if res < 0:
            raise error()

        return blendMode

    @draw_blend_mode.setter
    def draw_blend_mode(self, blendMode):
        # https://wiki.libsdl.org/SDL_SetRenderDrawBlendMode
        cdef int res = SDL_SetRenderDrawBlendMode(self._renderer, blendMode)
        if res < 0:
            raise error()

    @property
    def draw_color(self):
        """Get or set the color used for primitive drawing operations
        """
        cdef Uint8[4] rgba

        cdef int res = SDL_GetRenderDrawColor(self._renderer,
                                              &(rgba[0]),
                                              &(rgba[1]),
                                              &(rgba[2]),
                                              &(rgba[3]))

        if res < 0:
            raise error()

        return pgColor_NewLength(rgba, 4)

    @draw_color.setter
    def draw_color(self, new_value):
        cdef Uint8[4] rgba
        pg_RGBAFromObjEx(new_value, rgba, PG_COLOR_HANDLE_ALL)

        # https://wiki.libsdl.org/SDL_SetRenderDrawColor
        cdef int res = SDL_SetRenderDrawColor(self._renderer,
                                              rgba[0],
                                              rgba[1],
                                              rgba[2],
                                              rgba[3])

        if res < 0:
            raise error()

    def clear(self):
        """Clear the current rendering target with the drawing color
        """
        # https://wiki.libsdl.org/SDL_RenderClear
        cdef int res = SDL_RenderClear(self._renderer)
        if res < 0:
            raise error()

    def present(self):
        """Update the screen with any rendering performed since the previous call

        Presents the composed backbuffer to the screen.
        Updates the screen with any rendering performed since the previous call.
        """
        # https://wiki.libsdl.org/SDL_RenderPresent
        SDL_RenderPresent(self._renderer)

    cpdef get_viewport(self):
        """Get the drawing area on the rendering target
        """
        # https://wiki.libsdl.org/SDL_RenderGetViewport
        cdef SDL_Rect rect
        SDL_RenderGetViewport(self._renderer, &rect)
        return pgRect_New(&rect)

    def set_viewport(self, area):
        """Set the drawing area on the rendering target

        :param area: A :class:`pygame.Rect` or tuple representing the
                     drawing area on the target, or ``None`` to use the
                     entire area of the current rendering target.
        """
        # https://wiki.libsdl.org/SDL_RenderSetViewport
        if area is None:
            if SDL_RenderSetViewport(self._renderer, NULL) < 0:
                raise error()
            return

        cdef SDL_Rect tmprect
        cdef SDL_Rect *rectptr = pgRect_FromObject(area, &tmprect)
        if rectptr == NULL:
            raise TypeError('expected a rectangle')

        if SDL_RenderSetViewport(self._renderer, rectptr) < 0:
            raise error()

    @property
    def logical_size(self):
        """Get or set the logical Renderer size (a device independent resolution for rendering)
        """
        cdef int w
        cdef int h
        SDL_RenderGetLogicalSize(self._renderer, &w, &h)
        return (w, h)

    @logical_size.setter
    def logical_size(self, size):
        cdef int w = size[0]
        cdef int h = size[1]
        if (SDL_RenderSetLogicalSize(self._renderer, w, h) != 0):
            raise error()

    @property
    def scale(self):
        """Get the drawing scale for the current rendering target
        """
        cdef float x
        cdef float y
        SDL_RenderGetScale(self._renderer, &x, &y)
        return (x, y)

    @scale.setter
    def scale(self, scale):
        cdef float x = scale[0]
        cdef float y = scale[1]
        if (SDL_RenderSetScale(self._renderer, x, y) != 0):
            raise error()

    # TODO ifdef
    # def is_integer_scale(self):
    #     return SDL_RenderGetIntegerScale(self._renderer)

    @property
    def target(self):
        """`Get or set the current rendering target

        Gets or sets the current rendering target.
        A value of ``None`` means that no custom rendering target was set and the
        Renderer's window will be used as the target.
        """
        # https://wiki.libsdl.org/SDL_GetRenderTarget
        return self._target

    @target.setter
    def target(self, newtarget):
        # https://wiki.libsdl.org/SDL_SetRenderTarget
        if newtarget is None:
            self._target = None
            if SDL_SetRenderTarget(self._renderer, NULL) < 0:
                raise error()
        elif isinstance(newtarget, Texture):
            self._target = newtarget
            if SDL_SetRenderTarget(self._renderer,
                                   self._target._tex) < 0:
                raise error()
        else:
            raise TypeError('target must be a Texture or None')

    cpdef object blit(self, object source, Rect dest=None, Rect area=None, int special_flags=0):
        """Draw textures using a Surface-like API

        For compatibility purposes. Draws :class:`Texture` objects onto the
        Renderer using a method signature similar to :meth:`pygame.Surface.blit`.

        :param source: A :class:`Texture` or :class:`Image` to draw.
        :param dest: The drawing destination on the rendering target.
        :param area: The portion of the source texture or image to draw from.
        :param special_flags: have no effect at this moment.

        .. note:: Textures created by different Renderers cannot shared with each other!
        """
        if isinstance(source, Texture):
            (<Texture>source).draw(area, dest)
        elif isinstance(source, Image):
            (<Image>source).draw(area, dest)
        elif not hasattr(source, 'draw'):
            raise TypeError('source must be drawable')
        else:
            source.draw(area, dest)

        if not dest:
            return self.get_viewport()
        return dest

    def draw_line(self, p1, p2):
        """Draw a line

        :param p1: The line start point.
        :param p2: The line end point.
        """
        cdef int res
        res = SDL_RenderDrawLineF(self._renderer,
                                 p1[0], p1[1],
                                 p2[0], p2[1])
        if res < 0:
            raise error()

    def coordinates_to_window(self, point):
        """Translates renderer coordinates to window coordinates

        :param point: The coordinates in render space.
        """
        cdef int wx
        cdef int wy

        # Note: Must be changed to SDL_RenderCoordinatesToWindow for SDL3
        # https://wiki.libsdl.org/SDL3/SDL_RenderCoordinatesToWindow
        SDL_RenderLogicalToWindow(self._renderer, point[0], point[1], &wx, &wy);

        # Return float for future compatibility with SDL3's RenderCoordinatesToWindow
        return (float(wx), float(wy))

    def coordinates_from_window(self, point):
        """Translates window coordinates to renderer coordinates

        :param point: The coordinates in window space.
        """
        cdef float lx
        cdef float ly

        # Note: Must be changed to SDL_RenderCoordinatesFromWindow for SDL3
        # https://wiki.libsdl.org/SDL3/SDL_RenderCoordinatesFromWindow
        SDL_RenderWindowToLogical(self._renderer, point[0], point[1], &lx, &ly);

        return (lx, ly)

    def draw_point(self, point):
        """Draw a point

        :param point: The point's coordinate.
        """
        # https://wiki.libsdl.org/SDL_RenderDrawPointF
        cdef int res
        res = SDL_RenderDrawPointF(self._renderer,
                                point[0], point[1])
        if res < 0:
            raise error()

    def draw_rect(self, rect):
        """Draw a rectangle outline

        :param rect: The :class:`Rect`-like rectangle to draw.
        """
        # https://wiki.libsdl.org/SDL_RenderDrawRectF
        cdef SDL_FRect _frect
        cdef SDL_FRect *frectptr
        cdef int res

        frectptr = pgFRect_FromObject(rect, &_frect)
        if frectptr == NULL:
            raise TypeError('expected a rectangle')

        res = SDL_RenderDrawRectF(self._renderer, frectptr)

        if res < 0:
            raise error()

    def fill_rect(self, rect):
        """Draw a filled rectangle

        :param rect: The :class:`Rect`-like rectangle to draw.
        """
        # https://wiki.libsdl.org/SDL_RenderFillRectF
        cdef SDL_FRect _frect
        cdef SDL_FRect *frectptr
        cdef int res


        frectptr = pgFRect_FromObject(rect, &_frect)
        if frectptr == NULL:
            raise TypeError('expected a rectangle')

        res = SDL_RenderFillRectF(self._renderer, frectptr)

        if res < 0:
            raise error()

    def draw_triangle(self, p1, p2, p3):
        # https://wiki.libsdl.org/SDL_RenderDrawLinesF
        cdef SDL_FPoint fpoints[4]

        for i, pos in enumerate((p1, p2, p3, p1)):
            fpoints[i].x = pos[0]
            fpoints[i].y = pos[1]

        res = SDL_RenderDrawLinesF(self._renderer, fpoints, 4)

        if res < 0:
            raise error()

    def fill_triangle(self, p1, p2, p3):
        # https://wiki.libsdl.org/SDL_RenderGeometry
        if not SDL_VERSION_ATLEAST(2, 0, 18):
            raise error("fill_triangle requires SDL 2.0.18 or newer")

        cdef Uint8[4] rgba

        cdef int res = SDL_GetRenderDrawColor(self._renderer,
                                              &(rgba[0]),
                                              &(rgba[1]),
                                              &(rgba[2]),
                                              &(rgba[3]))

        if res < 0:
            raise error()

        cdef SDL_Vertex vertices[3]
        for i, pos in enumerate((p1, p2, p3)):
            vertices[i].position.x = pos[0]
            vertices[i].position.y = pos[1]
            vertices[i].color.r = rgba[0]
            vertices[i].color.g = rgba[1]
            vertices[i].color.b = rgba[2]
            vertices[i].color.a = rgba[3]

        res = SDL_RenderGeometry(self._renderer, NULL, vertices, 3, NULL, 0)
        if res < 0:
            raise error()

    def draw_quad(self, p1, p2, p3, p4):
        # https://wiki.libsdl.org/SDL_RenderDrawLinesF
        cdef SDL_FPoint fpoints[5]
        for i, pos in enumerate((p1, p2, p3, p4, p1)):
            fpoints[i].x = pos[0]
            fpoints[i].y = pos[1]

        res = SDL_RenderDrawLinesF(self._renderer, fpoints, 5)

        if res < 0:
            raise error()

    def fill_quad(self, p1, p2, p3, p4):
        # https://wiki.libsdl.org/SDL_RenderGeometry
        if not SDL_VERSION_ATLEAST(2, 0, 18):
            raise error("fill_quad requires SDL 2.0.18 or newer")

        cdef Uint8[4] rgba

        cdef int res = SDL_GetRenderDrawColor(self._renderer,
                                              &(rgba[0]),
                                              &(rgba[1]),
                                              &(rgba[2]),
                                              &(rgba[3]))

        if res < 0:
            raise error()

        cdef SDL_Vertex vertices[6]
        for i, pos in enumerate((p1, p2, p3, p3, p4, p1)):
            vertices[i].position.x = pos[0]
            vertices[i].position.y = pos[1]
            vertices[i].color.r = rgba[0]
            vertices[i].color.g = rgba[1]
            vertices[i].color.b = rgba[2]
            vertices[i].color.a = rgba[3]

        res = SDL_RenderGeometry(self._renderer, NULL, vertices, 6, NULL, 0)
        if res < 0:
            raise error()

    def to_surface(self, surface=None, area=None):
        # https://wiki.libsdl.org/SDL_RenderReadPixels
        """Read pixels from current rendering target and create a Surface (slow operation, use sparingly)

        Read pixel data from the current rendering target and return a
        :class:`pygame.Surface` containing it.

        :param Surface surface: A :class:`pygame.Surface` object to read the pixel
                                data into. It must be large enough to fit the area, otherwise
                                ``ValueError`` is raised.
                                If set to ``None``, a new surface will be created.
        :param area: The area of the screen to read pixels from. The area is
                     clipped to fit inside the viewport.
                     If ``None``, the entire viewport is used.

        .. note::
            This is a very slow operation, due to the overhead of the VRAM to RAM
            data transfer and the cost of creating a potentially large
            :class:`pygame.Surface`. It should not be used frequently.
        """
        cdef Uint32 format
        cdef SDL_Rect rarea
        cdef SDL_Rect tempviewport
        cdef SDL_Rect *areaparam
        cdef SDL_Surface *surf
        cdef SDL_Rect *rectptr

        # obtain area to use
        if area is not None:

            rectptr = pgRect_FromObject(area, &rarea)
            if rectptr == NULL:
                raise TypeError('area must be None or a rect')

            # clip area
            SDL_RenderGetViewport(self._renderer, &tempviewport)
            SDL_IntersectRect(rectptr, &tempviewport, rectptr)

            areaparam = rectptr
            rarea.x = rectptr.x
            rarea.y = rectptr.y
            rarea.w = rectptr.w
            rarea.h = rectptr.h
        else:
            SDL_RenderGetViewport(self._renderer, &rarea)
            areaparam = NULL

        # prepare surface
        if surface is None:
            # create a new surface
            format = SDL_GetWindowPixelFormat(self._win._win)
            if format == SDL_PIXELFORMAT_UNKNOWN:
                raise error()

            surf = SDL_CreateRGBSurfaceWithFormat(
                0,
                rarea.w, rarea.h,
                SDL_BITSPERPIXEL(format),
                format)
            if surf == NULL:
                raise MemoryError("not enough memory for the surface")

            surface = <object>pgSurface_New2(surf, 1)
            # casting to <object> makes cython assume reference counting of the object and it increments
            # it by one, however, pgSurface_New2 already returns a new reference to an object, so
            # we need to decrement that reference cython has added on top
            # see https://github.com/cython/cython/issues/2589#issuecomment-417604249 for additional context
            Py_DECREF(surface)
        elif pgSurface_Check(surface):
            surf = pgSurface_AsSurface(surface)
            if surf.w < rarea.w or surf.h < rarea.h:
                raise ValueError("the surface is too small")
            format = surf.format.format
        else:
            raise TypeError("'surface' must be a surface or None")

        if SDL_RenderReadPixels(self._renderer,
                                areaparam,
                                format, surf.pixels, surf.pitch) < 0:
            raise error()
        return surface

    @staticmethod
    def compose_custom_blend_mode(color_mode, alpha_mode):
        """Compose a custom blend mode

        Compose a custom blend mode.
        See https://wiki.libsdl.org/SDL2/SDL_ComposeCustomBlendMode for more information.

        :param color_mode: A tuple ``(srcColorFactor, dstColorFactor, colorOperation)``
        :param alpha_mode: A tuple ``(srcAlphaFactor, dstAlphaFactor, alphaOperation)``

        :return: A blend mode to be used with :meth:`Renderer.set_draw_blend_mode` and :meth:`Texture.set_blend_mode`.
        """
        # https://wiki.libsdl.org/SDL_ComposeCustomBlendMode
        cdef int res = SDL_ComposeCustomBlendMode(color_mode[0],
                                                  color_mode[1],
                                                  color_mode[2],
                                                  alpha_mode[0],
                                                  alpha_mode[1],
                                                  alpha_mode[2])
        if res < 0:
            raise error()
        return res
