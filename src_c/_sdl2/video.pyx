from cpython cimport PyObject
from . import error

WINDOW_FULLSCREEN = _SDL_WINDOW_FULLSCREEN
WINDOW_FULLSCREEN_DESKTOP = _SDL_WINDOW_FULLSCREEN_DESKTOP
WINDOW_OPENGL = _SDL_WINDOW_OPENGL
WINDOW_SHOWN = _SDL_WINDOW_SHOWN
WINDOW_HIDDEN = _SDL_WINDOW_HIDDEN
WINDOW_BORDERLESS = _SDL_WINDOW_BORDERLESS
WINDOW_RESIZABLE = _SDL_WINDOW_RESIZABLE
WINDOW_MINIMIZED = _SDL_WINDOW_MINIMIZED
WINDOW_MAXIMIZED = _SDL_WINDOW_MAXIMIZED
WINDOW_INPUT_GRABBED = _SDL_WINDOW_INPUT_GRABBED
WINDOW_INPUT_FOCUS = _SDL_WINDOW_INPUT_FOCUS
WINDOW_MOUSE_FOCUS = _SDL_WINDOW_MOUSE_FOCUS
WINDOW_FOREIGN = _SDL_WINDOW_FOREIGN
WINDOW_ALLOW_HIGHDPI = _SDL_WINDOW_ALLOW_HIGHDPI
WINDOW_MOUSE_CAPTURE = _SDL_WINDOW_MOUSE_CAPTURE
WINDOW_ALWAYS_ON_TOP = _SDL_WINDOW_ALWAYS_ON_TOP
WINDOW_SKIP_TASKBAR = _SDL_WINDOW_SKIP_TASKBAR
WINDOW_UTILITY = _SDL_WINDOW_UTILITY
WINDOW_TOOLTIP = _SDL_WINDOW_TOOLTIP
WINDOW_POPUP_MENU = _SDL_WINDOW_POPUP_MENU
WINDOW_VULKAN = _SDL_WINDOW_VULKAN

WINDOWPOS_UNDEFINED = _SDL_WINDOWPOS_UNDEFINED
WINDOWPOS_CENTERED = _SDL_WINDOWPOS_CENTERED

RENDERER_SOFTWARE = _SDL_RENDERER_SOFTWARE
RENDERER_ACCELERATED = _SDL_RENDERER_ACCELERATED
RENDERER_PRESENT_VSYNC = _SDL_RENDERER_PRESENTVSYNC
RENDERER_TARGETTEXTURE = _SDL_RENDERER_TARGETTEXTURE


cdef extern from "../pygame.h" nogil:
    int pgSurface_Check(object surf)
    SDL_Surface* pgSurface_AsSurface(object surf)
    void import_pygame_surface()
    SDL_Rect *pgRect_FromObject(object obj, SDL_Rect *temp)
    void import_pygame_rect()

import_pygame_surface()
import_pygame_rect()

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
    cdef int num = SDL_GetNumRenderDrivers()
    cdef SDL_RendererInfo info
    cdef int ind
    for ind from 0 <= ind < num:
        SDL_GetRenderDriverInfo(ind, &info)
        ret = RendererDriverInfo()
        ret.name = info.name
        ret.flags = info.flags
        ret.num_texture_formats = info.num_texture_formats
        ret.max_texture_width = info.max_texture_width
        ret.max_texture_height = info.max_texture_height
        yield ret

cdef class Window:
    DEFAULT_SIZE = 640, 480

    def __init__(self, title='pygame',
                 size=DEFAULT_SIZE, flags=0,
                 x=WINDOWPOS_UNDEFINED, y=WINDOWPOS_UNDEFINED):
        """ Create a window with the specified position, dimensions, and flags.

        :param title str: the title of the window, in UTF-8 encoding
        :param size tuple: the size of the window, in screen coordinates (width, height)
        :param flags int: 0, or one or more SDL_WindowFlags OR'd together
        :param x int: the x position of the window, WINDOWPOS_CENTERED, or WINDOWPOS_UNDEFINED
        :param y int: the y position of the window, WINDOWPOS_CENTERED, or WINDOWPOS_UNDEFINED

        WINDOW_FULLSCREEN
          fullscreen window

        WINDOW_FULLSCREEN_DESKTOP
          fullscreen window at the current desktop resolution

        WINDOW_OPENGL
          Window usable with OpenGL context. You will still need to create an OpenGL context.

        WINDOW_VULKAN
          window usable with a Vulkan instance

        WINDOW_HIDDEN
          window is not visible

        WINDOW_BORDERLESS
          no window decoration

        WINDOW_RESIZABLE
          window can be resized

        WINDOW_MINIMIZED
          window is minimized

        WINDOW_MAXIMIZED
          window is maximized

        WINDOW_INPUT_GRABBED
          window has grabbed input focus

        WINDOW_ALLOW_HIGHDPI
          window should be created in high-DPI mode if supported (>= SDL 2.0.1)
        """
        # https://wiki.libsdl.org/SDL_CreateWindow
        # https://wiki.libsdl.org/SDL_WindowFlags
        self._win = SDL_CreateWindow(title.encode('utf8'), x, y,
                                     size[0], size[1], flags)
        if not self._win:
            raise error()
        SDL_SetWindowData(self._win, "pg_window", <PyObject*>self)

    @property
    def title(self):
        return SDL_GetWindowTitle(self._win).decode('utf8')

    @title.setter
    def title(self, new_title):
        SDL_SetWindowTitle(self._win, new_title.encode('utf8'))

    def destroy(self):
        if self._win:
            SDL_SetWindowData(self._win, "pg_window", NULL)
            SDL_DestroyWindow(self._win)
            self._win = NULL

    def __del__(self):
        self.destroy()


cdef class Texture:
    def __init__(self, Renderer renderer, surface):
        """ Create a texture from an existing surface.

        :param renderer Renderer: Rendering context for the texture.
        :param surface Surface: The surface to create a texture from.
        """
        # https://wiki.libsdl.org/SDL_CreateTextureFromSurface
        if not pgSurface_Check(surface):
            raise error('2nd argument must be a surface')
        self.renderer = renderer
        cdef SDL_Renderer* _renderer = renderer._renderer
        cdef SDL_Surface *surf_ptr = pgSurface_AsSurface(surface)
        self._tex = SDL_CreateTextureFromSurface(_renderer,
                                                 surf_ptr)
        if not self._tex:
            raise error()
        self.width = surface.get_width()
        self.height = surface.get_height()

    def __del__(self):
        if self._tex:
            SDL_DestroyTexture(self._tex)

cdef class Renderer:
    def __init__(self, Window window, index=-1, flags=0):
        """ Create a 2D rendering context for a window.

        :param window Window: where rendering is displayed.
        :param index int: index of rendering driver to initialize,
                          or -1 to init the first supporting requested flags.
        :param flags int: 0, or one or more SDL_RendererFlags OR'd together.

        flags can be these OR'd together:

        0
          gives priority to available RENDERER_ACCELERATED renderers.

        RENDERER_SOFTWARE
          the renderer is a software fallback.

        RENDERER_ACCELERATED
          the renderer uses hardware acceleration.

        RENDERER_PRESENTVSYNC
          present is synchronized with the refresh rate.

        RENDERER_TARGETTEXTURE
          the renderer supports rendering to texture.
        """
        # https://wiki.libsdl.org/SDL_CreateRenderer
        # https://wiki.libsdl.org/SDL_RendererFlags
        self._renderer = SDL_CreateRenderer(window._win, index, flags)
        if not self._renderer:
            raise error()

        self._draw_color = (255, 255, 255, 255)

    def __del__(self):
        if self._renderer:
            SDL_DestroyRenderer(self._renderer)

    @property
    def draw_color(self):
        return self._draw_color

    @draw_color.setter
    def draw_color(self, new_value):
        self._draw_color = new_value[:]
        SDL_SetRenderDrawColor(self._renderer,
                               new_value[0],
                               new_value[1],
                               new_value[2],
                               new_value[3])

    def clear(self):
        SDL_RenderClear(self._renderer)

    def copy_pos(self, Texture texture, x, y):
        cdef SDL_Rect srcrect = SDL_Rect(0, 0, texture.width, texture.height)
        cdef SDL_Rect dstrect = SDL_Rect(x, y, texture.width, texture.height)
        SDL_RenderCopy(self._renderer, texture._tex, &srcrect, &dstrect)

    def copy(self, Texture texture, srcrect=None, dstrect=None):
        cdef SDL_Rect src, dst
        cdef SDL_Rect *csrcrect = pgRect_FromObject(srcrect, &src)
        cdef SDL_Rect *cdstrect = pgRect_FromObject(dstrect, &dst)
        SDL_RenderCopy(self._renderer, texture._tex, csrcrect, cdstrect)

    def present(self):
        SDL_RenderPresent(self._renderer)
