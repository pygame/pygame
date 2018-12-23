from cpython cimport PyObject
from . import error


WINDOWPOS_UNDEFINED = _SDL_WINDOWPOS_UNDEFINED
WINDOWPOS_CENTERED = _SDL_WINDOWPOS_CENTERED


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

def get_grabbed_window():
    """return the Window with input grab enabled,
       or None if input isn't grabbed."""
    cdef SDL_Window *win = SDL_GetGrabbedWindow()
    cdef void *ptr
    if win:
        ptr = SDL_GetWindowData(win, "pg_window")
        if not ptr:
            return None
        return <object>ptr
    return None

cdef class Window:
    DEFAULT_SIZE = 640, 480

    _kwarg_to_flag = {
        'opengl': _SDL_WINDOW_OPENGL,
        'vulkan': _SDL_WINDOW_VULKAN,
        'hidden': _SDL_WINDOW_HIDDEN,
        'borderless': _SDL_WINDOW_BORDERLESS,
        'resizable': _SDL_WINDOW_RESIZABLE,
        'minimized': _SDL_WINDOW_MINIMIZED,
        'maximized': _SDL_WINDOW_MAXIMIZED,
        'input_grabbed': _SDL_WINDOW_INPUT_GRABBED,
        'input_focus': _SDL_WINDOW_INPUT_FOCUS,
        'mouse_focus': _SDL_WINDOW_MOUSE_FOCUS,
        'allow_highdpi': _SDL_WINDOW_ALLOW_HIGHDPI,
        'foreign': _SDL_WINDOW_FOREIGN,
        'mouse_capture': _SDL_WINDOW_MOUSE_CAPTURE,
        'always_on_top': _SDL_WINDOW_ALWAYS_ON_TOP,
        'skip_taskbar': _SDL_WINDOW_SKIP_TASKBAR,
        'utility': _SDL_WINDOW_UTILITY,
        'tooltip': _SDL_WINDOW_TOOLTIP,
        'popup_menu': _SDL_WINDOW_POPUP_MENU,
    }

    def __init__(self, title='pygame',
                 size=DEFAULT_SIZE,
                 position=WINDOWPOS_UNDEFINED,
                 int fullscreen=0, **kwargs):
        """ Create a window with the specified position, dimensions, and flags.

        :param title str: the title of the window, in UTF-8 encoding
        :param size tuple: the size of the window, in screen coordinates (width, height)
        :param position: a tuple specifying the window position, WINDOWPOS_CENTERED, or WINDOWPOS_UNDEFINED.
        :param fullscreen int: 0: windowed mode
                               1: fullscreen window ("real" fullscreen with a videomode change)
                               2: fullscreen window at the current desktop resolution
                                  ("fake" fullscreen that takes the size of the desktop)
        :param opengl bool: Usable with OpenGL context. You will still need to create an OpenGL context.
        :param vulkan bool: usable with a Vulkan instance
        :param hidden bool: window is not visible
        :param borderless bool: no window decoration
        :param resizable bool: window can be resized
        :param minimized bool: window is minimized
        :param maximized bool: window is maximized
        :param input_grabbed bool: window has grabbed input focus
        :param input_focus bool: window has input focus
        :param mouse_focus bool: window has mouse focus
        :param foreign bool: window not created by SDL
        :param allow_highdpi bool: window should be created in high-DPI mode if supported (>= SDL 2.0.1)
        :param mouse_capture bool: window has mouse captured (unrelated to INPUT_GRABBED, >= SDL 2.0.4)
        :param always_on_top bool: window should always be above others (X11 only, >= SDL 2.0.5)
        :param skip_taskbar bool: window should not be added to the taskbar (X11 only, >= SDL 2.0.5)
        :param utility bool: window should be treated as a utility window (X11 only, >= SDL 2.0.5)
        :param tooltip bool: window should be treated as a tooltip (X11 only, >= SDL 2.0.5)
        :param popup_menu bool: window should be treated as a popup menu (X11 only, >= SDL 2.0.5)
        """
        # https://wiki.libsdl.org/SDL_CreateWindow
        # https://wiki.libsdl.org/SDL_WindowFlags
        if position == WINDOWPOS_UNDEFINED:
            x, y = WINDOWPOS_UNDEFINED, WINDOWPOS_UNDEFINED
        elif position == WINDOWPOS_CENTERED:
            x, y = WINDOWPOS_CENTERED, WINDOWPOS_CENTERED
        else:
            x, y = position

        flags = 0
        if fullscreen > 0:
            if fullscreen == 1:
                flags |= _SDL_WINDOW_FULLSCREEN
            else:
                flags |= _SDL_WINDOW_FULLSCREEN_DESKTOP

        _kwarg_to_flag = self._kwarg_to_flag
        for k, v in kwargs.items():
            try:
                flag = _kwarg_to_flag[k]
                if v:
                    flags |= flag
            except KeyError:
                raise error("unknown parameter: %s" % k)

        self._win = SDL_CreateWindow(title.encode('utf8'), x, y,
                                     size[0], size[1], flags)
        if not self._win:
            raise error()
        SDL_SetWindowData(self._win, "pg_window", <PyObject*>self)

        import pygame.pkgdata
        surf = pygame.image.load(pygame.pkgdata.getResource(
                                 'pygame_icon.bmp'))
        surf.set_colorkey(0)
        self.set_icon(surf)

    @property
    def grab(self):
        """get a window's input grab state (True or False).

        When input is grabbed the mouse is confined to the window.
        If the caller enables a grab while another window is currently grabbed,
        the other window loses its grab in favor of the caller's window."""
        return SDL_GetWindowGrab(self._win) != 0

    @grab.setter
    def grab(self, bint enabled):
        """set a window's input grab state (True or False).

        When input is grabbed the mouse is confined to the window.
        If the caller enables a grab while another window is currently grabbed,
        the other window loses its grab in favor of the caller's window."""
        SDL_SetWindowGrab(self._win, 1 if enabled else 0)

    def set_fullscreen(self, int mode):
        """set the window's fullscreen state.
        :param mode int: 0: windowed mode
                         1: fullscreen window ("real" fullscreen with a videomode change)
                         2: fullscreen window at the current desktop resolution
                            ("fake" fullscreen that takes the size of the desktop)
        """
        cdef int flags = 0
        if mode > 0:
            if mode == 1:
                flags = _SDL_WINDOW_FULLSCREEN
            else:
                flags = _SDL_WINDOW_FULLSCREEN_DESKTOP
        if SDL_SetWindowFullscreen(self._win, flags):
            raise error()

    @property
    def title(self):
        return SDL_GetWindowTitle(self._win).decode('utf8')

    @title.setter
    def title(self, new_title):
        SDL_SetWindowTitle(self._win, new_title.encode('utf8'))

    def destroy(self):
        if self._win:
            SDL_DestroyWindow(self._win)
            self._win = NULL

    def hide(self):
        SDL_HideWindow(self._win)

    def show(self):
        SDL_ShowWindow(self._win)

    def focus(self, input_only=False):
        """raise a window above other windows and set the input focus.
        If input_only is True, the window will be given input focus
        but may be completely obscured by other windows."""
        if input_only:
            if SDL_SetWindowInputFocus(self._win):
                raise error()
        else:
            SDL_RaiseWindow(self._win)

    def restore(self):
        """restore the size and position of a minimized or maximized window."""
        SDL_RestoreWindow(self._win)

    def maximize(self):
        SDL_MaximizeWindow(self._win)

    def minimize(self):
        SDL_MinimizeWindow(self._win)

    @property
    def resizable(self):
        return SDL_GetWindowFlags(self._win) & _SDL_WINDOW_RESIZABLE != 0

    @resizable.setter
    def resizable(self, enabled):
        SDL_SetWindowResizable(self._win, 1 if enabled else 0)

    @property
    def borderless(self):
        return SDL_GetWindowFlags(self._win) & _SDL_WINDOW_BORDERLESS != 0

    @borderless.setter
    def borderless(self, enabled):
        """add or remove the border from the actual window.
        You can't change the border state of a fullscreen window."""
        SDL_SetWindowBordered(self._win, 1 if enabled else 0)

    def set_icon(self, surface):
        """set the icon for the window.
        :param surface Surface: A Surface to use as the icon."""
        if not pgSurface_Check(surface):
            raise error('surface must be a Surface object')
        SDL_SetWindowIcon(self._win, pgSurface_AsSurface(surface))

    @property
    def id(self):
        return SDL_GetWindowID(self._win)

    @property
    def size(self):
        """get the size of a window's client area."""
        cdef int w, h
        SDL_GetWindowSize(self._win, &w, &h)
        return (w, h)

    @size.setter
    def size(self, size):
        """set the size of a window's client area."""
        SDL_SetWindowSize(self._win, size[0], size[1])

    @property
    def position(self):
        cdef int x, y
        SDL_GetWindowPosition(self._win, &x, &y)
        return (x, y)

    @position.setter
    def position(self, position):
        """screen coordinates, or WINDOWPOS_CENTERED or WINDOWPOS_UNDEFINED"""
        cdef int x, y
        if position == WINDOWPOS_UNDEFINED:
            x, y = WINDOWPOS_UNDEFINED, WINDOWPOS_UNDEFINED
        elif position == WINDOWPOS_CENTERED:
            x, y = WINDOWPOS_CENTERED, WINDOWPOS_CENTERED
        else:
            x, y = position
        SDL_SetWindowPosition(self._win, x, y)

    @property
    def opacity(self):
        """get window opacity in (0.0f transparent, 1.0f opaque)"""
        cdef float opacity
        if SDL_GetWindowOpacity(self._win, &opacity):
            raise error()
        return opacity

    @opacity.setter
    def opacity(self, opacity):
        """set window opacity in (0.0f transparent, 1.0f opaque)"""
        if SDL_SetWindowOpacity(self._win, opacity):
            raise error()

    @property
    def brightness(self):
        """set the brightness (gamma multiplier) for the display that owns a given window.
        the brightness (gamma multiplier) value to set where 0.0 is
        completely dark and 1.0 is normal brightness"""
        return SDL_GetWindowBrightness(self._win)

    @brightness.setter
    def brightness(self, float value):
        """set the brightness (gamma multiplier) for the display that owns a given window.
        the brightness (gamma multiplier) value to set where 0.0 is
        completely dark and 1.0 is normal brightness"""
        if SDL_SetWindowBrightness(self._win, value):
            raise error()

    @property
    def display_index(self):
        """get the index of the display associated with the window."""
        cdef int index = SDL_GetWindowDisplayIndex(self._win)
        if index < 0:
            raise error()
        return index

    def set_modal_for(self, Window parent):
        """set the window as a modal for a parent window
        This function is only supported on X11."""
        if SDL_SetWindowModalFor(self._win, parent._win):
            raise error()

    def __dealloc__(self):
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

    def __dealloc__(self):
        if self._tex:
            SDL_DestroyTexture(self._tex)

cdef class Renderer:
    def __init__(self, Window window, int index=-1,
                 int accelerated=-1, bint vsync=False,
                 bint target_texture=False):
        """ Create a 2D rendering context for a window.

        :param window Window: where rendering is displayed.
        :param index int: index of rendering driver to initialize,
                          or -1 to init the first supporting requested options.
        :param accelerated int: if 1, the renderer uses hardware acceleration.
                                if 0, the renderer is a software fallback.
                                -1 gives precedence to renderers using hardware acceleration.
        :param vsync bool: .present() is synchronized with the refresh rate.
        :param target_texture bool: the renderer supports rendering to texture.
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

        self._draw_color = (255, 255, 255, 255)

    def __dealloc__(self):
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
