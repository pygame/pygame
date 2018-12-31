from cpython cimport PyObject
from . import error
from . import error as errorfnc
from libc.stdlib cimport free, malloc


WINDOWPOS_UNDEFINED = _SDL_WINDOWPOS_UNDEFINED
WINDOWPOS_CENTERED = _SDL_WINDOWPOS_CENTERED

MESSAGEBOX_ERROR = _SDL_MESSAGEBOX_ERROR
MESSAGEBOX_WARNING = _SDL_MESSAGEBOX_WARNING
MESSAGEBOX_INFORMATION = _SDL_MESSAGEBOX_INFORMATION


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


def messagebox(title, message,
               Window window=None,
               bint info=False,
               bint warn=False,
               bint error=False,
               buttons=('OK', ),
               return_button=0,
               escape_button=0):
    """ Display a message box.

    :param str title: A title string or None.
    :param str message: A message string.
    :param bool info: If ``True``, display an info message.
    :param bool warn: If ``True``, display a warning message.
    :param bool error: If ``True``, display an error message.
    :param tuple buttons: An optional sequence of buttons to show to the user (strings).
    :param int return_button: Button index to use if the return key is hit (-1 for none).
    :param int escape_button: Button index to use if the escape key is hit (-1 for none).
    :return: The index of the button that was pushed.
    """
    # TODO: type check
    # TODO: color scheme
    cdef SDL_MessageBoxButtonData* c_buttons = NULL

    cdef SDL_MessageBoxData data
    data.flags = 0
    if warn:
        data.flags |= _SDL_MESSAGEBOX_WARNING
    if error:
        data.flags |= _SDL_MESSAGEBOX_ERROR
    if info:
        data.flags |= _SDL_MESSAGEBOX_INFORMATION
    if not window:
        data.window = NULL
    else:
        data.window = window._win
    if title is not None:
        title = title.encode('utf8')
        data.title = title
    else:
        data.title = NULL
    message = message.encode('utf8')
    data.message = message
    data.colorScheme = NULL

    cdef SDL_MessageBoxButtonData button
    if not buttons:
        button.flags |= _SDL_MESSAGEBOX_BUTTON_RETURNKEY_DEFAULT |\
                        _SDL_MESSAGEBOX_BUTTON_ESCAPEKEY_DEFAULT
        button.buttonid = 0
        button.text = "OK"
        data.buttons = &button
        data.numbuttons = 1
    else:
        buttons_utf8 = [s.encode('utf8') for s in buttons]
        data.numbuttons = len(buttons)
        c_buttons =\
            <SDL_MessageBoxButtonData*>malloc(data.numbuttons * sizeof(SDL_MessageBoxButtonData))
        if not c_buttons:
            raise MemoryError()
        for i, but in enumerate(reversed(buttons_utf8)):
            c_buttons[i].flags = 0
            c_buttons[i].buttonid = data.numbuttons - i - 1
            if c_buttons[i].buttonid == return_button:
                c_buttons[i].flags |= _SDL_MESSAGEBOX_BUTTON_RETURNKEY_DEFAULT
            if c_buttons[i].buttonid == escape_button:
                c_buttons[i].flags |= _SDL_MESSAGEBOX_BUTTON_ESCAPEKEY_DEFAULT
            c_buttons[i].text = but
        data.buttons = c_buttons

    cdef int buttonid
    if SDL_ShowMessageBox(&data, &buttonid):
        free(c_buttons)
        raise errorfnc()

    free(c_buttons)
    return buttonid


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
                 bint fullscreen=False,
                 bint fullscreen_desktop=False, **kwargs):
        """ Create a window with the specified position, dimensions, and flags.

        :param str title: the title of the window, in UTF-8 encoding
        :param tuple size: the size of the window, in screen coordinates (width, height)
        :param position: a tuple specifying the window position, WINDOWPOS_CENTERED, or WINDOWPOS_UNDEFINED.
        :param bool fullscreen: fullscreen window using the window size as the resolution (videomode change)
        :param bool fullscreen_desktop: fullscreen window using the current desktop resolution
        :param bool opengl: Usable with OpenGL context. You will still need to create an OpenGL context.
        :param bool vulkan: usable with a Vulkan instance
        :param bool hidden: window is not visible
        :param bool borderless: no window decoration
        :param bool resizable: window can be resized
        :param bool minimized: window is minimized
        :param bool maximized: window is maximized
        :param bool input_grabbed: window has grabbed input focus
        :param bool input_focus: window has input focus
        :param bool mouse_focus: window has mouse focus
        :param bool foreign: window not created by SDL
        :param bool allow_highdpi: window should be created in high-DPI mode if supported (>= SDL 2.0.1)
        :param bool mouse_capture: window has mouse captured (unrelated to INPUT_GRABBED, >= SDL 2.0.4)
        :param bool always_on_top: window should always be above others (X11 only, >= SDL 2.0.5)
        :param bool skip_taskbar: window should not be added to the taskbar (X11 only, >= SDL 2.0.5)
        :param bool utility: window should be treated as a utility window (X11 only, >= SDL 2.0.5)
        :param bool tooltip: window should be treated as a tooltip (X11 only, >= SDL 2.0.5)
        :param bool popup_menu: window should be treated as a popup menu (X11 only, >= SDL 2.0.5)
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
        if fullscreen and fullscreen_desktop:
            raise error("fullscreen and fullscreen_desktop cannot be used at the same time.")
        if fullscreen:
            flags |= _SDL_WINDOW_FULLSCREEN
        elif fullscreen_desktop:
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
        """ Get window's input grab state (``True`` or ``False``).

        When input is grabbed the mouse is confined to the window.
        If the caller enables a grab while another window is currently grabbed,
        the other window loses its grab in favor of the caller's window.

        :rtype: bool
        """
        return SDL_GetWindowGrab(self._win) != 0

    @grab.setter
    def grab(self, bint grabbed):
        """ Set window's input grab state (``True`` or ``False``).

        When input is grabbed the mouse is confined to the window.
        If the caller enables a grab while another window is currently grabbed,
        the other window loses its grab in favor of the caller's window.

        :param bool grabbed: ``True`` to grab, ``False`` to release.
        """
        # https://wiki.libsdl.org/SDL_SetWindowGrab
        SDL_SetWindowGrab(self._win, 1 if grabbed else 0)

    def set_windowed(self):
        """ Enable windowed mode

        .. seealso:: :func:`set_fullscreen`

        """
        # https://wiki.libsdl.org/SDL_SetWindowFullscreen
        if SDL_SetWindowFullscreen(self._win, 0):
            raise error()

    #TODO: not sure about this...
    # Perhaps this is more readable:
    #     window.fullscreen = True
    #     window.fullscreen_desktop = True
    #     window.windowed = True
    def set_fullscreen(self, bint desktop=False):
        """ Enable fullscreen for the window

        :param bool desktop: If ``True``: use the current desktop resolution.
                             If ``False``: change the fullscreen resolution to the window size.

        .. seealso:: :func:`set_windowed`
        """
        cdef int flags = 0
        if desktop:
            flags = _SDL_WINDOW_FULLSCREEN_DESKTOP
        else:
            flags = _SDL_WINDOW_FULLSCREEN
        if SDL_SetWindowFullscreen(self._win, flags):
            raise error()

    @property
    def title(self):
        """ Returns the title of the window or u"" if there is none.
        """
        # https://wiki.libsdl.org/SDL_GetWindowTitle
        return SDL_GetWindowTitle(self._win).decode('utf8')

    @title.setter
    def title(self, title):
        """ Set the window title.

        :param str title: the desired window title in UTF-8.
        """
        # https://wiki.libsdl.org/SDL_SetWindowTitle
        SDL_SetWindowTitle(self._win, title.encode('utf8'))

    def destroy(self):
        """ Destroys the window.
        """
        # https://wiki.libsdl.org/SDL_DestroyWindow
        if self._win:
            SDL_DestroyWindow(self._win)
            self._win = NULL

    def hide(self):
        """ Hide the window.
        """
        # https://wiki.libsdl.org/SDL_HideWindow
        SDL_HideWindow(self._win)

    def show(self):
        """ Show the window.
        """
        # https://wiki.libsdl.org/SDL_ShowWindow
        SDL_ShowWindow(self._win)

    def focus(self, input_only=False):
        """ Raise the window above other windows and set the input focus.

        :param bool input_only: if ``True``, the window will be given input focus
                                but may be completely obscured by other windows.
        """
        # https://wiki.libsdl.org/SDL_RaiseWindow
        if input_only:
            if SDL_SetWindowInputFocus(self._win):
                raise error()
        else:
            SDL_RaiseWindow(self._win)

    def restore(self):
        """ Restore the size and position of a minimized or maximized window.
        """
        SDL_RestoreWindow(self._win)

    def maximize(self):
        """ Maximize the window.
        """
        SDL_MaximizeWindow(self._win)

    def minimize(self):
        """ Minimize the window.
        """
        SDL_MinimizeWindow(self._win)

    @property
    def resizable(self):
        """ Sets whether the window is resizable.
        """
        return SDL_GetWindowFlags(self._win) & _SDL_WINDOW_RESIZABLE != 0

    @resizable.setter
    def resizable(self, enabled):
        SDL_SetWindowResizable(self._win, 1 if enabled else 0)

    @property
    def borderless(self):
        """ Add or remove the border from the actual window.

        .. note:: You can't change the border state of a fullscreen window.
        """
        return SDL_GetWindowFlags(self._win) & _SDL_WINDOW_BORDERLESS != 0

    @borderless.setter
    def borderless(self, enabled):
        SDL_SetWindowBordered(self._win, 1 if enabled else 0)

    def set_icon(self, surface):
        """ Set the icon for the window.

        :param pygame.Surface surface: A Surface to use as the icon.
        """
        if not pgSurface_Check(surface):
            raise error('surface must be a Surface object')
        SDL_SetWindowIcon(self._win, pgSurface_AsSurface(surface))

    @property
    def id(self):
        """ A unique window ID. *Read-only*.

        :rtype: int
        """
        return SDL_GetWindowID(self._win)

    @property
    def size(self):
        """ The size of the window's client area."""
        cdef int w, h
        SDL_GetWindowSize(self._win, &w, &h)
        return (w, h)

    @size.setter
    def size(self, size):
        SDL_SetWindowSize(self._win, size[0], size[1])

    @property
    def position(self):
        """ Window's screen coordinates, or WINDOWPOS_CENTERED or WINDOWPOS_UNDEFINED"""
        cdef int x, y
        SDL_GetWindowPosition(self._win, &x, &y)
        return (x, y)

    @position.setter
    def position(self, position):
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
        """ Window opacity. It ranges between 0.0 (fully transparent)
        and 1.0 (fully opaque)."""
        cdef float opacity
        if SDL_GetWindowOpacity(self._win, &opacity):
            raise error()
        return opacity

    @opacity.setter
    def opacity(self, opacity):
        if SDL_SetWindowOpacity(self._win, opacity):
            raise error()

    @property
    def brightness(self):
        """ The brightness (gamma multiplier) for the display that owns a given window.
        It ranges between 0.0 (completely dark) and 1.0 (normal brightness)."""
        return SDL_GetWindowBrightness(self._win)

    @brightness.setter
    def brightness(self, float value):
        if SDL_SetWindowBrightness(self._win, value):
            raise error()

    @property
    def display_index(self):
        """ The index of the display associated with the window. *Read-only*.

        :rtype: int
        """
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

        :param Renderer renderer: Rendering context for the texture.
        :param pygame.Surface surface: The surface to create a texture from.
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

        :param Window window: where rendering is displayed.
        :param int index: index of rendering driver to initialize,
                          or -1 to init the first supporting requested options.
        :param int accelerated: if 1, the renderer uses hardware acceleration.
                                if 0, the renderer is a software fallback.
                                -1 gives precedence to renderers using hardware acceleration.
        :param bool vsync: .present() is synchronized with the refresh rate.
        :param bool target_texture: the renderer supports rendering to texture.
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
        """ Color used by the drawing functions.
        """
        return self._draw_color

    @draw_color.setter
    def draw_color(self, new_value):
        """ color used by the drawing functions.
        """
        # https://wiki.libsdl.org/SDL_SetRenderDrawColor
        # TODO: this should probably be a pygame.Color.
        self._draw_color = new_value[:]
        res = SDL_SetRenderDrawColor(self._renderer,
                                     new_value[0],
                                     new_value[1],
                                     new_value[2],
                                     new_value[3])
        if res < 0:
            raise error()

    def clear(self):
        """ Clear the current rendering target with the drawing color.
        """
        # https://wiki.libsdl.org/SDL_RenderClear
        res = SDL_RenderClear(self._renderer)
        if res < 0:
            raise error()

    def copy(self, Texture texture, srcrect=None, dstrect=None):
        """ Copy portion of texture to rendering target.

        :param Texture texture: the source texture.
        :param srcrect: source rectangle on the texture, or None for the entire texture.
        :type srcrect: pygame.Rect or None
        :param dstrect: destination rectangle on the render target, or None for entire target.
                        The texture is stretched to fill dstrect.
        :type dstrect: pygame.Rect or None
        """
        # https://wiki.libsdl.org/SDL_RenderCopy
        cdef SDL_Rect src, dst
        cdef SDL_Rect *csrcrect = pgRect_FromObject(srcrect, &src)
        cdef SDL_Rect *cdstrect = pgRect_FromObject(dstrect, &dst)
        res = SDL_RenderCopy(self._renderer, texture._tex, csrcrect, cdstrect)
        if res < 0:
            raise error()

    def present(self):
        """ Present the composed backbuffer to the screen.

        Updates the screen with any rendering performed since previous call.
        """
        # https://wiki.libsdl.org/SDL_RenderPresent
        SDL_RenderPresent(self._renderer)
