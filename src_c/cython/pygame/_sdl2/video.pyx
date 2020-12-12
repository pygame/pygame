from cpython cimport PyObject
from . import error
from . import error as errorfnc
from libc.stdlib cimport free, malloc


WINDOWPOS_UNDEFINED = _SDL_WINDOWPOS_UNDEFINED
WINDOWPOS_CENTERED = _SDL_WINDOWPOS_CENTERED

MESSAGEBOX_ERROR = _SDL_MESSAGEBOX_ERROR
MESSAGEBOX_WARNING = _SDL_MESSAGEBOX_WARNING
MESSAGEBOX_INFORMATION = _SDL_MESSAGEBOX_INFORMATION


cdef extern from "SDL.h" nogil:
    Uint32 SDL_GetWindowPixelFormat(SDL_Window* window)
    SDL_bool SDL_IntersectRect(const SDL_Rect* A,
                               const SDL_Rect* B,
                               SDL_Rect*       result)
    void SDL_SetWindowResizable(SDL_Window *window, SDL_bool resizable)
    int SDL_GetWindowOpacity(SDL_Window *window, float *opacity)
    int SDL_SetWindowOpacity(SDL_Window *window, float opacity)
    int SDL_SetWindowModalFor(SDL_Window *modal_window, SDL_Window *parent_window)
    int SDL_SetWindowInputFocus(SDL_Window *window)
    int SDL_SetRelativeMouseMode(SDL_bool enabled)
    SDL_bool SDL_GetRelativeMouseMode()
    SDL_Renderer* SDL_GetRenderer(SDL_Window* window)
    SDL_Window* SDL_GetWindowFromID(Uint32 id)
    SDL_Surface * SDL_CreateRGBSurfaceWithFormat(Uint32 flags, int width, int height, int depth, Uint32 format)


cdef extern from "pygame.h" nogil:
    ctypedef struct pgSurfaceObject:
        pass

    int pgSurface_Check(object surf)
    SDL_Surface* pgSurface_AsSurface(object surf)
    void import_pygame_surface()

    SDL_Window* pg_GetDefaultWindow()
    void import_pygame_base()

    int pgRect_Check(object rect)
    SDL_Rect *pgRect_FromObject(object obj, SDL_Rect *temp)
    object pgRect_New(SDL_Rect *r)
    object pgRect_New4(int x, int y, int w, int h)
    SDL_Rect pgRect_AsRect(object rect)
    void import_pygame_rect()


    object pgColor_New(Uint8 rgba[])
    object pgColor_NewLength(Uint8 rgba[], Uint8 length)
    void import_pygame_color()
    pgSurfaceObject *pgSurface_New2(SDL_Surface *info, int owner)

cdef extern from "pgcompat.h" nogil:
    pass

import_pygame_base()
import_pygame_color()
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
        button.flags = _SDL_MESSAGEBOX_BUTTON_RETURNKEY_DEFAULT |\
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

    @classmethod
    def from_display_module(cls):
        cdef Window self = cls.__new__(cls)
        cdef SDL_Window* window = pg_GetDefaultWindow()
        if not window:
            raise error()
        self._win=window
        self._is_borrowed=1
        SDL_SetWindowData(window, "pg_window", <PyObject*>self)
        return self

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
            raise ValueError("fullscreen and fullscreen_desktop cannot be used at the same time.")
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
                raise TypeError("unknown parameter: %s" % k)

        self._win = SDL_CreateWindow(title.encode('utf8'), x, y,
                                     size[0], size[1], flags)
        self._is_borrowed=0
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
        """ Window's input grab state (``True`` or ``False``).

        Set it to ``True`` to grab, ``False`` to release.

        When input is grabbed the mouse is confined to the window.
        If the caller enables a grab while another window is currently grabbed,
        the other window loses its grab in favor of the caller's window.

        :rtype: bool
        """
        return SDL_GetWindowGrab(self._win) != 0

    @grab.setter
    def grab(self, bint grabbed):
        # https://wiki.libsdl.org/SDL_SetWindowGrab
        SDL_SetWindowGrab(self._win, 1 if grabbed else 0)

    @property
    def relative_mouse(self):
        """ Window's relative mouse motion state (``True`` or ``False``).

        Set it to ``True`` to enable, ``False`` to disable.
        If mouse.set_visible(True) is set the input will be grabbed,
        and the mouse will enter endless relative motion mode.

        :rtype: bool
        """
        return SDL_GetRelativeMouseMode()


    @relative_mouse.setter
    def relative_mouse(self, bint enable):
        # https://wiki.libsdl.org/SDL_SetRelativeMouseMode
        #SDL_SetWindowGrab(self._win, 1 if enable else 0)
        SDL_SetRelativeMouseMode(1 if enable else 0)

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
        """ The title of the window or u"" if there is none.
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
            raise TypeError('surface must be a Surface object')
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
        0.0 is completely dark and 1.0 is normal brightness."""
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
        if self._is_borrowed:
            return
        self.destroy()

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


cdef class Texture:
    def __cinit__(self):
        cdef Uint8[3] defaultColor = [255, 255, 255]
        self._color = pgColor_NewLength(defaultColor, 3)

    def __init__(self,
                 Renderer renderer,
                 size, int depth=0,
                 static=False, streaming=False,
                 target=False):
        """ Create an empty texture.

        :param Renderer renderer: Rendering context for the texture.
        :param tuple size: The width and height of the texture.
        :param int depth: The pixel format (0 to use the default).

        One of ``static``, ``streaming``, or ``target`` can be set
        to ``True``. If all are ``False``, then ``static`` is used.

        :param bool static: Changes rarely, not lockable.
        :param bool streaming: Changes frequently, lockable.
        :param bool target: Can be used as a render target.
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
        self.width, self.height = width, height

    @staticmethod
    def from_surface(Renderer renderer, surface):
        """ Create a texture from an existing surface.

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
        # https://wiki.libsdl.org/SDL_GetTextureAlphaMod
        cdef Uint8 alpha
        res = SDL_GetTextureAlphaMod(self._tex, &alpha)
        if res < 0:
            raise error()

        return alpha

    @alpha.setter
    def alpha(self, Uint8 new_value):
        # https://wiki.libsdl.org/SDL_SetTextureAlphaMod
        res = SDL_SetTextureAlphaMod(self._tex, new_value)
        if res < 0:
            raise error()

    @property
    def blend_mode(self):
        # https://wiki.libsdl.org/SDL_GetTextureBlendMode
        cdef SDL_BlendMode blendMode
        res = SDL_GetTextureBlendMode(self._tex, &blendMode)
        if res < 0:
            raise error()

        return blendMode

    @blend_mode.setter
    def blend_mode(self, blendMode):
        # https://wiki.libsdl.org/SDL_SetTextureBlendMode
        res = SDL_SetTextureBlendMode(self._tex, blendMode)
        if res < 0:
            raise error()

    @property
    def color(self):
        # https://wiki.libsdl.org/SDL_GetTextureColorMod
        res = SDL_GetTextureColorMod(self._tex,
            &self._color.data[0],
            &self._color.data[1],
            &self._color.data[2])
        if res < 0:
            raise error()

        return self._color

    @color.setter
    def color(self, new_value):
        # https://wiki.libsdl.org/SDL_SetTextureColorMod
        res = SDL_SetTextureColorMod(self._tex,
                                     new_value[0],
                                     new_value[1],
                                     new_value[2])
        if res < 0:
            raise error()

    def get_rect(self, **kwargs):
        """ Get the rectangular area of the texture.
        like surface.get_rect(), returns a new rectangle covering the entire surface.
        This rectangle will always start at 0, 0 with a width. and height the same size as the texture.
        """
        rect = pgRect_New4(0, 0, self.width, self.height)
        for key in kwargs:
            setattr(rect, key, kwargs[key])

        return rect

    cdef draw_internal(self, SDL_Rect *csrcrect, SDL_Rect *cdstrect, float angle=0, SDL_Point *originptr=NULL,
                       bint flipX=False, bint flipY=False):
        cdef int flip = SDL_FLIP_NONE
        if flipX:
            flip |= SDL_FLIP_HORIZONTAL
        if flipY:
            flip |= SDL_FLIP_VERTICAL

        res = SDL_RenderCopyEx(self.renderer._renderer, self._tex, csrcrect, cdstrect,
                               angle, originptr, <SDL_RendererFlip>flip)
        if res < 0:
            raise error()

    cpdef void draw(self, srcrect=None, dstrect=None, float angle=0, origin=None,
                    bint flipX=False, bint flipY=False):
        """ Copy a portion of the texture to the rendering target.

        :param srcrect: source rectangle on the texture, or None for the entire texture.
        :param dstrect: destination rectangle or position on the render target, or None for entire target.
                        The texture is stretched to fill dstrect.
        :param float angle: angle (in degrees) to rotate dstrect around (clockwise).
        :param origin: point around which dstrect will be rotated.
                       If None, it will equal the center: (dstrect.w/2, dstrect.h/2).
        :param bool flipX: flip horizontally.
        :param bool flipY: flip vertically.
        """
        cdef SDL_Rect src, dst
        cdef SDL_Rect *csrcrect = NULL
        cdef SDL_Rect *cdstrect = NULL
        cdef SDL_Point corigin
        cdef SDL_Point *originptr

        if srcrect is not None:
            csrcrect = pgRect_FromObject(srcrect, &src)
            if not csrcrect:
                raise TypeError("the argument is not a rectangle or None")

        if dstrect is not None:
            cdstrect = pgRect_FromObject(dstrect, &dst)
            if cdstrect == NULL:
                if len(dstrect) == 2:
                    dst.x = dstrect[0]
                    dst.y = dstrect[1]
                    dst.w = self.width
                    dst.h = self.height
                    cdstrect = &dst
                else:
                    raise TypeError('dstrect must be a position, rect, or None')

        if origin:
            originptr = &corigin
            corigin.x = origin[0]
            corigin.y = origin[1]
        else:
            originptr = NULL

        self.draw_internal(csrcrect, cdstrect, angle, originptr,
                           flipX, flipY)

    def update(self, surface, area=None):
        # https://wiki.libsdl.org/SDL_UpdateTexture
        # Should it accept a raw pixel data array too?
        """ Update the texture with Surface.
        This is a fairly slow function, intended for use with static textures that do not change often.

        If the texture is intended to be updated often,
        it is preferred to create the texture as streaming and use the locking functions.

        While this function will work with streaming textures,
        for optimization reasons you may not get the pixels back if you lock the texture afterward.

        :param surface: source Surface.
        """

        if not pgSurface_Check(surface):
            raise TypeError("update source should be a Surface.")

        cdef SDL_Rect rect
        cdef SDL_Rect *rectptr = pgRect_FromObject(area, &rect)
        cdef SDL_Surface *surf = pgSurface_AsSurface(surface)

        if rectptr == NULL and area is not None:
            raise TypeError('area must be a rectangle or None')

        res = SDL_UpdateTexture(self._tex, rectptr, surf.pixels, surf.pitch)
        if res < 0:
            raise error()

cdef class Image:

    def __cinit__(self):
        self.angle = 0
        self.origin[0] = 0
        self.origin[1] = 0
        self.flipX = False
        self.flipY = False

        cdef Uint8[4] defaultColor = [255, 255, 255, 255]
        self.color = pgColor_NewLength(defaultColor, 3)
        self.alpha = 255

    def __init__(self, textureOrImage, srcrect=None):
        cdef SDL_Rect temp
        cdef SDL_Rect *rectptr

        if isinstance(textureOrImage, Image):
            self.texture = textureOrImage.texture
            self.srcrect = pgRect_New(&(<Rect>textureOrImage.srcrect).r)
        else:
            self.texture = textureOrImage
            self.srcrect = textureOrImage.get_rect()

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

        self.origin[0] = self.srcrect.w / 2
        self.origin[1] = self.srcrect.h / 2

    def get_rect(self):
        return pgRect_New(&self.srcrect.r)

    cpdef void draw(self, srcrect=None, dstrect=None):
        """ Copy a portion of the image to the rendering target.

        :param srcrect: source rectangle specifying a sub-image, or None for the entire image.
        :param dstrect: destination rectangle or position on the render target, or None for entire target.
                        The image is stretched to fill dstrect.
        """
        cdef SDL_Rect src
        cdef SDL_Rect dst
        cdef SDL_Rect *csrcrect = NULL
        cdef SDL_Rect *cdstrect = NULL
        cdef SDL_Point origin
        cdef SDL_Rect *rectptr

        if srcrect is None:
            csrcrect = &self.srcrect.r
        else:
            if pgRect_Check(srcrect):
                src = (<Rect>srcrect).r
            else:

                rectptr = pgRect_FromObject(srcrect, &src)
                if rectptr == NULL:
                    raise TypeError('srcrect must be a rect or None')
                src.x = rectptr.x
                src.y = rectptr.y
                src.w = rectptr.w
                src.h = rectptr.h

            src.x += self.srcrect.x
            src.y += self.srcrect.y
            csrcrect = &src

        if dstrect is not None:
            cdstrect = pgRect_FromObject(dstrect, &dst)
            if cdstrect == NULL:
                if len(dstrect) == 2:
                    dst.x = dstrect[0]
                    dst.y = dstrect[1]
                    dst.w = self.srcrect.w
                    dst.h = self.srcrect.h
                    cdstrect = &dst
                else:
                    raise TypeError('dstrect must be a position, rect, or None')

        self.texture.color = self.color
        self.texture.alpha = self.alpha

        origin.x = <int>self.origin[0]
        origin.y = <int>self.origin[1]

        self.texture.draw_internal(csrcrect, cdstrect, self.angle,
                                   &origin, self.flipX, self.flipY)


cdef class Renderer:

    @classmethod
    def from_window(cls, Window window):
        cdef Renderer self = cls.__new__(cls)
        self._win = window
        if window._is_borrowed:
            self._is_borrowed=1
        else:
            raise error()
        if not self._win:
            raise error()

        self._renderer =  SDL_GetRenderer(self._win._win)
        if not self._renderer:
            raise error()

        cdef Uint8[4] defaultColor = [255, 255, 255, 255]
        self._draw_color = pgColor_NewLength(defaultColor, 4)
        self._target = None
        return self

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

        cdef Uint8[4] defaultColor = [255, 255, 255, 255]
        self._draw_color = pgColor_NewLength(defaultColor, 4)
        self._target = None
        self._win = window
        self._is_borrowed=0

    def __dealloc__(self):
        if self._is_borrowed:
            return
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
        self._draw_color[:] = new_value
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

    def present(self):
        """ Present the composed backbuffer to the screen.

        Updates the screen with any rendering performed since previous call.
        """
        # https://wiki.libsdl.org/SDL_RenderPresent
        SDL_RenderPresent(self._renderer)

    cpdef get_viewport(self):
        """ Returns the drawing area on the target.

        :rtype: pygame.Rect
        """
        # https://wiki.libsdl.org/SDL_RenderGetViewport
        cdef SDL_Rect rect
        SDL_RenderGetViewport(self._renderer, &rect)
        return pgRect_New(&rect)

    @property
    def logical_size(self):
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
        cdef float x
        cdef float y
        SDL_RenderGetScale(self._renderer, &x, &y);
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

    def set_viewport(self, area):
        """ Set the drawing area on the target.
        If this is set to ``None``, the entire target will be used.

        :param area: A ``pygame.Rect`` or tuple representing the
                     drawing area on the target, or None.
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
    def target(self):
        """ The current render target. Set to ``None`` for the default target.

        :rtype: Texture, None
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
        """ Only for compatibility.
        Textures created by different Renderers cannot shared with each other!
        :param source: A Texture or Image to draw.
        :param dest: destination on the render target.
        :param area: the portion of source texture.
        :param special_flags: have no effect at this moment.
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
        # https://wiki.libsdl.org/SDL_RenderDrawLine
        res = SDL_RenderDrawLine(self._renderer,
                                 p1[0], p1[1],
                                 p2[0], p2[1])
        if res < 0:
            raise error()

    def draw_point(self, point):
        # https://wiki.libsdl.org/SDL_RenderDrawPoint
        res = SDL_RenderDrawPoint(self._renderer,
                                  point[0], point[1])
        if res < 0:
            raise error()

    def draw_rect(self, rect):
        # https://wiki.libsdl.org/SDL_RenderDrawRect
        cdef SDL_Rect _rect
        cdef SDL_Rect *rectptr = pgRect_FromObject(rect, &_rect)
        if rectptr == NULL:
            raise TypeError('expected a rectangle')
        res = SDL_RenderDrawRect(self._renderer, rectptr)
        if res < 0:
            raise error()

    def fill_rect(self, rect):
        # https://wiki.libsdl.org/SDL_RenderFillRect
        cdef SDL_Rect _rect
        cdef SDL_Rect *rectptr = pgRect_FromObject(rect, &_rect)
        if rectptr == NULL:
            raise TypeError('expected a rectangle')
        res = SDL_RenderFillRect(self._renderer, rectptr)

        if res < 0:
            raise error()

    def to_surface(self, surface=None, area=None):
        # https://wiki.libsdl.org/SDL_RenderReadPixels
        """
            Read pixels from the current rendering target and create a pygame.Surface.
            WARNING: This is a very slow operation, and should not be used frequently.

        :param surface: A surface to read the pixel data into.
                        It must be large enough to fit the area, or ``ValueError`` is
                        raised.
                        If ``None``, a new surface is returned.
        :param area: The area of the screen to read pixels from. The area is
                     clipped to fit inside the viewport.
                     If ``None``, the entire viewport is used.
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
