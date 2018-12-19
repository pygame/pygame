from cpython cimport PyObject

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

cdef extern from "../_pygame.h" nogil:
    SDL_Surface* pgSurface_AsSurface(object surf)

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
    POSITION_UNDEFINED = _SDL_WINDOWPOS_UNDEFINED
    POSITION_CENTERED = _SDL_WINDOWPOS_CENTERED
    DEFAULT_WIDTH = 640
    DEFAULT_HEIGHT = 480

    def __init__(self, title='pygame',
                 x=POSITION_UNDEFINED, y=POSITION_UNDEFINED,
                 w=DEFAULT_WIDTH, h=DEFAULT_HEIGHT, flags=0):
        self._win = SDL_CreateWindow(title.encode('utf8'), x, y, w, h, flags)

    @property
    def title(self):
        return SDL_GetWindowTitle(self._win).value

    @title.setter
    def title(self, new_title):
        SDL_SetWindowTitle(self._win, new_title.encode('utf8'))

    def destroy(self):
        if self._win:
            SDL_DestroyWindow(self._win)
            self._win = NULL

    def __del__(self):
        self.destroy()


cdef class Texture:
    def __init__(self, Renderer renderer, surface):
        self.renderer = renderer
        cdef SDL_Renderer* _renderer = renderer._renderer
        cdef SDL_Surface *surf_ptr = pgSurface_AsSurface(surface)
        self._tex = SDL_CreateTextureFromSurface(_renderer,
                                                 surf_ptr)
        self.width = surface.get_width()
        self.height = surface.get_height()

    def __del__(self):
        SDL_DestroyTexture(self._tex)

cdef class Renderer:
    SOFTWARE = _SDL_RENDERER_SOFTWARE
    ACCELERATED = _SDL_RENDERER_ACCELERATED
    PRESENT_VSYNC = _SDL_RENDERER_PRESENTVSYNC
    TARGET_TEXTURE = _SDL_RENDERER_TARGETTEXTURE

    def __init__(self, Window window, index=-1, flags=ACCELERATED):
        self._renderer = SDL_CreateRenderer(window._win, index, flags)
        self._draw_color = (255, 255, 255, 255)

    def __del__(self):
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
        # TODO: use pgRect_FromObject()
        cdef SDL_Rect *csrcrect
        cdef SDL_Rect *cdstrect
        cdef SDL_Rect src, dst
        if srcrect:
            src.x, src.y = srcrect.x, srcrect.y
            src.w, src.h = srcrect.w, srcrect.h
            csrcrect = &src
        else:
            csrcrect = NULL
        if dstrect:
            dst.x, dst.y = dstrect.x, dstrect.y
            dst.w, dst.h = dstrect.w, dstrect.h
            cdstrect = &dst
        else:
            cdstrect = NULL
        SDL_RenderCopy(self._renderer, texture._tex, csrcrect, cdstrect)

    def present(self):
        SDL_RenderPresent(self._renderer)
