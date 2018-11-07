from ._sdl2 import *
from pygame import Surface


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
    num = SDL_GetNumRenderDrivers()
    for ind in range(num):
        info = SDL_RendererInfo()
        SDL_GetRenderDriverInfo(ind, byref(info))
        ret = RendererDriverInfo()
        for name, _ in info._fields_:
            setattr(ret, name, getattr(info, name))
        yield ret

class Renderer:
    SOFTWARE = SDL_RENDERER_SOFTWARE
    ACCELERATED = SDL_RENDERER_ACCELERATED
    PRESENT_VSYNC = SDL_RENDERER_PRESENTVSYNC
    TARGET_TEXTURE = SDL_RENDERER_TARGETTEXTURE

    def __init__(self, window, index=-1, flags=ACCELERATED):
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
        SDL_SetRenderDrawColor(self._renderer, *new_value)

    def clear(self):
        SDL_RenderClear(self._renderer)

    def copy_pos(self, texture, x, y):
        srcrect = byref(SDL_Rect(0, 0, texture.width, texture.height))
        dstrect = byref(SDL_Rect(x, y, texture.width, texture.height))
        SDL_RenderCopy(self._renderer, texture._tex, srcrect, dstrect)

    def copy(self, texture, srcrect=None, dstrect=None):
        srcrect = byref(SDL_Rect(*srcrect)) if srcrect else None
        dstrect = byref(SDL_Rect(*dstrect)) if dstrect else None
        SDL_RenderCopy(self._renderer, texture._tex, srcrect, dstrect)

    def present(self):
        SDL_RenderPresent(self._renderer)

class Texture:
    def __init__(self, renderer, surface):
        self.renderer = renderer
        self._tex = SDL_CreateTextureFromSurface(renderer._renderer, get_surface_ptr(surface))
        self.width = surface.get_width()
        self.height = surface.get_height()

    def __del__(self):
        SDL_DestroyTexture(self._tex)
