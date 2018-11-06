from ._sdl2 import *
from pygame import Surface


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

    def copy(self, texture, srcrect=None, dstrect=None):
        srcrect = byref(SDL_Rect(*srcrect)) if srcrect else None
        dstrect = byref(SDL_Rect(*dstrect)) if dstrect else None
        SDL_RenderCopy(self._renderer, texture._tex, byref(srcrect), byref(dstrect))

    def present(self):
        SDL_RenderPresent(self._renderer)

class Texture:
    def __init__(self, renderer, surface):
        self.renderer = renderer
        self._tex = SDL_CreateTextureFromSurface(renderer._renderer, get_surface_ptr(surface))

    def __del__(self):
        SDL_DestroyTexture(self._tex)
