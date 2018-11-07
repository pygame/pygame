from ._sdl2 import *


class Window:
    POSITION_UNDEFINED = SDL_WINDOWPOS_UNDEFINED
    POSITION_CENTERED = SDL_WINDOWPOS_CENTERED

    def __init__(self, title='pygame', x=POSITION_UNDEFINED, y=POSITION_UNDEFINED, w=640, h=480, flags=0):
        self._win = SDL_CreateWindow(title.encode(), x, y, w, h, flags)

    @property
    def title(self):
        return SDL_GetWindowTitle(self._win).value

    @title.setter
    def title(self, new_title):
        SDL_SetWindowTitle(self._win, new_title.encode())

    def destroy(self):
        if self._win:
            SDL_DestroyWindow(self._win)
            self._win = None

    def __del__(self):
        self.destroy()
