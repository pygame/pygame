from ._sdl2 import *


FULLSCREEN = 0x00000001
FULLSCREEN_DESKTOP = FULLSCREEN | 0x00001000
OPENGL = 0x00000002
SHOWN = 0x00000004
HIDDEN = 0x00000008
BORDERLESS = 0x00000010
RESIZABLE = 0x00000020
MINIMIZED = 0x00000040
MAXIMIZED = 0x00000080
INPUT_GRABBED = 0x00000100
INPUT_FOCUS = 0x00000200
MOUSE_FOCUS = 0x00000400
FOREIGN = 0x00000800
ALLOW_HIGHDPI = 0x00002000
MOUSE_CAPTURE = 0x00004000
ALWAYS_ON_TOP = 0x00008000
SKIP_TASKBAR = 0x00010000
UTILITY = 0x00020000
TOOLTIP = 0x00040000
POPUP_MENU = 0x00080000
VULKAN = 0x10000000

class Window:
    POSITION_UNDEFINED = SDL_WINDOWPOS_UNDEFINED
    POSITION_CENTERED = SDL_WINDOWPOS_CENTERED
    DEFAULT_WIDTH = 640
    DEFAULT_HEIGHT = 480

    def __init__(self, title='pygame', x=POSITION_UNDEFINED, y=POSITION_UNDEFINED, w=DEFAULT_WIDTH, h=DEFAULT_HEIGHT, flags=0):
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
