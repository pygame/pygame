# cython: language_level=2
#

from sdl2 cimport *

cdef extern from "SDL.h" nogil:
    ctypedef struct SDL_Window
    ctypedef struct SDL_Texture
    ctypedef struct SDL_Renderer
    ctypedef struct SDL_Surface
    ctypedef struct SDL_Rect:
        int x, y
        int w, h

    # RENDERER
    cdef Uint32 _SDL_RENDERER_SOFTWARE "SDL_RENDERER_SOFTWARE"
    cdef Uint32 _SDL_RENDERER_ACCELERATED "SDL_RENDERER_ACCELERATED"
    cdef Uint32 _SDL_RENDERER_PRESENTVSYNC "SDL_RENDERER_PRESENTVSYNC"
    cdef Uint32 _SDL_RENDERER_TARGETTEXTURE "SDL_RENDERER_TARGETTEXTURE"

    # https://wiki.libsdl.org/SDL_SetRenderDrawColor
    # https://wiki.libsdl.org/SDL_CreateRenderer
    # https://wiki.libsdl.org/SDL_DestroyRenderer
    # https://wiki.libsdl.org/SDL_RenderClear
    # https://wiki.libsdl.org/SDL_RenderCopy
    # https://wiki.libsdl.org/SDL_RenderPresent
    int SDL_SetRenderDrawColor(SDL_Renderer* renderer,
                               Uint8         r,
                               Uint8         g,
                               Uint8         b,
                               Uint8         a)
    SDL_Renderer* SDL_CreateRenderer(SDL_Window* window,
                                     int         index,
                                     Uint32      flags)
    void SDL_DestroyRenderer(SDL_Renderer* renderer)
    int SDL_RenderClear(SDL_Renderer* renderer)
    int SDL_RenderCopy(SDL_Renderer*   renderer,
                       SDL_Texture*    texture,
                       const SDL_Rect* srcrect,
                       const SDL_Rect* dstrect)
    void SDL_RenderPresent(SDL_Renderer* renderer)

    # https://wiki.libsdl.org/SDL_RendererInfo
    ctypedef struct SDL_RendererInfo:
        const char *name
        Uint32 flags
        Uint32 num_texture_formats
        Uint32[16] texture_formats
        int max_texture_width
        int max_texture_height
    # https://wiki.libsdl.org/SDL_GetNumRenderDrivers
    int SDL_GetNumRenderDrivers()
    # https://wiki.libsdl.org/SDL_GetRenderDriverInfo
    int SDL_GetRenderDriverInfo(int               index,
                                SDL_RendererInfo* info)

    # WINDOW
    # https://wiki.libsdl.org/SDL_CreateWindow
    # https://wiki.libsdl.org/SDL_DestroyWindow
    # https://wiki.libsdl.org/SDL_GetWindowTitle
    # https://wiki.libsdl.org/SDL_SetWindowTitle
    SDL_Window* SDL_CreateWindow(const char* title,
                                 int         x,
                                 int         y,
                                 int         w,
                                 int         h,
                                 Uint32      flags)
    void SDL_DestroyWindow(SDL_Window *window)
    const char* SDL_GetWindowTitle(SDL_Window* window)
    void SDL_SetWindowTitle(SDL_Window* window,
                            const char* title)

    cdef int _SDL_WINDOWPOS_UNDEFINED "SDL_WINDOWPOS_UNDEFINED"
    cdef int _SDL_WINDOWPOS_CENTERED "SDL_WINDOWPOS_CENTERED"
    cdef Uint32 _SDL_WINDOW_FULLSCREEN "SDL_WINDOW_FULLSCREEN"
    cdef Uint32 _SDL_WINDOW_FULLSCREEN_DESKTOP "SDL_WINDOW_FULLSCREEN_DESKTOP"
    cdef Uint32 _SDL_WINDOW_OPENGL "SDL_WINDOW_OPENGL"
    cdef Uint32 _SDL_WINDOW_SHOWN "SDL_WINDOW_SHOWN"
    cdef Uint32 _SDL_WINDOW_HIDDEN "SDL_WINDOW_HIDDEN"
    cdef Uint32 _SDL_WINDOW_BORDERLESS "SDL_WINDOW_BORDERLESS"
    cdef Uint32 _SDL_WINDOW_RESIZABLE "SDL_WINDOW_RESIZABLE"
    cdef Uint32 _SDL_WINDOW_MINIMIZED "SDL_WINDOW_MINIMIZED"
    cdef Uint32 _SDL_WINDOW_MAXIMIZED "SDL_WINDOW_MAXIMIZED"
    cdef Uint32 _SDL_WINDOW_INPUT_GRABBED "SDL_WINDOW_INPUT_GRABBED"
    cdef Uint32 _SDL_WINDOW_INPUT_FOCUS "SDL_WINDOW_INPUT_FOCUS"
    cdef Uint32 _SDL_WINDOW_MOUSE_FOCUS "SDL_WINDOW_MOUSE_FOCUS"
    cdef Uint32 _SDL_WINDOW_FOREIGN "SDL_WINDOW_FOREIGN"
    cdef Uint32 _SDL_WINDOW_ALLOW_HIGHDPI "SDL_WINDOW_ALLOW_HIGHDPI"
    cdef Uint32 _SDL_WINDOW_MOUSE_CAPTURE "SDL_WINDOW_MOUSE_CAPTURE"
    cdef Uint32 _SDL_WINDOW_ALWAYS_ON_TOP "SDL_WINDOW_ALWAYS_ON_TOP"
    cdef Uint32 _SDL_WINDOW_SKIP_TASKBAR "SDL_WINDOW_SKIP_TASKBAR"
    cdef Uint32 _SDL_WINDOW_UTILITY "SDL_WINDOW_UTILITY"
    cdef Uint32 _SDL_WINDOW_TOOLTIP "SDL_WINDOW_TOOLTIP"
    cdef Uint32 _SDL_WINDOW_POPUP_MENU "SDL_WINDOW_POPUP_MENU"
    cdef Uint32 _SDL_WINDOW_VULKAN "SDL_WINDOW_VULKAN"

    # TEXTURE
    # https://wiki.libsdl.org/SDL_CreateTexture
    # https://wiki.libsdl.org/SDL_CreateTextureFromSurface
    # https://wiki.libsdl.org/SDL_DestroyTexture
    SDL_Texture* SDL_CreateTexture(SDL_Renderer* renderer,
                                   Uint32        format,
                                   int           access,
                                   int           w,
                                   int           h)
    SDL_Texture* SDL_CreateTextureFromSurface(SDL_Renderer* renderer,
                                              SDL_Surface*  surface)
    void SDL_DestroyTexture(SDL_Texture* texture)

cdef class Window:
    cdef SDL_Window* _win

cdef class Renderer:
    cdef SDL_Renderer* _renderer
    cdef tuple _draw_color

cdef class Texture:
    cdef SDL_Texture* _tex
    cdef readonly Renderer renderer
    cdef readonly int width
    cdef readonly int height
