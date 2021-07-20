# cython: language_level=2
#

from sdl2 cimport *

cdef extern from "SDL.h" nogil:
    ctypedef struct SDL_Window
    ctypedef struct SDL_Texture
    ctypedef struct SDL_Renderer
    ctypedef struct SDL_Rect:
        int x, y
        int w, h

    ctypedef enum SDL_PixelFormatEnum:
        SDL_PIXELFORMAT_UNKNOWN

    int SDL_BITSPERPIXEL(Uint32 format)

    ctypedef struct SDL_PixelFormat:
        Uint32 format

    ctypedef struct SDL_Surface:
        Uint32 flags
        SDL_PixelFormat *format
        int w,h
        int pitch
        void* pixels
        void *userdata
        int locked
        void *lock_data
        SDL_Rect clip_rect

    ctypedef struct SDL_Point:
        int x, y
    ctypedef enum SDL_RendererFlip:
        SDL_FLIP_NONE,
        SDL_FLIP_HORIZONTAL,
        SDL_FLIP_VERTICAL
    ctypedef enum SDL_BlendMode:
        SDL_BLENDMODE_NONE = 0x00000000,
        SDL_BLENDMODE_BLEND = 0x00000001,
        SDL_BLENDMODE_ADD = 0x00000002,
        SDL_BLENDMODE_MOD = 0x00000004,
        SDL_BLENDMODE_INVALID = 0x7FFFFFFF

    # https://wiki.libsdl.org/SDL_MessageBoxData
    # https://wiki.libsdl.org/SDL_ShowMessageBox
    cdef Uint32 _SDL_MESSAGEBOX_ERROR "SDL_MESSAGEBOX_ERROR"
    cdef Uint32 _SDL_MESSAGEBOX_WARNING "SDL_MESSAGEBOX_WARNING"
    cdef Uint32 _SDL_MESSAGEBOX_INFORMATION "SDL_MESSAGEBOX_INFORMATION"

    cdef Uint32 _SDL_MESSAGEBOX_BUTTON_RETURNKEY_DEFAULT "SDL_MESSAGEBOX_BUTTON_RETURNKEY_DEFAULT"
    cdef Uint32 _SDL_MESSAGEBOX_BUTTON_ESCAPEKEY_DEFAULT "SDL_MESSAGEBOX_BUTTON_ESCAPEKEY_DEFAULT"

    ctypedef struct SDL_MessageBoxData:
        Uint32 flags
        SDL_Window* window
        const char* title
        const char* message
        int numbuttons
        const SDL_MessageBoxButtonData* buttons
        const SDL_MessageBoxColorScheme* colorScheme
    ctypedef struct SDL_MessageBoxButtonData:
        Uint32 flags
        int buttonid
        const char *text
    ctypedef struct SDL_MessageBoxColorScheme
    int SDL_ShowMessageBox(const SDL_MessageBoxData* messageboxdata,
                           int*                      buttonid)

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