# cython: language_level=3str
#

from .sdl2 cimport *

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

    # RENDERER
    cdef Uint32 _SDL_RENDERER_SOFTWARE "SDL_RENDERER_SOFTWARE"
    cdef Uint32 _SDL_RENDERER_ACCELERATED "SDL_RENDERER_ACCELERATED"
    cdef Uint32 _SDL_RENDERER_PRESENTVSYNC "SDL_RENDERER_PRESENTVSYNC"
    cdef Uint32 _SDL_RENDERER_TARGETTEXTURE "SDL_RENDERER_TARGETTEXTURE"

    # https://wiki.libsdl.org/SDL_SetRenderDrawColor
    # https://wiki.libsdl.org/SDL_SetRenderDrawBlendMode
    # https://wiki.libsdl.org/SDL_GetRenderDrawBlendMode
    # https://wiki.libsdl.org/SDL_CreateRenderer
    # https://wiki.libsdl.org/SDL_DestroyRenderer
    # https://wiki.libsdl.org/SDL_RenderClear
    # https://wiki.libsdl.org/SDL_RenderCopy
    # https://wiki.libsdl.org/SDL_RenderCopyEx
    # https://wiki.libsdl.org/SDL_RenderPresent
    int SDL_SetRenderDrawColor(SDL_Renderer* renderer,
                               Uint8         r,
                               Uint8         g,
                               Uint8         b,
                               Uint8         a)
    int SDL_GetRenderDrawBlendMode(SDL_Renderer*   renderer,
                                   SDL_BlendMode* blendMode)
    int SDL_SetRenderDrawBlendMode(SDL_Renderer*  renderer,
                                   SDL_BlendMode blendMode)
    SDL_Renderer* SDL_CreateRenderer(SDL_Window* window,
                                     int         index,
                                     Uint32      flags)
    void SDL_DestroyRenderer(SDL_Renderer* renderer)
    int SDL_RenderClear(SDL_Renderer* renderer)
    int SDL_RenderCopy(SDL_Renderer*   renderer,
                       SDL_Texture*    texture,
                       const SDL_Rect* srcrect,
                       const SDL_Rect* dstrect)
    int SDL_RenderCopyEx(SDL_Renderer*          renderer,
                         SDL_Texture*           texture,
                         const SDL_Rect*        srcrect,
                         const SDL_Rect*        dstrect,
                         const double           angle,
                         const SDL_Point*       center,
                         const SDL_RendererFlip flip)
    void SDL_RenderPresent(SDL_Renderer* renderer)
    # https://wiki.libsdl.org/SDL_RenderGetViewport
    # https://wiki.libsdl.org/SDL_RenderSetViewport
    void SDL_RenderGetViewport(SDL_Renderer* renderer,
                               SDL_Rect*     rect)
    int SDL_RenderSetViewport(SDL_Renderer*   renderer,
                              const SDL_Rect* rect)
    # https://wiki.libsdl.org/SDL_RenderReadPixels
    int SDL_RenderReadPixels(SDL_Renderer*   renderer,
                             const SDL_Rect* rect,
                             Uint32          format,
                             void*           pixels,
                             int             pitch)
    # https://wiki.libsdl.org/SDL_SetRenderTarget
    int SDL_SetRenderTarget(SDL_Renderer* renderer,
                            SDL_Texture*  texture)

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

    # https://wiki.libsdl.org/SDL_ComposeCustomBlendMode
    # https://wiki.libsdl.org/SDL_BlendFactor
    # https://wiki.libsdl.org/SDL_BlendOperation
    SDL_BlendMode SDL_ComposeCustomBlendMode(SDL_BlendFactor    srcColorFactor,
                                             SDL_BlendFactor    dstColorFactor,
                                             SDL_BlendOperation colorOperation,
                                             SDL_BlendFactor    srcAlphaFactor,
                                             SDL_BlendFactor    dstAlphaFactor,
                                             SDL_BlendOperation alphaOperation)
 
    ctypedef enum SDL_BlendOperation:
        SDL_BLENDOPERATION_ADD = 0x00000001,
        SDL_BLENDOPERATION_SUBTRACT = 0x00000002,
        SDL_BLENDOPERATION_REV_SUBTRACT = 0x00000003,
        SDL_BLENDOPERATION_MINIMUM = 0x00000004,
        SDL_BLENDOPERATION_MAXIMUM = 0x00000005

    ctypedef enum SDL_BlendFactor:
        SDL_BLENDFACTOR_ZERO = 0x00000001,
        SDL_BLENDFACTOR_ONE = 0x00000002,
        SDL_BLENDFACTOR_SRC_COLOR = 0x00000003,
        SDL_BLENDFACTOR_ONE_MINUS_SRC_COLOR = 0x00000004,
        SDL_BLENDFACTOR_SRC_ALPHA = 0x00000005,
        SDL_BLENDFACTOR_ONE_MINUS_SRC_ALPHA = 0x00000006,
        SDL_BLENDFACTOR_DST_COLOR = 0x00000007,
        SDL_BLENDFACTOR_ONE_MINUS_DST_COLOR = 0x00000008,
        SDL_BLENDFACTOR_DST_ALPHA = 0x00000009,
        SDL_BLENDFACTOR_ONE_MINUS_DST_ALPHA = 0x0000000A

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
    # https://wiki.libsdl.org/SDL_GetWindowData
    # https://wiki.libsdl.org/SDL_SetWindowData
    void* SDL_GetWindowData(SDL_Window* window,
                            const char* name)
    void* SDL_SetWindowData(SDL_Window* window,
                            const char* name,
                            void*       userdata)
    # https://wiki.libsdl.org/SDL_MaximizeWindow
    # https://wiki.libsdl.org/SDL_MinimizeWindow
    # https://wiki.libsdl.org/SDL_RestoreWindow
    # https://wiki.libsdl.org/SDL_ShowWindow
    # https://wiki.libsdl.org/SDL_HideWindow
    # https://wiki.libsdl.org/SDL_RaiseWindow
    # https://wiki.libsdl.org/SDL_SetWindowInputFocus
    # https://wiki.libsdl.org/SDL_SetWindowResizable
    # https://wiki.libsdl.org/SDL_SetWindowBordered
    # https://wiki.libsdl.org/SDL_SetWindowIcon
    void SDL_MaximizeWindow(SDL_Window* window)
    void SDL_MinimizeWindow(SDL_Window* window)
    void SDL_RestoreWindow(SDL_Window* window)
    void SDL_ShowWindow(SDL_Window* window)
    void SDL_HideWindow(SDL_Window* window)
    void SDL_RaiseWindow(SDL_Window* window)
    int SDL_SetWindowInputFocus(SDL_Window* window)
    void SDL_SetWindowResizable(SDL_Window* window,
                                SDL_bool    resizable)
    void SDL_SetWindowBordered(SDL_Window* window,
                               SDL_bool    bordered)
    void SDL_SetWindowIcon(SDL_Window*  window,
                           SDL_Surface* icon)
    # https://wiki.libsdl.org/SDL_GetWindowFlags
    # https://wiki.libsdl.org/SDL_GetWindowID
    Uint32 SDL_GetWindowFlags(SDL_Window* window)
    Uint32 SDL_GetWindowID(SDL_Window* window)
    # https://wiki.libsdl.org/SDL_GetWindowSize
    # https://wiki.libsdl.org/SDL_SetWindowSize
    # https://wiki.libsdl.org/SDL_GetWindowPosition
    # https://wiki.libsdl.org/SDL_SetWindowPosition
    void SDL_GetWindowSize(SDL_Window* window,
                           int*        w,
                           int*        h)
    void SDL_SetWindowSize(SDL_Window* window,
                           int         w,
                           int         h)
    void SDL_GetWindowPosition(SDL_Window* window,
                               int*        x,
                               int*        y)
    void SDL_SetWindowPosition(SDL_Window* window,
                               int         x,
                               int         y)
    # https://wiki.libsdl.org/SDL_GetWindowOpacity
    # https://wiki.libsdl.org/SDL_SetWindowOpacity
    int SDL_GetWindowOpacity(SDL_Window* window,
                             float*      opacity)
    int SDL_SetWindowOpacity(SDL_Window* window,
                             float       opacity)
    # https://wiki.libsdl.org/SDL_GetWindowBrightness
    # https://wiki.libsdl.org/SDL_SetWindowBrightness
    float SDL_GetWindowBrightness(SDL_Window* window)
    int SDL_SetWindowBrightness(SDL_Window* window,
                                float       brightness)
    # https://wiki.libsdl.org/SDL_GetWindowDisplayIndex
    # https://wiki.libsdl.org/SDL_GetGrabbedWindow
    # https://wiki.libsdl.org/SDL_GetWindowGrab
    # https://wiki.libsdl.org/SDL_SetWindowGrab
    # https://wiki.libsdl.org/SDL_SetWindowFullscreen
    # https://wiki.libsdl.org/SDL_SetWindowModalFor
    int SDL_GetWindowDisplayIndex(SDL_Window* window)
    SDL_Window* SDL_GetGrabbedWindow()
    SDL_bool SDL_GetWindowGrab(SDL_Window* window)
    void SDL_SetWindowGrab(SDL_Window* window,
                           SDL_bool    grabbed)
    int SDL_SetWindowFullscreen(SDL_Window* window,
                                Uint32      flags)
    int SDL_SetWindowModalFor(SDL_Window* modal_window,
                              SDL_Window* parent_window)

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
    # https://wiki.libsdl.org/SDL_GetTextureAlphaMod
    # https://wiki.libsdl.org/SDL_SetTextureAlphaMod
    # https://wiki.libsdl.org/SDL_GetTextureBlendMode
    # https://wiki.libsdl.org/SDL_SetTextureBlendMode
    # https://wiki.libsdl.org/SDL_GetTextureColorMod
    # https://wiki.libsdl.org/SDL_SetTextureColorMod
    SDL_Texture* SDL_CreateTexture(SDL_Renderer* renderer,
                                   Uint32        format,
                                   int           access,
                                   int           w,
                                   int           h)
    SDL_Texture* SDL_CreateTextureFromSurface(SDL_Renderer* renderer,
                                              SDL_Surface*  surface)
    void SDL_DestroyTexture(SDL_Texture* texture)
    # https://wiki.libsdl.org/SDL_TextureAccess
    cdef Uint32 _SDL_TEXTUREACCESS_STATIC "SDL_TEXTUREACCESS_STATIC"
    cdef Uint32 _SDL_TEXTUREACCESS_STREAMING "SDL_TEXTUREACCESS_STREAMING"
    cdef Uint32 _SDL_TEXTUREACCESS_TARGET "SDL_TEXTUREACCESS_TARGET"

    Uint32 SDL_MasksToPixelFormatEnum(int    bpp,
                                      Uint32 Rmask,
                                      Uint32 Gmask,
                                      Uint32 Bmask,
                                      Uint32 Amask)


    int SDL_GetTextureAlphaMod(SDL_Texture* texture,
                               Uint8*       alpha)
    int SDL_SetTextureAlphaMod(SDL_Texture* texture,
                               Uint8        alpha)

    int SDL_GetTextureBlendMode(SDL_Texture*   texture,
                                SDL_BlendMode* blendMode)
    int SDL_SetTextureBlendMode(SDL_Texture*  texture,
                                SDL_BlendMode blendMode)
    int SDL_GetTextureColorMod(SDL_Texture* texture,
                               Uint8*       r,
                               Uint8*       g,
                               Uint8*       b)
    int SDL_SetTextureColorMod(SDL_Texture* texture,
                               Uint8        r,
                               Uint8        g,
                               Uint8        b)

    # https://wiki.libsdl.org/SDL_UpdateTexture
    int SDL_UpdateTexture(SDL_Texture*    texture,
                          const SDL_Rect* rect,
                          const void*     pixels,
                          int             pitch)
    # https://wiki.libsdl.org/SDL_RenderReadPixels
    int SDL_RenderReadPixels(SDL_Renderer*   renderer,
                             const SDL_Rect* rect,
                             Uint32          format,
                             void*           pixels,
                             int             pitch)
    # https://wiki.libsdl.org/SDL_QueryTexture
    int SDL_QueryTexture(SDL_Texture* texture,
                         Uint32*      format,
                         int*         access,
                         int*         w,
                         int*         h)
    # https://wiki.libsdl.org/SDL_GetRenderTarget
    SDL_Texture* SDL_GetRenderTarget(SDL_Renderer* renderer)
    # https://wiki.libsdl.org/SDL_CreateRGBSurfaceWithFormat
    SDL_Surface* SDL_CreateRGBSurfaceWithFormat(Uint32 flags,
                                            int    width,
                                            int    height,
                                            int    depth,
                                            Uint32 format)
    # https://wiki.libsdl.org/SDL_RenderDrawLine
    # https://wiki.libsdl.org/SDL_RenderDrawPoint
    # https://wiki.libsdl.org/SDL_RenderDrawRect
    # https://wiki.libsdl.org/SDL_RenderFillRect
    int SDL_RenderDrawLine(SDL_Renderer* renderer,
                           int x1,
                           int y1,
                           int x2,
                           int y2)
    int SDL_RenderDrawPoint(SDL_Renderer* renderer,
                           int x,
                           int y)

    int SDL_RenderDrawRect(SDL_Renderer* renderer,
                           const SDL_Rect* rect)

    int SDL_RenderFillRect(SDL_Renderer*   renderer,
                           const SDL_Rect* rect)

    # https://wiki.libsdl.org/SDL_RenderSetScale
    # https://wiki.libsdl.org/SDL_RenderGetScale
    # https://wiki.libsdl.org/SDL_RenderSetLogicalSize
    # https://wiki.libsdl.org/SDL_RenderGetLogicalSize
    # https://wiki.libsdl.org/SDL_RenderGetIntegerScale
    int SDL_RenderSetScale(SDL_Renderer* renderer,
                           float         scaleX,
                           float         scaleY) 
    void SDL_RenderGetScale(SDL_Renderer* renderer,
                            float*        scaleX,
                            float*        scaleY)
    int SDL_RenderSetLogicalSize(SDL_Renderer* renderer,
                                 int           w,
                                 int           h)
    void SDL_RenderGetLogicalSize(SDL_Renderer* renderer,
                                  int*          w,
                                  int*          h)
    int SDL_RenderGetIntegerScale(SDL_Renderer* renderer)


cdef extern from "pygame.h" nogil:
    ctypedef class pygame.Color [object pgColorObject]:
        cdef Uint8 data[4]
        cdef Uint8 len

    ctypedef class pygame.Rect [object pgRectObject]:
        cdef SDL_Rect r
        cdef object weakreflist


cdef class Window:
    cdef SDL_Window* _win
    cdef int _is_borrowed

cdef class Renderer:
    cdef SDL_Renderer* _renderer
    cdef Color _draw_color
    cdef Texture _target
    cdef Window _win
    cdef int _is_borrowed

    cpdef object get_viewport(self)
    cpdef object blit(self, object source, Rect dest=*, Rect area=*, int special_flags=*)

cdef class Texture:
    cdef SDL_Texture* _tex
    cdef Color _color
    cdef readonly Renderer renderer
    cdef readonly int width
    cdef readonly int height

    cdef draw_internal(self, SDL_Rect *csrcrect, SDL_Rect *cdstrect, float angle=*, SDL_Point *originptr=*,
                       bint flip_x=*, bint flip_y=*)
    cpdef void draw(self, srcrect=*, dstrect=*, float angle=*, origin=*,
                    bint flip_x=*, bint flip_y=*)

cdef class Image:
    cdef Color _color
    cdef public float angle
    cdef SDL_Point _origin
    cdef SDL_Point* _originptr
    cdef public bint flip_x
    cdef public bint flip_y
    cdef public float alpha
    cdef public SDL_BlendMode blend_mode

    cdef public Texture texture
    cdef public Rect srcrect

    cpdef void draw(self, srcrect=*, dstrect=*)
