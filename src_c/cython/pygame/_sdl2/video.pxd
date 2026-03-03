# cython: language_level=3str
#

from .sdl2 cimport *

cdef extern from "pgcompat_rect.h" nogil:
    ctypedef struct SDL_FRect:
        float x, y
        float w, h


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
    ctypedef struct SDL_Color:
        Uint8 r, g, b, a

    SDL_bool SDL_SetHint(const char *name, const char *value)
    char* SDL_HINT_RENDER_SCALE_QUALITY b'SDL_HINT_RENDER_SCALE_QUALITY'

    cdef extern from *:
        """
        #if SDL_VERSION_ATLEAST(2, 0, 18)
            typedef SDL_FPoint _pgsdlFPoint;
            typedef SDL_Vertex _pgsdlVertex;
        #else
            typedef struct {
                float x;
                float y;
            } _pgsdlFPoint;

            typedef struct {
                _pgsdlFPoint position;
                SDL_Color color;
                _pgsdlFPoint tex_coord;
            } _pgsdlVertex;

            int SDL_RenderGeometry(SDL_Renderer *renderer, SDL_Texture *texture,
                  const _pgsdlVertex *vertices, int num_vertices,
                  const int *indices, int num_indices)
            {
                SDL_SetError("Needs SDL 2.0.18");
                return -1;
            }
        #endif

        #if SDL_VERSION_ATLEAST(2,0,12)
            typedef SDL_ScaleMode _pgsdlScaleMode;
        #else
            typedef enum {
                SDL_ScaleModeNearest,
                SDL_ScaleModeLinear,
                SDL_ScaleModeBest,
            } _pgsdlScaleMode;

            int SDL_SetTextureScaleMode(SDL_Texture * texture, _pgsdlScaleMode scaleMode){
                return 0;
            }
        #endif
        """
    ctypedef struct SDL_FPoint "_pgsdlFPoint":
        float x, y
    ctypedef struct SDL_Vertex "_pgsdlVertex":
        SDL_FPoint position
        SDL_Color color
        SDL_FPoint tex_coord

    ctypedef enum SDL_ScaleMode "_pgsdlScaleMode":
        SDL_ScaleModeNearest,
        SDL_ScaleModeLinear,
        SDL_ScaleModeBest

    int SDL_SetTextureScaleMode(SDL_Texture * texture, SDL_ScaleMode scaleMode)

    ctypedef enum SDL_RendererFlip:
        SDL_FLIP_NONE,
        SDL_FLIP_HORIZONTAL,
        SDL_FLIP_VERTICAL
    ctypedef enum SDL_BlendMode:
        SDL_BLENDMODE_NONE = 0x00000000,
        SDL_BLENDMODE_BLEND = 0x00000001,
        SDL_BLENDMODE_ADD = 0x00000002,
        SDL_BLENDMODE_MOD = 0x00000004,
        SDL_BLENDMODE_MUL = 0x00000008,
        SDL_BLENDMODE_INVALID = 0x7FFFFFFF

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
    # https://wiki.libsdl.org/SDL_RenderCopyExF
    # https://wiki.libsdl.org/SDL_RenderPresent
    int SDL_GetRenderDrawColor(SDL_Renderer* renderer,
                               Uint8*        r,
                               Uint8*        g,
                               Uint8*        b,
                               Uint8*        a)
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
    int SDL_RenderCopyExF(SDL_Renderer*          renderer,
                          SDL_Texture*           texture,
                          const SDL_Rect*        srcrect,
                          const SDL_FRect*       dstrect,
                          const double           angle,
                          const SDL_FPoint*      center,
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
    # https://wiki.libsdl.org/SDL_GetWindowData
    # https://wiki.libsdl.org/SDL_GetGrabbedWindow
    void* SDL_GetWindowData(SDL_Window* window,
                            const char* name)
    SDL_Window* SDL_GetGrabbedWindow()

    cdef int _SDL_WINDOWPOS_UNDEFINED "SDL_WINDOWPOS_UNDEFINED"
    cdef int _SDL_WINDOWPOS_CENTERED "SDL_WINDOWPOS_CENTERED"

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
    # https://wiki.libsdl.org/SDL_RenderDrawLineF
    # https://wiki.libsdl.org/SDL_RenderDrawLines
    # https://wiki.libsdl.org/SDL_RenderDrawLinesF
    # https://wiki.libsdl.org/SDL_RenderDrawPoint
    # https://wiki.libsdl.org/SDL_RenderDrawPointF
    # https://wiki.libsdl.org/SDL_RenderDrawRect
    # https://wiki.libsdl.org/SDL_RenderDrawRectF
    # https://wiki.libsdl.org/SDL_RenderFillRect
    # https://wiki.libsdl.org/SDL_RenderFillRectF
    # https://wiki.libsdl.org/SDL_RenderGeometry
    int SDL_RenderDrawLineF(SDL_Renderer* renderer,
                           float x1,
                           float y1,
                           float x2,
                           float y2)
    int SDL_RenderDrawLinesF(SDL_Renderer* renderer,
                            const SDL_FPoint* points,
                            int count)
    int SDL_RenderDrawPointF(SDL_Renderer* renderer,
                           float x,
                           float y)
    int SDL_RenderDrawRectF(SDL_Renderer* renderer,
                           const SDL_FRect* rect)
    int SDL_RenderFillRectF(SDL_Renderer*   renderer,
                           const SDL_FRect* rect)
    int SDL_RenderGeometry(SDL_Renderer* renderer,
                           SDL_Texture* texture,
                           const SDL_Vertex* vertices,
                           int num_vertices,
                           const int* indices,
                           int num_indices)

    # https://wiki.libsdl.org/SDL_RenderSetScale
    # https://wiki.libsdl.org/SDL_RenderGetScale
    # https://wiki.libsdl.org/SDL_RenderSetLogicalSize
    # https://wiki.libsdl.org/SDL_RenderGetLogicalSize
    # https://wiki.libsdl.org/SDL_RenderGetIntegerScale
    # https://wiki.libsdl.org/SDL2/SDL_RenderLogicalToWindow
    # https://wiki.libsdl.org/SDL2/SDL_RenderWindowToLogical
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
    # Note: Must be changed to SDL_RenderCoordinatesToWindow for SDL3
    # https://wiki.libsdl.org/SDL3/SDL_RenderCoordinatesToWindow
    void SDL_RenderLogicalToWindow(SDL_Renderer* renderer,
                                   float lx,
                                   float ly,
                                   int *wx,
                                   int *wy);
    # Note: Must be changed to SDL_RenderCoordinatesFromWindow for SDL3
    # https://wiki.libsdl.org/SDL3/SDL_RenderCoordinatesFromWindow
    void SDL_RenderWindowToLogical(SDL_Renderer* renderer,
                                   int wx,
                                   int wy,
                                   float *lx,
                                   float *ly);

    int SDL_VERSION_ATLEAST(int major, int minor, int patch)

    # https://wiki.libsdl.org/SDL_GetWindowPixelFormat
    # https://wiki.libsdl.org/SDL_IntersectRect
    # https://wiki.libsdl.org/SDL_SetRelativeMouseMode
    # https://wiki.libsdl.org/SDL_GetRelativeMouseMode
    # https://wiki.libsdl.org/SDL_GetRenderer
    # https://wiki.libsdl.org/SDL_ConvertSurface
    # https://wiki.libsdl.org/SDL_FreeSurface
    # https://wiki.libsdl.org/SDL_AllocFormat
    # https://wiki.libsdl.org/SDL_FreeFormat
    # https://wiki.libsdl.org/SDL_SetSurfaceBlendMode
    # https://wiki.libsdl.org/SDL_GetSurfaceBlendMode
    Uint32 SDL_GetWindowPixelFormat(SDL_Window* window)
    SDL_bool SDL_IntersectRect(const SDL_Rect* A,
                               const SDL_Rect* B,
                               SDL_Rect*       result)
    int SDL_SetRelativeMouseMode(SDL_bool enabled)
    SDL_bool SDL_GetRelativeMouseMode()
    SDL_Renderer* SDL_GetRenderer(SDL_Window* window)
    SDL_Surface* SDL_ConvertSurface(SDL_Surface * src, const SDL_PixelFormat * fmt, Uint32 flags)
    void SDL_FreeSurface(SDL_Surface * surface)
    SDL_PixelFormat * SDL_AllocFormat(Uint32 pixel_format)
    void SDL_FreeFormat(SDL_PixelFormat *format)
    int SDL_SetSurfaceBlendMode(SDL_Surface * surface, SDL_BlendMode blendMode)
    int SDL_GetSurfaceBlendMode(SDL_Surface * surface, SDL_BlendMode *blendMode)


cdef extern from "pygame.h" nogil:
    ctypedef class pygame.color.Color [object pgColorObject]:
        cdef Uint8 data[4]
        cdef Uint8 len

    ctypedef enum pgColorHandleFlags:
        PG_COLOR_HANDLE_SIMPLE
        PG_COLOR_HANDLE_STR
        PG_COLOR_HANDLE_INT
        PG_COLOR_HANDLE_RESTRICT_SEQ
        PG_COLOR_HANDLE_ALL

    ctypedef class pygame.rect.Rect [object pgRectObject]:
        cdef SDL_Rect r
        cdef object weakreflist

    ctypedef class pygame.window.Window [object pgWindowObject]:
        cdef SDL_Window *_win
        cdef SDL_bool _is_borrowed

    ctypedef struct pgSurfaceObject

    int pgSurface_Check(object surf)
    SDL_Surface* pgSurface_AsSurface(object surf)
    void import_pygame_surface()

    SDL_Window* pg_GetDefaultWindow()
    void import_pygame_base()

    int pgRect_Check(object rect)
    SDL_Rect *pgRect_FromObject(object obj, SDL_Rect *temp)
    object pgRect_New(SDL_Rect *r)
    object pgRect_New4(int x, int y, int w, int h)
    int pgFRect_Check(object rect)
    SDL_FRect *pgFRect_FromObject(object obj, SDL_FRect *temp)
    object pgFRect_New(SDL_FRect *r)
    object pgFRect_New4(float x, float y, float w, float h)
    void import_pygame_rect()

    int pg_RGBAFromObjEx(object color, Uint8 rgba[], pgColorHandleFlags handle_flags) except 0
    object pgColor_NewLength(Uint8 rgba[], Uint8 length)
    void import_pygame_color()
    pgSurfaceObject *pgSurface_New2(SDL_Surface *info, int owner)

    int pgWindow_Check(object win)
    void import_pygame_window()

cdef class Renderer:
    cdef SDL_Renderer* _renderer
    cdef Texture _target
    cdef Window _win
    cdef int _is_borrowed

    cpdef object get_viewport(self)
    cpdef object blit(self, object source, Rect dest=*, Rect area=*, int special_flags=*)

cdef class Texture:
    cdef SDL_Texture* _tex
    cdef readonly Renderer renderer
    cdef readonly int width
    cdef readonly int height

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
