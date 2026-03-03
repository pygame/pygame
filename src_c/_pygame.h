/*
    pygame - Python Game Library
    Copyright (C) 2000-2001  Pete Shinners

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Library General Public License for more details.

    You should have received a copy of the GNU Library General Public
    License along with this library; if not, write to the Free
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Pete Shinners
    pete@shinners.org
*/

/* This will use PYGAMEAPI_EXTERN_SLOTS instead
 * of PYGAMEAPI_DEFINE_SLOTS for base modules.
 */
#ifndef _PYGAME_INTERNAL_H
#define _PYGAME_INTERNAL_H

#include "pgplatform.h"
/*
    If PY_SSIZE_T_CLEAN is defined before including Python.h, length is a
    Py_ssize_t rather than an int for all # variants of formats (s#, y#, etc.)
*/
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "include/pythoncapi_compat.h"

/* Ensure PyPy-specific code is not in use when running on GraalPython (PR
 * #2580) */
#if defined(GRAALVM_PYTHON) && defined(PYPY_VERSION)
#undef PYPY_VERSION
#endif

#ifdef PG_SDL3
#include <SDL3/SDL.h>
#else
#include <SDL.h>
#endif

#include "stdbool.h"

#if SDL_VERSION_ATLEAST(3, 0, 0)

#include "include/SDL_gesture.h"

#define SDL_DOLLARGESTURE GESTURE_DOLLARGESTURE
#define SDL_DOLLARRECORD GESTURE_DOLLARRECORD
#define SDL_MULTIGESTURE GESTURE_MULTIGESTURE

#define PG_ShowCursor SDL_ShowCursor
#define PG_HideCursor SDL_HideCursor
#define PG_CursorVisible SDL_CursorVisible

#define PG_INIT_NOPARACHUTE 0

// UINT16 audio no longer exists in SDL3
#define PG_AUDIO_U16LSB 0
#define PG_AUDIO_U16MSB 0
#define PG_AUDIO_U16SYS 0
#define PG_AUDIO_U16 0

// Allowed changes no longer exists, your request gets emulated if unavailable
#define PG_AUDIO_ALLOW_FREQUENCY_CHANGE 0
#define PG_AUDIO_ALLOW_FORMAT_CHANGE 0
#define PG_AUDIO_ALLOW_CHANNELS_CHANGE 0
#define PG_AUDIO_ALLOW_ANY_CHANGE 0

#define PG_CreateSurface SDL_CreateSurface
#define PG_CreateSurfaceFrom SDL_CreateSurfaceFrom
#define PG_ConvertSurface SDL_ConvertSurface
#define PG_ConvertSurfaceFormat SDL_ConvertSurface

#define PG_PixelFormatEnum SDL_PixelFormat

#define PG_SurfaceHasRLE SDL_SurfaceHasRLE
#define PG_SetSurfaceRLE SDL_SetSurfaceRLE

#define PG_SoftStretchNearest(src, srcrect, dst, dstrect) \
    SDL_StretchSurface(src, srcrect, dst, dstrect, SDL_SCALEMODE_NEAREST)

#define PG_UpdateWindowSurface SDL_UpdateWindowSurface

/* Emulating SDL2 SDL_LockMutex API. In SDL3, it returns void. */
static inline int
PG_LockMutex(SDL_mutex *mutex)
{
    SDL_LockMutex(mutex);
    return 0;
}

/* Emulating SDL2 SDL_UnlockMutex API. In SDL3, it returns void. */
static inline int
PG_UnlockMutex(SDL_mutex *mutex)
{
    SDL_UnlockMutex(mutex);
    return 0;
}

// Implementation from SDL_GetMasksForPixelFormat, which is used by
// SDL_InitPixelFormatDetails in SDL_pixels.c
// Created to match surf->format->BitsPerPixel in SDL2,
// details->bits_per_pixel in SDL3.
static inline int
PG_SURF_BitsPerPixel(SDL_Surface *surf)
{
    if (SDL_ISPIXELFORMAT_FOURCC(surf->format)) {
        // however, some of these are packed formats, and can legit declare
        // bits-per-pixel!
        switch (surf->format) {
            case SDL_PIXELFORMAT_YUY2:
            case SDL_PIXELFORMAT_UYVY:
            case SDL_PIXELFORMAT_YVYU:
                return 32;
            default:
                return 0;  // oh well.
        }
    }

    if (SDL_BYTESPERPIXEL(surf->format) <= 2) {
        return SDL_BITSPERPIXEL(surf->format);
    }
    return SDL_BYTESPERPIXEL(surf->format) * 8;
}

#define PG_SURF_BytesPerPixel(surf) SDL_BYTESPERPIXEL(surf->format)
#define PG_FORMAT_BitsPerPixel(format) format->bits_per_pixel
#define PG_FORMAT_BytesPerPixel(format) format->bytes_per_pixel
#define PG_SURF_FORMATENUM(surf) surf->format

#define PG_FORMAT_R_LOSS(format) (8 - format->Rbits)
#define PG_FORMAT_G_LOSS(format) (8 - format->Gbits)
#define PG_FORMAT_B_LOSS(format) (8 - format->Bbits)
#define PG_FORMAT_A_LOSS(format) (8 - format->Abits)

#define PG_PixelFormat const SDL_PixelFormatDetails

static inline bool
PG_GetSurfaceDetails(SDL_Surface *surf, PG_PixelFormat **format_p,
                     SDL_Palette **palette_p)
{
    *palette_p = SDL_GetSurfacePalette(surf);
    *format_p = SDL_GetPixelFormatDetails(surf->format);
    return *format_p != NULL;
}

static inline PG_PixelFormat *
PG_GetSurfaceFormat(SDL_Surface *surf)
{
    return SDL_GetPixelFormatDetails(surf->format);
}

#define PG_GetSurfacePalette SDL_GetSurfacePalette
#define PG_SetPaletteColors SDL_SetPaletteColors
#define PG_SetSurfacePalette SDL_SetSurfacePalette
#define PG_SetSurfaceColorKey SDL_SetSurfaceColorKey
#define PG_SetSurfaceBlendMode SDL_SetSurfaceBlendMode
#define PG_GetSurfaceBlendMode SDL_GetSurfaceBlendMode
#define PG_GetSurfaceAlphaMod SDL_GetSurfaceAlphaMod
#define PG_SetSurfaceAlphaMod SDL_SetSurfaceAlphaMod
#define PG_FillSurfaceRect SDL_FillSurfaceRect
#define PG_LockSurface SDL_LockSurface

#define PG_GetRGBA SDL_GetRGBA
#define PG_GetRGB SDL_GetRGB
#define PG_MapRGBA SDL_MapRGBA
#define PG_MapRGB SDL_MapRGB

/* Mask to test if surface flags are in a fullscreen window. */
#define PG_WINDOW_FULLSCREEN_INCLUSIVE SDL_WINDOW_FULLSCREEN

#define PG_SetEventEnabled(type, enabled) SDL_SetEventEnabled(type, enabled)
#define PG_EventEnabled(type) SDL_EventEnabled(type)
#define PG_SetJoystickEventsEnabled(enabled) \
    SDL_SetJoystickEventsEnabled(enabled)
#define PG_InitSubSystem(flags) SDL_InitSubSystem(flags)

#define PG_FIND_VNUM_MAJOR(ver) SDL_VERSIONNUM_MAJOR(ver)
#define PG_FIND_VNUM_MINOR(ver) SDL_VERSIONNUM_MINOR(ver)
#define PG_FIND_VNUM_MICRO(ver) SDL_VERSIONNUM_MICRO(ver)

#define PG_INIT_TIMER 0

#define PG_GetSurfaceClipRect SDL_GetSurfaceClipRect

#define PG_GL_SetSwapInterval SDL_GL_SetSwapInterval

#else /* ~SDL_VERSION_ATLEAST(3, 0, 0)*/
#define PG_ShowCursor() SDL_ShowCursor(SDL_ENABLE)
#define PG_HideCursor() SDL_ShowCursor(SDL_DISABLE)
#define PG_CursorVisible() SDL_ShowCursor(SDL_QUERY)

#define PG_INIT_NOPARACHUTE SDL_INIT_NOPARACHUTE

#define PG_AUDIO_U16LSB AUDIO_U16LSB
#define PG_AUDIO_U16MSB AUDIO_U16MSB
#define PG_AUDIO_U16SYS AUDIO_U16SYS
#define PG_AUDIO_U16 AUDIO_U16

#define PG_AUDIO_ALLOW_FREQUENCY_CHANGE SDL_AUDIO_ALLOW_FREQUENCY_CHANGE
#define PG_AUDIO_ALLOW_FORMAT_CHANGE SDL_AUDIO_ALLOW_FORMAT_CHANGE
#define PG_AUDIO_ALLOW_CHANNELS_CHANGE SDL_AUDIO_ALLOW_CHANNELS_CHANGE
#define PG_AUDIO_ALLOW_ANY_CHANGE SDL_AUDIO_ALLOW_ANY_CHANGE

#define PG_CreateSurface(width, height, format) \
    SDL_CreateRGBSurfaceWithFormat(0, width, height, 0, format)
#define PG_CreateSurfaceFrom(width, height, format, pixels, pitch) \
    SDL_CreateRGBSurfaceWithFormatFrom(pixels, width, height, 0, pitch, format)
#define PG_ConvertSurface(src, fmt) SDL_ConvertSurface(src, fmt, 0)
#define PG_ConvertSurfaceFormat(src, pixel_format) \
    SDL_ConvertSurfaceFormat(src, pixel_format, 0)

#define PG_PixelFormatEnum SDL_PixelFormatEnum

#define PG_SoftStretchNearest(src, srcrect, dst, dstrect) \
    SDL_SoftStretch(src, srcrect, dst, dstrect)

static inline bool
PG_UpdateWindowSurface(SDL_Window *window)
{
    return SDL_UpdateWindowSurface(window) == 0;
}

static inline int
PG_LockMutex(SDL_mutex *mutex)
{
    return SDL_LockMutex(mutex);
}

static inline int
PG_UnlockMutex(SDL_mutex *mutex)
{
    return SDL_UnlockMutex(mutex);
}

#define PG_SURF_BitsPerPixel(surf) surf->format->BitsPerPixel
#define PG_SURF_BytesPerPixel(surf) surf->format->BytesPerPixel
#define PG_FORMAT_BitsPerPixel(format) format->BitsPerPixel
#define PG_FORMAT_BytesPerPixel(format) format->BytesPerPixel
#define PG_SURF_FORMATENUM(surf) surf->format->format

#define PG_FORMAT_R_LOSS(format) format->Rloss
#define PG_FORMAT_G_LOSS(format) format->Gloss
#define PG_FORMAT_B_LOSS(format) format->Bloss
#define PG_FORMAT_A_LOSS(format) format->Aloss

#define PG_PixelFormat SDL_PixelFormat

static inline bool
PG_GetSurfaceDetails(SDL_Surface *surf, PG_PixelFormat **format_p,
                     SDL_Palette **palette_p)
{
    *format_p = surf->format;
    *palette_p = surf->format->palette;
    return true;
}

static inline PG_PixelFormat *
PG_GetSurfaceFormat(SDL_Surface *surf)
{
    return surf->format;
}

static inline SDL_Palette *
PG_GetSurfacePalette(SDL_Surface *surf)
{
    return surf->format->palette;
}

static inline bool
PG_SetPaletteColors(SDL_Palette *palette, const SDL_Color *colors,
                    int firstcolor, int ncolors)
{
    return SDL_SetPaletteColors(palette, colors, firstcolor, ncolors) == 0;
}

static inline bool
PG_SetSurfacePalette(SDL_Surface *surface, SDL_Palette *palette)
{
    return SDL_SetSurfacePalette(surface, palette) == 0;
}

static inline bool
PG_SetSurfaceColorKey(SDL_Surface *surface, bool enabled, Uint32 key)
{
    return SDL_SetColorKey(surface, enabled, key) == 0;
}

static inline bool
PG_SetSurfaceBlendMode(SDL_Surface *surface, SDL_BlendMode blendMode)
{
    return SDL_SetSurfaceBlendMode(surface, blendMode) == 0;
}

static inline bool
PG_GetSurfaceBlendMode(SDL_Surface *surface, SDL_BlendMode *blendMode)
{
    return SDL_GetSurfaceBlendMode(surface, blendMode) == 0;
}

static inline bool
PG_GetSurfaceAlphaMod(SDL_Surface *surface, Uint8 *alpha)
{
    return SDL_GetSurfaceAlphaMod(surface, alpha) == 0;
}

static inline bool
PG_SetSurfaceAlphaMod(SDL_Surface *surface, Uint8 alpha)
{
    return SDL_SetSurfaceAlphaMod(surface, alpha) == 0;
}

static inline bool
PG_FillSurfaceRect(SDL_Surface *dst, const SDL_Rect *rect, Uint32 color)
{
    return SDL_FillRect(dst, rect, color) == 0;
}

static inline bool
PG_LockSurface(SDL_Surface *surface)
{
    return SDL_LockSurface(surface) == 0;
}

// NOTE:
// palette is part of the format in SDL2, so these functions below have it
// as a separate parameter to be consistent with the SDL3 signature.
// They are ignoring the palette parameter, but not the palette data.

static inline void
PG_GetRGBA(Uint32 pixel, PG_PixelFormat *format, const SDL_Palette *palette,
           Uint8 *r, Uint8 *g, Uint8 *b, Uint8 *a)
{
    SDL_GetRGBA(pixel, format, r, g, b, a);
}

static inline void
PG_GetRGB(Uint32 pixel, PG_PixelFormat *format, const SDL_Palette *palette,
          Uint8 *r, Uint8 *g, Uint8 *b)
{
    SDL_GetRGB(pixel, format, r, g, b);
}

static inline Uint32
PG_MapRGBA(PG_PixelFormat *format, const SDL_Palette *palette, Uint8 r,
           Uint8 g, Uint8 b, Uint8 a)
{
    return SDL_MapRGBA(format, r, g, b, a);
}

static inline Uint32
PG_MapRGB(PG_PixelFormat *format, const SDL_Palette *palette, Uint8 r, Uint8 g,
          Uint8 b)
{
    return SDL_MapRGB(format, r, g, b);
}

static inline bool
PG_InitSubSystem(Uint32 flags)
{
    return SDL_InitSubSystem(flags) == 0;
}

/* Mask to test if surface flags are in a fullscreen window.
 * SDL_WINDOW_FULLSCREEN_DESKTOP works here because it also contains
 * SDL_WINDOW_FULLSCREEN. */
#define PG_WINDOW_FULLSCREEN_INCLUSIVE SDL_WINDOW_FULLSCREEN_DESKTOP

/* SDL_EventState is meant to take SDL_IGNORE or SDL_ENABLE, but it also
 * works identically with SDL_FALSE and SDL_TRUE, because they evaluate to
 * the same values, respectively. */
#define PG_SetEventEnabled(type, enabled) SDL_EventState(type, enabled)
#define PG_EventEnabled(type) SDL_EventState(type, SDL_QUERY)
#define PG_SetJoystickEventsEnabled(enabled) SDL_JoystickEventState(enabled)

#define PG_FIND_VNUM_MAJOR(ver) ver.major
#define PG_FIND_VNUM_MINOR(ver) ver.minor
#define PG_FIND_VNUM_MICRO(ver) ver.patch

#define PG_INIT_TIMER SDL_INIT_TIMER

#define PG_SurfaceHasRLE SDL_HasSurfaceRLE

static inline bool
PG_SetSurfaceRLE(SDL_Surface *surface, bool enabled)
{
    return SDL_SetSurfaceRLE(surface, enabled) == 0;
}

static inline bool
PG_GetSurfaceClipRect(SDL_Surface *surface, SDL_Rect *rect)
{
    *rect = surface->clip_rect;
    return true;
}

static inline bool
PG_GL_SetSwapInterval(int interval)
{
    return SDL_GL_SetSwapInterval(interval) == 0;
}
#endif

/* DictProxy is useful for event posting with an arbitrary dict. Maintains
 * state of number of events on queue and whether the owner of this struct
 * wants this dict freed. This DictProxy is only to be freed when there are no
 * more instances of this DictProxy on the event queue. Access to this is
 * safeguarded with a per-proxy spinlock, which is more optimal than having
 * to hold GIL in case of event timers */
typedef struct _pgEventDictProxy {
    PyObject *dict;
    SDL_SpinLock lock;
    int num_on_queue;
    Uint8 do_free_at_end;
} pgEventDictProxy;

/* SDL 1.2 constants removed from SDL 2 */
typedef enum {
    SDL_HWSURFACE = 0,
    SDL_RESIZABLE = SDL_WINDOW_RESIZABLE,
    SDL_ASYNCBLIT = 0,
    SDL_OPENGL = SDL_WINDOW_OPENGL,
    SDL_OPENGLBLIT = 0,
    SDL_ANYFORMAT = 0,
    SDL_HWPALETTE = 0,
    SDL_DOUBLEBUF = 0,
    SDL_FULLSCREEN = SDL_WINDOW_FULLSCREEN,
    SDL_HWACCEL = 0,
    SDL_SRCCOLORKEY = 0,
    SDL_RLEACCELOK = 0,
    SDL_SRCALPHA = 0,
    SDL_NOFRAME = SDL_WINDOW_BORDERLESS,
    SDL_GL_SWAP_CONTROL = 0,
    TIMER_RESOLUTION = 0
} PygameVideoFlags;

/* the wheel button constants were removed from SDL 2 */
typedef enum {
    PGM_BUTTON_LEFT = SDL_BUTTON_LEFT,
    PGM_BUTTON_RIGHT = SDL_BUTTON_RIGHT,
    PGM_BUTTON_MIDDLE = SDL_BUTTON_MIDDLE,
    PGM_BUTTON_WHEELUP = 4,
    PGM_BUTTON_WHEELDOWN = 5,
    PGM_BUTTON_X1 = SDL_BUTTON_X1 + 2,
    PGM_BUTTON_X2 = SDL_BUTTON_X2 + 2,
    PGM_BUTTON_KEEP = 0x80
} PygameMouseFlags;

#if SDL_VERSION_ATLEAST(3, 0, 0)
typedef enum {
    PGE_WINDOWSHOWN = SDL_EVENT_WINDOW_SHOWN,
    PGE_WINDOWHIDDEN = SDL_EVENT_WINDOW_HIDDEN,
    PGE_WINDOWEXPOSED = SDL_EVENT_WINDOW_EXPOSED,
    PGE_WINDOWMOVED = SDL_EVENT_WINDOW_MOVED,
    PGE_WINDOWRESIZED = SDL_EVENT_WINDOW_RESIZED,
    PGE_WINDOWSIZECHANGED = SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED,
    PGE_WINDOWMINIMIZED = SDL_EVENT_WINDOW_MINIMIZED,
    PGE_WINDOWMAXIMIZED = SDL_EVENT_WINDOW_MAXIMIZED,
    PGE_WINDOWRESTORED = SDL_EVENT_WINDOW_RESTORED,
    PGE_WINDOWENTER = SDL_EVENT_WINDOW_MOUSE_ENTER,
    PGE_WINDOWLEAVE = SDL_EVENT_WINDOW_MOUSE_LEAVE,
    PGE_WINDOWFOCUSGAINED = SDL_EVENT_WINDOW_FOCUS_GAINED,
    PGE_WINDOWFOCUSLOST = SDL_EVENT_WINDOW_FOCUS_LOST,
    PGE_WINDOWCLOSE = SDL_EVENT_WINDOW_CLOSE_REQUESTED,
    PGE_WINDOWTAKEFOCUS = -1, /* No SDL3 equivalent */
    PGE_WINDOWHITTEST = SDL_EVENT_WINDOW_HIT_TEST,
    PGE_WINDOWICCPROFCHANGED = SDL_EVENT_WINDOW_ICCPROF_CHANGED,
    PGE_WINDOWDISPLAYCHANGED = SDL_EVENT_WINDOW_DISPLAY_CHANGED,
} PygameWindowEventCode;
/*
TODO: expose these window events in pygame API
    SDL_EVENT_WINDOW_METAL_VIEW_RESIZED,
    SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED,
    SDL_EVENT_WINDOW_SAFE_AREA_CHANGED,
    SDL_EVENT_WINDOW_OCCLUDED,
    SDL_EVENT_WINDOW_ENTER_FULLSCREEN,
    SDL_EVENT_WINDOW_LEAVE_FULLSCREEN,
    SDL_EVENT_WINDOW_DESTROYED,
    SDL_EVENT_WINDOW_HDR_STATE_CHANGED,
*/
#endif

typedef enum {
    /* Any SDL_* events here are for backward compatibility. */
    SDL_NOEVENT = 0,

    SDL_ACTIVEEVENT = SDL_USEREVENT,
    SDL_VIDEORESIZE,
    SDL_VIDEOEXPOSE,
#if SDL_VERSION_ATLEAST(3, 0, 0)
    /* SDL_SYSWMEVENT removed in SDL3, define it here for compat */
    SDL_SYSWMEVENT,
#endif

    PGE_MIDIIN,
    PGE_MIDIOUT,
    PGE_KEYREPEAT, /* Special internal pygame event, for managing key-presses */

/* These PGE events are only needed on SDL2: SDL3 has dedicated events for
 * these */
#if !SDL_VERSION_ATLEAST(3, 0, 0)
    /* DO NOT CHANGE THE ORDER OF EVENTS HERE */
    PGE_WINDOWSHOWN,
    PGE_WINDOWHIDDEN,
    PGE_WINDOWEXPOSED,
    PGE_WINDOWMOVED,
    PGE_WINDOWRESIZED,
    PGE_WINDOWSIZECHANGED,
    PGE_WINDOWMINIMIZED,
    PGE_WINDOWMAXIMIZED,
    PGE_WINDOWRESTORED,
    PGE_WINDOWENTER,
    PGE_WINDOWLEAVE,
    PGE_WINDOWFOCUSGAINED,
    PGE_WINDOWFOCUSLOST,
    PGE_WINDOWCLOSE,
    PGE_WINDOWTAKEFOCUS,
    PGE_WINDOWHITTEST,
    PGE_WINDOWICCPROFCHANGED,
    PGE_WINDOWDISPLAYCHANGED,
#endif

    /* Here we define PGPOST_* events, events that act as a one-to-one
     * proxy for SDL events (and some extra events too!), the proxy is used
     * internally when pygame users use event.post()
     *
     * At a first glance, these may look redundant, but they are really
     * important, especially with event blocking. If proxy events are
     * not there, blocked events don't make it to our event filter, and
     * that can break a lot of stuff.
     *
     * IMPORTANT NOTE: Do not post events directly with these proxy types,
     * use the appropriate functions from event.c, that handle these proxy
     * events for you.
     * Proxy events are for internal use only */
    PGPOST_EVENTBEGIN, /* mark start of proxy-events */
    PGPOST_ACTIVEEVENT = PGPOST_EVENTBEGIN,
    PGPOST_APP_TERMINATING,
    PGPOST_APP_LOWMEMORY,
    PGPOST_APP_WILLENTERBACKGROUND,
    PGPOST_APP_DIDENTERBACKGROUND,
    PGPOST_APP_WILLENTERFOREGROUND,
    PGPOST_APP_DIDENTERFOREGROUND,
    PGPOST_AUDIODEVICEADDED,
    PGPOST_AUDIODEVICEREMOVED,
    PGPOST_CLIPBOARDUPDATE,
    PGPOST_CONTROLLERAXISMOTION,
    PGPOST_CONTROLLERBUTTONDOWN,
    PGPOST_CONTROLLERBUTTONUP,
    PGPOST_CONTROLLERDEVICEADDED,
    PGPOST_CONTROLLERDEVICEREMOVED,
    PGPOST_CONTROLLERDEVICEREMAPPED,
    PGPOST_CONTROLLERTOUCHPADDOWN,
    PGPOST_CONTROLLERTOUCHPADMOTION,
    PGPOST_CONTROLLERTOUCHPADUP,
    PGPOST_CONTROLLERSENSORUPDATE,
    PGPOST_DOLLARGESTURE,
    PGPOST_DOLLARRECORD,
    PGPOST_DROPFILE,
    PGPOST_DROPTEXT,
    PGPOST_DROPBEGIN,
    PGPOST_DROPCOMPLETE,
    PGPOST_FINGERMOTION,
    PGPOST_FINGERDOWN,
    PGPOST_FINGERUP,
    PGPOST_KEYDOWN,
    PGPOST_KEYMAPCHANGED,
    PGPOST_KEYUP,
    PGPOST_JOYAXISMOTION,
    PGPOST_JOYBALLMOTION,
    PGPOST_JOYHATMOTION,
    PGPOST_JOYBUTTONDOWN,
    PGPOST_JOYBUTTONUP,
    PGPOST_JOYDEVICEADDED,
    PGPOST_JOYDEVICEREMOVED,
    PGPOST_LOCALECHANGED,
    PGPOST_MIDIIN,
    PGPOST_MIDIOUT,
    PGPOST_MOUSEMOTION,
    PGPOST_MOUSEBUTTONDOWN,
    PGPOST_MOUSEBUTTONUP,
    PGPOST_MOUSEWHEEL,
    PGPOST_MULTIGESTURE,
    PGPOST_NOEVENT,
    PGPOST_QUIT,
    PGPOST_RENDER_TARGETS_RESET,
    PGPOST_RENDER_DEVICE_RESET,
    PGPOST_SYSWMEVENT,
    PGPOST_TEXTEDITING,
    PGPOST_TEXTINPUT,
    PGPOST_VIDEORESIZE,
    PGPOST_VIDEOEXPOSE,
    PGPOST_WINDOWSHOWN,
    PGPOST_WINDOWHIDDEN,
    PGPOST_WINDOWEXPOSED,
    PGPOST_WINDOWMOVED,
    PGPOST_WINDOWRESIZED,
    PGPOST_WINDOWSIZECHANGED,
    PGPOST_WINDOWMINIMIZED,
    PGPOST_WINDOWMAXIMIZED,
    PGPOST_WINDOWRESTORED,
    PGPOST_WINDOWENTER,
    PGPOST_WINDOWLEAVE,
    PGPOST_WINDOWFOCUSGAINED,
    PGPOST_WINDOWFOCUSLOST,
    PGPOST_WINDOWCLOSE,
    PGPOST_WINDOWTAKEFOCUS,
    PGPOST_WINDOWHITTEST,
    PGPOST_WINDOWICCPROFCHANGED,
    PGPOST_WINDOWDISPLAYCHANGED,

    PGE_USEREVENT, /* this event must stay in this position only */

    PG_NUMEVENTS =
        SDL_LASTEVENT /* Not an event. Indicates end of user events. */
} PygameEventCode;

/* SDL1 ACTIVEEVENT state attribute can take the following values */
/* These constant values are directly picked from SDL1 source */
#define SDL_APPMOUSEFOCUS 0x01
#define SDL_APPINPUTFOCUS 0x02
#define SDL_APPACTIVE 0x04

/* Surface flags: based on SDL 1.2 flags */
typedef enum {
    PGS_SWSURFACE = 0x00000000,
    PGS_HWSURFACE = 0x00000001,
    PGS_ASYNCBLIT = 0x00000004,

    PGS_ANYFORMAT = 0x10000000,
    PGS_HWPALETTE = 0x20000000,
    PGS_DOUBLEBUF = 0x40000000,
    PGS_FULLSCREEN = 0x80000000,
    PGS_SCALED = 0x00000200,

    PGS_OPENGL = 0x00000002,
    PGS_OPENGLBLIT = 0x0000000A,
    PGS_RESIZABLE = 0x00000010,
    PGS_NOFRAME = 0x00000020,
    PGS_SHOWN = 0x00000040,  /* Added from SDL 2 */
    PGS_HIDDEN = 0x00000080, /* Added from SDL 2 */

    PGS_HWACCEL = 0x00000100,
    PGS_SRCCOLORKEY = 0x00001000,
    PGS_RLEACCELOK = 0x00002000,
    PGS_RLEACCEL = 0x00004000,
    PGS_SRCALPHA = 0x00010000,
    PGS_PREALLOC = 0x01000000
} PygameSurfaceFlags;

typedef enum {
    PGS_SCROLL_DEFAULT = 0x00000000,
    PGS_SCROLL_REPEAT = 0x00000001,
    PGS_SCROLL_ERASE = 0x00000004
} PygameScrollSurfaceFlags;

#define RAISE(x, y) (PyErr_SetString((x), (y)), NULL)
#define RAISERETURN(x, y, r)   \
    PyErr_SetString((x), (y)); \
    return r;
#define DEL_ATTR_NOT_SUPPORTED_CHECK(name, value)                            \
    do {                                                                     \
        if (!value) {                                                        \
            PyErr_Format(PyExc_AttributeError, "Cannot delete attribute %s", \
                         name);                                              \
            return -1;                                                       \
        }                                                                    \
    } while (0)

#define DEL_ATTR_NOT_SUPPORTED_CHECK_NO_NAME(value)                           \
    do {                                                                      \
        if (!value) {                                                         \
            PyErr_SetString(PyExc_AttributeError, "Cannot delete attribute"); \
            return -1;                                                        \
        }                                                                     \
    } while (0)

/*
 * Initialization checks
 */

#define VIDEO_INIT_CHECK()            \
    if (!SDL_WasInit(SDL_INIT_VIDEO)) \
    return RAISE(pgExc_SDLError, "video system not initialized")

#define JOYSTICK_INIT_CHECK()            \
    if (!SDL_WasInit(SDL_INIT_JOYSTICK)) \
    return RAISE(pgExc_SDLError, "joystick system not initialized")

/* thread check */
#ifdef WITH_THREAD
#define PG_CHECK_THREADS() (1)
#else /* ~WITH_THREAD */
#define PG_CHECK_THREADS() \
    (RAISE(PyExc_NotImplementedError, "Python built without thread support"))
#endif /* ~WITH_THREAD */

/* Update this function if new sequences are added to the fast sequence
 * type. */
#ifndef pgSequenceFast_Check
#define pgSequenceFast_Check(o) (PyList_Check(o) || PyTuple_Check(o))
#endif /* ~pgSequenceFast_Check */

/*
 * event module internals
 */
struct pgEventObject {
    PyObject_HEAD int type;
    PyObject *dict;
};

/*
 * surface module internals
 */
struct pgSubSurface_Data {
    PyObject *owner;
    int offsetx, offsety;
};

/*
 * color module internals
 */
struct pgColorObject {
    PyObject_HEAD Uint8 data[4];
    Uint8 len;
};

typedef enum {
    /* 0b000: Only handle RGB[A] sequence (which includes pygame.Color) */
    PG_COLOR_HANDLE_SIMPLE = 0,

    /* 0b001: In addition to PG_COLOR_HANDLE_SIMPLE, also handle str */
    PG_COLOR_HANDLE_STR = 1,

    /* 0b010: In addition to PG_COLOR_HANDLE_SIMPLE, also handles int */
    PG_COLOR_HANDLE_INT = (PG_COLOR_HANDLE_STR << 1),

    /* 0b100: A specialised flag, used to indicate that only tuple,
       pygame.Color or subtypes of these both are allowed */
    PG_COLOR_HANDLE_RESTRICT_SEQ = (PG_COLOR_HANDLE_INT << 1),

    /* 0b011: equivalent to PG_COLOR_HANDLE_STR | PG_COLOR_HANDLE_INT */
    PG_COLOR_HANDLE_ALL = PG_COLOR_HANDLE_STR | PG_COLOR_HANDLE_INT,
} pgColorHandleFlags;

/*
 * include public API
 */
#include "include/_pygame.h"

/* Slot counts.
 * Remember to keep these constants up to date.
 */

#define PYGAMEAPI_RECT_NUMSLOTS 10
#define PYGAMEAPI_JOYSTICK_NUMSLOTS 3
#define PYGAMEAPI_DISPLAY_NUMSLOTS 2
#define PYGAMEAPI_SURFACE_NUMSLOTS 4
#define PYGAMEAPI_SURFLOCK_NUMSLOTS 6
#define PYGAMEAPI_RWOBJECT_NUMSLOTS 5
#define PYGAMEAPI_PIXELARRAY_NUMSLOTS 2
#define PYGAMEAPI_COLOR_NUMSLOTS 5
#define PYGAMEAPI_BASE_NUMSLOTS 30
#define PYGAMEAPI_EVENT_NUMSLOTS 10
#define PYGAMEAPI_WINDOW_NUMSLOTS 1
#define PYGAMEAPI_RENDER_NUMSLOTS 3
#define PYGAMEAPI_GEOMETRY_NUMSLOTS 2
#define PYGAMEAPI_BUFFERPROXY_NUMSLOTS 4

#endif /* _PYGAME_INTERNAL_H */
