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
#include <SDL.h>

/* IS_SDLv1 is 1 if SDL 1.x.x, 0 otherwise */
/* IS_SDLv2 is 1 if at least SDL 2.0.0, 0 otherwise */
#if (SDL_VERSION_ATLEAST(2, 0, 0))
#define IS_SDLv2 1
#define IS_SDLv1 0
#else
#define IS_SDLv2 0
#define IS_SDLv1 1
#endif

/*#if IS_SDLv1 && PG_MAJOR_VERSION >= 2
#error pygame 2 requires SDL 2
#endif*/

#if SDL_VERSION_ATLEAST(2, 0, 0)
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

typedef enum {
    /* Any SDL_* events here are for backward compatibility. */
    SDL_NOEVENT = 0,

    /* pygame events */
    PGE_EVENTBEGIN = SDL_USEREVENT, /* Not an event. Indicates start of pygame events. */
    SDL_ACTIVEEVENT = PGE_EVENTBEGIN,
    SDL_VIDEORESIZE,
    SDL_VIDEOEXPOSE,
    PGE_KEYREPEAT,
    PGE_MIDIIN,
    PGE_MIDIOUT,
    PGE_EVENTEND, /* Not an event. Indicates end of pygame events. */

    /* User event range. */
    /* SDL 1.2 allowed for 8 user defined events. */
    PGE_USEREVENT = PGE_EVENTEND,
    PG_NUMEVENTS = PGE_USEREVENT + 0x2000 /* Not an event. Indicates end of user events. */
} PygameEventCode;

#define PGE_NUMRESERVED (PGE_EVENTEND - PGE_EVENTBEGIN)

typedef enum {
    SDL_APPFOCUSMOUSE,
    SDL_APPINPUTFOCUS,
    SDL_APPACTIVE
} PygameAppCode;

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
#else /* ~SDL_VERSION_ATLEAST(2, 0, 0) */
/* To maintain SDL 1.2 build support. */
#define PGE_USEREVENT SDL_USEREVENT
#define PG_NUMEVENTS SDL_NUMEVENTS
/* These midi events were originally defined in midi.py.
 * Note: They are outside the SDL_USEREVENT/SDL_NUMEVENTS event range for
 * SDL 1.2. */
#define PGE_MIDIIN PGE_USEREVENT + 10
#define PGE_MIDIOUT PGE_USEREVENT + 11
#endif /* ~SDL_VERSION_ATLEAST(2, 0, 0) */

#define RAISE(x, y) (PyErr_SetString((x), (y)), (PyObject *)NULL)
#define DEL_ATTR_NOT_SUPPORTED_CHECK(name, value)           \
    do {                                                    \
       if (!value) {                                        \
           if (name) {                                      \
               PyErr_Format(PyExc_AttributeError,           \
                            "Cannot delete attribute %s",   \
                            name);                          \
           } else {                                         \
               PyErr_SetString(PyExc_AttributeError,        \
                               "Cannot delete attribute");  \
           }                                                \
           return -1;                                       \
       }                                                    \
    } while (0)

/*
 * Initialization checks
 */

#define VIDEO_INIT_CHECK()            \
    if (!SDL_WasInit(SDL_INIT_VIDEO)) \
    return RAISE(pgExc_SDLError, "video system not initialized")

#define CDROM_INIT_CHECK()            \
    if (!SDL_WasInit(SDL_INIT_CDROM)) \
    return RAISE(pgExc_SDLError, "cdrom system not initialized")

#define JOYSTICK_INIT_CHECK()            \
    if (!SDL_WasInit(SDL_INIT_JOYSTICK)) \
    return RAISE(pgExc_SDLError, "joystick system not initialized")

/* thread check */
#ifdef WITH_THREAD
#define PG_CHECK_THREADS() (1)
#else /* ~WITH_THREAD */
#define PG_CHECK_THREADS()                        \
    (RAISE(PyExc_NotImplementedError,             \
          "Python built without thread support"))
#endif /* ~WITH_THREAD */

#define PyType_Init(x) (((x).ob_type) = &PyType_Type)

/*
 * event module internals
 */
struct pgEventObject {
    PyObject_HEAD int type;
    PyObject *dict;
};

/*
 * surflock module internals
 */
typedef struct {
    PyObject_HEAD PyObject *surface;
    PyObject *lockobj;
    PyObject *weakrefs;
} pgLifetimeLockObject;

/*
 * surface module internals
 */
struct pgSubSurface_Data {
    PyObject *owner;
    int pixeloffset;
    int offsetx, offsety;
};

/*
 * color module internals
 */
struct pgColorObject {
    PyObject_HEAD
    Uint8 data[4];
    Uint8 len;
};

/*
 * include public API
 */
#include "include/_pygame.h"

#include "pgimport.h"

/* Slot counts.
 * Remember to keep these constants up to date.
 */

#define PYGAMEAPI_RECT_NUMSLOTS 5
#define PYGAMEAPI_JOYSTICK_NUMSLOTS 2
#define PYGAMEAPI_DISPLAY_NUMSLOTS 2
#define PYGAMEAPI_SURFACE_NUMSLOTS 4
#define PYGAMEAPI_SURFLOCK_NUMSLOTS 8
#define PYGAMEAPI_RWOBJECT_NUMSLOTS 6
#define PYGAMEAPI_PIXELARRAY_NUMSLOTS 2
#define PYGAMEAPI_COLOR_NUMSLOTS 5
#define PYGAMEAPI_MATH_NUMSLOTS 2
#define PYGAMEAPI_CDROM_NUMSLOTS 2

#if PG_API_VERSION == 1
#define PYGAMEAPI_BASE_NUMSLOTS 19
#define PYGAMEAPI_EVENT_NUMSLOTS 4
#else /* PG_API_VERSION == 2 */
#define PYGAMEAPI_BASE_NUMSLOTS 23
#define PYGAMEAPI_EVENT_NUMSLOTS 6
#endif /* PG_API_VERSION == 2 */

#endif /* _PYGAME_INTERNAL_H */
