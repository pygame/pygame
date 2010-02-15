/*
  pygame - Python Game Library
  Copyright (C) 2000-2001 Pete Shinners, 2008 Marcus von Appen

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

*/
#ifndef _PYGAME_SDL_H_
#define _PYGAME_SDL_H_

#include <SDL.h>

#include "pgbase.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PYGAME_SDLBASE_FIRSTSLOT 0
#define PYGAME_SDLBASE_NUMSLOTS 10
#ifndef PYGAME_SDLBASE_INTERNAL
#define Uint8FromObj                                                    \
    (*(int(*)(PyObject*,Uint8*))PyGameSDLBase_C_API[PYGAME_SDLBASE_FIRSTSLOT])
#define Uint16FromObj                                                   \
    (*(int(*)(PyObject*,Uint16*))PyGameSDLBase_C_API[PYGAME_SDLBASE_FIRSTSLOT+1])
#define Sint16FromObj                                                   \
    (*(int(*)(PyObject*,Sint16*))PyGameSDLBase_C_API[PYGAME_SDLBASE_FIRSTSLOT+2])
#define Uint32FromObj                                                   \
    (*(int(*)(PyObject*,Uint32*))PyGameSDLBase_C_API[PYGAME_SDLBASE_FIRSTSLOT+3])
#define Uint8FromSeqIndex                                               \
    (*(int(*)(PyObject*,Py_ssize_t,Uint8*))PyGameSDLBase_C_API[PYGAME_SDLBASE_FIRSTSLOT+4])
#define Uint16FromSeqIndex                                              \
    (*(int(*)(PyObject*,Py_ssize_t,Uint16*))PyGameSDLBase_C_API[PYGAME_SDLBASE_FIRSTSLOT+5])
#define Sint16FromSeqIndex                                              \
    (*(int(*)(PyObject*,Py_ssize_t,Sint16*))PyGameSDLBase_C_API[PYGAME_SDLBASE_FIRSTSLOT+6])
#define Uint32FromSeqIndex                                              \
    (*(int(*)(PyObject*,Py_ssize_t,Uint32*))PyGameSDLBase_C_API[PYGAME_SDLBASE_FIRSTSLOT+7])
#define IsValidRect                                                     \
    (*(int(*)(PyObject*))PyGameSDLBase_C_API[PYGAME_SDLBASE_FIRSTSLOT+8])
#define SDLRect_FromRect                                                \
    (*(int(*)(PyObject*,SDL_Rect*))PyGameSDLBase_C_API[PYGAME_SDLBASE_FIRSTSLOT+9])
#endif /* PYGAME_SDLBASE_INTERNAL */

/**
 * C API export.
 */
#ifdef PYGAME_INTERNAL
void **PyGameSDLBase_C_API;
#else
static void **PyGameSDLBase_C_API;
#endif

#define PYGAME_SDLBASE_SLOTS                                    \
    (PYGAME_SDLBASE_FIRSTSLOT + PYGAME_SDLBASE_NUMSLOTS)
#define PYGAME_SDLBASE_ENTRY "_PYGAME_SDLBASE_CAPI"

static int
import_pygame2_sdl_base (void)
{
    PyObject *_module = PyImport_ImportModule ("pygame2.sdl.base");
    if (_module != NULL)
    {
        PyObject *_capi = PyObject_GetAttrString(_module, PYGAME_SDLBASE_ENTRY);
        if (!PyCObject_Check (_capi))
        {
            Py_DECREF (_module);
            return -1;
        }
        PyGameSDLBase_C_API = (void**) PyCObject_AsVoidPtr (_capi);
        Py_DECREF (_capi);
        return 0;
    }
    return -1;
}

/* Video system */
#define ASSERT_VIDEO_INIT(x)                                            \
    if (!SDL_WasInit(SDL_INIT_VIDEO))                                   \
    {                                                                   \
        PyErr_SetString(PyExc_PyGameError, "video subsystem not initialized"); \
        return (x);                                                     \
    }

#define ASSERT_VIDEO_SURFACE_SET(x)                                     \
    ASSERT_VIDEO_INIT(x);                                               \
    if (!SDL_GetVideoSurface ())                                        \
    {                                                                   \
        PyErr_SetString(PyExc_PyGameError, "display surface not set");  \
        return (x);                                                     \
    }

#define PYGAME_SDLVIDEO_FIRSTSLOT 0
#define PYGAME_SDLVIDEO_NUMSLOTS 1
#ifndef PYGAME_SDLVIDEO_INTERNAL
#define SDLColorFromObj                                                    \
    (*(int(*)(PyObject*,SDL_PixelFormat*,Uint32*))PyGameSDLVideo_C_API[PYGAME_SDLVIDEO_FIRSTSLOT+0])
#endif /* PYGAME_SDLVIDEO_INTERNAL */

typedef struct
{
    PyObject_HEAD
    SDL_PixelFormat *format;
    int              readonly;
} PyPixelFormat;
#define PyPixelFormat_AsPixelFormat(x) (((PyPixelFormat*)x)->format)
#define PYGAME_SDLPXFMT_FIRSTSLOT \
    (PYGAME_SDLVIDEO_FIRSTSLOT + PYGAME_SDLVIDEO_NUMSLOTS)
#define PYGAME_SDLPXFMT_NUMSLOTS 3
#ifndef PYGAME_SDLPXFMT_INTERNAL
#define PyPixelFormat_Type                                              \
    (*(PyTypeObject*)PyGameSDLVideo_C_API[PYGAME_SDLPXFMT_FIRSTSLOT+0])
#define PyPixelFormat_Check(x)                                          \
    (PyObject_TypeCheck(x,                                              \
        (PyTypeObject*)PyGameSDLVideo_C_API[PYGAME_SDLPXFMT_FIRSTSLOT+0]))
#define PyPixelFormat_New                                               \
    (*(PyObject*(*)(void))PyGameSDLVideo_C_API[PYGAME_SDLPXFMT_FIRSTSLOT+1])
#define PyPixelFormat_NewFromSDLPixelFormat                             \
    (*(PyObject*(*)(SDL_PixelFormat*))PyGameSDLVideo_C_API[PYGAME_SDLPXFMT_FIRSTSLOT+2])

#endif /* PYGAME_SDLPXFMT_INTERNAL */

typedef struct
{
    PyObject *surface;
    PyObject *lockobj;
} SDLSurfaceLock;

typedef struct
{
    PySurface   pysurface;
    
    SDL_Surface *surface;
    int         isdisplay : 1;

    PyObject    *dict;
    PyObject    *weakrefs;
    PyObject    *locklist;
    pguint16     intlocks; /* Internally hold locks */
} PySDLSurface;
#define PySDLSurface_AsSDLSurface(x) (((PySDLSurface*)x)->surface)
#define PySDLSurface_AsPySurface(x) (&(((PySDLSurface*)x)->pysurface))
#define PYGAME_SDLSURFACE_FIRSTSLOT                             \
    (PYGAME_SDLPXFMT_FIRSTSLOT + PYGAME_SDLPXFMT_NUMSLOTS)
#define PYGAME_SDLSURFACE_NUMSLOTS 7
#ifndef PYGAME_SDLSURFACE_INTERNAL
#define PySDLSurface_Type \
    (*(PyTypeObject*)PyGameSDLVideo_C_API[PYGAME_SDLSURFACE_FIRSTSLOT+0])
#define PySDLSurface_Check(x)                                              \
    (PyObject_TypeCheck(x,                                              \
        (PyTypeObject*)PyGameSDLVideo_C_API[PYGAME_SDLSURFACE_FIRSTSLOT+0]))
#define PySDLSurface_New                                                \
    (*(PyObject*(*)(int,int))PyGameSDLVideo_C_API[PYGAME_SDLSURFACE_FIRSTSLOT+1])
#define PySDLSurface_NewFromSDLSurface                                  \
    (*(PyObject*(*)(SDL_Surface*))PyGameSDLVideo_C_API[PYGAME_SDLSURFACE_FIRSTSLOT+2])
#define PySDLSurface_Copy                                               \
    (*(PyObject*(*)(PyObject*))PyGameSDLVideo_C_API[PYGAME_SDLSURFACE_FIRSTSLOT+3])
/* Note: this increments the surface refcount. */
#define PySDLSurface_AddRefLock                                         \
    (*(int(*)(PyObject*,PyObject*))PyGameSDLVideo_C_API[PYGAME_SDLSURFACE_FIRSTSLOT+4])
/* Note: this decrements the surface refcount. */
#define PySDLSurface_RemoveRefLock                                      \
    (*(int(*)(PyObject*,PyObject*))PyGameSDLVideo_C_API[PYGAME_SDLSURFACE_FIRSTSLOT+5])
#define PySDLSurface_AcquireLockObj                                     \
    (*(PyObject*(*)(PyObject*,PyObject*))PyGameSDLVideo_C_API[PYGAME_SDLSURFACE_FIRSTSLOT+6])

#endif /* PYGAME_SDLSURFACE_INTERNAL */

typedef struct
{
    PyObject_HEAD
    
    SDL_Overlay *overlay;
    PyObject    *surface;
    PyObject    *dict;
    PyObject    *weakrefs;
    PyObject    *locklist;
} PyOverlay;
#define PyOverlay_AsOverlay(x) (((PyOverlay*)x)->overlay)
#define PYGAME_SDLOVERLAY_FIRSTSLOT                             \
    (PYGAME_SDLSURFACE_FIRSTSLOT + PYGAME_SDLSURFACE_NUMSLOTS)
#define PYGAME_SDLOVERLAY_NUMSLOTS 4
#ifndef PYGAME_SDLOVERLAY_INTERNAL
#define PyOverlay_Type \
    (*(PyTypeObject*)PyGameSDLVideo_C_API[PYGAME_SDLOVERLAY_FIRSTSLOT+0])
#define PyOverlay_Check(x)                                              \
    (PyObject_TypeCheck(x,                                              \
        (PyTypeObject*)PyGameSDLVideo_C_API[PYGAME_SDLOVERLAY_FIRSTSLOT+0]))
#define PyOverlay_New                                                   \
    (*(PyObject*(*)(PyObject*,int,int, Uint32))PyGameSDLVideo_C_API[PYGAME_SDLOVERLAY_FIRSTSLOT+1])
/* Note: this increments the surface refcount. */
#define PyOverlay_AddRefLock                                            \
    (*(int(*)(PyObject*,PyObject*))PyGameSDLVideo_C_API[PYGAME_SDLOVERLAY_FIRSTSLOT+2])
/* Note: this decrements the surface refcount. */
#define PyOverlay_RemoveRefLock                                         \
    (*(int(*)(PyObject*,PyObject*))PyGameSDLVideo_C_API[PYGAME_SDLOVERLAY_FIRSTSLOT+3])

#endif /* PYGAME_SDLOVERLAY_INTERNAL */

/**
 * C API export.
 */
#ifdef PYGAME_INTERNAL
void **PyGameSDLVideo_C_API;
#else
static void **PyGameSDLVideo_C_API;
#endif

#define PYGAME_SDLVIDEO_SLOTS                                   \
    (PYGAME_SDLOVERLAY_FIRSTSLOT + PYGAME_SDLOVERLAY_NUMSLOTS)
#define PYGAME_SDLVIDEO_ENTRY "_PYGAME_SDLVIDEO_CAPI"

static int
import_pygame2_sdl_video (void)
{
    PyObject *_module = PyImport_ImportModule ("pygame2.sdl.video");
    if (_module != NULL)
    {
        PyObject *_capi = PyObject_GetAttrString(_module, PYGAME_SDLVIDEO_ENTRY);
        if (!PyCObject_Check (_capi))
        {
            Py_DECREF (_module);
            return -1;
        }
        PyGameSDLVideo_C_API = (void**) PyCObject_AsVoidPtr (_capi);
        Py_DECREF (_capi);
        return 0;
    }
    return -1;
}

/* Mouse input */
typedef struct
{
    PyObject_HEAD
    SDL_Cursor *cursor;
} PyCursor;
#define PyCursor_AsCursor(x) (((PyCursor*)x)->cursor)
#define PYGAME_SDLCURSOR_FIRSTSLOT 0
#define PYGAME_SDLCURSOR_NUMSLOTS 1
#ifndef PYGAME_SDLCURSOR_INTERNAL
#define PyCursor_Type \
    (*(PyTypeObject*)PyGameSDLMouse_C_API[PYGAME_SDLCURSOR_FIRSTSLOT+0])
#define PyCursor_Check(x)                                               \
    (PyObject_TypeCheck(x,                                              \
        (PyTypeObject*)PyGameSDLMouse_C_API[PYGAME_SDLCURSOR_FIRSTSLOT+0]))
#endif /* PYGAME_SDLCURSOR_INTERNAL */

/**
 * C API export.
 */
#ifdef PYGAME_INTERNAL
void **PyGameSDLMouse_C_API;
#else
static void **PyGameSDLMouse_C_API;
#endif

#define PYGAME_SDLMOUSE_SLOTS                                   \
    (PYGAME_SDLCURSOR_FIRSTSLOT + PYGAME_SDLCURSOR_NUMSLOTS)
#define PYGAME_SDLMOUSE_ENTRY "_PYGAME_SDLMOUSE_CAPI"

static int
import_pygame2_sdl_mouse (void)
{
    PyObject *_module = PyImport_ImportModule ("pygame2.sdl.mouse");
    if (_module != NULL)
    {
        PyObject *_capi = PyObject_GetAttrString(_module, PYGAME_SDLMOUSE_ENTRY);
        if (!PyCObject_Check (_capi))
        {
            Py_DECREF (_module);
            return -1;
        }
        PyGameSDLMouse_C_API = (void**) PyCObject_AsVoidPtr (_capi);
        Py_DECREF (_capi);
        return 0;
    }
    return -1;
}

/* Event */
typedef struct
{
    PyObject_HEAD
    Uint8     type;
    PyObject *dict;
} PyEvent;

/*
 * Define a specialised user event constant to check against.
 * >>> hex (hash ("PYGAME_SDLEVENT_USEREVENT"))
 * '0x5F79938C'
 * 
 */
#define PYGAME_USEREVENT 0x5F79938C
#define PYGAME_USEREVENT_CODE 0x5F

#define PYGAME_SDLEVENT_FIRSTSLOT 0
#define PYGAME_SDLEVENT_NUMSLOTS 3
#ifndef PYGAME_SDLEVENT_INTERNAL
#define PyEvent_Type \
    (*(PyTypeObject*)PyGameSDLEvent_C_API[PYGAME_SDLEVENT_FIRSTSLOT+0])
#define PyEvent_Check(x)                                                \
    (PyObject_TypeCheck(x,                                              \
        (PyTypeObject*)PyGameSDLEvent_C_API[PYGAME_SDLEVENT_FIRSTSLOT+0]))
#define PyEvent_New                                                     \
    (*(PyObject*(*)(SDL_Event*))PyGameSDLEvent_C_API[PYGAME_SDLEVENT_FIRSTSLOT+1])
#define PyEvent_SDLEventFromEvent                                       \
    (*(int(*)(PyObject*,SDL_Event*))PyGameSDLEvent_C_API[PYGAME_SDLEVENT_FIRSTSLOT+2])
#endif /* PYGAME_SDLEVENT_INTERNAL */

/**
 * C API export.
 */
#ifdef PYGAME_INTERNAL
void **PyGameSDLEvent_C_API;
#else
static void **PyGameSDLEvent_C_API;
#endif

#define PYGAME_SDLEVENT_SLOTS                                   \
    (PYGAME_SDLEVENT_FIRSTSLOT + PYGAME_SDLEVENT_NUMSLOTS)
#define PYGAME_SDLEVENT_ENTRY "_PYGAME_SDLEVENT_CAPI"

static int
import_pygame2_sdl_event (void)
{
    PyObject *_module = PyImport_ImportModule ("pygame2.sdl.event");
    if (_module != NULL)
    {
        PyObject *_capi = PyObject_GetAttrString(_module, PYGAME_SDLEVENT_ENTRY);
        if (!PyCObject_Check (_capi))
        {
            Py_DECREF (_module);
            return -1;
        }
        PyGameSDLEvent_C_API = (void**) PyCObject_AsVoidPtr (_capi);
        Py_DECREF (_capi);
        return 0;
    }
    return -1;
}

/* Joystick */
#define ASSERT_JOYSTICK_INIT(x)                                         \
    if (!SDL_WasInit(SDL_INIT_JOYSTICK))                                \
    {                                                                   \
        PyErr_SetString(PyExc_PyGameError,                              \
            "joystick subsystem not initialized");                      \
        return (x);                                                     \
    }
#define ASSERT_JOYSTICK_OPEN(x,r)                                   \
    ASSERT_JOYSTICK_INIT(r);                                        \
    if (!SDL_JoystickOpened(((PyJoystick*)(x))->index))             \
    {                                                               \
        PyErr_SetString(PyExc_PyGameError, "joystick is not open"); \
        return (r);                                                 \
    }

typedef struct
{
    PyObject_HEAD
    int           index;
    SDL_Joystick *joystick;
} PyJoystick;
#define PyJoystick_AsJoystick(x) (((PyJoystick*)x)->joystick)
#define PYGAME_SDLJOYSTICK_FIRSTSLOT 0
#define PYGAME_SDLJOYSTICK_NUMSLOTS 2
#ifndef PYGAME_SDLJOYSTICK_INTERNAL
#define PyJoystick_Type \
    (*(PyTypeObject*)PyGameSDLJoystick_C_API[PYGAME_SDLJOYSTICK_FIRSTSLOT+0])
#define PyJoystick_Check(x)                                             \
    (PyObject_TypeCheck(x,                                              \
        (PyTypeObject*)PyGameSDLJoystick_C_API[PYGAME_SDLJOYSTICK_FIRSTSLOT+0]))
#define PyJoystick_New                                                  \
    (*(PyObject*(*)(int))PyGameSDLJoystick_C_API[PYGAME_SDLJOYSTICK_FIRSTSLOT+1])

#endif /* PYGAME_SDLJOYSTICK_INTERNAL */

/**
 * C API export.
 */
#ifdef PYGAME_INTERNAL
void **PyGameSDLJoystick_C_API;
#else
static void **PyGameSDLJoystick_C_API;
#endif

#define PYGAME_SDLJOYSTICK_SLOTS                                        \
    (PYGAME_SDLJOYSTICK_FIRSTSLOT + PYGAME_SDLJOYSTICK_NUMSLOTS)
#define PYGAME_SDLJOYSTICK_ENTRY "_PYGAME_SDLJOYSTICK_CAPI"
    
static int
import_pygame2_sdl_joystick (void)
{
    PyObject *_module = PyImport_ImportModule ("pygame2.sdl.joystick");
    if (_module != NULL)
    {
        PyObject *_capi = PyObject_GetAttrString(_module, PYGAME_SDLJOYSTICK_ENTRY);
        if (!PyCObject_Check (_capi))
        {
            Py_DECREF (_module);
            return -1;
        }
        PyGameSDLJoystick_C_API = (void**) PyCObject_AsVoidPtr (_capi);
        Py_DECREF (_capi);
        return 0;
    }
    return -1;
}

/* Timer */
#define ASSERT_TIME_INIT(x)                                             \
    if (!SDL_WasInit(SDL_INIT_TIMER))                                   \
    {                                                                   \
        PyErr_SetString(PyExc_PyGameError, "time subsystem not initialized"); \
        return (x);                                                     \
    }

/* Cdrom */
#define ASSERT_CDROM_INIT(x)                                            \
    if (!SDL_WasInit(SDL_INIT_CDROM))                                   \
    {                                                                   \
        PyErr_SetString(PyExc_PyGameError, "cdrom subsystem not initialized"); \
        return (x);                                                     \
    }

#define ASSERT_CDROM_OPEN(x,r)                                      \
    ASSERT_CDROM_INIT(r);                                           \
    if (!((PyCD*)(x))->cd)                                          \
    {                                                               \
        PyErr_SetString(PyExc_PyGameError, "cdrom is not open");    \
        return (r);                                                 \
    }

typedef struct
{
    PyObject_HEAD
    int     index;
    SDL_CD *cd;
} PyCD;
#define PyCD_AsCD(x) (((PyCD*)x)->cd)
#define PYGAME_SDLCDROM_FIRSTSLOT 0
#define PYGAME_SDLCDROM_NUMSLOTS 2
#ifndef PYGAME_SDLCDROM_INTERNAL
#define PyCD_Type \
    (*(PyTypeObject*)PyGameSDLCD_C_API[PYGAME_SDLCDROM_FIRSTSLOT+0])
#define PyCD_Check(x)                                                   \
    (PyObject_TypeCheck(x,                                              \
        (PyTypeObject*)PyGameSDLCD_C_API[PYGAME_SDLCDROM_FIRSTSLOT+0]))
#define PyCD_New                                                        \
    (*(PyObject*(*)(int))PyGameSDLCD_C_API[PYGAME_SDLCDROM_FIRSTSLOT+1])

#endif /* PYGAME_SDLCDROM_INTERNAL */

typedef struct
{
    PyObject_HEAD
    SDL_CDtrack track;
} PyCDTrack;
#define PyCDTrack_AsCDTrack(x) (((PyCDTrack*)x)->track)
#define PYGAME_SDLCDTRACK_FIRSTSLOT \
    (PYGAME_SDLCDROM_FIRSTSLOT + PYGAME_SDLCDROM_NUMSLOTS)
#define PYGAME_SDLCDTRACK_NUMSLOTS 2
#ifndef PYGAME_SDLCDTRACK_INTERNAL
#define PyCDTrack_Type \
    (*(PyTypeObject*)PyGameSDLCD_C_API[PYGAME_SDLCDTRACK_FIRSTSLOT+0])
#define PyCDTrack_Check(x)                                              \
    (PyObject_TypeCheck(x,                                              \
        (PyTypeObject*)PyGameSDLCD_C_API[PYGAME_SDLCDTRACK_FIRSTSLOT+0]))
#define PyCDTrack_New                                                   \
    (*(PyObject*(*)(SDL_CDtrack))PyGameSDLCD_C_API[PYGAME_SDLCDTRACK_FIRSTSLOT+1])
#endif /* PYGAME_SDLCDTRACK_INTERNAL */

/**
 * C API export.
 */
#ifdef PYGAME_INTERNAL
void **PyGameSDLCD_C_API;
#else
static void **PyGameSDLCD_C_API;
#endif

#define PYGAME_SDLCDROM_SLOTS \
    (PYGAME_SDLCDTRACK_FIRSTSLOT + PYGAME_SDLCDTRACK_NUMSLOTS)
#define PYGAME_SDLCDROM_ENTRY "_PYGAME_SDLCDROM_CAPI"
    
static int
import_pygame2_sdl_cdrom (void)
{
    PyObject *_module = PyImport_ImportModule ("pygame2.sdl.cdrom");
    if (_module != NULL)
    {
        PyObject *_capi = PyObject_GetAttrString(_module, PYGAME_SDLCDROM_ENTRY);
        if (!PyCObject_Check (_capi))
        {
            Py_DECREF (_module);
            return -1;
        }
        PyGameSDLCD_C_API = (void**) PyCObject_AsVoidPtr (_capi);
        Py_DECREF (_capi);
        return 0;
    }
    return -1;
}

/* RWops */
#define PYGAME_SDLRWOPS_FIRSTSLOT 0
#define PYGAME_SDLRWOPS_NUMSLOTS 5
#ifndef PYGAME_SDLRWOPS_INTERNAL
#define PyRWops_NewRO                                                 \
    (*(SDL_RWops*(*)(PyObject*,int*))PyGameSDLRWops_C_API[PYGAME_SDLRWOPS_FIRSTSLOT])
#define PyRWops_NewRW                                                 \
    (*(SDL_RWops*(*)(PyObject*,int*))PyGameSDLRWops_C_API[PYGAME_SDLRWOPS_FIRSTSLOT+1])
#define PyRWops_Close                                                 \
    (*(void(*)(SDL_RWops*,int))PyGameSDLRWops_C_API[PYGAME_SDLRWOPS_FIRSTSLOT+2])
#define PyRWops_NewRO_Threaded                                        \
    (*(SDL_RWops*(*)(PyObject*,int*))PyGameSDLRWops_C_API[PYGAME_SDLRWOPS_FIRSTSLOT+3])
#define PyRWops_NewRW_Threaded                                        \
    (*(SDL_RWops*(*)(PyObject*,int*))PyGameSDLRWops_C_API[PYGAME_SDLRWOPS_FIRSTSLOT+4])
#endif /* PYGAME_SDLRWOPS_INTERNAL */

/**
 * C API export.
 */
#ifdef PYGAME_INTERNAL
void **PyGameSDLRWops_C_API;
#else
static void **PyGameSDLRWops_C_API;
#endif

#define PYGAME_SDLRWOPS_SLOTS \
    (PYGAME_SDLRWOPS_FIRSTSLOT + PYGAME_SDLRWOPS_NUMSLOTS)
#define PYGAME_SDLRWOPS_ENTRY "_PYGAME_SDLRWOPS_CAPI"
    
static int
import_pygame2_sdl_rwops (void)
{
    PyObject *_module = PyImport_ImportModule ("pygame2.sdl.rwops");
    if (_module != NULL)
    {
        PyObject *_capi = PyObject_GetAttrString(_module, PYGAME_SDLRWOPS_ENTRY);
        if (!PyCObject_Check (_capi))
        {
            Py_DECREF (_module);
            return -1;
        }
        PyGameSDLRWops_C_API = (void**) PyCObject_AsVoidPtr (_capi);
        Py_DECREF (_capi);
        return 0;
    }
    return -1;
}

#ifdef __cplusplus
}
#endif

#endif /* _PYGAME_SDL_H_ */
