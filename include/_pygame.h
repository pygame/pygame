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

#ifndef _PYGAME_H
#define _PYGAME_H

/** This header file includes all the definitions for the
 ** base pygame extensions. This header only requires
 ** SDL and Python includes. The reason for functions
 ** prototyped with #define's is to allow for maximum
 ** python portability. It also uses python as the
 ** runtime linker, which allows for late binding. For more
 ** information on this style of development, read the Python
 ** docs on this subject.
 ** http://www.python.org/doc/current/ext/using-cobjects.html
 **
 ** If using this to build your own derived extensions,
 ** you'll see that the functions available here are mainly
 ** used to help convert between python objects and SDL objects.
 ** Since this library doesn't add a lot of functionality to
 ** the SDL library, it doesn't need to offer a lot either.
 **
 ** When initializing your extension module, you must manually
 ** import the modules you want to use. (this is the part about
 ** using python as the runtime linker). Each module has its
 ** own import_xxx() routine. You need to perform this import
 ** after you have initialized your own module, and before
 ** you call any routines from that module. Since every module
 ** in pygame does this, there are plenty of examples.
 **
 ** The base module does include some useful conversion routines
 ** that you are free to use in your own extension.
 **
 ** When making changes, it is very important to keep the
 ** FIRSTSLOT and NUMSLOT constants up to date for each
 ** section. Also be sure not to overlap any of the slots.
 ** When you do make a mistake with this, it will result
 ** is a dereferenced NULL pointer that is easier to diagnose
 ** than it could be :]
 **/

#include "pgplatform.h"
#include <Python.h>
#include <SDL.h>

/* version macros (defined since version 1.9.5) */
#define PG_MAJOR_VERSION 2
#define PG_MINOR_VERSION 0
#define PG_PATCH_VERSION 0
#define PG_VERSIONNUM(MAJOR, MINOR, PATCH) (1000*(MAJOR) + 100*(MINOR) + (PATCH))
#define PG_VERSION_ATLEAST(MAJOR, MINOR, PATCH)                             \
    (PG_VERSIONNUM(PG_MAJOR_VERSION, PG_MINOR_VERSION, PG_PATCH_VERSION) >= \
     PG_VERSIONNUM(MAJOR, MINOR, PATCH))

/* SDL 1.x/2.x
 */

/* IS_SDLv1 is 1 if SDL 1.x.x, 0 otherwise */
/* IS_SDLv2 is 1 if at least SDL 2.0.0, 0 otherwise */

#if (SDL_VERSION_ATLEAST(2, 0, 0))
#define IS_SDLv1 0
#define IS_SDLv2 1
#else
#define IS_SDLv1 1
#define IS_SDLv2 0
#endif

/* Pygame uses Py_buffer (PEP 3118) to exchange array information internally;
 * define here as needed.
 */
#if !defined(PyBUF_SIMPLE)
typedef struct bufferinfo {
    void *buf;
    PyObject *obj;
    Py_ssize_t len;
    Py_ssize_t itemsize;
    int readonly;
    int ndim;
    char *format;
    Py_ssize_t *shape;
    Py_ssize_t *strides;
    Py_ssize_t *suboffsets;
    void *internal;
} Py_buffer;

/* Flags for getting buffers */
#define PyBUF_SIMPLE 0
#define PyBUF_WRITABLE 0x0001
/*  we used to include an E, backwards compatible alias  */
#define PyBUF_WRITEABLE PyBUF_WRITABLE
#define PyBUF_FORMAT 0x0004
#define PyBUF_ND 0x0008
#define PyBUF_STRIDES (0x0010 | PyBUF_ND)
#define PyBUF_C_CONTIGUOUS (0x0020 | PyBUF_STRIDES)
#define PyBUF_F_CONTIGUOUS (0x0040 | PyBUF_STRIDES)
#define PyBUF_ANY_CONTIGUOUS (0x0080 | PyBUF_STRIDES)
#define PyBUF_INDIRECT (0x0100 | PyBUF_STRIDES)

#define PyBUF_CONTIG (PyBUF_ND | PyBUF_WRITABLE)
#define PyBUF_CONTIG_RO (PyBUF_ND)

#define PyBUF_STRIDED (PyBUF_STRIDES | PyBUF_WRITABLE)
#define PyBUF_STRIDED_RO (PyBUF_STRIDES)

#define PyBUF_RECORDS (PyBUF_STRIDES | PyBUF_WRITABLE | PyBUF_FORMAT)
#define PyBUF_RECORDS_RO (PyBUF_STRIDES | PyBUF_FORMAT)

#define PyBUF_FULL (PyBUF_INDIRECT | PyBUF_WRITABLE | PyBUF_FORMAT)
#define PyBUF_FULL_RO (PyBUF_INDIRECT | PyBUF_FORMAT)

#define PyBUF_READ 0x100
#define PyBUF_WRITE 0x200
#define PyBUF_SHADOW 0x400

typedef int (*getbufferproc)(PyObject *, Py_buffer *, int);
typedef void (*releasebufferproc)(Py_buffer *);
#endif /* ~defined(PyBUF_SIMPLE) */

/* Flag indicating a pg_buffer; used for assertions within callbacks */
#ifndef NDEBUG
#define PyBUF_PYGAME 0x4000
#endif

#define PyBUF_HAS_FLAG(f, F) (((f) & (F)) == (F))

/* Array information exchange struct C type; inherits from Py_buffer
 *
 * Pygame uses its own Py_buffer derived C struct as an internal representation
 * of an imported array buffer. The extended Py_buffer allows for a
 * per-instance release callback,
 */
typedef void (*pybuffer_releaseproc)(Py_buffer *);

typedef struct pg_bufferinfo_s {
    Py_buffer view;
    PyObject *consumer; /* Input: Borrowed reference */
    pybuffer_releaseproc release_buffer;
} pg_buffer;

/* macros used throughout the source */
#define RAISE(x, y) (PyErr_SetString((x), (y)), (PyObject *)NULL)

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

#include "pgimport.h"

/*
 * BASE module
 */
#define PYGAMEAPI_BASE_FIRSTSLOT 0
#if IS_SDLv1
#define PYGAMEAPI_BASE_NUMSLOTS 19
#else /* IS_SDLv2 */
#define PYGAMEAPI_BASE_NUMSLOTS 23
#endif /* IS_SDLv2 */

#ifndef PYGAMEAPI_BASE_INTERNAL
#define pgExc_SDLError \
    ((PyObject *)      \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_BASE_FIRSTSLOT))

#define pg_RegisterQuit          \
    (*(void (*)(void (*)(void))) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_BASE_FIRSTSLOT + 1))

#define pg_IntFromObj              \
    (*(int (*)(PyObject *, int *)) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_BASE_FIRSTSLOT + 2))

#define pg_IntFromObjIndex              \
    (*(int (*)(PyObject *, int, int *)) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_BASE_FIRSTSLOT + 3))

#define pg_TwoIntsFromObj                 \
    (*(int (*)(PyObject *, int *, int *)) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_BASE_FIRSTSLOT + 4))

#define pg_FloatFromObj \
    (*(int (*)(PyObject *, float *)) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_BASE_FIRSTSLOT + 5))

#define pg_FloatFromObjIndex              \
    (*(int (*)(PyObject *, int, float *)) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_BASE_FIRSTSLOT + 6))

#define pg_TwoFloatsFromObj                   \
    (*(int (*)(PyObject *, float *, float *)) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_BASE_FIRSTSLOT + 7))

#define pg_UintFromObj                \
    (*(int (*)(PyObject *, Uint32 *)) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_BASE_FIRSTSLOT + 8))

#define pg_UintFromObjIndex     \
    (*(int (*)(PyObject *, int, Uint32 *)) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_BASE_FIRSTSLOT + 9))

#define pgVideo_AutoQuit \
    (*(void (*)(void)) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_BASE_FIRSTSLOT + 10))

#define pgVideo_AutoInit \
    (*(int (*)(void))    \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_BASE_FIRSTSLOT + 11))

#define pg_RGBAFromObj               \
    (*(int (*)(PyObject *, Uint8 *)) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_BASE_FIRSTSLOT + 12))

#define pgBuffer_AsArrayInterface   \
    (*(PyObject * (*)(Py_buffer *)) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_BASE_FIRSTSLOT + 13))

#define pgBuffer_AsArrayStruct      \
    (*(PyObject * (*)(Py_buffer *)) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_BASE_FIRSTSLOT + 14))

#define pgObject_GetBuffer                    \
    (*(int (*)(PyObject *, pg_buffer *, int)) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_BASE_FIRSTSLOT + 15))

#define pgBuffer_Release      \
    (*(void (*)(pg_buffer *)) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_BASE_FIRSTSLOT + 16))

#define pgDict_AsBuffer                       \
    (*(int (*)(pg_buffer *, PyObject *, int)) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_BASE_FIRSTSLOT + 17))

#define pgExc_BufferError \
    ((PyObject *)         \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_BASE_FIRSTSLOT + 18))

#if IS_SDLv2
#define pg_GetDefaultWindow     \
    (*(SDL_Window * (*)(void))  \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_BASE_FIRSTSLOT + 19))

#define pg_SetDefaultWindow    \
    (*(void (*)(SDL_Window *)) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_BASE_FIRSTSLOT + 20))

#define pg_GetDefaultWindowSurface \
    (*(PyObject * (*)(void))       \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_BASE_FIRSTSLOT + 21))

#define pg_SetDefaultWindowSurface \
    (*(void (*)(PyObject *))       \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_BASE_FIRSTSLOT + 22))

#endif /* IS_SDLv2 */

#define import_pygame_base() IMPORT_PYGAME_MODULE(base, BASE)
#endif /* ~PYGAMEAPI_BASE_INTERNAL */

/*
 * RECT module
 */
#define PYGAMEAPI_RECT_FIRSTSLOT \
    (PYGAMEAPI_BASE_FIRSTSLOT + PYGAMEAPI_BASE_NUMSLOTS)
#define PYGAMEAPI_RECT_NUMSLOTS 4

#if IS_SDLv1
typedef struct {
    int x, y;
    int w, h;
} GAME_Rect;
#else
typedef SDL_Rect GAME_Rect;
#endif

typedef struct {
    PyObject_HEAD GAME_Rect r;
    PyObject *weakreflist;
} pgRectObject;

#define pgRect_AsRect(x) (((pgRectObject *)x)->r)
#ifndef PYGAMEAPI_RECT_INTERNAL
#define pgRect_Type    \
    (*(PyTypeObject *) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_RECT_FIRSTSLOT + 0))

#define pgRect_Check(x) \
    ((x)->ob_type == &pgRect_Type)
#define pgRect_New                  \
    (*(PyObject * (*)(SDL_Rect *))  \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_RECT_FIRSTSLOT + 1))

#define pgRect_New4                        \
    (*(PyObject * (*)(int, int, int, int)) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_RECT_FIRSTSLOT + 2))

#define pgRect_FromObject                        \
    (*(GAME_Rect * (*)(PyObject *, GAME_Rect *)) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_RECT_FIRSTSLOT + 3))

#define import_pygame_rect() IMPORT_PYGAME_MODULE(rect, RECT)
#endif

/*
 * CDROM module
 */
#define PYGAMEAPI_CDROM_FIRSTSLOT \
    (PYGAMEAPI_RECT_FIRSTSLOT + PYGAMEAPI_RECT_NUMSLOTS)
#define PYGAMEAPI_CDROM_NUMSLOTS 2

typedef struct {
    PyObject_HEAD int id;
} pgCDObject;

#define pgCD_AsID(x) (((pgCDObject *)x)->id)
#ifndef PYGAMEAPI_CDROM_INTERNAL
#define pgCD_Type      \
    (*(PyTypeObject *) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_CDROM_FIRSTSLOT + 0))

#define pgCD_Check(x) \
    ((x)->ob_type == &pgCD_Type)
#define pgCD_New             \
    (*(PyObject * (*)(int))  \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_CDROM_FIRSTSLOT + 1))

#define import_pygame_cd() IMPORT_PYGAME_MODULE(cdrom, CDROM)
#endif

/*
 * JOYSTICK module
 */
#define PYGAMEAPI_JOYSTICK_FIRSTSLOT \
    (PYGAMEAPI_CDROM_FIRSTSLOT + PYGAMEAPI_CDROM_NUMSLOTS)
#define PYGAMEAPI_JOYSTICK_NUMSLOTS 2

typedef struct {
    PyObject_HEAD int id;
} pgJoystickObject;

#define pgJoystick_AsID(x) (((pgJoystickObject *)x)->id)

#ifndef PYGAMEAPI_JOYSTICK_INTERNAL
#define pgJoystick_Type \
    (*(PyTypeObject *)  \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_JOYSTICK_FIRSTSLOT + 0))

#define pgJoystick_Check(x) \
    ((x)->ob_type == &pgJoystick_Type)
#define pgJoystick_New       \
    (*(PyObject * (*)(int))  \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_JOYSTICK_FIRSTSLOT + 1))

#define import_pygame_joystick() IMPORT_PYGAME_MODULE(joystick, JOYSTICK)
#endif

/*
 * DISPLAY module
 */
#define PYGAMEAPI_DISPLAY_FIRSTSLOT \
    (PYGAMEAPI_JOYSTICK_FIRSTSLOT + PYGAMEAPI_JOYSTICK_NUMSLOTS)
#define PYGAMEAPI_DISPLAY_NUMSLOTS 2

#if IS_SDLv2
typedef struct {
    Uint32 hw_available:1;
    Uint32 wm_available:1;
    Uint32 blit_hw:1;
    Uint32 blit_hw_CC:1;
    Uint32 blit_hw_A:1;
    Uint32 blit_sw:1;
    Uint32 blit_sw_CC:1;
    Uint32 blit_sw_A:1;
    Uint32 blit_fill:1;
    Uint32 video_mem;
    SDL_PixelFormat *vfmt;
    SDL_PixelFormat vfmt_data;
    int current_w;
    int current_h;
} pg_VideoInfo;
#endif /* IS_SDLv2 */

typedef struct {
#if IS_SDLv1
    PyObject_HEAD SDL_VideoInfo info;
#else
    PyObject_HEAD pg_VideoInfo info;
#endif
} pgVidInfoObject;

#define pgVidInfo_AsVidInfo(x) (((pgVidInfoObject *)x)->info)
#ifndef PYGAMEAPI_DISPLAY_INTERNAL
#define pgVidInfo_Type \
    (*(PyTypeObject *) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_DISPLAY_FIRSTSLOT + 0))

#define pgVidInfo_Check(x) \
    ((x)->ob_type == &pgVidInfo_Type)

#if IS_SDLv1
#define pgVidInfo_New                   \
    (*(PyObject * (*)(SDL_VideoInfo *)) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_DISPLAY_FIRSTSLOT + 1))
#else
#define pgVidInfo_New                   \
    (*(PyObject * (*)(pg_VideoInfo *))  \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_DISPLAY_FIRSTSLOT + 1))
#endif

#define import_pygame_display() IMPORT_PYGAME_MODULE(display, DISPLAY)
#endif

/*
 * SURFACE module
 */
#define PYGAMEAPI_SURFACE_FIRSTSLOT \
    (PYGAMEAPI_DISPLAY_FIRSTSLOT + PYGAMEAPI_DISPLAY_NUMSLOTS)
#define PYGAMEAPI_SURFACE_NUMSLOTS 3
struct pgSubSurface_Data;

typedef struct {
    PyObject_HEAD SDL_Surface *surf;
#if IS_SDLv2
    int owner;
#endif /* IS_SDLv2 */
    struct pgSubSurface_Data *subsurface; /* ptr to subsurface data (if a
                                           * subsurface)*/
    PyObject *weakreflist;
    PyObject *locklist;
    PyObject *dependency;
} pgSurfaceObject;
#define pgSurface_AsSurface(x) (((pgSurfaceObject *)x)->surf)
#ifndef PYGAMEAPI_SURFACE_INTERNAL
#define pgSurface_Type \
    (*(PyTypeObject *) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_SURFACE_FIRSTSLOT + 0))

#define pgSurface_Check(x)    \
    (PyObject_IsInstance((x), (PyObject *) &pgSurface_Type))
#if IS_SDLv1
#define pgSurface_New                 \
    (*(PyObject * (*)(SDL_Surface *)) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_SURFACE_FIRSTSLOT + 1))

#else /* IS_SDLv2 */
#define pgSurface_New2                     \
    (*(PyObject * (*)(SDL_Surface *, int)) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_SURFACE_FIRSTSLOT + 1))

#endif /* IS_SDLv2 */
#define pgSurface_Blit                                                \
    (*(int (*)(PyObject *, PyObject *, SDL_Rect *, SDL_Rect *, int))  \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_SURFACE_FIRSTSLOT + 2))

#define import_pygame_surface()                   \
    do {                                          \
        IMPORT_PYGAME_MODULE(surface, SURFACE);   \
        if (PyErr_Occurred() != NULL)             \
            break;                                \
        IMPORT_PYGAME_MODULE(surflock, SURFLOCK); \
    } while (0)

#if IS_SDLv2
#define pgSurface_New(surface) pgSurface_New2((surface), 1)
#define pgSurface_NewNoOwn(surface) pgSurface_New2((surface), 0)
#endif /* IS_SDLv2 */

#endif

/*
 * SURFLOCK module
 * auto imported/initialized by surface
 */
#define PYGAMEAPI_SURFLOCK_FIRSTSLOT \
    (PYGAMEAPI_SURFACE_FIRSTSLOT + PYGAMEAPI_SURFACE_NUMSLOTS)
#define PYGAMEAPI_SURFLOCK_NUMSLOTS 8

#ifndef PYGAMEAPI_SURFLOCK_INTERNAL
#define pgLifetimeLock_Type \
    (*(PyTypeObject *)      \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_SURFLOCK_FIRSTSLOT + 0))

#define pgLifetimeLock_Check(x) \
    ((x)->ob_type == &pgLifetimeLock_Type)
#define pgSurface_Prep(x)                   \
    if (((pgSurfaceObject *)x)->subsurface) \
    (*(*(void (*)(PyObject *)) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_SURFLOCK_FIRSTSLOT + 1)))(x)

#define pgSurface_Unprep(x)                 \
    if (((pgSurfaceObject *)x)->subsurface) \
    (*(*(void (*)(PyObject *)) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_SURFLOCK_FIRSTSLOT + 2)))(x)

#define pgSurface_Lock \
    (*(int (*)(PyObject *)) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_SURFLOCK_FIRSTSLOT + 3))

#define pgSurface_Unlock \
    (*(int (*)(PyObject *)) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_SURFLOCK_FIRSTSLOT + 4))

#define pgSurface_LockBy   \
    (*(int (*)(PyObject *, PyObject *)) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_SURFLOCK_FIRSTSLOT + 5))

#define pgSurface_UnlockBy \
    (*(int (*)(PyObject *, PyObject *)) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_SURFLOCK_FIRSTSLOT + 6))

#define pgSurface_LockLifetime                 \
    (*(PyObject * (*)(PyObject *, PyObject *)) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_SURFLOCK_FIRSTSLOT + 7))
#endif

/*
 * EVENT module
 */
#define PYGAMEAPI_EVENT_FIRSTSLOT \
    (PYGAMEAPI_SURFLOCK_FIRSTSLOT + PYGAMEAPI_SURFLOCK_NUMSLOTS)
#if IS_SDLv1
#define PYGAMEAPI_EVENT_NUMSLOTS 4
#else /* IS_SDLv2 */
#define PYGAMEAPI_EVENT_NUMSLOTS 6
#endif /* IS_SDLv2 */

typedef struct pgEventObject pgEventObject;

#ifndef PYGAMEAPI_EVENT_INTERNAL
#define pgEvent_Type \
    (*(PyTypeObject *) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_EVENT_FIRSTSLOT + 0))

#define pgEvent_Check(x) \
    ((x)->ob_type == &pgEvent_Type)
#define pgEvent_New                 \
    (*(PyObject * (*)(SDL_Event *)) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_EVENT_FIRSTSLOT + 1))

#define pgEvent_New2                    \
    (*(PyObject * (*)(int, PyObject *)) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_EVENT_FIRSTSLOT + 2))

#define pgEvent_FillUserEvent   \
    (*(int (*)(pgEventObject *, SDL_Event *)) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_EVENT_FIRSTSLOT + 3))

#if IS_SDLv2
#define pg_EnableKeyRepeat \
    (*(int (*)(int, int)) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_EVENT_FIRSTSLOT + 4))

#define pg_GetKeyRepeat \
    (*(void (*)(int *, int *)) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_EVENT_FIRSTSLOT + 5))

#endif /* IS_SDLv2 */
#define import_pygame_event() IMPORT_PYGAME_MODULE(event, EVENT)
#endif

/*
 * RWOBJECT module
 * the rwobject are only needed for C side work, not accessable from python.
 */
#define PYGAMEAPI_RWOBJECT_FIRSTSLOT \
    (PYGAMEAPI_EVENT_FIRSTSLOT + PYGAMEAPI_EVENT_NUMSLOTS)
#define PYGAMEAPI_RWOBJECT_NUMSLOTS 6
#ifndef PYGAMEAPI_RWOBJECT_INTERNAL
#define pgRWops_FromObject           \
    (*(SDL_RWops * (*)(PyObject *))  \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_RWOBJECT_FIRSTSLOT + 0))

#define pgRWops_IsFileObject \
    (*(int (*)(SDL_RWops *)) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_RWOBJECT_FIRSTSLOT + 1))

#define pg_EncodeFilePath                       \
    (*(PyObject * (*)(PyObject *, PyObject *))  \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_RWOBJECT_FIRSTSLOT + 2))

#define pg_EncodeString                                                     \
    (*(PyObject * (*)(PyObject *, const char *, const char *, PyObject *))  \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_RWOBJECT_FIRSTSLOT + 3))

#define pgRWops_FromFileObject       \
    (*(SDL_RWops * (*)(PyObject *))  \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_RWOBJECT_FIRSTSLOT + 4))

#define pgRWops_ReleaseObject       \
    (*(int (*)(SDL_RWops *))        \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_RWOBJECT_FIRSTSLOT + 5))

#define import_pygame_rwobject() IMPORT_PYGAME_MODULE(rwobject, RWOBJECT)

#endif

/*
 * PixelArray module
 */
#define PYGAMEAPI_PIXELARRAY_FIRSTSLOT \
    (PYGAMEAPI_RWOBJECT_FIRSTSLOT + PYGAMEAPI_RWOBJECT_NUMSLOTS)
#define PYGAMEAPI_PIXELARRAY_NUMSLOTS 2
#ifndef PYGAMEAPI_PIXELARRAY_INTERNAL
#define PyPixelArray_Type \
    ((PyTypeObject *) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_PIXELARRAY_FIRSTSLOT + 0))

#define PyPixelArray_Check(x) \
    ((x)->ob_type == &PyPixelArray_Type)
#define PyPixelArray_New \
    (*(PyObject * (*))  \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_PIXELARRAY_FIRSTSLOT + 1))

#define import_pygame_pixelarray() IMPORT_PYGAME_MODULE(pixelarray, PIXELARRAY)
#endif /* PYGAMEAPI_PIXELARRAY_INTERNAL */

/*
 * Color module
 */
#define PYGAMEAPI_COLOR_FIRSTSLOT \
    (PYGAMEAPI_PIXELARRAY_FIRSTSLOT + PYGAMEAPI_PIXELARRAY_NUMSLOTS)
#define PYGAMEAPI_COLOR_NUMSLOTS 4
#ifndef PYGAMEAPI_COLOR_INTERNAL
#define pgColor_Type (*(PyObject *) \
    PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_COLOR_FIRSTSLOT))

#define pgColor_Check(x) \
    ((x)->ob_type == &pgColor_Type)
#define pgColor_New \
    (*(PyObject * (*)(Uint8 *))  \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_COLOR_FIRSTSLOT + 1))

#define pgColor_NewLength              \
    (*(PyObject * (*)(Uint8 *, Uint8)) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_COLOR_FIRSTSLOT + 3))

#define pg_RGBAFromColorObj \
    (*(int (*)(PyObject *, Uint8 *)) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_COLOR_FIRSTSLOT + 2))

#define import_pygame_color() IMPORT_PYGAME_MODULE(color, COLOR)
#endif /* PYGAMEAPI_COLOR_INTERNAL */

/*
 * Math module
 */
#define PYGAMEAPI_MATH_FIRSTSLOT \
    (PYGAMEAPI_COLOR_FIRSTSLOT + PYGAMEAPI_COLOR_NUMSLOTS)
#define PYGAMEAPI_MATH_NUMSLOTS 2
#ifndef PYGAMEAPI_MATH_INTERNAL
#define pgVector2_Check(x) \
    ((x)->ob_type == (PyTypeObject *) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_MATH_FIRSTSLOT + 0))

#define pgVector3_Check(x) \
    ((x)->ob_type == (PyTypeObject *) \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_MATH_FIRSTSLOT + 1))
/*
#define pgVector2_New                                             \
    (*(PyObject*(*))  \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, PYGAMEAPI_MATH_FIRSTSLOT + 1))
*/
#define import_pygame_math() IMPORT_PYGAME_MODULE(math, MATH)
#endif /* PYGAMEAPI_MATH_INTERNAL */

#define IMPORT_PYGAME_MODULE(module, MODULE) \
    _IMPORT_PYGAME_MODULE(module, MODULE, PyGAME_C_API)
#define PYGAMEAPI_TOTALSLOTS \
    (PYGAMEAPI_MATH_FIRSTSLOT + PYGAMEAPI_MATH_NUMSLOTS)

/*
 * base pygame API slots
 * disable slots with NO_PYGAME_C_API
 */
#ifdef PYGAME_H
PYGAMEAPI_DEFINE_SLOTS( PyGAME_C_API, PYGAMEAPI_TOTALSLOTS );
#else /* ~PYGAME_H */
PYGAMEAPI_EXTERN_SLOTS( PyGAME_C_API, PYGAMEAPI_TOTALSLOTS );
#endif /* ~PYGAME_H */

#endif /* PYGAME_H */
