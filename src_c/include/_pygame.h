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
 ** Python includes (and SDL.h for functions that use SDL types).
 ** The reason for functions prototyped with #define's is
 ** to allow for maximum Python portability. It also uses
 ** Python as the runtime linker, which allows for late binding.
 '' For more information on this style of development, read
 ** the Python docs on this subject.
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
 **/

#include "pgplatform.h"
#include <Python.h>

/* version macros (defined since version 1.9.5) */
#define PG_MAJOR_VERSION 2
#define PG_MINOR_VERSION 1
#define PG_PATCH_VERSION 3
#define PG_VERSIONNUM(MAJOR, MINOR, PATCH) \
    (1000 * (MAJOR) + 100 * (MINOR) + (PATCH))
#define PG_VERSION_ATLEAST(MAJOR, MINOR, PATCH)                             \
    (PG_VERSIONNUM(PG_MAJOR_VERSION, PG_MINOR_VERSION, PG_PATCH_VERSION) >= \
     PG_VERSIONNUM(MAJOR, MINOR, PATCH))

#include "pgcompat.h"

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

#include "pgimport.h"

/*
 * BASE module
 */
#ifndef PYGAMEAPI_BASE_INTERNAL
#define pgExc_SDLError ((PyObject *)PYGAMEAPI_GET_SLOT(base, 0))

#define pg_RegisterQuit \
    (*(void (*)(void (*)(void)))PYGAMEAPI_GET_SLOT(base, 1))

#define pg_IntFromObj \
    (*(int (*)(PyObject *, int *))PYGAMEAPI_GET_SLOT(base, 2))

#define pg_IntFromObjIndex \
    (*(int (*)(PyObject *, int, int *))PYGAMEAPI_GET_SLOT(base, 3))

#define pg_TwoIntsFromObj \
    (*(int (*)(PyObject *, int *, int *))PYGAMEAPI_GET_SLOT(base, 4))

#define pg_FloatFromObj \
    (*(int (*)(PyObject *, float *))PYGAMEAPI_GET_SLOT(base, 5))

#define pg_FloatFromObjIndex \
    (*(int (*)(PyObject *, int, float *))PYGAMEAPI_GET_SLOT(base, 6))

#define pg_TwoFloatsFromObj \
    (*(int (*)(PyObject *, float *, float *))PYGAMEAPI_GET_SLOT(base, 7))

#define pg_UintFromObj \
    (*(int (*)(PyObject *, Uint32 *))PYGAMEAPI_GET_SLOT(base, 8))

#define pg_UintFromObjIndex \
    (*(int (*)(PyObject *, int, Uint32 *))PYGAMEAPI_GET_SLOT(base, 9))

#define pg_mod_autoinit (*(int (*)(const char *))PYGAMEAPI_GET_SLOT(base, 10))

#define pg_mod_autoquit (*(void (*)(const char *))PYGAMEAPI_GET_SLOT(base, 11))

#define pg_RGBAFromObj \
    (*(int (*)(PyObject *, Uint8 *))PYGAMEAPI_GET_SLOT(base, 12))

#define pgBuffer_AsArrayInterface \
    (*(PyObject * (*)(Py_buffer *)) PYGAMEAPI_GET_SLOT(base, 13))

#define pgBuffer_AsArrayStruct \
    (*(PyObject * (*)(Py_buffer *)) PYGAMEAPI_GET_SLOT(base, 14))

#define pgObject_GetBuffer \
    (*(int (*)(PyObject *, pg_buffer *, int))PYGAMEAPI_GET_SLOT(base, 15))

#define pgBuffer_Release (*(void (*)(pg_buffer *))PYGAMEAPI_GET_SLOT(base, 16))

#define pgDict_AsBuffer \
    (*(int (*)(pg_buffer *, PyObject *, int))PYGAMEAPI_GET_SLOT(base, 17))

#define pgExc_BufferError ((PyObject *)PYGAMEAPI_GET_SLOT(base, 18))

#define pg_GetDefaultWindow \
    (*(SDL_Window * (*)(void)) PYGAMEAPI_GET_SLOT(base, 19))

#define pg_SetDefaultWindow \
    (*(void (*)(SDL_Window *))PYGAMEAPI_GET_SLOT(base, 20))

#define pg_GetDefaultWindowSurface \
    (*(pgSurfaceObject * (*)(void)) PYGAMEAPI_GET_SLOT(base, 21))

#define pg_SetDefaultWindowSurface \
    (*(void (*)(pgSurfaceObject *))PYGAMEAPI_GET_SLOT(base, 22))

#define pg_EnvShouldBlendAlphaSDL2 \
    (*(char *(*)(void))PYGAMEAPI_GET_SLOT(base, 23))

#define import_pygame_base() IMPORT_PYGAME_MODULE(base)
#endif /* ~PYGAMEAPI_BASE_INTERNAL */

typedef struct {
    PyObject_HEAD SDL_Rect r;
    PyObject *weakreflist;
} pgRectObject;

#define pgRect_AsRect(x) (((pgRectObject *)x)->r)
#ifndef PYGAMEAPI_RECT_INTERNAL
#define pgRect_Type (*(PyTypeObject *)PYGAMEAPI_GET_SLOT(rect, 0))

#define pgRect_Check(x) ((x)->ob_type == &pgRect_Type)
#define pgRect_New (*(PyObject * (*)(SDL_Rect *)) PYGAMEAPI_GET_SLOT(rect, 1))

#define pgRect_New4 \
    (*(PyObject * (*)(int, int, int, int)) PYGAMEAPI_GET_SLOT(rect, 2))

#define pgRect_FromObject \
    (*(SDL_Rect * (*)(PyObject *, SDL_Rect *)) PYGAMEAPI_GET_SLOT(rect, 3))

#define pgRect_Normalize (*(void (*)(SDL_Rect *))PYGAMEAPI_GET_SLOT(rect, 4))

#define import_pygame_rect() IMPORT_PYGAME_MODULE(rect)
#endif /* ~PYGAMEAPI_RECT_INTERNAL */

/*
 * JOYSTICK module
 */
typedef struct pgJoystickObject {
    PyObject_HEAD int id;
    SDL_Joystick *joy;

    /* Joysticks form an intrusive linked list.
     *
     * Note that we don't maintain refcounts for these so they are weakrefs
     * from the Python side.
     */
    struct pgJoystickObject *next;
    struct pgJoystickObject *prev;
} pgJoystickObject;

#define pgJoystick_AsID(x) (((pgJoystickObject *)x)->id)
#define pgJoystick_AsSDL(x) (((pgJoystickObject *)x)->joy)

#ifndef PYGAMEAPI_JOYSTICK_INTERNAL
#define pgJoystick_Type (*(PyTypeObject *)PYGAMEAPI_GET_SLOT(joystick, 0))

#define pgJoystick_Check(x) ((x)->ob_type == &pgJoystick_Type)
#define pgJoystick_New (*(PyObject * (*)(int)) PYGAMEAPI_GET_SLOT(joystick, 1))

#define import_pygame_joystick() IMPORT_PYGAME_MODULE(joystick)
#endif

/*
 * DISPLAY module
 */

typedef struct {
    Uint32 hw_available : 1;
    Uint32 wm_available : 1;
    Uint32 blit_hw : 1;
    Uint32 blit_hw_CC : 1;
    Uint32 blit_hw_A : 1;
    Uint32 blit_sw : 1;
    Uint32 blit_sw_CC : 1;
    Uint32 blit_sw_A : 1;
    Uint32 blit_fill : 1;
    Uint32 video_mem;
    SDL_PixelFormat *vfmt;
    SDL_PixelFormat vfmt_data;
    int current_w;
    int current_h;
} pg_VideoInfo;

typedef struct {
    PyObject_HEAD pg_VideoInfo info;
} pgVidInfoObject;

#define pgVidInfo_AsVidInfo(x) (((pgVidInfoObject *)x)->info)

#ifndef PYGAMEAPI_DISPLAY_INTERNAL
#define pgVidInfo_Type (*(PyTypeObject *)PYGAMEAPI_GET_SLOT(display, 0))

#define pgVidInfo_Check(x) ((x)->ob_type == &pgVidInfo_Type)
#define pgVidInfo_New \
    (*(PyObject * (*)(pg_VideoInfo *)) PYGAMEAPI_GET_SLOT(display, 1))

#define import_pygame_display() IMPORT_PYGAME_MODULE(display)
#endif /* ~PYGAMEAPI_DISPLAY_INTERNAL */

/*
 * SURFACE module
 */
struct pgSubSurface_Data;
struct SDL_Surface;

typedef struct {
    PyObject_HEAD struct SDL_Surface *surf;
    int owner;
    struct pgSubSurface_Data *subsurface; /* ptr to subsurface data (if a
                                           * subsurface)*/
    PyObject *weakreflist;
    PyObject *locklist;
    PyObject *dependency;
} pgSurfaceObject;
#define pgSurface_AsSurface(x) (((pgSurfaceObject *)x)->surf)

#ifndef PYGAMEAPI_SURFACE_INTERNAL
#define pgSurface_Type (*(PyTypeObject *)PYGAMEAPI_GET_SLOT(surface, 0))

#define pgSurface_Check(x) \
    (PyObject_IsInstance((x), (PyObject *)&pgSurface_Type))
#define pgSurface_New2                            \
    (*(pgSurfaceObject * (*)(SDL_Surface *, int)) \
         PYGAMEAPI_GET_SLOT(surface, 1))

#define pgSurface_SetSurface                                              \
    (*(int (*)(pgSurfaceObject *, SDL_Surface *, int))PYGAMEAPI_GET_SLOT( \
        surface, 3))

#define pgSurface_Blit                                                       \
    (*(int (*)(pgSurfaceObject *, pgSurfaceObject *, SDL_Rect *, SDL_Rect *, \
               int))PYGAMEAPI_GET_SLOT(surface, 2))

#define import_pygame_surface()         \
    do {                                \
        IMPORT_PYGAME_MODULE(surface);  \
        if (PyErr_Occurred() != NULL)   \
            break;                      \
        IMPORT_PYGAME_MODULE(surflock); \
    } while (0)

#define pgSurface_New(surface) pgSurface_New2((surface), 1)
#define pgSurface_NewNoOwn(surface) pgSurface_New2((surface), 0)

#endif /* ~PYGAMEAPI_SURFACE_INTERNAL */

/*
 * SURFLOCK module
 * auto imported/initialized by surface
 */
#ifndef PYGAMEAPI_SURFLOCK_INTERNAL
#define pgLifetimeLock_Type (*(PyTypeObject *)PYGAMEAPI_GET_SLOT(surflock, 0))

#define pgLifetimeLock_Check(x) ((x)->ob_type == &pgLifetimeLock_Type)

#define pgSurface_Prep(x) \
    if ((x)->subsurface)  \
    (*(*(void (*)(pgSurfaceObject *))PYGAMEAPI_GET_SLOT(surflock, 1)))(x)

#define pgSurface_Unprep(x) \
    if ((x)->subsurface)    \
    (*(*(void (*)(pgSurfaceObject *))PYGAMEAPI_GET_SLOT(surflock, 2)))(x)

#define pgSurface_Lock \
    (*(int (*)(pgSurfaceObject *))PYGAMEAPI_GET_SLOT(surflock, 3))

#define pgSurface_Unlock \
    (*(int (*)(pgSurfaceObject *))PYGAMEAPI_GET_SLOT(surflock, 4))

#define pgSurface_LockBy \
    (*(int (*)(pgSurfaceObject *, PyObject *))PYGAMEAPI_GET_SLOT(surflock, 5))

#define pgSurface_UnlockBy \
    (*(int (*)(pgSurfaceObject *, PyObject *))PYGAMEAPI_GET_SLOT(surflock, 6))

#define pgSurface_LockLifetime \
    (*(PyObject * (*)(PyObject *, PyObject *)) PYGAMEAPI_GET_SLOT(surflock, 7))
#endif

/*
 * EVENT module
 */
typedef struct pgEventObject pgEventObject;

#ifndef PYGAMEAPI_EVENT_INTERNAL
#define pgEvent_Type (*(PyTypeObject *)PYGAMEAPI_GET_SLOT(event, 0))

#define pgEvent_Check(x) ((x)->ob_type == &pgEvent_Type)

#define pgEvent_New \
    (*(PyObject * (*)(SDL_Event *)) PYGAMEAPI_GET_SLOT(event, 1))

#define pgEvent_New2 \
    (*(PyObject * (*)(int, PyObject *)) PYGAMEAPI_GET_SLOT(event, 2))

#define pgEvent_FillUserEvent \
    (*(int (*)(pgEventObject *, SDL_Event *))PYGAMEAPI_GET_SLOT(event, 3))

#define pg_EnableKeyRepeat (*(int (*)(int, int))PYGAMEAPI_GET_SLOT(event, 4))

#define pg_GetKeyRepeat (*(void (*)(int *, int *))PYGAMEAPI_GET_SLOT(event, 5))

#define import_pygame_event() IMPORT_PYGAME_MODULE(event)
#endif

/*
 * RWOBJECT module
 * the rwobject are only needed for C side work, not accessible from python.
 */
#ifndef PYGAMEAPI_RWOBJECT_INTERNAL
#define pgRWops_FromObject \
    (*(SDL_RWops * (*)(PyObject *)) PYGAMEAPI_GET_SLOT(rwobject, 0))

#define pgRWops_IsFileObject \
    (*(int (*)(SDL_RWops *))PYGAMEAPI_GET_SLOT(rwobject, 1))

#define pg_EncodeFilePath \
    (*(PyObject * (*)(PyObject *, PyObject *)) PYGAMEAPI_GET_SLOT(rwobject, 2))

#define pg_EncodeString                                                    \
    (*(PyObject * (*)(PyObject *, const char *, const char *, PyObject *)) \
         PYGAMEAPI_GET_SLOT(rwobject, 3))

#define pgRWops_FromFileObject \
    (*(SDL_RWops * (*)(PyObject *)) PYGAMEAPI_GET_SLOT(rwobject, 4))

#define pgRWops_ReleaseObject \
    (*(int (*)(SDL_RWops *))PYGAMEAPI_GET_SLOT(rwobject, 5))

#define pgRWops_GetFileExtension \
    (*(char *(*)(SDL_RWops *))PYGAMEAPI_GET_SLOT(rwobject, 6))

#define import_pygame_rwobject() IMPORT_PYGAME_MODULE(rwobject)

#endif

/*
 * PixelArray module
 */
#ifndef PYGAMEAPI_PIXELARRAY_INTERNAL
#define PyPixelArray_Type ((PyTypeObject *)PYGAMEAPI_GET_SLOT(pixelarray, 0))

#define PyPixelArray_Check(x) ((x)->ob_type == &PyPixelArray_Type)
#define PyPixelArray_New (*(PyObject * (*)) PYGAMEAPI_GET_SLOT(pixelarray, 1))

#define import_pygame_pixelarray() IMPORT_PYGAME_MODULE(pixelarray)
#endif /* PYGAMEAPI_PIXELARRAY_INTERNAL */

/*
 * Color module
 */
typedef struct pgColorObject pgColorObject;

#ifndef PYGAMEAPI_COLOR_INTERNAL
#define pgColor_Type (*(PyObject *)PYGAMEAPI_GET_SLOT(color, 0))

#define pgColor_Check(x) ((x)->ob_type == &pgColor_Type)
#define pgColor_New (*(PyObject * (*)(Uint8 *)) PYGAMEAPI_GET_SLOT(color, 1))

#define pgColor_NewLength \
    (*(PyObject * (*)(Uint8 *, Uint8)) PYGAMEAPI_GET_SLOT(color, 3))

#define pg_RGBAFromColorObj \
    (*(int (*)(PyObject *, Uint8 *))PYGAMEAPI_GET_SLOT(color, 2))

#define pg_RGBAFromFuzzyColorObj \
    (*(int (*)(PyObject *, Uint8 *))PYGAMEAPI_GET_SLOT(color, 4))

#define pgColor_AsArray(x) (((pgColorObject *)x)->data)
#define pgColor_NumComponents(x) (((pgColorObject *)x)->len)

#define import_pygame_color() IMPORT_PYGAME_MODULE(color)
#endif /* PYGAMEAPI_COLOR_INTERNAL */

/*
 * Math module
 */
#ifndef PYGAMEAPI_MATH_INTERNAL
#define pgVector2_Check(x) \
    ((x)->ob_type == (PyTypeObject *)PYGAMEAPI_GET_SLOT(math, 0))

#define pgVector3_Check(x) \
    ((x)->ob_type == (PyTypeObject *)PYGAMEAPI_GET_SLOT(math, 1))
/*
#define pgVector2_New                                             \
    (*(PyObject*(*))  \
        PYGAMEAPI_GET_SLOT(PyGAME_C_API, 1))
*/
#define import_pygame_math() IMPORT_PYGAME_MODULE(math)
#endif /* PYGAMEAPI_MATH_INTERNAL */

#define IMPORT_PYGAME_MODULE _IMPORT_PYGAME_MODULE

/*
 * base pygame API slots
 * disable slots with NO_PYGAME_C_API
 */
#ifdef PYGAME_H
PYGAMEAPI_DEFINE_SLOTS(base);
PYGAMEAPI_DEFINE_SLOTS(rect);
PYGAMEAPI_DEFINE_SLOTS(cdrom);
PYGAMEAPI_DEFINE_SLOTS(joystick);
PYGAMEAPI_DEFINE_SLOTS(display);
PYGAMEAPI_DEFINE_SLOTS(surface);
PYGAMEAPI_DEFINE_SLOTS(surflock);
PYGAMEAPI_DEFINE_SLOTS(event);
PYGAMEAPI_DEFINE_SLOTS(rwobject);
PYGAMEAPI_DEFINE_SLOTS(pixelarray);
PYGAMEAPI_DEFINE_SLOTS(color);
PYGAMEAPI_DEFINE_SLOTS(math);
#else  /* ~PYGAME_H */
PYGAMEAPI_EXTERN_SLOTS(base);
PYGAMEAPI_EXTERN_SLOTS(rect);
PYGAMEAPI_EXTERN_SLOTS(cdrom);
PYGAMEAPI_EXTERN_SLOTS(joystick);
PYGAMEAPI_EXTERN_SLOTS(display);
PYGAMEAPI_EXTERN_SLOTS(surface);
PYGAMEAPI_EXTERN_SLOTS(surflock);
PYGAMEAPI_EXTERN_SLOTS(event);
PYGAMEAPI_EXTERN_SLOTS(rwobject);
PYGAMEAPI_EXTERN_SLOTS(pixelarray);
PYGAMEAPI_EXTERN_SLOTS(color);
PYGAMEAPI_EXTERN_SLOTS(math);
#endif /* ~PYGAME_H */

#endif /* PYGAME_H */
