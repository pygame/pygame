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
#ifndef PG_MAJOR_VERSION
#define PG_MAJOR_VERSION 2
#endif

#ifndef PG_MINOR_VERSION
#define PG_MINOR_VERSION 6
#endif

#ifndef PG_PATCH_VERSION
#define PG_PATCH_VERSION 1
#endif

#ifndef PG_VERSION_TAG
#define PG_VERSION_TAG ""
#endif

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
#include "../pgcompat_rect.h"

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

#define pg_DoubleFromObj \
    (*(int (*)(PyObject *, double *))PYGAMEAPI_GET_SLOT(base, 24))

#define pg_TwoDoublesFromObj \
    (*(int (*)(PyObject *, double *, double *))PYGAMEAPI_GET_SLOT(base, 25))

#define pg_TwoDoublesFromFastcallArgs                   \
    (*(int (*)(PyObject *const *, Py_ssize_t, double *, \
               double *))PYGAMEAPI_GET_SLOT(base, 26))

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
    (*(int (*)(void))PYGAMEAPI_GET_SLOT(base, 23))

#define pg_GetDefaultConvertFormat \
    (*(PG_PixelFormatEnum (*)(void))PYGAMEAPI_GET_SLOT(base, 27))

#define pg_SetDefaultConvertFormat \
    (*(void (*)(Uint32))PYGAMEAPI_GET_SLOT(base, 28))

#define pgObject_getRectHelper                                               \
    (*(PyObject * (*)(PyObject *, PyObject *const *, Py_ssize_t, PyObject *, \
                      char *)) PYGAMEAPI_GET_SLOT(base, 29))

#define import_pygame_base() IMPORT_PYGAME_MODULE(base)
#endif /* ~PYGAMEAPI_BASE_INTERNAL */

typedef struct {
    PyObject_HEAD SDL_Rect r;
    PyObject *weakreflist;
} pgRectObject;

typedef struct {
    PyObject_HEAD SDL_FRect r;
    PyObject *weakreflist;
} pgFRectObject;

#define pgRect_AsRect(x) (((pgRectObject *)x)->r)
#define pgFRect_AsRect(x) (((pgFRectObject *)x)->r)

#ifndef PYGAMEAPI_RECT_INTERNAL
#define pgRect_Type (*(PyTypeObject *)PYGAMEAPI_GET_SLOT(rect, 0))

#define pgRect_Check(x) ((x)->ob_type == &pgRect_Type)
#define pgRect_New (*(PyObject * (*)(SDL_Rect *)) PYGAMEAPI_GET_SLOT(rect, 1))

#define pgRect_New4 \
    (*(PyObject * (*)(int, int, int, int)) PYGAMEAPI_GET_SLOT(rect, 2))

#define pgRect_FromObject \
    (*(SDL_Rect * (*)(PyObject *, SDL_Rect *)) PYGAMEAPI_GET_SLOT(rect, 3))

#define pgRect_Normalize (*(void (*)(SDL_Rect *))PYGAMEAPI_GET_SLOT(rect, 4))

#define pgFRect_Type (*(PyTypeObject *)PYGAMEAPI_GET_SLOT(rect, 5))

#define pgFRect_Check(x) ((x)->ob_type == &pgFRect_Type)
#define pgFRect_New \
    (*(PyObject * (*)(SDL_FRect *)) PYGAMEAPI_GET_SLOT(rect, 6))

#define pgFRect_New4 \
    (*(PyObject * (*)(float, float, float, float)) PYGAMEAPI_GET_SLOT(rect, 7))

#define pgFRect_FromObject \
    (*(SDL_FRect * (*)(PyObject *, SDL_FRect *)) PYGAMEAPI_GET_SLOT(rect, 8))

#define pgFRect_Normalize (*(void (*)(SDL_FRect *))PYGAMEAPI_GET_SLOT(rect, 9))

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
#define pgJoystick_GetDeviceIndexByInstanceID \
    (*(int (*)(int))PYGAMEAPI_GET_SLOT(joystick, 2))

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
#if PG_SDL3
    /* We cannot use PG_PixelFormat here because it aliases to const */
    SDL_PixelFormatDetails *vfmt;
    SDL_PixelFormatDetails vfmt_data;
#else
    SDL_PixelFormat *vfmt;
    SDL_PixelFormat vfmt_data;
#endif
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
#define pgSurface_Prep(x) \
    if ((x)->subsurface)  \
    (*(*(void (*)(pgSurfaceObject *))PYGAMEAPI_GET_SLOT(surflock, 0)))(x)

#define pgSurface_Unprep(x) \
    if ((x)->subsurface)    \
    (*(*(void (*)(pgSurfaceObject *))PYGAMEAPI_GET_SLOT(surflock, 1)))(x)

#define pgSurface_Lock \
    (*(int (*)(pgSurfaceObject *))PYGAMEAPI_GET_SLOT(surflock, 2))

#define pgSurface_Unlock \
    (*(int (*)(pgSurfaceObject *))PYGAMEAPI_GET_SLOT(surflock, 3))

#define pgSurface_LockBy \
    (*(int (*)(pgSurfaceObject *, PyObject *))PYGAMEAPI_GET_SLOT(surflock, 4))

#define pgSurface_UnlockBy \
    (*(int (*)(pgSurfaceObject *, PyObject *))PYGAMEAPI_GET_SLOT(surflock, 5))
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

#define pg_post_event \
    (*(int (*)(Uint32, PyObject *))PYGAMEAPI_GET_SLOT(event, 2))

#define pg_post_event_dictproxy \
    (*(int (*)(Uint32, pgEventDictProxy *))PYGAMEAPI_GET_SLOT(event, 3))

#define pg_EnableKeyRepeat (*(int (*)(int, int))PYGAMEAPI_GET_SLOT(event, 4))

#define pg_GetKeyRepeat (*(void (*)(int *, int *))PYGAMEAPI_GET_SLOT(event, 5))

#define pgEvent_GetKeyDownInfo (*(char *(*)(void))PYGAMEAPI_GET_SLOT(event, 6))

#define pgEvent_GetKeyUpInfo (*(char *(*)(void))PYGAMEAPI_GET_SLOT(event, 7))

#define pgEvent_GetMouseButtonDownInfo \
    (*(char *(*)(void))PYGAMEAPI_GET_SLOT(event, 8))

#define pgEvent_GetMouseButtonUpInfo \
    (*(char *(*)(void))PYGAMEAPI_GET_SLOT(event, 9))

#define import_pygame_event() IMPORT_PYGAME_MODULE(event)
#endif

/*
 * RWOBJECT module
 * the rwobject are only needed for C side work, not accessible from python.
 */
#ifndef PYGAMEAPI_RWOBJECT_INTERNAL
#define pgRWops_FromObject \
    (*(SDL_RWops * (*)(PyObject *, char **)) PYGAMEAPI_GET_SLOT(rwobject, 0))

#define pgRWops_IsFileObject \
    (*(int (*)(SDL_RWops *))PYGAMEAPI_GET_SLOT(rwobject, 1))

#define pg_EncodeFilePath \
    (*(PyObject * (*)(PyObject *, PyObject *)) PYGAMEAPI_GET_SLOT(rwobject, 2))

#define pg_EncodeString                                                    \
    (*(PyObject * (*)(PyObject *, const char *, const char *, PyObject *)) \
         PYGAMEAPI_GET_SLOT(rwobject, 3))

#define pgRWops_FromFileObject \
    (*(SDL_RWops * (*)(PyObject *)) PYGAMEAPI_GET_SLOT(rwobject, 4))

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
 * BufferProxy module
 */
typedef struct {
    PyObject_HEAD PyObject *obj; /* Wrapped object (parent)     */
    pg_buffer *pg_view_p;        /* For array interface export  */
    getbufferproc get_buffer;    /* pg_buffer get callback      */
    PyObject *dict;              /* Allow arbitrary attributes  */
    PyObject *weakrefs;          /* Reference cycles can happen */
} pgBufferProxyObject;

#ifndef PYGAMEAPI_BUFFERPROXY_INTERNAL
#define pgBufferProxy_Type \
    (*(PyTypeObject *)PYGAMEAPI_GET_SLOT(bufferproxy, 0))

#define pgBufferProxy_Check(x) ((x)->ob_type == &pgBufferProxy_Type)
#define pgBufferProxy_New                         \
    (*(PyObject * (*)(PyObject *, getbufferproc)) \
         PYGAMEAPI_GET_SLOT(bufferproxy, 1))

#define pgBufferProxy_GetParent \
    (*(PyObject * (*)(PyObject *)) PYGAMEAPI_GET_SLOT(bufferproxy, 2))
#define pgBufferProxy_Trip \
    (*(int (*)(PyObject *))PYGAMEAPI_GET_SLOT(bufferproxy, 3))

#define import_pygame_bufferproxy() _IMPORT_PYGAME_MODULE(bufferproxy)
#endif /* ~PYGAMEAPI_BUFFERPROXY_INTERNAL */

/*
 * Color module
 */
typedef struct pgColorObject pgColorObject;

#ifndef PYGAMEAPI_COLOR_INTERNAL
#define pgColor_Type (*(PyObject *)PYGAMEAPI_GET_SLOT(color, 0))

#define pgColor_CheckExact(x) ((x)->ob_type == &pgColor_Type)
#define pgColor_New (*(PyObject * (*)(Uint8 *)) PYGAMEAPI_GET_SLOT(color, 1))

#define pgColor_NewLength \
    (*(PyObject * (*)(Uint8 *, Uint8)) PYGAMEAPI_GET_SLOT(color, 3))

#define pg_RGBAFromObjEx                                                    \
    (*(int (*)(PyObject *, Uint8 *, pgColorHandleFlags))PYGAMEAPI_GET_SLOT( \
        color, 2))

#define pg_MappedColorFromObj                       \
    (*(int (*)(PyObject *, SDL_Surface *, Uint32 *, \
               pgColorHandleFlags))PYGAMEAPI_GET_SLOT(color, 4))

#define pgColor_AsArray(x) (((pgColorObject *)x)->data)
#define pgColor_NumComponents(x) (((pgColorObject *)x)->len)

#define import_pygame_color() IMPORT_PYGAME_MODULE(color)
#endif /* PYGAMEAPI_COLOR_INTERNAL */

/*
 * Geometry module
 */
#ifndef PYGAMEAPI_GEOMETRY_INTERNAL
#define import_pygame_geometry() IMPORT_PYGAME_MODULE(geometry)
#endif /* ~PYGAMEAPI_GEOMETRY_INTERNAL */

/*
 * Window module
 */
typedef struct {
    PyObject_HEAD SDL_Window *_win;
    SDL_bool _is_borrowed;
    pgSurfaceObject *surf;
    SDL_GLContext context;
} pgWindowObject;

#ifndef PYGAMEAPI_WINDOW_INTERNAL
#define pgWindow_Type (*(PyTypeObject *)PYGAMEAPI_GET_SLOT(window, 0))
#define pgWindow_Check(x) \
    (PyObject_IsInstance((x), (PyObject *)&pgWindow_Type))
#define import_pygame_window() IMPORT_PYGAME_MODULE(window)
#endif

typedef struct pgTextureObject pgTextureObject;

/*
 * Render module
 */
typedef struct {
    PyObject_HEAD SDL_Renderer *renderer;
    pgWindowObject *window;
    pgTextureObject *target;
    SDL_bool _is_borrowed;
} pgRendererObject;

struct pgTextureObject {
    PyObject_HEAD SDL_Texture *texture;
    pgRendererObject *renderer;
    int width;
    int height;
    PyObject *weakreflist;
};

typedef struct {
    PyObject_HEAD pgTextureObject *texture;
    pgRectObject *srcrect;
    pgColorObject *color;
    float angle;
    float alpha;
    SDL_bool has_origin;
    SDL_FPoint origin;
    SDL_bool flip_x;
    SDL_bool flip_y;
    SDL_BlendMode blend_mode;
} pgImageObject;

#ifndef PYGAMEAPI_RENDER_INTERNAL
#define pgRenderer_Type (*(PyTypeObject *)PYGAMEAPI_GET_SLOT(_render, 0))
#define pgTexture_Type (*(PyTypeObject *)PYGAMEAPI_GET_SLOT(_render, 1))
#define pgImage_Type (*(PyTypeObject *)PYGAMEAPI_GET_SLOT(_render, 2))
#define pgRenderer_Check(x) \
    (PyObject_IsInstance((x), (PyObject *)&pgRender_Type))
#define pgTexture_Check(x) \
    (PyObject_IsInstance((x), (PyObject *)&pgTexture_Type))
#define pgImage_Check(x) (PyObject_IsInstance((x), (PyObject *)&pgImage_Type))
#define import_pygame_render() IMPORT_PYGAME_MODULE(_render)
#endif

#define IMPORT_PYGAME_MODULE _IMPORT_PYGAME_MODULE

/*
 * base pygame API slots
 * disable slots with NO_PYGAME_C_API
 */
#ifdef PYGAME_H
PYGAMEAPI_DEFINE_SLOTS(base);
PYGAMEAPI_DEFINE_SLOTS(rect);
PYGAMEAPI_DEFINE_SLOTS(joystick);
PYGAMEAPI_DEFINE_SLOTS(display);
PYGAMEAPI_DEFINE_SLOTS(surface);
PYGAMEAPI_DEFINE_SLOTS(surflock);
PYGAMEAPI_DEFINE_SLOTS(event);
PYGAMEAPI_DEFINE_SLOTS(rwobject);
PYGAMEAPI_DEFINE_SLOTS(pixelarray);
PYGAMEAPI_DEFINE_SLOTS(bufferproxy);
PYGAMEAPI_DEFINE_SLOTS(color);
PYGAMEAPI_DEFINE_SLOTS(math);
PYGAMEAPI_DEFINE_SLOTS(window);
PYGAMEAPI_DEFINE_SLOTS(_render);
PYGAMEAPI_DEFINE_SLOTS(geometry);
#else /* ~PYGAME_H */
PYGAMEAPI_EXTERN_SLOTS(base);
PYGAMEAPI_EXTERN_SLOTS(rect);
PYGAMEAPI_EXTERN_SLOTS(joystick);
PYGAMEAPI_EXTERN_SLOTS(display);
PYGAMEAPI_EXTERN_SLOTS(surface);
PYGAMEAPI_EXTERN_SLOTS(surflock);
PYGAMEAPI_EXTERN_SLOTS(event);
PYGAMEAPI_EXTERN_SLOTS(rwobject);
PYGAMEAPI_EXTERN_SLOTS(pixelarray);
PYGAMEAPI_EXTERN_SLOTS(bufferproxy);
PYGAMEAPI_EXTERN_SLOTS(color);
PYGAMEAPI_EXTERN_SLOTS(math);
PYGAMEAPI_EXTERN_SLOTS(window);
PYGAMEAPI_EXTERN_SLOTS(_render);
PYGAMEAPI_EXTERN_SLOTS(geometry);

#endif /* ~PYGAME_H */

#endif /* PYGAME_H */

/*  Use the end of this file for other cross module inline utility
 *  functions There seems to be no good reason to stick to macro only
 *  functions in Python 3.
 */

#define SURF_INIT_CHECK(surf)                                           \
    {                                                                   \
        if (!surf) {                                                    \
            return RAISE(pgExc_SDLError, "Surface is not initialized"); \
        }                                                               \
    }

static PG_INLINE PyObject *
pg_tuple_couple_from_values_int(int val1, int val2)
{
    /* This function turns two input integers into a python tuple object.
     * Currently, 5th November 2022, this is faster than using Py_BuildValue
     * to do the same thing.
     */
    PyObject *tup = PyTuple_New(2);
    if (!tup) {
        return NULL;
    }

    PyObject *tmp = PyLong_FromLong(val1);
    if (!tmp) {
        Py_DECREF(tup);
        return NULL;
    }
    PyTuple_SET_ITEM(tup, 0, tmp);

    tmp = PyLong_FromLong(val2);
    if (!tmp) {
        Py_DECREF(tup);
        return NULL;
    }
    PyTuple_SET_ITEM(tup, 1, tmp);

    return tup;
}

static PG_INLINE PyObject *
pg_tuple_triple_from_values_int(int val1, int val2, int val3)
{
    /* This function turns three input integers into a python tuple object.
     * Currently, 5th November 2022, this is faster than using Py_BuildValue
     * to do the same thing.
     */
    PyObject *tup = PyTuple_New(3);
    if (!tup) {
        return NULL;
    }

    PyObject *tmp = PyLong_FromLong(val1);
    if (!tmp) {
        Py_DECREF(tup);
        return NULL;
    }
    PyTuple_SET_ITEM(tup, 0, tmp);

    tmp = PyLong_FromLong(val2);
    if (!tmp) {
        Py_DECREF(tup);
        return NULL;
    }
    PyTuple_SET_ITEM(tup, 1, tmp);

    tmp = PyLong_FromLong(val3);
    if (!tmp) {
        Py_DECREF(tup);
        return NULL;
    }
    PyTuple_SET_ITEM(tup, 2, tmp);

    return tup;
}

static PG_INLINE PyObject *
pg_tuple_couple_from_values_double(double val1, double val2)
{
    /* This function turns two input doubles into a python tuple object.
     * Currently, 5th November 2022, this is faster than using Py_BuildValue
     * to do the same thing.
     */
    PyObject *tuple = PyTuple_New(2);
    if (!tuple) {
        return NULL;
    }

    PyObject *tmp = PyFloat_FromDouble(val1);
    if (!tmp) {
        Py_DECREF(tuple);
        return NULL;
    }
    PyTuple_SET_ITEM(tuple, 0, tmp);

    tmp = PyFloat_FromDouble(val2);
    if (!tmp) {
        Py_DECREF(tuple);
        return NULL;
    }
    PyTuple_SET_ITEM(tuple, 1, tmp);

    return tuple;
}

static PG_INLINE PyObject *
pg_PointList_FromArrayDouble(double const *array, int arr_length)
{
    if (arr_length % 2) {
        return RAISE(PyExc_ValueError, "array length must be even");
    }

    int num_points = arr_length / 2;
    PyObject *sequence = PyList_New(num_points);
    if (!sequence) {
        return NULL;
    }

    int i;
    PyObject *point = NULL;
    for (i = 0; i < num_points; i++) {
        point =
            pg_tuple_couple_from_values_double(array[i * 2], array[i * 2 + 1]);
        if (!point) {
            Py_DECREF(sequence);
            return NULL;
        }
        PyList_SET_ITEM(sequence, i, point);
        point = NULL;
    }

    return sequence;
}

#if PY_VERSION_HEX >= 0x030C0000

#define PG_DECLARE_EXCEPTION_SAVER static PyObject *__cached_exception = NULL;

#define PG_SAVE_EXCEPTION __cached_exception = PyErr_GetRaisedException();

#define PG_UNSAVE_EXCEPTION                       \
    PyErr_SetRaisedException(__cached_exception); \
    __cached_exception = NULL;

#else

#define PG_DECLARE_EXCEPTION_SAVER                    \
    static PyObject *__cached_exception_type = NULL;  \
    static PyObject *__cached_exception_value = NULL; \
    static PyObject *__cached_exception_traceback = NULL;

#define PG_SAVE_EXCEPTION                                            \
    PyErr_Fetch(&__cached_exception_type, &__cached_exception_value, \
                &__cached_exception_traceback);

#define PG_UNSAVE_EXCEPTION                                          \
    PyErr_Restore(__cached_exception_type, __cached_exception_value, \
                  __cached_exception_traceback);                     \
    __cached_exception_type = NULL;                                  \
    __cached_exception_value = NULL;                                 \
    __cached_exception_traceback = NULL;

#endif
