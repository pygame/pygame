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

#ifndef PYGAME_H
#define PYGAME_H

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
 ** the SDL libarary, it doesn't need to offer a lot either.
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
#if defined(HAVE_SNPRINTF)  /* defined in python.h (pyerrors.h) and SDL.h (SDL_config.h) */
#undef HAVE_SNPRINTF        /* remove GCC redefine warning */
#endif

// This must be before all else
#if defined(__SYMBIAN32__) && defined( OPENC )
#include <sys/types.h>

#if defined(__WINS__)
void* _alloca(size_t size);
#  define alloca _alloca
#endif

#endif

/* This is unconditionally defined in Python.h */
#if defined(_POSIX_C_SOURCE)
#undef _POSIX_C_SOURCE
#endif

#include <Python.h>

// No signal()
#if defined(__SYMBIAN32__) && defined(HAVE_SIGNAL_H)
#undef HAVE_SIGNAL_H
#endif

#if defined(HAVE_SNPRINTF)
#undef HAVE_SNPRINTF
#endif

#ifdef MS_WIN32 /*Python gives us MS_WIN32, SDL needs just WIN32*/
#ifndef WIN32
#define WIN32
#endif
#endif


/// Prefix when initializing module
#define MODPREFIX ""
/// Prefix when importing module
#define IMPPREFIX "pygame."

#ifdef __SYMBIAN32__
#undef MODPREFIX
#undef IMPPREFIX
// On Symbian there is no pygame package. The extensions are built-in or in sys\bin.
#define MODPREFIX "pygame_"
#define IMPPREFIX "pygame_"
#endif

#include <SDL.h>

/* macros used throughout the source */
#define RAISE(x,y) (PyErr_SetString((x), (y)), (PyObject*)NULL)

#if PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION == 3
#  define Py_RETURN_NONE return Py_INCREF(Py_None), Py_None
#  define Py_RETURN_TRUE return Py_INCREF(Py_True), Py_True
#  define Py_RETURN_FALSE return Py_INCREF(Py_False), Py_False
#endif

/* Py_ssize_t availability. */
#if PY_VERSION_HEX < 0x02050000 && !defined(PY_SSIZE_T_MIN)
typedef int Py_ssize_t;
#define PY_SSIZE_T_MAX INT_MAX
#define PY_SSIZE_T_MIN INT_MIN
typedef inquiry lenfunc;
typedef intargfunc ssizeargfunc;
typedef intobjargproc ssizeobjargproc;
typedef intintargfunc ssizessizeargfunc;
typedef intintobjargproc ssizessizeobjargproc;
typedef getreadbufferproc readbufferproc;
typedef getwritebufferproc writebufferproc;
typedef getsegcountproc segcountproc;
typedef getcharbufferproc charbufferproc;
#endif

#define PyType_Init(x) (((x).ob_type) = &PyType_Type)
#define PYGAMEAPI_LOCAL_ENTRY "_PYGAME_C_API"

#ifndef MIN
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a,b) ( (a) > (b) ? (a) : (b))
#endif

#ifndef ABS
#define ABS(a) (((a) < 0) ? -(a) : (a))
#endif

/* test sdl initializations */
#define VIDEO_INIT_CHECK()                                              \
    if(!SDL_WasInit(SDL_INIT_VIDEO))                                    \
        return RAISE(PyExc_SDLError, "video system not initialized")

#define CDROM_INIT_CHECK()                                              \
    if(!SDL_WasInit(SDL_INIT_CDROM))                                    \
        return RAISE(PyExc_SDLError, "cdrom system not initialized")

#define JOYSTICK_INIT_CHECK()                                           \
    if(!SDL_WasInit(SDL_INIT_JOYSTICK))                                 \
        return RAISE(PyExc_SDLError, "joystick system not initialized")

/* BASE */
#define PYGAMEAPI_BASE_FIRSTSLOT 0
#define PYGAMEAPI_BASE_NUMSLOTS 13
#ifndef PYGAMEAPI_BASE_INTERNAL
#define PyExc_SDLError ((PyObject*)PyGAME_C_API[PYGAMEAPI_BASE_FIRSTSLOT])

#define PyGame_RegisterQuit                                             \
    (*(void(*)(void(*)(void)))PyGAME_C_API[PYGAMEAPI_BASE_FIRSTSLOT + 1])

#define IntFromObj                                                      \
    (*(int(*)(PyObject*, int*))PyGAME_C_API[PYGAMEAPI_BASE_FIRSTSLOT + 2])

#define IntFromObjIndex                                                 \
    (*(int(*)(PyObject*, int, int*))PyGAME_C_API[PYGAMEAPI_BASE_FIRSTSLOT + 3])

#define TwoIntsFromObj                                                  \
    (*(int(*)(PyObject*, int*, int*))PyGAME_C_API[PYGAMEAPI_BASE_FIRSTSLOT + 4])

#define FloatFromObj                                                    \
    (*(int(*)(PyObject*, float*))PyGAME_C_API[PYGAMEAPI_BASE_FIRSTSLOT + 5])

#define FloatFromObjIndex                                               \
    (*(float(*)(PyObject*, int, float*))                                \
     PyGAME_C_API[PYGAMEAPI_BASE_FIRSTSLOT + 6])

#define TwoFloatsFromObj                                \
    (*(int(*)(PyObject*, float*, float*))               \
     PyGAME_C_API[PYGAMEAPI_BASE_FIRSTSLOT + 7])

#define UintFromObj                                                     \
    (*(int(*)(PyObject*, Uint32*))PyGAME_C_API[PYGAMEAPI_BASE_FIRSTSLOT + 8])

#define UintFromObjIndex                                                \
    (*(int(*)(PyObject*, int, Uint32*))                                 \
     PyGAME_C_API[PYGAMEAPI_BASE_FIRSTSLOT + 9])

#define PyGame_Video_AutoQuit                                           \
    (*(void(*)(void))PyGAME_C_API[PYGAMEAPI_BASE_FIRSTSLOT + 10])

#define PyGame_Video_AutoInit                                           \
    (*(int(*)(void))PyGAME_C_API[PYGAMEAPI_BASE_FIRSTSLOT + 11])

#define RGBAFromObj                                                     \
    (*(int(*)(PyObject*, Uint8*))PyGAME_C_API[PYGAMEAPI_BASE_FIRSTSLOT + 12])

#define import_pygame_base() {                                          \
	PyObject *_module = PyImport_ImportModule(IMPPREFIX "base");        \
	if (_module != NULL) {                                           \
            PyObject *_dict = PyModule_GetDict(_module);                  \
            PyObject *_c_api = PyDict_GetItemString(_dict,                \
                                                   PYGAMEAPI_LOCAL_ENTRY); \
            if(PyCObject_Check(_c_api)) {                                \
                int i; void** localptr = (void**)PyCObject_AsVoidPtr(_c_api); \
                for(i = 0; i < PYGAMEAPI_BASE_NUMSLOTS; ++i)            \
                    PyGAME_C_API[i + PYGAMEAPI_BASE_FIRSTSLOT] = localptr[i]; \
            }                                                           \
            Py_DECREF(_module);                                          \
        }                                                               \
    }
#endif


/* RECT */
#define PYGAMEAPI_RECT_FIRSTSLOT                                \
    (PYGAMEAPI_BASE_FIRSTSLOT + PYGAMEAPI_BASE_NUMSLOTS)
#define PYGAMEAPI_RECT_NUMSLOTS 4

typedef struct {
    int x, y;
    int w, h;
}GAME_Rect;

typedef struct {
    PyObject_HEAD
    GAME_Rect r;
    PyObject *weakreflist;
} PyRectObject;

#define PyRect_AsRect(x) (((PyRectObject*)x)->r)
#ifndef PYGAMEAPI_RECT_INTERNAL
#define PyRect_Check(x) \
    ((x)->ob_type == (PyTypeObject*)PyGAME_C_API[PYGAMEAPI_RECT_FIRSTSLOT + 0])
#define PyRect_Type (*(PyTypeObject*)PyGAME_C_API[PYGAMEAPI_RECT_FIRSTSLOT + 0])
#define PyRect_New                                                      \
    (*(PyObject*(*)(SDL_Rect*))PyGAME_C_API[PYGAMEAPI_RECT_FIRSTSLOT + 1])
#define PyRect_New4                                                     \
    (*(PyObject*(*)(int,int,int,int))PyGAME_C_API[PYGAMEAPI_RECT_FIRSTSLOT + 2])
#define GameRect_FromObject                                             \
    (*(GAME_Rect*(*)(PyObject*, GAME_Rect*))                            \
     PyGAME_C_API[PYGAMEAPI_RECT_FIRSTSLOT + 3])

#define import_pygame_rect() {                                          \
	PyObject *_module = PyImport_ImportModule(IMPPREFIX "rect");        \
	if (_module != NULL) {                                         \
            PyObject *_dict = PyModule_GetDict(_module);                  \
            PyObject *_c_api = PyDict_GetItemString(_dict,                \
                                                   PYGAMEAPI_LOCAL_ENTRY); \
            if(PyCObject_Check(_c_api)) {                                \
                int i; void** localptr = (void**)PyCObject_AsVoidPtr(_c_api); \
                for(i = 0; i < PYGAMEAPI_RECT_NUMSLOTS; ++i)            \
                    PyGAME_C_API[i + PYGAMEAPI_RECT_FIRSTSLOT] = localptr[i]; \
            }                                                           \
            Py_DECREF(_module);                                          \
        }                                                               \
    }
#endif


/* CDROM */
#define PYGAMEAPI_CDROM_FIRSTSLOT                               \
    (PYGAMEAPI_RECT_FIRSTSLOT + PYGAMEAPI_RECT_NUMSLOTS)
#define PYGAMEAPI_CDROM_NUMSLOTS 2

typedef struct {
    PyObject_HEAD
    int id;
} PyCDObject;

#define PyCD_AsID(x) (((PyCDObject*)x)->id)
#ifndef PYGAMEAPI_CDROM_INTERNAL
#define PyCD_Check(x)                                                   \
    ((x)->ob_type == (PyTypeObject*)PyGAME_C_API[PYGAMEAPI_CDROM_FIRSTSLOT + 0])
#define PyCD_Type (*(PyTypeObject*)PyGAME_C_API[PYGAMEAPI_CDROM_FIRSTSLOT + 0])
#define PyCD_New                                                        \
    (*(PyObject*(*)(int))PyGAME_C_API[PYGAMEAPI_CDROM_FIRSTSLOT + 1])

#define import_pygame_cd() {                                      \
	PyObject *_module = PyImport_ImportModule(IMPPREFIX "cdrom"); \
	if (_module != NULL) {                                     \
            PyObject *_dict = PyModule_GetDict(_module);                  \
            PyObject *_c_api = PyDict_GetItemString(_dict,                \
                                                   PYGAMEAPI_LOCAL_ENTRY); \
            if(PyCObject_Check(_c_api)) {                                \
                int i; void** localptr = (void**)PyCObject_AsVoidPtr(_c_api); \
                for(i = 0; i < PYGAMEAPI_CDROM_NUMSLOTS; ++i)           \
                    PyGAME_C_API[i + PYGAMEAPI_CDROM_FIRSTSLOT] = localptr[i]; \
            }                                                           \
            Py_DECREF(_module);                                          \
        }                                                               \
    }
#endif


/* JOYSTICK */
#define PYGAMEAPI_JOYSTICK_FIRSTSLOT \
    (PYGAMEAPI_CDROM_FIRSTSLOT + PYGAMEAPI_CDROM_NUMSLOTS)
#define PYGAMEAPI_JOYSTICK_NUMSLOTS 2

typedef struct {
    PyObject_HEAD
    int id;
} PyJoystickObject;

#define PyJoystick_AsID(x) (((PyJoystickObject*)x)->id)

#ifndef PYGAMEAPI_JOYSTICK_INTERNAL
#define PyJoystick_Check(x)                                             \
    ((x)->ob_type == (PyTypeObject*)                                    \
     PyGAME_C_API[PYGAMEAPI_JOYSTICK_FIRSTSLOT + 0])

#define PyJoystick_Type                                                 \
    (*(PyTypeObject*)PyGAME_C_API[PYGAMEAPI_JOYSTICK_FIRSTSLOT + 0])
#define PyJoystick_New                                                  \
    (*(PyObject*(*)(int))PyGAME_C_API[PYGAMEAPI_JOYSTICK_FIRSTSLOT + 1])

#define import_pygame_joystick() {                                      \
	PyObject *_module = PyImport_ImportModule(IMPPREFIX "joystick");    \
	if (_module != NULL) {                                           \
            PyObject *_dict = PyModule_GetDict(_module);                  \
            PyObject *_c_api = PyDict_GetItemString(_dict,                \
                                                   PYGAMEAPI_LOCAL_ENTRY); \
            if(PyCObject_Check(_c_api)) {                                \
                int i; void** localptr = (void**)PyCObject_AsVoidPtr(_c_api); \
                for(i = 0; i < PYGAMEAPI_JOYSTICK_NUMSLOTS; ++i)        \
                    PyGAME_C_API[i + PYGAMEAPI_JOYSTICK_FIRSTSLOT] =    \
                        localptr[i];                                    \
            }                                                           \
            Py_DECREF(_module);                                          \
        }                                                               \
    }
#endif


/* DISPLAY */
#define PYGAMEAPI_DISPLAY_FIRSTSLOT \
    (PYGAMEAPI_JOYSTICK_FIRSTSLOT + PYGAMEAPI_JOYSTICK_NUMSLOTS)
#define PYGAMEAPI_DISPLAY_NUMSLOTS 2
typedef struct {
    PyObject_HEAD
    SDL_VideoInfo info;
} PyVidInfoObject;

#define PyVidInfo_AsVidInfo(x) (((PyVidInfoObject*)x)->info)
#ifndef PYGAMEAPI_DISPLAY_INTERNAL
#define PyVidInfo_Check(x)                                              \
    ((x)->ob_type == (PyTypeObject*)                                    \
     PyGAME_C_API[PYGAMEAPI_DISPLAY_FIRSTSLOT + 0])

#define PyVidInfo_Type                                                  \
    (*(PyTypeObject*)PyGAME_C_API[PYGAMEAPI_DISPLAY_FIRSTSLOT + 0])
#define PyVidInfo_New                                   \
    (*(PyObject*(*)(SDL_VideoInfo*))                    \
     PyGAME_C_API[PYGAMEAPI_DISPLAY_FIRSTSLOT + 1])
#define import_pygame_display() {                                   \
	PyObject *_module = PyImport_ImportModule(IMPPREFIX "display"); \
	if (_module != NULL) {                                       \
            PyObject *_dict = PyModule_GetDict(_module);                  \
            PyObject *_c_api = PyDict_GetItemString(_dict,                \
                                                   PYGAMEAPI_LOCAL_ENTRY); \
            if(PyCObject_Check(_c_api)) {                                \
                int i; void** localptr = (void**)PyCObject_AsVoidPtr(_c_api); \
                for(i = 0; i < PYGAMEAPI_DISPLAY_NUMSLOTS; ++i)         \
                    PyGAME_C_API[i + PYGAMEAPI_DISPLAY_FIRSTSLOT] =     \
                        localptr[i];                                    \
            }                                                           \
            Py_DECREF(_module);                                          \
        }                                                               \
    }
#endif


/* SURFACE */
#define PYGAMEAPI_SURFACE_FIRSTSLOT                             \
    (PYGAMEAPI_DISPLAY_FIRSTSLOT + PYGAMEAPI_DISPLAY_NUMSLOTS)
#define PYGAMEAPI_SURFACE_NUMSLOTS 3
typedef struct {
    PyObject_HEAD
    SDL_Surface* surf;
    struct SubSurface_Data* subsurface;  /*ptr to subsurface data (if a
                                          * subsurface)*/
    PyObject *weakreflist;
    PyObject *locklist;
    PyObject *dependency;
} PySurfaceObject;
#define PySurface_AsSurface(x) (((PySurfaceObject*)x)->surf)
#ifndef PYGAMEAPI_SURFACE_INTERNAL
#define PySurface_Check(x)                                              \
    ((x)->ob_type == (PyTypeObject*)                                    \
     PyGAME_C_API[PYGAMEAPI_SURFACE_FIRSTSLOT + 0])
#define PySurface_Type                                                  \
    (*(PyTypeObject*)PyGAME_C_API[PYGAMEAPI_SURFACE_FIRSTSLOT + 0])
#define PySurface_New                                                   \
    (*(PyObject*(*)(SDL_Surface*))                                      \
     PyGAME_C_API[PYGAMEAPI_SURFACE_FIRSTSLOT + 1])
#define PySurface_Blit                                                  \
    (*(int(*)(PyObject*,PyObject*,SDL_Rect*,SDL_Rect*,int))             \
     PyGAME_C_API[PYGAMEAPI_SURFACE_FIRSTSLOT + 2])

#define import_pygame_surface() do {                                   \
	PyObject *_module = PyImport_ImportModule(IMPPREFIX "surface"); \
	if (_module != NULL) {                                       \
            PyObject *_dict = PyModule_GetDict(_module);                  \
            PyObject *_c_api = PyDict_GetItemString(_dict,                \
                                                   PYGAMEAPI_LOCAL_ENTRY); \
            if(PyCObject_Check(_c_api)) {                                \
                int i; void** localptr = (void**)PyCObject_AsVoidPtr(_c_api); \
                for(i = 0; i < PYGAMEAPI_SURFACE_NUMSLOTS; ++i)         \
                    PyGAME_C_API[i + PYGAMEAPI_SURFACE_FIRSTSLOT] =     \
                        localptr[i];                                    \
				}                                                           \
				Py_DECREF(_module);                                          \
			}                                                               \
			else                                                            \
			{                                                               \
				break;                                                      \
			}                                                               \
			_module = PyImport_ImportModule(IMPPREFIX "surflock");          \
			if (_module != NULL) {                                           \
            PyObject *_dict = PyModule_GetDict(_module);                  \
            PyObject *_c_api = PyDict_GetItemString(_dict,                \
                                                   PYGAMEAPI_LOCAL_ENTRY); \
            if(PyCObject_Check(_c_api)) {                                \
                int i; void** localptr = (void**)PyCObject_AsVoidPtr(_c_api); \
                for(i = 0; i < PYGAMEAPI_SURFLOCK_NUMSLOTS; ++i)        \
                    PyGAME_C_API[i + PYGAMEAPI_SURFLOCK_FIRSTSLOT] =    \
                        localptr[i];                                    \
            }                                                           \
            Py_DECREF(_module);                                          \
        }                                                               \
    } while (0)
#endif


/* SURFLOCK */    /*auto import/init by surface*/
#define PYGAMEAPI_SURFLOCK_FIRSTSLOT                            \
    (PYGAMEAPI_SURFACE_FIRSTSLOT + PYGAMEAPI_SURFACE_NUMSLOTS)
#define PYGAMEAPI_SURFLOCK_NUMSLOTS 8
struct SubSurface_Data
{
    PyObject* owner;
    int pixeloffset;
    int offsetx, offsety;
};

typedef struct
{
    PyObject_HEAD
    PyObject *surface;
    PyObject *lockobj;
    PyObject *weakrefs;
} PyLifetimeLock;

#ifndef PYGAMEAPI_SURFLOCK_INTERNAL
#define PyLifetimeLock_Check(x)                         \
    ((x)->ob_type == (PyTypeObject*)                    \
        PyGAME_C_API[PYGAMEAPI_SURFLOCK_FIRSTSLOT + 0])
#define PySurface_Prep(x)                                               \
    if(((PySurfaceObject*)x)->subsurface)                               \
        (*(*(void(*)(PyObject*))                                        \
           PyGAME_C_API[PYGAMEAPI_SURFLOCK_FIRSTSLOT + 1]))(x)

#define PySurface_Unprep(x)                                             \
    if(((PySurfaceObject*)x)->subsurface)                               \
        (*(*(void(*)(PyObject*))                                        \
           PyGAME_C_API[PYGAMEAPI_SURFLOCK_FIRSTSLOT + 2]))(x)

#define PySurface_Lock                                                  \
    (*(int(*)(PyObject*))PyGAME_C_API[PYGAMEAPI_SURFLOCK_FIRSTSLOT + 3])
#define PySurface_Unlock                                                \
    (*(int(*)(PyObject*))PyGAME_C_API[PYGAMEAPI_SURFLOCK_FIRSTSLOT + 4])
#define PySurface_LockBy                                                \
    (*(int(*)(PyObject*,PyObject*))                                     \
        PyGAME_C_API[PYGAMEAPI_SURFLOCK_FIRSTSLOT + 5])
#define PySurface_UnlockBy                                              \
    (*(int(*)(PyObject*,PyObject*))                                     \
        PyGAME_C_API[PYGAMEAPI_SURFLOCK_FIRSTSLOT + 6])
#define PySurface_LockLifetime                                          \
    (*(PyObject*(*)(PyObject*,PyObject*))                               \
        PyGAME_C_API[PYGAMEAPI_SURFLOCK_FIRSTSLOT + 7])
#endif


/* EVENT */
#define PYGAMEAPI_EVENT_FIRSTSLOT                                       \
    (PYGAMEAPI_SURFLOCK_FIRSTSLOT + PYGAMEAPI_SURFLOCK_NUMSLOTS)
#define PYGAMEAPI_EVENT_NUMSLOTS 4

typedef struct {
    PyObject_HEAD
    int type;
    PyObject* dict;
} PyEventObject;

#ifndef PYGAMEAPI_EVENT_INTERNAL
#define PyEvent_Check(x)                                                \
    ((x)->ob_type == (PyTypeObject*)PyGAME_C_API[PYGAMEAPI_EVENT_FIRSTSLOT + 0])
#define PyEvent_Type \
    (*(PyTypeObject*)PyGAME_C_API[PYGAMEAPI_EVENT_FIRSTSLOT + 0])
#define PyEvent_New \
    (*(PyObject*(*)(SDL_Event*))PyGAME_C_API[PYGAMEAPI_EVENT_FIRSTSLOT + 1])
#define PyEvent_New2                                                    \
    (*(PyObject*(*)(int, PyObject*))PyGAME_C_API[PYGAMEAPI_EVENT_FIRSTSLOT + 2])
#define PyEvent_FillUserEvent                           \
    (*(int (*)(PyEventObject*, SDL_Event*))             \
     PyGAME_C_API[PYGAMEAPI_EVENT_FIRSTSLOT + 3])
#define import_pygame_event() {                                   \
	PyObject *_module = PyImport_ImportModule(IMPPREFIX "event"); \
	if (_module != NULL) {                                     \
            PyObject *_dict = PyModule_GetDict(_module);                  \
            PyObject *_c_api = PyDict_GetItemString(_dict,                \
                                                   PYGAMEAPI_LOCAL_ENTRY); \
            if(PyCObject_Check(_c_api)) {                                \
                int i; void** localptr = (void**)PyCObject_AsVoidPtr(_c_api); \
                for(i = 0; i < PYGAMEAPI_EVENT_NUMSLOTS; ++i)           \
                    PyGAME_C_API[i + PYGAMEAPI_EVENT_FIRSTSLOT] = localptr[i]; \
            }                                                           \
            Py_DECREF(_module);                                          \
        }                                                               \
    }
#endif


/* RWOBJECT */
/*the rwobject are only needed for C side work, not accessable from python*/
#define PYGAMEAPI_RWOBJECT_FIRSTSLOT                            \
    (PYGAMEAPI_EVENT_FIRSTSLOT + PYGAMEAPI_EVENT_NUMSLOTS)
#define PYGAMEAPI_RWOBJECT_NUMSLOTS 7
#ifndef PYGAMEAPI_RWOBJECT_INTERNAL
#define RWopsFromObject \
    (*(SDL_RWops*(*)(PyObject*))PyGAME_C_API[PYGAMEAPI_RWOBJECT_FIRSTSLOT + 0])
#define RWopsCheckObject                                               \
    (*(int(*)(SDL_RWops*))PyGAME_C_API[PYGAMEAPI_RWOBJECT_FIRSTSLOT + 1])
#define RWopsFromFileObjectThreaded                                         \
    (*(SDL_RWops*(*)(PyObject*))PyGAME_C_API[PYGAMEAPI_RWOBJECT_FIRSTSLOT + 2])
#define RWopsCheckObjectThreaded                                        \
    (*(int(*)(SDL_RWops*))PyGAME_C_API[PYGAMEAPI_RWOBJECT_FIRSTSLOT + 3])
#define RWopsEncodeFilePath \
    (*(PyObject*(*)(PyObject*, PyObject*)) \
        PyGAME_C_API[PYGAMEAPI_RWOBJECT_FIRSTSLOT + 4])
#define RWopsEncodeString \
    (*(PyObject*(*)(PyObject*, const char*, const char*, PyObject*)) \
        PyGAME_C_API[PYGAMEAPI_RWOBJECT_FIRSTSLOT + 5])
#define RWopsFromFileObject                                         \
    (*(SDL_RWops*(*)(PyObject*))PyGAME_C_API[PYGAMEAPI_RWOBJECT_FIRSTSLOT + 6])
#define import_pygame_rwobject() {                                   \
	PyObject *_module = PyImport_ImportModule(IMPPREFIX "rwobject"); \
	if (_module != NULL) {                                        \
            PyObject *_dict = PyModule_GetDict(_module);                  \
            PyObject *_c_api = PyDict_GetItemString(_dict,                \
                                                   PYGAMEAPI_LOCAL_ENTRY); \
            if(PyCObject_Check(_c_api)) {                                \
                int i; void** localptr = (void**)PyCObject_AsVoidPtr(_c_api); \
                for(i = 0; i < PYGAMEAPI_RWOBJECT_NUMSLOTS; ++i)        \
                    PyGAME_C_API[i + PYGAMEAPI_RWOBJECT_FIRSTSLOT] =    \
                        localptr[i];                                    \
            }                                                           \
            Py_DECREF(_module);                                          \
        }                                                               \
    }
/* For backward compatibility */
#define RWopsFromPython RWopsFromObject
#define RWopsCheckPython RWopsCheckObject
#define RWopsFromPythonThreaded RWopsFromFileObjectThreaded
#define RWopsCheckPythonThreaded RWopsCheckObjectThreaded
#endif

/* BufferProxy */
typedef struct
{
    PyObject_HEAD
    PyObject *dict;     /* dict for subclassing */
    PyObject *weakrefs; /* Weakrefs for subclassing */
    void *buffer;       /* Pointer to the buffer of the parent object. */
    Py_ssize_t length;  /* Length of the buffer. */
    PyObject *parent;   /* Parent object associated with this object. */
    PyObject *lock;     /* Lock object for the surface. */

} PyBufferProxy;

#define PYGAMEAPI_BUFFERPROXY_FIRSTSLOT                                 \
    (PYGAMEAPI_RWOBJECT_FIRSTSLOT + PYGAMEAPI_RWOBJECT_NUMSLOTS)
#define PYGAMEAPI_BUFFERPROXY_NUMSLOTS 2
#ifndef PYGAMEAPI_BUFFERPROXY_INTERNAL
#define PyBufferProxy_Check(x)                                          \
    ((x)->ob_type == (PyTypeObject*)                                    \
     PyGAME_C_API[PYGAMEAPI_BUFFERPROXY_FIRSTSLOT + 0])
#define PyBufferProxy_New                                               \
    (*(PyObject*(*)(PyObject*, void*, Py_ssize_t, PyObject*))           \
    PyGAME_C_API[PYGAMEAPI_BUFFERPROXY_FIRSTSLOT + 1])
#define import_pygame_bufferproxy()                                      \
    {                                                                    \
	PyObject *_module = PyImport_ImportModule (IMPPREFIX "bufferproxy");\
	if (_module != NULL)                                             \
        {                                                                \
            PyObject *_dict = PyModule_GetDict (_module);                \
            PyObject *_c_api = PyDict_GetItemString                      \
                (_dict, PYGAMEAPI_LOCAL_ENTRY);                          \
            if (PyCObject_Check (_c_api))                                \
            {                                                            \
                int i;                                                   \
                void** localptr = (void**) PyCObject_AsVoidPtr (_c_api); \
                for (i = 0; i < PYGAMEAPI_BUFFERPROXY_NUMSLOTS; ++i)     \
                    PyGAME_C_API[i + PYGAMEAPI_BUFFERPROXY_FIRSTSLOT] =  \
                        localptr[i];                                     \
            }                                                            \
            Py_DECREF (_module);                                         \
        }                                                                \
    }
#endif /* PYGAMEAPI_BUFFERPROXY_INTERNAL */

/* PixelArray */
#define PYGAMEAPI_PIXELARRAY_FIRSTSLOT                                 \
    (PYGAMEAPI_BUFFERPROXY_FIRSTSLOT + PYGAMEAPI_BUFFERPROXY_NUMSLOTS)
#define PYGAMEAPI_PIXELARRAY_NUMSLOTS 2
#ifndef PYGAMEAPI_PIXELARRAY_INTERNAL
#define PyPixelArray_Check(x)                                           \
    ((x)->ob_type == (PyTypeObject*)                                    \
     PyGAME_C_API[PYGAMEAPI_PIXELARRAY_FIRSTSLOT + 0])
#define PyPixelArray_New                                                \
    (*(PyObject*(*)) PyGAME_C_API[PYGAMEAPI_PIXELARRAY_FIRSTSLOT + 1])
#define import_pygame_pixelarray()                                       \
    {                                                                    \
	PyObject *_module = PyImport_ImportModule (IMPPREFIX "pixelarray"); \
	if (_module != NULL)                                             \
        {                                                                \
            PyObject *_dict = PyModule_GetDict (_module);                \
            PyObject *_c_api = PyDict_GetItemString                      \
                (_dict, PYGAMEAPI_LOCAL_ENTRY);                          \
            if (PyCObject_Check (_c_api))                                \
            {                                                            \
                int i;                                                   \
                void** localptr = (void**) PyCObject_AsVoidPtr (_c_api); \
                for (i = 0; i < PYGAMEAPI_PIXELARRAY_NUMSLOTS; ++i)      \
                    PyGAME_C_API[i + PYGAMEAPI_PIXELARRAY_FIRSTSLOT] =   \
                        localptr[i];                                     \
            }                                                            \
            Py_DECREF (_module);                                         \
        }                                                                \
    }
#endif /* PYGAMEAPI_PIXELARRAY_INTERNAL */

/* Color */
#define PYGAMEAPI_COLOR_FIRSTSLOT                                       \
    (PYGAMEAPI_PIXELARRAY_FIRSTSLOT + PYGAMEAPI_PIXELARRAY_NUMSLOTS)
#define PYGAMEAPI_COLOR_NUMSLOTS 4
#ifndef PYGAMEAPI_COLOR_INTERNAL
#define PyColor_Check(x)                                                \
    ((x)->ob_type == (PyTypeObject*)                                    \
        PyGAME_C_API[PYGAMEAPI_COLOR_FIRSTSLOT + 0])
#define PyColor_New                                                     \
    (*(PyObject *(*)(Uint8*)) PyGAME_C_API[PYGAMEAPI_COLOR_FIRSTSLOT + 1])
#define PyColor_NewLength                                               \
    (*(PyObject *(*)(Uint8*, Uint8)) PyGAME_C_API[PYGAMEAPI_COLOR_FIRSTSLOT + 3])

#define RGBAFromColorObj                                                \
    (*(int(*)(PyObject*, Uint8*)) PyGAME_C_API[PYGAMEAPI_COLOR_FIRSTSLOT + 2])
#define import_pygame_color()                                           \
    {                                                                   \
	PyObject *_module = PyImport_ImportModule (IMPPREFIX "color");     \
	if (_module != NULL)                                            \
        {                                                               \
            PyObject *_dict = PyModule_GetDict (_module);               \
            PyObject *_c_api = PyDict_GetItemString                     \
                (_dict, PYGAMEAPI_LOCAL_ENTRY);                         \
            if (PyCObject_Check (_c_api))                               \
            {                                                           \
                int i;                                                  \
                void** localptr = (void**) PyCObject_AsVoidPtr (_c_api); \
                for (i = 0; i < PYGAMEAPI_COLOR_NUMSLOTS; ++i)          \
                    PyGAME_C_API[i + PYGAMEAPI_COLOR_FIRSTSLOT] =       \
                        localptr[i];                                    \
            }                                                           \
            Py_DECREF (_module);                                        \
        }                                                               \
    }
#endif /* PYGAMEAPI_COLOR_INTERNAL */


/* Math */
#define PYGAMEAPI_MATH_FIRSTSLOT                                       \
    (PYGAMEAPI_COLOR_FIRSTSLOT + PYGAMEAPI_COLOR_NUMSLOTS)
#define PYGAMEAPI_MATH_NUMSLOTS 2
#ifndef PYGAMEAPI_MATH_INTERNAL
#define PyVector2_Check(x)                                                \
    ((x)->ob_type == (PyTypeObject*)                                    \
        PyGAME_C_API[PYGAMEAPI_MATH_FIRSTSLOT + 0])
#define PyVector3_Check(x)                                                \
    ((x)->ob_type == (PyTypeObject*)                                    \
        PyGAME_C_API[PYGAMEAPI_MATH_FIRSTSLOT + 1])
/*
#define PyVector2_New                                             \
    (*(PyObject*(*)) PyGAME_C_API[PYGAMEAPI_MATH_FIRSTSLOT + 1])
*/
#define import_pygame_math()                                           \
    {                                                                   \
	PyObject *_module = PyImport_ImportModule (IMPPREFIX "math");     \
	if (_module != NULL)                                            \
        {                                                               \
            PyObject *_dict = PyModule_GetDict (_module);               \
            PyObject *_c_api = PyDict_GetItemString                     \
                (_dict, PYGAMEAPI_LOCAL_ENTRY);                         \
            if (PyCObject_Check (_c_api))                               \
            {                                                           \
                int i;                                                  \
                void** localptr = (void**) PyCObject_AsVoidPtr (_c_api); \
                for (i = 0; i < PYGAMEAPI_MATH_NUMSLOTS; ++i)          \
                    PyGAME_C_API[i + PYGAMEAPI_MATH_FIRSTSLOT] =       \
                        localptr[i];                                    \
            }                                                           \
            Py_DECREF (_module);                                        \
        }                                                               \
    }
#endif /* PYGAMEAPI_MATH_INTERNAL */

#ifndef NO_PYGAME_C_API
#define PYGAMEAPI_TOTALSLOTS                                            \
    (PYGAMEAPI_MATH_FIRSTSLOT + PYGAMEAPI_MATH_NUMSLOTS)
static void* PyGAME_C_API[PYGAMEAPI_TOTALSLOTS] = { NULL };
#endif

/*last platform compiler stuff*/
#if defined(macintosh) && defined(__MWERKS__) || defined(__SYMBIAN32__)
#define PYGAME_EXPORT __declspec(export)
#else
#define PYGAME_EXPORT
#endif

#if defined(__SYMBIAN32__) && PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION == 2

// These are missing from Python 2.2
#ifndef Py_RETURN_NONE

#define Py_RETURN_NONE     return Py_INCREF(Py_None), Py_None
#define Py_RETURN_TRUE     return Py_INCREF(Py_True), Py_True
#define Py_RETURN_FALSE    return Py_INCREF(Py_False), Py_False

#ifndef intrptr_t
#define intptr_t int

// No PySlice_GetIndicesEx on Py 2.2        
#define PySlice_GetIndicesEx(a,b,c,d,e,f) PySlice_GetIndices(a,b,c,d,e)

#define PyBool_FromLong(x) 	Py_BuildValue("b", x)
#endif

// _symport_free and malloc are not exported in python.dll
// See http://discussion.forum.nokia.com/forum/showthread.php?t=57874
#undef PyObject_NEW
#define PyObject_NEW PyObject_New
#undef PyMem_MALLOC
#define PyMem_MALLOC PyMem_Malloc
#undef PyObject_DEL
#define PyObject_DEL PyObject_Del

#endif // intptr_t

#endif // __SYMBIAN32__ Python 2.2.2

#endif /* PYGAME_H */
