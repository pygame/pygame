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
#include <Python.h>

#ifdef MS_WIN32 /*Python gives us MS_WIN32, SDL needs just WIN32*/
#define WIN32
#endif

#include <SDL.h>


/* older python compatability */
#if PYTHON_API_VERSION < 1009
#define PyObject_DEL(op)		free(op)
#define PyMem_New(type, n)  	((type*)PyMem_Malloc((n) * sizeof(type)))
#define PyMem_Del(p)			PyMem_Free((char*)p)
static int PyModule_AddObject(PyObject *m, char *name, PyObject *o)
{
	PyObject *dict;
    if (!PyModule_Check(m) || o == NULL)
		return -1;
	dict = PyModule_GetDict(m);
	if (dict == NULL)
		return -1;
    if (PyDict_SetItemString(dict, name, o))
		return -1;
	Py_DECREF(o);
	return 0;
}
#define PyModule_AddIntConstant(m, name, value) \
	PyModule_AddObject(m, name, PyInt_FromLong(value))
#define PyString_AsStringAndSize(o,ppc,pn) (*ppc = PyString_AsString(o), *pn = PyString_Size(o))
#define PySequence_Size(x) PySequence_Length(x)

#define PyUnicode_Check(text) 0
#endif

/* macros used throughout the source */
#define RAISE(x,y) (PyErr_SetString((x), (y)), (PyObject*)NULL)
#define RETURN_NONE return (Py_INCREF(Py_None), Py_None);
#define PyType_Init(x) (((x).ob_type) = &PyType_Type)
#define PYGAMEAPI_LOCAL_ENTRY "_PYGAME_C_API"
#ifndef min
#define min(a,b) ((a)<=(b)?(a):(b))
#endif
#ifndef max
#define max(a,b) ((a)>=(b)?(a):(b))
#endif

/* test sdl initializations */
#define VIDEO_INIT_CHECK() \
	if(!SDL_WasInit(SDL_INIT_VIDEO)) \
		return RAISE(PyExc_SDLError, "video system not initialized")
#define CDROM_INIT_CHECK() \
	if(!SDL_WasInit(SDL_INIT_CDROM)) \
		return RAISE(PyExc_SDLError, "cdrom system not initialized")
#define JOYSTICK_INIT_CHECK() \
	if(!SDL_WasInit(SDL_INIT_JOYSTICK)) \
		return RAISE(PyExc_SDLError, "joystick system not initialized")




/* BASE */
#define PYGAMEAPI_BASE_FIRSTSLOT 0
#define PYGAMEAPI_BASE_NUMSLOTS 13
#ifndef PYGAMEAPI_BASE_INTERNAL
#define PyExc_SDLError ((PyObject*)PyGAME_C_API[PYGAMEAPI_BASE_FIRSTSLOT])
#define PyGame_RegisterQuit \
			(*(void(*)(void(*)(void)))PyGAME_C_API[PYGAMEAPI_BASE_FIRSTSLOT + 1])
#define IntFromObj \
			(*(int(*)(PyObject*, int*))PyGAME_C_API[PYGAMEAPI_BASE_FIRSTSLOT + 2])
#define IntFromObjIndex \
			(*(int(*)(PyObject*, int, int*))PyGAME_C_API[PYGAMEAPI_BASE_FIRSTSLOT + 3])
#define TwoIntsFromObj \
			(*(int(*)(PyObject*, int*, int*))PyGAME_C_API[PYGAMEAPI_BASE_FIRSTSLOT + 4])
#define FloatFromObj \
			(*(int(*)(PyObject*, float*))PyGAME_C_API[PYGAMEAPI_BASE_FIRSTSLOT + 5])
#define FloatFromObjIndex \
			(*(float(*)(PyObject*, int, float*))PyGAME_C_API[PYGAMEAPI_BASE_FIRSTSLOT + 6])
#define TwoFloatsFromObj \
			(*(int(*)(PyObject*, float*, float*))PyGAME_C_API[PYGAMEAPI_BASE_FIRSTSLOT + 7])
#define UintFromObj \
			(*(int(*)(PyObject*, Uint32*))PyGAME_C_API[PYGAMEAPI_BASE_FIRSTSLOT + 8])
#define UintFromObjIndex \
			(*(int(*)(PyObject*, int, Uint32*))PyGAME_C_API[PYGAMEAPI_BASE_FIRSTSLOT + 9])
#define PyGame_Video_AutoQuit \
			(*(void(*)(void))PyGAME_C_API[PYGAMEAPI_BASE_FIRSTSLOT + 10])
#define PyGame_Video_AutoInit \
			(*(int(*)(void))PyGAME_C_API[PYGAMEAPI_BASE_FIRSTSLOT + 11])
#define RGBAFromObj \
			(*(int(*)(PyObject*, Uint8*))PyGAME_C_API[PYGAMEAPI_BASE_FIRSTSLOT + 12])
#define import_pygame_base() { \
	PyObject *module = PyImport_ImportModule("pygame.base"); \
	if (module != NULL) { \
		PyObject *dict = PyModule_GetDict(module); \
		PyObject *c_api = PyDict_GetItemString(dict, PYGAMEAPI_LOCAL_ENTRY); \
		if(PyCObject_Check(c_api)) {\
			int i; void** localptr = (void**)PyCObject_AsVoidPtr(c_api); \
			for(i = 0; i < PYGAMEAPI_BASE_NUMSLOTS; ++i) \
				PyGAME_C_API[i + PYGAMEAPI_BASE_FIRSTSLOT] = localptr[i]; \
} Py_DECREF(module); } }
#endif



/* RECT */
#define PYGAMEAPI_RECT_FIRSTSLOT 20
#define PYGAMEAPI_RECT_NUMSLOTS 4
typedef struct {
	int x, y;
	int w, h;
}GAME_Rect;
typedef struct {
  PyObject_HEAD
  GAME_Rect r;
} PyRectObject;
#define PyRect_AsRect(x) (((PyRectObject*)x)->r)
#ifndef PYGAMEAPI_RECT_INTERNAL
#define PyRect_Check(x) ((x)->ob_type == (PyTypeObject*)PyGAME_C_API[PYGAMEAPI_RECT_FIRSTSLOT + 0])
#define PyRect_Type (*(PyTypeObject*)PyGAME_C_API[PYGAMEAPI_RECT_FIRSTSLOT + 0])
#define PyRect_New (*(PyObject*(*)(SDL_Rect*))PyGAME_C_API[PYGAMEAPI_RECT_FIRSTSLOT + 1])
#define PyRect_New4 \
			(*(PyObject*(*)(int,int,int,int))PyGAME_C_API[PYGAMEAPI_RECT_FIRSTSLOT + 2])
#define GameRect_FromObject \
			(*(GAME_Rect*(*)(PyObject*, GAME_Rect*))PyGAME_C_API[PYGAMEAPI_RECT_FIRSTSLOT + 3])
#define import_pygame_rect() { \
	PyObject *module = PyImport_ImportModule("pygame.rect"); \
	if (module != NULL) { \
		PyObject *dict = PyModule_GetDict(module); \
		PyObject *c_api = PyDict_GetItemString(dict, PYGAMEAPI_LOCAL_ENTRY); \
		if(PyCObject_Check(c_api)) {\
			int i; void** localptr = (void**)PyCObject_AsVoidPtr(c_api); \
			for(i = 0; i < PYGAMEAPI_RECT_NUMSLOTS; ++i) \
				PyGAME_C_API[i + PYGAMEAPI_RECT_FIRSTSLOT] = localptr[i]; \
} Py_DECREF(module); } }
#endif




/* CDROM */
#define PYGAMEAPI_CDROM_FIRSTSLOT 30
#define PYGAMEAPI_CDROM_NUMSLOTS 2
typedef struct {
	PyObject_HEAD
	int id;
} PyCDObject;
#define PyCD_AsID(x) (((PyCDObject*)x)->id)
#ifndef PYGAMEAPI_CDROM_INTERNAL
#define PyCD_Check(x) ((x)->ob_type == (PyTypeObject*)PyGAME_C_API[PYGAMEAPI_CDROM_FIRSTSLOT + 0])
#define PyCD_Type (*(PyTypeObject*)PyGAME_C_API[PYGAMEAPI_CDROM_FIRSTSLOT + 0])
#define PyCD_New (*(PyObject*(*)(int))PyGAME_C_API[PYGAMEAPI_CDROM_FIRSTSLOT + 1])
#define import_pygame_cd() { \
	PyObject *module = PyImport_ImportModule("pygame.cdrom"); \
	if (module != NULL) { \
		PyObject *dict = PyModule_GetDict(module); \
		PyObject *c_api = PyDict_GetItemString(dict, PYGAMEAPI_LOCAL_ENTRY); \
		if(PyCObject_Check(c_api)) {\
			int i; void** localptr = (void**)PyCObject_AsVoidPtr(c_api); \
			for(i = 0; i < PYGAMEAPI_CDROM_NUMSLOTS; ++i) \
				PyGAME_C_API[i + PYGAMEAPI_CDROM_FIRSTSLOT] = localptr[i]; \
} Py_DECREF(module); } }
#endif


/* JOYSTICK */
#define PYGAMEAPI_JOYSTICK_FIRSTSLOT 32
#define PYGAMEAPI_JOYSTICK_NUMSLOTS 2
typedef struct {
	PyObject_HEAD
	int id;
} PyJoystickObject;
#define PyJoystick_AsID(x) (((PyJoystickObject*)x)->id)
#ifndef PYGAMEAPI_JOYSTICK_INTERNAL
#define PyJoystick_Check(x) ((x)->ob_type == (PyTypeObject*)PyGAME_C_API[PYGAMEAPI_JOYSTICK_FIRSTSLOT + 0])
#define PyJoystick_Type (*(PyTypeObject*)PyGAME_C_API[PYGAMEAPI_JOYSTICK_FIRSTSLOT + 0])
#define PyJoystick_New (*(PyObject*(*)(int))PyGAME_C_API[PYGAMEAPI_JOYSTICK_FIRSTSLOT + 1])
#define import_pygame_joystick() { \
	PyObject *module = PyImport_ImportModule("pygame.joystick"); \
	if (module != NULL) { \
		PyObject *dict = PyModule_GetDict(module); \
		PyObject *c_api = PyDict_GetItemString(dict, PYGAMEAPI_LOCAL_ENTRY); \
		if(PyCObject_Check(c_api)) {\
			int i; void** localptr = (void**)PyCObject_AsVoidPtr(c_api); \
			for(i = 0; i < PYGAMEAPI_JOYSTICK_NUMSLOTS; ++i) \
				PyGAME_C_API[i + PYGAMEAPI_JOYSTICK_FIRSTSLOT] = localptr[i]; \
} Py_DECREF(module); } }
#endif



/* DISPLAY */
#define PYGAMEAPI_DISPLAY_FIRSTSLOT 35
#define PYGAMEAPI_DISPLAY_NUMSLOTS 2
typedef struct {
	PyObject_HEAD
	SDL_VideoInfo info;
} PyVidInfoObject;
#define PyVidInfo_AsVidInfo(x) (((PyVidInfoObject*)x)->info)
#ifndef PYGAMEAPI_DISPLAY_INTERNAL
#define PyVidInfo_Check(x) ((x)->ob_type == (PyTypeObject*)PyGAME_C_API[PYGAMEAPI_DISPLAY_FIRSTSLOT + 0])
#define PyVidInfo_Type (*(PyTypeObject*)PyGAME_C_API[PYGAMEAPI_SURFACE_FIRSTSLOT + 0])
#define PyVidInfo_New (*(PyObject*(*)(SDL_VideoInfo*))PyGAME_C_API[PYGAMEAPI_DISPLAY_FIRSTSLOT + 1])
#define import_pygame_display() { \
	PyObject *module = PyImport_ImportModule("pygame.display"); \
	if (module != NULL) { \
		PyObject *dict = PyModule_GetDict(module); \
		PyObject *c_api = PyDict_GetItemString(dict, PYGAMEAPI_LOCAL_ENTRY); \
		if(PyCObject_Check(c_api)) {\
			int i; void** localptr = (void**)PyCObject_AsVoidPtr(c_api); \
			for(i = 0; i < PYGAMEAPI_DISPLAY_NUMSLOTS; ++i) \
				PyGAME_C_API[i + PYGAMEAPI_DISPLAY_FIRSTSLOT] = localptr[i]; \
} Py_DECREF(module); } }
#endif



/* SURFACE */
#define PYGAMEAPI_SURFACE_FIRSTSLOT 40
#define PYGAMEAPI_SURFACE_NUMSLOTS 3
typedef struct {
	PyObject_HEAD
	SDL_Surface* surf;
	struct SubSurface_Data* subsurface;  /*ptr to subsurface data (if a subsurface)*/
	int lockcount;
	int didlock;
} PySurfaceObject;
#define PySurface_AsSurface(x) (((PySurfaceObject*)x)->surf)
#ifndef PYGAMEAPI_SURFACE_INTERNAL
#define PySurface_Check(x) ((x)->ob_type == (PyTypeObject*)PyGAME_C_API[PYGAMEAPI_SURFACE_FIRSTSLOT + 0])
#define PySurface_Type (*(PyTypeObject*)PyGAME_C_API[PYGAMEAPI_SURFACE_FIRSTSLOT + 0])
#define PySurface_New (*(PyObject*(*)(SDL_Surface*))PyGAME_C_API[PYGAMEAPI_SURFACE_FIRSTSLOT + 1])
#define PySurface_Blit (*(int(*)(PyObject*,PyObject*,SDL_Rect*,SDL_Rect*))PyGAME_C_API[PYGAMEAPI_SURFACE_FIRSTSLOT + 2])
#define import_pygame_surface() { \
	PyObject *module = PyImport_ImportModule("pygame.surface"); \
	if (module != NULL) { \
		PyObject *dict = PyModule_GetDict(module); \
		PyObject *c_api = PyDict_GetItemString(dict, PYGAMEAPI_LOCAL_ENTRY); \
		if(PyCObject_Check(c_api)) {\
			int i; void** localptr = (void**)PyCObject_AsVoidPtr(c_api); \
			for(i = 0; i < PYGAMEAPI_SURFACE_NUMSLOTS; ++i) \
				PyGAME_C_API[i + PYGAMEAPI_SURFACE_FIRSTSLOT] = localptr[i]; \
	} Py_DECREF(module); } \
	module = PyImport_ImportModule("pygame.surflock"); \
	if (module != NULL) { \
		PyObject *dict = PyModule_GetDict(module); \
		PyObject *c_api = PyDict_GetItemString(dict, PYGAMEAPI_LOCAL_ENTRY); \
		if(PyCObject_Check(c_api)) {\
			int i; void** localptr = (void**)PyCObject_AsVoidPtr(c_api); \
			for(i = 0; i < PYGAMEAPI_SURFLOCK_NUMSLOTS; ++i) \
				PyGAME_C_API[i + PYGAMEAPI_SURFLOCK_FIRSTSLOT] = localptr[i]; \
} Py_DECREF(module); } }
#endif



/* SURFLOCK */    /*auto import/init by surface*/
#define PYGAMEAPI_SURFLOCK_FIRSTSLOT 44
#define PYGAMEAPI_SURFLOCK_NUMSLOTS 5
struct SubSurface_Data
{
	PyObject* owner;
	int pixeloffset;
	int offsetx, offsety;
};
#ifndef PYGAMEAPI_SURFLOCK_INTERNAL
#define PySurface_Prep(x) if(((PySurfaceObject*)x)->subsurface)(*(*(void(*)(PyObject*))PyGAME_C_API[PYGAMEAPI_SURFLOCK_FIRSTSLOT + 0]))(x)
#define PySurface_Unprep(x) if(((PySurfaceObject*)x)->subsurface)(*(*(void(*)(PyObject*))PyGAME_C_API[PYGAMEAPI_SURFLOCK_FIRSTSLOT + 1]))(x)
#define PySurface_Lock (*(int(*)(PyObject*))PyGAME_C_API[PYGAMEAPI_SURFLOCK_FIRSTSLOT + 2])
#define PySurface_Unlock (*(int(*)(PyObject*))PyGAME_C_API[PYGAMEAPI_SURFLOCK_FIRSTSLOT + 3])
#define PySurface_LockLifetime (*(PyObject*(*)(PyObject*))PyGAME_C_API[PYGAMEAPI_SURFLOCK_FIRSTSLOT + 4])
#endif



/* EVENT */
#define PYGAMEAPI_EVENT_FIRSTSLOT 49
#define PYGAMEAPI_EVENT_NUMSLOTS 2
typedef struct {
	PyObject_HEAD
	int type;
	PyObject* dict;
} PyEventObject;
#ifndef PYGAMEAPI_EVENT_INTERNAL
#define PyEvent_Check(x) ((x)->ob_type == (PyTypeObject*)PyGAME_C_API[PYGAMEAPI_EVENT_FIRSTSLOT + 0])
#define PyEvent_Type (*(PyTypeObject*)PyGAME_C_API[PYGAMEAPI_EVENT_FIRSTSLOT + 0])
#define PyEvent_New (*(PyObject*(*)(int, PyObject*))PyGAME_C_API[PYGAMEAPI_EVENT_FIRSTSLOT + 1])
#define import_pygame_event() { \
	PyObject *module = PyImport_ImportModule("pygame.event"); \
	if (module != NULL) { \
		PyObject *dict = PyModule_GetDict(module); \
		PyObject *c_api = PyDict_GetItemString(dict, PYGAMEAPI_LOCAL_ENTRY); \
		if(PyCObject_Check(c_api)) {\
			int i; void** localptr = (void**)PyCObject_AsVoidPtr(c_api); \
			for(i = 0; i < PYGAMEAPI_EVENT_NUMSLOTS; ++i) \
				PyGAME_C_API[i + PYGAMEAPI_EVENT_FIRSTSLOT] = localptr[i]; \
} Py_DECREF(module); } }
#endif


/* RWOBJECT */
/*the rwobject are only needed for C side work, not accessable from python*/
#define PYGAMEAPI_RWOBJECT_FIRSTSLOT 53
#define PYGAMEAPI_RWOBJECT_NUMSLOTS 4
#ifndef PYGAMEAPI_RWOBJECT_INTERNAL
#define RWopsFromPython (*(SDL_RWops*(*)(PyObject*))PyGAME_C_API[PYGAMEAPI_RWOBJECT_FIRSTSLOT + 0])
#define RWopsCheckPython (*(int(*)(SDL_RWops*))PyGAME_C_API[PYGAMEAPI_RWOBJECT_FIRSTSLOT + 1])
#define RWopsFromPythonThreaded (*(SDL_RWops*(*)(PyObject*))PyGAME_C_API[PYGAMEAPI_RWOBJECT_FIRSTSLOT + 2])
#define RWopsCheckPythonThreaded (*(int(*)(SDL_RWops*))PyGAME_C_API[PYGAMEAPI_RWOBJECT_FIRSTSLOT + 3])
#define import_pygame_rwobject() { \
	PyObject *module = PyImport_ImportModule("pygame.rwobject"); \
	if (module != NULL) { \
		PyObject *dict = PyModule_GetDict(module); \
		PyObject *c_api = PyDict_GetItemString(dict, PYGAMEAPI_LOCAL_ENTRY); \
		if(PyCObject_Check(c_api)) {\
			int i; void** localptr = (void**)PyCObject_AsVoidPtr(c_api); \
			for(i = 0; i < PYGAMEAPI_RWOBJECT_NUMSLOTS; ++i) \
				PyGAME_C_API[i + PYGAMEAPI_RWOBJECT_FIRSTSLOT] = localptr[i]; \
} Py_DECREF(module); } }
#endif




#ifndef NO_PYGAME_C_API
#define PYGAMEAPI_TOTALSLOTS 60
static void* PyGAME_C_API[PYGAMEAPI_TOTALSLOTS] = {NULL};
#endif


/*last platform compiler stuff*/
#if defined(macintosh) && defined(__MWERKS__)
#define PYGAME_EXPORT __declspec(export)
#else
#define PYGAME_EXPORT
#endif

