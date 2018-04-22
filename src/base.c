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
#define NO_PYGAME_C_API
#define PYGAMEAPI_BASE_INTERNAL
#include "pygame.h"
#include "pgarrinter.h"
#include "pgcompat.h"
#include "doc/pygame_doc.h"
#include <signal.h>


/* This file controls all the initialization of
 * the module and the various SDL subsystems
 */

/*platform specific init stuff*/

#ifdef MS_WIN32 /*python gives us MS_WIN32*/
#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#include<windows.h>
extern int SDL_RegisterApp(char*, Uint32, void*);
#endif

#if defined(macintosh)
#if(!defined(__MWERKS__) && !TARGET_API_MAC_CARBON)
QDGlobals pg_qd;
#endif
#endif

#if SDL_BYTEORDER == SDL_LIL_ENDIAN
#define PAI_MY_ENDIAN '<'
#define PAI_OTHER_ENDIAN '>'
#define BUF_OTHER_ENDIAN '>'
#else
#define PAI_MY_ENDIAN '>'
#define PAI_OTHER_ENDIAN '<'
#define BUF_OTHER_ENDIAN '<'
#endif
#define BUF_MY_ENDIAN '='

#if PY3
#define INT_CHECK(o) PyLong_Check(o)
#else
#define INT_CHECK(o) (PyInt_Check(o) || PyLong_Check(o))
#endif

/* Extended array struct */
typedef struct pg_capsule_interface_s {
    PyArrayInterface inter;
    Py_intptr_t imem[1];
} pgCapsuleInterface;

/* Py_buffer internal data for an array interface/struct */
typedef struct pg_view_internals_s {
    char format[4];      /* make 4 byte word sized */
    Py_ssize_t imem[1];
} pgViewInternals;

/* Custom exceptions */
static PyObject* pgExc_BufferError = NULL;

/* Only one instance of the state per process. */
static PyObject* pg_quit_functions = NULL;
static int pg_sdl_was_init = 0;
#if IS_SDLv2
SDL_Window* pg_default_window = NULL;
PyObject* pg_default_screen = NULL;
#endif /* IS_SDLv2 */

static void pg_install_parachute(void);
static void pg_uninstall_parachute(void);
static void _pg_quit(void);
static void pg_atexit_quit(void);
static int pgVideo_AutoInit(void);
static void pgVideo_AutoQuit(void);
static int pgGetArrayStruct(PyObject*, PyObject**, PyArrayInterface**);
static PyObject* pgArrayStruct_AsDict(PyArrayInterface*);
static PyObject* pgBuffer_AsArrayInterface(Py_buffer*);
static PyObject* pgBuffer_AsArrayStruct(Py_buffer*);
static int _pg_buffer_is_byteswapped(Py_buffer*);
static void pgBuffer_Release(pg_buffer*);
static int pgObject_GetBuffer(PyObject*, pg_buffer*, int);
static int pgGetArrayInterface(PyObject**, PyObject*);
static int pgArrayStruct_AsBuffer(pg_buffer*,
                                  PyObject*,
                                  PyArrayInterface*,
                                  int);
static int _pg_arraystruct_as_buffer(Py_buffer*,
                                     PyObject*,
                                     PyArrayInterface*,
                                     int);
static int _pg_arraystruct_to_format(char*, PyArrayInterface*, int);
static int pgDict_AsBuffer(pg_buffer*, PyObject*, int);
static int _pg_shape_check(PyObject*);
static int _pg_typestr_check(PyObject*);
static int _pg_strides_check(PyObject*);
static int _pg_data_check(PyObject*);
static int _pg_is_int_tuple(PyObject*);
static int _pg_values_as_buffer(Py_buffer*,
                                int,
                                PyObject*,
                                PyObject*,
                                PyObject*,
                                PyObject*);
static int _pg_int_tuple_as_ssize_arr(PyObject*, Py_ssize_t*);
static int _pg_typestr_as_format(PyObject*, char*, Py_ssize_t*);
static PyObject* pg_view_get_typestr_obj(Py_buffer*);
static PyObject* pg_view_get_shape_obj(Py_buffer*);
static PyObject* pg_view_get_strides_obj(Py_buffer*);
static PyObject* pg_view_get_data_obj(Py_buffer*);
static char _pg_as_arrayinter_typekind(Py_buffer*);
static char _pg_as_arrayinter_byteorder(Py_buffer*);
static int _pg_as_arrayinter_flags(Py_buffer*);
static pgCapsuleInterface* _pg_new_capsuleinterface(Py_buffer*);
#if PY3
static void _pg_capsule_PyMem_Free(PyObject*);
#endif
static PyObject* _pg_shape_as_tuple(PyArrayInterface*);
static PyObject* _pg_typekind_as_str(PyArrayInterface*);
static PyObject* _pg_strides_as_tuple(PyArrayInterface*);
static PyObject* _pg_data_as_tuple(PyArrayInterface*);
static PyObject* pg_get_array_interface(PyObject*, PyObject*);
static void _pg_release_buffer_array(Py_buffer*);
static void _pg_release_buffer_generic(Py_buffer*);
#if IS_SDLv2
static SDL_Window* pg_GetDefaultWindow(void);
static void pg_SetDefaultWindow(SDL_Window*);
static PyObject* pg_GetDefaultWindowSurface(void);
static void pg_SetDefaultWindowSurface(PyObject*);
#endif /* IS_SDLv2 */

#if PY_VERSION_HEX < 0x02060000
static int
_IsFortranContiguous(Py_buffer *view)
{
    Py_ssize_t sd, dim;
    int i;

    if (view->ndim == 0) return 1;
    if (view->strides == NULL) {
        return view->ndim == 1
    };

    sd = view->itemsize;
    if (view->ndim == 1) {
        return view->shape[0] == 1 || sd == view->strides[0];
    }
    for (i=0; i<view->ndim; i++) {
        dim = view->shape[i];
        if (dim == 0) {
            return 1;
        }
        if (view->strides[i] != sd) {
            return 0;
        }
        sd *= dim;
    }
    return 1;
}

static int
_IsCContiguous(Py_buffer *view)
{
    Py_ssize_t sd, dim;
    int i;

    if (view->ndim == 0) {
        return 1
    };
    if (view->strides == NULL) {
        return 1
    };

    sd = view->itemsize;
    if (view->ndim == 1) {
        return view->shape[0] == 1 || sd == view->strides[0];
    }
    for (i=view->ndim-1; i>=0; i--) {
        dim = view->shape[i];
        if (dim == 0) {
            return 1;
        }
        if (view->strides[i] != sd) {
            return 0;
        }
        sd *= dim;
    }
    return 1;
}

static int
PyBuffer_IsContiguous(Py_buffer *view, char fort)
{

    if (view->suboffsets != NULL) {
        return 0;
    }

    if (fort == 'C') {
        return _IsCContiguous(view);
    }
    else if (fort == 'F') {
        return _IsFortranContiguous(view);
    }
    else if (fort == 'A') {
        return _IsCContiguous(view) || _IsFortranContiguous(view);
    }
    return 0;
}
#endif /* #if PY_VERSION_HEX < 0x02060000 */

static int
pg_CheckSDLVersions(void) /*compare compiled to linked*/
{
#if IS_SDLv1
    SDL_version compiled;
    const SDL_version* linked;

    SDL_VERSION(&compiled);
    linked = SDL_Linked_Version();

    /*only check the major and minor version numbers.
      we will relax any differences in 'patch' version.*/

    if (compiled.major != linked->major || compiled.minor != linked->minor) {
        PyErr_Format(PyExc_RuntimeError,
                     "SDL compiled with version %d.%d.%d, linked to %d.%d.%d",
                     compiled.major, compiled.minor, compiled.patch,
                     linked->major, linked->minor, linked->patch);
        return 0;
    }
#else /* IS_SDLv2 */
    SDL_version compiled;
    SDL_version linked;

    SDL_VERSION(&compiled);
    SDL_GetVersion(&linked);

    /*only check the major and minor version numbers.
      we will relax any differences in 'patch' version.*/

    if (compiled.major != linked.major || compiled.minor != linked.minor) {
        PyErr_Format(PyExc_RuntimeError,
                     "SDL compiled with version %d.%d.%d, linked to %d.%d.%d",
                     compiled.major, compiled.minor, compiled.patch,
                     linked.major, linked.minor, linked.patch);
        return 0;
    }
#endif /* IS_SDLv2 */

    return 1;
}

void
pg_RegisterQuit(void(*func)(void))
{
    PyObject* obj;

    if (!pg_quit_functions) {
        pg_quit_functions = PyList_New(0);
        if (!pg_quit_functions) {
            return;
        }
    }
    if (func) {
        obj = PyCapsule_New(func, "quit", NULL);
        PyList_Append(pg_quit_functions, obj);
        Py_DECREF(obj);
    }
}

static PyObject*
pg_register_quit(PyObject* self, PyObject* value)
{
    if (!pg_quit_functions) {
        pg_quit_functions = PyList_New(0);
        if (!pg_quit_functions) {
            return NULL;
        }
    }
    PyList_Append(pg_quit_functions, value);

    Py_RETURN_NONE;
}

static PyObject*
pg_init(PyObject* self)
{
    PyObject *allmodules, *moduleslist, *dict, *func, *result, *mod;
    int loop, num;

    int success=0, fail=0;

    if (!pg_CheckSDLVersions()) {
        return NULL;
    }

    /*nice to initialize timer, so startup time will reflec pg_init() time*/
#if defined(WITH_THREAD) && !defined(MS_WIN32) && defined(SDL_INIT_EVENTTHREAD)
    pg_sdl_was_init = SDL_Init(SDL_INIT_EVENTTHREAD |
                            SDL_INIT_TIMER |
                            SDL_INIT_NOPARACHUTE) == 0;
#else
    pg_sdl_was_init = SDL_Init(SDL_INIT_TIMER | SDL_INIT_NOPARACHUTE) == 0;
#endif

    /* initialize all pygame modules */
    allmodules = PyImport_GetModuleDict();
    moduleslist = PyDict_Values(allmodules);
    if (!allmodules || !moduleslist) {
        return Py_BuildValue("(ii)", 0, 0);
    }

    if (pgVideo_AutoInit()) {
        ++success;
    }
    else {
        ++fail;
    }

    num = PyList_Size(moduleslist);
    for (loop = 0; loop < num; ++loop) {
        mod = PyList_GET_ITEM(moduleslist, loop);
        if (!mod || !PyModule_Check(mod)) {
            continue;
        }
        dict = PyModule_GetDict(mod);
        func = PyDict_GetItemString(dict, "__PYGAMEinit__");
        if(func && PyCallable_Check(func)) {
            result = PyObject_CallObject(func, NULL);
            if (result && PyObject_IsTrue(result)) {
                ++success;
            }
            else {
                PyErr_Clear();
                ++fail;
            }
            Py_XDECREF(result);
        }
    }
    Py_DECREF(moduleslist);

    return Py_BuildValue("(ii)", success, fail);
}

static void
pg_atexit_quit(void)
{
    pgVideo_AutoQuit();

    /* Maybe it is safe to call SDL_quit more than once after an SDL_Init,
       but this is undocumented. So play it safe and only call after a
       successful SDL_Init.
    */
    if (pg_sdl_was_init) {
        pg_sdl_was_init = 0;
        SDL_Quit();
    }
}

static PyObject*
pg_get_sdl_version(PyObject* self)
{
#if IS_SDLv1
    const SDL_version *v;

    v = SDL_Linked_Version();
    return Py_BuildValue("iii", v->major, v->minor, v->patch);
#else /* IS_SDLv2 */
    SDL_version v;

    SDL_GetVersion(&v);
    return Py_BuildValue("iii", v.major, v.minor, v.patch);
#endif /* IS_SDLv2 */
}

static PyObject*
pg_get_sdl_byteorder(PyObject *self)
{
    return PyLong_FromLong(SDL_BYTEORDER);
}

static PyObject*
pg_quit(PyObject* self)
{
    _pg_quit();
    Py_RETURN_NONE;
}

static void
_pg_quit(void)
{
    PyObject* quit;
    PyObject* privatefuncs;
    int num;

    if (!pg_quit_functions) {
        return;
    }

    privatefuncs = pg_quit_functions;
    pg_quit_functions = NULL;

    pg_uninstall_parachute();
    num = PyList_Size(privatefuncs);

    /*quit in reverse order*/
    while (num--) {
        quit = PyList_GET_ITEM(privatefuncs, num);
        if (PyCallable_Check(quit)) {
            PyObject_CallObject(quit, NULL);
        }
        else if (PyCapsule_CheckExact(quit)) {
            void* ptr = PyCapsule_GetPointer(quit, "quit");
            (*(void(*)(void)) ptr)();
        }
    }
    Py_DECREF(privatefuncs);

    pg_atexit_quit();
}

/* internal C API utility functions */
static int
pg_IntFromObj(PyObject* obj, int* val) {
    int tmp_val = PyInt_AsLong(obj);

    if (tmp_val == -1 && PyErr_Occurred()) {
        PyErr_Clear();
        return 0;
    }
    *val = tmp_val;
    return 1;
}

static int
pg_IntFromObjIndex(PyObject* obj, int _index, int* val)
{
    int result = 0;
    PyObject* item = PySequence_GetItem(obj, _index);

    if (item) {
        result = pg_IntFromObj(item, val);
        Py_DECREF(item);
    }
    return result;
}

static int
pg_TwoIntsFromObj(PyObject* obj, int* val1, int* val2)
{
    if (PyTuple_Check(obj) && PyTuple_Size(obj) == 1) {
        return pg_TwoIntsFromObj(PyTuple_GET_ITEM(obj, 0), val1, val2);
    }
    if (!PySequence_Check(obj) || PySequence_Length(obj) != 2) {
        return 0;
    }
    if (!pg_IntFromObjIndex(obj, 0, val1) || !pg_IntFromObjIndex(obj, 1, val2)) {
        return 0;
    }
    return 1;
}

static int
pg_FloatFromObj(PyObject* obj, float* val)
{
    float f= (float)PyFloat_AsDouble(obj);

    if (f==-1 && PyErr_Occurred()) {
        PyErr_Clear();
        return 0;
    }

    *val = f;
    return 1;
}

static int
pg_FloatFromObjIndex(PyObject* obj, int _index, float* val)
{
    int result = 0;
    PyObject* item = PySequence_GetItem(obj, _index);

    if (item) {
        result = pg_FloatFromObj(item, val);
        Py_DECREF(item);
    }
    return result;
}

static int
pg_TwoFloatsFromObj(PyObject* obj, float* val1, float* val2)
{
    if (PyTuple_Check(obj) && PyTuple_Size(obj) == 1) {
        return pg_TwoFloatsFromObj(PyTuple_GET_ITEM(obj, 0), val1, val2);
    }
    if (!PySequence_Check(obj) || PySequence_Length(obj) != 2) {
        return 0;
    }
    if (!pg_FloatFromObjIndex(obj, 0, val1) ||
        !pg_FloatFromObjIndex(obj, 1, val2)    ) {
        return 0;
    }
    return 1;
}

static int
pg_UintFromObj(PyObject* obj, Uint32* val)
{
    PyObject* longobj;

    if (PyNumber_Check(obj)) {
        if (!(longobj = PyNumber_Long(obj))) {
            return 0;
        }
        *val = (Uint32) PyLong_AsUnsignedLong(longobj);
        Py_DECREF(longobj);
        return 1;
    }
    return 0;
}

static int
pg_UintFromObjIndex(PyObject* obj, int _index, Uint32* val)
{
    int result = 0;
    PyObject* item = PySequence_GetItem(obj, _index);

    if (item) {
        result = pg_UintFromObj(item, val);
        Py_DECREF(item);
    }
    return result;
}

static int
pg_RGBAFromObj(PyObject* obj, Uint8* RGBA)
{
    int length;
    Uint32 val;

    if (PyTuple_Check(obj) && PyTuple_Size(obj) == 1) {
        return pg_RGBAFromObj(PyTuple_GET_ITEM(obj, 0), RGBA);
    }
    if (!PySequence_Check(obj)) {
        return 0;
    }
    length = PySequence_Length(obj);
    if (length < 3 || length > 4) {
        return 0;
    }
    if (!pg_UintFromObjIndex(obj, 0, &val) || val > 255) {
        return 0;
    }
    RGBA[0] = (Uint8)val;
    if (!pg_UintFromObjIndex(obj, 1, &val) || val > 255) {
        return 0;
    }
    RGBA[1] = (Uint8)val;
    if (!pg_UintFromObjIndex(obj, 2, &val) || val > 255) {
        return 0;
    }
    RGBA[2] = (Uint8)val;
    if (length == 4) {
        if (!pg_UintFromObjIndex(obj, 3, &val) || val > 255) {
            return 0;
        }
        RGBA[3] = (Uint8)val;
    }
    else {
        RGBA[3] = (Uint8)255;
    }
    return 1;
}

static PyObject*
pg_get_error(PyObject* self)
{
    return Text_FromUTF8(SDL_GetError());
}

static PyObject*
pg_set_error(PyObject *s, PyObject *args)
{
    char *errstring = NULL;

    if (!PyArg_ParseTuple(args, "s", &errstring)) {
        return NULL;
    }
    SDL_SetError("%s", errstring);
    Py_RETURN_NONE;
}


/*video init needs to be here, because of it's
 *important init order priority
 */
static void
pgVideo_AutoQuit(void)
{
    if (SDL_WasInit(SDL_INIT_VIDEO)) {
        SDL_QuitSubSystem(SDL_INIT_VIDEO);
    }
}

static int
pgVideo_AutoInit(void)
{
    if (!SDL_WasInit(SDL_INIT_VIDEO)) {
        int status;
#if defined(__APPLE__) && defined(darwin)
        PyObject *module = PyImport_ImportModule("pygame.macosx");
        PyObject *rval;

        if (!module) {
            printf("ERROR: pygame.macosx import FAILED\n");
            return -1;
        }

        rval = PyObject_CallMethod(module, "Video_AutoInit", "");
        Py_DECREF(module);
        if (!rval) {
            printf("ERROR: pygame.macosx.Video_AutoInit() call FAILED\n");
            return -1;
        }

        status = PyObject_IsTrue(rval);
        Py_DECREF(rval);
        if (status != 1) {
            return 0;
        }
#endif
        status = SDL_InitSubSystem(SDL_INIT_VIDEO);
        if (status) {
            return 0;
        }

#if IS_SDLv1
        SDL_EnableUNICODE(1);
#endif /* IS_SDLv1 */

        /*we special case the video quit to last now*/
        /*pg_RegisterQuit(pgVideo_AutoQuit);*/
    }
    return 1;
}

/*array interface*/

static int
pgGetArrayStruct(PyObject* obj,
                 PyObject** cobj_p,
                 PyArrayInterface** inter_p)
{
    PyObject* cobj = PyObject_GetAttrString(obj, "__array_struct__");
    PyArrayInterface* inter = NULL;

    if (cobj == NULL) {
        if (PyErr_ExceptionMatches(PyExc_AttributeError)) {
            PyErr_Clear();
            PyErr_SetString(PyExc_ValueError,
                            "no C-struct array interface");
        }
        return -1;
    }

#if PG_HAVE_COBJECT
    if (PyCObject_Check(cobj)) {
        inter = (PyArrayInterface *)PyCObject_AsVoidPtr(cobj);
    }
#endif
#if PG_HAVE_CAPSULE
    if (PyCapsule_IsValid(cobj, NULL)) {
        inter = (PyArrayInterface*)PyCapsule_GetPointer(cobj, NULL);
    }
#endif
    if (inter == NULL || inter->two != 2 /* conditional or */) {
        Py_DECREF(cobj);
        PyErr_SetString(PyExc_ValueError, "invalid array interface");
        return -1;
    }

    *cobj_p = cobj;
    *inter_p = inter;
    return 0;
}

static PyObject*
pgArrayStruct_AsDict(PyArrayInterface* inter_p)
{
    PyObject *dictobj = Py_BuildValue("{sisNsNsNsN}",
                                      "version", (int)3,
                                      "typestr", _pg_typekind_as_str(inter_p),
                                      "shape", _pg_shape_as_tuple(inter_p),
                                      "strides", _pg_strides_as_tuple(inter_p),
                                      "data", _pg_data_as_tuple(inter_p));

    if (!dictobj) {
        return 0;
    }
    if (inter_p->flags & PAI_ARR_HAS_DESCR) {
        if (!inter_p->descr) {
            Py_DECREF(dictobj);
            PyErr_SetString(PyExc_ValueError,
                            "Array struct has descr flag set"
                            " but no descriptor");
            return 0;
        }
        if (PyDict_SetItemString(dictobj, "descr", inter_p->descr)) {
            Py_DECREF(dictobj);
            return 0;
        }
    }
    return dictobj;
}

static PyObject*
pgBuffer_AsArrayInterface(Py_buffer* view_p)
{
    return Py_BuildValue("{sisNsNsNsN}",
                         "version", (int)3,
                         "typestr", pg_view_get_typestr_obj(view_p),
                         "shape", pg_view_get_shape_obj(view_p),
                         "strides", pg_view_get_strides_obj(view_p),
                         "data", pg_view_get_data_obj(view_p));
}

static PyObject*
pgBuffer_AsArrayStruct(Py_buffer* view_p)
{
    void *cinter_p  = _pg_new_capsuleinterface(view_p);
    PyObject *capsule;

    if (!cinter_p) {
        return 0;
    }
#if PY3
    capsule = PyCapsule_New(cinter_p, 0, _pg_capsule_PyMem_Free);
#else
    capsule = PyCObject_FromVoidPtr(cinter_p, PyMem_Free);
#endif
    if (!capsule) {
        PyMem_Free(cinter_p);
        return 0;
    }
    return capsule;
}

static pgCapsuleInterface*
_pg_new_capsuleinterface(Py_buffer *view_p)
{
    int ndim = view_p->ndim;
    Py_ssize_t cinter_size;
    pgCapsuleInterface *cinter_p;
    int i;

    cinter_size = (sizeof(pgCapsuleInterface) +
                   sizeof(Py_intptr_t) * (2 * ndim - 1));
    cinter_p = (pgCapsuleInterface *)PyMem_Malloc(cinter_size);
    if (!cinter_p) {
        PyErr_NoMemory();
        return 0;
    }
    cinter_p->inter.two = 2;
    cinter_p->inter.nd = ndim;
    cinter_p->inter.typekind = _pg_as_arrayinter_typekind(view_p);
    cinter_p->inter.itemsize = view_p->itemsize;
    cinter_p->inter.flags = _pg_as_arrayinter_flags(view_p);
    if (view_p->shape) {
        cinter_p->inter.shape = cinter_p->imem;
        for (i = 0; i < ndim; ++i) {
            cinter_p->inter.shape[i] = (Py_intptr_t)view_p->shape[i];
        }
    }
    if (view_p->strides) {
        cinter_p->inter.strides = cinter_p->imem + ndim;
        for (i = 0; i < ndim; ++i) {
            cinter_p->inter.strides[i] = (Py_intptr_t)view_p->strides[i];
        }
    }
    cinter_p->inter.data = view_p->buf;
    cinter_p->inter.descr = 0;
    return cinter_p;
}

#if PY3
static void
_pg_capsule_PyMem_Free(PyObject *capsule)
{
    PyMem_Free(PyCapsule_GetPointer(capsule, 0));
}
#endif

static int
_pg_as_arrayinter_flags(Py_buffer* view_p)
{
    int inter_flags = PAI_ALIGNED; /* atomic int types always aligned */

    if (!view_p->readonly) {
        inter_flags |= PAI_WRITEABLE;
    }
    inter_flags |= _pg_buffer_is_byteswapped(view_p) ? 0 : PAI_NOTSWAPPED;
    if (PyBuffer_IsContiguous(view_p, 'C')) {
        inter_flags |= PAI_CONTIGUOUS;
    }
    if (PyBuffer_IsContiguous(view_p, 'F')) {
        inter_flags |= PAI_FORTRAN;
    }
    return inter_flags;
}

static PyObject*
pg_view_get_typestr_obj(Py_buffer* view)
{
    return Text_FromFormat("%c%c%i",
                           _pg_as_arrayinter_byteorder(view),
                           _pg_as_arrayinter_typekind(view),
                           (int)view->itemsize);
}

static PyObject*
pg_view_get_shape_obj(Py_buffer* view)
{
    PyObject *shapeobj = PyTuple_New(view->ndim);
    PyObject *lengthobj;
    Py_ssize_t i;

    if (!shapeobj) {
        return 0;
    }
    for (i = 0; i < view->ndim; ++i) {
        lengthobj = PyInt_FromLong((long)view->shape[i]);
        if (!lengthobj) {
            Py_DECREF(shapeobj);
            return 0;
        }
        PyTuple_SET_ITEM(shapeobj, i, lengthobj);
    }
    return shapeobj;
}

static PyObject*
pg_view_get_strides_obj(Py_buffer* view)
{
    PyObject *shapeobj = PyTuple_New(view->ndim);
    PyObject *lengthobj;
    Py_ssize_t i;

    if (!shapeobj) {
        return 0;
    }
    for (i = 0; i < view->ndim; ++i) {
        lengthobj = PyInt_FromLong((long)view->strides[i]);
        if (!lengthobj) {
            Py_DECREF(shapeobj);
            return 0;
        }
        PyTuple_SET_ITEM(shapeobj, i, lengthobj);
    }
    return shapeobj;
}

static PyObject*
pg_view_get_data_obj(Py_buffer* view)
{
    return Py_BuildValue("NN",
                         PyLong_FromVoidPtr(view->buf),
                         PyBool_FromLong((long)view->readonly));
}

static char
_pg_as_arrayinter_typekind(Py_buffer* view)
{
    char type = view->format ? view->format[0] : 'B';
    char typekind = 'V';

    switch (type) {

    case '<':
    case '>':
    case '=':
    case '@':
    case '!':
        type = view->format[1];
    }
    switch (type) {

    case 'b':
    case 'h':
    case 'i':
    case 'l':
    case 'q':
        typekind = 'i';
        break;
    case 'B':
    case 'H':
    case 'I':
    case 'L':
    case 'Q':
        typekind = 'u';
        break;
    case 'f':
    case 'd':
        typekind = 'f';
        break;
    default:
        /* Unknown type */
        typekind = 'V';
    }
    return typekind;
}

static char
_pg_as_arrayinter_byteorder(Py_buffer* view)
{
    char format_0 = view->format ? view->format[0] : 'B';
    char byteorder;

    if (view->itemsize == 1) {
        byteorder = '|';
    }
    else {
        switch (format_0) {

        case '<':
        case '>':
            byteorder = format_0;
            break;
        case '!':
            byteorder = '>';
            break;
        case 'c':
        case 's':
        case 'p':
        case 'b':
        case 'B':
            byteorder = '|';
            break;
        default:
            byteorder = PAI_MY_ENDIAN;
        }
    }
    return byteorder;
}


static PyObject*
_pg_shape_as_tuple(PyArrayInterface* inter_p)
{
    PyObject *shapeobj = PyTuple_New((Py_ssize_t)inter_p->nd);
    PyObject *lengthobj;
    Py_ssize_t i;

    if (!shapeobj) {
        return 0;
    }
    for (i = 0; i < inter_p->nd; ++i) {
        lengthobj = PyInt_FromLong((long)inter_p->shape[i]);
        if (!lengthobj) {
            Py_DECREF(shapeobj);
            return 0;
        }
        PyTuple_SET_ITEM(shapeobj, i, lengthobj);
    }
    return shapeobj;
}

static PyObject*
_pg_typekind_as_str(PyArrayInterface* inter_p)
{
    return Text_FromFormat("%c%c%i",
                           inter_p->itemsize > 1 ?
                               (inter_p->flags & PAI_NOTSWAPPED ?
                                    PAI_MY_ENDIAN :
                                    PAI_OTHER_ENDIAN) :
                               '|',
                           inter_p->typekind, inter_p->itemsize);
}

static PyObject*
_pg_strides_as_tuple(PyArrayInterface* inter_p)
{
    PyObject *stridesobj = PyTuple_New((Py_ssize_t)inter_p->nd);
    PyObject *lengthobj;
    Py_ssize_t i;

    if (!stridesobj) {
        return 0;
    }
    for (i = 0; i < inter_p->nd; ++i) {
        lengthobj = PyInt_FromLong((long)inter_p->strides[i]);
        if (!lengthobj) {
            Py_DECREF(stridesobj);
            return 0;
        }
        PyTuple_SET_ITEM(stridesobj, i, lengthobj);
    }
    return stridesobj;
}

static PyObject*
_pg_data_as_tuple(PyArrayInterface* inter_p)
{
    long readonly = (inter_p->flags & PAI_WRITEABLE) == 0;

    return Py_BuildValue("NN",
                         PyLong_FromVoidPtr(inter_p->data),
                         PyBool_FromLong(readonly));
}

static PyObject*
pg_get_array_interface(PyObject* self, PyObject* arg)
{
    PyObject *cobj;
    PyArrayInterface *inter_p;
    PyObject *dictobj;

    if (pgGetArrayStruct(arg, &cobj, &inter_p)) {
        return 0;
    }
    dictobj = pgArrayStruct_AsDict(inter_p);
    Py_DECREF(cobj);
    return dictobj;
}

static int
pgObject_GetBuffer(PyObject* obj, pg_buffer* pg_view_p, int flags)
{
    Py_buffer* view_p = (Py_buffer*)pg_view_p;
    PyObject* cobj = 0;
    PyObject* dict = 0;
    PyArrayInterface* inter_p = 0;
    char *fchar_p;
    int success = 0;

    pg_view_p->release_buffer = _pg_release_buffer_generic;
    view_p->len = 0;

#ifndef NDEBUG
    /* Allow a callback to assert that it recieved a pg_buffer,
       not a Py_buffer */
    flags |= PyBUF_PYGAME;
#endif

#if PG_ENABLE_NEWBUF

    if (PyObject_CheckBuffer(obj)) {
        if (PyObject_GetBuffer(obj, view_p, flags)) {
            return -1;
        }
        pg_view_p->release_buffer = PyBuffer_Release;

        /* Check the format is a numeric type or pad bytes
         */
        fchar_p = view_p->format;
        /* Skip valid size/byte order code */
        switch (*fchar_p) {

        case '@':
        case '=':
        case '<':
        case '>':
        case '!':
            ++fchar_p;
            break;

        /* default: assume it is a format type character or item count */
        }
        /* Skip a leading digit */
        switch (*fchar_p) {

        case '1':
            /* valid count for all item types */
            ++fchar_p;
            break;
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
        case '8':
        case '9':
            /* only valid as a pad byte count */
            if (*(fchar_p + 1) == 'x') {
                ++fchar_p;
            }
            break;

        /* default: assume it is a format character */
        }
        /* verify is a format type character */
        switch (*fchar_p) {

        case 'b':
        case 'B':
        case 'h':
        case 'H':
        case 'i':
        case 'I':
        case 'l':
        case 'L':
        case 'q':
        case 'Q':
        case 'f':
        case 'd':
        case 'x':
            ++fchar_p;
            break;
        default:
            pgBuffer_Release(pg_view_p);
            PyErr_SetString(PyExc_ValueError,
                            "Unsupported array element type");
            return -1;
        }
        if (*fchar_p != '\0') {
            pgBuffer_Release(pg_view_p);
            PyErr_SetString(PyExc_ValueError,
                            "Arrays of records are unsupported");
            return -1;
        }
        success = 1;
    }

#endif
    if (!success && pgGetArrayStruct(obj, &cobj, &inter_p) == 0) {
        if (pgArrayStruct_AsBuffer(pg_view_p, cobj, inter_p, flags)) {
            Py_DECREF(cobj);
            return -1;
        }
        Py_INCREF(obj);
        view_p->obj = obj;
        Py_DECREF(cobj);
        success = 1;
    }
    else if (!success) {
        PyErr_Clear();
    }

    if (!success && pgGetArrayInterface(&dict, obj) == 0) {
        if (pgDict_AsBuffer(pg_view_p, dict, flags)) {
            Py_DECREF(dict);
            return -1;
        }
        Py_INCREF(obj);
        view_p->obj = obj;
        Py_DECREF(dict);
        success = 1;
    }
    else if (!success) {
        PyErr_Clear();
    }

    if (!success) {
        PyErr_Format(PyExc_ValueError,
                      "%s object does not export an array buffer",
                      Py_TYPE(obj)->tp_name);
        return -1;
    }
    return 0;
}

static void
pgBuffer_Release(pg_buffer* pg_view_p)
{
    assert(pg_view_p && pg_view_p->release_buffer);
    pg_view_p->release_buffer((Py_buffer*)pg_view_p);
}

static void
_pg_release_buffer_generic(Py_buffer* view_p)
{
    if (view_p->obj) {
        Py_XDECREF(view_p->obj);
        view_p->obj = 0;
    }
}

static void
_pg_release_buffer_array(Py_buffer* view_p)
{
    /* This is deliberately made safe for use on an unitialized *view_p */
    if (view_p->internal) {
        PyMem_Free(view_p->internal);
        view_p->internal = 0;
    }
    _pg_release_buffer_generic(view_p);
}

static int
_pg_buffer_is_byteswapped(Py_buffer* view)
{
    if (view->format) {
        switch (view->format[0]) {

        case '<':
            return SDL_BYTEORDER != SDL_LIL_ENDIAN;
        case '>':
        case '!':
            return SDL_BYTEORDER != SDL_BIG_ENDIAN;
        }
    }
    return 0;
}

static int
pgGetArrayInterface(PyObject **dict, PyObject *obj)
{
    PyObject* inter = PyObject_GetAttrString(obj, "__array_interface__");

    if (inter == NULL) {
        if (PyErr_ExceptionMatches(PyExc_AttributeError)) {
                PyErr_Clear();
                PyErr_SetString(PyExc_ValueError, "no array interface");
        }
        return -1;
    }
    if (!PyDict_Check(inter)) {
        PyErr_Format(PyExc_ValueError,
                     "expected '__array_interface__' to return a dict: got %s",
                     Py_TYPE(dict)->tp_name);
        Py_DECREF(inter);
        return -1;
    }
    *dict = inter;
    return 0;
}

static int
pgArrayStruct_AsBuffer(pg_buffer* pg_view_p, PyObject* cobj,
                       PyArrayInterface* inter_p, int flags)
{
    pg_view_p->release_buffer = _pg_release_buffer_array;
    if (_pg_arraystruct_as_buffer((Py_buffer*)pg_view_p,
                                cobj, inter_p, flags)) {
        pgBuffer_Release(pg_view_p);
        return -1;
    }
    return 0;
}

static int
_pg_arraystruct_as_buffer(Py_buffer* view_p, PyObject* cobj,
                          PyArrayInterface* inter_p, int flags)
{
    pgViewInternals* internal_p;
    Py_ssize_t sz = (sizeof(pgViewInternals)
                     + (2 * inter_p->nd - 1) * sizeof(Py_ssize_t));
    int readonly = inter_p->flags & PAI_WRITEABLE ? 0 : 1;
    Py_ssize_t i;

    view_p->obj = 0;
    view_p->internal = 0;
    if (PyBUF_HAS_FLAG(flags, PyBUF_WRITABLE) && readonly) {
        PyErr_SetString(pgExc_BufferError,
                        "require writable buffer, but it is read-only");
        return -1;
    }
    if (PyBUF_HAS_FLAG(flags, PyBUF_ANY_CONTIGUOUS)) {
        if (!(inter_p->flags & (PAI_CONTIGUOUS | PAI_FORTRAN))) {
            PyErr_SetString(pgExc_BufferError,
                            "buffer data is not contiguous");
            return -1;
        }
    }
    else if (PyBUF_HAS_FLAG(flags, PyBUF_C_CONTIGUOUS)) {
        if (!(inter_p->flags & PAI_CONTIGUOUS)) {
            PyErr_SetString(pgExc_BufferError,
                            "buffer data is not C contiguous");
            return -1;
        }
    }
    else if (PyBUF_HAS_FLAG(flags, PyBUF_F_CONTIGUOUS)) {
        if (!(inter_p->flags & PAI_FORTRAN)) {
            PyErr_SetString(pgExc_BufferError,
                            "buffer data is not F contiguous");
            return -1;
        }
    }
    internal_p = (pgViewInternals*)PyMem_Malloc(sz);
    if (!internal_p) {
        PyErr_NoMemory();
        return -1;
    }
    view_p->internal = internal_p;
    if (PyBUF_HAS_FLAG(flags, PyBUF_FORMAT)) {
        if (_pg_arraystruct_to_format(internal_p->format,
                                   inter_p,
                                   sizeof(internal_p->format))) {
            return -1;
        }
    view_p->format = internal_p->format;
    }
    else {
        view_p->format = 0;
    }
    view_p->buf = inter_p->data;
    view_p->itemsize = (Py_ssize_t)inter_p->itemsize;
    view_p->readonly = readonly;
    if (PyBUF_HAS_FLAG(flags, PyBUF_ND)) {
        view_p->ndim = (Py_ssize_t)inter_p->nd;
        view_p->shape = internal_p->imem;
        for (i = 0; i < view_p->ndim; ++i) {
            view_p->shape[i] = (Py_ssize_t)inter_p->shape[i];
        }
    }
    else if (inter_p->flags & PAI_CONTIGUOUS) {
        view_p->ndim = 0;
        view_p->shape = 0;
    }
    else {
        PyErr_SetString(pgExc_BufferError,
                        "buffer data is not C contiguous, shape needed");
        return -1;
    }
    if (PyBUF_HAS_FLAG(flags, PyBUF_STRIDES)) {
        view_p->strides = view_p->shape + inter_p->nd;
        for (i = 0; i < view_p->ndim; ++i) {
            view_p->strides[i] = (Py_ssize_t)inter_p->strides[i];
        }
    }
    else if (inter_p->flags & (PAI_CONTIGUOUS | PAI_FORTRAN)) {
        view_p->strides = 0;
    }
    else {
        PyErr_SetString(pgExc_BufferError,
                        "buffer is not contiguous, strides needed");
        return -1;
    }
    view_p->suboffsets = 0;
    view_p->len = view_p->itemsize;
    for (i = 0; i < inter_p->nd; ++i) {
        view_p->len *= (Py_ssize_t)inter_p->shape[i];
    }
    return 0;
}

static int _pg_arraystruct_to_format(char* format,
                                     PyArrayInterface* inter_p,
                                     int max_format_len)
{
    char* fchar_p = format;

    assert(max_format_len >= 4);
    switch (inter_p->typekind) {

    case 'i':
        *fchar_p = (inter_p->flags & PAI_NOTSWAPPED ?
                    BUF_MY_ENDIAN : BUF_OTHER_ENDIAN);
        ++fchar_p;
        switch (inter_p->itemsize) {

        case 1:
            *fchar_p = 'b';
            break;
        case 2:
            *fchar_p = 'h';
            break;
        case 4:
            *fchar_p = 'i';
            break;
        case 8:
            *fchar_p = 'q';
            break;
        default:
            PyErr_Format(PyExc_ValueError,
                         "Unsupported signed integer size %d",
                         (int)inter_p->itemsize);
            return -1;
        }
        break;
    case 'u':
        *fchar_p = (inter_p->flags & PAI_NOTSWAPPED ?
                    BUF_MY_ENDIAN : BUF_OTHER_ENDIAN);
        ++fchar_p;
        switch (inter_p->itemsize) {

        case 1:
            *fchar_p = 'B';
            break;
        case 2:
            *fchar_p = 'H';
            break;
        case 4:
            *fchar_p = 'I';
            break;
        case 8:
            *fchar_p = 'Q';
            break;
        default:
            PyErr_Format(PyExc_ValueError,
                         "Unsupported unsigned integer size %d",
                         (int)inter_p->itemsize);
            return -1;
        }
        break;
    case 'f':
        *fchar_p = (inter_p->flags & PAI_NOTSWAPPED ?
                    BUF_MY_ENDIAN : BUF_OTHER_ENDIAN);
        ++fchar_p;
        switch (inter_p->itemsize) {

        case 4:
            *fchar_p = 'f';
            break;
        case 8:
            *fchar_p = 'd';
            break;
        default:
            PyErr_Format(PyExc_ValueError,
                         "Unsupported float size %d",
                         (int)inter_p->itemsize);
            return -1;
        }
        break;
    case 'V':
        if (inter_p->itemsize > 9) {
            PyErr_Format(PyExc_ValueError,
                         "Unsupported void size %d",
                         (int)inter_p->itemsize);
            return -1;
        }
        switch (inter_p->itemsize) {

        case 1:
            *fchar_p = '1';
            break;
        case 2:
            *fchar_p = '2';
            break;
        case 3:
            *fchar_p = '3';
            break;
        case 4:
            *fchar_p = '4';
            break;
        case 5:
            *fchar_p = '5';
            break;
        case 6:
            *fchar_p = '6';
            break;
        case 7:
            *fchar_p = '7';
            break;
        case 8:
            *fchar_p = '8';
            break;
        case 9:
            *fchar_p = '9';
            break;
        default:
            PyErr_Format(PyExc_ValueError,
                         "Unsupported void size %d",
                         (int)inter_p->itemsize);
            return -1;
        }
        ++fchar_p;
        *fchar_p = 'x';
        break;
    default:
        PyErr_Format(PyExc_ValueError,
                     "Unsupported value type '%c'",
                     (int)inter_p->typekind);
        return -1;
    }
    ++fchar_p;
    *fchar_p = '\0';
    return 0;
}

static int
pgDict_AsBuffer(pg_buffer* pg_view_p, PyObject* dict, int flags)
{
    PyObject* shape = PyDict_GetItemString(dict, "shape");
    PyObject* typestr = PyDict_GetItemString(dict, "typestr");
    PyObject* data = PyDict_GetItemString(dict, "data");
    PyObject* strides = PyDict_GetItemString(dict, "strides");

    if (_pg_shape_check(shape)) {
        return -1;
    }
    if (_pg_typestr_check(typestr)) {
        return -1;
    }
    if (_pg_data_check(data)) {
        return -1;
    }
    if (_pg_strides_check(strides)) {
        return -1;
    }
    pg_view_p->release_buffer = _pg_release_buffer_array;
    if (_pg_values_as_buffer((Py_buffer*)pg_view_p, flags,
                             typestr, shape, data, strides)) {
        pgBuffer_Release(pg_view_p);
        return -1;
    }
    return 0;
}

static int
_pg_shape_check(PyObject* op)
{
    if (!op) {
        PyErr_SetString(PyExc_ValueError,
                        "required 'shape' item is missing");
        return -1;
    }
    if (!_pg_is_int_tuple(op)) {
        PyErr_SetString(PyExc_ValueError,
                        "expected a tuple of ints for 'shape'");
        return -1;
    }
    if (PyTuple_GET_SIZE(op) == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "expected 'shape' to be at least one-dimensional");
        return -1;
    }
    return 0;
}

static int
_pg_typestr_check(PyObject *op)
{
    if (!op) {
        PyErr_SetString(PyExc_ValueError,
                        "required 'typestr' item is missing");
        return -1;
    }
    if (PyUnicode_Check(op)) {
        if (PyUnicode_GET_SIZE(op) != 3) {
            PyErr_SetString(PyExc_ValueError,
                            "expected 'typestr' to be length 3");
            return -1;
        }
    }
    else if (Bytes_Check(op)) {
        if (Bytes_GET_SIZE(op) != 3) {
            PyErr_SetString(PyExc_ValueError,
                            "expected 'typestr' to be length 3");
            return -1;
        }
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                        "expected a string for 'typestr'");
        return -1;
    }
    return 0;
}

static int
_pg_data_check(PyObject *op)
{
    PyObject* item;

    if (!op) {
        PyErr_SetString(PyExc_ValueError,
                        "required 'data' item is missing");
        return -1;
    }
    if (!PyTuple_Check(op)) {
        PyErr_SetString(PyExc_ValueError,
                        "expected a tuple for 'data'");
        return -1;
    }
    if (PyTuple_GET_SIZE(op) != 2) {
        PyErr_SetString(PyExc_ValueError,
                        "expected a length 2 tuple for 'data'");
        return -1;
    }
    item = PyTuple_GET_ITEM(op, 0);
    if (!INT_CHECK(item)) {
        PyErr_SetString(PyExc_ValueError,
                        "expected an int for item 0 of 'data'");
        return -1;
    }
    return 0;
}

static int
_pg_strides_check(PyObject *op)
{
    if (op && !_pg_is_int_tuple(op) /* Conditional && */) {
        PyErr_SetString(PyExc_ValueError,
                        "expected a tuple of ints for 'strides'");
        return -1;
    }
    return 0;
}

static int
_pg_is_int_tuple(PyObject* op)
{
    Py_ssize_t i;
    Py_ssize_t n;
    PyObject* ip;

    if (!PyTuple_Check(op)) {
        return 0;
    }
    n = PyTuple_GET_SIZE(op);
    for (i = 0; i != n; ++i) {
        ip = PyTuple_GET_ITEM(op, i);
        if (!INT_CHECK(ip)) {
            return 0;
        }
    }
    return 1;
}

static int
_pg_values_as_buffer(Py_buffer* view_p, int flags,
                     PyObject* typestr,
                     PyObject* shape,
                     PyObject* data,
                     PyObject* strides)
{
    Py_ssize_t ndim = PyTuple_GET_SIZE(shape);
    pgViewInternals* internal_p;
    Py_ssize_t sz;
    int i;

    assert(ndim > 0);
    view_p->obj = 0;
    view_p->internal = 0;
    if (strides && PyTuple_GET_SIZE(strides) != ndim /* Cond. && */) {
        PyErr_SetString(PyExc_ValueError,
                        "'shape' and 'strides' are not the same length");
        return -1;
    }
    view_p->ndim = ndim;
    view_p->buf = PyLong_AsVoidPtr(PyTuple_GET_ITEM(data, 0));
    if (!view_p->buf && PyErr_Occurred()) {
        return -1;
    }
    view_p->readonly = PyObject_IsTrue(PyTuple_GET_ITEM(data, 1));
    if (view_p->readonly == -1) {
        return -1;
    }
    if (PyBUF_HAS_FLAG(flags, PyBUF_WRITABLE) && view_p->readonly) {
        PyErr_SetString(pgExc_BufferError,
                        "require writable buffer, but it is read-only");
        return -1;
    }
    sz = sizeof (pgViewInternals) + (2 * ndim - 1) * sizeof(Py_ssize_t);
    internal_p = (pgViewInternals*)PyMem_Malloc(sz);
    if (!internal_p) {
        PyErr_NoMemory();
        return -1;
    }
    view_p->internal = internal_p;
    view_p->format = internal_p->format;
    view_p->shape = internal_p->imem;
    view_p->strides = internal_p->imem + ndim;
    if (_pg_typestr_as_format(typestr, view_p->format, &view_p->itemsize)) {
        return -1;
    }
    if (_pg_int_tuple_as_ssize_arr(shape, view_p->shape)) {
        return -1;
    }
    if (strides) {
        if (_pg_int_tuple_as_ssize_arr(strides, view_p->strides)) {
            return -1;
        }
    }
    else if (PyBUF_HAS_FLAG(flags, PyBUF_STRIDES)) {
        view_p->strides[ndim - 1] = view_p->itemsize;
        for (i = ndim - 1; i != 0; --i) {
            view_p->strides[i - 1] = view_p->shape[i] * view_p->strides[i];
        }
    }
    else {
        view_p->strides = 0;
    }
    view_p->suboffsets = 0;
    view_p->len = view_p->itemsize;
    for (i = 0; i < view_p->ndim; ++i) {
        view_p->len *= view_p->shape[i];
    }
    if (PyBUF_HAS_FLAG(flags, PyBUF_ANY_CONTIGUOUS)) {
        if (!PyBuffer_IsContiguous(view_p, 'A')) {
            PyErr_SetString(pgExc_BufferError,
                            "buffer data is not contiguous");
            return -1;
        }
    }
    else if (PyBUF_HAS_FLAG(flags, PyBUF_C_CONTIGUOUS)) {
        if (!PyBuffer_IsContiguous(view_p, 'C')) {
            PyErr_SetString(pgExc_BufferError,
                            "buffer data is not C contiguous");
            return -1;
        }
    }
    else if (PyBUF_HAS_FLAG(flags, PyBUF_F_CONTIGUOUS)) {
        if (!PyBuffer_IsContiguous(view_p, 'F')) {
            PyErr_SetString(pgExc_BufferError,
                            "buffer data is not F contiguous");
            return -1;
        }
    }
    if (!PyBUF_HAS_FLAG(flags, PyBUF_STRIDES)) {
        if (PyBuffer_IsContiguous(view_p, 'C')) {
            view_p->strides = 0;
        }
        else {
            PyErr_SetString(pgExc_BufferError,
                            "buffer data is not C contiguous, strides needed");
            return -1;
        }
    }
    if (!PyBUF_HAS_FLAG(flags, PyBUF_ND)) {
        if (PyBuffer_IsContiguous(view_p, 'C')) {
            view_p->shape = 0;
        }
        else {
            PyErr_SetString(pgExc_BufferError,
                            "buffer data is not C contiguous, shape needed");
            return -1;
        }
    }
    if (!PyBUF_HAS_FLAG(flags, PyBUF_FORMAT)) {
        view_p->format = 0;
    }
    if (!PyBUF_HAS_FLAG(flags, PyBUF_ND)) {
        view_p->ndim = 0;
    }
    return 0;
}

static int
_pg_int_tuple_as_ssize_arr(PyObject *tp, Py_ssize_t* arr)
{
    Py_ssize_t i;
    Py_ssize_t n = PyTuple_GET_SIZE(tp);

    for (i = 0; i != n; ++i) {
        arr[i] = PyInt_AsSsize_t(PyTuple_GET_ITEM(tp, i));
        if (arr[i] == -1 && PyErr_Occurred()) {
            return -1;
        }
    }
    return 0;
}

static int
_pg_typestr_as_format(PyObject* sp, char* format, Py_ssize_t* itemsize_p)
{
    const char* typestr;
    char *fchar_p = format;
    int is_swapped = 0;
    Py_ssize_t itemsize = 0;

    if (PyUnicode_Check(sp)) {
        sp = PyUnicode_AsASCIIString(sp);
        if (!sp) {
            return -1;
        }
    }
    else {
        Py_INCREF(sp);
    }
    typestr = Bytes_AsString(sp);
    switch (typestr[0]) {

    case PAI_MY_ENDIAN:
    case '|':
        break;
    case PAI_OTHER_ENDIAN:
        is_swapped = 1;
        break;
    default:
        PyErr_Format(PyExc_ValueError, "unsupported typestr %s", typestr);
        Py_DECREF(sp);
        return -1;
    }
    switch (typestr[1]) {

    case 'i':
    case 'u':
        switch (typestr[2]) {

        case '1':
            *fchar_p = 'B';
            itemsize = 1;
            break;
        case '2':
            *fchar_p = is_swapped ? BUF_OTHER_ENDIAN : BUF_MY_ENDIAN;
            ++fchar_p;
            *fchar_p = 'H';
            itemsize = 2;
            break;
        case '3':
            *fchar_p = '3';
            ++fchar_p;
            *fchar_p = 'x';
            itemsize = 3;
            break;
        case '4':
            *fchar_p = is_swapped ? BUF_OTHER_ENDIAN : BUF_MY_ENDIAN;
            ++fchar_p;
            *fchar_p = 'I';
            itemsize = 4;
            break;
        case '5':
            *fchar_p = '5';
            ++fchar_p;
            *fchar_p = 'x';
            itemsize = 5;
            break;
        case '6':
            *fchar_p = '6';
            ++fchar_p;
            *fchar_p = 'x';
            itemsize = 6;
            break;
        case '7':
            *fchar_p = '7';
            ++fchar_p;
            *fchar_p = 'x';
            itemsize = 7;
            break;
        case '8':
            *fchar_p = is_swapped ? BUF_OTHER_ENDIAN : BUF_MY_ENDIAN;
            ++fchar_p;
            *fchar_p = 'Q';
            itemsize = 8;
            break;
        case '9':
            *fchar_p = '9';
            ++fchar_p;
            *fchar_p = 'x';
            itemsize = 9;
            break;
        default:
            PyErr_Format(PyExc_ValueError, "unsupported typestr %s", typestr);
            Py_DECREF(sp);
            return -1;
        }
        if (typestr[1] == 'i') {
            /* This leaves 'x' uneffected. */
            *fchar_p = tolower(*fchar_p);
        }
        break;
    case 'f':
        *fchar_p = is_swapped ? BUF_OTHER_ENDIAN : BUF_MY_ENDIAN;
        ++fchar_p;
        switch (typestr[2]) {

        case '4':
            *fchar_p = 'f';
            itemsize = 4;
            break;
        case '8':
            *fchar_p = 'd';
            itemsize = 8;
            break;
        default:
            PyErr_Format(PyExc_ValueError, "unsupported typestr %s", typestr);
            Py_DECREF(sp);
            return -1;
        }
        break;
    case 'V':
        switch (typestr[2]) {

        case '1':
            *fchar_p = '1';
            itemsize = 1;
            break;
        case '2':
            *fchar_p = '2';
            itemsize = 2;
            break;
        case '3':
            *fchar_p = '3';
            itemsize = 3;
            break;
        case '4':
            *fchar_p = '4';
            itemsize = 4;
            break;
        case '5':
            *fchar_p = '5';
            itemsize = 5;
            break;
        case '6':
            *fchar_p = '6';
            itemsize = 6;
            break;
        case '7':
            *fchar_p = '7';
            itemsize = 7;
            break;
        case '8':
            *fchar_p = '8';
            itemsize = 8;
            break;
        case '9':
            *fchar_p = '9';
            itemsize = 9;
            break;
        default:
            PyErr_Format(PyExc_ValueError, "unsupported typestr %s", typestr);
            Py_DECREF(sp);
            return -1;
        }
        ++fchar_p;
        *fchar_p = 'x';
        break;
    default:
        PyErr_Format(PyExc_ValueError, "unsupported typestr %s", typestr);
        Py_DECREF(sp);
        return -1;
    }
    Py_DECREF(sp);
    ++fchar_p;
    *fchar_p = '\0';
    *itemsize_p = itemsize;
    return 0;
}


#if IS_SDLv2
/*Default window(display)*/
static SDL_Window*
pg_GetDefaultWindow(void)
{
    return pg_default_window;
}

static void
pg_SetDefaultWindow(SDL_Window* win)
{
    /*Allows a window to be replaced by itself*/
    if (win == pg_default_window) {
        return;
    }
    if (pg_default_window) {
        SDL_DestroyWindow(pg_default_window);
    }
    pg_default_window = win;
}

static PyObject*
pg_GetDefaultWindowSurface(void)
{
    /*return a borrowed reference*/
    return pg_default_screen;
}

static void
pg_SetDefaultWindowSurface(PyObject* screen)
{
    /*a screen surface can be replaced with itself*/
    if (screen == pg_default_screen) {
        return;
    }
    Py_XINCREF(screen);
    if (pg_default_screen) {
        PySurface_AsSurface(pg_default_screen) = NULL;
        Py_DECREF(pg_default_screen);
    }
    pg_default_screen = screen;
}
#endif /* IS_SDLv2 */

/*error signal handlers(replacing SDL parachute)*/
static void
pygame_parachute(int sig)
{
#ifdef HAVE_SIGNAL_H
    char* signaltype;

    signal(sig, SIG_DFL);
    switch (sig) {

    case SIGSEGV:
        signaltype = "(pygame parachute) Segmentation Fault";
        break;
#ifdef SIGBUS
#if SIGBUS != SIGSEGV
    case SIGBUS:
        signaltype = "(pygame parachute) Bus Error";
        break;
#endif
#endif
#ifdef SIGFPE
    case SIGFPE:
        signaltype = "(pygame parachute) Floating Point Exception";
        break;
#endif
#ifdef SIGQUIT
    case SIGQUIT:
        signaltype = "(pygame parachute) Keyboard Abort";
        break;
#endif
    default:
        signaltype = "(pygame parachute) Unknown Signal";
        break;
    }

    _pg_quit();
    Py_FatalError(signaltype);
#endif
}


static int fatal_signals[] =
{
    SIGSEGV,
#ifdef SIGBUS
    SIGBUS,
#endif
#ifdef SIGFPE
    SIGFPE,
#endif
#ifdef SIGQUIT
    SIGQUIT,
#endif
    0 /*end of list*/
};

static int parachute_installed = 0;
static void
pg_install_parachute(void)
{
#ifdef HAVE_SIGNAL_H
    int i;
    void (*ohandler)(int);

    if (parachute_installed) {
        return;
    }
    parachute_installed = 1;

    /* Set a handler for any fatal signal not already handled */
    for (i = 0; fatal_signals[i]; ++i) {
        ohandler = (void(*)(int))signal(fatal_signals[i], pygame_parachute);
        if (ohandler != SIG_DFL) {
            signal(fatal_signals[i], ohandler);
        }
    }

#if defined(SIGALRM) && defined(HAVE_SIGACTION)
    {/* Set SIGALRM to be ignored -- necessary on Solaris */
        struct sigaction action, oaction;

        /* Set SIG_IGN action */
        memset(&action, 0, (sizeof action));
        action.sa_handler = SIG_IGN;
        sigaction(SIGALRM, &action, &oaction);
        /* Reset original action if it was already being handled */
        if (oaction.sa_handler != SIG_DFL) {
            sigaction(SIGALRM, &oaction, NULL);
        }
    }
#endif
#endif
    return;
}

static void
pg_uninstall_parachute(void)
{
#ifdef HAVE_SIGNAL_H
    int i;
    void (*ohandler)(int);

    if (!parachute_installed) {
        return;
    }
    parachute_installed = 0;

    /* Remove a handler for any fatal signal handled */
    for (i = 0; fatal_signals[i]; ++i) {
        ohandler = (void(*)(int))signal(fatal_signals[i], SIG_DFL);
        if (ohandler != pygame_parachute) {
            signal(fatal_signals[i], ohandler);
        }
    }
#endif
}

/* bind functions to python */

static PyObject*
pg_do_segfault(PyObject* self)
{
    //force crash
    *((int*)1) = 45;
    memcpy((char*)2, (char*)3, 10);
    Py_RETURN_NONE;
}

static PyMethodDef _base_methods[] =
{
    { "init", (PyCFunction) pg_init, METH_NOARGS, DOC_PYGAMEINIT },
    { "quit", (PyCFunction) pg_quit, METH_NOARGS, DOC_PYGAMEQUIT },
    { "register_quit", pg_register_quit, METH_O, DOC_PYGAMEREGISTERQUIT },
    { "get_error", (PyCFunction) pg_get_error, METH_NOARGS, DOC_PYGAMEGETERROR },
    { "set_error", (PyCFunction) pg_set_error, METH_VARARGS, DOC_PYGAMESETERROR },
    { "get_sdl_version", (PyCFunction) pg_get_sdl_version, METH_NOARGS,
      DOC_PYGAMEGETSDLVERSION },
    { "get_sdl_byteorder", (PyCFunction) pg_get_sdl_byteorder, METH_NOARGS,
      DOC_PYGAMEGETSDLBYTEORDER },

    { "get_array_interface", (PyCFunction) pg_get_array_interface, METH_O,
      "return an array struct interface as an interface dictionary" },

    { "segfault", (PyCFunction) pg_do_segfault, METH_NOARGS, "crash" },
    { NULL, NULL, 0, NULL }
};

MODINIT_DEFINE(base)
{
    static int is_loaded = 0;
    PyObject *module, *dict, *apiobj;
    PyObject *atexit, *atexit_register = NULL, *quit, *rval;
    PyObject *pgExc_SDLError;
    int ecode;
    static void* c_api[PYGAMEAPI_BASE_NUMSLOTS];

#if PY3
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "base",
        "",
        -1,
        _base_methods,
        NULL, NULL, NULL, NULL
    };
#endif

    if (!is_loaded) {
        /* import need modules. Do this first so if there is an error
           the module is not loaded.
        */
        atexit = PyImport_ImportModule("atexit");
        if (!atexit) {
            MODINIT_ERROR;
        }
        atexit_register = PyObject_GetAttrString(atexit, "register");
        Py_DECREF(atexit);
        if (!atexit_register) {
            MODINIT_ERROR;
        }
    }

    /* create the module */
#if PY3
    module = PyModule_Create(&_module);
#else
    module = Py_InitModule3(MODPREFIX "base", _base_methods, DOC_PYGAME);
#endif
    if (module == NULL) {
        MODINIT_ERROR;
    }
    dict = PyModule_GetDict(module);

    /* create the exceptions */
    pgExc_SDLError = PyErr_NewException("pygame.error", PyExc_RuntimeError,
                                         NULL);
    if (pgExc_SDLError == NULL) {
        Py_XDECREF(atexit_register);
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    ecode = PyDict_SetItemString(dict, "error", pgExc_SDLError);
    Py_DECREF(pgExc_SDLError);
    if (ecode) {
        Py_XDECREF(atexit_register);
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

#if PG_ENABLE_NEWBUF
    pgExc_BufferError = PyErr_NewException("pygame.BufferError",
                                           PyExc_BufferError, NULL);
#else
    pgExc_BufferError = PyErr_NewException("pygame.BufferError",
                                            PyExc_RuntimeError, NULL);
#endif
    if (pgExc_SDLError == NULL) {
        Py_XDECREF(atexit_register);
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    ecode = PyDict_SetItemString(dict, "BufferError", pgExc_BufferError);
    if (ecode) {
        Py_DECREF(pgExc_BufferError);
        Py_XDECREF(atexit_register);
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

    /* export the c api */
    c_api[0]  = pgExc_SDLError;
    c_api[1]  = pg_RegisterQuit;
    c_api[2]  = pg_IntFromObj;
    c_api[3]  = pg_IntFromObjIndex;
    c_api[4]  = pg_TwoIntsFromObj;
    c_api[5]  = pg_FloatFromObj;
    c_api[6]  = pg_FloatFromObjIndex;
    c_api[7]  = pg_TwoFloatsFromObj;
    c_api[8]  = pg_UintFromObj;
    c_api[9]  = pg_UintFromObjIndex;
    c_api[10] = pgVideo_AutoQuit;
    c_api[11] = pgVideo_AutoInit;
    c_api[12] = pg_RGBAFromObj;
    c_api[13] = pgBuffer_AsArrayInterface;
    c_api[14] = pgBuffer_AsArrayStruct;
    c_api[15] = pgObject_GetBuffer;
    c_api[16] = pgBuffer_Release;
    c_api[17] = pgDict_AsBuffer;
    c_api[18] = pgExc_BufferError;
#if IS_SDLv1
#define FILLED_SLOTS 19
#else /* IS_SDLv2 */
    c_api[19] = pg_GetDefaultWindow;
    c_api[20] = pg_SetDefaultWindow;
    c_api[21] = pg_GetDefaultWindowSurface;
    c_api[22] = pg_SetDefaultWindowSurface;
#define FILLED_SLOTS 23
#endif /* IS_SDLv2 */

#if PYGAMEAPI_BASE_NUMSLOTS != FILLED_SLOTS
#error export slot count mismatch
#endif

    apiobj = encapsulate_api(c_api, "base");
    if (apiobj == NULL) {
        Py_XDECREF(atexit_register);
        Py_DECREF(pgExc_BufferError);
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    ecode = PyDict_SetItemString(dict, PYGAMEAPI_LOCAL_ENTRY, apiobj);
    Py_DECREF(apiobj);
    if (ecode) {
        Py_XDECREF(atexit_register);
        Py_DECREF(pgExc_BufferError);
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

    if (PyModule_AddIntConstant(module, "HAVE_NEWBUF", PG_ENABLE_NEWBUF)) {
        Py_XDECREF(atexit_register);
        Py_DECREF(pgExc_BufferError);
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

    if (!is_loaded) {
        /*some intialization*/
        quit = PyObject_GetAttrString(module, "quit");
        if (quit == NULL) {  /* assertion */
            Py_DECREF(atexit_register);
            Py_DECREF(pgExc_BufferError);
            DECREF_MOD(module);
            MODINIT_ERROR;
        }
        rval = PyObject_CallFunctionObjArgs(atexit_register, quit, NULL);
        Py_DECREF(atexit_register);
        Py_DECREF(quit);
        if (rval == NULL) {
            DECREF_MOD(module);
            Py_DECREF(pgExc_BufferError);
            MODINIT_ERROR;
        }
        Py_DECREF(rval);
        Py_AtExit(pg_atexit_quit);
#ifdef HAVE_SIGNAL_H
        pg_install_parachute();
#endif


#ifdef MS_WIN32
        SDL_RegisterApp("pygame", 0, GetModuleHandle(NULL));
#endif
#if defined(macintosh)
#if(!defined(__MWERKS__) && !TARGET_API_MAC_CARBON)
        SDL_InitQuickDraw(&pg_qd);
#endif
#endif
        }
    is_loaded = 1;
    MODINIT_RETURN(module);
}
