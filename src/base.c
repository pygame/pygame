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
extern int SDL_RegisterApp (char*, Uint32, void*);
#endif

#if defined(macintosh)
#if(!defined(__MWERKS__) && !TARGET_API_MAC_CARBON)
QDGlobals qd;
#endif
#endif

#if SDL_BYTEORDER == SDL_LIL_ENDIAN
#define PAI_MY_ENDIAN '<'
#define PAI_OTHER_ENDIAN '>'
#else
#define PAI_MY_ENDIAN '>'
#define PAI_OTHER_ENDIAN '<'
#endif

/* Extended array struct */
typedef struct capsule_interface_s {
    PyArrayInterface inter;
    PyObject *parent;
    Py_intptr_t imem[1];
} CapsuleInterface;

/* Py_buffer internal data for an array struct */
typedef struct view_internals_s {
    PyObject* cobj;
    char format[4];      /* make 4 byte word sized */
    Py_ssize_t imem[1];
} ViewInternals;

/* Custom exceptions */
static PyObject* PgExc_BufferError = NULL;

/* Only one instance of the state per process. */
static PyObject* quitfunctions = NULL;
static int sdl_was_init = 0;

static void install_parachute (void);
static void uninstall_parachute (void);
static void _quit (void);
static void atexit_quit (void);
static int PyGame_Video_AutoInit (void);
static void PyGame_Video_AutoQuit (void);
static int GetArrayStruct (PyObject*, PyObject**, PyArrayInterface**);
static PyObject* ArrayStructAsDict (PyArrayInterface*);
static PyObject* PgBuffer_AsArrayInterface (Py_buffer*);
static PyObject* PgBuffer_AsArrayStruct (Py_buffer*);
static int _buffer_is_byteswapped (Py_buffer*);
static void PgBuffer_Release (Pg_buffer*);
static int PgObject_GetBuffer (PyObject*, Pg_buffer*, int);
static int GetArrayInterface (PyObject**, PyObject*);
static int PgDict_AsBuffer (Pg_buffer*, PyObject*, int);
static int _shape_arg_convert (PyObject *, Py_buffer*);
static int _typestr_arg_convert (PyObject *, Py_buffer*);
static int _data_arg_convert (PyObject*, Py_buffer*);
static int _strides_arg_convert (PyObject*, Py_buffer*);
static PyObject* view_get_typestr_obj (Py_buffer*);
static PyObject* view_get_shape_obj (Py_buffer*);
static PyObject* view_get_strides_obj (Py_buffer*);
static PyObject* view_get_data_obj (Py_buffer*);
static char _as_arrayinter_typekind (Py_buffer*);
static char _as_arrayinter_byteorder (Py_buffer*);
static int _as_arrayinter_flags (Py_buffer*);
static CapsuleInterface* _new_capsuleinterface (Py_buffer*);
static void _free_capsuleinterface (void*);
#if PY3
static void _capsule_free_capsuleinterface (PyObject*);
#endif
static PyObject* _shape_as_tuple (PyArrayInterface*);
static PyObject* _typekind_as_str (PyArrayInterface*);
static PyObject* _strides_as_tuple (PyArrayInterface*);
static PyObject* _data_as_tuple (PyArrayInterface*);
static PyObject* get_array_interface (PyObject*, PyObject*);
static void _release_buffer_array (Py_buffer*);
static void _release_buffer_generic (Py_buffer*);

#if PY_VERSION_HEX < 0x02060000
static int
_IsFortranContiguous(Py_buffer *view)
{
    Py_ssize_t sd, dim;
    int i;

    if (view->ndim == 0) return 1;
    if (view->strides == NULL) return (view->ndim == 1);

    sd = view->itemsize;
    if (view->ndim == 1) return (view->shape[0] == 1 ||
                               sd == view->strides[0]);
    for (i=0; i<view->ndim; i++) {
        dim = view->shape[i];
        if (dim == 0) return 1;
        if (view->strides[i] != sd) return 0;
        sd *= dim;
    }
    return 1;
}

static int
_IsCContiguous(Py_buffer *view)
{
    Py_ssize_t sd, dim;
    int i;

    if (view->ndim == 0) return 1;
    if (view->strides == NULL) return 1;

    sd = view->itemsize;
    if (view->ndim == 1) return (view->shape[0] == 1 ||
                               sd == view->strides[0]);
    for (i=view->ndim-1; i>=0; i--) {
        dim = view->shape[i];
        if (dim == 0) return 1;
        if (view->strides[i] != sd) return 0;
        sd *= dim;
    }
    return 1;
}

static int
PyBuffer_IsContiguous(Py_buffer *view, char fort)
{

    if (view->suboffsets != NULL) return 0;

    if (fort == 'C')
        return _IsCContiguous(view);
    else if (fort == 'F')
        return _IsFortranContiguous(view);
    else if (fort == 'A')
        return (_IsCContiguous(view) || _IsFortranContiguous(view));
    return 0;
}
#endif /* #if PY_VERSION_HEX < 0x02060000 */

static int
CheckSDLVersions (void) /*compare compiled to linked*/
{
    SDL_version compiled;
    const SDL_version* linked;
    SDL_VERSION (&compiled);
    linked = SDL_Linked_Version ();

    /*only check the major and minor version numbers.
      we will relax any differences in 'patch' version.*/

    if (compiled.major != linked->major || compiled.minor != linked->minor)
    {
		PyErr_Format(PyExc_RuntimeError, "SDL compiled with version %d.%d.%d, linked to %d.%d.%d",
                 compiled.major, compiled.minor, compiled.patch,
                 linked->major, linked->minor, linked->patch);
        return 0;
    }
    return 1;
}

void
PyGame_RegisterQuit (void(*func)(void))
{
    PyObject* obj;
    if (!quitfunctions)
    {
        quitfunctions = PyList_New (0);
        if (!quitfunctions)
            return;
    }
    if (func)
    {
        obj = PyCapsule_New (func, "quit", NULL);
        PyList_Append (quitfunctions, obj);
        Py_DECREF (obj);
    }
}

static PyObject*
register_quit (PyObject* self, PyObject* value)
{
    if (!quitfunctions)
    {
        quitfunctions = PyList_New (0);
        if (!quitfunctions)
            return NULL;
    }
    PyList_Append (quitfunctions, value);

    Py_RETURN_NONE;
}

static PyObject*
init (PyObject* self)
{
    PyObject *allmodules, *moduleslist, *dict, *func, *result, *mod;
    int loop, num;
    int success=0, fail=0;

    if (!CheckSDLVersions ())
        return NULL;


    /*nice to initialize timer, so startup time will reflec init() time*/
    sdl_was_init = SDL_Init (
#if defined(WITH_THREAD) && !defined(MS_WIN32) && defined(SDL_INIT_EVENTTHREAD)
        SDL_INIT_EVENTTHREAD |
#endif
        SDL_INIT_TIMER |
        SDL_INIT_NOPARACHUTE) == 0;


    /* initialize all pygame modules */
    allmodules = PyImport_GetModuleDict ();
    moduleslist = PyDict_Values (allmodules);
    if (!allmodules || !moduleslist)
        return Py_BuildValue ("(ii)", 0, 0);

    if (PyGame_Video_AutoInit ())
        ++success;
    else
        ++fail;

    num = PyList_Size (moduleslist);
    for (loop = 0; loop < num; ++loop)
    {
        mod = PyList_GET_ITEM (moduleslist, loop);
        if (!mod || !PyModule_Check (mod))
            continue;
        dict = PyModule_GetDict (mod);
        func = PyDict_GetItemString (dict, "__PYGAMEinit__");
        if(func && PyCallable_Check (func))
        {
            result = PyObject_CallObject (func, NULL);
            if (result && PyObject_IsTrue (result))
                ++success;
            else
            {
                PyErr_Clear ();
                ++fail;
            }
            Py_XDECREF (result);
        }
    }
    Py_DECREF (moduleslist);

    return Py_BuildValue ("(ii)", success, fail);
}

static void
atexit_quit (void)
{
    PyGame_Video_AutoQuit ();

    /* Maybe it is safe to call SDL_quit more than once after an SDL_Init,
       but this is undocumented. So play it safe and only call after a
       successful SDL_Init.
    */
    if (sdl_was_init) {
        sdl_was_init = 0;
        SDL_Quit ();
    }
}

static PyObject*
get_sdl_version (PyObject* self)
{
    const SDL_version *v;
	
    v = SDL_Linked_Version ();
    return Py_BuildValue ("iii", v->major, v->minor, v->patch);
}

static PyObject*
get_sdl_byteorder (PyObject *self)
{
    return PyLong_FromLong (SDL_BYTEORDER);
}

static PyObject*
quit (PyObject* self)
{
    _quit ();
    Py_RETURN_NONE;
}

static void
_quit (void)
{
    PyObject* quit;
    PyObject* privatefuncs;
    int num;

    if (!quitfunctions) {
        return;
    }

    privatefuncs = quitfunctions;
    quitfunctions = NULL;

    uninstall_parachute ();
    num = PyList_Size (privatefuncs);

    while (num--) /*quit in reverse order*/
    {
        quit = PyList_GET_ITEM (privatefuncs, num);
        if (PyCallable_Check (quit))
            PyObject_CallObject (quit, NULL);
        else if (PyCapsule_CheckExact (quit))
        {
            void* ptr = PyCapsule_GetPointer (quit, "quit");
            (*(void(*)(void)) ptr) ();
        }
    }
    Py_DECREF (privatefuncs);

    atexit_quit ();
}

/* internal C API utility functions */
static int
IntFromObj (PyObject* obj, int* val) {
    int tmp_val;
    tmp_val = PyInt_AsLong (obj);
    if (tmp_val == -1 && PyErr_Occurred ())
    {
        PyErr_Clear ();
        return 0;
    }
    *val = tmp_val;
    return 1;
}

static int
IntFromObjIndex (PyObject* obj, int _index, int* val)
{
    int result = 0;
    PyObject* item;
    item = PySequence_GetItem (obj, _index);
    if (item)
    {
        result = IntFromObj (item, val);
        Py_DECREF (item);
    }
    return result;
}

static int
TwoIntsFromObj (PyObject* obj, int* val1, int* val2)
{
    if (PyTuple_Check (obj) && PyTuple_Size (obj) == 1)
        return TwoIntsFromObj (PyTuple_GET_ITEM (obj, 0), val1, val2);

    if (!PySequence_Check (obj) || PySequence_Length (obj) != 2)
        return 0;

    if (!IntFromObjIndex (obj, 0, val1) || !IntFromObjIndex (obj, 1, val2))
        return 0;

    return 1;
}

static int
FloatFromObj (PyObject* obj, float* val)
{
    float f= (float)PyFloat_AsDouble (obj);

    if (f==-1 && PyErr_Occurred()) {
		PyErr_Clear ();
        return 0;
	}
    
    *val = f;
    return 1;
}

static int
FloatFromObjIndex (PyObject* obj, int _index, float* val)
{
    int result = 0;
    PyObject* item;
    item = PySequence_GetItem (obj, _index);
    if (item)
    {
        result = FloatFromObj (item, val);
        Py_DECREF (item);
    }
    return result;
}

static int
TwoFloatsFromObj (PyObject* obj, float* val1, float* val2)
{
    if (PyTuple_Check (obj) && PyTuple_Size (obj) == 1)
        return TwoFloatsFromObj (PyTuple_GET_ITEM (obj, 0), val1, val2);

    if (!PySequence_Check (obj) || PySequence_Length (obj) != 2)
        return 0;

    if (!FloatFromObjIndex (obj, 0, val1) || !FloatFromObjIndex (obj, 1, val2))
        return 0;

    return 1;
}

static int
UintFromObj (PyObject* obj, Uint32* val)
{
    PyObject* longobj;

    if (PyNumber_Check (obj))
    {
        if (!(longobj = PyNumber_Long (obj)))
            return 0;
        *val = (Uint32) PyLong_AsUnsignedLong (longobj);
        Py_DECREF (longobj);
        return 1;
    }
    return 0;
}

static int
UintFromObjIndex (PyObject* obj, int _index, Uint32* val)
{
    int result = 0;
    PyObject* item;
    item = PySequence_GetItem (obj, _index);
    if (item)
    {
        result = UintFromObj (item, val);
        Py_DECREF (item);
    }
    return result;
}

static int
RGBAFromObj (PyObject* obj, Uint8* RGBA)
{
    int length;
    Uint32 val;
    if (PyTuple_Check (obj) && PyTuple_Size (obj) == 1)
        return RGBAFromObj (PyTuple_GET_ITEM (obj, 0), RGBA);

    if (!PySequence_Check (obj))
        return 0;

    length = PySequence_Length (obj);
    if (length < 3 || length > 4)
        return 0;

    if (!UintFromObjIndex (obj, 0, &val) || val > 255)
        return 0;
    RGBA[0] = (Uint8) val;
    if (!UintFromObjIndex (obj, 1, &val) || val > 255)
        return 0;
    RGBA[1] = (Uint8) val;
    if (!UintFromObjIndex (obj, 2, &val) || val > 255)
        return 0;
    RGBA[2] = (Uint8) val;
    if (length == 4)
    {
        if (!UintFromObjIndex (obj, 3, &val) || val > 255)
            return 0;
        RGBA[3] = (Uint8) val;
    }
    else RGBA[3] = (Uint8) 255;

    return 1;
}

static PyObject*
get_error (PyObject* self)
{
    return Text_FromUTF8 (SDL_GetError ());
}

static PyObject*
set_error (PyObject *s, PyObject *args)
{
    char *errstring = NULL;

    if (!PyArg_ParseTuple (args, "s", &errstring))
        return NULL;

    SDL_SetError(errstring);

    Py_RETURN_NONE;
}




/*video init needs to be here, because of it's
 *important init order priority
 */
static void
PyGame_Video_AutoQuit (void)
{
    if (SDL_WasInit (SDL_INIT_VIDEO))
        SDL_QuitSubSystem (SDL_INIT_VIDEO);
}

static int
PyGame_Video_AutoInit (void)
{
    if (!SDL_WasInit (SDL_INIT_VIDEO))
    {
        int status;
#if defined(__APPLE__) && defined(darwin)
        PyObject *module;
        PyObject *rval;
        module = PyImport_ImportModule ("pygame.macosx");
        if (!module)
        {
        	printf("ERROR: pygame.macosx import FAILED\n");
            return -1;
        }

        rval = PyObject_CallMethod (module, "Video_AutoInit", "");
        Py_DECREF (module);
        if (!rval)
        {
        	printf("ERROR: pygame.macosx.Video_AutoInit() call FAILED\n");
            return -1;
        }

        status = PyObject_IsTrue (rval);
        Py_DECREF (rval);
        if (status != 1)
            return 0;
#endif
        status = SDL_InitSubSystem (SDL_INIT_VIDEO);
        if (status)
            return 0;
        SDL_EnableUNICODE (1);
        /*we special case the video quit to last now*/
        /*PyGame_RegisterQuit(PyGame_Video_AutoQuit);*/
    }
    return 1;
}

/*array interface*/

static int
GetArrayStruct (PyObject* obj,
                   PyObject** cobj_p,
                   PyArrayInterface** inter_p)
{
    PyObject* cobj = PyObject_GetAttrString (obj, "__array_struct__");
    PyArrayInterface* inter = NULL;

    if (cobj == NULL) {
        if (PyErr_ExceptionMatches (PyExc_AttributeError)) {
                PyErr_Clear ();
                PyErr_SetString (PyExc_ValueError,
                                 "no C-struct array interface");
        }
        return -1;
    }

#if PG_HAVE_COBJECT
    if (PyCObject_Check (cobj)) {
        inter = (PyArrayInterface *)PyCObject_AsVoidPtr (cobj);
    }
#endif
#if PG_HAVE_CAPSULE
    if (PyCapsule_IsValid (cobj, NULL)) {
        inter = (PyArrayInterface*)PyCapsule_GetPointer (cobj, NULL);
    }
#endif
    if (inter == NULL || inter->two != 2 /* conditional or */) {
        Py_DECREF (cobj);
        PyErr_SetString (PyExc_ValueError, "invalid array interface");
        return -1;
    }

    *cobj_p = cobj;
    *inter_p = inter;
    return 0;
}

static PyObject*
ArrayStructAsDict (PyArrayInterface* inter_p)
{
    PyObject *dictobj = Py_BuildValue ("{sisNsNsNsN}",
                                       "version", (int)3,
                                       "typestr", _typekind_as_str (inter_p),
                                       "shape", _shape_as_tuple (inter_p),
                                       "strides", _strides_as_tuple (inter_p),
                                       "data", _data_as_tuple (inter_p));

    if (!dictobj) {
        return 0;
    }
    if (inter_p->flags & PAI_ARR_HAS_DESCR) {
        if (!inter_p->descr) {
            Py_DECREF (dictobj);
            PyErr_SetString (PyExc_ValueError,
                             "Array struct has descr flag set"
                             " but no descriptor");
            return 0;
        }
        if (PyDict_SetItemString (dictobj, "descr", inter_p->descr)) {
            Py_DECREF (dictobj);
            return 0;
        }
    }
    return dictobj;
}

static PyObject*
PgBuffer_AsArrayInterface (Py_buffer* view_p)
{
    return Py_BuildValue ("{sisNsNsNsN}",
                          "version", (int)3,
                          "typestr", view_get_typestr_obj (view_p),
                          "shape", view_get_shape_obj (view_p),
                          "strides", view_get_strides_obj (view_p),
                          "data", view_get_data_obj (view_p));
}

static PyObject*
PgBuffer_AsArrayStruct (Py_buffer* view_p)
{
    void *cinter_p  = _new_capsuleinterface (view_p);
    PyObject *capsule;

    if (!cinter_p) {
        return 0;
    }
#if PY3
    capsule = PyCapsule_New (cinter_p, 0, _capsule_free_capsuleinterface);
#else
    capsule = PyCObject_FromVoidPtr (cinter_p, _free_capsuleinterface);
#endif
    if (!capsule) {
        _free_capsuleinterface ((void*)cinter_p);
        return 0;
    }
    return capsule;
}

static CapsuleInterface*
_new_capsuleinterface (Py_buffer *view_p)
{
    int ndim = view_p->ndim;
    Py_ssize_t cinter_size;
    CapsuleInterface *cinter_p;
    int i;

    cinter_size = (sizeof (CapsuleInterface) +
                   sizeof (Py_intptr_t) * (2 * ndim - 1));
    cinter_p = (CapsuleInterface *)PyMem_Malloc (cinter_size);
    if (!cinter_p) {
        PyErr_NoMemory ();
        return 0;
    }
    cinter_p->inter.two = 2;
    cinter_p->inter.nd = ndim;
    cinter_p->inter.typekind = _as_arrayinter_typekind (view_p);
    cinter_p->inter.itemsize = view_p->itemsize;
    cinter_p->inter.flags = _as_arrayinter_flags (view_p);
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
    cinter_p->parent = view_p->obj;
    Py_XINCREF (cinter_p->parent);
    return cinter_p;
}

static void
_free_capsuleinterface (void *p)
{
    CapsuleInterface *cinter_p = (CapsuleInterface *)p;

    Py_XDECREF (cinter_p->parent);
    PyMem_Free (p);
}

#if PY3
static void
_capsule_free_capsuleinterface (PyObject *capsule)
{
    _free_capsuleinterface (PyCapsule_GetPointer (capsule, 0));
}
#endif

static int
_as_arrayinter_flags (Py_buffer* view_p)
{
    int inter_flags = PAI_ALIGNED; /* atomic int types always aligned */

    if (!view_p->readonly) {
        inter_flags |= PAI_WRITEABLE;
    }
    inter_flags |= _buffer_is_byteswapped (view_p) ? 0 : PAI_NOTSWAPPED;
    if (PyBuffer_IsContiguous (view_p, 'C')) {
        inter_flags |= PAI_CONTIGUOUS;
    }
    if (PyBuffer_IsContiguous (view_p, 'F')) {
        inter_flags |= PAI_FORTRAN;
    }
    return inter_flags;
}

static PyObject*
view_get_typestr_obj (Py_buffer* view)
{
    return Text_FromFormat ("%c%c%i",
                            _as_arrayinter_byteorder (view),
                            _as_arrayinter_typekind (view),
                            (int)view->itemsize);
}

static PyObject*
view_get_shape_obj (Py_buffer* view)
{
    PyObject *shapeobj = PyTuple_New (view->ndim);
    PyObject *lengthobj;
    Py_ssize_t i;

    if (!shapeobj) {
        return 0;
    }
    for (i = 0; i < view->ndim; ++i) {
        lengthobj = PyInt_FromLong ((long)view->shape[i]);
        if (!lengthobj) {
            Py_DECREF (shapeobj);
            return 0;
        }
        PyTuple_SET_ITEM (shapeobj, i, lengthobj);
    }
    return shapeobj;
}

static PyObject*
view_get_strides_obj (Py_buffer* view)
{
    PyObject *shapeobj = PyTuple_New (view->ndim);
    PyObject *lengthobj;
    Py_ssize_t i;

    if (!shapeobj) {
        return 0;
    }
    for (i = 0; i < view->ndim; ++i) {
        lengthobj = PyInt_FromLong ((long)view->strides[i]);
        if (!lengthobj) {
            Py_DECREF (shapeobj);
            return 0;
        }
        PyTuple_SET_ITEM (shapeobj, i, lengthobj);
    }
    return shapeobj;
}

static PyObject*
view_get_data_obj (Py_buffer* view)
{
    return Py_BuildValue ("NN",
                          PyLong_FromVoidPtr (view->buf),
                          PyBool_FromLong ((long)view->readonly));
}

static char
_as_arrayinter_typekind (Py_buffer* view)
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
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
        typekind = (view->format[1] == 'x') ? 'u' : 'V';
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
_as_arrayinter_byteorder (Py_buffer* view)
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
_shape_as_tuple (PyArrayInterface* inter_p)
{
    PyObject *shapeobj = PyTuple_New ((Py_ssize_t)inter_p->nd);
    PyObject *lengthobj;
    Py_ssize_t i;

    if (!shapeobj) {
        return 0;
    }
    for (i = 0; i < inter_p->nd; ++i) {
        lengthobj = PyInt_FromLong ((long)inter_p->shape[i]);
        if (!lengthobj) {
            Py_DECREF (shapeobj);
            return 0;
        }
        PyTuple_SET_ITEM (shapeobj, i, lengthobj);
    }
    return shapeobj;
}

static PyObject*
_typekind_as_str (PyArrayInterface* inter_p)
{
    return Text_FromFormat ("%c%c%i",
                            inter_p->itemsize > 1 ?
                                (inter_p->flags & PAI_NOTSWAPPED ?
                                     PAI_MY_ENDIAN :
                                     PAI_OTHER_ENDIAN) :
                                '|',
                            inter_p->typekind, inter_p->itemsize);
}

static PyObject*
_strides_as_tuple (PyArrayInterface* inter_p)
{
    PyObject *stridesobj = PyTuple_New ((Py_ssize_t)inter_p->nd);
    PyObject *lengthobj;
    Py_ssize_t i;

    if (!stridesobj) {
        return 0;
    }
    for (i = 0; i < inter_p->nd; ++i) {
        lengthobj = PyInt_FromLong ((long)inter_p->strides[i]);
        if (!lengthobj) {
            Py_DECREF (stridesobj);
            return 0;
        }
        PyTuple_SET_ITEM (stridesobj, i, lengthobj);
    }
    return stridesobj;
}

static PyObject*
_data_as_tuple (PyArrayInterface* inter_p)
{
    long readonly = (inter_p->flags & PAI_WRITEABLE) == 0;

    return Py_BuildValue ("NN",
                          PyLong_FromVoidPtr (inter_p->data),
                          PyBool_FromLong (readonly));
}

static PyObject*
get_array_interface (PyObject* self, PyObject* arg)
{
    PyObject *cobj;
    PyArrayInterface *inter_p;
    PyObject *dictobj;

    if (GetArrayStruct (arg, &cobj, &inter_p)) {
        return 0;
    }
    dictobj = ArrayStructAsDict (inter_p);
    Py_DECREF (cobj);
    return dictobj;
}

static int
PgObject_GetBuffer (PyObject* obj, Pg_buffer* pg_view_p, int flags)
{
    Py_buffer* view_p = (Py_buffer*)pg_view_p;
    PyObject* cobj = 0;
    PyObject* dict = 0;
    PyArrayInterface* inter_p = 0;
    ViewInternals* internal_p;
    size_t sz;
    char *fchar_p;
    Py_ssize_t i;
    int success = 0;

    pg_view_p->release_buffer = _release_buffer_generic;
    view_p->len = 0;

#if PG_ENABLE_NEWBUF

    if (PyObject_CheckBuffer (obj)) {
        if (PyObject_GetBuffer (obj, view_p, flags)) {
            return -1;
        }
        pg_view_p->release_buffer = PyBuffer_Release;
        fchar_p = view_p->format;
        switch (*fchar_p) {

        case '@':
        case '=':
        case '<':
        case '>':
        case '!':
            ++fchar_p;
            break;
        default:
            break;
        }
        if (*fchar_p == 1) {
            ++fchar_p;
        }
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
            ++fchar_p;
            break;
        case '0':
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
        case '8':
        case '9':
            ++fchar_p;
            if (*fchar_p == 'x') {
                ++fchar_p;
            }
            break;
        default:
            PgBuffer_Release (pg_view_p);
            PyErr_SetString (PyExc_ValueError,
                             "Unsupported array element type");
            return -1;
        }
        if (*fchar_p != '\0') {
            PgBuffer_Release (pg_view_p);
            PyErr_SetString (PyExc_ValueError,
                             "Arrays of records are unsupported");
            return -1;
        }
        success = 1;
    }

#endif
    if (!success && GetArrayStruct (obj, &cobj, &inter_p) == 0) {
        sz = (sizeof (ViewInternals) + 
              (2 * inter_p->nd - 1) * sizeof (Py_ssize_t));
        internal_p = (ViewInternals*)PyMem_Malloc (sz);
        if (!internal_p) {
            Py_DECREF (cobj);
            PyErr_NoMemory ();
            return -1;
        }
        fchar_p = internal_p->format;
        switch (inter_p->typekind) {

        case 'i':
            *fchar_p = (inter_p->flags & PAI_NOTSWAPPED ?
                        PAI_MY_ENDIAN : PAI_OTHER_ENDIAN);
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
                PyErr_Format (PyExc_ValueError,
                              "Unsupported signed interger size %d",
                              (int)inter_p->itemsize);
                Py_DECREF (cobj);
                return -1;
            }
            break;
        case 'u':
            *fchar_p = (inter_p->flags & PAI_NOTSWAPPED ?
                        PAI_MY_ENDIAN : PAI_OTHER_ENDIAN);
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
                PyErr_Format (PyExc_ValueError,
                              "Unsupported unsigned interger size %d",
                              (int)inter_p->itemsize);
                Py_DECREF (cobj);
                return -1;
            }
            break;
        case 'f':
            *fchar_p = (inter_p->flags & PAI_NOTSWAPPED ?
                        PAI_MY_ENDIAN : PAI_OTHER_ENDIAN);
            ++fchar_p;
            switch (inter_p->itemsize) {

            case 4:
                *fchar_p = 'f';
                break;
            case 8:
                *fchar_p = 'd';
                break;
            default:
                PyErr_Format (PyExc_ValueError,
                              "Unsupported float size %d",
                              (int)inter_p->itemsize);
                Py_DECREF (cobj);
                return -1;
            }
            break;
        case 'V':
            if (inter_p->itemsize > 9) {
                PyErr_Format (PyExc_ValueError,
                              "Unsupported void size %d",
                              (int)inter_p->itemsize);
                Py_DECREF (cobj);
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
                PyErr_Format (PyExc_ValueError,
                              "Unsupported void size %d",
                              (int)inter_p->itemsize);
                Py_DECREF (cobj);
                return -1;
            }
            ++fchar_p;
            *fchar_p = 'x';
            break;
        default:
            PyErr_Format (PyExc_ValueError,
                          "Unsupported value type '%c'",
                          (int)inter_p->typekind);
            Py_DECREF (cobj);
            return -1;
        }
        ++fchar_p;
        *fchar_p = '\0';
        view_p->internal = internal_p;
        view_p->format = internal_p->format;
        view_p->shape = internal_p->imem;
        view_p->strides = view_p->shape + inter_p->nd;
        internal_p->cobj = cobj;
        view_p->buf = inter_p->data;
        Py_INCREF (obj);
        view_p->obj = obj;
        view_p->ndim = (Py_ssize_t)inter_p->nd;
        view_p->itemsize = (Py_ssize_t)inter_p->itemsize;
        view_p->readonly = inter_p->flags & PAI_WRITEABLE ? 0 : 1;
        for (i = 0; i < view_p->ndim; ++i) {
            view_p->shape[i] = (Py_ssize_t)inter_p->shape[i];
            view_p->strides[i] = (Py_ssize_t)inter_p->strides[i];
        }
        view_p->suboffsets = 0;
        view_p->len = view_p->itemsize;
        for (i = 0; i < view_p->ndim; ++i) {
            view_p->len *= view_p->shape[i];
        }
        pg_view_p->release_buffer = _release_buffer_array;
        success = 1;
    }
    else if (!success) {
        PyErr_Clear ();
    }

    if (!success && GetArrayInterface (&dict, obj) == 0) {
        if (PgDict_AsBuffer (pg_view_p, dict, flags)) {
            return -1;
        }
        Py_INCREF (obj);
        view_p->obj = obj;
        success = 1;
    }
    else if (!success) {
        PyErr_Clear ();
    }

    if (!success) {
        PyErr_Format (PyExc_TypeError,
                      "%s object does not export an array buffer",
                      Py_TYPE (obj)->tp_name);
        return -1;
    }

    if (!view_p->len) {
        view_p->len = view_p->itemsize;
        for (i = 0; i < view_p->ndim; ++i) {
            view_p->len *= view_p->shape[i];
        }
    }
    return 0;
}

static void
PgBuffer_Release (Pg_buffer* pg_view_p)
{
    assert(pg_view_p && pg_view_p->release_buffer);
    pg_view_p->release_buffer ((Py_buffer*)pg_view_p);
}

static void
_release_buffer_generic (Py_buffer* view_p)
{
    if (view_p->obj) {
        Py_XDECREF (view_p->obj);
        view_p->obj = 0;
    }
}

static void
_release_buffer_array (Py_buffer* view_p)
{
    /* This is deliberately made safe for use on an unitialized *view_p */
    if (view_p->internal) {
        Py_XDECREF (((ViewInternals*)view_p->internal)->cobj);
        PyMem_Free (view_p->internal);
        view_p->internal = 0;
    }
    if (view_p->obj) {
        Py_DECREF (view_p->obj);
        view_p->obj = 0;
    }
}

static int
_buffer_is_byteswapped (Py_buffer* view)
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
GetArrayInterface (PyObject **dict, PyObject *obj)
{
    PyObject* inter = PyObject_GetAttrString (obj, "__array_interface__");

    if (inter == NULL) {
        if (PyErr_ExceptionMatches (PyExc_AttributeError)) {
                PyErr_Clear ();
                PyErr_SetString (PyExc_ValueError, "no array interface");
        }
        return -1;
    }
    if (!PyDict_Check (inter)) {
        PyErr_Format (PyExc_ValueError,
                      "expected __array_interface__ to return a dict: got a %s",
                      Py_TYPE (dict)->tp_name);
        Py_DECREF (inter);
        return -1;
    }
    *dict = inter;
    return 0;
}

static int
PgDict_AsBuffer (Pg_buffer* pg_view_p, PyObject* dict, int flags)
{
    Py_buffer* view_p = (Py_buffer*)pg_view_p;
    PyObject* pyshape = PyDict_GetItemString (dict, "shape");
    PyObject* pytypestr = PyDict_GetItemString (dict, "typestr");
    PyObject* pydata = PyDict_GetItemString (dict, "data");
    PyObject* pystrides = PyDict_GetItemString (dict, "strides");
    int i;

#warning I really do not know what to do with the flags argument.
    if (!pyshape) {
        PyErr_SetString (PyExc_ValueError,
                         "required \"shape\" item is missing");
        return -1;
    }
    if (!pytypestr) {
        PyErr_SetString (PyExc_ValueError,
                         "required \"typestr\" item is missing");
        return -1;
    }
    if (!pydata) {
        PyErr_SetString (PyExc_ValueError,
                         "required \"data\" item is missing");
        return -1;
    }
    /* The item processing order is important:
       "strides" and "typestr" must follow "shape". */
    view_p->internal = 0;
    view_p->obj = 0;
    pg_view_p->release_buffer = _release_buffer_array;
    if (_shape_arg_convert (pyshape, view_p)) {
        PgBuffer_Release (pg_view_p);
        return -1;
    }
    if (_typestr_arg_convert (pytypestr, view_p)) {
        PgBuffer_Release (pg_view_p);
        return -1;
    }
    if (_data_arg_convert (pydata, view_p)) {
        PgBuffer_Release (pg_view_p);
        return -1;
    }
    if (_strides_arg_convert (pystrides, view_p)) {
        PgBuffer_Release (pg_view_p);
        return -1;
    }
    view_p->len = view_p->itemsize;
    for (i = 0; i < view_p->ndim; ++i) {
        view_p->len *= view_p->shape[i];
    }
    return 0;
}

static int
_shape_arg_convert (PyObject* o, Py_buffer* view)
{
    /* Convert o as a C array of integers and return 1, otherwise
     * raise a Python exception and return 0.
     */
    ViewInternals* internal_p;
    Py_ssize_t i, n;
    Py_ssize_t* a;
    size_t sz;

    view->obj = 0;
    view->internal = 0;
    if (!PyTuple_Check (o)) {
        PyErr_Format (PyExc_TypeError,
                      "Expected a tuple for shape: found %s",
                      Py_TYPE (o)->tp_name);
        return -1;
    }
    n = PyTuple_GET_SIZE (o);
    sz = sizeof (ViewInternals) + (2 * n - 1) * sizeof (Py_ssize_t);
    internal_p = (ViewInternals*)PyMem_Malloc (sz);
    if (!internal_p) {
        PyErr_NoMemory ();
        return -1;
    }
    internal_p->cobj = 0;
    a = internal_p->imem;
    for (i = 0; i < n; ++i) {
        a[i] = PyInt_AsSsize_t (PyTuple_GET_ITEM (o, i));
        if (a[i] == -1 && PyErr_Occurred () /* conditional && */) {
            PyMem_Free (internal_p);
            PyErr_Format (PyExc_TypeError,
                          "shape tuple has a non-integer at position %d",
                          (int)i);
            return -1;
        }
    }
    view->ndim = n;
    view->internal = internal_p;
    view->shape = a;
    view->strides = a + n;
    view->format = internal_p->format;
    return 0;
}

static int
_typestr_arg_convert (PyObject* o, Py_buffer* view)
{
    /* Due to incompatibilities between the array and new buffer interfaces,
     * as well as significant unicode changes in Python 3.3, this will
     * only handle integer types. Must call _shape_arg_convert first.
     */
    char *fchar_p;
    int is_swapped;
    int itemsize = 0;
    PyObject* s;
    const char* typestr;

    if (PyUnicode_Check (o)) {
        s = PyUnicode_AsASCIIString (o);
        if (!s) {
            return -1;
        }
    }
    else {
        Py_INCREF (o);
        s = o;
    }
    if (!Bytes_Check (s)) {
        PyErr_Format( PyExc_TypeError, "Expected a string for typestr: got %s",
                     Py_TYPE (s)->tp_name);
        Py_DECREF (s);
        return -1;
    }
    if (Bytes_GET_SIZE (s) != 3) {
        PyErr_SetString (PyExc_TypeError, "Expected typestr to be length 3");
        Py_DECREF (s);
        return -1;
    }
    typestr = Bytes_AsString (s);
    fchar_p = view->format;
    switch (typestr[0]) {

    case PAI_MY_ENDIAN:
        is_swapped = 0;
        break;
    case PAI_OTHER_ENDIAN:
        is_swapped = 1;
        break;
    case '|':
        is_swapped = 0;
        break;
    default:
        PyErr_Format (PyExc_ValueError, "unsupported typestr %s", typestr);
        Py_DECREF (s);
        return -1;
    }
    switch (typestr[1]) {

    case 'i':
    case 'u':
        switch (typestr[2]) {

        case '1':
            *fchar_p = '=';
            ++fchar_p;
            *fchar_p = 'B';
            itemsize = 1;
            break;
        case '2':
            *fchar_p = is_swapped ? PAI_OTHER_ENDIAN : '=';
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
            *fchar_p = is_swapped ? PAI_OTHER_ENDIAN : '=';
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
            *fchar_p = is_swapped ? PAI_OTHER_ENDIAN : '=';
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
            PyErr_Format (PyExc_ValueError, "unsupported typestr %s", typestr);
            Py_DECREF (s);
            return -1;
        }
        if (typestr[1] == 'i') {
            /* This leaves 'x' uneffected. */
            *fchar_p = tolower(*fchar_p);
        }
        break;
    case 'f':
        *fchar_p = is_swapped ? PAI_OTHER_ENDIAN : '=';
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
            PyErr_Format (PyExc_ValueError, "unsupported typestr %s", typestr);
            Py_DECREF (s);
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
            PyErr_Format (PyExc_ValueError, "unsupported typestr %s", typestr);
            Py_DECREF (s);
            return -1;
        }
        ++fchar_p;
        *fchar_p = 'x';
        break;
    default:
        PyErr_Format (PyExc_ValueError, "unsupported typestr %s", typestr);
        Py_DECREF (s);
        return -1;
    }
    ++fchar_p;
    *fchar_p = '\0';
    view->itemsize = itemsize;
    return 0;
}

static int
_strides_arg_convert (PyObject* o, Py_buffer* view)
{
    /* Must called _shape_arg_convert first.
     */
    int n;
    Py_ssize_t* a;
    size_t i;

    if (o == Py_None) {
        /* no strides (optional) given */
        view->strides = 0;
        return 0;
    }
    a = view->strides;
    
    if (!PyTuple_Check (o)) {
        PyErr_Format (PyExc_TypeError,
                      "Expected a tuple for strides: found %s",
                      Py_TYPE (o)->tp_name);
        return -1;
    }
    n = PyTuple_GET_SIZE (o);
    if (n != view->ndim) {
        PyErr_SetString (PyExc_TypeError,
                         "Missmatch in strides and shape length");
        return -1;
    }
    for (i = 0; i < n; ++i) {
        a[i] = PyInt_AsSsize_t (PyTuple_GET_ITEM (o, i));
        if (a[i] == -1 && PyErr_Occurred () /* conditional && */) {
            PyErr_Format (PyExc_TypeError,
                          "strides tuple has a non-integer at position %d",
                          (int)i);
            return -1;
        }
    }
    if (n != view->ndim) {
        PyErr_SetString (PyExc_TypeError,
                         "strides and shape tuple lengths differ");
        return -1;
    }
    return 0;
}

static int
_data_arg_convert(PyObject *o, Py_buffer *view)
{
    void* address;
    int readonly;

    if (!PyTuple_Check (o)) {
        PyErr_Format (PyExc_TypeError, "expected a tuple for data: got %s",
                      Py_TYPE (o)->tp_name);
        return -1;
    }
    if (PyTuple_GET_SIZE (o) != 2) {
        PyErr_SetString (PyExc_TypeError, "expected a length 2 tuple for data");
        return -1;
    }
    address = PyLong_AsVoidPtr (PyTuple_GET_ITEM (o, 0));
    if (!address && PyErr_Occurred ()) {
        PyErr_Clear ();
        PyErr_Format (PyExc_TypeError,
                      "expected an integer address for data item 0: got %s",
                      Py_TYPE (PyTuple_GET_ITEM (o, 0))->tp_name);
        return -1;
    }
    readonly = PyObject_IsTrue (PyTuple_GET_ITEM (o, 1));
    if (readonly == -1) {
        PyErr_Clear ();
        PyErr_Format (PyExc_TypeError,
                      "expected a boolean flag for data item 1: got %s",
                      Py_TYPE (PyTuple_GET_ITEM (o, 0))->tp_name);
        return -1;
    }
    view->buf = address;
    view->readonly = readonly;
    return 0;
}

/*error signal handlers (replacing SDL parachute)*/
static void
pygame_parachute (int sig)
{
#ifdef HAVE_SIGNAL_H
    char* signaltype;
    
    signal (sig, SIG_DFL);
    switch (sig)
    {
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

    _quit ();
    Py_FatalError (signaltype);
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
install_parachute (void)
{
#ifdef HAVE_SIGNAL_H
    int i;
    void (*ohandler)(int);

    if (parachute_installed)
        return;
    parachute_installed = 1;

    /* Set a handler for any fatal signal not already handled */
    for (i = 0; fatal_signals[i]; ++i)
    {
        ohandler = (void(*)(int))signal (fatal_signals[i], pygame_parachute);
        if (ohandler != SIG_DFL)
            signal (fatal_signals[i], ohandler);
    }
    
#if defined(SIGALRM) && defined(HAVE_SIGACTION) 
    {/* Set SIGALRM to be ignored -- necessary on Solaris */
        struct sigaction action, oaction;
        /* Set SIG_IGN action */
        memset (&action, 0, (sizeof action));
        action.sa_handler = SIG_IGN;
        sigaction (SIGALRM, &action, &oaction);
        /* Reset original action if it was already being handled */
        if (oaction.sa_handler != SIG_DFL)
            sigaction (SIGALRM, &oaction, NULL);
    }
#endif
#endif    
    return;
}

static void
uninstall_parachute (void)
{
#ifdef HAVE_SIGNAL_H
    int i;
    void (*ohandler)(int);

    if (!parachute_installed)
        return;
    parachute_installed = 0;

    /* Remove a handler for any fatal signal handled */
    for (i = 0; fatal_signals[i]; ++i)
    {
        ohandler = (void(*)(int))signal (fatal_signals[i], SIG_DFL);
        if (ohandler != pygame_parachute)
            signal (fatal_signals[i], ohandler);
    }
#endif    
}

/* bind functions to python */

static PyObject*
do_segfault (PyObject* self)
{
    //force crash
    *((int*)1) = 45;
    memcpy ((char*)2, (char*)3, 10);
    Py_RETURN_NONE;
}

static PyMethodDef _base_methods[] =
{
    { "init", (PyCFunction) init, METH_NOARGS, DOC_PYGAMEINIT },
    { "quit", (PyCFunction) quit, METH_NOARGS, DOC_PYGAMEQUIT },
    { "register_quit", register_quit, METH_O, DOC_PYGAMEREGISTERQUIT },
    { "get_error", (PyCFunction) get_error, METH_NOARGS, DOC_PYGAMEGETERROR },
    { "set_error", (PyCFunction) set_error, METH_VARARGS, DOC_PYGAMESETERROR },
    { "get_sdl_version", (PyCFunction) get_sdl_version, METH_NOARGS,
      DOC_PYGAMEGETSDLVERSION },
    { "get_sdl_byteorder", (PyCFunction) get_sdl_byteorder, METH_NOARGS,
      DOC_PYGAMEGETSDLBYTEORDER },

    { "get_array_interface", (PyCFunction) get_array_interface, METH_O,
      "return an array struct interface as an interface dictionary" },

    { "segfault", (PyCFunction) do_segfault, METH_NOARGS, "crash" },
    { NULL, NULL, 0, NULL }
};

MODINIT_DEFINE(base)
{
    static int is_loaded = 0;
    PyObject *module, *dict, *apiobj;
    PyObject *atexit, *atexit_register = NULL, *quit, *rval;
    PyObject *PyExc_SDLError;
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
        atexit = PyImport_ImportModule ("atexit");
        if (!atexit) {
            MODINIT_ERROR;
        }
        atexit_register = PyObject_GetAttrString (atexit, "register");
        Py_DECREF (atexit);
        if (!atexit_register) {
            MODINIT_ERROR;
        }
    }

    /* create the module */
#if PY3
    module = PyModule_Create (&_module);
#else
    module = Py_InitModule3 (MODPREFIX "base", _base_methods, DOC_PYGAME);
#endif
    if (module == NULL) {
        MODINIT_ERROR;
    }
    dict = PyModule_GetDict (module);

    /* create the exceptions */
    PyExc_SDLError = PyErr_NewException ("pygame.error", PyExc_RuntimeError,
                                         NULL);
    if (PyExc_SDLError == NULL) {
        Py_XDECREF (atexit_register);
        DECREF_MOD (module);
        MODINIT_ERROR;
    }
    ecode = PyDict_SetItemString (dict, "error", PyExc_SDLError);
    Py_DECREF (PyExc_SDLError);
    if (ecode) {
        Py_XDECREF (atexit_register);
        DECREF_MOD (module);
        MODINIT_ERROR;
    }

#if PG_ENABLE_NEWBUF
    PgExc_BufferError = PyErr_NewException ("pygame.BufferError",
                                            PyExc_BufferError, NULL);
#else
    PgExc_BufferError = PyErr_NewException ("pygame.BufferError",
                                            PyExc_RuntimeError, NULL);
#endif
    if (PyExc_SDLError == NULL) {
        Py_XDECREF (atexit_register);
        DECREF_MOD (module);
        MODINIT_ERROR;
    }
    ecode = PyDict_SetItemString (dict, "BufferError", PyExc_SDLError);
    if (ecode) {
        Py_DECREF (PgExc_BufferError);
        Py_XDECREF (atexit_register);
        DECREF_MOD (module);
        MODINIT_ERROR;
    }

    /* export the c api */
#if PYGAMEAPI_BASE_NUMSLOTS != 21
#warning export slot count mismatch
#endif
    c_api[0] = PyExc_SDLError;
    c_api[1] = PyGame_RegisterQuit;
    c_api[2] = IntFromObj;
    c_api[3] = IntFromObjIndex;
    c_api[4] = TwoIntsFromObj;
    c_api[5] = FloatFromObj;
    c_api[6] = FloatFromObjIndex;
    c_api[7] = TwoFloatsFromObj;
    c_api[8] = UintFromObj;
    c_api[9] = UintFromObjIndex;
    c_api[10] = PyGame_Video_AutoQuit;
    c_api[11] = PyGame_Video_AutoInit;
    c_api[12] = RGBAFromObj;
    c_api[13] = ArrayStructAsDict;
    c_api[14] = PgBuffer_AsArrayInterface;
    c_api[15] = GetArrayStruct;
    c_api[16] = PgBuffer_AsArrayStruct;
    c_api[17] = PgObject_GetBuffer;
    c_api[18] = PgBuffer_Release;
    c_api[19] = PgDict_AsBuffer;
    c_api[20] = PgExc_BufferError;
    apiobj = encapsulate_api (c_api, "base");
    if (apiobj == NULL) {
        Py_XDECREF (atexit_register);
        Py_DECREF (PgExc_BufferError);
        DECREF_MOD (module);
        MODINIT_ERROR;
    }
    ecode = PyDict_SetItemString (dict, PYGAMEAPI_LOCAL_ENTRY, apiobj);
    Py_DECREF (apiobj);
    if (ecode) {
        Py_XDECREF (atexit_register);
        Py_DECREF (PgExc_BufferError);
        DECREF_MOD (module);
        MODINIT_ERROR;
    }

    if (PyModule_AddIntConstant (module, "HAVE_NEWBUF", PG_ENABLE_NEWBUF)) {
        Py_XDECREF (atexit_register);
        Py_DECREF (PgExc_BufferError);
        DECREF_MOD (module);
        MODINIT_ERROR;
    }

    if (!is_loaded) {
        /*some intialization*/
        quit = PyObject_GetAttrString (module, "quit");
        if (quit == NULL) {  /* assertion */
            Py_DECREF (atexit_register);
            Py_DECREF (PgExc_BufferError);
            DECREF_MOD (module);
            MODINIT_ERROR;
        }
        rval = PyObject_CallFunctionObjArgs (atexit_register, quit, NULL);
        Py_DECREF (atexit_register);
        Py_DECREF (quit);
        if (rval == NULL) {
            DECREF_MOD (module);
            Py_DECREF (PgExc_BufferError);
            MODINIT_ERROR;
        }
        Py_DECREF (rval);
        Py_AtExit (atexit_quit);
#ifdef HAVE_SIGNAL_H    
        install_parachute ();
#endif


#ifdef MS_WIN32
        SDL_RegisterApp ("pygame", 0, GetModuleHandle (NULL));
#endif
#if defined(macintosh)
#if(!defined(__MWERKS__) && !TARGET_API_MAC_CARBON)
        SDL_InitQuickDraw (&qd);
#endif
#endif
        }
    is_loaded = 1;
    MODINIT_RETURN (module);
}
