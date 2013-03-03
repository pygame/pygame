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
    PyObject *obj;
    char format[4];      /* make 4 byte word sized */
    Py_ssize_t imem[1];
} ViewInternals;

/* Only one instance of the state per process. */
static PyObject* quitfunctions = NULL;
static int sdl_was_init = 0;

static void install_parachute (void);
static void uninstall_parachute (void);
static void _quit (void);
static void atexit_quit (void);
static int PyGame_Video_AutoInit (void);
static void PyGame_Video_AutoQuit (void);
static int GetArrayInterface (PyObject*, PyObject**, PyArrayInterface**);
static PyObject* ArrayStructAsDict (PyArrayInterface*);
static PyObject* ViewAsDict (Py_buffer*);
static PyObject* ViewAndFlagsAsArrayStruct (Py_buffer*, int);
static int ViewIsByteSwapped (const Py_buffer*);
static int GetView (PyObject*, Py_buffer*);
static void ReleaseView (Py_buffer*);
static PyObject* view_get_typestr_obj (Py_buffer*);
static PyObject* view_get_shape_obj (Py_buffer*);
static PyObject* view_get_strides_obj (Py_buffer*);
static PyObject* view_get_data_obj (Py_buffer*);
static char _as_arrayinter_typekind (const Py_buffer*);
static char _as_arrayinter_byteorder (const Py_buffer*);
static int _as_arrayinter_flags (const Py_buffer*, int);
static CapsuleInterface* _new_capsuleinterface (const Py_buffer*, int);
static void _free_capsuleinterface (void*);
#if PY3
static void _capsule_free_capsuleinterface (PyObject*);
#endif
static PyObject* _shape_as_tuple (PyArrayInterface*);
static PyObject* _typekind_as_str (PyArrayInterface*);
static PyObject* _strides_as_tuple (PyArrayInterface*);
static PyObject* _data_as_tuple (PyArrayInterface*);
static PyObject* get_array_interface (PyObject*, PyObject*);

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
GetArrayInterface (PyObject* obj,
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
ViewAsDict (Py_buffer* view)
{
    PyObject *dictobj =
        Py_BuildValue ("{sisNsNsNsN}",
                       "version", (int)3,
                       "typestr", view_get_typestr_obj (view),
                       "shape", view_get_shape_obj (view),
                       "strides", view_get_strides_obj (view),
                       "data", view_get_data_obj (view));
    PyObject *obj = (PyObject *)view->obj;

    if (!dictobj) {
        return 0;
    }
    if (obj) {
        if (PyDict_SetItemString (dictobj, "__obj", obj)) {
            Py_DECREF (dictobj);
            return 0;
        }
    }
    return dictobj;
}

static PyObject*
ViewAndFlagsAsArrayStruct (Py_buffer* view, int flags)
{
    void *cinter_p  = _new_capsuleinterface (view, flags);
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
        _free_capsuleinterface ((void *)cinter_p);
        return 0;
    }
    return capsule;
}

static CapsuleInterface*
_new_capsuleinterface (const Py_buffer *view, int flags)
{
    int ndim = view->ndim;
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
    cinter_p->inter.typekind = _as_arrayinter_typekind (view);
    cinter_p->inter.itemsize = view->itemsize;
    cinter_p->inter.flags = _as_arrayinter_flags (view, flags);
    if (view->shape) {
        cinter_p->inter.shape = cinter_p->imem;
        for (i = 0; i < ndim; ++i) {
            cinter_p->inter.shape[i] = (Py_intptr_t)view->shape[i];
        }
    }
    if (view->strides) {
        cinter_p->inter.strides = cinter_p->imem + ndim;
        for (i = 0; i < ndim; ++i) {
            cinter_p->inter.strides[i] = (Py_intptr_t)view->strides[i];
        }
    }
    cinter_p->inter.data = view->buf;
    cinter_p->inter.descr = 0;
    cinter_p->parent = (PyObject *)view->obj;
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
_as_arrayinter_flags (const Py_buffer* view, int flags)
{
    int inter_flags = PAI_ALIGNED; /* atomic int types always aligned */

    if (!view->readonly) {
        inter_flags |= PAI_WRITEABLE;
    }
    inter_flags |= ViewIsByteSwapped (view) ? 0 : PAI_NOTSWAPPED;
    if (flags & VIEW_CONTIGUOUS) {
        inter_flags |= PAI_CONTIGUOUS;
    }
    if (flags & VIEW_F_ORDER) {
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
_as_arrayinter_typekind (const Py_buffer* view)
{
    char type = view->format[0];
    char typekind;

    switch (type) {

    case '<':
    case '>':
    case '=':
    case '@':
    case '!':
        type = view->format[1];
    }
    switch (type) {

    case 'c':
    case 'h':
    case 'i':
    case 'l':
    case 'q':
        typekind = 'i';
        break;
    case 'b':
    case 'B':
    case 'H':
    case 'I':
    case 'L':
    case 'Q':
    case 's':
        typekind = 'u';
        break;
    default:
        /* Unknown type */
        typekind = 's';
    }
    return typekind;
}

static char
_as_arrayinter_byteorder (const Py_buffer* view)
{
    char format_0 = view->format[0];
    char byteorder;

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

    if (GetArrayInterface (arg, &cobj, &inter_p)) {
        return 0;
    }
    dictobj = ArrayStructAsDict (inter_p);
    Py_DECREF (cobj);
    return dictobj;
}

static int
GetView (PyObject* obj, Py_buffer* view)
{
    PyObject* cobj = 0;
    PyArrayInterface* inter_p = 0;
    ViewInternals* internal_p;
    Py_ssize_t sz;
    char fchar;
    Py_ssize_t i;

#if PY3
    char *fchar_p;

    if (PyObject_CheckBuffer (obj)) {
        if (PyObject_GetBuffer (obj, &view, PyBUF_RECORDS)) {
            return -1;
        }
        fchar_p = view->format;
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
            ++fchar_p;
            break;
        case 'q':
        case 'Q':
            PyErr_FromString (PyExc_ValueError,
                              "Unsupported integer size of 8 bytes");
            PyBuffer_Release (view);
            return -1;
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
            /* A record: will raise exception later */
            break;
        default:
            PyErr_FromString (PyExc_ValueError,
                              "Unsupported array element type");
            PyBuffer_Release (view);
            return -1;
        }
        if (fchar_p != '\0') {
            PyErr_FromString (PyExc_ValueError,
                              "Arrays of records are unsupported");
            PyBuffer_Release (view);
            return -1;
        }
    }
    else
#endif
    if (PyObject_HasAttrString (obj, "__array_struct__")) {
        if (!GetArrayInterface (obj, &cobj, &inter_p)) {
            return -1;
        }
        sz = (sizeof (ViewInternals) + 
              (2 * inter_p->nd - 1) * sizeof (Py_ssize_t));
        internal_p = (ViewInternals*)PyMem_Malloc (sz);
        if (!internal_p) {
            Py_DECREF (cobj);
            PyErr_NoMemory ();
            return -1;
        }
        switch (inter_p->typekind) {

        case 'i':
            switch (inter_p->itemsize) {

            case 1:
                fchar = 'b';
                break; 
            case 2:
                fchar = 'h';
                break;
            case 4:
                fchar = 'i';
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
            switch (inter_p->itemsize) {

            case 1:
                fchar = 'B';
                break; 
            case 2:
                fchar = 'H';
                break;
            case 4:
                fchar = 'I';
                break;
            default:
                PyErr_Format (PyExc_ValueError,
                              "Unsupported unsigned interger size %d",
                              (int)inter_p->itemsize);
                Py_DECREF (cobj);
                return -1;
            }
            break;
        default:
            PyErr_Format (PyExc_ValueError,
                          "Unsupported value type '%c'",
                          (int)inter_p->typekind);
            Py_DECREF (cobj);
            return -1;
        }
        view->internal = internal_p;
        view->format = internal_p->format;
        view->shape = internal_p->imem;
        view->strides = view->shape + view->ndim;
        internal_p->obj = obj;
        Py_INCREF (obj);
        view->obj = cobj;
        view->ndim = (Py_ssize_t)inter_p->nd;
        view->itemsize = (Py_ssize_t)inter_p->itemsize;
        view->readonly = inter_p->flags & PAI_WRITEABLE ? 0 : 1;
        view->format[0] = (inter_p->flags & PAI_NOTSWAPPED ?
                           PAI_MY_ENDIAN : PAI_OTHER_ENDIAN);
        view->format[1] = fchar;
        view->format[2] = '\0';
        for (i = 0; i < view->ndim; ++i) {
            view->shape[i] = (Py_ssize_t)inter_p->shape[i];
            view->strides[i] = (Py_ssize_t)inter_p->strides[i];
        }
        view->suboffsets = 0;
    }
    else {
        PyErr_Format (PyExc_TypeError,
                      "%s object does not export a C level array buffer",
                      Py_TYPE (obj)->tp_name);
        return -1;
    }
    return 0;
}

static void
ReleaseView (Py_buffer* view)
{
#if PY3
    if (PyCapsule_CheckExact (view->obj)) {
        Py_DECREF (view->obj);
        Py_XDECREF (((ViewInternals*)view->internal)->obj);
        PyMem_Free (view->internal);
    }
    else {
        PyBuffer_Release (view);
    }
#else
    Py_DECREF (view->obj);
    Py_XDECREF (((ViewInternals*)view->internal)->obj);
    PyMem_Free (view->internal);
#endif
}

static int
ViewIsByteSwapped (const Py_buffer* view)
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

    /* export the c api */
#if PYGAMEAPI_BASE_NUMSLOTS != 20
#error export slot count mismatch
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
    c_api[14] = ViewAsDict;
    c_api[15] = GetArrayInterface;
    c_api[16] = ViewAndFlagsAsArrayStruct;
    c_api[17] = ViewIsByteSwapped;
    c_api[18] = GetView;
    c_api[19] = ReleaseView;
    apiobj = encapsulate_api (c_api, "base");
    if (apiobj == NULL) {
        Py_XDECREF (atexit_register);
        DECREF_MOD (module);
        MODINIT_ERROR;
    }
    ecode = PyDict_SetItemString (dict, PYGAMEAPI_LOCAL_ENTRY, apiobj);
    Py_DECREF (apiobj);
    if (ecode) {
        Py_XDECREF (atexit_register);
        DECREF_MOD (module);
        MODINIT_ERROR;
    }

    if (!is_loaded) {
        /*some intialization*/
        quit = PyObject_GetAttrString (module, "quit");
        if (quit == NULL) {  /* assertion */
            Py_DECREF (atexit_register);
            DECREF_MOD (module);
            MODINIT_ERROR;
        }
        rval = PyObject_CallFunctionObjArgs (atexit_register, quit, NULL);
        Py_DECREF (atexit_register);
        Py_DECREF (quit);
        if (rval == NULL) {
            DECREF_MOD (module);
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
