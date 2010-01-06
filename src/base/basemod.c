/*
  pygame - Python Game Library
  Copyright (C) 2000-2001  Pete Shinners, 2008 Marcus von Appen

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
#define PYGAME_BASE_INTERNAL

#include "internals.h"
#include "pgbase.h"
#include "base_doc.h"

static int _base_traverse (PyObject *mod, visitproc visit, void *arg);
static int _base_clear (PyObject *mod);

static PyMethodDef _base_methods[] = {
    { NULL, NULL, 0, NULL }
};

int
DoubleFromObj (PyObject* obj, double* val)
{
    double tmp;

    if (!obj || !val)
    {
        PyErr_SetString (PyExc_TypeError, "argument is NULL");
        return 0;
    }

    tmp = PyFloat_AsDouble (obj);
    if (tmp == -1 && PyErr_Occurred ())
        return 0;
    *val = tmp;
    return 1;
}

int
IntFromObj (PyObject* obj, int* val)
{
    PyObject* intobj;
    long tmp;

    if (!obj || !val)
    {
        PyErr_SetString (PyExc_TypeError, "argument is NULL");
        return 0;
    }
    
    if (PyNumber_Check (obj))
    {
        if (!(intobj = PyNumber_Int (obj)))
            return 0;
        tmp = PyInt_AsLong (intobj);
        Py_DECREF (intobj);
        if (tmp == -1 && PyErr_Occurred ())
            return 0;
        if (tmp > INT_MAX)
        {
            PyErr_SetString (PyExc_ValueError, "value exceeds allowed range");
            return 0;
        }
        *val = (int)tmp;
        return 1;
    }
    PyErr_SetString (PyExc_TypeError, "value must be a number object");
    return 0;
}

int
UintFromObj (PyObject* obj, unsigned int* val)
{
    PyObject* intobj;
    long tmp;
    
    if (!obj || !val)
    {
        PyErr_SetString (PyExc_TypeError, "argument is NULL");
        return 0;
    }

    if (PyNumber_Check (obj))
    {
        if (!(intobj = PyNumber_Int (obj)))
            return 0;
        tmp = PyInt_AsLong (intobj);
        Py_DECREF (intobj);
        if (tmp == -1 && PyErr_Occurred ())
            return 0;
        if (tmp < 0)
        {
            PyErr_SetString (PyExc_ValueError, "value must not be negative");
            return 0;
        }
        if (tmp > UINT_MAX)
        {
            PyErr_SetString (PyExc_ValueError, "value exceeds allowed range");
            return 0;
        }

        *val = tmp;
        return 1;
    }
    PyErr_SetString (PyExc_TypeError, "value must be a number object");
    return 0;
}

int
IntFromSeqIndex (PyObject* obj, Py_ssize_t _index, int* val)
{
    int result = 0;
    PyObject* item;

    if (!obj || !val)
    {
        PyErr_SetString (PyExc_TypeError, "argument is NULL");
        return 0;
    }

    item = PySequence_GetItem (obj, _index);
    if (item)
    {
        result = IntFromObj (item, val);
        Py_DECREF (item);
    }
    return result;
}

int
UintFromSeqIndex (PyObject* obj, Py_ssize_t _index, unsigned int* val)
{
    int result = 0;
    PyObject* item;

    if (!obj || !val)
    {
        PyErr_SetString (PyExc_TypeError, "argument is NULL");
        return 0;
    }

    item = PySequence_GetItem (obj, _index);
    if (item)
    {
        result = UintFromObj (item, val);
        Py_DECREF (item);
    }
    return result;
}

int
DoubleFromSeqIndex (PyObject* obj, Py_ssize_t _index, double* val)
{
    int result = 0;
    PyObject* item;

    if (!obj || !val)
    {
        PyErr_SetString (PyExc_TypeError, "argument is NULL");
        return 0;
    }

    item = PySequence_GetItem (obj, _index);
    if (item)
    {
        result = DoubleFromObj (item, val);
        Py_DECREF (item);
    }
    return result;
}

int
PointFromObject (PyObject *obj, int *x, int *y)
{
    if (!obj || !x || !y)
    {
        PyErr_SetString (PyExc_TypeError, "argument is NULL");
        return 0;
    }

    if (PyRect_Check (obj))
    {
        *x = (int) ((PyRect*)obj)->x;
        *y = (int) ((PyRect*)obj)->y;
        return 1;
    }
    else if (PyFRect_Check (obj))
    {
        *x = (int) round (((PyFRect*)obj)->x);
        *y = (int) round (((PyFRect*)obj)->y);
        return 1;
    }
    else if (PySequence_Check (obj) && PySequence_Size (obj) >= 2)
    {
        if (!IntFromSeqIndex (obj, 0, x))
            goto failed;
        if (!IntFromSeqIndex (obj, 1, y))
            goto failed;
        return 1;
    }
failed:
    if (!PyErr_Occurred ())
        PyErr_SetString (PyExc_TypeError,
            "object must be a Rect, FRect or 2-value sequence");
    return 0;
}

int
FPointFromObject (PyObject *obj, double *x, double *y)
{
    if (!obj || !x || !y)
    {
        PyErr_SetString (PyExc_TypeError, "argument is NULL");
        return 0;
    }

    if (PyRect_Check (obj))
    {
        *x = (double) ((PyRect*)obj)->x;
        *y = (double) ((PyRect*)obj)->y;
        return 1;
    }
    else if (PyFRect_Check (obj))
    {
        *x = ((PyFRect*)obj)->x;
        *y = ((PyFRect*)obj)->y;
        return 1;
    }
    else if (PySequence_Check (obj) && PySequence_Size (obj) >= 2)
    {
        if (!DoubleFromSeqIndex (obj, 0, x))
            goto failed;
        if (!DoubleFromSeqIndex (obj, 1, y))
            goto failed;
        return 1;
    }
failed:
    if (!PyErr_Occurred ())
        PyErr_SetString (PyExc_TypeError,
            "object must be a Rect, FRect or 2-value sequence");
    return 0;
}

int
SizeFromObject (PyObject *obj, pgint32 *w, pgint32 *h)
{
    if (!obj || !w || !h)
    {
        PyErr_SetString (PyExc_TypeError, "argument is NULL");
        return 0;
    }

    if (PyRect_Check (obj))
    {
        *w = (pgint32) ((PyRect*)obj)->w;
        *h = (pgint32) ((PyRect*)obj)->h;
        return 1;
    }
    else if (PyFRect_Check (obj))
    {
        *w = (pgint32) round (((PyFRect*)obj)->w);
        *h = (pgint32) round (((PyFRect*)obj)->h);
        return 1;
    }
    else if (PySequence_Check (obj) && PySequence_Size (obj) >= 2)
    {
        if (!IntFromSeqIndex (obj, 0, (int*)w))
            goto failed;
        if (!IntFromSeqIndex (obj, 1, (int*)h))
            goto failed;
        return 1;
    }
failed:
    if (!PyErr_Occurred ())
        PyErr_SetString (PyExc_TypeError,
            "object must be a Rect, FRect or 2-value sequence");
    return 0;
}

int
FSizeFromObject (PyObject *obj, double *w, double *h)
{
    if (!obj || !w || !h)
    {
        PyErr_SetString (PyExc_TypeError, "argument is NULL");
        return 0;
    }

    if (PyRect_Check (obj))
    {
        *w = (double) ((PyRect*)obj)->w;
        *h = (double) ((PyRect*)obj)->h;
        return 1;
    }
    else if (PyFRect_Check (obj))
    {
        *w = ((PyFRect*)obj)->w;
        *h = ((PyFRect*)obj)->h;
        return 1;
    }
    else if (PySequence_Check (obj) && PySequence_Size (obj) >= 2)
    {
        if (!DoubleFromSeqIndex (obj, 0, w))
            goto failed;
        if (!DoubleFromSeqIndex (obj, 1, h))
            goto failed;
        return 1;
    }
failed:
    if (!PyErr_Occurred ())
        PyErr_SetString (PyExc_TypeError,
            "object must be a Rect, FRect or 2-value sequence");
    return 0;
}

int
ASCIIFromObject (PyObject *obj, char **text, PyObject **freeme)
{
    if (!obj || !text || !freeme)
    {
        PyErr_SetString (PyExc_TypeError, "argument is NULL");
        return 0;
    }

    *freeme = NULL;
    *text = NULL;

    if (PyUnicode_Check (obj))
    {
        *freeme = PyUnicode_AsEncodedString (obj, "ascii", NULL);
        if (!(*freeme))
            return 0;
        *text = Bytes_AS_STRING (*freeme);
    }
    else if (Bytes_Check (obj))
        *text = Bytes_AS_STRING (obj);
    else
        return 0;

    return 1;
}

int
UTF8FromObject (PyObject *obj, char **text, PyObject **freeme)
{
    if (!obj || !text || !freeme)
    {
        PyErr_SetString (PyExc_TypeError, "argument is NULL");
        return 0;
    }

    *freeme = NULL;
    *text = NULL;

    if (PyUnicode_Check (obj))
    {
        *freeme = PyUnicode_AsUTF8String (obj);
        if (!(*freeme))
            return 0;
        *text = Bytes_AS_STRING (*freeme);
    }
    else if (Bytes_Check (obj))
        *text = Bytes_AS_STRING (obj);
    else
        return 0;

    return 1;
}

static int
_base_traverse (PyObject *mod, visitproc visit, void *arg)
{
    Py_VISIT (BASE_MOD_STATE(mod)->error);
    return 0;
}

static int
_base_clear (PyObject *mod)
{
    Py_CLEAR (BASE_MOD_STATE(mod)->error);
    return 0;
}

#ifdef IS_PYTHON_3
struct PyModuleDef _basemodule = {
    PyModuleDef_HEAD_INIT,
    "base",
    DOC_BASE, 
    sizeof (_BaseState),
    _base_methods,
    NULL,
    _base_traverse,
    _base_clear,
    NULL
};
#else
_BaseState _modstate;
#endif

#ifdef IS_PYTHON_3
PyMODINIT_FUNC PyInit_base (void)
#else
PyMODINIT_FUNC initbase (void)
#endif
{
    static void* c_api[PYGAME_BASE_SLOTS];
    _BaseState *state;
    PyObject *mod, *c_api_obj;
    
    /* Complete types */
    PyColor_Type.tp_new = PyType_GenericNew;
    if (PyType_Ready (&PyColor_Type) < 0)
        MODINIT_RETURN(NULL);
    PyFRect_Type.tp_new = PyType_GenericNew;
    if (PyType_Ready (&PyFRect_Type) < 0)
        MODINIT_RETURN(NULL);
    PyRect_Type.tp_new = PyType_GenericNew;
    if (PyType_Ready (&PyRect_Type) < 0)
        MODINIT_RETURN(NULL);
    if (PyType_Ready (&PyBufferProxy_Type) < 0)
        MODINIT_RETURN(NULL);
    if (PyType_Ready (&PySurface_Type) < 0)
        MODINIT_RETURN(NULL);
    if (PyType_Ready(&PyFont_Type) < 0)
        MODINIT_RETURN(NULL);

    Py_INCREF (&PyColor_Type);
    Py_INCREF (&PyFRect_Type);
    Py_INCREF (&PyRect_Type);
    Py_INCREF (&PyBufferProxy_Type);
    Py_INCREF (&PySurface_Type);
    Py_INCREF (&PyFont_Type);

#ifdef IS_PYTHON_3
    mod = PyModule_Create (&_basemodule);
#else
    mod = Py_InitModule3 ("base", _base_methods, DOC_BASE);
#endif
    if (!mod)
        MODINIT_RETURN(NULL);
    state = BASE_MOD_STATE(mod);

    /* Setup the pygame exeption */
    state->error = PyErr_NewException ("base.Error", NULL, NULL);
    if (!state->error)
    {
        Py_DECREF (mod);
        MODINIT_RETURN(NULL);
    }

    Py_INCREF(state->error);
    PyModule_AddObject (mod, "Error", state->error);

    PyModule_AddObject (mod, "Color", (PyObject *) &PyColor_Type);
    PyModule_AddObject (mod, "Rect", (PyObject *) &PyRect_Type);
    PyModule_AddObject (mod, "FRect", (PyObject *) &PyFRect_Type);
    PyModule_AddObject (mod, "BufferProxy", (PyObject *) &PyBufferProxy_Type);
    PyModule_AddObject (mod, "Surface", (PyObject *) &PySurface_Type);
    PyModule_AddObject (mod, "Font", (PyObject *) &PyFont_Type);
    
    /* Export C API */
    c_api[PYGAME_BASE_FIRSTSLOT] = state->error;
    c_api[PYGAME_BASE_FIRSTSLOT+1] = DoubleFromObj;
    c_api[PYGAME_BASE_FIRSTSLOT+2] = IntFromObj;
    c_api[PYGAME_BASE_FIRSTSLOT+3] = UintFromObj;
    c_api[PYGAME_BASE_FIRSTSLOT+4] = DoubleFromSeqIndex;
    c_api[PYGAME_BASE_FIRSTSLOT+5] = IntFromSeqIndex;
    c_api[PYGAME_BASE_FIRSTSLOT+6] = UintFromSeqIndex;
    c_api[PYGAME_BASE_FIRSTSLOT+7] = PointFromObject;
    c_api[PYGAME_BASE_FIRSTSLOT+8] = SizeFromObject;
    c_api[PYGAME_BASE_FIRSTSLOT+9] = FPointFromObject;
    c_api[PYGAME_BASE_FIRSTSLOT+10] = FSizeFromObject;
    c_api[PYGAME_BASE_FIRSTSLOT+11] = ASCIIFromObject;
    c_api[PYGAME_BASE_FIRSTSLOT+12] = UTF8FromObject;

    color_export_capi (c_api);
    rect_export_capi (c_api);
    floatrect_export_capi (c_api);
    bufferproxy_export_capi (c_api);
    surface_export_capi (c_api);
    font_export_capi (c_api);

    c_api_obj = PyCObject_FromVoidPtr ((void *) c_api, NULL);
    if (c_api_obj)
        PyModule_AddObject (mod, PYGAME_BASE_ENTRY, c_api_obj);
    MODINIT_RETURN(mod);
}
