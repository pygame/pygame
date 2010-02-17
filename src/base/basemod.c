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

#include "pymacros.h"
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
LongFromObj (PyObject* obj, long* val)
{
    PyObject* longobj;
    long tmp;

    if (!obj || !val)
    {
        PyErr_SetString (PyExc_TypeError, "argument is NULL");
        return 0;
    }
    
    if (PyNumber_Check (obj))
    {
        if (!(longobj = PyNumber_Long (obj)))
            return 0;
        tmp = PyLong_AsLong (longobj);
        Py_DECREF (longobj);
        if (tmp == -1 && PyErr_Occurred ())
            return 0;
        *val = tmp;
        return 1;
    }
    PyErr_SetString (PyExc_TypeError, "value must be a number object");
    return 0;
}

unsigned long
UlongFromObj (PyObject* obj, long* val)
{
    PyObject* longobj;
    unsigned long tmp;
    
    if (!obj || !val)
    {
        PyErr_SetString (PyExc_TypeError, "argument is NULL");
        return 0;
    }
    
    if (PyNumber_Check (obj))
    {
        if (!(longobj = PyNumber_Long (obj)))
            return 0;
        tmp = PyLong_AsUnsignedLong (longobj);
        Py_DECREF (longobj);
        if (PyErr_Occurred ())
            return 0;
        *val = tmp;
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
PointFromObj (PyObject *obj, int *x, int *y)
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
        if (!IntFromSeqIndex (obj, (Py_ssize_t)0, x))
            goto failed;
        if (!IntFromSeqIndex (obj, (Py_ssize_t)1, y))
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
FPointFromObj (PyObject *obj, double *x, double *y)
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
SizeFromObj (PyObject *obj, pgint32 *w, pgint32 *h)
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
FSizeFromObj (PyObject *obj, double *w, double *h)
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
ASCIIFromObj (PyObject *obj, char **text, PyObject **freeme)
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
UTF8FromObj (PyObject *obj, char **text, PyObject **freeme)
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

int
ColorFromObj (PyObject *obj, pguint32 *val)
{
    if (!obj || !val)
    {
        PyErr_SetString (PyExc_TypeError, "argument is NULL");
        return 0;
    }
    
    if (PyColor_Check (obj))
    {
        PyColor *color = (PyColor*) obj;
        *val = ((pguint32) color->r << 24) | ((pguint32) color->g << 16) |
            ((pguint32) color->b << 8) | ((pguint32) color->a);
        return 1;
    }
    else if (PyLong_Check (obj))
    {
        unsigned long longval = PyLong_AsUnsignedLong (obj);
        if (PyErr_Occurred ())
        {
            PyErr_Clear ();
            PyErr_SetString(PyExc_ValueError, "invalid color argument");
            return 0;
        }
        *val = (pguint32) longval;
        return 1;
    }
    else if (PyInt_Check (obj))
    {
        long intval = PyInt_AsLong (obj);
        if (intval == -1 && PyErr_Occurred ())
        {
            PyErr_Clear ();
            PyErr_SetString (PyExc_ValueError, "invalid color argument");
            return 0;
        }
        *val = (pguint32) intval;
        return 1;
    }
    else
        PyErr_SetString (PyExc_TypeError, "invalid color argument");
    return 0;
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
    
#ifdef IS_PYTHON_3
    mod = PyModule_Create (&_basemodule);
#else
    mod = Py_InitModule3 ("base", _base_methods, DOC_BASE);
#endif
    if (!mod)
        MODINIT_RETURN(NULL);
    state = BASE_MOD_STATE(mod);

    /* Setup the pygame exeption */
    state->error = PyErr_NewException ("pygame2.Error", NULL, NULL);
    if (!state->error)
    {
        Py_DECREF (mod);
        MODINIT_RETURN(NULL);
    }

    /* Complete types */
    PyColor_Type.tp_new = PyType_GenericNew;
    if (PyType_Ready (&PyColor_Type) < 0)
        goto failed;
    PyFRect_Type.tp_new = PyType_GenericNew;
    if (PyType_Ready (&PyFRect_Type) < 0)
        goto failed;
    PyRect_Type.tp_new = PyType_GenericNew;
    if (PyType_Ready (&PyRect_Type) < 0)
        goto failed;
    if (PyType_Ready (&PyBufferProxy_Type) < 0)
        goto failed;
    if (PyType_Ready (&PySurface_Type) < 0)
        goto failed;
    if (PyType_Ready(&PyFont_Type) < 0)
        goto failed;

    Py_INCREF(state->error);
    if (PyModule_AddObject (mod, "Error", state->error) == -1)
    {
        Py_DECREF (state->error);
        goto failed;
    }

    ADD_OBJ_OR_FAIL (mod, "Color", PyColor_Type, failed);
    ADD_OBJ_OR_FAIL (mod, "Rect", PyRect_Type, failed);
    ADD_OBJ_OR_FAIL (mod, "FRect", PyFRect_Type, failed);
    ADD_OBJ_OR_FAIL (mod, "BufferProxy", PyBufferProxy_Type, failed);
    ADD_OBJ_OR_FAIL (mod, "Surface", PySurface_Type, failed);
    ADD_OBJ_OR_FAIL (mod, "Font", PyFont_Type, failed);
    
    /* Export C API */
    c_api[PYGAME_BASE_FIRSTSLOT] = state->error;
    c_api[PYGAME_BASE_FIRSTSLOT+1] = DoubleFromObj;
    c_api[PYGAME_BASE_FIRSTSLOT+2] = IntFromObj;
    c_api[PYGAME_BASE_FIRSTSLOT+3] = UintFromObj;
    c_api[PYGAME_BASE_FIRSTSLOT+4] = DoubleFromSeqIndex;
    c_api[PYGAME_BASE_FIRSTSLOT+5] = IntFromSeqIndex;
    c_api[PYGAME_BASE_FIRSTSLOT+6] = UintFromSeqIndex;
    c_api[PYGAME_BASE_FIRSTSLOT+7] = PointFromObj;
    c_api[PYGAME_BASE_FIRSTSLOT+8] = SizeFromObj;
    c_api[PYGAME_BASE_FIRSTSLOT+9] = FPointFromObj;
    c_api[PYGAME_BASE_FIRSTSLOT+10] = FSizeFromObj;
    c_api[PYGAME_BASE_FIRSTSLOT+11] = ASCIIFromObj;
    c_api[PYGAME_BASE_FIRSTSLOT+12] = UTF8FromObj;
    c_api[PYGAME_BASE_FIRSTSLOT+13] = UlongFromObj;
    c_api[PYGAME_BASE_FIRSTSLOT+14] = LongFromObj;
    c_api[PYGAME_BASE_FIRSTSLOT+15] = ColorFromObj;

    streamwrapper_export_capi (c_api);
    color_export_capi (c_api);
    rect_export_capi (c_api);
    floatrect_export_capi (c_api);
    bufferproxy_export_capi (c_api);
    surface_export_capi (c_api);
    font_export_capi (c_api);

    c_api_obj = PyCObject_FromVoidPtr ((void *) c_api, NULL);
    if (c_api_obj)
    {
        if (PyModule_AddObject (mod, PYGAME_BASE_ENTRY, c_api_obj) == -1)
        {
            Py_DECREF (c_api_obj);
            goto failed;
        }
    }
    MODINIT_RETURN(mod);

failed:
    Py_DECREF (mod);
    MODINIT_RETURN (NULL);
}
