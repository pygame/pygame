/*
  pygame - Python Game Library
  Copyright (C) 2010 Marcus von Appen

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
#define PYGAME_OPENALLISTENER_INTERNAL

#include "openalmod.h"
#include "pgopenal.h"

static int _listener_init (PyObject *self, PyObject *args, PyObject *kwds);
static void _listener_dealloc (PyListener *self);
static PyObject* _listener_repr (PyObject *self);

static PyObject* _listener_setprop (PyObject *self, PyObject *args);
static PyObject* _listener_getprop (PyObject *self, PyObject *args);

/**
 */
static PyMethodDef _listener_methods[] = {
    { "set_prop", _listener_setprop, METH_VARARGS, NULL },
    { "get_prop", _listener_getprop, METH_VARARGS, NULL },
    { NULL, NULL, 0, NULL }
};

/**
 */
static PyGetSetDef _listener_getsets[] = {
    { NULL, NULL, NULL, NULL, NULL }
};

/**
 */
PyTypeObject PyListener_Type =
{
    TYPE_HEAD(NULL, 0)
    "base.Listener",            /* tp_name */
    sizeof (PyListener),        /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor) _listener_dealloc, /* tp_dealloc */
    0,                          /* tp_print */
    0,                          /* tp_getattr */
    0,                          /* tp_setattr */
    0,                          /* tp_compare */
    (reprfunc)_listener_repr,   /* tp_repr */
    0,                          /* tp_as_number */
    0,                          /* tp_as_sequence */
    0,                          /* tp_as_mapping */
    0,                          /* tp_hash */
    0,                          /* tp_call */
    0,                          /* tp_str */
    0,                          /* tp_getattro */
    0,                          /* tp_setattro */
    0,                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    0/*DOC_BASE_DEVICE*/,
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    _listener_methods,          /* tp_methods */
    0,                          /* tp_members */
    _listener_getsets,          /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc) _listener_init,  /* tp_init */
    0,                          /* tp_alloc */
    0,                          /* tp_new */
    0,                          /* tp_free */
    0,                          /* tp_is_gc */
    0,                          /* tp_bases */
    0,                          /* tp_mro */
    0,                          /* tp_cache */
    0,                          /* tp_subclasses */
    0,                          /* tp_weaklist */
    0,                          /* tp_del */
#if PY_VERSION_HEX >= 0x02060000
    0                           /* tp_version_tag */
#endif
};

static void
_listener_dealloc (PyListener *self)
{
    ((PyObject*)self)->ob_type->tp_free ((PyObject *) self);
}

static int
_listener_init (PyObject *self, PyObject *args, PyObject *kwds)
{
    PyErr_SetString (PyExc_NotImplementedError,
        "Listener cannot be created dirrectly - use the Context instead");
    return -1;
}

static PyObject*
_listener_repr (PyObject *self)
{
    return Text_FromUTF8 ("<Listener>");
}

/* Listener getters/setters */

/* Listener methods */
static PyObject*
_listener_setprop (PyObject *self, PyObject *args)
{
    ALenum param;
    PyObject *values;
    char *type;
    PropType ptype = INVALID;

    if (!CONTEXT_IS_CURRENT (((PyListener*)self)->context))
    {
        PyErr_SetString (PyExc_PyGameError, "buffer context is not current");
        return NULL;
    }

    if (!PyArg_ParseTuple (args, "lO|s:set_prop", &param, &values, &type))
        return NULL;

    if (type)
    {
        ptype = GetPropTypeFromStr (type);
        if (ptype == INVALID)
        {
            PyErr_SetString (PyExc_RuntimeError,
                "passing a sequence requires passing a type specifier");
            return NULL;
        }
    }

    if (PySequence_Check (values))
    {
        Py_ssize_t size, cnt;
        if (!type)
        {
            PyErr_SetString (PyExc_RuntimeError,
                "passing a sequence requires passing a type specifier");
            return NULL;
        }
        if (ptype == INT || ptype == FLOAT)
        {
            PyErr_SetString (PyExc_TypeError,
                "cannot use single value type and sequence together");
            return NULL;
        }

        size = PySequence_Size (values);
        switch (ptype)
        {
        case INT3:
        case INTARRAY:
        {
            ALint *vals;
            int tmp;
            if (ptype == INT3 && size < 3)
            {
                PyErr_SetString (PyExc_ValueError,
                    "sequence too small for 'i3'");
                return NULL;
            }
            vals = PyMem_New (ALint, size);
            if (!vals)
                return NULL;
            for (cnt = 0; cnt < size; cnt++)
            {
                if (!IntFromSeqIndex (values, cnt, &tmp))
                {
                    PyMem_Free (vals);
                    return NULL;
                }
                vals[cnt] = (ALint) tmp;
            }

            CLEAR_ERROR_STATE ();
            if (ptype == INT3)
                alListener3i (param, vals[0], vals[1], vals[2]);
            else
                alListeneriv (param, vals);
            PyMem_Free (vals);
            /* Error will be set at the end */
            break;
        }
        case FLOAT3:
        case FLOATARRAY:
        {
            ALfloat *vals;
            double tmp;
            if (ptype == FLOAT3 && size < 3)
            {
                PyErr_SetString (PyExc_ValueError,
                    "sequence too small for 'f3'");
                return NULL;
            }
            vals = PyMem_New (ALfloat, size);
            if (!vals)
                return NULL;
            for (cnt = 0; cnt < size; cnt++)
            {
                if (!DoubleFromSeqIndex (values, cnt, &tmp))
                {
                    PyMem_Free (vals);
                    return NULL;
                }
                vals[cnt] = (ALfloat) tmp;
            }

            CLEAR_ERROR_STATE ();
            if (ptype == FLOAT3)
                alListener3f (param, vals[0], vals[1], vals[2]);
            else
                alListenerfv (param, vals);
            PyMem_Free (vals);
            /* Error will be set at the end */
            break;
        }
        default:
            PyErr_SetString (PyExc_TypeError, "unsupported value");
            return NULL;
        }
    }
    else
    {
        int ival = 0;
        double fval = 0;

        if (!type)
        {
            if (IntFromObj (values, &ival))
                ptype = INT;
            else
                PyErr_Clear ();
            if (DoubleFromObj (values, &fval))
                ptype = FLOAT;
            else
            {
                PyErr_Clear ();
                PyErr_SetString (PyExc_TypeError, "unsupported value");
                return NULL;
            }
        }
        
        switch (ptype)
        {
        case INT:
            if (!IntFromObj (values, &ival))
                return NULL;
            CLEAR_ERROR_STATE ();
            alListeneri (param, (ALint) ival);
            break;
        case FLOAT:
            if (!DoubleFromObj (values, &fval))
                return NULL;
            CLEAR_ERROR_STATE ();
            alListenerf (param, (ALfloat) fval);
            break;
        default:
            PyErr_SetString (PyExc_TypeError, "value type mismatch");
            return NULL;
        }
    }

    if (SetALErrorException (alGetError ()))
        return NULL;
    Py_RETURN_NONE;
}

static PyObject*
_listener_getprop (PyObject *self, PyObject *args)
{
    ALenum param;
    char *type;
    int size = 0, cnt;
    PropType ptype = INVALID;

    if (!CONTEXT_IS_CURRENT (((PyListener*)self)->context))
    {
        PyErr_SetString (PyExc_PyGameError, "buffer context is not current");
        return NULL;
    }

    if (!PyArg_ParseTuple (args, "ls", &param, &type))
    {
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "l|si", &param, &type, &size))
            return NULL;
        if (size <= 0)
        {
            PyErr_SetString (PyExc_ValueError, "size must not smaller than 0");
            return NULL;
        }
    }

    ptype = GetPropTypeFromStr (type);
    CLEAR_ERROR_STATE ();
    switch (ptype)
    {
    case INT:
    {
        ALint val;
        alGetListeneri (param, &val);
        if (SetALErrorException (alGetError ()))
            return NULL;
        return PyLong_FromLong ((long)val);
    }
    case FLOAT:
    {
        ALfloat val;
        alGetListenerf (param, &val);
        if (SetALErrorException (alGetError ()))
            return NULL;
        return PyFloat_FromDouble ((double)val);
    }
    case INT3:
    {
        ALint val[3];
        alGetListener3i (param, &val[0], &val[1], &val[2]);
        if (SetALErrorException (alGetError ()))
            return NULL;
        return Py_BuildValue ("(lll)", (long)val[0], (long)val[1],
            (long)val[2]);
    }
    case FLOAT3:
    {
        ALfloat val[3];
        alGetListener3f (param, &val[0], &val[1], &val[2]);
        if (SetALErrorException (alGetError ()))
            return NULL;
        return Py_BuildValue ("(ddd)", (double)val[0], (double)val[1],
            (double)val[2]);
    }
    case INTARRAY:
    {
        PyObject *tuple, *item;
        ALint* val = PyMem_New (ALint, size);
        if (!val)
            return NULL;
        alGetListeneriv (param, val);
        if (SetALErrorException (alGetError ()))
        {
            PyMem_Free (val);
            return NULL;
        }
        tuple = PyTuple_New ((Py_ssize_t) size);
        if (!tuple)
            return NULL;
        for (cnt = 0; cnt < size; cnt++)
        {
            item = PyLong_FromLong ((long)val[cnt]);
            if (!item)
            {
                PyMem_Free (val);
                Py_DECREF (tuple);
                return NULL;
            }
            PyTuple_SET_ITEM (tuple, (Py_ssize_t) cnt, item);
        }
        return tuple;
    }
    case FLOATARRAY:
    {
        PyObject *tuple, *item;
        ALfloat* val = PyMem_New (ALfloat, size);
        if (!val)
            return NULL;
        alGetListenerfv (param, val);
        if (SetALErrorException (alGetError ()))
        {
            PyMem_Free (val);
            return NULL;
        }
        tuple = PyTuple_New ((Py_ssize_t) size);
        if (!tuple)
            return NULL;
        for (cnt = 0; cnt < size; cnt++)
        {
            item = PyFloat_FromDouble ((double)val[cnt]);
            if (!item || PyTuple_SET_ITEM (tuple, (Py_ssize_t) cnt, item) != 0)
            {
                PyMem_Free (val);
                Py_XDECREF (item);
                Py_DECREF (tuple);
                return NULL;
            }
        }
        return tuple;
    }
    default:
        PyErr_SetString (PyExc_ValueError, "invalid type specifier");
        return NULL;
    }
}

/* C API */
PyObject*
PyListener_New (PyObject *context)
{
    PyObject *listener;

    if (!context || !PyContext_Check (context))
    {
        PyErr_SetString (PyExc_TypeError, "context is not a valid Context");
        return NULL;
    }

    listener = PyListener_Type.tp_new (&PyListener_Type, NULL, NULL);
    if (!listener)
        return NULL;
    ((PyListener*)listener)->context = context;
    return listener;
}

void
listener_export_capi (void **capi)
{
    capi[PYGAME_OPENALLISTENER_FIRSTSLOT] = &PyListener_Type;
}
