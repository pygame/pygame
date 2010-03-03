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
#define PYGAME_OPENALBUFFERS_INTERNAL

#include "openalmod.h"
#include "pgopenal.h"

static int _buffers_init (PyObject *self, PyObject *args, PyObject *kwds);
static void _buffers_dealloc (PyBuffers *self);
static PyObject* _buffers_repr (PyObject *self);

static PyObject* _buffers_setprop (PyObject *self, PyObject *args);
static PyObject* _buffers_getprop (PyObject *self, PyObject *args);
static PyObject* _buffers_bufferdata (PyObject *self, PyObject *args);

static PyObject* _buffers_getcount (PyObject* self, void *closure);
static PyObject* _buffers_getbuffers (PyObject* self, void *closure);

/**
 */
static PyMethodDef _buffers_methods[] = {
    { "set_prop", _buffers_setprop, METH_VARARGS, NULL },
    { "get_prop", _buffers_getprop, METH_VARARGS, NULL },
    { "buffer_data", _buffers_bufferdata, METH_VARARGS, NULL },
    { NULL, NULL, 0, NULL }
};

/**
 */
static PyGetSetDef _buffers_getsets[] = {
    { "count", _buffers_getcount, NULL, NULL, NULL },
    { "buffers", _buffers_getbuffers, NULL, NULL, NULL },
    { NULL, NULL, NULL, NULL, NULL }
};

/**
 */
PyTypeObject PyBuffers_Type =
{
    TYPE_HEAD(NULL, 0)
    "base.Buffers",              /* tp_name */
    sizeof (PyBuffers),          /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor) _buffers_dealloc, /* tp_dealloc */
    0,                          /* tp_print */
    0,                          /* tp_getattr */
    0,                          /* tp_setattr */
    0,                          /* tp_compare */
    (reprfunc)_buffers_repr,     /* tp_repr */
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
    _buffers_methods,            /* tp_methods */
    0,                          /* tp_members */
    _buffers_getsets,            /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc) _buffers_init,   /* tp_init */
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
_buffers_dealloc (PyBuffers *self)
{
    int switched = 0;

    if (self->buffers != NULL)
    {
        ALCcontext *ctxt = alcGetCurrentContext ();
        if (ctxt != PyContext_AsContext (self->context))
        {
            /* Switch the current context to release the correct buffers. */
            alcMakeContextCurrent (PyContext_AsContext (self->context));
            switched = 1;
        }
        alDeleteBuffers (self->count, self->buffers);
        PyMem_Free (self->buffers);
        if (switched)
            alcMakeContextCurrent (ctxt);
    }
    self->buffers = NULL;
    self->count = 0;

    ((PyObject*)self)->ob_type->tp_free ((PyObject *) self);
}

static int
_buffers_init (PyObject *self, PyObject *args, PyObject *kwds)
{
    PyErr_SetString (PyExc_NotImplementedError,
        "Buffers cannot be created dirrectly - use the Context instead");
    return -1;
}

static PyObject*
_buffers_repr (PyObject *self)
{
    return Text_FromUTF8 ("<Buffers>");
}

/* Buffers getters/setters */
static PyObject*
_buffers_getcount (PyObject* self, void *closure)
{
    return PyInt_FromLong ((long)(((PyBuffers*)self)->count));
}

static PyObject*
_buffers_getbuffers (PyObject* self, void *closure)
{
    PyBuffers *buffers = (PyBuffers*)self;
    PyObject *tuple, *item;
    ALsizei i;

    tuple = PyTuple_New ((Py_ssize_t)(buffers->count));
    if (!tuple)
        return NULL;

    for (i = 0; i < buffers->count; i++)
    {
        item = PyInt_FromLong ((long)buffers->buffers[i]);
        if (!item)
        {
            Py_DECREF (tuple);
            return NULL;
        }
        PyTuple_SET_ITEM (tuple, (Py_ssize_t)i, item);
    }
    return tuple;
}


/* Buffers methods */
static PyObject*
_buffers_setprop (PyObject *self, PyObject *args)
{
    long bufnum;
    ALenum param;
    PyObject *values;
    char *type;
    PropType ptype = INVALID;

    if (!CONTEXT_IS_CURRENT (((PyBuffers*)self)->context))
    {
        PyErr_SetString (PyExc_PyGameError, "buffer context is not current");
        return NULL;
    }

    if (!PyArg_ParseTuple (args, "llO|s:set_prop", &bufnum, &param, &values,
            &type))
        return NULL;

    if (bufnum < 0 || bufnum > ((PyBuffers*)self)->count)
    {
        PyErr_SetString (PyExc_ValueError, "buffer index out of range");
        return NULL;
    }

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
                alBuffer3i ((ALuint)bufnum, param, vals[0], vals[1], vals[2]);
            else
                alBufferiv ((ALuint)bufnum, param, vals);
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
                alBuffer3f ((ALuint)bufnum, param, vals[0], vals[1], vals[2]);
            else
                alBufferfv ((ALuint)bufnum, param, vals);
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
            alBufferi ((ALuint)bufnum, param, (ALint) ival);
            break;
        case FLOAT:
            if (!DoubleFromObj (values, &fval))
                return NULL;
            CLEAR_ERROR_STATE ();
            alBufferf ((ALuint)bufnum, param, (ALfloat) fval);
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
_buffers_getprop (PyObject *self, PyObject *args)
{
    long bufnum;
    ALenum param;
    char *type;
    int size = 0, cnt;
    PropType ptype = INVALID;

    if (!CONTEXT_IS_CURRENT (((PyBuffers*)self)->context))
    {
        PyErr_SetString (PyExc_PyGameError, "buffer context is not current");
        return NULL;
    }

    if (!PyArg_ParseTuple (args, "lls:get_prop", &bufnum, &param, &type))
    {
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "ll|si:get_prop", &bufnum, &param, &type,
            &size))
            return NULL;
        if (size <= 0)
        {
            PyErr_SetString (PyExc_ValueError, "size must not smaller than 0");
            return NULL;
        }
    }

    if (bufnum < 0 || bufnum > ((PyBuffers*)self)->count)
    {
        PyErr_SetString (PyExc_ValueError, "buffer index out of range");
        return NULL;
    }

    ptype = GetPropTypeFromStr (type);
    CLEAR_ERROR_STATE ();
    switch (ptype)
    {
    case INT:
    {
        ALint val;
        alGetBufferi ((ALuint)bufnum, param, &val);
        if (SetALErrorException (alGetError ()))
            return NULL;
        return PyLong_FromLong ((long)val);
    }
    case FLOAT:
    {
        ALfloat val;
        alGetBufferf ((ALuint)bufnum, param, &val);
        if (SetALErrorException (alGetError ()))
            return NULL;
        return PyFloat_FromDouble ((double)val);
    }
    case INT3:
    {
        ALint val[3];
        alGetBuffer3i ((ALuint)bufnum, param, &val[0], &val[1], &val[2]);
        if (SetALErrorException (alGetError ()))
            return NULL;
        return Py_BuildValue ("(lll)", (long)val[0], (long)val[1],
            (long)val[2]);
    }
    case FLOAT3:
    {
        ALfloat val[3];
        alGetBuffer3f ((ALuint)bufnum, param, &val[0], &val[1], &val[2]);
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
        alGetBufferiv ((ALuint)bufnum, param, val);
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
        alGetBufferfv ((ALuint)bufnum, param, val);
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

static PyObject*
_buffers_bufferdata (PyObject *self, PyObject *args)
{
    long bufnum;
    ALenum format;
    const char *buf;
    Py_ssize_t len;
    ALsizei freq;

    if (!CONTEXT_IS_CURRENT (((PyBuffers*)self)->context))
    {
        PyErr_SetString (PyExc_PyGameError, "buffer context is not current");
        return NULL;
    }

    if (!PyArg_ParseTuple (args, "lls#l", &bufnum, &format, &buf, &len, &freq))
        return NULL;
    
    /* TODO */
    if (bufnum < 0 || bufnum > ((PyBuffers*)self)->count)
    {
        PyErr_SetString (PyExc_ValueError, "buffer index out of range");
        return NULL;
    }
    CLEAR_ERROR_STATE ();
    alBufferData ((ALuint) bufnum, format, (const void*) buf, (ALsizei)len,
        freq);
    if (SetALErrorException (alGetError ()))
        return NULL;
    Py_RETURN_NONE;
}

/* C API */
PyObject*
PyBuffers_New (PyObject *context, ALsizei count)
{
    ALuint *buf;
    PyObject *buffers;

    if (!context || !PyContext_Check (context))
    {
        PyErr_SetString (PyExc_TypeError, "context is not a valid Context");
        return NULL;
    }

    if (count < 1)
    {
        PyErr_SetString (PyExc_ValueError, "cannot create less than 1 buffer");
        return NULL;
    }

    buffers = PyBuffers_Type.tp_new (&PyBuffers_Type, NULL, NULL);
    if (!buffers)
        return NULL;
    ((PyBuffers*)buffers)->context = context;
    ((PyBuffers*)buffers)->count = count;
    ((PyBuffers*)buffers)->buffers = NULL;
    
    buf = PyMem_New (ALuint, count);
    if (!buf)
    {
        Py_DECREF (buffers);
        return NULL;
    }
    CLEAR_ERROR_STATE ();
    alGenBuffers (count, buf);
    if (SetALErrorException (alGetError ()))
    {
        Py_DECREF (buffers);
        PyMem_Free (buf);
        return NULL;
    }

    ((PyBuffers*)buffers)->buffers = buf;
    return buffers;
}

void
buffers_export_capi (void **capi)
{
    capi[PYGAME_OPENALBUFFERS_FIRSTSLOT] = &PyBuffers_Type;
}
