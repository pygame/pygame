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
#define PYGAME_OPENALSOURCES_INTERNAL

#include "openalmod.h"
#include "pgopenal.h"

static int _sources_init (PyObject *self, PyObject *args, PyObject *kwds);
static void _sources_dealloc (PySources *self);
static PyObject* _sources_repr (PyObject *self);

typedef enum
{
    PLAY,
    PAUSE,
    STOP,
    REWIND
} SourcesAction;
static PyObject* _sources_action (PyObject *self, PyObject *args,
    SourcesAction action);

static PyObject* _sources_setprop (PyObject *self, PyObject *args);
static PyObject* _sources_getprop (PyObject *self, PyObject *args);
static PyObject* _sources_play (PyObject *self, PyObject *args);
static PyObject* _sources_pause (PyObject *self, PyObject *args);
static PyObject* _sources_stop (PyObject *self, PyObject *args);
static PyObject* _sources_rewind (PyObject *self, PyObject *args);
static PyObject* _sources_queuebuffers (PyObject *self, PyObject *args);
static PyObject* _sources_unqueuebuffers (PyObject *self, PyObject *args);

static PyObject* _sources_getcount (PyObject* self, void *closure);
static PyObject* _sources_getsources (PyObject* self, void *closure);

/**
 */
static PyMethodDef _sources_methods[] = {
    { "set_prop", _sources_setprop, METH_VARARGS, NULL },
    { "get_prop", _sources_getprop, METH_VARARGS, NULL },
    { "play", _sources_play, METH_O, NULL },
    { "pause", _sources_pause, METH_O, NULL },
    { "stop", _sources_stop, METH_O, NULL },
    { "rewind", _sources_rewind, METH_O, NULL },
    { "queue_buffers", _sources_queuebuffers, METH_VARARGS, NULL },
    { "unqueue_buffers", _sources_unqueuebuffers, METH_VARARGS, NULL },
    { NULL, NULL, 0, NULL }
};

/**
 */
static PyGetSetDef _sources_getsets[] = {
    { "count", _sources_getcount, NULL, NULL, NULL },
    { "sources", _sources_getsources, NULL, NULL, NULL },
    { NULL, NULL, NULL, NULL, NULL }
};

/**
 */
PyTypeObject PySources_Type =
{
    TYPE_HEAD(NULL, 0)
    "base.Sources",              /* tp_name */
    sizeof (PySources),          /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor) _sources_dealloc, /* tp_dealloc */
    0,                          /* tp_print */
    0,                          /* tp_getattr */
    0,                          /* tp_setattr */
    0,                          /* tp_compare */
    (reprfunc)_sources_repr,     /* tp_repr */
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
    _sources_methods,            /* tp_methods */
    0,                          /* tp_members */
    _sources_getsets,            /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc) _sources_init,   /* tp_init */
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
_sources_dealloc (PySources *self)
{
    int switched = 0;

    if (self->sources != NULL)
    {
        ALCcontext *ctxt = alcGetCurrentContext ();
        if (ctxt != PyContext_AsContext (self->context))
        {
            /* Switch the current context to release the correct buffers. */
            alcMakeContextCurrent (PyContext_AsContext (self->context));
            switched = 1;
        }
        alDeleteSources (self->count, self->sources);
        PyMem_Free (self->sources);
        if (switched)
            alcMakeContextCurrent (ctxt);
    }
    self->sources = NULL;
    self->count = 0;

    ((PyObject*)self)->ob_type->tp_free ((PyObject *) self);
}

static int
_sources_init (PyObject *self, PyObject *args, PyObject *kwds)
{
    PyErr_SetString (PyExc_NotImplementedError,
        "Sources cannot be created dirrectly - use the Context instead");
    return -1;
}

static PyObject*
_sources_repr (PyObject *self)
{
    return Text_FromUTF8 ("<Sources>");
}

/* Sources getters/setters */
static PyObject*
_sources_getcount (PyObject* self, void *closure)
{
    return PyInt_FromLong ((long)(((PySources*)self)->count));
}

static PyObject*
_sources_getsources (PyObject* self, void *closure)
{
    PySources *sources = (PySources*)self;
    PyObject *tuple, *item;
    ALsizei i;

    tuple = PyTuple_New ((Py_ssize_t)(sources->count));
    if (!tuple)
        return NULL;

    for (i = 0; i < sources->count; i++)
    {
        item = PyInt_FromLong ((long)sources->sources[i]);
        if (!item)
        {
            Py_DECREF (tuple);
            return NULL;
        }
        PyTuple_SET_ITEM (tuple, (Py_ssize_t)i, item);
    }
    return tuple;
}

/* Sources methods */
static PyObject*
_sources_setprop (PyObject *self, PyObject *args)
{
    long bufnum;
    ALenum param;
    PyObject *values;
    char *type;
    PropType ptype = INVALID;

    if (!CONTEXT_IS_CURRENT (((PySources*)self)->context))
    {
        PyErr_SetString (PyExc_PyGameError, "source context is not current");
        return NULL;
    }

    if (!PyArg_ParseTuple (args, "llO|s:set_prop", &bufnum, &param, &values,
            &type))
        return NULL;

    if (bufnum < 0 || bufnum > ((PySources*)self)->count)
    {
        PyErr_SetString (PyExc_ValueError, "source index out of range");
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

            CLEAR_ALERROR_STATE ();
            if (ptype == INT3)
                alSource3i ((ALuint)bufnum, param, vals[0], vals[1], vals[2]);
            else
                alSourceiv ((ALuint)bufnum, param, vals);
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

            CLEAR_ALERROR_STATE ();
            if (ptype == FLOAT3)
                alSource3f ((ALuint)bufnum, param, vals[0], vals[1], vals[2]);
            else
                alSourcefv ((ALuint)bufnum, param, vals);
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
            CLEAR_ALERROR_STATE ();
            alSourcei ((ALuint)bufnum, param, (ALint) ival);
            break;
        case FLOAT:
            if (!DoubleFromObj (values, &fval))
                return NULL;
            CLEAR_ALERROR_STATE ();
            alSourcef ((ALuint)bufnum, param, (ALfloat) fval);
            break;
        default:
            PyErr_SetString (PyExc_TypeError, "value type mismatch");
            return NULL;
        }
    }

    if (SetALErrorException (alGetError (), 0))
        return NULL;
    Py_RETURN_NONE;
}

static PyObject*
_sources_getprop (PyObject *self, PyObject *args)
{
    long bufnum;
    ALenum param;
    char *type;
    int size = 0, cnt;
    PropType ptype = INVALID;

    if (!CONTEXT_IS_CURRENT (((PySources*)self)->context))
    {
        PyErr_SetString (PyExc_PyGameError, "source context is not current");
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

    if (bufnum < 0 || bufnum > ((PySources*)self)->count)
    {
        PyErr_SetString (PyExc_ValueError, "source index out of range");
        return NULL;
    }

    ptype = GetPropTypeFromStr (type);
    CLEAR_ALERROR_STATE ();
    switch (ptype)
    {
    case INT:
    {
        ALint val;
        alGetSourcei ((ALuint)bufnum, param, &val);
        if (SetALErrorException (alGetError (), 0))
            return NULL;
        return PyLong_FromLong ((long)val);
    }
    case FLOAT:
    {
        ALfloat val;
        alGetSourcef ((ALuint)bufnum, param, &val);
        if (SetALErrorException (alGetError (), 0))
            return NULL;
        return PyFloat_FromDouble ((double)val);
    }
    case INT3:
    {
        ALint val[3];
        alGetSource3i ((ALuint)bufnum, param, &val[0], &val[1], &val[2]);
        if (SetALErrorException (alGetError (), 0))
            return NULL;
        return Py_BuildValue ("(lll)", (long)val[0], (long)val[1],
            (long)val[2]);
    }
    case FLOAT3:
    {
        ALfloat val[3];
        alGetSource3f ((ALuint)bufnum, param, &val[0], &val[1], &val[2]);
        if (SetALErrorException (alGetError (), 0))
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
        alGetSourceiv ((ALuint)bufnum, param, val);
        if (SetALErrorException (alGetError (), 0))
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
        alGetSourcefv ((ALuint)bufnum, param, val);
        if (SetALErrorException (alGetError (), 0))
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
_sources_action (PyObject *self, PyObject *args, SourcesAction action)
{
    unsigned int source;
    
    if (!CONTEXT_IS_CURRENT (((PySources*)self)->context))
    {
        PyErr_SetString (PyExc_PyGameError, "source context is not current");
        return NULL;
    }

    if (PySequence_Check (args))
    {
        ALuint *sources;
        Py_ssize_t i;
        Py_ssize_t len = PySequence_Size (args);

        if (len > ((PySources*)self)->count)
        {
            PyErr_SetString (PyExc_ValueError,
                "sequence size exceeds the available sources");
            return NULL;
        }
        sources = PyMem_New (ALuint, (ALsizei) len);
        if (!sources)
            return NULL;
        for (i = 0; i < len; i++)
        {
            if (!UintFromSeqIndex (args, i, &source))
            {
                PyMem_Free (sources);
                return NULL;
            }
            sources[i] = source;
        }
        CLEAR_ALERROR_STATE ();
        switch (action)
        {
        case PLAY:
            alSourcePlayv ((ALsizei) len, sources);
            break;
        case PAUSE:
            alSourcePausev ((ALsizei) len, sources);
            break;
        case STOP:
            alSourceStopv ((ALsizei) len, sources);
            break;
        case REWIND:
            alSourceRewindv ((ALsizei) len, sources);
            break;
        default:
            break;
        }
        PyMem_Free (sources);
        if (SetALErrorException (alGetError (), 0))
            return NULL;
        Py_RETURN_NONE;
    }
    else if (UintFromObj (args, &source))
    {
        CLEAR_ALERROR_STATE ();
        switch (action)
        {
        case PLAY:
            alSourcePlay ((ALuint)source);
            break;
        case PAUSE:
            alSourcePause ((ALuint)source);
            break;
        case STOP:
            alSourceStop ((ALuint)source);
            break;
        case REWIND:
            alSourceRewind ((ALuint)source);
            break;
        }
        if (SetALErrorException (alGetError (), 0))
            return NULL;
        Py_RETURN_NONE;
    }

    PyErr_SetString (PyExc_TypeError,
        "argument must be a sequence or positive integer");
    return NULL;
}

static PyObject*
_sources_play (PyObject *self, PyObject *args)
{
    return _sources_action (self, args, PLAY);
}

static PyObject*
_sources_pause (PyObject *self, PyObject *args)
{
    return _sources_action (self, args, PAUSE);
}

static PyObject*
_sources_stop (PyObject *self, PyObject *args)
{
    return _sources_action (self, args, STOP);
}

static PyObject*
_sources_rewind (PyObject *self, PyObject *args)
{
    return _sources_action (self, args, REWIND);
}

static PyObject*
_sources_queuebuffers (PyObject *self, PyObject *args)
{
    PyObject *buffers;
    long bufnum;
    
    if (!CONTEXT_IS_CURRENT (((PySources*)self)->context))
    {
        PyErr_SetString (PyExc_PyGameError, "source context is not current");
        return NULL;
    }
    
    if (!PyArg_ParseTuple (args, "lO:queue_buffers", &bufnum, &buffers))
        return NULL;
    
    if (bufnum < 0 || bufnum > ((PySources*)self)->count)
    {
        PyErr_SetString (PyExc_ValueError, "source index out of range");
        return NULL;
    }
    if (!PyBuffers_Check (buffers))
    {
        PyErr_SetString (PyExc_TypeError, "argument must be a Buffers object");
        return NULL;
    }
    
    CLEAR_ALERROR_STATE ();
    alSourceQueueBuffers ((ALuint) bufnum, ((PyBuffers*)buffers)->count,
        PyBuffers_AsBuffers(buffers));
    if (SetALErrorException (alGetError (), 0))
        return NULL;
    
    Py_RETURN_NONE;
}

static PyObject*
_sources_unqueuebuffers (PyObject *self, PyObject *args)
{
    PyObject *buffers;
    long bufnum;
    
    if (!CONTEXT_IS_CURRENT (((PySources*)self)->context))
    {
        PyErr_SetString (PyExc_PyGameError, "source context is not current");
        return NULL;
    }
    
    if (!PyArg_ParseTuple (args, "lO:unqueue_buffers", &bufnum, &buffers))
        return NULL;
    
    if (bufnum < 0 || bufnum > ((PySources*)self)->count)
    {
        PyErr_SetString (PyExc_ValueError, "source index out of range");
        return NULL;
    }
    if (!PyBuffers_Check (buffers))
    {
        PyErr_SetString (PyExc_TypeError, "argument must be a Buffers object");
        return NULL;
    }
    
    CLEAR_ALERROR_STATE ();
    alSourceUnqueueBuffers ((ALuint) bufnum, ((PyBuffers*)buffers)->count,
        PyBuffers_AsBuffers(buffers));
    if (SetALErrorException (alGetError (), 0))
        return NULL;
    
    Py_RETURN_NONE;
}
/* C API */
PyObject*
PySources_New (PyObject *context, ALsizei count)
{
    ALuint *buf;
    PyObject *sources;

    if (!context || !PyContext_Check (context))
    {
        PyErr_SetString (PyExc_TypeError, "context is not a valid Context");
        return NULL;
    }

    if (count < 1)
    {
        PyErr_SetString (PyExc_ValueError, "cannot create less than 1 sources");
        return NULL;
    }

    sources = PySources_Type.tp_new (&PySources_Type, NULL, NULL);
    if (!sources)
        return NULL;
    ((PySources*)sources)->context = context;
    ((PySources*)sources)->count = count;
    ((PySources*)sources)->sources = NULL;
    
    buf = PyMem_New (ALuint, count);
    if (!buf)
    {
        Py_DECREF (sources);
        return NULL;
    }
    CLEAR_ALERROR_STATE ();
    alGenSources (count, buf);
    if (SetALErrorException (alGetError (), 0))
    {
        Py_DECREF (sources);
        PyMem_Free (buf);
        return NULL;
    }

    ((PySources*)sources)->sources = buf;
    return sources;
}

void
sources_export_capi (void **capi)
{
    capi[PYGAME_OPENALSOURCES_FIRSTSLOT] = &PySources_Type;
}
