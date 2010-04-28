/*
  pygame - Python Game Library
  Copyright (C) 2007 Marcus von Appen

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
#define PYGAME_BUFFERPROXY_INTERNAL


#include "internals.h"
#include "pgbase.h"
#include "base_doc.h"

#include <structmember.h> /* Somehow offsetof's not recognized in this file */

static PyObject* _bufferproxy_new (PyTypeObject *type, PyObject *args,
    PyObject *kwds);
static void _bufferproxy_dealloc (PyBufferProxy *self);
static PyObject* _bufferproxy_get_dict (PyBufferProxy *self, void *closure);
static PyObject* _bufferproxy_get_raw (PyBufferProxy *self, void *closure);
static PyObject* _bufferproxy_get_length (PyBufferProxy *self, void *closure);
static PyObject* _bufferproxy_repr (PyBufferProxy *self);
static PyObject* _bufferproxy_write (PyBufferProxy *buffer, PyObject *args);

/* Buffer methods */
#ifndef IS_PYTHON_3
static Py_ssize_t _bufferproxy_getreadbuf (PyBufferProxy *buffer,
    Py_ssize_t _index, const void **ptr);
static Py_ssize_t _bufferproxy_getwritebuf (PyBufferProxy *buffer,
    Py_ssize_t _index, const void **ptr);
static Py_ssize_t _bufferproxy_getsegcount (PyBufferProxy *buffer,
    Py_ssize_t *lenp);
#else
static int _bufferproxy_getbuffer (PyBufferProxy *self, Py_buffer *view,
    int flags);
static void _bufferproxy_releasebuffer (PyBufferProxy *self, Py_buffer *view);
#endif

/* C API interfaces */
static PyObject* PyBufferProxy_New (PyObject *object, void *buffer,
    Py_ssize_t length, bufferunlock_func unlock_func);

/**
 * Methods, which are bound to the PyBufferProxy type.
 */
static PyMethodDef _bufferproxy_methods[] =
{
    { "write", (PyCFunction) _bufferproxy_write, METH_VARARGS,
      DOC_BASE_BUFFERPROXY_WRITE},
    { NULL, NULL, 0, NULL }
};

/**
 * Getters and setters for the PyBufferProxy.
 */
static PyGetSetDef _bufferproxy_getsets[] =
{
    { "__dict__", (getter) _bufferproxy_get_dict, NULL, NULL, NULL },
    { "raw", (getter) _bufferproxy_get_raw, NULL, DOC_BASE_BUFFERPROXY_RAW,
      NULL },
    { "length", (getter) _bufferproxy_get_length, NULL,
      DOC_BASE_BUFFERPROXY_LENGTH, NULL },
    { NULL, NULL, NULL, NULL, NULL }
};

/**
 * Buffer interface support for the PyBufferProxy.
 */
#ifndef IS_PYTHON_3
static PyBufferProcs _bufferproxy_as_buffer =
{
    (readbufferproc) _bufferproxy_getreadbuf,
    (writebufferproc) _bufferproxy_getwritebuf,
    (segcountproc) _bufferproxy_getsegcount,
    NULL,
#if PY_VERSION_HEX >= 0x02060000
    NULL,
    NULL
#endif
};
#else /* IS_PYTHON_3 */
static PyBufferProcs _bufferproxy_as_buffer =
{
    (getbufferproc) _bufferproxy_getbuffer,
    (releasebufferproc) _bufferproxy_releasebuffer
};
#endif /* IS_PYTHON_3 */

PyTypeObject PyBufferProxy_Type =
{
    TYPE_HEAD(NULL,0)
    "base.BufferProxy",       /* tp_name */
    sizeof (PyBufferProxy),     /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor) _bufferproxy_dealloc, /* tp_dealloc */
    0,                          /* tp_print */
    0,                          /* tp_getattr */
    0,                          /* tp_setattr */
    0,                          /* tp_compare */
    (reprfunc) &_bufferproxy_repr,          /* tp_repr */
    0,                          /* tp_as_number */
    0,                          /* tp_as_sequence */
    0,                          /* tp_as_mapping */
    0,                          /* tp_hash */
    0,                          /* tp_call */
    0,                          /* tp_str */
    0,                          /* tp_getattro */
    0,                          /* tp_setattro */
    &_bufferproxy_as_buffer,    /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    DOC_BASE_BUFFERPROXY,
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    offsetof (PyBufferProxy, weakrefs),  /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    _bufferproxy_methods,       /* tp_methods */
    0,                          /* tp_members */
    _bufferproxy_getsets,       /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    offsetof (PyBufferProxy, dict), /* tp_dictoffset */
    0,                          /* tp_init */
    0,                          /* tp_alloc */
    _bufferproxy_new,           /* tp_new */
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

/**
 * Creates a new PyBufferProxy.
 */
static PyObject*
_bufferproxy_new (PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyBufferProxy *self = (PyBufferProxy *) type->tp_alloc (type, 0);
    if (!self)
        return NULL;

    self->object = NULL;
    self->weakrefs = NULL;
    self->dict = NULL;
    self->buffer = NULL;
    self->unlock_func = NULL;
    return (PyObject *) self;
}

/**
 * Deallocates the PyBufferProxy and its members.
 */
static void
_bufferproxy_dealloc (PyBufferProxy *self)
{
    if (self->unlock_func)
        (*self->unlock_func) (self->object, (PyObject*)self);
    else
        Py_XDECREF (self->object);

    if (self->weakrefs)
        PyObject_ClearWeakRefs ((PyObject *) self);
    Py_XDECREF (self->dict);
    ((PyObject*)self)->ob_type->tp_free ((PyObject *) self);
}

/* Getters/Setters */

/**
 * Getter for PyBufferProxy.__dict__.
 */
static PyObject*
_bufferproxy_get_dict (PyBufferProxy *self, void *closure)
{
    if (!self->dict)
    {
	self->dict = PyDict_New ();
        if (!self->dict)
            return NULL;
    }

    Py_INCREF (self->dict);
    return self->dict;
}

/**
 * Getter for PyBufferProxy.raw.
 */
static PyObject*
_bufferproxy_get_raw (PyBufferProxy *self, void *closure)
{
    return Bytes_FromStringAndSize (self->buffer, self->length);
}

/**
 * Getter for PyBufferProxy.length
 */
static PyObject*
_bufferproxy_get_length (PyBufferProxy *self, void *closure)
{
    return PyInt_FromLong (self->length);
}

/* Methods */

/**
 * Representation method.
 */
static PyObject*
_bufferproxy_repr (PyBufferProxy *self)
{
    /* zd is for Py_size_t which python < 2.5 doesn't have. */
#if PY_VERSION_HEX < 0x02050000
    return PyString_FromFormat ("<BufferProxy(%d)>", self->length);
#else
    return Text_FromFormat ("<BufferProxy(%zd)>", self->length);
#endif
}

/**
 * Writes raw data to the buffer.
 */
static PyObject*
_bufferproxy_write (PyBufferProxy *buffer, PyObject *args)
{
    Py_ssize_t offset;
    Py_ssize_t length;
    char *buf;

#ifdef IS_PYTHON_24
    if (!PyArg_ParseTuple (args, "s#i", &buf, &length, &offset))
        return NULL;
#else
    if (!PyArg_ParseTuple (args, "s#n", &buf, &length, &offset))
        return NULL;
#endif

    if (offset + length > buffer->length)
    {
        PyErr_SetString (PyExc_IndexError, "bytes to write exceed buffer size");
        return NULL;
    }

    memcpy (((char *)buffer->buffer) + offset, buf, (size_t) length);

    Py_RETURN_NONE;
}   

/*Buffer interfaces */
#ifndef IS_PYTHON_3
static Py_ssize_t
_bufferproxy_getreadbuf (PyBufferProxy *buffer, Py_ssize_t _index,
    const void **ptr)
{
    if (_index != 0)
    {
        PyErr_SetString (PyExc_SystemError,
            "Accessing non-existent buffer segment");
        return -1;
    }

    *ptr = buffer->buffer;
    return buffer->length;
}

static Py_ssize_t
_bufferproxy_getwritebuf (PyBufferProxy *buffer, Py_ssize_t _index,
    const void **ptr)
{
    if (_index != 0)
    {
        PyErr_SetString (PyExc_SystemError, 
            "Accessing non-existent array segment");
        return -1;
    }

    *ptr = buffer->buffer;
    return buffer->length;
}

static Py_ssize_t
_bufferproxy_getsegcount (PyBufferProxy *buffer, Py_ssize_t *lenp)
{
    if (lenp)
        *lenp = buffer->length;
    return 1;
}

#else /* !IS_PYTHON_3 */
static int
_bufferproxy_getbuffer (PyBufferProxy *self, Py_buffer *view, int flags)
{
    if (!view)
        return 0;
    Py_INCREF (self); /* Guarantee that the object does not get destroyed */
    return PyBuffer_FillInfo (view, (PyObject*)self, self->buffer,
        self->length, 0, flags);
}

static void
_bufferproxy_releasebuffer (PyBufferProxy *self, Py_buffer *view)
{
    Py_DECREF (self);
}
#endif /* IS_PYTHON_3 */

/* C API */
static PyObject*
PyBufferProxy_New (PyObject *object, void *buffer, Py_ssize_t length,
    bufferunlock_func unlock_func)
{
    PyBufferProxy *buf = (PyBufferProxy*) PyBufferProxy_Type.tp_new
        (&PyBufferProxy_Type, NULL, NULL);
    if (!buf)
        return NULL;

    buf->object = object;
    buf->buffer = buffer;
    buf->length = length;
    buf->unlock_func = unlock_func;
    return (PyObject *) buf;
}

void
bufferproxy_export_capi (void **capi)
{
    capi[PYGAME_BUFFERPROXY_FIRSTSLOT] = &PyBufferProxy_Type;
    capi[PYGAME_BUFFERPROXY_FIRSTSLOT+1] = (void *)PyBufferProxy_New;
}
