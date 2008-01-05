/*
  pygame - Python Game Library
  Copyright (C) 2007  Marcus von Appen

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

#define PYGAMEAPI_BUFFERPROXY_INTERNAL

#include "pygame.h"
#include "pygamedocs.h"

typedef struct
{
    PyObject_HEAD
    PyObject *dict;     /* dict for subclassing */
    PyObject *weakrefs; /* Weakrefs for subclassing */
    void *buffer;       /* Pointer to the buffer of the parent object. */
    Py_ssize_t length;  /* Length of the buffer. */
    PyObject *parent;   /* Parent object associated with this object. */
    PyObject *lock;     /* Lock object for the surface. */

} PyBufferProxy;

static PyObject* _bufferproxy_new (PyTypeObject *type, PyObject *args,
                                   PyObject *kwds);
static void _bufferproxy_dealloc (PyBufferProxy *self);
static PyObject* _bufferproxy_get_dict (PyBufferProxy *self, void *closure);
static PyObject* _bufferproxy_get_raw (PyBufferProxy *buffer, void *closure);
static PyObject* _bufferproxy_repr (PyBufferProxy *self);
static PyObject* _bufferproxy_write (PyBufferProxy *buffer, PyObject *args);

/* Buffer methods */
static Py_ssize_t _bufferproxy_getreadbuf (PyBufferProxy *buffer,
                                           Py_ssize_t _index,
                                           const void **ptr);
static Py_ssize_t _bufferproxy_getwritebuf (PyBufferProxy *buffer,
                                            Py_ssize_t _index,
                                            const void **ptr);
static Py_ssize_t _bufferproxy_getsegcount (PyBufferProxy *buffer,
                                            Py_ssize_t *lenp);

/* C API interfaces */
static PyObject* PyBufferProxy_New (PyObject *parent, void *buffer,
                                    Py_ssize_t length, PyObject *lock);

/**
 * Methods, which are bound to the PyBufferProxy type.
 */
static PyMethodDef _bufferproxy_methods[] =
{
    { "write", (PyCFunction) _bufferproxy_write, METH_VARARGS,
      "Writes raw data to the buffer" },
    { NULL, NULL, 0, NULL }
};

/**
 * Getters and setters for the PyBufferProxy.
 */
static PyGetSetDef _bufferproxy_getsets[] =
{
    { "__dict__", (getter) _bufferproxy_get_dict, NULL, NULL, NULL },
    { "raw", (getter) _bufferproxy_get_raw, NULL,
      "The raw buffer data as string", NULL },
    { NULL, NULL, NULL, NULL, NULL }
};


/**
 * Buffer interface support for the PyBufferProxy.
 */
static PyBufferProcs _bufferproxy_as_buffer =
{
        (readbufferproc) _bufferproxy_getreadbuf,
        (writebufferproc) _bufferproxy_getwritebuf,
        (segcountproc) _bufferproxy_getsegcount,
        NULL,
};

static PyTypeObject PyBufferProxy_Type =
{
    PyObject_HEAD_INIT(NULL)
    0,
    "BufferProxy",              /* tp_name */
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
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_WEAKREFS,
    "",                         /* tp_doc */
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
    0                           /* tp_del */
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

    self->weakrefs = NULL;
    self->dict = NULL;
    self->parent = NULL;
    self->buffer = NULL;
    self->lock = NULL;
    return (PyObject *) self;
}

/**
 * Deallocates the PyBufferProxy and its members.
 */
static void
_bufferproxy_dealloc (PyBufferProxy *self)
{
    if (self->weakrefs)
        PyObject_ClearWeakRefs ((PyObject *) self);
    if (self->lock)
    {
        Py_DECREF (self->lock);
    }
    Py_XDECREF (self->dict);
    self->ob_type->tp_free ((PyObject *) self);
}

/**** Getter and setter access ****/

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
_bufferproxy_get_raw (PyBufferProxy *buffer, void *closure)
{
    return PyString_FromStringAndSize (buffer->buffer, buffer->length);
}

/**** Methods ****/

/**
 * Representation method.
 */
static PyObject*
_bufferproxy_repr (PyBufferProxy *self)
{
    /* zd is for Py_size_t which python < 2.5 doesn't have. */
#if PY_VERSION_HEX < 0x02050000
    return PyString_FromFormat("<BufferProxy(%d)>", self->length);
#else
    return PyString_FromFormat("<BufferProxy(%zd)>", self->length);
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

    if (!PyArg_ParseTuple (args, "s#i", &buf, &length, &offset))
        return NULL;

    if (offset + length > buffer->length)
    {
        return RAISE (PyExc_IndexError, "bytes to write exceed buffer size");
    }

    memcpy (buffer->buffer + offset, buf, (size_t) length);

    Py_RETURN_NONE;
}   

/**** Buffer interfaces ****/

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

    if (!buffer->parent)
    {
        *ptr = NULL;
        return 0;
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

    if (!buffer->parent)
    {
        *ptr = NULL;
        return 0;
    }

    *ptr = buffer->buffer;
    return buffer->length;
}

static Py_ssize_t
_bufferproxy_getsegcount (PyBufferProxy *buffer, Py_ssize_t *lenp)
{
    if (!buffer->parent)
    {
        *lenp = 0;
        return 0;
    }

    if (lenp)
        *lenp = buffer->length;
    return 1;
}

static PyObject*
PyBufferProxy_New (PyObject *parent, void *buffer, Py_ssize_t length,
                   PyObject *lock)
{
    PyBufferProxy *buf;

    buf = (PyBufferProxy *) _bufferproxy_new (&PyBufferProxy_Type, NULL, NULL);

    if (!buf)
        return NULL;
    buf->buffer = buffer;
    buf->length = length;
    buf->lock = lock;
    buf->parent = parent;
    return (PyObject *) buf;
}

PYGAME_EXPORT
void initbufferproxy (void)
{
    PyObject *module;
    PyObject *dict;
    PyObject *apiobj;
    static void* c_api[PYGAMEAPI_BUFFERPROXY_NUMSLOTS];

    if (PyType_Ready (&PyBufferProxy_Type) < 0)
        return;

    /* create the module */
    module = Py_InitModule3 ("bufferproxy", NULL,
        "A generic proxy module that can spend arbitrary objects a buffer " \
        "interface");
    PyBufferProxy_Type.tp_getattro = PyObject_GenericGetAttr;
    Py_INCREF (&PyBufferProxy_Type);
    PyModule_AddObject (module, "BufferProxy", (PyObject *)&PyBufferProxy_Type);
    dict = PyModule_GetDict (module);

    c_api[0] = &PyBufferProxy_Type;
    c_api[1] = PyBufferProxy_New;
    apiobj = PyCObject_FromVoidPtr (c_api, NULL);
    PyDict_SetItemString (dict, PYGAMEAPI_LOCAL_ENTRY, apiobj);
    Py_DECREF (apiobj);


}
