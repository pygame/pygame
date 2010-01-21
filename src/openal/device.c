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
#define PYGAME_OPENALDEVICE_INTERNAL

#include "openalmod.h"
#include "pgopenal.h"

static int _device_init (PyObject *self, PyObject *args, PyObject *kwds);
static void _device_dealloc (PyDevice *self);
static PyObject* _device_repr (PyObject *self);

static PyObject* _device_hasextension (PyObject *self, PyObject *args);

static PyObject* _device_getname (PyObject* self, void *closure);
/**
 */
static PyMethodDef _device_methods[] = {
    { "has_extension", _device_hasextension, METH_VARARGS, NULL },
    { NULL, NULL, 0, NULL }
};

/**
 */
static PyGetSetDef _device_getsets[] = {
    { "name", _device_getname, NULL, NULL/*DOC_BASEDEVICE_NAME*/, NULL },
    { NULL, NULL, NULL, NULL, NULL }
};

/**
 */
PyTypeObject PyDevice_Type =
{
    TYPE_HEAD(NULL, 0)
    "base.Device",              /* tp_name */
    sizeof (PyDevice),          /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor) _device_dealloc, /* tp_dealloc */
    0,                          /* tp_print */
    0,                          /* tp_getattr */
    0,                          /* tp_setattr */
    0,                          /* tp_compare */
    (reprfunc)_device_repr,     /* tp_repr */
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
    _device_methods,            /* tp_methods */
    0,                          /* tp_members */
    _device_getsets,            /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc) _device_init,    /* tp_init */
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
_device_dealloc (PyDevice *self)
{
    if (self->device)
        alcCloseDevice (self->device);
    self->device = NULL;
    ((PyObject*)self)->ob_type->tp_free ((PyObject *) self);
}

static int
_device_init (PyObject *self, PyObject *args, PyObject *kwds)
{
    char *name = NULL;
    ALCdevice *device = NULL;
    
    if (!PyArg_ParseTuple (args, "|s", &name))
        return -1;
    
    device = alcOpenDevice (name);
    if (!device)
    {
        SetALErrorException (alGetError ());
        return -1;
    }
    return 0;
}

static PyObject*
_device_repr (PyObject *self)
{
    return Text_FromUTF8 ("<alcDevice>");
}

/* Device getters/setters */
static PyObject*
_device_getname (PyObject* self, void *closure)
{
    const ALCchar *name = alcGetString (PyDevice_AsDevice (self),
        ALC_DEVICE_SPECIFIER);
    if (!name)
    {
        SetALErrorException (alGetError ());
        return NULL;
    }
    return Text_FromUTF8 ((const char*)name);
}

/* Device methods */
static PyObject*
_device_hasextension (PyObject *self, PyObject *args)
{
    char *extname = NULL;
    ALCboolean present;
    
    if(!PyArg_ParseTuple (args, "s:has_extension", &extname))
        return NULL;
    
    present = alcIsExtensionPresent (PyDevice_AsDevice (self),
        (const ALchar*)extname);
    if (SetALErrorException (alGetError ()))
        return NULL;
    if (present == ALC_FALSE)
        Py_RETURN_FALSE;
    Py_RETURN_TRUE;
}

/* C API */
PyObject*
PyDevice_New (const char* name)
{
    ALCdevice *dev;
    PyObject *device = PyDevice_Type.tp_new (&PyDevice_Type, NULL, NULL);

    if (!device)
        return NULL;

    dev = alcOpenDevice (name);
    if (!dev)
    {
        SetALErrorException (alGetError ());
        Py_DECREF (device);
        return NULL;
    }
    ((PyDevice*)device)->device = dev;
    return device;
}

void
device_export_capi (void **capi)
{
    capi[PYGAME_OPENALDEVICE_FIRSTSLOT] = &PyDevice_Type;
    capi[PYGAME_OPENALDEVICE_FIRSTSLOT+1] = PyDevice_New;
}
