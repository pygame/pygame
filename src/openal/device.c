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
#include "openalbase_doc.h"

static int _device_init (PyObject *self, PyObject *args, PyObject *kwds);
static void _device_dealloc (PyDevice *self);
static PyObject* _device_repr (PyObject *self);

static PyObject* _device_hasextension (PyObject *self, PyObject *args);
static PyObject* _device_geterror (PyObject* self);
static PyObject* _device_getenumvalue (PyObject *self, PyObject *args);

static PyObject* _device_getname (PyObject* self, void *closure);
static PyObject* _device_getextensions (PyObject *self, void *closure);

/**
 */
static PyMethodDef _device_methods[] = {
    { "has_extension", _device_hasextension, METH_VARARGS,
      DOC_BASE_DEVICE_HAS_EXTENSION },
    { "get_error", (PyCFunction)_device_geterror, METH_NOARGS,
      DOC_BASE_DEVICE_GET_ERROR },
    { "get_enum_value", _device_getenumvalue, METH_O,
      DOC_BASE_DEVICE_GET_ENUM_VALUE },
    { NULL, NULL, 0, NULL }
};

/**
 */
static PyGetSetDef _device_getsets[] = {
    { "name", _device_getname, NULL, DOC_BASE_DEVICE_NAME, NULL },
    { "extensions", _device_getextensions, NULL, DOC_BASE_DEVICE_EXTENSIONS,
      NULL },
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
    DOC_BASE_DEVICE,
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
    {
        Py_BEGIN_ALLOW_THREADS;
        alcCloseDevice (self->device);
        Py_END_ALLOW_THREADS;
    }
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
    
    CLEAR_ALCERROR_STATE ();
    Py_BEGIN_ALLOW_THREADS;
    device = alcOpenDevice (name);
    Py_END_ALLOW_THREADS;
    if (!device)
    {
        SetALCErrorException (alcGetError (NULL), 1);
        return -1;
    }
    ((PyDevice*)self)->device = device;
    return 0;
}

static PyObject*
_device_repr (PyObject *self)
{
    PyObject *retval;
    const ALCchar *name;
    size_t len;
    char *str;
    
    CLEAR_ALCERROR_STATE ();
    name = alcGetString (PyDevice_AsDevice (self), ALC_DEVICE_SPECIFIER);
    if (!name)
    {
        SetALCErrorException (alcGetError (PyDevice_AsDevice (self)), 1);
        return NULL;
    }

    /* Device('') == 10 */
    len = strlen ((const char*) name) + 11;
    str = malloc (len);
    if (!str)
        return NULL;

    snprintf (str, len, "Device('%s')", (const char*) name);
    retval = Text_FromUTF8 (str);
    free (str);
    return retval;
}

/* Device getters/setters */
static PyObject*
_device_getname (PyObject* self, void *closure)
{
    const ALCchar *name;

    CLEAR_ALCERROR_STATE ();
    name = alcGetString (PyDevice_AsDevice (self), ALC_DEVICE_SPECIFIER);
    if (!name)
    {
        SetALCErrorException (alcGetError (PyDevice_AsDevice (self)), 1);
        return NULL;
    }
    return Text_FromUTF8 ((const char*)name);
}

static PyObject*
_device_getextensions (PyObject *self, void *closure)
{
    ALCchar tmp[4096] = { '\0' };
    int i = 0;
    const ALCchar *dptr;
    PyObject *list, *item;
    const ALCchar *devices;

    CLEAR_ALCERROR_STATE ();
    devices = alcGetString (PyDevice_AsDevice (self), ALC_DEVICE_SPECIFIER);
    if (!devices)
    {
        SetALCErrorException (alcGetError (PyDevice_AsDevice (self)), 1);
        return NULL;
    }
    
    list = PyList_New (0);
    dptr = devices;
    while (*dptr)
    {
        if (*dptr == ' ') /* list entry end */
        {
            item = Text_FromUTF8 ((const char*) tmp);
            if (!item)
            {
                Py_DECREF (list);
                return NULL;
            }
            if (PyList_Append (list, item) == -1)
            {
                Py_DECREF (item);
                Py_DECREF (list);
                return NULL;
            }
            Py_DECREF (item);
            memset (tmp, '\0', (size_t)4096);
            i = 0;
        }
        else if (i < 4096)
        {
            tmp[i] = *dptr;
        }
        dptr++;
        i++;
    }
    /* Sentinel append */
    if (i != 0)
    {
        item = Text_FromUTF8 ((const char*) tmp);
        if (!item)
        {
            Py_DECREF (list);
            return NULL;
        }
        if (PyList_Append (list, item) == -1)
        {
            Py_DECREF (item);
            Py_DECREF (list);
            return NULL;
        }
        Py_DECREF (item);
    }
    return list;
}

/* Device methods */
static PyObject*
_device_hasextension (PyObject *self, PyObject *args)
{
    char *extname = NULL;
    ALCboolean present;
    
    if(!PyArg_ParseTuple (args, "s:has_extension", &extname))
        return NULL;
    
    CLEAR_ALCERROR_STATE ();
    present = alcIsExtensionPresent (PyDevice_AsDevice (self),
        (const ALchar*)extname);
    if (SetALCErrorException (alcGetError (PyDevice_AsDevice (self)), 0))
        return NULL;
    if (present == ALC_FALSE)
        Py_RETURN_FALSE;
    Py_RETURN_TRUE;
}

static PyObject*
_device_geterror (PyObject* self)
{
    ALCenum error = alcGetError (PyDevice_AsDevice (self));
    switch (error)
    {
    case ALC_INVALID_ENUM:
        return Text_FromUTF8 ("invalid enumeration value");
    case ALC_INVALID_VALUE:
        return Text_FromUTF8 ("invalid value");
    case ALC_INVALID_DEVICE:
        return Text_FromUTF8 ("invalid device");
    case ALC_INVALID_CONTEXT:
        return Text_FromUTF8 ("invalid context");
    case ALC_OUT_OF_MEMORY:
        return Text_FromUTF8 ("insufficient memory");
    default:
        Py_RETURN_NONE;
    }
}

static PyObject*
_device_getenumvalue (PyObject *self, PyObject *args)
{
    ALCenum val;
    PyObject *freeme;
    char *enumname;
    
    if (!ASCIIFromObj (args, &enumname, &freeme))
        return NULL;
    CLEAR_ALCERROR_STATE ();
    val = alcGetEnumValue (PyDevice_AsDevice (self), (const ALchar*)enumname);
    Py_XDECREF (freeme);
    if (SetALCErrorException (alcGetError (PyDevice_AsDevice (self)), 0))
        return NULL;
    return PyInt_FromLong ((long)val);
}

/* C API */
PyObject*
PyDevice_New (const char* name)
{
    ALCdevice *dev;
    PyObject *device = PyDevice_Type.tp_new (&PyDevice_Type, NULL, NULL);

    if (!device)
        return NULL;

    CLEAR_ALCERROR_STATE ();
    dev = alcOpenDevice (name);
    if (!dev)
    {
        SetALCErrorException (alcGetError (NULL), 1);
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
    capi[PYGAME_OPENALDEVICE_FIRSTSLOT+1] = (void *)PyDevice_New;
}
