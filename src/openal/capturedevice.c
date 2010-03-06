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
#define PYGAME_OPENALCAPTUREDEVICE_INTERNAL

#include "openalmod.h"
#include "pgopenal.h"

static PyObject* _capturedevice_new (PyTypeObject *type, PyObject *args,
    PyObject *kwds);
static int _capturedevice_init (PyObject *self, PyObject *args, PyObject *kwds);
static void _capturedevice_dealloc (PyCaptureDevice *self);
static PyObject* _capturedevice_repr (PyObject *self);

static PyObject* _capturedevice_start (PyObject* self);
static PyObject* _capturedevice_stop (PyObject* self);
static PyObject* _capturedevice_getsamples (PyObject* self, PyObject *args);

static PyObject* _capturedevice_getsize (PyObject *self, void *closure);
static PyObject* _capturedevice_getfrequency (PyObject *self, void *closure);
static PyObject* _capturedevice_getformat (PyObject *self, void *closure);

/**
 */
static PyMethodDef _capturedevice_methods[] = {
    { "start", (PyCFunction) _capturedevice_start, METH_NOARGS, NULL },
    { "stop", (PyCFunction) _capturedevice_stop, METH_NOARGS, NULL },
    { "get_samples", (PyCFunction) _capturedevice_getsamples, METH_VARARGS, NULL },
    { NULL, NULL, 0, NULL }
};

/**
 */
static PyGetSetDef _capturedevice_getsets[] = {
    { "size", _capturedevice_getsize, NULL, NULL, NULL },
    { "frequency", _capturedevice_getfrequency, NULL, NULL, NULL },
    { "format", _capturedevice_getformat, NULL, NULL, NULL },
    { NULL, NULL, NULL, NULL, NULL }
};

/**
 */
PyTypeObject PyCaptureDevice_Type =
{
    TYPE_HEAD(NULL, 0)
    "base.CaptureDevice",       /* tp_name */
    sizeof (PyCaptureDevice),   /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor) _capturedevice_dealloc, /* tp_dealloc */
    0,                          /* tp_print */
    0,                          /* tp_getattr */
    0,                          /* tp_setattr */
    0,                          /* tp_compare */
    (reprfunc)_capturedevice_repr,     /* tp_repr */
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
    _capturedevice_methods,     /* tp_methods */
    0,                          /* tp_members */
    _capturedevice_getsets,     /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc) _capturedevice_init,    /* tp_init */
    0,                          /* tp_alloc */
    _capturedevice_new,         /* tp_new */
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

static PyObject*
_capturedevice_new (PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyCaptureDevice *device = (PyCaptureDevice *)type->tp_alloc (type, 0);
    if (!device)
        return NULL;
    device->size = 0;
    device->format = 0;
    device->frequency = 0;
    
    device->device.device = NULL;
    return (PyObject*) device;
}

static void
_capturedevice_dealloc (PyCaptureDevice *self)
{
    if (self->device.device)
    {
        Py_BEGIN_ALLOW_THREADS;
        alcCaptureCloseDevice (self->device.device);
        Py_END_ALLOW_THREADS;
    }
    self->device.device = NULL;
    ((PyObject*)self)->ob_type->tp_free ((PyObject *) self);
}

static int
_capturedevice_init (PyObject *self, PyObject *args, PyObject *kwds)
{
    char *name = NULL;
    ALCdevice *device = NULL;
    long bufsize, freq, format;
    
    if (!PyArg_ParseTuple (args, "slll", &name, &freq, &format, &bufsize))
    {
        name = NULL;
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "lll", &freq, &format, &bufsize))
            return -1;
    }
    if (bufsize <= 0)
    {
        PyErr_SetString (PyExc_ValueError,
            "bufsize must not be smaller than 1");
        return -1;
    }

    CLEAR_ALCERROR_STATE ();
    device = alcCaptureOpenDevice ((const ALCchar*)name, (ALCuint) freq,
        (ALCenum)format, (ALCsizei) bufsize);
    if (!device)
    {
        SetALCErrorException (alcGetError (NULL), 1);
        return -1;
    }
    ((PyCaptureDevice*)self)->device.device = device;
    ((PyCaptureDevice*)self)->format = (ALCenum)format;
    ((PyCaptureDevice*)self)->frequency = (ALCuint) freq;
    ((PyCaptureDevice*)self)->size = (ALCsizei) bufsize;

    return 0;
}

static PyObject*
_capturedevice_repr (PyObject *self)
{
    PyObject *retval;
    const ALCchar *name;
    size_t len;
    char *str;
    
    CLEAR_ALCERROR_STATE ();
    name = alcGetString (PyCaptureDevice_AsDevice (self),
        ALC_CAPTURE_DEVICE_SPECIFIER);
    if (!name)
    {
        SetALCErrorException (alcGetError (PyCaptureDevice_AsDevice (self)), 1);
        return NULL;
    }        
    /* CaptureDevice('') == 17 */
    len = strlen ((const char*) name) + 18;
    str = malloc (len);
    if (!str)
        return NULL;

    snprintf (str, len, "CaptureDevice('%s')", (const char*) name);
    retval = Text_FromUTF8 (str);
    free (str);
    return retval;
}

/* CaptureDevice getters/setters */
static PyObject*
_capturedevice_getsize (PyObject *self, void *closure)
{
    return PyInt_FromLong ((long) ((PyCaptureDevice*)self)->size);
}

static PyObject*
_capturedevice_getfrequency (PyObject *self, void *closure)
{
    return PyInt_FromLong ((long) ((PyCaptureDevice*)self)->frequency);
}

static PyObject*
_capturedevice_getformat (PyObject *self, void *closure)
{
    return PyInt_FromLong ((long) ((PyCaptureDevice*)self)->format);
}

/* CaptureDevice methods */
static PyObject*
_capturedevice_start (PyObject* self)
{
    CLEAR_ALCERROR_STATE ();
    alcCaptureStart (PyCaptureDevice_AsDevice(self));
    if (SetALCErrorException (alcGetError (PyCaptureDevice_AsDevice(self)), 0))
        return NULL;
    Py_RETURN_NONE;
}

static PyObject*
_capturedevice_stop (PyObject* self)
{
    CLEAR_ALCERROR_STATE ();
    alcCaptureStop (PyCaptureDevice_AsDevice(self));
    if (SetALCErrorException (alcGetError (PyCaptureDevice_AsDevice(self)), 0))
        return NULL;
    Py_RETURN_NONE;
}

static PyObject*
_capturedevice_getsamples (PyObject* self, PyObject *args)
{
    PyObject *buffer = NULL;
    long offset = 0;
    ALCvoid *buf;
    ALCsizei count;
    
    if (!PyArg_ParseTuple (args, "|Ol:get_samples", &buffer, &offset))
        return NULL;
    
    CLEAR_ALCERROR_STATE ();
    alcGetIntegerv (PyCaptureDevice_AsDevice (self), ALC_CAPTURE_SAMPLES,
        (ALCsizei)(sizeof (ALCsizei)), &count);
    if (SetALCErrorException (alcGetError (PyCaptureDevice_AsDevice(self)), 0))
        return NULL;
    if (count == 0)
    {
        Py_RETURN_NONE;
    }
    
    alcCaptureSamples (PyCaptureDevice_AsDevice (self), &buf, count);
    if (SetALCErrorException (alcGetError (PyCaptureDevice_AsDevice(self)), 0))
        return NULL;
    Py_RETURN_NONE;
}

/* C API */
PyObject*
PyCaptureDevice_New (const char* name, ALCuint frequency, ALCenum format,
    ALCsizei bufsize)
{
    ALCdevice *dev;
    PyObject *device = PyCaptureDevice_Type.tp_new (&PyCaptureDevice_Type,
        NULL, NULL);

    if (!device)
        return NULL;

    CLEAR_ALCERROR_STATE ();
    dev = alcCaptureOpenDevice (name, frequency, format, bufsize);
    if (!dev)
    {
        SetALErrorException (alGetError (), 1);
        Py_DECREF (device);
        return NULL;
    }
    ((PyCaptureDevice*)device)->device.device = dev;
    return device;
}

void
capturedevice_export_capi (void **capi)
{
    capi[PYGAME_OPENALDEVICE_FIRSTSLOT] = &PyCaptureDevice_Type;
    capi[PYGAME_OPENALDEVICE_FIRSTSLOT+1] = (void *)PyCaptureDevice_New;
}
