/*
  pygame - Python Game Library
  Copyright (C) 2009 Marcus von Appen

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
#define PYGAME_SURFACE_INTERNAL

#include "internals.h"
#include "pgbase.h"
#include "base_doc.h"

static PyObject* _surface_new (PyTypeObject *type, PyObject *args,
    PyObject *kwds);
static int _surface_init (PyObject *cursor, PyObject *args, PyObject *kwds);
static void _surface_dealloc (PySurface *self);
static PyObject* _surface_repr (PyObject *self);

static PyObject* _surface_getwidth (PyObject *self, void *closure);
static PyObject* _surface_getheight (PyObject *self, void *closure);
static PyObject* _surface_getsize (PyObject *self, void *closure);
static PyObject* _surface_getpixels (PyObject *self, void *closure);

static PyObject* _surface_blit (PyObject* self, PyObject *args, PyObject *kwds);
static PyObject* _surface_copy (PyObject* self);

/**
 */
static PyMethodDef _surface_methods[] = {
    { "blit", (PyCFunction) _surface_blit, METH_VARARGS | METH_KEYWORDS,
      DOC_BASE_SURFACE_BLIT },
    { "copy", (PyCFunction) _surface_copy, METH_NOARGS, DOC_BASE_SURFACE_COPY },
    { NULL, NULL, 0, NULL }
};

/**
 */
static PyGetSetDef _surface_getsets[] = {
    { "pixels", _surface_getpixels, NULL, DOC_BASE_SURFACE_PIXELS, NULL },
    { "width", _surface_getwidth, NULL, DOC_BASE_SURFACE_WIDTH, NULL },
    { "height", _surface_getheight, NULL, DOC_BASE_SURFACE_HEIGHT, NULL },
    { "size", _surface_getsize, NULL, DOC_BASE_SURFACE_SIZE, NULL },

    { NULL, NULL, NULL, NULL, NULL }
};

/**
 */
PyTypeObject PySurface_Type =
{
    TYPE_HEAD(NULL, 0)
    "base.Surface",             /* tp_name */
    sizeof (PySurface),         /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor) _surface_dealloc, /* tp_dealloc */
    0,                          /* tp_print */
    0,                          /* tp_getattr */
    0,                          /* tp_setattr */
    0,                          /* tp_compare */
    (reprfunc)_surface_repr,    /* tp_repr */
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
    DOC_BASE_SURFACE,
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    _surface_methods,           /* tp_methods */
    0,                          /* tp_members */
    _surface_getsets,           /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc) _surface_init,   /* tp_init */
    0,                          /* tp_alloc */
    _surface_new,               /* tp_new */
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
_surface_new (PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PySurface* sf = (PySurface*) type->tp_alloc (type, 0);
    if (!sf)
        return NULL;
    sf->get_width = NULL;
    sf->get_height = NULL;
    sf->get_size = NULL;
    sf->get_pixels = NULL;
    sf->get_size = NULL;
    return (PyObject*) sf;
}

static void
_surface_dealloc (PySurface *self)
{
    ((PyObject*)self)->ob_type->tp_free ((PyObject *) self);
}

static int
_surface_init (PyObject *self, PyObject *args, PyObject *kwds)
{
    return 0;
}

static PyObject*
_surface_repr (PyObject *self)
{
    return Text_FromUTF8 ("Surface");
}

/* Surface getters/setters */
static PyObject*
_surface_getwidth (PyObject *self, void *closure)
{
    return ((PySurface*)self)->get_width (self, closure);
}

static PyObject*
_surface_getheight (PyObject *self, void *closure)
{
    return ((PySurface*)self)->get_height (self, closure);
}

static PyObject*
_surface_getsize (PyObject *self, void *closure)
{
    return ((PySurface*)self)->get_size (self, closure);
}

static PyObject*
_surface_getpixels (PyObject *self, void *closure)
{
    return ((PySurface*)self)->get_pixels (self, closure);
}

/* Surface methods */
static PyObject*
_surface_blit (PyObject* self, PyObject *args, PyObject *kwds)
{
    return ((PySurface*)self)->blit (self, args, kwds);
}

static PyObject*
_surface_copy (PyObject* self)
{
    return ((PySurface*)self)->copy (self);
}

/* C API */
PyObject*
PySurface_New (void)
{
    return PySurface_Type.tp_new (&PySurface_Type, NULL, NULL);
}

void
surface_export_capi (void **capi)
{
    capi[PYGAME_SURFACE_FIRSTSLOT] = &PySurface_Type;
    capi[PYGAME_SURFACE_FIRSTSLOT+1] = PySurface_New;
}
