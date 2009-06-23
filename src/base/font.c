/*
  pygame - Python Game Library
  Copyright (C) 2000-2001 Pete Shinners
  Copyright (C) 2008 Marcus von Appen
  Copyright (C) 2009 Vicent Marti

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

#define PYGAME_FONT_INTERNAL

#include "internals.h"
#include "pgbase.h"
#include "base_doc.h"

static PyObject* _font_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
static int _font_init(PyObject *cursor, PyObject *args, PyObject *kwds);
static void _font_dealloc(PyFont *self);
static PyObject* _font_repr(PyObject *self);


/*
 * Get/set attributes
 */
static PyObject* _font_getheight(PyObject *self, void *closure);
static PyObject* _font_getname(PyObject *self, void *closure);
static PyObject* _font_getstyle(PyObject *self, void *closure);
static int       _font_setstyle(PyObject *self, PyObject *arg, void *closure);

/*
 * Class methods
 */
static PyObject* _font_getsize(PyObject *self, PyObject *args, PyObject *kwds);
static PyObject* _font_render(PyObject* self, PyObject *args, PyObject *kwds);
static PyObject* _font_copy(PyObject* self);

/*
 * Get/set attributes (default fallbacks)
 */
static PyObject* _def_f_getheight(PyObject *self, void *closure);
static PyObject* _def_f_getname(PyObject *self, void *closure);
static PyObject* _def_f_getstyle(PyObject *self, void *closure);
static int       _def_f_setstyle(PyObject *self, PyObject *arg, void *closure);

/*
 * Class methods (default fallbacks)
 */
static PyObject* _def_f_getsize(PyObject *self, PyObject *args, PyObject *kwds);
static PyObject* _def_f_render (PyObject *self, PyObject *args, PyObject *kwds); 
static PyObject* _def_f_copy (PyObject *self); 


/*
 * METHODS TABLE
 */
static PyMethodDef _font_methods[] = 
{
    {
        "render", 
        (PyCFunction) _font_render, 
        METH_VARARGS | METH_KEYWORDS,
        DOC_BASE_FONT_RENDER 
    },
    {
        "get_size", 
        (PyCFunction) _font_getsize, 
        METH_VARARGS | METH_KEYWORDS,
        DOC_BASE_FONT_SIZE 
    },
    {
        "copy", 
        (PyCFunction) _font_copy, 
        METH_NOARGS, 
        DOC_BASE_FONT_COPY
    },
    { NULL, NULL, 0, NULL }
};

/*
 * GET/SETS TABLE
 */
static PyGetSetDef _font_getsets[] = {
    { "height", _font_getheight,    NULL, DOC_BASE_FONT_HEIGHT, NULL },
    { "name",   _font_getname,      NULL, DOC_BASE_FONT_NAME,   NULL },
    { "style",  _font_getstyle,     _font_setstyle, DOC_BASE_FONT_STYLE, NULL },
    { NULL, NULL, NULL, NULL, NULL }
};


/*
 * TYPE OBJECT TABLE
 */
PyTypeObject PyFont_Type =
{
    TYPE_HEAD(NULL, 0)
    "base.Font",                /* tp_name */
    sizeof (PyFont),            /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor) _font_dealloc, /* tp_dealloc */
    0,                          /* tp_print */
    0,                          /* tp_getattr */
    0,                          /* tp_setattr */
    0,                          /* tp_compare */
    (reprfunc)_font_repr,       /* tp_repr */
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
    DOC_BASE_FONT,              /* docstring */
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    _font_methods,              /* tp_methods */
    0,                          /* tp_members */
    _font_getsets,              /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc) _font_init,      /* tp_init */
    0,                          /* tp_alloc */
    _font_new,                  /* tp_new */
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
_font_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyFont *font = (PyFont *)type->tp_alloc(type, 0);

    if (!font)
        return NULL;

    font->get_height =  _def_f_getheight;
    font->get_size =    _def_f_getsize;
    font->render =      _def_f_render;
    font->copy =        _def_f_copy;
    font->get_name =    _def_f_getname;
    font->get_style =   _def_f_getstyle;
    font->set_style =   _def_f_setstyle;

    return (PyObject*) font;
}

static void
_font_dealloc(PyFont *self)
{
    ((PyObject*)self)->ob_type->tp_free((PyObject *)self);
}

static int
_font_init(PyObject *self, PyObject *args, PyObject *kwds)
{
    return 0;
}

static PyObject*
_font_repr(PyObject *self)
{
    return Text_FromUTF8("<Generic Font>");
}

/* Font getters/setters */
static PyObject*
_font_getheight(PyObject *self, void *closure)
{
    if (((PyFont *)self)->get_height &&
        ((PyFont *)self)->get_height != _def_f_getheight)
        return ((PyFont *)self)->get_height(self, closure);

    PyErr_SetString(PyExc_NotImplementedError,
        "height attribute not implemented");

    return NULL;
}


static PyObject*
_font_getname(PyObject *self, void *closure)
{
    if (((PyFont *)self)->get_name &&
        ((PyFont *)self)->get_name != _def_f_getname)
        return ((PyFont *)self)->get_name(self, closure);

    PyErr_SetString (PyExc_NotImplementedError,
        "name attribute not implemented");

    return NULL;
}

static PyObject*
_font_getstyle(PyObject *self, void *closure)
{
    if (((PyFont *)self)->get_style &&
        ((PyFont *)self)->get_style != _def_f_getstyle)
        return ((PyFont *)self)->get_style(self, closure);

    PyErr_SetString (PyExc_NotImplementedError,
        "style attribute not implemented");

    return NULL;
}

static int
_font_setstyle(PyObject *self, PyObject *args, void *closure)
{
    if (((PyFont *)self)->set_style &&
        ((PyFont *)self)->set_style != _def_f_setstyle)
        return ((PyFont *)self)->set_style(self, args, closure);

    PyErr_SetString (PyExc_NotImplementedError,
        "style attribute not implemented");

    return -1;
}



/* Font methods */
static PyObject*
_font_render(PyObject* self, PyObject *args, PyObject *kwds)
{
    if (((PyFont *)self)->render && ((PyFont *)self)->render != _def_f_render)
        return ((PyFont *)self)->render(self, args, kwds);

    PyErr_SetString(PyExc_NotImplementedError, "render method not implemented");

    return NULL;
}

static PyObject*
_font_copy(PyObject* self)
{
    if (((PyFont *)self)->copy && ((PyFont *)self)->copy != _def_f_copy)
        return ((PyFont *)self)->copy (self);

    PyErr_SetString(PyExc_NotImplementedError, "copy method not implemented");

    return NULL;
}

static PyObject*
_font_getsize(PyObject *self, PyObject *args, PyObject *kwds)
{
    if (((PyFont *)self)->get_size &&
        ((PyFont *)self)->get_size != _def_f_getsize)
        return ((PyFont *)self)->get_size(self, args, kwds);

    PyErr_SetString (PyExc_NotImplementedError,
        "size attribute not implemented");

    return NULL;
}


/*
 * Default fallbacks 
 */

static PyObject*
_def_f_getheight(PyObject *self, void *closure)
{
    return PyObject_GetAttrString(self, "height");
}

static PyObject*
_def_f_getname(PyObject *self, void *closure)
{
    return PyObject_GetAttrString(self, "name");
}

static PyObject*
_def_f_getstyle(PyObject *self, void *closure)
{
    return PyObject_GetAttrString(self, "style");
}

static int
_def_f_setstyle(PyObject *self, PyObject *arg, void *closure)
{
    return PyObject_SetAttrString(self, "style", arg);
}

static PyObject*
_def_f_render(PyObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *retval, *method;
    
    method = PyObject_GetAttrString(self, "render");

    if (!method)
        return NULL;

    retval = PyObject_Call(method, args, kwds);
    Py_DECREF (method);
    return retval;
}

static PyObject*
_def_f_copy(PyObject *self)
{
    return PyObject_CallMethod(self, "copy", NULL, NULL);
}

static PyObject*
_def_f_getsize(PyObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *retval, *method;
    
    method = PyObject_GetAttrString(self, "getsize");

    if (!method)
        return NULL;

    retval = PyObject_Call(method, args, kwds);
    Py_DECREF (method);
    return retval;
}


/* C API */
PyObject*
PyFont_New(void)
{
    return PyFont_Type.tp_new(&PyFont_Type, NULL, NULL);
}

void
font_export_capi(void **capi)
{
    capi[PYGAME_FONT_FIRSTSLOT + 0] = &PyFont_Type;
    capi[PYGAME_FONT_FIRSTSLOT + 1] = PyFont_New;
}
