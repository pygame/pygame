/*
  pygame - Python Game Library
  Copyright (C) 2008 Marcus von Appen

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
#define PYGAMEAPI_COLOR_INTERNAL

#include "pygamedocs.h"
#include "pygame.h"
#include "pgcompat.h"
#include <ctype.h>

typedef struct
{
    PyObject_HEAD
    /* RGBA */
    Uint8 r;
    Uint8 g;
    Uint8 b;
    Uint8 a;
    Uint8 len;
} PyColor;

typedef enum {
    TRISTATE_SUCCESS,
    TRISTATE_FAIL,
    TRISTATE_ERROR
} tristate;

static PyObject *_COLORDICT = NULL;

static int _get_double (PyObject *obj, double *val);
static int _get_color (PyObject *val, Uint32 *color);
static int _hextoint (char *hex, Uint8 *val);
static tristate _hexcolor (PyObject *color, Uint8 rgba[]);
static int _coerce_obj(PyObject *obj, Uint8 rgba[]);

static PyColor* _color_new_internal (PyTypeObject *type, Uint8 rgba[]);
static PyColor* _color_new_internal_length (PyTypeObject *type, Uint8 rgba[], Uint8 length);

static PyObject* _color_new (PyTypeObject *type, PyObject *args,
    PyObject *kwds);
static void _color_dealloc (PyColor *color);
static PyObject* _color_repr (PyColor *color);
static PyObject* _color_normalize (PyColor *color);
static PyObject* _color_correct_gamma (PyColor *color, PyObject *args);
static PyObject* _color_set_length (PyColor *color, PyObject *args);

/* Getters/setters */
static PyObject* _color_get_r (PyColor *color, void *closure);
static int _color_set_r (PyColor *color, PyObject *value, void *closure);
static PyObject* _color_get_g (PyColor *color, void *closure);
static int _color_set_g (PyColor *color, PyObject *value, void *closure);
static PyObject* _color_get_b (PyColor *color, void *closure);
static int _color_set_b (PyColor *color, PyObject *value, void *closure);
static PyObject* _color_get_a (PyColor *color, void *closure);
static int _color_set_a (PyColor *color, PyObject *value, void *closure);
static PyObject* _color_get_hsva (PyColor *color, void *closure);
static int _color_set_hsva (PyColor *color, PyObject *value, void *closure);
static PyObject* _color_get_hsla (PyColor *color, void *closure);
static int _color_set_hsla (PyColor *color, PyObject *value, void *closure);
static PyObject* _color_get_i1i2i3 (PyColor *color, void *closure);
static int _color_set_i1i2i3 (PyColor *color, PyObject *value, void *closure);
static PyObject* _color_get_cmy (PyColor *color, void *closure);
static int _color_set_cmy (PyColor *color, PyObject *value, void *closure);

/* Number protocol methods */
static PyObject* _color_add (PyColor *color1, PyColor *color2);
static PyObject* _color_sub (PyColor *color1, PyColor *color2);
static PyObject* _color_mul (PyColor *color1, PyColor *color2);
static PyObject* _color_div (PyColor *color1, PyColor *color2);
static PyObject* _color_mod (PyColor *color1, PyColor *color2);
static PyObject* _color_inv (PyColor *color);
#if !PY3
static int _color_coerce (PyObject **pv, PyObject **pw);
#endif
static PyObject* _color_int (PyColor *color);
static PyObject* _color_long (PyColor *color);
static PyObject* _color_float (PyColor *color);
#if !PY3
static PyObject* _color_oct (PyColor *color);
static PyObject* _color_hex (PyColor *color);
#endif

/* Sequence protocol methods */
static Py_ssize_t _color_length (PyColor *color);
static PyObject* _color_item (PyColor *color, Py_ssize_t _index);
static int _color_ass_item (PyColor *color, Py_ssize_t _index, PyObject *value);
static PyObject * _color_slice(register PyColor *a, 
                               register Py_ssize_t ilow, 
                               register Py_ssize_t ihigh);

/* Mapping protocol methods. */
static PyObject * _color_subscript(PyColor* self, PyObject* item);


/* Comparison */
static PyObject* _color_richcompare(PyObject *o1, PyObject *o2, int opid);

/* C API interfaces */
static PyObject* PyColor_New (Uint8 rgba[]);
static PyObject* PyColor_NewLength (Uint8 rgba[], Uint8 length);
static int RGBAFromColorObj (PyObject *color, Uint8 rgba[]);

/**
 * Methods, which are bound to the PyColor type.
 */
static PyMethodDef _color_methods[] =
{
    { "normalize", (PyCFunction) _color_normalize, METH_NOARGS,
      DOC_COLORNORMALIZE },
    { "correct_gamma", (PyCFunction) _color_correct_gamma, METH_VARARGS,
      DOC_COLORCORRECTGAMMA },
    { "set_length", (PyCFunction) _color_set_length, METH_VARARGS,
      DOC_COLORSETLENGTH },
    { NULL, NULL, 0, NULL }
};

/**
 * Getters and setters for the PyColor.
 */
static PyGetSetDef _color_getsets[] =
{
    { "r", (getter) _color_get_r, (setter) _color_set_r, DOC_COLORR, NULL },
    { "g", (getter) _color_get_g, (setter) _color_set_g, DOC_COLORG, NULL },
    { "b", (getter) _color_get_b, (setter) _color_set_b, DOC_COLORB, NULL },
    { "a", (getter) _color_get_a, (setter) _color_set_a, DOC_COLORA, NULL },
    { "hsva", (getter) _color_get_hsva, (setter) _color_set_hsva, DOC_COLORHSVA,
      NULL },
    { "hsla", (getter) _color_get_hsla, (setter) _color_set_hsla, DOC_COLORHSLA,
      NULL },
    { "i1i2i3", (getter) _color_get_i1i2i3, (setter) _color_set_i1i2i3,
      DOC_COLORI1I2I3, NULL },
    { "cmy", (getter) _color_get_cmy, (setter) _color_set_cmy, DOC_COLORCMY,
      NULL },
    { NULL, NULL, NULL, NULL, NULL }
};


static PyNumberMethods _color_as_number =
{
    (binaryfunc) _color_add, /* nb_add */
    (binaryfunc) _color_sub, /* nb_subtract */
    (binaryfunc) _color_mul, /* nb_multiply */
#if !PY3
    (binaryfunc) _color_div, /* nb_divide */
#endif
    (binaryfunc) _color_mod, /* nb_remainder */
    0,                       /* nb_divmod */
    0,                       /* nb_power */
    0,                       /* nb_negative */
    0,                       /* nb_positive */
    0,                       /* nb_absolute */
    0,                       /* nb_nonzero / nb_bool*/
    (unaryfunc) _color_inv,  /* nb_invert */
    0,                       /* nb_lshift */
    0,                       /* nb_rshift */
    0,                       /* nb_and */
    0,                       /* nb_xor */
    0,                       /* nb_or */
#if !PY3
    _color_coerce,           /* nb_coerce */
#endif
    (unaryfunc) _color_int,  /* nb_int */
    (unaryfunc) _color_long, /* nb_long */
    (unaryfunc) _color_float,/* nb_float */
#if !PY3
    (unaryfunc) _color_oct,  /* nb_oct */
    (unaryfunc) _color_hex,  /* nb_hex */
#endif
    0,                       /* nb_inplace_add */
    0,                       /* nb_inplace_subtract */
    0,                       /* nb_inplace_multiply */
#if !PY3
    0,                       /* nb_inplace_divide */
#endif
    0,                       /* nb_inplace_remainder */
    0,                       /* nb_inplace_power */
    0,                       /* nb_inplace_lshift */
    0,                       /* nb_inplace_rshift */
    0,                       /* nb_inplace_and */
    0,                       /* nb_inplace_xor */
    0,                       /* nb_inplace_or */
    (binaryfunc) _color_div, /* nb_floor_divide */
    0,                       /* nb_true_divide */
    0,                       /* nb_inplace_floor_divide */
    0,                       /* nb_inplace_true_divide */
#if PY_VERSION_HEX >= 0x02050000
    (unaryfunc) _color_int,  /* nb_index */
#endif
};

/**
 * Sequence interface support for PyColor.
 */
static PySequenceMethods _color_as_sequence =
{
    (lenfunc) _color_length,           /* sq_length */
    NULL,                              /* sq_concat */
    NULL,                              /* sq_repeat */
    (ssizeargfunc) _color_item,        /* sq_item */
    (ssizessizeargfunc)_color_slice,    /* sq_slice */
    (ssizeobjargproc) _color_ass_item, /* sq_ass_item */
    NULL, /* sq_ass_slice */
    NULL,                              /* sq_contains */
    NULL,                              /* sq_inplace_concat */
    NULL,                              /* sq_inplace_repeat */
};



static PyMappingMethods _color_as_mapping = {
        (lenfunc) _color_length,
        (binaryfunc)_color_subscript,
        NULL
};







#define DEFERRED_ADDRESS(ADDR) 0

static PyTypeObject PyColor_Type =
{
    TYPE_HEAD (NULL, 0)
    "pygame.Color",             /* tp_name */
    sizeof (PyColor),           /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor) _color_dealloc,/* tp_dealloc */
    0,                          /* tp_print */
    0,                          /* tp_getattr */
    0,                          /* tp_setattr */
    0,                          /* tp_compare */
    (reprfunc) _color_repr,     /* tp_repr */
    &_color_as_number,          /* tp_as_number */
    &_color_as_sequence,        /* tp_as_sequence */
    &_color_as_mapping,          /* tp_as_mapping */
    0,                          /* tp_hash */
    0,                          /* tp_call */
    0,                          /* tp_str */
    0,                          /* tp_getattro */
    0,                          /* tp_setattro */
    0,                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    DOC_PYGAMECOLOR,            /* tp_doc */
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    _color_richcompare,         /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    _color_methods,             /* tp_methods */
    0,                          /* tp_members */
    _color_getsets,             /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    0,                          /* tp_init */
    0,                          /* tp_alloc */
    _color_new,                 /* tp_new */
#ifndef __SYMBIAN32__
    0,                          /* tp_free */
    0,                          /* tp_is_gc */
    0,                          /* tp_bases */
    0,                          /* tp_mro */
    0,                          /* tp_cache */
    0,                          /* tp_subclasses */
    0,                          /* tp_weaklist */
    0                           /* tp_del */
#endif    
};

#define PyColor_Check(o) \
    ((o)->ob_type == (PyTypeObject *) &PyColor_Type)

#define RGB_EQUALS(x,y)                          \
    ((((PyColor *)x)->r == ((PyColor *)y)->r) && \
     (((PyColor *)x)->g == ((PyColor *)y)->g) && \
     (((PyColor *)x)->b == ((PyColor *)y)->b) && \
     (((PyColor *)x)->a == ((PyColor *)y)->a))

static int
_get_double (PyObject *obj, double *val)
{
    PyObject *floatobj;
    if (!(floatobj = PyNumber_Float (obj)))
        return 0;
    *val = PyFloat_AsDouble (floatobj);
    Py_DECREF (floatobj);
    return 1;
}

static int
_get_color (PyObject *val, Uint32 *color)
{
    if (!val || !color)
        return 0;

#if !PY3
    if (PyInt_Check (val))
    {
        long intval = PyInt_AsLong (val);
        if (intval == -1 && PyErr_Occurred ())
        {
            PyErr_SetString (PyExc_ValueError, "invalid color argument");
            return 0;
        }
        *color = (Uint32) intval;
        return 1;
    }
#endif
    if (PyLong_Check (val))
    {
        unsigned long longval = PyLong_AsUnsignedLong (val);
        if (PyErr_Occurred ())
        {
            PyErr_SetString (PyExc_ValueError, "invalid color argument");
            return 0;
        }
        *color = (Uint32) longval;
        return 1;
    }
    
    /* Failed */
    PyErr_SetString (PyExc_ValueError, "invalid color argument");
    return 0;
}

static int
_hextoint (char *hex, Uint8 *val)
{
    /* 'hex' is a two digit hexadecimal number, no spaces, no signs.
     * This algorithm is brute force, but it is character system agnostic.
     * It is definitely not a general purpose solution.
     */
    Uint8 temp = 0;

    switch (toupper(hex[0]))
    {
    case '0':
        break;
    case '1':
        temp += 0x10;
        break;
    case '2':
        temp += 0x20;
        break;
    case '3':
        temp += 0x30;
        break;
    case '4':
        temp += 0x40;
        break;
    case '5':
        temp += 0x50;
        break;
    case '6':
        temp += 0x60;
        break;
    case '7':
        temp += 0x70;
        break;
    case '8':
        temp += 0x80;
        break;
    case '9':
        temp += 0x90;
        break;
    case 'A':
        temp += 0xA0;
        break;
    case 'B':
        temp += 0xB0;
        break;
    case 'C':
        temp += 0xC0;
        break;
    case 'D':
        temp += 0xD0;
        break;
    case 'E':
        temp += 0xE0;
        break;
    case 'F':
        temp += 0xF0;
        break;
    default:
        return 0;
    }

    switch (toupper(hex[1]))
    {
    case '0':
        break;
    case '1':
        temp += 0x01;
        break;
    case '2':
        temp += 0x02;
        break;
    case '3':
        temp += 0x03;
        break;
    case '4':
        temp += 0x04;
        break;
    case '5':
        temp += 0x05;
        break;
    case '6':
        temp += 0x06;
        break;
    case '7':
        temp += 0x07;
        break;
    case '8':
        temp += 0x08;
        break;
    case '9':
        temp += 0x09;
        break;
    case 'A':
        temp += 0x0A;
        break;
    case 'B':
        temp += 0x0B;
        break;
    case 'C':
        temp += 0x0C;
        break;
    case 'D':
        temp += 0x0D;
        break;
    case 'E':
        temp += 0x0E;
        break;
    case 'F':
        temp += 0x0F;
        break;
    default:
        return 0;
    }

    *val = temp;
    return 1;
}

static tristate
_hexcolor (PyObject *color, Uint8 rgba[])
{
    size_t len;
    tristate rcode = TRISTATE_FAIL;
    char *name;
#if PY3
    PyObject* ascii = PyUnicode_AsASCIIString (color);
    if (ascii == NULL) {
        rcode = TRISTATE_ERROR;
        goto Fail;
    }
    name = PyBytes_AsString (ascii);
#else
    name = PyString_AsString (color);
#endif
    if (name == NULL) {
        goto Fail;
    }

    len = strlen (name);
    /* hex colors can be
     * #RRGGBB
     * #RRGGBBAA
     * 0xRRGGBB
     * 0xRRGGBBAA
     */
    if (len < 7) {
        goto Fail;
    }

    if (name[0] == '#')
    {
        if (len != 7 && len != 9)
            goto Fail;
        if (!_hextoint (name + 1, &rgba[0]))
            goto Fail;
        if (!_hextoint (name + 3, &rgba[1]))
            goto Fail;
        if (!_hextoint (name + 5, &rgba[2]))
            goto Fail;
        rgba[3] = 255;
        if (len == 9 && !_hextoint (name + 7, &rgba[3]))
            goto Fail;
        goto Success;
    }
    else if (name[0] == '0' && name[1] == 'x')
    {
        if (len != 8 && len != 10)
            goto Fail;
        if (!_hextoint (name + 2, &rgba[0]))
            goto Fail;
        if (!_hextoint (name + 4, &rgba[1]))
            goto Fail;
        if (!_hextoint (name + 6, &rgba[2]))
            goto Fail;
        rgba[3] = 255;
        if (len == 10 && !_hextoint (name + 8, &rgba[3]))
            goto Fail;
        goto Success;
    }
    goto Fail;

Success:
    rcode = TRISTATE_SUCCESS;
Fail:
#if PY3
    Py_XDECREF (ascii);
#endif
    return rcode;
}

static int
_coerce_obj (PyObject *obj, Uint8 rgba[])
{
    if (PyType_IsSubtype (obj->ob_type, &PyColor_Type))
    {
	rgba[0] = ((PyColor *) obj)->r;
	rgba[1] = ((PyColor *) obj)->g;
	rgba[2] = ((PyColor *) obj)->b;
	rgba[3] = ((PyColor *) obj)->a;
	return 1;
    }
    else if (PyType_IsSubtype (obj->ob_type, &PyTuple_Type))
    {
        if (RGBAFromObj (obj, rgba))
        {
            return 1;
        }
        else if (PyErr_Occurred ())
        {
            return -1;
        }
    }

    return 0;
}

static PyColor*
_color_new_internal (PyTypeObject *type, Uint8 rgba[])
{
    /* default length of 4 - r,g,b,a. */
    return _color_new_internal_length(type, rgba, 4);
}


static PyColor*
_color_new_internal_length (PyTypeObject *type, Uint8 rgba[], Uint8 length)
{
    PyColor *color = (PyColor *) type->tp_alloc (type, 0);
    if (!color)
        return NULL;

    color->r = rgba[0];
    color->g = rgba[1];
    color->b = rgba[2];
    color->a = rgba[3];
    color->len = length;

    return color;
}

/**
 * Creates a new PyColor.
 */
static PyObject*
_color_new (PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyObject *obj = NULL, *obj1 = NULL, *obj2 = NULL, *obj3 = NULL;
    Uint8 rgba[4];

    if (!PyArg_ParseTuple (args, "O|OOO", &obj, &obj1, &obj2, &obj3))
        return NULL;

    if (Text_Check (obj))
    {
        /* Named color */
        PyObject *color = NULL;
        PyObject *name1 = NULL, *name2 = NULL;
        if (obj1 || obj2 || obj3)
            return RAISE (PyExc_ValueError, "invalid arguments");
        
        name1 = PyObject_CallMethod(obj, "replace", "(ss)", " ", "");
        if (!name1)
        {
            return NULL;
        }
        name2 = PyObject_CallMethod(name1, "lower", NULL);
        Py_DECREF(name1);
        if (!name2)
        {
            return NULL;
        }
        color = PyDict_GetItem (_COLORDICT, name2);
        Py_DECREF(name2);
        if (!color)
        {
            switch (_hexcolor (obj, rgba))
            {
                case TRISTATE_FAIL:
                return RAISE (PyExc_ValueError, "invalid color name");
                case TRISTATE_ERROR:
                return NULL;
                default:
                break;
            }
        }
        else if (!RGBAFromObj (color, rgba))
        {
            return RAISE (PyExc_ValueError, "invalid color");
        }

        return (PyObject *) _color_new_internal (type, rgba);
    }
    else if (!obj1)
    {
        /* Single integer color value or tuple */
        Uint32 color;
        if (_get_color (obj, &color))
        {
            rgba[0] = (Uint8) (color >> 24);
            rgba[1] = (Uint8) (color >> 16);
            rgba[2] = (Uint8) (color >> 8);
            rgba[3] = (Uint8) color;
        }
        else if (!RGBAFromObj (obj, rgba))
            return RAISE (PyExc_ValueError, "invalid argument");
        else
            return RAISE (PyExc_ValueError, "invalid argument");
        return (PyObject *) _color_new_internal (type, rgba);
    }
    else
    {
        Uint32 color = 0;
        
        /* Color (R,G,B[,A]) */
        if (!_get_color (obj, &color) || color > 255)
            return RAISE (PyExc_ValueError, "invalid color argument");
        rgba[0] = (Uint8) color;
        if (!_get_color (obj1, &color) || color > 255)
            return RAISE (PyExc_ValueError, "invalid color argument");
        rgba[1] = (Uint8) color;
        if (!obj2 || !_get_color (obj2, &color) || color > 255)
            return RAISE (PyExc_ValueError, "invalid color argument");
        rgba[2] = (Uint8) color;

        if (obj3)
        {
            if (!_get_color (obj3, &color) || color > 255)
                return RAISE (PyExc_ValueError, "invalid color argument");
            rgba[3] = (Uint8) color;
        }
        else /* No alpha */
            rgba[3] = 255;
        return (PyObject *) _color_new_internal (type, rgba);
    }
}

/**
 * Deallocates the PyColor.
 */
static void
_color_dealloc (PyColor *color)
{
    Py_TYPE(color)->tp_free ((PyObject *) color);
}

/**
 * repr(color)
 */
static PyObject*
_color_repr (PyColor *color)
{
    /* Max. would be (255, 255, 255, 255) */
    char buf[21];
    PyOS_snprintf (buf, sizeof (buf), "(%d, %d, %d, %d)",
        color->r, color->g, color->b, color->a);
    return Text_FromUTF8 (buf);
}

/**
 * color.normalize ()
 */
static PyObject*
_color_normalize (PyColor *color)
{
    double rgba[4];
    rgba[0] = color->r / 255.0;
    rgba[1] = color->g / 255.0;
    rgba[2] = color->b / 255.0;
    rgba[3] = color->a / 255.0;
    return Py_BuildValue ("(ffff)", rgba[0], rgba[1], rgba[2], rgba[3]);
}

/**
 * color.correct_gamma (x)
 */
static PyObject*
_color_correct_gamma (PyColor *color, PyObject *args)
{
    double frgba[4];
    Uint8 rgba[4];
    double _gamma;
    
    if (!PyArg_ParseTuple (args, "d", &_gamma))
        return NULL;

    frgba[0] = pow (color->r / 255.0, _gamma);
    frgba[1] = pow (color->g / 255.0, _gamma);
    frgba[2] = pow (color->b / 255.0, _gamma);
    frgba[3] = pow (color->a / 255.0, _gamma);

    /* visual studio doesn't have a round func, so doing it with +.5 and
     * truncaction */
    rgba[0] = (frgba[0] > 1.0) ? 255 : ((frgba[0] < 0.0) ? 0 :
        (Uint8) (frgba[0] * 255 + .5));
    rgba[1] = (frgba[1] > 1.0) ? 255 : ((frgba[1] < 0.0) ? 0 :
        (Uint8) (frgba[1] * 255 + .5));
    rgba[2] = (frgba[2] > 1.0) ? 255 : ((frgba[2] < 0.0) ? 0 :
        (Uint8) (frgba[2] * 255 + .5));
    rgba[3] = (frgba[3] > 1.0) ? 255 : ((frgba[3] < 0.0) ? 0 :
        (Uint8) (frgba[3] * 255 + .5));
    return (PyObject *) _color_new_internal (&PyColor_Type, rgba);
}

/**
 * color.r
 */
static PyObject*
_color_get_r (PyColor *color, void *closure)
{
    return PyInt_FromLong (color->r);
}

/**
 * color.r = x
 */
static int
_color_set_r (PyColor *color, PyObject *value, void *closure)
{
    Uint32 c;
    if (!_get_color (value, &c))
        return -1;
    if (c > 255)
    {
        PyErr_SetString (PyExc_ValueError, "color exceeds allowed range");
        return -1;
    }
    color->r = c;
    return 0;
}

/**
 * color.g
 */
static PyObject*
_color_get_g (PyColor *color, void *closure)
{
    return PyInt_FromLong (color->g);
}

/**
 * color.g = x
 */
static int
_color_set_g (PyColor *color, PyObject *value, void *closure)
{
    Uint32 c;
    if (!_get_color (value, &c))
        return -1;
    if (c > 255)
    {
        PyErr_SetString (PyExc_ValueError, "color exceeds allowed range");
        return -1;
    }
    color->g = c;
    return 0;
}

/**
 * color.b
 */
static PyObject*
_color_get_b (PyColor *color, void *closure)
{
    return PyInt_FromLong (color->b);
}

/**
 * color.b = x
 */
static int
_color_set_b (PyColor *color, PyObject *value, void *closure)
{
    Uint32 c;
    if (!_get_color (value, &c))
        return -1;
    if (c > 255)
    {
        PyErr_SetString (PyExc_ValueError, "color exceeds allowed range");
        return -1;
    }
    color->b = c;
    return 0;
}

/**
 * color.a
 */
static PyObject*
_color_get_a (PyColor *color, void *closure)
{
    return PyInt_FromLong (color->a);
}

/**
 * color.a = x
 */
static int
_color_set_a (PyColor *color, PyObject *value, void *closure)
{
    Uint32 c;
    if (!_get_color (value, &c))
        return -1;
    if (c > 255)
    {
        PyErr_SetString (PyExc_ValueError, "color exceeds allowed range");
        return -1;
    }
    color->a = c;
    return 0;
}

/**
 * color.hsva
 */
static PyObject*
_color_get_hsva (PyColor *color, void *closure)
{
    double hsv[3] = { 0, 0, 0 };
    double frgb[4];
    double minv, maxv, diff;

    /* Normalize */
    frgb[0] = color->r / 255.0;
    frgb[1] = color->g / 255.0;
    frgb[2] = color->b / 255.0;
    frgb[3] = color->a / 255.0;

    maxv = MAX (MAX (frgb[0], frgb[1]), frgb[2]);
    minv = MIN (MIN (frgb[0], frgb[1]), frgb[2]);
    diff = maxv - minv;

    /* Calculate V */
    hsv[2] = 100. * maxv;

    if (maxv == minv)
    {
        hsv[0] = 0;
        hsv[1] = 0;
        return Py_BuildValue ("(ffff)", hsv[0], hsv[1], hsv[2], frgb[3] * 100);
    }
    /* Calculate S */
    hsv[1] = 100. * (maxv - minv) / maxv;
    
    /* Calculate H */
    if (maxv == frgb[0])
        hsv[0] = fmod ((60 * ((frgb[1] - frgb[2]) / diff)), 360.f);
    else if (maxv == frgb[1])
        hsv[0] = (60 * ((frgb[2] - frgb[0]) / diff)) + 120.f;
    else
        hsv[0] = (60 * ((frgb[0] - frgb[1]) / diff)) + 240.f;

    if (hsv[0] < 0)
        hsv[0] += 360.f;

    /* H,S,V,A */
    return Py_BuildValue ("(ffff)", hsv[0], hsv[1], hsv[2], frgb[3] * 100);
}

static int
_color_set_hsva (PyColor *color, PyObject *value, void *closure)
{
    PyObject *item;
    double hsva[4] = { 0, 0, 0, 0 };
    double f, p, q, t, v, s;
    int hi;

    if (!PySequence_Check (value) || PySequence_Size (value) < 3)
    {
        PyErr_SetString (PyExc_ValueError, "invalid HSVA value");
        return -1;
    }

    /* H */
    item = PySequence_GetItem (value, 0);
    if (!item || !_get_double (item, &(hsva[0])) ||
        hsva[0] < 0 || hsva[0] > 360)
    {
        Py_XDECREF (item);
        PyErr_SetString (PyExc_ValueError, "invalid HSVA value");
        return -1;
    }
    Py_DECREF (item);

    /* S */
    item = PySequence_GetItem (value, 1);
    if (!item || !_get_double (item, &(hsva[1])) ||
        hsva[1] < 0 || hsva[1] > 100)
    {
        Py_XDECREF (item);
        PyErr_SetString (PyExc_ValueError, "invalid HSVA value");
        return -1;
    }
    Py_DECREF (item);

    /* V */
    item = PySequence_GetItem (value, 2);
    if (!item || !_get_double (item, &(hsva[2])) ||
        hsva[2] < 0 || hsva[2] > 100)
    {
        Py_XDECREF (item);
        PyErr_SetString (PyExc_ValueError, "invalid HSVA value");
        return -1;
    }
    Py_DECREF (item);

    /* A */
    if (PySequence_Size (value) > 3)
    {
        item = PySequence_GetItem (value, 3);
        if (!item || !_get_double (item, &(hsva[3])) ||
            hsva[3] < 0 || hsva[3] > 100)
        {
            Py_DECREF (item);
            PyErr_SetString (PyExc_ValueError, "invalid HSVA value");
            return -1;
        }
        Py_DECREF (item);
    }

    color->a = (Uint8) ((hsva[3] / 100.0f) * 255);

    s = hsva[1] / 100.f;
    v = hsva[2] / 100.f;

    hi = (int) floor (hsva[0] / 60.f);
    f = (hsva[0] / 60.f) - hi;
    p = v * (1 - s);
    q = v * (1 - s * f);
    t = v * (1 - s * (1 - f));

    switch (hi)
    {
    case 0:
        color->r = (Uint8) (v * 255);
        color->g = (Uint8) (t * 255);
        color->b = (Uint8) (p * 255);
        break;
    case 1:
        color->r = (Uint8) (q * 255);
        color->g = (Uint8) (v * 255);
        color->b = (Uint8) (p * 255);
        break;
    case 2:
        color->r = (Uint8) (p * 255);
        color->g = (Uint8) (v * 255);
        color->b = (Uint8) (t * 255);
        break;
    case 3:
        color->r = (Uint8) (p * 255);
        color->g = (Uint8) (q * 255);
        color->b = (Uint8) (v * 255);
        break;
    case 4:
        color->r = (Uint8) (t * 255);
        color->g = (Uint8) (p * 255);
        color->b = (Uint8) (v * 255);
        break;
    case 5:
        color->r = (Uint8) (v * 255);
        color->g = (Uint8) (p * 255);
        color->b = (Uint8) (q * 255);
        break;
    default:
        PyErr_SetString (PyExc_OverflowError,
            "this is not allowed to happen ever");
        return -1;
    }

    return 0;
}

/**
 * color.hsla
 */
static PyObject*
_color_get_hsla (PyColor *color, void *closure)
{
    double hsl[3] = { 0, 0, 0 };
    double frgb[4];
    double minv, maxv, diff;

    /* Normalize */
    frgb[0] = color->r / 255.0;
    frgb[1] = color->g / 255.0;
    frgb[2] = color->b / 255.0;
    frgb[3] = color->a / 255.0;

    maxv = MAX (MAX (frgb[0], frgb[1]), frgb[2]);
    minv = MIN (MIN (frgb[0], frgb[1]), frgb[2]);

    diff = maxv - minv;

    /* Calculate L */
    hsl[2] = 50.f * (maxv + minv); /* 1/2 (max + min) */

    if (maxv == minv)
    {
        hsl[1] = 0;
        hsl[0] = 0;
        return Py_BuildValue ("(ffff)", hsl[0], hsl[1], hsl[2], frgb[3] * 100);
    }

    /* Calculate S */
    if (hsl[2] <= 50)
        hsl[1] = diff / (maxv + minv);
    else
        hsl[1] = diff / (2 - maxv - minv);
    hsl[1] *= 100.f;
    
    /* Calculate H */
    if (maxv == frgb[0])
        hsl[0] = fmod ((60 * ((frgb[1] - frgb[2]) / diff)), 360.f);
    else if (maxv == frgb[1])
        hsl[0] = (60 * ((frgb[2] - frgb[0]) / diff)) + 120.f;
    else
        hsl[0] = (60 * ((frgb[0] - frgb[1]) / diff)) + 240.f;
    if (hsl[0] < 0)
        hsl[0] += 360.f;

    /* H,S,L,A */
    return Py_BuildValue ("(ffff)", hsl[0], hsl[1], hsl[2], frgb[3] * 100);
}

/**
 * color.hsla = x
 */
static int
_color_set_hsla (PyColor *color, PyObject *value, void *closure)
{
    PyObject *item;
    double hsla[4] = { 0, 0, 0, 0 };
    double ht, h, q, p = 0, s, l = 0;
    static double onethird = 1.0 / 3.0f;

    if (!PySequence_Check (value) || PySequence_Size (value) < 3)
    {
        PyErr_SetString (PyExc_ValueError, "invalid HSLA value");
        return -1;
    }

    /* H */
    item = PySequence_GetItem (value, 0);
    if (!item || !_get_double (item, &(hsla[0])) ||
        hsla[0] < 0 || hsla[0] > 360)
    {
        Py_XDECREF (item);
        PyErr_SetString (PyExc_ValueError, "invalid HSLA value");
        return -1;
    }
    Py_DECREF (item);

    /* S */
    item = PySequence_GetItem (value, 1);
    if (!item || !_get_double (item, &(hsla[1])) ||
        hsla[1] < 0 || hsla[1] > 100)
    {
        Py_XDECREF (item);
        PyErr_SetString (PyExc_ValueError, "invalid HSLA value");
        return -1;
    }
    Py_DECREF (item);

    /* L */
    item = PySequence_GetItem (value, 2);
    if (!item || !_get_double (item, &(hsla[2])) ||
        hsla[2] < 0 || hsla[2] > 100)
    {
        Py_XDECREF (item);
        PyErr_SetString (PyExc_ValueError, "invalid HSLA value");
        return -1;
    }
    Py_DECREF (item);

    /* A */
    if (PySequence_Size (value) > 3)
    {
        item = PySequence_GetItem (value, 3);
        if (!item || !_get_double (item, &(hsla[3])) ||
            hsla[3] < 0 || hsla[3] > 100)
        {
            Py_DECREF (item);
            PyErr_SetString (PyExc_ValueError, "invalid HSLA value");
            return -1;
        }
        Py_DECREF (item);
    }

    color->a = (Uint8) ((hsla[3] / 100.f) * 255);

    s = hsla[1] / 100.f;
    l = hsla[2] / 100.f;

    if (s == 0)
    {
        color->r = (Uint8) (l * 255);
        color->g = (Uint8) (l * 255);
        color->b = (Uint8) (l * 255);
        return 0;
    }

    if (l < 0.5f)
        q = l * (1 + s);
    else
        q = l + s - (l * s);
    p = 2 * l - q;

    ht = hsla[0] / 360.f;

    /* Calulate R */
    h = ht + onethird;
    if (h < 0)
        h += 1;
    else if (h > 1)
        h -= 1;

    if (h < 1./6.f)
        color->r = (Uint8) ((p + ((q - p) * 6 * h)) * 255);
    else if (h < 0.5f)
        color->r = (Uint8) (q * 255);
    else if (h < 2./3.f)
        color->r = (Uint8) ((p + ((q - p) * 6 * (2./3.f - h))) * 255);
    else
        color->r = (Uint8) (p * 255);

    /* Calculate G */
    h = ht;
    if (h < 0)
        h += 1;
    else if (h > 1)
        h -= 1;

    if (h < 1./6.f)
        color->g = (Uint8) ((p + ((q - p) * 6 * h)) * 255);
    else if (h < 0.5f)
        color->g = (Uint8) (q * 255);
    else if (h < 2./3.f)
        color->g = (Uint8) ((p + ((q - p) * 6 * (2./3.f - h))) * 255);
    else
        color->g = (Uint8) (p * 255);

    /* Calculate B */
    h = ht - onethird;
    if (h < 0)
        h += 1;
    else if (h > 1)
        h -= 1;

    if (h < 1./6.f)
        color->b = (Uint8) ((p + ((q - p) * 6 * h)) * 255);
    else if (h < 0.5f)
        color->b = (Uint8) (q * 255);
    else if (h < 2./3.f)
        color->b = (Uint8) ((p + ((q - p) * 6 * (2./3.f - h))) * 255);
    else
        color->b = (Uint8) (p * 255);

    return 0;
}

static PyObject*
_color_get_i1i2i3 (PyColor *color, void *closure)
{
    double i1i2i3[3] = { 0, 0, 0 };
    double frgb[3];

    /* Normalize */
    frgb[0] = color->r / 255.0;
    frgb[1] = color->g / 255.0;
    frgb[2] = color->b / 255.0;
    
    i1i2i3[0] = (frgb[0] + frgb[1] + frgb[2]) / 3.0f;
    i1i2i3[1] = (frgb[0] - frgb[2]) / 2.0f;
    i1i2i3[2] = (2 * frgb[1] - frgb[0] - frgb[2]) / 4.0f;
 
    return Py_BuildValue ("(fff)", i1i2i3[0], i1i2i3[1], i1i2i3[2]);
}

static int
_color_set_i1i2i3 (PyColor *color, PyObject *value, void *closure)
{
    PyObject *item;
    double i1i2i3[3] = { 0, 0, 0 };
    double ar, ag, ab;

    /* I1 */
    item = PySequence_GetItem (value, 0);
    if (!item || !_get_double (item, &(i1i2i3[0])) ||
        i1i2i3[0] < 0 || i1i2i3[0] > 1)
    {
        Py_XDECREF (item);
        PyErr_SetString (PyExc_ValueError, "invalid I1I2I3 value");
        return -1;
    }
    Py_DECREF (item);

    /* I2 */
    item = PySequence_GetItem (value, 1);
    if (!item || !_get_double (item, &(i1i2i3[1])) ||
        i1i2i3[1] < -0.5f || i1i2i3[1] > 0.5f)
    {
        Py_XDECREF (item);
        PyErr_SetString (PyExc_ValueError, "invalid I1I2I3 value");
        return -1;
    }
    Py_DECREF (item);

    /* I3 */
    item = PySequence_GetItem (value, 2);
    if (!item || !_get_double (item, &(i1i2i3[2])) ||
        i1i2i3[2] < -0.5f || i1i2i3[2] > 0.5f)
    {
        Py_XDECREF (item);
        PyErr_SetString (PyExc_ValueError, "invalid I1I2I3 value");
        return -1;
    }
    Py_DECREF (item);

    ab = i1i2i3[0] - i1i2i3[1] - 2 * i1i2i3[2] / 3.f;
    ar = 2 * i1i2i3[1] + ab;
    ag = 3 * i1i2i3[0] - ar - ab;

    color->r = (Uint8) (ar * 255);
    color->g = (Uint8) (ag * 255);
    color->b = (Uint8) (ab * 255);

    return 0;
}

static PyObject*
_color_get_cmy (PyColor *color, void *closure)
{
    double cmy[3] = { 0, 0, 0 };
    double frgb[3];

    /* Normalize */
    frgb[0] = color->r / 255.0;
    frgb[1] = color->g / 255.0;
    frgb[2] = color->b / 255.0;
    
    cmy[0] = 1.0 - frgb[0];
    cmy[1] = 1.0 - frgb[1];
    cmy[2] = 1.0 - frgb[2];
    
    return Py_BuildValue ("(fff)", cmy[0], cmy[1], cmy[2]);
}

static int
_color_set_cmy (PyColor *color, PyObject *value, void *closure)
{
    PyObject *item;
    double cmy[3] = { 0, 0, 0 };

    /* I1 */
    item = PySequence_GetItem (value, 0);
    if (!item || !_get_double (item, &(cmy[0])) || cmy[0] < 0 || cmy[0] > 1)
    {
        Py_XDECREF (item);
        PyErr_SetString (PyExc_ValueError, "invalid CMY value");
        return -1;
    }
    Py_DECREF (item);

    /* I2 */
    item = PySequence_GetItem (value, 1);
    if (!item || !_get_double (item, &(cmy[1])) || cmy[1] < 0 || cmy[1] > 1)
    {
        Py_XDECREF (item);
        PyErr_SetString (PyExc_ValueError, "invalid CMY value");
        return -1;
    }
    Py_DECREF (item);

    /* I2 */
    item = PySequence_GetItem (value, 2);
    if (!item || !_get_double (item, &(cmy[2])) || cmy[2] < 0 || cmy[2] > 1)
    {
        Py_XDECREF (item);
        PyErr_SetString (PyExc_ValueError, "invalid CMY value");
        return -1;
    }
    Py_DECREF (item);
    
    color->r = (Uint8) ((1.0 - cmy[0]) * 255);
    color->g = (Uint8) ((1.0 - cmy[1]) * 255);
    color->b = (Uint8) ((1.0 - cmy[2]) * 255);

    return 0;
}

/* Number protocol methods */

/**
 * color1 + color2
 */
static PyObject*
_color_add (PyColor *color1, PyColor *color2)
{
    Uint8 rgba[4];
    rgba[0] = MIN (color1->r + color2->r, 255);
    rgba[1] = MIN (color1->g + color2->g, 255);
    rgba[2] = MIN (color1->b + color2->b, 255);
    rgba[3] = MIN (color1->a + color2->a, 255);
    return (PyObject*) _color_new_internal (&PyColor_Type, rgba);
}

/**
 * color1 - color2
 */
static PyObject*
_color_sub (PyColor *color1, PyColor *color2)
{
    Uint8 rgba[4];
    rgba[0] = MAX (color1->r - color2->r, 0);
    rgba[1] = MAX (color1->g - color2->g, 0);
    rgba[2] = MAX (color1->b - color2->b, 0);
    rgba[3] = MAX (color1->a - color2->a, 0);
    return (PyObject*) _color_new_internal (&PyColor_Type, rgba);
}

/**
 * color1 * color2
 */
static PyObject*
_color_mul (PyColor *color1, PyColor *color2)
{
    Uint8 rgba[4];
    rgba[0] = MIN (color1->r * color2->r, 255);
    rgba[1] = MIN (color1->g * color2->g, 255);
    rgba[2] = MIN (color1->b * color2->b, 255);
    rgba[3] = MIN (color1->a * color2->a, 255);
    return (PyObject*) _color_new_internal (&PyColor_Type, rgba);
}

/**
 * color1 / color2
 */
static PyObject*
_color_div (PyColor *color1, PyColor *color2)
{
    Uint8 rgba[4] = { 0, 0, 0, 0 };
    if (color2->r != 0)
        rgba[0] = color1->r / color2->r;
    if (color2->g != 0)
        rgba[1] = color1->g / color2->g;
    if (color2->b)
        rgba[2] = color1->b / color2->b;
    if (color2->a)
        rgba[3] = color1->a / color2->a;
    return (PyObject*) _color_new_internal (&PyColor_Type, rgba);
}

/**
 * color1 % color2
 */
static PyObject*
_color_mod (PyColor *color1, PyColor *color2)
{
    Uint8 rgba[4];
    rgba[0] = color1->r % color2->r;
    rgba[1] = color1->g % color2->g;
    rgba[2] = color1->b % color2->b;
    rgba[3] = color1->a % color2->a;
    return (PyObject*) _color_new_internal (&PyColor_Type, rgba);
}

/**
 * ~color
 */
static PyObject*
_color_inv (PyColor *color)
{
    Uint8 rgba[4];
    rgba[0] = 255 - color->r;
    rgba[1] = 255 - color->g;
    rgba[2] = 255 - color->b;
    rgba[3] = 255 - color->a;
    return (PyObject*) _color_new_internal (&PyColor_Type, rgba);
}

#if !PY3
/**
 * coerce (color1, color2)
 */
static int
_color_coerce (PyObject **pv, PyObject **pw)
{
    if (PyColor_Check (*pw))
    {
        Py_INCREF (*pv);
        Py_INCREF (*pw);
        return 0;
    }
    return 1;
}
#endif

/**
 * int(color)
 */
static PyObject*
_color_int (PyColor *color)
{
    Uint32 tmp = (color->r << 24) + (color->g << 16) + 
                 (color->b << 8) + color->a;
#if !PY3
    if (tmp < LONG_MAX)
        return PyInt_FromLong ((long) tmp);
#endif
    return PyLong_FromUnsignedLong (tmp);
}

/**
 * long(color)
 */
static PyObject*
_color_long (PyColor *color)
{
    Uint32 tmp = (color->r << 24) + (color->g << 16) + (color->b << 8) +
        color->a;
    return PyLong_FromUnsignedLong (tmp);
}

/**
 * float(color)
 */
static PyObject*
_color_float (PyColor *color)
{
    Uint32 tmp = (color->r << 24) + (color->g << 16) + (color->b << 8) +
        color->a;
    return PyFloat_FromDouble ((double) tmp);
}

#if !PY3
/**
 * oct(color)
 */
static PyObject*
_color_oct (PyColor *color)
{
    char buf[100];
    Uint32 tmp = (color->r << 24) + (color->g << 16) + (color->b << 8) +
        color->a;

    if (tmp < LONG_MAX)
        PyOS_snprintf (buf, sizeof (buf), "0%lo", (unsigned long) tmp);
    else
        PyOS_snprintf (buf, sizeof (buf), "0%loL", (unsigned long) tmp);
    return PyString_FromString (buf);
}

/**
 * hex(color)
 */
static PyObject*
_color_hex (PyColor *color)
{
    char buf[100];
    Uint32 tmp = (color->r << 24) + (color->g << 16) + (color->b << 8) +
        color->a;
    if (tmp < LONG_MAX)
        PyOS_snprintf (buf, sizeof (buf), "0x%lx", (unsigned long) tmp);
    else
    {
#if PY_VERSION_HEX >= 0x02050000
        PyOS_snprintf (buf, sizeof (buf), "0x%lxL", (unsigned long) tmp);
#else
        /* <= 2.4 uses capitalised hex chars. */
        PyOS_snprintf (buf, sizeof (buf), "0x%lXL", (unsigned long) tmp);
#endif
    }
    return Text_FromUTF8 (buf);
}
#endif

/* Sequence protocol methods */

/**
 * len (color)
 */
static Py_ssize_t
_color_length (PyColor *color)
{
    return color->len;
}

/**
 * color.set_length(3)
 */

static PyObject*
_color_set_length (PyColor *color, PyObject *args)
{
    Py_ssize_t clength;

    if (!PyArg_ParseTuple (args, "k", &clength))
        return NULL;

    if (clength > 4 || clength < 1) {
        return RAISE (PyExc_ValueError, "Length needs to be 1,2,3, or 4.");
    }

    color->len = clength;

    Py_RETURN_NONE;
}




/**
 * color[x]
 */
static PyObject*
_color_item (PyColor *color, Py_ssize_t _index)
{

    if((_index > (color->len-1)) ) {
        return RAISE (PyExc_IndexError, "invalid index");
    }

    switch (_index)
    {
    case 0:
        return PyInt_FromLong (color->r);
    case 1:
        return PyInt_FromLong (color->g);
    case 2:
        return PyInt_FromLong (color->b);
    case 3:
        return PyInt_FromLong (color->a);
    default:
        return RAISE (PyExc_IndexError, "invalid index");
    }
}







static PyObject * _color_subscript(PyColor* self, PyObject* item) {

    if (PyIndex_Check(item)) {
        Py_ssize_t i;
        i = PyNumber_AsSsize_t(item, PyExc_IndexError);
        if (i == -1 && PyErr_Occurred())
            return NULL;
        /*
        if (i < 0)
            i += PyList_GET_SIZE(self);
        */
        return _color_item(self, i);
    }
    if (PySlice_Check(item)) {
        int len= 4;
        Py_ssize_t start, stop, step, slicelength;

        if (PySlice_GetIndicesEx((PySliceObject*)item, len, &start, &stop, &step, &slicelength) < 0)
            return NULL;

        if (slicelength <= 0) {
            return PyTuple_New(0);
        }
        else if (step == 1) {
            return _color_slice(self, start, stop);
        }
        else {
            PyErr_SetString(PyExc_TypeError, "slice steps not supported");
            return NULL;
        }
    }
    else {
        PyErr_Format(PyExc_TypeError,
                     "Color indices must be integers, not %.200s",
                     item->ob_type->tp_name);
        return NULL;
    }
}





/**
 * color[x] = y
 */
static int
_color_ass_item (PyColor *color, Py_ssize_t _index, PyObject *value)
{

    switch (_index)
    {
    case 0:
        return _color_set_r (color, value, NULL);
    case 1:
        return _color_set_g (color, value, NULL);
    case 2:
        return _color_set_b (color, value, NULL);
    case 3:
        return _color_set_a (color, value, NULL);
    default:
        PyErr_SetString (PyExc_IndexError, "invalid index");
        break;
    }
    return -1;
}



static PyObject *
_color_slice(register PyColor *a, 
             register Py_ssize_t ilow, 
             register Py_ssize_t ihigh)
{

        Py_ssize_t len;
        Py_ssize_t c1, c2, c3, c4;
        c1=0;c2=0;c3=0;c4=0;

        /* printf("ilow :%d:, ihigh:%d:\n", ilow, ihigh); */

        if (ilow < 0)
                ilow = 0;
        if (ihigh > 3)
                ihigh = 4;
        if (ihigh < ilow)
                ihigh = ilow;

        len = ihigh - ilow;
        /* printf("2 ilow :%d:, ihigh:%d: len:%d:\n", ilow, ihigh, len); */
        
        if(ilow == 0) {
            c1 = a->r;
            c2 = a->g;
            c3 = a->b;
            c4 = a->a;
        } else if(ilow == 1) {
            c1 = a->g;
            c2 = a->b;
            c3 = a->a;

        } else if(ilow == 2) {
            c1 = a->b;
            c2 = a->a;

        } else if(ilow == 3) {
            c1 = a->a;
        }



        /* return a tuple depending on which elements are wanted.  */
        if(len == 4) {
            return Py_BuildValue ("(iiii)",c1,c2,c3,c4);
        } else if(len == 3) {
            return Py_BuildValue ("(iii)",c1,c2,c3);
        } else if(len == 2) {
            return Py_BuildValue ("(ii)",c1,c2);
        } else if(len == 1) {
            return Py_BuildValue ("(i)",c1);
        } else {
            return Py_BuildValue ("()");
        }
}








/*
 * colorA == colorB
 * colorA != colorB
 */
static PyObject*
_color_richcompare(PyObject *o1, PyObject *o2, int opid)
{
    Uint8 rgba1[4], rgba2[4];

    switch (_coerce_obj (o1, rgba1))
    {
    case -1:
	return 0;
    case 0:
	goto Unimplemented;
    default:
        break;
    }
    switch (_coerce_obj (o2, rgba2))
    {
    case -1:
	return 0;
    case 0:
	goto Unimplemented;
    default:
        break;
    }

    switch (opid)
    {
    case Py_EQ:
        return PyBool_FromLong (*((Uint32 *) rgba1) == *((Uint32 *) rgba2));
    case Py_NE:
        return PyBool_FromLong (*((Uint32 *) rgba1) != *((Uint32 *) rgba2));
    default:
        break;
    }

Unimplemented:
    Py_INCREF (Py_NotImplemented);
    return Py_NotImplemented;
}

/**** C API interfaces ****/
static PyObject*
PyColor_New (Uint8 rgba[])
{
    return (PyObject *) _color_new_internal (&PyColor_Type, rgba);
}

static PyObject*
PyColor_NewLength (Uint8 rgba[], Uint8 length)
{
    if(length < 1 || length > 4) {
        return NULL;
    }

    return (PyObject *) _color_new_internal_length (&PyColor_Type, rgba, length);
}




static int
RGBAFromColorObj (PyObject *color, Uint8 rgba[])
{
    if (PyColor_Check (color))
    {
        rgba[0] = ((PyColor *) color)->r;
        rgba[1] = ((PyColor *) color)->g;
        rgba[2] = ((PyColor *) color)->b;
        rgba[3] = ((PyColor *) color)->a;
        return 1;
    }
    else
        return RGBAFromObj (color, rgba);
}

/*DOC*/ static char _color_doc[] =
/*DOC*/    "color module for pygame";

MODINIT_DEFINE (color)
{
    PyObject *colordict;
    PyObject *module;
    PyObject *apiobj;
    static void* c_api[PYGAMEAPI_COLOR_NUMSLOTS];
    
#if PY3
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "color",
        _color_doc,
        -1,
        _color_methods,
        NULL, NULL, NULL, NULL
    };
#endif

    /* imported needed apis; Do this first so if there is an error
       the module is not loaded.
    */
    import_pygame_base ();
    if (PyErr_Occurred ()) {
        MODINIT_ERROR;
    }

    colordict = PyImport_ImportModule ("pygame.colordict");
    if (colordict)
    {
        PyObject *_dict = PyModule_GetDict (colordict);
        PyObject *colors = PyDict_GetItemString (_dict, "THECOLORS");
        Py_INCREF (colors);
        _COLORDICT = colors;
        Py_DECREF (colordict);
    }
    else
    {
        MODINIT_ERROR;
    }
    
    /* type preparation */
    if (PyType_Ready (&PyColor_Type) < 0)
    {
        Py_DECREF (_COLORDICT);
        MODINIT_ERROR;
    }
    
    /* create the module */
#if PY3
    module = PyModule_Create (&_module);
#else
    module = Py_InitModule3 ( MODPREFIX "color", NULL, _color_doc);
#endif
    if (module == NULL) {
        Py_DECREF (_COLORDICT);
        MODINIT_ERROR;
    }
    PyColor_Type.tp_getattro = PyObject_GenericGetAttr;
    Py_INCREF (&PyColor_Type);
    if (PyModule_AddObject (module, "Color", (PyObject *) &PyColor_Type)) {
        Py_DECREF (&PyColor_Type);
        Py_DECREF (_COLORDICT);
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    Py_INCREF (_COLORDICT);
    if (PyModule_AddObject (module, "THECOLORS", _COLORDICT)) {
        Py_DECREF (_COLORDICT);
        Py_DECREF (_COLORDICT);
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

    c_api[0] = &PyColor_Type;
    c_api[1] = PyColor_New;
    c_api[2] = RGBAFromColorObj;
    c_api[3] = PyColor_NewLength;

    apiobj = PyCObject_FromVoidPtr (c_api, NULL);
    if (apiobj == NULL) {
        Py_DECREF (_COLORDICT);
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    if (PyModule_AddObject (module, PYGAMEAPI_LOCAL_ENTRY, apiobj)) {
        Py_DECREF (apiobj);
        Py_DECREF (_COLORDICT);
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    MODINIT_RETURN (module);
}
