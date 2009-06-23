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
#define PYGAME_COLOR_INTERNAL

#include <ctype.h>
#include "internals.h"
#include "pgbase.h"
#include "base_doc.h"

static int _get_color (PyObject *val, pguint32 *color);
static int _hextoint (char *hex, pgbyte *val);
static int _hexcolor (PyObject *color, pgbyte rgba[]);
static int _resolve_colorname (PyObject *name, pgbyte rgba[]);

static int _color_init (PyObject *color, PyObject *args, PyObject *kwds);
static void _color_dealloc (PyColor *color);
static PyObject* _color_repr (PyColor *color);
static PyObject* _color_normalize (PyColor *color);
static PyObject* _color_correct_gamma (PyColor *color, PyObject *args);

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
static int _color_coerce (PyObject **pv, PyObject **pw);
static PyObject* _color_int (PyColor *color);
static PyObject* _color_long (PyColor *color);
static PyObject* _color_float (PyColor *color);
static PyObject* _color_oct (PyColor *color);
static PyObject* _color_hex (PyColor *color);

/* Sequence protocol methods */
static Py_ssize_t _color_length (PyColor *color);
static PyObject* _color_item (PyColor *color, Py_ssize_t _index);
static int _color_ass_item (PyColor *color, Py_ssize_t _index, PyObject *value);

static PyObject* _color_richcompare (PyObject *o1, PyObject *o2, int opid);

/* C API */
static PyObject* PyColor_New (pgbyte rgba[]);
static PyObject* PyColor_NewFromNumber (pguint32 val);
static pguint32 PyColor_AsNumber (PyObject *color);

/**
 * Methods, which are bound to the PyColor type.
 */
static PyMethodDef _color_methods[] =
{
    { "normalize", (PyCFunction) _color_normalize, METH_NOARGS,
      DOC_BASE_COLOR_NORMALIZE },
    { "correct_gamma", (PyCFunction) _color_correct_gamma, METH_VARARGS,
      DOC_BASE_COLOR_CORRECT_GAMMA },
    { NULL, NULL, 0, NULL }
};

/**
 * Getters and setters for the PyColor.
 */
static PyGetSetDef _color_getsets[] =
{
    { "r", (getter) _color_get_r, (setter) _color_set_r, DOC_BASE_COLOR_R,
      NULL },
    { "g", (getter) _color_get_g, (setter) _color_set_g, DOC_BASE_COLOR_G,
      NULL },
    { "b", (getter) _color_get_b, (setter) _color_set_b, DOC_BASE_COLOR_B,
      NULL },
    { "a", (getter) _color_get_a, (setter) _color_set_a, DOC_BASE_COLOR_A,
      NULL },
    { "hsva", (getter) _color_get_hsva, (setter) _color_set_hsva,
      DOC_BASE_COLOR_HSVA, NULL },
    { "hsla", (getter) _color_get_hsla, (setter) _color_set_hsla,
      DOC_BASE_COLOR_HSLA, NULL },
    { "i1i2i3", (getter) _color_get_i1i2i3, (setter) _color_set_i1i2i3,
      DOC_BASE_COLOR_I1I2I3, NULL },
    { "cmy", (getter) _color_get_cmy, (setter) _color_set_cmy,
      DOC_BASE_COLOR_CMY, NULL },
    { NULL, NULL, NULL, NULL, NULL }
};

static PyNumberMethods _color_as_number =
{
#ifndef IS_PYTHON_3
    (binaryfunc) _color_add, /* nb_add */
    (binaryfunc) _color_sub, /* nb_subtract */
    (binaryfunc) _color_mul, /* nb_multiply */
    (binaryfunc) _color_div, /* nb_divide */
    (binaryfunc) _color_mod, /* nb_remainder */
    0,                       /* nb_divmod */
    0,                       /* nb_power */
    0,                       /* nb_negative */
    0,                       /* nb_positive */
    0,                       /* nb_absolute */
    0,                       /* nb_nonzero */
    (unaryfunc) _color_inv,  /* nb_invert */
    0,                       /* nb_lshift */
    0,                       /* nb_rshift */
    0,                       /* nb_and */
    0,                       /* nb_xor */
    0,                       /* nb_or */
    _color_coerce,           /* nb_coerce */
    (unaryfunc) _color_int,  /* nb_int */
    (unaryfunc) _color_long, /* nb_long */
    (unaryfunc) _color_float,/* nb_float */
    (unaryfunc) _color_oct,  /* nb_oct */
    (unaryfunc) _color_hex,  /* nb_hex */
    0,                       /* nb_inplace_add */
    0,                       /* nb_inplace_subtract */
    0,                       /* nb_inplace_multiply */
    0,                       /* nb_inplace_divide */
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
#else /* 0x03000000 */
    (binaryfunc) _color_add, /* nb_add */
    (binaryfunc) _color_sub, /* nb_subtract */
    (binaryfunc) _color_mul, /* nb_multiply */
    (binaryfunc) _color_mod, /* nb_remainder */
    0,                       /* nb_divmod */
    0,                       /* nb_power */
    0,                       /* nb_negative */
    0,                       /* nb_positive */
    0,                       /* nb_absolute */
    0,                       /* nb_bool */
    (unaryfunc) _color_inv,  /* nb_invert */
    0,                       /* nb_lshift */
    0,                       /* nb_rshift */
    0,                       /* nb_and */
    0,                       /* nb_xor */
    0,                       /* nb_or */
    (unaryfunc) _color_int,  /* nb_int */
    (unaryfunc) _color_long, /* nb_long */
    (unaryfunc) _color_float,/* nb_float */
    0,                       /* nb_inplace_add */
    0,                       /* nb_inplace_subtract */
    0,                       /* nb_inplace_multiply */
    0,                       /* nb_inplace_remainder */
    0,                       /* nb_inplace_power */
    0,                       /* nb_inplace_lshift */
    0,                       /* nb_inplace_rshift */
    0,                       /* nb_inplace_and */
    0,                       /* nb_inplace_xor */
    0,                       /* nb_inplace_or */
    (binaryfunc) _color_div, /* nb_floor_divide */
    (binaryfunc) _color_div, /* nb_true_divide */
    0,                       /* nb_inplace_floor_divide */
    0,                       /* nb_inplace_true_divide */
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
    NULL,                              /* sq_slice */
    (ssizeobjargproc) _color_ass_item, /* sq_ass_item */
    NULL,                              /* sq_ass_slice */
    NULL,                              /* sq_contains */
    NULL,                              /* sq_inplace_concat */
    NULL,                              /* sq_inplace_repeat */
};

PyTypeObject PyColor_Type =
{
    TYPE_HEAD(NULL,0)
    "base.Color",             /* tp_name */
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
    0,                          /* tp_as_mapping */
    0,                          /* tp_hash */
    0,                          /* tp_call */
    0,                          /* tp_str */
    0,                          /* tp_getattro */
    0,                          /* tp_setattro */
    0,                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    DOC_BASE_COLOR,
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
    (initproc)_color_init,      /* tp_init */
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

#define RGB_EQUALS(x,y)                          \
    ((((PyColor *)x)->r == ((PyColor *)y)->r) && \
     (((PyColor *)x)->g == ((PyColor *)y)->g) && \
     (((PyColor *)x)->b == ((PyColor *)y)->b) && \
     (((PyColor *)x)->a == ((PyColor *)y)->a))

static int
_get_color (PyObject *val, pguint32 *color)
{
    unsigned int intval;

    if (!val || !color)
        return 0;
    
    if (PyLong_Check (val))
    {
        unsigned long longval;
        longval = PyLong_AsUnsignedLong (val);
        if (PyErr_Occurred ())
        {
            PyErr_Clear ();
            PyErr_SetString (PyExc_ValueError, "color argument too large");
            return 0;
        }
        *color = (pguint32) longval;
        return 1;
    }
    else if (UintFromObj (val, &intval))
        *color = (pguint32) intval;
    else
        return 0;
    return 1;
}

static int
_hextoint (char *hex, pgbyte *val)
{
    pgbyte temp = 0;

    switch (toupper (hex[0]))
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

static int
_hexcolor (PyObject *color, pgbyte rgba[])
{
    size_t len;
    char *name;
    PyObject *tmp;

    if (!ASCIIFromObject (color, &name, &tmp))
        return 0;

    len = strlen (name);
    /* hex colors can be
     * #RRGGBB
     * #RRGGBBAA
     * 0xRRGGBB
     * 0xRRGGBBAA
     */
    if (len < 7)
        goto fail;

    if (name[0] == '#')
    {
        if (len != 7 && len != 9)
            goto fail;
        if (!_hextoint (name + 1, &rgba[0]))
            goto fail;
        if (!_hextoint (name + 3, &rgba[1]))
            goto fail;
        if (!_hextoint (name + 5, &rgba[2]))
            goto fail;
        rgba[3] = 255;
        if (len == 9 && !_hextoint (name + 7, &rgba[3]))
            goto fail;
        goto success;
    }
    else if (name[0] == '0' && name[1] == 'x')
    {
        if (len != 8 && len != 10)
            goto fail;
        if (!_hextoint (name + 2, &rgba[0]))
            goto fail;
        if (!_hextoint (name + 4, &rgba[1]))
            goto fail;
        if (!_hextoint (name + 6, &rgba[2]))
            goto fail;
        rgba[3] = 255;
        if (len == 10 && !_hextoint (name + 8, &rgba[3]))
            goto fail;
        goto success;
    }
fail:
    Py_XDECREF (tmp);
    return 0;

success:
    Py_XDECREF (tmp);
    return 1;
}

static int
_resolve_colorname (PyObject *name, pgbyte rgba[])
{
    static PyObject *_COLORDICT = NULL;

    if (!_COLORDICT)
    {
        PyObject *colordict = PyImport_ImportModule ("pygame2.colordict");
        if (colordict)
        {
            PyObject *_dict = PyModule_GetDict (colordict);
            PyObject *colors = PyDict_GetItemString (_dict, "THECOLORS");
            Py_INCREF (colors);
            _COLORDICT = colors;
            Py_DECREF (colordict);
        }
    }
    
    if (_COLORDICT)
    {
        int i;
        pguint32 c;
        PyObject *color, *item;
        PyObject *name2 = PyObject_CallMethod (name, "lower", NULL);
        if (!name2)
            return 0;
        color = PyDict_GetItem (_COLORDICT, name2);
        Py_DECREF (name2);

        if (!color)
        {
            PyErr_SetString (PyExc_ValueError, "invalid color name");
            return 0;
        }
        if (!PySequence_Check (color) || PySequence_Size (color) != 4)
        {
            PyErr_SetString (PyExc_ValueError, "invalid color value");
            return 0;
        }

        for (i = 0; i < 4; i++)
        {
            item = PySequence_GetItem (color, i);
            if (!_get_color (item, &c) || c > 255)
            {
                Py_XDECREF (item);
                PyErr_SetString (PyExc_ValueError, "invalid color value");
                return 0;
            }
            Py_DECREF (item);
            rgba[i] = (pgbyte) c;
        }
        return 1;
    }

    PyErr_SetString (PyExc_PyGameError, "colordict package could not be found");
    return 0;
}

/**
 * Creates a new PyColor.
 */
static int
_color_init (PyObject *color, PyObject *args, PyObject *kwds)
{
    PyObject *obj = NULL, *obj1 = NULL, *obj2 = NULL, *obj3 = NULL;
    pgbyte rgba[4];

    if (!PyArg_ParseTuple (args, "O|OOO", &obj, &obj1, &obj2, &obj3))
        return -1;

    if (IsTextObj (obj))
    {
        /* Named color */
        if (obj1 || obj2 || obj3)
        {
            PyErr_SetString (PyExc_ValueError, "invalid arguments");
            return -1;
        }

        if (!_hexcolor (obj, rgba))
            if (!_resolve_colorname (obj, rgba))
                return -1;

        ((PyColor*)color)->r = rgba[0];
        ((PyColor*)color)->g = rgba[1];
        ((PyColor*)color)->b = rgba[2];
        ((PyColor*)color)->a = rgba[3];
        return 0;
    }
    else if (!obj1)
    {
        /* Single integer color value or tuple */
        pguint32 c;
        if (_get_color (obj, &c))
        {
            ((PyColor*)color)->a = (pgbyte) (c >> 24);
            ((PyColor*)color)->r = (pgbyte) (c >> 16);
            ((PyColor*)color)->g = (pgbyte) (c >> 8);
            ((PyColor*)color)->b = (pgbyte) c;
        }
        else
            return -1;
        return 0;
    }
    else
    {
        pguint32 c = 0;
        
        /* Color (R,G,B[,A]) */
        if (!_get_color (obj, &c) || c > 255)
        {
            PyErr_SetString (PyExc_ValueError, "invalid color argument red");
            return -1;
        }
        rgba[0] = (pgbyte) c;
        if (!_get_color (obj1, &c) || c > 255)
        {
            PyErr_SetString (PyExc_ValueError, "invalid color argument green");
            return -1;
        }
        rgba[1] = (pgbyte) c;
        if (!obj2 || !_get_color (obj2, &c) || c > 255)
        {
            PyErr_SetString (PyExc_ValueError, "invalid color argument blue");
            return -1;
        }
        rgba[2] = (pgbyte) c;

        if (obj3)
        {
            if (!_get_color (obj3, &c) || c > 255)
            {
                PyErr_SetString (PyExc_ValueError,
                    "invalid color argument alpha");
                return -1;
            }
            rgba[3] = (pgbyte) c;
        }
        else /* No alpha */
            rgba[3] = 255;
        ((PyColor*)color)->r = rgba[0];
        ((PyColor*)color)->g = rgba[1];
        ((PyColor*)color)->b = rgba[2];
        ((PyColor*)color)->a = rgba[3];
        return 0;
    }
}

/**
 * Deallocates the PyColor.
 */
static void
_color_dealloc (PyColor *color)
{
    ((PyObject*)color)->ob_type->tp_free ((PyObject *) color);
}

/**
 * repr(color)
 */
static PyObject*
_color_repr (PyColor *color)
{
    /* Max. would be Color(255, 255, 255, 255) */
    return Text_FromFormat("Color(%d, %d, %d, %d)",
        color->r, color->g, color->b, color->a);
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
    pgbyte rgba[4];
    double _gamma;
    
    if (!PyArg_ParseTuple (args, "d:correct_gamma", &_gamma))
        return NULL;

    frgba[0] = pow (color->r / 255.0, _gamma);
    frgba[1] = pow (color->g / 255.0, _gamma);
    frgba[2] = pow (color->b / 255.0, _gamma);
    frgba[3] = pow (color->a / 255.0, _gamma);

    /* visual studio doesn't have a round func, so doing it with +.5 and
     * truncaction */
    rgba[0] = (frgba[0] > 1.0) ? 255 : ((frgba[0] < 0.0) ? 0 :
        (pgbyte) (frgba[0] * 255 + .5));
    rgba[1] = (frgba[1] > 1.0) ? 255 : ((frgba[1] < 0.0) ? 0 :
        (pgbyte) (frgba[1] * 255 + .5));
    rgba[2] = (frgba[2] > 1.0) ? 255 : ((frgba[2] < 0.0) ? 0 :
        (pgbyte) (frgba[2] * 255 + .5));
    rgba[3] = (frgba[3] > 1.0) ? 255 : ((frgba[3] < 0.0) ? 0 :
        (pgbyte) (frgba[3] * 255 + .5));
    return PyColor_New (rgba);
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
    pguint32 c;
    if (!_get_color (value, &c))
        return -1;
    if (c > 255)
    {
        PyErr_SetString (PyExc_ValueError, "color exceeds allowed range");
        return -1;
    }
    color->r = (pgbyte) c;
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
    pguint32 c;
    if (!_get_color (value, &c))
        return -1;
    if (c > 255)
    {
        PyErr_SetString (PyExc_ValueError, "color exceeds allowed range");
        return -1;
    }
    color->g = (pgbyte) c;
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
    pguint32 c;
    if (!_get_color (value, &c))
        return -1;
    if (c > 255)
    {
        PyErr_SetString (PyExc_ValueError, "color exceeds allowed range");
        return -1;
    }
    color->b = (pgbyte) c;
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
    pguint32 c;
    if (!_get_color (value, &c))
        return -1;
    if (c > 255)
    {
        PyErr_SetString (PyExc_ValueError, "color exceeds allowed range");
        return -1;
    }
    color->a = (pgbyte) c;
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
    double hsva[4] = { 0, 0, 0, 0 };
    double f, p, q, t, v, s;
    int hi;

    if (!PySequence_Check (value) || PySequence_Size (value) < 3)
    {
        PyErr_SetString (PyExc_ValueError, "invalid HSVA value");
        return -1;
    }

    /* H */
    if (!DoubleFromSeqIndex (value, 0, &(hsva[0]))
        || hsva[0] < 0 || hsva[0] > 360)
    {
        PyErr_SetString (PyExc_ValueError, "invalid HSVA value");
        return -1;
    }

    /* S */
    if (!DoubleFromSeqIndex (value, 1, &(hsva[1]))
        || hsva[1] < 0 || hsva[1] > 100)
    {
        PyErr_SetString (PyExc_ValueError, "invalid HSVA value");
        return -1;
    }

    /* V */
    if (!DoubleFromSeqIndex (value, 2, &(hsva[2]))
        || hsva[2] < 0 || hsva[2] > 100)
    {
        PyErr_SetString (PyExc_ValueError, "invalid HSVA value");
        return -1;
    }

    /* A */
    if (PySequence_Size (value) > 3)
    {
        if (!DoubleFromSeqIndex (value, 3, &(hsva[3]))
            || hsva[3] < 0 || hsva[3] > 100)
        {
            PyErr_SetString (PyExc_ValueError, "invalid HSVA value");
            return -1;
        }
    }

    color->a = (pgbyte) ((hsva[3] / 100.0f) * 255);

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
        color->r = (pgbyte) (v * 255);
        color->g = (pgbyte) (t * 255);
        color->b = (pgbyte) (p * 255);
        break;
    case 1:
        color->r = (pgbyte) (q * 255);
        color->g = (pgbyte) (v * 255);
        color->b = (pgbyte) (p * 255);
        break;
    case 2:
        color->r = (pgbyte) (p * 255);
        color->g = (pgbyte) (v * 255);
        color->b = (pgbyte) (t * 255);
        break;
    case 3:
        color->r = (pgbyte) (p * 255);
        color->g = (pgbyte) (q * 255);
        color->b = (pgbyte) (v * 255);
        break;
    case 4:
        color->r = (pgbyte) (t * 255);
        color->g = (pgbyte) (p * 255);
        color->b = (pgbyte) (v * 255);
        break;
    case 5:
        color->r = (pgbyte) (v * 255);
        color->g = (pgbyte) (p * 255);
        color->b = (pgbyte) (q * 255);
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
    double hsla[4] = { 0, 0, 0, 0 };
    double ht, h, q, p = 0, s, l = 0;
    static double onethird = 1.0 / 3.0f;

    if (!PySequence_Check (value) || PySequence_Size (value) < 3)
    {
        PyErr_SetString (PyExc_ValueError, "invalid HSLA value");
        return -1;
    }

    /* H */
    if (!DoubleFromSeqIndex (value, 0, &(hsla[0]))
        || hsla[0] < 0 || hsla[0] > 360)
    {
        PyErr_SetString (PyExc_ValueError, "invalid HSLA value");
        return -1;
    }

    /* S */
    if (!DoubleFromSeqIndex (value, 1, &(hsla[1]))
        || hsla[1] < 0 || hsla[1] > 100)
    {
        PyErr_SetString (PyExc_ValueError, "invalid HSLA value");
        return -1;
    }

    /* L */
    if (!DoubleFromSeqIndex (value, 2, &(hsla[2]))
        || hsla[2] < 0 || hsla[2] > 100)
    {
        PyErr_SetString (PyExc_ValueError, "invalid HSLA value");
        return -1;
    }

    /* A */
    if (PySequence_Size (value) > 3)
    {
        if (!DoubleFromSeqIndex (value, 3, &(hsla[3]))
            || hsla[3] < 0 || hsla[3] > 100)
        {
            PyErr_SetString (PyExc_ValueError, "invalid HSLA value");
            return -1;
        }
    }

    color->a = (pgbyte) ((hsla[3] / 100.f) * 255);

    s = hsla[1] / 100.f;
    l = hsla[2] / 100.f;

    if (s == 0)
    {
        color->r = (pgbyte) (l * 255);
        color->g = (pgbyte) (l * 255);
        color->b = (pgbyte) (l * 255);
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
        color->r = (pgbyte) ((p + ((q - p) * 6 * h)) * 255);
    else if (h < 0.5f)
        color->r = (pgbyte) (q * 255);
    else if (h < 2./3.f)
        color->r = (pgbyte) ((p + ((q - p) * 6 * (2./3.f - h))) * 255);
    else
        color->r = (pgbyte) (p * 255);

    /* Calculate G */
    h = ht;
    if (h < 0)
        h += 1;
    else if (h > 1)
        h -= 1;

    if (h < 1./6.f)
        color->g = (pgbyte) ((p + ((q - p) * 6 * h)) * 255);
    else if (h < 0.5f)
        color->g = (pgbyte) (q * 255);
    else if (h < 2./3.f)
        color->g = (pgbyte) ((p + ((q - p) * 6 * (2./3.f - h))) * 255);
    else
        color->g = (pgbyte) (p * 255);

    /* Calculate B */
    h = ht - onethird;
    if (h < 0)
        h += 1;
    else if (h > 1)
        h -= 1;

    if (h < 1./6.f)
        color->b = (pgbyte) ((p + ((q - p) * 6 * h)) * 255);
    else if (h < 0.5f)
        color->b = (pgbyte) (q * 255);
    else if (h < 2./3.f)
        color->b = (pgbyte) ((p + ((q - p) * 6 * (2./3.f - h))) * 255);
    else
        color->b = (pgbyte) (p * 255);

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
    double i1i2i3[3] = { 0, 0, 0 };
    double ar, ag, ab;

    /* I1 */
    if (!DoubleFromSeqIndex (value, 0, &(i1i2i3[0]))
        || i1i2i3[0] < 0 || i1i2i3[0] > 1)
    {
        PyErr_SetString (PyExc_ValueError, "invalid I1I2I3 value");
        return -1;
    }

    /* I2 */
    if (!DoubleFromSeqIndex (value, 1, &(i1i2i3[1]))
        || i1i2i3[1] < -0.5f || i1i2i3[1] > 0.5f)
    {
        PyErr_SetString (PyExc_ValueError, "invalid I1I2I3 value");
        return -1;
    }

    /* I3 */
    if (!DoubleFromSeqIndex (value, 2, &(i1i2i3[2]))
        || i1i2i3[2] < -0.5f || i1i2i3[2] > 0.5f)
    {
        PyErr_SetString (PyExc_ValueError, "invalid I1I2I3 value");
        return -1;
    }

    ab = i1i2i3[0] - i1i2i3[1] - 2 * i1i2i3[2] / 3.f;
    ar = 2 * i1i2i3[1] + ab;
    ag = 3 * i1i2i3[0] - ar - ab;

    color->r = (pgbyte) (ar * 255);
    color->g = (pgbyte) (ag * 255);
    color->b = (pgbyte) (ab * 255);

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
    double cmy[3] = { 0, 0, 0 };

    /* I1 */
    if (!DoubleFromSeqIndex (value, 0, &(cmy[0])) || cmy[0] < 0 || cmy[0] > 1)
    {
        PyErr_SetString (PyExc_ValueError, "invalid CMY value");
        return -1;
    }

    /* I2 */
    if (!DoubleFromSeqIndex (value, 1, &(cmy[1])) || cmy[1] < 0 || cmy[1] > 1)
    {
        PyErr_SetString (PyExc_ValueError, "invalid CMY value");
        return -1;
    }

    /* I2 */
    if (!DoubleFromSeqIndex (value, 2, &(cmy[2])) || cmy[2] < 0 || cmy[2] > 1)
    {
        PyErr_SetString (PyExc_ValueError, "invalid CMY value");
        return -1;
    }
    
    color->r = (pgbyte) ((1.0 - cmy[0]) * 255);
    color->g = (pgbyte) ((1.0 - cmy[1]) * 255);
    color->b = (pgbyte) ((1.0 - cmy[2]) * 255);

    return 0;
}

/* Number protocol methods */

/**
 * color1 + color2
 */
static PyObject*
_color_add (PyColor *color1, PyColor *color2)
{
    pgbyte rgba[4];
    rgba[0] = (pgbyte) MIN (color1->r + color2->r, 255);
    rgba[1] = (pgbyte) MIN (color1->g + color2->g, 255);
    rgba[2] = (pgbyte) MIN (color1->b + color2->b, 255);
    rgba[3] = (pgbyte) MIN (color1->a + color2->a, 255);
    return PyColor_New (rgba);
}

/**
 * color1 - color2
 */
static PyObject*
_color_sub (PyColor *color1, PyColor *color2)
{
    pgbyte rgba[4];
    rgba[0] = (pgbyte) MAX (color1->r - color2->r, 0);
    rgba[1] = (pgbyte) MAX (color1->g - color2->g, 0);
    rgba[2] = (pgbyte) MAX (color1->b - color2->b, 0);
    rgba[3] = (pgbyte) MAX (color1->a - color2->a, 0);
    return PyColor_New (rgba);
}

/**
 * color1 * color2
 */
static PyObject*
_color_mul (PyColor *color1, PyColor *color2)
{
    pgbyte rgba[4];
    rgba[0] = (pgbyte) MIN (color1->r * color2->r, 255);
    rgba[1] = (pgbyte) MIN (color1->g * color2->g, 255);
    rgba[2] = (pgbyte) MIN (color1->b * color2->b, 255);
    rgba[3] = (pgbyte) MIN (color1->a * color2->a, 255);
    return PyColor_New (rgba);
}

/**
 * color1 / color2
 */
static PyObject*
_color_div (PyColor *color1, PyColor *color2)
{
    pgbyte rgba[4] = { 0, 0, 0, 0 };
    if (color2->r != 0)
        rgba[0] = color1->r / color2->r;
    if (color2->g != 0)
        rgba[1] = color1->g / color2->g;
    if (color2->b)
        rgba[2] = color1->b / color2->b;
    if (color2->a)
        rgba[3] = color1->a / color2->a;
    return PyColor_New (rgba);
}

/**
 * color1 % color2
 */
static PyObject*
_color_mod (PyColor *color1, PyColor *color2)
{
    pgbyte rgba[4];
    rgba[0] = color1->r % color2->r;
    rgba[1] = color1->g % color2->g;
    rgba[2] = color1->b % color2->b;
    rgba[3] = color1->a % color2->a;
    return PyColor_New (rgba);
}

/**
 * ~color
 */
static PyObject*
_color_inv (PyColor *color)
{
    pgbyte rgba[4];
    rgba[0] = (pgbyte) (255 - color->r);
    rgba[1] = (pgbyte) (255 - color->g);
    rgba[2] = (pgbyte) (255 - color->b);
    rgba[3] = (pgbyte) (255 - color->a);
    return PyColor_New (rgba);
}

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

/**
 * int(color)
 */
static PyObject*
_color_int (PyColor *color)
{
    unsigned long tmp = ((unsigned long) color->a << 24) +
        ((unsigned long) color->r << 16) +
        ((unsigned long) color->g << 8) +
        (unsigned long) color->b;
    if (tmp < INT_MAX)
        return PyInt_FromLong ((long) tmp);
    return PyLong_FromUnsignedLong (tmp);
}

/**
 * long(color)
 */
static PyObject*
_color_long (PyColor *color)
{
    unsigned long tmp = ((unsigned long) color->a << 24) +
        ((unsigned long) color->r << 16) +
        ((unsigned long) color->g << 8) +
        (unsigned long) color->b;
    return PyLong_FromUnsignedLong (tmp);
}

/**
 * float(color)
 */
static PyObject*
_color_float (PyColor *color)
{
    unsigned long tmp = ((unsigned long) color->a << 24) +
        ((unsigned long) color->r << 16) +
        ((unsigned long) color->g << 8) +
        (unsigned long) color->b;
    return PyFloat_FromDouble ((double) tmp);
}

/**
 * oct(color)
 */
static PyObject*
_color_oct (PyColor *color)
{
    char buf[100];
    unsigned long tmp = ((unsigned long) color->a << 24) +
        ((unsigned long) color->r << 16) +
        ((unsigned long) color->g << 8) +
        (unsigned long) color->b;
    if (tmp < LONG_MAX)
        PyOS_snprintf (buf, sizeof (buf), "0%lo", tmp);
    else
        PyOS_snprintf (buf, sizeof (buf), "0%loL", tmp);
    return Text_FromUTF8 (buf);
}

/**
 * hex(color)
 */
static PyObject*
_color_hex (PyColor *color)
{
    char buf[100];
    unsigned long tmp = ((unsigned long) color->a << 24) +
        ((unsigned long) color->r << 16) +
        ((unsigned long) color->g << 8) +
        (unsigned long) color->b;
    if (tmp < LONG_MAX)
        PyOS_snprintf (buf, sizeof (buf), "0x%lx", tmp);
    else
    {
#if PY_VERSION_HEX >= 0x02050000
        PyOS_snprintf (buf, sizeof (buf), "0x%lxL", tmp);
#else
        /* <= 2.4 uses capitalised hex chars. */
        PyOS_snprintf (buf, sizeof (buf), "0x%lXL", tmp);
#endif
    }
    return Text_FromUTF8 (buf);
}

/* Sequence protocol methods */

/**
 * len (color)
 */
static Py_ssize_t
_color_length (PyColor *color)
{
    return 4;
}

/**
 * color[x]
 */
static PyObject*
_color_item (PyColor *color, Py_ssize_t _index)
{
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
        PyErr_SetString (PyExc_IndexError, "invalid index");
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

/**
 * colora == colorb
 * colora != colorb
 */
static PyObject*
_color_richcompare (PyObject *o1, PyObject *o2, int opid)
{
    PyColor *c1, *c2;
    int equal;
    
    if (!(PyColor_Check(o1) && PyColor_Check (o2)))
    {
		Py_INCREF (Py_NotImplemented);
		return Py_NotImplemented;
	}
    c1 = (PyColor*) o1;
    c2 = (PyColor*) o2;
    
    equal = c1->r == c2->r && c1->g == c2->g &&
        c1->b == c2->b && c1->a == c2->a;

    switch (opid)
    {
    case Py_EQ:
        return PyBool_FromLong (equal);
    case Py_NE:
        return PyBool_FromLong (!equal);
    default:
        break;
    }
    Py_INCREF (Py_NotImplemented);
    return Py_NotImplemented;
}

/* C API */
static PyObject*
PyColor_New (pgbyte rgba[])
{
    PyColor *color;
    if (!rgba)
    {
        PyErr_SetString (PyExc_TypeError, "rgba must not be NULL");
        return NULL;
    }
    color = (PyColor*) PyColor_Type.tp_new (&PyColor_Type, NULL, NULL);
    if (!color)
        return NULL;

    color->r = rgba[0];
    color->g = rgba[1];
    color->b = rgba[2];
    color->a = rgba[3];
    return (PyObject*)color;
}

static PyObject*
PyColor_NewFromNumber (pguint32 val)
{
    pgbyte rgba[4];
    
    rgba[3] = (pgbyte) (val >> 24);
    rgba[0] = (pgbyte) (val >> 16);
    rgba[1] = (pgbyte) (val >> 8);
    rgba[2] = (pgbyte) val;
    return PyColor_New (rgba);
}

static PyObject*
PyColor_NewFromRGBA (pgbyte r, pgbyte g, pgbyte b, pgbyte a)
{
    PyColor *color = (PyColor*) PyColor_Type.tp_new (&PyColor_Type, NULL, NULL);
    if (!color)
        return NULL;

    color->r = r;
    color->g = g;
    color->b = b;
    color->a = a;
    return (PyObject*)color;
}

static pguint32
PyColor_AsNumber (PyObject *color)
{
    pguint32 tmp = 0;
    PyColor *c = (PyColor*) color;
    
    if (!color)
    {
        PyErr_SetString (PyExc_ValueError, "color must not be NULL");
        return 0;
    }

    if(!PyColor_Check (color))
    {
        PyErr_SetString (PyExc_TypeError, "color must be a Color");
        return 0;
    }

    tmp = ((pguint32) c->a << 24) + ((pguint32) c->r << 16) +
        ((pguint32) c->g << 8) + (pguint32) c->b;
    return tmp;
}

void
color_export_capi (void **capi)
{
    capi[PYGAME_COLOR_FIRSTSLOT] = &PyColor_Type;
    capi[PYGAME_COLOR_FIRSTSLOT+1] = PyColor_New;
    capi[PYGAME_COLOR_FIRSTSLOT+2] = PyColor_NewFromNumber;
    capi[PYGAME_COLOR_FIRSTSLOT+3] = PyColor_NewFromRGBA;
    capi[PYGAME_COLOR_FIRSTSLOT+4] = PyColor_AsNumber;
}
