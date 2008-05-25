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

typedef struct
{
    PyObject_HEAD
    /* RGBA */
    Uint8 r;
    Uint8 g;
    Uint8 b;
    Uint8 a;
} PyColor;

static PyObject *_COLORDICT = NULL;

static int _get_color (PyObject *val, Uint32 *color);

static PyColor* _color_new_internal (PyTypeObject *type, Uint8 rgba[]);
static PyObject* _color_new (PyTypeObject *type, PyObject *args,
    PyObject *kwds);
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
static PyObject* _color_get_hlsa (PyColor *color, void *closure);
static int _color_set_hlsa (PyColor *color, PyObject *value, void *closure);
static PyObject* _color_get_yuv (PyColor *color, void *closure);
static int _color_set_yuv (PyColor *color, PyObject *value, void *closure);

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

/* C API interfaces */
static PyObject* PyColor_New (Uint8 rgba[]);
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
    { "hlsa", (getter) _color_get_hlsa, (setter) _color_set_hlsa, DOC_COLORHLSA,
      NULL },
    { "yuv", (getter) _color_get_yuv, (setter) _color_set_yuv, DOC_COLORYUV,
      NULL },
    { NULL, NULL, NULL, NULL, NULL }
};


static PyNumberMethods _color_as_number =
{
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

static PyTypeObject PyColor_Type =
{
    PyObject_HEAD_INIT(NULL)
    0,
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
    0,                          /* tp_as_mapping */
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
    0,                          /* tp_richcompare */
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
    0,                          /* tp_free */
    0,                          /* tp_is_gc */
    0,                          /* tp_bases */
    0,                          /* tp_mro */
    0,                          /* tp_cache */
    0,                          /* tp_subclasses */
    0,                          /* tp_weaklist */
    0                           /* tp_del */
};

#define PyColor_Check(o) \
    ((o)->ob_type == (PyTypeObject *) &PyColor_Type)

#define RGB_EQUALS(x,y)                          \
    ((((PyColor *)x)->r == ((PyColor *)y)->r) && \
     (((PyColor *)x)->g == ((PyColor *)y)->g) && \
     (((PyColor *)x)->b == ((PyColor *)y)->b) && \
     (((PyColor *)x)->a == ((PyColor *)y)->a))

static int
_get_color (PyObject *val, Uint32 *color)
{
    if (!val || !color)
        return 0;

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
    else if (PyLong_Check (val))
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
    else
        PyErr_SetString (PyExc_ValueError, "invalid color argument");
    return 0;
}

static PyColor*
_color_new_internal (PyTypeObject *type, Uint8 rgba[])
{
    PyColor *color = (PyColor *) type->tp_alloc (type, 0);
    if (!color)
        return NULL;

    color->r = rgba[0];
    color->g = rgba[1];
    color->b = rgba[2];
    color->a = rgba[3];

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

    if (PyString_Check (obj))
    {
        /* Named color */
        PyObject *color = NULL;
        if (obj1 || obj2 || obj3)
            return RAISE (PyExc_ValueError, "invalid arguments");
        
        color = PyDict_GetItem (_COLORDICT, obj);
        if (!color)
            return RAISE (PyExc_ValueError, "invalid color name");
        if (!RGBAFromObj (color, rgba))
            return RAISE (PyExc_ValueError, "invalid color");

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
    color->ob_type->tp_free ((PyObject *) color);
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
    return PyString_FromString (buf);
}

/**
 * color.notmalize ()
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

    rgba[0] = (frgba[0] > 1.0) ? 255 : ((frgba[0] < 0.0) ? 0 :
        (Uint8) round (frgba[0] * 255));
    rgba[1] = (frgba[1] > 1.0) ? 255 : ((frgba[1] < 0.0) ? 0 :
        (Uint8) round (frgba[1] * 255));
    rgba[2] = (frgba[2] > 1.0) ? 255 : ((frgba[2] < 0.0) ? 0 :
        (Uint8) round (frgba[2] * 255));
    rgba[3] = (frgba[3] > 1.0) ? 255 : ((frgba[3] < 0.0) ? 0 :
        (Uint8) round (frgba[3] * 255));
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
    double hsv[3];
    double frgb[4];
    double minv, maxv, diff;

    /* Normalize */
    frgb[0] = color->r / 255.0;
    frgb[1] = color->g / 255.0;
    frgb[2] = color->b / 255.0;
    frgb[3] = color->a / 255.0;

    maxv = MAX (MAX (frgb[0], frgb[1]), frgb[2]);
    minv = MIN (MIN (frgb[0], frgb[1]), frgb[2]);

    if (minv == maxv)
    {
        hsv[0] = 0;
        hsv[1] = 0;
        hsv[2] = maxv;
    }
    else
    {
        diff = maxv - minv;
        hsv[1] = (maxv == 0) ? 0 : diff / maxv;
        hsv[2] = maxv;
        
        if (frgb[0] == maxv)
            hsv[0] = (frgb[1] - frgb[2]) / diff;
        else if (frgb[1] == maxv)
            hsv[0] = 2.0 + (frgb[2] - frgb[0]) / diff;
        else
            hsv[0] = 4.0 + (frgb[0] - frgb[1]) / diff;
        hsv[0] = hsv[0] / 6.0;
    }

    /* H,S,V,A */
    return Py_BuildValue ("(ffff)", hsv[0], hsv[1], hsv[2], frgb[3]);
}

static int
_color_set_hsva (PyColor *color, PyObject *value, void *closure)
{
    PyObject *item;
    float hsva[4] = { 0, 0, 0, 0 };
    float h, f, p, q, t, v;

    if (!PySequence_Check (value) || PySequence_Size (value) < 3)
    {
        PyErr_SetString (PyExc_ValueError, "invalid HSVA value");
        return -1;
    }

    /* H */
    item = PySequence_GetItem (value, 0);
    if (!item || !FloatFromObj (item, &(hsva[0])) || hsva[0] < 0 || hsva[0] > 1)
    {
        Py_XDECREF (item);
        PyErr_SetString (PyExc_ValueError, "invalid HSVA value");
        return -1;
    }

    /* S */
    item = PySequence_GetItem (value, 1);
    if (!item || !FloatFromObj (item, &(hsva[1])) || hsva[1] < 0 || hsva[1] > 1)
    {
        Py_XDECREF (item);
        PyErr_SetString (PyExc_ValueError, "invalid HSVA value");
        return -1;
    }

    /* V */
    item = PySequence_GetItem (value, 2);
    if (!item || !FloatFromObj (item, &(hsva[2])) || hsva[2] < 0 || hsva[2] > 1)
    {
        Py_XDECREF (item);
        PyErr_SetString (PyExc_ValueError, "invalid HSVA value");
        return -1;
    }

    /* A */
    if (PySequence_Size (value) > 3)
    {
        item = PySequence_GetItem (value, 3);
        if (!item || !FloatFromObj (item, &(hsva[3])) ||
            hsva[3] < 0 || hsva[3] > 1)
        {
            Py_DECREF (item);
            PyErr_SetString (PyExc_ValueError, "invalid HSVA value");
            return -1;
        }
    }

    color->a = (Uint8) (hsva[3] * 255);

    v = (Uint8) (hsva[2] * 255);
    if (hsva[1] == 0)
    {
        color->r = v;
        color->g = v;
        color->b = v;
        return 0;
    }

    h = floor (hsva[0] * 6);
    f = hsva[0] * 6 - h;
    p = hsva[2] * (1.0 - hsva[1]);
    q = hsva[2] * (1.0 - hsva[1] * f);
    t = hsva[2] * (1.0 - hsva[1] * (1.0 - f));

    switch (((int)h) % 6)
    {
    case 0:
        color->r = v;
        color->g = (Uint8) (t * 255);
        color->b = (Uint8) (p * 255);
        break;
    case 1:
        color->r = (Uint8) (q * 255);
        color->g = v;
        color->b = (Uint8) (p * 255);
        break;
    case 2:
        color->r = (Uint8) (p * 255);
        color->g = v;
        color->b = (Uint8) (t * 255);
        break;
    case 3:
        color->r = (Uint8) (p * 255);
        color->g = (Uint8) (q * 255);
        color->b = v;
        break;
    case 4:
        color->r = (Uint8) (t * 255);
        color->g = (Uint8) (p * 255);
        color->b = v;
        break;
    case 5:
        color->r = v;
        color->g = (Uint8) (p * 255);
        color->b = (Uint8) (q * 255);
        break;
    default:
        PyErr_SetString (PyExc_ValueError, "unpredictable error");
        return -1;
    }
    return 0;
}

/**
 * color.hlsa
 */
static PyObject*
_color_get_hlsa (PyColor *color, void *closure)
{
    double hls[3] = { 0, 0, 0 };
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
    hls[1] = (maxv + minv) / 2.0;
    if (maxv == minv)
    {
        hls[0] = 0;
        hls[2] = 0;
        return Py_BuildValue ("(ffff)", hls[0], hls[1], hls[2], frgb[3]);
    }
    else if (hls[1] <= 0.5)
    {
        hls[2] = diff / (maxv + minv);
    }
    else
    {
        hls[2] = diff / (2.0 - maxv - minv);
    }

    
    if (frgb[0] == maxv)
    {
        hls[0] = (frgb[1] - frgb[2]) / diff;
    }
    else if (frgb[1] == maxv)
    {
        hls[0] = 2.0 + (frgb[2] - frgb[0]) / diff;
    }
    else
    {
        hls[0] = 4.0 + (frgb[0] - frgb[1]) / diff;
    }
    hls[0] = hls[0] / 6.0;

    
    /* H,L,S,A */
    return Py_BuildValue ("(ffff)", hls[0], hls[1], hls[2], frgb[3]);
}

/**
 * color.hlsa = x
 */
static int
_color_set_hlsa (PyColor *color, PyObject *value, void *closure)
{
    PyObject *item;
    float hlsa[4] = { 0, 0, 0, 0 };
    float h, q, p = 0;

    if (!PySequence_Check (value) || PySequence_Size (value) < 3)
    {
        PyErr_SetString (PyExc_ValueError, "invalid HLSA value");
        return -1;
    }

    /* H */
    item = PySequence_GetItem (value, 0);
    if (!item || !FloatFromObj (item, &(hlsa[0])) || hlsa[0] < 0 || hlsa[0] > 1)
    {
        Py_XDECREF (item);
        PyErr_SetString (PyExc_ValueError, "invalid HLSA value");
        return -1;
    }

    /* L */
    item = PySequence_GetItem (value, 1);
    if (!item || !FloatFromObj (item, &(hlsa[1])) || hlsa[1] < 0 || hlsa[1] > 1)
    {
        Py_XDECREF (item);
        PyErr_SetString (PyExc_ValueError, "invalid HLSA value");
        return -1;
    }

    /* S */
    item = PySequence_GetItem (value, 2);
    if (!item || !FloatFromObj (item, &(hlsa[2])) || hlsa[2] < 0 || hlsa[2] > 1)
    {
        Py_XDECREF (item);
        PyErr_SetString (PyExc_ValueError, "invalid HLSA value");
        return -1;
    }

    /* A */
    if (PySequence_Size (value) > 3)
    {
        item = PySequence_GetItem (value, 3);
        if (!item || !FloatFromObj (item, &(hlsa[3])) ||
            hlsa[3] < 0 || hlsa[3] > 1)
        {
            Py_DECREF (item);
            PyErr_SetString (PyExc_ValueError, "invalid HLSA value");
            return -1;
        }
    }

    color->a = (Uint8) (hlsa[3] * 255);

    if (hlsa[2] == 0)
    {
        color->r = (Uint8) hlsa[1] * 255;
        color->g = (Uint8) hlsa[1] * 255;
        color->b = (Uint8) hlsa[1] * 255;
    }
    else if (hlsa[1] <= 0.5)
        p = hlsa[1] * (1.0 + hlsa[2]);
    else
        p = hlsa[1] + hlsa[2] - (hlsa[1] * hlsa[2]);
    q = 2.0 * hlsa[1] - p;
    
    /* R channel */
    h = hlsa[0] + 1.0 / 3.0;
    if (h < 1.0/6.0)
        color->r = (Uint8) ((q + (p - q) * h * 6.0) * 255);
    else if (h < 0.5)
        color->r = (Uint8) (p * 255);
    else if (h < 2.0 / 3.0)
        color->r = (Uint8) ((q + (p - q) * ((2.0 / 3.0) - h) * 6.0) * 255);
    else
        color->r = (Uint8) (q * 255);

    /* G channel */
    h = hlsa[0];
    if (h < 1.0/6.0)
        color->g = (Uint8) ((q + (p - q) * h * 6.0) * 255);
    else if (h < 0.5)
        color->g = (Uint8) (p * 255);
    else if (h < 2.0 / 3.0)
        color->g = (Uint8) ((q + (p - q) * ((2.0 / 3.0) - h) * 6.0) * 255);
    else
        color->g = (Uint8) (q * 255);

    /* B channel */
    h = hlsa[0] - 1.0 / 3.0;
    if (h < 1.0/6.0)
        color->b = (Uint8) ((q + (p - q) * h * 6.0) * 255);
    else if (h < 0.5)
        color->b = (Uint8) (p * 255);
    else if (h < 2.0 / 3.0)
        color->b = (Uint8) ((q + (p - q) * ((2.0 / 3.0) - h) * 6.0) * 255);
    else
        color->b = (Uint8) (q * 255);

    return 0;
}

/**
 * color.yuv
 */
static PyObject*
_color_get_yuv (PyColor *color, void *closure)
{
    double yuv[3] = { 0, 0, 0 };
    double frgb[3];

    /* Normalize */
    frgb[0] = color->r / 255.0;
    frgb[1] = color->g / 255.0;
    frgb[2] = color->b / 255.0;

    yuv[0] = 0.299 * frgb[0] + 0.587 * frgb[1] + 0.114 * frgb[2];
    yuv[1] = 0.493 * (frgb[2] - yuv[0]);
    yuv[2] = 0.877 * (frgb[0] - yuv[0]);

    /* Y,U,V */
    return Py_BuildValue ("(fff)", yuv[0], yuv[1], yuv[2]);
}

/**
 * color.yuv = x
 */
static int
_color_set_yuv (PyColor *color, PyObject *value, void *closure)
{
    PyObject *item;
    float yuv[3] = { 0, 0, 0 };

    if (!PySequence_Check (value) || PySequence_Size (value) < 3)
    {
        PyErr_SetString (PyExc_ValueError, "invalid YUV value");
        return -1;
    }

    /* Y */
    item = PySequence_GetItem (value, 0);
    if (!item || !FloatFromObj (item, &(yuv[0])) || yuv[0] < 0 || yuv[0] > 1)
    {
        Py_XDECREF (item);
        PyErr_SetString (PyExc_ValueError, "invalid YUV value");
        return -1;
    }

    /* U */
    item = PySequence_GetItem (value, 1);
    if (!item || !FloatFromObj (item, &(yuv[1])) || yuv[1] < 0 || yuv[1] > 1)
    {
        Py_XDECREF (item);
        PyErr_SetString (PyExc_ValueError, "invalid YUV value");
        return -1;
    }

    /* V */
    item = PySequence_GetItem (value, 2);
    if (!item || !FloatFromObj (item, &(yuv[2])) || yuv[2] < 0 || yuv[2] > 1)
    {
        Py_XDECREF (item);
        PyErr_SetString (PyExc_ValueError, "invalid YUV value");
        return -1;
    }

    color->r = (Uint8) ((yuv[0] + yuv[2] / 0.877) * 255);
    color->g = (Uint8) ((yuv[0] - 0.39466 * yuv[1] - 0.5806 * yuv[2]) * 255);
    color->b = (Uint8) ((yuv[0] + yuv[2] / 0.493) * 255);
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
    unsigned long tmp = (color->r << 24) + (color->g << 16) + (color->b << 8) +
        color->a;
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
    unsigned long tmp = (color->r << 24) + (color->g << 16) + (color->b << 8) +
        color->a;
    return PyLong_FromUnsignedLong (tmp);
}

/**
 * float(color)
 */
static PyObject*
_color_float (PyColor *color)
{
    unsigned long tmp = (color->r << 24) + (color->g << 16) + (color->b << 8) +
        color->a;
    return PyFloat_FromDouble ((double) tmp);
}

/**
 * oct(color)
 */
static PyObject*
_color_oct (PyColor *color)
{
    char buf[100];
    unsigned long tmp = (color->r << 24) + (color->g << 16) + (color->b << 8) +
        color->a;
    if (tmp < INT_MAX)
        PyOS_snprintf (buf, sizeof (buf), "0%lo", tmp);
    else
        PyOS_snprintf (buf, sizeof (buf), "0%loL", tmp);
    return PyString_FromString (buf);
}

/**
 * hex(color)
 */
static PyObject*
_color_hex (PyColor *color)
{
    char buf[100];
    unsigned long tmp = (color->r << 24) + (color->g << 16) + (color->b << 8) +
        color->a;
    if (tmp < INT_MAX)
        PyOS_snprintf (buf, sizeof (buf), "0x%lx", tmp);
    else
        PyOS_snprintf (buf, sizeof (buf), "0x%lxL", tmp);
    return PyString_FromString (buf);
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
        return RAISE (PyExc_IndexError, "invalid index");
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

/**** C API interfaces ****/
static PyObject*
PyColor_New (Uint8 rgba[])
{
    return (PyObject *) _color_new_internal (&PyColor_Type, rgba);
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

PYGAME_EXPORT
void initcolor (void)
{
    PyObject *colordict;
    PyObject *module;
    PyObject *dict;
    PyObject *apiobj;
    static void* c_api[PYGAMEAPI_COLOR_NUMSLOTS];

    if (PyType_Ready (&PyColor_Type) < 0)
        return;
    
    /* create the module */
    module = Py_InitModule3 ("color", NULL, "color module for pygame");
    PyColor_Type.tp_getattro = PyObject_GenericGetAttr;
    Py_INCREF (&PyColor_Type);
    PyModule_AddObject (module, "Color", (PyObject *) &PyColor_Type);
    dict = PyModule_GetDict (module);

    colordict = PyImport_ImportModule ("pygame.colordict");
    if (colordict)
    {
        PyObject *_dict = PyModule_GetDict (colordict);
        PyObject *colors = PyDict_GetItemString (_dict, "THECOLORS");
        Py_INCREF (colors);
        Py_INCREF (colors);
        _COLORDICT = colors;
        PyModule_AddObject (module, "THECOLORS", colors);
        Py_DECREF (colordict);
    }

    import_pygame_base ();

    c_api[0] = &PyColor_Type;
    c_api[1] = PyColor_New;
    c_api[2] = RGBAFromColorObj;

    apiobj = PyCObject_FromVoidPtr (c_api, NULL);
    PyDict_SetItemString (dict, PYGAMEAPI_LOCAL_ENTRY, apiobj);
    Py_DECREF (apiobj);
}
