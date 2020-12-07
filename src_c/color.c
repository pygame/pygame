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

/* The follow bug was reported for the pygame.math module:
 *
 * Adjust gcc 4.4 optimization for floating point on x86-32 PCs running Linux.
 * This addresses bug 52:
 * http://pygame.motherhamster.org/bugzilla/show_bug.cgi?id=52
 *
 * Apparently, the same problem plagues pygame.color, as it failed the
 * test_hsva__all_elements_within_limits and
 * test_hsva__sanity_testing_converted_should_not_raise test cases due
 * to slight, on the order of 10e-14, discrepencies in calculated double
 * values.
 */
#if defined(__GNUC__) && defined(__linux__) && defined(__i386__) && \
    __SIZEOF_POINTER__ == 4 && __GNUC__ == 4 && __GNUC_MINOR__ >= 4
#pragma GCC optimize("float-store")
#endif

#define PYGAMEAPI_COLOR_INTERNAL

#include "doc/color_doc.h"

#include "pygame.h"

#include "pgcompat.h"

#include <ctype.h>


#if (!defined(__STDC_VERSION__) || __STDC_VERSION__ < 199901L) && !defined(round)
#define pg_round(d) (((d < 0) ? (ceil((d)-0.5)) : (floor((d)+0.5))))
#else
#define pg_round(d) round(d)
#endif

typedef enum { TRISTATE_SUCCESS, TRISTATE_FAIL, TRISTATE_ERROR } tristate;

static PyObject *_COLORDICT = NULL;

static int
_get_double(PyObject *, double *);
static int
_get_color(PyObject *, Uint32 *);
static int
_hextoint(char *, Uint8 *);
static tristate
_hexcolor(PyObject *, Uint8 *);
static int
_coerce_obj(PyObject *, Uint8 *);

static pgColorObject *
_color_new_internal(PyTypeObject *, const Uint8 *);
static pgColorObject *
_color_new_internal_length(PyTypeObject *, const Uint8 *, Uint8);

static PyObject *
_color_new(PyTypeObject *, PyObject *, PyObject *);
static int
_color_init(pgColorObject *, PyObject *, PyObject *);
static void
_color_dealloc(pgColorObject *);
static PyObject *
_color_repr(pgColorObject *);
static PyObject *
_color_normalize(pgColorObject *, PyObject *);
static PyObject *
_color_correct_gamma(pgColorObject *, PyObject *);
static PyObject *
_color_set_length(pgColorObject *, PyObject *);
static PyObject *
_color_lerp(pgColorObject *, PyObject *, PyObject *);
static PyObject *
_premul_alpha(pgColorObject *, PyObject *);
static PyObject *
_color_update(pgColorObject *, PyObject *, PyObject *);

/* Getters/setters */
static PyObject *
_color_get_r(pgColorObject *, void *);
static int
_color_set_r(pgColorObject *, PyObject *, void *);
static PyObject *
_color_get_g(pgColorObject *, void *);
static int
_color_set_g(pgColorObject *, PyObject *, void *);
static PyObject *
_color_get_b(pgColorObject *, void *);
static int
_color_set_b(pgColorObject *, PyObject *, void *);
static PyObject *
_color_get_a(pgColorObject *, void *);
static int
_color_set_a(pgColorObject *, PyObject *, void *);
static PyObject *
_color_get_hsva(pgColorObject *, void *);
static int
_color_set_hsva(pgColorObject *, PyObject *, void *);
static PyObject *
_color_get_hsla(pgColorObject *, void *);
static int
_color_set_hsla(pgColorObject *, PyObject *, void *);
static PyObject *
_color_get_i1i2i3(pgColorObject *, void *);
static int
_color_set_i1i2i3(pgColorObject *, PyObject *, void *);
static PyObject *
_color_get_cmy(pgColorObject *, void *);
static int
_color_set_cmy(pgColorObject *, PyObject *, void *);
static PyObject *
_color_get_arraystruct(pgColorObject *, void *);

/* Number protocol methods */
static PyObject *
_color_add(PyObject *, PyObject *);
static PyObject *
_color_sub(PyObject *, PyObject *);
static PyObject *
_color_mul(PyObject *, PyObject *);
static PyObject *
_color_div(PyObject *, PyObject *);
static PyObject *
_color_mod(PyObject *, PyObject *);
static PyObject *
_color_inv(pgColorObject *);
static PyObject *
_color_int(pgColorObject *);
static PyObject *
_color_float(pgColorObject *);
#if !PY3
static PyObject *
_color_long(pgColorObject *);
static PyObject *
_color_oct(pgColorObject *);
static PyObject *
_color_hex(pgColorObject *);
#endif

/* Sequence protocol methods */
static Py_ssize_t
_color_length(pgColorObject *);
static PyObject *
_color_item(pgColorObject *, Py_ssize_t);
static int
_color_ass_item(pgColorObject *, Py_ssize_t, PyObject *);
static PyObject *
_color_slice(register pgColorObject *, register Py_ssize_t,
             register Py_ssize_t);

/* Mapping protocol methods. */
static PyObject *
_color_subscript(pgColorObject *, PyObject *);
static int
_color_set_slice(pgColorObject *, PyObject *, PyObject *);

/* Comparison */
static PyObject *
_color_richcompare(PyObject *, PyObject *, int);

/* New buffer protocol methods. */
static int
_color_getbuffer(pgColorObject *, Py_buffer *, int);

/* C API interfaces */
static PyObject *
pgColor_New(Uint8 rgba[]);
static PyObject *
pgColor_NewLength(Uint8 rgba[], Uint8 length);
static int
pg_RGBAFromColorObj(PyObject *color, Uint8 rgba[]);
static int
pg_RGBAFromFuzzyColorObj(PyObject *color, Uint8 rgba[]);


/**
 * Methods, which are bound to the pgColorObject type.
 */
static PyMethodDef _color_methods[] = {
    {"normalize", (PyCFunction)_color_normalize, METH_NOARGS,
     DOC_COLORNORMALIZE},
    {"correct_gamma", (PyCFunction)_color_correct_gamma, METH_VARARGS,
     DOC_COLORCORRECTGAMMA},
    {"set_length", (PyCFunction)_color_set_length, METH_VARARGS,
     DOC_COLORSETLENGTH},
    {"lerp", (PyCFunction)_color_lerp, METH_VARARGS | METH_KEYWORDS,
     DOC_COLORLERP},
    {"premul_alpha", (PyCFunction)_premul_alpha, METH_NOARGS,
     DOC_COLORPREMULALPHA},
    {"update", (PyCFunction)_color_update, METH_VARARGS,
     DOC_COLORUPDATE},
    {NULL, NULL, 0, NULL}};

/**
 * Getters and setters for the pgColorObject.
 */
static PyGetSetDef _color_getsets[] = {
    {"r", (getter)_color_get_r, (setter)_color_set_r, DOC_COLORR, NULL},
    {"g", (getter)_color_get_g, (setter)_color_set_g, DOC_COLORG, NULL},
    {"b", (getter)_color_get_b, (setter)_color_set_b, DOC_COLORB, NULL},
    {"a", (getter)_color_get_a, (setter)_color_set_a, DOC_COLORA, NULL},
    {"hsva", (getter)_color_get_hsva, (setter)_color_set_hsva, DOC_COLORHSVA,
     NULL},
    {"hsla", (getter)_color_get_hsla, (setter)_color_set_hsla, DOC_COLORHSLA,
     NULL},
    {"i1i2i3", (getter)_color_get_i1i2i3, (setter)_color_set_i1i2i3,
     DOC_COLORI1I2I3, NULL},
    {"cmy", (getter)_color_get_cmy, (setter)_color_set_cmy, DOC_COLORCMY,
     NULL},
    {"__array_struct__", (getter)_color_get_arraystruct, NULL,
     "array structure interface, read only", NULL},
    {NULL, NULL, NULL, NULL, NULL}};

static PyNumberMethods _color_as_number = {
    (binaryfunc)_color_add, /* nb_add */
    (binaryfunc)_color_sub, /* nb_subtract */
    (binaryfunc)_color_mul, /* nb_multiply */
#if !PY3
    (binaryfunc)_color_div, /* nb_divide */
#endif
    (binaryfunc)_color_mod, /* nb_remainder */
    NULL,                   /* nb_divmod */
    NULL,                   /* nb_power */
    NULL,                   /* nb_negative */
    NULL,                   /* nb_positive */
    NULL,                   /* nb_absolute */
    NULL,                   /* nb_nonzero / nb_bool*/
    (unaryfunc)_color_inv,  /* nb_invert */
    NULL,                   /* nb_lshift */
    NULL,                   /* nb_rshift */
    NULL,                   /* nb_and */
    NULL,                   /* nb_xor */
    NULL,                   /* nb_or */
#if !PY3
    NULL, /* nb_coerce */
#endif
    (unaryfunc)_color_int, /* nb_int */
#if PY3
    NULL, /* nb_reserved */
#else
    (unaryfunc)_color_long, /* nb_long */
#endif
    (unaryfunc)_color_float, /* nb_float */
#if !PY3
    (unaryfunc)_color_oct, /* nb_oct */
    (unaryfunc)_color_hex, /* nb_hex */
#endif
    NULL, /* nb_inplace_add */
    NULL, /* nb_inplace_subtract */
    NULL, /* nb_inplace_multiply */
#if !PY3
    NULL, /* nb_inplace_divide */
#endif
    NULL,                   /* nb_inplace_remainder */
    NULL,                   /* nb_inplace_power */
    NULL,                   /* nb_inplace_lshift */
    NULL,                   /* nb_inplace_rshift */
    NULL,                   /* nb_inplace_and */
    NULL,                   /* nb_inplace_xor */
    NULL,                   /* nb_inplace_or */
    (binaryfunc)_color_div, /* nb_floor_divide */
    NULL,                   /* nb_true_divide */
    NULL,                   /* nb_inplace_floor_divide */
    NULL,                   /* nb_inplace_true_divide */
    (unaryfunc)_color_int,  /* nb_index */
};

/**
 * Sequence interface support for pgColorObject.
 */
static PySequenceMethods _color_as_sequence = {
    (lenfunc)_color_length,           /* sq_length */
    NULL,                             /* sq_concat */
    NULL,                             /* sq_repeat */
    (ssizeargfunc)_color_item,        /* sq_item */
    (ssizessizeargfunc)_color_slice,  /* sq_slice */
    (ssizeobjargproc)_color_ass_item, /* sq_ass_item */
    NULL,                             /* sq_ass_slice */
    NULL,                             /* sq_contains */
    NULL,                             /* sq_inplace_concat */
    NULL,                             /* sq_inplace_repeat */
};

static PyMappingMethods _color_as_mapping = {
    (lenfunc)_color_length,
    (binaryfunc)_color_subscript,
    (objobjargproc)_color_set_slice
};

static PyBufferProcs _color_as_buffer = {
#if HAVE_OLD_BUFPROTO
    NULL,
    NULL,
    NULL,
    NULL,
#endif
    (getbufferproc)_color_getbuffer,
    NULL};

#define COLOR_TPFLAGS_COMMON \
    (Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_CHECKTYPES)
#if PY2
#define COLOR_TPFLAGS (COLOR_TPFLAGS_COMMON | Py_TPFLAGS_HAVE_NEWBUFFER)
#else
#define COLOR_TPFLAGS COLOR_TPFLAGS_COMMON
#endif

#define DEFERRED_ADDRESS(ADDR) 0

static PyTypeObject pgColor_Type = {
    PyVarObject_HEAD_INIT(NULL,0)
    "pygame.Color",                    /* tp_name */
    sizeof(pgColorObject),             /* tp_basicsize */
    0,                                 /* tp_itemsize */
    (destructor)_color_dealloc,        /* tp_dealloc */
    0,                                 /* tp_print */
    NULL,                              /* tp_getattr */
    NULL,                              /* tp_setattr */
    NULL,                              /* tp_compare */
    (reprfunc)_color_repr,             /* tp_repr */
    &_color_as_number,                 /* tp_as_number */
    &_color_as_sequence,               /* tp_as_sequence */
    &_color_as_mapping,                /* tp_as_mapping */
    NULL,                              /* tp_hash */
    NULL,                              /* tp_call */
    NULL,                              /* tp_str */
    NULL,                              /* tp_getattro */
    NULL,                              /* tp_setattro */
    &_color_as_buffer, /* tp_as_buffer */
    COLOR_TPFLAGS,
    DOC_PYGAMECOLOR,       /* tp_doc */
    NULL,                  /* tp_traverse */
    NULL,                  /* tp_clear */
    _color_richcompare,    /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    NULL,                  /* tp_iter */
    NULL,                  /* tp_iternext */
    _color_methods,        /* tp_methods */
    NULL,                  /* tp_members */
    _color_getsets,        /* tp_getset */
    NULL,                  /* tp_base */
    NULL,                  /* tp_dict */
    NULL,                  /* tp_descr_get */
    NULL,                  /* tp_descr_set */
    0,                     /* tp_dictoffset */
    (initproc)_color_init, /* tp_init */
    NULL,                  /* tp_alloc */
    _color_new,            /* tp_new */
#ifndef __SYMBIAN32__
    NULL, /* tp_free */
    NULL, /* tp_is_gc */
    NULL, /* tp_bases */
    NULL, /* tp_mro */
    NULL, /* tp_cache */
    NULL, /* tp_subclasses */
    NULL, /* tp_weaklist */
    NULL  /* tp_del */
#endif
};

#define PyColor_Check(o) ((o)->ob_type == (PyTypeObject *)&pgColor_Type)

#define RGB_EQUALS(x, y)                                              \
    ((((pgColorObject *)x)->data[0] == ((pgColorObj *)y)->data[0]) && \
     (((pgColorObject *)x)->data[1] == ((pgColorObj *)y)->data[1]) && \
     (((pgColorObject *)x)->data[2] == ((pgColorObj *)y)->data[2]) && \
     (((pgColorObject *)x)->data[3] == ((pgColorObj *)y)->data[3]))

static int
_get_double(PyObject *obj, double *val)
{
    PyObject *floatobj;
    if (!(floatobj = PyNumber_Float(obj))) {
        return 0;
    }
    *val = PyFloat_AsDouble(floatobj);
    Py_DECREF(floatobj);
    return 1;
}

static int
_get_color(PyObject *val, Uint32 *color)
{
    if (!val || !color) {
        return 0;
    }

#if !PY3
    if (PyInt_Check(val)) {
        long intval = PyInt_AsLong(val);
        if ((intval == -1 && PyErr_Occurred()) || (intval > 0xFFFFFFFF)) {
            PyErr_SetString(PyExc_ValueError, "invalid color argument");
            return 0;
        }
        *color = (Uint32)intval;
        return 1;
    }
#endif
    if (PyLong_Check(val)) {
        unsigned long longval = PyLong_AsUnsignedLong(val);
        if (PyErr_Occurred() || (longval > 0xFFFFFFFF)) {
            PyErr_SetString(PyExc_ValueError, "invalid color argument");
            return 0;
        }
        *color = (Uint32)longval;
        return 1;
    }

    /* Failed */
    PyErr_SetString(PyExc_TypeError, "invalid color argument");
    return 0;
}

static int
_hextoint(char *hex, Uint8 *val)
{
    /* 'hex' is a two digit hexadecimal number, no spaces, no signs.
     * This algorithm is brute force, but it is character system agnostic.
     * It is definitely not a general purpose solution.
     */
    Uint8 temp = 0;

    switch (toupper(hex[0])) {
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

    switch (toupper(hex[1])) {
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
_hexcolor(PyObject *color, Uint8 rgba[])
{
    size_t len;
    tristate rcode = TRISTATE_FAIL;
    char *name;
#if PY3
    PyObject *ascii = PyUnicode_AsASCIIString(color);
    if (ascii == NULL) {
        rcode = TRISTATE_ERROR;
        goto Fail;
    }
    name = PyBytes_AsString(ascii);
#else
    name = PyString_AsString(color);
#endif
    if (name == NULL) {
        goto Fail;
    }

    len = strlen(name);
    /* hex colors can be
     * #RRGGBB
     * #RRGGBBAA
     * 0xRRGGBB
     * 0xRRGGBBAA
     */
    if (len < 7) {
        goto Fail;
    }

    if (name[0] == '#') {
        if (len != 7 && len != 9)
            goto Fail;
        if (!_hextoint(name + 1, &rgba[0]))
            goto Fail;
        if (!_hextoint(name + 3, &rgba[1]))
            goto Fail;
        if (!_hextoint(name + 5, &rgba[2]))
            goto Fail;
        rgba[3] = 255;
        if (len == 9 && !_hextoint(name + 7, &rgba[3])) {
            goto Fail;
        }
        goto Success;
    }
    else if (name[0] == '0' && name[1] == 'x') {
        if (len != 8 && len != 10)
            goto Fail;
        if (!_hextoint(name + 2, &rgba[0]))
            goto Fail;
        if (!_hextoint(name + 4, &rgba[1]))
            goto Fail;
        if (!_hextoint(name + 6, &rgba[2]))
            goto Fail;
        rgba[3] = 255;
        if (len == 10 && !_hextoint(name + 8, &rgba[3])) {
            goto Fail;
        }
        goto Success;
    }
    goto Fail;

Success:
    rcode = TRISTATE_SUCCESS;
Fail:
#if PY3
    Py_XDECREF(ascii);
#endif
    return rcode;
}

static int
_coerce_obj(PyObject *obj, Uint8 rgba[])
{
    if (PyType_IsSubtype(obj->ob_type, &pgColor_Type)) {
        rgba[0] = ((pgColorObject *)obj)->data[0];
        rgba[1] = ((pgColorObject *)obj)->data[1];
        rgba[2] = ((pgColorObject *)obj)->data[2];
        rgba[3] = ((pgColorObject *)obj)->data[3];
        return 1;
    }
    else if (PyType_IsSubtype(obj->ob_type, &PyTuple_Type)) {
        if (pg_RGBAFromObj(obj, rgba)) {
            return 1;
        }
        else if (PyErr_Occurred()) {
            return -1;
        }
    }

    return 0;
}

static pgColorObject *
_color_new_internal(PyTypeObject *type, const Uint8 rgba[])
{
    /* default length of 4 - r,g,b,a. */
    return _color_new_internal_length(type, rgba, 4);
}

static pgColorObject *
_color_new_internal_length(PyTypeObject *type, const Uint8 rgba[],
                           Uint8 length)
{
    pgColorObject *color = (pgColorObject *)type->tp_alloc(type, 0);
    if (!color) {
        return NULL;
    }

    color->data[0] = rgba[0];
    color->data[1] = rgba[1];
    color->data[2] = rgba[2];
    color->data[3] = rgba[3];
    color->len = length;

    return color;
}

/**
 * Creates a new pgColorObject.
 */
static PyObject *
_color_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    static const Uint8 DEFAULT_RGBA[4] = {0, 0, 0, 255};

    return (PyObject *)_color_new_internal_length(type, DEFAULT_RGBA, 4);
}

static int
_parse_color_from_text(PyObject *str_obj, Uint8 *rgba) {
    /* Named color */
    PyObject *color = NULL;
    PyObject *name1 = NULL, *name2 = NULL;

    /* We assume the caller handled this check for us. */
    assert(Text_Check(str_obj) || PyUnicode_Check(str_obj));

    name1 = PyObject_CallMethod(str_obj, "replace", "(ss)", " ", "");
    if (!name1) {
        return -1;
    }
    name2 = PyObject_CallMethod(name1, "lower", NULL);
    Py_DECREF(name1);
    if (!name2) {
        return -1;
    }
    color = PyDict_GetItem(_COLORDICT, name2);
    Py_DECREF(name2);
    if (!color) {
        switch (_hexcolor(str_obj, rgba)) {
            case TRISTATE_FAIL:
                PyErr_SetString(PyExc_ValueError, "invalid color name");
                return -1;
            case TRISTATE_ERROR:
                return -1;
            default:
                break;
        }
    } else if (!pg_RGBAFromObj(color, rgba)) {
        PyErr_SetString(PyExc_ValueError, "invalid color");
        return -1;
    }
    return 0;
}

static int
_parse_color_from_single_object(PyObject *obj, Uint8 *rgba) {

    if (Text_Check(obj) || PyUnicode_Check(obj)) {
        if (_parse_color_from_text(obj, rgba)) {
            return -1;
        }
    } else {
        /* At this point color is either tuple-like or a single integer. */
        if (!pg_RGBAFromColorObj(obj, rgba)) {
            /* Color is not a valid tuple-like. */
            Uint32 color;
            if (PyTuple_Check(obj) || PySequence_Check(obj)) {
                /* It was a tuple-like; raise a ValueError
                 * - if we pass it to _get_color, we will get a TypeError
                 *   instead, which is wrong.  The type is correct, but it
                 *   had the wrong number of arguments.
                 */
                PyErr_SetString(PyExc_ValueError, "invalid color argument");
                return -1;
            }

            if (_get_color(obj, &color)) {
                /* Color is a single integer. */
                rgba[0] = (Uint8)(color >> 24);
                rgba[1] = (Uint8)(color >> 16);
                rgba[2] = (Uint8)(color >> 8);
                rgba[3] = (Uint8)color;
            } else {
                /* Exception already set by _get_color(). */
                return -1;
            }
        }
    }
    return 0;
}

static int
_color_init(pgColorObject *self, PyObject *args, PyObject *kwds)
{
    Uint8 *rgba = self->data;
    PyObject *obj;
    PyObject *obj1 = NULL;
    PyObject *obj2 = NULL;
    PyObject *obj3 = NULL;

    if (!PyArg_ParseTuple(args, "O|OOO", &obj, &obj1, &obj2, &obj3)) {
        return -1;
    }

    if (!obj1) {
        if (_parse_color_from_single_object(obj, rgba)) {
            return -1;
        }
    } else {
        Uint32 color = 0;

        /* Color(R,G,B[,A]) */
        if (!_get_color(obj, &color) || color > 255) {
            PyErr_SetString(PyExc_ValueError, "invalid color argument");
            return -1;
        }
        rgba[0] = (Uint8)color;
        if (!_get_color(obj1, &color) || color > 255) {
            PyErr_SetString(PyExc_ValueError, "invalid color argument");
            return -1;
        }
        rgba[1] = (Uint8)color;
        if (!obj2 || !_get_color(obj2, &color) || color > 255) {
            PyErr_SetString(PyExc_ValueError, "invalid color argument");
            return -1;
        }
        rgba[2] = (Uint8)color;

        if (obj3) {
            if (!_get_color(obj3, &color) || color > 255) {
                PyErr_SetString(PyExc_ValueError, "invalid color argument");
                return -1;
            }
            rgba[3] = (Uint8)color;
        }
        else { /* No alpha */
            rgba[3] = 255;
        }
    }

    self->len = 4;
    return 0;
}

/**
 * Deallocates the pgColorObject.
 */
static void
_color_dealloc(pgColorObject *color)
{
    Py_TYPE(color)->tp_free((PyObject *)color);
}

/**
 * repr(color)
 */
static PyObject *
_color_repr(pgColorObject *color)
{
    /* Max. would be(255, 255, 255, 255) */
    char buf[21];
    PyOS_snprintf(buf, sizeof(buf), "(%d, %d, %d, %d)", color->data[0],
                  color->data[1], color->data[2], color->data[3]);
    return Text_FromUTF8(buf);
}

/**
 * color.normalize()
 */
static PyObject *
_color_normalize(pgColorObject *color, PyObject *args)
{
    double rgba[4];
    rgba[0] = color->data[0] / 255.0;
    rgba[1] = color->data[1] / 255.0;
    rgba[2] = color->data[2] / 255.0;
    rgba[3] = color->data[3] / 255.0;
    return Py_BuildValue("(ffff)", rgba[0], rgba[1], rgba[2], rgba[3]);
}

/**
 * color.correct_gamma(x)
 */
static PyObject *
_color_correct_gamma(pgColorObject *color, PyObject *args)
{
    double frgba[4];
    Uint8 rgba[4];
    double _gamma;

    if (!PyArg_ParseTuple(args, "d", &_gamma)) {
        return NULL;
    }

    frgba[0] = pow(color->data[0] / 255.0, _gamma);
    frgba[1] = pow(color->data[1] / 255.0, _gamma);
    frgba[2] = pow(color->data[2] / 255.0, _gamma);
    frgba[3] = pow(color->data[3] / 255.0, _gamma);

    /* visual studio doesn't have a round func, so doing it with +.5 and
     * truncaction */
    rgba[0] = (frgba[0] > 1.0)
                  ? 255
                  : ((frgba[0] < 0.0) ? 0 : (Uint8)(frgba[0] * 255 + .5));
    rgba[1] = (frgba[1] > 1.0)
                  ? 255
                  : ((frgba[1] < 0.0) ? 0 : (Uint8)(frgba[1] * 255 + .5));
    rgba[2] = (frgba[2] > 1.0)
                  ? 255
                  : ((frgba[2] < 0.0) ? 0 : (Uint8)(frgba[2] * 255 + .5));
    rgba[3] = (frgba[3] > 1.0)
                  ? 255
                  : ((frgba[3] < 0.0) ? 0 : (Uint8)(frgba[3] * 255 + .5));
    return (PyObject *)_color_new_internal(Py_TYPE(color), rgba);
}

/**
 * color.lerp(other, x)
 */
static PyObject *
_color_lerp(pgColorObject *self, PyObject *args, PyObject *kw)
{
    Uint8 rgba[4];
    Uint8 new_rgba[4];
    PyObject* colobj;
    double amt;
    static char *keywords[] = {"color", "amount", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kw, "Od", keywords,
                                     &colobj, &amt)) {
        return NULL;
    }

    if (!pg_RGBAFromFuzzyColorObj(colobj, rgba)) {
        /* Exception already set for us */
        return NULL;
    }

    if (amt < 0 || amt > 1) {
        return RAISE(PyExc_ValueError,
                        "Argument 2 must be in range [0, 1]");
    }

    new_rgba[0] = (Uint8)pg_round(self->data[0] * (1 - amt) + rgba[0] * amt);
    new_rgba[1] = (Uint8)pg_round(self->data[1] * (1 - amt) + rgba[1] * amt);
    new_rgba[2] = (Uint8)pg_round(self->data[2] * (1 - amt) + rgba[2] * amt);
    new_rgba[3] = (Uint8)pg_round(self->data[3] * (1 - amt) + rgba[3] * amt);

    return (PyObject *)_color_new_internal(Py_TYPE(self), new_rgba);
}

/**
 * color.premul_alpha()
 */
static PyObject *
_premul_alpha(pgColorObject *color, PyObject *args)
{
    Uint8 new_rgba[4];
    new_rgba[0] = (Uint8)(((color->data[0] + 1) * color->data[3]) >> 8);
    new_rgba[1] = (Uint8)(((color->data[1] + 1) * color->data[3]) >> 8);
    new_rgba[2] = (Uint8)(((color->data[2] + 1) * color->data[3]) >> 8);
    new_rgba[3] = color->data[3];

    return (PyObject *)_color_new_internal(Py_TYPE(color), new_rgba);
}

static PyObject *
_color_update(pgColorObject *self, PyObject *args, PyObject *kwargs)
{
    Uint8 *rgba = self->data;
    PyObject *r_or_obj;
    PyObject *g = NULL;
    PyObject *b = NULL;
    PyObject *a = NULL;

    if (!PyArg_ParseTuple(args, "O|OOO", &r_or_obj, &g, &b, &a)) {
        return NULL;
    }

    if (!g) {
        if (_parse_color_from_single_object(r_or_obj, rgba)) {
            return NULL;
        }
    }
    else {
        Uint32 color = 0;

        /* Color(R,G,B[,A]) */
        if (!_get_color(r_or_obj, &color) || color > 255) {
            return RAISE(PyExc_ValueError, "invalid color argument");
        }
        rgba[0] = (Uint8)color;
        if (!_get_color(g, &color) || color > 255) {
            return RAISE(PyExc_ValueError, "invalid color argument");
        }
        rgba[1] = (Uint8)color;
        if (!b || !_get_color(b, &color) || color > 255) {
            return RAISE(PyExc_ValueError, "invalid color argument");
        }
        rgba[2] = (Uint8)color;

        if (a) {
            if (!_get_color(a, &color) || color > 255) {
                return RAISE(PyExc_ValueError, "invalid color argument");
            }
            self->len = 4;
            rgba[3] = (Uint8)color;
        }
    }
    Py_RETURN_NONE;
}

/**
 * color.r
 */
static PyObject *
_color_get_r(pgColorObject *color, void *closure)
{
    return PyInt_FromLong(color->data[0]);
}

/**
 * color.r = x
 */
static int
_color_set_r(pgColorObject *color, PyObject *value, void *closure)
{
    Uint32 c;

    DEL_ATTR_NOT_SUPPORTED_CHECK("r", value);

    if (!_get_color(value, &c)) {
        return -1;
    }
    if (c > 255) {
        PyErr_SetString(PyExc_ValueError, "color exceeds allowed range");
        return -1;
    }
    color->data[0] = c;
    return 0;
}

/**
 * color.g
 */
static PyObject *
_color_get_g(pgColorObject *color, void *closure)
{
    return PyInt_FromLong(color->data[1]);
}

/**
 * color.g = x
 */
static int
_color_set_g(pgColorObject *color, PyObject *value, void *closure)
{
    Uint32 c;

    DEL_ATTR_NOT_SUPPORTED_CHECK("g", value);

    if (!_get_color(value, &c)) {
        return -1;
    }
    if (c > 255) {
        PyErr_SetString(PyExc_ValueError, "color exceeds allowed range");
        return -1;
    }
    color->data[1] = c;
    return 0;
}

/**
 * color.b
 */
static PyObject *
_color_get_b(pgColorObject *color, void *closure)
{
    return PyInt_FromLong(color->data[2]);
}

/**
 * color.b = x
 */
static int
_color_set_b(pgColorObject *color, PyObject *value, void *closure)
{
    Uint32 c;

    DEL_ATTR_NOT_SUPPORTED_CHECK("b", value);

    if (!_get_color(value, &c)) {
        return -1;
    }
    if (c > 255) {
        PyErr_SetString(PyExc_ValueError, "color exceeds allowed range");
        return -1;
    }
    color->data[2] = c;
    return 0;
}

/**
 * color.a
 */
static PyObject *
_color_get_a(pgColorObject *color, void *closure)
{
    return PyInt_FromLong(color->data[3]);
}

/**
 * color.a = x
 */
static int
_color_set_a(pgColorObject *color, PyObject *value, void *closure)
{
    Uint32 c;

    DEL_ATTR_NOT_SUPPORTED_CHECK("a", value);

    if (!_get_color(value, &c)) {
        return -1;
    }
    if (c > 255) {
        PyErr_SetString(PyExc_ValueError, "color exceeds allowed range");
        return -1;
    }
    color->data[3] = c;
    return 0;
}

/**
 * color.hsva
 */
static PyObject *
_color_get_hsva(pgColorObject *color, void *closure)
{
    double hsv[3] = {0, 0, 0};
    double frgb[4];
    double minv, maxv, diff;

    /* Normalize */
    frgb[0] = color->data[0] / 255.0;
    frgb[1] = color->data[1] / 255.0;
    frgb[2] = color->data[2] / 255.0;
    frgb[3] = color->data[3] / 255.0;

    maxv = MAX(MAX(frgb[0], frgb[1]), frgb[2]);
    minv = MIN(MIN(frgb[0], frgb[1]), frgb[2]);
    diff = maxv - minv;

    /* Calculate V */
    hsv[2] = 100. * maxv;

    if (maxv == minv) {
        hsv[0] = 0;
        hsv[1] = 0;
        return Py_BuildValue("(ffff)", hsv[0], hsv[1], hsv[2], frgb[3] * 100);
    }
    /* Calculate S */
    hsv[1] = 100. * (maxv - minv) / maxv;

    /* Calculate H */
    if (maxv == frgb[0]) {
        hsv[0] = fmod((60 * ((frgb[1] - frgb[2]) / diff)), 360.f);
    }
    else if (maxv == frgb[1]) {
        hsv[0] = (60 * ((frgb[2] - frgb[0]) / diff)) + 120.f;
    }
    else {
        hsv[0] = (60 * ((frgb[0] - frgb[1]) / diff)) + 240.f;
    }

    if (hsv[0] < 0) {
        hsv[0] += 360.f;
    }

    /* H,S,V,A */
    return Py_BuildValue("(ffff)", hsv[0], hsv[1], hsv[2], frgb[3] * 100);
}

static int
_color_set_hsva(pgColorObject *color, PyObject *value, void *closure)
{
    PyObject *item;
    double hsva[4] = {0, 0, 0, 0};
    double f, p, q, t, v, s;
    int hi;


    DEL_ATTR_NOT_SUPPORTED_CHECK("hsva", value);

    if (!PySequence_Check(value) || PySequence_Size(value) < 3) {
        PyErr_SetString(PyExc_ValueError, "invalid HSVA value");
        return -1;
    }

    /* H */
    item = PySequence_GetItem(value, 0);
    if (!item || !_get_double(item, &(hsva[0])) || hsva[0] < 0 ||
        hsva[0] > 360) {
        Py_XDECREF(item);
        PyErr_SetString(PyExc_ValueError, "invalid HSVA value");
        return -1;
    }
    Py_DECREF(item);

    /* S */
    item = PySequence_GetItem(value, 1);
    if (!item || !_get_double(item, &(hsva[1])) || hsva[1] < 0 ||
        hsva[1] > 100) {
        Py_XDECREF(item);
        PyErr_SetString(PyExc_ValueError, "invalid HSVA value");
        return -1;
    }
    Py_DECREF(item);

    /* V */
    item = PySequence_GetItem(value, 2);
    if (!item || !_get_double(item, &(hsva[2])) || hsva[2] < 0 ||
        hsva[2] > 100) {
        Py_XDECREF(item);
        PyErr_SetString(PyExc_ValueError, "invalid HSVA value");
        return -1;
    }
    Py_DECREF(item);

    /* A */
    if (PySequence_Size(value) > 3) {
        item = PySequence_GetItem(value, 3);
        if (!item || !_get_double(item, &(hsva[3])) || hsva[3] < 0 ||
            hsva[3] > 100) {
            Py_XDECREF(item);
            PyErr_SetString(PyExc_ValueError, "invalid HSVA value");
            return -1;
        }
        Py_DECREF(item);
    }

    color->data[3] = (Uint8)((hsva[3] / 100.0f) * 255);

    s = hsva[1] / 100.f;
    v = hsva[2] / 100.f;

    hi = (int)floor(hsva[0] / 60.f);
    f = (hsva[0] / 60.f) - hi;
    p = v * (1 - s);
    q = v * (1 - s * f);
    t = v * (1 - s * (1 - f));

    switch (hi) {
        case 1:
            color->data[0] = (Uint8)(q * 255);
            color->data[1] = (Uint8)(v * 255);
            color->data[2] = (Uint8)(p * 255);
            break;
        case 2:
            color->data[0] = (Uint8)(p * 255);
            color->data[1] = (Uint8)(v * 255);
            color->data[2] = (Uint8)(t * 255);
            break;
        case 3:
            color->data[0] = (Uint8)(p * 255);
            color->data[1] = (Uint8)(q * 255);
            color->data[2] = (Uint8)(v * 255);
            break;
        case 4:
            color->data[0] = (Uint8)(t * 255);
            color->data[1] = (Uint8)(p * 255);
            color->data[2] = (Uint8)(v * 255);
            break;
        case 5:
            color->data[0] = (Uint8)(v * 255);
            color->data[1] = (Uint8)(p * 255);
            color->data[2] = (Uint8)(q * 255);
            break;
        default:
            /* 0 or 6, which are equivalent. */
            assert(hi == 0 || hi == 6);
            color->data[0] = (Uint8)(v * 255);
            color->data[1] = (Uint8)(t * 255);
            color->data[2] = (Uint8)(p * 255);
    }

    return 0;
}

/**
 * color.hsla
 */
static PyObject *
_color_get_hsla(pgColorObject *color, void *closure)
{
    double hsl[3] = {0, 0, 0};
    double frgb[4];
    double minv, maxv, diff;

    /* Normalize */
    frgb[0] = color->data[0] / 255.0;
    frgb[1] = color->data[1] / 255.0;
    frgb[2] = color->data[2] / 255.0;
    frgb[3] = color->data[3] / 255.0;

    maxv = MAX(MAX(frgb[0], frgb[1]), frgb[2]);
    minv = MIN(MIN(frgb[0], frgb[1]), frgb[2]);

    diff = maxv - minv;

    /* Calculate L */
    hsl[2] = 50.f * (maxv + minv); /* 1/2 (max + min) */

    if (maxv == minv) {
        hsl[1] = 0;
        hsl[0] = 0;
        return Py_BuildValue("(ffff)", hsl[0], hsl[1], hsl[2], frgb[3] * 100);
    }

    /* Calculate S */
    if (hsl[2] <= 50) {
        hsl[1] = diff / (maxv + minv);
    }
    else {
        hsl[1] = diff / (2 - maxv - minv);
    }
    hsl[1] *= 100.f;

    /* Calculate H */
    if (maxv == frgb[0]) {
        hsl[0] = fmod((60 * ((frgb[1] - frgb[2]) / diff)), 360.f);
    }
    else if (maxv == frgb[1]) {
        hsl[0] = (60 * ((frgb[2] - frgb[0]) / diff)) + 120.f;
    }
    else {
        hsl[0] = (60 * ((frgb[0] - frgb[1]) / diff)) + 240.f;
    }
    if (hsl[0] < 0) {
        hsl[0] += 360.f;
    }

    /* H,S,L,A */
    return Py_BuildValue("(ffff)", hsl[0], hsl[1], hsl[2], frgb[3] * 100);
}

/**
 * color.hsla = x
 */
static int
_color_set_hsla(pgColorObject *color, PyObject *value, void *closure)
{
    PyObject *item;
    double hsla[4] = {0, 0, 0, 0};
    double ht, h, q, p = 0, s, l = 0;
    static double onethird = 1.0 / 3.0f;

    DEL_ATTR_NOT_SUPPORTED_CHECK("hsla", value);

    if (!PySequence_Check(value) || PySequence_Size(value) < 3) {
        PyErr_SetString(PyExc_ValueError, "invalid HSLA value");
        return -1;
    }

    /* H */
    item = PySequence_GetItem(value, 0);
    if (!item || !_get_double(item, &(hsla[0])) || hsla[0] < 0 ||
        hsla[0] > 360) {
        Py_XDECREF(item);
        PyErr_SetString(PyExc_ValueError, "invalid HSLA value");
        return -1;
    }
    Py_DECREF(item);

    /* S */
    item = PySequence_GetItem(value, 1);
    if (!item || !_get_double(item, &(hsla[1])) || hsla[1] < 0 ||
        hsla[1] > 100) {
        Py_XDECREF(item);
        PyErr_SetString(PyExc_ValueError, "invalid HSLA value");
        return -1;
    }
    Py_DECREF(item);

    /* L */
    item = PySequence_GetItem(value, 2);
    if (!item || !_get_double(item, &(hsla[2])) || hsla[2] < 0 ||
        hsla[2] > 100) {
        Py_XDECREF(item);
        PyErr_SetString(PyExc_ValueError, "invalid HSLA value");
        return -1;
    }
    Py_DECREF(item);

    /* A */
    if (PySequence_Size(value) > 3) {
        item = PySequence_GetItem(value, 3);
        if (!item || !_get_double(item, &(hsla[3])) || hsla[3] < 0 ||
            hsla[3] > 100) {
            Py_XDECREF(item);
            PyErr_SetString(PyExc_ValueError, "invalid HSLA value");
            return -1;
        }
        Py_DECREF(item);
    }

    color->data[3] = (Uint8)((hsla[3] / 100.f) * 255);

    s = hsla[1] / 100.f;
    l = hsla[2] / 100.f;

    if (s == 0) {
        color->data[0] = (Uint8)(l * 255);
        color->data[1] = (Uint8)(l * 255);
        color->data[2] = (Uint8)(l * 255);
        return 0;
    }

    if (l < 0.5f) {
        q = l * (1 + s);
    }
    else {
        q = l + s - (l * s);
    }
    p = 2 * l - q;

    ht = hsla[0] / 360.f;

    /* Calulate R */
    h = ht + onethird;
    if (h < 0) {
        h += 1;
    }
    else if (h > 1) {
        h -= 1;
    }

    if (h < 1. / 6.f) {
        color->data[0] = (Uint8)((p + ((q - p) * 6 * h)) * 255);
    }
    else if (h < 0.5f) {
        color->data[0] = (Uint8)(q * 255);
    }
    else if (h < 2. / 3.f) {
        color->data[0] = (Uint8)((p + ((q - p) * 6 * (2. / 3.f - h))) * 255);
    }
    else {
        color->data[0] = (Uint8)(p * 255);
    }

    /* Calculate G */
    h = ht;
    if (h < 0) {
        h += 1;
    }
    else if (h > 1) {
        h -= 1;
    }

    if (h < 1. / 6.f) {
        color->data[1] = (Uint8)((p + ((q - p) * 6 * h)) * 255);
    }
    else if (h < 0.5f) {
        color->data[1] = (Uint8)(q * 255);
    }
    else if (h < 2. / 3.f) {
        color->data[1] = (Uint8)((p + ((q - p) * 6 * (2. / 3.f - h))) * 255);
    }
    else {
        color->data[1] = (Uint8)(p * 255);
    }

    /* Calculate B */
    h = ht - onethird;
    if (h < 0) {
        h += 1;
    }
    else if (h > 1) {
        h -= 1;
    }

    if (h < 1. / 6.f) {
        color->data[2] = (Uint8)((p + ((q - p) * 6 * h)) * 255);
    }
    else if (h < 0.5f) {
        color->data[2] = (Uint8)(q * 255);
    }
    else if (h < 2. / 3.f) {
        color->data[2] = (Uint8)((p + ((q - p) * 6 * (2. / 3.f - h))) * 255);
    }
    else {
        color->data[2] = (Uint8)(p * 255);
    }

    return 0;
}

static PyObject *
_color_get_i1i2i3(pgColorObject *color, void *closure)
{
    double i1i2i3[3] = {0, 0, 0};
    double frgb[3];

    /* Normalize */
    frgb[0] = color->data[0] / 255.0;
    frgb[1] = color->data[1] / 255.0;
    frgb[2] = color->data[2] / 255.0;

    i1i2i3[0] = (frgb[0] + frgb[1] + frgb[2]) / 3.0f;
    i1i2i3[1] = (frgb[0] - frgb[2]) / 2.0f;
    i1i2i3[2] = (2 * frgb[1] - frgb[0] - frgb[2]) / 4.0f;

    return Py_BuildValue("(fff)", i1i2i3[0], i1i2i3[1], i1i2i3[2]);
}

static int
_color_set_i1i2i3(pgColorObject *color, PyObject *value, void *closure)
{
    PyObject *item;
    double i1i2i3[3] = {0, 0, 0};
    double ar, ag, ab;


    DEL_ATTR_NOT_SUPPORTED_CHECK("i1i2i3", value);

    /* I1 */
    item = PySequence_GetItem(value, 0);
    if (!item || !_get_double(item, &(i1i2i3[0])) || i1i2i3[0] < 0 ||
        i1i2i3[0] > 1) {
        Py_XDECREF(item);
        PyErr_SetString(PyExc_ValueError, "invalid I1I2I3 value");
        return -1;
    }
    Py_DECREF(item);

    /* I2 */
    item = PySequence_GetItem(value, 1);
    if (!item || !_get_double(item, &(i1i2i3[1])) || i1i2i3[1] < -0.5f ||
        i1i2i3[1] > 0.5f) {
        Py_XDECREF(item);
        PyErr_SetString(PyExc_ValueError, "invalid I1I2I3 value");
        return -1;
    }
    Py_DECREF(item);

    /* I3 */
    item = PySequence_GetItem(value, 2);
    if (!item || !_get_double(item, &(i1i2i3[2])) || i1i2i3[2] < -0.5f ||
        i1i2i3[2] > 0.5f) {
        Py_XDECREF(item);
        PyErr_SetString(PyExc_ValueError, "invalid I1I2I3 value");
        return -1;
    }
    Py_DECREF(item);

    ab = i1i2i3[0] - i1i2i3[1] - 2 * i1i2i3[2] / 3.f;
    ar = 2 * i1i2i3[1] + ab;
    ag = 3 * i1i2i3[0] - ar - ab;

    color->data[0] = (Uint8)(ar * 255);
    color->data[1] = (Uint8)(ag * 255);
    color->data[2] = (Uint8)(ab * 255);

    return 0;
}

static PyObject *
_color_get_cmy(pgColorObject *color, void *closure)
{
    double cmy[3] = {0, 0, 0};
    double frgb[3];

    /* Normalize */
    frgb[0] = color->data[0] / 255.0;
    frgb[1] = color->data[1] / 255.0;
    frgb[2] = color->data[2] / 255.0;

    cmy[0] = 1.0 - frgb[0];
    cmy[1] = 1.0 - frgb[1];
    cmy[2] = 1.0 - frgb[2];

    return Py_BuildValue("(fff)", cmy[0], cmy[1], cmy[2]);
}

static int
_color_set_cmy(pgColorObject *color, PyObject *value, void *closure)
{
    PyObject *item;
    double cmy[3] = {0, 0, 0};

    DEL_ATTR_NOT_SUPPORTED_CHECK("cmy", value);

    /* I1 */
    item = PySequence_GetItem(value, 0);
    if (!item || !_get_double(item, &(cmy[0])) || cmy[0] < 0 || cmy[0] > 1) {
        Py_XDECREF(item);
        PyErr_SetString(PyExc_ValueError, "invalid CMY value");
        return -1;
    }
    Py_DECREF(item);

    /* I2 */
    item = PySequence_GetItem(value, 1);
    if (!item || !_get_double(item, &(cmy[1])) || cmy[1] < 0 || cmy[1] > 1) {
        Py_XDECREF(item);
        PyErr_SetString(PyExc_ValueError, "invalid CMY value");
        return -1;
    }
    Py_DECREF(item);

    /* I2 */
    item = PySequence_GetItem(value, 2);
    if (!item || !_get_double(item, &(cmy[2])) || cmy[2] < 0 || cmy[2] > 1) {
        Py_XDECREF(item);
        PyErr_SetString(PyExc_ValueError, "invalid CMY value");
        return -1;
    }
    Py_DECREF(item);

    color->data[0] = (Uint8)((1.0 - cmy[0]) * 255);
    color->data[1] = (Uint8)((1.0 - cmy[1]) * 255);
    color->data[2] = (Uint8)((1.0 - cmy[2]) * 255);

    return 0;
}

static PyObject *
_color_get_arraystruct(pgColorObject *color, void *closure)
{
    Py_buffer view;
    PyObject *capsule;

    if (_color_getbuffer(color, &view, PyBUF_FULL_RO)) {
        return 0;
    }
    capsule = pgBuffer_AsArrayStruct(&view);
    Py_DECREF(color);
    return capsule;
}

/* Number protocol methods */

/**
 * color1 + color2
 */
static PyObject *
_color_add(PyObject *obj1, PyObject *obj2)
{
    Uint8 rgba[4];
    pgColorObject *color1 = (pgColorObject *)obj1;
    pgColorObject *color2 = (pgColorObject *)obj2;
    if (!PyObject_IsInstance(obj1, (PyObject *)&pgColor_Type) ||
        !PyObject_IsInstance(obj2, (PyObject *)&pgColor_Type)) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    rgba[0] = MIN(color1->data[0] + color2->data[0], 255);
    rgba[1] = MIN(color1->data[1] + color2->data[1], 255);
    rgba[2] = MIN(color1->data[2] + color2->data[2], 255);
    rgba[3] = MIN(color1->data[3] + color2->data[3], 255);
    return (PyObject *)_color_new_internal(Py_TYPE(obj1), rgba);
}

/**
 * color1 - color2
 */
static PyObject *
_color_sub(PyObject *obj1, PyObject *obj2)
{
    Uint8 rgba[4];
    pgColorObject *color1 = (pgColorObject *)obj1;
    pgColorObject *color2 = (pgColorObject *)obj2;
    if (!PyObject_IsInstance(obj1, (PyObject *)&pgColor_Type) ||
        !PyObject_IsInstance(obj2, (PyObject *)&pgColor_Type)) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    rgba[0] = MAX(color1->data[0] - color2->data[0], 0);
    rgba[1] = MAX(color1->data[1] - color2->data[1], 0);
    rgba[2] = MAX(color1->data[2] - color2->data[2], 0);
    rgba[3] = MAX(color1->data[3] - color2->data[3], 0);
    return (PyObject *)_color_new_internal(Py_TYPE(obj1), rgba);
}

/**
 * color1 * color2
 */
static PyObject *
_color_mul(PyObject *obj1, PyObject *obj2)
{
    Uint8 rgba[4];
    pgColorObject *color1 = (pgColorObject *)obj1;
    pgColorObject *color2 = (pgColorObject *)obj2;
    if (!PyObject_IsInstance(obj1, (PyObject *)&pgColor_Type) ||
        !PyObject_IsInstance(obj2, (PyObject *)&pgColor_Type)) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    rgba[0] = MIN(color1->data[0] * color2->data[0], 255);
    rgba[1] = MIN(color1->data[1] * color2->data[1], 255);
    rgba[2] = MIN(color1->data[2] * color2->data[2], 255);
    rgba[3] = MIN(color1->data[3] * color2->data[3], 255);
    return (PyObject *)_color_new_internal(Py_TYPE(obj1), rgba);
}

/**
 * color1 / color2
 */
static PyObject *
_color_div(PyObject *obj1, PyObject *obj2)
{
    Uint8 rgba[4] = {0, 0, 0, 0};
    pgColorObject *color1 = (pgColorObject *)obj1;
    pgColorObject *color2 = (pgColorObject *)obj2;
    if (!PyObject_IsInstance(obj1, (PyObject *)&pgColor_Type) ||
        !PyObject_IsInstance(obj2, (PyObject *)&pgColor_Type)) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    if (color2->data[0] != 0) {
        rgba[0] = color1->data[0] / color2->data[0];
    }
    if (color2->data[1] != 0) {
        rgba[1] = color1->data[1] / color2->data[1];
    }
    if (color2->data[2]) {
        rgba[2] = color1->data[2] / color2->data[2];
    }
    if (color2->data[3]) {
        rgba[3] = color1->data[3] / color2->data[3];
    }
    return (PyObject *)_color_new_internal(Py_TYPE(obj1), rgba);
}

/**
 * color1 % color2
 */
static PyObject *
_color_mod(PyObject *obj1, PyObject *obj2)
{
    Uint8 rgba[4] = {0, 0, 0, 0};
    pgColorObject *color1 = (pgColorObject *)obj1;
    pgColorObject *color2 = (pgColorObject *)obj2;
    if (!PyObject_IsInstance(obj1, (PyObject *)&pgColor_Type) ||
        !PyObject_IsInstance(obj2, (PyObject *)&pgColor_Type)) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    if (color2->data[0] != 0) {
        rgba[0] = color1->data[0] % color2->data[0];
    }
    if (color2->data[1] != 0) {
        rgba[1] = color1->data[1] % color2->data[1];
    }
    if (color2->data[2]) {
        rgba[2] = color1->data[2] % color2->data[2];
    }
    if (color2->data[3]) {
        rgba[3] = color1->data[3] % color2->data[3];
    }
    return (PyObject *)_color_new_internal(Py_TYPE(obj1), rgba);
}

/**
 * ~color
 */
static PyObject *
_color_inv(pgColorObject *color)
{
    Uint8 rgba[4];
    rgba[0] = 255 - color->data[0];
    rgba[1] = 255 - color->data[1];
    rgba[2] = 255 - color->data[2];
    rgba[3] = 255 - color->data[3];
    return (PyObject *)_color_new_internal(Py_TYPE(color), rgba);
}

/**
 * int(color)
 */
static PyObject *
_color_int(pgColorObject *color)
{
    Uint32 tmp = (color->data[0] << 24) + (color->data[1] << 16) +
                 (color->data[2] << 8) + color->data[3];
#if !PY3
#if LONG_MAX == 2147483647
    if (tmp < LONG_MAX) {
        return PyInt_FromLong((long)tmp);
    }
#endif
#endif
    return PyLong_FromUnsignedLong(tmp);
}

/**
 * float(color)
 */
static PyObject *
_color_float(pgColorObject *color)
{
    Uint32 tmp = ((color->data[0] << 24) + (color->data[1] << 16) +
                  (color->data[2] << 8) + color->data[3]);
    return PyFloat_FromDouble((double)tmp);
}

#if !PY3
/**
 * long(color)
 */
static PyObject *
_color_long(pgColorObject *color)
{
    Uint32 tmp = ((color->data[0] << 24) + (color->data[1] << 16) +
                  (color->data[2] << 8) + color->data[3]);
    return PyLong_FromUnsignedLong(tmp);
}

/**
 * oct(color)
 */
static PyObject *
_color_oct(pgColorObject *color)
{
    char buf[100];
    Uint32 tmp = ((color->data[0] << 24) + (color->data[1] << 16) +
                  (color->data[2] << 8) + color->data[3]);
#if !PY3
#if LONG_MAX == 2147483647
    if (tmp < LONG_MAX) {
        PyOS_snprintf(buf, sizeof(buf), "0%lo", (unsigned long)tmp);
    } else {
        PyOS_snprintf(buf, sizeof(buf), "0%loL", (unsigned long)tmp);
    }
    return PyString_FromString(buf);
#endif
#endif
    PyOS_snprintf(buf, sizeof(buf), "0%lo", (unsigned long)tmp);

    return PyString_FromString(buf);
}

/**
 * hex(color)
 */
static PyObject *
_color_hex(pgColorObject *color)
{
    char buf[100];
    Uint32 tmp = ((color->data[0] << 24) + (color->data[1] << 16) +
                  (color->data[2] << 8) + color->data[3]);
#if !PY3
#if LONG_MAX == 2147483647
    if (tmp < LONG_MAX) {
        PyOS_snprintf(buf, sizeof(buf), "0x%lx", (unsigned long)tmp);
    } else {
        PyOS_snprintf(buf, sizeof(buf), "0x%lxL", (unsigned long)tmp);
    }
    return Text_FromUTF8(buf);
#endif
#endif
    PyOS_snprintf(buf, sizeof(buf), "0x%lx", (unsigned long)tmp);
    return Text_FromUTF8(buf);
}
#endif

/* Sequence protocol methods */

/**
 * len(color)
 */
static Py_ssize_t
_color_length(pgColorObject *color)
{
    return color->len;
}

/**
 * color.set_length(3)
 */

static PyObject *
_color_set_length(pgColorObject *color, PyObject *args)
{
    int clength;

    if (!PyArg_ParseTuple(args, "i", &clength)) {
        if (!PyErr_ExceptionMatches(PyExc_OverflowError)) {
            return NULL;
        }
        /* OverflowError also means the value is out-of-range */
        PyErr_Clear();
        clength = INT_MAX;
    }

    if (clength > 4 || clength < 1) {
        return RAISE(PyExc_ValueError, "Length needs to be 1,2,3, or 4.");
    }

    color->len = clength;

    Py_RETURN_NONE;
}

/**
 * color[x]
 */
static PyObject *
_color_item(pgColorObject *color, Py_ssize_t _index)
{
    if ((_index > (color->len - 1))) {
        return RAISE(PyExc_IndexError, "invalid index");
    }

    switch (_index) {
        case 0:
            return PyInt_FromLong(color->data[0]);
        case 1:
            return PyInt_FromLong(color->data[1]);
        case 2:
            return PyInt_FromLong(color->data[2]);
        case 3:
            return PyInt_FromLong(color->data[3]);
        default:
            return RAISE(PyExc_IndexError, "invalid index");
    }
}

static PyObject *
_color_subscript(pgColorObject *self, PyObject *item)
{
    if (PyIndex_Check(item)) {
        Py_ssize_t i;
        i = PyNumber_AsSsize_t(item, PyExc_IndexError);

        if (i == -1 && PyErr_Occurred()) {
            return NULL;
        }
        /*
        if (i < 0)
            i += PyList_GET_SIZE(self);
        */
        return _color_item(self, i);
    }
    if (PySlice_Check(item)) {
        int len = 4;
        Py_ssize_t start, stop, step, slicelength;

        if (Slice_GET_INDICES_EX(item, len, &start, &stop, &step,
                                 &slicelength) < 0) {
            return NULL;
        }

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
_color_ass_item(pgColorObject *color, Py_ssize_t _index, PyObject *value)
{
    switch (_index) {
        case 0:
            return _color_set_r(color, value, NULL);
        case 1:
            return _color_set_g(color, value, NULL);
        case 2:
            return _color_set_b(color, value, NULL);
        case 3:
            return _color_set_a(color, value, NULL);
        default:
            PyErr_SetString(PyExc_IndexError, "invalid index");
            break;
    }
    return -1;
}

static PyObject *
_color_slice(register pgColorObject *a, register Py_ssize_t ilow,
             register Py_ssize_t ihigh)
{
    Py_ssize_t len;
    Py_ssize_t c1 = 0;
    Py_ssize_t c2 = 0;
    Py_ssize_t c3 = 0;
    Py_ssize_t c4 = 0;

    /* printf("ilow :%d:, ihigh:%d:\n", ilow, ihigh); */

    if (ilow < 0) {
        ilow = 0;
    }
    if (ihigh > 3) {
        ihigh = 4;
    }
    if (ihigh < ilow) {
        ihigh = ilow;
    }

    len = ihigh - ilow;
    /* printf("2 ilow :%d:, ihigh:%d: len:%d:\n", ilow, ihigh, len); */

    if (ilow == 0) {
        c1 = a->data[0];
        c2 = a->data[1];
        c3 = a->data[2];
        c4 = a->data[3];
    }
    else if (ilow == 1) {
        c1 = a->data[1];
        c2 = a->data[2];
        c3 = a->data[3];
    }
    else if (ilow == 2) {
        c1 = a->data[2];
        c2 = a->data[3];
    }
    else if (ilow == 3) {
        c1 = a->data[3];
    }

    /* return a tuple depending on which elements are wanted.  */
    if (len == 4) {
        return Py_BuildValue("(iiii)", c1, c2, c3, c4);
    }
    else if (len == 3) {
        return Py_BuildValue("(iii)", c1, c2, c3);
    }
    else if (len == 2) {
        return Py_BuildValue("(ii)", c1, c2);
    }
    else if (len == 1) {
        return Py_BuildValue("(i)", c1);
    }
    else {
        return Py_BuildValue("()");
    }
}

static int
_color_set_slice(pgColorObject *color, PyObject *idx, PyObject *val)
{
    if (val == NULL) {
        PyErr_SetString(PyExc_TypeError,
            "Color object doesn't support item deletion");
        return -1;
    }
#if PY2
    if (PyInt_Check(idx)) {
        return _color_ass_item(color, PyInt_AS_LONG(idx), val);
    }
#endif
    if (PyLong_Check(idx)) {
        return _color_ass_item(color, PyLong_AsLong(idx), val);
    }
    else if (PySlice_Check(idx)) {
        Py_ssize_t start, stop, step, slicelength;
        PyObject *fastitems;
        int c;
        Py_ssize_t i, cur;

        if (Slice_GET_INDICES_EX(idx, color->len, &start, &stop, &step,
                                 &slicelength) < 0) {
            return -1;
        }
        if ((step < 0 && start < stop) || (step > 0 && start > stop))
            stop = start;

        if (!(fastitems = PySequence_Fast(val, "expected sequence"))) {
            return -1;
        }
        if (PySequence_Fast_GET_SIZE(fastitems) != slicelength) {
            PyErr_Format(PyExc_ValueError,
                "attempting to assign sequence of length %zd "
                "to slice of length %zd",
                PySequence_Fast_GET_SIZE(fastitems), slicelength);
            Py_DECREF(fastitems);
            return -1;
        }

        for (cur = start, i = 0; i < slicelength; cur += step, i++) {
            PyObject *obj = PySequence_Fast_GET_ITEM(fastitems, i);
            if (PyLong_Check(obj)) {
                c = PyLong_AsLong(obj);
            }
#if PY2
            else if (PyInt_Check(obj)) {
                c = PyInt_AS_LONG(obj);
            }
#endif /* PY2 */
            else {
                PyErr_SetString(PyExc_TypeError, "color components must be integers");
                Py_DECREF(fastitems);
                return -1;
            }
            if (c < 0 || c > 255) {
                PyErr_SetString(PyExc_ValueError, "color component must be 0-255");
                Py_DECREF(fastitems);
                return -1;
            }
            color->data[cur] = (Uint8)c;
        }

        Py_DECREF(fastitems);
        return 0;
    }
    PyErr_SetString(PyExc_IndexError,
        "Index must be an integer or slice");
    return -1;
}

/*
 * colorA == colorB
 * colorA != colorB
 */
static PyObject *
_color_richcompare(PyObject *o1, PyObject *o2, int opid)
{
    typedef union {
        Uint32 pixel;
        Uint8 bytes[4];
    } _rgba_t;
    _rgba_t rgba1, rgba2;

    switch (_coerce_obj(o1, rgba1.bytes)) {
        case -1:
            return 0;
        case 0:
            goto Unimplemented;
        default:
            break;
    }
    switch (_coerce_obj(o2, rgba2.bytes)) {
        case -1:
            return 0;
        case 0:
            goto Unimplemented;
        default:
            break;
    }

    switch (opid) {
        case Py_EQ:
            return PyBool_FromLong(rgba1.pixel == rgba2.pixel);
        case Py_NE:
            return PyBool_FromLong(rgba1.pixel != rgba2.pixel);
        default:
            break;
    }

Unimplemented:
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}

static int
_color_getbuffer(pgColorObject *color, Py_buffer *view, int flags)
{
    static char format[] = "B";

    if (PyBUF_HAS_FLAG(flags, PyBUF_WRITABLE)) {
        PyErr_SetString(pgExc_BufferError, "color buffer is read-only");
        return -1;
    }
    view->buf = color->data;
    view->ndim = 1;
    view->itemsize = 1;
    view->len = color->len;
    view->readonly = 1;
    if (PyBUF_HAS_FLAG(flags, PyBUF_ND)) {
        view->ndim = 1;
        view->shape = &view->len;
    }
    else {
        view->ndim = 0;
        view->shape = 0;
    }
    if (PyBUF_HAS_FLAG(flags, PyBUF_FORMAT)) {
        view->format = format;
    }
    else {
        view->format = 0;
    }
    if (PyBUF_HAS_FLAG(flags, PyBUF_STRIDES)) {
        view->strides = &view->itemsize;
    }
    else {
        view->strides = 0;
    }
    view->suboffsets = 0;
    Py_INCREF(color);
    view->obj = (PyObject *)color;
    return 0;
}

/**** C API interfaces ****/
static PyObject *
pgColor_New(Uint8 rgba[])
{
    return (PyObject *)_color_new_internal(&pgColor_Type, rgba);
}

static PyObject *
pgColor_NewLength(Uint8 rgba[], Uint8 length)
{
    if (length < 1 || length > 4) {
        return PyErr_Format(PyExc_ValueError,
                            "Expected length within range [1,4]: got %d",
                            (int)length);
    }

    return (PyObject *)_color_new_internal_length(&pgColor_Type, rgba, length);
}

static int
pg_RGBAFromColorObj(PyObject *color, Uint8 rgba[])
{
    if (PyColor_Check(color)) {
        rgba[0] = ((pgColorObject *)color)->data[0];
        rgba[1] = ((pgColorObject *)color)->data[1];
        rgba[2] = ((pgColorObject *)color)->data[2];
        rgba[3] = ((pgColorObject *)color)->data[3];
        return 1;
    }

    /* Default action */
    return pg_RGBAFromObj(color, rgba);
}

static int
pg_RGBAFromFuzzyColorObj(PyObject * color, Uint8 rgba[])
{
    return _parse_color_from_single_object(color, rgba) == 0;
}

/*DOC*/ static char _color_doc[] =
    /*DOC*/ "color module for pygame";

MODINIT_DEFINE(color)
{
    PyObject *colordict;
    PyObject *module;
    PyObject *apiobj;
    static void *c_api[PYGAMEAPI_COLOR_NUMSLOTS];

#if PY3
    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "color",
                                         _color_doc,
                                         -1,
                                         _color_methods,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL};
#endif

    /* imported needed apis; Do this first so if there is an error
       the module is not loaded.
    */
    import_pygame_base();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }

    colordict = PyImport_ImportModule("pygame.colordict");
    if (colordict) {
        PyObject *_dict = PyModule_GetDict(colordict);
        PyObject *colors = PyDict_GetItemString(_dict, "THECOLORS");
        Py_INCREF(colors);
        _COLORDICT = colors;
        Py_DECREF(colordict);
    }
    else {
        MODINIT_ERROR;
    }

    /* type preparation */
    if (PyType_Ready(&pgColor_Type) < 0) {
        Py_DECREF(_COLORDICT);
        MODINIT_ERROR;
    }

    /* create the module */
#if PY3
    module = PyModule_Create(&_module);
#else
    module = Py_InitModule3(MODPREFIX "color", NULL, _color_doc);
#endif
    if (module == NULL) {
        Py_DECREF(_COLORDICT);
        MODINIT_ERROR;
    }
    pgColor_Type.tp_getattro = PyObject_GenericGetAttr;
    Py_INCREF(&pgColor_Type);
    if (PyModule_AddObject(module, "Color", (PyObject *)&pgColor_Type)) {
        Py_DECREF(&pgColor_Type);
        Py_DECREF(_COLORDICT);
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    Py_INCREF(_COLORDICT);
    if (PyModule_AddObject(module, "THECOLORS", _COLORDICT)) {
        Py_DECREF(_COLORDICT);
        Py_DECREF(_COLORDICT);
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

    c_api[0] = &pgColor_Type;
    c_api[1] = pgColor_New;
    c_api[2] = pg_RGBAFromColorObj;
    c_api[3] = pgColor_NewLength;
    c_api[4] = pg_RGBAFromFuzzyColorObj;

    apiobj = encapsulate_api(c_api, "color");
    if (apiobj == NULL) {
        Py_DECREF(_COLORDICT);
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    if (PyModule_AddObject(module, PYGAMEAPI_LOCAL_ENTRY, apiobj)) {
        Py_DECREF(apiobj);
        Py_DECREF(_COLORDICT);
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    MODINIT_RETURN(module);
}
