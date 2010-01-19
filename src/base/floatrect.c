/*
  pygame - Python Game Library
  Copyright (C) 2000-2001  Pete Shinners, 2008 Marcus von Appen

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
#define PYGAME_FRECT_INTERNAL

#include "internals.h"
#include "pgbase.h"
#include "base_doc.h"

#define INTERSECT(A,B) \
    (((A->x >= B->x && A->x < B->x + B->w) ||        \
     (B->x >= A->x && B->x < A->x + A->w)) &&        \
    ((A->y >= B->y && A->y < B->y + B->h)   ||       \
     (B->y >= A->y && B->y < A->y + A->h)))

static int _frect_init (PyObject *self, PyObject *args, PyObject *kwds);
static void _frect_dealloc (PyFRect *self);
static PyObject* _frect_repr (PyObject *self);

static PyObject* _frect_getx (PyObject *self, void *closure);
static int _frect_setx (PyObject *self, PyObject *value, void *closure);
static PyObject* _frect_gety (PyObject *self, void *closure);
static int _frect_sety (PyObject *self, PyObject *value, void *closure);
static PyObject* _frect_getwidth (PyObject *self, void *closure);
static int _frect_setwidth (PyObject *self, PyObject *value, void *closure);
static PyObject* _frect_getheight (PyObject *self, void *closure);
static int _frect_setheight (PyObject *self, PyObject *value, void *closure);
static PyObject* _frect_getbottom (PyObject *self, void *closure);
static int _frect_setbottom (PyObject *self, PyObject *value, void *closure);
static PyObject* _frect_getright (PyObject *self, void *closure);
static int _frect_setright (PyObject *self, PyObject *value, void *closure);
static PyObject* _frect_getcenterx (PyObject *self, void *closure);
static int _frect_setcenterx (PyObject *self, PyObject *value, void *closure);
static PyObject* _frect_getcentery (PyObject *self, void *closure);
static int _frect_setcentery (PyObject *self, PyObject *value, void *closure);
static PyObject* _frect_getcenter (PyObject *self, void *closure);
static int _frect_setcenter (PyObject *self, PyObject *value, void *closure);
static PyObject* _frect_getsize (PyObject *self, void *closure);
static int _frect_setsize (PyObject *self, PyObject *value, void *closure);
static PyObject* _frect_getmidtop (PyObject *self, void *closure);
static int _frect_setmidtop (PyObject *self, PyObject *value, void *closure);
static PyObject* _frect_getmidleft (PyObject *self, void *closure);
static int _frect_setmidleft (PyObject *self, PyObject *value, void *closure);
static PyObject* _frect_getmidbottom (PyObject *self, void *closure);
static int _frect_setmidbottom (PyObject *self, PyObject *value, void *closure);
static PyObject* _frect_getmidright (PyObject *self, void *closure);
static int _frect_setmidright (PyObject *self, PyObject *value, void *closure);
static PyObject* _frect_gettopleft (PyObject *self, void *closure);
static int _frect_settopleft (PyObject *self, PyObject *value, void *closure);
static PyObject* _frect_gettopright (PyObject *self, void *closure);
static int _frect_settopright (PyObject *self, PyObject *value, void *closure);
static PyObject* _frect_getbottomleft (PyObject *self, void *closure);
static int _frect_setbottomleft (PyObject *self, PyObject *value,
    void *closure);
static PyObject* _frect_getbottomright (PyObject *self, void *closure);
static int _frect_setbottomright (PyObject *self, PyObject *value,
    void *closure);

static PyObject* _frect_clip (PyObject* self, PyObject *args);
static PyObject* _frect_copy (PyObject* self);
static PyObject* _frect_move (PyObject* self, PyObject *args);
static PyObject* _frect_move_ip (PyObject* self, PyObject *args);
static PyObject* _frect_union (PyObject* self, PyObject *args);
static PyObject* _frect_union_ip (PyObject* self, PyObject *args);
static PyObject* _frect_inflate (PyObject* self, PyObject *args);
static PyObject* _frect_inflate_ip (PyObject* self, PyObject *args);
static PyObject* _frect_clamp (PyObject* self, PyObject *args);
static PyObject* _frect_clamp_ip (PyObject* self, PyObject *args);
static PyObject* _frect_fit (PyObject* self, PyObject *args);
static PyObject* _frect_contains (PyObject* self, PyObject *args);
static PyObject* _frect_collidepoint (PyObject *self, PyObject *args);
static PyObject* _frect_colliderect (PyObject *self, PyObject *args);
static PyObject* _frect_collidelist (PyObject *self, PyObject *args,
    PyObject *kwds);
static PyObject* _frect_collidelistall (PyObject *self, PyObject *args,
    PyObject *kwds);
static PyObject* _frect_collidedict (PyObject *self, PyObject *args,
    PyObject *kwds);
static PyObject* _frect_collidedictall (PyObject *self, PyObject *args,
    PyObject *kwds);
static PyObject* _frect_round (PyObject *self);
static PyObject* _frect_ceil (PyObject *self);
static PyObject* _frect_floor (PyObject *self);
static PyObject* _frect_trunc (PyObject *self);

static int _frect_compare (PyObject *self, PyObject *other);
static PyObject* _frect_richcompare (PyObject *o1, PyObject *o2, int opid);

/**
 */
static PyMethodDef _frect_methods[] = {
    { "clip", _frect_clip, METH_O, DOC_BASE_FRECT_CLIP },
    { "copy", (PyCFunction)_frect_copy, METH_NOARGS, DOC_BASE_FRECT_COPY },
    { "move", _frect_move, METH_VARARGS, DOC_BASE_FRECT_MOVE },
    { "move_ip",  _frect_move_ip, METH_VARARGS, DOC_BASE_FRECT_MOVE_IP },
    { "union",  _frect_union, METH_O, DOC_BASE_FRECT_UNION},
    { "union_ip", _frect_union_ip, METH_O, DOC_BASE_FRECT_UNION_IP },
    { "inflate",  _frect_inflate, METH_VARARGS, DOC_BASE_FRECT_INFLATE },
    { "inflate_ip", _frect_inflate_ip, METH_VARARGS,
      DOC_BASE_FRECT_INFLATE_IP },
    { "clamp", _frect_clamp, METH_O, DOC_BASE_FRECT_CLAMP },
    { "clamp_ip", _frect_clamp_ip, METH_O, DOC_BASE_FRECT_CLAMP_IP },
    { "fit", _frect_fit, METH_O, DOC_BASE_FRECT_FIT },
    { "contains", _frect_contains, METH_O, DOC_BASE_FRECT_CONTAINS },
    { "collidepoint", _frect_collidepoint, METH_VARARGS,
      DOC_BASE_FRECT_COLLIDEPOINT },
    { "colliderect", _frect_colliderect, METH_O, DOC_BASE_FRECT_COLLIDERECT},
    { "collidelist", (PyCFunction) _frect_collidelist,
      METH_VARARGS | METH_KEYWORDS, DOC_BASE_FRECT_COLLIDELIST},
    { "collidelistall", (PyCFunction) _frect_collidelistall,
      METH_VARARGS | METH_KEYWORDS, DOC_BASE_FRECT_COLLIDELISTALL },
    { "collidedict", (PyCFunction) _frect_collidedict,
      METH_VARARGS | METH_KEYWORDS, DOC_BASE_FRECT_COLLIDEDICT },
    { "collidedictall", (PyCFunction) _frect_collidedictall,
      METH_VARARGS | METH_KEYWORDS, DOC_BASE_FRECT_COLLIDEDICTALL},
    { "round", (PyCFunction) _frect_round, METH_NOARGS, DOC_BASE_FRECT_ROUND },
    { "ceil", (PyCFunction) _frect_ceil, METH_NOARGS, DOC_BASE_FRECT_CEIL },
    { "floor", (PyCFunction) _frect_floor, METH_NOARGS, DOC_BASE_FRECT_FLOOR },
    { "trunc", (PyCFunction) _frect_trunc, METH_NOARGS, DOC_BASE_FRECT_TRUNC },
    { NULL, NULL, 0, NULL }
};

/**
 */
static PyGetSetDef _frect_getsets[] = {
    { "x", _frect_getx, _frect_setx, DOC_BASE_FRECT_X, NULL },
    { "y", _frect_gety, _frect_sety, DOC_BASE_FRECT_Y, NULL },
    { "width", _frect_getwidth, _frect_setwidth, DOC_BASE_FRECT_WIDTH, NULL },
    { "w", _frect_getwidth, _frect_setwidth, DOC_BASE_FRECT_WIDTH, NULL },
    { "height", _frect_getheight, _frect_setheight, DOC_BASE_FRECT_HEIGHT,
      NULL },
    { "h", _frect_getheight, _frect_setheight, DOC_BASE_FRECT_HEIGHT, NULL },
    { "size", _frect_getsize, _frect_setsize, DOC_BASE_FRECT_SIZE, NULL },

    { "left", _frect_getx, _frect_setx, DOC_BASE_FRECT_LEFT, NULL },
    { "top", _frect_gety, _frect_sety, DOC_BASE_FRECT_TOP, NULL },
    { "bottom", _frect_getbottom, _frect_setbottom, DOC_BASE_FRECT_BOTTOM,
      NULL },
    { "right", _frect_getright, _frect_setright, DOC_BASE_FRECT_RIGHT, NULL },

    { "centerx", _frect_getcenterx, _frect_setcenterx, DOC_BASE_FRECT_CENTERX,
      NULL },
    { "centery", _frect_getcentery, _frect_setcentery, DOC_BASE_FRECT_CENTERY,
      NULL },
    { "center", _frect_getcenter, _frect_setcenter, DOC_BASE_FRECT_CENTER,
      NULL },

    { "midtop", _frect_getmidtop, _frect_setmidtop, DOC_BASE_FRECT_MIDTOP,
      NULL },
    { "midleft", _frect_getmidleft, _frect_setmidleft, DOC_BASE_FRECT_MIDLEFT,
      NULL },
    { "midbottom", _frect_getmidbottom, _frect_setmidbottom,
      DOC_BASE_FRECT_MIDBOTTOM, NULL },
    { "midright", _frect_getmidright, _frect_setmidright,
      DOC_BASE_FRECT_MIDRIGHT, NULL },

    { "topleft", _frect_gettopleft, _frect_settopleft, DOC_BASE_FRECT_TOPLEFT,
      NULL },
    { "topright", _frect_gettopright, _frect_settopright,
      DOC_BASE_FRECT_TOPRIGHT, NULL },
    { "bottomleft", _frect_getbottomleft, _frect_setbottomleft,
      DOC_BASE_FRECT_BOTTOMLEFT, NULL },
    { "bottomright", _frect_getbottomright, _frect_setbottomright,
      DOC_BASE_FRECT_BOTTOMRIGHT, NULL },

    { NULL, NULL, NULL, NULL, NULL }
};

/**
 */
PyTypeObject PyFRect_Type =
{
    TYPE_HEAD(NULL,0)
    "base.FRect",               /* tp_name */
    sizeof (PyFRect),           /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor) _frect_dealloc, /* tp_dealloc */
    0,                          /* tp_print */
    0,                          /* tp_getattr */
    0,                          /* tp_setattr */
#ifdef IS_PYTHON_3
    0,                          /* tp_compare is now tp_reserved */
#else
    (cmpfunc)_frect_compare,    /* tp_compare */
#endif
    (reprfunc)_frect_repr,      /* tp_repr */
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
    DOC_BASE_FRECT,
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    _frect_richcompare,         /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    _frect_methods,             /* tp_methods */
    0,                          /* tp_members */
    _frect_getsets,             /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc) _frect_init,     /* tp_init */
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
_frect_dealloc (PyFRect *self)
{
    ((PyObject*)self)->ob_type->tp_free ((PyObject *) self);
}

static int
_frect_init (PyObject *self, PyObject *args, PyObject *kwds)
{
    double x, y;
    double w, h;

    if (!PyArg_ParseTuple (args, "dddd", &x, &y, &w, &h))
    {
        x = y = 0;
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "dd", &w, &h))
        {
            PyObject *pt, *rect;
            PyErr_Clear ();
            /* Rect ((x,y),(w,h)) */
            if (!PyArg_ParseTuple (args, "OO", &pt, &rect))
            {
                PyErr_Clear ();
                /* Rect ((w,h)) || Rect (rect) */
                if (!PyArg_ParseTuple (args, "O", &rect))
                {
                    return -1;
                }
                if (PyRect_Check (rect))
                {
                    x = ((PyRect*)rect)->x;
                    y = ((PyRect*)rect)->y;
                    w = ((PyRect*)rect)->w;
                    h = ((PyRect*)rect)->h;
                }
                else if (PyFRect_Check (rect))
                {
                    x = ((PyFRect*)rect)->x;
                    y = ((PyFRect*)rect)->y;
                    w = ((PyFRect*)rect)->w;
                    h = ((PyFRect*)rect)->h;
                }
                else if (!FSizeFromObject (rect, &w, &h))
                    return -1;
            }
            else
            {
                if (!FPointFromObject (pt, &x, &y))
                    return -1;
                if (!FSizeFromObject (rect, &w, &h))
                    return -1;
            }
        }
    }

    if (w < 0 || h < 0)
    {
        PyErr_SetString (PyExc_ValueError,
            "width and height must not be negative");
        return -1;
    }

    ((PyFRect*)self)->x = x;
    ((PyFRect*)self)->y = y;
    ((PyFRect*)self)->w = w;
    ((PyFRect*)self)->h = h;
    return 0;
}

static PyObject*
_frect_repr (PyObject *self)
{
    PyFRect *r = (PyFRect*) self;
#ifdef IS_PYTHON_3
    PyObject *retval;
    int i;
    char *b[4] = { NULL, NULL, NULL, NULL };
    
    b[0] = PyOS_double_to_string (r->x, 'f', 8, 0, NULL);
    if (!b[0])
        goto failure;
    b[1] = PyOS_double_to_string (r->y, 'f', 8, 0, NULL);
    if (!b[1])
        goto failure;
    b[2] = PyOS_double_to_string (r->w, 'f', 8, 0, NULL);
    if (!b[2])
        goto failure;
    b[3] = PyOS_double_to_string (r->h, 'f', 8, 0, NULL);
    if (!b[3])
        goto failure;

    retval = Text_FromFormat ("FRect(%s, %s, %s, %s)", b[0], b[1], b[2], b[3]);
    goto success;

failure:
    retval = Text_FromUTF8 ("FRect(???)");
success:
    for (i = 0; i < 4; i++)
    {
        if (b[i])
            PyMem_Free (b[i]);
    }
    return retval;
#else /* !IS_PYTHON_3 */
    char b1[32], b2[32], b3[32], b4[32];
    if (!PyOS_ascii_formatd (b1, 32, "%.8f", r->x) ||
        !PyOS_ascii_formatd (b2, 32, "%.8f", r->y) ||
        !PyOS_ascii_formatd (b3, 32, "%.8f", r->w) ||
        !PyOS_ascii_formatd (b4, 32, "%.8f", r->h))
        return Text_FromUTF8 ("FRect(???)");
    return Text_FromFormat ("FRect(%s, %s, %s, %s)", b1, b2, b3, b4);
#endif
}

/* FRect getters/setters */
static PyObject*
_frect_getx (PyObject *self, void *closure)
{
    return PyFloat_FromDouble (((PyFRect*)self)->x);
}

static int
_frect_setx (PyObject *self, PyObject *value, void *closure)
{
    double x;
    if (!DoubleFromObj (value, &x))
        return -1;
    ((PyFRect*)self)->x = x;
    return 0;
}

static PyObject*
_frect_gety (PyObject *self, void *closure)
{
    return PyFloat_FromDouble (((PyFRect*)self)->y);
}

static int
_frect_sety (PyObject *self, PyObject *value, void *closure)
{
    double y;
    if (!DoubleFromObj (value, &y))
        return -1;
    ((PyFRect*)self)->y = y;
    return 0;
}

static PyObject*
_frect_getwidth (PyObject *self, void *closure)
{
    return PyFloat_FromDouble (((PyFRect*)self)->w);
}

static int
_frect_setwidth (PyObject *self, PyObject *value, void *closure)
{
    double w;
    if (!DoubleFromObj (value, &w))
        return -1;
    if (w < 0)
    {
        PyErr_SetString (PyExc_ValueError, "width must not be negative");
        return -1;
    }
    ((PyFRect*)self)->w = w;
    return 0;
}

static PyObject*
_frect_getheight (PyObject *self, void *closure)
{
    return PyFloat_FromDouble (((PyFRect*)self)->h);
}

static int
_frect_setheight (PyObject *self, PyObject *value, void *closure)
{
    double h;
    if (!DoubleFromObj (value, &h))
        return -1;
    if (h < 0)
    {
        PyErr_SetString (PyExc_ValueError, "height must not be negative");
        return -1;
    }
    ((PyFRect*)self)->h = h;
    return 0;
}

static PyObject*
_frect_getbottom (PyObject *self, void *closure)
{
    PyFRect *r = (PyFRect*) self;
    return PyFloat_FromDouble (r->y + r->h);
}

static int
_frect_setbottom (PyObject *self, PyObject *value, void *closure)
{
    double bottom;
    if (!DoubleFromObj (value, &bottom))
        return -1;
    ((PyFRect*)self)->y = DBL_SUB_LIMIT (bottom, ((PyFRect*)self)->h);
    return 0;
}

static PyObject*
_frect_getright (PyObject *self, void *closure)
{
    PyFRect *r = (PyFRect*) self;
    return PyFloat_FromDouble (r->x + r->w);
}

static int
_frect_setright (PyObject *self, PyObject *value, void *closure)
{
    double right;
    if (!DoubleFromObj (value, &right))
        return -1;
    ((PyFRect*)self)->x = DBL_SUB_LIMIT (right, ((PyFRect*)self)->w);
    return 0;
}

static PyObject*
_frect_getcenterx (PyObject *self, void *closure)
{
    PyFRect *r = (PyFRect*) self;
    return PyFloat_FromDouble (r->x + (r->w / 2));
}

static int
_frect_setcenterx (PyObject *self, PyObject *value, void *closure)
{
    double centerx;
    if (!DoubleFromObj (value, &centerx))
        return -1;

    ((PyFRect*)self)->x = DBL_SUB_LIMIT (centerx, (((PyFRect*)self)->w / 2));
    return 0;
}

static PyObject*
_frect_getcentery (PyObject *self, void *closure)
{
    PyFRect *r = (PyFRect*) self;
    return PyFloat_FromDouble (r->y + (r->h / 2));
}

static int
_frect_setcentery (PyObject *self, PyObject *value, void *closure)
{
    double centery;
    if (!DoubleFromObj (value, &centery))
        return -1;

    ((PyFRect*)self)->y = DBL_SUB_LIMIT (centery, (((PyFRect*)self)->h / 2));
    return 0;
}

static PyObject*
_frect_getcenter (PyObject *self, void *closure)
{
    PyFRect *r = (PyFRect*) self;
    return Py_BuildValue ("(dd)", r->x + (r->w / 2), r->y + (r->h / 2));
}

static int
_frect_setcenter (PyObject *self, PyObject *value, void *closure)
{
    double x, y;
    if (!FPointFromObject (value, &x, &y))
        return -1;
    
    ((PyFRect*)self)->x = DBL_SUB_LIMIT (x, (((PyFRect*)self)->w / 2));
    ((PyFRect*)self)->y = DBL_SUB_LIMIT (y, (((PyFRect*)self)->h / 2));
    return 0;
}

static PyObject*
_frect_getsize (PyObject *self, void *closure)
{
    PyFRect *r = (PyFRect*) self;
    return Py_BuildValue ("(dd)", r->w, r->h);
}

static int
_frect_setsize (PyObject *self, PyObject *value, void *closure)
{
    double w, h;

    if (!FSizeFromObject (value, &w, &h))
        return -1;

    if (w < 0 || h < 0)
    {
        PyErr_SetString (PyExc_ValueError,
            "width and height must not be negative");
        return -1;
    }

    ((PyFRect*)self)->w = w;
    ((PyFRect*)self)->h = h;
    return 0;
}

static PyObject*
_frect_getmidtop (PyObject *self, void *closure)
{
    PyFRect *r = (PyFRect*) self;
    return Py_BuildValue ("(dd)", r->x + (r->w / 2), r->y);
}

static int
_frect_setmidtop (PyObject *self, PyObject *value, void *closure)
{
    double x, y;
    if (!FPointFromObject (value, &x, &y))
        return -1;

    ((PyFRect*)self)->x = DBL_SUB_LIMIT (x, (((PyFRect*)self)->w / 2));
    ((PyFRect*)self)->y = y;
    return 0;
}

static PyObject*
_frect_getmidleft (PyObject *self, void *closure)
{
    PyFRect *r = (PyFRect*) self;
    return Py_BuildValue ("(dd)", r->x, r->y +  (r->h / 2));
}

static int
_frect_setmidleft (PyObject *self, PyObject *value, void *closure)
{
    double x, y;
    if (!FPointFromObject (value, &x, &y))
        return -1;

    ((PyFRect*)self)->x = x;
    ((PyFRect*)self)->y = DBL_SUB_LIMIT (y, (((PyFRect*)self)->h / 2));
    return 0;
}

static PyObject*
_frect_getmidbottom (PyObject *self, void *closure)
{
    PyFRect *r = (PyFRect*) self;
    return Py_BuildValue ("(dd)", r->x + (r->w / 2), r->y + r->h);
}

static int
_frect_setmidbottom (PyObject *self, PyObject *value, void *closure)
{
    double x, y;
    if (!FPointFromObject (value, &x, &y))
        return -1;
    
    ((PyFRect*)self)->x = DBL_SUB_LIMIT (x, (((PyFRect*)self)->w / 2));
    ((PyFRect*)self)->y = DBL_SUB_LIMIT (y, ((PyFRect*)self)->h);
    return 0;
}

static PyObject*
_frect_getmidright (PyObject *self, void *closure)
{
    PyFRect *r = (PyFRect*) self;
    return Py_BuildValue ("(dd)", r->x + r->w, r->y + (r->h / 2));
}

static int
_frect_setmidright (PyObject *self, PyObject *value, void *closure)
{
    double x, y;
    if (!FPointFromObject (value, &x, &y))
        return -1;

    ((PyFRect*)self)->x = DBL_SUB_LIMIT (x, ((PyFRect*)self)->w);
    ((PyFRect*)self)->y = DBL_SUB_LIMIT (y, (((PyFRect*)self)->h / 2));
    return 0;
}

static PyObject*
_frect_gettopleft (PyObject *self, void *closure)
{
    PyFRect *r = (PyFRect*) self;
    return Py_BuildValue ("(dd)", r->x, r->y);
}
static int
_frect_settopleft (PyObject *self, PyObject *value, void *closure)
{
    double x, y;
    if (!FPointFromObject (value, &x, &y))
        return -1;

    ((PyFRect*)self)->x = x;
    ((PyFRect*)self)->y = y;
    return 0;
}

static PyObject*
_frect_gettopright (PyObject *self, void *closure)
{
    PyFRect *r = (PyFRect*) self;
    return Py_BuildValue ("(dd)", r->x + r->w, r->y);
}

static int
_frect_settopright (PyObject *self, PyObject *value, void *closure)
{
    double x, y;
    if (!FPointFromObject (value, &x, &y))
        return -1;

    ((PyFRect*)self)->x = DBL_SUB_LIMIT (x, ((PyFRect*)self)->w);
    ((PyFRect*)self)->y = y;
    return 0;
}

static PyObject*
_frect_getbottomleft (PyObject *self, void *closure)
{
    PyFRect *r = (PyFRect*) self;
    return Py_BuildValue ("(dd)", r->x, r->y + r->h);
}

static int
_frect_setbottomleft (PyObject *self, PyObject *value, void *closure)
{
    double x, y;
    if (!FPointFromObject (value, &x, &y))
        return -1;

    ((PyFRect*)self)->x = x;
    ((PyFRect*)self)->y = DBL_SUB_LIMIT (y, ((PyFRect*)self)->h);
    return 0;
}

static PyObject*
_frect_getbottomright (PyObject *self, void *closure)
{
    PyFRect *r = (PyFRect*) self;
    return Py_BuildValue ("(dd)", r->x + r->w, r->y + r->h);
}

static int
_frect_setbottomright (PyObject *self, PyObject *value, void *closure)
{
    double x, y;
    if (!FPointFromObject (value, &x, &y))
        return -1;

    ((PyFRect*)self)->x = DBL_SUB_LIMIT (x, ((PyFRect*)self)->w);
    ((PyFRect*)self)->y = DBL_SUB_LIMIT (y, ((PyFRect*)self)->h);
    return 0;
}

/* FRect methods */
static PyObject*
_frect_clip (PyObject* self, PyObject *args)
{
    PyFRect *rself, *rarg;

    double x, y, w, h;
    double selfright, argright;
    double selfbottom, argbottom;

    if (!PyFRect_Check (args))
    {
        PyErr_SetString (PyExc_TypeError, "argument must be a FRect");
        return NULL;
    }

    rself = (PyFRect*) self;
    rarg = (PyFRect*) args;

    selfright = DBL_ADD_LIMIT (rself->x, rself->w);
    selfbottom = DBL_ADD_LIMIT (rself->y, rself->h);
    argright = DBL_ADD_LIMIT (rarg->x, rarg->w);
    argbottom = DBL_ADD_LIMIT (rarg->y, rarg->h);

    /* Check left and right non-overlaps */
    if (rarg->x > selfright || rself->x > argright)
        return PyFRect_New (0., 0., 0., 0.);

    /* Check bottom and top non-overlaps */
    if (rarg->y > selfbottom || rself->y > argbottom)
        return PyFRect_New (0., 0., 0., 0.);

    /* Clip x and y by testing self in arg overlap */
    x = (rself->x >= rarg->x) ? rself->x : rarg->x;
    y = (rself->y >= rarg->y) ? rself->y : rarg->y;
    
    /* Clip width and height */
    if (selfright <= argright)
        w = selfright - x;
    else
        w = argright - x;

    if (selfbottom <= argbottom)
        h = selfbottom - y;
    else
        h = argbottom - y;

    return PyFRect_New (x, y, w, h);
}

static PyObject*
_frect_copy (PyObject* self)
{
    return PyFRect_New (((PyFRect*)self)->x, ((PyFRect*)self)->y,
        ((PyFRect*)self)->w, ((PyFRect*)self)->h);
}

static PyObject*
_frect_move (PyObject* self, PyObject *args)
{
    PyFRect *frect = (PyFRect*) self;
    double x, y;

    if (!PyArg_ParseTuple (args, "dd:move", &x, &y))
    {
        PyObject *pos;
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "O:move", &pos))
            return NULL;
        if (!FPointFromObject (pos, &x, &y))
            return NULL;
    }

    return PyFRect_New (frect->x + x, frect->y + y, frect->w, frect->h);
}

static PyObject*
_frect_move_ip (PyObject* self, PyObject *args)
{
    PyFRect *frect = (PyFRect*) self;
    double x, y;

    if (!PyArg_ParseTuple (args, "dd:move_ip", &x, &y))
    {
        PyObject *pos;
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "O:move_ip", &pos))
            return NULL;
        if (!FPointFromObject (pos, &x, &y))
            return NULL;
    }

    frect->x = DBL_ADD_LIMIT (frect->x, x);
    frect->y = DBL_ADD_LIMIT (frect->y, y);
    Py_RETURN_NONE;
}

static PyObject*
_frect_union (PyObject* self, PyObject *args)
{
    PyObject *frect;
    PyFRect *rself, *rarg;
    Py_ssize_t count, i;
    double x, y;
    double r, b;

    rself = (PyFRect*) self;

    if (!PySequence_Check (args))
    {
        if (!PyFRect_Check (args))
        {
            PyErr_SetString (PyExc_TypeError, "argument must be a FRect");
            return NULL;
        }
        rarg = (PyFRect*) args;

        x = MIN (rself->x, rarg->x);
        y = MIN (rself->y, rarg->y);
        r = MAX (DBL_ADD_LIMIT (rself->x, rself->w),
            DBL_ADD_LIMIT (rarg->x, rarg->w));
        b = MAX (DBL_ADD_LIMIT (rself->y, rself->h),
            DBL_ADD_LIMIT (rarg->y, rarg->h));
        return PyFRect_New (x, y, r - x, b - y);
    }

    /* Sequence of frects. */
    x = rself->x;
    y = rself->y;
    r = DBL_ADD_LIMIT (rself->x, rself->w);
    b = DBL_ADD_LIMIT (rself->y, rself->h);
    count = PySequence_Size (args);
    if (count == -1)
        return NULL;

    for (i = 0; i < count; i++)
    {
        frect = PySequence_ITEM (args, i);
        if (!PyFRect_Check (frect))
        {
            Py_XDECREF (frect);
            PyErr_SetString (PyExc_TypeError,
                "argument must be a sequence of FRect objects.");
            return NULL;
        }
        rarg = (PyFRect*) frect;

        x = MIN (x, rarg->x);
        y = MIN (y, rarg->y);
        r = MAX (r, DBL_ADD_LIMIT (rarg->x, rarg->w));
        b = MAX (b, DBL_ADD_LIMIT (rarg->y, rarg->h));

        Py_DECREF (frect);
    }
    return PyFRect_New (x, y, r - x, b - y);
}

static PyObject*
_frect_union_ip (PyObject* self, PyObject *args)
{
    PyObject *frect;
    PyFRect *rself, *rarg;
    Py_ssize_t count, i;
    double x, y;
    double r, b;

    rself = (PyFRect*) self;
    
    if (!PySequence_Check (args))
    {
        if (!PyFRect_Check (args))
        {
            PyErr_SetString (PyExc_TypeError, "argument must be a FRect");
            return NULL;
        }
        rarg = (PyFRect*) args;

        x = MIN (rself->x, rarg->x);
        y = MIN (rself->y, rarg->y);
        r = MAX (DBL_ADD_LIMIT (rself->x, rself->w),
            DBL_ADD_LIMIT (rarg->x, rarg->w));
        b = MAX (DBL_ADD_LIMIT (rself->y, rself->h),
            DBL_ADD_LIMIT (rarg->y, rarg->h));

        rself->x = x;
        rself->y = y;
        rself->w = r - x;
        rself->h = b - y;
        Py_RETURN_NONE;
    }

    /* Sequence of frects. */
    x = rself->x;
    y = rself->y;
    r = DBL_ADD_LIMIT (rself->x, rself->w);
    b = DBL_ADD_LIMIT (rself->y, rself->h);
    count = PySequence_Size (args);
    if (count == -1)
        return NULL;

    for (i = 0; i < count; i++)
    {
        frect = PySequence_ITEM (args, i);
        if (!PyFRect_Check (frect))
        {
            Py_XDECREF (frect);
            PyErr_SetString (PyExc_TypeError,
                "argument must be a sequence of FRect objects.");
            return NULL;
        }
        rarg = (PyFRect*) frect;

        x = MIN (x, rarg->x);
        y = MIN (y, rarg->y);
        r = MAX (r, DBL_ADD_LIMIT (rarg->x, rarg->w));
        b = MAX (b, DBL_ADD_LIMIT (rarg->y, rarg->h));

        Py_DECREF (frect);
    }
    rself->x = x;
    rself->y = y;
    rself->w = r - x;
    rself->h = b - y;

    Py_RETURN_NONE;
}

static PyObject*
_frect_inflate (PyObject* self, PyObject *args)
{
    PyFRect *frect = (PyFRect*) self;
    double x, y;

    if (!PyArg_ParseTuple (args, "dd:inflate", &x, &y))
    {
        PyObject *pos;
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "O:inflate", &pos))
            return NULL;
        if (!FPointFromObject (pos, &x, &y))
            return NULL;
    }

    return PyFRect_New (DBL_SUB_LIMIT (frect->x, x / 2),
        DBL_SUB_LIMIT (frect->y, y / 2), DBL_ADD_LIMIT (frect->w, x),
        DBL_ADD_LIMIT (frect->h, y));
}

static PyObject*
_frect_inflate_ip (PyObject* self, PyObject *args)
{
    PyFRect *frect = (PyFRect*) self;
    double x, y;

    if (!PyArg_ParseTuple (args, "dd:inflate_ip", &x, &y))
    {
        PyObject *pos;
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "O:inflate_ip", &pos))
            return NULL;
        if (!FPointFromObject (pos, &x, &y))
            return NULL;
    }

    frect->x = DBL_SUB_LIMIT (frect->x, x / 2);
    frect->y = DBL_SUB_LIMIT (frect->y, y / 2);
    frect->w = DBL_ADD_LIMIT (frect->w, x);
    frect->h = DBL_ADD_LIMIT (frect->h, y);

    Py_RETURN_NONE;
}

static PyObject*
_frect_clamp (PyObject* self, PyObject *args)
{
    PyFRect *rself, *rarg;
    double x, y, t;

    if (!PyFRect_Check (args))
    {
        PyErr_SetString (PyExc_TypeError, "argument must be a FRect");
        return NULL;
    }
    rself = (PyFRect*) self;
    rarg = (PyFRect*) args;

    if (rself->w >= rarg->w)
    {
        t = DBL_ADD_LIMIT (rarg->x, rarg->w / 2);
        x = DBL_SUB_LIMIT (t, rself->w / 2);
    }
    else if (rself->x < rarg->x)
        x = rarg->x;
    else if (rself->x + rself->w > rarg->x + rarg->w)
    {
        t = DBL_ADD_LIMIT (rarg->x, rarg->w);
        x = DBL_SUB_LIMIT (t, rself->w);
    }
    else
        x = rself->x;

    if (rself->h >= rarg->h)
    {
        t = DBL_ADD_LIMIT (rarg->y, rarg->h / 2);
        y = DBL_SUB_LIMIT (t, rself->h / 2);
    }
    else if (rself->y < rarg->y)
        y = rarg->y;
    else if (rself->y + rself->h > rarg->y + rarg->h)
    {
        t = DBL_ADD_LIMIT (rarg->y, rarg->h);
        y = DBL_SUB_LIMIT (t, rself->h);
    }
    else
        y = rself->y;

    return PyFRect_New (x, y, rself->w, rself->h);
}

static PyObject*
_frect_clamp_ip (PyObject* self, PyObject *args)
{
    PyFRect *rself, *rarg;
    double t;

    if (!PyFRect_Check (args))
    {
        PyErr_SetString (PyExc_TypeError, "argument must be a FRect");
        return NULL;
    }
    rself = (PyFRect*) self;
    rarg = (PyFRect*) args;

    if (rself->w >= rarg->w)
    {
        t = DBL_ADD_LIMIT (rarg->x, rarg->w / 2);
        rself->x = DBL_SUB_LIMIT (t, rself->w / 2);
    }
    else if (rself->x < rarg->x)
        rself->x = rarg->x;
    else if (rself->x + rself->w > rarg->x + rarg->w)
    {
        t = DBL_ADD_LIMIT (rarg->x, rarg->w);
        rself->x = DBL_SUB_LIMIT (t, rself->w);
    }
    else
        rself->x = rself->x;

    if (rself->h >= rarg->h)
    {
        t = DBL_ADD_LIMIT (rarg->y, rarg->h / 2);
        rself->y = DBL_SUB_LIMIT (t, rself->h / 2);
    }
    else if (rself->y < rarg->y)
        rself->y = rarg->y;
    else if (rself->y + rself->h > rarg->y + rarg->h)
    {
        t = DBL_ADD_LIMIT (rarg->y, rarg->h);
        rself->y = DBL_SUB_LIMIT (t, rself->h);
    }
    else
        rself->y = rself->y;
    Py_RETURN_NONE;
}

static PyObject*
_frect_fit (PyObject* self, PyObject *args)
{
    PyFRect *rself, *rarg;
    double xratio, yratio, maxratio;
    double x, y, w, h;
    
    rself = (PyFRect*) self;

    if (!PyFRect_Check (args))
    {
        PyErr_SetString (PyExc_TypeError, "argument must be a FRect");
        return NULL;
    }
    rarg = (PyFRect*) args;

    xratio = rself->w / rarg->w;
    yratio = rself->h / rarg->h;
    maxratio = (xratio > yratio) ? xratio : yratio;

    w = rself->w / maxratio;
    h = rself->h / maxratio;
    x = DBL_ADD_LIMIT (rarg->x, (rarg->w - w) / 2);
    y = DBL_ADD_LIMIT (rarg->y, (rarg->h - h) / 2);

    return PyFRect_New (x, y, w, h);
}

static PyObject*
_frect_contains (PyObject* self, PyObject *args)
{
    PyFRect* rself, *rarg;

    if (!PyFRect_Check (args))
    {
        PyErr_SetString (PyExc_TypeError, "argument must be a FRect");
        return NULL;
    }
    rself = (PyFRect*) self;
    rarg = (PyFRect*) args;

    if ((rself->x <= rarg->x) && (rself->y <= rarg->y) &&
        (rself->x + rself->w >= rarg->x + rarg->w) &&
        (rself->y + rself->h >= rarg->y + rarg->h) &&
        (rself->x + rself->w > rarg->x) && (rself->y + rself->h > rarg->y))
    {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyObject*
_frect_collidepoint (PyObject *self, PyObject *args)
{
    PyFRect *rself = (PyFRect*) self;
    double x, y;

    if (!PyArg_ParseTuple (args, "dd:collidepoint", &x, &y))
    {
        PyObject *pos;
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "O:collidepoint", &pos))
            return NULL;
        if (!FPointFromObject (pos, &x, &y))
            return NULL;
    }

    if (x >= rself->x && x < rself->x + rself->w &&
        y >= rself->y && y < rself->y + rself->h)
    {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyObject*
_frect_colliderect (PyObject *self, PyObject *args)
{
    PyFRect *rarg, *rself = (PyFRect*) self;

    if (!PyFRect_Check (args))
    {
        PyErr_SetString (PyExc_TypeError, "argument must be a FRect");
        return NULL;
    }
    rarg = (PyFRect*) args;

    if (INTERSECT (rself, rarg))
    {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyObject*
_frect_collidelist (PyObject *self, PyObject *args, PyObject *kwds)
{
    PyFRect *rarg, *rself = (PyFRect*) self;
    PyObject *list, *frect, *compare = NULL;
    Py_ssize_t i, count;
    
    static char *keys[] = { "rects", "key", NULL };
    if (!PyArg_ParseTupleAndKeywords (args, kwds, "O|O:collidelist", keys,
            &list, &compare))
        return NULL;

    if (!PySequence_Check (list))
    {
        PyErr_SetString (PyExc_TypeError,
            "rects argument must be a sequence of FRect objects");
        return NULL;
    }

    if (compare == Py_None)
        compare = NULL;

    if (compare && !PyCallable_Check (compare))
    {
        PyErr_SetString (PyExc_TypeError, "key argument must be callable");
        return NULL;
    }

    count = PySequence_Size (list);
    if (count == -1)
        return NULL;

    if (compare)
    {
        PyObject *ret;
        int retval;

        for (i = 0; i < count; i++)
        {
            frect = PySequence_ITEM (list, i);
            if (!PyFRect_Check (frect))
            {
                Py_XDECREF (frect);
                PyErr_SetString (PyExc_TypeError,
                    "rects argument must be a sequence of FRect objects.");
                return NULL;
            }

            ret = PyObject_CallFunctionObjArgs (compare, self, frect, NULL);
            Py_DECREF (frect);
            if (!ret)
                return NULL;
            
            retval = PyObject_IsTrue (ret);
            Py_DECREF (ret);
            if (retval == 1)
                return PyInt_FromSsize_t (i);
            else if (retval == -1)
                return NULL;
        }
    }
    else
    {
        for (i = 0; i < count; i++)
        {
            frect = PySequence_ITEM (list, i);
            if (!PyFRect_Check (frect))
            {
                Py_XDECREF (frect);
                PyErr_SetString (PyExc_TypeError,
                    "rects argument must be a sequence of FRect objects.");
                return NULL;
            }
            rarg = (PyFRect*) frect;
            
            if (INTERSECT (rself, rarg))
            {
                Py_DECREF (frect);
                return PyInt_FromSsize_t (i);
            }
            Py_DECREF (frect);
        }
    }
    return PyFloat_FromDouble (-1.);
}

static PyObject*
_frect_collidelistall (PyObject *self, PyObject *args, PyObject *kwds)
{
    PyFRect *rarg, *rself = (PyFRect*) self;
    PyObject *list, *frect, *indices, *compare = NULL;
    Py_ssize_t i, count;
    
    static char *keys[] = { "rects", "key", NULL };
    if (!PyArg_ParseTupleAndKeywords (args, kwds, "O|O:collidelistall", keys,
            &list, &compare))
        return NULL;

    if (!PySequence_Check (list))
    {
        PyErr_SetString (PyExc_TypeError,
            "rects argument must be a sequence of FRect objects");
        return NULL;
    }

    if (compare == Py_None)
        compare = NULL;
    if (compare && !PyCallable_Check (compare))
    {
        PyErr_SetString (PyExc_TypeError, "key argument must be callable");
        return NULL;
    }

    count = PySequence_Size (list);
    if (count == -1)
        return NULL;

    indices = PyList_New (0);
    if (!indices)
        return NULL;

    if (compare)
    {
        PyObject *ret;
        int retval;

        for (i = 0; i < count; i++)
        {
            frect = PySequence_ITEM (list, i);
            if (!PyFRect_Check (frect))
            {
                Py_XDECREF (frect);
                Py_DECREF (indices);
                PyErr_SetString (PyExc_TypeError,
                    "rects argument must be a sequence of FRect objects.");
                return NULL;
            }

            ret = PyObject_CallFunctionObjArgs (compare, self, frect, NULL);
            Py_DECREF (frect);
            if (!ret)
            {
                Py_DECREF (indices);
                return NULL;
            }
            retval = PyObject_IsTrue (ret);
            Py_DECREF (ret);

            if (retval == 1)
            {
                PyObject *obj =  PyInt_FromSsize_t (i);
                if (PyList_Append (indices, obj) == -1)
                {
                    Py_DECREF (obj);
                    Py_DECREF (indices);
                    return NULL;
                }
                Py_DECREF (obj);
            }
            else if (retval == -1)
            {
                Py_DECREF (indices);
                return NULL;
            }
        }
    }
    else
    {
        for (i = 0; i < count; i++)
        {
            frect = PySequence_ITEM (list, i);
            if (!PyFRect_Check (frect))
            {
                Py_XDECREF (frect);
                Py_DECREF (indices);
                PyErr_SetString (PyExc_TypeError,
                    "rects argument must be a sequence of FRect objects.");
                return NULL;
            }
            rarg = (PyFRect*) frect;
            
            if (INTERSECT (rself, rarg))
            {
                PyObject *obj =  PyInt_FromSsize_t (i);
                if (PyList_Append (indices, obj) == -1)
                {
                    Py_DECREF (obj);
                    Py_DECREF (indices);
                    Py_DECREF (frect);
                    return NULL;
                }
                Py_DECREF (obj);
            }
            Py_DECREF (frect);
        }
    }
    return indices;
}

static PyObject*
_frect_collidedict (PyObject *self, PyObject *args, PyObject *kwds)
{
    PyFRect *rarg, *rself = (PyFRect*) self;
    PyObject *dict, *key, *val, *check = NULL, *compare = NULL;
    Py_ssize_t pos = 0;
    int cvalues = 0;

    static char *keys[] = { "rects", "checkvals", "key", NULL };
    if (!PyArg_ParseTupleAndKeywords (args, kwds, "O|OO:collidedict", keys,
            &dict, &check, &compare))
        return NULL;

    if (!PyDict_Check (dict))
    {
        PyErr_SetString (PyExc_TypeError, "rects argument must be a dict.");
        return NULL;
    }

    if (compare == Py_None)
        compare = NULL;
    if (compare && !PyCallable_Check (compare))
    {
        PyErr_SetString (PyExc_TypeError, "key argument must be callable");
        return NULL;
    }

    if (check)
    {
        cvalues = PyObject_IsTrue (check);
        if (cvalues == -1)
            return NULL;
    }

    if (compare)
    {
        PyObject *ret;
        int retval;

        while (PyDict_Next (dict, &pos, &key, &val))
        {
            if (cvalues)
            {
                if (!PyFRect_Check (val))
                {
                    PyErr_SetString (PyExc_TypeError, 
                        "rects argument must be a dict with FRect values.");
                    return NULL;
                }
                rarg = (PyFRect*) val;
            }
            else
            {
                if (!PyFRect_Check (key))
                {
                    PyErr_SetString (PyExc_TypeError, 
                        "rects argument must be a dict with FRect keys.");
                    return NULL;
                }
                rarg = (PyFRect*) key;
            }
        
            ret = PyObject_CallFunctionObjArgs (compare, self, (PyObject*)rarg,
                NULL);
            if (!ret)
                return NULL;
            retval = PyObject_IsTrue (ret);
            Py_DECREF (ret);

            if (retval == 1)
                return Py_BuildValue ("(OO)", key, val);
            else if (retval == -1)
                return NULL;
        }
    }
    else
    {
        while (PyDict_Next (dict, &pos, &key, &val))
        {
            if (cvalues)
            {
                if (!PyFRect_Check (val))
                {
                    PyErr_SetString (PyExc_TypeError, 
                        "argument must be a dict with FRect values.");
                    return NULL;
                }
                rarg = (PyFRect*) val;
            }
            else
            {
                if (!PyFRect_Check (key))
                {
                    PyErr_SetString (PyExc_TypeError, 
                        "argument must be a dict with FRect keys.");
                    return NULL;
                }
                rarg = (PyFRect*) key;
            }
            if (INTERSECT (rself, rarg))
                return Py_BuildValue ("(OO)", key, val);
    }
    }
    Py_RETURN_NONE;
}

static PyObject*
_frect_collidedictall (PyObject *self, PyObject *args, PyObject *kwds)
{
    PyFRect *rarg, *rself = (PyFRect*) self;
    PyObject *dict, *key, *val, *list, *check = NULL, *compare = NULL;
    Py_ssize_t pos = 0;
    int cvalues = 0;

    static char *keys[] = { "rects", "checkvals", "key", NULL };
    if (!PyArg_ParseTupleAndKeywords (args, kwds, "O|OO:collidedictall", keys,
            &dict, &check, &compare))
        return NULL;

    if (!PyDict_Check (dict))
    {
        PyErr_SetString (PyExc_TypeError, "rects argument must be a dict.");
        return NULL;
    }

    if (compare == Py_None)
        compare = NULL;
    if (compare && !PyCallable_Check (compare))
    {
        PyErr_SetString (PyExc_TypeError, "key argument must be callable");
        return NULL;
    }

    if (check)
    {
        cvalues = PyObject_IsTrue (check);
        if (cvalues == -1)
            return NULL;
    }

    list = PyList_New (0);
    if (!list)
        return NULL;
    
    if (compare)
    {
        PyObject *ret;
        int retval;

        while (PyDict_Next (dict, &pos, &key, &val))
        {
            if (cvalues)
            {
                if (!PyFRect_Check (val))
                {
                    Py_DECREF (list);
                    PyErr_SetString (PyExc_TypeError, 
                        "rects argument must be a dict with FRect values.");
                    return NULL;
                }
                rarg = (PyFRect*) val;
            }
            else
            {
                if (!PyFRect_Check (key))
                {
                    Py_DECREF (list);
                    PyErr_SetString (PyExc_TypeError, 
                        "rects argument must be a dict with FRect keys.");
                    return NULL;
                }
                rarg = (PyFRect*) key;
            }

            ret = PyObject_CallFunctionObjArgs (compare, self, (PyObject*)rarg,
                NULL);
            if (!ret)
            {
                Py_DECREF (list);
                return NULL;
            }
            retval = PyObject_IsTrue (ret);
            Py_DECREF (ret);

            if (retval == -1)
            {
                Py_DECREF (list);
                return NULL;
            }
            else if (retval == 1)
            {
                PyObject *obj = Py_BuildValue ("(OO)", key, val);
                if (!obj)
                {
                    Py_DECREF (list);
                    return NULL;
                }
                
                if (PyList_Append (list, obj) == -1)
                {
                    Py_DECREF (obj);
                    Py_DECREF (list);
                    return NULL;
                }
                Py_DECREF (obj);
            }
        }
    }
    else
    {
        while (PyDict_Next (dict, &pos, &key, &val))
        {
            if (cvalues)
            {
                if (!PyFRect_Check (val))
                {
                    Py_DECREF (list);
                    PyErr_SetString (PyExc_TypeError, 
                        "rects argument must be a dict with FRect values.");
                    return NULL;
                }
                rarg = (PyFRect*) val;
            }
            else
            {
                if (!PyFRect_Check (key))
                {
                    Py_DECREF (list);
                    PyErr_SetString (PyExc_TypeError, 
                        "rects argument must be a dict with FRect keys.");
                    return NULL;
                }
                rarg = (PyFRect*) key;
            }
            
            if (INTERSECT (rself, rarg))
            {
                PyObject *obj = Py_BuildValue ("(OO)", key, val);
                if (!obj)
                {
                    Py_DECREF (list);
                    return NULL;
                }
                
                if (PyList_Append (list, obj) == -1)
                {
                    Py_DECREF (obj);
                    Py_DECREF (list);
                    return NULL;
                }
                Py_DECREF (obj);
            }
        }
    }
    return list;
}

static PyObject*
_frect_round (PyObject *self)
{
    PyFRect *frect = (PyFRect*) self;
    PyRect *rect = (PyRect*) PyRect_Type.tp_new (&PyRect_Type, NULL, NULL);
    if (!rect)
        return NULL;
    rect->x = (pgint16) round (frect->x);
    rect->y = (pgint16) round (frect->y);
    rect->w = (pguint16) ((frect->w < 0) ? 0 : round (frect->w));
    rect->h = (pguint16) ((frect->h < 0) ? 0 : round (frect->h));

    return (PyObject*) rect;
}

static PyObject*
_frect_ceil (PyObject *self)
{
    PyFRect *frect = (PyFRect*) self;
    PyRect *rect = (PyRect*) PyRect_Type.tp_new (&PyRect_Type, NULL, NULL);
    if (!rect)
        return NULL;
    rect->x = (pgint16) ceil (frect->x);
    rect->y = (pgint16) ceil (frect->y);
    rect->w = (pguint16) ((frect->w < 0) ? 0 : ceil (frect->w));
    rect->h = (pguint16) ((frect->h < 0) ? 0 : ceil (frect->h));

    return (PyObject*) rect;
}

static PyObject*
_frect_floor (PyObject *self)
{
    PyFRect *frect = (PyFRect*) self;
    PyRect *rect = (PyRect*) PyRect_Type.tp_new (&PyRect_Type, NULL, NULL);
    if (!rect)
        return NULL;
    rect->x = (pgint16) floor (frect->x);
    rect->y = (pgint16) floor (frect->y);
    rect->w = (pguint16) ((frect->w < 0) ? 0 : floor (frect->w));
    rect->h = (pguint16) ((frect->h < 0) ? 0 : floor (frect->h));

    return (PyObject*) rect;
}

static PyObject*
_frect_trunc (PyObject *self)
{
    PyFRect *frect = (PyFRect*) self;
    PyRect *rect = (PyRect*) PyRect_Type.tp_new (&PyRect_Type, NULL, NULL);
    if (!rect)
        return NULL;
    rect->x = (pgint16) trunc (frect->x);
    rect->y = (pgint16) trunc (frect->y);
    rect->w = (pguint16) ((frect->w < 0) ? 0 : trunc (frect->w));
    rect->h = (pguint16) ((frect->h < 0) ? 0 : trunc (frect->h));

    return (PyObject*) rect;
}

static int
_frect_compare (PyObject *self, PyObject *other)
{
    PyFRect *rect = (PyFRect*) self;

    if (PyFRect_Check (other))
    {
        PyFRect *rect2 = (PyFRect*) other;

        if (rect->x != rect2->x)
            return rect->x < rect2->x ? -1 : 1;
        if (rect->y != rect2->y)
            return rect->y < rect2->y ? -1 : 1;
        if (rect->w != rect2->w)
            return rect->w < rect2->w ? -1 : 1;
        if (rect->h != rect2->h)
            return rect->h < rect2->h ? -1 : 1;
        return 0;
    }
    else if (PyRect_Check (other))
    {
        PyRect *rect2 = (PyRect*) other;
        double rx = (double) rect2->x;
        double ry = (double) rect2->y;
        double rw = (double) rect2->w;
        double rh = (double) rect2->h;

        if (rect2->x != rx)
            return rect2->x < rx ? -1 : 1;
        if (rect2->y != ry)
            return rect2->y < ry ? -1 : 1;
        if (rect2->w != rw)
            return rect2->w < rw ? -1 : 1;
        if (rect2->h != rh)
            return rect2->h < rh ? -1 : 1;
        return 0;
    }
     PyErr_SetString (PyExc_TypeError,
        "comparision value should be a Rect or FRect");
    return -1;
}

static PyObject*
_frect_richcompare (PyObject *o1, PyObject *o2, int opid)
{
    PyFRect tmp1, tmp2;
    PyRect *r = NULL;
    PyFRect *fr1 = NULL, *fr2 = NULL;
    int equal;

    if (PyRect_Check (o1))
    {
        r = (PyRect *) o1;
        tmp1.x = (double) r->x;
        tmp1.y = (double) r->y;
        tmp1.w = (double) r->w;
        tmp1.h = (double) r->h;
        fr1 = &tmp1;
    }
    else if (PyFRect_Check (o1))
        fr1 = (PyFRect *) o1;
    else
    {
        Py_INCREF (Py_NotImplemented);
        return Py_NotImplemented;
    }

    if (PyRect_Check (o2))
    {
        r = (PyRect *) o2;
        tmp2.x = (double) r->x;
        tmp2.y = (double) r->y;
        tmp2.w = (double) r->w;
        tmp2.h = (double) r->h;
        fr2 = &tmp2;
    }
    else if (PyFRect_Check(o2))
        fr2 = (PyFRect *) o2;
    else
    {
        Py_INCREF (Py_NotImplemented);
        return Py_NotImplemented;
    }

    equal = fr1->x == fr2->x && fr1->y == fr2->y &&
        fr1->w == fr2->w && fr1->h == fr2->h;

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
PyObject*
PyFRect_New (double x, double y, double w, double h)
{
    PyFRect *frect;

    if (w < 0 || h < 0)
    {
        PyErr_SetString (PyExc_ValueError,
            "width and height must not be negative");
        return NULL;
    }

    frect = (PyFRect*) PyFRect_Type.tp_new (&PyFRect_Type, NULL, NULL);
    if (!frect)
        return NULL;

    frect->x = x;
    frect->y = y;
    frect->w = w;
    frect->h = h;
    return (PyObject*) frect;
}

void
floatrect_export_capi (void **capi)
{
    capi[PYGAME_FRECT_FIRSTSLOT] = &PyFRect_Type;
    capi[PYGAME_FRECT_FIRSTSLOT+1] = PyFRect_New;
}
