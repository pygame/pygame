/*
  pygame - Python Game Library
  Copyright (C) 2000-2001 Pete Shinners

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
#define PYGAME_RECT_INTERNAL

#include "internals.h"
#include "pgbase.h"
#include "base_doc.h"

#define INTERSECT(A,B)                                          \
    (((A->x >= B->x && A->x < (pgint32)(B->x + B->w)) ||        \
        (B->x >= A->x && B->x < (pgint32)(A->x + A->w))) &&     \
        ((A->y >= B->y && A->y < (pgint32) (B->y + B->h))   ||  \
            (B->y >= A->y && B->y < (pgint32)(A->y + A->h))))

static int _rect_init (PyObject *cursor, PyObject *args, PyObject *kwds);
static void _rect_dealloc (PyRect *self);
static PyObject* _rect_repr (PyObject *self);

static PyObject* _rect_getx (PyObject *self, void *closure);
static int _rect_setx (PyObject *self, PyObject *value, void *closure);
static PyObject* _rect_gety (PyObject *self, void *closure);
static int _rect_sety (PyObject *self, PyObject *value, void *closure);
static PyObject* _rect_getwidth (PyObject *self, void *closure);
static int _rect_setwidth (PyObject *self, PyObject *value, void *closure);
static PyObject* _rect_getheight (PyObject *self, void *closure);
static int _rect_setheight (PyObject *self, PyObject *value, void *closure);
static PyObject* _rect_getbottom (PyObject *self, void *closure);
static int _rect_setbottom (PyObject *self, PyObject *value, void *closure);
static PyObject* _rect_getright (PyObject *self, void *closure);
static int _rect_setright (PyObject *self, PyObject *value, void *closure);
static PyObject* _rect_getcenterx (PyObject *self, void *closure);
static int _rect_setcenterx (PyObject *self, PyObject *value, void *closure);
static PyObject* _rect_getcentery (PyObject *self, void *closure);
static int _rect_setcentery (PyObject *self, PyObject *value, void *closure);
static PyObject* _rect_getcenter (PyObject *self, void *closure);
static int _rect_setcenter (PyObject *self, PyObject *value, void *closure);
static PyObject* _rect_getsize (PyObject *self, void *closure);
static int _rect_setsize (PyObject *self, PyObject *value, void *closure);
static PyObject* _rect_getmidtop (PyObject *self, void *closure);
static int _rect_setmidtop (PyObject *self, PyObject *value, void *closure);
static PyObject* _rect_getmidleft (PyObject *self, void *closure);
static int _rect_setmidleft (PyObject *self, PyObject *value, void *closure);
static PyObject* _rect_getmidbottom (PyObject *self, void *closure);
static int _rect_setmidbottom (PyObject *self, PyObject *value, void *closure);
static PyObject* _rect_getmidright (PyObject *self, void *closure);
static int _rect_setmidright (PyObject *self, PyObject *value, void *closure);
static PyObject* _rect_gettopleft (PyObject *self, void *closure);
static int _rect_settopleft (PyObject *self, PyObject *value, void *closure);
static PyObject* _rect_gettopright (PyObject *self, void *closure);
static int _rect_settopright (PyObject *self, PyObject *value, void *closure);
static PyObject* _rect_getbottomleft (PyObject *self, void *closure);
static int _rect_setbottomleft (PyObject *self, PyObject *value, void *closure);
static PyObject* _rect_getbottomright (PyObject *self, void *closure);
static int _rect_setbottomright (PyObject *self, PyObject *value,
    void *closure);

static PyObject* _rect_clip (PyObject* self, PyObject *args);
static PyObject* _rect_copy (PyObject* self);
static PyObject* _rect_move (PyObject* self, PyObject *args);
static PyObject* _rect_move_ip (PyObject* self, PyObject *args);
static PyObject* _rect_union (PyObject* self, PyObject *args);
static PyObject* _rect_union_ip (PyObject* self, PyObject *args);
static PyObject* _rect_inflate (PyObject* self, PyObject *args);
static PyObject* _rect_inflate_ip (PyObject* self, PyObject *args);
static PyObject* _rect_clamp (PyObject* self, PyObject *args);
static PyObject* _rect_clamp_ip (PyObject* self, PyObject *args);
static PyObject* _rect_fit (PyObject* self, PyObject *args);
static PyObject* _rect_contains (PyObject* self, PyObject *args);
static PyObject* _rect_collidepoint (PyObject *self, PyObject *args);
static PyObject* _rect_colliderect (PyObject *self, PyObject *args);
static PyObject* _rect_collidelist (PyObject *self, PyObject *args,
    PyObject *kwds);
static PyObject* _rect_collidelistall (PyObject *self, PyObject *args,
    PyObject *kwds);
static PyObject* _rect_collidedict (PyObject *self, PyObject *args,
    PyObject *kwds);
static PyObject* _rect_collidedictall (PyObject *self, PyObject *args,
    PyObject *kwds);

static int _rect_compare (PyObject *self, PyObject *other);
static PyObject* _rect_richcompare (PyObject *o1, PyObject *o2, int opid);

/**
 */
static PyMethodDef _rect_methods[] = {
    { "clip", _rect_clip, METH_VARARGS, DOC_BASE_RECT_CLIP },
    { "copy", (PyCFunction)_rect_copy, METH_NOARGS, DOC_BASE_RECT_COPY },
    { "move", _rect_move, METH_VARARGS, DOC_BASE_RECT_MOVE },
    { "move_ip",  _rect_move_ip, METH_VARARGS, DOC_BASE_RECT_MOVE_IP },
    { "union",  _rect_union, METH_VARARGS, DOC_BASE_RECT_UNION },
    { "union_ip", _rect_union_ip, METH_VARARGS, DOC_BASE_RECT_UNION_IP },
    { "inflate",  _rect_inflate, METH_VARARGS, DOC_BASE_RECT_INFLATE },
    { "inflate_ip", _rect_inflate_ip, METH_VARARGS, DOC_BASE_RECT_INFLATE_IP },
    { "clamp", _rect_clamp, METH_VARARGS, DOC_BASE_RECT_CLAMP },
    { "clamp_ip", _rect_clamp_ip, METH_VARARGS, DOC_BASE_RECT_CLAMP_IP },
    { "fit", _rect_fit, METH_VARARGS, DOC_BASE_RECT_FIT },
    { "contains", _rect_contains, METH_VARARGS, DOC_BASE_RECT_CONTAINS },
    { "collidepoint", _rect_collidepoint, METH_VARARGS,
      DOC_BASE_RECT_COLLIDEPOINT },
    { "colliderect", _rect_colliderect, METH_VARARGS,
      DOC_BASE_RECT_COLLIDERECT },
    { "collidelist", (PyCFunction) _rect_collidelist,
      METH_VARARGS | METH_KEYWORDS, DOC_BASE_RECT_COLLIDELIST },
    { "collidelistall", (PyCFunction) _rect_collidelistall,
      METH_VARARGS | METH_KEYWORDS, DOC_BASE_RECT_COLLIDELISTALL },
    { "collidedict", (PyCFunction) _rect_collidedict,
      METH_VARARGS | METH_KEYWORDS, DOC_BASE_RECT_COLLIDEDICT },
    { "collidedictall", (PyCFunction) _rect_collidedictall,
      METH_VARARGS | METH_KEYWORDS, DOC_BASE_RECT_COLLIDEDICTALL },
    { NULL, NULL, 0, NULL }
};

/**
 */
static PyGetSetDef _rect_getsets[] = {
    { "x", _rect_getx, _rect_setx, DOC_BASE_RECT_X, NULL },
    { "y", _rect_gety, _rect_sety, DOC_BASE_RECT_Y, NULL },
    { "width", _rect_getwidth, _rect_setwidth, DOC_BASE_RECT_WIDTH, NULL },
    { "w", _rect_getwidth, _rect_setwidth, DOC_BASE_RECT_WIDTH, NULL },
    { "height", _rect_getheight, _rect_setheight, DOC_BASE_RECT_HEIGHT, NULL },
    { "h", _rect_getheight, _rect_setheight, DOC_BASE_RECT_HEIGHT, NULL },
    { "size", _rect_getsize, _rect_setsize, DOC_BASE_RECT_SIZE, NULL },

    { "left", _rect_getx, _rect_setx, DOC_BASE_RECT_LEFT, NULL },
    { "top", _rect_gety, _rect_sety, DOC_BASE_RECT_TOP, NULL },
    { "bottom", _rect_getbottom, _rect_setbottom, DOC_BASE_RECT_BOTTOM, NULL },
    { "right", _rect_getright, _rect_setright, DOC_BASE_RECT_RIGHT, NULL },

    { "centerx", _rect_getcenterx, _rect_setcenterx, DOC_BASE_RECT_CENTERX,
      NULL },
    { "centery", _rect_getcentery, _rect_setcentery, DOC_BASE_RECT_CENTERY,
      NULL },
    { "center", _rect_getcenter, _rect_setcenter, DOC_BASE_RECT_CENTER, NULL },

    { "midtop", _rect_getmidtop, _rect_setmidtop, DOC_BASE_RECT_MIDTOP, NULL },
    { "midleft", _rect_getmidleft, _rect_setmidleft, DOC_BASE_RECT_MIDLEFT,
      NULL },
    { "midbottom", _rect_getmidbottom, _rect_setmidbottom,
      DOC_BASE_RECT_MIDBOTTOM, NULL },
    { "midright", _rect_getmidright, _rect_setmidright,
      DOC_BASE_RECT_MIDRIGHT, NULL },

    { "topleft", _rect_gettopleft, _rect_settopleft, DOC_BASE_RECT_TOPLEFT,
      NULL },
    { "topright", _rect_gettopright, _rect_settopright, DOC_BASE_RECT_TOPRIGHT,
      NULL },
    { "bottomleft", _rect_getbottomleft, _rect_setbottomleft,
      DOC_BASE_RECT_BOTTOMLEFT, NULL },
    { "bottomright", _rect_getbottomright, _rect_setbottomright,
      DOC_BASE_RECT_BOTTOMRIGHT, NULL },

    { NULL, NULL, NULL, NULL, NULL }
};

/**
 */
PyTypeObject PyRect_Type =
{
    TYPE_HEAD(NULL, 0)
    "base.Rect",                /* tp_name */
    sizeof (PyRect),            /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor) _rect_dealloc, /* tp_dealloc */
    0,                          /* tp_print */
    0,                          /* tp_getattr */
    0,                          /* tp_setattr */
#ifdef IS_PYTHON_3
    0,                          /* tp_compare is now tp_rserved */
#else
    (cmpfunc)_rect_compare,     /* tp_compare */
#endif
    (reprfunc)_rect_repr,       /* tp_repr */
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
    DOC_BASE_RECT,
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    _rect_richcompare,          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    _rect_methods,              /* tp_methods */
    0,                          /* tp_members */
    _rect_getsets,              /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc) _rect_init,      /* tp_init */
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
_rect_dealloc (PyRect *self)
{
    ((PyObject*)self)->ob_type->tp_free ((PyObject *) self);
}

static int
_rect_init (PyObject *self, PyObject *args, PyObject *kwds)
{
    pgint16 x, y;
    pgint32 w, h;

    if (!PyArg_ParseTuple (args, "iiii", &x, &y, &w, &h))
    {
        x = y = 0;
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "ii", &w, &h))
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
                    x = (pgint16) trunc (((PyFRect*)rect)->x);
                    y = (pgint16) trunc (((PyFRect*)rect)->y);
                    w = (pgint32) trunc (((PyFRect*)rect)->w);
                    h = (pgint32) trunc (((PyFRect*)rect)->h);
                }
                else if (!SizeFromObject (rect, &w, &h))
                    return -1;
            }
            else
            {
                if (!PointFromObject (pt, (int*)&x, (int*)&y))
                    return -1;
                if (!SizeFromObject (rect, &w, &h))
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

    ((PyRect*)self)->x = x;
    ((PyRect*)self)->y = y;
    ((PyRect*)self)->w = (pguint16)w;
    ((PyRect*)self)->h = (pguint16)h;

    return 0;
}

static PyObject*
_rect_repr (PyObject *self)
{
    PyRect *r = (PyRect*) self;
#if PY_VERSION_HEX < 0x02050000
    return Text_FromFormat ("Rect(%d, %d, %ld, %ld)", r->x, r->y, r->w, r->h);
#else
    return Text_FromFormat ("Rect(%d, %d, %u, %u)", r->x, r->y, r->w, r->h);
#endif
}

/* Rect getters/setters */
static PyObject*
_rect_getx (PyObject *self, void *closure)
{
    return PyLong_FromLong (((PyRect*)self)->x);
}

static int
_rect_setx (PyObject *self, PyObject *value, void *closure)
{
    pgint16 x;
    if (!IntFromObj (value, &x))
        return -1;
    ((PyRect*)self)->x = x;
    return 0;
}

static PyObject*
_rect_gety (PyObject *self, void *closure)
{
    return PyLong_FromLong (((PyRect*)self)->y);
}

static int
_rect_sety (PyObject *self, PyObject *value, void *closure)
{
    pgint16 y;
    if (!IntFromObj (value, &y))
        return -1;
    ((PyRect*)self)->y = y;
    return 0;
}

static PyObject*
_rect_getwidth (PyObject *self, void *closure)
{
    return PyLong_FromUnsignedLong ((unsigned long)((PyRect*)self)->w);
}

static int
_rect_setwidth (PyObject *self, PyObject *value, void *closure)
{
    pguint16 w;
    if (!UintFromObj (value, &w))
        return -1;
    ((PyRect*)self)->w = w;
    return 0;
}

static PyObject*
_rect_getheight (PyObject *self, void *closure)
{
    return PyLong_FromUnsignedLong ((unsigned long)((PyRect*)self)->h);
}

static int
_rect_setheight (PyObject *self, PyObject *value, void *closure)
{
    pguint16 h;
    if (!UintFromObj (value, &h))
        return -1;
    ((PyRect*)self)->h = h;
    return 0;
}

static PyObject*
_rect_getbottom (PyObject *self, void *closure)
{
    PyRect *r = (PyRect*) self;
    return PyLong_FromUnsignedLong ((unsigned long) (r->y + r->h));
}

static int
_rect_setbottom (PyObject *self, PyObject *value, void *closure)
{
    pgint16 bottom;
    if (!IntFromObj (value, &bottom))
        return -1;
    INT16_SUB_UINT16_LIMIT (bottom, ((PyRect*)self)->h, ((PyRect*)self)->y);
    return 0;
}

static PyObject*
_rect_getright (PyObject *self, void *closure)
{
    PyRect *r = (PyRect*) self;
    return PyLong_FromUnsignedLong ((unsigned long) (r->x + r->w));
}

static int
_rect_setright (PyObject *self, PyObject *value, void *closure)
{
    pgint16 right;
    if (!IntFromObj (value, &right))
        return -1;
    INT16_SUB_UINT16_LIMIT (right, ((PyRect*)self)->w, ((PyRect*)self)->x);
    return 0;
}

static PyObject*
_rect_getcenterx (PyObject *self, void *closure)
{
    PyRect *r = (PyRect*) self;
    return PyLong_FromUnsignedLong ((unsigned long) (r->x + (r->w >> 1)));
}

static int
_rect_setcenterx (PyObject *self, PyObject *value, void *closure)
{
    int centerx;
    if (!IntFromObj (value, &centerx))
        return -1;

    INT16_SUB_UINT16_LIMIT (centerx, (((PyRect*)self)->w >> 1),
        ((PyRect*)self)->x);
    return 0;
}

static PyObject*
_rect_getcentery (PyObject *self, void *closure)
{
    PyRect *r = (PyRect*) self;
    return PyLong_FromUnsignedLong ((unsigned long) (r->y + (r->h >> 1)));
}

static int
_rect_setcentery (PyObject *self, PyObject *value, void *closure)
{
    int centery;
    if (!IntFromObj (value, &centery))
        return -1;

    INT16_SUB_UINT16_LIMIT (centery, (((PyRect*)self)->h >> 1),
        ((PyRect*)self)->y);
    return 0;
}

static PyObject*
_rect_getcenter (PyObject *self, void *closure)
{
    PyRect *r = (PyRect*) self;
    return Py_BuildValue ("(ii)", r->x + (r->w >> 1), r->y + (r->h >> 1));
}

static int
_rect_setcenter (PyObject *self, PyObject *value, void *closure)
{
    pgint16 x, y;
    if (!PointFromObject (value, (int*)&x, (int*)&y))
        return -1;

    INT16_SUB_UINT16_LIMIT (x, (((PyRect*)self)->w >> 1), ((PyRect*)self)->x);
    INT16_SUB_UINT16_LIMIT (y, (((PyRect*)self)->h >> 1), ((PyRect*)self)->y);
    return 0;
}

static PyObject*
_rect_getsize (PyObject *self, void *closure)
{
    PyRect *r = (PyRect*) self;
    return Py_BuildValue ("(ii)", r->w, r->h);
}

static int
_rect_setsize (PyObject *self, PyObject *value, void *closure)
{
    pguint16 w, h;

    if (!SizeFromObject (value, (pgint32*)&w, (pgint32*)&h))
        return -1;

    ((PyRect*)self)->w = w;
    ((PyRect*)self)->h = h;
    return 0;
}

static PyObject*
_rect_getmidtop (PyObject *self, void *closure)
{
    PyRect *r = (PyRect*) self;
    return Py_BuildValue ("(ii)", r->x + (r->w >> 1), r->y);
}

static int
_rect_setmidtop (PyObject *self, PyObject *value, void *closure)
{
    pgint16 x, y;
    if (!PointFromObject (value, (int*)&x, (int*)&y))
        return -1;

    INT16_SUB_UINT16_LIMIT (x, (((PyRect*)self)->w >> 1), ((PyRect*)self)->x);
    ((PyRect*)self)->y = y;
    return 0;
}

static PyObject*
_rect_getmidleft (PyObject *self, void *closure)
{
    PyRect *r = (PyRect*) self;
    return Py_BuildValue ("(ii)", r->x, r->y +  (r->h >> 1));
}

static int
_rect_setmidleft (PyObject *self, PyObject *value, void *closure)
{
    pgint16 x, y;
    if (!PointFromObject (value, (int*)&x, (int*)&y))
        return -1;

    ((PyRect*)self)->x = x;
    INT16_SUB_UINT16_LIMIT (y, (((PyRect*)self)->h >> 1), ((PyRect*)self)->y);
    return 0;
}

static PyObject*
_rect_getmidbottom (PyObject *self, void *closure)
{
    PyRect *r = (PyRect*) self;
    return Py_BuildValue ("(ii)", r->x + (r->w >> 1), r->y + r->h);
}

static int
_rect_setmidbottom (PyObject *self, PyObject *value, void *closure)
{
    pgint16 x, y;
    if (!PointFromObject (value, (int*)&x, (int*)&y))
        return -1;

    INT16_SUB_UINT16_LIMIT (x, (((PyRect*)self)->w >> 1), ((PyRect*)self)->x);
    INT16_SUB_UINT16_LIMIT (y, ((PyRect*)self)->h, ((PyRect*)self)->y);
    return 0;
}

static PyObject*
_rect_getmidright (PyObject *self, void *closure)
{
    PyRect *r = (PyRect*) self;
    return Py_BuildValue ("(ii)", r->x + r->w, r->y + (r->h >> 1));
}

static int
_rect_setmidright (PyObject *self, PyObject *value, void *closure)
{
    pgint16 x, y;
    if (!PointFromObject (value, (int*)&x, (int*)&y))
        return -1;

    INT16_SUB_UINT16_LIMIT (x, ((PyRect*)self)->w, ((PyRect*)self)->x);
    INT16_SUB_UINT16_LIMIT (y, (((PyRect*)self)->h >> 1), ((PyRect*)self)->y);
    return 0;
}

static PyObject*
_rect_gettopleft (PyObject *self, void *closure)
{
    PyRect *r = (PyRect*) self;
    return Py_BuildValue ("(ii)", r->x, r->y);
}
static int
_rect_settopleft (PyObject *self, PyObject *value, void *closure)
{
    pgint16 x, y;
    if (!PointFromObject (value, (int*)&x, (int*)&y))
        return -1;

    ((PyRect*)self)->x = x;
    ((PyRect*)self)->y = y;
    return 0;
}

static PyObject*
_rect_gettopright (PyObject *self, void *closure)
{
    PyRect *r = (PyRect*) self;
    return Py_BuildValue ("(ii)", r->x + r->w, r->y);
}

static int
_rect_settopright (PyObject *self, PyObject *value, void *closure)
{
    pgint16 x, y;
    if (!PointFromObject (value, (int*)&x, (int*)&y))
        return -1;

    INT16_SUB_UINT16_LIMIT (x, ((PyRect*)self)->w, ((PyRect*)self)->x);
    ((PyRect*)self)->y = y;
    return 0;
}

static PyObject*
_rect_getbottomleft (PyObject *self, void *closure)
{
    PyRect *r = (PyRect*) self;
    return Py_BuildValue ("(ii)", r->x, r->y + r->h);
}

static int
_rect_setbottomleft (PyObject *self, PyObject *value, void *closure)
{
    pgint16 x, y;
    if (!PointFromObject (value, (int*)&x, (int*)&y))
        return -1;

    ((PyRect*)self)->x = x;
    INT16_SUB_UINT16_LIMIT (y, ((PyRect*)self)->h, ((PyRect*)self)->y);
    return 0;
}

static PyObject*
_rect_getbottomright (PyObject *self, void *closure)
{
    PyRect *r = (PyRect*) self;
    return Py_BuildValue ("(ii)", r->x + r->w, r->y + r->h);
}

static int
_rect_setbottomright (PyObject *self, PyObject *value, void *closure)
{
    pgint16 x, y;
    if (!PointFromObject (value, (int*)&x, (int*)&y))
        return -1;

    INT16_SUB_UINT16_LIMIT (x, ((PyRect*)self)->w, ((PyRect*)self)->x);
    INT16_SUB_UINT16_LIMIT (y, ((PyRect*)self)->h, ((PyRect*)self)->y);
    return 0;
}

/* Rect methods */
static PyObject*
_rect_clip (PyObject* self, PyObject *args)
{
    PyObject *rect;
    PyRect *rself, *rarg;

    pgint16 x, y;
    pguint16 w, h;

    pgint32 selfright, argright;
    pgint32 selfbottom, argbottom;

    if (!PyArg_ParseTuple (args, "O:clip", &rect))
        return NULL;
    if (!PyRect_Check (rect))
    {
        PyErr_SetString (PyExc_TypeError, "argument must be a Rect");
        return NULL;
    }

    rself = (PyRect*) self;
    rarg = (PyRect*) rect;

    INT16_ADD_UINT16_LIMIT (rself->x, rself->w, selfright);
    INT16_ADD_UINT16_LIMIT (rself->y, rself->h, selfbottom);
    INT16_ADD_UINT16_LIMIT (rarg->x, rarg->w, argright);
    INT16_ADD_UINT16_LIMIT (rarg->y, rarg->h, argbottom);

    /* Check left and right non-overlaps */
    if (rarg->x > selfright || rself->x > argright)
        return PyRect_New (0, 0, 0, 0);

    /* Check bottom and top non-overlaps */
    if (rarg->y > selfbottom || rself->y > argbottom)
        return PyRect_New (0, 0, 0, 0);

    /* Clip x and y by testing self in arg overlap */
    x = (rself->x >= rarg->x) ? rself->x : rarg->x;
    y = (rself->y >= rarg->y) ? rself->y : rarg->y;
    
    /* Clip width and height */
    if (selfright <= argright)
        w = (pguint16)(selfright - x);
    else
        w = (pguint16)(argright - x);

    if (selfbottom <= argbottom)
        h = (pguint16)(selfbottom - y);
    else
        h = (pguint16)(argbottom - y);

    return PyRect_New (x, y, w, h);
}

static PyObject*
_rect_move (PyObject* self, PyObject *args)
{
    PyRect *rect = (PyRect*) self;
    pgint16 x, y;

    if (!PyArg_ParseTuple (args, "ii:move", &x, &y))
    {
        PyObject *pos;
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "O:move", &pos))
            return NULL;
        if (!PointFromObject (pos, (int*)&x, (int*)&y))
            return NULL;
    }

    return PyRect_New (rect->x + x, rect->y + y, rect->w, rect->h);
}

static PyObject*
_rect_copy (PyObject* self)
{
    return PyRect_New (((PyRect*)self)->x, ((PyRect*)self)->y,
        ((PyRect*)self)->w, ((PyRect*)self)->h);
}

static PyObject*
_rect_move_ip (PyObject* self, PyObject *args)
{
    PyRect *rect = (PyRect*) self;
    pgint16 x, y;

    if (!PyArg_ParseTuple (args, "ii:move_ip", &x, &y))
    {
        PyObject *pos;
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "O:move_ip", &pos))
            return NULL;
        if (!PointFromObject (pos, (int*)&x, (int*)&y))
            return NULL;
    }
    rect->x = INT16_ADD_LIMIT (rect->x, x);
    rect->y = INT16_ADD_LIMIT (rect->y, y);
    Py_RETURN_NONE;
}

static PyObject*
_rect_union (PyObject* self, PyObject *args)
{
    PyObject *rect, *list;
    PyRect *rself, *rarg;
    Py_ssize_t count, i;
    pgint16 x, y;
    pgint32 r, b, t, q;

    rself = (PyRect*) self;
    if (!PyArg_ParseTuple (args, "O:union", &list))
        return NULL;

    if (!PySequence_Check (list))
    {
        if (!PyRect_Check (list))
        {
            PyErr_SetString (PyExc_TypeError, "argument must be a Rect");
            return NULL;
        }
        rarg = (PyRect*) list;

        x = MIN (rself->x, rarg->x);
        y = MIN (rself->y, rarg->y);
        INT16_ADD_UINT16_LIMIT (rself->x, rself->w, t);
        INT16_ADD_UINT16_LIMIT (rarg->x, rarg->w, q);
        r = MAX (t, q);
        INT16_ADD_UINT16_LIMIT (rself->y, rself->h, t);
        INT16_ADD_UINT16_LIMIT (rarg->y, rarg->h, q)
        b = MAX (t, q);
        return PyRect_New (x, y, (pguint16)(r - x), (pguint16)(b - y));
    }

    /* Sequence of rects. */
    x = rself->x;
    y = rself->y;
    INT16_ADD_UINT16_LIMIT (rself->x, rself->w, r);
    INT16_ADD_UINT16_LIMIT (rself->y, rself->h, b);
    count = PySequence_Size (list);
    if (count == -1)
        return NULL;

    for (i = 0; i < count; i++)
    {
        rect = PySequence_ITEM (list, i);
        if (!PyRect_Check (rect))
        {
            Py_XDECREF (rect);
            PyErr_SetString (PyExc_TypeError,
                "argument must be a sequence of Rect objects.");
            return NULL;
        }
        rarg = (PyRect*) rect;

        x = MIN (x, rarg->x);
        y = MIN (y, rarg->y);
        INT16_ADD_UINT16_LIMIT (rarg->x, rarg->w, t);
        r = MAX (r, t);
        INT16_ADD_UINT16_LIMIT (rarg->y, rarg->h, t);
        b = MAX (b, t);

        Py_DECREF (rect);
    }
    return PyRect_New (x, y, (pguint16)(r - x), (pguint16)(b - y));
}

static PyObject*
_rect_union_ip (PyObject* self, PyObject *args)
{
    PyObject *rect, *list;
    PyRect *rself, *rarg;
    Py_ssize_t count, i;
    pgint16 x, y;
    pgint32 r, b, t, q;

    rself = (PyRect*) self;
    
    if (!PyArg_ParseTuple (args, "O:union_ip", &list))
        return NULL;
    if (!PySequence_Check (list))
    {
        if (!PyRect_Check (list))
        {
            PyErr_SetString (PyExc_TypeError, "argument must be a Rect");
            return NULL;
        }
        rarg = (PyRect*) list;

        x = MIN (rself->x, rarg->x);
        y = MIN (rself->y, rarg->y);
        INT16_ADD_UINT16_LIMIT (rself->x, rself->w, t);
        INT16_ADD_UINT16_LIMIT (rarg->x, rarg->w, q);
        r = MAX (t, q);
        INT16_ADD_UINT16_LIMIT (rself->y, rself->h, t);
        INT16_ADD_UINT16_LIMIT (rarg->y, rarg->h, q)
        b = MAX (t, q);

        rself->x = x;
        rself->y = y;
        rself->w = (pguint16) (r - x);
        rself->h = (pguint16) (b - y);
        Py_RETURN_NONE;
    }

    /* Sequence of rects. */
    x = rself->x;
    y = rself->y;
    INT16_ADD_UINT16_LIMIT (rself->x, rself->w, r);
    INT16_ADD_UINT16_LIMIT (rself->y, rself->h, b);
    count = PySequence_Size (list);
    if (count == -1)
        return NULL;

    for (i = 0; i < count; i++)
    {
        rect = PySequence_ITEM (list, i);
        if (!PyRect_Check (rect))
        {
            Py_XDECREF (rect);
            PyErr_SetString (PyExc_TypeError,
                "argument must be a sequence of Rect objects.");
            return NULL;
        }
        rarg = (PyRect*) rect;

        x = MIN (x, rarg->x);
        y = MIN (y, rarg->y);
        INT16_ADD_UINT16_LIMIT (rarg->x, rarg->w, t);
        r = MAX (r, t);
        INT16_ADD_UINT16_LIMIT (rarg->y, rarg->h, t);
        b = MAX (b, t);

        Py_DECREF (rect);
    }
    rself->x = x;
    rself->y = y;
    rself->w = (pguint16)(r - x);
    rself->h = (pguint16)(b - y);

    Py_RETURN_NONE;
}

static PyObject*
_rect_inflate (PyObject* self, PyObject *args)
{
    PyRect *rect = (PyRect*) self;
    pgint16 x, y;
    pgint32 w, h;

    if (!PyArg_ParseTuple (args, "ii:inflate", &x, &y))
    {
        PyObject *pos;
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "O:inflate", &pos))
            return NULL;
        if (!PointFromObject (pos, (int*)&x, (int*)&y))
            return NULL;
    }

    w = (pgint32)(rect->w + x);
    h = (pgint32)(rect->h + y);
    
    return PyRect_New (INT16_SUB_LIMIT (rect->x, x / 2),
        INT16_SUB_LIMIT (rect->y, y / 2),
        MIN ((pguint16)w, UINT_MAX),
        MIN ((pguint16)h, UINT_MAX));
}

static PyObject*
_rect_inflate_ip (PyObject* self, PyObject *args)
{
    PyRect *rect = (PyRect*) self;
    pgint16 x, y;
    pgint32 w, h;

    if (!PyArg_ParseTuple (args, "ii:inflate_ip", &x, &y))
    {
        PyObject *pos;
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "O:inflate_ip", &pos))
            return NULL;
        if (!PointFromObject (pos, (int*)&x, (int*)&y))
            return NULL;
    }

    rect->x = INT16_SUB_LIMIT (rect->x, x / 2);
    rect->y = INT16_SUB_LIMIT (rect->y, y / 2);
    w = (pgint32)(rect->w + x);
    h = (pgint32)(rect->h + y);
    rect->w = MIN ((pguint16)w, UINT_MAX);
    rect->h = MIN ((pguint16)h, UINT_MAX);

    Py_RETURN_NONE;
}

static PyObject*
_rect_clamp (PyObject* self, PyObject *args)
{
    PyRect *rself, *rarg;
    pgint16 x, y, t;

    if (!PyArg_ParseTuple (args, "O:clamp", &rarg))
        return NULL;
    if (!PyRect_Check (rarg))
    {
        PyErr_SetString (PyExc_TypeError, "argument must be a Rect");
        return NULL;
    }
    rself = (PyRect*) self;

    if (rself->w >= rarg->w)
    {
        INT16_ADD_UINT16_LIMIT (rarg->x, rarg->w / 2, t);
        INT16_SUB_UINT16_LIMIT (t, rself->w / 2, x);
    }
    else if (rself->x < rarg->x)
        x = rarg->x;
    else if (rself->x + rself->w > rarg->x + rarg->w)
    {
        INT16_ADD_UINT16_LIMIT (rarg->x, rarg->w, t);
        INT16_SUB_UINT16_LIMIT (t, rself->w, x);
    }
    else
        x = rself->x;

    if (rself->h >= rarg->h)
    {
        INT16_ADD_UINT16_LIMIT (rarg->y, rarg->h / 2, t);
        INT16_SUB_UINT16_LIMIT (t, rself->h / 2, y);
    }
    else if (rself->y < rarg->y)
        y = rarg->y;
    else if (rself->y + rself->h > rarg->y + rarg->h)
    {
        INT16_ADD_UINT16_LIMIT (rarg->y, rarg->h, t)
        INT16_SUB_UINT16_LIMIT (t, rself->h, y);
    }
    else
        y = rself->y;

    return PyRect_New (x, y, rself->w, rself->h);
}

static PyObject*
_rect_clamp_ip (PyObject* self, PyObject *args)
{
    PyRect *rself, *rarg;
    pgint16 t;

    if (!PyArg_ParseTuple (args, "O:clamp_ip", &rarg))
        return NULL;
    if (!PyRect_Check (rarg))
    {
        PyErr_SetString (PyExc_TypeError, "argument must be a Rect");
        return NULL;
    }
    rself = (PyRect*) self;

    if (rself->w >= rarg->w)
    {
        INT16_ADD_UINT16_LIMIT (rarg->x, rarg->w / 2, t);
        INT16_SUB_UINT16_LIMIT (t, rself->w / 2, rself->x);
    }
    else if (rself->x < rarg->x)
        rself->x = rarg->x;
    else if (rself->x + rself->w > rarg->x + rarg->w)
    {
        INT16_ADD_UINT16_LIMIT (rarg->x, rarg->w, t);
        INT16_SUB_UINT16_LIMIT (t, rself->w, rself->x);
    }
    else
        rself->x = rself->x;

    if (rself->h >= rarg->h)
    {
        INT16_ADD_UINT16_LIMIT (rarg->y, rarg->h / 2, t);
        INT16_SUB_UINT16_LIMIT (t, rself->h / 2, rself->y);
    }
    else if (rself->y < rarg->y)
        rself->y = rarg->y;
    else if (rself->y + rself->h > rarg->y + rarg->h)
    {
        INT16_ADD_UINT16_LIMIT (rarg->y, rarg->h, t);
        INT16_SUB_UINT16_LIMIT (t, rself->h, rself->y);
    }
    else
        rself->y = rself->y;
    Py_RETURN_NONE;
}

static PyObject*
_rect_fit (PyObject* self, PyObject *args)
{
    PyRect *rself, *rarg;
    float xratio, yratio, maxratio;
    pgint16 x, y;
    pguint16 w, h;
    
    rself = (PyRect*) self;

    if (!PyArg_ParseTuple (args, "O:fit", &rarg))
        return NULL;

    if (!PyRect_Check (rarg))
    {
        PyErr_SetString (PyExc_TypeError, "argument must be a Rect");
        return NULL;
    }

    xratio = (float) rself->w / (float) rarg->w;
    yratio = (float) rself->h / (float) rarg->h;
    maxratio = (xratio > yratio) ? xratio : yratio;

    w = (pguint16) (rself->w / maxratio);
    h = (pguint16) (rself->h / maxratio);
    INT16_ADD_UINT16_LIMIT (rarg->x, (rarg->w - w) / 2, x);
    INT16_ADD_UINT16_LIMIT (rarg->y, (rarg->h - h) / 2, y);

    return PyRect_New (x, y, w, h);
}

static PyObject*
_rect_contains (PyObject* self, PyObject *args)
{
    PyRect* rself, *rarg;
    pgint32 ar, br, ab, bb;

    if (!PyArg_ParseTuple (args, "O:contains", &rarg))
        return NULL;

    if (!PyRect_Check (rarg))
    {
        PyErr_SetString (PyExc_TypeError, "argument must be a Rect");
        return NULL;
    }
    rself = (PyRect*) self;

    ar = rself->x + rself->w;
    ab = rself->y + rself->h;
    br = rarg->x + rarg->w;
    bb = rarg->y + rarg->h;

    if ((rself->x <= rarg->x) && (rself->y <= rarg->y) && (ar >= br) &&
        (ab >= bb) && (ar > rarg->x) && (ab > rarg->y))
    {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyObject*
_rect_collidepoint (PyObject *self, PyObject *args)
{
    PyRect *rself = (PyRect*) self;
    pgint16 x, y;
    pgint32 r, b;

    if (!PyArg_ParseTuple (args, "ii:collidepoint", &x, &y))
    {
        PyObject *pos;
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "O:collidepoint", &pos))
            return NULL;
        if (!PointFromObject (pos, (int*)&x, (int*)&y))
            return NULL;
    }

    r = rself->x + rself->w;
    b = rself->y + rself->h;

    if (x >= rself->x && x < r && y >= rself->y && y < b)
    {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyObject*
_rect_colliderect (PyObject *self, PyObject *args)
{
    PyRect *rarg, *rself = (PyRect*) self;
    PyObject *rect;

    if (!PyArg_ParseTuple (args, "O:colliderect", &rect))
        return NULL;
    if (!PyRect_Check (rect))
    {
        PyErr_SetString (PyExc_TypeError, "argument must be a Rect");
        return NULL;
    }
    rarg = (PyRect*) rect;

    if (INTERSECT (rself, rarg))
    {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyObject*
_rect_collidelist (PyObject *self, PyObject *args, PyObject *kwds)
{
    PyRect *rarg, *rself = (PyRect*) self;
    PyObject *list, *rect, *compare = NULL;
    Py_ssize_t i, count;

    static char *keys[] = { "rects", "key", NULL };
    if (!PyArg_ParseTupleAndKeywords (args, kwds, "O|O:collidelist", keys,
            &list, &compare))
        return NULL;
    if (!PySequence_Check (list))
    {
        PyErr_SetString (PyExc_TypeError,
            "rects argument must be a sequence of Rect objects");
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
            rect = PySequence_ITEM (list, i);
            if (!PyRect_Check (rect))
            {
                Py_XDECREF (rect);
                PyErr_SetString (PyExc_TypeError,
                    "rects argument must be a sequence of Rect objects.");
                return NULL;
            }

            ret = PyObject_CallFunctionObjArgs (compare, self, rect, NULL);
            Py_DECREF (rect);
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
            rect = PySequence_ITEM (list, i);
            if (!PyRect_Check (rect))
            {
                Py_XDECREF (rect);
                PyErr_SetString (PyExc_TypeError,
                    "rects argument must be a sequence of Rect objects.");
                return NULL;
            }
            rarg = (PyRect*) rect;

            if (INTERSECT (rself, rarg))
            {
                Py_DECREF (rect);
                return PyInt_FromSsize_t (i);
            }
            Py_DECREF (rect);
        }
    }
    return PyLong_FromLong (-1);
}

static PyObject*
_rect_collidelistall (PyObject *self, PyObject *args, PyObject *kwds)
{
    PyRect *rarg, *rself = (PyRect*) self;
    PyObject *list, *rect, *indices, *compare = NULL;
    Py_ssize_t i, count;
    
    static char *keys[] = { "rects", "key", NULL };
    if (!PyArg_ParseTupleAndKeywords (args, kwds, "O|O:collidelistall", keys,
            &list, &compare))
        return NULL;

    if (!PySequence_Check (list))
    {
        PyErr_SetString (PyExc_TypeError,
            "rects must be a sequence of Rect objects");
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
            rect = PySequence_ITEM (list, i);
            if (!PyRect_Check (rect))
            {
                Py_XDECREF (rect);
                Py_DECREF (indices);
                PyErr_SetString (PyExc_TypeError,
                    "rects argument must be a sequence of Rect objects.");
                return NULL;
            }
            
            ret = PyObject_CallFunctionObjArgs (compare, self, rect, NULL);
            Py_DECREF (rect);
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
            rect = PySequence_ITEM (list, i);
            if (!PyRect_Check (rect))
            {
                Py_XDECREF (rect);
                Py_DECREF (indices);
                PyErr_SetString (PyExc_TypeError,
                    "rects argument must be a sequence of Rect objects.");
                return NULL;
            }
            rarg = (PyRect*) rect;
            
            if (INTERSECT (rself, rarg))
            {
                PyObject *obj =  PyInt_FromSsize_t (i);
                if (PyList_Append (indices, obj) == -1)
                {
                    Py_DECREF (obj);
                    Py_DECREF (indices);
                    Py_DECREF (rect);
                    return NULL;
                }
                Py_DECREF (obj);
            }
            Py_DECREF (rect);
        }
    }

    return indices;
}

static PyObject*
_rect_collidedict (PyObject *self, PyObject *args, PyObject *kwds)
{
    PyRect *rarg, *rself = (PyRect*) self;
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
                if (!PyRect_Check (val))
                {
                    PyErr_SetString (PyExc_TypeError, 
                        "rects argument must be a dict with Rect values.");
                    return NULL;
                }
                rarg = (PyRect*) val;
            }
            else
            {
                if (!PyRect_Check (key))
                {
                    PyErr_SetString (PyExc_TypeError, 
                        "rects argument must be a dict with Rect keys.");
                    return NULL;
                }
                rarg = (PyRect*) key;
            }

            ret = PyObject_CallFunctionObjArgs (compare, self,
                (PyObject*)rarg, NULL);
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
                if (!PyRect_Check (val))
                {
                    PyErr_SetString (PyExc_TypeError, 
                        "rects argument must be a dict with Rect values.");
                    return NULL;
                }
                rarg = (PyRect*) val;
            }
            else
            {
                if (!PyRect_Check (key))
                {
                    PyErr_SetString (PyExc_TypeError, 
                        "rects argument must be a dict with Rect keys.");
                    return NULL;
                }
                rarg = (PyRect*) key;
            }
            
            if (INTERSECT (rself, rarg))
                return Py_BuildValue ("(OO)", key, val);
        }
    }
    Py_RETURN_NONE;
}

static PyObject*
_rect_collidedictall (PyObject *self, PyObject *args, PyObject *kwds)
{
    PyRect *rarg, *rself = (PyRect*) self;
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
                if (!PyRect_Check (val))
                {
                    Py_DECREF (list);
                    PyErr_SetString (PyExc_TypeError, 
                        "rects argument must be a dict with Rect values.");
                    return NULL;
                }
                rarg = (PyRect*) val;
            }
            else
            {
                if (!PyRect_Check (key))
                {
                    Py_DECREF (list);
                    PyErr_SetString (PyExc_TypeError, 
                        "rects argument must be a dict with Rect keys.");
                    return NULL;
                }
                rarg = (PyRect*) key;
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
                if (!PyRect_Check (val))
                {
                    PyErr_SetString (PyExc_TypeError, 
                        "rects argument must be a dict with Rect values.");
                    return NULL;
                }
                rarg = (PyRect*) val;
            }
            else
            {
                if (!PyRect_Check (key))
                {
                    PyErr_SetString (PyExc_TypeError, 
                        "rects argument must be a dict with Rect keys.");
                    return NULL;
                }
                rarg = (PyRect*) key;
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


static int
_rect_compare (PyObject *self, PyObject *other)
{
    PyRect *rect = (PyRect*) self;

    if (PyFRect_Check (other))
    {
        PyFRect *rect2 = (PyFRect*) other;
        pgint16 rx = (pgint16) trunc(rect2->x);
        pgint16 ry = (pgint16) trunc(rect2->y);
        pguint16 rw = (pguint16) trunc(rect2->w);
        pguint16 rh = (pguint16) trunc(rect2->h);

        if (rect->x != rx)
            return rect->x < rx ? -1 : 1;
        if (rect->y != ry)
            return rect->y < ry ? -1 : 1;
        if (rect->w != rw)
            return rect->w < rw ? -1 : 1;
        if (rect->h != rh)
            return rect->h < rh ? -1 : 1;
        return 0;
    }
    else if (PyRect_Check (other))
    {
        PyRect *rect2 = (PyRect*) other;

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
     PyErr_SetString (PyExc_TypeError,
        "comparision value should be a Rect or FRect");
    return -1;
}

static PyObject*
_rect_richcompare (PyObject *o1, PyObject *o2, int opid)
{
    PyRect tmp1, tmp2;
    PyRect *r1 = NULL, *r2 = NULL;
    PyFRect *fr = NULL;
    int equal;

    if (PyFRect_Check (o1))
    {
        fr = (PyFRect *) o1;
        tmp1.x = (pgint16) trunc (fr->x);
        tmp1.y = (pgint16) trunc (fr->y);
        tmp1.w = (pguint16) trunc (fr->w);
        tmp1.h = (pguint16) trunc (fr->h);
        r1 = &tmp1;
    }
    else if (PyRect_Check (o1))
        r1 = (PyRect *) o1;
    else
    {
        Py_INCREF (Py_NotImplemented);
        return Py_NotImplemented;
    }

    if (PyFRect_Check (o2))
    {
        fr = (PyFRect *) o2;
        tmp2.x = (pgint16) trunc (fr->x);
        tmp2.y = (pgint16) trunc (fr->y);
        tmp2.w = (pguint16) trunc (fr->w);
        tmp2.h = (pguint16) trunc (fr->h);
        r2 = &tmp2;
    }
    else if (PyRect_Check(o2))
        r2 = (PyRect *) o2;
    else
    {
        Py_INCREF (Py_NotImplemented);
        return Py_NotImplemented;
    }

    equal = r1->x == r2->x && r1->y == r2->y &&
        r1->w == r2->w && r1->h == r2->h;

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
PyRect_New (pgint16 x, pgint16 y, pguint16 w, pguint16 h)
{
    PyRect *rect = (PyRect*) PyRect_Type.tp_new (&PyRect_Type, NULL, NULL);
    if (!rect)
        return NULL;

    rect->x = x;
    rect->y = y;
    rect->w = w;
    rect->h = h;
    return (PyObject*) rect;
}

void
rect_export_capi (void **capi)
{
    capi[PYGAME_RECT_FIRSTSLOT] = &PyRect_Type;
    capi[PYGAME_RECT_FIRSTSLOT+1] = PyRect_New;
}
