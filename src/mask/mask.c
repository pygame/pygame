/*
  Copyright (C) 2002-2007 Ulf Ekstrom except for the bitcount function.
  This wrapper code was originally written by Danny van Bruggen(?) for
  the SCAM library, it was then converted by Ulf Ekstrom to wrap the
  bitmask library, a spinoff from SCAM.

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
#define PYGAME_MASK_INTERNAL

#include "maskmod.h"
#include "pgmask.h"
#include "pgbase.h"
#include "mask_doc.h"

static int _largest_connected_comp (bitmask_t* input, bitmask_t* output,
    int ccx, int ccy);
static unsigned int _cc_label (bitmask_t *input, unsigned int* image,
    unsigned int* ufind, unsigned int* largest);
static int _get_bounding_rects (bitmask_t *input,
    int *num_bounding_boxes, CRect** ret_rects);

static int _get_connected_components (bitmask_t *mask, bitmask_t ***components,
    int min);

static PyObject *_mask_new (PyTypeObject *type, PyObject *args, PyObject *kwds);
static int _mask_init (PyObject *mask, PyObject *args, PyObject *kwds);
static void _mask_dealloc (PyMask *self);
static PyObject *_mask_repr (PyObject *self);

static PyObject* _mask_getsize (PyObject* self, void *closure);
static PyObject* _mask_getwidth (PyObject* self, void *closure);
static PyObject* _mask_getheight (PyObject* self, void *closure);
static PyObject* _mask_count (PyObject* self, void *closure);
static PyObject* _mask_centroid (PyObject* self, void *closure);
static PyObject* _mask_angle (PyObject* self, void *closure);

static PyObject* _mask_getat (PyObject* self, PyObject* args);
static PyObject* _mask_setat (PyObject* self, PyObject* args);
static PyObject* _mask_overlap (PyObject* self, PyObject* args);
static PyObject* _mask_overlaparea (PyObject* self, PyObject* args);
static PyObject* _mask_overlapmask (PyObject* self, PyObject* args);
static PyObject* _mask_fill (PyObject* self);
static PyObject* _mask_clear (PyObject* self);
static PyObject* _mask_invert (PyObject* self);
static PyObject* _mask_scale (PyObject* self, PyObject* args);
static PyObject* _mask_draw (PyObject* self, PyObject* args);
static PyObject* _mask_erase (PyObject* self, PyObject* args);
static PyObject* _mask_outline(PyObject* self, PyObject* args);
static PyObject* _mask_connectedcomponent (PyObject* self, PyObject* args);
static PyObject* _mask_connectedcomponents (PyObject* self, PyObject* args);
static PyObject* _mask_getboundingrects (PyObject* self);
static PyObject* _mask_convolve (PyObject* self, PyObject* args);

static PyGetSetDef _mask_getsets[] = {
    { "size", _mask_getsize, NULL, DOC_MASK_MASK, NULL },
    { "width", _mask_getwidth, NULL, DOC_MASK_MASK_WIDTH, NULL },
    { "height", _mask_getheight, NULL, DOC_MASK_MASK_HEIGHT, NULL },
    { "count", _mask_count, NULL, DOC_MASK_MASK_COUNT, NULL },
    { "centroid", _mask_centroid, NULL, DOC_MASK_MASK_CENTROID, NULL },
    { "angle", _mask_angle, NULL, DOC_MASK_MASK_ANGLE, NULL },
    { NULL, NULL, NULL, NULL, NULL },
};

static PyMethodDef _mask_methods[] = {
    { "get_at", _mask_getat, METH_VARARGS, DOC_MASK_MASK_GET_AT },
    { "set_at", _mask_setat, METH_VARARGS, DOC_MASK_MASK_SET_AT },
    { "overlap", _mask_overlap, METH_VARARGS, DOC_MASK_MASK_OVERLAP },
    { "overlap_area", _mask_overlaparea, METH_VARARGS,
      DOC_MASK_MASK_OVERLAP_AREA },
    { "overlap_mask", _mask_overlapmask, METH_VARARGS,
      DOC_MASK_MASK_OVERLAP_MASK },
    { "fill", (PyCFunction) _mask_fill, METH_NOARGS, DOC_MASK_MASK_FILL },
    { "clear", (PyCFunction) _mask_clear, METH_NOARGS, DOC_MASK_MASK_CLEAR },
    { "invert", (PyCFunction) _mask_invert, METH_NOARGS, DOC_MASK_MASK_INVERT },
    { "scale", _mask_scale, METH_VARARGS, DOC_MASK_MASK_SCALE },
    { "draw", _mask_draw, METH_VARARGS, DOC_MASK_MASK_DRAW },
    { "erase", _mask_erase, METH_VARARGS, DOC_MASK_MASK_ERASE },
    { "outline", _mask_outline, METH_VARARGS, "" },
    { "connected_component", _mask_connectedcomponent, METH_VARARGS,
      DOC_MASK_MASK_CONNECTED_COMPONENT },
    { "connected_components", _mask_connectedcomponents, METH_VARARGS, 
      DOC_MASK_MASK_CONNECTED_COMPONENT },
    { "get_bounding_rects",(PyCFunction) _mask_getboundingrects, METH_NOARGS,
      DOC_MASK_MASK_GET_BOUNDING_RECTS },
    { "convolve", _mask_convolve, METH_VARARGS, DOC_MASK_MASK_CONVOLVE },
    { NULL, NULL, 0, NULL }
};

PyTypeObject PyMask_Type =
{
    TYPE_HEAD(NULL,0)
    "mask.Mask",                /* tp_name */
    sizeof (PyMask),            /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor) _mask_dealloc, /* tp_dealloc */
    0,                          /* tp_print */
    0,                          /* tp_getattr */
    0,                          /* tp_setattr */
    0,                          /* tp_compare */
    (reprfunc)_mask_repr,       /* tp_repr */
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
    DOC_MASK_MASK,
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    _mask_methods,              /* tp_methods */
    0,                          /* tp_members */
    _mask_getsets,              /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc) _mask_init,      /* tp_init */
    0,                          /* tp_alloc */
    _mask_new,                  /* tp_new */
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
_mask_dealloc (PyMask *self)
{
    if (self->mask)
        bitmask_free (self->mask);
    ((PyObject*)self)->ob_type->tp_free ((PyObject *) self);
}

static PyObject*
_mask_new (PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyMask* mask = (PyMask*) type->tp_alloc (type, 0);
    if (!mask)
        return NULL;
    mask->mask = NULL;
    return (PyObject*) mask;
}

static int
_mask_init (PyObject *mask, PyObject *args, PyObject *kwds)
{
    pgint32 w, h;
    bitmask_t *m;

    if (!PyArg_ParseTuple (args, "ii", &w, &h))
    {
        PyObject *size;
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "O", &size))
            return -1;
        if (!SizeFromObj (size, &w, &h))
            return -1;
    }
    if (w <= 0 || h <= 0)
    {
        PyErr_SetString (PyExc_ValueError,
            "bitmask dimensions must be positive");
        return -1;
    }

    m = bitmask_create ((int)w, (int)h);
    if (!m)
    {
        PyErr_SetString (PyExc_MemoryError, "memory allocation failed");
        return -1;
    }
    ((PyMask*)mask)->mask = m;
    return 0;
}

static PyObject*
_mask_repr (PyObject *self)
{
    PyMask *mask = (PyMask*) self;
    return Text_FromFormat ("<Mask(%d, %d)>", mask->mask->w, mask->mask->h);
}

/* Getters/Setters */
static PyObject*
_mask_getsize (PyObject* self, void *closure)
{
    bitmask_t *mask = PyMask_AsBitmask (self);
    return Py_BuildValue ("(ii)", mask->w, mask->h);
}

static PyObject*
_mask_getwidth (PyObject* self, void *closure)
{
    bitmask_t *mask = PyMask_AsBitmask (self);
    return PyInt_FromLong (mask->w);
}

static PyObject*
_mask_getheight (PyObject* self, void *closure)
{
    bitmask_t *mask = PyMask_AsBitmask (self);
    return PyInt_FromLong (mask->h);
}

static PyObject*
_mask_count (PyObject* self, void *closure)
{
    bitmask_t *m = PyMask_AsBitmask (self);
    return PyLong_FromUnsignedLong ((unsigned long) bitmask_count (m));
}

static PyObject*
_mask_centroid (PyObject* self, void *closure)
{
    bitmask_t *mask = PyMask_AsBitmask(self);
    int x, y;
    long int m10, m01, m00;
    PyObject *ret, *xobj, *yobj;

    m10 = m01 = m00 = 0;
    
    for (x = 0; x < mask->w; x++)
    {
        for (y = 0; y < mask->h; y++)
        {
            if (bitmask_getbit (mask, x, y))
            {
                m10 += x;
                m01 += y;
                m00++;
            }
        }
    }
    
    if (m00)
    {
        xobj = PyInt_FromLong (m10/m00);
        yobj = PyInt_FromLong (m01/m00);
    }
    else
    {
        xobj = PyInt_FromLong (0);
        yobj = PyInt_FromLong (0);
    }
    
    ret = Py_BuildValue ("(NN)", xobj, yobj);
    if (!ret)
        return NULL;

    return ret;
}

static PyObject*
_mask_angle (PyObject* self, void *closure)
{
    bitmask_t *mask = PyMask_AsBitmask (self);
    int x, y, xc, yc;
    long int m10, m01, m00, m20, m02, m11;
    double theta;

    m10 = m01 = m00 = m20 = m02 = m11 = 0;
    
    for (x = 0; x < mask->w; x++)
    {
        for (y = 0; y < mask->h; y++)
        {
            if (bitmask_getbit(mask, x, y))
            {
                m10 += x;
                m20 += x*x;
                m11 += x*y;
                m02 += y*y;
                m01 += y;
                m00++;
            }
        }
    }
    
    if (m00)
    {
        xc = m10/m00;
        yc = m01/m00;
        theta = -90.0 * atan2(2. * (m11/m00 - xc*yc),
            (float)((m20/m00 - xc*xc) - (m02/m00 - yc*yc))) / M_PI;
        return PyFloat_FromDouble (theta);
    }
    return PyFloat_FromDouble(0.);
}

/* Methods */
static PyObject*
_mask_getat (PyObject* self, PyObject* args)
{
    bitmask_t *mask = PyMask_AsBitmask (self);
    int x, y, val;

    if (!PyArg_ParseTuple (args, "ii", &x, &y))
    {
        PyObject *pt;
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "O", &pt))
            return NULL;
        if (!PointFromObj (pt, &x, &y))
            return NULL;
    }

    if (x >= 0 && x < mask->w && y >= 0 && y < mask->h)
    {
        val = bitmask_getbit (mask, x, y);
    }
    else
    {
        PyErr_Format (PyExc_IndexError, "%d, %d is out of bounds", x, y);
        return NULL;
    }

    return PyInt_FromLong (val);
}

static PyObject*
_mask_setat (PyObject* self, PyObject* args)
{
    bitmask_t *mask = PyMask_AsBitmask (self);
    int x, y, value = 1;

    if (!PyArg_ParseTuple (args, "ii|i", &x, &y, &value))
    {
        PyObject *pt;
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "O|i", &pt, &value))
            return NULL;
        if (!PointFromObj (pt, &x, &y))
            return NULL;
    }

    if (x >= 0 && x < mask->w && y >= 0 && y < mask->h)
    {
        if (value)
        {
            bitmask_setbit (mask, x, y);
        }
        else
        {
            bitmask_clearbit (mask, x, y);
        }
    }
    else
    {
        PyErr_Format (PyExc_IndexError, "%d, %d is out of bounds", x, y);
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_mask_overlap (PyObject* self, PyObject* args)
{
    bitmask_t *mask = PyMask_AsBitmask (self);
    bitmask_t *othermask;
    PyObject *maskobj;
    int x, y, val;
    int xp,yp;

    if (!PyArg_ParseTuple (args, "O!ii", &PyMask_Type, &maskobj, &x, &y))
    {
        PyObject *pt;
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "O!O", &PyMask_Type, &maskobj, &pt))
            return NULL;
        if (!PointFromObj (pt, &x, &y))
            return NULL;
    }

    othermask = PyMask_AsBitmask(maskobj);

    val = bitmask_overlap_pos (mask, othermask, x, y, &xp, &yp);
    if (val)
        return Py_BuildValue ("(ii)", xp,yp);
    Py_RETURN_NONE;
}


static PyObject*
_mask_overlaparea (PyObject* self, PyObject* args)
{
    bitmask_t *mask = PyMask_AsBitmask (self);
    bitmask_t *othermask;
    PyObject *maskobj;
    int x, y, val;

    if (!PyArg_ParseTuple (args, "O!ii", &PyMask_Type, &maskobj, &x, &y))
    {
        PyObject *pt;
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "O!O", &PyMask_Type, &maskobj, &pt))
            return NULL;
        if (!PointFromObj (pt, &x, &y))
            return NULL;
    }

    othermask = PyMask_AsBitmask (maskobj);

    val = bitmask_overlap_area (mask, othermask, x, y);
    return PyInt_FromLong (val);
}

static PyObject*
_mask_overlapmask (PyObject* self, PyObject* args)
{
    int x, y;
    bitmask_t *mask = PyMask_AsBitmask (self);
    bitmask_t *output;
    bitmask_t *othermask;
    PyObject *maskobj;
    PyMask *maskobj2;

    if (!PyArg_ParseTuple (args, "O!ii", &PyMask_Type, &maskobj, &x, &y))
    {
        PyObject *pt;
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "O!O", &PyMask_Type, &maskobj, &pt))
            return NULL;
        if (!PointFromObj (pt, &x, &y))
            return NULL;
    }

    output = bitmask_create (mask->w, mask->h);
    if (!output)
    {
        PyErr_SetString (PyExc_MemoryError, "memory allocation failed");
        return NULL;
    }

    maskobj2 = (PyMask*) PyMask_Type.tp_new (&PyMask_Type, NULL, NULL);
    if (!maskobj2)
    {
        bitmask_free (output);
        return NULL;
    }

    othermask = PyMask_AsBitmask (maskobj);
    bitmask_overlap_mask (mask, othermask, output, x, y);
    maskobj2->mask = output;

    return (PyObject*) maskobj2;
}

static PyObject* 
_mask_fill (PyObject* self)
{
    bitmask_t *mask = PyMask_AsBitmask (self);
    bitmask_fill (mask);
    Py_RETURN_NONE;
}

static PyObject*
_mask_clear(PyObject* self)
{
    bitmask_t *mask = PyMask_AsBitmask (self);
    bitmask_clear (mask);
    Py_RETURN_NONE;
}

static PyObject*
_mask_invert (PyObject* self)
{
    bitmask_t *mask = PyMask_AsBitmask (self);
    bitmask_invert (mask);
    Py_RETURN_NONE;
}

static PyObject*
_mask_scale (PyObject* self, PyObject *args)
{
    int x, y;
    bitmask_t *input = PyMask_AsBitmask(self);
    bitmask_t *output;
    PyMask *maskobj;
    
    if (!PyArg_ParseTuple (args, "ii", &x, &y))
    {
        PyObject *pt;
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "O", &pt))
            return NULL;
        if (!PointFromObj (pt, &x, &y))
            return NULL;
    }

    maskobj = (PyMask*)PyMask_Type.tp_new (&PyMask_Type, NULL, NULL);
    maskobj->mask = NULL;
    if (!maskobj)
        return NULL;

    output = bitmask_scale (input, x, y);
    if (!output)
    {
        Py_DECREF (maskobj);
        PyErr_SetString (PyExc_MemoryError, "memory allocation failed");
        return NULL;
    }
    maskobj->mask = output;

    return (PyObject*) maskobj;
}

static PyObject*
_mask_draw (PyObject* self, PyObject* args)
{
    bitmask_t *mask = PyMask_AsBitmask (self);
    bitmask_t *othermask;
    PyObject *maskobj;
    int x, y;

    if (!PyArg_ParseTuple (args, "O!ii", &PyMask_Type, &maskobj, &x, &y))
    {
        PyObject *pt;
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "O!O", &PyMask_Type, &maskobj, &pt))
            return NULL;
        if (!PointFromObj (pt, &x, &y))
            return NULL;
    }

    othermask = PyMask_AsBitmask (maskobj);
    bitmask_draw(mask, othermask, x, y);
    Py_RETURN_NONE;
}

static PyObject*
_mask_erase (PyObject* self, PyObject* args)
{
    bitmask_t *mask = PyMask_AsBitmask(self);
    bitmask_t *othermask;
    PyObject *maskobj;
    int x, y;

    if (!PyArg_ParseTuple (args, "O!ii", &PyMask_Type, &maskobj, &x, &y))
    {
        PyObject *pt;
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "O!O", &PyMask_Type, &maskobj, &pt))
            return NULL;
        if (!PointFromObj (pt, &x, &y))
            return NULL;
    }
    
    othermask = PyMask_AsBitmask(maskobj);
    bitmask_erase (mask, othermask, x, y);
    Py_RETURN_NONE;
}

static PyObject*
_mask_outline (PyObject* self, PyObject* args)
{
    bitmask_t* c = PyMask_AsBitmask (self);
    bitmask_t* m;
    PyObject *plist, *value;
    int secx = 0, secy = 0, currx = 0, curry = 0;
    int x, y, every, e, firstx, firsty, nextx, nexty, n;
    int a[14], b[14];
    a[0] = a[1] = a[7] = a[8] = a[9] = b[1] = b[2] = b[3] = b[9] = b[10] =
        b[11]= 1;
    a[2] = a[6] = a[10] = b[4] = b[0] = b[12] = b[8] = 0;
    a[3] = a[4] = a[5] = a[11] = a[12] = a[13] = b[5] = b[6] = b[7] = 
        b[13] = -1;
    
    every = 1;
    n = firstx = firsty = secx = x = 0;

    if (!PyArg_ParseTuple (args, "|i", &every))
        return NULL;

    plist = PyList_New (0);
    if (!plist)
        return NULL;
    m = bitmask_create (c->w + 2, c->h + 2);     
    if (!m)
    {
        Py_DECREF (plist);
        PyErr_SetString (PyExc_MemoryError, "memory allocation failed");
        return NULL;
    }

    /* by copying to a new, larger mask, we avoid having to check if we
       are at a border pixel every time.  */
    bitmask_draw (m, c, 1, 1);

    e = every;
    
    /* find the first set pixel in the mask */
    for (y = 1; y < m->h-1; y++)
    {
        for (x = 1; x < m->w-1; x++)
        {
            if (bitmask_getbit(m, x, y))
            {
                firstx = x;
                firsty = y;
                value = Py_BuildValue ("(ii)", x - 1, y - 1);
                if (!value || PyList_Append (plist, value) == -1)
                {
                    Py_XDECREF (value);
                    Py_DECREF (plist);
                    bitmask_free (m);
                    return NULL;
                }
                Py_DECREF (value);
                break;
            }
        }
        if (bitmask_getbit (m, x, y))
            break;
    }
    
    /* covers the mask having zero pixels or only the final pixel */
    if ((x == m->w-1) && (y == m->h-1))
    {
        bitmask_free (m);
        return plist;
    }        
    
    /* check just the first pixel for neighbors */
    for (n = 0;n < 8;n++)
    {
        if (bitmask_getbit (m, x+a[n], y+b[n]))
        {
            currx = secx = x+a[n];
            curry = secy = y+b[n];
            e--;
            if (!e)
            {
                e = every;
                value = Py_BuildValue ("(ii)", secx-1, secy-1);
                if (!value || PyList_Append (plist, value) == -1)
                {
                    Py_XDECREF (value);
                    Py_DECREF (plist);
                    bitmask_free (m);
                    return NULL;
                }
                Py_DECREF (value);
            }
            break;
        }
    }
    
    /* if there are no neighbors, return */
    if (!secx)
    {
        bitmask_free (m);
        return plist;
    }
    
    /* the outline tracing loop */
    for (;;)
    {
        /* look around the pixel, it has to have a neighbor */
        for (n = (n + 6) & 7;; n++)
        {
            if (bitmask_getbit (m, currx+a[n], curry+b[n]))
            {
                nextx = currx+a[n];
                nexty = curry+b[n];
                e--;
                if (!e)
                {
                    e = every;
                    if ((curry == firsty && currx == firstx) &&
                        (secx == nextx && secy == nexty))
                    {
                        break;
                    }
                    value = Py_BuildValue ("(ii)", nextx-1, nexty-1);
                    if (!value || PyList_Append (plist, value) == -1)
                    {
                        Py_XDECREF (value);
                        Py_DECREF (plist);
                        bitmask_free (m);
                        return NULL;
                    }
                    Py_DECREF (value);
                }
                break;
            }
        }
        /* if we are back at the first pixel, and the next one will be
           the second one we visited, we are done */
        if ((curry == firsty && currx == firstx) &&
            (secx == nextx && secy == nexty))
        {
            break;
        }

        curry = nexty;
        currx = nextx;
    }
    
    bitmask_free (m);
    return plist;
}

static PyObject*
_mask_connectedcomponents (PyObject* self, PyObject* args)
{
    PyObject* ret;
    PyMask *maskobj;
    bitmask_t **components;
    bitmask_t *mask = PyMask_AsBitmask (self);
    int i, num_components, min;
    
    min = 0;
    components = NULL;
    
    if(!PyArg_ParseTuple(args, "|i", &min))
        return NULL;
    
    Py_BEGIN_ALLOW_THREADS;
    num_components = _get_connected_components (mask, &components, min);
    Py_END_ALLOW_THREADS;
    
    if (num_components == -1)
    {
        PyErr_SetString (PyExc_MemoryError, "memory allocation failed");
        return NULL;
    }

    ret = PyList_New(0);
    if (!ret)
        return NULL;    
    
    for (i = 1; i <= num_components; i++)
    {
        maskobj = (PyMask*)PyMask_Type.tp_new (&PyMask_Type, NULL, NULL);
        if (!maskobj)
        {
            int j;
            Py_DECREF (ret);
            for (j = i; j <= num_components; j++)
            {
                bitmask_free (components[j]);
            }
            free (components);
            return NULL;
        }

        maskobj->mask = components[i];
        if (PyList_Append (ret, (PyObject *) maskobj) == -1)
        {
            int j;
            Py_DECREF((PyObject *) maskobj);
            Py_DECREF (ret);
            for (j = i; j <= num_components; j++)
            {
                bitmask_free (components[j]);
            }
            free (components);
            return NULL;
        }
        Py_DECREF (maskobj);
    }
    
    free (components);
    return ret;
}

static PyObject*
_mask_connectedcomponent (PyObject* self, PyObject* args)
{
    bitmask_t *input = PyMask_AsBitmask (self);
    bitmask_t *output;
    PyMask *maskobj;
    int x, y;
    
    y = x = -1;

    if (!PyArg_ParseTuple (args, "|ii", &x, &y))
    {
        PyObject *pt;
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "|O", &pt))
            return NULL;
        if (!PointFromObj (pt, &x, &y))
            return NULL;
    }
    
    output = bitmask_create (input->w, input->h);
    if (!output)
    {
        PyErr_SetString (PyExc_MemoryError, "memory allocation failed");
        return NULL;
    }

    maskobj = (PyMask*)PyMask_Type.tp_new (&PyMask_Type, NULL, NULL);
    maskobj->mask = NULL;
    if (!maskobj)
    {
        bitmask_free (output);
        return NULL;
    }
    /* if a coordinate is specified, make the pixel there is actually set */
    if (x == -1 || bitmask_getbit(input, x, y))
    {
        if (_largest_connected_comp (input, output, x, y) == -2)
        {
            PyErr_SetString (PyExc_MemoryError, "memory allocation failed");
            bitmask_free (output);
            Py_DECREF (maskobj);
            return NULL;
        }
    }
    
    maskobj->mask = output;
    return (PyObject*)maskobj;
}

static PyObject*
_mask_getboundingrects (PyObject* self)
{
    CRect *regions;
    CRect *aregion;
    int num_bounding_boxes, i, r;
    PyObject* ret;
    PyObject* rect;
    bitmask_t *mask = PyMask_AsBitmask(self);

    ret = NULL;
    regions = NULL;
    aregion = NULL;

    num_bounding_boxes = 0;

    Py_BEGIN_ALLOW_THREADS;
    r = _get_bounding_rects (mask, &num_bounding_boxes, &regions);
    Py_END_ALLOW_THREADS;

    if (r == 0)
    {
        /* memory out failure */
        PyErr_SetString (PyExc_MemoryError, "memory allocation failed");
        return NULL;
    }

    ret = PyList_New (0);
    if (!ret)
    {
        free (regions);
        return NULL;
    }

    /* build a list of rects to return.  Starts at 1 because we never use 0. */
    for(i = 1; i <= num_bounding_boxes; i++)
    {
        aregion = regions + i;
        rect = PyRect_New (aregion->x, aregion->y, aregion->w, aregion->h);
        if (!rect || PyList_Append (ret, rect) == -1)
        {
            Py_XDECREF (rect);
            Py_DECREF (ret);
            free (regions);
            return NULL;
        }
        Py_DECREF (rect);
    }

    free (regions);

    return ret;
}

static PyObject*
_mask_convolve (PyObject* self, PyObject* args)
{
    PyObject *pt = NULL, *cmask = NULL, *outmask = NULL;
    bitmask_t *a, *b, *o;
    int x = 0, y = 0;

    if (!PyArg_ParseTuple (args, "O|OO", &cmask, &outmask, &pt))
    {
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "O|Oii", &cmask, &outmask, &x, &y))
            return NULL;
    }
    else if (pt)
    {
        if (!PointFromObj (pt, &x, &y))
            return NULL;
    }
    
    if (!PyMask_Check (cmask))
    {
        PyErr_SetString (PyExc_TypeError, "mask must be a Mask");
        return NULL;
    }

    if (outmask && outmask != Py_None && !PyMask_Check (outmask))
    {
        PyErr_SetString (PyExc_TypeError, "outputmask must be a Mask");
        return NULL;
    }

    a = PyMask_AsBitmask (self);
    b = PyMask_AsBitmask (cmask);

    /* outmask->w < a->w + b->w - 1 && outmask->h < a->h + b->h - 1 is
     * automatically handled by the convolve/bitmask_draw functions */
    if (!outmask || outmask == Py_None)
    {
        outmask = PyMask_New (a->w + b->w - 1, a->h + b->h  - 1);
        if (!outmask)
            return NULL;
    }
    else
        Py_INCREF (outmask);

    o = PyMask_AsBitmask (outmask);

    bitmask_convolve(a, b, o, x, y);

    return outmask;
}

/* Connected component labeling based on the SAUF algorithm by Kesheng Wu,
   Ekow Otoo, and Kenji Suzuki.  The algorithm is best explained by their paper,
   "Two Strategies to Speed up Connected Component Labeling Algorithms", but in
   summary, it is a very efficient two pass method for 8-connected components.
   It uses a decision tree to minimize the number of neighbors that need to be
   checked.  It stores equivalence information in an array based union-find.
   This implementation also has a final step of finding bounding boxes. */

/* 
   returns 0 on memory allocation error, otherwise 1 on success.

   input - the input mask.
   num_bounding_boxes - returns the number of bounding rects found.
   rects - returns the rects that are found.  Allocates the memory for
   the rects.
*/
static int
_get_bounding_rects (bitmask_t *input, int *num_bounding_boxes,
    CRect** ret_rects)
{
    unsigned int *image, *ufind, *largest, *buf;
    int x, y, w, h, temp, label, relabel;
    CRect* rects;

    rects = NULL;
    label = 0;

    w = input->w;
    h = input->h;

    /* a temporary image to assign labels to each bit of the mask */
    image = (unsigned int *) malloc (sizeof (int) * w * h);
    if (!image)
        return 0;

    /* allocate enough space for the maximum possible connected components */
    /* the union-find array. see wikipedia for info on union find */
    ufind = (unsigned int *) malloc (sizeof (int) * (w / 2 + 1)* ( h / 2 + 1));
    if (!ufind)
    {
        free (image);
        return 0;
    }

    largest = (unsigned int *) malloc (sizeof (int) * (w/2 + 1) * (h/2 + 1));
    if (!largest)
    {
        free (image);
        free (ufind);
        return 0;
    }

    /* do the initial labelling */
    label = _cc_label (input, image, ufind, largest);

    relabel = 0;
    /* flatten and relabel the union-find equivalence array.  Start at label 1
       because label 0 indicates an unset pixel.  For this reason, we also use
       <= label rather than < label. */
    for (x = 1; x <= label; x++)
    {
        if (ufind[x] < (unsigned int) x)
        {
            /* is it a union find root? */
            ufind[x] = ufind[ufind[x]]; /* relabel it to its root */
        } else {                 /* its a root */
            relabel++;                      
            ufind[x] = relabel;  /* assign the lowest label available */
        }
    }

    *num_bounding_boxes = relabel;

    if (relabel == 0)
    {
        /* early out, as we didn't find anything. */
        free(image);
        free(ufind);
        free(largest);
        *ret_rects = NULL;
        return 1;
    }

    /* the bounding rects, need enough space for the number of labels */
    rects = (CRect *) malloc (sizeof (CRect) * (relabel + 1));
    if (!rects)
    {
        free(image);
        free(ufind);
        free(largest);
        return 0;
    }

    for (temp = 0; temp <= relabel; temp++)
    {
        rects[temp].h = 0;        /* so we know if its a new rect or not */
    }

    /* find the bounding rect of each connected component */
    buf = image;
    for (y = 0; y < h; y++)
    {
        for (x = 0; x < w; x++)
        {
            if (ufind[*buf])
            {
                /* if the pixel is part of a component */
                if (rects[ufind[*buf]].h)
                {
                    /* the component has a rect */
                    temp = rects[ufind[*buf]].x;
                    rects[ufind[*buf]].x = MIN(x, temp);
                    rects[ufind[*buf]].y = MIN(y, rects[ufind[*buf]].y);
                    rects[ufind[*buf]].w = MAX(rects[ufind[*buf]].w + temp,
                        (unsigned int) x + 1) - rects[ufind[*buf]].x;
                    rects[ufind[*buf]].h = MAX(rects[ufind[*buf]].h,
                        (unsigned int) y - rects[ufind[*buf]].y + 1);
                }
                else
                {
                    /* otherwise, start the rect */
                    rects[ufind[*buf]].x = x;
                    rects[ufind[*buf]].y = y;
                    rects[ufind[*buf]].w = 1;
                    rects[ufind[*buf]].h = 1;
                }
            }
            buf++;
        }
    }
	
    free(image);
    free(ufind);
    free(largest);
    *ret_rects = rects;

    return 1;
}

static int
_get_connected_components (bitmask_t *mask, bitmask_t ***components, int min)
{
    unsigned int *image, *ufind, *largest, *buf;
    int x, y, w, h, label, relabel;
    bitmask_t** comps;

    label = 0;

    w = mask->w;
    h = mask->h;

    /* a temporary image to assign labels to each bit of the mask */
    image = (unsigned int *) malloc (sizeof (int) * w * h);
    if (!image)
        return -1;

    /* allocate enough space for the maximum possible connected components */
    /* the union-find array. see wikipedia for info on union find */
    ufind = (unsigned int *) malloc (sizeof (int) * (w/2 + 1) * (h/2 + 1));
    if (!ufind)
    {
        free (image);
        return -1;
    }

    largest = (unsigned int *) malloc (sizeof (int) * (w/2 + 1) * (h/2 + 1));
    if (!largest)
    {
        free (image);
        free (ufind);
        return -1;
    }
	
    /* do the initial labelling */
    label = _cc_label (mask, image, ufind, largest);
    
    for (x = 1; x <= label; x++)
    {
        if (ufind[x] < (unsigned int) x)
        {
            largest[ufind[x]] += largest[x];
        }
    }

    relabel = 0;
    /* flatten and relabel the union-find equivalence array.  Start at label 1
       because label 0 indicates an unset pixel.  For this reason, we also use
       <= label rather than < label. */
    for (x = 1; x <= label; x++)
    {
        if (ufind[x] < (unsigned int) x)
        {
            /* is it a union find root? */
            ufind[x] = ufind[ufind[x]]; /* relabel it to its root */
        } else {                 /* its a root */
            if (largest[x] >= (unsigned int) min) {
                relabel++;                      
                ufind[x] = relabel;  /* assign the lowest label available */
            } else {
                ufind[x] = 0;
            }
        }
    }

    if (relabel == 0)
    {
        /* early out, as we didn't find anything. */
        free (image);
        free (ufind);
        free (largest);
        return 0;
    }

    /* allocate space for the mask array */
    comps = (bitmask_t **) malloc (sizeof (bitmask_t *) * (relabel +1));
    if (!comps)
    {
        free (image);
        free (ufind);
        free (largest);
        return -1;
    }
    
    /* create the empty masks */
    for (x = 1; x <= relabel; x++)
    {
        comps[x] = bitmask_create (w, h);
        if (!comps[x])
        {
            int wx;
            for (wx = 1; wx < x; wx++)
            {
                bitmask_free (comps[wx]);
            }
            free (image);
            free (ufind);
            free (largest);
            free (comps);
            return -1;
        }
    }

    /* set the bits in each mask */
    buf = image;
    for (y = 0; y < h; y++)
    {
        for (x = 0; x < w; x++)
        {
            if (ufind[*buf])
            {
                /* if the pixel is part of a component */
                bitmask_setbit(comps[ufind[*buf]], x, y);
            }
            buf++;
        }
    }
    
    free (image);
    free (ufind);
    free (largest);

    *components = comps;

    return relabel;
}    

/* the initial labelling phase of the connected components algorithm 

   Returns: The highest label in the labelled image

   input - The input Mask
   image - An array to store labelled pixels
   ufind - The union-find label equivalence array
   largest - An array to store the number of pixels for each label

*/
static unsigned int
_cc_label (bitmask_t *input, unsigned int* image, unsigned int* ufind,
    unsigned int* largest)
{
    unsigned int *buf;
    unsigned int x, y, w, h, root, aroot, croot, temp, label;
    
    label = 0;
    w = input->w;
    h = input->h;

    ufind[0] = 0;
    buf = image;

    /* special case for first pixel */
    if (bitmask_getbit (input, 0, 0)) { /* process for a new connected comp: */
        label++;              /* create a new label */
        *buf = label;         /* give the pixel the label */
        ufind[label] = label; /* put the label in the equivalence array */
        largest[label] = 1;   /* the label has 1 pixel associated with it */
    } else {
        *buf = 0;
    }
    buf++;



    /* special case for first row.  
       Go over the first row except the first pixel. 
    */
    for (x = 1; x < w; x++) {
        if (bitmask_getbit (input, (int) x, 0)) {
            if (*(buf-1)) {                    /* d label */
                *buf = *(buf-1);
            } else {                           /* create label */
                label++;
                *buf = label;
                ufind[label] = label;
                largest[label] = 0;
            }
            largest[*buf]++;
        } else {
            *buf = 0;
        }
        buf++;
    }



    /* the rest of the image */
    for(y = 1; y < h; y++) {
        /* first pixel of the row */
        if (bitmask_getbit(input, 0, (int) y)) {
            if (*(buf-w)) {                    /* b label */
                *buf = *(buf-w);
            } else if (*(buf-w+1)) {           /* c label */
                *buf = *(buf-w+1);
            } else {                           /* create label */
                label++;
                *buf = label;
                ufind[label] = label;
                largest[label] = 0;
            }
            largest[*buf]++;
        } else {
            *buf = 0;
        }
        buf++;
        /* middle pixels of the row */
        for(x = 1; x < (w-1); x++) {
            if (bitmask_getbit(input, (int) x, (int) y)) {
                if (*(buf-w)) {                /* b label */
                    *buf = *(buf-w);
                } else if (*(buf-w+1)) {       /* c branch of tree */
                    if (*(buf-w-1)) {                      /* union(c, a) */
                        croot = root = *(buf-w+1);
                        while (ufind[root] < root) {       /* find root */
                            root = ufind[root];
                        }
                        if (croot != *(buf-w-1)) {
                            temp = aroot = *(buf-w-1);
                            while (ufind[aroot] < aroot) { /* find root */
                                aroot = ufind[aroot];
                            }
                            if (root > aroot) {
                                root = aroot;
                            }
                            while (ufind[temp] > root) {   /* set root */
                                aroot = ufind[temp];
                                ufind[temp] = root;
                                temp = aroot;
                            }
                        }
                        while (ufind[croot] > root) {      /* set root */
                            temp = ufind[croot];
                            ufind[croot] = root;
                            croot = temp;
                        }
                        *buf = root;
                    } else if (*(buf-1)) {                 /* union(c, d) */
                        croot = root = *(buf-w+1);
                        while (ufind[root] < root) {       /* find root */
                            root = ufind[root];
                        }
                        if (croot != *(buf-1)) {
                            temp = aroot = *(buf-1);
                            while (ufind[aroot] < aroot) { /* find root */
                                aroot = ufind[aroot];
                            }
                            if (root > aroot) {
                                root = aroot;
                            }
                            while (ufind[temp] > root) {   /* set root */
                                aroot = ufind[temp];
                                ufind[temp] = root;
                                temp = aroot;
                            }
                        }
                        while (ufind[croot] > root) {      /* set root */
                            temp = ufind[croot];
                            ufind[croot] = root;
                            croot = temp;
                        }
                        *buf = root;
                    } else {                   /* c label */
                        *buf = *(buf-w+1);
                    }
                } else if (*(buf-w-1)) {       /* a label */
                    *buf = *(buf-w-1);
                } else if (*(buf-1)) {         /* d label */
                    *buf = *(buf-1);
                } else {                       /* create label */
                    label++;
                    *buf = label;
                    ufind[label] = label;
                    largest[label] = 0;
                }
                largest[*buf]++;
            } else {
                *buf = 0;
            }
            buf++;
        }
        /* last pixel of the row, if its not also the first pixel of the row */
        if (w > 1) {
            if (bitmask_getbit(input, (int) x, (int) y)) {
                if (*(buf-w)) {                /* b label */
                    *buf = *(buf-w);
                } else if (*(buf-w-1)) {       /* a label */
                    *buf = *(buf-w-1);
                } else if (*(buf-1)) {         /* d label */
                    *buf = *(buf-1);
                } else {                       /* create label */
                    label++;
                    *buf = label;
                    ufind[label] = label;
                    largest[label] = 0;
                }
                largest[*buf]++;
            } else {
                *buf = 0;
            }
            buf++;
        }
    }
    
    return label;
}

/* Connected component labeling based on the SAUF algorithm by Kesheng Wu,
   Ekow Otoo, and Kenji Suzuki.  The algorithm is best explained by their paper,
   "Two Strategies to Speed up Connected Component Labeling Algorithms", but in
   summary, it is a very efficient two pass method for 8-connected components.
   It uses a decision tree to minimize the number of neighbors that need to be
   checked.  It stores equivalence information in an array based union-find.
   This implementation also tracks the number of pixels in each label, finding 
   the biggest one while flattening the union-find equivalence array.  It then 
   writes an output mask containing only the largest connected component. */
static int
_largest_connected_comp (bitmask_t* input, bitmask_t* output, int ccx, int ccy)
{
    unsigned int *image, *ufind, *largest, *buf;
    unsigned int max, x, y, w, h, label;
    
    w = input->w;
    h = input->h;

    /* a temporary image to assign labels to each bit of the mask */
    image = (unsigned int *) malloc(sizeof (int) * w * h);
    if(!image)
        return -2;

    /* allocate enough space for the maximum possible connected components */
    /* the union-find array. see wikipedia for info on union find */
    ufind = (unsigned int *) malloc (sizeof (int) * (w/2 + 1) * (h/2 + 1));
    if(!ufind)
    {
        free (image);
        return -2;
    }
    
    /* an array to track the number of pixels associated with each label */
    largest = (unsigned int *) malloc (sizeof (int) * (w/2 + 1) * (h/2 + 1));
    if (!largest)
    {
        free (image);
        free (ufind);
        return -2;
    }
    
    /* do the initial labelling */
    label = _cc_label (input, image, ufind, largest);

    max = 1;
    /* flatten the union-find equivalence array */
    for (x = 2; x <= label; x++) {
        if (ufind[x] != x) {                 /* is it a union find root? */
            largest[ufind[x]] += largest[x]; /* add its pixels to its root */
            ufind[x] = ufind[ufind[x]];      /* relabel it to its root */
        }
        if (largest[ufind[x]] > largest[max]) { /* is it the new biggest? */
            max = ufind[x];
        }
    }
    
    /* write out the final image */
    buf = image;
    if (ccx >= 0)
        max = ufind[*(buf+ccy*w+ccx)];
    for (y = 0; y < h; y++) {
        for (x = 0; x < w; x++) {
            if (ufind[*buf] == max) {         /* if the label is the max one */
                /* set the bit in the mask */
                bitmask_setbit(output, (int)x, (int)y);
            }
            buf++;
        }
    }
    
    free (image);
    free (ufind);
    free (largest);
    
    return 0;
}

/* C API */
PyObject*
PyMask_New (int w, int h)
{
    bitmask_t *m;
    PyMask *mask = (PyMask*) PyMask_Type.tp_new (&PyMask_Type, NULL, NULL);
    mask->mask = NULL;
    if (!mask)
        return NULL;

    m = bitmask_create (w, h);
    if (!m)
    {
        PyErr_SetString (PyExc_MemoryError, "memory allocation failed");
        Py_DECREF ((PyObject*) mask);
        return NULL;
    }
    mask->mask = m;
    return (PyObject*) mask;
}

void
mask_export_capi (void **capi)
{
    capi[PYGAME_MASK_FIRSTSLOT+0] = &PyMask_Type;
    capi[PYGAME_MASK_FIRSTSLOT+1] = (void *)PyMask_New;
}
