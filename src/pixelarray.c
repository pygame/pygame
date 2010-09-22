/*
  pygame - Python Game Library
  Copyright (C) 2007-2008 Marcus von Appen

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

#define PYGAMEAPI_PIXELARRAY_INTERNAL

#include "pygame.h"
#include "pgcompat.h"
#include "doc/pixelarray_doc.h"
#include "surface.h"

#if PY_VERSION_HEX < 0x02050000
#define PyIndex_Check(op) 0
#endif

typedef struct
{
    PyObject_HEAD
    PyObject *dict;     /* dict for subclassing */
    PyObject *weakrefs; /* Weakrefs for subclassing */
    PyObject *surface;  /* Surface associated with the array. */
    PyObject *lock;     /* Lock object for the surface. */
    Uint32 xstart;      /* X offset for subarrays */
    Uint32 ystart;      /* Y offset for subarrays */
    Uint32 xlen;        /* X segment length. */
    Uint32 ylen;        /* Y segment length. */
    Sint32 xstep;       /* X offset step width. */
    Sint32 ystep;       /* Y offset step width. */
    Uint32 padding;     /* Padding to get to the next x offset. */
    PyObject *parent;   /* Parent pixel array */

} PyPixelArray;

static PyPixelArray* _pxarray_new_internal (PyTypeObject *type,
    PyObject *surface, Uint32 xstart, Uint32 ystart, Uint32 xlen, Uint32 ylen,
    Sint32 xstep, Sint32 ystep, Uint32 padding, PyObject *parent);
static PyObject* _pxarray_new (PyTypeObject *type, PyObject *args,
    PyObject *kwds);
static void _pxarray_dealloc (PyPixelArray *self);

static PyObject* _pxarray_get_dict (PyPixelArray *self, void *closure);
static PyObject* _pxarray_get_surface (PyPixelArray *self, void *closure);
static PyObject* _pxarray_repr (PyPixelArray *array);

static PyObject* _array_slice_internal (PyPixelArray *array, Sint32 _start,
    Sint32 _end, Sint32 _step);

/* Sequence methods */
static Py_ssize_t _pxarray_length (PyPixelArray *array);
static PyObject* _pxarray_item (PyPixelArray *array, Py_ssize_t _index);
#if !PY3
static PyObject* _pxarray_slice (PyPixelArray *array, Py_ssize_t low,
    Py_ssize_t high);
#endif
static int _array_assign_array (PyPixelArray *array, Py_ssize_t low,
    Py_ssize_t high, PyPixelArray *val);
static int _array_assign_sequence (PyPixelArray *array, Py_ssize_t low,
    Py_ssize_t high, PyObject *val);
static int _array_assign_slice (PyPixelArray *array, Py_ssize_t low,
    Py_ssize_t high, Uint32 color);
static int _pxarray_ass_item (PyPixelArray *array, Py_ssize_t _index,
    PyObject *value);
static int _pxarray_ass_slice (PyPixelArray *array, Py_ssize_t low,
    Py_ssize_t high, PyObject *value);
static int _pxarray_contains (PyPixelArray *array, PyObject *value);
static PyObject* _pxarray_iter (PyPixelArray *array);

/* Mapping methods */
static int _get_subslice (PyObject *op, Py_ssize_t length, Py_ssize_t *start,
    Py_ssize_t *stop, Py_ssize_t *step);
static PyObject* _pxarray_subscript (PyPixelArray *array, PyObject *op);
static int _pxarray_ass_subscript (PyPixelArray *array, PyObject* op,
    PyObject* value);

/* C API interfaces */
static PyObject* PyPixelArray_New (PyObject *surfobj);

/* Incomplete forward declaration so we can use it in the methods included
 * below.
 */
static PyTypeObject PyPixelArray_Type;
#define PyPixelArray_Check(o) \
    (Py_TYPE (o) == (PyTypeObject *) &PyPixelArray_Type)

#define SURFACE_EQUALS(x,y) \
    (((PyPixelArray *)x)->surface == ((PyPixelArray *)y)->surface)

#include "pixelarray_methods.c"
    
/**
 * Methods, which are bound to the PyPixelArray type.
 */
static PyMethodDef _pxarray_methods[] =
{
    { "compare", (PyCFunction) _compare, METH_VARARGS | METH_KEYWORDS,
      DOC_PIXELARRAYCOMPARE },
    { "extract", (PyCFunction) _extract_color, METH_VARARGS | METH_KEYWORDS,
      DOC_PIXELARRAYEXTRACT },
    { "make_surface", (PyCFunction) _make_surface, METH_VARARGS | METH_NOARGS,
      DOC_PIXELARRAYMAKESURFACE },
    { "replace", (PyCFunction) _replace_color, METH_VARARGS | METH_KEYWORDS,
      DOC_PIXELARRAYREPLACE },
    { NULL, NULL, 0, NULL }
};

#if PY3
static void
Text_ConcatAndDel (PyObject **string, PyObject *newpart)
{
    PyObject *result = NULL;
    if (*string != NULL && newpart != NULL) {
        PyUnicode_Concat (*string, newpart);
        Py_DECREF (*string);
        Py_DECREF (newpart);
    }
    else {
        Py_XDECREF (*string);
        Py_XDECREF (newpart);
    }
    *string = result;
}
#else
#define Text_ConcatAndDel PyString_ConcatAndDel
#endif

/**
 * Getters and setters for the PyPixelArray.
 */
static PyGetSetDef _pxarray_getsets[] =
{
    { "__dict__", (getter) _pxarray_get_dict, NULL, NULL, NULL },
    { "surface", (getter) _pxarray_get_surface, NULL, DOC_PIXELARRAYSURFACE,
      NULL }, 
    { NULL, NULL, NULL, NULL, NULL }
};

/**
 * Sequence interface support for the PyPixelArray.
 * concat and repeat are not implemented due to the possible confusion
 * of their behaviour (see lists numpy array).
 */
static PySequenceMethods _pxarray_sequence =
{
    (lenfunc) _pxarray_length,                  /*sq_length*/
    NULL, /*sq_concat*/
    NULL, /*sq_repeat*/
    (ssizeargfunc) _pxarray_item,               /*sq_item*/
#if PY3
    NULL,
#else
    (ssizessizeargfunc) _pxarray_slice,         /*sq_slice*/
#endif
    (ssizeobjargproc) _pxarray_ass_item,        /*sq_ass_item*/
#if PY3
    NULL,
#else
    (ssizessizeobjargproc) _pxarray_ass_slice,  /*sq_ass_slice*/
#endif
    (objobjproc) _pxarray_contains,             /*sq_contains*/
    NULL, /*sq_inplace_concat*/
    NULL, /*sq_inplace_repeat*/
};

/**
 * Mapping interface support for the PyPixelArray.
 */
static PyMappingMethods _pxarray_mapping =
{
    (inquiry) _pxarray_length,              /*mp_length*/
    (binaryfunc) _pxarray_subscript,        /*mp_subscript*/
    (objobjargproc) _pxarray_ass_subscript, /*mp_ass_subscript*/
};

static PyTypeObject PyPixelArray_Type =
{
    TYPE_HEAD (NULL, 0)
    "pygame.PixelArray",        /* tp_name */
    sizeof (PyPixelArray),      /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor) _pxarray_dealloc, /* tp_dealloc */
    0,                          /* tp_print */
    0,                          /* tp_getattr */
    0,                          /* tp_setattr */
    0,                          /* tp_compare */
    (reprfunc) _pxarray_repr,   /* tp_repr */
    0,                          /* tp_as_number */
    &_pxarray_sequence,         /* tp_as_sequence */
    &_pxarray_mapping,          /* tp_as_mapping */
    0,                          /* tp_hash */
    0,                          /* tp_call */
    0,                          /* tp_str */
    0,                          /* tp_getattro */
    0,                          /* tp_setattro */
    0,                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_WEAKREFS,
    DOC_PYGAMEPIXELARRAY,       /* tp_doc */
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    offsetof (PyPixelArray, weakrefs),  /* tp_weaklistoffset */
    (getiterfunc) _pxarray_iter, /* tp_iter */
    0,                          /* tp_iternext */
    _pxarray_methods,           /* tp_methods */
    0,                          /* tp_members */
    _pxarray_getsets,           /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    offsetof (PyPixelArray, dict), /* tp_dictoffset */
    0,                          /* tp_init */
    0,                          /* tp_alloc */
    _pxarray_new,               /* tp_new */
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

static PyPixelArray*
_pxarray_new_internal (PyTypeObject *type, PyObject *surface,
    Uint32 xstart, Uint32 ystart, Uint32 xlen, Uint32 ylen,
    Sint32 xstep, Sint32 ystep, Uint32 padding, PyObject *parent)
{
    PyPixelArray *self = (PyPixelArray *) type->tp_alloc (type, 0);
    if (!self)
        return NULL;

    self->surface = (PyObject *) surface;
    self->lock = NULL;
    self->parent = NULL;
    Py_INCREF (surface);

    if (!parent)
    {
        /* Initial PixelArray */
        if (surface)
        {
            self->lock = PySurface_LockLifetime (surface, (PyObject *) self);
            if (!self->lock)
            {
                Py_DECREF (surface);
                Py_TYPE (self)->tp_free ((PyObject *) self);
                return NULL;
            }
        }
    }
    else
    {
        self->parent = parent;
        Py_INCREF (parent);
        self->lock = ((PyPixelArray *)parent)->lock;
        Py_INCREF (self->lock);
    }

    self->weakrefs = NULL;
    self->dict = NULL;
    self->xstart = xstart;
    self->ystart = ystart;
    self->xlen = xlen;
    self->ylen = ylen;
    self->xstep = xstep;
    self->ystep = ystep;
    self->padding = padding;
    return self;
}

/**
 * Creates a new PyPixelArray.
 */
static PyObject*
_pxarray_new (PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyObject *surfobj;
    SDL_Surface* surface;

    if (!PyArg_ParseTuple (args, "O!", &PySurface_Type, &surfobj))
        return NULL;

    surface = PySurface_AsSurface (surfobj);
    if (surface->format->BytesPerPixel < 1  ||
        surface->format->BytesPerPixel > 4)
        return RAISE (PyExc_ValueError,
            "unsupport bit depth for reference array");

    return (PyObject *) _pxarray_new_internal
        (type, surfobj, 0, 0, (Uint32) surface->w, (Uint32) surface->h, 1, 1,
            surface->pitch, NULL);
}

/**
 * Deallocates the PyPixelArray and its members.
 */
static void
_pxarray_dealloc (PyPixelArray *self)
{
    if (self->weakrefs)
        PyObject_ClearWeakRefs ((PyObject *) self);
    Py_XDECREF (self->lock);
    Py_XDECREF (self->parent);
    Py_XDECREF (self->dict);
    Py_DECREF (self->surface);
    Py_TYPE (self)->tp_free ((PyObject *) self);
}

/**** Getter and setter access ****/

/**
 * Getter for PixelArray.__dict__.
 */
static PyObject*
_pxarray_get_dict (PyPixelArray *self, void *closure)
{
    if (!self->dict)
    {
	self->dict = PyDict_New ();
        if (!self->dict)
            return NULL;
    }

    Py_INCREF (self->dict);
    return self->dict;
}

/**
 * Getter for PixelArray.surface
 */
static PyObject*
_pxarray_get_surface (PyPixelArray *self, void *closure)
{
    Py_XINCREF (self->surface);
    return self->surface;
}

/**** Methods ****/

/**
 * repr(PixelArray)
 */
static PyObject*
_pxarray_repr (PyPixelArray *array)
{
    PyObject *string;
    SDL_Surface *surface;
    int bpp;
    Uint8 *pixels;
    Uint8 *px24;
    Uint32 pixel;
    Uint32 x = 0;
    Uint32 y = 0;
    Sint32 xlen = 0;
    Sint32 absxstep;
    Sint32 absystep;
    Uint32 posx = 0;
    Uint32 posy = 0;

    surface = PySurface_AsSurface (array->surface);
    bpp = surface->format->BytesPerPixel;
    pixels = (Uint8 *) surface->pixels;

/*
    printf ("::ARRAY: %d:%d:%d,  %d:%d:%d %d\n",
        array->xstart, array->xlen, array->xstep, array->ystart, 
        array->ylen, array->ystep, array->padding);
*/
    string = Text_FromUTF8 ("PixelArray(");

    absxstep = ABS (array->xstep);
    absystep = ABS (array->ystep);
    xlen = (Sint32) array->xlen - absxstep;

    y = array->ystart;
    
    switch (bpp)
    {
    case 1:
        while (posy < array->ylen)
        {
            /* Construct the rows */
            Text_ConcatAndDel (&string, Text_FromUTF8 ("\n  ["));
            if (string == NULL)
            {
                return NULL;
            }
            posx = 0;
            x = array->xstart;
            while (posx < (Uint32)xlen)
            {
                /* Construct the columns */
                pixel = (Uint32) *((Uint8 *) pixels + x + y * array->padding);
                Text_ConcatAndDel (&string,
                                   Text_FromFormat ("%ld, ", (long)pixel));
                if (string == NULL)
                {
                    return NULL;
                }
                x += array->xstep;
                posx += absxstep;
            }
            pixel = (Uint32) *((Uint8 *) pixels + x + y * array->padding);
            Text_ConcatAndDel (&string,
                               Text_FromFormat ("%ld]", (long)pixel));
            if (string == NULL)
            {
                return NULL;
            }
            y += array->ystep;
            posy += absystep;
        }
        break;
    case 2:
        while (posy < array->ylen)
        {
            /* Construct the rows */
            Text_ConcatAndDel (&string, Text_FromUTF8 ("\n  ["));
            if (string == NULL)
            {
                return NULL;
            }
            posx = 0;
            x = array->xstart;
            while (posx < (Uint32)xlen)
            {
                /* Construct the columns */
                pixel = (Uint32)
                    *((Uint16 *) (pixels + y * array->padding) + x);
                Text_ConcatAndDel (&string,
                                   Text_FromFormat ("%ld, ", (long)pixel));
                if (string == NULL)
                {
                    return NULL;
                }
                x += array->xstep;
                posx += absxstep;
            }
            pixel = (Uint32) *((Uint16 *) (pixels + y * array->padding) + x);
            Text_ConcatAndDel (&string,
                               Text_FromFormat ("%ld]", (long)pixel));
            if (string == NULL)
            {
                return NULL;
            }
            y += array->ystep;
            posy += absystep;
        }
        break;
    case 3:
        while (posy < array->ylen)
        {
            /* Construct the rows */
            Text_ConcatAndDel (&string, Text_FromUTF8 ("\n  ["));
            if (string == NULL)
            {
                return NULL;
            }
            posx = 0;
            x = array->xstart;
            while (posx < (Uint32)xlen)
            {
                /* Construct the columns */
                px24 = ((Uint8 *) (pixels + y * array->padding) + x * 3);
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                pixel = (px24[0]) + (px24[1] << 8) + (px24[2] << 16);
#else
                pixel = (px24[2]) + (px24[1] << 8) + (px24[0] << 16);
#endif
                Text_ConcatAndDel (&string,
                                   Text_FromFormat ("%ld, ", (long)pixel));
                if (string == NULL)
                {
                    return NULL;
                }
                x += array->xstep;
                posx += absxstep;
            }
            px24 = ((Uint8 *) (pixels + y * array->padding) + x * 3);
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
            pixel = (px24[0]) + (px24[1] << 8) + (px24[2] << 16);
#else
            pixel = (px24[2]) + (px24[1] << 8) + (px24[0] << 16);
#endif
            Text_ConcatAndDel (&string,
                               Text_FromFormat ("%ld]", (long)pixel));
            if (string == NULL)
            {
                return NULL;
            }
            y += array->ystep;
            posy += absystep;
        }
        break;
    default: /* 4bpp */
        while (posy < array->ylen)
        {
            /* Construct the rows */
            Text_ConcatAndDel (&string, Text_FromUTF8 ("\n  ["));
            if (string == NULL)
            {
                return NULL;
            }
            posx = 0;
            x = array->xstart;
            while (posx < (Uint32)xlen)
            {
                /* Construct the columns */
                pixel = *((Uint32 *) (pixels + y * array->padding) + x);
                Text_ConcatAndDel (&string,
                                   Text_FromFormat ("%ld, ", (long)pixel));
                if (string == NULL)
                {
                    return NULL;
                }
                x += array->xstep;
                posx += absxstep;
            }
            pixel = *((Uint32 *) (pixels + y * array->padding) + x);
            Text_ConcatAndDel (&string,
                               Text_FromFormat ("%ld]", (long)pixel));
            if (string == NULL)
            {
                return NULL;
            }
            y += array->ystep;
            posy += absystep;
        }
        break;
    }
    Text_ConcatAndDel (&string, Text_FromUTF8 ("\n)"));
    return string;
}


/**
 * Creates a 2D slice of the array.
 */
static PyObject*
_array_slice_internal (PyPixelArray *array, Sint32 _start, Sint32 _end,
    Sint32 _step)
{
    Uint32 xstart = 0;
    Uint32 ystart = 0;
    Uint32 xlen;
    Uint32 ylen;
    Sint32 xstep;
    Sint32 ystep;
    Uint32 padding;

    if (_end == _start)
        return RAISE (PyExc_IndexError, "array size must not be 0");

    if (array->xlen == 1)
    {
        ystart = array->ystart + _start * array->ystep;
        xstart = array->xstart;
        xlen = array->xlen;
        ylen = ABS (_end - _start);
        ystep = _step;
        xstep = array->xstep;
        padding = array->padding;

        /* Out of bounds? */
        if (_start >= (Sint32) array->ylen && ystep > 0)
            return RAISE (PyExc_IndexError, "array index out of range");
    }
    else
    {
        xstart = array->xstart + _start * array->xstep;
        ystart = array->ystart;
        xlen = ABS (_end - _start);
        ylen = array->ylen;
        xstep = _step;
        ystep = array->ystep;
        padding = array->padding;

        /* Out of bounds? */
        if (_start >= (Sint32) array->xlen && xstep > 0)
            return RAISE (PyExc_IndexError, "array index out of range");
    }

/*
    printf ("VALS: %d:%d, %d:%d %d:%d %d\n",
        xstart, ystart, xlen, xstep, ylen, ystep, padding);
*/
    return (PyObject *) _pxarray_new_internal
        (&PyPixelArray_Type, array->surface, xstart, ystart, xlen, ylen,
         xstep, ystep, padding, (PyObject *) array);
}


/**** Sequence interfaces ****/

/**
 * len (array)
 */
static Py_ssize_t
_pxarray_length (PyPixelArray *array)
{
    if (array->xlen > 1)
        return array->xlen / ABS (array->xstep);
    return array->ylen / ABS (array->ystep);
}

/**
 * array[x]
 */
static PyObject*
_pxarray_item (PyPixelArray *array, Py_ssize_t _index)
{
    SDL_Surface *surface;
    int bpp;

    if (_index < 0)
        return RAISE (PyExc_IndexError, "array index out of range");

    surface = PySurface_AsSurface (array->surface);
    bpp = surface->format->BytesPerPixel;

     /* Access of a single column. */
    if (array->xlen == 1)
    {
        if ((Uint32) _index >= array->ystart + array->ylen)
            return RAISE (PyExc_IndexError, "array index out of range");

        return _get_single_pixel ((Uint8 *) surface->pixels, bpp,
            array->xstart, _index * array->padding * array->ystep);
    }
    if (array->ylen == 1)
    {
        if ((Uint32) _index >= array->xstart + array->xlen)
            return RAISE (PyExc_IndexError, "array index out of range");

        return _get_single_pixel ((Uint8 *) surface->pixels, bpp,
            array->xstart + _index * array->xstep,
            array->ystart * array->padding * array->ystep);
    }

    return _array_slice_internal (array, _index, _index + 1, 1);
}

#if !PY3
/**
 * array[x:y]
 */
static PyObject*
_pxarray_slice (PyPixelArray *array, Py_ssize_t low, Py_ssize_t high)
{
    if (low < 0)
        low = 0;
    else if (low > (Sint32) array->xlen)
        low = array->xlen;

    if (high < low)
        high = low;
    else if (high > (Sint32) array->xlen)
        high = array->xlen;

    if (low == high)
        Py_RETURN_NONE;

    return _array_slice_internal (array, low, high, 1);
}
#endif

static int
_array_assign_array (PyPixelArray *array, Py_ssize_t low, Py_ssize_t high,
    PyPixelArray *val)
{
    SDL_Surface *surface;
    SDL_Surface *valsf = NULL;
    Uint32 x = 0;
    Uint32 y = 0;
    Uint32 vx = 0;
    Uint32 vy = 0;
    int bpp;
    int valbpp;
    Uint8 *pixels;
    Uint8 *valpixels;
    int copied = 0;

    Uint32 xstart = 0;
    Uint32 ystart = 0;
    Uint32 xlen;
    Uint32 ylen;
    Sint32 xstep;
    Sint32 ystep;
    Uint32 padding;
    Uint32 posx = 0;
    Uint32 posy = 0;
    Sint32 absxstep;
    Sint32 absystep;

    /* Set the correct slice indices */
    surface = PySurface_AsSurface (array->surface);

    if (array->xlen == 1)
    {
        xstart = array->xstart;
        ystart = array->ystart + low * array->ystep;
        xlen = array->xlen;
        ylen = ABS (high - low);
        ystep = array->ystep;
        xstep = array->xstep;
        padding = array->padding;
    }
    else
    {
        xstart = array->xstart + low * array->xstep;
        ystart = array->ystart;
        xlen = ABS (high - low);
        ylen = array->ylen;
        xstep = array->xstep;
        ystep = array->ystep;
        padding = array->padding;
    }

/*
    printf ("ARRAY: %d:%d:%d, %d:%d:%d -- VAL: %d:%d:%d, %d:%d:%d\n",
        xstart, xlen, xstep, ystart, ylen, ystep,
        val->xstart, val->xlen, val->xstep,
        val->ystart, val->ylen, val->ystep);
*/
    if (val->ylen / ABS (val->ystep) != ylen / ABS (ystep) ||
        val->xlen / ABS (val->xstep) != xlen / ABS (xstep))
    {
        /* Bounds do not match. */
        PyErr_SetString (PyExc_ValueError, "array sizes do not match");
        return -1;
    }

    valsf = PySurface_AsSurface (val->surface);
    bpp = surface->format->BytesPerPixel;
    valbpp = valsf->format->BytesPerPixel;
    pixels = (Uint8 *) surface->pixels;
    valpixels = valsf->pixels;

    if (bpp != valbpp)
    {
        /* bpp do not match. We cannot guarantee that the padding and co
         * would be set correctly. */
        PyErr_SetString (PyExc_ValueError, "bit depths do not match");
        return -1;
    }

    /* If we reassign the same array, we need to copy the pixels
     * first. */
    if (SURFACE_EQUALS (array, val))
    {
        /* We assign a different view or so. Copy the source buffer. */
        valpixels = malloc ((size_t) (surface->pitch * surface->h));
        if (!valpixels)
        {
            PyErr_SetString (PyExc_ValueError, "could not copy pixels");
            return -1;
        }
        valpixels = memcpy (valpixels, pixels,
            (size_t) (surface->pitch * surface->h));
        copied = 1;
    }

    absxstep = ABS (xstep);
    absystep = ABS (ystep);

    y = ystart;
    vy = val->ystart;

    Py_BEGIN_ALLOW_THREADS;
    /* Single value assignment. */
    switch (bpp)
    {
    case 1:
        while (posy < ylen)
        {
            vx = val->xstart;
            posx = 0;
            x = xstart;
            while (posx < xlen)
            {
                *((Uint8 *) pixels + y * padding + x) =
                    (Uint8)*((Uint8 *) valpixels + vy * val->padding + vx);
                vx += val->xstep;
                x += xstep;
                posx += absxstep;
            }
            vy += val->ystep;
            y += ystep;
            posy += absystep;
        }
        break;
    case 2:
        while (posy < ylen)
        {
            vx = val->xstart;
            posx = 0;
            x = xstart;
            while (posx < xlen)
            {
                *((Uint16 *) (pixels + y * padding) + x) =
                    (Uint16)*((Uint16 *) (valpixels + vy * val->padding) + vx);
                vx += val->xstep;
                x += xstep;
                posx += absxstep;
            }
            vy += val->ystep;
            y += ystep;
            posy += absystep;
        }
        break;
    case 3:
    {
        Uint8 *px;
        Uint8 *vpx;
        SDL_PixelFormat *format = surface->format;
        SDL_PixelFormat *vformat = valsf->format;

        while (posy < ylen)
        {
            vx = val->xstart;
            posx = 0;
            x = xstart;
            while (posx < xlen)
            {
                px = ((Uint8 *) (pixels + y * padding) + x * 3);
                vpx = ((Uint8 *) (valpixels + vy * val->padding) + vx * 3);

#if (SDL_BYTEORDER == SDL_LIL_ENDIAN)
                *(px + (format->Rshift >> 3)) =
                    *(vpx + (vformat->Rshift >> 3));
                *(px + (format->Gshift >> 3)) =
                    *(vpx + (vformat->Gshift >> 3));
                *(px + (format->Bshift >> 3)) =
                    *(vpx + (vformat->Bshift >> 3));
#else
                *(px + 2 - (format->Rshift >> 3)) =
                    *(vpx + 2 - (vformat->Rshift >> 3));
                *(px + 2 - (format->Gshift >> 3)) =
                    *(vpx + 2 - (vformat->Gshift >> 3));
                *(px + 2 - (format->Bshift >> 3)) =
                    *(vpx + 2 - (vformat->Bshift >> 3));
#endif
                vx += val->xstep;
                x += xstep;
                posx += absxstep;
            }
            vy += val->ystep;
            y += ystep;
            posy += absystep;
        }
        break;
    }
    default:
        while (posy < ylen)
        {
            vx = val->xstart;
            posx = 0;
            x = xstart;
            while (posx < xlen)
            {
                *((Uint32 *) (pixels + y * padding) + x) =
                    (Uint32)*((Uint32 *) (valpixels + vy * val->padding) + vx);
                vx += val->xstep;
                x += xstep;
                posx += absxstep;
            }
            vy += val->ystep;
            y += ystep;
            posy += absystep;
        }
        break;
    }
    Py_END_ALLOW_THREADS;
    
    if (copied)
    {
        free (valpixels);
    }
    return 0;
}

static int
_array_assign_sequence (PyPixelArray *array, Py_ssize_t low, Py_ssize_t high,
    PyObject *val)
{
    SDL_Surface *surface;
    Uint32 x = 0;
    Uint32 y = 0;
    int bpp;
    int gooverx = 0;
    Uint8 *pixels;
    Uint32 color = 0;
    Uint32 *colorvals = NULL;
    Uint32 *nextcolor = NULL;
    Py_ssize_t offset = 0;
    Py_ssize_t seqsize = PySequence_Size (val);

    Uint32 xstart = 0;
    Uint32 ystart = 0;
    Uint32 xlen;
    Uint32 ylen;
    Sint32 xstep;
    Sint32 ystep;
    Uint32 padding;
    Uint32 posx = 0;
    Uint32 posy = 0;
    Sint32 absxstep;
    Sint32 absystep;

    surface = PySurface_AsSurface (array->surface);
    bpp = surface->format->BytesPerPixel;
    pixels = (Uint8 *) surface->pixels;

    /* Set the correct slice indices */
    if (array->xlen == 1)
    {
        xstart = array->xstart;
        ystart = array->ystart + low * array->ystep;
        xlen = array->xlen;
        ylen = ABS (high - low);
        ystep = array->ystep;
        xstep = array->xstep;
        padding = array->padding;
    }
    else
    {
        xstart = array->xstart + low * array->xstep;
        ystart = array->ystart;
        xlen = ABS (high - low);
        ylen = array->ylen;
        xstep = array->xstep;
        ystep = array->ystep;
        padding = array->padding;
    }

    if ((Uint32)seqsize != ylen / ABS (ystep))
    {
        if ((Uint32)seqsize != xlen / ABS (xstep))
        {
            PyErr_SetString(PyExc_ValueError, "sequence size mismatch");
            return -1;
        }
        gooverx = 1; /* We have to iterate over the x axis. */
    }

    if (seqsize == 1)
    {
        /* Single value assignment. */
        _set_single_pixel (pixels, bpp, xstart, ystart + padding * ystep,
            surface->format, color);
        return 0;
    }

    /* Copy the values. */
    colorvals = malloc (sizeof (Uint32) * seqsize);
    if (!colorvals)
    {
        PyErr_SetString (PyExc_ValueError, "could not copy colors");
        return -1;
    }

    for (offset = 0; offset < seqsize; offset++)
    {
        if (!_get_color_from_object (PySequence_Fast_GET_ITEM (val, offset),
                surface->format, &color))
        {
            free (colorvals);
            return -1;
        }
        colorvals[offset] = color;
    }

    absxstep = ABS (xstep);
    absystep = ABS (ystep);
    y = ystart;
    nextcolor = colorvals;

    Py_BEGIN_ALLOW_THREADS;
    switch (bpp)
    {
    case 1:
        if (gooverx)
        {
            while (posy < ylen)
            {
                posx = 0;
                x = xstart;
                nextcolor = colorvals;
                while (posx < xlen)
                {
                    color = *nextcolor++;
                    *((Uint8 *) pixels + y * padding + x) = (Uint8) color;
                    x += xstep;
                    posx += absxstep;
                }
                y += ystep;
                posy += absystep;
            }
        }
        else
        {
            while (posy < ylen)
            {
                posx = 0;
                x = xstart;
                color = *nextcolor++;
                while (posx < xlen)
                {
                    *((Uint8 *) pixels + y * padding + x) = (Uint8) color;
                    x += xstep;
                    posx += absxstep;
                }
                y += ystep;
                posy += absystep;
            }
        }        
        break;
    case 2:
        if (gooverx)
        {
            while (posy < ylen)
            {
                posx = 0;
                x = xstart;
                nextcolor = colorvals;
                while (posx < xlen)
                {
                    color = *nextcolor++;
                    *((Uint16 *) (pixels + y * padding) + x) = (Uint16) color;
                    x += xstep;
                    posx += absxstep;
                }
                y += ystep;
                posy += absystep;
            }
        }
        else
        {
            while (posy < ylen)
            {
                posx = 0;
                x = xstart;
                color = *nextcolor++;
                while (posx < xlen)
                {
                    *((Uint16 *) (pixels + y * padding) + x) = (Uint16) color;
                    x += xstep;
                    posx += absxstep;
                }
                y += ystep;
                posy += absystep;
            }
        }
        break;
    case 3:
    {
        Uint8 *px;
        SDL_PixelFormat *format = surface->format;

        if (gooverx)
        {
            while (posy < ylen)
            {
                posx = 0;
                x = xstart;
                nextcolor = colorvals;
                while (posx < xlen)
                {
                    color = *nextcolor++;
                    px = ((Uint8 *) (pixels + y * padding) + x * 3);
#if (SDL_BYTEORDER == SDL_LIL_ENDIAN)
                    *(px + (format->Rshift >> 3)) = (Uint8) (color >> 16);
                    *(px + (format->Gshift >> 3)) = (Uint8) (color >> 8);
                    *(px + (format->Bshift >> 3)) = (Uint8) color;
#else
                    *(px + 2 - (format->Rshift >> 3)) = (Uint8) (color >> 16);
                    *(px + 2 - (format->Gshift >> 3)) = (Uint8) (color >> 8);
                    *(px + 2 - (format->Bshift >> 3)) = (Uint8) color;
#endif
                    x += xstep;
                    posx += absxstep;
                }
                y += ystep;
                posy += absystep;
            }
        }
        else
        {
            while (posy < ylen)
            {
                posx = 0;
                x = xstart;
                color = *nextcolor++;
                while (posx < xlen)
                {
                    px = ((Uint8 *) (pixels + y * padding) + x * 3);
#if (SDL_BYTEORDER == SDL_LIL_ENDIAN)
                    *(px + (format->Rshift >> 3)) = (Uint8) (color >> 16);
                    *(px + (format->Gshift >> 3)) = (Uint8) (color >> 8);
                    *(px + (format->Bshift >> 3)) = (Uint8) color;
#else
                    *(px + 2 - (format->Rshift >> 3)) = (Uint8) (color >> 16);
                    *(px + 2 - (format->Gshift >> 3)) = (Uint8) (color >> 8);
                    *(px + 2 - (format->Bshift >> 3)) = (Uint8) color;
#endif
                    x += xstep;
                    posx += absxstep;
                }
                y += ystep;
                posy += absystep;
            }
        }
        break;
    }
    default:
        if (gooverx)
        {
            while (posy < ylen)
            {
                posx = 0;
                x = xstart;
                nextcolor = colorvals;
                while (posx < xlen)
                {
                    color = *nextcolor++;
                    *((Uint32 *) (pixels + y * padding) + x) = color;
                    x += xstep;
                    posx += absxstep;
                }
                y += ystep;
                posy += absystep;
            }
        }
        else
        {
            while (posy < ylen)
            {
                posx = 0;
                x = xstart;
                color = *nextcolor++;
                while (posx < xlen)
                {
                    *((Uint32 *) (pixels + y * padding) + x) = color;
                    x += xstep;
                    posx += absxstep;
                }
                y += ystep;
                posy += absystep;
            }
        }
        break;
    }
    free (colorvals);
    Py_END_ALLOW_THREADS;
    return 0;
}

static int
_array_assign_slice (PyPixelArray *array, Py_ssize_t low, Py_ssize_t high,
    Uint32 color)
{
    SDL_Surface *surface;
    Uint32 x = 0;
    Uint32 y = 0;
    int bpp;
    Uint8 *pixels;

    Uint32 xstart = 0;
    Uint32 ystart = 0;
    Uint32 xlen;
    Uint32 ylen;
    Sint32 xstep;
    Sint32 ystep;
    Uint32 padding;
    Uint32 posx = 0;
    Uint32 posy = 0;
    Sint32 absxstep;
    Sint32 absystep;

    surface = PySurface_AsSurface (array->surface);
    bpp = surface->format->BytesPerPixel;
    pixels = (Uint8 *) surface->pixels;

    /* Set the correct slice indices */
    if (array->xlen == 1)
    {
        xstart = array->xstart;
        ystart = array->ystart + low * array->ystep;
        xlen = array->xlen;
        ylen = ABS (high - low);
        ystep = array->ystep;
        xstep = array->xstep;
        padding = array->padding;
    }
    else
    {
        xstart = array->xstart + low * array->xstep;
        ystart = array->ystart;
        xlen = ABS (high - low);
        ylen = array->ylen;
        xstep = array->xstep;
        ystep = array->ystep;
        padding = array->padding;
    }

    absxstep = ABS (xstep);
    absystep = ABS (ystep);
    y = ystart;

    Py_BEGIN_ALLOW_THREADS;
    /* Single value assignment. */
    switch (bpp)
    {
    case 1:
        while (posy < ylen)
        {
            posx = 0;
            x = xstart;
            while (posx < xlen)
            {
                *((Uint8 *) pixels + y * padding + x) = (Uint8) color;
                x += xstep;
                posx += absxstep;
            }
            y += ystep;
            posy += absystep;
        }
        break;
    case 2:
        while (posy < ylen)
        {
            posx = 0;
            x = xstart;
            while (posx < xlen)
            {
                *((Uint16 *) (pixels + y * padding) + x) = (Uint16) color;
                x += xstep;
                posx += absxstep;
            }
            y += ystep;
            posy += absystep;
        }
        break;
    case 3:
    {
        Uint8 *px;
        SDL_PixelFormat *format = surface->format;

        while (posy < ylen)
        {
            posx = 0;
            x = xstart;
            while (posx < xlen)
            {
                px = ((Uint8 *) (pixels + y * padding) + x * 3);
#if (SDL_BYTEORDER == SDL_LIL_ENDIAN)
                *(px + (format->Rshift >> 3)) = (Uint8) (color >> 16);
                *(px + (format->Gshift >> 3)) = (Uint8) (color >> 8);
                *(px + (format->Bshift >> 3)) = (Uint8) color;
#else
                *(px + 2 - (format->Rshift >> 3)) = (Uint8) (color >> 16);
                *(px + 2 - (format->Gshift >> 3)) = (Uint8) (color >> 8);
                *(px + 2 - (format->Bshift >> 3)) = (Uint8) color;
#endif
                x += xstep;
                posx += absxstep;
            }
            y += ystep;
            posy += absystep;
        }
        break;
    }
    default: /* 4 bpp */
        while (posy < ylen)
        {
            posx = 0;
            x = xstart;
            while (posx < xlen)
            {
                *((Uint32 *) (pixels + y * padding) + x) = color;
                x += xstep;
                posx += absxstep;
            }
            y += ystep;
            posy += absystep;
        }
        break;
    }
    Py_END_ALLOW_THREADS;
    return 0;
}

/**
 * array[x] = ...
 */
static int
_pxarray_ass_item (PyPixelArray *array, Py_ssize_t _index, PyObject *value)
{
    SDL_Surface *surface;
    Uint32 x = 0;
    Uint32 y = 0;
    int bpp;
    Uint8 *pixels;
    Uint32 color = 0;

    Uint32 xstart = 0;
    Uint32 ystart = 0;
    Uint32 xlen;
    Uint32 ylen;
    Sint32 xstep;
    Sint32 ystep;
    Uint32 padding;
    Uint32 posx = 0;
    Uint32 posy = 0;
    Sint32 absxstep;
    Sint32 absystep;

    surface = PySurface_AsSurface (array->surface);
    bpp = surface->format->BytesPerPixel;
    pixels = (Uint8 *) surface->pixels;

    if (!_get_color_from_object (value, surface->format, &color))
    {
        if (PyPixelArray_Check (value))
        {
            PyErr_Clear (); /* _get_color_from_object */
            return _array_assign_array (array, _index, _index + 1,
                (PyPixelArray *) value);
        }
        else if (PySequence_Check (value))
        {
            PyErr_Clear (); /* _get_color_from_object */
            return _array_assign_sequence (array, _index, _index + 1, value);
        }
        else /* Error already set by _get_color_from_object(). */
            return -1;
    }

    /* Set the correct slice indices */
    if (array->xlen == 1)
    {
        xstart = array->xstart;
        ystart = array->ystart + _index * array->ystep;
        xlen = array->xlen;
        ylen = 1;
        ystep = array->ystep;
        xstep = array->xstep;
        padding = array->padding;
    }
    else
    {
        xstart = array->xstart + _index * array->xstep;
        ystart = array->ystart;
        xlen = 1;
        ylen = array->ylen;
        xstep = array->xstep;
        ystep = array->ystep;
        padding = array->padding;
    }

/*
    printf ("ITEM: %d:%d, %d:%d %d:%d %d\n",
        xstart, ystart, xlen, xstep, ylen, ystep, padding);
*/
    absxstep = ABS (xstep);
    absystep = ABS (ystep);
    y = ystart;

    Py_BEGIN_ALLOW_THREADS;
    /* Single value assignment. */
    switch (bpp)
    {
    case 1:
        while (posy < ylen)
        {
            posx = 0;
            x = xstart;
            while (posx < xlen)
            {
                *((Uint8 *) pixels + y * padding + x) = (Uint8) color;
                x += xstep;
                posx += absxstep;
            }
            y += ystep;
            posy += absystep;
        }
        break;
    case 2:
        while (posy < ylen)
        {
            posx = 0;
            x = xstart;
            while (posx < xlen)
            {
                *((Uint16 *) (pixels + y * padding) + x) = (Uint16) color;
                x += xstep;
                posx += absxstep;
            }
            y += ystep;
            posy += absystep;
        }
        break;
    case 3:
    {
        Uint8 *px;
        SDL_PixelFormat *format = surface->format;

        while (posy < ylen)
        {
            posx = 0;
            x = xstart;
            while (posx < xlen)
            {
                px = ((Uint8 *) (pixels + y * padding) + x * 3);
#if (SDL_BYTEORDER == SDL_LIL_ENDIAN)
                *(px + (format->Rshift >> 3)) = (Uint8) (color >> 16);
                *(px + (format->Gshift >> 3)) = (Uint8) (color >> 8);
                *(px + (format->Bshift >> 3)) = (Uint8) color;
#else
                *(px + 2 - (format->Rshift >> 3)) = (Uint8) (color >> 16);
                *(px + 2 - (format->Gshift >> 3)) = (Uint8) (color >> 8);
                *(px + 2 - (format->Bshift >> 3)) = (Uint8) color;
#endif
                x += xstep;
                posx += absxstep;
            }
            y += ystep;
            posy += absystep;
        }
        break;
    }
    default: /* 4 bpp */
        while (posy < ylen)
        {
            posx = 0;
            x = xstart;
            while (posx < xlen)
            {
                *((Uint32 *) (pixels + y * padding) + x) = color;
                x += xstep;
                posx += absxstep;
            }
            y += ystep;
            posy += absystep;
        }
        break;
    }
    Py_END_ALLOW_THREADS;
    return 0;
}

/**
 * array[x:y] = ....
 */
static int
_pxarray_ass_slice (PyPixelArray *array, Py_ssize_t low, Py_ssize_t high,
    PyObject *value)
{
    SDL_Surface *surface;
    Uint32 color;

    if (array->xlen != 1)
    {
        if (low < 0)
            low = 0;
        else if (low > (Sint32) array->xlen)
            low = array->xlen;
        
        if (high < low)
            high = low;
        else if (high > (Sint32) array->xlen)
            high = array->xlen;
    }
    else
    {
        if (low < 0)
            low = 0;
        else if (low > (Sint32) array->ylen)
            low = array->ylen;
        
        if (high < low)
            high = low;
        else if (high > (Sint32) array->ylen)
            high = array->ylen;
    }

    surface = PySurface_AsSurface (array->surface);
/*
    printf ("SLICE IS: %d:%d\n", low, high);
*/
    if (PyPixelArray_Check (value))
    {
        return _array_assign_array (array, low, high, (PyPixelArray *) value);
    }
    else if (_get_color_from_object (value, surface->format, &color))
    {
        return _array_assign_slice (array, low, high, color);
    }
    else if (PySequence_Check (value))
    {
        PyErr_Clear (); /* In case _get_color_from_object set it */
        return _array_assign_sequence (array, low, high, value);
    }
    return 0;
}

/**
 * x in array
 */
static int
_pxarray_contains (PyPixelArray *array, PyObject *value)
{
    SDL_Surface *surface;
    Uint32 x = 0;
    Uint32 y = 0;
    Uint8 *pixels;
    int bpp;
    Uint32 color;

    Uint32 posx = 0;
    Uint32 posy = 0;
    Sint32 absxstep;
    Sint32 absystep;
    int found = 0;

    surface = PySurface_AsSurface (array->surface);
    bpp = surface->format->BytesPerPixel;
    pixels = (Uint8 *) surface->pixels;

    if (!_get_color_from_object (value, surface->format, &color))
        return -1;

    absxstep = ABS (array->xstep);
    absystep = ABS (array->ystep);
    y = array->ystart;

    Py_BEGIN_ALLOW_THREADS;
    switch (bpp)
    {
    case 1:
        while (posy < array->ylen && !found)
        {
            posx = 0;
            x = array->xstart;
            while (posx < array->xlen)
            {
                if (*((Uint8 *) pixels + y * array->padding + x)
                    == (Uint8) color)
                {
                    found = 1;
                    break;
                }
                x += array->xstep;
                posx += absxstep;
            }
            y += array->ystep;
            posy += absystep;
        }
        break;
    case 2:
        while (posy < array->ylen && !found)
        {
            posx = 0;
            x = array->xstart;
            while (posx < array->xlen)
            {
                if (*((Uint16 *) (pixels + y * array->padding) + x)
                    == (Uint16) color)
                {
                    found = 1;
                    break;
                }
                x += array->xstep;
                posx += absxstep;
            }
            y += array->ystep;
            posy += absystep;
        }
        break;
    case 3:
    {
        Uint32 pxcolor;
        Uint8 *pix;

        while (posy < array->ylen && !found)
        {
            posx = 0;
            x = array->xstart;
            while (posx < array->xlen)
            {
                pix = ((Uint8 *) (pixels + y * array->padding) + x * 3);
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                pxcolor = (pix[0]) + (pix[1] << 8) + (pix[2] << 16);
#else
                pxcolor = (pix[2]) + (pix[1] << 8) + (pix[0] << 16);
#endif
                if (pxcolor == color)
                {
                    found = 1;
                    break;
                }
                x += array->xstep;
                posx += absxstep;
            }
            y += array->ystep;
            posy += absystep;
        }
        break;
    }
    default: /* 4 bpp */
        while (posy < array->ylen && !found)
        {
            posx = 0;
            x = array->xstart;
            while (posx < array->xlen)
            {
                if (*((Uint32 *) (pixels + y * array->padding) + x)
                    == color)
                {
                    found = 1;
                    break;
                }
                x += array->xstep;
                posx += absxstep;
            }
            y += array->ystep;
            posy += absystep;
        }
        break;
    }

    Py_END_ALLOW_THREADS;
    return found;
}

/**
 * iter (arrray), for x in array
 */
static PyObject*
_pxarray_iter (PyPixelArray *array)
{
/*
    printf ("Iter ARRAY: %d:%d:%d %d:%d:%d\n",
        array->xstart, array->xlen, array->xstep,
        array->ystart, array->ylen, array->ystep);
*/
    return PySeqIter_New ((PyObject *) array);
}

/**** Mapping interfaces ****/

/**
 * Internally used parser function for the 2D slices:
 * array[x,y], array[:,:], ...
 */
static int
_get_subslice (PyObject *op, Py_ssize_t length, Py_ssize_t *start,
    Py_ssize_t *stop, Py_ssize_t *step)
{
    *start = -1;
    *stop = -1;
    *step = -1;

    if (PySlice_Check (op))
    {
        Py_ssize_t slicelen;

        /* Operator is a slice: array[x::, */
        if (PySlice_GetIndicesEx ((PySliceObject *) op, length,
                start, stop, step, &slicelen) < 0)
        {
            return 0;
        }
    }
    else if (PyInt_Check (op))
    {
        /* Plain index: array[x, */
        *start = PyInt_AsLong (op);
        if (*start < 0)
            *start += length;
        if (*start >= length || *start < 0)
        {
            PyErr_SetString(PyExc_IndexError, "invalid index");
            return 0;
        }   
        *stop = (*start) + 1;
        *step = 1;
    }
    else if (PyLong_Check (op))
    {
        long long val = -1;
        /* Plain index: array[x, */

        val = PyLong_AsLong (op);
        if ((val < INT_MIN) || (val > INT_MAX))
        {
            PyErr_SetString(PyExc_ValueError,
                "index too big for array access");
            return 0;
        }
        *start = (int) val;
        if (*start < 0)
            *start += length;
        if (*start >= length || *start < 0)
        {
            PyErr_SetString(PyExc_IndexError, "invalid index");
            return 0;
        }   
        *stop = (*start) + 1;
        *step = 1;
    }
    return 1;
}

/**
 * Slicing support for 1D and 2D access.
 * array[x,y] is only supported for 2D arrays.
 */
static PyObject*
_pxarray_subscript (PyPixelArray *array, PyObject *op)
{
    SDL_Surface *surface = PySurface_AsSurface (array->surface);

    /* Note: order matters here.
     * First check array[x,y], then array[x:y:z], then array[x]
     * Otherwise it'll fail.
     */
    if (PySequence_Check (op))
    {
        PyObject *obj;
        Py_ssize_t size = PySequence_Size (op);
        Py_ssize_t xstart, xstop, xstep;
        Py_ssize_t ystart, ystop, ystep;
        Py_ssize_t lenx, leny;
        
        if (size == 0)
        {
            /* array[,], array[()] ... */
            Py_INCREF (array);
            return (PyObject *) array;
        }
        if (size > 2 || (size == 2 && array->xlen == 1))
            return RAISE (PyExc_IndexError, "too many indices for the array");

        lenx = (array->xlen > 1) ? array->xlen / ABS (array->xstep) : 0;
        leny = array->ylen / ABS (array->ystep);

        obj = PySequence_Fast_GET_ITEM (op, 0);
        if (obj == Py_Ellipsis || obj == Py_None)
        {
            /* Operator is the ellipsis or None
             * array[...,XXX], array[None,XXX]
             */
            xstart = 0;
            xstop = array->xlen;
            xstep = array->xstep;
        }
        else if (!_get_subslice (obj, lenx, &xstart, &xstop, &xstep))
        {
            /* Error on retrieving the subslice. */
            return NULL;
        }

        if (size == 2)
        {
            obj = PySequence_Fast_GET_ITEM (op, 1);
            if (obj == Py_Ellipsis || obj == Py_None)
            {
                /* Operator is the ellipsis or None
                 * array[XXX,...], array[XXX,None]
                 */
                ystart = array->ystart;
                ystop = array->ylen;
                ystep = array->ystep;
            }
            else if (!_get_subslice (obj, leny, &ystart, &ystop, &ystep))
            {
                /* Error on retrieving the subslice. */
                return NULL;
            }
        }
        else
        {
            ystart = array->ystart;
            ystop = array->ylen;
            ystep = array->ystep;
        }

        /* Null value? */
        if (xstart == xstop || ystart == ystop)
            Py_RETURN_NONE;
/*
        printf ("X: %d:%d:%d Y: %d:%d:%d\n", xstart, xstop, xstep,
            ystart, ystop, ystep);
*/
        /* Single value? */
        if (ABS (xstop - xstart) == 1 && ABS (ystop - ystart) == 1)
        {
            return  _get_single_pixel ((Uint8 *) surface->pixels,
                surface->format->BytesPerPixel, array->xstart + xstart,
                ystart * array->padding * array->ystep);
        }

/*
        printf ("NEW ARRAY: %d:%d:%d %d:%d:%d\n",
            array->xstart + xstart, ABS (xstop - xstart), xstep,
            array->ystart + ystart, ABS (ystop - ystart), ystep);
*/
        return (PyObject *) _pxarray_new_internal (&PyPixelArray_Type,
            array->surface,
            (Uint32) array->xstart + xstart,
            (Uint32) array->ystart + ystart,
            (Uint32) ABS (xstop - xstart),
            (Uint32) ABS (ystop - ystart),
            (Sint32) xstep,
            (Sint32) ystep,
            (Uint32) array->padding, (PyObject *) array);
    }
    else if (PySlice_Check (op))
    {
        /* A slice */
        Py_ssize_t slicelen;
        Py_ssize_t step;
        Py_ssize_t start;
        Py_ssize_t stop;
        int retval;

        if (array->xlen > 1)
        {
            /* 2D array - slice along the x axis */
            retval = PySlice_GetIndicesEx ((PySliceObject *) op,
                (Py_ssize_t) (array->xlen / ABS (array->xstep)), &start, &stop,
                &step, &slicelen);
        }
        else
        {
            /* 1D array - use the y axis. */
            retval = PySlice_GetIndicesEx ((PySliceObject *) op,
                (Py_ssize_t) (array->ylen / ABS (array->ystep)), &start, &stop,
                &step, &slicelen);
        }
        if (retval < 0 || slicelen < 0)
            return NULL;
        if (slicelen == 0)
            Py_RETURN_NONE;
/*
        printf ("start: %d, stop: %d, step: %d, len: %d\n", start, stop,
            step, slicelen);
*/
        return (PyObject *) _array_slice_internal (array, start, stop, step);
    }
    else if (PyIndex_Check (op) || PyInt_Check (op) || PyLong_Check (op))
    {
        Py_ssize_t i;
#if PY_VERSION_HEX >= 0x02050000
        PyObject *val = PyNumber_Index (op);
        if (!val)
            return NULL;
        /* A simple index. */
        i = PyNumber_AsSsize_t (val, PyExc_IndexError);
#else
        if (PyInt_Check (op))
            i = PyInt_AsLong (op);
        else
            i = PyLong_AsLong (op);
#endif 
        if (i == -1 && PyErr_Occurred ())
            return NULL;
        if (i < 0)
            i += (array->xlen > 1) ? array->xlen / ABS (array->xstep) :
                array->ylen / ABS (array->ystep);

        return _pxarray_item (array, i);
    }

    return RAISE (PyExc_TypeError,
        "index must be an integer, sequence or slice");
}

static int
_pxarray_ass_subscript (PyPixelArray *array, PyObject* op, PyObject* value)
{
    /* TODO: by time we can make this faster by avoiding the creation of
     * temporary subarrays.
     */
    
    /* Note: order matters here.
     * First check array[x,y], then array[x:y:z], then array[x]
     * Otherwise it'll fail.
     */
    if (PySequence_Check (op))
    {
        PyPixelArray *tmparray;
        PyObject *obj;
        Py_ssize_t size = PySequence_Size (op);
        Py_ssize_t xstart, xstop, xstep;
        Py_ssize_t ystart, ystop, ystep;
        Py_ssize_t lenx, leny;
        int retval;
        
        if (size == 0)
        {
            /* array[,], array[()] ... */
            if (array->xlen == 1)
                return _pxarray_ass_slice (array, 0, (Py_ssize_t) array->ylen,
                    value);
            else
                return _pxarray_ass_slice (array, 0, (Py_ssize_t) array->xlen,
                    value);
        }
        if (size > 2 || (size == 2 && array->xlen == 1))
        {
            PyErr_SetString (PyExc_IndexError,
                "too many indices for the array");
            return -1;
        }

        lenx = (array->xlen > 1) ? array->xlen / ABS (array->xstep) : 0;
        leny = array->ylen / ABS (array->ystep);

        obj = PySequence_Fast_GET_ITEM (op, 0);
        if (obj == Py_Ellipsis || obj == Py_None)
        {
            /* Operator is the ellipsis or None
             * array[...,XXX], array[None,XXX]
             */
            xstart = 0;
            xstop = array->xlen;
            xstep = array->xstep;
        }
        else if (!_get_subslice (obj, lenx, &xstart, &xstop, &xstep))
        {
            /* Error on retrieving the subslice. */
            return -1;
        }

        if (size == 2)
        {
            obj = PySequence_Fast_GET_ITEM (op, 1);
            if (obj == Py_Ellipsis || obj == Py_None)
            {
                /* Operator is the ellipsis or None
                 * array[XXX,...], array[XXX,None]
                 */
                ystart = array->ystart;
                ystop = array->ylen;
                ystep = array->ystep;
            }
            else if (!_get_subslice (obj, leny, &ystart, &ystop, &ystep))
            {
                /* Error on retrieving the subslice. */
                return -1;
            }
        }
        else
        {
            ystart = array->ystart;
            ystop = array->ylen;
            ystep = array->ystep;
        }

        /* Null value? Do nothing then. */
        if (xstart == xstop || ystart == ystop)
            return 0;

        /* Single value? */
        if (ABS (xstop - xstart) == 1 && ABS (ystop - ystart) == 1)
        {
            tmparray = _pxarray_new_internal (&PyPixelArray_Type,
                array->surface,
                (Uint32) array->xstart + xstart,
                (Uint32) array->ystart + ystart,
                1, 1, 1, 1, (Uint32) array->padding, (PyObject *) array);
            if (!tmparray)
                return -1;
            retval = _pxarray_ass_item (tmparray, 0, value);
            Py_DECREF (tmparray);
            return retval;
        }
        tmparray =_pxarray_new_internal (&PyPixelArray_Type,
            array->surface,
            (Uint32) array->xstart + xstart, (Uint32) array->ystart + ystart,
            (Uint32) ABS (xstop - xstart), (Uint32) ABS (ystop - ystart),
            (Sint32) xstep, (Sint32) ystep,
            (Uint32) array->padding, (PyObject *) array);
        if (!tmparray)
            return -1;

        if (tmparray->xlen == 1)
            retval = _pxarray_ass_slice (tmparray, 0,
                (Py_ssize_t) tmparray->ylen, value);
        else
            retval = _pxarray_ass_slice (tmparray, 0,
                (Py_ssize_t) tmparray->xlen, value);
        Py_DECREF (tmparray);
        return retval;
    }
    else if (PySlice_Check (op))
    {
        /* A slice */
        PyPixelArray *tmparray;
        Py_ssize_t slicelen;
        Py_ssize_t step;
        Py_ssize_t start;
        Py_ssize_t stop;
        int retval;

        if (array->xlen > 1)
        {
            /* 2D array - slice along the x axis */
            retval = PySlice_GetIndicesEx ((PySliceObject *) op,
                (Py_ssize_t) (array->xlen / ABS (array->xstep)), &start, &stop,
                &step, &slicelen);
        }
        else
        {
            /* 1D array - use the y axis. */
            retval = PySlice_GetIndicesEx ((PySliceObject *) op,
                (Py_ssize_t) (array->ylen / ABS (array->ystep)), &start, &stop,
                &step, &slicelen);
        }
        if (retval < 0 || slicelen < 0)
            return -1;
        if (slicelen == 0)
            return 0;

/*
        printf ("start: %d, stop: %d, step: %d, len: %d\n", start, stop,
            step, slicelen);
*/
        tmparray = (PyPixelArray *) _array_slice_internal (array, start, stop,
            step);
        if (!tmparray)
            return -1;
        if (tmparray->xlen == 1)
            retval = _pxarray_ass_slice (tmparray, 0,
                (Py_ssize_t) tmparray->ylen, value);
        else
            retval = _pxarray_ass_slice (tmparray, 0,
                (Py_ssize_t) tmparray->xlen, value);
        Py_DECREF (tmparray);
        return retval;
    }
    else if (PyIndex_Check (op) || PyInt_Check (op) || PyLong_Check (op))
    {
        Py_ssize_t i;
#if PY_VERSION_HEX >= 0x02050000
        PyObject *val = PyNumber_Index (op);
        if (!val)
            return -1;
        /* A simple index. */
        i = PyNumber_AsSsize_t (val, PyExc_IndexError);
#else
        if (PyInt_Check (op))
            i = PyInt_AsLong (op);
        else
            i = PyLong_AsLong (op);
#endif 
        if (i == -1 && PyErr_Occurred ())
            return -1;
        if (i < 0)
            i += (array->xlen > 1) ? array->xlen / ABS (array->xstep) :
                array->ylen / ABS (array->ystep);

        return _pxarray_ass_item (array, i, value);
    }

    PyErr_SetString (PyExc_TypeError,
        "index must be an integer, sequence or slice");
    return -1;
}

/**** C API interfaces ****/
static PyObject* PyPixelArray_New (PyObject *surfobj)
{
    SDL_Surface *surface;

    if (!PySurface_Check (surfobj))
        return RAISE (PyExc_TypeError, "argument is no a Surface");

    surface = PySurface_AsSurface (surfobj);
    if (surface->format->BytesPerPixel < 1  ||
        surface->format->BytesPerPixel > 4)
        return RAISE (PyExc_ValueError,
            "unsupported bit depth for reference array");

    return (PyObject *) _pxarray_new_internal
        (&PyPixelArray_Type, surfobj, 0, 0,
            (Uint32) surface->w, (Uint32) surface->h, 1, 1,
            (Uint32) surface->pitch, NULL);
}

MODINIT_DEFINE (pixelarray)
{
    PyObject *module;
    PyObject *dict;
    PyObject *apiobj;
    int ecode;
    static void* c_api[PYGAMEAPI_PIXELARRAY_NUMSLOTS];

#if PY3
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "pixelarray",
        NULL,
        -1,
        NULL,
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
    import_pygame_color();
    if (PyErr_Occurred ()) {
        MODINIT_ERROR;
    }
    import_pygame_surface ();
    if (PyErr_Occurred ()) {
        MODINIT_ERROR;
    }

    /* type preparation */
    if (PyType_Ready (&PyPixelArray_Type) < 0)
    {
        MODINIT_ERROR;
    }
    
    /* create the module */
#if PY3
    module = PyModule_Create (&_module);
#else
    module = Py_InitModule3 (MODPREFIX "pixelarray", NULL, NULL);
#endif
    if (module == NULL)
    {
        MODINIT_ERROR;
    }
    Py_INCREF (&PyPixelArray_Type);
    if (PyModule_AddObject (module, "PixelArray",
                            (PyObject *) &PyPixelArray_Type) == -1)
    {
        Py_DECREF ((PyObject *) &PyPixelArray_Type);
        DECREF_MOD (module);
        MODINIT_ERROR;
    }
    PyPixelArray_Type.tp_getattro = PyObject_GenericGetAttr;
    dict = PyModule_GetDict (module);

    c_api[0] = &PyPixelArray_Type;
    c_api[1] = PyPixelArray_New;
    apiobj = encapsulate_api (c_api, "pixelarray");
    if (apiobj == NULL)
    {
        DECREF_MOD (module);
        MODINIT_ERROR;
    }
    ecode = PyDict_SetItemString (dict, PYGAMEAPI_LOCAL_ENTRY, apiobj);
    Py_DECREF (apiobj);
    if (ecode == -1)
    {
        DECREF_MOD (module);
        MODINIT_ERROR;
    }
    MODINIT_RETURN (module);
}
