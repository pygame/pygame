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
#include "pgarrinter.h"
#include "surface.h"

#if PY_VERSION_HEX < 0x02050000
#define PyIndex_Check(op) 0
#endif

struct _pixelarray_t;

/* The array, like its surface, is in column-major (FORTRAN) order.
   This is the reverse of C's row-major order. All array descriptor
   fields are relative to the original surface data. The pixels
   field is adjusted for any offsets of the array view into the
   surface. If dim1 is 0 the array is one dimensional.
 */
typedef struct _pixelarray_t {
    PyObject_HEAD
    PyObject *dict;        /* dict for subclassing */
    PyObject *weakrefs;    /* Weakrefs for subclassing */
    PyObject *surface;     /* Surface associated with the array */
    PyObject *lock;        /* Lock object for the surface */
    Py_ssize_t dim0;       /* Shape 0 of array, in pixels */
    Py_ssize_t dim1;       /* Shape 1 of array, in pixels */
    Py_ssize_t stride0;    /* Offset, in bytes, between pixels within a row */
    Py_ssize_t stride1;    /* Offset, in bytes, between rows */
    Uint8  *pixels;        /* Start of array data */
    struct _pixelarray_t *parent;
                           /* Parent pixel array: NULL if no parent */
} PyPixelArray;

static PyPixelArray *_pxarray_new_internal(
    PyTypeObject *type, PyObject *surface, PyPixelArray *parent, Uint8 *pixels,
    Py_ssize_t dim0, Py_ssize_t dim1, Py_ssize_t stride0, Py_ssize_t stride1);

static PyObject *_pxarray_new(
    PyTypeObject *type, PyObject *args, PyObject *kwds);
static void _pxarray_dealloc(PyPixelArray *self);

static PyObject *_pxarray_get_dict(PyPixelArray *self, void *closure);
static PyObject *_pxarray_get_surface(PyPixelArray *self, void *closure);
static PyObject *_pxarray_get_itemsize(PyPixelArray *self, void *closure);
static PyObject *_pxarray_get_shape(PyPixelArray *self, void *closure);
static PyObject *_pxarray_get_strides(PyPixelArray *self, void *closure);
static PyObject *_pxarray_get_ndim(PyPixelArray *self, void *closure);
static PyObject *_pxarray_get_arraystruct(PyPixelArray *self, void *closure);
static PyObject *_pxarray_get_pixelsaddress(PyPixelArray *self, void *closure);
static PyObject *_pxarray_repr(PyPixelArray *array);

static PyObject *_array_slice_internal(
    PyPixelArray *array, Sint32 start, Sint32 end, Sint32 step);

#if PY3
static void _pxarray_capsule_destr(PyObject *capsule);
#endif

/* Sequence methods */
static Py_ssize_t _pxarray_length(PyPixelArray *array);
static PyObject *_pxarray_item(PyPixelArray *array, Py_ssize_t index);
static int _array_assign_array(
    PyPixelArray *array, Py_ssize_t low, Py_ssize_t high, PyPixelArray *val);
static int _array_assign_sequence(
    PyPixelArray *array, Py_ssize_t low, Py_ssize_t high, PyObject *val);
static int _array_assign_slice(
    PyPixelArray *array, Py_ssize_t low, Py_ssize_t high, Uint32 color);
static int _pxarray_ass_item(
    PyPixelArray *array, Py_ssize_t index, PyObject *value);
static int _pxarray_ass_slice(
    PyPixelArray *array, Py_ssize_t low, Py_ssize_t high, PyObject *value);
static int _pxarray_contains(PyPixelArray *array, PyObject *value);

/* Mapping methods */
static int _get_subslice(PyObject *op, Py_ssize_t length,
                         Py_ssize_t *start, Py_ssize_t *stop, Py_ssize_t *step);
static PyObject *_pxarray_subscript(PyPixelArray *array, PyObject *op);
static int _pxarray_ass_subscript(
    PyPixelArray *array, PyObject* op, PyObject* value);

static PyObject *_pxarray_subscript_internal(
    PyPixelArray *array,
    Py_ssize_t xstart, Py_ssize_t xstop, Py_ssize_t xstep,
    Py_ssize_t ystart, Py_ssize_t ystop, Py_ssize_t ystep);

/* C API interfaces */
static PyObject* PyPixelArray_New(PyObject *surfobj);

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
    { "compare", (PyCFunction)_compare, METH_VARARGS | METH_KEYWORDS,
      DOC_PIXELARRAYCOMPARE },
    { "extract", (PyCFunction)_extract_color, METH_VARARGS | METH_KEYWORDS,
      DOC_PIXELARRAYEXTRACT },
    { "make_surface", (PyCFunction)_make_surface, METH_NOARGS,
      DOC_PIXELARRAYMAKESURFACE },
    { "replace", (PyCFunction)_replace_color, METH_VARARGS | METH_KEYWORDS,
      DOC_PIXELARRAYREPLACE },
    { NULL, NULL, 0, NULL }
};

#if PY3
static void
Text_ConcatAndDel(PyObject **string, PyObject *newpart)
{
    PyObject *result = 0;
    if (*string && newpart) {
        PyUnicode_Concat(*string, newpart);
        Py_DECREF(*string);
        Py_DECREF(newpart);
    }
    else {
        Py_XDECREF(*string);
        Py_XDECREF(newpart);
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
    { "__dict__", (getter)_pxarray_get_dict, 0, 0, 0 },
    { "surface", (getter)_pxarray_get_surface, 0, DOC_PIXELARRAYSURFACE, 0 }, 
    { "itemsize", (getter)_pxarray_get_itemsize, 0, DOC_PIXELARRAYITEMSIZE, 0 },
    { "shape", (getter)_pxarray_get_shape, 0, DOC_PIXELARRAYSHAPE, 0 },
    { "strides", (getter)_pxarray_get_strides, 0, DOC_PIXELARRAYSTRIDES, 0 },
    { "ndim", (getter)_pxarray_get_ndim, 0, DOC_PIXELARRAYNDIM, 0 },
    { "__array_struct__", (getter)_pxarray_get_arraystruct, 0, "Version 3", 0 },
    { "_pixels_address", (getter)_pxarray_get_pixelsaddress, 0,
          "pixel buffer address (readonly)", 0 },
    { 0, 0, 0, 0, 0 }
};

/**
 * Sequence interface support for the PyPixelArray.
 * concat and repeat are not implemented due to the possible confusion
 * of their behaviour (see lists numpy array).
 */
static PySequenceMethods _pxarray_sequence =
{
    (lenfunc)_pxarray_length,              /*sq_length*/
    0,                                     /*sq_concat*/
    0,                                     /*sq_repeat*/
    (ssizeargfunc)_pxarray_item,           /*sq_item*/
    0,                                     /*reserved*/
    (ssizeobjargproc)_pxarray_ass_item,    /*sq_ass_item*/
    0,                                     /*reserved*/
    (objobjproc)_pxarray_contains,         /*sq_contains*/
    0,                                     /*sq_inplace_concat*/
    0                                      /*sq_inplace_repeat*/
};

/**
 * Mapping interface support for the PyPixelArray.
 */
static PyMappingMethods _pxarray_mapping =
{
    (inquiry)_pxarray_length,              /*mp_length*/
    (binaryfunc)_pxarray_subscript,        /*mp_subscript*/
    (objobjargproc)_pxarray_ass_subscript, /*mp_ass_subscript*/
};

static PyTypeObject PyPixelArray_Type =
{
    TYPE_HEAD (NULL, 0)
    "pygame.PixelArray",        /* tp_name */
    sizeof (PyPixelArray),      /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor)_pxarray_dealloc, /* tp_dealloc */
    0,                          /* tp_print */
    0,                          /* tp_getattr */
    0,                          /* tp_setattr */
    0,                          /* tp_compare */
    (reprfunc)_pxarray_repr,    /* tp_repr */
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
    offsetof(PyPixelArray, weakrefs),  /* tp_weaklistoffset */
    PySeqIter_New,              /* tp_iter */
    0,                          /* tp_iternext */
    _pxarray_methods,           /* tp_methods */
    0,                          /* tp_members */
    _pxarray_getsets,           /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    offsetof(PyPixelArray, dict), /* tp_dictoffset */
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

static PyPixelArray *
_pxarray_new_internal(PyTypeObject *type,
                      PyObject *surface, PyPixelArray *parent, Uint8 *pixels,
                      Py_ssize_t dim0, Py_ssize_t dim1,
                      Py_ssize_t stride0, Py_ssize_t stride1)
{
    PyPixelArray *self;

    self = (PyPixelArray *)type->tp_alloc(type, 0);
    if (!self) {
        return 0;
    }

    if (!parent) {
        if (!surface) {
            Py_TYPE(self)->tp_free((PyObject *)self);
            PyErr_SetString(PyExc_SystemError,
                            "Pygame internal error in _pxarray_new_internal: "
                            "no parent or surface.");
            return 0;
        }
        self->parent = 0;
        self->surface = surface;
        Py_INCREF(surface);
        self->lock = PySurface_LockLifetime(surface, (PyObject *)self);
        if (!self->lock) {
            Py_DECREF(surface);
            Py_TYPE(self)->tp_free((PyObject *)self);
            return 0;
        }
    }
    else {
        self->parent = parent;
        Py_INCREF(parent);
        surface = parent->surface;
        self->surface = surface;
        Py_INCREF(surface);
        self->lock = parent->lock;
        Py_INCREF(self->lock);
    }
    self->dim0 = dim0;
    self->dim1 = dim1;
    self->stride0 = stride0;
    self->stride1 = stride1;
    self->pixels = pixels;
    self->weakrefs = 0;
    self->dict = 0;

    return self;
}

/**
 * Creates a new PyPixelArray.
 */
static PyObject *
_pxarray_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyObject *surfobj;
    SDL_Surface *surf;
    Py_ssize_t dim0;
    Py_ssize_t dim1;
    Py_ssize_t stride0;
    Py_ssize_t stride1;
    Uint8 *pixels;

    if (!PyArg_ParseTuple(args, "O!", &PySurface_Type, &surfobj)) {
        return 0;
    }

    surf = PySurface_AsSurface(surfobj);
    dim0 = (Py_ssize_t)surf->w;
    dim1 = (Py_ssize_t)surf->h;
    stride0 = (Py_ssize_t)surf->format->BytesPerPixel;
    stride1 = (Py_ssize_t)surf->pitch;
    pixels = surf->pixels;
    if (stride0 < 1  || stride0 > 4) {
        return RAISE(PyExc_ValueError,
                     "unsupport bit depth for reference array");
    }

    return (PyObject *)_pxarray_new_internal(type, surfobj, 0, pixels,
                                             dim0, dim1, stride0, stride1);
}

/**
 * Deallocates the PyPixelArray and its members.
 */
static void
_pxarray_dealloc(PyPixelArray *self)
{
    if (self->weakrefs) {
        PyObject_ClearWeakRefs ((PyObject *) self);
    }
    Py_XDECREF(self->lock);
    Py_XDECREF(self->parent);
    Py_XDECREF(self->dict);
    Py_DECREF(self->surface);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/**** Getter and setter access ****/

/**
 * Getter for PixelArray.__dict__.
 */
static PyObject *
_pxarray_get_dict(PyPixelArray *self, void *closure)
{
    if (!self->dict) {
	self->dict = PyDict_New();
        if (!self->dict) {
            return 0;
        }
    }

    Py_INCREF(self->dict);
    return self->dict;
}

/**
 * Getter for PixelArray.surface
 */
static PyObject*
_pxarray_get_surface(PyPixelArray *self, void *closure)
{
    Py_INCREF(self->surface);
    return self->surface;
}

/**
 * Getter for PixelArray.itemsize
 * (pixel size in bytes)
 */
static PyObject *
_pxarray_get_itemsize(PyPixelArray *self, void *closure)
{
    SDL_Surface *surf = PySurface_AsSurface(self->surface);

    return PyInt_FromLong((long)surf->format->BytesPerPixel);
}

/**
 * Getter for PixelArray.shape
 */
static PyObject *
_pxarray_get_shape(PyPixelArray *self, void *closure)
{
    if (self->dim1) {
        return Py_BuildValue("(ll)", (long)self->dim0, (long)self->dim1);
    }
    return Py_BuildValue("(l)", (long)self->dim0);
}

/**
 * Getter for PixelArray.strides
 */
static PyObject *
_pxarray_get_strides(PyPixelArray *self, void *closure)
{
    if (self->dim1) {
        return Py_BuildValue("(ll)", (long)self->stride0, (long)self->stride1);
    }
    return Py_BuildValue("(l)", (long)self->stride0);
}

/**
 * Getter for PixelArray.ndim
 */
static PyObject *
_pxarray_get_ndim(PyPixelArray *self, void *closure)
{
    return PyInt_FromLong(self->dim1 ? 2L : 1L);
}

/**
 * Getter for PixelArray.__array_struct__
 * (array struct interface)
 */
static PyObject *
_pxarray_get_arraystruct(PyPixelArray *self, void *closure)
{
    typedef struct {
        PyArrayInterface inter;
        Py_intptr_t shape_mem[2];
        Py_intptr_t strides_mem[2];
    } Interface;
    int itemsize =
        (int)PySurface_AsSurface(self->surface)->format->BytesPerPixel;
    Interface *allocation;
    PyArrayInterface *inter;
    PyObject *capsule;

    allocation = PyMem_New(Interface, 1);
    if (!allocation) {
        return PyErr_NoMemory();
    }
    inter = (PyArrayInterface *)allocation;
    inter->shape = allocation->shape_mem;
    inter->strides = allocation->strides_mem;
#if PY3
    capsule = PyCapsule_New(inter, 0, _pxarray_capsule_destr);
#else
    capsule = PyCObject_FromVoidPtr(inter, PyMem_Free);
#endif
    if (!capsule) {
        PyMem_Free(inter);
        return 0;
    }
    inter->two = 2;
    inter->typekind = 'u';
    inter->itemsize = itemsize;
    inter->flags = PAI_ALIGNED | PAI_NOTSWAPPED | PAI_WRITEABLE | PAI_FORTRAN;
    inter->descr = 0;
    inter->nd = self->dim1 ? 2 : 1;
    inter->shape[0] = (Py_intptr_t)self->dim0;
    inter->shape[1] = (Py_intptr_t)self->dim1;
    inter->strides[0] = (Py_intptr_t)self->stride0;
    inter->strides[1] = (Py_intptr_t)self->stride1;
    if (self->stride0 == itemsize &&
        (!self->dim1 || self->stride1 == self->dim0 * itemsize)) {
        inter->flags |= PAI_CONTIGUOUS;
    }
    inter->data = self->pixels;
    return capsule;
}

#if PY3
static void
_pxarray_capsule_destr (PyObject *capsule)
{
    PyMem_Free (PyCapsule_GetPointer (capsule, 0));
}
#endif

/**
 * Getter for PixelArray._pixels_address
 * (address of the array's pointer into its surface's pixel data)
 */
static PyObject *
_pxarray_get_pixelsaddress(PyPixelArray *self, void *closure)
{
    void *address = self->pixels;

#if SIZEOF_VOID_P > SIZEOF_LONG
    return PyLong_FromUnsignedLongLong((unsigned PY_LONG_LONG)address);
#else
    return PyLong_FromUnsignedLong((unsigned long)address);
#endif
}


/**** Methods ****/

/**
 * repr(PixelArray)
 */
static PyObject *
_pxarray_repr(PyPixelArray *array)
{
    SDL_Surface *surf = PySurface_AsSurface(array->surface);
    PyObject *string;
    int bpp;
    Uint8 *pixels = array->pixels;
    int ndim = array->dim1 ? 2 : 1;
    Py_ssize_t dim0 = array->dim0;
    Py_ssize_t dim1 = array->dim1 ? array->dim1 : 1;
    Py_ssize_t stride0 = array->stride0;
    Py_ssize_t stride1 = array->stride1;
    Uint32 pixel;
    Py_ssize_t x;
    Py_ssize_t y;
    Uint8 *pixelrow;
    Uint8 *pixel_p;

    bpp = surf->format->BytesPerPixel;

    string = Text_FromUTF8 ("PixelArray(");
    if (!string) {
        return 0;
    }

    pixelrow = pixels;
    if (ndim == 2) {
        Text_ConcatAndDel(&string, Text_FromUTF8("["));
        if (!string) {
            return 0;
        }
    }

    switch (bpp) {

    case 1:
        for (y = 0; y < dim1; ++y) {
            Text_ConcatAndDel(&string, Text_FromUTF8("\n  ["));
            if (!string) {
                return 0;
            }
            pixel_p = pixelrow;
            for (x = 0; x < dim0 - 1; ++x) {
                Text_ConcatAndDel(&string,
                                  Text_FromFormat("%ld, ", (long)*pixel_p));
                if (!string) {
                    return 0;
                }
                pixel_p += stride0;
            }
            Text_ConcatAndDel(&string,
                              Text_FromFormat("%ld]", (long)*pixel_p));
            if (!string) {
                return 0;
            }
            pixelrow += stride1;
        }
        break;
    case 2:
        for (y = 0; y < dim1; ++y) {
            Text_ConcatAndDel(&string, Text_FromUTF8("\n  ["));
            if (!string) {
                return 0;
            }
            pixel_p = pixelrow;
            for (x = 0; x < dim0 - 1; ++x) {
                Text_ConcatAndDel(&string,
                                  Text_FromFormat("%ld, ",
                                                  (long)*(Uint16 *)pixel_p));
                if (!string) {
                    return 0;
                }
                pixel_p += stride0;
            }
            Text_ConcatAndDel(&string,
                              Text_FromFormat("%ld]",
                                              (long)*(Uint16 *)pixel_p));
            if (string == NULL) {
                return NULL;
            }
            pixelrow += stride1;
        }
        break;
    case 3:
        for (y = 0; y < dim1; ++y) {
            Text_ConcatAndDel(&string, Text_FromUTF8("\n  ["));
            pixel_p = pixelrow;
            for (x = 0; x < dim0 - 1; ++x) {
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                pixel = (pixel_p[0]) + (pixel_p[1] << 8) + (pixel_p[2] << 16);
#else
                pixel = (pixel_p[2]) + (pixel_p[1] << 8) + (pixel_p[0] << 16);
#endif
                Text_ConcatAndDel(&string,
                                  Text_FromFormat("%ld, ", (long)pixel));
                if (!string) {
                    return 0;
                }
                pixel_p += stride0;
            }
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
            pixel = (pixel_p[0]) + (pixel_p[1] << 8) + (pixel_p[2] << 16);
#else
            pixel = (pixel_p[2]) + (pixel_p[1] << 8) + (pixel_p[0] << 16);
#endif
            Text_ConcatAndDel(&string,
                              Text_FromFormat("%ld]", (long)pixel));
            if (!string) {
                return 0;
            }
            pixelrow += stride1;
        }
        break;
    default: /* case 4: */
        for (y = 0; y < dim1; ++y) {
            Text_ConcatAndDel(&string, Text_FromUTF8("\n  ["));
            if (!string) {
                return 0;
            }
            pixel_p = pixelrow;
            for (x = 0; x < dim0 - 1; ++x) {
                Text_ConcatAndDel(&string,
                                  Text_FromFormat("%ld, ",
                                                  (long)*(Uint32 *)pixel_p));
                if (!string)
                {
                    return 0;
                }
                pixel_p += stride0;
            }
            Text_ConcatAndDel(&string,
                              Text_FromFormat("%ld]",
                                              (long)*(Uint32 *)pixel_p));
            if (string == NULL)
            {
                return NULL;
            }
            pixelrow += stride1;
        }
        break;
    }

    if (ndim == 2) {
        Text_ConcatAndDel(&string, Text_FromUTF8 ("]\n)"));
    }
    else {
        Text_ConcatAndDel (&string, Text_FromUTF8 ("\n)"));
    }
    return string;
}

static PyObject *
_pxarray_subscript_internal(PyPixelArray *array,
                            Py_ssize_t xstart,
			    Py_ssize_t xstop,
			    Py_ssize_t xstep,
			    Py_ssize_t ystart,
			    Py_ssize_t ystop,
			    Py_ssize_t ystep)
{
    /* Special case: if xstep or ystep are zero, then the corresponding
     * dimension is removed. If both are zero, then a single integer
     * pixel value is returned.
     */
    Py_ssize_t dim0;
    Py_ssize_t dim1;
    Py_ssize_t stride0;
    Py_ssize_t stride1;
    Uint8 *pixels;
    Py_ssize_t absxstep = ABS(xstep);
    Py_ssize_t absystep = ABS(ystep);
    Py_ssize_t dx = xstop - xstart;
    Py_ssize_t dy = ystop - ystart;

    if (!array->dim1) {
        ystart = 0;
        ystop = 1;
        ystep = 0;
    }
    if (!(xstep || ystep)) {
        return _get_single_pixel(array, xstart, ystart);
    }
    if (xstep) {
        dim0 = (ABS(dx) + absxstep - 1) / absxstep;
        stride0 = array->stride0 * xstep;
    }
    else {
        /* Move dimension 2 into 1 */
        dim0 = (ABS(dy) + absystep - 1) / absystep;
        stride0 = array->stride1 * ystep;
    }
    if (xstep && ystep) {
        dim1 = (ABS(dy) + absystep - 1) / absystep;
        stride1 = array->stride1 * ystep;
    }
    else {
        /* Only a one dimensional array */
        dim1 = 0;
        stride1 = 0;
    }
    pixels = array->pixels + xstart * array->stride0 + ystart * array->stride1;
    return (PyObject *)_pxarray_new_internal(&PyPixelArray_Type, 0,
					     array, pixels,
					     dim0, dim1, stride0, stride1);
}

/**
 * Creates a 2D slice of the array.
 */
static PyObject *
_array_slice_internal(PyPixelArray *array,
                      Py_ssize_t start, Py_ssize_t end, Py_ssize_t step)
{
    if (end == start) {
        return RAISE(PyExc_IndexError, "array size must not be 0");
    }

    if (start >= array->dim0) {
        return RAISE(PyExc_IndexError, "array index out of range");
    }
    return _pxarray_subscript_internal(array,
                                       start, end, step,
                                       0, array->dim1, 1);
}


/**** Sequence interfaces ****/

/**
 * len (array)
 */
static Py_ssize_t
_pxarray_length(PyPixelArray *array)
{
    return array->dim0;
}

/**
 * array[x]
 */
static PyObject *
_pxarray_item(PyPixelArray *array, Py_ssize_t index)
{
    if (index < 0) {
        index = array->dim0 - index;
        if (index < 0) {
            return RAISE(PyExc_IndexError, "array index out of range");
        }
    }
    if (index >= array->dim0) {
        return RAISE(PyExc_IndexError, "array index out of range");
    }
    return _pxarray_subscript_internal(array, index, 0, 0, 0, array->dim1, 1);
}

static int
_array_assign_array(PyPixelArray *array,
                    Py_ssize_t low, Py_ssize_t high,
                    PyPixelArray *val)
{
    SDL_Surface *surf = PySurface_AsSurface(array->surface);
    Py_ssize_t dim0 = ABS(high - low);
    Py_ssize_t dim1 = array->dim1;
    Py_ssize_t stride0 = high >= low ? array->stride0 : -array->stride0;
    Py_ssize_t stride1 = array->stride1;
    Uint8 *pixels = array->pixels + low * array->stride0;
    int bpp;
    Py_ssize_t val_dim0 = val->dim0;
    Py_ssize_t val_dim1 = val->dim1;
    Py_ssize_t val_stride0 = val->stride0;
    Py_ssize_t val_stride1 = val->stride1;
    Uint8 *val_pixels = val->pixels;
    SDL_Surface *val_surf = PySurface_AsSurface(val->surface);
    int val_bpp;
    Uint8 *pixelrow;
    Uint8 *pixel_p;
    Uint8 *val_pixelrow;
    Uint8 *val_pixel_p;
    Uint8 *copied_pixels = 0;
    Py_ssize_t x;
    Py_ssize_t y;

    if (!((val_dim1 && dim0 == val_dim0 && dim1 == val_dim1) ||
	  (dim1 && dim1 == val_dim0)                         ||
	  (dim0 == val_dim0)                                    )) {
         /* Bounds do not match. */
        PyErr_SetString(PyExc_ValueError, "array sizes do not match");
        return -1;
    }
    bpp = surf->format->BytesPerPixel;
    val_bpp = val_surf->format->BytesPerPixel;
    if (val_bpp != bpp) {
        /* bpp do not match. We cannot guarantee that the padding and co
         * would be set correctly. */
        PyErr_SetString(PyExc_ValueError, "bit depths do not match");
        return -1;
    }

    /* If we reassign the same array, we need to copy the pixels
     * first. */
    if (SURFACE_EQUALS(array, val)) {
        /* We assign a different view or so. Copy the source buffer. */
        size_t size = val_surf->h * val_surf->pitch;
        int val_offset = val_pixels - (Uint8 *)val_surf->pixels;

        copied_pixels = (Uint8 *)malloc(size);
        if (!copied_pixels) {
            PyErr_NoMemory();
            return -1;
        }
        val_pixels = memcpy(copied_pixels, val_surf->pixels, size) + val_offset;
    }
                     
    if (!dim1) {
        dim1 = 1;
    }
    pixelrow = pixels;
    val_pixelrow = val_pixels;

    switch (bpp) {

    case 1:
        for (y = 0; y < dim1; ++y) {
            pixel_p = pixelrow;
            val_pixel_p = val_pixelrow;
            for (x = 0; x < dim0; ++x) {
                *pixel_p = *val_pixel_p;
                pixel_p += stride0;
                val_pixel_p += val_stride0;
            }
            pixelrow += stride1;
            val_pixelrow += val_stride1;
        }
        break;
    case 2:
        for (y = 0; y < dim1; ++y) {
            pixel_p = pixelrow;
            val_pixel_p = val_pixelrow;
            for (x = 0; x < dim0; ++x) {
                *((Uint16 *)pixel_p) = *((Uint16 *)val_pixel_p);
                pixel_p += stride0;
                val_pixel_p += val_stride0;
            }
            pixelrow += stride1;
            val_pixelrow += val_stride1;
        }
        break;
    case 3:
    {
#if (SDL_BYTEORDER == SDL_LIL_ENDIAN)
        Uint32 Roffset = surf->format->Rshift >> 3;
        Uint32 Goffset = surf->format->Gshift >> 3;
        Uint32 Boffset = surf->format->Bshift >> 3;
        Uint32 vRoffset = val_surf->format->Rshift >> 3;
        Uint32 vGoffset = val_surf->format->Gshift >> 3;
        Uint32 vBoffset = val_surf->format->Bshift >> 3;
#else
        Uint32 Roffset = 2 - surf->format->Rshift >> 3;
        Uint32 Goffset = 2 - surf->format->Gshift >> 3;
        Uint32 Boffset = 2 - surf->format->Bshift >> 3;
        Uint32 vRoffset = 2 - val_surf->format->Rshift >> 3;
        Uint32 vGoffset = 2 - val_surf->format->Gshift >> 3;
        Uint32 vBoffset = 2 - val_surf->format->Bshift >> 3;
#endif
        for (y = 0; y < dim1; ++y) {
            pixel_p = pixelrow;
            val_pixel_p = val_pixelrow;
            for (x = 0; x < dim0; ++x) {
                pixel_p[Roffset] = val_pixel_p[vRoffset];
                pixel_p[Goffset] = val_pixel_p[vGoffset];
                pixel_p[Boffset] = val_pixel_p[vBoffset];
                pixel_p += stride0;
                val_pixel_p += val_stride0;
            }
            pixelrow += stride1;
            val_pixelrow += val_stride1;
        }
    }
        break;
    default: /* case 4: */
        for (y = 0; y < dim1; ++y) {
            pixel_p = pixelrow;
            val_pixel_p = val_pixelrow;
            for (x = 0; x < dim0; ++x) {
                *((Uint32 *)pixel_p) = *((Uint32 *)val_pixel_p);
                pixel_p += stride0;
                val_pixel_p += val_stride0;
            }
            pixelrow += stride1;
            val_pixelrow += val_stride1;
        }
        break;
    }

    if (copied_pixels) {
        free(copied_pixels);
    }
    return 0;
}

static int
_array_assign_sequence(PyPixelArray *array, Py_ssize_t low, Py_ssize_t high,
                       PyObject *val)
{ 
    SDL_Surface *surf = PySurface_AsSurface(array->surface);
    SDL_PixelFormat *format;
    Py_ssize_t dim0 = ABS (high - low);
    Py_ssize_t dim1 = array->dim1;
    Py_ssize_t stride0 = high >= low ? array->stride0 : -array->stride0;
    Py_ssize_t stride1 = array->stride1;
    Uint8 *pixels = array->pixels + low * array->stride0;
    int bpp;
    Py_ssize_t val_dim0 = PySequence_Size(val);
    Uint32 *val_colors;
    Uint8 *pixelrow;
    Uint8 *pixel_p;
    Uint32 *val_color_p;
    Py_ssize_t x;
    Py_ssize_t y;

    if (val_dim0 != dim0) {
        PyErr_SetString(PyExc_ValueError, "sequence size mismatch");
        return -1;
    }

    format = surf->format;
    bpp = format->BytesPerPixel;

    if (!dim1) {
        dim1 = 1;
    }

    /* Copy the values. */
    val_colors = malloc(sizeof(Uint32) * val_dim0);
    if (!val_colors) {
        PyErr_NoMemory();
        return -1;
    }
    for (x = 0; x < val_dim0; ++x) {
        if (!_get_color_from_object(PySequence_Fast_GET_ITEM(val, x),
                                    format, (val_colors + x))) {
            free (val_colors);
            return -1;
        }
    }

    pixelrow = pixels;

    Py_BEGIN_ALLOW_THREADS;
    switch (bpp) {

    case 1:
        for (y = 0; y < dim1; ++y) {
            pixel_p = pixelrow;
            val_color_p = val_colors;
            for (x = 0; x < dim0; ++x) {
                *pixel_p = (Uint8)*val_color_p;
                pixel_p += stride0;
                ++val_color_p;
            }
            pixelrow += stride1;
        }
        break;
    case 2:
        for (y = 0; y < dim1; ++y) {
            pixel_p = pixelrow;
            val_color_p = val_colors;
            for (x = 0; x < dim0; ++x) {
                *((Uint16 *)pixel_p) = (Uint16)*val_color_p;
                pixel_p += stride0;
                ++val_color_p;
            }
            pixelrow += stride1;
        }
        break;
    case 3:
    {
#if (SDL_BYTEORDER == SDL_LIL_ENDIAN)
        Uint32 Roffset = surf->format->Rshift >> 3;
        Uint32 Goffset = surf->format->Gshift >> 3;
        Uint32 Boffset = surf->format->Bshift >> 3;
#else
        Uint32 Roffset = 2 - surf->format->Rshift >> 3;
        Uint32 Goffset = 2 - surf->format->Gshift >> 3;
        Uint32 Boffset = 2 - surf->format->Bshift >> 3;
#endif
        for (y = 0; y < dim1; ++y) {
            pixel_p = pixelrow;
            val_color_p = val_colors;
            for (x = 0; x < dim0; ++x) {
                pixel_p[Roffset] = (Uint8)(*val_color_p >> 16);
                pixel_p[Goffset] = (Uint8)(*val_color_p >> 8);
                pixel_p[Boffset] = (Uint8)*val_color_p;
                pixel_p += stride0;
                ++val_color_p;
            }
            pixelrow += stride1;
        }
    }
        break;
    default: /* case 4: */
        for (y = 0; y < dim1; ++y) {
            pixel_p = pixelrow;
            val_color_p = val_colors;
            for (x = 0; x < dim0; ++x) {
                *((Uint32 *)pixel_p) = *val_color_p;
                pixel_p += stride0;
                ++val_color_p;
            }
            pixelrow += stride1;
        }
        break;
    }
    Py_END_ALLOW_THREADS;

    free(val_colors);
    return 0;
}

static int
_array_assign_slice(PyPixelArray *array, Py_ssize_t low, Py_ssize_t high,
                    Uint32 color)
{
    SDL_Surface *surf = PySurface_AsSurface(array->surface);
    Py_ssize_t dim0 = ABS (high - low);
    Py_ssize_t dim1 = array->dim1;
    Py_ssize_t stride0 = high >= low ? array->stride0 : -array->stride0;
    Py_ssize_t stride1 = array->stride1;
    Uint8 *pixels = array->pixels + low * array->stride0;
    int bpp;
    Uint8 *pixelrow;
    Uint8 *pixel_p;
    Py_ssize_t x;
    Py_ssize_t y;

    bpp = surf->format->BytesPerPixel;

    if (!dim1) {
        dim1 = 1;
    }
    pixelrow = pixels;

    Py_BEGIN_ALLOW_THREADS;
    switch (bpp) {

    case 1:
    {
        Uint8 c = (Uint8)color;

        for (y = 0; y < dim1; ++y) {
            pixel_p = pixelrow;
            for (x = 0; x < dim0; ++x) {
                *pixel_p = c;
                pixel_p += stride0;
            }
            pixelrow += stride1;
        }
    }
        break;
    case 2:
    {
        Uint16 c = (Uint16)color;

        for (y = 0; y < dim1; ++y) {
            pixel_p = pixelrow;
            for (x = 0; x < dim0; ++x) {
                *((Uint16 *)pixel_p) = c;
                pixel_p += stride0;
            }
            pixelrow += stride1;
        }
    }
        break;
    case 3:
    {
#if (SDL_BYTEORDER == SDL_LIL_ENDIAN)
        Uint32 Roffset = surf->format->Rshift >> 3;
        Uint32 Goffset = surf->format->Gshift >> 3;
        Uint32 Boffset = surf->format->Bshift >> 3;
#else
        Uint32 Roffset = 2 - surf->format->Rshift >> 3;
        Uint32 Goffset = 2 - surf->format->Gshift >> 3;
        Uint32 Boffset = 2 - surf->format->Bshift >> 3;
#endif
        Uint8 r = (Uint8)(color >> 16);
        Uint8 g = (Uint8)(color >> 8);
        Uint8 b = (Uint8)(color);

        for (y = 0; y < dim1; ++y) {
            pixel_p = pixelrow;
            for (x = 0; x < dim0; ++x) {
                pixel_p[Roffset] = r;
                pixel_p[Goffset] = g;
                pixel_p[Boffset] = b;
                pixel_p += stride0;
            }
            pixelrow += stride1;
        }
    }
        break;
    default: /* case 4: */
        for (y = 0; y < dim1; ++y) {
            pixel_p = pixelrow;
            for (x = 0; x < dim0; ++x) {
                *((Uint32 *)pixel_p) = color;
                pixel_p += stride0;
            }
            pixelrow += stride1;
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
_pxarray_ass_item(PyPixelArray *array, Py_ssize_t index, PyObject *value)
{
    SDL_Surface *surf = PySurface_AsSurface(array->surface);
    Py_ssize_t y = 0;
    int bpp;
    Uint8 *pixels = array->pixels;
    Uint8 *pixel_p;
    Uint32 color = 0;
    Py_ssize_t dim0 = array->dim0;
    Py_ssize_t dim1 = array->dim1;
    Py_ssize_t stride0 = array->stride0;
    Py_ssize_t stride1 = array->stride1;
    PyPixelArray *tmparray = 0;
    int retval;

    bpp = surf->format->BytesPerPixel;

    if (!_get_color_from_object(value, surf->format, &color)) {
        if (PyPixelArray_Check(value)) {
            PyErr_Clear(); /* _get_color_from_object */
            return _array_assign_array(array, index, index + 1,
                                       (PyPixelArray *)value);
        }
        else if (PySequence_Check(value)) {
            PyErr_Clear(); /* _get_color_from_object */
            tmparray = (PyPixelArray *)
                _pxarray_subscript_internal(array,
                                            index, 0, 0,
                                            0, array->dim1, 1);
            if (!tmparray) {
                return 0;
            }
            retval = _array_assign_sequence(tmparray,
                                            0, tmparray->dim0, value);
            Py_DECREF(tmparray);
            return retval;
        }
        else { /* Error already set by _get_color_from_object(). */
            return -1;
        }
    }

    if (index < 0) {
        index += dim0;
        if (index < 0) {
            PyErr_SetString(PyExc_IndexError, "array index out of range");
            return -1;
        }
    }
    if (index >= dim0) {
        PyErr_SetString(PyExc_IndexError, "array index out of range");
    }
    pixels += index * stride0;

    pixel_p = pixels;
    if (!dim1) {
        dim1 = 1;
    }
    
    Py_BEGIN_ALLOW_THREADS;
    /* Single value assignment. */
    switch (bpp) {

    case 1:
        for (y = 0; y < dim1; ++y) {
            *((Uint8 *)pixel_p) = (Uint8)color;
            pixel_p += stride1;
        }
        break;
    case 2:
        for (y = 0; y < dim1; ++y) {
            *((Uint16 *)pixel_p) = (Uint16)color;
            pixel_p += stride1;
        }
        break;
    case 3:
    {
#if (SDL_BYTEORDER == SDL_LIL_ENDIAN)
        Uint32 Roffset = surf->format->Rshift >> 3;
        Uint32 Goffset = surf->format->Gshift >> 3;
        Uint32 Boffset = surf->format->Bshift >> 3;
#else
        Uint32 Roffset = 2 - (surf->format->Rshift >> 3);
        Uint32 Goffset = 2 - (surf->format->Gshift >> 3);
        Uint32 Boffset = 2 - (surf->format->Bshift >> 3);
#endif
        for (y = 0; y < dim1; ++y) {
            pixel_p[Roffset] = (Uint8)(color >> 16);
            pixel_p[Goffset] = (Uint8)(color >> 8);
            pixel_p[Boffset] = (Uint8)color;
            pixel_p += stride1;
        }
        break;
    }
    default: /* case 4: */
        for (y = 0; y < dim1; ++y) {
            *((Uint32 *)pixel_p) = color;
            pixel_p += stride1;
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
_pxarray_ass_slice(PyPixelArray *array, Py_ssize_t low, Py_ssize_t high,
                   PyObject *value)
{
    SDL_Surface *surf = PySurface_AsSurface(array->surface);
    Uint32 color;

    if (low < 0) {
        low = 0;
    }
    else if (low > (Sint32)array->dim0) {
        low = array->dim0;
    }
        
    if (high < low) {
        high = low;
    }
    else if (high > (Sint32)array->dim0) {
        high = array->dim0;
    }

    if (PyPixelArray_Check(value)) {
        return _array_assign_array(array, low, high, (PyPixelArray *)value);
    }
    if (_get_color_from_object(value, surf->format, &color)) {
        return _array_assign_slice(array, low, high, color);
    }
    PyErr_Clear(); /* In case _get_color_from_object set it */
    if (PySequence_Check(value)) {
        return _array_assign_sequence(array, low, high, value);
    }
    return 0;
}

/**
 * x in array
 */
static int
_pxarray_contains(PyPixelArray *array, PyObject *value)
{
    SDL_Surface *surf = PySurface_AsSurface(array->surface);
    Py_ssize_t dim0 = array->dim0;
    Py_ssize_t dim1 = array->dim1;
    Py_ssize_t stride0 = array->stride0;
    Py_ssize_t stride1 = array->stride1;
    Uint8 *pixels = array->pixels;
    int bpp;
    Uint32 color;
    Uint8 *pixelrow;
    Uint8 *pixel_p;
    Py_ssize_t x;
    Py_ssize_t y;
    int found = 0;

    bpp = surf->format->BytesPerPixel;

    if (!_get_color_from_object(value, surf->format, &color)) {
        return -1;
    }

    if (!dim1) {
        dim1 = 1;
    }
    pixelrow = pixels;

    Py_BEGIN_ALLOW_THREADS;
    switch (bpp) {

    case 1:
    {
        Uint8 c = (Uint8)color;

        for (y = 0; !found && y < dim1; ++y) {
            pixel_p = pixelrow;
            for (x = 0; !found && x < dim0; ++x) {
                found = *pixel_p == c ? 1 : 0;
                pixel_p += stride0;
            }
            pixelrow += stride1;
        }
    }
        break;
    case 2:
    {
        Uint16 c = (Uint16)color;

        for (y = 0; !found && y < dim1; ++y) {
            pixel_p = pixelrow;
            for (x = 0; !found && x < dim0; ++x) {
                found = *((Uint16 *)pixel_p) == c ? 1 : 0;
                pixel_p += stride0;
            }
            pixelrow += stride1;
        }
    }
        break;
    case 3:
        for (y = 0; !found && y < dim1; ++y) {
            pixel_p = pixelrow;
            for (x = 0; !found && x < dim0; ++x) {
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                found = (((Uint32)pixel_p[0]) +
                         ((Uint32)pixel_p[1] << 8) +
                         ((Uint32)pixel_p[2] << 16)  ) == color;
#else
                found = (((Uint32)pixel_p[2]) +
                         ((Uint32)pixel_p[1] << 8) +
                         ((Uint32)pixel_p[0] << 16)  ) == color;
#endif
                pixel_p += stride0;
            }
            pixelrow += stride1;
        }
        break;
    default: /* case 4: */
        for (y = 0; !found && y < dim1; ++y) {
            pixel_p = pixelrow;
            for (x = 0; !found && x < dim0; ++x) {
                found = *((Uint32 *)pixel_p) == color ? 1 : 0;
                pixel_p += stride0;
            }
            pixelrow += stride1;
        }
        break;
    }
    Py_END_ALLOW_THREADS;

    return found;
}


/**** Mapping interfaces ****/

/**
 * Internally used parser function for the 2D slices:
 * array[x,y], array[:,:], ...
 */
static int
_get_subslice(PyObject *op, Py_ssize_t length, Py_ssize_t *start,
              Py_ssize_t *stop, Py_ssize_t *step)
{
    /* Special case: return step as 0 for an integer op.
     */
    *start = -1;
    *stop = -1;
    *step = -1;

    if (PySlice_Check(op)) {
        Py_ssize_t slicelen;

        /* Operator is a slice: array[x::, */
        if (Slice_GET_INDICES_EX(op, length, start, stop, step, &slicelen)) {
            return -1;
        }
    }
    else if (PyInt_Check(op)) {
        /* Plain index: array[x, */
        *start = PyInt_AsLong(op);
        if (*start < 0) {
            *start += length;
        }
        if (*start >= length || *start < 0) {
            PyErr_SetString(PyExc_IndexError, "invalid index");
            return -1;
        }   
        *stop = (*start) + 1;
        *step = 0;
    }
    else if (PyLong_Check(op)) {
        long long val = -1;
        /* Plain index: array[x, */

        val = PyLong_AsLong(op);
        if ((val < INT_MIN) || (val > INT_MAX)) {
            PyErr_SetString(PyExc_ValueError,
                            "index too big for array access");
            return -1;
        }
        *start = (int) val;
        if (*start < 0) {
            *start += length;
        }
        if (*start >= length || *start < 0) {
            PyErr_SetString(PyExc_IndexError, "invalid index");
            return -1;
        }   
        *stop = (*start) + 1;
        *step = 0;
    }
    /* No errors. */
    return 0;
}

/**
 * Slicing support for 1D and 2D access.
 * array[x,y] is only supported for 2D arrays.
 */
static PyObject *
_pxarray_subscript(PyPixelArray *array, PyObject *op)
{
    Py_ssize_t dim0 = array->dim0;
    Py_ssize_t dim1 = array->dim1;

    /* Note: order matters here.
     * First check array[x,y], then array[x:y:z], then array[x]
     * Otherwise it'll fail.
     */
    if (PySequence_Check(op)) {
        PyObject *obj;
        Py_ssize_t size = PySequence_Size(op);
        Py_ssize_t xstart, xstop, xstep;
        Py_ssize_t ystart, ystop, ystep;
        
        if (size == 0) {
            /* array[,], array[()] ... */
            Py_INCREF(array);
            return (PyObject *)array;
        }
        if (size > 2 || (size == 2 && !dim1)) {
            return RAISE(PyExc_IndexError, "too many indices for the array");
        }

        obj = PySequence_Fast_GET_ITEM(op, 0);
        if (obj == Py_Ellipsis || obj == Py_None) {
            /* Operator is the ellipsis or None
             * array[...,XXX], array[None,XXX]
             */
            xstart = 0;
            xstop = dim0;
            xstep = 1;
        }
        else if (_get_subslice(obj, dim0, &xstart, &xstop, &xstep)) {
            /* Error on retrieving the subslice. */
            return 0;
        }

        if (size == 2) {
            obj = PySequence_Fast_GET_ITEM(op, 1);
            if (obj == Py_Ellipsis || obj == Py_None) {
                /* Operator is the ellipsis or None
                 * array[XXX,...], array[XXX,None]
                 */
                ystart = 0;
                ystop = dim1;
                ystep = 1;
            }
            else if (_get_subslice(obj, dim1, &ystart, &ystop, &ystep)) {
                /* Error on retrieving the subslice. */
                return 0;
            }
        }
        else {
            ystart = 0;
            ystop = dim1;
            ystep = 1;
        }

        /* Null value? */
        if (xstart == xstop || ystart == ystop) {
            Py_RETURN_NONE;
        }

        return _pxarray_subscript_internal(array,
					   xstart, xstop, xstep,
					   ystart, ystop, ystep);
    }
    else if (op == Py_Ellipsis) {
        Py_INCREF(array);
        return (PyObject *)array;
    }
    else if (PySlice_Check(op)) {
        /* A slice */
        Py_ssize_t slicelen;
        Py_ssize_t step;
        Py_ssize_t start;
        Py_ssize_t stop;

        if (Slice_GET_INDICES_EX(op, dim0, &start, &stop, &step, &slicelen)) {
            return 0;
        }
        if (slicelen < 0) {
            return RAISE(PyExc_IndexError, "Unable to handle negative slice");
        }
        if (slicelen == 0) {
            Py_RETURN_NONE;
        }
        return _pxarray_subscript_internal(array,
					   start, stop, step, 0, dim1, 1);
    }
    else if (PyIndex_Check(op) || PyInt_Check(op) || PyLong_Check(op)) {
        Py_ssize_t i;
#if PY_VERSION_HEX >= 0x02050000
        PyObject *val = PyNumber_Index (op);
        if (!val) {
            return 0;
        }
        /* A simple index. */
        i = PyNumber_AsSsize_t(val, PyExc_IndexError);
#else
        i = PyInt_Check(op) ? PyInt_AsLong(op) : PyLong_AsLong(op);
#endif
        if (i == -1 && PyErr_Occurred()) {
            return 0;
        }
	if (i < 0) {
            i += dim0;
        }  
        if (i < 0 || i >= dim0) {
            return RAISE(PyExc_IndexError, "array index out of range");
        }
       return _pxarray_subscript_internal(array, i, i + 1, 0, 0, dim1, 1);
    }

    return RAISE(PyExc_TypeError,
                 "index must be an integer, sequence or slice");
}

static int
_pxarray_ass_subscript(PyPixelArray *array, PyObject* op, PyObject* value)
{
    /* TODO: by time we can make this faster by avoiding the creation of
     * temporary subarrays.
     */
    Uint32 dim0 = array->dim0;
    Uint32 dim1 = array->dim1;
    
    /* Note: order matters here.
     * First check array[x,y], then array[x:y:z], then array[x]
     * Otherwise it'll fail.
     */
    if (PySequence_Check(op))
    {
        PyPixelArray *tmparray;
        PyObject *obj;
        Py_ssize_t size = PySequence_Size(op);
        Py_ssize_t xstart, xstop, xstep;
        Py_ssize_t ystart, ystop, ystep;
        int retval;

        if (size > 2 || (size == 2 && !dim1)) {
            PyErr_SetString(PyExc_IndexError, "too many indices for the array");
            return -1;
        }

        obj = PySequence_Fast_GET_ITEM(op, 0);
        if (obj == Py_Ellipsis || obj == Py_None) {
            /* Operator is the ellipsis or None
             * array[...,XXX], array[None,XXX]
             */
            xstart = 0;
            xstop = dim0;
            xstep = 1;
        }
        else if (_get_subslice(obj, dim0, &xstart, &xstop, &xstep)) {
            /* Error on retrieving the subslice. */
            return -1;
        }

        if (size == 2) {
            obj = PySequence_Fast_GET_ITEM(op, 1);
            if (obj == Py_Ellipsis || obj == Py_None) {
                /* Operator is the ellipsis or None
                 * array[XXX,...], array[XXX,None]
                 */
                ystart = 0;
                ystop = dim1;
                ystep = 1;
            }
            else if (_get_subslice(obj, dim1, &ystart, &ystop, &ystep)) {
                /* Error on retrieving the subslice. */
                return -1;
            }
        }
        else {
            ystart = 0;
            ystop = dim1;
            ystep = 1;
        }

        /* Null value? Do nothing then. */
        if (xstart == xstop || ystart == ystop) {
            return 0;
        }

        /* Single value? */
        if (ABS(xstop - xstart) == 1 && ABS(ystop - ystart) == 1) {
            tmparray = (PyPixelArray *)
                _pxarray_subscript_internal(array,
                                            xstart, xstart + 1, 1,
                                            ystart, ystart + 1, 1);
            if (!tmparray) {
                return -1;
            }
            retval = _pxarray_ass_item(tmparray, 0, value);
            Py_DECREF(tmparray);
            return retval;
        }
        tmparray = (PyPixelArray *)
            _pxarray_subscript_internal(array,
                                        xstart, xstop, xstep,
                                        ystart, ystop, ystep);
        if (!tmparray) {
            return -1;
        }

        retval = _pxarray_ass_slice(tmparray, 0,
                                    (Py_ssize_t)tmparray->dim0, value);
        Py_DECREF(tmparray);
        return retval;
    }
    else if (op == Py_Ellipsis) {
        /* A slice */
        PyPixelArray *tmparray =
             (PyPixelArray *) _pxarray_subscript_internal(array,
                                                          0, array->dim0, 1,
                                                          0, array->dim1, 1);
        int retval;

        if (!tmparray) {
            return -1;
        }

        retval = _pxarray_ass_slice(tmparray, 0,
                                    (Py_ssize_t)tmparray->dim0, value);
        Py_DECREF(tmparray);
        return retval;
    }
    else if (PySlice_Check(op)) {
        /* A slice */
        PyPixelArray *tmparray;
        Py_ssize_t slicelen;
        Py_ssize_t step;
        Py_ssize_t start;
        Py_ssize_t stop;
        int retval;

        if (Slice_GET_INDICES_EX(op, array->dim0,
                                 &start, &stop, &step, &slicelen)) {
            return -1;
        }
        if (slicelen < 0) {
            PyErr_SetString(PyExc_IndexError,
                            "Unable to handle negative slice");
            return -1;
        }
        if (slicelen == 0) {
            return 0;
        }
        tmparray = (PyPixelArray *)_array_slice_internal(array,
                                                         start, stop, step);
        if (!tmparray) {
            return -1;
        }
        retval = _pxarray_ass_slice(tmparray,
                                    0, (Py_ssize_t)tmparray->dim0, value);
        Py_DECREF(tmparray);
        return retval;
    }
    else if (PyIndex_Check(op) || PyInt_Check(op) || PyLong_Check(op)) {
        Py_ssize_t i;
#if PY_VERSION_HEX >= 0x02050000
        PyObject *val = PyNumber_Index(op);
        if (!val) {
            return -1;
        }
        /* A simple index. */
        i = PyNumber_AsSsize_t(val, PyExc_IndexError);
#else
        i = PyInt_Check(op) ? PyInt_AsLong (op) : PyLong_AsLong (op);
#endif 
        if (i == -1 && PyErr_Occurred()) {
            return -1;
        }
        return _pxarray_ass_item(array, i, value);
    }

    PyErr_SetString(PyExc_TypeError,
                    "index must be an integer, sequence or slice");
    return -1;
}


/**** C API interfaces ****/
static PyObject* PyPixelArray_New(PyObject *surfobj)
{
    SDL_Surface *surf;
    Py_ssize_t dim0;
    Py_ssize_t dim1;
    Py_ssize_t stride0;
    Py_ssize_t stride1;
    Uint8 *pixels;

    if (!PySurface_Check(surfobj)) {
        return RAISE(PyExc_TypeError, "argument is no a Surface");
    }

    surf = PySurface_AsSurface(surfobj);
    dim0 = (Py_ssize_t)surf->w;
    dim1 = (Py_ssize_t)surf->h;
    stride0 = (Py_ssize_t)surf->format->BytesPerPixel;
    stride1 = (Py_ssize_t)surf->pitch;
    pixels = surf->pixels;
    if (stride0 < 1  || stride0 > 4) {
        return RAISE(PyExc_ValueError,
                     "unsupported bit depth for reference array");
    }

    return (PyObject *)_pxarray_new_internal(&PyPixelArray_Type, surfobj, 0,
                                             pixels,
                                             dim0, dim1, stride0, stride1);
}


MODINIT_DEFINE(pixelarray)
{
    PyObject *module;
    PyObject *apiobj;
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
    import_pygame_base();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }
    import_pygame_color();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }
    import_pygame_surface();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }

    /* type preparation */
    if (PyType_Ready(&PyPixelArray_Type)) {
        MODINIT_ERROR;
    }
    
    /* create the module */
#if PY3
    module = PyModule_Create(&_module);
#else
    module = Py_InitModule3(MODPREFIX "pixelarray", 0, 0);
#endif
    if (!module) {
        MODINIT_ERROR;
    }
    Py_INCREF(&PyPixelArray_Type);
    if (PyModule_AddObject(module, "PixelArray",
                           (PyObject *)&PyPixelArray_Type)) {
        Py_DECREF((PyObject *)&PyPixelArray_Type);
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    PyPixelArray_Type.tp_getattro = PyObject_GenericGetAttr;

    c_api[0] = &PyPixelArray_Type;
    c_api[1] = PyPixelArray_New;
    apiobj = encapsulate_api (c_api, "pixelarray");
    if (apiobj == NULL) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    if (PyModule_AddObject(module, PYGAMEAPI_LOCAL_ENTRY, apiobj)) {
        Py_DECREF(apiobj);
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    MODINIT_RETURN(module);
}
