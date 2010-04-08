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
#define PYGAME_SDLPXFMT_INTERNAL

#include "videomod.h"
#include "pgsdl.h"
#include "surface.h"
#include "sdlvideo_doc.h"

#define IS_READONLY(x) (((PyPixelFormat*)(x))->readonly == 1)

static SDL_PixelFormat* _sdlpixelformat_new (void);
static PyObject* _pixelformat_new (PyTypeObject *type, PyObject *args,
    PyObject *kwds);
static int _pixelformat_init (PyObject *self, PyObject *args, PyObject *kwds);
static void _pixelformat_dealloc (PyPixelFormat *self);

static PyObject* _pixelformat_getbpp (PyObject *self, void *closure);
static int _pixelformat_setbpp (PyObject *self, PyObject *value, void *closure);
static PyObject* _pixelformat_getbytes (PyObject *self, void *closure);
static int _pixelformat_setbytes (PyObject *self, PyObject *value,
    void *closure);
static PyObject* _pixelformat_getlosses (PyObject *self, void *closure);
static int _pixelformat_setlosses (PyObject *self, PyObject *value,
    void *closure);
static PyObject* _pixelformat_getshifts (PyObject *self, void *closure);
static int _pixelformat_setshifts (PyObject *self, PyObject *value,
    void *closure);
static PyObject* _pixelformat_getmasks (PyObject *self, void *closure);
static int _pixelformat_setmasks (PyObject *self, PyObject *value,
    void *closure);
static PyObject* _pixelformat_getcolorkey (PyObject *self, void *closure);
static int _pixelformat_setcolorkey (PyObject *self, PyObject *value,
    void *closure);
static PyObject* _pixelformat_getalpha (PyObject *self, void *closure);
static int _pixelformat_setalpha (PyObject *self, PyObject *value,
    void *closure);
static PyObject* _pixelformat_getpalette (PyObject *self, void *closure);
static PyObject* _pixelformat_getreadonly (PyObject *self, void *closure);

static PyObject *_pixelformat_maprgba (PyObject *self, PyObject *args);
static PyObject *_pixelformat_getrgba (PyObject *self, PyObject *args);

/**
 */
static PyMethodDef _pixelformat_methods[] = {
    { "map_rgba", _pixelformat_maprgba, METH_VARARGS,
      DOC_VIDEO_PIXELFORMAT_MAP_RGBA },
    { "get_rgba", _pixelformat_getrgba, METH_O,
      DOC_VIDEO_PIXELFORMAT_GET_RGBA },
    { NULL, NULL, 0, NULL }
};

/**
 */
static PyGetSetDef _pixelformat_getsets[] = {
    { "palette", _pixelformat_getpalette, NULL, DOC_VIDEO_PIXELFORMAT_PALETTE,
      NULL },
    { "bits_per_pixel", _pixelformat_getbpp, _pixelformat_setbpp,
      DOC_VIDEO_PIXELFORMAT_BITS_PER_PIXEL, NULL },
    { "bytes_per_pixel", _pixelformat_getbytes, _pixelformat_setbytes,
      DOC_VIDEO_PIXELFORMAT_BYTES_PER_PIXEL, NULL },
    { "losses", _pixelformat_getlosses, _pixelformat_setlosses,
      DOC_VIDEO_PIXELFORMAT_LOSSES, NULL },
    { "shifts", _pixelformat_getshifts, _pixelformat_setshifts,
      DOC_VIDEO_PIXELFORMAT_SHIFTS, NULL },
    { "masks", _pixelformat_getmasks, _pixelformat_setmasks,
      DOC_VIDEO_PIXELFORMAT_MASKS, NULL },
    { "colorkey", _pixelformat_getcolorkey, _pixelformat_setcolorkey,
      DOC_VIDEO_PIXELFORMAT_COLORKEY, NULL },
    { "alpha", _pixelformat_getalpha, _pixelformat_setalpha,
      DOC_VIDEO_PIXELFORMAT_ALPHA, NULL },
    { "readonly", _pixelformat_getreadonly, NULL,
      DOC_VIDEO_PIXELFORMAT_READONLY, NULL },
    { NULL, NULL, NULL, NULL, NULL }
};

/**
 */
PyTypeObject PyPixelFormat_Type =
{
    TYPE_HEAD(NULL,0)
    "video.PixelFormat",              /* tp_name */
    sizeof (PyPixelFormat),   /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor) _pixelformat_dealloc, /* tp_dealloc */
    0,                          /* tp_print */
    0,                          /* tp_getattr */
    0,                          /* tp_setattr */
    0,                          /* tp_compare */
    0,                          /* tp_repr */
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
    DOC_VIDEO_PIXELFORMAT,
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    _pixelformat_methods,       /* tp_methods */
    0,                          /* tp_members */
    _pixelformat_getsets,       /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc) _pixelformat_init,    /* tp_init */
    0,                          /* tp_alloc */
    _pixelformat_new,           /* tp_new */
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

static SDL_PixelFormat*
_sdlpixelformat_new (void)
{
    SDL_PixelFormat *fmt = PyMem_New (SDL_PixelFormat, 1);
    if (!fmt)
        return NULL;
    
    fmt->palette = NULL;
    fmt->BitsPerPixel = 0;
    fmt->BytesPerPixel = 0;
    fmt->Rloss = fmt->Gloss = fmt->Bloss = fmt->Aloss = 0;
    fmt->Rshift = fmt->Gshift = fmt->Bshift = fmt->Ashift = 0;
    fmt->Rmask = fmt->Gmask = fmt->Bmask = fmt->Amask = 0;
    fmt->colorkey = 0;
    fmt->alpha = 0;
    return fmt;
}

static PyObject*
_pixelformat_new (PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    SDL_PixelFormat *fmt;
    PyPixelFormat *pxfmt;

    fmt = _sdlpixelformat_new ();
    if (!fmt)
        return NULL;

    pxfmt = (PyPixelFormat *)type->tp_alloc (type, 0);
    if (!pxfmt)
    {
        PyMem_Free (fmt);
        return NULL;
    }

    pxfmt->format = fmt;
    pxfmt->readonly = 0;
    return (PyObject *) pxfmt;
}

static void
_pixelformat_dealloc (PyPixelFormat *self)
{
    if (self->format && !self->readonly)
        PyMem_Free (self->format);

    ((PyObject*)self)->ob_type->tp_free ((PyObject *) self);
}

static int
_pixelformat_init (PyObject *self, PyObject *args, PyObject *kwds)
{
    SDL_PixelFormat *fmt = _sdlpixelformat_new ();
    if (!fmt)
        return -1;
    ((PyPixelFormat*)self)->format = fmt;
    ((PyPixelFormat*)self)->readonly = 0;
    
    return 0;
}

/* PixelFormat getters/setters */
static PyObject*
_pixelformat_getbpp (PyObject *self, void *closure)
{
    SDL_PixelFormat *fmt = ((PyPixelFormat*)self)->format;
    return PyInt_FromLong (fmt->BitsPerPixel);
}

static int
_pixelformat_setbpp (PyObject *self, PyObject *value, void *closure)
{
    Uint8 bpp;
    SDL_PixelFormat *fmt = ((PyPixelFormat*)self)->format;
    
    if (IS_READONLY(self))
    {
        PyErr_SetString (PyExc_PyGameError, "PixelFormat is readonly");
        return -1;
    }
    if (!Uint8FromObj (value, &bpp))
    {
        PyErr_SetString (PyExc_ValueError,
            "bpp must be an integer in the range 0-255");
        return -1;
    }
    fmt->BitsPerPixel = bpp;
    return 0;
}

static PyObject*
_pixelformat_getbytes (PyObject *self, void *closure)
{
    SDL_PixelFormat *fmt = ((PyPixelFormat*)self)->format;
    return PyInt_FromLong (fmt->BytesPerPixel);
}

static int
_pixelformat_setbytes (PyObject *self, PyObject *value, void *closure)
{
    Uint8 bpp;
    SDL_PixelFormat *fmt = ((PyPixelFormat*)self)->format;
    
    if (IS_READONLY(self))
    {
        PyErr_SetString (PyExc_PyGameError, "PixelFormat is readonly");
        return -1;
    }
    if (!Uint8FromObj (value, &bpp))
    {
        PyErr_SetString (PyExc_ValueError,
            "bytes must be an integer in the range 0-255");
        return -1;
    }
    fmt->BytesPerPixel = bpp;
    return 0;
}

static PyObject*
_pixelformat_getlosses (PyObject *self, void *closure)
{
    SDL_PixelFormat *fmt = ((PyPixelFormat*)self)->format;
    return Py_BuildValue ("(iiii)", fmt->Rloss, fmt->Gloss, fmt->Bloss,
        fmt->Aloss);
}

static int
_pixelformat_setlosses (PyObject *self, PyObject *value, void *closure)
{
    SDL_PixelFormat *fmt = ((PyPixelFormat*)self)->format;
    Uint8 rloss, gloss, bloss, aloss;
    
    if (IS_READONLY(self))
    {
        PyErr_SetString (PyExc_PyGameError, "PixelFormat is readonly");
        return -1;
    }
    
    if (!PySequence_Check (value) || PySequence_Size (value) != 4)
    {
        PyErr_SetString (PyExc_ValueError, "losses must be a 4-value sequence");
        return -1;
    }
    if (!Uint8FromSeqIndex (value, 0, &rloss) ||
        !Uint8FromSeqIndex (value, 1, &gloss) ||
        !Uint8FromSeqIndex (value, 2, &bloss) ||
        !Uint8FromSeqIndex (value, 3, &aloss))
    {
        PyErr_SetString (PyExc_ValueError,
            "invalid loss values in losses sequence");
        return -1;
    }
    fmt->Rloss = rloss;
    fmt->Gloss = gloss;
    fmt->Bloss = bloss;
    fmt->Aloss = aloss;
    return 0;
}
static PyObject*
_pixelformat_getshifts (PyObject *self, void *closure)
{
    SDL_PixelFormat *fmt = ((PyPixelFormat*)self)->format;
    return Py_BuildValue ("(iiii)", fmt->Rshift, fmt->Gshift, fmt->Bshift,
        fmt->Ashift);
}

static int
_pixelformat_setshifts (PyObject *self, PyObject *value, void *closure)
{
    SDL_PixelFormat *fmt = ((PyPixelFormat*)self)->format;
    Uint8 rshift, gshift, bshift, ashift;
    
    if (IS_READONLY(self))
    {
        PyErr_SetString (PyExc_PyGameError, "PixelFormat is readonly");
        return -1;
    }
    
    if (!PySequence_Check (value) || PySequence_Size (value) != 4)
    {
        PyErr_SetString (PyExc_ValueError, "shifts must be a 4-value sequence");
        return -1;
    }
    if (!Uint8FromSeqIndex (value, 0, &rshift) ||
        !Uint8FromSeqIndex (value, 1, &gshift) ||
        !Uint8FromSeqIndex (value, 2, &bshift) ||
        !Uint8FromSeqIndex (value, 3, &ashift))
    {
        PyErr_SetString (PyExc_ValueError,
            "invalid shift values in shifts sequence");
        return -1;
    }
    fmt->Rshift = rshift;
    fmt->Gshift = gshift;
    fmt->Bshift = bshift;
    fmt->Ashift = ashift;
    return 0;
}

static PyObject*
_pixelformat_getmasks (PyObject *self, void *closure)
{
    SDL_PixelFormat *fmt = ((PyPixelFormat*)self)->format;
    return Py_BuildValue ("(kkkk)", fmt->Rmask, fmt->Gmask, fmt->Bmask,
        fmt->Amask);
}
static int
_pixelformat_setmasks (PyObject *self, PyObject *value, void *closure)
{
    SDL_PixelFormat *fmt = ((PyPixelFormat*)self)->format;
    Uint32 rmask, gmask, bmask, amask;
    
    if (IS_READONLY(self))
    {
        PyErr_SetString (PyExc_PyGameError, "PixelFormat is readonly");
        return -1;
    }
    
    if (!PySequence_Check (value) || PySequence_Size (value) != 4)
    {
        PyErr_SetString (PyExc_ValueError, "masks must be a 4-value sequence");
        return -1;
    }
    if (!Uint32FromSeqIndex (value, 0, &rmask) ||
        !Uint32FromSeqIndex (value, 1, &gmask) ||
        !Uint32FromSeqIndex (value, 2, &bmask) ||
        !Uint32FromSeqIndex (value, 3, &amask))
    {
        PyErr_SetString (PyExc_ValueError,
            "invalid mask values in masks sequence");
        return -1;
    }
    fmt->Rmask = rmask;
    fmt->Gmask = gmask;
    fmt->Bmask = bmask;
    fmt->Amask = amask;
    return 0;
}

static PyObject*
_pixelformat_getcolorkey (PyObject *self, void *closure)
{
    SDL_PixelFormat *fmt = ((PyPixelFormat*)self)->format;
    PyObject *color = PyColor_NewFromNumber((pguint32) fmt->colorkey);
    return color;
}

static int
_pixelformat_setcolorkey (PyObject *self, PyObject *value, void *closure)
{
    SDL_PixelFormat *fmt = ((PyPixelFormat*)self)->format;
    if (IS_READONLY(self))
    {
        PyErr_SetString (PyExc_PyGameError, "PixelFormat is readonly");
        return -1;
    }
    
    if (!PyColor_Check (value))
    {
        PyErr_SetString (PyExc_ValueError, "color must be a Color");
        return -1;
    }
    
    fmt->colorkey = PyColor_AsNumber (value);
    ARGB2FORMAT (fmt->colorkey, fmt);
    return 0;
}

static PyObject*
_pixelformat_getalpha (PyObject *self, void *closure)
{
    return PyInt_FromLong (((PyPixelFormat*)self)->format->alpha);
}

static int
_pixelformat_setalpha (PyObject *self, PyObject *value, void *closure)
{
    SDL_PixelFormat *fmt = ((PyPixelFormat*)self)->format;
    Uint8 alpha;
    
    if (IS_READONLY(self))
    {
        PyErr_SetString (PyExc_PyGameError, "PixelFormat is readonly");
        return -1;
    }
    if (!Uint8FromObj (value, &alpha))
    {
        PyErr_SetString (PyExc_ValueError,
            "alpha must be an integer in the range 0-255");
        return -1;
    }
    fmt->alpha = alpha;
    return 0;

}

static PyObject*
_pixelformat_getpalette (PyObject *self, void *closure)
{
    SDL_PixelFormat *fmt = ((PyPixelFormat*)self)->format;
    SDL_Palette *pal = fmt->palette;
    SDL_Color *c;
    PyObject *tuple, *color;
    Py_ssize_t i;

    if (!pal)
        Py_RETURN_NONE;

    tuple = PyTuple_New ((Py_ssize_t) pal->ncolors);
    if (!tuple)
        return NULL;

    for (i = 0; i < pal->ncolors; i++)
    {
        c = &pal->colors[i];
        color = PyColor_NewFromRGBA (c->r, c->g, c->b, 255);
        if (!color)
        {
            Py_DECREF (tuple);
            return NULL;
        }
        PyTuple_SET_ITEM (tuple, i, color);
    }
    return tuple;
}

static PyObject*
_pixelformat_getreadonly (PyObject *self, void *closure)
{
    return PyBool_FromLong (IS_READONLY(self));
}

/* Methods */
static PyObject*
_pixelformat_maprgba (PyObject *self, PyObject *args)
{
    PyObject *color = NULL;
    Uint8 r, g, b, a = 0;
    
    int _a = -1;
    Uint32 val;

    if (!PyArg_ParseTuple (args, "iii|i:map_rgba", &r, &g, &b, &_a))
    {
        PyErr_Clear ();
        if (PyArg_ParseTuple (args, "O:map_rgba", &color))
        {
            if (!PyColor_Check (color))
            {
                PyErr_SetString (PyExc_TypeError, "argument must be a color");
                return NULL;
            }
        }
        else
            return NULL;
    }

    if (color)
    {
        r = (Uint8) ((PyColor*)color)->r;
        g = (Uint8) ((PyColor*)color)->g;
        b = (Uint8) ((PyColor*)color)->b;
        _a = a = (Uint8) ((PyColor*)color)->a;
    }

    /* Only check for the alpha value, if there is a per-pixel alpha mask set
     * and an alpha value was requested.
     */
    if (((PyPixelFormat*)self)->format->Amask != 0 && _a != -1)
        val = SDL_MapRGBA (((PyPixelFormat*)self)->format, r, g, b, a);
    else
        val = SDL_MapRGB (((PyPixelFormat*)self)->format, r, g, b);

    return PyLong_FromUnsignedLong ((unsigned long) val);
}

static PyObject*
_pixelformat_getrgba (PyObject *self, PyObject *args)
{
    Uint8 r = 0, g = 0, b = 0, a = 255;
    Uint32 val;

    if (!Uint32FromObj (args, &val))
        return NULL;
    if (PyColor_Check (args))
    {
        ARGB2FORMAT (val, ((PyPixelFormat*)self)->format);
    }

    if (((PyPixelFormat*)self)->format->Amask != 0)
        SDL_GetRGBA (val, ((PyPixelFormat*)self)->format, &r, &g, &b, &a);
    else
        SDL_GetRGB (val, ((PyPixelFormat*)self)->format, &r, &g, &b);

    return PyColor_NewFromRGBA ((pgbyte)r, (pgbyte)g, (pgbyte)b, (pgbyte)a);
}

/* C API */
PyObject*
PyPixelFormat_New (void)
{
    PyPixelFormat *format;
    format = (PyPixelFormat*) PyPixelFormat_Type.tp_new (&PyPixelFormat_Type,
        NULL, NULL);
    return (PyObject*) format;
}

PyObject*
PyPixelFormat_NewFromSDLPixelFormat (SDL_PixelFormat *fmt)
{
    PyPixelFormat *format;
    if (!fmt)
    {
        PyErr_SetString (PyExc_ValueError, "fmt must not be NULL");
        return NULL;
    }
   
    format = (PyPixelFormat*) PyPixelFormat_Type.tp_new (&PyPixelFormat_Type,
        NULL, NULL);
    if (!format)
        return NULL;

    PyMem_Free (format->format);
    format->readonly = 1;
    format->format = fmt;
    return (PyObject*) format;
}

void
pixelformat_export_capi (void **capi)
{
    capi[PYGAME_SDLPXFMT_FIRSTSLOT] = &PyPixelFormat_Type;
    capi[PYGAME_SDLPXFMT_FIRSTSLOT+1] = (void *)PyPixelFormat_New;
    capi[PYGAME_SDLPXFMT_FIRSTSLOT+2] =
        (void *)PyPixelFormat_NewFromSDLPixelFormat;
}
