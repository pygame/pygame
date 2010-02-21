/*
  pygame - Python Game Library
  Copyright (C) 2000-2001 Pete Shinners, 2007-2008 Marcus von Appen

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
#define PYGAME_SDLSURFACE_INTERNAL

#include "videomod.h"
#include "surface.h"
#include "pgsdl.h"
#include "tga.h"
#include "sdlvideo_doc.h"

#ifdef HAVE_PNG
#include "pgpng.h"
#endif

static PyObject* _surface_new (PyTypeObject *type, PyObject *args,
    PyObject *kwds);
static int _surface_init (PyObject *surface, PyObject *args, PyObject *kwds);
static void _surface_dealloc (PySDLSurface *self);
static PyObject* _surface_repr (PyObject *self);

static PyObject* _surface_getdict (PyObject *self, void *closure);
static PyObject* _surface_getcliprect (PyObject *self, void *closure);
static int _surface_setcliprect (PyObject *self, PyObject *value,
    void *closure);
static PyObject* _surface_getsize (PyObject *self, void *closure);
static PyObject* _surface_getflags (PyObject *self, void *closure);
static PyObject* _surface_getformat (PyObject *self, void *closure);
static PyObject* _surface_getpitch (PyObject *self, void *closure);
static PyObject* _surface_getpixels (PyObject *self, void *closure);
static PyObject* _surface_getwidth (PyObject *self, void *closure);
static PyObject* _surface_getheight (PyObject *self, void *closure);
static PyObject* _surface_getlocked (PyObject *self, void *closure);

static PyObject* _surface_update (PyObject *self, PyObject *args);
static PyObject* _surface_flip (PyObject *self);
static PyObject* _surface_setcolors (PyObject *self, PyObject *args);
static PyObject* _surface_getpalette (PyObject *self);
static PyObject* _surface_setpalette (PyObject *self, PyObject *args);
static PyObject* _surface_lock (PyObject *self);
static PyObject* _surface_unlock (PyObject *self);
static PyObject* _surface_getcolorkey (PyObject *self);
static PyObject* _surface_setcolorkey (PyObject *self, PyObject *args);
static PyObject* _surface_getalpha (PyObject *self);
static PyObject* _surface_setalpha (PyObject *self, PyObject *args);
static PyObject* _surface_convert (PyObject *self, PyObject *args,
    PyObject *kwds);
static PyObject* _surface_copy (PyObject *self);
static PyObject* _surface_blit (PyObject *self, PyObject *args, PyObject *kwds);
static PyObject* _surface_fill (PyObject *self, PyObject *args, PyObject *kwds);
static PyObject* _surface_save (PyObject *self, PyObject *args);
static PyObject* _surface_getat (PyObject *self, PyObject *args);
static PyObject* _surface_setat (PyObject *self, PyObject *args);
static PyObject* _surface_scroll (PyObject *self, PyObject *args,
    PyObject *kwds);

static void _release_c_lock (void *ptr);

/**
 */
static PyMethodDef _surface_methods[] = {
    { "update", _surface_update, METH_O, DOC_VIDEO_SURFACE_UPDATE },
    { "flip", (PyCFunction)_surface_flip, METH_NOARGS, DOC_VIDEO_SURFACE_FLIP },
    { "set_colors", _surface_setcolors, METH_VARARGS,
      DOC_VIDEO_SURFACE_SET_COLORS },
    { "get_palette", (PyCFunction) _surface_getpalette, METH_NOARGS,
      DOC_VIDEO_SURFACE_GET_PALETTE},
    { "set_palette", _surface_setpalette, METH_VARARGS,
      DOC_VIDEO_SURFACE_SET_PALETTE },
    { "lock", (PyCFunction)_surface_lock, METH_NOARGS, DOC_VIDEO_SURFACE_LOCK },
    { "unlock", (PyCFunction)_surface_unlock, METH_NOARGS,
      DOC_VIDEO_SURFACE_UNLOCK },
    { "get_colorkey", (PyCFunction) _surface_getcolorkey, METH_NOARGS,
      DOC_VIDEO_SURFACE_GET_COLORKEY },
    { "set_colorkey", _surface_setcolorkey, METH_VARARGS,
      DOC_VIDEO_SURFACE_SET_COLORKEY },
    { "get_alpha", (PyCFunction) _surface_getalpha, METH_NOARGS,
      DOC_VIDEO_SURFACE_GET_ALPHA },
    { "set_alpha", _surface_setalpha, METH_VARARGS,
      DOC_VIDEO_SURFACE_SET_ALPHA },
    { "get_at", _surface_getat, METH_VARARGS, DOC_VIDEO_SURFACE_GET_AT },
    { "set_at", _surface_setat, METH_VARARGS, DOC_VIDEO_SURFACE_SET_AT },
    { "convert", (PyCFunction) _surface_convert, METH_VARARGS | METH_KEYWORDS,
      DOC_VIDEO_SURFACE_CONVERT },
    { "copy", (PyCFunction)_surface_copy, METH_NOARGS, DOC_VIDEO_SURFACE_COPY },
    { "blit", (PyCFunction)_surface_blit, METH_VARARGS | METH_KEYWORDS,
      DOC_VIDEO_SURFACE_BLIT },
    { "fill", (PyCFunction)_surface_fill, METH_VARARGS | METH_KEYWORDS,
      DOC_VIDEO_SURFACE_FILL },
    { "save", _surface_save, METH_VARARGS, DOC_VIDEO_SURFACE_SAVE },
    { "scroll", (PyCFunction) _surface_scroll, METH_VARARGS | METH_KEYWORDS,
      DOC_VIDEO_SURFACE_SCROLL },
    { NULL, NULL, 0, NULL }
};

/**
 */
static PyGetSetDef _surface_getsets[] = {
    { "__dict__", _surface_getdict, NULL, "", NULL },
    { "clip_rect", _surface_getcliprect, _surface_setcliprect,
      DOC_VIDEO_SURFACE_CLIP_RECT, NULL },
    { "w", _surface_getwidth, NULL, DOC_VIDEO_SURFACE_W, NULL },
    { "width", _surface_getwidth, NULL, DOC_VIDEO_SURFACE_WIDTH, NULL },
    { "h", _surface_getheight, NULL, DOC_VIDEO_SURFACE_H, NULL },
    { "height", _surface_getheight, NULL, DOC_VIDEO_SURFACE_HEIGHT, NULL },
    { "size", _surface_getsize, NULL, DOC_VIDEO_SURFACE_SIZE, NULL },
    { "flags", _surface_getflags, NULL, DOC_VIDEO_SURFACE_FLAGS, NULL },
    { "format", _surface_getformat, NULL, DOC_VIDEO_SURFACE_FORMAT, NULL },
    { "pitch", _surface_getpitch, NULL, DOC_VIDEO_SURFACE_PITCH, NULL },
    { "pixels", _surface_getpixels, NULL, DOC_VIDEO_SURFACE_PIXELS, NULL },
    { "locked", _surface_getlocked, NULL, DOC_VIDEO_SURFACE_LOCKED, NULL },
    { NULL, NULL, NULL, NULL, NULL }
};

/**
 */
PyTypeObject PySDLSurface_Type =
{
    TYPE_HEAD(NULL, 0)
    "video.Surface",              /* tp_name */
    sizeof (PySDLSurface),   /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor) _surface_dealloc, /* tp_dealloc */
    0,                          /* tp_print */
    0,                          /* tp_getattr */
    0,                          /* tp_setattr */
    0,                          /* tp_compare */
    (reprfunc) _surface_repr,   /* tp_repr */
    0,                          /* tp_as_number */
    0,                          /* tp_as_sequence */
    0,                          /* tp_as_mapping */
    0,                          /* tp_hash */
    0,                          /* tp_call */
    0,                          /* tp_str */
    0,                          /* tp_getattro */
    0,                          /* tp_setattro */
    0,                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_WEAKREFS,
    DOC_VIDEO_SURFACE,
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    offsetof (PySDLSurface, weakrefs), /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    _surface_methods,           /* tp_methods */
    0,                          /* tp_members */
    _surface_getsets,           /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    offsetof (PySDLSurface, dict), /* tp_dictoffset */
    (initproc) _surface_init,   /* tp_init */
    0,                          /* tp_alloc */
    _surface_new,               /* tp_new */
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
_surface_dealloc (PySDLSurface *self)
{
    PyObject *obj;

    Py_XDECREF (self->dict);
    Py_XDECREF (self->locklist);
    if (self->weakrefs)
        PyObject_ClearWeakRefs ((PyObject *) self);

    if (self->surface && !self->isdisplay)
        SDL_FreeSurface (self->surface);
    self->surface = NULL;

    obj = (PyObject*) &(self->pysurface);
    obj->ob_type->tp_free ((PyObject*)self);
}

static PyObject*
_surface_new (PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PySDLSurface *surface = (PySDLSurface *)type->tp_alloc (type, 0);
    if (!surface)
        return NULL;

    surface->locklist = NULL;
    surface->dict = NULL;
    surface->weakrefs = NULL;
    surface->intlocks = 0;
    surface->isdisplay = 0;

    surface->pysurface.get_width = _surface_getwidth;
    surface->pysurface.get_height = _surface_getheight;
    surface->pysurface.get_size = _surface_getsize;
    surface->pysurface.get_pixels = _surface_getpixels;
    surface->pysurface.blit = _surface_blit;
    surface->pysurface.copy = _surface_copy;

    return (PyObject*) surface;
}

static int
_surface_init (PyObject *self, PyObject *args, PyObject *kwds)
{
    Uint32 flags = 0;
    int width, height, depth = 0;
    Uint32 rmask, gmask, bmask, amask;
    SDL_Surface *surface;
    PyObject *masks = NULL;
    
    static char *keys[] = { "width", "height", "depth", "flags", "masks",
                            NULL };
    static char *keys2[] = { "size", "depth", "flags", "masks", NULL };

    /* The SDL docs require SDL_SetVideoMode() to be called beforehand. It
     * works nicely without for SDL >= 1.2.10, though */
    /* ASSERT_VIDEO_SURFACE_SET (-1); */
    ASSERT_VIDEO_INIT (-1);

    if (PySurface_Type.tp_init ((PyObject *) self, args, kwds) < 0)
        return -1;

    if (!PyArg_ParseTupleAndKeywords (args, kwds, "ii|ilO", keys, &width,
            &height, &depth, &flags, &masks))
    {
        PyObject *size;

        PyErr_Clear ();
        if (!PyArg_ParseTupleAndKeywords (args, kwds, "O|ilO", keys2, &size,
                &depth, &flags, &masks))
            return -1;
        if (!SizeFromObj (size, (pgint32*)&width, (pgint32*)&height))
            return -1;
    }

    if (width < 0 || height < 0)
    {
        PyErr_SetString (PyExc_ValueError,
            "width and height must not be negative");
        return -1;
    }

    if (!depth && !masks && !flags)
    {
        /* Use the optimal video depth or set default depth. */
        const SDL_VideoInfo *info = SDL_GetVideoInfo ();
        depth = info->vfmt->BitsPerPixel;
    }
    
    if (depth && !masks)
    {
        /* Using depth we let the screen resolution decide about the
         * rgba mask order */
        rmask = gmask = bmask = amask = 0;

        if (flags & SDL_SRCALPHA)
        {
            /* The user wants an alpha component. In that case we will
             * set a default RGBA mask for 16 and 32 bpp. */
            switch (depth)
            {
            case 16:
                amask = 0xf000;
                rmask = 0x0f00;
                gmask = 0x00f0;
                bmask = 0x000f;
                break;
            case 32:
                amask = 0xff000000;
                rmask = 0x00ff0000;
                gmask = 0x0000ff00;
                bmask = 0x000000ff;
                break;
            default:
                PyErr_SetString (PyExc_PyGameError,
                    "Per-pixel alpha requires masks for that depth");
                return -1;
            }
        }
    }
    else if (masks)
    {
        if (!PySequence_Check (masks) || PySequence_Size (masks) != 4)
        {
            PyErr_SetString (PyExc_ValueError,
                "masks must be a 4-value sequence");
            return -1;
        }
        if (!Uint32FromSeqIndex (masks, 0, &rmask) ||
            !Uint32FromSeqIndex (masks, 1, &gmask) ||
            !Uint32FromSeqIndex (masks, 2, &bmask) ||
            !Uint32FromSeqIndex (masks, 3, &amask))
        {
            PyErr_SetString (PyExc_ValueError,
                "invalid mask values in masks sequence");
            return -1;
        }
    }
    
    surface = SDL_CreateRGBSurface (flags, width, height, depth, rmask, gmask,
        bmask, amask);
    if (!surface)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return -1;
    }
    
    ((PySDLSurface*)self)->surface = surface;
    return 0;
}

static PyObject*
_surface_repr (PyObject *self)
{
    SDL_Surface *sf = PySDLSurface_AsSDLSurface (self);
#if PY_VERSION_HEX < 0x02050000
    return Text_FromFormat ("<Surface %ldx%ld@%dbpp>", sf->w, sf->h,
        sf->format->BitsPerPixel);
#else
    return Text_FromFormat ("<Surface %ux%u@%ubpp>", sf->w, sf->h,
        sf->format->BitsPerPixel);
#endif
}

/* Surface getters/setters */
static PyObject*
_surface_getdict (PyObject *self, void *closure)
{
    PySDLSurface *surface = (PySDLSurface*) self;
    if (!surface->dict)
    {
        surface->dict = PyDict_New ();
        if (!surface->dict)
            return NULL;
    }
    Py_INCREF (surface->dict);
    return surface->dict;
}

static PyObject*
_surface_getcliprect (PyObject *self, void *closure)
{
    SDL_Surface *surface = PySDLSurface_AsSDLSurface (self);
    SDL_Rect sdlrect;

    SDL_GetClipRect (surface, &sdlrect);
    return PyRect_New (sdlrect.x, sdlrect.y, sdlrect.w, sdlrect.h);
}

static int
_surface_setcliprect (PyObject *self, PyObject *value, void *closure)
{
    SDL_Surface *surface = PySDLSurface_AsSDLSurface (self);
    SDL_Rect rect;
    
    if (value == Py_None)
    {
        SDL_SetClipRect (surface, NULL);
        return 0;
    }
    if (!SDLRectFromRect (value, &rect))
        return -1;
    SDL_SetClipRect (surface, &rect);
    return 0;
}

static PyObject*
_surface_getwidth (PyObject *self, void *closure)
{
    SDL_Surface *surface = PySDLSurface_AsSDLSurface (self);
    return PyInt_FromLong (surface->w);
}

static PyObject*
_surface_getheight (PyObject *self, void *closure)
{
    SDL_Surface *surface = PySDLSurface_AsSDLSurface (self);
    return PyInt_FromLong (surface->h);
}

static PyObject*
_surface_getsize (PyObject *self, void *closure)
{
    SDL_Surface *surface = PySDLSurface_AsSDLSurface (self);
    return Py_BuildValue ("(ii)", surface->w, surface->h);
}

static PyObject*
_surface_getflags (PyObject *self, void *closure)
{
    SDL_Surface *surface = PySDLSurface_AsSDLSurface (self);
    return PyLong_FromLong ((long)surface->flags);
}

static PyObject*
_surface_getformat (PyObject *self, void *closure)
{
    SDL_Surface *surface = PySDLSurface_AsSDLSurface (self);
    return PyPixelFormat_NewFromSDLPixelFormat (surface->format);
}

static PyObject*
_surface_getpitch (PyObject *self, void *closure)
{
    SDL_Surface *surface = PySDLSurface_AsSDLSurface (self);
    return PyInt_FromLong (surface->pitch);
}

static PyObject*
_surface_getpixels (PyObject *self, void *closure)
{
    PyObject *buffer;
    SDL_Surface *surface = PySDLSurface_AsSDLSurface (self);

    buffer = PyBufferProxy_New (NULL, NULL, 0, PySDLSurface_RemoveRefLock);
    if (!buffer)
        return NULL;
    if (!PySDLSurface_AddRefLock (self, buffer))
        return NULL;
    ((PyBufferProxy*)buffer)->object = self;
    ((PyBufferProxy*)buffer)->buffer = surface->pixels;
    ((PyBufferProxy*)buffer)->length = (Py_ssize_t) surface->pitch * surface->h;

    return buffer;
}

static PyObject*
_surface_getlocked (PyObject *self, void *closure)
{
    PySDLSurface *sf = (PySDLSurface*) self;

    if (sf->intlocks != 0)
        Py_RETURN_TRUE;

    if (sf->locklist && PyList_Size (sf->locklist) != 0)
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

/* Surface methods */
static PyObject*
_surface_update (PyObject *self, PyObject *args)
{
    PyObject *item;
    SDL_Rect *rects, r;
    Py_ssize_t count, i;
    
    if (SDLRectFromRect (args, &r))
    {
        CLIP_RECT_TO_SURFACE (PySDLSurface_AsSDLSurface (self), &r);

        /* Single rect to update */
        Py_BEGIN_ALLOW_THREADS;
        SDL_UpdateRect (PySDLSurface_AsSDLSurface (self), r.x, r.y, r.w, r.h);
        Py_END_ALLOW_THREADS;
        Py_RETURN_NONE;
    }
    else
        PyErr_Clear (); /* From SDLRectFromRect */
    
    if (!PySequence_Check (args))
    {
        PyErr_SetString (PyExc_TypeError,
            "argument must be a Rect or list of Rect objects");
        return NULL;
    }

    /* Sequence of rects only? */
    count = PySequence_Size (args);
    rects = PyMem_New (SDL_Rect, (size_t) count);
    if (!rects)
        return NULL;

    for (i = 0; i < count; i++)
    {
        item = PySequence_ITEM (args, i);

        if (!SDLRectFromRect (item, &(rects[i])))
        {
            Py_XDECREF (item);
            PyMem_Free (rects);
            PyErr_Clear ();
            PyErr_SetString (PyExc_ValueError,
                "list may only contain Rect objects");
            return NULL;
        }
        CLIP_RECT_TO_SURFACE (PySDLSurface_AsSDLSurface (self), &(rects[i]));
        Py_DECREF (item);
    }

    Py_BEGIN_ALLOW_THREADS;
    SDL_UpdateRects (PySDLSurface_AsSDLSurface (self), (int)count, rects);
    Py_END_ALLOW_THREADS;
    PyMem_Free (rects);

    Py_RETURN_NONE;
}

static PyObject*
_surface_flip (PyObject *self)
{
    int ret;
    
    Py_BEGIN_ALLOW_THREADS;
    ret = SDL_Flip (PySDLSurface_AsSDLSurface (self));
    Py_END_ALLOW_THREADS;
    
    if (ret == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_surface_setcolors (PyObject *self, PyObject *args)
{
    PyObject *colorlist, *item;
    Py_ssize_t count, i;
    SDL_Color *colors;
    int ret, first = 0;
    
    if (!PyArg_ParseTuple (args, "O|i:set_colors", &colorlist, &first))
        return NULL;

    if (!PySequence_Check (colorlist))
    {
        PyErr_SetString (PyExc_TypeError,
            "argument must be a list of Color objects");
        return NULL;
    }

    if (first < 0)
    {
        PyErr_SetString (PyExc_ValueError,
            "first position must not be negative");
        return NULL;
    }

    count = PySequence_Size (colorlist);
    colors = PyMem_New (SDL_Color, (size_t) count);
    if (!colors)
        return NULL;

    for (i = 0; i < count; i++)
    {
        item = PySequence_ITEM (colorlist, i);

        if (!PyColor_Check (item))
        {
            Py_XDECREF (item);
            PyMem_Free (colors);

            PyErr_SetString (PyExc_ValueError,
                "list may only contain Color objects");
            return NULL;
        }
        colors[i].r = ((PyColor*)item)->r;
        colors[i].g = ((PyColor*)item)->g;
        colors[i].b = ((PyColor*)item)->b;
        Py_DECREF (item);
    }
    
    ret = SDL_SetColors (PySDLSurface_AsSDLSurface (self), colors, first,
        (int)count);
    PyMem_Free (colors);

    if (!ret)
        Py_RETURN_FALSE;
    Py_RETURN_TRUE;
}

static PyObject*
_surface_getpalette (PyObject *self)
{
    SDL_Surface *surface = PySDLSurface_AsSDLSurface (self);
    SDL_Palette *pal = surface->format->palette;
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
_surface_setpalette (PyObject *self, PyObject *args)
{
    PyObject *colorlist, *item;
    Py_ssize_t count, i;
    SDL_Color *palette;
    int ret, flags = SDL_LOGPAL | SDL_PHYSPAL, first = 0;

    if (!PyArg_ParseTuple (args, "O|ii:set_palette", &colorlist, &flags,
            &first))
        return NULL;

    if (!PySequence_Check (colorlist))
    {
        PyErr_SetString (PyExc_TypeError,
            "argument must be a list of Color objects");
        return NULL;
    }

    count = PySequence_Size (colorlist);
    palette = PyMem_New (SDL_Color, (size_t) count);
    if (!palette)
        return NULL;

    for (i = 0; i < count; i++)
    {
        item = PySequence_ITEM (colorlist, i);

        if (!PyColor_Check (item))
        {
            Py_XDECREF (item);
            PyMem_Free (palette);

            PyErr_SetString (PyExc_ValueError,
                "list may only contain Color objects");
            return NULL;
        }
        palette[i].r = ((PyColor*)item)->r;
        palette[i].g = ((PyColor*)item)->g;
        palette[i].b = ((PyColor*)item)->b;
        Py_DECREF (item);
    }
    
    ret = SDL_SetPalette (PySDLSurface_AsSDLSurface (self), (int)flags, palette,
        first, (int)count);
    PyMem_Free (palette);

    if (!ret)
        Py_RETURN_FALSE;
    Py_RETURN_TRUE;
}

static PyObject*
_surface_lock (PyObject *self)
{
    if (SDL_LockSurface (PySDLSurface_AsSDLSurface (self)) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    ((PySDLSurface*)self)->intlocks++;
    Py_RETURN_NONE;
}

static PyObject*
_surface_unlock (PyObject *self)
{
    if (((PySDLSurface*)self)->intlocks == 0)
        Py_RETURN_NONE;

    SDL_UnlockSurface (PySDLSurface_AsSDLSurface (self));
    ((PySDLSurface*)self)->intlocks--;
    Py_RETURN_NONE;
}

static PyObject*
_surface_getcolorkey (PyObject *self)
{
    SDL_Surface *surface = PySDLSurface_AsSDLSurface (self);
    Uint8 rgba[4] = { 0 };

    if (!(surface->flags & SDL_SRCCOLORKEY))
        Py_RETURN_NONE;

    SDL_GetRGBA (surface->format->colorkey, surface->format,
        &(rgba[0]), &(rgba[1]), &(rgba[2]), &(rgba[3]));
    return PyColor_New ((pgbyte*)rgba);
}

static PyObject*
_surface_setcolorkey (PyObject *self, PyObject *args)
{
    Uint32 flags = SDL_SRCCOLORKEY, key;
    PyObject *colorkey;

    if (!PyArg_ParseTuple (args, "O|l:set_colorkey", &colorkey, &flags))
        return NULL;

    if (PyColor_Check (colorkey))
    {
        key = (Uint32) PyColor_AsNumber (colorkey);
        ARGB2FORMAT (key, PySDLSurface_AsSDLSurface (self)->format);
    }
    else if (!Uint32FromObj (colorkey, &key))
        return NULL;
    
    if (SDL_SetColorKey (PySDLSurface_AsSDLSurface (self), flags, key) == -1)
        Py_RETURN_FALSE;
    Py_RETURN_TRUE;
}

static PyObject*
_surface_getalpha (PyObject *self)
{
    SDL_Surface *surface = PySDLSurface_AsSDLSurface (self);
    return PyInt_FromLong (surface->format->alpha);
}

static PyObject*
_surface_setalpha (PyObject *self, PyObject *args)
{
    Uint32 flags = SDL_SRCALPHA;
    Uint8 alpha;

    if (!PyArg_ParseTuple (args, "b|l:set_alpha", &alpha, &flags))
        return NULL;

    if (SDL_SetAlpha (PySDLSurface_AsSDLSurface (self), flags, alpha) == -1)
        Py_RETURN_FALSE;
    Py_RETURN_TRUE;
}

static PyObject*
_surface_convert (PyObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *pxfmt = NULL, *sf;
    Uint32 flags = 0;
    SDL_Surface *surface;
    SDL_PixelFormat *fmt;
    
    static char *keys[] = { "format", "flags", NULL };
    if (!PyArg_ParseTupleAndKeywords (args, kwds, "|Ol:convert", keys, &pxfmt,
        &flags))
        return NULL;

    if (pxfmt && !PyPixelFormat_Check (pxfmt))
    {
        PyErr_SetString (PyExc_TypeError, "format must be a PixelFormat");
        return NULL;
    }
    
    if (!pxfmt && flags == 0)
    {
        surface = SDL_DisplayFormat (PySDLSurface_AsSDLSurface (self));
    }
    else
    {
        if (pxfmt)
            fmt = ((PyPixelFormat*)pxfmt)->format;
        else
            fmt = PySDLSurface_AsSDLSurface (self)->format;

        surface = SDL_ConvertSurface (PySDLSurface_AsSDLSurface (self), fmt,
            flags);
    }

    if (!surface)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    
    sf = PySDLSurface_NewFromSDLSurface (surface);
    if (!sf)
    {
        SDL_FreeSurface (surface);
        return NULL;
    }
    return sf;
}

static PyObject*
_surface_getat (PyObject *self, PyObject *args)
{
    int x, y;
    SDL_Surface *surface = PySDLSurface_AsSDLSurface (self);
    SDL_PixelFormat *fmt = surface->format;
    Uint8 rgba[4] = { 0 };
    Uint32 value;

    if (!PyArg_ParseTuple (args, "ii", &x, &y))
    {
        PyObject *pos;
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "O", &pos))
            return NULL;
        if (!PointFromObj (pos, &x, &y))
            return NULL;
    }

    if (x < 0 || x > surface->w || y < 0 || y >= surface->h)
    {
        PyErr_SetString (PyExc_IndexError, "pixel index out of range");
        return NULL;
    }
    
    if (fmt->BytesPerPixel < 1 || fmt->BytesPerPixel > 4)
    {
        PyErr_SetString (PyExc_TypeError, "invalid bit depth for surface");
        return NULL;
    }
    if (SDL_LockSurface (surface) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    GET_PIXEL_AT (value, surface, fmt->BytesPerPixel, x, y);
    SDL_UnlockSurface (surface);
    SDL_GetRGBA (value, fmt, &rgba[0], &rgba[1], &rgba[2], &rgba[3]);
    return PyColor_New ((pgbyte*)rgba);
}

static PyObject*
_surface_setat (PyObject *self, PyObject *args)
{
    int x, y;
    SDL_Surface *surface = PySDLSurface_AsSDLSurface (self);
    SDL_PixelFormat *fmt = surface->format;
    PyObject *color;
    Uint32 value;

    if (!PyArg_ParseTuple (args, "iiO", &x, &y, &color))
    {
        PyObject *pos;
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "OO", &pos, &color))
            return NULL;
        if (!PointFromObj (pos, &x, &y))
            return NULL;
    }

    if (x < 0 || x > surface->w || y < 0 || y >= surface->h)
    {
        PyErr_SetString (PyExc_IndexError, "pixel index out of range");
        return NULL;
    }

    if (!Uint32FromObj (color, &value))
        return NULL;
    if (PyColor_Check (color))
    {
        ARGB2FORMAT (value, surface->format);
    }

    if (fmt->BytesPerPixel < 1 || fmt->BytesPerPixel > 4)
    {
        PyErr_SetString (PyExc_TypeError, "invalid bit depth for surface");
        return NULL;
    }
    if (SDL_LockSurface (surface) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    SET_PIXEL_AT (surface, fmt, x, y, value);
    SDL_UnlockSurface (surface);
    Py_RETURN_NONE;
}

static PyObject*
_surface_copy (PyObject *self)
{
    return PySDLSurface_Copy (self);
}

static PyObject*
_surface_blit (PyObject *self, PyObject *args, PyObject *kwds)
{
    SDL_Surface *src, *dst;
    SDL_Rect srcrect, dstrect;
    PyObject *srcsf, *srcr = NULL, *dstr = NULL;
    int blitargs = 0;

    static char *keys[] = { "surface", "dstrect", "srcrect", "blendargs",
                            NULL };
    if (!PyArg_ParseTupleAndKeywords (args, kwds, "O|OOi:blit", keys, &srcsf,
            &dstr, &srcr, &blitargs))
        return NULL;
   
    if (!PySDLSurface_Check (srcsf))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }

    if (srcr && !SDLRectFromRect (srcr, &srcrect))
    {
        PyErr_Clear ();
        PyErr_SetString (PyExc_TypeError, "srcrect must be a Rect or FRect");
        return NULL;
    }

    if (dstr && !PointFromObj (dstr, (int*)&(dstrect.x), (int*)&(dstrect.y)))
        return NULL;

    src = PySDLSurface_AsSDLSurface (srcsf);
    dst = PySDLSurface_AsSDLSurface (self);
    
    if (!srcr)
    {
        srcrect.x = srcrect.y = 0;
        srcrect.w = src->w;
        srcrect.h = src->h;
    }

    if (!dstr)
    {
        dstrect.x = dstrect.y = 0;
        dstrect.w = dst->w;
        dstrect.h = dst->h;
    }

    if (dst->flags & SDL_OPENGL &&
        !(dst->flags & (SDL_OPENGLBLIT & ~SDL_OPENGL)))
    {
        PyErr_SetString (PyExc_PyGameError,
            "cannot blit to OPENGL Surfaces (OPENGLBLIT is ok)");
        return NULL;
    }

    /* TODO: Check if all blit combinations work. */
    if (blitargs != 0)
    {
        if (!pyg_sdlsoftware_blit (src, &srcrect, dst, &dstrect, blitargs))
        {
            PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
            return NULL;
        }
    }
    else if (SDL_BlitSurface (src, &srcrect, dst, &dstrect) == -1)
        Py_RETURN_NONE;
    return PyRect_New (dstrect.x, dstrect.y, dstrect.w, dstrect.h);
}

static PyObject*
_surface_fill (PyObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *color, *dstrect = NULL;
    SDL_Rect rect;
    SDL_Surface *surface;
    Uint32 col;
    int ret, blendargs = -1;
    
    static char *keys[] = { "color", "rect", "blendargs", NULL };
    if (!PyArg_ParseTupleAndKeywords (args, kwds, "O|Oi", keys, &color,
            &dstrect, &blendargs))
        return NULL;
        
    if (!PyColor_Check (color))
    {
        PyErr_SetString (PyExc_TypeError, "color must be a Color");
        return NULL;
    }
    if (dstrect && !SDLRectFromRect (dstrect, &rect))
    {
        PyErr_Clear ();
        PyErr_SetString (PyExc_TypeError, "rect must be a Rect");
        return NULL;
    }
    
    surface = PySDLSurface_AsSDLSurface (self);
    if (!dstrect)
    {
        rect.x = rect.y = 0;
        rect.w = surface->w;
        rect.h = surface->h;
    }
    
    col = (Uint32) PyColor_AsNumber (color);
    ARGB2FORMAT (col, surface->format);

    Py_BEGIN_ALLOW_THREADS;
    if (blendargs == -1)
        ret = SDL_FillRect (surface, &rect, col);
    else
        ret =  pyg_sdlsurface_fill_blend (surface, &rect, col, blendargs);
    Py_END_ALLOW_THREADS;
        
    if (ret == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_surface_save (PyObject *self, PyObject *args)
{
    SDL_Surface *surface;
    PyObject *file;
    char *type = NULL;
    int retval;
    SDL_RWops *rw;
    int autoclose;

    if (!PyArg_ParseTuple (args, "O|s", &file, &type))
        return NULL;

    surface = PySDLSurface_AsSDLSurface (self);

    rw = PyRWops_NewRW_Threaded (file, &autoclose);
    if (!rw)
        return NULL;

    if (IsTextObj (file) && !type)
    {
        char *filename;
        size_t len;
        PyObject *tmp;

        if (!UTF8FromObj (file, &filename, &tmp))
            return NULL;
        Py_XDECREF (tmp);

        len = strlen (filename);
        if (len < 4)
        {
            PyErr_SetString (PyExc_PyGameError, "unknown file type");
            return NULL;
        }
        type = filename + (len - 3);
    }

    Py_BEGIN_ALLOW_THREADS;
    retval = pyg_sdlsurface_save_rw (surface, rw, type, autoclose);
    Py_END_ALLOW_THREADS;

    if (!autoclose)
        PyRWops_Close (rw, autoclose);
    
    if (!retval)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_surface_scroll (PyObject *self, PyObject *args, PyObject *kwds)
{
    int dx = 0, dy = 0;
    SDL_Surface *surface;
    
    static char *keys[] = { "dx", "dy", NULL };
    if (!PyArg_ParseTupleAndKeywords (args, kwds, "|ii", keys, &dx, &dy))
        return NULL;

    surface = PySDLSurface_AsSDLSurface (self);

    if (!pyg_sdlsurface_scroll (surface, dx, dy))
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
                    
}

/* C API */
static void
_release_c_lock (void *ptr)
{
    SDLSurfaceLock* c_lock = (SDLSurfaceLock*) ptr;

    SDL_UnlockSurface (((PySDLSurface*)c_lock->surface)->surface);
    Py_XDECREF (c_lock->surface);
    Py_XDECREF (c_lock->lockobj);
    PyMem_Free (c_lock);
}

PyObject*
PySDLSurface_New (int w, int h)
{
    SDL_Surface *sf, *video;
    PyObject *surface;

    ASSERT_VIDEO_INIT (NULL);
    video = SDL_GetVideoSurface ();
    if (!video)
    {
        PyErr_SetString (PyExc_PyGameError, "display surface not set");
        return NULL;
    }

    sf = SDL_CreateRGBSurface (video->flags, w, h, video->format->BitsPerPixel,
        video->format->Rmask, video->format->Gmask, video->format->Bmask,
        video->format->Amask);
    if (!sf)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }

    surface = (PyObject*) PySDLSurface_Type.tp_new
        (&PySDLSurface_Type, NULL, NULL);
    if (!surface)
    {
        SDL_FreeSurface (sf);
        return NULL;
    }
    ((PySDLSurface*)surface)->surface = sf;
    return surface;
}

PyObject*
PySDLSurface_NewFromSDLSurface (SDL_Surface *sf)
{
    PySDLSurface *surface;
    if (!sf)
    {
        PyErr_SetString (PyExc_ValueError, "sf must not be NULL");
        return NULL;
    }
    surface = (PySDLSurface*) PySDLSurface_Type.tp_new
        (&PySDLSurface_Type, NULL, NULL);
    if (!surface)
        return NULL;

    surface->surface = sf;
    return (PyObject*) surface;
}

int
PySDLSurface_AddRefLock (PyObject *surface, PyObject *lock)
{
    PySDLSurface *sf = (PySDLSurface*)surface;
    PyObject *wkref;

    if (!surface || !PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return 0;
    }
    if (!lock)
    {
        PyErr_SetString (PyExc_TypeError, "lock must not be NULL");
        return 0;
    }

    if (!sf->locklist)
    {
        sf->locklist = PyList_New (0);
        if (!sf->locklist)
            return 0;
    }

    if (SDL_LockSurface (sf->surface) == -1)
        return 0;

    wkref = PyWeakref_NewRef (lock, NULL);
    if (!wkref)
        return 0;

    if (PyList_Append (sf->locklist, wkref) == -1)
    {
        SDL_UnlockSurface (sf->surface);
        Py_DECREF (wkref);
        return 0;
    }
    Py_DECREF (wkref);

    return 1;
}

int
PySDLSurface_RemoveRefLock (PyObject *surface, PyObject *lock)
{
    PySDLSurface *sf = (PySDLSurface*)surface;
    PyObject *ref, *item;
    Py_ssize_t size;
    int found = 0, noerror = 1;

    if (!lock)
    {
        PyErr_SetString (PyExc_TypeError, "lock must not be NULL");
        return 0;
    }

    if (!surface || !PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return 0;
    }

    if (!sf->locklist)
    {
        PyErr_SetString (PyExc_ValueError, "no locks are hold by the object");
        return 0;
    }

    size = PyList_Size (sf->locklist);
    if (size == 0)
    {
        PyErr_SetString (PyExc_ValueError, "no locks are hold by the object");
        return 0;
    }
    
    while (--size >= 0)
    {
        ref = PyList_GET_ITEM (sf->locklist, size);
        item = PyWeakref_GET_OBJECT (ref);
        if (item == lock)
        {
            if (PySequence_DelItem (sf->locklist, size) == -1)
                return 0;
            found++;
        }
        else if (item == Py_None)
        {
            /* Clear dead references */
            if (PySequence_DelItem (sf->locklist, size) != -1)
                found++;
            else
                noerror = 0;
        }
    }
    if (!found)
        return noerror;

    /* Release all locks on the surface.
     * In case we are deallocating the surface, sf->surface may become
     * invalid, then skip the unlocking process. */
    while (found > 0 && sf->surface)
    {
        SDL_UnlockSurface (sf->surface);
        found--;
    }
    return noerror;
}

PyObject*
PySDLSurface_AcquireLockObj (PyObject *surface, PyObject *lock)
{
    PyObject *cobj;
    SDLSurfaceLock *c_lock;

    if (!lock)
    {
        PyErr_SetString (PyExc_TypeError, "lock must not be NULL");
        return 0;
    }

    if (!surface || !PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return 0;
    }

    c_lock = PyMem_New (SDLSurfaceLock, 1);
    if (!c_lock)
        return NULL;

    Py_INCREF (surface);
    Py_XINCREF (lock);
    c_lock->surface = surface;
    c_lock->lockobj = lock;

    if (SDL_LockSurface (((PySDLSurface*)surface)->surface) == -1)
    {
        PyMem_Free (c_lock);
        Py_DECREF (surface);
        Py_XDECREF (lock);
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }

    cobj = PyCObject_FromVoidPtr (c_lock, _release_c_lock);
    if (!cobj)
    {
        SDL_UnlockSurface (((PySDLSurface*)surface)->surface);
        PyMem_Free (c_lock);
        Py_DECREF (surface);
        Py_XDECREF (lock);
        return NULL;
    }

    return cobj;
}

PyObject*
PySDLSurface_Copy (PyObject *source)
{
    PyObject *surfobj;
    SDL_Surface *surface, *newsurface;

    if (!source || !PySDLSurface_Check (source))
    {
        PyErr_SetString (PyExc_TypeError, "source must be a Surface");
        return NULL;
    }

    surface = ((PySDLSurface*)source)->surface;

    /* TODO: does this really copy anything? */
    newsurface = SDL_ConvertSurface (surface, surface->format, surface->flags);
    if (!newsurface)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    surfobj = PySDLSurface_NewFromSDLSurface (newsurface);
    if (!surfobj)
    {
        SDL_FreeSurface (newsurface);
        return NULL;
    }
    return surfobj;
}

void
surface_export_capi (void **capi)
{
    capi[PYGAME_SDLSURFACE_FIRSTSLOT] = &PySDLSurface_Type;
    capi[PYGAME_SDLSURFACE_FIRSTSLOT+1] = (void *)PySDLSurface_New;
    capi[PYGAME_SDLSURFACE_FIRSTSLOT+2] =
        (void *)PySDLSurface_NewFromSDLSurface;
    capi[PYGAME_SDLSURFACE_FIRSTSLOT+3] = (void *)PySDLSurface_Copy;
    capi[PYGAME_SDLSURFACE_FIRSTSLOT+4] = (void *)PySDLSurface_AddRefLock;
    capi[PYGAME_SDLSURFACE_FIRSTSLOT+5] = (void *)PySDLSurface_RemoveRefLock;
    capi[PYGAME_SDLSURFACE_FIRSTSLOT+6] = (void *)PySDLSurface_AcquireLockObj;
}
