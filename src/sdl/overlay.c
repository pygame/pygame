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
#define PYGAME_SDLOVERLAY_INTERNAL

#include "videomod.h"
#include "pgsdl.h"
#include "sdlvideo_doc.h"

static PyObject* _overlay_new (PyTypeObject *type, PyObject *args,
    PyObject *kwds);
static int _overlay_init (PyObject *overlay, PyObject *args, PyObject *kwds);
static void _overlay_dealloc (PyOverlay *overlay);

static PyObject* _overlay_getdict (PyObject *self, void *closure);
static PyObject* _overlay_getformat (PyObject *self, void *closure);
static PyObject* _overlay_getwidth (PyObject *self, void *closure);
static PyObject* _overlay_getheight (PyObject *self, void *closure);
static PyObject* _overlay_getsize (PyObject *self, void *closure);
static PyObject* _overlay_getplanes (PyObject *self, void *closure);
static PyObject* _overlay_getpitches (PyObject *self, void *closure);
static PyObject* _overlay_getpixels (PyObject *self, void *closure);
static PyObject* _overlay_gethwoverlay (PyObject *self, void *closure);
static PyObject* _overlay_getlocked (PyObject *self, void *closure);

static PyObject* _overlay_lock (PyObject *self);
static PyObject* _overlay_unlock (PyObject *self);
static PyObject* _overlay_display (PyObject *self, PyObject *args);

/**
 */
static PyMethodDef _overlay_methods[] = {
    { "lock", (PyCFunction) _overlay_lock, METH_NOARGS,
      DOC_VIDEO_OVERLAY_LOCK },
    { "unlock", (PyCFunction) _overlay_unlock, METH_NOARGS,
      DOC_VIDEO_OVERLAY_UNLOCK },
    { "display", _overlay_display , METH_VARARGS,
      DOC_VIDEO_OVERLAY_DISPLAY },
    { NULL, NULL, 0, NULL }
};

/**
 */
static PyGetSetDef _overlay_getsets[] = {
    { "__dict__", _overlay_getdict, NULL, "", NULL },
    { "format", _overlay_getformat, NULL, DOC_VIDEO_OVERLAY_FORMAT, NULL },
    { "w", _overlay_getwidth, NULL, DOC_VIDEO_OVERLAY_W, NULL },
    { "width", _overlay_getwidth, NULL, DOC_VIDEO_OVERLAY_WIDTH, NULL },
    { "h", _overlay_getheight, NULL, DOC_VIDEO_OVERLAY_H, NULL },
    { "height", _overlay_getheight, NULL, DOC_VIDEO_OVERLAY_HEIGHT, NULL },
    { "size", _overlay_getsize, NULL, DOC_VIDEO_OVERLAY_SIZE, NULL },
    { "planes", _overlay_getplanes, NULL, DOC_VIDEO_OVERLAY_PLANES, NULL },
    { "pitches", _overlay_getpitches, NULL, DOC_VIDEO_OVERLAY_PITCHES, NULL },
    { "pixels", _overlay_getpixels, NULL, DOC_VIDEO_OVERLAY_PIXELS, NULL },
    { "hw_overlay", _overlay_gethwoverlay, NULL, DOC_VIDEO_OVERLAY_HW_OVERLAY,
      NULL },
    { "locked", _overlay_getlocked, NULL, DOC_VIDEO_OVERLAY_LOCKED, NULL },
    { NULL, NULL, NULL, NULL, NULL }
};

/**
 */
PyTypeObject PyOverlay_Type =
{
    TYPE_HEAD(NULL, 0)
    "video.Overlay",         /* tp_name */
    sizeof (PyOverlay),      /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor) _overlay_dealloc, /* tp_dealloc */
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
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_WEAKREFS,
    DOC_VIDEO_OVERLAY,
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    offsetof (PyOverlay, weakrefs), /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    _overlay_methods,           /* tp_methods */
    0,                          /* tp_members */
    _overlay_getsets,           /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    offsetof (PyOverlay, dict), /* tp_dictoffset */
    (initproc) _overlay_init,   /* tp_init */
    0,                          /* tp_alloc */
    _overlay_new,               /* tp_new */
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
_overlay_dealloc (PyOverlay *self)
{
    Py_XDECREF (self->dict);
    Py_XDECREF (self->locklist);
    if (self->weakrefs)
        PyObject_ClearWeakRefs ((PyObject *) self);

    if (self->overlay)
        SDL_FreeYUVOverlay (self->overlay);

    ((PyObject*)self)->ob_type->tp_free ((PyObject *) self);
}

static PyObject*
_overlay_new (PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyOverlay *overlay = (PyOverlay *)type->tp_alloc (type, 0);
    if (!overlay)
        return NULL;
    overlay->locklist = NULL;
    overlay->dict = NULL;
    overlay->weakrefs = NULL;
    return (PyObject*) overlay;
}

static int
_overlay_init (PyObject *self, PyObject *args, PyObject *kwds)
{
    int width, height;
    Uint32 format;
    PyObject *surf, *size;
    SDL_Overlay *overlay;
    SDL_Surface *surface;

    ASSERT_VIDEO_INIT(-1);
    
    if (!PyArg_ParseTuple (args, "OOl", &surf, &size, &format))
    {
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "Oiil", &surf, &width, &height, &format))
            return -1;
    }
    else
    {
        if (!SizeFromObject (size, (pgint32*)&width, (pgint32*)&height))
            return -1;
    }
    
    if (!PySDLSurface_Check (surf))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return -1;
    }
    if (width < 0 || height < 0)
    {
        PyErr_SetString (PyExc_ValueError,
            "width and height must not be negative");
        return -1;
    }

    surface = ((PySDLSurface*)surf)->surface;
    overlay = SDL_CreateYUVOverlay (width, height, format, surface);
    if (!overlay)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return -1;
    }

    ((PyOverlay*)self)->surface = surf;
    ((PyOverlay*)self)->overlay = overlay;
    ((PyOverlay*)self)->locklist = NULL;
    ((PyOverlay*)self)->dict = NULL;
    ((PyOverlay*)self)->weakrefs = NULL;
    Py_INCREF (surf);

    return 0;
}

/* Overlay getters/setters */
static PyObject*
_overlay_getdict (PyObject *self, void *closure)
{
    PyOverlay *overlay = (PyOverlay*) self;
    if (!overlay->dict)
    {
        overlay->dict = PyDict_New ();
        if (!overlay->dict)
            return NULL;
    }
    Py_INCREF (overlay->dict);
    return overlay->dict;
}

static PyObject*
_overlay_getformat (PyObject *self, void *closure)
{
    SDL_Overlay *overlay = ((PyOverlay*)self)->overlay;
    ASSERT_VIDEO_INIT(NULL);
    
    return PyLong_FromUnsignedLong (overlay->format);
}

static PyObject*
_overlay_getwidth (PyObject *self, void *closure)
{
    SDL_Overlay *overlay = ((PyOverlay*)self)->overlay;
    ASSERT_VIDEO_INIT(NULL);
    
    return PyInt_FromLong (overlay->w);
}

static PyObject*
_overlay_getheight (PyObject *self, void *closure)
{
    SDL_Overlay *overlay = ((PyOverlay*)self)->overlay;
    ASSERT_VIDEO_INIT(NULL);
    
    return PyInt_FromLong (overlay->h);
}

static PyObject*
_overlay_getsize (PyObject *self, void *closure)
{
    SDL_Overlay *overlay = ((PyOverlay*)self)->overlay;
    ASSERT_VIDEO_INIT(NULL);
    
    return Py_BuildValue ("(ii)", overlay->w, overlay->h);
}

static PyObject*
_overlay_getplanes (PyObject *self, void *closure)
{
    SDL_Overlay *overlay = ((PyOverlay*)self)->overlay;
    ASSERT_VIDEO_INIT(NULL);
    
    return PyInt_FromLong (overlay->planes);
}

static PyObject*
_overlay_getpitches (PyObject *self, void *closure)
{
    PyObject *tuple;
    Py_ssize_t count, i;
    
    SDL_Overlay *overlay = ((PyOverlay*)self)->overlay;
    ASSERT_VIDEO_INIT(NULL);
    
    count = (Py_ssize_t) overlay->planes;
    tuple = PyTuple_New (count);
    if (!tuple)
        return NULL;
    
    for (i = 0; i < count; i++)
        PyTuple_SET_ITEM (tuple, i, PyInt_FromLong (overlay->pitches[i]));
    return tuple;
}

static PyObject*
_overlay_getpixels (PyObject *self, void *closure)
{
    PyObject *buffer, *tuple;
    Py_ssize_t count, i;
    
    SDL_Overlay *overlay = ((PyOverlay*)self)->overlay;
    ASSERT_VIDEO_INIT(NULL);
    
    count = (Py_ssize_t) overlay->planes;
    tuple = PyTuple_New (count);
    if (!tuple)
        return NULL;

    for (i = 0; i < count; i++)
    {
        buffer = PyBufferProxy_New (NULL, NULL, 0, PyOverlay_RemoveRefLock);
        if (!buffer)
        {
            Py_DECREF (tuple);
            return NULL;
        }
        if (!PyOverlay_AddRefLock (self, buffer))
        {
            ((PyBufferProxy*)buffer)->unlock_func = NULL;
            Py_DECREF (buffer);
            Py_DECREF (tuple);
            return NULL;
        }
        ((PyBufferProxy*)buffer)->object = self;
        ((PyBufferProxy*)buffer)->buffer = overlay->pixels[i];
        ((PyBufferProxy*)buffer)->length =
            (Py_ssize_t) overlay->pitches[i] * overlay->h;
        PyTuple_SET_ITEM (tuple, i, buffer);
    }
    return tuple;
}

static PyObject*
_overlay_gethwoverlay (PyObject *self, void *closure)
{
    SDL_Overlay *overlay = ((PyOverlay*)self)->overlay;
    ASSERT_VIDEO_INIT(NULL);
    
    return PyBool_FromLong (overlay->hw_overlay);
}

static PyObject*
_overlay_getlocked (PyObject *self, void *closure)
{
    PyOverlay *overlay = (PyOverlay*) self;

    if (overlay->locklist && PyList_Size (overlay->locklist) != 0)
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

/* Overlay methods */
static PyObject*
_overlay_lock (PyObject *self)
{
    SDL_Overlay *overlay = ((PyOverlay*)self)->overlay;
    ASSERT_VIDEO_INIT(NULL);
    
    if (SDL_LockYUVOverlay (overlay) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_overlay_unlock (PyObject *self)
{
    SDL_Overlay *overlay = ((PyOverlay*)self)->overlay;
    ASSERT_VIDEO_INIT(NULL);

    SDL_UnlockYUVOverlay (overlay);
    Py_RETURN_NONE;
}

static PyObject*
_overlay_display (PyObject *self, PyObject *args)
{
    SDL_Rect sdlrect;
    PyObject *rect = NULL;
    SDL_Overlay *overlay = ((PyOverlay*)self)->overlay;
    ASSERT_VIDEO_INIT(NULL);

    if (!PyArg_ParseTuple (args, "|O:display", &rect))
        return NULL;
    
    if (rect && !SDLRect_FromRect (rect, &sdlrect))
        return NULL;
    if (!rect)
    {
        sdlrect.x = sdlrect.y = 0;
        sdlrect.w = overlay->w;
        sdlrect.h = overlay->h;
    }
    if (SDL_DisplayYUVOverlay (overlay, &sdlrect) != 0)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

/* C API */
PyObject*
PyOverlay_New (PyObject *surface, int width, int height, Uint32 format)
{
    PyObject *overlay;
    SDL_Surface *sdlsurface;
    SDL_Overlay *sdloverlay;

    if (!surface || !PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (width < 0 || height < 0)
    {
        PyErr_SetString (PyExc_ValueError,
            "width and height must not be negative");
        return NULL;
    }

    sdlsurface = ((PySDLSurface*)surface)->surface;
    sdloverlay = SDL_CreateYUVOverlay (width, height, format, sdlsurface);
    if (!sdloverlay)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }

    overlay = PyOverlay_Type.tp_new (&PyOverlay_Type, NULL, NULL);
    if (!overlay)
    {
        SDL_FreeYUVOverlay (sdloverlay);
        return NULL;
    }
    ((PyOverlay*)overlay)->overlay = sdloverlay;
    ((PyOverlay*)overlay)->surface = surface;
    Py_INCREF (surface);
    return overlay;
}

int
PyOverlay_AddRefLock (PyObject *overlay, PyObject *lock)
{
    PyOverlay *ov = (PyOverlay*)overlay;
    PyObject *wkref;

    if (!overlay || !PyOverlay_Check (overlay))
    {
        PyErr_SetString (PyExc_TypeError, "overlay must be an Overlay");
        return 0;
    }
    if (!lock)
    {
        PyErr_SetString (PyExc_TypeError, "lock must not be NULL");
        return 0;
    }

    if (!ov->locklist)
    {
        ov->locklist = PyList_New (0);
        if (!ov->locklist)
            return 0;
    }

    if (SDL_LockYUVOverlay (ov->overlay) == -1)
        return 0;

    wkref = PyWeakref_NewRef (lock, NULL);
    if (!wkref)
        return 0;

    if (PyList_Append (ov->locklist, wkref) == -1)
    {
        SDL_UnlockYUVOverlay (ov->overlay);
        Py_DECREF (wkref);
        return 0;
    }
    Py_DECREF (wkref);

    return 1;
}

int
PyOverlay_RemoveRefLock (PyObject *overlay, PyObject *lock)
{
    PyOverlay *ov = (PyOverlay*)overlay;
    PyObject *ref, *item;
    Py_ssize_t size;
    int found = 0, noerror = 1;

    if (!lock)
    {
        PyErr_SetString (PyExc_TypeError, "lock must not be NULL");
        return 0;
    }

    if (!overlay || !PyOverlay_Check (overlay))
    {
        PyErr_SetString (PyExc_TypeError, "overlay must be an Overlay");
        return 0;
    }

    if (!ov->locklist)
    {
        PyErr_SetString (PyExc_ValueError, "no locks are hold by the object");
        return 0;
    }

    size = PyList_Size (ov->locklist);
    if (size == 0)
    {
        PyErr_SetString (PyExc_ValueError, "no locks are hold by the object");
        return 0;
    }
    
    while (--size >= 0)
    {
        ref = PyList_GET_ITEM (ov->locklist, size);
        item = PyWeakref_GET_OBJECT (ref);
        if (item == lock)
        {
            if (PySequence_DelItem (ov->locklist, size) == -1)
                return 0;
            found++;
        }
        else if (item == Py_None)
        {
            /* Clear dead references */
            if (PySequence_DelItem (ov->locklist, size) != -1)
                found++;
            else
                noerror = 0;
        }
    }
    if (!found)
        return noerror;

    /* Release all locks on the overlay */
    while (found > 0)
    {
        SDL_UnlockYUVOverlay (ov->overlay);
        found--;
    }
    return noerror;
}

void
overlay_export_capi (void **capi)
{
    capi[PYGAME_SDLOVERLAY_FIRSTSLOT] = &PyOverlay_Type;
    capi[PYGAME_SDLOVERLAY_FIRSTSLOT+1] = PyOverlay_New;
    capi[PYGAME_SDLOVERLAY_FIRSTSLOT+2] = PyOverlay_AddRefLock;
    capi[PYGAME_SDLOVERLAY_FIRSTSLOT+3] = PyOverlay_RemoveRefLock;
}
