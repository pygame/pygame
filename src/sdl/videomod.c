/*
  pygame - Python Game Library
  Copyright (C) 2000-2001 Pete Shinners, 2008 Marcus von Appen

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
#define PYGAME_SDLVIDEO_INTERNAL

#include "videomod.h"
#include "pgsdl.h"

static void _quit (void);
static int _seq_to_uint16 (PyObject *seq, Uint16 *array, Py_ssize_t array_size);

static PyObject* _sdl_videoinit (PyObject *self);
static PyObject* _sdl_videowasinit (PyObject *self);
static PyObject* _sdl_videoquit (PyObject *self);
static PyObject* _sdl_getvideosurface (PyObject *self);
static PyObject* _sdl_getvideoinfo (PyObject *self);
static PyObject* _sdl_videodrivername (PyObject *self);
static PyObject* _sdl_setgamma (PyObject *self, PyObject *args);
static PyObject* _sdl_getgammaramp (PyObject *self);
static PyObject* _sdl_setgammaramp (PyObject *self, PyObject *args);
static PyObject* _sdl_videomodeok (PyObject *self, PyObject *args);
static PyObject* _sdl_listmodes (PyObject *self, PyObject *args,
    PyObject *kwds);
static PyObject* _sdl_setvideomode (PyObject *self, PyObject *args,
    PyObject *kwds);

static PyMethodDef _video_methods[] = {
    { "init", (PyCFunction) _sdl_videoinit, METH_NOARGS, "" },
    { "was_init", (PyCFunction) _sdl_videowasinit, METH_NOARGS, "" },
    { "quit", (PyCFunction) _sdl_videoquit, METH_NOARGS, "" },
    { "get_videosurface", (PyCFunction)_sdl_getvideosurface, METH_NOARGS, "" },
    { "get_info", (PyCFunction)_sdl_getvideoinfo, METH_NOARGS, "" },
    { "get_drivername", (PyCFunction)_sdl_videodrivername, METH_NOARGS, "" },
    { "set_gamma", _sdl_setgamma, METH_VARARGS, "" },
    { "get_gammaramp", (PyCFunction)_sdl_getgammaramp, METH_NOARGS, "" },
    { "set_gammaramp", _sdl_setgammaramp, METH_VARARGS, "" },
    { "is_mode_ok", _sdl_videomodeok, METH_VARARGS, "" },
    { "list_modes", (PyCFunction) _sdl_listmodes, METH_VARARGS | METH_KEYWORDS,
      "" },
    { "set_mode", (PyCFunction)_sdl_setvideomode, METH_VARARGS | METH_KEYWORDS,
      "" },
    { NULL, NULL, 0, NULL }
};

static void
_quit (void)
{
    if (SDL_WasInit (SDL_INIT_VIDEO))
        SDL_QuitSubSystem (SDL_INIT_VIDEO);
}

static int
_seq_to_uint16 (PyObject *seq, Uint16 *array, Py_ssize_t array_size)
{
    Py_ssize_t count, i;
    Uint16 v;

    if (!PySequence_Check (seq))
    {
        PyErr_SetString (PyExc_TypeError, "array must be a sequence");
        return 0;
    }

    count = PySequence_Size (seq);
    if (count != array_size)
    {
        PyErr_SetString (PyExc_ValueError, "array does not match needed size");
        return 0;
    }

    for (i = 0; i < count; i++)
    {
        if (!Uint16FromSeqIndex (seq, i, &v))
            return 0;
        array[i] = v;
    }
    return 1;
}

static PyObject*
_sdl_videoinit (PyObject *self)
{
    if (SDL_WasInit (SDL_INIT_VIDEO))
        Py_RETURN_NONE;
    if (SDL_InitSubSystem (SDL_INIT_VIDEO) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    /* Enable unicode by default */
    SDL_EnableUNICODE (1);

    Py_RETURN_NONE;
}

static PyObject*
_sdl_videowasinit (PyObject *self)
{
    if (SDL_WasInit (SDL_INIT_VIDEO))
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject*
_sdl_videoquit (PyObject *self)
{
    _quit ();
    Py_RETURN_NONE;
}

static PyObject*
_sdl_getvideosurface (PyObject *self)
{
    SDL_Surface *surface;

    ASSERT_VIDEO_INIT(NULL);
    
    surface = SDL_GetVideoSurface ();
    if (!surface)
        Py_RETURN_NONE;
    return PySDLSurface_NewFromSDLSurface (surface);
}

static PyObject*
_sdl_getvideoinfo (PyObject *self)
{
    const SDL_VideoInfo *info;
    PyObject *vfmt, *dict, *val;

    ASSERT_VIDEO_INIT(NULL);

    dict = PyDict_New ();
    if (!dict)
        return NULL;

    info = SDL_GetVideoInfo ();
    vfmt = PyPixelFormat_NewFromSDLPixelFormat (info->vfmt);
    if (!vfmt)
    {
        Py_DECREF (dict);
        return NULL;
    }

    val = PyInt_FromLong (info->hw_available);
    PyDict_SetItemString (dict, "hw_available", val);
    Py_DECREF (val);

    val = PyInt_FromLong (info->wm_available);
    PyDict_SetItemString (dict, "wm_available", val);
    Py_DECREF (val);

    val = PyInt_FromLong (info->blit_hw);
    PyDict_SetItemString (dict, "blit_hw", val);
    Py_DECREF (val);

    val = PyInt_FromLong (info->blit_hw_CC);
    PyDict_SetItemString (dict, "blit_hw_CC", val);
    Py_DECREF (val);

    val = PyInt_FromLong (info->blit_hw_A);
    PyDict_SetItemString (dict, "blit_hw_A", val);
    Py_DECREF (val);

    val = PyInt_FromLong (info->blit_sw);
    PyDict_SetItemString (dict, "blit_sw", val);
    Py_DECREF (val);

    val = PyInt_FromLong (info->blit_sw_CC);
    PyDict_SetItemString (dict, "blit_sw_CC", val);
    Py_DECREF (val);

    val = PyInt_FromLong (info->blit_sw_A);
    PyDict_SetItemString (dict, "blit_sw_A", val);
    Py_DECREF (val);

    val = PyInt_FromLong (info->blit_fill);
    PyDict_SetItemString (dict, "blit_fill", val);
    Py_DECREF (val);

    val = PyInt_FromLong ((long)info->video_mem);
    PyDict_SetItemString (dict, "video_mem", val);
    Py_DECREF (val);

    PyDict_SetItemString (dict, "vfmt", vfmt);
    Py_DECREF (vfmt);
    return dict;
}

static PyObject*
_sdl_videodrivername (PyObject *self)
{
    char buf[256];
    if (!SDL_VideoDriverName (buf, sizeof (buf)))
        Py_RETURN_NONE;
    return Text_FromUTF8 (buf);
}

static PyObject*
_sdl_setgamma (PyObject *self, PyObject *args)
{
    float r,g,b;

    ASSERT_VIDEO_INIT(NULL);

    if (!PyArg_ParseTuple (args, "fff:set_gamma", &r, &g, &b))
        return NULL;

    if (SDL_SetGamma (r, g, b) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *_sdl_getgammaramp (PyObject *self)
{
    PyObject *rr, *rg, *rb;
    Uint16 r[256], g[256], b[256];
    int i;

    ASSERT_VIDEO_INIT(NULL);
  
    rr = PyTuple_New (256);
    if (!rr)
        return NULL;
    rg = PyTuple_New (256);
    if (!rg)
    {
        Py_DECREF (rr);
        return NULL;
    }
    rb = PyTuple_New (256);
    if (!rb)
    {
        Py_DECREF (rr);
        Py_DECREF (rg);
        return NULL;
    }

    if (SDL_GetGammaRamp (r, g, b) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    for (i = 0; i < 256; i++)
    {
        PyTuple_SET_ITEM (rr, i, PyInt_FromLong (r[i])); 
        PyTuple_SET_ITEM (rg, i, PyInt_FromLong (g[i])); 
        PyTuple_SET_ITEM (rb, i, PyInt_FromLong (b[i])); 
    }
    return Py_BuildValue ("(NNN)", rr, rg, rb);
}


static PyObject*
_sdl_setgammaramp (PyObject *self, PyObject *args)
{
    PyObject *rr, *rg, *rb;
    Uint16 r[256], g[256], b[256];
    Uint16 *rp, *gp, *bp;

    ASSERT_VIDEO_INIT(NULL);

    if (!PyArg_ParseTuple (args, "OOO:set_gammaramp", &rr, &rg, &rb))
        return NULL;

    if (rr != Py_None && !_seq_to_uint16 (rr, r, 256))
        return NULL;
    if (rg != Py_None && !_seq_to_uint16 (rg, g, 256))
        return NULL;
    if (rb != Py_None && !_seq_to_uint16 (rb, b, 256))
        return NULL;

    rp = (rr == Py_None) ? NULL : r;
    gp = (rg == Py_None) ? NULL : g;
    bp = (rb == Py_None) ? NULL : b;

    if (SDL_SetGammaRamp (rp, gp, bp) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_sdl_videomodeok (PyObject *self, PyObject *args)
{
    int width, height;
    int bpp;
    Uint32 flags;
    int retbpp;

    ASSERT_VIDEO_INIT(NULL);

    if (!PyArg_ParseTuple (args, "iiil:is_mode_ok", &width, &height, &bpp,
            &flags))
    {
        PyObject *size;
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "Oil:is_mode_ok", &size, &bpp, &flags))
            return NULL;
        if (!SizeFromObject (size, (pgint32*)&width, (pgint32*)&height))
            return NULL;
    }

    retbpp = SDL_VideoModeOK (width, height, bpp, flags);
    if (!retbpp)
        Py_RETURN_FALSE;
    if (retbpp == bpp)
        Py_RETURN_TRUE;
    return PyInt_FromLong (retbpp);
}

static PyObject*
_sdl_listmodes (PyObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *pxfmt = NULL;
    Uint32 flags = 0;
    SDL_Rect **modes;
    int i;
    PyObject *list, *rect;

    static char *kwlist[] = { "format", "flags", NULL };
    ASSERT_VIDEO_INIT(NULL);
    
    if (!PyArg_ParseTupleAndKeywords (args, kwds, "|Ol:list_modes", kwlist,
        &pxfmt, &flags))
        return NULL;
    
    if (pxfmt && !PyPixelFormat_Check (pxfmt))
    {
        PyErr_SetString (PyExc_TypeError, "format must be a PixelFormat");
        return NULL;
    }

    if (pxfmt)
        modes = SDL_ListModes (((PyPixelFormat*)pxfmt)->format, flags);
    else
        modes = SDL_ListModes (NULL, flags);

    if (modes == NULL)
        Py_RETURN_NONE; /* No format applicable */

    list = PyList_New (0);
    if (!list)
        return NULL;

    if (modes == (SDL_Rect**)-1)
        return list; /* All modes okay */

    for (i = 0; modes[i]; i++)
    {
        rect = PyRect_New (modes[i]->x, modes[i]->y, modes[i]->w, modes[i]->h);
        if (!rect)
        {
            Py_DECREF (list);
            return NULL;
        }

        if (PyList_Append (list, rect) == -1)
        {
            Py_DECREF (rect);
            Py_DECREF (list);
            return NULL;
        }
        Py_DECREF (rect);
    }
    return list;
}

static PyObject*
_sdl_setvideomode (PyObject *self, PyObject *args, PyObject *kwds)
{
    int width, height;
    int bpp = 32;
    Uint32 flags = 0;
    SDL_Surface *surface;
    PyObject *sf;

    static char *kwlist[] = { "width", "height", "bpp", "flags", NULL };
    
    /* Not necessary usually. SDL_SetVideoMode() seems to do that
     * implicitly for recent versions of SDL. Though we'll force users
     * to do it explicitly. */
    ASSERT_VIDEO_INIT(NULL);

    if (!PyArg_ParseTupleAndKeywords (args, kwds, "ii|il:set_mode", kwlist,
        &width, &height, &bpp, &flags))
    {
        PyObject *size;
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "O|il:set_mode", &size, &bpp, &flags))
            return NULL;
        if (!SizeFromObject (size, (pgint32*)&width, (pgint32*)&height))
            return NULL;
    }

    Py_BEGIN_ALLOW_THREADS;
    surface = SDL_SetVideoMode (width, height, bpp, flags);
    Py_END_ALLOW_THREADS;
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

/* C API */
int
ColorFromObj (PyObject *value, SDL_PixelFormat *format, Uint32 *color)
{
    Uint8 rgba[4];

    if (!value || !format || !color)
        return 0;

    if (PyInt_Check (value))
    {
        long intval = PyInt_AsLong (value);
        if (intval == -1 && PyErr_Occurred ())
        {
            PyErr_Clear ();
            PyErr_SetString (PyExc_ValueError, "invalid color argument");
            return 0;
        }
        *color = (Uint32) intval;
        return 1;
    }
    else if (PyLong_Check (value))
    {
        unsigned long longval = PyLong_AsUnsignedLong (value);
        if (PyErr_Occurred ())
        {
            PyErr_Clear ();
            PyErr_SetString(PyExc_ValueError, "invalid color argument");
            return 0;
        }
        *color = (Uint32) longval;
        return 1;
    }
    else if (PyColor_Check (value))
    {
        rgba[0] = ((PyColor*)value)->r;
        rgba[1] = ((PyColor*)value)->g;
        rgba[2] = ((PyColor*)value)->b;
        rgba[3] = ((PyColor*)value)->a;

        *color = (Uint32) SDL_MapRGBA
            (format, rgba[0], rgba[1], rgba[2], rgba[3]);
        return 1;
    }
    else
        PyErr_SetString (PyExc_ValueError, "invalid color argument");
    return 0;
}

#if PY_VERSION_HEX >= 0x03000000
PyMODINIT_FUNC PyInit_video (void)
#else
PyMODINIT_FUNC initvideo (void)
#endif
{
    PyObject *mod = NULL;
    PyObject *c_api_obj;
    static void *c_api[PYGAME_SDLVIDEO_SLOTS];
    
#if PY_VERSION_HEX >= 0x03000000
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "video",
        "",
        -1,
        _video_methods,
        NULL, NULL, NULL, NULL
    };
#endif

    if (import_pygame2_base () < 0)
        goto fail;
    if (import_pygame2_sdl_base () < 0)
        goto fail;

    /* Complete types */
    PySDLSurface_Type.tp_base = &PySurface_Type;
    if (PyType_Ready (&PySDLSurface_Type) < 0)
        goto fail;
    if (PyType_Ready (&PyOverlay_Type) < 0)
        goto fail;
    if (PyType_Ready (&PyPixelFormat_Type) < 0)
        goto fail;

    Py_INCREF (&PySDLSurface_Type);
    Py_INCREF (&PyOverlay_Type);
    Py_INCREF (&PyPixelFormat_Type);

#if PY_VERSION_HEX < 0x03000000
    mod = Py_InitModule3 ("video", _video_methods, "");
#else
    mod = PyModule_Create (&_module);
#endif
    if (!mod)
        goto fail;
    
    PyModule_AddObject (mod, "PixelFormat", (PyObject *) &PyPixelFormat_Type);
    PyModule_AddObject (mod, "Surface", (PyObject *) &PySDLSurface_Type);
    PyModule_AddObject (mod, "Overlay", (PyObject *) &PyOverlay_Type);
    
    c_api[PYGAME_SDLVIDEO_FIRSTSLOT+0] = ColorFromObj;

    pixelformat_export_capi (c_api);
    surface_export_capi (c_api);
    overlay_export_capi (c_api);

    c_api_obj = PyCObject_FromVoidPtr ((void *) c_api, NULL);
    if (c_api_obj)
        PyModule_AddObject (mod, PYGAME_SDLVIDEO_ENTRY, c_api_obj);    

    RegisterQuitCallback (_quit);
    MODINIT_RETURN(mod);

fail:
    Py_XDECREF (mod);
    MODINIT_RETURN (NULL);
}
