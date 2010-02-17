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
#define PYGAME_SDLGFXROTOZOOM_INTERNAL

#include "pgsdl.h"
#include "pggfx.h"
#include "sdlgfxrotozoom_doc.h"
#include <SDL_rotozoom.h>

static PyObject* _gfx_rotozoom (PyObject *self, PyObject *args);
static PyObject* _gfx_rotozoomxy (PyObject *self, PyObject *args);
static PyObject* _gfx_rotozoomsize (PyObject *self, PyObject *args);
static PyObject* _gfx_rotozoomsizexy (PyObject *self, PyObject *args);
static PyObject* _gfx_zoom (PyObject *self, PyObject *args);
static PyObject* _gfx_zoomsize (PyObject *self, PyObject *args);
static PyObject* _gfx_shrink (PyObject *self, PyObject *args);
static PyObject* _gfx_rotate90 (PyObject *self, PyObject *args);

static PyMethodDef _gfx_methods[] = {
    { "rotozoom", (PyCFunction) _gfx_rotozoom, METH_VARARGS,
      DOC_ROTOZOOM_ROTOZOOM },
    { "rotozoom_xy", (PyCFunction) _gfx_rotozoomxy, METH_VARARGS,
      DOC_ROTOZOOM_ROTOZOOM_XY },
    { "rotozoom_size", (PyCFunction) _gfx_rotozoomsize, METH_VARARGS,
      DOC_ROTOZOOM_ROTOZOOM_SIZE },
    { "rotozoom_size_xy", (PyCFunction) _gfx_rotozoomsizexy, METH_VARARGS,
      DOC_ROTOZOOM_ROTOZOOM_SIZE_XY },
    { "zoom", (PyCFunction) _gfx_zoom, METH_VARARGS, DOC_ROTOZOOM_ZOOM },
    { "zoom_size", (PyCFunction) _gfx_zoomsize, METH_VARARGS,
      DOC_ROTOZOOM_ZOOM_SIZE },
    { "shrink", (PyCFunction) _gfx_shrink, METH_VARARGS, DOC_ROTOZOOM_SHRINK },
    { "rotate_90", (PyCFunction) _gfx_rotate90, METH_VARARGS,
      DOC_ROTOZOOM_ROTATE_90 },
    { NULL, NULL, 0, NULL },
};

static PyObject*
_gfx_rotozoom (PyObject *self, PyObject *args)
{
    SDL_Surface *orig, *result;
    PyObject *surface, *retval, *aa = NULL;
    double angle, zoom;
    int smooth = 0;

    if (!PyArg_ParseTuple (args, "Odd|O:rotozoom", &surface, &angle, &zoom,
            &aa))
        return NULL;

    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    orig = ((PySDLSurface*)surface)->surface;

    if (aa)
    {
        smooth = PyObject_IsTrue (aa);
        if (smooth == -1)
            return NULL;
    }
    
    result = rotozoomSurface (orig, angle, zoom, smooth);
    if (!result)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    
    retval = PySDLSurface_NewFromSDLSurface (result);
    if (!retval)
    {
        SDL_FreeSurface (result);
        return NULL;
    }
    return retval;
}

static PyObject*
_gfx_rotozoomxy (PyObject *self, PyObject *args)
{
    SDL_Surface *orig, *result;
    PyObject *surface, *retval, *aa = NULL;
    double angle, zoomx, zoomy;
    int smooth = 0;

    if (!PyArg_ParseTuple (args, "Oddd|O:rotozoom_xy", &surface, &angle,
            &zoomx, &zoomy, &aa))
        return NULL;

    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    orig = ((PySDLSurface*)surface)->surface;

    if (aa)
    {
        smooth = PyObject_IsTrue (aa);
        if (smooth == -1)
            return NULL;
    }
    
    result = rotozoomSurfaceXY (orig, angle, zoomx, zoomy, smooth);
    if (!result)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    
    retval = PySDLSurface_NewFromSDLSurface (result);
    if (!retval)
    {
        SDL_FreeSurface (result);
        return NULL;
    }
    return retval;
}

static PyObject*
_gfx_rotozoomsize (PyObject *self, PyObject *args)
{
    PyObject *obj = NULL;
    int w, h, dstw, dsth;
    double angle, zoom;

    if (!PyArg_ParseTuple (args, "iidd:rotozoom_size", &w, &h, &angle, &zoom))
    {
        PyErr_Clear ();
        if (PyArg_ParseTuple (args, "Odd:rotozoom_size", &obj, &angle, &zoom))
        {
            if (PySDLSurface_Check (obj))
            {
                w = ((PySDLSurface *)obj)->surface->w;
                h = ((PySDLSurface *)obj)->surface->w;
            }
            else
            {
                if (!SizeFromObj (obj, (pgint32*)&w, (pgint32*)&h))
                    return NULL;
            }
        }
        else
            return NULL;
    }
    rotozoomSurfaceSize (w, h, angle, zoom, &dstw, &dsth);
    return Py_BuildValue ("(ii)", dstw, dsth);
}

static PyObject*
_gfx_rotozoomsizexy (PyObject *self, PyObject *args)
{
    PyObject *obj = NULL;
    int w, h, dstw, dsth;
    double angle, zoomx, zoomy;

    if (!PyArg_ParseTuple (args, "iiddd:rotozoom_size_xy", &w, &h, &angle,
            &zoomx, &zoomy))
    {
        PyErr_Clear ();
        if (PyArg_ParseTuple (args, "Oddd:rotozoom_size", &obj, &angle,
                &zoomx, &zoomy))
        {
            if (PySDLSurface_Check (obj))
            {
                w = ((PySDLSurface *)obj)->surface->w;
                h = ((PySDLSurface *)obj)->surface->w;
            }
            else
            {
                if (!SizeFromObj (obj, (pgint32*)&w, (pgint32*)&h))
                    return NULL;
            }
        }
        else
            return NULL;
    }
    rotozoomSurfaceSizeXY (w, h, angle, zoomx, zoomy, &dstw, &dsth);
    return Py_BuildValue ("(ii)", dstw, dsth);
}

static PyObject*
_gfx_zoom (PyObject *self, PyObject *args)
{
    PyObject *surface, *retval, *aa = NULL;
    SDL_Surface *orig, *result;
    double zoomx, zoomy;
    int smooth = 0;

    if (!PyArg_ParseTuple (args, "Odd|O:zoom", &surface, &zoomx, &zoomy, &aa))
        return NULL;

    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    orig = ((PySDLSurface*)surface)->surface;

    if (aa)
    {
        smooth = PyObject_IsTrue (aa);
        if (smooth == -1)
            return NULL;
    }
    
    result = zoomSurface (orig, zoomx, zoomy, smooth);
    if (!result)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    
    retval = PySDLSurface_NewFromSDLSurface (result);
    if (!retval)
    {
        SDL_FreeSurface (result);
        return NULL;
    }
    return retval;
}

static PyObject*
_gfx_zoomsize (PyObject *self, PyObject *args)
{
    PyObject *obj = NULL;
    int w, h, dstw, dsth;
    double zoomx, zoomy;

    if (!PyArg_ParseTuple (args, "iidd:zoom_size", &w, &h, &zoomx, &zoomy))
    {
        PyErr_Clear ();
        if (PyArg_ParseTuple (args, "Odd:zoom_size", &obj, &zoomx, &zoomy))
        {
            if (PySDLSurface_Check (obj))
            {
                w = ((PySDLSurface *)obj)->surface->w;
                h = ((PySDLSurface *)obj)->surface->w;
            }
            else
            {
                if (!SizeFromObj (obj, (pgint32*)&w, (pgint32*)&h))
                    return NULL;
            }
        }
        else
            return NULL;
    }
    zoomSurfaceSize (w, h, zoomx, zoomy, &dstw, &dsth);
    return Py_BuildValue ("(ii)", dstw, dsth);
}

static PyObject*
_gfx_shrink (PyObject *self, PyObject *args)
{
    PyObject *surface, *retval;
    SDL_Surface *orig, *result;
    int facx, facy;

    if (!PyArg_ParseTuple (args, "Oii:shrink", &surface, &facx, &facy))
        return NULL;

    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    orig = ((PySDLSurface*)surface)->surface;

    result = shrinkSurface (orig, facx, facy);
    if (!result)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    
    retval = PySDLSurface_NewFromSDLSurface (result);
    if (!retval)
    {
        SDL_FreeSurface (result);
        return NULL;
    }
    return retval;
}

static PyObject*
_gfx_rotate90 (PyObject *self, PyObject *args)
{
    PyObject *surface, *retval;
    SDL_Surface *orig, *result;
    int times;

    if (!PyArg_ParseTuple (args, "Oi:rotate_90", &surface, &times))
        return NULL;

    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    orig = ((PySDLSurface*)surface)->surface;

    result = rotateSurface90Degrees (orig, times);
    if (!result)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    
    retval = PySDLSurface_NewFromSDLSurface (result);
    if (!retval)
    {
        SDL_FreeSurface (result);
        return NULL;
    }
    return retval;
}

#ifdef IS_PYTHON_3
PyMODINIT_FUNC PyInit_rotozoom (void)
#else
PyMODINIT_FUNC initrotozoom (void)
#endif
{
    PyObject *mod;

#ifdef IS_PYTHON_3
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "rotozoom",
        DOC_ROTOZOOM,
        -1,
        _gfx_methods,
        NULL, NULL, NULL, NULL
    };
    mod = PyModule_Create (&_module);
#else
    mod = Py_InitModule3 ("rotozoom", _gfx_methods, DOC_ROTOZOOM);
#endif
    if (!mod)
        goto fail;

    if (import_pygame2_base () < 0)
        goto fail;
    if (import_pygame2_sdl_base () < 0)
        goto fail;
    if (import_pygame2_sdl_video () < 0)
        goto fail;
    MODINIT_RETURN(mod);
fail:
    Py_XDECREF (mod);
    MODINIT_RETURN (NULL);
}
