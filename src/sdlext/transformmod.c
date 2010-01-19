/*
  pygame - Python Game Library
  Copyright (C) 2000-2001  Pete Shinners, 2007  Rene Dudfield, Richard Goedeken 

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

#define PYGAME_SDLEXTTRANSFORM_INTERNAL

#include "pgsdlext.h"
#include "pgsdl.h"
#include "transform.h"
#include "filters.h"
#include "surface.h"
#include "sdlexttransform_doc.h"

typedef struct {
    FilterFuncs filterfuncs;
} _TransformState;

#ifdef IS_PYTHON_3
struct PyModuleDef _transformmodule; /* Forward declaration */
#define TRANSFORM_MOD_STATE(mod) ((_TransformState*)PyModule_GetState(mod))
#define TRANSFORM_STATE \
    TRANSFORM_MOD_STATE(PyState_FindModule(&_transformmodule))
#else
static _TransformState _modstate;
#define TRANSFORM_MOD_STATE(mod) (&_modstate)
#define TRANSFORM_STATE TRANSFORM_MOD_STATE(NULL)
#endif

static PyObject* _transform_scale (PyObject* self, PyObject* args);
static PyObject* _transform_rotate (PyObject* self, PyObject* args);
static PyObject* _transform_flip (PyObject* self, PyObject* args);
static PyObject* _transform_chop (PyObject* self, PyObject* args);
static PyObject* _transform_scale2x (PyObject* self, PyObject* args);
static PyObject* _transform_smoothscale (PyObject* self, PyObject* args);
static PyObject* _transform_threshold (PyObject* self, PyObject* args);
static PyObject* _transform_laplacian (PyObject* self, PyObject* args);
static PyObject* _transform_averagesurfaces (PyObject* self, PyObject* args);
static PyObject* _transform_averagecolor (PyObject* self, PyObject* args);
static PyObject* _transform_getfiltertype (PyObject* self);
static PyObject* _transform_setfiltertype (PyObject* self, PyObject* args);

static PyMethodDef _transform_methods[] =
{
    { "scale", _transform_scale, METH_VARARGS, DOC_TRANSFORM_SCALE },
    { "rotate", _transform_rotate, METH_VARARGS, DOC_TRANSFORM_ROTATE },
    { "flip", _transform_flip, METH_VARARGS, DOC_TRANSFORM_FLIP },
    { "chop", _transform_chop, METH_VARARGS, DOC_TRANSFORM_CHOP},
    { "scale2x", _transform_scale2x, METH_VARARGS, DOC_TRANSFORM_SCALE2X },
    { "smoothscale", _transform_smoothscale, METH_VARARGS,
      DOC_TRANSFORM_SMOOTHSCALE },
    { "threshold", _transform_threshold, METH_VARARGS,
      DOC_TRANSFORM_THRESHOLD },
    { "laplacian", _transform_laplacian, METH_VARARGS,
      DOC_TRANSFORM_LAPLACIAN },
    { "average_surfaces", _transform_averagesurfaces, METH_VARARGS,
      DOC_TRANSFORM_AVERAGE_SURFACES },
    { "average_color", _transform_averagecolor, METH_VARARGS,
      DOC_TRANSFORM_AVERAGE_COLOR },
    { "get_filtertype", (PyCFunction)_transform_getfiltertype, METH_NOARGS,
      DOC_TRANSFORM_GET_FILTERTYPE },
    { "set_filtertype", _transform_setfiltertype, METH_O,
      DOC_TRANSFORM_SET_FILTERTYPE },
    { NULL, NULL, 0, NULL }
};

static PyObject*
_transform_scale (PyObject* self, PyObject* args)
{
    PyObject *srcobj, *dstobj = NULL;
    PyObject *size;
    SDL_Surface *srcsurface, *dstsurface = NULL;
    int width, height;

    /*get all the arguments*/
    if (!PyArg_ParseTuple (args, "OO|O", &srcobj, &size, &dstobj))
    {
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "Oii|O", &srcobj, &width, &height,
                &dstobj))
            return NULL;
    }
    else
    {
        if (!SizeFromObject (size, (pgint32*)&width, (pgint32*)&height))
            return NULL;
    }
    
    if (!PySDLSurface_Check (srcobj))
    {
        PyErr_SetString (PyExc_TypeError, "source surface must be a Surface");
        return NULL;
    }
    if (dstobj && !PySDLSurface_Check (dstobj))
    {
        PyErr_SetString (PyExc_TypeError,
            "destination surface must be a Surface");
        return NULL;
    }
    if (width < 0 || height < 0)
    {
        PyErr_SetString (PyExc_ValueError, "cannot scale to negative size");
        return NULL;
    }

    srcsurface = ((PySDLSurface*)srcobj)->surface;
    if (dstobj)
    {
        dstsurface = ((PySDLSurface*)dstobj)->surface;
        if (dstsurface->w != width || dstsurface->h != height)
        {
            PyErr_SetString (PyExc_ValueError,
                "destination surface does not match the given width or height");
            return NULL;
        }
    }
    Py_BEGIN_ALLOW_THREADS;
    dstsurface = pyg_transform_scale (srcsurface, dstsurface, width, height);
    Py_END_ALLOW_THREADS;

    if (!dstsurface)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }

    if (dstobj)
    {
        Py_INCREF (dstobj);
        return dstobj;
    }

    dstobj = PySDLSurface_NewFromSDLSurface (dstsurface);
    if (!dstobj)
    {
        SDL_FreeSurface (dstsurface);
        return NULL;
    }
    
    return dstobj;
}

static PyObject*
_transform_rotate (PyObject* self, PyObject* args)
{
    PyObject *surfobj, *newsurf;
    SDL_Surface *surface, *newsurface;
    double angle;
    
    if (!PyArg_ParseTuple (args, "Od", &surfobj, &angle))
        return NULL;
    if (!PySDLSurface_Check (surfobj))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }

    surface = ((PySDLSurface*)surfobj)->surface;

    if (fmod ((double)angle, (double)90.0f) == 0)
    {
        Py_BEGIN_ALLOW_THREADS;
        newsurface = pyg_transform_rotate90 (surface, (int)angle);
        Py_END_ALLOW_THREADS;
        if (!newsurface)
        {
            PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
            return NULL;
        }

        newsurf = PySDLSurface_NewFromSDLSurface (newsurface);
        if (!newsurf)
        {
            SDL_FreeSurface (newsurface);
            return NULL;
        }
        return newsurf;
    }

    Py_BEGIN_ALLOW_THREADS;
    newsurface = pyg_transform_rotate (surface, DEG2RAD(angle));
    Py_END_ALLOW_THREADS;
    if (!newsurface)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }

    newsurf = PySDLSurface_NewFromSDLSurface (newsurface);
    if (!newsurf)
    {
        SDL_FreeSurface (newsurface);
        return NULL;
    }
    return newsurf;
}

static PyObject*
_transform_flip (PyObject* self, PyObject* args)
{
    PyObject *surfobj, *newsurf, *x, *y;
    SDL_Surface* surface, *newsurface;
    int xaxis, yaxis;

    if (!PyArg_ParseTuple (args, "OOO", &surfobj, &x, &y))
        return NULL;
    if (!PySDLSurface_Check (surfobj))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }

    xaxis = PyObject_IsTrue (x);
    if (xaxis == -1)
        return NULL;
    yaxis = PyObject_IsTrue (y);
    if (yaxis == -1)
        return NULL;

    surface = ((PySDLSurface*)surfobj)->surface;
    
    if (!xaxis && !yaxis)
    {
        /* No changes. */
        return PySDLSurface_Copy (surfobj);
    }

    Py_BEGIN_ALLOW_THREADS;
    newsurface = pyg_transform_flip (surface, xaxis, yaxis);
    Py_END_ALLOW_THREADS;
    if (!newsurface)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    
    newsurf = PySDLSurface_NewFromSDLSurface (newsurface);
    if (!newsurf)
    {
        SDL_FreeSurface (newsurface);
        return NULL;
    }
    return newsurf;
}

static PyObject*
_transform_chop (PyObject* self, PyObject* args)
{
    PyObject *surfobj, *newsurf, *rectobj;
    SDL_Surface* surface, *newsurface;
    SDL_Rect rect;

    if (!PyArg_ParseTuple (args, "OO", &surfobj, &rectobj))
        return NULL;
    if (!PySDLSurface_Check (surfobj))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!SDLRect_FromRect (rectobj, &rect))
        return NULL;

    surface = ((PySDLSurface*)surfobj)->surface;

    Py_BEGIN_ALLOW_THREADS;
    newsurface = pyg_transform_chop (surface, &rect);
    Py_END_ALLOW_THREADS;
    if (!newsurface)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    
    newsurf = PySDLSurface_NewFromSDLSurface (newsurface);
    if (!newsurf)
    {
        SDL_FreeSurface (newsurface);
        return NULL;
    }
    return newsurf;
}

static PyObject*
_transform_scale2x (PyObject* self, PyObject* args)
{
    PyObject *surfobj, *dstobj = NULL;
    SDL_Surface *src, *dst = NULL;

    if (!PyArg_ParseTuple (args, "O|O", &surfobj, &dstobj))
        return NULL;
    if (!PySDLSurface_Check (surfobj))
    {
        PyErr_SetString (PyExc_TypeError, "source surface must be a Surface");
        return NULL;
    }
    if (dstobj && !PySDLSurface_Check (dstobj))
    {
        PyErr_SetString (PyExc_TypeError,
            "destination surface must be a Surface");
        return NULL;
    }
    
    src = ((PySDLSurface*)surfobj)->surface;
    if (dstobj)
        dst = ((PySDLSurface*)dstobj)->surface;

    Py_BEGIN_ALLOW_THREADS;
    dst = pyg_transform_scale2x (src, dst);
    Py_END_ALLOW_THREADS;
    if (!dst)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }

    if (dstobj)
    {
        Py_INCREF (dstobj);
        return dstobj;
    }

    dstobj = PySDLSurface_NewFromSDLSurface (dst);
    if (!dstobj)
    {
        SDL_FreeSurface (dst);
        return NULL;
    }
    return dstobj;
}

static PyObject*
_transform_smoothscale (PyObject* self, PyObject* args)
{
    PyObject *surfobj, *dstobj = NULL;
    PyObject *size;
    SDL_Surface *src, *dst = NULL;
    int width, height;

    if (!PyArg_ParseTuple (args, "OO|O", &surfobj, &size, &dstobj))
    {
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "Oii|O", &surfobj, &width, &height,
                &dstobj))
            return NULL;
    }
    else
    {
        if (!SizeFromObject (size, (pgint32*)&width, (pgint32*)&height))
            return NULL;
    }
    
    if (!PySDLSurface_Check (surfobj))
    {
        PyErr_SetString (PyExc_TypeError, "source surface must be a Surface");
        return NULL;
    }
    if (!width < 0 || height < 0)
    {
        PyErr_SetString (PyExc_ValueError,
            "width and height must not be negative");
        return NULL;
    }
    if (dstobj && !PySDLSurface_Check (dstobj))
    {
        PyErr_SetString (PyExc_TypeError,
            "destination surface must be a Surface");
        return NULL;
    }
    
    src = ((PySDLSurface*)surfobj)->surface;
    if (dstobj)
        dst = ((PySDLSurface*)dstobj)->surface;

    Py_BEGIN_ALLOW_THREADS;
    dst = pyg_transform_smoothscale (src, dst, width, height,
        &(TRANSFORM_MOD_STATE (self)->filterfuncs));
    Py_END_ALLOW_THREADS;
    if (!dst)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }

    if (dstobj)
    {
        Py_INCREF (dstobj);
        return dstobj;
    }

    dstobj = PySDLSurface_NewFromSDLSurface (dst);
    if (!dstobj)
    {
        SDL_FreeSurface (dst);
        return NULL;
    }
    return dstobj;
}

static PyObject*
_transform_threshold (PyObject* self, PyObject* args)
{
    PyObject *surfobj, *colorobj, *thresobj, *destobj = NULL;
    int diff_is_color = 0;
    SDL_Surface *srcsurface, *diffsurface = NULL, *dstsurface = NULL;
    Uint32 diffcolor = 0, threscolor;
    int changed;

    if (!PyArg_ParseTuple (args, "OOO|O:threshold", &surfobj, &colorobj,
            &thresobj, &destobj))
        return NULL;
    if (!PySDLSurface_Check (surfobj))
    {
        PyErr_SetString (PyExc_TypeError, "source surface must be a Surface");
        return NULL;
    }
    if (!PySDLSurface_Check (colorobj))
    {
        if (PyColor_Check (colorobj))
        {
            diff_is_color = 1;
            diffcolor = PyColor_AsNumber (colorobj);
        }
        else
        {
            PyErr_SetString (PyExc_TypeError,
                "diffobj must be a Surface or Color");
            return NULL;
        }
    }
    else
    {
        diffsurface = ((PySDLSurface*)colorobj)->surface;
    }
    if (!PyColor_Check (thresobj))
    {
        PyErr_SetString (PyExc_TypeError, "threshold must be a Color");
        return NULL;
    }
    if (destobj && !PySDLSurface_Check (destobj))
    {
        PyErr_SetString (PyExc_TypeError,
            "destination surface must be a Surface");
        return NULL;
    }

    srcsurface = ((PySDLSurface*)surfobj)->surface;
    if (destobj)
        dstsurface = ((PySDLSurface*)destobj)->surface;
    threscolor = PyColor_AsNumber (thresobj);

    ARGB2FORMAT (diffcolor, srcsurface->format);
    ARGB2FORMAT (threscolor, srcsurface->format);

    if (srcsurface->w != dstsurface->w || srcsurface->h != dstsurface->h)
    {
        PyErr_SetString (PyExc_ValueError,
            "destination surface and source surface must have the same size");
        return NULL;
    }
    if (diffsurface &&
        (srcsurface->w != diffsurface->w || srcsurface->h != diffsurface->h))
    {
        PyErr_SetString (PyExc_ValueError,
            "diff surface and source surface must have the same size");
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS;
    if (diff_is_color)
    {
        changed = pyg_transform_threshold_color (srcsurface, diffcolor,
            threscolor, dstsurface);
    }
    else
    {
        changed = pyg_transform_threshold_surface (srcsurface, diffsurface,
            threscolor, dstsurface);
    }
    Py_END_ALLOW_THREADS;

    if (changed == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }

    return PyInt_FromLong (changed);
}

static PyObject*
_transform_laplacian (PyObject* self, PyObject* args)
{
    PyObject *surfobj, *dstobj = NULL;
    SDL_Surface *src, *dst = NULL;

    if (!PyArg_ParseTuple (args, "O|O", &surfobj, &dstobj))
        return NULL;
    if (!PySDLSurface_Check (surfobj))
    {
        PyErr_SetString (PyExc_TypeError, "source surface must be a Surface");
        return NULL;
    }
    if (dstobj && !PySDLSurface_Check (dstobj))
    {
        PyErr_SetString (PyExc_TypeError,
            "destination surface must be a Surface");
        return NULL;
    }

    src = ((PySDLSurface*)surfobj)->surface;
    if (dstobj)
        dst = ((PySDLSurface*)dstobj)->surface;

    Py_BEGIN_ALLOW_THREADS;
    dst = pyg_transform_laplacian (src, dst);
    Py_END_ALLOW_THREADS;
    if (!dst)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }

    if (dstobj)
    {
        Py_INCREF (dstobj);
        return dstobj;
    }

    dstobj = PySDLSurface_NewFromSDLSurface (dst);
    if (!dstobj)
    {
        SDL_FreeSurface (dst);
        return NULL;
    }
    return dstobj;
}

static PyObject*
_transform_averagesurfaces (PyObject* self, PyObject* args)
{
    PyObject *list, *surfobj, *dstobj = NULL;
    SDL_Surface *surface, **surfaces, *dst = NULL;
    Py_ssize_t i, count;
    int width =0, height = 0;

    if (!PyArg_ParseTuple (args, "O|O", &list, &dstobj))
        return NULL;

    if (!PySequence_Check (list))
    {
        PyErr_SetString (PyExc_TypeError, 
            "surfaces must be a sequence of Surface objects");
        return NULL;
    }
    if (dstobj && !PySDLSurface_Check (dstobj))
    {
        PyErr_SetString (PyExc_TypeError,
            "destination surface must be a Surface");
        return NULL;
    }

    if (dstobj)
        dst = ((PySDLSurface*)dstobj)->surface;
    
    count = PySequence_Size (list);
    surfaces = PyMem_New (SDL_Surface*, (size_t) count);
    if (!surfaces)
        return NULL;

    for (i = 0; i < count; i++)
    {
        surfobj = PySequence_ITEM (list, i);
        if (!PySDLSurface_Check (surfobj))
        {
            Py_XDECREF (surfobj);
            PyMem_Free (surfaces);
            PyErr_SetString (PyExc_TypeError,
                "surfaces must be a sequence of Surface objects");
            return NULL;
        }
        surfaces[i] = ((PySDLSurface*)surfobj)->surface;
        width = MAX (surfaces[i]->w, width);
        height = MAX (surfaces[i]->w, width);
        Py_DECREF (surfobj);
    }

    if (dst && (dst->w != width || dst->h != height))
    {
        PyMem_Free (surfaces);
        PyErr_SetString (PyExc_ValueError,
            "destination surface size must match the biggest surface size");
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS;
    dst = pyg_transform_average_surfaces (surfaces, count, dst);
    Py_END_ALLOW_THREADS;
    PyMem_Free (surfaces);
    if (!dst)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }

    if (dstobj)
    {
        Py_INCREF (dstobj);
        return dstobj;
    }

    /* We received a 32 bit SW_SURFACE, convert it to the display format. */
    surface = SDL_DisplayFormat (dst);
    SDL_FreeSurface (dst);
    if (!surface)
        return NULL;

    dstobj = PySDLSurface_NewFromSDLSurface (surface);
    if (!dstobj)
    {
        SDL_FreeSurface (surface);
        return NULL;
    }
    return dstobj;
}

static PyObject*
_transform_averagecolor (PyObject* self, PyObject* args)
{
    PyObject *surfobj, *rectobj = NULL;
    SDL_Surface* surface;
    SDL_Rect sdlrect;
    Uint8 rgba[4] = { 0 };
    int retval;

    if (!PyArg_ParseTuple (args, "O|O", &surfobj, &rectobj))
        return NULL;
    if (!PySDLSurface_Check (surfobj))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be Surface");
        return NULL;
    }
    if (rectobj && !SDLRect_FromRect (rectobj, &sdlrect))
        return NULL;

    surface = ((PySDLSurface*)surfobj)->surface;
    if (!rectobj)
    {
        sdlrect.x = sdlrect.y = 0;
        sdlrect.w = surface->w;
        sdlrect.h = surface->h;
    }
    
    Py_BEGIN_ALLOW_THREADS;    
    retval = pyg_transform_average_color (surface, &sdlrect, &rgba[0],
        &rgba[1], &rgba[2], &rgba[3]);
    Py_END_ALLOW_THREADS;

    if (!retval)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    return PyColor_New ((pgbyte*)rgba);
}   

static PyObject*
_transform_getfiltertype (PyObject* self)
{
    return PyInt_FromLong (TRANSFORM_MOD_STATE (self)->filterfuncs.type);
}

static PyObject*
_transform_setfiltertype (PyObject* self, PyObject* args)
{
    FilterType type;

    if (!IntFromObj (args, (int*)&type))
        return NULL;
    return PyInt_FromLong (pyg_filter_init_filterfuncs
        (&(TRANSFORM_MOD_STATE (self)->filterfuncs), type));
}

#ifdef IS_PYTHON_3
struct PyModuleDef _transformmodule = {
    PyModuleDef_HEAD_INIT,
    "transform",
    DOC_TRANSFORM,
    sizeof (_TransformState),
    _transform_methods,
    NULL,
    NULL,
    NULL,
    NULL
};
#endif

#ifdef IS_PYTHON_3
PyMODINIT_FUNC PyInit_transform (void)
#else
PyMODINIT_FUNC inittransform (void)
#endif
{
    PyObject *mod;
    _TransformState *state;

#ifdef IS_PYTHON_3
    mod = PyModule_Create (&_transformmodule);
#else
    mod = Py_InitModule3 ("transform", _transform_methods, DOC_TRANSFORM);
#endif
    if (!mod)
        goto fail;
    state = TRANSFORM_MOD_STATE (mod);
    
    if (import_pygame2_base () < 0)
        goto fail;
    if (import_pygame2_sdl_video () < 0)
        goto fail;
    
    pyg_filter_init_filterfuncs (&(state->filterfuncs), FILTER_C);
    
    MODINIT_RETURN(mod);
fail:
    Py_XDECREF (mod);
    MODINIT_RETURN (NULL);
}
