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
#define PYGAME_SDLGFXPRIM_INTERNAL

#include "pgsdl.h"
#include "pggfx.h"
#include "surface.h"
#include <SDL_gfxPrimitives.h>

static PyObject* _gfx_pixelcolor (PyObject *self, PyObject* args);
static PyObject* _gfx_hlinecolor (PyObject *self, PyObject* args);
static PyObject* _gfx_vlinecolor (PyObject *self, PyObject* args);
static PyObject* _gfx_rectanglecolor (PyObject *self, PyObject* args);
static PyObject* _gfx_boxcolor (PyObject *self, PyObject* args);
static PyObject* _gfx_linecolor (PyObject *self, PyObject* args);
static PyObject* _gfx_circlecolor (PyObject *self, PyObject* args);
static PyObject* _gfx_arccolor (PyObject *self, PyObject* args);
static PyObject* _gfx_aacirclecolor (PyObject *self, PyObject* args);
static PyObject* _gfx_filledcirclecolor (PyObject *self, PyObject* args);
static PyObject* _gfx_ellipsecolor (PyObject *self, PyObject* args);
static PyObject* _gfx_aaellipsecolor (PyObject *self, PyObject* args);
static PyObject* _gfx_filledellipsecolor (PyObject *self, PyObject* args);
static PyObject* _gfx_piecolor (PyObject *self, PyObject* args);
static PyObject* _gfx_trigoncolor (PyObject *self, PyObject* args);
static PyObject* _gfx_aatrigoncolor (PyObject *self, PyObject* args);
static PyObject* _gfx_filledtrigoncolor (PyObject *self, PyObject* args);
static PyObject* _gfx_polygoncolor (PyObject *self, PyObject* args);
static PyObject* _gfx_aapolygoncolor (PyObject *self, PyObject* args);
static PyObject* _gfx_filledpolygoncolor (PyObject *self, PyObject* args);
static PyObject* _gfx_texturedpolygon (PyObject *self, PyObject* args);
static PyObject* _gfx_beziercolor (PyObject *self, PyObject* args);

static PyMethodDef _gfx_methods[] = {
    { "pixel", _gfx_pixelcolor, METH_VARARGS, "" },
    { "hline", _gfx_hlinecolor, METH_VARARGS, "" },
    { "vline", _gfx_vlinecolor, METH_VARARGS, "" },
    { "rectangle", _gfx_rectanglecolor, METH_VARARGS, "" },
    { "box", _gfx_boxcolor, METH_VARARGS, "" },
    { "line", _gfx_linecolor, METH_VARARGS, "" },
    { "arc", _gfx_arccolor, METH_VARARGS, "" },
    { "circle", _gfx_circlecolor, METH_VARARGS, "" },
    { "aacircle", _gfx_aacirclecolor, METH_VARARGS, "" },
    { "filled_circle", _gfx_filledcirclecolor, METH_VARARGS, "" },
    { "ellipse", _gfx_ellipsecolor, METH_VARARGS, "" },
    { "aaellipse", _gfx_aaellipsecolor, METH_VARARGS, "" },
    { "filled_ellipse", _gfx_filledellipsecolor, METH_VARARGS, "" },
    { "pie", _gfx_piecolor, METH_VARARGS, "" },
    { "trigon", _gfx_trigoncolor, METH_VARARGS, "" },
    { "aatrigon", _gfx_aatrigoncolor, METH_VARARGS, "" },
    { "filled_trigon", _gfx_filledtrigoncolor, METH_VARARGS, "" },
    { "polygon", _gfx_polygoncolor, METH_VARARGS, "" },
    { "aapolygon", _gfx_aapolygoncolor, METH_VARARGS, "" },
    { "filled_polygon", _gfx_filledpolygoncolor, METH_VARARGS, "" },
    { "textured_polygon", _gfx_texturedpolygon, METH_VARARGS, "" },
    { "bezier", _gfx_beziercolor, METH_VARARGS, "" },
    { NULL, NULL, 0, NULL },
};

static PyObject*
_gfx_pixelcolor (PyObject *self, PyObject* args)
{
    PyObject *surface, *color;
    Sint16 x, y;
    Uint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OiiO:pixel", &surface, &x, &y, &color))
        return NULL;
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!PyColor_Check (color))
    {
        PyErr_SetString (PyExc_TypeError, "color must be a Color");
        return NULL;
    }

    c = (Uint32) PyColor_AsNumber (color);
    ARGB2FORMAT (c, ((PySDLSurface*)surface)->surface->format);

    if (pixelColor (((PySDLSurface*)surface)->surface, x, y, c) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_gfx_hlinecolor (PyObject *self, PyObject* args)
{
    PyObject *surface, *color;
    Sint16 x1, x2, y;
    Uint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OiiiO:hline", &surface, &x1, &x2, &y, &color))
        return NULL;
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!PyColor_Check (color))
    {
        PyErr_SetString (PyExc_TypeError, "color must be a Color");
        return NULL;
    }

    c = (Uint32) PyColor_AsNumber (color);
    ARGB2FORMAT (c, ((PySDLSurface*)surface)->surface->format);

    if (hlineColor (((PySDLSurface*)surface)->surface, x1, x2, y, c) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_gfx_vlinecolor (PyObject *self, PyObject* args)
{
    PyObject *surface, *color;
    Sint16 x, _y1, y2;
    Uint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OiiiO:vline", &surface, &x, &_y1, &y2,
            &color))
        return NULL;
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!PyColor_Check (color))
    {
        PyErr_SetString (PyExc_TypeError, "color must be a Color");
        return NULL;
    }

    c = (Uint32) PyColor_AsNumber (color);
    ARGB2FORMAT (c, ((PySDLSurface*)surface)->surface->format);

    if (vlineColor (((PySDLSurface*)surface)->surface, x, _y1, y2, c) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_gfx_rectanglecolor (PyObject *self, PyObject* args)
{
    PyObject *surface, *color, *rect;
    SDL_Rect sdlrect;
    Sint16 x1, x2, _y1, y2;
    Uint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OOO:rectangle", &surface, &rect, &color))
        return NULL;
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!SDLRect_FromRect (rect, &sdlrect))
        return NULL;

    if (!PyColor_Check (color))
    {
        PyErr_SetString (PyExc_TypeError, "color must be a Color");
        return NULL;
    }

    x1 = sdlrect.x;
    _y1 = sdlrect.y;
    x2 = (Sint16) (sdlrect.x + sdlrect.w);
    y2 = (Sint16) (sdlrect.y + sdlrect.h);
    c = (Uint32) PyColor_AsNumber (color);
    ARGB2FORMAT (c, ((PySDLSurface*)surface)->surface->format);

    if (rectangleColor (((PySDLSurface*)surface)->surface, x1, x2,_y1, y2, c) ==
        -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_gfx_boxcolor (PyObject *self, PyObject* args)
{
    PyObject *surface, *color, *rect;
    SDL_Rect sdlrect;
    Sint16 x1, x2, _y1, y2;
    Uint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OOO:box", &surface, &rect, &color))
        return NULL;
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!SDLRect_FromRect (rect, &sdlrect))
        return NULL;

    if (!PyColor_Check (color))
    {
        PyErr_SetString (PyExc_TypeError, "color must be a Color");
        return NULL;
    }

    x1 = sdlrect.x;
    _y1 = sdlrect.y;
    x2 = (Sint16) (sdlrect.x + sdlrect.w);
    y2 = (Sint16) (sdlrect.y + sdlrect.h);
    c = (Uint32) PyColor_AsNumber (color);
    ARGB2FORMAT (c, ((PySDLSurface*)surface)->surface->format);

    if (rectangleColor (((PySDLSurface*)surface)->surface, x1, x2,_y1, y2, c) ==
        -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_gfx_linecolor (PyObject *self, PyObject* args)
{
    PyObject *surface, *color;
    Sint16 x1, x2, _y1, y2;
    Uint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OiiiiO:line", &surface, &x1, &_y1, &x2, &y2,
            &color))
        return NULL;
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!PyColor_Check (color))
    {
        PyErr_SetString (PyExc_TypeError, "color must be a Color");
        return NULL;
    }

    c = (Uint32) PyColor_AsNumber (color);
    ARGB2FORMAT (c, ((PySDLSurface*)surface)->surface->format);

    if (lineColor (((PySDLSurface*)surface)->surface, x1, _y1, x2, y2, c) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_gfx_circlecolor (PyObject *self, PyObject* args)
{
    PyObject *surface, *color;
    Sint16 x, y, r;
    Uint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OiiiO:circle", &surface, &x, &y, &r, &color))
        return NULL;
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!PyColor_Check (color))
    {
        PyErr_SetString (PyExc_TypeError, "color must be a Color");
        return NULL;
    }

    c = (Uint32) PyColor_AsNumber (color);
    ARGB2FORMAT (c, ((PySDLSurface*)surface)->surface->format);

    if (circleColor (((PySDLSurface*)surface)->surface, x, y, r, c) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_gfx_arccolor (PyObject *self, PyObject* args)
{
    PyObject *surface, *color;
    Sint16 x, y, r, start, end;
    Uint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OiiiiiO:arc", &surface, &x, &y, &r,
            &start, &end, &color))
        return NULL;
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!PyColor_Check (color))
    {
        PyErr_SetString (PyExc_TypeError, "color must be a Color");
        return NULL;
    }

    c = (Uint32) PyColor_AsNumber (color);
    ARGB2FORMAT (c, ((PySDLSurface*)surface)->surface->format);

    if (arcColor (((PySDLSurface*)surface)->surface, x, y, r, start, end, c)
        == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_gfx_aacirclecolor (PyObject *self, PyObject* args)
{
    PyObject *surface, *color;
    Sint16 x, y, r;
    Uint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OiiiO:aacircle", &surface, &x, &y, &r,
            &color))
        return NULL;
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!PyColor_Check (color))
    {
        PyErr_SetString (PyExc_TypeError, "color must be a Color");
        return NULL;
    }

    c = (Uint32) PyColor_AsNumber (color);
    ARGB2FORMAT (c, ((PySDLSurface*)surface)->surface->format);

    if (aacircleColor (((PySDLSurface*)surface)->surface, x, y, r, c) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_gfx_filledcirclecolor (PyObject *self, PyObject* args)
{
    PyObject *surface, *color;
    Sint16 x, y, r;
    Uint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OiiiO:filledcircle", &surface, &x, &y, &r,
            &color))
        return NULL;
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!PyColor_Check (color))
    {
        PyErr_SetString (PyExc_TypeError, "color must be a Color");
        return NULL;
    }

    c = (Uint32) PyColor_AsNumber (color);
    ARGB2FORMAT (c, ((PySDLSurface*)surface)->surface->format);

    if (filledCircleColor (((PySDLSurface*)surface)->surface, x, y, r, c) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_gfx_ellipsecolor (PyObject *self, PyObject* args)
{
    PyObject *surface, *color;
    Sint16 x, y, rx, ry;
    Uint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OiiiiO:ellipse", &surface, &x, &y, &rx, &ry,
            &color))
        return NULL;
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!PyColor_Check (color))
    {
        PyErr_SetString (PyExc_TypeError, "color must be a Color");
        return NULL;
    }

    c = (Uint32) PyColor_AsNumber (color);
    ARGB2FORMAT (c, ((PySDLSurface*)surface)->surface->format);

    if (ellipseColor (((PySDLSurface*)surface)->surface, x, y, rx, ry, c) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_gfx_aaellipsecolor (PyObject *self, PyObject* args)
{
    PyObject *surface, *color;
    Sint16 x, y, rx, ry;
    Uint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OiiiiO:aaellipse", &surface, &x, &y, &rx, &ry,
            &color))
        return NULL;
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!PyColor_Check (color))
    {
        PyErr_SetString (PyExc_TypeError, "color must be a Color");
        return NULL;
    }

    c = (Uint32) PyColor_AsNumber (color);
    ARGB2FORMAT (c, ((PySDLSurface*)surface)->surface->format);

    if (aaellipseColor (((PySDLSurface*)surface)->surface, x, y, rx, ry, c)
        == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_gfx_filledellipsecolor (PyObject *self, PyObject* args)
{
    PyObject *surface, *color;
    Sint16 x, y, rx, ry;
    Uint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OiiiiO:filled_ellipse", &surface, &x, &y,
            &rx, &ry, &color))
        return NULL;
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!PyColor_Check (color))
    {
        PyErr_SetString (PyExc_TypeError, "color must be a Color");
        return NULL;
    }

    c = (Uint32) PyColor_AsNumber (color);
    ARGB2FORMAT (c, ((PySDLSurface*)surface)->surface->format);

    if (filledEllipseColor (((PySDLSurface*)surface)->surface, x, y, rx, ry, c)
        == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_gfx_piecolor (PyObject *self, PyObject* args)
{
    PyObject *surface, *color;
    Sint16 x, y, r, start, end;
    Uint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OiiiiiO:pie", &surface, &x, &y, &r,
            &start, &end, &color))
        return NULL;
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!PyColor_Check (color))
    {
        PyErr_SetString (PyExc_TypeError, "color must be a Color");
        return NULL;
    }

    c = (Uint32) PyColor_AsNumber (color);
    ARGB2FORMAT (c, ((PySDLSurface*)surface)->surface->format);

    if (pieColor (((PySDLSurface*)surface)->surface, x, y, r, start, end, c)
        == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_gfx_trigoncolor (PyObject *self, PyObject* args)
{
    PyObject *surface, *color;
    Sint16 x1, x2, x3, _y1, y2, y3;
    Uint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OiiiiiiO:trigon", &surface, &x1, &_y1, &x2,
            &y2, &x3, &y3, &color))
        return NULL;
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!PyColor_Check (color))
    {
        PyErr_SetString (PyExc_TypeError, "color must be a Color");
        return NULL;
    }

    c = (Uint32) PyColor_AsNumber (color);
    ARGB2FORMAT (c, ((PySDLSurface*)surface)->surface->format);

    if (trigonColor (((PySDLSurface*)surface)->surface, x1, _y1, x2, y2, x3,
            y3, c) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_gfx_aatrigoncolor (PyObject *self, PyObject* args)
{
    PyObject *surface, *color;
    Sint16 x1, x2, x3, _y1, y2, y3;
    Uint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OiiiiiiO:aatrigon", &surface, &x1, &_y1, &x2,
            &y2, &x3, &y3, &color))
        return NULL;
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!PyColor_Check (color))
    {
        PyErr_SetString (PyExc_TypeError, "color must be a Color");
        return NULL;
    }

    c = (Uint32) PyColor_AsNumber (color);
    ARGB2FORMAT (c, ((PySDLSurface*)surface)->surface->format);

    if (aatrigonColor (((PySDLSurface*)surface)->surface, x1, _y1, x2, y2, x3,
            y3, c) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_gfx_filledtrigoncolor (PyObject *self, PyObject* args)
{
    PyObject *surface, *color;
    Sint16 x1, x2, x3, _y1, y2, y3;
    Uint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OiiiiiiO:filled_trigon", &surface, &x1, &_y1,
            &x2, &y2, &x3, &y3, &color))
        return NULL;
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!PyColor_Check (color))
    {
        PyErr_SetString (PyExc_TypeError, "color must be a Color");
        return NULL;
    }

    c = (Uint32) PyColor_AsNumber (color);
    ARGB2FORMAT (c, ((PySDLSurface*)surface)->surface->format);

    if (filledTrigonColor (((PySDLSurface*)surface)->surface, x1, _y1, x2, y2,
            x3, y3, c) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_gfx_polygoncolor (PyObject *self, PyObject* args)
{
    PyObject *surface, *color, *points, *item;
    Sint16 *vx, *vy, x, y;
    Py_ssize_t count, i;
    int ret;
    Uint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OOO:polygon", &surface, &points, &color))
        return NULL;
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!PyColor_Check (color))
    {
        PyErr_SetString (PyExc_TypeError, "color must be a Color");
        return NULL;
    }
    if (!PySequence_Check (points))
    {
        PyErr_SetString (PyExc_TypeError, "points must be a sequence");
        return NULL;
    }

    count = PySequence_Size (points);
    if (count < 3)
    {
        PyErr_SetString (PyExc_ValueError,
            "points must contain more than 2 points");
        return NULL;
    }
    c = (Uint32) PyColor_AsNumber (color);
    ARGB2FORMAT (c, ((PySDLSurface*)surface)->surface->format);

    vx = PyMem_New (Sint16, (size_t) count);
    vy = PyMem_New (Sint16, (size_t) count);
    if (!vx || !vy)
    {
        if (vx)
            PyMem_Free (vx);
        if (vy)
            PyMem_Free (vy);
        return NULL;
    }

    for (i = 0; i < count; i++)
    {
        item = PySequence_ITEM (points, i);
        if (!Sint16FromSeqIndex (item, 0, &x))
        {
            PyMem_Free (vx);
            PyMem_Free (vy);
            Py_XDECREF (item);
            return NULL;
        }
        if (!Sint16FromSeqIndex (item, 1, &y))
        {
            PyMem_Free (vx);
            PyMem_Free (vy);
            Py_XDECREF (item);
            return NULL;
        }
        Py_DECREF (item);
        vx[i] = x;
        vy[i] = y;
    }

    Py_BEGIN_ALLOW_THREADS;
    ret = polygonColor (((PySDLSurface*)surface)->surface, vx, vy, (int)count,
        c);
    Py_END_ALLOW_THREADS;

    PyMem_Free (vx);
    PyMem_Free (vy);

    if (ret == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_gfx_aapolygoncolor (PyObject *self, PyObject* args)
{
    PyObject *surface, *color, *points, *item;
    Sint16 *vx, *vy, x, y;
    Py_ssize_t count, i;
    int ret;
    Uint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OOO:aapolygon", &surface, &points, &color))
        return NULL;
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!PyColor_Check (color))
    {
        PyErr_SetString (PyExc_TypeError, "color must be a Color");
        return NULL;
    }
    if (!PySequence_Check (points))
    {
        PyErr_SetString (PyExc_TypeError, "points must be a sequence");
        return NULL;
    }

    count = PySequence_Size (points);
    if (count < 3)
    {
        PyErr_SetString (PyExc_ValueError,
            "points must contain more than 2 points");
        return NULL;
    }
    c = (Uint32) PyColor_AsNumber (color);
    ARGB2FORMAT (c, ((PySDLSurface*)surface)->surface->format);

    vx = PyMem_New (Sint16, (size_t) count);
    vy = PyMem_New (Sint16, (size_t) count);
    if (!vx || !vy)
    {
        if (vx)
            PyMem_Free (vx);
        if (vy)
            PyMem_Free (vy);
        return NULL;
    }

    for (i = 0; i < count; i++)
    {
        item = PySequence_ITEM (points, i);
        if (!Sint16FromSeqIndex (item, 0, &x))
        {
            PyMem_Free (vx);
            PyMem_Free (vy);
            Py_XDECREF (item);
            return NULL;
        }
        if (!Sint16FromSeqIndex (item, 1, &y))
        {
            PyMem_Free (vx);
            PyMem_Free (vy);
            Py_XDECREF (item);
            return NULL;
        }
        Py_DECREF (item);
        vx[i] = x;
        vy[i] = y;
    }

    Py_BEGIN_ALLOW_THREADS;
    ret = aapolygonColor (((PySDLSurface*)surface)->surface, vx, vy, (int)count,
        c);
    Py_END_ALLOW_THREADS;

    PyMem_Free (vx);
    PyMem_Free (vy);

    if (ret == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_gfx_filledpolygoncolor (PyObject *self, PyObject* args)
{
    PyObject *surface, *color, *points, *item;
    Sint16 *vx, *vy, x, y;
    Py_ssize_t count, i;
    int ret;
    Uint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OOO:filled_polygon", &surface, &points,
            &color))
        return NULL;
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!PyColor_Check (color))
    {
        PyErr_SetString (PyExc_TypeError, "color must be a Color");
        return NULL;
    }
    if (!PySequence_Check (points))
    {
        PyErr_SetString (PyExc_TypeError, "points must be a sequence");
        return NULL;
    }

    count = PySequence_Size (points);
    if (count < 3)
    {
        PyErr_SetString (PyExc_ValueError,
            "points must contain more than 2 points");
        return NULL;
    }
    c = (Uint32) PyColor_AsNumber (color);
    ARGB2FORMAT (c, ((PySDLSurface*)surface)->surface->format);

    vx = PyMem_New (Sint16, (size_t) count);
    vy = PyMem_New (Sint16, (size_t) count);
    if (!vx || !vy)
    {
        if (vx)
            PyMem_Free (vx);
        if (vy)
            PyMem_Free (vy);
        return NULL;
    }

    for (i = 0; i < count; i++)
    {
        item = PySequence_ITEM (points, i);
        if (!Sint16FromSeqIndex (item, 0, &x))
        {
            PyMem_Free (vx);
            PyMem_Free (vy);
            Py_XDECREF (item);
            return NULL;
        }
        if (!Sint16FromSeqIndex (item, 1, &y))
        {
            PyMem_Free (vx);
            PyMem_Free (vy);
            Py_XDECREF (item);
            return NULL;
        }
        Py_DECREF (item);
        vx[i] = x;
        vy[i] = y;
    }

    Py_BEGIN_ALLOW_THREADS;
    ret = filledPolygonColor (((PySDLSurface*)surface)->surface, vx, vy,
        (int)count, c);
    Py_END_ALLOW_THREADS;

    PyMem_Free (vx);
    PyMem_Free (vy);

    if (ret == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_gfx_texturedpolygon (PyObject *self, PyObject* args)
{
    PyObject *surface, *texture, *points, *item;
    Sint16 *vx, *vy, x, y, tdx, tdy;
    Py_ssize_t count, i;
    int ret;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OOOii:textured_polygon", &surface, &points,
            &texture, &tdx, &tdy))
        return NULL;
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!PySDLSurface_Check (texture))
    {
        PyErr_SetString (PyExc_TypeError, "texture must be a Surface");
        return NULL;
    }
    if (!PySequence_Check (points))
    {
        PyErr_SetString (PyExc_TypeError, "points must be a sequence");
        return NULL;
    }

    count = PySequence_Size (points);
    if (count < 3)
    {
        PyErr_SetString (PyExc_ValueError,
            "points must contain more than 2 points");
        return NULL;
    }

    vx = PyMem_New (Sint16, (size_t) count);
    vy = PyMem_New (Sint16, (size_t) count);
    if (!vx || !vy)
    {
        if (vx)
            PyMem_Free (vx);
        if (vy)
            PyMem_Free (vy);
        return NULL;
    }

    for (i = 0; i < count; i++)
    {
        item = PySequence_ITEM (points, i);
        if (!Sint16FromSeqIndex (item, 0, &x))
        {
            PyMem_Free (vx);
            PyMem_Free (vy);
            Py_XDECREF (item);
            return NULL;
        }
        if (!Sint16FromSeqIndex (item, 1, &y))
        {
            PyMem_Free (vx);
            PyMem_Free (vy);
            Py_XDECREF (item);
            return NULL;
        }
        Py_DECREF (item);
        vx[i] = x;
        vy[i] = y;
    }

    Py_BEGIN_ALLOW_THREADS;
    ret = texturedPolygon (((PySDLSurface*)surface)->surface, vx, vy,
        (int)count, ((PySDLSurface*)texture)->surface, tdx, tdy);
    Py_END_ALLOW_THREADS;

    PyMem_Free (vx);
    PyMem_Free (vy);

    if (ret == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}


static PyObject*
_gfx_beziercolor (PyObject *self, PyObject* args)
{
    PyObject *surface, *color, *points, *item;
    Sint16 *vx, *vy, x, y;
    Py_ssize_t count, i;
    int ret, steps;
    Uint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OOiO:polygon", &surface, &points, &steps,
            &color))
        return NULL;
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!PyColor_Check (color))
    {
        PyErr_SetString (PyExc_TypeError, "color must be a Color");
        return NULL;
    }
    if (!PySequence_Check (points))
    {
        PyErr_SetString (PyExc_TypeError, "points must be a sequence");
        return NULL;
    }

    count = PySequence_Size (points);
    if (count < 3)
    {
        PyErr_SetString (PyExc_ValueError,
            "points must contain more than 2 points");
        return NULL;
    }
    c = (Uint32) PyColor_AsNumber (color);
    ARGB2FORMAT (c, ((PySDLSurface*)surface)->surface->format);

    vx = PyMem_New (Sint16, (size_t) count);
    vy = PyMem_New (Sint16, (size_t) count);
    if (!vx || !vy)
    {
        PyErr_SetString (PyExc_MemoryError, "memory allocation failed");
        if (vx)
            PyMem_Free (vx);
        if (vy)
            PyMem_Free (vy);
        return NULL;
    }

    for (i = 0; i < count; i++)
    {
        item = PySequence_ITEM (points, i);
        if (!Sint16FromSeqIndex (item, 0, &x))
        {
            PyMem_Free (vx);
            PyMem_Free (vy);
            Py_XDECREF (item);
            return NULL;
        }
        if (!Sint16FromSeqIndex (item, 1, &y))
        {
            PyMem_Free (vx);
            PyMem_Free (vy);
            Py_XDECREF (item);
            return NULL;
        }
        Py_DECREF (item);
        vx[i] = x;
        vy[i] = y;
    }

    Py_BEGIN_ALLOW_THREADS;
    ret = bezierColor (((PySDLSurface*)surface)->surface, vx, vy, (int)count,
        steps, c);
    Py_END_ALLOW_THREADS;

    PyMem_Free (vx);
    PyMem_Free (vy);

    if (ret == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

#if PY_VERSION_HEX >= 0x03000000
PyMODINIT_FUNC PyInit_primitives (void)
#else
PyMODINIT_FUNC initprimitives (void)
#endif
{
    PyObject *mod;
    
#if PY_VERSION_HEX >= 0x03000000
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "primitives",
        "",
        -1,
        _gfx_methods,
        NULL, NULL, NULL, NULL
    };
#endif

#if PY_VERSION_HEX < 0x03000000
    mod = Py_InitModule3 ("primitives", _gfx_methods, "");
#else
    mod = PyModule_Create (&_module);
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
