/*
  pygame - Python Game Library
  Copyright (C) 2008-2010 Marcus von Appen

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

#include "pgbase.h"
#include "pgsdl.h"
#include "pggfx.h"
#include "surface.h"
#include "sdlgfxprimitives_doc.h"
#include <SDL_gfxPrimitives.h>

static PyObject* _gfx_pixelcolor (PyObject *self, PyObject* args);
static PyObject* _gfx_hlinecolor (PyObject *self, PyObject* args);
static PyObject* _gfx_vlinecolor (PyObject *self, PyObject* args);
static PyObject* _gfx_rectanglecolor (PyObject *self, PyObject* args);
static PyObject* _gfx_boxcolor (PyObject *self, PyObject* args);
static PyObject* _gfx_aalinecolor (PyObject *self, PyObject* args);
static PyObject* _gfx_linecolor (PyObject *self, PyObject* args);
static PyObject* _gfx_circlecolor (PyObject *self, PyObject* args);
static PyObject* _gfx_arccolor (PyObject *self, PyObject* args);
static PyObject* _gfx_aacirclecolor (PyObject *self, PyObject* args);
static PyObject* _gfx_filledcirclecolor (PyObject *self, PyObject* args);
static PyObject* _gfx_ellipsecolor (PyObject *self, PyObject* args);
static PyObject* _gfx_aaellipsecolor (PyObject *self, PyObject* args);
static PyObject* _gfx_filledellipsecolor (PyObject *self, PyObject* args);
static PyObject* _gfx_piecolor (PyObject *self, PyObject* args);
static PyObject* _gfx_filledpiecolor (PyObject *self, PyObject* args);
static PyObject* _gfx_trigoncolor (PyObject *self, PyObject* args);
static PyObject* _gfx_aatrigoncolor (PyObject *self, PyObject* args);
static PyObject* _gfx_filledtrigoncolor (PyObject *self, PyObject* args);
static PyObject* _gfx_polygoncolor (PyObject *self, PyObject* args);
static PyObject* _gfx_aapolygoncolor (PyObject *self, PyObject* args);
static PyObject* _gfx_filledpolygoncolor (PyObject *self, PyObject* args);
static PyObject* _gfx_texturedpolygon (PyObject *self, PyObject* args);
static PyObject* _gfx_beziercolor (PyObject *self, PyObject* args);

static PyMethodDef _gfx_methods[] = {
    { "pixel", _gfx_pixelcolor, METH_VARARGS, DOC_PRIMITIVES_PIXEL },
    { "hline", _gfx_hlinecolor, METH_VARARGS, DOC_PRIMITIVES_HLINE },
    { "vline", _gfx_vlinecolor, METH_VARARGS, DOC_PRIMITIVES_VLINE },
    { "rectangle", _gfx_rectanglecolor, METH_VARARGS,
      DOC_PRIMITIVES_RECTANGLE },
    { "box", _gfx_boxcolor, METH_VARARGS, DOC_PRIMITIVES_BOX },
    { "aaline", _gfx_aalinecolor, METH_VARARGS, DOC_PRIMITIVES_AALINE },
    { "line", _gfx_linecolor, METH_VARARGS, DOC_PRIMITIVES_LINE },
    { "arc", _gfx_arccolor, METH_VARARGS, DOC_PRIMITIVES_ARC },
    { "circle", _gfx_circlecolor, METH_VARARGS, DOC_PRIMITIVES_CIRCLE },
    { "aacircle", _gfx_aacirclecolor, METH_VARARGS, DOC_PRIMITIVES_AACIRCLE },
    { "filled_circle", _gfx_filledcirclecolor, METH_VARARGS,
      DOC_PRIMITIVES_FILLED_CIRCLE },
    { "ellipse", _gfx_ellipsecolor, METH_VARARGS, DOC_PRIMITIVES_ELLIPSE },
    { "aaellipse", _gfx_aaellipsecolor, METH_VARARGS,
      DOC_PRIMITIVES_AAELLIPSE },
    { "filled_ellipse", _gfx_filledellipsecolor, METH_VARARGS, 
      DOC_PRIMITIVES_FILLED_ELLIPSE },
    { "pie", _gfx_piecolor, METH_VARARGS, DOC_PRIMITIVES_PIE },
    { "filled_pie", _gfx_filledpiecolor, METH_VARARGS,
      DOC_PRIMITIVES_FILLED_PIE },
    { "trigon", _gfx_trigoncolor, METH_VARARGS, DOC_PRIMITIVES_TRIGON },
    { "aatrigon", _gfx_aatrigoncolor, METH_VARARGS, DOC_PRIMITIVES_AATRIGON },
    { "filled_trigon", _gfx_filledtrigoncolor, METH_VARARGS,
      DOC_PRIMITIVES_FILLED_TRIGON },
    { "polygon", _gfx_polygoncolor, METH_VARARGS, DOC_PRIMITIVES_POLYGON },
    { "aapolygon", _gfx_aapolygoncolor, METH_VARARGS,
      DOC_PRIMITIVES_AAPOLYGON },
    { "filled_polygon", _gfx_filledpolygoncolor, METH_VARARGS,
      DOC_PRIMITIVES_FILLED_POLYGON },
    { "textured_polygon", _gfx_texturedpolygon, METH_VARARGS,
      DOC_PRIMITIVES_TEXTURED_POLYGON },
    { "bezier", _gfx_beziercolor, METH_VARARGS, DOC_PRIMITIVES_BEZIER },
    { NULL, NULL, 0, NULL },
};

static PyObject*
_gfx_pixelcolor (PyObject *self, PyObject* args)
{
    PyObject *surface, *color, *pt;
    int x, y;
    pguint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OOO:pixel", &surface, &pt, &color))
    {
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "OiiO:pixel", &surface, &x, &y, &color))
            return NULL;
    }
    else
    {
        if (!PointFromObj (pt, &x, &y))
            return NULL;
    }
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }

    if (!ColorFromObj (color, &c))
        return NULL;

    if (pixelColor (((PySDLSurface*)surface)->surface, (Sint16)x, (Sint16)y,
            (Uint32)c)== -1)
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
    int x1, x2, y;
    pguint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OiiiO:hline", &surface, &x1, &x2, &y, &color))
        return NULL;
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }

    if (!ColorFromObj (color, &c))
        return NULL;

    if (hlineColor (((PySDLSurface*)surface)->surface,
            (Sint16)x1, (Sint16)x2, y, (Uint32)c) == -1)
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
    int x, _y1, y2;
    pguint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OiiiO:vline", &surface, &x, &_y1, &y2,
            &color))
        return NULL;
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }

    if (!ColorFromObj (color, &c))
        return NULL;

    if (vlineColor (((PySDLSurface*)surface)->surface, (Sint16) x, (Sint16)_y1,
            (Sint16)y2, (Uint32)c) == -1)
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
    pguint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OOO:rectangle", &surface, &rect, &color))
        return NULL;
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!SDLRectFromRect (rect, &sdlrect))
        return NULL;

    x1 = sdlrect.x;
    _y1 = sdlrect.y;
    x2 = (Sint16) (sdlrect.x + sdlrect.w);
    y2 = (Sint16) (sdlrect.y + sdlrect.h);
    if (!ColorFromObj (color, &c))
        return NULL;

    if (rectangleColor (((PySDLSurface*)surface)->surface, x1, _y1, x2, y2,
            (Uint32)c) == -1)
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
    pguint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OOO:box", &surface, &rect, &color))
        return NULL;
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!SDLRectFromRect (rect, &sdlrect))
        return NULL;

    x1 = sdlrect.x;
    _y1 = sdlrect.y;
    x2 = (Sint16) (sdlrect.x + sdlrect.w);
    y2 = (Sint16) (sdlrect.y + sdlrect.h);
    if (!ColorFromObj (color, &c))
        return NULL;
 
    if (boxColor (((PySDLSurface*)surface)->surface, x1, _y1, x2, y2,
            (Uint32)c) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_gfx_aalinecolor (PyObject *self, PyObject* args)
{
    PyObject *surface, *color;
    PyObject *p1, *p2;
    int x1, x2, _y1, y2;
    pguint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OOOO:aaline", &surface, &p1, &p2, &color))
    {
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "OiiiiO:aaline", &surface, &x1, &_y1,
            &x2, &y2, &color))
        return NULL;
    }
    else
    {
        if (!PointFromObj (p1, &x1, &_y1) || !PointFromObj (p2, &x2, &y2))
            return NULL;
    }
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }

    if (!ColorFromObj (color, &c))
        return NULL;

    if (aalineColor (((PySDLSurface*)surface)->surface, 
            (Sint16)x1, (Sint16)_y1, (Sint16)x2, (Sint16)y2, (Uint32)c) == -1)
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
    PyObject *p1, *p2;
    int x1, x2, _y1, y2;
    pguint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OOOO:line", &surface, &p1, &p2, &color))
    {
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "OiiiiO:line", &surface, &x1, &_y1,
            &x2, &y2, &color))
        return NULL;
    }
    else
    {
        if (!PointFromObj (p1, &x1, &_y1) || !PointFromObj (p2, &x2, &y2))
            return NULL;
    }
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }

    if (!ColorFromObj (color, &c))
        return NULL;

    if (lineColor (((PySDLSurface*)surface)->surface, 
            (Sint16)x1, (Sint16)_y1, (Sint16)x2, (Sint16)y2, (Uint32)c) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_gfx_circlecolor (PyObject *self, PyObject* args)
{
    PyObject *surface, *color, *pt;
    int x, y, r;
    pguint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OOiO:circle", &surface, &pt, &r, &color))
    {
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "OiiiO:circle", &surface, &x, &y, &r,
            &color))
        return NULL;
    }
    else
    {
        if (!PointFromObj (pt, &x, &y))
            return NULL;
    }
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }

    if (!ColorFromObj (color, &c))
        return NULL;

    if (circleColor (((PySDLSurface*)surface)->surface,
            (Sint16)x, (Sint16)y, (Sint16)r, (Uint32)c) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_gfx_arccolor (PyObject *self, PyObject* args)
{
    PyObject *surface, *color, *pt;
    int x, y, r, start, end;
    pguint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OOiiiO:arc", &surface, &pt, &r, &start, &end,
        &color))
    {
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "OiiiiiO:arc", &surface, &x, &y, &r,
                &start, &end, &color))
            return NULL;
    }
    else
    {
        if (!PointFromObj (pt, &x, &y))
            return NULL;
    }

    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }

    if (!ColorFromObj (color, &c))
        return NULL;

    if (arcColor (((PySDLSurface*)surface)->surface, (Sint16)x, (Sint16)y,
            (Sint16)r, (Sint16)start, (Sint16)end, (Uint32)c) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_gfx_aacirclecolor (PyObject *self, PyObject* args)
{
    PyObject *surface, *color, *pt;
    int x, y, r;
    pguint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OOiO:aacircle", &surface, &pt, &r, &color))
    {
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "OiiiO:aacircle", &surface, &x, &y, &r,
                &color))
            return NULL;
    }
    else
    {
        if (!PointFromObj (pt, &x, &y))
            return NULL;
    }

    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!ColorFromObj (color, &c))
        return NULL;

    if (aacircleColor (((PySDLSurface*)surface)->surface,
            (Sint16)x, (Sint16)y, (Sint16)r, (Uint32)c) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_gfx_filledcirclecolor (PyObject *self, PyObject* args)
{
    PyObject *surface, *color, *pt;
    int x, y, r;
    pguint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OOiO:filledcircle", &surface, &pt, &r,
        &color))
    {
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "OiiiO:filledcircle", &surface, &x, &y,
            &r, &color))
            return NULL;
    }
    else
    {
        if (!PointFromObj (pt, (int*)&x, (int*)&y))
            return NULL;
    }
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!ColorFromObj (color, &c))
        return NULL;

    if (filledCircleColor (((PySDLSurface*)surface)->surface,
            (Sint16)x, (Sint16)y, (Sint16)r, (Uint32)c) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_gfx_ellipsecolor (PyObject *self, PyObject* args)
{
    PyObject *surface, *color, *pt, *rd;
    int x, y, rx, ry;
    pguint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OOOO:ellipse", &surface, &pt, &rd, &color))
    {
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "OiiiiO:ellipse", &surface, &x, &y, &rx,
            &ry, &color))
            return NULL;
    }
    else
    {
        if (!PointFromObj (pt, &x, &y) || !PointFromObj (rd, &rx, &ry))
            return NULL;
    }
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!ColorFromObj (color, &c))
        return NULL;

    if (ellipseColor (((PySDLSurface*)surface)->surface, (Sint16)x, (Sint16)y,
            (Sint16)rx, (Sint16)ry, (Uint32)c) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_gfx_aaellipsecolor (PyObject *self, PyObject* args)
{
    PyObject *surface, *color, *pt, *rd;
    int x, y, rx, ry;
    pguint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OOOO:aaellipse", &surface, &pt, &rd, &color))
    {
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "OiiiiO:aaellipse", &surface, &x, &y, &rx,
            &ry, &color))
            return NULL;
    }
    else
    {
        if (!PointFromObj (pt, &x, &y) || !PointFromObj (rd, &rx, &ry))
            return NULL;
    }
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!ColorFromObj (color, &c))
        return NULL;

    if (aaellipseColor (((PySDLSurface*)surface)->surface,
            (Sint16)x, (Sint16)y, (Sint16)rx, (Sint16)ry, (Uint32)c) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_gfx_filledellipsecolor (PyObject *self, PyObject* args)
{
    PyObject *surface, *color, *pt, *rd;
    int x, y, rx, ry;
    pguint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OOOO:filled_ellipse", &surface, &pt, &rd,
        &color))
    {
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "OiiiiO:filled_ellipse", &surface, &x, &y,
            &rx, &ry, &color))
            return NULL;
    }
    else
    {
        if (!PointFromObj (pt, &x, &y) || !PointFromObj (rd, &rx, &ry))
            return NULL;
    }

    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!ColorFromObj (color, &c))
        return NULL;

    if (filledEllipseColor (((PySDLSurface*)surface)->surface,
            (Sint16)x, (Sint16)y, (Sint16)rx, (Sint16)ry, (Uint32)c) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_gfx_piecolor (PyObject *self, PyObject* args)
{
    PyObject *surface, *color, *pt;
    int x, y, r, start, end;
    pguint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OOiiiO:pie", &surface, &pt, &r,
            &start, &end, &color))
    {
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "OiiiiiO:pie", &surface, &x, &y, &r,
                &start, &end, &color))
            return NULL;
    }
    else
    {
        if (!PointFromObj (pt, &x, &y))
            return NULL;
    }
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!ColorFromObj (color, &c))
        return NULL;

    if (pieColor (((PySDLSurface*)surface)->surface, (Sint16)x, (Sint16)y,
            (Sint16)r, (Sint16)start, (Sint16)end, (Uint32)c) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_gfx_filledpiecolor (PyObject *self, PyObject* args)
{
    PyObject *surface, *color, *pt;
    int x, y, r, start, end;
    pguint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OOiiiO:filled_pie", &surface, &pt, &r,
            &start, &end, &color))
    {
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "OiiiiiO:filled_pie", &surface, &x, &y, &r,
                &start, &end, &color))
            return NULL;
    }
    else
    {
        if (!PointFromObj (pt, (int*)&x, (int*)&y))
            return NULL;
    }
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }

    if (!ColorFromObj (color, &c))
        return NULL;

    if (filledPieColor (((PySDLSurface*)surface)->surface, (Sint16)x,
            (Sint16)y, (Sint16)r, (Sint16)start, (Sint16)end, (Uint32)c) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_gfx_trigoncolor (PyObject *self, PyObject* args)
{
    PyObject *surface, *color, *p1, *p2, *p3;
    int x1, x2, x3, _y1, y2, y3;
    pguint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OOOOO:trigon", &surface, &p1, &p2, &p3,
        &color))
    {
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "OiiiiiiO:trigon", &surface, &x1, &_y1,
            &x2, &y2, &x3, &y3, &color))
            return NULL;
    }
    else
    {
        if (!PointFromObj (p1, &x1, &_y1) ||
            !PointFromObj (p2, &x2, &y2) ||
            !PointFromObj (p3, &x3, &y3))
            return NULL;
    }
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }

    if (!ColorFromObj (color, &c))
        return NULL;

    if (trigonColor (((PySDLSurface*)surface)->surface,
            (Sint16)x1, (Sint16)_y1, (Sint16)x2, (Sint16)y2,
            (Sint16)x3, (Sint16)y3, (Uint32)c) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_gfx_aatrigoncolor (PyObject *self, PyObject* args)
{
    PyObject *surface, *color, *p1, *p2, *p3;
    int x1, x2, x3, _y1, y2, y3;
    pguint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OOOOO:aatrigon", &surface, &p1, &p2, &p3,
        &color))
    {
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "OiiiiiiO:aatrigon", &surface, &x1, &_y1,
            &x2, &y2, &x3, &y3, &color))
            return NULL;
    }
    else
    {
        if (!PointFromObj (p1, &x1, &_y1) ||
            !PointFromObj (p2, &x2, &y2) ||
            !PointFromObj (p3, &x3, &y3))
            return NULL;
    }
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }

    if (!ColorFromObj (color, &c))
        return NULL;

    if (aatrigonColor (((PySDLSurface*)surface)->surface,
            (Sint16)x1, (Sint16)_y1, (Sint16)x2, (Sint16)y2,
            (Sint16)x3, (Sint16)y3, (Uint32)c) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_gfx_filledtrigoncolor (PyObject *self, PyObject* args)
{
    PyObject *surface, *color, *p1, *p2, *p3;
    int x1, x2, x3, _y1, y2, y3;
    pguint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OOOOO:filled_trigon", &surface, &p1, &p2, &p3,
        &color))
    {
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "OiiiiiiO:filled_trigon", &surface, &x1,
            &_y1, &x2, &y2, &x3, &y3, &color))
            return NULL;
    }
    else
    {
        if (!PointFromObj (p1, &x1, &_y1) ||
            !PointFromObj (p2, &x2, &y2) ||
            !PointFromObj (p3, &x3, &y3))
            return NULL;
    }
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }

    if (!ColorFromObj (color, &c))
        return NULL;

    if (filledTrigonColor (((PySDLSurface*)surface)->surface,
            (Sint16)x1, (Sint16)_y1, (Sint16)x2, (Sint16)y2,
            (Sint16)x3, (Sint16)y3, (Uint32)c) == -1)
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
    Sint16 *vx, *vy;
    int tmp1, tmp2;
    Py_ssize_t count, i;
    int ret;
    pguint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OOO:polygon", &surface, &points, &color))
        return NULL;
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
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

    if (!ColorFromObj (color, &c))
        return NULL;

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
        if (!PointFromObj (item, &tmp1, &tmp2))
        {
            PyMem_Free (vx);
            PyMem_Free (vy);
            Py_XDECREF (item);
            return NULL;
        }
        Py_DECREF (item);
        vx[i] = (Sint16)tmp1;
        vy[i] = (Sint16)tmp2;
    }

    Py_BEGIN_ALLOW_THREADS;
    ret = polygonColor (((PySDLSurface*)surface)->surface, vx, vy, (int)count,
        (Uint32)c);
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
    Sint16 *vx, *vy;
    int tmp1, tmp2;
    Py_ssize_t count, i;
    int ret;
    pguint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OOO:aapolygon", &surface, &points, &color))
        return NULL;
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
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

    if (!ColorFromObj (color, &c))
        return NULL;

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
        if (!PointFromObj (item, &tmp1, &tmp2))
        {
            PyMem_Free (vx);
            PyMem_Free (vy);
            Py_XDECREF (item);
            return NULL;
        }
        Py_DECREF (item);
        vx[i] = (Sint16)tmp1;
        vy[i] = (Sint16)tmp2;
    }

    Py_BEGIN_ALLOW_THREADS;
    ret = aapolygonColor (((PySDLSurface*)surface)->surface, vx, vy, (int)count,
        (Uint32)c);
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
    Sint16 *vx, *vy;
    int tmp1, tmp2;
    Py_ssize_t count, i;
    int ret;
    pguint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OOO:filled_polygon", &surface, &points,
            &color))
        return NULL;
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
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

    if (!ColorFromObj (color, &c))
        return NULL;

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
        if (!PointFromObj (item, &tmp1, &tmp2))
        {
            PyMem_Free (vx);
            PyMem_Free (vy);
            Py_XDECREF (item);
            return NULL;
        }
        Py_DECREF (item);
        vx[i] = (Sint16)tmp1;
        vy[i] = (Sint16)tmp2;
    }

    Py_BEGIN_ALLOW_THREADS;
    ret = filledPolygonColor (((PySDLSurface*)surface)->surface, vx, vy,
        (int)count, (Uint32)c);
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
    PyObject *surface, *texture, *points, *item, *pt;
    Sint16 *vx, *vy;
    int tmp1, tmp2, tdx, tdy;
    Py_ssize_t count, i;
    int ret;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OOOO:textured_polygon", &surface, &points,
            &texture, &pt))
    {
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "OOOii:textured_polygon", &surface,
            &points, &texture, &tdx, &tdy))
            return NULL;
    }
    else
    {
        if (!PointFromObj (pt, &tdx, &tdy))
            return NULL;
    }
    
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
        if (!PointFromObj (item, &tmp1, &tmp2))
        {
            PyMem_Free (vx);
            PyMem_Free (vy);
            Py_XDECREF (item);
            return NULL;
        }
        Py_DECREF (item);
        vx[i] = (Sint16)tmp1;
        vy[i] = (Sint16)tmp2;
    }

    Py_BEGIN_ALLOW_THREADS;
    ret = texturedPolygon (((PySDLSurface*)surface)->surface, vx, vy,
        (int)count, ((PySDLSurface*)texture)->surface, (Sint16)tdx,
        (Sint16)tdy);
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
    Sint16 *vx, *vy;
    int x, y;
    Py_ssize_t count, i;
    int ret, steps;
    pguint32 c;

    ASSERT_VIDEO_INIT (NULL);

    if (!PyArg_ParseTuple (args, "OOiO:bezier", &surface, &points, &steps,
            &color))
        return NULL;
    
    if (!PySDLSurface_Check (surface))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
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

    if (!ColorFromObj (color, &c))
        return NULL;

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
        if (!PointFromObj (item, &x, &y))
        {
            PyMem_Free (vx);
            PyMem_Free (vy);
            Py_XDECREF (item);
            return NULL;
        }
        Py_DECREF (item);
        vx[i] = (Sint16)x;
        vy[i] = (Sint16)y;
    }

    Py_BEGIN_ALLOW_THREADS;
    ret = bezierColor (((PySDLSurface*)surface)->surface, vx, vy, (int)count,
        steps, (Uint32)c);
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

#ifdef IS_PYTHON_3
PyMODINIT_FUNC PyInit_primitives (void)
#else
PyMODINIT_FUNC initprimitives (void)
#endif
{
    PyObject *mod;
    
#ifdef IS_PYTHON_3
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "primitives",
        DOC_PRIMITIVES,
        -1,
        _gfx_methods,
        NULL, NULL, NULL, NULL
    };
    mod = PyModule_Create (&_module);
#else
    mod = Py_InitModule3 ("primitives", _gfx_methods, DOC_PRIMITIVES);
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
