/*
    pygame - Python Game Library
    Copyright (C) 2000-2001  Pete Shinners

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

#define PYGAME_SDLEXTDRAW_INTERNAL

#include "pgsdlext.h"
#include "pgsdl.h"
#include "draw.h"
#include "sdlextdraw_doc.h"
#include "surface.h"

static PyObject* _draw_aaline (PyObject* self, PyObject* args);
static PyObject* _draw_line (PyObject* self, PyObject* args);
static PyObject* _draw_aalines (PyObject* self, PyObject* args);
static PyObject* _draw_lines (PyObject* self, PyObject* args);
static PyObject* _draw_ellipse (PyObject* self, PyObject* args);
static PyObject* _draw_arc (PyObject* self, PyObject* args);
static PyObject* _draw_circle (PyObject* self, PyObject* args);
static PyObject* _draw_polygon (PyObject* self, PyObject* args);
static PyObject* _draw_aapolygon (PyObject* self, PyObject* args);
static PyObject* _draw_rect (PyObject* self, PyObject* args);

static PyMethodDef _draw_methods[] =
{
    { "aaline", _draw_aaline, METH_VARARGS, DOC_DRAW_AALINE },
    { "line", _draw_line, METH_VARARGS, DOC_DRAW_LINE },
    { "aalines", _draw_aalines, METH_VARARGS, DOC_DRAW_AALINES },
    { "lines", _draw_lines, METH_VARARGS, DOC_DRAW_LINES },
    { "ellipse", _draw_ellipse, METH_VARARGS, DOC_DRAW_ELLIPSE },
    { "arc", _draw_arc, METH_VARARGS, DOC_DRAW_ARC },
    { "circle", _draw_circle, METH_VARARGS, DOC_DRAW_CIRCLE },
    { "polygon", _draw_polygon, METH_VARARGS, DOC_DRAW_POLYGON },
    { "aapolygon", _draw_aapolygon, METH_VARARGS, DOC_DRAW_AAPOLYGON },
    { "rect", _draw_rect, METH_VARARGS, DOC_DRAW_RECT },
    { NULL, NULL, 0, NULL }
};

static PyObject*
_draw_aaline (PyObject* self, PyObject* args)
{
    PyObject *surfobj, *colorobj;
    PyObject *p1, *p2;
    SDL_Surface* surface;
    SDL_Rect area;
    Uint32 color;
    int x1, _y1, x2, y2, blend = 0;
    int drawn = 0;
    
    if (!PyArg_ParseTuple (args, "OOOO|i:aaline", &surfobj, &colorobj, &p1,
        &p2, &blend))
    {
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "OOiiii|i:aaline", &surfobj, &colorobj,
                &x1, &_y1, &x2, &y2, &blend))
            return NULL;
    }
    else
    {
        if (!PointFromObject (p1, &x1, &_y1) ||
            !PointFromObject (p2, &x2, &y2))
            return NULL;
    }
    
    if (!PySDLSurface_Check (surfobj))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    surface = ((PySDLSurface*)surfobj)->surface;
    
    if (!ColorFromObj (colorobj, surface->format, &color))
        return NULL;
    
    if (surface->format->BytesPerPixel != 3 &&
        surface->format->BytesPerPixel != 4)
    {
        PyErr_SetString (PyExc_ValueError,
            "unsupported bit depth for aaline draw (supports 32 & 24 bit)");
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS;
    drawn = pyg_draw_aaline (surface, &surface->clip_rect, color, x1, _y1,
        x2, y2, blend, &area);
    Py_END_ALLOW_THREADS;

    if (!drawn)
        return PyRect_New (x1, _y1, 0, 0);
    return PyRect_New (area.x, area.y, area.w, area.h);
}

static PyObject*
_draw_line (PyObject* self, PyObject* args)
{
    PyObject *surfobj, *colorobj;
    PyObject *p1, *p2;
    SDL_Surface* surface;
    SDL_Rect area;
    Uint32 color;
    int x1, _y1, x2, y2, width = 1;
    int drawn = 0;
    
    if (!PyArg_ParseTuple (args, "OOOO|i:line", &surfobj, &colorobj, &p1, &p2,
        &width))
    {
        PyErr_Clear ();
        if (!PyArg_ParseTuple (args, "OOiiii|i:line", &surfobj, &colorobj, &x1,
                &_y1, &x2, &y2, &width))
            return NULL;
    }
    else
    {
        if (!PointFromObject (p1, &x1, &_y1) ||
            !PointFromObject (p2, &x2, &y2))
            return NULL;
    }
    
    if (!PySDLSurface_Check (surfobj))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    surface = ((PySDLSurface*)surfobj)->surface;
    
    if (!ColorFromObj (colorobj, surface->format, &color))
        return NULL;
    
    if (width < 1)
    {
        PyErr_SetString (PyExc_ValueError, "width must be >= 1");
        return NULL;
    }
    
    Py_BEGIN_ALLOW_THREADS;
    drawn = pyg_draw_line (surface, &surface->clip_rect, color, x1, _y1,
        x2, y2, width, &area);
    Py_END_ALLOW_THREADS;

    if (!drawn)
        return PyRect_New (x1, _y1, 0, 0);
    return PyRect_New (area.x, area.y, area.w, area.h);
}

static PyObject*
_draw_aalines (PyObject* self, PyObject* args)
{
    PyObject *surfobj, *colorobj, *list, *item;
    SDL_Surface* surface;
    SDL_Rect area;
    Uint32 color;
    int *xpts, *ypts, x, y;
    Py_ssize_t i, count;
    int left, right, bottom, top;
    int drawn = 0, blend = 0;
    
    if (!PyArg_ParseTuple (args, "OOO|i:aalines", &surfobj, &colorobj, &list,
            &blend))
        return NULL;
    if (!PySDLSurface_Check (surfobj))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    surface = ((PySDLSurface*)surfobj)->surface;
    
    if (!ColorFromObj (colorobj, surface->format, &color))
        return NULL;
    
    if (!PySequence_Check (list))
    {
        PyErr_SetString (PyExc_TypeError,
            "points must be a sequence of points");
        return NULL;
    }

    if (surface->format->BytesPerPixel != 3 &&
        surface->format->BytesPerPixel != 4)
    {
        PyErr_SetString (PyExc_ValueError,
            "unsupported bit depth for aalines draw (supports 32 & 24 bit)");
        return NULL;
    }

    count = PySequence_Size (list);
    if (count < 2)
    {
        PyErr_SetString (PyExc_ValueError,
            "points must contain at least two points");
        return NULL;
    }

    /* Get all points */
    xpts = PyMem_New (int, (size_t) count);
    ypts = PyMem_New (int, (size_t) count);
    if (!xpts || !ypts)
    {
        if (xpts)
            PyMem_Free (xpts);
        if (ypts)
            PyMem_Free (ypts);
        return NULL;
    }

    left = right = bottom = top = 0;
    for (i = 0; i < count; i++)
    {
        item = PySequence_ITEM (list, i);
        if (!PointFromObject (item, &x, &y))
        {
            Py_XDECREF (item);
            PyMem_Free (xpts);
            PyMem_Free (ypts);
            PyErr_SetString (PyExc_ValueError, "invalid point list");
            return NULL;
        }
        left = MIN (x, left);
        right = MAX (x, right);
        xpts[i] = x;

        top = MIN (y, top);
        bottom = MAX (y, bottom);
        ypts[i] = y;
        
        Py_DECREF (item);
    }

    Py_BEGIN_ALLOW_THREADS;
    drawn = pyg_draw_aalines (surface, &surface->clip_rect, color, xpts, ypts,
        (unsigned int)count, blend, &area);
    Py_END_ALLOW_THREADS;

    PyMem_Free (xpts);
    PyMem_Free (ypts);

    if (!drawn)
        PyRect_New (left, top, 0, 0);
    return PyRect_New (area.x, area.y, area.w, area.h);
}

static PyObject*
_draw_lines (PyObject* self, PyObject* args)
{
    PyObject *surfobj, *colorobj, *list, *item;
    SDL_Surface* surface;
    SDL_Rect area;
    Uint32 color;
    int *xpts, *ypts, x, y;
    Py_ssize_t i, count;
    int left, right, bottom, top;
    int drawn = 0, width = 1;
    
    if (!PyArg_ParseTuple (args, "OOO|i:lines", &surfobj, &colorobj, &list,
            &width))
        return NULL;
    if (!PySDLSurface_Check (surfobj))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    surface = ((PySDLSurface*)surfobj)->surface;
    
    if (!ColorFromObj (colorobj, surface->format, &color))
        return NULL;
    
    if (!PySequence_Check (list))
    {
        PyErr_SetString (PyExc_TypeError,
            "points must be a sequence of points");
        return NULL;
    }
    if (width < 1)
    {
        PyErr_SetString (PyExc_ValueError, "width must be >= 1");
        return NULL;
    }
    
    if (surface->format->BytesPerPixel != 3 &&
        surface->format->BytesPerPixel != 4)
    {
        PyErr_SetString (PyExc_ValueError,
            "unsupported bit depth for aalines draw (supports 32 & 24 bit)");
        return NULL;
    }
    
    count = PySequence_Size (list);
    if (count < 2)
    {
        PyErr_SetString (PyExc_ValueError,
            "points must contain at least two points");
        return NULL;
    }

    /* Get all points */
    xpts = PyMem_New (int, (size_t) count);
    ypts = PyMem_New (int, (size_t) count);
    if (!xpts || !ypts)
    {
        if (xpts)
            PyMem_Free (xpts);
        if (ypts)
            PyMem_Free (ypts);
        return NULL;
    }

    left = right = bottom = top = 0;
    for (i = 0; i < count; i++)
    {
        item = PySequence_ITEM (list, i);
        if (!PointFromObject (item, &x, &y))
        {
            Py_XDECREF (item);
            PyMem_Free (xpts);
            PyMem_Free (ypts);
            PyErr_SetString (PyExc_ValueError, "invalid point list");
            return NULL;
        }

        left = MIN (x, left);
        right = MAX (x, right);
        xpts[i] = x;

        top = MIN (y, top);
        bottom = MAX (y, bottom);
        ypts[i] = y;

        Py_DECREF (item);
    }

    Py_BEGIN_ALLOW_THREADS;
    drawn = pyg_draw_lines (surface, &surface->clip_rect, color, xpts, ypts,
        (unsigned int)count, width, &area);
    Py_END_ALLOW_THREADS;

    PyMem_Free (xpts);
    PyMem_Free (ypts);

    if (!drawn)
        return PyRect_New (left, top, 0, 0);
    return PyRect_New (area.x, area.y, area.w, area.h);
}

static PyObject*
_draw_ellipse (PyObject* self, PyObject* args)
{
    PyObject *surfobj, *colorobj, *rectobj;
    SDL_Surface* surface;
    Uint32 color;
    int width = 0, loop;
    pgint16 l, t, r, b;
    pguint16 w, h;
    SDL_Rect rect;

    if (!PyArg_ParseTuple (args, "OOO|i:ellipse", &surfobj, &colorobj, &rectobj,
            &width))
        return NULL;
    if (!PySDLSurface_Check (surfobj))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    surface = ((PySDLSurface*)surfobj)->surface;
    
    if (!ColorFromObj (colorobj, surface->format, &color))
        return NULL;

    if (!SDLRect_FromRect (rectobj, &rect))
        return NULL;

    if (width < 0)
    {
        PyErr_SetString (PyExc_ValueError, "width must not be negative");
        return NULL;
    }

    if (((pguint16)width) > rect.w / 2 || ((pguint16)width) > rect.h / 2)
    {
        PyErr_SetString (PyExc_ValueError,
            "width must not be greater than ellipse radius");
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS;
    w = rect.w / 2;
    h = rect.h / 2;
    if (width == 0)
    {
        pyg_draw_filled_ellipse (surface, &surface->clip_rect, color,
            (int) (rect.x + w), (int) (rect.y + h), (int) w, (int) h, NULL);
    }
    else
    {
        width = MIN (((pguint16)width), MIN (rect.w, rect.h) / 2);
        for (loop = 0; loop < width; loop++)
        {
            pyg_draw_ellipse (surface, &surface->clip_rect, color,
                (int) (rect.x + w), (int) (rect.y + h),
                (int) (w - loop), (int) (h - loop), NULL);
        }

    }
    Py_END_ALLOW_THREADS;

    l = MAX (rect.x, surface->clip_rect.x);
    t = MAX (rect.y, surface->clip_rect.y);
    r = MIN ((int) (rect.x + rect.w),
        (int)(surface->clip_rect.x + surface->clip_rect.w));
    b = MIN ((int)(rect.y + rect.h),
        (int)(surface->clip_rect.y + surface->clip_rect.h));

    return PyRect_New (l, t, (pguint16)MAX (r - l, 0),
        (pguint16)MAX (b - t, 0));
}

static PyObject*
_draw_arc (PyObject* self, PyObject* args)
{
    PyObject *surfobj, *colorobj, *rectobj;
    SDL_Surface* surface;
    Uint32 color;
    int width = 1, loop;
    pgint16 l, t;
    pguint16 w, h, r, b;
    SDL_Rect rect;
    double astart, astop;

    if (!PyArg_ParseTuple (args, "OOOdd|i:arc", &surfobj, &colorobj, &rectobj,
            &astart, &astop, &width))
        return NULL;
    if (!PySDLSurface_Check (surfobj))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    surface = ((PySDLSurface*)surfobj)->surface;
    
    if (!ColorFromObj (colorobj, surface->format, &color))
        return NULL;

    if (!SDLRect_FromRect (rectobj, &rect))
        return NULL;

    if (width < 0)
    {
        PyErr_SetString (PyExc_ValueError, "width must not be negative");
        return NULL;
    }

    if (((pguint16)width) > rect.w / 2 || ((pguint16)width) > rect.h / 2)
    {
        PyErr_SetString (PyExc_ValueError,
            "width must not be greater than ellipse radius");
        return NULL;
    }

    while (astop < astart)
        astop += 360;
    width = MIN (((pguint16)width), MIN (rect.w, rect.h) / 2);

    Py_BEGIN_ALLOW_THREADS;
    w = rect.w / 2;
    h = rect.h / 2;
    for (loop = 0; loop < width; loop++)
    {
        pyg_draw_arc (surface, &surface->clip_rect, color,
            (int) (rect.x + w), (int)(rect.y + h), (int) (w - loop),
            (int) (h - loop), DEG2RAD(astart), DEG2RAD(astop), NULL);
    }
    Py_END_ALLOW_THREADS;

    l = MAX (rect.x, surface->clip_rect.x);
    t = MAX (rect.y, surface->clip_rect.y);
    r = MIN ((int)(rect.x + rect.w),
        (int)(surface->clip_rect.x + surface->clip_rect.w));
    b = MIN ((int)(rect.y + rect.h),
        (int)(surface->clip_rect.y + surface->clip_rect.h));

    return PyRect_New (l, t, (pguint16)MAX (r - l, 0),
        (pguint16)MAX (b - t, 0));
}

static PyObject*
_draw_circle (PyObject* self, PyObject* args)
{
    PyObject *surfobj, *colorobj;
    PyObject *pt;
    SDL_Surface* surface;
    Uint32 color;
    int px, py, radius, loop, width = 0;
    pgint16 l, t;
    pguint16 r, b;

    if (!PyArg_ParseTuple (args, "OOOi|i:circle", &surfobj, &colorobj, &pt,
        &radius, &width))
    {
        if (!PyArg_ParseTuple (args, "OOiii|i:circle", &surfobj, &colorobj,
                &px, &py, &radius, &width))
            return NULL;
    }
    else
    {
        if (!PointFromObject (pt, &px, &py))
            return NULL;
    }

    if (!PySDLSurface_Check (surfobj))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    surface = ((PySDLSurface*)surfobj)->surface;
    
    if (!ColorFromObj (colorobj, surface->format, &color))
        return NULL;

    if (radius < 0)
    {
        PyErr_SetString (PyExc_ValueError, "radius must not be negative");
        return NULL;
    }
    if (width < 0)
    {
        PyErr_SetString (PyExc_ValueError, "width must not be negative");
        return NULL;
    }
    if (width > radius)
    {
        PyErr_SetString (PyExc_ValueError,
            "width must not be greater than radius");
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS;
    if (width == 0)
    {
        pyg_draw_filled_ellipse (surface, &surface->clip_rect, color, px, py,
            radius, radius, NULL);
    }
    else
    {
        for (loop = 0; loop < width; loop++)
            pyg_draw_ellipse (surface, &surface->clip_rect, color, px, py,
                radius - loop, radius - loop, NULL);
    }
    Py_END_ALLOW_THREADS;

    l = MAX (px - radius, surface->clip_rect.x);
    t = MAX (py - radius, surface->clip_rect.y);
    r = MIN (px + radius, surface->clip_rect.x + surface->clip_rect.w);
    b = MIN (py + radius, surface->clip_rect.y + surface->clip_rect.h);

    return PyRect_New (l, t, (pguint16)MAX (r - l, 0),
        (pguint16)MAX (b - t, 0));
}

static PyObject*
_draw_polygon (PyObject* self, PyObject* args)
{
    PyObject *surfobj, *colorobj, *list, *item;
    SDL_Surface* surface;
    SDL_Rect area;
    Uint32 color;
    int *xpts, *ypts, x, y;
    Py_ssize_t i, count;
    int left, right, bottom, top;
    int drawn = 0, width = 1;
    
    if (!PyArg_ParseTuple (args, "OOO|i:polygon", &surfobj, &colorobj, &list,
            &width))
        return NULL;
    if (!PySDLSurface_Check (surfobj))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    surface = ((PySDLSurface*)surfobj)->surface;
    
    if (!ColorFromObj (colorobj, surface->format, &color))
        return NULL;

    if (!PySequence_Check (list))
    {
        PyErr_SetString (PyExc_TypeError,
            "points must be a sequence of points");
        return NULL;
    }
    if (width < 0)
    {
        PyErr_SetString (PyExc_ValueError, "width must not be negative");
        return NULL;
    }

    count = PySequence_Size (list);
    if (count < 2)
    {
        PyErr_SetString (PyExc_ValueError,
            "points must contain at least two points");
        return NULL;
    }

    /* Get all points */
    xpts = PyMem_New (int, (size_t) count);
    ypts = PyMem_New (int, (size_t) count);
    if (!xpts || !ypts)
    {
        if (xpts)
            PyMem_Free (xpts);
        if (ypts)
            PyMem_Free (ypts);
        return NULL;
    }

    left = right = bottom = top = 0;
    for (i = 0; i < count; i++)
    {
        item = PySequence_ITEM (list, i);
        if (!PointFromObject (item, &x, &y))
        {
            Py_XDECREF (item);
            PyMem_Free (xpts);
            PyMem_Free (ypts);
            PyErr_SetString (PyExc_ValueError, "invalid point list");
            return NULL;
        }

        left = MIN (x, left);
        right = MAX (x, right);
        xpts[i] = x;

        top = MIN (y, top);
        bottom = MAX (y, bottom);
        ypts[i] = y;

        Py_DECREF (item);
    }

    Py_BEGIN_ALLOW_THREADS;
    if (width == 0)
    {
        drawn = pyg_draw_filled_polygon (surface, &surface->clip_rect, color,
            xpts, ypts, (unsigned int) count, &area);
    }
    else
    {
        drawn = pyg_draw_polygon (surface, &surface->clip_rect, color, xpts,
            ypts, (unsigned int)count, width, &area);
    }
    Py_END_ALLOW_THREADS;

    PyMem_Free (xpts);
    PyMem_Free (ypts);

    if (!drawn)
        Py_RETURN_NONE;
    return PyRect_New (area.x, area.y, area.w, area.h);
}

static PyObject*
_draw_aapolygon (PyObject* self, PyObject* args)
{
    PyObject *surfobj, *colorobj, *list, *item;
    SDL_Surface* surface;
    SDL_Rect area;
    Uint32 color;
    int *xpts, *ypts, x, y;
    Py_ssize_t i, count;
    int left, right, bottom, top;
    int drawn = 0, blendargs = 0;
    
    if (!PyArg_ParseTuple (args, "OOO|i:aapolygon", &surfobj, &colorobj, &list,
            &blendargs))
        return NULL;
    if (!PySDLSurface_Check (surfobj))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    surface = ((PySDLSurface*)surfobj)->surface;
    
    if (!ColorFromObj (colorobj, surface->format, &color))
        return NULL;
    
    if (!PySequence_Check (list))
    {
        PyErr_SetString (PyExc_TypeError,
            "points must be a sequence of points");
        return NULL;
    }

    count = PySequence_Size (list);
    if (count < 2)
    {
        PyErr_SetString (PyExc_ValueError,
            "points must contain at least two points");
        return NULL;
    }

    /* Get all points */
    xpts = PyMem_New (int, (size_t) count);
    ypts = PyMem_New (int, (size_t) count);
    if (!xpts || !ypts)
    {
        if (xpts)
            PyMem_Free (xpts);
        if (ypts)
            PyMem_Free (ypts);
        return NULL;
    }

    left = right = bottom = top = 0;
    for (i = 0; i < count; i++)
    {
        item = PySequence_ITEM (list, i);
        if (!PointFromObject (item, &x, &y))
        {
            Py_XDECREF (item);
            PyMem_Free (xpts);
            PyMem_Free (ypts);
            PyErr_SetString (PyExc_ValueError, "invalid point list");
            return NULL;
        }

        left = MIN (x, left);
        right = MAX (x, right);
        xpts[i] = x;

        top = MIN (y, top);
        bottom = MAX (y, bottom);
        ypts[i] = y;

        Py_DECREF (item);
    }

    Py_BEGIN_ALLOW_THREADS;
    drawn = pyg_draw_aapolygon (surface, &surface->clip_rect, color, xpts, ypts,
        (unsigned int)count, blendargs, &area);
    Py_END_ALLOW_THREADS;

    PyMem_Free (xpts);
    PyMem_Free (ypts);

    if (!drawn)
        Py_RETURN_NONE;
    return PyRect_New (area.x, area.y, area.w, area.h);
}

static PyObject*
_draw_rect (PyObject* self, PyObject* args)
{
    PyObject *surfobj, *colorobj, *rectobj;
    SDL_Surface *surface;
    SDL_Rect rect;
    Uint32 color;
    int drawn, width = 0;
    int xpts[4], ypts[4];

    if (!PyArg_ParseTuple (args, "OOO|i:rect", &surfobj, &colorobj, &rectobj,
            &width))
        return NULL;
    if (!PySDLSurface_Check (surfobj))
    {
        PyErr_SetString (PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    surface = ((PySDLSurface*)surfobj)->surface;
    
    if (!ColorFromObj (colorobj, surface->format,  &color))
        return NULL;

    if (!SDLRect_FromRect (rectobj, &rect))
        return NULL;

    xpts[0] = rect.x; ypts[0] = rect.y;
    xpts[1] = rect.x + rect.w - 1; ypts[1] = rect.y;
    xpts[2] = rect.x + rect.w - 1; ypts[2] = rect.y + rect.h - 1;
    xpts[3] = rect.x; ypts[3] = rect.y + rect.h - 1;

    Py_BEGIN_ALLOW_THREADS;
    if (width == 0)
    {
        drawn = pyg_draw_filled_polygon (surface, &surface->clip_rect, color,
            xpts, ypts, 4, NULL);
    }
    else
    {
        drawn = pyg_draw_polygon (surface, &surface->clip_rect, color,
            xpts, ypts, 4, width, NULL);
    }
    Py_END_ALLOW_THREADS;

    if (!drawn)
        PyRect_New (rect.x, rect.y, 0, 0);
    return PyRect_New (rect.x, rect.y, rect.w, rect.h);
}

#ifdef IS_PYTHON_3
PyMODINIT_FUNC PyInit_draw (void)
#else
PyMODINIT_FUNC initdraw (void)
#endif
{
    PyObject *mod;

#ifdef IS_PYTHON_3
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "draw",
        DOC_DRAW,
        -1,
        _draw_methods,
        NULL, NULL, NULL, NULL
    };
    mod = PyModule_Create (&_module);
#else
    mod = Py_InitModule3 ("draw", _draw_methods, DOC_DRAW);
#endif
    if (!mod)
        goto fail;

    if (import_pygame2_base () < 0)
        goto fail;
    if (import_pygame2_sdl_base () < 0)
        goto fail;
    if (import_pygame2_sdl_video () < 0)
        goto fail;

    MODINIT_RETURN (mod);
fail:
    Py_XDECREF (mod);
    MODINIT_RETURN (NULL);
}
