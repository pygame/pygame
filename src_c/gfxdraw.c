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
/*
  This is a proposed SDL_gfx draw module for Pygame. It is backported
  from Pygame 2.

  TODO:
  - fix filledPolygonRGBA to use MT versions for threaded use.
  - do a filled pie version using filledPieColor
  - Determine if SDL video must be initiated for all routines to work.
    Add check if required, else remove ASSERT_VIDEO_INIT.
  - Example (Maybe).
*/
#define PYGAME_SDLGFXPRIM_INTERNAL

#include "pygame.h"

#include "doc/gfxdraw_doc.h"

#include "surface.h"

#include "pgcompat.h"

#include "SDL_gfx/SDL_gfxPrimitives.h"

static PyObject *
_gfx_pixelcolor(PyObject *self, PyObject *args);
static PyObject *
_gfx_hlinecolor(PyObject *self, PyObject *args);
static PyObject *
_gfx_vlinecolor(PyObject *self, PyObject *args);
static PyObject *
_gfx_rectanglecolor(PyObject *self, PyObject *args);
static PyObject *
_gfx_boxcolor(PyObject *self, PyObject *args);
static PyObject *
_gfx_linecolor(PyObject *self, PyObject *args);
static PyObject *
_gfx_circlecolor(PyObject *self, PyObject *args);
static PyObject *
_gfx_arccolor(PyObject *self, PyObject *args);
static PyObject *
_gfx_aacirclecolor(PyObject *self, PyObject *args);
static PyObject *
_gfx_filledcirclecolor(PyObject *self, PyObject *args);
static PyObject *
_gfx_ellipsecolor(PyObject *self, PyObject *args);
static PyObject *
_gfx_aaellipsecolor(PyObject *self, PyObject *args);
static PyObject *
_gfx_filledellipsecolor(PyObject *self, PyObject *args);
static PyObject *
_gfx_piecolor(PyObject *self, PyObject *args);
static PyObject *
_gfx_trigoncolor(PyObject *self, PyObject *args);
static PyObject *
_gfx_aatrigoncolor(PyObject *self, PyObject *args);
static PyObject *
_gfx_filledtrigoncolor(PyObject *self, PyObject *args);
static PyObject *
_gfx_polygoncolor(PyObject *self, PyObject *args);
static PyObject *
_gfx_aapolygoncolor(PyObject *self, PyObject *args);
static PyObject *
_gfx_filledpolygoncolor(PyObject *self, PyObject *args);
static PyObject *
_gfx_texturedpolygon(PyObject *self, PyObject *args);
static PyObject *
_gfx_beziercolor(PyObject *self, PyObject *args);

static PyMethodDef _gfxdraw_methods[] = {
    {"pixel", _gfx_pixelcolor, METH_VARARGS, DOC_PYGAMEGFXDRAWPIXEL},
    {"hline", _gfx_hlinecolor, METH_VARARGS, DOC_PYGAMEGFXDRAWHLINE},
    {"vline", _gfx_vlinecolor, METH_VARARGS, DOC_PYGAMEGFXDRAWVLINE},
    {"rectangle", _gfx_rectanglecolor, METH_VARARGS,
     DOC_PYGAMEGFXDRAWRECTANGLE},
    {"box", _gfx_boxcolor, METH_VARARGS, DOC_PYGAMEGFXDRAWRECTANGLE},
    {"line", _gfx_linecolor, METH_VARARGS, DOC_PYGAMEGFXDRAWLINE},
    {"circle", _gfx_circlecolor, METH_VARARGS, DOC_PYGAMEGFXDRAWCIRCLE},
    {"arc", _gfx_arccolor, METH_VARARGS, DOC_PYGAMEGFXDRAWARC},
    {"aacircle", _gfx_aacirclecolor, METH_VARARGS, DOC_PYGAMEGFXDRAWAACIRCLE},
    {"filled_circle", _gfx_filledcirclecolor, METH_VARARGS,
     DOC_PYGAMEGFXDRAWFILLEDCIRCLE},
    {"ellipse", _gfx_ellipsecolor, METH_VARARGS, DOC_PYGAMEGFXDRAWELLIPSE},
    {"aaellipse", _gfx_aaellipsecolor, METH_VARARGS,
     DOC_PYGAMEGFXDRAWAAELLIPSE},
    {"filled_ellipse", _gfx_filledellipsecolor, METH_VARARGS,
     DOC_PYGAMEGFXDRAWFILLEDELLIPSE},
    {"pie", _gfx_piecolor, METH_VARARGS, DOC_PYGAMEGFXDRAWPIE},
    {"trigon", _gfx_trigoncolor, METH_VARARGS, DOC_PYGAMEGFXDRAWTRIGON},
    {"aatrigon", _gfx_aatrigoncolor, METH_VARARGS, DOC_PYGAMEGFXDRAWAATRIGON},
    {"filled_trigon", _gfx_filledtrigoncolor, METH_VARARGS,
     DOC_PYGAMEGFXDRAWFILLEDTRIGON},
    {"polygon", _gfx_polygoncolor, METH_VARARGS, DOC_PYGAMEGFXDRAWPOLYGON},
    {"aapolygon", _gfx_aapolygoncolor, METH_VARARGS,
     DOC_PYGAMEGFXDRAWAAPOLYGON},
    {"filled_polygon", _gfx_filledpolygoncolor, METH_VARARGS,
     DOC_PYGAMEGFXDRAWFILLEDPOLYGON},
    {"textured_polygon", _gfx_texturedpolygon, METH_VARARGS,
     DOC_PYGAMEGFXDRAWTEXTUREDPOLYGON},
    {"bezier", _gfx_beziercolor, METH_VARARGS, DOC_PYGAMEGFXDRAWBEZIER},
    {NULL, NULL, 0, NULL},
};

#define ASSERT_VIDEO_INIT(unused) /* Is video really needed for gfxdraw? */

static int
Sint16FromObj(PyObject *item, Sint16 *val)
{
    if (PyNumber_Check(item)) {
        PyObject *intobj;
        long tmp;

        if (!(intobj = PyNumber_Long(item)))
            return 0;
        tmp = PyLong_AsLong(intobj);
        Py_DECREF(intobj);
        if (tmp == -1 && PyErr_Occurred())
            return 0;
        *val = (Sint16)tmp;
        return 1;
    }
    return 0;
}

static int
Sint16FromSeqIndex(PyObject *obj, Py_ssize_t _index, Sint16 *val)
{
    int result = 0;
    PyObject *item;
    item = PySequence_GetItem(obj, _index);
    if (item) {
        result = Sint16FromObj(item, val);
        Py_DECREF(item);
    }
    return result;
}

static PyObject *
_gfx_pixelcolor(PyObject *self, PyObject *args)
{
    PyObject *surface, *color;
    Sint16 x, y;
    Uint8 rgba[4];

    ASSERT_VIDEO_INIT(NULL);

    if (!PyArg_ParseTuple(args, "OhhO:pixel", &surface, &x, &y, &color))
        return NULL;

    if (!pgSurface_Check(surface)) {
        PyErr_SetString(PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!pg_RGBAFromObj(color, rgba)) {
        PyErr_SetString(PyExc_TypeError, "invalid color argument");
        return NULL;
    }

    if (pixelRGBA(pgSurface_AsSurface(surface), x, y, rgba[0], rgba[1],
                  rgba[2], rgba[3]) == -1) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
_gfx_hlinecolor(PyObject *self, PyObject *args)
{
    PyObject *surface, *color;
    Sint16 x1, x2, y;
    Uint8 rgba[4];

    ASSERT_VIDEO_INIT(NULL);

    if (!PyArg_ParseTuple(args, "OhhhO:hline", &surface, &x1, &x2, &y, &color))
        return NULL;

    if (!pgSurface_Check(surface)) {
        PyErr_SetString(PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!pg_RGBAFromObj(color, rgba)) {
        PyErr_SetString(PyExc_TypeError, "invalid color argument");
        return NULL;
    }

    if (hlineRGBA(pgSurface_AsSurface(surface), x1, x2, y, rgba[0], rgba[1],
                  rgba[2], rgba[3]) == -1) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
_gfx_vlinecolor(PyObject *self, PyObject *args)
{
    PyObject *surface, *color;
    Sint16 x, _y1, y2;
    Uint8 rgba[4];

    ASSERT_VIDEO_INIT(NULL);

    if (!PyArg_ParseTuple(args, "OhhhO:vline", &surface, &x, &_y1, &y2,
                          &color))
        return NULL;

    if (!pgSurface_Check(surface)) {
        PyErr_SetString(PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!pg_RGBAFromObj(color, rgba)) {
        PyErr_SetString(PyExc_TypeError, "invalid color argument");
        return NULL;
    }

    if (vlineRGBA(pgSurface_AsSurface(surface), x, _y1, y2, rgba[0], rgba[1],
                  rgba[2], rgba[3]) == -1) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
_gfx_rectanglecolor(PyObject *self, PyObject *args)
{
    PyObject *surface, *color, *rect;
    SDL_Rect temprect, *sdlrect;
    Sint16 x1, x2, _y1, y2;
    Uint8 rgba[4];

    ASSERT_VIDEO_INIT(NULL);

    if (!PyArg_ParseTuple(args, "OOO:rectangle", &surface, &rect, &color)) {
        /* Exception already set */
        return NULL;
    }

    if (!pgSurface_Check(surface)) {
        PyErr_SetString(PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    sdlrect = pgRect_FromObject(rect, &temprect);
    if (sdlrect == NULL) {
        PyErr_SetString(PyExc_TypeError, "invalid rect style argument");
        return NULL;
    }

    if (!pg_RGBAFromObj(color, rgba)) {
        PyErr_SetString(PyExc_TypeError, "invalid color argument");
        return NULL;
    }

    x1 = sdlrect->x;
    _y1 = sdlrect->y;
    x2 = (Sint16)(sdlrect->x + sdlrect->w - 1);
    y2 = (Sint16)(sdlrect->y + sdlrect->h - 1);

    if (rectangleRGBA(pgSurface_AsSurface(surface), x1, _y1, x2, y2, rgba[0],
                      rgba[1], rgba[2], rgba[3]) == -1) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
_gfx_boxcolor(PyObject *self, PyObject *args)
{
    PyObject *surface, *color, *rect;
    SDL_Rect temprect, *sdlrect;
    Sint16 x1, x2, _y1, y2;
    Uint8 rgba[4];

    ASSERT_VIDEO_INIT(NULL);

    if (!PyArg_ParseTuple(args, "OOO:box", &surface, &rect, &color)) {
        /* Exception already set */
        return NULL;
    }

    if (!pgSurface_Check(surface)) {
        PyErr_SetString(PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    sdlrect = pgRect_FromObject(rect, &temprect);
    if (sdlrect == NULL) {
        PyErr_SetString(PyExc_TypeError, "invalid rect style argument");
        return NULL;
    }
    if (!pg_RGBAFromObj(color, rgba)) {
        PyErr_SetString(PyExc_TypeError, "invalid color argument");
        return NULL;
    }

    x1 = sdlrect->x;
    _y1 = sdlrect->y;
    x2 = (Sint16)(sdlrect->x + sdlrect->w - 1);
    y2 = (Sint16)(sdlrect->y + sdlrect->h - 1);

    if (boxRGBA(pgSurface_AsSurface(surface), x1, _y1, x2, y2, rgba[0],
                rgba[1], rgba[2], rgba[3]) == -1) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
_gfx_linecolor(PyObject *self, PyObject *args)
{
    PyObject *surface, *color;
    Sint16 x1, x2, _y1, y2;
    Uint8 rgba[4];

    ASSERT_VIDEO_INIT(NULL);

    if (!PyArg_ParseTuple(args, "OhhhhO:line", &surface, &x1, &_y1, &x2, &y2,
                          &color))
        return NULL;

    if (!pgSurface_Check(surface)) {
        PyErr_SetString(PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!pg_RGBAFromObj(color, rgba)) {
        PyErr_SetString(PyExc_TypeError, "invalid color argument");
        return NULL;
    }

    if (lineRGBA(pgSurface_AsSurface(surface), x1, _y1, x2, y2, rgba[0],
                 rgba[1], rgba[2], rgba[3]) == -1) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
_gfx_circlecolor(PyObject *self, PyObject *args)
{
    PyObject *surface, *color;
    Sint16 x, y, r;
    Uint8 rgba[4];

    ASSERT_VIDEO_INIT(NULL);

    if (!PyArg_ParseTuple(args, "OhhhO:circle", &surface, &x, &y, &r, &color))
        return NULL;

    if (!pgSurface_Check(surface)) {
        PyErr_SetString(PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!pg_RGBAFromObj(color, rgba)) {
        PyErr_SetString(PyExc_TypeError, "invalid color argument");
        return NULL;
    }

    if (circleRGBA(pgSurface_AsSurface(surface), x, y, r, rgba[0], rgba[1],
                   rgba[2], rgba[3]) == -1) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
_gfx_arccolor(PyObject *self, PyObject *args)
{
    PyObject *surface, *color;
    Sint16 x, y, r, start, end;
    Uint8 rgba[4];

    ASSERT_VIDEO_INIT(NULL);

    if (!PyArg_ParseTuple(args, "OhhhhhO:arc", &surface, &x, &y, &r, &start,
                          &end, &color))
        return NULL;

    if (!pgSurface_Check(surface)) {
        PyErr_SetString(PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!pg_RGBAFromObj(color, rgba)) {
        PyErr_SetString(PyExc_TypeError, "invalid color argument");
        return NULL;
    }

    if (arcRGBA(pgSurface_AsSurface(surface), x, y, r, start, end, rgba[0],
                rgba[1], rgba[2], rgba[3]) == -1) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
_gfx_aacirclecolor(PyObject *self, PyObject *args)
{
    PyObject *surface, *color;
    Sint16 x, y, r;
    Uint8 rgba[4];

    ASSERT_VIDEO_INIT(NULL);

    if (!PyArg_ParseTuple(args, "OhhhO:aacircle", &surface, &x, &y, &r,
                          &color))
        return NULL;

    if (!pgSurface_Check(surface)) {
        PyErr_SetString(PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!pg_RGBAFromObj(color, rgba)) {
        PyErr_SetString(PyExc_TypeError, "invalid color argument");
        return NULL;
    }

    if (aacircleRGBA(pgSurface_AsSurface(surface), x, y, r, rgba[0], rgba[1],
                     rgba[2], rgba[3]) == -1) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
_gfx_filledcirclecolor(PyObject *self, PyObject *args)
{
    PyObject *surface, *color;
    Sint16 x, y, r;
    Uint8 rgba[4];

    ASSERT_VIDEO_INIT(NULL);

    if (!PyArg_ParseTuple(args, "OhhhO:filledcircle", &surface, &x, &y, &r,
                          &color))
        return NULL;

    if (!pgSurface_Check(surface)) {
        PyErr_SetString(PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!pg_RGBAFromObj(color, rgba)) {
        PyErr_SetString(PyExc_TypeError, "invalid color argument");
        return NULL;
    }

    if (filledCircleRGBA(pgSurface_AsSurface(surface), x, y, r, rgba[0],
                         rgba[1], rgba[2], rgba[3]) == -1) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
_gfx_ellipsecolor(PyObject *self, PyObject *args)
{
    PyObject *surface, *color;
    Sint16 x, y, rx, ry;
    Uint8 rgba[4];

    ASSERT_VIDEO_INIT(NULL);

    if (!PyArg_ParseTuple(args, "OhhhhO:ellipse", &surface, &x, &y, &rx, &ry,
                          &color))
        return NULL;

    if (!pgSurface_Check(surface)) {
        PyErr_SetString(PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!pg_RGBAFromObj(color, rgba)) {
        PyErr_SetString(PyExc_TypeError, "invalid color argument");
        return NULL;
    }

    if (ellipseRGBA(pgSurface_AsSurface(surface), x, y, rx, ry, rgba[0],
                    rgba[1], rgba[2], rgba[3]) == -1) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
_gfx_aaellipsecolor(PyObject *self, PyObject *args)
{
    PyObject *surface, *color;
    Sint16 x, y, rx, ry;
    Uint8 rgba[4];

    ASSERT_VIDEO_INIT(NULL);

    if (!PyArg_ParseTuple(args, "OhhhhO:aaellipse", &surface, &x, &y, &rx, &ry,
                          &color))
        return NULL;

    if (!pgSurface_Check(surface)) {
        PyErr_SetString(PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!pg_RGBAFromObj(color, rgba)) {
        PyErr_SetString(PyExc_TypeError, "invalid color argument");
        return NULL;
    }

    if (aaellipseRGBA(pgSurface_AsSurface(surface), x, y, rx, ry, rgba[0],
                      rgba[1], rgba[2], rgba[3]) == -1) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
_gfx_filledellipsecolor(PyObject *self, PyObject *args)
{
    PyObject *surface, *color;
    Sint16 x, y, rx, ry;
    Uint8 rgba[4];

    ASSERT_VIDEO_INIT(NULL);

    if (!PyArg_ParseTuple(args, "OhhhhO:filled_ellipse", &surface, &x, &y, &rx,
                          &ry, &color))
        return NULL;

    if (!pgSurface_Check(surface)) {
        PyErr_SetString(PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!pg_RGBAFromObj(color, rgba)) {
        PyErr_SetString(PyExc_TypeError, "invalid color argument");
        return NULL;
    }

    if (filledEllipseRGBA(pgSurface_AsSurface(surface), x, y, rx, ry, rgba[0],
                          rgba[1], rgba[2], rgba[3]) == -1) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
_gfx_piecolor(PyObject *self, PyObject *args)
{
    PyObject *surface, *color;
    Sint16 x, y, r, start, end;
    Uint8 rgba[4];

    ASSERT_VIDEO_INIT(NULL);

    if (!PyArg_ParseTuple(args, "OhhhhhO:pie", &surface, &x, &y, &r, &start,
                          &end, &color))
        return NULL;

    if (!pgSurface_Check(surface)) {
        PyErr_SetString(PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!pg_RGBAFromObj(color, rgba)) {
        PyErr_SetString(PyExc_TypeError, "invalid color argument");
        return NULL;
    }

    if (pieRGBA(pgSurface_AsSurface(surface), x, y, r, start, end, rgba[0],
                rgba[1], rgba[2], rgba[3]) == -1) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
_gfx_trigoncolor(PyObject *self, PyObject *args)
{
    PyObject *surface, *color;
    Sint16 x1, x2, x3, _y1, y2, y3;
    Uint8 rgba[4];

    ASSERT_VIDEO_INIT(NULL);

    if (!PyArg_ParseTuple(args, "OhhhhhhO:trigon", &surface, &x1, &_y1, &x2,
                          &y2, &x3, &y3, &color))
        return NULL;

    if (!pgSurface_Check(surface)) {
        PyErr_SetString(PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!pg_RGBAFromObj(color, rgba)) {
        PyErr_SetString(PyExc_TypeError, "invalid color argument");
        return NULL;
    }

    if (trigonRGBA(pgSurface_AsSurface(surface), x1, _y1, x2, y2, x3, y3,
                   rgba[0], rgba[1], rgba[2], rgba[3]) == -1) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
_gfx_aatrigoncolor(PyObject *self, PyObject *args)
{
    PyObject *surface, *color;
    Sint16 x1, x2, x3, _y1, y2, y3;
    Uint8 rgba[4];

    ASSERT_VIDEO_INIT(NULL);

    if (!PyArg_ParseTuple(args, "OhhhhhhO:aatrigon", &surface, &x1, &_y1, &x2,
                          &y2, &x3, &y3, &color))
        return NULL;

    if (!pgSurface_Check(surface)) {
        PyErr_SetString(PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!pg_RGBAFromObj(color, rgba)) {
        PyErr_SetString(PyExc_TypeError, "invalid color argument");
        return NULL;
    }

    if (aatrigonRGBA(pgSurface_AsSurface(surface), x1, _y1, x2, y2, x3, y3,
                     rgba[0], rgba[1], rgba[2], rgba[3]) == -1) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
_gfx_filledtrigoncolor(PyObject *self, PyObject *args)
{
    PyObject *surface, *color;
    Sint16 x1, x2, x3, _y1, y2, y3;
    Uint8 rgba[4];

    ASSERT_VIDEO_INIT(NULL);

    if (!PyArg_ParseTuple(args, "OhhhhhhO:filled_trigon", &surface, &x1, &_y1,
                          &x2, &y2, &x3, &y3, &color))
        return NULL;

    if (!pgSurface_Check(surface)) {
        PyErr_SetString(PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!pg_RGBAFromObj(color, rgba)) {
        PyErr_SetString(PyExc_TypeError, "invalid color argument");
        return NULL;
    }

    if (filledTrigonRGBA(pgSurface_AsSurface(surface), x1, _y1, x2, y2, x3, y3,
                         rgba[0], rgba[1], rgba[2], rgba[3]) == -1) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
_gfx_polygoncolor(PyObject *self, PyObject *args)
{
    PyObject *surface, *color, *points, *item;
    Sint16 *vx, *vy, x, y;
    Py_ssize_t count, i;
    int ret;
    Uint8 rgba[4];

    ASSERT_VIDEO_INIT(NULL);

    if (!PyArg_ParseTuple(args, "OOO:polygon", &surface, &points, &color))
        return NULL;

    if (!pgSurface_Check(surface)) {
        PyErr_SetString(PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!pg_RGBAFromObj(color, rgba)) {
        PyErr_SetString(PyExc_TypeError, "invalid color argument");
        return NULL;
    }
    if (!PySequence_Check(points)) {
        PyErr_SetString(PyExc_TypeError, "points must be a sequence");
        return NULL;
    }

    count = PySequence_Size(points);
    if (count < 3) {
        PyErr_SetString(PyExc_ValueError,
                        "points must contain more than 2 points");
        return NULL;
    }

    vx = PyMem_New(Sint16, (size_t)count);
    vy = PyMem_New(Sint16, (size_t)count);
    if (!vx || !vy) {
        if (vx)
            PyMem_Free(vx);
        if (vy)
            PyMem_Free(vy);
        return NULL;
    }

    for (i = 0; i < count; i++) {
        item = PySequence_ITEM(points, i);
        if (!Sint16FromSeqIndex(item, 0, &x)) {
            PyMem_Free(vx);
            PyMem_Free(vy);
            Py_XDECREF(item);
            return NULL;
        }
        if (!Sint16FromSeqIndex(item, 1, &y)) {
            PyMem_Free(vx);
            PyMem_Free(vy);
            Py_XDECREF(item);
            return NULL;
        }
        Py_DECREF(item);
        vx[i] = x;
        vy[i] = y;
    }

    Py_BEGIN_ALLOW_THREADS;
    ret = polygonRGBA(pgSurface_AsSurface(surface), vx, vy, (int)count,
                      rgba[0], rgba[1], rgba[2], rgba[3]);
    Py_END_ALLOW_THREADS;

    PyMem_Free(vx);
    PyMem_Free(vy);

    if (ret == -1) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
_gfx_aapolygoncolor(PyObject *self, PyObject *args)
{
    PyObject *surface, *color, *points, *item;
    Sint16 *vx, *vy, x, y;
    Py_ssize_t count, i;
    int ret;
    Uint8 rgba[4];

    ASSERT_VIDEO_INIT(NULL);

    if (!PyArg_ParseTuple(args, "OOO:aapolygon", &surface, &points, &color))
        return NULL;

    if (!pgSurface_Check(surface)) {
        PyErr_SetString(PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!pg_RGBAFromObj(color, rgba)) {
        PyErr_SetString(PyExc_TypeError, "invalid color argument");
        return NULL;
    }
    if (!PySequence_Check(points)) {
        PyErr_SetString(PyExc_TypeError, "points must be a sequence");
        return NULL;
    }

    count = PySequence_Size(points);
    if (count < 3) {
        PyErr_SetString(PyExc_ValueError,
                        "points must contain more than 2 points");
        return NULL;
    }

    vx = PyMem_New(Sint16, (size_t)count);
    vy = PyMem_New(Sint16, (size_t)count);
    if (!vx || !vy) {
        if (vx)
            PyMem_Free(vx);
        if (vy)
            PyMem_Free(vy);
        return NULL;
    }

    for (i = 0; i < count; i++) {
        item = PySequence_ITEM(points, i);
        if (!Sint16FromSeqIndex(item, 0, &x)) {
            PyMem_Free(vx);
            PyMem_Free(vy);
            Py_XDECREF(item);
            return NULL;
        }
        if (!Sint16FromSeqIndex(item, 1, &y)) {
            PyMem_Free(vx);
            PyMem_Free(vy);
            Py_XDECREF(item);
            return NULL;
        }
        Py_DECREF(item);
        vx[i] = x;
        vy[i] = y;
    }

    Py_BEGIN_ALLOW_THREADS;
    ret = aapolygonRGBA(pgSurface_AsSurface(surface), vx, vy, (int)count,
                        rgba[0], rgba[1], rgba[2], rgba[3]);
    Py_END_ALLOW_THREADS;

    PyMem_Free(vx);
    PyMem_Free(vy);

    if (ret == -1) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
_gfx_filledpolygoncolor(PyObject *self, PyObject *args)
{
    PyObject *surface, *color, *points, *item;
    Sint16 *vx, *vy, x, y;
    Py_ssize_t count, i;
    int ret;
    Uint8 rgba[4];

    ASSERT_VIDEO_INIT(NULL);

    if (!PyArg_ParseTuple(args, "OOO:filled_polygon", &surface, &points,
                          &color))
        return NULL;

    if (!pgSurface_Check(surface)) {
        PyErr_SetString(PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!pg_RGBAFromObj(color, rgba)) {
        PyErr_SetString(PyExc_TypeError, "invalid color argument");
        return NULL;
    }
    if (!PySequence_Check(points)) {
        PyErr_SetString(PyExc_TypeError, "points must be a sequence");
        return NULL;
    }

    count = PySequence_Size(points);
    if (count < 3) {
        PyErr_SetString(PyExc_ValueError,
                        "points must contain more than 2 points");
        return NULL;
    }

    vx = PyMem_New(Sint16, (size_t)count);
    vy = PyMem_New(Sint16, (size_t)count);
    if (!vx || !vy) {
        if (vx)
            PyMem_Free(vx);
        if (vy)
            PyMem_Free(vy);
        return NULL;
    }

    for (i = 0; i < count; i++) {
        item = PySequence_ITEM(points, i);
        if (!Sint16FromSeqIndex(item, 0, &x)) {
            PyMem_Free(vx);
            PyMem_Free(vy);
            Py_XDECREF(item);
            return NULL;
        }
        if (!Sint16FromSeqIndex(item, 1, &y)) {
            PyMem_Free(vx);
            PyMem_Free(vy);
            Py_XDECREF(item);
            return NULL;
        }
        Py_DECREF(item);
        vx[i] = x;
        vy[i] = y;
    }

    Py_BEGIN_ALLOW_THREADS;
    ret = filledPolygonRGBA(pgSurface_AsSurface(surface), vx, vy, (int)count,
                            rgba[0], rgba[1], rgba[2], rgba[3]);
    Py_END_ALLOW_THREADS;

    PyMem_Free(vx);
    PyMem_Free(vy);

    if (ret == -1) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
_gfx_texturedpolygon(PyObject *self, PyObject *args)
{
    PyObject *surface, *texture, *points, *item;
    SDL_Surface *s_surface, *s_texture;
    Sint16 *vx, *vy, x, y, tdx, tdy;
    Py_ssize_t count, i;
    int ret;

    ASSERT_VIDEO_INIT(NULL);

    if (!PyArg_ParseTuple(args, "OOOhh:textured_polygon", &surface, &points,
                          &texture, &tdx, &tdy))
        return NULL;

    if (!pgSurface_Check(surface)) {
        PyErr_SetString(PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    s_surface = pgSurface_AsSurface(surface);
    if (!pgSurface_Check(texture)) {
        PyErr_SetString(PyExc_TypeError, "texture must be a Surface");
        return NULL;
    }
    s_texture = pgSurface_AsSurface(texture);
    if (!PySequence_Check(points)) {
        PyErr_SetString(PyExc_TypeError, "points must be a sequence");
        return NULL;
    }
    if (s_surface->format->BytesPerPixel == 1 &&
        (s_texture->format->Amask || s_texture->flags & SDL_SRCALPHA)) {
        PyErr_SetString(PyExc_ValueError,
                        "Per-byte alpha texture unsupported "
                        "for 8 bit surfaces");
        return NULL;
    }

    count = PySequence_Size(points);
    if (count < 3) {
        PyErr_SetString(PyExc_ValueError,
                        "points must contain more than 2 points");
        return NULL;
    }

    vx = PyMem_New(Sint16, (size_t)count);
    vy = PyMem_New(Sint16, (size_t)count);
    if (!vx || !vy) {
        if (vx)
            PyMem_Free(vx);
        if (vy)
            PyMem_Free(vy);
        return NULL;
    }

    for (i = 0; i < count; i++) {
        item = PySequence_ITEM(points, i);
        if (!Sint16FromSeqIndex(item, 0, &x)) {
            PyMem_Free(vx);
            PyMem_Free(vy);
            Py_XDECREF(item);
            return NULL;
        }
        if (!Sint16FromSeqIndex(item, 1, &y)) {
            PyMem_Free(vx);
            PyMem_Free(vy);
            Py_XDECREF(item);
            return NULL;
        }
        Py_DECREF(item);
        vx[i] = x;
        vy[i] = y;
    }

    Py_BEGIN_ALLOW_THREADS;
    ret = texturedPolygon(s_surface, vx, vy, (int)count, s_texture, tdx, tdy);
    Py_END_ALLOW_THREADS;

    PyMem_Free(vx);
    PyMem_Free(vy);

    if (ret == -1) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
_gfx_beziercolor(PyObject *self, PyObject *args)
{
    PyObject *surface, *color, *points, *item;
    Sint16 *vx, *vy, x, y;
    Py_ssize_t count, i;
    int ret, steps;
    Uint8 rgba[4];

    ASSERT_VIDEO_INIT(NULL);

    if (!PyArg_ParseTuple(args, "OOiO:bezier", &surface, &points, &steps,
                          &color))
        return NULL;

    if (!pgSurface_Check(surface)) {
        PyErr_SetString(PyExc_TypeError, "surface must be a Surface");
        return NULL;
    }
    if (!pg_RGBAFromObj(color, rgba)) {
        PyErr_SetString(PyExc_TypeError, "invalid color argument");
        return NULL;
    }
    if (!PySequence_Check(points)) {
        PyErr_SetString(PyExc_TypeError, "points must be a sequence");
        return NULL;
    }

    count = PySequence_Size(points);
    if (count < 3) {
        PyErr_SetString(PyExc_ValueError,
                        "points must contain more than 2 points");
        return NULL;
    }

    if (steps < 2) {
        PyErr_SetString(PyExc_ValueError,
                        "steps parameter must be greater than 1");
        return NULL;
    }

    vx = PyMem_New(Sint16, (size_t)count);
    vy = PyMem_New(Sint16, (size_t)count);
    if (!vx || !vy) {
        PyErr_SetString(PyExc_MemoryError, "memory allocation failed");
        if (vx)
            PyMem_Free(vx);
        if (vy)
            PyMem_Free(vy);
        return NULL;
    }

    for (i = 0; i < count; i++) {
        item = PySequence_ITEM(points, i);
        if (!Sint16FromSeqIndex(item, 0, &x)) {
            PyMem_Free(vx);
            PyMem_Free(vy);
            Py_XDECREF(item);
            return NULL;
        }
        if (!Sint16FromSeqIndex(item, 1, &y)) {
            PyMem_Free(vx);
            PyMem_Free(vy);
            Py_XDECREF(item);
            return NULL;
        }
        Py_DECREF(item);
        vx[i] = x;
        vy[i] = y;
    }

    Py_BEGIN_ALLOW_THREADS;
    ret = bezierRGBA(pgSurface_AsSurface(surface), vx, vy, (int)count, steps,
                     rgba[0], rgba[1], rgba[2], rgba[3]);
    Py_END_ALLOW_THREADS;

    PyMem_Free(vx);
    PyMem_Free(vy);

    if (ret == -1) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return NULL;
    }
    Py_RETURN_NONE;
}

MODINIT_DEFINE(gfxdraw)
{
    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "gfxdraw",
                                         DOC_PYGAMEGFXDRAW,
                                         -1,
                                         _gfxdraw_methods,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL};

    /* import needed APIs; Do this first so if there is an error
       the module is not loaded.
    */
    import_pygame_base();
    if (PyErr_Occurred()) {
        return NULL;
    }
    import_pygame_color();
    if (PyErr_Occurred()) {
        return NULL;
    }
    import_pygame_rect();
    if (PyErr_Occurred()) {
        return NULL;
    }
    import_pygame_surface();
    if (PyErr_Occurred()) {
        return NULL;
    }

    return PyModule_Create(&_module);
}
