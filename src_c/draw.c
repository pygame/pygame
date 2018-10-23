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

    Pete Shinners
    pete@shinners.org
*/

/*
 *  drawing module for pygame
 */
#include "pygame.h"

#include "pgcompat.h"

#include "doc/draw_doc.h"

#include <math.h>

/* Many C libraries seem to lack the trunc call (added in C99) */
#define trunc(d) (((d) >= 0.0) ? (floor(d)) : (ceil(d)))
#define FRAC(z) ((z)-trunc(z))
#define INVFRAC(z) (1 - FRAC(z))

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int
clip_and_draw_line(SDL_Surface *surf, SDL_Rect *rect, Uint32 color, int *pts);
static int
clip_and_draw_aaline(SDL_Surface *surf, SDL_Rect *rect, Uint32 color,
                     float *pts, int blend);
static int
clip_and_draw_line_width(SDL_Surface *surf, SDL_Rect *rect, Uint32 color,
                         int width, int *pts);
static int
clipline(int *pts, int left, int top, int right, int bottom);
static int
clipaaline(float *pts, int left, int top, int right, int bottom);
static void
drawline(SDL_Surface *surf, Uint32 color, int startx, int starty, int endx,
         int endy);
static void
drawaaline(SDL_Surface *surf, Uint32 color, float startx, float starty,
           float endx, float endy, int blend);
static void
drawhorzline(SDL_Surface *surf, Uint32 color, int startx, int starty,
             int endx);
static void
drawvertline(SDL_Surface *surf, Uint32 color, int x1, int y1, int y2);
static void
draw_arc(SDL_Surface *dst, int x, int y, int radius1, int radius2,
         double angle_start, double angle_stop, Uint32 color);
static void
draw_ellipse(SDL_Surface *dst, int x, int y, int width, int height, int solid,
             Uint32 color);
static void
draw_fillpoly(SDL_Surface *dst, int *vx, int *vy, int n, Uint32 color);

static PyObject *
aaline(PyObject *self, PyObject *arg)
{
    PyObject *surfobj, *colorobj, *start, *end;
    SDL_Surface *surf;
    float startx, starty, endx, endy;
    int top, left, bottom, right;
    int blend = 1;
    float pts[4];
    Uint8 rgba[4];
    Uint32 color;
    int anydraw;

    /*get all the arguments*/
    if (!PyArg_ParseTuple(arg, "O!OOO|i", &pgSurface_Type, &surfobj, &colorobj,
                          &start, &end, &blend))
        return NULL;
    surf = pgSurface_AsSurface(surfobj);

    if (surf->format->BytesPerPixel != 3 && surf->format->BytesPerPixel != 4)
        return RAISE(
            PyExc_ValueError,
            "unsupported bit depth for aaline draw (supports 32 & 24 bit)");

    if (pg_RGBAFromColorObj(colorobj, rgba))
        color = SDL_MapRGBA(surf->format, rgba[0], rgba[1], rgba[2], rgba[3]);
    else
        return RAISE(PyExc_TypeError, "invalid color argument");

    if (!pg_TwoFloatsFromObj(start, &startx, &starty))
        return RAISE(PyExc_TypeError, "Invalid start position argument");
    if (!pg_TwoFloatsFromObj(end, &endx, &endy))
        return RAISE(PyExc_TypeError, "Invalid end position argument");

    if (!pgSurface_Lock(surfobj))
        return NULL;

    pts[0] = startx;
    pts[1] = starty;
    pts[2] = endx;
    pts[3] = endy;
    anydraw = clip_and_draw_aaline(surf, &surf->clip_rect, color, pts, blend);

    if (!pgSurface_Unlock(surfobj))
        return NULL;

    /*compute return rect*/
    if (!anydraw)
        return pgRect_New4(startx, starty, 0, 0);
    if (pts[0] < pts[2]) {
        left = (int)(pts[0]);
        right = (int)(pts[2]);
    }
    else {
        left = (int)(pts[2]);
        right = (int)(pts[0]);
    }
    if (pts[1] < pts[3]) {
        top = (int)(pts[1]);
        bottom = (int)(pts[3]);
    }
    else {
        top = (int)(pts[3]);
        bottom = (int)(pts[1]);
    }
    return pgRect_New4(left, top, right - left + 2, bottom - top + 2);
}

static PyObject *
line(PyObject *self, PyObject *arg)
{
    PyObject *surfobj, *colorobj, *start, *end;
    SDL_Surface *surf;
    int startx, starty, endx, endy;
    int dx, dy;
    int rtop, rleft, rwidth, rheight;
    int width = 1;
    int pts[4];
    Uint8 rgba[4];
    Uint32 color;
    int anydraw;

    /*get all the arguments*/
    if (!PyArg_ParseTuple(arg, "O!OOO|i", &pgSurface_Type, &surfobj, &colorobj,
                          &start, &end, &width))
        return NULL;
    surf = pgSurface_AsSurface(surfobj);

    if (surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4)
        return RAISE(PyExc_ValueError, "unsupport bit depth for line draw");

    if (PyInt_Check(colorobj))
        color = (Uint32)PyInt_AsLong(colorobj);
    else if (pg_RGBAFromColorObj(colorobj, rgba))
        color = SDL_MapRGBA(surf->format, rgba[0], rgba[1], rgba[2], rgba[3]);
    else
        return RAISE(PyExc_TypeError, "invalid color argument");

    if (!pg_TwoIntsFromObj(start, &startx, &starty))
        return RAISE(PyExc_TypeError, "Invalid start position argument");
    if (!pg_TwoIntsFromObj(end, &endx, &endy))
        return RAISE(PyExc_TypeError, "Invalid end position argument");

    if (width < 1)
        return pgRect_New4(startx, starty, 0, 0);

    if (!pgSurface_Lock(surfobj))
        return NULL;

    pts[0] = startx;
    pts[1] = starty;
    pts[2] = endx;
    pts[3] = endy;
    anydraw =
        clip_and_draw_line_width(surf, &surf->clip_rect, color, width, pts);

    if (!pgSurface_Unlock(surfobj))
        return NULL;

    /*compute return rect*/
    if (!anydraw)
        return pgRect_New4(startx, starty, 0, 0);
    rleft = MIN(startx, endx);
    rtop = MIN(starty, endy);
    dx = abs(startx - endx);
    dy = abs(starty - endy);
    if (dx > dy) {
        rwidth = dx + 1;
        rheight = dy + width;
    }
    else {
        rwidth = dx + width;
        rheight = dy + 1;
    }
    return pgRect_New4(rleft, rtop, rwidth, rheight);
}

static PyObject *
aalines(PyObject *self, PyObject *arg)
{
    PyObject *surfobj, *colorobj, *closedobj, *points, *item;
    SDL_Surface *surf;
    float x, y;
    int top, left, bottom, right;
    float pts[4];
    Uint8 rgba[4];
    Uint32 color;
    int closed, blend;
    int result, loop, length, drawn;
    float startx, starty;

    /*get all the arguments*/
    if (!PyArg_ParseTuple(arg, "O!OOO|i", &pgSurface_Type, &surfobj, &colorobj,
                          &closedobj, &points, &blend))
        return NULL;
    surf = pgSurface_AsSurface(surfobj);

    if (surf->format->BytesPerPixel != 3 && surf->format->BytesPerPixel != 4)
        return RAISE(
            PyExc_ValueError,
            "unsupported bit depth for aaline draw (supports 32 & 24 bit)");

    if (pg_RGBAFromColorObj(colorobj, rgba))
        color = SDL_MapRGBA(surf->format, rgba[0], rgba[1], rgba[2], rgba[3]);
    else
        return RAISE(PyExc_TypeError, "invalid color argument");

    closed = PyObject_IsTrue(closedobj);

    if (!PySequence_Check(points))
        return RAISE(PyExc_TypeError,
                     "points argument must be a sequence of number pairs");
    length = PySequence_Length(points);
    if (length < 2)
        return RAISE(PyExc_ValueError,
                     "points argument must contain more than 1 points");

    for(loop = 0; loop < length; ++loop) {
        item = PySequence_GetItem(points, loop);
        result = pg_TwoIntsFromObj(item, &x, &y);
        Py_DECREF(item);
        if(!result)
            return RAISE(PyExc_TypeError, "all points must be number pairs");
    }

    item = PySequence_GetItem(points, 0);
    result = pg_TwoFloatsFromObj(item, &x, &y);
    Py_DECREF(item);

    startx = pts[0] = x;
    starty = pts[1] = y;
    left = right = (int)x;
    top = bottom = (int)y;

    if (!pgSurface_Lock(surfobj))
        return NULL;

    drawn = 1;
    for (loop = 1; loop < length; ++loop) {
        item = PySequence_GetItem(points, loop);
        result = pg_TwoFloatsFromObj(item, &x, &y);
        Py_DECREF(item);
        ++drawn;
        pts[0] = startx;
        pts[1] = starty;
        startx = pts[2] = x;
        starty = pts[3] = y;
        if (clip_and_draw_aaline(surf, &surf->clip_rect, color, pts, blend)) {
            left = MIN((int)MIN(pts[0], pts[2]), left);
            top = MIN((int)MIN(pts[1], pts[3]), top);
            right = MAX((int)MAX(pts[0], pts[2]), right);
            bottom = MAX((int)MAX(pts[1], pts[3]), bottom);
        }
    }
    if (closed && drawn > 2) {
        item = PySequence_GetItem(points, 0);
        result = pg_TwoFloatsFromObj(item, &x, &y);
        Py_DECREF(item);
        if (result) {
            pts[0] = startx;
            pts[1] = starty;
            pts[2] = x;
            pts[3] = y;
            clip_and_draw_aaline(surf, &surf->clip_rect, color, pts, blend);
        }
    }

    if (!pgSurface_Unlock(surfobj))
        return NULL;

    /*compute return rect*/
    return pgRect_New4(left, top, right - left + 2, bottom - top + 2);
}

static PyObject *
lines(PyObject *self, PyObject *arg)
{
    PyObject *surfobj, *colorobj, *closedobj, *points, *item;
    SDL_Surface *surf;
    int x, y;
    int top, left, bottom, right;
    int pts[4], width = 1;
    Uint8 rgba[4];
    Uint32 color;
    int closed;
    int result, loop, length, drawn;
    int startx, starty;

    /*get all the arguments*/
    if (!PyArg_ParseTuple(arg, "O!OOO|i", &pgSurface_Type, &surfobj, &colorobj,
                          &closedobj, &points, &width))
        return NULL;
    surf = pgSurface_AsSurface(surfobj);

    if (surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4)
        return RAISE(PyExc_ValueError, "unsupport bit depth for line draw");

    if (PyInt_Check(colorobj))
        color = (Uint32)PyInt_AsLong(colorobj);
    else if (pg_RGBAFromColorObj(colorobj, rgba))
        color = SDL_MapRGBA(surf->format, rgba[0], rgba[1], rgba[2], rgba[3]);
    else
        return RAISE(PyExc_TypeError, "invalid color argument");

    closed = PyObject_IsTrue(closedobj);

    if (!PySequence_Check(points))
        return RAISE(PyExc_TypeError,
                     "points argument must be a sequence of number pairs");
    length = PySequence_Length(points);
    if (length < 2)
        return RAISE(PyExc_ValueError,
                     "points argument must contain more than 1 points");

    for(loop = 0; loop < length; ++loop) {
        item = PySequence_GetItem(points, loop);
        result = pg_TwoIntsFromObj(item, &x, &y);
        Py_DECREF(item);
        if(!result)
            return RAISE(PyExc_TypeError, "all points must be number pairs");
    }

    item = PySequence_GetItem(points, 0);
    result = pg_TwoIntsFromObj(item, &x, &y);
    Py_DECREF(item);

    startx = pts[0] = left = right = x;
    starty = pts[1] = top = bottom = y;

    if (width < 1)
        return pgRect_New4(left, top, 0, 0);

    if (!pgSurface_Lock(surfobj))
        return NULL;

    drawn = 1;
    for (loop = 1; loop < length; ++loop) {
        item = PySequence_GetItem(points, loop);
        result = pg_TwoIntsFromObj(item, &x, &y);
        Py_DECREF(item);
        ++drawn;
        pts[0] = startx;
        pts[1] = starty;
        startx = pts[2] = x;
        starty = pts[3] = y;
        if (clip_and_draw_line_width(surf, &surf->clip_rect, color, width,
                                     pts)) {
            left = MIN(MIN(pts[0], pts[2]), left);
            top = MIN(MIN(pts[1], pts[3]), top);
            right = MAX(MAX(pts[0], pts[2]), right);
            bottom = MAX(MAX(pts[1], pts[3]), bottom);
        }
    }
    if (closed && drawn > 2) {
        item = PySequence_GetItem(points, 0);
        result = pg_TwoIntsFromObj(item, &x, &y);
        Py_DECREF(item);
        if (result) {
            pts[0] = startx;
            pts[1] = starty;
            pts[2] = x;
            pts[3] = y;
            clip_and_draw_line_width(surf, &surf->clip_rect, color, width,
                                     pts);
        }
    }

    if (!pgSurface_Unlock(surfobj))
        return NULL;

    /*compute return rect*/
    return pgRect_New4(left, top, right - left + 1, bottom - top + 1);
}

static PyObject *
arc(PyObject *self, PyObject *arg)
{
    PyObject *surfobj, *colorobj, *rectobj;
    GAME_Rect *rect, temp;
    SDL_Surface *surf;
    Uint8 rgba[4];
    Uint32 color;
    int width = 1, loop, t, l, b, r;
    double angle_start, angle_stop;

    /*get all the arguments*/
    if (!PyArg_ParseTuple(arg, "O!OOdd|i", &pgSurface_Type, &surfobj,
                          &colorobj, &rectobj, &angle_start, &angle_stop,
                          &width))
        return NULL;
    rect = pgRect_FromObject(rectobj, &temp);
    if (!rect)
        return RAISE(PyExc_TypeError, "Invalid recstyle argument");

    surf = pgSurface_AsSurface(surfobj);
    if (surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4)
        return RAISE(PyExc_ValueError, "unsupport bit depth for drawing");

    if (PyInt_Check(colorobj))
        color = (Uint32)PyInt_AsLong(colorobj);
    else if (pg_RGBAFromColorObj(colorobj, rgba))
        color = SDL_MapRGBA(surf->format, rgba[0], rgba[1], rgba[2], rgba[3]);
    else
        return RAISE(PyExc_TypeError, "invalid color argument");

    if (width < 0)
        return RAISE(PyExc_ValueError, "negative width");
    if (width > rect->w / 2 || width > rect->h / 2)
        return RAISE(PyExc_ValueError, "width greater than ellipse radius");
    if (angle_stop < angle_start)
        // Angle is in radians
        angle_stop += 2 * M_PI;

    if (!pgSurface_Lock(surfobj))
        return NULL;

    width = MIN(width, MIN(rect->w, rect->h) / 2);
    for (loop = 0; loop < width; ++loop) {
        draw_arc(surf, rect->x + rect->w / 2, rect->y + rect->h / 2,
                 rect->w / 2 - loop, rect->h / 2 - loop, angle_start,
                 angle_stop, color);
    }

    if (!pgSurface_Unlock(surfobj))
        return NULL;

    l = MAX(rect->x, surf->clip_rect.x);
    t = MAX(rect->y, surf->clip_rect.y);
    r = MIN(rect->x + rect->w, surf->clip_rect.x + surf->clip_rect.w);
    b = MIN(rect->y + rect->h, surf->clip_rect.y + surf->clip_rect.h);
    return pgRect_New4(l, t, MAX(r - l, 0), MAX(b - t, 0));
}

static PyObject *
ellipse(PyObject *self, PyObject *arg)
{
    PyObject *surfobj, *colorobj, *rectobj;
    GAME_Rect *rect, temp;
    SDL_Surface *surf;
    Uint8 rgba[4];
    Uint32 color;
    int width = 0, loop, t, l, b, r;

    /*get all the arguments*/
    if (!PyArg_ParseTuple(arg, "O!OO|i", &pgSurface_Type, &surfobj, &colorobj,
                          &rectobj, &width))
        return NULL;
    rect = pgRect_FromObject(rectobj, &temp);
    if (!rect)
        return RAISE(PyExc_TypeError, "Invalid recstyle argument");

    surf = pgSurface_AsSurface(surfobj);
    if (surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4)
        return RAISE(PyExc_ValueError, "unsupport bit depth for drawing");

    if (PyInt_Check(colorobj))
        color = (Uint32)PyInt_AsLong(colorobj);
    else if (pg_RGBAFromColorObj(colorobj, rgba))
        color = SDL_MapRGBA(surf->format, rgba[0], rgba[1], rgba[2], rgba[3]);
    else
        return RAISE(PyExc_TypeError, "invalid color argument");

    if (width < 0)
        return RAISE(PyExc_ValueError, "negative width");
    if (width > rect->w / 2 || width > rect->h / 2)
        return RAISE(PyExc_ValueError, "width greater than ellipse radius");

    if (!pgSurface_Lock(surfobj))
        return NULL;

    if (!width) {
        draw_ellipse(surf, (Sint16)(rect->x + rect->w / 2),
                     (Sint16)(rect->y + rect->h / 2), (Sint16)(rect->w),
                     (Sint16)(rect->h), 1, color);
    }
    else {
        width = MIN(width, MIN(rect->w, rect->h) / 2);
        for (loop = 0; loop < width; ++loop) {
            draw_ellipse(surf, rect->x + rect->w / 2, rect->y + rect->h / 2,
                         rect->w - loop, rect->h - loop, 0, color);
        }
    }

    if (!pgSurface_Unlock(surfobj))
        return NULL;

    l = MAX(rect->x, surf->clip_rect.x);
    t = MAX(rect->y, surf->clip_rect.y);
    r = MIN(rect->x + rect->w, surf->clip_rect.x + surf->clip_rect.w);
    b = MIN(rect->y + rect->h, surf->clip_rect.y + surf->clip_rect.h);
    return pgRect_New4(l, t, MAX(r - l, 0), MAX(b - t, 0));
}

static PyObject *
circle(PyObject *self, PyObject *arg)
{
    PyObject *surfobj, *colorobj;
    SDL_Surface *surf;
    Uint8 rgba[4];
    Uint32 color;
    int posx, posy, radius, t, l, b, r;
    int width = 0, loop;

    /*get all the arguments*/
    if (!PyArg_ParseTuple(arg, "O!O(ii)i|i", &pgSurface_Type, &surfobj,
                          &colorobj, &posx, &posy, &radius, &width))
        return NULL;

    surf = pgSurface_AsSurface(surfobj);
    if (surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4)
        return RAISE(PyExc_ValueError, "unsupport bit depth for drawing");

    if (PyInt_Check(colorobj))
        color = (Uint32)PyInt_AsLong(colorobj);
    else if (pg_RGBAFromColorObj(colorobj, rgba))
        color = SDL_MapRGBA(surf->format, rgba[0], rgba[1], rgba[2], rgba[3]);
    else
        return RAISE(PyExc_TypeError, "invalid color argument");

    if (radius < 0)
        return RAISE(PyExc_ValueError, "negative radius");
    if (width < 0)
        return RAISE(PyExc_ValueError, "negative width");
    if (width > radius)
        return RAISE(PyExc_ValueError, "width greater than radius");

    if (!pgSurface_Lock(surfobj))
        return NULL;

    if (!width) {
        draw_ellipse(surf, (Sint16)posx, (Sint16)posy, (Sint16)radius * 2,
                     (Sint16)radius * 2, 1, color);
    }
    else {
        for (loop = 0; loop < width; ++loop) {
            draw_ellipse(surf, posx, posy, 2 * (radius - loop),
                         2 * (radius - loop), 0, color);
            /* To avoid moiré pattern. Don't do an extra one on the outer
             * ellipse.  We draw another ellipse offset by a pixel, over
             * drawing the missed spots in the filled circle caused by which
             * pixels are filled.
            */
            if (width > 1 && loop > 0)
                draw_ellipse(surf, posx + 1, posy, 2 * (radius - loop),
                             2 * (radius - loop), 0, color);
        }
    }

    if (!pgSurface_Unlock(surfobj))
        return NULL;

    l = MAX(posx - radius, surf->clip_rect.x);
    t = MAX(posy - radius, surf->clip_rect.y);
    r = MIN(posx + radius, surf->clip_rect.x + surf->clip_rect.w);
    b = MIN(posy + radius, surf->clip_rect.y + surf->clip_rect.h);
    return pgRect_New4(l, t, MAX(r - l, 0), MAX(b - t, 0));
}

static PyObject *
polygon(PyObject *self, PyObject *arg)
{
    PyObject *surfobj, *colorobj, *points, *item;
    SDL_Surface *surf;
    Uint8 rgba[4];
    Uint32 color;
    int width = 0, length, loop, numpoints;
    int *xlist, *ylist;
    int x, y, top, left, bottom, right, result;

    /*get all the arguments*/
    if (!PyArg_ParseTuple(arg, "O!OO|i", &pgSurface_Type, &surfobj, &colorobj,
                          &points, &width))
        return NULL;

    if (width) {
        PyObject *args, *ret;
        args = Py_BuildValue("(OOiOi)", surfobj, colorobj, 1, points, width);
        if (!args)
            return NULL;
        ret = lines(NULL, args);
        Py_DECREF(args);
        return ret;
    }

    surf = pgSurface_AsSurface(surfobj);

    if (surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4)
        return RAISE(PyExc_ValueError, "unsupport bit depth for line draw");

    if (PyInt_Check(colorobj))
        color = (Uint32)PyInt_AsLong(colorobj);
    else if (pg_RGBAFromColorObj(colorobj, rgba))
        color = SDL_MapRGBA(surf->format, rgba[0], rgba[1], rgba[2], rgba[3]);
    else
        return RAISE(PyExc_TypeError, "invalid color argument");

    if (!PySequence_Check(points))
        return RAISE(PyExc_TypeError,
                     "points argument must be a sequence of number pairs");
    length = PySequence_Length(points);
    if (length < 3)
        return RAISE(PyExc_ValueError,
                     "points argument must contain more than 2 points");

    for(loop = 0; loop < length; ++loop) {
        item = PySequence_GetItem(points, loop);
        result = pg_TwoIntsFromObj(item, &x, &y);
        Py_DECREF(item);
        if(!result)
            return RAISE(PyExc_TypeError, "all points must be number pairs");
    }

    item = PySequence_GetItem(points, 0);
    result = pg_TwoIntsFromObj(item, &x, &y);
    Py_DECREF(item);
    left = right = x;
    top = bottom = y;

    xlist = PyMem_New(int, length);
    ylist = PyMem_New(int, length);

    numpoints = 0;
    for (loop = 0; loop < length; ++loop) {
        item = PySequence_GetItem(points, loop);
        result = pg_TwoIntsFromObj(item, &x, &y);
        Py_DECREF(item);
        xlist[numpoints] = x;
        ylist[numpoints] = y;
        ++numpoints;
        left = MIN(x, left);
        top = MIN(y, top);
        right = MAX(x, right);
        bottom = MAX(y, bottom);
    }

    if (!pgSurface_Lock(surfobj)) {
        PyMem_Del(xlist);
        PyMem_Del(ylist);
        return NULL;
    }

    draw_fillpoly(surf, xlist, ylist, numpoints, color);

    PyMem_Del(xlist);
    PyMem_Del(ylist);
    if (!pgSurface_Unlock(surfobj))
        return NULL;

    left = MAX(left, surf->clip_rect.x);
    top = MAX(top, surf->clip_rect.y);
    right = MIN(right, surf->clip_rect.x + surf->clip_rect.w);
    bottom = MIN(bottom, surf->clip_rect.y + surf->clip_rect.h);
    return pgRect_New4(left, top, right - left + 1, bottom - top + 1);
}

static PyObject *
rect(PyObject *self, PyObject *arg)
{
    PyObject *surfobj, *colorobj, *rectobj, *points, *args, *ret = NULL;
    GAME_Rect *rect, temp;
    int t, l, b, r, width = 0;

    /*get all the arguments*/
    if (!PyArg_ParseTuple(arg, "O!OO|i", &pgSurface_Type, &surfobj, &colorobj,
                          &rectobj, &width))
        return NULL;

    if (!(rect = pgRect_FromObject(rectobj, &temp)))
        return RAISE(PyExc_TypeError, "Rect argument is invalid");

    l = rect->x;
    r = rect->x + rect->w - 1;
    t = rect->y;
    b = rect->y + rect->h - 1;

    /*build the pointlist*/
    points = Py_BuildValue("((ii)(ii)(ii)(ii))", l, t, r, t, r, b, l, b);

    args = Py_BuildValue("(OONi)", surfobj, colorobj, points, width);
    if (args)
        ret = polygon(NULL, args);

    Py_XDECREF(args);
    return ret;
}

/*internal drawing tools*/

static int
clip_and_draw_aaline(SDL_Surface *surf, SDL_Rect *rect, Uint32 color,
                     float *pts, int blend)
{
    if (!clipaaline(pts, rect->x + 1, rect->y + 1, rect->x + rect->w - 2,
                    rect->y + rect->h - 2))
        return 0;
    drawaaline(surf, color, pts[0], pts[1], pts[2], pts[3], blend);
    return 1;
}

static int
clip_and_draw_line(SDL_Surface *surf, SDL_Rect *rect, Uint32 color, int *pts)
{
    if (!clipline(pts, rect->x, rect->y, rect->x + rect->w - 1,
                  rect->y + rect->h - 1))
        return 0;
    if (pts[1] == pts[3])
        drawhorzline(surf, color, pts[0], pts[1], pts[2]);
    else if (pts[0] == pts[2])
        drawvertline(surf, color, pts[0], pts[1], pts[3]);
    else
        drawline(surf, color, pts[0], pts[1], pts[2], pts[3]);
    return 1;
}

static int
clip_and_draw_line_width(SDL_Surface *surf, SDL_Rect *rect, Uint32 color,
                         int width, int *pts)
{
    int loop;
    int xinc = 0, yinc = 0;
    int newpts[4];
    int range[4];
    int anydrawn = 0;

    if (abs(pts[0] - pts[2]) > abs(pts[1] - pts[3]))
        yinc = 1;
    else
        xinc = 1;

    memcpy(newpts, pts, sizeof(int) * 4);
    if (clip_and_draw_line(surf, rect, color, newpts)) {
        anydrawn = 1;
        memcpy(range, newpts, sizeof(int) * 4);
    }
    else {
        range[0] = range[1] = 10000;
        range[2] = range[3] = -10000;
    }

    for (loop = 1; loop < width; loop += 2) {
        newpts[0] = pts[0] + xinc * (loop / 2 + 1);
        newpts[1] = pts[1] + yinc * (loop / 2 + 1);
        newpts[2] = pts[2] + xinc * (loop / 2 + 1);
        newpts[3] = pts[3] + yinc * (loop / 2 + 1);
        if (clip_and_draw_line(surf, rect, color, newpts)) {
            anydrawn = 1;
            range[0] = MIN(newpts[0], range[0]);
            range[1] = MIN(newpts[1], range[1]);
            range[2] = MAX(newpts[2], range[2]);
            range[3] = MAX(newpts[3], range[3]);
        }
        if (loop + 1 < width) {
            newpts[0] = pts[0] - xinc * (loop / 2 + 1);
            newpts[1] = pts[1] - yinc * (loop / 2 + 1);
            newpts[2] = pts[2] - xinc * (loop / 2 + 1);
            newpts[3] = pts[3] - yinc * (loop / 2 + 1);
            if (clip_and_draw_line(surf, rect, color, newpts)) {
                anydrawn = 1;
                range[0] = MIN(newpts[0], range[0]);
                range[1] = MIN(newpts[1], range[1]);
                range[2] = MAX(newpts[2], range[2]);
                range[3] = MAX(newpts[3], range[3]);
            }
        }
    }
    if (anydrawn)
        memcpy(pts, range, sizeof(int) * 4);
    return anydrawn;
}

#define SWAP(a, b, tmp) \
    tmp = b;            \
    b = a;              \
    a = tmp;            \

/*this line clipping based heavily off of code from
http://www.ncsa.uiuc.edu/Vis/Graphics/src/clipCohSuth.c */
#define LEFT_EDGE 0x1
#define RIGHT_EDGE 0x2
#define BOTTOM_EDGE 0x4
#define TOP_EDGE 0x8
#define INSIDE(a) (!a)
#define REJECT(a, b) (a & b)
#define ACCEPT(a, b) (!(a | b))

static int
encode(int x, int y, int left, int top, int right, int bottom)
{
    int code = 0;
    if (x < left)
        code |= LEFT_EDGE;
    if (x > right)
        code |= RIGHT_EDGE;
    if (y < top)
        code |= TOP_EDGE;
    if (y > bottom)
        code |= BOTTOM_EDGE;
    return code;
}

static int
encodeFloat(float x, float y, int left, int top, int right, int bottom)
{
    int code = 0;
    if (x < left)
        code |= LEFT_EDGE;
    if (x > right)
        code |= RIGHT_EDGE;
    if (y < top)
        code |= TOP_EDGE;
    if (y > bottom)
        code |= BOTTOM_EDGE;
    return code;
}

static int
clipaaline(float *pts, int left, int top, int right, int bottom)
{
    float x1 = pts[0];
    float y1 = pts[1];
    float x2 = pts[2];
    float y2 = pts[3];
    int code1, code2;
    int draw = 0;
    float swaptmp;
    int intswaptmp;
    float m; /*slope*/

    while (1) {
        code1 = encodeFloat(x1, y1, left, top, right, bottom);
        code2 = encodeFloat(x2, y2, left, top, right, bottom);
        if (ACCEPT(code1, code2)) {
            draw = 1;
            break;
        }
        else if (REJECT(code1, code2)) {
            break;
        }
        else {
            if (INSIDE(code1)) {
                swaptmp = x2;
                x2 = x1;
                x1 = swaptmp;
                swaptmp = y2;
                y2 = y1;
                y1 = swaptmp;
                intswaptmp = code2;
                code2 = code1;
                code1 = intswaptmp;
            }
            if (x2 != x1)
                m = (y2 - y1) / (x2 - x1);
            else
                m = 1.0f;
            if (code1 & LEFT_EDGE) {
                y1 += ((float)left - x1) * m;
                x1 = (float)left;
            }
            else if (code1 & RIGHT_EDGE) {
                y1 += ((float)right - x1) * m;
                x1 = (float)right;
            }
            else if (code1 & BOTTOM_EDGE) {
                if (x2 != x1)
                    x1 += ((float)bottom - y1) / m;
                y1 = (float)bottom;
            }
            else if (code1 & TOP_EDGE) {
                if (x2 != x1)
                    x1 += ((float)top - y1) / m;
                y1 = (float)top;
            }
        }
    }
    if (draw) {
        pts[0] = x1;
        pts[1] = y1;
        pts[2] = x2;
        pts[3] = y2;
    }
    return draw;
}

static int
clipline(int *pts, int left, int top, int right, int bottom)
{
    /*
     * Algorithm to calculate the clipped line.
     *
     * We write the coordinates of the part of the line
     * segment within the bounding box defined
     * by (left, top, right, bottom) into the "pts" array.
     * Returns 0 if we don't have to draw anything, eg if the
     * segment defined throuth "pts" doesn't cross the bounding box.
     */
    int x1 = pts[0];
    int y1 = pts[1];
    int x2 = pts[2];
    int y2 = pts[3];
    int code1, code2;
    int draw = 0;
    int swaptmp;
    float m; /*slope*/

    while (1) {
        code1 = encode(x1, y1, left, top, right, bottom);
        code2 = encode(x2, y2, left, top, right, bottom);
        if (ACCEPT(code1, code2)) {
            draw = 1;
            pts[0] = x1;
            pts[1] = y1;
            pts[2] = x2;
            pts[3] = y2;
            break;
        }
        else if (REJECT(code1, code2))
            break;
        else {
            if (INSIDE(code1)) {
                SWAP(x1, x2, swaptmp)
                SWAP(y1, y2, swaptmp)
                SWAP(code1, code2, swaptmp)
            }
            if (x2 != x1)
                m = (y2 - y1) / (float)(x2 - x1);
            else
                m = 1.0f;
            if (code1 & LEFT_EDGE) {
                y1 += (int)((left - x1) * m);
                x1 = left;
            }
            else if (code1 & RIGHT_EDGE) {
                y1 += (int)((right - x1) * m);
                x1 = right;
            }
            else if (code1 & BOTTOM_EDGE) {
                if (x2 != x1)
                    x1 += (int)((bottom - y1) / m);
                y1 = bottom;
            }
            else if (code1 & TOP_EDGE) {
                if (x2 != x1)
                    x1 += (int)((top - y1) / m);
                y1 = top;
            }
        }
    }
    return draw;
}

static int
set_at(SDL_Surface *surf, int x, int y, Uint32 color)
{
    SDL_PixelFormat *format = surf->format;
    Uint8 *pixels = (Uint8 *)surf->pixels;
    Uint8 *byte_buf, rgb[4];

    if (x < surf->clip_rect.x || x >= surf->clip_rect.x + surf->clip_rect.w ||
        y < surf->clip_rect.y || y >= surf->clip_rect.y + surf->clip_rect.h)
        return 0;

    switch (format->BytesPerPixel) {
        case 1:
            *((Uint8 *)pixels + y * surf->pitch + x) = (Uint8)color;
            break;
        case 2:
            *((Uint16 *)(pixels + y * surf->pitch) + x) = (Uint16)color;
            break;
        case 4:
            *((Uint32 *)(pixels + y * surf->pitch) + x) = color;
            /*              *((Uint32*)(pixels + y * surf->pitch) + x) =
                            ~(*((Uint32*)(pixels + y * surf->pitch) + x)) * 31;
            */
            break;
        default: /*case 3:*/
            SDL_GetRGB(color, format, rgb, rgb + 1, rgb + 2);
            byte_buf = (Uint8 *)(pixels + y * surf->pitch) + x * 3;
#if (SDL_BYTEORDER == SDL_LIL_ENDIAN)
            *(byte_buf + (format->Rshift >> 3)) = rgb[0];
            *(byte_buf + (format->Gshift >> 3)) = rgb[1];
            *(byte_buf + (format->Bshift >> 3)) = rgb[2];
#else
            *(byte_buf + 2 - (format->Rshift >> 3)) = rgb[0];
            *(byte_buf + 2 - (format->Gshift >> 3)) = rgb[1];
            *(byte_buf + 2 - (format->Bshift >> 3)) = rgb[2];
#endif
            break;
    }
    return 1;
}

#define DRAWPIX32(pixel, colorptr, br, blend)                               \
    if (blend) {                                                            \
        SDL_GetRGBA(*pixel, surf->format, &pixel_r, &pixel_g, &pixel_b,     \
                    &pixel_a);                                              \
        tmp_r = color_r * br + pixel_r * nbr;                               \
        tmp_g = color_g * br + pixel_g * nbr;                               \
        tmp_b = color_b * br + pixel_b * nbr;                               \
        tmp_a = color_a * br + pixel_a * nbr;                               \
        *((Uint32 *)pixel) =                                                \
            SDL_MapRGBA(surf->format, (Uint8)((tmp_r > 254) ? 255 : tmp_r), \
                        (Uint8)((tmp_g > 254) ? 255 : tmp_g),               \
                        (Uint8)((tmp_b > 254) ? 255 : tmp_b),               \
                        (Uint8)((tmp_a > 254) ? 255 : tmp_a));              \
    }                                                                       \
    else {                                                                  \
        pixel[0] = (Uint8)(colorptr[0] * br);                               \
        pixel[1] = (Uint8)(colorptr[1] * br);                               \
        pixel[2] = (Uint8)(colorptr[2] * br);                               \
        if (hasalpha)                                                       \
            pixel[3] = br * 255;                                            \
    }

/* Adapted from http://freespace.virgin.net/hugo.elias/graphics/x_wuline.htm */
static void
drawaaline(SDL_Surface *surf, Uint32 color, float x1, float y1, float x2,
           float y2, int blend)
{
    float grad, xd, yd;
    float xgap, ygap, xend, yend, xf, yf;
    float brightness1, brightness2;
    float swaptmp;
    int x, y, ix1, ix2, iy1, iy2;
    int pixx, pixy;

    /* for D-RAWPIX32 */
    int tmp_r, tmp_g, tmp_b, tmp_a;
    float nbr = 0.0f;
    Uint8 pixel_r, pixel_g, pixel_b, pixel_a;
    Uint8 color_r, color_g, color_b, color_a;

    Uint8 *pixel;
    Uint8 *pm = (Uint8 *)surf->pixels;
    Uint8 *colorptr = (Uint8 *)&color;
    const int hasalpha = surf->format->Amask;

    if (hasalpha) {
        SDL_GetRGBA(color, surf->format, &color_r, &color_g, &color_b,
                    &color_a);
    }
    else {
        SDL_GetRGB(color, surf->format, &color_r, &color_g, &color_b);
    }
    pixx = surf->format->BytesPerPixel;
    pixy = surf->pitch;

    xd = x2 - x1;
    yd = y2 - y1;

    if (xd == 0 && yd == 0) {
        /* Single point. Due to the nature of the aaline clipping, this
         * is less exact than the normal line. */
        set_at(surf, x1, y1, color);
        return;
    }

    if (fabs(xd) > fabs(yd)) {
        if (x1 > x2) {
            SWAP(x1, x2, swaptmp)
            SWAP(y1, y2, swaptmp)
            xd = (x2 - x1);
            yd = (y2 - y1);
        }
        grad = yd / xd;
        xend = trunc(x1) + 0.5; /* This makes more sense than trunc(x1+0.5) */
        yend = y1 + grad * (xend - x1);
        xgap = INVFRAC(x1);
        ix1 = (int)xend;
        iy1 = (int)yend;
        yf = yend + grad;
        brightness1 = INVFRAC(yend) * xgap;
        brightness2 = FRAC(yend) * xgap;
        pixel = pm + pixx * ix1 + pixy * iy1;
        DRAWPIX32(pixel, colorptr, brightness1, blend)
        pixel += pixy;
        DRAWPIX32(pixel, colorptr, brightness2, blend)
        xend = trunc(x2) + 0.5;
        yend = y2 + grad * (xend - x2);
        xgap = FRAC(x2); /* this also differs from Hugo's description. */
        ix2 = (int)xend;
        iy2 = (int)yend;
        brightness1 = INVFRAC(yend) * xgap;
        brightness2 = FRAC(yend) * xgap;
        pixel = pm + pixx * ix2 + pixy * iy2;
        DRAWPIX32(pixel, colorptr, brightness1, blend)
        pixel += pixy;
        DRAWPIX32(pixel, colorptr, brightness2, blend)
        for (x = ix1 + 1; x < ix2; ++x) {
            brightness1 = INVFRAC(yf);
            brightness2 = FRAC(yf);
            pixel = pm + pixx * x + pixy * (int)yf;
            DRAWPIX32(pixel, colorptr, brightness1, blend)
            pixel += pixy;
            DRAWPIX32(pixel, colorptr, brightness2, blend)
            yf += grad;
        }
    }
    else {
        if (y1 > y2) {
            SWAP(x1, x2, swaptmp)
            SWAP(y1, y2, swaptmp)
            yd = (y2 - y1);
            xd = (x2 - x1);
        }
        grad = xd / yd;
        yend = trunc(y1) + 0.5; /* This makes more sense than trunc(x1+0.5) */
        xend = x1 + grad * (yend - y1);
        ygap = INVFRAC(y1);
        iy1 = (int)yend;
        ix1 = (int)xend;
        xf = xend + grad;
        brightness1 = INVFRAC(xend) * ygap;
        brightness2 = FRAC(xend) * ygap;
        pixel = pm + pixx * ix1 + pixy * iy1;
        DRAWPIX32(pixel, colorptr, brightness1, blend)
        pixel += pixx;
        DRAWPIX32(pixel, colorptr, brightness2, blend)
        yend = trunc(y2) + 0.5;
        xend = x2 + grad * (yend - y2);
        ygap = FRAC(y2);
        iy2 = (int)yend;
        ix2 = (int)xend;
        brightness1 = INVFRAC(xend) * ygap;
        brightness2 = FRAC(xend) * ygap;
        pixel = pm + pixx * ix2 + pixy * iy2;
        DRAWPIX32(pixel, colorptr, brightness1, blend)
        pixel += pixx;
        DRAWPIX32(pixel, colorptr, brightness2, blend)
        for (y = iy1 + 1; y < iy2; ++y) {
            brightness1 = INVFRAC(xf);
            brightness2 = FRAC(xf);
            pixel = pm + pixx * (int)xf + pixy * y;
            DRAWPIX32(pixel, colorptr, brightness1, blend)
            pixel += pixx;
            DRAWPIX32(pixel, colorptr, brightness2, blend)
            xf += grad;
        }
    }
}

/*here's my sdl'ized version of bresenham*/
static void
drawline(SDL_Surface *surf, Uint32 color, int x1, int y1, int x2, int y2)
{
    int deltax, deltay, signx, signy;
    int pixx, pixy;
    int x = 0, y = 0;
    int swaptmp;
    Uint8 *pixel;
    Uint8 *colorptr;

    deltax = x2 - x1;
    deltay = y2 - y1;
    signx = (deltax < 0) ? -1 : 1;
    signy = (deltay < 0) ? -1 : 1;
    deltax = signx * deltax + 1;
    deltay = signy * deltay + 1;

    pixx = surf->format->BytesPerPixel;
    pixy = surf->pitch;
    pixel = ((Uint8 *)surf->pixels) + pixx * x1 + pixy * y1;

    pixx *= signx;
    pixy *= signy;
    if (deltax < deltay) /*swap axis if rise > run*/
    {
        SWAP(deltax, deltay, swaptmp)
        SWAP(pixx, pixy, swaptmp)
    }

    switch (surf->format->BytesPerPixel) {
        case 1:
            for (; x < deltax; x++, pixel += pixx) {
                *pixel = (Uint8)color;
                y += deltay;
                if (y >= deltax) {
                    y -= deltax;
                    pixel += pixy;
                }
            }
            break;
        case 2:
            for (; x < deltax; x++, pixel += pixx) {
                *(Uint16 *)pixel = (Uint16)color;
                y += deltay;
                if (y >= deltax) {
                    y -= deltax;
                    pixel += pixy;
                }
            }
            break;
        case 3:
            if (SDL_BYTEORDER == SDL_BIG_ENDIAN)
                color <<= 8;
            colorptr = (Uint8 *)&color;
            for (; x < deltax; x++, pixel += pixx) {
                pixel[0] = colorptr[0];
                pixel[1] = colorptr[1];
                pixel[2] = colorptr[2];
                y += deltay;
                if (y >= deltax) {
                    y -= deltax;
                    pixel += pixy;
                }
            }
            break;
        default: /*case 4*/
            for (; x < deltax; x++, pixel += pixx) {
                *(Uint32 *)pixel = (Uint32)color;
                y += deltay;
                if (y >= deltax) {
                    y -= deltax;
                    pixel += pixy;
                }
            }
            break;
    }
}

static void
drawhorzline(SDL_Surface *surf, Uint32 color, int x1, int y1, int x2)
{
    Uint8 *pixel, *end;
    Uint8 *colorptr;

    if (x1 == x2) {
        set_at(surf, x1, y1, color);
        return;
    }

    pixel = ((Uint8 *)surf->pixels) + surf->pitch * y1;
    if (x1 < x2) {
        end = pixel + x2 * surf->format->BytesPerPixel;
        pixel += x1 * surf->format->BytesPerPixel;
    }
    else {
        end = pixel + x1 * surf->format->BytesPerPixel;
        pixel += x2 * surf->format->BytesPerPixel;
    }
    switch (surf->format->BytesPerPixel) {
        case 1:
            for (; pixel <= end; ++pixel) {
                *pixel = (Uint8)color;
            }
            break;
        case 2:
            for (; pixel <= end; pixel += 2) {
                *(Uint16 *)pixel = (Uint16)color;
            }
            break;
        case 3:
            if (SDL_BYTEORDER == SDL_BIG_ENDIAN)
                color <<= 8;
            colorptr = (Uint8 *)&color;
            for (; pixel <= end; pixel += 3) {
                pixel[0] = colorptr[0];
                pixel[1] = colorptr[1];
                pixel[2] = colorptr[2];
            }
            break;
        default: /*case 4*/
            for (; pixel <= end; pixel += 4) {
                *(Uint32 *)pixel = color;
            }
            break;
    }
}

static void
drawhorzlineclip(SDL_Surface *surf, Uint32 color, int x1, int y1, int x2)
{
    if (y1 < surf->clip_rect.y || y1 >= surf->clip_rect.y + surf->clip_rect.h)
        return;

    if (x2 < x1) {
        int temp = x1;
        x1 = x2;
        x2 = temp;
    }

    x1 = MAX(x1, surf->clip_rect.x);
    x2 = MIN(x2, surf->clip_rect.x + surf->clip_rect.w - 1);

    if (x2 < surf->clip_rect.x || x1 >= surf->clip_rect.x + surf->clip_rect.w)
        return;

    drawhorzline(surf, color, x1, y1, x2);
}

static void
drawvertline(SDL_Surface *surf, Uint32 color, int x1, int y1, int y2)
{
    Uint8 *pixel, *end;
    Uint8 *colorptr;
    Uint32 pitch = surf->pitch;

    if (y1 == y2) {
        set_at(surf, x1, y1, color);
        return;
    }

    pixel = ((Uint8 *)surf->pixels) + x1 * surf->format->BytesPerPixel;
    if (y1 < y2) {
        end = pixel + surf->pitch * y2;
        pixel += surf->pitch * y1;
    }
    else {
        end = pixel + surf->pitch * y1;
        pixel += surf->pitch * y2;
    }

    switch (surf->format->BytesPerPixel) {
        case 1:
            for (; pixel <= end; pixel += pitch) {
                *pixel = (Uint8)color;
            }
            break;
        case 2:
            for (; pixel <= end; pixel += pitch) {
                *(Uint16 *)pixel = (Uint16)color;
            }
            break;
        case 3:
            if (SDL_BYTEORDER == SDL_BIG_ENDIAN)
                color <<= 8;
            colorptr = (Uint8 *)&color;
            for (; pixel <= end; pixel += pitch) {
                pixel[0] = colorptr[0];
                pixel[1] = colorptr[1];
                pixel[2] = colorptr[2];
            }
            break;
        default: /*case 4*/
            for (; pixel <= end; pixel += pitch) {
                *(Uint32 *)pixel = color;
            }
            break;
    }
}

static void
drawvertlineclip(SDL_Surface *surf, Uint32 color, int x1, int y1, int y2)
{
    if (x1 < surf->clip_rect.x || x1 >= surf->clip_rect.x + surf->clip_rect.w)
        return;
    if (y2 < y1) {
        int temp = y1;
        y1 = y2;
        y2 = temp;
    }
    y1 = MAX(y1, surf->clip_rect.y);
    y2 = MIN(y2, surf->clip_rect.y + surf->clip_rect.h - 1);

    drawvertline(surf, color, x1, y1, y2);
}

static void
draw_arc(SDL_Surface *dst, int x, int y, int radius1, int radius2,
         double angle_start, double angle_stop, Uint32 color)
{
    double aStep;  // Angle Step (rad)
    double a;      // Current Angle (rad)
    int x_last, x_next, y_last, y_next;

    // Angle step in rad
    if (radius1 < radius2) {
        if (radius1 < 1.0e-4) {
            aStep = 1.0;
        }
        else {
            aStep = asin(2.0 / radius1);
        }
    }
    else {
        if (radius2 < 1.0e-4) {
            aStep = 1.0;
        }
        else {
            aStep = asin(2.0 / radius2);
        }
    }

    if (aStep < 0.05) {
        aStep = 0.05;
    }

    x_last = x + cos(angle_start) * radius1;
    y_last = y - sin(angle_start) * radius2;
    for (a = angle_start + aStep; a <= angle_stop; a += aStep) {
        int points[4];
        x_next = x + cos(a) * radius1;
        y_next = y - sin(a) * radius2;
        points[0] = x_last;
        points[1] = y_last;
        points[2] = x_next;
        points[3] = y_next;
        clip_and_draw_line(dst, &dst->clip_rect, color, points);
        x_last = x_next;
        y_last = y_next;
    }
}

static void
draw_ellipse(SDL_Surface *dst, int x, int y, int width, int height, int solid,
             Uint32 color)
{
    int ix, iy;
    int h, i, j, k;
    int oh, oi, oj, ok;
    int xoff = (width & 1) ^ 1;
    int yoff = (height & 1) ^ 1;
    int rx = (width >> 1);
    int ry = (height >> 1) - yoff + (solid & 1);

    if (rx == 0 && ry == 0) { /* Special case - draw a single pixel */
        set_at(dst, x, y, color);
        return;
    }
    if (rx == 0) { /* Special case for rx=0 - draw a vline */
        drawvertlineclip(dst, color, x, (Sint16)(y - ry), (Sint16)(y + ry));
        return;
    }
    if (ry == 0) { /* Special case for ry=0 - draw a hline */
        drawhorzlineclip(dst, color, (Sint16)(x - rx), y, (Sint16)(x + rx));
        return;
    }

    /* Init vars */
    oh = oi = oj = ok = 0xFFFF;

    /* Draw */
    if (rx >= ry) {
        ix = 0;
        iy = rx * 64;

        do {
            h = (ix + 8) >> 6;
            i = (iy + 8) >> 6;
            j = (h * ry) / rx;
            k = (i * ry) / rx;
            if (((ok != k) && (oj != k) && (k < ry)) || !solid) {
                if (solid) {
                    drawhorzlineclip(dst, color, x - h, y - k - yoff,
                                     x + h - xoff);
                    drawhorzlineclip(dst, color, x - h, y + k, x + h - xoff);
                }
                else {
                    set_at(dst, x - h, y - k - yoff, color);
                    set_at(dst, x + h - xoff, y - k - yoff, color);
                    set_at(dst, x - h, y + k, color);
                    set_at(dst, x + h - xoff, y + k, color);
                }
                ok = k;
            }
            if (((oj != j) && (ok != j) && (k != j)) || !solid) {
                if (solid) {
                    drawhorzlineclip(dst, color, x - i, y + j, x + i - xoff);
                    drawhorzlineclip(dst, color, x - i, y - j - yoff,
                                     x + i - xoff);
                }
                else {
                    set_at(dst, x - i, y + j, color);
                    set_at(dst, x + i - xoff, y + j, color);
                    set_at(dst, x - i, y - j - yoff, color);
                    set_at(dst, x + i - xoff, y - j - yoff, color);
                }
                oj = j;
            }
            ix = ix + iy / rx;
            iy = iy - ix / rx;

        } while (i > h);
    }
    else {
        ix = 0;
        iy = ry * 64;

        do {
            h = (ix + 8) >> 6;
            i = (iy + 8) >> 6;
            j = (h * rx) / ry;
            k = (i * rx) / ry;

            if (((oi != i) && (oh != i) && (i < ry)) || !solid) {
                if (solid) {
                    drawhorzlineclip(dst, color, x - j, y + i, x + j - xoff);
                    drawhorzlineclip(dst, color, x - j, y - i - yoff,
                                     x + j - xoff);
                }
                else {
                    set_at(dst, x - j, y + i, color);
                    set_at(dst, x + j - xoff, y + i, color);
                    set_at(dst, x - j, y - i - yoff, color);
                    set_at(dst, x + j - xoff, y - i - yoff, color);
                }
                oi = i;
            }
            if (((oh != h) && (oi != h) && (i != h)) || !solid) {
                if (solid) {
                    drawhorzlineclip(dst, color, x - k, y + h, x + k - xoff);
                    drawhorzlineclip(dst, color, x - k, y - h - yoff,
                                     x + k - xoff);
                }
                else {
                    set_at(dst, x - k, y + h, color);
                    set_at(dst, x + k - xoff, y + h, color);
                    set_at(dst, x - k, y - h - yoff, color);
                    set_at(dst, x + k - xoff, y - h - yoff, color);
                }
                oh = h;
            }

            ix = ix + iy / ry;
            iy = iy - ix / ry;

        } while (i > h);
    }
}

static int
compare_int(const void *a, const void *b)
{
    return (*(const int *)a) - (*(const int *)b);
}

static void
draw_fillpoly(SDL_Surface *dst, int *point_x, int *point_y, int num_points,
              Uint32 color)
{
    /* point_x : x coordinates of the points
     * point-y : the y coordinates of the points
     * num_points : the number of points
     */
    int i, i_previous, y;  // i_previous is the index of the point before i
    int miny, maxy;
    int x1, y1;
    int x2, y2;
    /* x_intersect are the x-coordinates of intersections of the polygon
     * with some horizontal line */
    int *x_intersect = PyMem_New(int, num_points);
    if (x_intersect == NULL) {
        PyErr_NoMemory();
        return;
    }

    /* Determine Y maxima */
    miny = point_y[0];
    maxy = point_y[0];
    for (i = 1; (i < num_points); i++) {
        miny = MIN(miny, point_y[i]);
        maxy = MAX(maxy, point_y[i]);
    }

    if (miny == maxy) {
        /* Special case: polygon only 1 pixel high. */

        /* Determine X bounds */
        int minx = point_x[0];
        int maxx = point_x[0];
        for (i = 1; (i < num_points); i++) {
            minx = MIN(minx, point_x[i]);
            maxx = MAX(maxx, point_x[i]);
        }
        drawhorzlineclip(dst, color, minx, miny, maxx);
        PyMem_Free(x_intersect);
        return;
    }

    /* Draw, scanning y
     * ----------------
     * The algorithm uses a horizontal line (y) that moves from top to the
     * bottom of the polygon:
     *
     * 1. search intersections with the border lines
     * 2. sort intersections (x_intersect)
     * 3. each two x-coordinates in x_intersect are then inside the polygon
     *    (drawhorzlineclip for a pair of two such points)
     */
    for (y = miny; (y <= maxy); y++) {
        // n_intersections is the number of intersections with the polygon
        int n_intersections = 0;
        for (i = 0; (i < num_points); i++) {
            i_previous = ((i) ? (i - 1) : (num_points - 1));

            y1 = point_y[i_previous];
            y2 = point_y[i];
            if (y1 < y2) {
                x1 = point_x[i_previous];
                x2 = point_x[i];
            }
            else if (y1 > y2) {
                y2 = point_y[i_previous];
                y1 = point_y[i];
                x2 = point_x[i_previous];
                x1 = point_x[i];
            }
            else {  // y1 == y2 : has to be handled as special case (below)
                continue;
            }
            if (((y >= y1) && (y < y2)) || ((y == maxy) && (y2 == maxy))) {
                // add intersection if y crosses the edge (excluding the lower
                // end), or when we are on the lowest line (maxy)
                x_intersect[n_intersections++] =
                    (y - y1) * (x2 - x1) / (y2 - y1) + x1;
            }
        }
        qsort(x_intersect, n_intersections, sizeof(int), compare_int);

        for (i = 0; (i < n_intersections); i += 2) {
            drawhorzlineclip(dst, color, x_intersect[i], y,
                             x_intersect[i + 1]);
        }
    }

    /* Finally, a special case is not handled by above algorithm:
     *
     * For two border points with same height miny < y < maxy,
     * sometimes the line between them is not colored:
     * this happens when the line will be a lower border line of the polygon
     * (eg we are inside the polygon with a smaller y, and outside with a
     * bigger y),
     * So we loop for border lines that are horizontal.
     */
    for (i = 0; (i < num_points); i++) {
        i_previous = ((i) ? (i - 1) : (num_points - 1));
        y = point_y[i];

        if ((miny < y) && (point_y[i_previous] == y) && (y < maxy)) {
            drawhorzlineclip(dst, color, point_x[i], y, point_x[i_previous]);
        }
    }
    PyMem_Free(x_intersect);
}

static PyMethodDef _draw_methods[] = {
    {"aaline", aaline, METH_VARARGS, DOC_PYGAMEDRAWAALINE},
    {"line", line, METH_VARARGS, DOC_PYGAMEDRAWLINE},
    {"aalines", aalines, METH_VARARGS, DOC_PYGAMEDRAWAALINES},
    {"lines", lines, METH_VARARGS, DOC_PYGAMEDRAWLINES},
    {"ellipse", ellipse, METH_VARARGS, DOC_PYGAMEDRAWELLIPSE},
    {"arc", arc, METH_VARARGS, DOC_PYGAMEDRAWARC},
    {"circle", circle, METH_VARARGS, DOC_PYGAMEDRAWCIRCLE},
    {"polygon", polygon, METH_VARARGS, DOC_PYGAMEDRAWPOLYGON},
    {"rect", rect, METH_VARARGS, DOC_PYGAMEDRAWRECT},

    {NULL, NULL, 0, NULL}};

MODINIT_DEFINE(draw)
{
#if PY3
    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "draw",
                                         DOC_PYGAMEDRAW,
                                         -1,
                                         _draw_methods,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL};
#endif

    /* imported needed apis; Do this first so if there is an error
       the module is not loaded.
    */
    import_pygame_base();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }
    import_pygame_color();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }
    import_pygame_rect();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }
    import_pygame_surface();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }

/* create the module */
#if PY3
    return PyModule_Create(&_module);
#else
    Py_InitModule3(MODPREFIX "draw", _draw_methods, DOC_PYGAMEDRAW);
#endif
}
