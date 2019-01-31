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

/*
    Many C libraries seem to lack the trunc call (added in C99).

    Not sure int() is usable for all cases where trunc is used in this code?
    However casting to int gives quite a speedup over the one defined.
    Now sure how it compares to the trunc built into the C library.
    #define trunc(d) ((int)(d))
*/
#if (!defined(__STDC_VERSION__) || __STDC_VERSION__ < 199901L) && !defined(trunc)
#define trunc(d) (((d) >= 0.0) ? (floor(d)) : (ceil(d)))
#endif

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
clip_aaline(float *pts, int left, int top, int right, int bottom);
static void
drawline(SDL_Surface *surf, Uint32 color, int startx, int starty, int endx,
         int endy);
static void
draw_aaline(SDL_Surface *surf, Uint32 color, float startx, float starty,
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

// validation of a draw color
#define CHECK_LOAD_COLOR(colorobj)                                         \
    if (PyInt_Check(colorobj))                                             \
        color = (Uint32)PyInt_AsLong(colorobj);                            \
    else if (pg_RGBAFromColorObj(colorobj, rgba))                          \
        color =                                                            \
            SDL_MapRGBA(surf->format, rgba[0], rgba[1], rgba[2], rgba[3]); \
    else                                                                   \
        return RAISE(PyExc_TypeError, "invalid color argument");

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

    CHECK_LOAD_COLOR(colorobj)

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

    CHECK_LOAD_COLOR(colorobj)

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
    float top, left, bottom, right;
    float pts[4];
    Uint8 rgba[4];
    Uint32 color;
    int closed, blend=1;
    int result, loop, length;
    float *xlist, *ylist;

    /*get all the arguments*/
    if (!PyArg_ParseTuple(arg, "O!OOO|i", &pgSurface_Type, &surfobj, &colorobj,
                          &closedobj, &points, &blend))
        return NULL;
    surf = pgSurface_AsSurface(surfobj);

    CHECK_LOAD_COLOR(colorobj)

    closed = PyObject_IsTrue(closedobj);

    if (!PySequence_Check(points))
        return RAISE(PyExc_TypeError,
                     "points argument must be a sequence of number pairs");
    length = PySequence_Length(points);
    if (length < 2)
        return RAISE(PyExc_ValueError,
                     "points argument must contain more than 1 points");

    xlist = PyMem_New(float, length);
    ylist = PyMem_New(float, length);

    left = top = 10000;
    right = bottom = -10000;

    for (loop = 0; loop < length; ++loop) {
        item = PySequence_GetItem(points, loop);
        result = pg_TwoFloatsFromObj(item, &x, &y);
        Py_DECREF(item);
        if (!result) {
            PyMem_Del(xlist);
            PyMem_Del(ylist);
            return RAISE(PyExc_TypeError, "points must be number pairs");
        }
        xlist[loop] = x;
        ylist[loop] = y;
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

    for (loop = 1; loop < length; ++loop) {
        pts[0] = xlist[loop - 1];
        pts[1] = ylist[loop - 1];
        pts[2] = xlist[loop];
        pts[3] = ylist[loop];
        clip_and_draw_aaline(surf, &surf->clip_rect, color, pts, blend);
    }
    if (closed && length > 2) {
        pts[0] = xlist[length - 1];
        pts[1] = ylist[length - 1];
        pts[2] = xlist[0];
        pts[3] = ylist[0];
        clip_and_draw_aaline(surf, &surf->clip_rect, color, pts, blend);
    }

    PyMem_Del(xlist);
    PyMem_Del(ylist);
    if (!pgSurface_Unlock(surfobj))
        return NULL;

    /*compute return rect*/
    return pgRect_New4((int)left, (int)top, (int)(right - left + 2),
                       (int)(bottom - top + 2));
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
    int result, loop, length;
    int *xlist, *ylist;

    /*get all the arguments*/
    if (!PyArg_ParseTuple(arg, "O!OOO|i", &pgSurface_Type, &surfobj, &colorobj,
                          &closedobj, &points, &width))
        return NULL;
    surf = pgSurface_AsSurface(surfobj);

    if (surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4)
        return RAISE(PyExc_ValueError, "unsupport bit depth for line draw");

    CHECK_LOAD_COLOR(colorobj)

    closed = PyObject_IsTrue(closedobj);

    if (!PySequence_Check(points))
        return RAISE(PyExc_TypeError,
                     "points argument must be a sequence of number pairs");
    length = PySequence_Length(points);
    if (length < 2)
        return RAISE(PyExc_ValueError,
                     "points argument must contain more than 1 points");

    left = top = 10000;
    right = bottom = -10000;

    xlist = PyMem_New(int, length);
    ylist = PyMem_New(int, length);

    for (loop = 0; loop < length; ++loop) {
        item = PySequence_GetItem(points, loop);
        result = pg_TwoIntsFromObj(item, &x, &y);
        Py_DECREF(item);
        if (!result) {
            PyMem_Del(xlist);
            PyMem_Del(ylist);
            return RAISE(PyExc_TypeError, "points must be number pairs");
        }
        xlist[loop] = x;
        ylist[loop] = y;
        left = MIN(x, left);
        top = MIN(y, top);
        right = MAX(x, right);
        bottom = MAX(y, bottom);
    }

    if (width < 1)
        return pgRect_New4(left, top, 0, 0);

    if (!pgSurface_Lock(surfobj)) {
        PyMem_Del(xlist);
        PyMem_Del(ylist);
        return NULL;
    }

    for (loop = 1; loop < length; ++loop) {
        pts[0] = xlist[loop - 1];
        pts[1] = ylist[loop - 1];
        pts[2] = xlist[loop];
        pts[3] = ylist[loop];
        clip_and_draw_line_width(surf, &surf->clip_rect, color, width, pts);
    }
    if (closed && length > 2) {
        pts[0] = xlist[length - 1];
        pts[1] = ylist[length - 1];
        pts[2] = xlist[0];
        pts[3] = ylist[0];
        clip_and_draw_line_width(surf, &surf->clip_rect, color, width, pts);
    }

    PyMem_Del(xlist);
    PyMem_Del(ylist);
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

    CHECK_LOAD_COLOR(colorobj)

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

    CHECK_LOAD_COLOR(colorobj)

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

    CHECK_LOAD_COLOR(colorobj)

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
            // if (width > 1 && loop > 0)       // removed due to: 'Gaps in circle for width greater than 1 #736'
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
    int width = 0, length, loop;
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

    CHECK_LOAD_COLOR(colorobj)

    if (!PySequence_Check(points))
        return RAISE(PyExc_TypeError,
                     "points argument must be a sequence of number pairs");
    length = PySequence_Length(points);
    if (length < 3)
        return RAISE(PyExc_ValueError,
                     "points argument must contain more than 2 points");

    left = top = 10000;
    right = bottom = -10000;

    xlist = PyMem_New(int, length);
    ylist = PyMem_New(int, length);

    for (loop = 0; loop < length; ++loop) {
        item = PySequence_GetItem(points, loop);
        result = pg_TwoIntsFromObj(item, &x, &y);
        Py_DECREF(item);
        if (!result) {
            PyMem_Del(xlist);
            PyMem_Del(ylist);
            return RAISE(PyExc_TypeError, "points must be number pairs");
        }
        xlist[loop] = x;
        ylist[loop] = y;
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

    draw_fillpoly(surf, xlist, ylist, length, color);

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
    if (!clip_aaline(pts, rect->x + 1, rect->y + 1, rect->x + rect->w - 2,
                    rect->y + rect->h - 2))
        return 0;
    draw_aaline(surf, color, pts[0], pts[1], pts[2], pts[3], blend);
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
    a = tmp;

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
clip_aaline(float *segment, int left, int top, int right, int bottom)
{
    /*
     * Algorithm to calculate the clipped anti-aliased line.
     *
     * We write the coordinates of the part of the line
     * segment within the bounding box defined
     * by (left, top, right, bottom) into the "segment" array.
     * Returns 0 if we don't have to draw anything, eg if the
     * segment = [from_x, from_y, to_x, to_y]
     * doesn't cross the bounding box.
     */
    float x1 = segment[0];
    float y1 = segment[1];
    float x2 = segment[2];
    float y2 = segment[3];
    int code1, code2;
    float swaptmp;
    int intswaptmp;
    float m; /*slope*/

    while (1) {
        code1 = encodeFloat(x1, y1, left, top, right, bottom);
        code2 = encodeFloat(x2, y2, left, top, right, bottom);
        if (ACCEPT(code1, code2)) {
            segment[0] = x1;
            segment[1] = y1;
            segment[2] = x2;
            segment[3] = y2;
            return 1;
        }
        else if (REJECT(code1, code2)) {
            return 0;
        }
        else {
            if (INSIDE(code1)) {
                SWAP(x1, x2, swaptmp)
                SWAP(y1, y2, swaptmp)
                SWAP(code1, code2, intswaptmp)
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
}

static int
clipline(int *segment, int left, int top, int right, int bottom)
{
    /*
     * Algorithm to calculate the clipped line.
     * It's like clip_aaline, but for integer coordinate endpoints.
     */
    int x1 = segment[0];
    int y1 = segment[1];
    int x2 = segment[2];
    int y2 = segment[3];
    int code1, code2;
    int swaptmp;
    float m; /*slope*/

    while (1) {
        code1 = encode(x1, y1, left, top, right, bottom);
        code2 = encode(x2, y2, left, top, right, bottom);
        if (ACCEPT(code1, code2)) {
            segment[0] = x1;
            segment[1] = y1;
            segment[2] = x2;
            segment[3] = y2;
            return 1;
        }
        else if (REJECT(code1, code2))
            return 0;
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

static Uint32
get_pixel_32(Uint8 *pixels, SDL_PixelFormat *format)
{
    switch (format->BytesPerPixel) {
        case 4:
            return *((Uint32 *)pixels);
        case 3:
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
            return *pixels | *(pixels+1) << 8 | *(pixels+2) << 16;
#else
            return *pixels << 16 | *(pixels + 1) << 8 | *(pixels + 2);
#endif
        case 2:
            return *((Uint16 *)pixels);
        case 1:
            return *pixels;
    }
    return 0;
}

static void
set_pixel_32(Uint8 *pixels, SDL_PixelFormat *format, Uint32 pixel)
{
    switch (format->BytesPerPixel) {
        case 4:
            *(Uint32 *)pixels = pixel;
            break;
        case 3:
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
            *(Uint16*)pixels = pixel;
            pixels[2] = pixel >> 16;
#else
            pixels[2] = pixel;
            pixels[1] = pixel >> 8;
            pixels[0] = pixel >> 16;
#endif
            break;
        case 2:
            *(Uint16 *)pixels = pixel;
            break;
        case 1:
            *pixels = pixel;
            break;
    }
}

static void
draw_pixel_blended_32(Uint8 *pixels, Uint8 *colors, float br,
                      SDL_PixelFormat *format)
{
    Uint8 pixel32[4];
    SDL_GetRGBA(get_pixel_32(pixels, format), format, &pixel32[0], &pixel32[1],
                &pixel32[2], &pixel32[3]);
    *(Uint32 *)pixel32 =
        SDL_MapRGBA(format, br * colors[0] + (1 - br) * pixel32[0],
                    br * colors[1] + (1 - br) * pixel32[1],
                    br * colors[2] + (1 - br) * pixel32[2],
                    br * colors[3] + (1 - br) * pixel32[3]);
    set_pixel_32(pixels, format, *(Uint32 *)pixel32);
}

#define DRAWPIX32(pixels, colorptr, br, blend)                              \
    {                                                                       \
        if (blend)                                                          \
            draw_pixel_blended_32(pixels, colorptr, br, surf->format);      \
        else {                                                              \
            set_pixel_32(                                                   \
                pixels, surf->format,                                       \
                SDL_MapRGBA(surf->format, br *colorptr[0], br *colorptr[1], \
                            br *colorptr[2], br *colorptr[3]));             \
        }                                                                   \
    }

/* Adapted from http://freespace.virgin.net/hugo.elias/graphics/x_wuline.htm */
static void
draw_aaline(SDL_Surface *surf, Uint32 color, float from_x, float from_y, float to_x,
           float to_y, int blend)
{
    float slope, dx, dy;
    float xgap, ygap, pt_x, pt_y, xf, yf;
    float brightness1, brightness2;
    float swaptmp;
    int x, y, ifrom_x, ito_x, ifrom_y, ito_y;
    int pixx, pixy;
    Uint8 colorptr[4];

    Uint8 *pixel;
    Uint8 *surf_pmap = (Uint8 *)surf->pixels;
    SDL_GetRGBA(color, surf->format, &colorptr[0], &colorptr[1], &colorptr[2],
                &colorptr[3]);
    if (!blend)
        colorptr[3] = 255;

    pixx = surf->format->BytesPerPixel;
    pixy = surf->pitch;

    dx = to_x - from_x;
    dy = to_y - from_y;

    if (dx == 0 && dy == 0) {
        /* Single point. Due to the nature of the aaline clipping, this
         * is less exact than the normal line. */
        set_at(surf, from_x, from_y, color);
        return;
    }

    if (fabs(dx) > fabs(dy)) {
        if (from_x > to_x) {
            SWAP(from_x, to_x, swaptmp)
            SWAP(from_y, to_y, swaptmp)
            dx = -dx;
            dy = -dy;
        }
        slope = dy / dx;
        // 1. Draw start of the segment
        pt_x = trunc(from_x) + 0.5; /* This makes more sense than trunc(from_x+0.5) */
        pt_y = from_y + slope * (pt_x - from_x);
        xgap = INVFRAC(from_x);
        ifrom_x = (int)pt_x;
        ifrom_y = (int)pt_y;
        yf = pt_y + slope;
        brightness1 = INVFRAC(pt_y) * xgap;
        brightness2 = FRAC(pt_y) * xgap;
        pixel = surf_pmap + pixx * ifrom_x + pixy * ifrom_y;
        DRAWPIX32(pixel, colorptr, brightness1, blend)
        pixel += pixy;
        DRAWPIX32(pixel, colorptr, brightness2, blend)
        // 2. Draw end of the segment
        pt_x = trunc(to_x) + 0.5;
        pt_y = to_y + slope * (pt_x - to_x);
        xgap = FRAC(to_x); /* this also differs from Hugo's description. */
        ito_x = (int)pt_x;
        ito_y = (int)pt_y;
        brightness1 = INVFRAC(pt_y) * xgap;
        brightness2 = FRAC(pt_y) * xgap;
        pixel = surf_pmap + pixx * ito_x + pixy * ito_y;
        DRAWPIX32(pixel, colorptr, brightness1, blend)
        pixel += pixy;
        DRAWPIX32(pixel, colorptr, brightness2, blend)
        // 3. loop for other points
        for (x = ifrom_x + 1; x < ito_x; ++x) {
            brightness1 = INVFRAC(yf);
            brightness2 = FRAC(yf);
            pixel = surf_pmap + pixx * x + pixy * (int)yf;
            DRAWPIX32(pixel, colorptr, brightness1, blend)
            pixel += pixy;
            DRAWPIX32(pixel, colorptr, brightness2, blend)
            yf += slope;
        }
    }
    else {
        if (from_y > to_y) {
            SWAP(from_x, to_x, swaptmp)
            SWAP(from_y, to_y, swaptmp)
            dx = -dx;
            dy = -dy;
        }
        slope = dx / dy;
        // 1. Draw start of the segment
        pt_y = trunc(from_y) + 0.5; /* This makes more sense than trunc(from_x+0.5) */
        pt_x = from_x + slope * (pt_y - from_y);
        ygap = INVFRAC(from_y);
        ifrom_y = (int)pt_y;
        ifrom_x = (int)pt_x;
        xf = pt_x + slope;
        brightness1 = INVFRAC(pt_x) * ygap;
        brightness2 = FRAC(pt_x) * ygap;
        pixel = surf_pmap + pixx * ifrom_x + pixy * ifrom_y;
        DRAWPIX32(pixel, colorptr, brightness1, blend)
        pixel += pixx;
        DRAWPIX32(pixel, colorptr, brightness2, blend)
        // 2. Draw end of the segment
        pt_y = trunc(to_y) + 0.5;
        pt_x = to_x + slope * (pt_y - to_y);
        ygap = FRAC(to_y);
        ito_y = (int)pt_y;
        ito_x = (int)pt_x;
        brightness1 = INVFRAC(pt_x) * ygap;
        brightness2 = FRAC(pt_x) * ygap;
        pixel = surf_pmap + pixx * ito_x + pixy * ito_y;
        DRAWPIX32(pixel, colorptr, brightness1, blend)
        pixel += pixx;
        DRAWPIX32(pixel, colorptr, brightness2, blend)
        // 3. loop for other points
        for (y = ifrom_y + 1; y < ito_y; ++y) {
            brightness1 = INVFRAC(xf);
            brightness2 = FRAC(xf);
            pixel = surf_pmap + pixx * (int)xf + pixy * y;
            DRAWPIX32(pixel, colorptr, brightness1, blend)
            pixel += pixx;
            DRAWPIX32(pixel, colorptr, brightness2, blend)
            xf += slope;
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
