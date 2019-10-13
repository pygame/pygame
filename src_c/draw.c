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

#include <float.h>

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

/* Float versions.
 *
 * See comment above about some C libraries lacking the trunc function. The
 * functions truncf, floorf, and ceilf could also be missing as they were
 * added in C99 as well. Just use the double functions and cast to a float.
 */
#if (!defined(__STDC_VERSION__) || __STDC_VERSION__ < 199901L) && \
    !defined(truncf)
#define truncf(x) ((float)(((x) >= 0.0f) ? (floor(x)) : (ceil(x))))
#endif

#define FRAC_FLT(z) ((z)-truncf(z))
#define INVFRAC_FLT(z) (1 - FRAC_FLT(z))

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int
clip_and_draw_line(SDL_Surface *surf, Uint32 color, int *pts);
static int
clip_and_draw_aaline(SDL_Surface *surf, SDL_Rect *rect, Uint32 color,
                     float *pts, int blend);
static int
clip_and_draw_line_width(SDL_Surface *surf, Uint32 color,
                         int width, int *pts);
static int
clip_aaline(float *pts, int left, int top, int right, int bottom);
static int
drawline(SDL_Surface *surf, int* points, Uint32 color);
static void
draw_aaline(SDL_Surface *surf, Uint32 color, float startx, float starty,
           float endx, float endy, int blend);
static int
drawhorzline(SDL_Surface *surf, Uint32 color, int startx, int starty,
             int endx);
static int
drawvertline(SDL_Surface *surf, Uint32 color, int x1, int y1, int y2);
static void
draw_arc(SDL_Surface *dst, int x, int y, int radius1, int radius2,
         double angle_start, double angle_stop, Uint32 color);
static void
draw_circle_bresenham(SDL_Surface *dst, int x0, int y0, int radius, int thickness, Uint32 color);
static void
draw_circle_filled(SDL_Surface *dst, int x0, int y0, int radius, Uint32 color);
static void
draw_ellipse(SDL_Surface *dst, int x, int y, int width, int height, int solid,
             Uint32 color);
static void
draw_fillpoly(SDL_Surface *dst, int *vx, int *vy, Py_ssize_t n, Uint32 color);

// validation of a draw color
#define CHECK_LOAD_COLOR(colorobj)                                         \
    if (PyInt_Check(colorobj))                                             \
        color = (Uint32)PyInt_AsLong(colorobj);                            \
    else if (pg_RGBAFromColorObj(colorobj, rgba))                          \
        color =                                                            \
            SDL_MapRGBA(surf->format, rgba[0], rgba[1], rgba[2], rgba[3]); \
    else                                                                   \
        return RAISE(PyExc_TypeError, "invalid color argument");

/* Draws an antialiased line on the given surface.
 *
 * Returns a Rect bounding the drawn area.
 */
static PyObject *
aaline(PyObject *self, PyObject *arg, PyObject *kwargs)
{
    PyObject *surfobj = NULL, *colorobj = NULL, *start = NULL, *end = NULL;
    SDL_Surface *surf = NULL;
    float startx, starty, endx, endy;
    int top, left, bottom, right, anydraw;
    int blend = 1; /* Default blend. */
    float pts[4];
    Uint8 rgba[4];
    Uint32 color;
    static char *keywords[] = {"surface", "color", "start_pos",
                               "end_pos", "blend", NULL};

    if (!PyArg_ParseTupleAndKeywords(arg, kwargs, "O!OOO|i", keywords,
                                     &pgSurface_Type, &surfobj, &colorobj,
                                     &start, &end, &blend)) {
        return NULL; /* Exception already set. */
    }

    surf = pgSurface_AsSurface(surfobj);

    if (surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4) {
        return PyErr_Format(PyExc_ValueError,
                            "unsupported surface bit depth (%d) for drawing",
                            surf->format->BytesPerPixel);
    }

    CHECK_LOAD_COLOR(colorobj)

    if (!pg_TwoFloatsFromObj(start, &startx, &starty)) {
        return RAISE(PyExc_TypeError, "invalid start_pos argument");
    }

    if (!pg_TwoFloatsFromObj(end, &endx, &endy)) {
        return RAISE(PyExc_TypeError, "invalid end_pos argument");
    }

    if (!pgSurface_Lock(surfobj)) {
        return RAISE(PyExc_RuntimeError, "error locking surface");
    }

    pts[0] = startx;
    pts[1] = starty;
    pts[2] = endx;
    pts[3] = endy;
    anydraw = clip_and_draw_aaline(surf, &surf->clip_rect, color, pts, blend);

    if (!pgSurface_Unlock(surfobj)) {
        return RAISE(PyExc_RuntimeError, "error unlocking surface");
    }

    /* Compute return rect. */
    if (!anydraw) {
        return pgRect_New4((int)startx, (int)starty, 0, 0);
    }

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

/* Draws a line on the given surface.
 *
 * Returns a Rect bounding the drawn area.
 */
static PyObject *
line(PyObject *self, PyObject *arg, PyObject *kwargs)
{
    PyObject *surfobj = NULL, *colorobj = NULL, *start = NULL, *end = NULL;
    SDL_Surface *surf = NULL;
    int startx, starty, endx, endy, anydraw;
    int pts[4];
    Uint8 rgba[4];
    Uint32 color;
    int width = 1; /* Default width. */
    static char *keywords[] = {"surface", "color", "start_pos",
                               "end_pos", "width", NULL};

    if (!PyArg_ParseTupleAndKeywords(arg, kwargs, "O!OOO|i", keywords,
                                     &pgSurface_Type, &surfobj, &colorobj,
                                     &start, &end, &width)) {
        return NULL; /* Exception already set. */
    }

    surf = pgSurface_AsSurface(surfobj);

    if (surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4) {
        return PyErr_Format(PyExc_ValueError,
                            "unsupported surface bit depth (%d) for drawing",
                            surf->format->BytesPerPixel);
    }

    CHECK_LOAD_COLOR(colorobj)

    if (!pg_TwoIntsFromObj(start, &startx, &starty)) {
        return RAISE(PyExc_TypeError, "invalid start_pos argument");
    }

    if (!pg_TwoIntsFromObj(end, &endx, &endy)) {
        return RAISE(PyExc_TypeError, "invalid end_pos argument");
    }

    if (width < 1) {
        return pgRect_New4(startx, starty, 0, 0);
    }

    if (!pgSurface_Lock(surfobj)) {
        return RAISE(PyExc_RuntimeError, "error locking surface");
    }

    pts[0] = startx;
    pts[1] = starty;
    pts[2] = endx;
    pts[3] = endy;
    anydraw =
        clip_and_draw_line_width(surf, color, width, pts);

    if (!pgSurface_Unlock(surfobj)) {
        return RAISE(PyExc_RuntimeError, "error unlocking surface");
    }

    if (!anydraw) {
        return pgRect_New4(startx, starty, 0, 0);
    }

    /* The pts array was updated with the top left and bottom right corners
     * of the bounding rect: {left, top, right, bottom}. That is used to
     * construct the rect bounding the changed area. */
    return pgRect_New4(pts[0], pts[1], pts[2] - pts[0] + 1,
                       pts[3] - pts[1] + 1);
}

/* Draws a series of antialiased lines on the given surface.
 *
 * Returns a Rect bounding the drawn area.
 */
static PyObject *
aalines(PyObject *self, PyObject *arg, PyObject *kwargs)
{
    PyObject *surfobj = NULL, *colorobj = NULL, *closedobj = NULL;
    PyObject *points = NULL, *item = NULL;
    SDL_Surface *surf = NULL;
    Uint32 color;
    Uint8 rgba[4];
    float pts[4];
    float *xlist, *ylist;
    float x, y;
    float top = FLT_MAX, left = FLT_MAX;
    float bottom = FLT_MIN, right = FLT_MIN;
    int result;
    int closed = 0; /* Default closed. */
    int blend = 1;  /* Default blend. */
    Py_ssize_t loop, length;
    static char *keywords[] = {"surface", "color", "closed",
                               "points",  "blend", NULL};

    if (!PyArg_ParseTupleAndKeywords(arg, kwargs, "O!OOO|i", keywords,
                                     &pgSurface_Type, &surfobj, &colorobj,
                                     &closedobj, &points, &blend)) {
        return NULL; /* Exception already set. */
    }

    surf = pgSurface_AsSurface(surfobj);

    if (surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4) {
        return PyErr_Format(PyExc_ValueError,
                            "unsupported surface bit depth (%d) for drawing",
                            surf->format->BytesPerPixel);
    }

    CHECK_LOAD_COLOR(colorobj)

    closed = PyObject_IsTrue(closedobj);

    if (-1 == closed) {
        return RAISE(PyExc_TypeError, "closed argument is invalid");
    }

    if (!PySequence_Check(points)) {
        return RAISE(PyExc_TypeError,
                     "points argument must be a sequence of number pairs");
    }

    length = PySequence_Length(points);

    if (length < 2) {
        return RAISE(PyExc_ValueError,
                     "points argument must contain 2 or more points");
    }

    xlist = PyMem_New(float, length);
    ylist = PyMem_New(float, length);

    if (NULL == xlist || NULL == ylist) {
        return RAISE(PyExc_MemoryError,
                     "cannot allocate memory to draw aalines");
    }

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
        return RAISE(PyExc_RuntimeError, "error locking surface");
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

    if (!pgSurface_Unlock(surfobj)) {
        return RAISE(PyExc_RuntimeError, "error unlocking surface");
    }

    /* Compute return rect. */
    return pgRect_New4((int)left, (int)top, (int)(right - left + 2),
                       (int)(bottom - top + 2));
}

/* Draws a series of lines on the given surface.
 *
 * Returns a Rect bounding the drawn area.
 */
static PyObject *
lines(PyObject *self, PyObject *arg, PyObject *kwargs)
{
    PyObject *surfobj = NULL, *colorobj = NULL, *closedobj = NULL;
    PyObject *points = NULL, *item = NULL;
    SDL_Surface *surf = NULL;
    Uint32 color;
    Uint8 rgba[4];
    int pts[4];
    int x, y, closed, result;
    int top = INT_MAX, left = INT_MAX;
    int bottom = INT_MIN, right = INT_MIN;
    int *xlist = NULL, *ylist = NULL;
    int width = 1; /* Default width. */
    Py_ssize_t loop, length;
    static char *keywords[] = {"surface", "color", "closed",
                               "points",  "width", NULL};

    if (!PyArg_ParseTupleAndKeywords(arg, kwargs, "O!OOO|i", keywords,
                                     &pgSurface_Type, &surfobj, &colorobj,
                                     &closedobj, &points, &width)) {
        return NULL; /* Exception already set. */
    }

    surf = pgSurface_AsSurface(surfobj);

    if (surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4) {
        return PyErr_Format(PyExc_ValueError,
                            "unsupported surface bit depth (%d) for drawing",
                            surf->format->BytesPerPixel);
    }

    CHECK_LOAD_COLOR(colorobj)

    closed = PyObject_IsTrue(closedobj);

    if (-1 == closed) {
        return RAISE(PyExc_TypeError, "closed argument is invalid");
    }

    if (!PySequence_Check(points)) {
        return RAISE(PyExc_TypeError,
                     "points argument must be a sequence of number pairs");
    }

    length = PySequence_Length(points);

    if (length < 2) {
        return RAISE(PyExc_ValueError,
                     "points argument must contain 2 or more points");
    }

    xlist = PyMem_New(int, length);
    ylist = PyMem_New(int, length);

    if (NULL == xlist || NULL == ylist) {
        return RAISE(PyExc_MemoryError,
                     "cannot allocate memory to draw lines");
    }

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
    }

    x = xlist[0];
    y = ylist[0];

    if (width < 1) {
        PyMem_Del(xlist);
        PyMem_Del(ylist);
        return pgRect_New4(x, y, 0, 0);
    }

    if (!pgSurface_Lock(surfobj)) {
        PyMem_Del(xlist);
        PyMem_Del(ylist);
        return RAISE(PyExc_RuntimeError, "error locking surface");
    }

    for (loop = 1; loop < length; ++loop) {
        pts[0] = xlist[loop - 1];
        pts[1] = ylist[loop - 1];
        pts[2] = xlist[loop];
        pts[3] = ylist[loop];

        if (clip_and_draw_line_width(surf, color, width,
                                     pts)) {
            /* The pts array was updated with the top left and bottom right
             * corners of the bounding box: {left, top, right, bottom}. */
            left = MIN(pts[0], left);
            top = MIN(pts[1], top);
            right = MAX(pts[2], right);
            bottom = MAX(pts[3], bottom);
        }
    }

    if (closed && length > 2) {
        pts[0] = xlist[length - 1];
        pts[1] = ylist[length - 1];
        pts[2] = xlist[0];
        pts[3] = ylist[0];

        if (clip_and_draw_line_width(surf, color, width,
                                     pts)) {
            left = MIN(pts[0], left);
            top = MIN(pts[1], top);
            right = MAX(pts[2], right);
            bottom = MAX(pts[3], bottom);
        }
    }

    PyMem_Del(xlist);
    PyMem_Del(ylist);

    if (!pgSurface_Unlock(surfobj)) {
        return RAISE(PyExc_RuntimeError, "error unlocking surface");
    }

    if (INT_MAX == left) {
        /* Nothing was drawn. */
        return pgRect_New4(x, y, 0, 0);
    }

    /* Compute return rect. */
    return pgRect_New4(left, top, right - left + 1, bottom - top + 1);
}

static PyObject *
arc(PyObject *self, PyObject *arg, PyObject *kwargs)
{
    PyObject *surfobj = NULL, *colorobj = NULL, *rectobj = NULL;
    GAME_Rect *rect = NULL, temp;
    SDL_Surface *surf = NULL;
    Uint8 rgba[4];
    Uint32 color;
    int loop, t, l, b, r;
    int width = 1; /* Default width. */
    double angle_start, angle_stop;
    static char *keywords[] = {"surface", "color", "rect", "start_angle",
                               "stop_angle", "width", NULL};

    if (!PyArg_ParseTupleAndKeywords(arg, kwargs, "O!OOdd|i", keywords,
                                     &pgSurface_Type, &surfobj, &colorobj,
                                     &rectobj, &angle_start, &angle_stop,
                                     &width)) {
        return NULL; /* Exception already set. */
    }

    rect = pgRect_FromObject(rectobj, &temp);

    if (!rect) {
        return RAISE(PyExc_TypeError, "rect argument is invalid");
    }

    surf = pgSurface_AsSurface(surfobj);

    if (surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4) {
        return PyErr_Format(PyExc_ValueError,
                            "unsupported surface bit depth (%d) for drawing",
                            surf->format->BytesPerPixel);
    }

    CHECK_LOAD_COLOR(colorobj)

    if (width < 0) {
        return pgRect_New4(rect->x, rect->y, 0, 0);
    }

    if (width > rect->w / 2 || width > rect->h / 2) {
        width = MAX(rect->w / 2, rect->h / 2);
    }

    if (angle_stop < angle_start) {
        // Angle is in radians
        angle_stop += 2 * M_PI;
    }

    if (!pgSurface_Lock(surfobj)) {
        return RAISE(PyExc_RuntimeError, "error locking surface");
    }

    width = MIN(width, MIN(rect->w, rect->h) / 2);

    for (loop = 0; loop < width; ++loop) {
        draw_arc(surf, rect->x + rect->w / 2, rect->y + rect->h / 2,
                 rect->w / 2 - loop, rect->h / 2 - loop, angle_start,
                 angle_stop, color);
    }

    if (!pgSurface_Unlock(surfobj)) {
        return RAISE(PyExc_RuntimeError, "error unlocking surface");
    }

    l = MAX(rect->x, surf->clip_rect.x);
    t = MAX(rect->y, surf->clip_rect.y);
    r = MIN(rect->x + rect->w, surf->clip_rect.x + surf->clip_rect.w);
    b = MIN(rect->y + rect->h, surf->clip_rect.y + surf->clip_rect.h);
    return pgRect_New4(l, t, MAX(r - l, 0), MAX(b - t, 0));
}

static PyObject *
ellipse(PyObject *self, PyObject *arg, PyObject *kwargs)
{
    PyObject *surfobj = NULL, *colorobj = NULL, *rectobj = NULL;
    GAME_Rect *rect = NULL, temp;
    SDL_Surface *surf = NULL;
    Uint8 rgba[4];
    Uint32 color;
    int loop, t, l, b, r;
    int width = 0;  /* Default width. */
    static char *keywords[] = {"surface", "color", "rect", "width", NULL};

    if (!PyArg_ParseTupleAndKeywords(arg, kwargs, "O!OO|i", keywords,
                                     &pgSurface_Type, &surfobj, &colorobj,
                                     &rectobj, &width)) {
        return NULL; /* Exception already set. */
    }

    rect = pgRect_FromObject(rectobj, &temp);

    if (!rect) {
        return RAISE(PyExc_TypeError, "rect argument is invalid");
    }

    surf = pgSurface_AsSurface(surfobj);

    if (surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4) {
        return PyErr_Format(PyExc_ValueError,
                            "unsupported surface bit depth (%d) for drawing",
                            surf->format->BytesPerPixel);
    }

    CHECK_LOAD_COLOR(colorobj)

    if (width < 0) {
        return pgRect_New4(rect->x, rect->y, 0, 0);
    }

    if (width > rect->w / 2 || width > rect->h / 2) {
        width = MAX(rect->w / 2, rect->h / 2);
    }

    if (!pgSurface_Lock(surfobj)) {
        return RAISE(PyExc_RuntimeError, "error locking surface");
    }

    if (!width) {
        /* Draw a filled ellipse. */
        draw_ellipse(surf, rect->x + rect->w / 2, rect->y + rect->h / 2,
                     rect->w, rect->h, 1, color);
    }
    else {
        width = MIN(width, MIN(rect->w, rect->h) / 2);
        for (loop = 0; loop < width; ++loop) {
            draw_ellipse(surf, rect->x + rect->w / 2, rect->y + rect->h / 2,
                         rect->w - loop, rect->h - loop, 0, color);
        }
    }

    if (!pgSurface_Unlock(surfobj)) {
        return RAISE(PyExc_RuntimeError, "error unlocking surface");
    }

    l = MAX(rect->x, surf->clip_rect.x);
    t = MAX(rect->y, surf->clip_rect.y);
    r = MIN(rect->x + rect->w, surf->clip_rect.x + surf->clip_rect.w);
    b = MIN(rect->y + rect->h, surf->clip_rect.y + surf->clip_rect.h);
    return pgRect_New4(l, t, MAX(r - l, 0), MAX(b - t, 0));
}

static PyObject *
circle(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *surfobj = NULL, *colorobj = NULL;
    SDL_Surface *surf = NULL;
    Uint8 rgba[4];
    Uint32 color;
    PyObject *posobj, *radiusobj;
    int posx, posy, radius, t, l, b, r;
    int width = 0; /* Default width. */
    static char *keywords[] = {"surface", "color", "center",
                               "radius",  "width", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!OOO|i", keywords,
                          &pgSurface_Type, &surfobj,
                          &colorobj,
                          &posobj,
                          &radiusobj, &width))
        return NULL; /* Exception already set. */

    if (!pg_TwoIntsFromObj(posobj, &posx, &posy)) {
        PyErr_SetString(PyExc_TypeError,
                        "center argument must be a pair of numbers");
        return 0;
    }

    if (!pg_IntFromObj (radiusobj, &radius)) {
        PyErr_SetString(PyExc_TypeError,
                        "radius argument must be a number");
        return 0;
    }

    surf = pgSurface_AsSurface(surfobj);

    if (surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4) {
        return PyErr_Format(PyExc_ValueError,
                            "unsupported surface bit depth (%d) for drawing",
                            surf->format->BytesPerPixel);
    }

    CHECK_LOAD_COLOR(colorobj)

    if (radius < 1 || width < 0) {
        return pgRect_New4(posx, posy, 0, 0);
    }

    if (width > radius) {
        width = radius;
    }

    if (!pgSurface_Lock(surfobj)) {
        return RAISE(PyExc_RuntimeError, "error locking surface");
    }

    if (!width) {
        //draw_ellipse(surf, posx, posy, radius * 2, radius * 2, 1, color);
        draw_circle_filled(surf, posx, posy,
                              radius, color);
    } else {
        draw_circle_bresenham(surf, posx, posy,
                              radius, width, color);
    }

    if (!pgSurface_Unlock(surfobj)) {
        return RAISE(PyExc_RuntimeError, "error unlocking surface");
    }

    l = MAX(posx - radius, surf->clip_rect.x);
    t = MAX(posy - radius, surf->clip_rect.y);
    r = MIN(posx + radius, surf->clip_rect.x + surf->clip_rect.w);
    b = MIN(posy + radius, surf->clip_rect.y + surf->clip_rect.h);

    return pgRect_New4(l, t, MAX(r - l, 0), MAX(b - t, 0));
}

static PyObject *
polygon(PyObject *self, PyObject *arg, PyObject *kwargs)
{
    PyObject *surfobj = NULL, *colorobj = NULL, *points = NULL, *item = NULL;
    SDL_Surface *surf = NULL;
    Uint8 rgba[4];
    Uint32 color;
    int *xlist = NULL, *ylist = NULL;
    int width = 0; /* Default width. */
    int top = INT_MAX, left = INT_MAX;
    int bottom = INT_MIN, right = INT_MIN;
    int x, y, result;
    Py_ssize_t loop, length;
    static char *keywords[] = {"surface", "color", "points", "width", NULL};

    if (!PyArg_ParseTupleAndKeywords(arg, kwargs, "O!OO|i", keywords,
                                     &pgSurface_Type, &surfobj, &colorobj,
                                     &points, &width)) {
        return NULL; /* Exception already set. */
    }

    if (width) {
        PyObject *ret = NULL;
        PyObject *args =
            Py_BuildValue("(OOiOi)", surfobj, colorobj, 1, points, width);

        if (!args) {
            return NULL; /* Exception already set. */
        }

        ret = lines(NULL, args, NULL);
        Py_DECREF(args);
        return ret;
    }

    surf = pgSurface_AsSurface(surfobj);

    if (surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4) {
        return PyErr_Format(PyExc_ValueError,
                            "unsupported surface bit depth (%d) for drawing",
                            surf->format->BytesPerPixel);
    }

    CHECK_LOAD_COLOR(colorobj)

    if (!PySequence_Check(points)) {
        return RAISE(PyExc_TypeError,
                     "points argument must be a sequence of number pairs");
    }

    length = PySequence_Length(points);

    if (length < 3) {
        return RAISE(PyExc_ValueError,
                     "points argument must contain more than 2 points");
    }

    xlist = PyMem_New(int, length);
    ylist = PyMem_New(int, length);

    if (NULL == xlist || NULL == ylist) {
        return RAISE(PyExc_MemoryError,
                     "cannot allocate memory to draw polygon");
    }

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
        return RAISE(PyExc_RuntimeError, "error locking surface");
    }

    draw_fillpoly(surf, xlist, ylist, length, color);
    PyMem_Del(xlist);
    PyMem_Del(ylist);

    if (!pgSurface_Unlock(surfobj)) {
        return RAISE(PyExc_RuntimeError, "error unlocking surface");
    }

    left = MAX(left, surf->clip_rect.x);
    top = MAX(top, surf->clip_rect.y);
    right = MIN(right, surf->clip_rect.x + surf->clip_rect.w);
    bottom = MIN(bottom, surf->clip_rect.y + surf->clip_rect.h);
    return pgRect_New4(left, top, right - left + 1, bottom - top + 1);
}

static PyObject *
rect(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *surfobj = NULL, *colorobj = NULL, *rectobj = NULL;
    PyObject *points = NULL, *poly_args = NULL, *ret = NULL;
    GAME_Rect *rect = NULL, temp;
    int t, l, b, r;
    int width = 0; /* Default width. */
    static char *keywords[] = {"surface", "color", "rect", "width", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!OO|i", keywords,
                                     &pgSurface_Type, &surfobj, &colorobj,
                                     &rectobj, &width)) {
        return NULL; /* Exception already set. */
    }

    if (!(rect = pgRect_FromObject(rectobj, &temp))) {
        return RAISE(PyExc_TypeError, "rect argument is invalid");
    }

    l = rect->x;
    r = rect->x + rect->w - 1;
    t = rect->y;
    b = rect->y + rect->h - 1;

    points = Py_BuildValue("((ii)(ii)(ii)(ii))", l, t, r, t, r, b, l, b);
    poly_args = Py_BuildValue("(OONi)", surfobj, colorobj, points, width);
    if (NULL == poly_args) {
        return NULL; /* Exception already set. */
    }

    ret = polygon(NULL, poly_args, NULL);
    Py_DECREF(poly_args);
    return ret;
}

/*internal drawing tools*/

static int
clip_and_draw_aaline(SDL_Surface *surf, SDL_Rect *rect, Uint32 color,
                     float *pts, int blend)
{
    if (!clip_aaline(pts, rect->x, rect->y, rect->x + rect->w - 1,
                     rect->y + rect->h - 1))
        return 0;

    draw_aaline(surf, color, pts[0], pts[1], pts[2], pts[3], blend);
    return 1;
}

/* This function is used to draw a line that has the width of 1. First it checks
 * is the line horizontal or vertical and draws it (because the algorithm for drawing
 * straight line is faster). This function also draws line from any 2 points on the surface
 * Parameters are surface where line will be drawn. pts it the array of 4 ints that contains
 * the ending points of the line. PTS WILL BE MODIFIED AFTER THIS FUNCTION (for horizontal
 * and vertical lines inside of this function, and for other lines in drawline function)
 * so be careful when pasing points through multiple functions (at the end of the function
 * it will contain cordinates of the bounding_rect). Function returns 1 if it draws
 * anything on the screen otherwise it returns 0
 */
static int
clip_and_draw_line(SDL_Surface *surf, Uint32 color, int *pts)
{
    if (pts[1] == pts[3] && drawhorzline(surf, color, pts[0], pts[1], pts[2])) {
        int old_pts_zero = pts[0];
        *(pts) = MAX(0, MIN(pts[0], pts[2]));
        if ((pts[2] >= surf->clip_rect.w)||(old_pts_zero >= surf->clip_rect.w))
            *(pts+2) = surf->clip_rect.w - 1;
        else {
            *(pts+2) = MIN(surf->clip_rect.w, MAX(old_pts_zero, pts[2]));
        }
        return 1;
    }
    else if (pts[0] == pts[2] && drawvertline(surf, color, pts[0], pts[1], pts[3])) {
        int old_pts_one = pts[1];
        *(pts+1) = MAX(0, MIN(pts[1], pts[3]));
        if ((pts[3] >= surf->clip_rect.h)||(old_pts_one >= surf->clip_rect.h))
            *(pts+3) = surf->clip_rect.h - 1;
        else {
            *(pts+3) = MIN(surf->clip_rect.h, MAX(old_pts_one, pts[3]));
        }
        return 1;
    }
    else if (drawline(surf, pts, color)) {
        return 1;
    }
    else {
        return 0;
    }
}

/* This is an internal helper function.
 *
 * This function draws a line that is clipped by the given rect. To draw thick
 * lines (width > 1), multiple parallel lines are drawn.
 *
 * Params:
 *     surf - pointer to surface to draw on
 *     color - color of line to draw
 *     width - width/thickness of line to draw (expected to be > 0)
 *     pts - array of 4 points which are the endpoints of the line to
 *         draw: {x0, y0, x1, y1}
 *
 * Returns:
 *     int - 1 indicates that something was drawn on the surface
 *           0 indicates that nothing was drawn
 *
 *     If something was drawn, the 'pts' parameter is changed to contain the
 *     min/max x/y values of the pixels changed: {xmin, ymin, xmax, ymax}.
 *     These points represent the minimum bounding box of the affected area.
 *     The top left corner is xmin, ymin and the bottom right corner is
 *     xmax, ymax.
 */
static int
clip_and_draw_line_width(SDL_Surface *surf, Uint32 color,
                         int width, int *pts)
{
    int xinc = 0, yinc = 0;
    int bounding_rect[4];
    int original_values[4];
    int anydrawn = 0;
    int loop;
    bounding_rect[0] = INT_MAX;
    bounding_rect[1] = INT_MAX;
    bounding_rect[2] = 0;
    bounding_rect[3] = 0;
    memcpy(original_values, pts, sizeof(int) * 4);
    /* Decide which direction to grow (width/thickness). */
    if (abs(pts[0] - pts[2]) > abs(pts[1] - pts[3])) {
        /* The line's thickness will be in the y direction. The left/right
         * ends of the line will be flat. */
        yinc = 1;
    }
    else {
        /* The line's thickness will be in the x direction. The top/bottom
         * ends of the line will be flat. */
        xinc = 1;
    }
    /* Draw central line and calculate bounding rect of the line (just copy values
     * already stored in pts, possible that this doesn't need if/else) */
    if (clip_and_draw_line(surf, color, pts)) {
        anydrawn = 1;
        if (pts[0] > pts[2]) {
            bounding_rect[0] = pts[2];
            bounding_rect[2] = pts[0];
        }
        else {
            bounding_rect[0] = pts[0];
            bounding_rect[2] = pts[2];
        }

        if (pts[1] > pts[3]) {
            bounding_rect[1] = pts[3];
            bounding_rect[3] = pts[1];
        }
        else {
            bounding_rect[1] = pts[1];
            bounding_rect[3] = pts[3];
        }
    }
    /* If width is > 1 start drawing lines connected to the central line, first try to draw
     * to the right / down, and then to the left / right. Meanwhile every time calculate rect
     * (Possible that it can be improved to not calculate it every loop) */
    if (width != 1) {
        for (loop = 1; loop < width; loop += 2) {
            pts[0] = original_values[0] + xinc * (loop / 2 + 1);
            pts[1] = original_values[1] + yinc * (loop / 2 + 1);
            pts[2] = original_values[2] + xinc * (loop / 2 + 1);
            pts[3] = original_values[3] + yinc * (loop / 2 + 1);
            if (clip_and_draw_line(surf, color, pts)) {
                anydrawn = 1;
                bounding_rect[0] = MIN(bounding_rect[0], MIN(pts[0], pts[2]));
                bounding_rect[1] = MIN(bounding_rect[1], MIN(pts[1], pts[3]));
                bounding_rect[2] = MAX(bounding_rect[2], MAX(pts[0], pts[2]));
                bounding_rect[3] = MAX(bounding_rect[3], MAX(pts[1], pts[3]));
            }
            if (loop + 1 < width) {
                pts[0] = original_values[0] - xinc * (loop / 2 + 1);
                pts[1] = original_values[1] - yinc * (loop / 2 + 1);
                pts[2] = original_values[2] - xinc * (loop / 2 + 1);
                pts[3] = original_values[3] - yinc * (loop / 2 + 1);
                if (clip_and_draw_line(surf, color, pts)) {
                    anydrawn = 1;
                    bounding_rect[0] = MIN(bounding_rect[0], MIN(pts[0], pts[2]));
                    bounding_rect[1] = MIN(bounding_rect[1], MIN(pts[1], pts[3]));
                    bounding_rect[2] = MAX(bounding_rect[2], MAX(pts[0], pts[2]));
                    bounding_rect[3] = MAX(bounding_rect[3], MAX(pts[1], pts[3]));
                }
            }
        }
        /* After you draw rect you don't need pts array any more so it is used to store rect */
        memcpy(pts, bounding_rect, sizeof(int) * 4);
    }
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
        SDL_MapRGBA(format, (Uint8)(br * colors[0] + (1 - br) * pixel32[0]),
                    (Uint8)(br * colors[1] + (1 - br) * pixel32[1]),
                    (Uint8)(br * colors[2] + (1 - br) * pixel32[2]),
                    (Uint8)(br * colors[3] + (1 - br) * pixel32[3]));

    set_pixel_32(pixels, format, *(Uint32 *)pixel32);
}

#define DRAWPIX32(pixels, colorptr, br, blend)                                \
    {                                                                         \
        if (blend)                                                            \
            draw_pixel_blended_32(pixels, colorptr, br, surf->format);        \
        else {                                                                \
            set_pixel_32(pixels, surf->format,                                \
                         SDL_MapRGBA(surf->format, (Uint8)(br * colorptr[0]), \
                                     (Uint8)(br * colorptr[1]),               \
                                     (Uint8)(br * colorptr[2]),               \
                                     (Uint8)(br * colorptr[3])));             \
        }                                                                     \
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
    SDL_Rect *rect = &surf->clip_rect;
    int max_x = rect->x + rect->w - 1;
    int max_y = rect->y + rect->h - 1;

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
        set_at(surf, (int)truncf(from_x), (int)truncf(from_y), color);
        return;
    }

    if (fabs(dx) > fabs(dy)) {
        /* Lines tending to be more horizontal (run > rise) handled here. */
        if (from_x > to_x) {
            SWAP(from_x, to_x, swaptmp)
            SWAP(from_y, to_y, swaptmp)
            dx = -dx;
            dy = -dy;
        }
        slope = dy / dx;

        // 1. Draw start of the segment
        /* This makes more sense than truncf(from_x + 0.5f) */
        pt_x = truncf(from_x) + 0.5f;
        pt_y = from_y + slope * (pt_x - from_x);
        xgap = INVFRAC_FLT(from_x);
        ifrom_x = (int)pt_x;
        ifrom_y = (int)pt_y;
        yf = pt_y + slope;
        brightness1 = INVFRAC_FLT(pt_y) * xgap;

        pixel = surf_pmap + pixx * ifrom_x + pixy * ifrom_y;
        DRAWPIX32(pixel, colorptr, brightness1, blend)

        /* Skip if ifrom_y+1 is not on the surface. */
        if (ifrom_y < max_y) {
            brightness2 = FRAC_FLT(pt_y) * xgap;
            pixel += pixy;
            DRAWPIX32(pixel, colorptr, brightness2, blend)
        }

        // 2. Draw end of the segment
        pt_x = truncf(to_x) + 0.5f;
        pt_y = to_y + slope * (pt_x - to_x);
        xgap = INVFRAC_FLT(to_x);
        ito_x = (int)pt_x;
        ito_y = (int)pt_y;
        brightness1 = INVFRAC_FLT(pt_y) * xgap;

        pixel = surf_pmap + pixx * ito_x + pixy * ito_y;
        DRAWPIX32(pixel, colorptr, brightness1, blend)

        /* Skip if ito_y+1 is not on the surface. */
        if (ito_y < max_y) {
            brightness2 = FRAC_FLT(pt_y) * xgap;
            pixel += pixy;
            DRAWPIX32(pixel, colorptr, brightness2, blend)
        }

        // 3. loop for other points
        for (x = ifrom_x + 1; x < ito_x; ++x) {
            brightness1 = INVFRAC_FLT(yf);
            y = (int)yf;

            pixel = surf_pmap + pixx * x + pixy * y;
            DRAWPIX32(pixel, colorptr, brightness1, blend)

            /* Skip if y+1 is not on the surface. */
            if (y < max_y) {
                brightness2 = FRAC_FLT(yf);
                pixel += pixy;
                DRAWPIX32(pixel, colorptr, brightness2, blend)
            }
            yf += slope;
        }
    }
    else {
        /* Lines tending to be more vertical (rise >= run) handled here. */
        if (from_y > to_y) {
            SWAP(from_x, to_x, swaptmp)
            SWAP(from_y, to_y, swaptmp)
            dx = -dx;
            dy = -dy;
        }
        slope = dx / dy;

        // 1. Draw start of the segment
        /* This makes more sense than truncf(from_x + 0.5f) */
        pt_y = truncf(from_y) + 0.5f;
        pt_x = from_x + slope * (pt_y - from_y);
        ygap = INVFRAC_FLT(from_y);
        ifrom_y = (int)pt_y;
        ifrom_x = (int)pt_x;
        xf = pt_x + slope;
        brightness1 = INVFRAC_FLT(pt_x) * ygap;

        pixel = surf_pmap + pixx * ifrom_x + pixy * ifrom_y;
        DRAWPIX32(pixel, colorptr, brightness1, blend)

        /* Skip if ifrom_x+1 is not on the surface. */
        if (ifrom_x < max_x) {
            brightness2 = FRAC_FLT(pt_x) * ygap;
            pixel += pixx;
            DRAWPIX32(pixel, colorptr, brightness2, blend)
        }

        // 2. Draw end of the segment
        pt_y = truncf(to_y) + 0.5f;
        pt_x = to_x + slope * (pt_y - to_y);
        ygap = INVFRAC_FLT(to_y);
        ito_y = (int)pt_y;
        ito_x = (int)pt_x;
        brightness1 = INVFRAC_FLT(pt_x) * ygap;

        pixel = surf_pmap + pixx * ito_x + pixy * ito_y;
        DRAWPIX32(pixel, colorptr, brightness1, blend)

        /* Skip if ito_x+1 is not on the surface. */
        if (ito_x < max_x) {
            brightness2 = FRAC_FLT(pt_x) * ygap;
            pixel += pixx;
            DRAWPIX32(pixel, colorptr, brightness2, blend)
        }

        // 3. loop for other points
        for (y = ifrom_y + 1; y < ito_y; ++y) {
            x = (int)xf;
            brightness1 = INVFRAC_FLT(xf);

            pixel = surf_pmap + pixx * x + pixy * y;
            DRAWPIX32(pixel, colorptr, brightness1, blend)

            /* Skip if x+1 is not on the surface. */
            if (x < max_x) {
                brightness2 = FRAC_FLT(xf);
                pixel += pixx;
                DRAWPIX32(pixel, colorptr, brightness2, blend)
            }
            xf += slope;
        }
    }
}

/* Algorithm modified from
 * https://stackoverflow.com/questions/11678693/all-cases-covered-bresenhams-line-algorithm */
static int
drawline(SDL_Surface *surf, int* pts, Uint32 color)
{
    int i, numerator;
    int lowest_x = INT_MAX;
    int lowest_y = INT_MAX;
    int highest_x = 0;
    int highest_y = 0;
    int anydraw = 0;
    int x = *(pts);
    int y = *(pts+1);
    int w = *(pts+2) - x;
    int h = *(pts+3) - y;
    int dx1 = 0, dy1 = 0, dx2 = 0, dy2 = 0;
    int longest = abs(w);
    int shortest = abs(h);
    if (w<0) dx1 = -1; else if (w>0) dx1 = 1;
    if (h<0) dy1 = -1; else if (h>0) dy1 = 1;
    if (w<0) dx2 = -1; else if (w>0) dx2 = 1;
    if (!(longest>shortest)) {
        longest = abs(h);
        shortest = abs(w);
        if (h<0) dy2 = -1; else if (h>0) dy2 = 1;
        dx2 = 0;
    }
    numerator = longest >> 1;
    for (i=0;i<=longest;i++) {
        if (set_at(surf, x, y, color)) {
            anydraw = 1;
            if (x < lowest_x) {
                lowest_x = x;
            }
            if (y < lowest_y) {
                lowest_y = y;
            }
            if (x > highest_x) {
                highest_x = x;
            }
            if (y > highest_y) {
                highest_y = y;
            }
        }
        numerator += shortest;
        if (!(numerator<longest)) {
            numerator -= longest;
            x += dx1;
            y += dy1;
        } else {
            x += dx2;
            y += dy2;
        }
    }
    *(pts) = lowest_x;
    *(pts+1) = lowest_y;
    *(pts+2) = highest_x;
    *(pts+3) = highest_y;
    return anydraw;
}

/* Draw line between (x1, y1) and (x2, y2) */
static int
drawhorzline(SDL_Surface *surf, Uint32 color, int x1, int y1, int x2)
{
    int i, direction;
    int anydraw = 0;
    if (x1 == x2 && set_at(surf, x1, y1, color)) /* Draw only one pixel */
        return 1;
    else {
        direction = (x1 < x2) ? 1 : -1; /* Decide to go left or right */
        for (i = 0; i <= abs(x1 - x2); i++) {
            if (set_at(surf, x1 + direction * i, y1, color)) {
                anydraw = 1;
            }
        }
    }
    return anydraw;
}

/* Draw line between (x1, y1) and (x2, y2) */
static int
drawvertline(SDL_Surface *surf, Uint32 color, int x1, int y1, int y2)
{
    int i, direction;
    int anydraw = 0;
    if (y1 == y2 && set_at(surf, x1, y1, color)) /* Draw only one pixel */
        return 1;
    else {
        direction = (y1 < y2) ? 1 : -1;
        for (i = 0; i <= abs(y1 - y2); i++) {
            if (set_at(surf, x1, y1 + direction * i, color)) {
                anydraw = 1;
            }
        }
    }
    return anydraw;
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

    x_last = (int)(x + cos(angle_start) * radius1);
    y_last = (int)(y - sin(angle_start) * radius2);
    for (a = angle_start + aStep; a <= angle_stop; a += aStep) {
        int points[4];
        x_next = (int)(x + cos(a) * radius1);
        y_next = (int)(y - sin(a) * radius2);
        points[0] = x_last;
        points[1] = y_last;
        points[2] = x_next;
        points[3] = y_next;
        clip_and_draw_line(dst, color, points);
        x_last = x_next;
        y_last = y_next;
    }
}

/* This function is used just for circle drawing because Bresenham Circle Algorithm
 * draws the circle with the odd diameter and in the original pygame it had even diameter
 * It does scaling by that one pixel (direction based on relative quadrant to the center
 * to keep symmetry). In any other drawing function use normal set_at function
 */
static void
draw_circle_pixel(SDL_Surface *dst, int x0, int y0, int x1, int y1, Uint32 color)
{
    // (x0, y0) -> center of the circle, (x1, y1) -> pixel to bve drawn
    // leading_bit -> used for fast quadrant calculation, 1 for negative and 0 for positive
    // number (number is "vector" from center to the pixel)
    int x_leading_bit = ((x1-x0) > 0) - ((x1-x0) < 0) == 1 ? 0 : 1;
    int y_leading_bit = ((y1-y0) > 0) - ((y1-y0) < 0) == 1 ? 0 : 1;
    int quadrant = (x_leading_bit != y_leading_bit) + y_leading_bit + y_leading_bit + 1;
    if (quadrant == 1) {
        set_at(dst, x1 - 1, y1 - 1, color);  // Move one to the left and up
    }
    else if (quadrant == 2) {
        set_at(dst, x1, y1 - 1, color);      // Move one to the up
    }
    else if (quadrant == 3) {
        set_at(dst, x1, y1, color);
    }
    else {
        set_at(dst, x1 - 1, y1, color);      // Move one to the left
    }
}


/* Bresenham Circle Algorithm
 * adapted from: https://de.wikipedia.org/wiki/Bresenham-Algorithmus
 * with additional line width parameter
 */
static void
draw_circle_bresenham(SDL_Surface *dst, int x0, int y0, int radius, int thickness, Uint32 color)
{
    int f = 1 - radius;
    int ddF_x = 0;
    int ddF_y = -2 * radius;
    int x = 0;
    int y = radius;
    int radius1, y1;
    int i_y = radius-thickness;
    int i_f = 1 - i_y;
    int i_ddF_x = 0;
    int i_ddF_y = -2 * i_y;
    int i;

    /* to avoid holes/moire, draw thick line in inner loop,
     * instead of concentric circles in outer loop */
    for (i=0; i<thickness; i++){
        radius1=radius - i;
        draw_circle_pixel(dst, x0, y0, x0, y0 + radius1, color);
        draw_circle_pixel(dst, x0, y0, x0, y0 - radius1, color);
        draw_circle_pixel(dst, x0, y0, x0 + radius1, y0, color);
        draw_circle_pixel(dst, x0, y0, x0 - radius1, y0, color);
    }

    while(x < y)
    {
      if(f >= 0)
      {
        y--;
        ddF_y += 2;
        f += ddF_y;
      }
      if(i_f >= 0)
      {
        i_y--;
        i_ddF_y += 2;
        i_f += i_ddF_y;
      }
      x++;
      ddF_x += 2;
      f += ddF_x + 1;

      i_ddF_x += 2;
      i_f += i_ddF_x + 1;

      if(thickness>1)
          thickness=y-i_y;

      /* as above:
       * to avoid holes/moire, draw thick line in inner loop,
       * instead of concentric circles in outer loop */
      for (i=0; i<thickness; i++){
          y1=y-i;

          draw_circle_pixel(dst, x0, y0, x0 + x, y0 + y1, color);
          draw_circle_pixel(dst, x0, y0, x0 - x, y0 + y1, color);
          draw_circle_pixel(dst, x0, y0, x0 + x, y0 - y1, color);
          draw_circle_pixel(dst, x0, y0, x0 - x, y0 - y1, color);
          draw_circle_pixel(dst, x0, y0, x0 + y1, y0 + x, color);
          draw_circle_pixel(dst, x0, y0, x0 - y1, y0 + x, color);
          draw_circle_pixel(dst, x0, y0, x0 + y1, y0 - x, color);
          draw_circle_pixel(dst, x0, y0, x0 - y1, y0 - x, color);
      }
    }
}

static void
draw_circle_filled(SDL_Surface *dst, int x0, int y0, int radius, Uint32 color)
{
    int f = 1 - radius;
    int ddF_x = 0;
    int ddF_y = -2 * radius;
    int x = 0;
    int y = radius;
    int y1;

    for (y1=y0 - y; y1 <= y0 + y; y1++) {
	    draw_circle_pixel(dst, x0, y0, x0, y1, color);
    }
    draw_circle_pixel(dst, x0, y0, x0 + radius, y0, color);
    draw_circle_pixel(dst, x0, y0, x0 - radius, y0, color);

    while(x < y)
    {
      if(f >= 0)
      {
        y--;
        ddF_y += 2;
        f += ddF_y;
      }
      x++;
      ddF_x += 2;
      f += ddF_x + 1;

      for (y1=y0 - y; y1 <= y0 + y; y1++){
        draw_circle_pixel(dst, x0, y0, x0+x, y1, color);
        draw_circle_pixel(dst, x0, y0, x0-x, y1, color);
      }
      for (y1=y0 - x; y1 <= y0 + x; y1++){
        draw_circle_pixel(dst, x0, y0, x0+y, y1, color);
        draw_circle_pixel(dst, x0, y0, x0-y, y1, color);
      }
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
    int ry = (height >> 1);

    /* Special case: draw a single pixel */
    if (rx == 0 && ry == 0) {
        set_at(dst, x, y, color);
        return;
    }

    /* Special case: draw a vertical line */
    if (rx == 0) {
        drawvertline(dst, color, x, (Sint16)(y - ry),
                         (Sint16)(y + ry + (height & 1)));
        return;
    }

    /* Special case: draw a horizontal line */
    if (ry == 0) {
        drawhorzline(dst, color, (Sint16)(x - rx), y,
                         (Sint16)(x + rx + (width & 1)));
        return;
    }

    /* Adjust ry for the rest of the ellipses (non-special cases). */
    ry += (solid & 1) - yoff;

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
                    drawhorzline(dst, color, x - h, y - k - yoff,
                                     x + h - xoff);
                    drawhorzline(dst, color, x - h, y + k, x + h - xoff);
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
                    drawhorzline(dst, color, x - i, y + j, x + i - xoff);
                    drawhorzline(dst, color, x - i, y - j - yoff,
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
                    drawhorzline(dst, color, x - j, y + i, x + j - xoff);
                    drawhorzline(dst, color, x - j, y - i - yoff,
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
                    drawhorzline(dst, color, x - k, y + h, x + k - xoff);
                    drawhorzline(dst, color, x - k, y - h - yoff,
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
draw_fillpoly(SDL_Surface *dst, int *point_x, int *point_y,
              Py_ssize_t num_points, Uint32 color)
{
    /* point_x : x coordinates of the points
     * point-y : the y coordinates of the points
     * num_points : the number of points
     */
    Py_ssize_t i, i_previous;  // i_previous is the index of the point before i
    int y, miny, maxy;
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
        drawhorzline(dst, color, minx, miny, maxx);
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
     *    (drawhorzline for a pair of two such points)
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
            drawhorzline(dst, color, x_intersect[i], y,
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
            drawhorzline(dst, color, point_x[i], y, point_x[i_previous]);
        }
    }
    PyMem_Free(x_intersect);
}

static PyMethodDef _draw_methods[] = {
    {"aaline", (PyCFunction)aaline, METH_VARARGS | METH_KEYWORDS,
     DOC_PYGAMEDRAWAALINE},
    {"line", (PyCFunction)line, METH_VARARGS | METH_KEYWORDS,
     DOC_PYGAMEDRAWLINE},
    {"aalines", (PyCFunction)aalines, METH_VARARGS | METH_KEYWORDS,
     DOC_PYGAMEDRAWAALINES},
    {"lines", (PyCFunction)lines, METH_VARARGS | METH_KEYWORDS,
     DOC_PYGAMEDRAWLINES},
    {"ellipse", (PyCFunction)ellipse, METH_VARARGS | METH_KEYWORDS,
     DOC_PYGAMEDRAWELLIPSE},
    {"arc", (PyCFunction)arc, METH_VARARGS | METH_KEYWORDS, DOC_PYGAMEDRAWARC},
    {"circle", (PyCFunction)circle, METH_VARARGS | METH_KEYWORDS,
     DOC_PYGAMEDRAWCIRCLE},
    {"polygon", (PyCFunction)polygon, METH_VARARGS | METH_KEYWORDS,
     DOC_PYGAMEDRAWPOLYGON},
    {"rect", (PyCFunction)rect, METH_VARARGS | METH_KEYWORDS,
     DOC_PYGAMEDRAWRECT},

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
