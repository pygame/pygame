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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Declaration of drawing algorithms */
static void
draw_line_width(SDL_Surface *surf, Uint32 color, int x1, int y1, int x2, int y2, int width,
                         int *drawn_area);
static void
draw_line(SDL_Surface *surf, int x1, int y1, int x2, int y2, Uint32 color,
         int *drawn_area);
static void
draw_aaline(SDL_Surface *surf, Uint32 color, float startx, float starty,
            float endx, float endy, int blend, int *drawn_area);
static void
draw_arc(SDL_Surface *surf, int x, int y, int radius1, int radius2,
         double angle_start, double angle_stop, Uint32 color, int *drawn_area);
static void
draw_circle_bresenham(SDL_Surface *surf, int x0, int y0, int radius,
                      int thickness, Uint32 color, int *drawn_area);
static void
draw_circle_bresenham_thin(SDL_Surface *surf, int x0, int y0, int radius,
                      Uint32 color, int *drawn_area);
static void
draw_circle_filled(SDL_Surface *surf, int x0, int y0, int radius, Uint32 color,
                   int *drawn_area);
static void
draw_circle_quadrant(SDL_Surface *surf, int x0, int y0, int radius,
                     int thickness, Uint32 color, int top_right, int top_left,
                     int bottom_left, int bottom_right, int *drawn_area);
static void
draw_ellipse_filled(SDL_Surface *surf, int x0, int y0, int width, int height,
                    Uint32 color, int *drawn_area);
static void
draw_ellipse_thickness(SDL_Surface *surf, int x0, int y0, int width, int height,
                       int thickness, Uint32 color, int *drawn_area);
static void
draw_fillpoly(SDL_Surface *surf, int *vx, int *vy, Py_ssize_t n, Uint32 color,
              int *drawn_area);
static void
draw_round_rect(SDL_Surface *surf, int x1, int y1, int x2, int y2, int radius,
                int width, Uint32 color, int top_left, int top_right,
                int bottom_left, int bottom_right, int *drawn_area);

// validation of a draw color
#define CHECK_LOAD_COLOR(colorobj)                                         \
    if (PyInt_Check(colorobj))                                             \
        color = (Uint32)PyInt_AsLong(colorobj);                            \
    else if (pg_RGBAFromFuzzyColorObj(colorobj, rgba))                     \
        color =                                                            \
            SDL_MapRGBA(surf->format, rgba[0], rgba[1], rgba[2], rgba[3]); \
    else                                                                   \
        return NULL; /* pg_RGBAFromFuzzyColorObj sets the exception for us */

/* Definition of functions that get called in Python */

/* Draws an antialiased line on the given surface.
 *
 * Returns a Rect bounding the drawn area.
 */
static PyObject *
aaline(PyObject *self, PyObject *arg, PyObject *kwargs)
{
    pgSurfaceObject *surfobj = NULL;
    PyObject* colorobj = NULL, *start = NULL, *end = NULL;
    SDL_Surface *surf = NULL;
    float startx, starty, endx, endy;
    int blend = 1; /* Default blend. */
    float pts[4];
    int drawn_area[4] = {INT_MAX, INT_MAX, INT_MIN,
                         INT_MIN}; /* Used to store bounding box values */
    Uint8 rgba[4];
    Uint32 color;
    static char *keywords[] = {"surface", "color", "start_pos",
                               "end_pos", "blend", NULL};

    if (!PyArg_ParseTupleAndKeywords(arg, kwargs, "O!OOO|i", keywords,
                                     &pgSurface_Type, &surfobj, &colorobj,
                                     &start, &end, &blend)) {
        return NULL; /* Exception already set. */
    }

    if (!blend) {
        if (PyErr_WarnEx(PyExc_DeprecationWarning,
                "blend=False will be deprecated in pygame 2.2 and will "
                "default to True",
                1) == -1) {
            return NULL;
        }
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
    draw_aaline(surf, color, pts[0], pts[1], pts[2], pts[3], blend,
                drawn_area);

    if (!pgSurface_Unlock(surfobj)) {
        return RAISE(PyExc_RuntimeError, "error unlocking surface");
    }

    if (drawn_area[0] != INT_MAX && drawn_area[1] != INT_MAX &&
        drawn_area[2] != INT_MIN && drawn_area[3] != INT_MIN)
        return pgRect_New4(drawn_area[0], drawn_area[1],
                           drawn_area[2] - drawn_area[0] + 1,
                           drawn_area[3] - drawn_area[1] + 1);
    else
        return pgRect_New4((int)startx, (int)starty, 0, 0);
}

/* Draws a line on the given surface.
 *
 * Returns a Rect bounding the drawn area.
 */
static PyObject *
line(PyObject *self, PyObject *arg, PyObject *kwargs)
{
    pgSurfaceObject *surfobj = NULL;
    PyObject *colorobj = NULL, *start = NULL, *end = NULL;
    SDL_Surface *surf = NULL;
    int startx, starty, endx, endy;
    Uint8 rgba[4];
    Uint32 color;
    int width = 1; /* Default width. */
    int drawn_area[4] = {INT_MAX, INT_MAX, INT_MIN,
                         INT_MIN}; /* Used to store bounding box values */
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

    draw_line_width(surf, color, startx, starty, endx, endy, width, drawn_area);

    if (!pgSurface_Unlock(surfobj)) {
        return RAISE(PyExc_RuntimeError, "error unlocking surface");
    }

    /* Compute return rect. */
    if (drawn_area[0] != INT_MAX && drawn_area[1] != INT_MAX &&
        drawn_area[2] != INT_MIN && drawn_area[3] != INT_MIN)
        return pgRect_New4(drawn_area[0], drawn_area[1],
                           drawn_area[2] - drawn_area[0] + 1,
                           drawn_area[3] - drawn_area[1] + 1);
    else
        return pgRect_New4(startx, starty, 0, 0);
}

/* Draws a series of antialiased lines on the given surface.
 *
 * Returns a Rect bounding the drawn area.
 */
static PyObject *
aalines(PyObject *self, PyObject *arg, PyObject *kwargs)
{
    pgSurfaceObject *surfobj = NULL;
    PyObject *colorobj = NULL, *closedobj = NULL;
    PyObject *points = NULL, *item = NULL;
    SDL_Surface *surf = NULL;
    Uint32 color;
    Uint8 rgba[4];
    float pts[4];
    float *xlist, *ylist;
    float x, y;
    int l, t;
    int drawn_area[4] = {INT_MAX, INT_MAX, INT_MIN,
                         INT_MIN}; /* Used to store bounding box values */
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

    if (!blend) {
        if (PyErr_WarnEx(
                PyExc_DeprecationWarning,
                "blend=False will be deprecated in pygame 2.2 and will "
                "default to True",
                1) == -1) {
            return NULL;
        }
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
        if (xlist) {
            PyMem_Del(xlist);
        }
        if (ylist) {
            PyMem_Del(ylist);
        }
        return RAISE(PyExc_MemoryError,
                     "cannot allocate memory to draw aalines");
    }

    for (loop = 0; loop < length; ++loop) {
        item = PySequence_GetItem(points, loop);
        result = pg_TwoFloatsFromObj(item, &x, &y);
        if (loop == 0) {
            l = (int) x;
            t = (int) y;
        }
        Py_DECREF(item);

        if (!result) {
            PyMem_Del(xlist);
            PyMem_Del(ylist);
            return RAISE(PyExc_TypeError, "points must be number pairs");
        }

        xlist[loop] = x;
        ylist[loop] = y;
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
        draw_aaline(surf, color, pts[0], pts[1], pts[2], pts[3], blend,
            drawn_area);
    }
    if (closed && length > 2) {
        pts[0] = xlist[length - 1];
        pts[1] = ylist[length - 1];
        pts[2] = xlist[0];
        pts[3] = ylist[0];
        draw_aaline(surf, color, pts[0], pts[1], pts[2], pts[3], blend,
                    drawn_area);
    }

    PyMem_Del(xlist);
    PyMem_Del(ylist);

    if (!pgSurface_Unlock(surfobj)) {
        return RAISE(PyExc_RuntimeError, "error unlocking surface");
    }

    /* Compute return rect. */
    if (drawn_area[0] != INT_MAX && drawn_area[1] != INT_MAX &&
        drawn_area[2] != INT_MIN && drawn_area[3] != INT_MIN)
        return pgRect_New4(drawn_area[0], drawn_area[1],
                           drawn_area[2] - drawn_area[0] + 1,
                           drawn_area[3] - drawn_area[1] + 1);
    else
        return pgRect_New4(l, t, 0, 0);
}

/* Draws a series of lines on the given surface.
 *
 * Returns a Rect bounding the drawn area.
 */
static PyObject *
lines(PyObject *self, PyObject *arg, PyObject *kwargs)
{
    pgSurfaceObject *surfobj = NULL;
    PyObject *colorobj = NULL, *closedobj = NULL;
    PyObject *points = NULL, *item = NULL;
    SDL_Surface *surf = NULL;
    Uint32 color;
    Uint8 rgba[4];
    int x, y, closed, result;
    int *xlist = NULL, *ylist = NULL;
    int width = 1; /* Default width. */
    Py_ssize_t loop, length;
    int drawn_area[4] = {INT_MAX, INT_MAX, INT_MIN,
                         INT_MIN}; /* Used to store bounding box values */
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
        if (xlist) {
            PyMem_Del(xlist);
        }
        if (ylist) {
            PyMem_Del(ylist);
        }
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
        draw_line_width(surf, color, xlist[loop - 1], ylist[loop - 1], xlist[loop], ylist[loop], width, drawn_area);
    }

    if (closed && length > 2) {
        draw_line_width(surf, color, xlist[length - 1], ylist[length - 1], xlist[0], ylist[0], width, drawn_area);
    }

    PyMem_Del(xlist);
    PyMem_Del(ylist);

    if (!pgSurface_Unlock(surfobj)) {
        return RAISE(PyExc_RuntimeError, "error unlocking surface");
    }

    /* Compute return rect. */
    if (drawn_area[0] != INT_MAX && drawn_area[1] != INT_MAX &&
        drawn_area[2] != INT_MIN && drawn_area[3] != INT_MIN)
        return pgRect_New4(drawn_area[0], drawn_area[1],
                           drawn_area[2] - drawn_area[0] + 1,
                           drawn_area[3] - drawn_area[1] + 1);
    else
        return pgRect_New4(x, y, 0, 0);
}

static PyObject *
arc(PyObject *self, PyObject *arg, PyObject *kwargs)
{
    pgSurfaceObject *surfobj = NULL;
    PyObject *colorobj = NULL, *rectobj = NULL;
    GAME_Rect *rect = NULL, temp;
    SDL_Surface *surf = NULL;
    Uint8 rgba[4];
    Uint32 color;
    int loop;
    int width = 1; /* Default width. */
    int drawn_area[4] = {INT_MAX, INT_MAX, INT_MIN,
                         INT_MIN}; /* Used to store bounding box values */
    double angle_start, angle_stop;
    static char *keywords[] = {"surface",    "color", "rect", "start_angle",
                               "stop_angle", "width", NULL};

    if (!PyArg_ParseTupleAndKeywords(
            arg, kwargs, "O!OOdd|i", keywords, &pgSurface_Type, &surfobj,
            &colorobj, &rectobj, &angle_start, &angle_stop, &width)) {
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
                 angle_stop, color, drawn_area);
    }

    if (!pgSurface_Unlock(surfobj)) {
        return RAISE(PyExc_RuntimeError, "error unlocking surface");
    }

    /* Compute return rect. */
    if (drawn_area[0] != INT_MAX && drawn_area[1] != INT_MAX &&
        drawn_area[2] != INT_MIN && drawn_area[3] != INT_MIN)
        return pgRect_New4(drawn_area[0], drawn_area[1],
                           drawn_area[2] - drawn_area[0] + 1,
                           drawn_area[3] - drawn_area[1] + 1);
    else
        return pgRect_New4(rect->x, rect->y, 0, 0);
}

static PyObject *
ellipse(PyObject *self, PyObject *arg, PyObject *kwargs)
{
    pgSurfaceObject *surfobj = NULL;
    PyObject *colorobj = NULL, *rectobj = NULL;
    GAME_Rect *rect = NULL, temp;
    SDL_Surface *surf = NULL;
    Uint8 rgba[4];
    Uint32 color;
    int width = 0; /* Default width. */
    int drawn_area[4] = {INT_MAX, INT_MAX, INT_MIN,
                         INT_MIN}; /* Used to store bounding box values */
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

    if (!pgSurface_Lock(surfobj)) {
        return RAISE(PyExc_RuntimeError, "error locking surface");
    }

    if (!width || width >= MIN(rect->w / 2 + rect->w % 2, rect->h / 2 + rect->h % 2)) {
        draw_ellipse_filled(surf, rect->x, rect->y, rect->w, rect->h, color, drawn_area);
    }
    else {
        draw_ellipse_thickness(surf, rect->x, rect->y, rect->w, rect->h, width - 1,
                               color, drawn_area);
    }

    if (!pgSurface_Unlock(surfobj)) {
        return RAISE(PyExc_RuntimeError, "error unlocking surface");
    }

    if (drawn_area[0] != INT_MAX && drawn_area[1] != INT_MAX &&
        drawn_area[2] != INT_MIN && drawn_area[3] != INT_MIN)
        return pgRect_New4(drawn_area[0], drawn_area[1],
                           drawn_area[2] - drawn_area[0] + 1,
                           drawn_area[3] - drawn_area[1] + 1);
    else
        return pgRect_New4(rect->x, rect->y, 0, 0);
}

static PyObject *
circle(PyObject *self, PyObject *args, PyObject *kwargs)
{
    pgSurfaceObject *surfobj = NULL;
    PyObject *colorobj = NULL;
    SDL_Surface *surf = NULL;
    Uint8 rgba[4];
    Uint32 color;
    PyObject *posobj, *radiusobj;
    int posx, posy, radius;
    int width = 0; /* Default values. */
    int top_right = 0, top_left = 0, bottom_left = 0, bottom_right = 0;
    int drawn_area[4] = {INT_MAX, INT_MAX, INT_MIN,
                         INT_MIN}; /* Used to store bounding box values */
    static char *keywords[] = {"surface",
                               "color",
                               "center",
                               "radius",
                               "width",
                               "draw_top_right",
                               "draw_top_left",
                               "draw_bottom_left",
                               "draw_bottom_right",
                               NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!OOO|iiiii", keywords,
                                     &pgSurface_Type, &surfobj, &colorobj,
                                     &posobj, &radiusobj, &width, &top_right,
                                     &top_left, &bottom_left, &bottom_right))
        return NULL; /* Exception already set. */

    if (!pg_TwoIntsFromObj(posobj, &posx, &posy)) {
        PyErr_SetString(PyExc_TypeError,
                        "center argument must be a pair of numbers");
        return 0;
    }

    if (!pg_IntFromObj(radiusobj, &radius)) {
        PyErr_SetString(PyExc_TypeError, "radius argument must be a number");
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

    if ((top_right == 0 && top_left == 0 && bottom_left == 0 &&
         bottom_right == 0)) {
        if (!width || width == radius) {
            draw_circle_filled(surf, posx, posy, radius, color, drawn_area);
        } else if (width == 1) {
            draw_circle_bresenham_thin(surf, posx, posy, radius, color,
                                  drawn_area);
        } else {
            draw_circle_bresenham(surf, posx, posy, radius, width, color,
                                  drawn_area);
        }
    }
    else {
        draw_circle_quadrant(surf, posx, posy, radius, width, color, top_right,
                             top_left, bottom_left, bottom_right, drawn_area);
    }

    if (!pgSurface_Unlock(surfobj)) {
        return RAISE(PyExc_RuntimeError, "error unlocking surface");
    }
    if (drawn_area[0] != INT_MAX && drawn_area[1] != INT_MAX &&
        drawn_area[2] != INT_MIN && drawn_area[3] != INT_MIN)
        return pgRect_New4(drawn_area[0], drawn_area[1],
                           drawn_area[2] - drawn_area[0] + 1,
                           drawn_area[3] - drawn_area[1] + 1);
    else
        return pgRect_New4(posx, posy, 0, 0);
}

static PyObject *
polygon(PyObject *self, PyObject *arg, PyObject *kwargs)
{
    pgSurfaceObject *surfobj = NULL;
    PyObject *colorobj = NULL, *points = NULL, *item = NULL;
    SDL_Surface *surf = NULL;
    Uint8 rgba[4];
    Uint32 color;
    int *xlist = NULL, *ylist = NULL;
    int width = 0; /* Default width. */
    int x, y, result, l, t;
    int drawn_area[4] = {INT_MAX, INT_MAX, INT_MIN,
                         INT_MIN}; /* Used to store bounding box values */
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
        if (xlist) {
            PyMem_Del(xlist);
        }
        if (ylist) {
            PyMem_Del(ylist);
        }
        return RAISE(PyExc_MemoryError,
                     "cannot allocate memory to draw polygon");
    }

    for (loop = 0; loop < length; ++loop) {
        item = PySequence_GetItem(points, loop);
        result = pg_TwoIntsFromObj(item, &x, &y);
        if (loop == 0) {
            l = x;
            t = y;
        }
        Py_DECREF(item);

        if (!result) {
            PyMem_Del(xlist);
            PyMem_Del(ylist);
            return RAISE(PyExc_TypeError, "points must be number pairs");
        }

        xlist[loop] = x;
        ylist[loop] = y;
    }

    if (!pgSurface_Lock(surfobj)) {
        PyMem_Del(xlist);
        PyMem_Del(ylist);
        return RAISE(PyExc_RuntimeError, "error locking surface");
    }

    draw_fillpoly(surf, xlist, ylist, length, color, drawn_area);
    PyMem_Del(xlist);
    PyMem_Del(ylist);

    if (!pgSurface_Unlock(surfobj)) {
        return RAISE(PyExc_RuntimeError, "error unlocking surface");
    }

    if (drawn_area[0] != INT_MAX && drawn_area[1] != INT_MAX &&
        drawn_area[2] != INT_MIN && drawn_area[3] != INT_MIN)
        return pgRect_New4(drawn_area[0], drawn_area[1],
                           drawn_area[2] - drawn_area[0] + 1,
                           drawn_area[3] - drawn_area[1] + 1);
    else
        return pgRect_New4(l, t, 0, 0);
}

static PyObject *
rect(PyObject *self, PyObject *args, PyObject *kwargs)
{
    pgSurfaceObject *surfobj = NULL;
    PyObject *colorobj = NULL, *rectobj = NULL;
    PyObject *points = NULL, *poly_args = NULL, *ret = NULL;
    GAME_Rect *rect = NULL, temp;
    SDL_Surface *surf = NULL;
    Uint8 rgba[4];
    Uint32 color;
    int t, l, b, r, width = 0, radius = 0; /* Default values. */
    int x, y, w, h; /* Fields for the rounded rect draw to "normalize" into */
    int top_left_radius = -1, top_right_radius = -1, bottom_left_radius = -1,
        bottom_right_radius = -1;
#if IS_SDLv2
    SDL_Rect sdlrect;
    SDL_Rect cliprect;
    int result;
    SDL_Rect clipped;
#endif /* IS_SDLv2 */
    int drawn_area[4] = {INT_MAX, INT_MAX, INT_MIN,
                         INT_MIN}; /* Used to store bounding box values */
    static char *keywords[] = {"surface",
                               "color",
                               "rect",
                               "width",
                               "border_radius",
                               "border_top_left_radius",
                               "border_top_right_radius",
                               "border_bottom_left_radius",
                               "border_bottom_right_radius",
                               NULL};
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "O!OO|iiiiii", keywords, &pgSurface_Type, &surfobj,
            &colorobj, &rectobj, &width, &radius, &top_left_radius,
            &top_right_radius, &bottom_left_radius, &bottom_right_radius)) {
        return NULL; /* Exception already set. */
    }

    if (!(rect = pgRect_FromObject(rectobj, &temp))) {
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

    /* If there isn't any rounded rect-ness OR the rect is really thin in one direction.
       The "really thin in one direction" check is necessary because draw_round_rect
       fails (draws something bad) on rects with a dimension that is 0 or 1 pixels across.*/
    if ((radius <= 0 && top_left_radius <= 0 && top_right_radius <= 0 &&
        bottom_left_radius <= 0 && bottom_right_radius <= 0) || 
        abs(rect->w) < 2 || abs(rect->h) < 2) {
#if IS_SDLv2
        if(width > 0){
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
        else {
            sdlrect.x = rect->x;
            sdlrect.y = rect->y;
            sdlrect.w = rect->w;
            sdlrect.h = rect->h;

            SDL_GetClipRect(surf, &cliprect);

            /* SDL_FillRect respects the clip rect already, but in order to
               return the drawn area, we need to do this here, and keep the
               pointer to the result in clipped */
            if (!SDL_IntersectRect(&sdlrect,
                                   &cliprect,
                                   &clipped)) {
                return pgRect_New4(rect->x, rect->y, 0, 0);
            }
            pgSurface_Prep(surfobj);
            pgSurface_Lock(surfobj);
            result = SDL_FillRect(surf, &clipped, color);
            pgSurface_Unlock(surfobj);
            pgSurface_Unprep(surfobj);
            if (result != 0)
                return RAISE(pgExc_SDLError, SDL_GetError());
            return pgRect_New(&clipped);
        }
#else
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
#endif
    }
    else {
        if (!pgSurface_Lock(surfobj)) {
            return RAISE(PyExc_RuntimeError, "error locking surface");
        }

        /* Little bit to normalize the rect: this matters for the rounded
           rects, despite not mattering for the normal rects. */
        x = rect->x;
        y = rect->y;
        w = rect->w;
        h = rect->h;

        if (w < 0) {
            x += w;
            w = -w;
        }

        if (h < 0) {
            y += h;
            h = -h;
        }

        if (width > w / 2 || width > h / 2) {
            width = MAX(w / 2, h / 2);
        }

        draw_round_rect(surf, x, y, x + w - 1, y + h - 1, radius, width, color,
                        top_left_radius, top_right_radius, bottom_left_radius,
                        bottom_right_radius, drawn_area);
        if (!pgSurface_Unlock(surfobj)) {
            return RAISE(PyExc_RuntimeError, "error unlocking surface");
        }
    }

    if (drawn_area[0] != INT_MAX && drawn_area[1] != INT_MAX &&
        drawn_area[2] != INT_MIN && drawn_area[3] != INT_MIN)
        return pgRect_New4(drawn_area[0], drawn_area[1],
                           drawn_area[2] - drawn_area[0] + 1,
                           drawn_area[3] - drawn_area[1] + 1);
    else
        return pgRect_New4(rect->x, rect->y, 0, 0);
}

/* Functions used in drawing algorithms */

static void
swap(float *a, float *b)
{
    float temp = *a;
    *a = *b;
    *b = temp;
}

static int
compare_int(const void *a, const void *b)
{
    return (*(const int *)a) - (*(const int *)b);
}

static int
sign(int x, int y)
{
    return (x > 0) ? 1 : ((x < 0) ? -1 : y);
}

static Uint32
get_antialiased_color(SDL_Surface *surf, int x, int y, Uint32 original_color,
                      float brightness, int blend)
{
    Uint8 color_part[4], background_color[4];
    Uint32 *pixels = (Uint32 *)surf->pixels;
    SDL_GetRGBA(original_color, surf->format, &color_part[0], &color_part[1],
                &color_part[2], &color_part[3]);
    if (blend) {
        if (x < surf->clip_rect.x || x >= surf->clip_rect.x + surf->clip_rect.w ||
            y < surf->clip_rect.y || y >= surf->clip_rect.y + surf->clip_rect.h)
            return original_color;
        SDL_GetRGBA(pixels[(y * surf->w) + x], surf->format, &background_color[0],
                    &background_color[1], &background_color[2], &background_color[3]);
        color_part[0] = (Uint8) (brightness * color_part[0] +
                                (1 - brightness) * background_color[0]);
        color_part[1] = (Uint8) (brightness * color_part[1] +
                                (1 - brightness) * background_color[1]);
        color_part[2] = (Uint8) (brightness * color_part[2] +
                                (1 - brightness) * background_color[2]);
        color_part[3] = (Uint8) (brightness * color_part[3] +
                                (1 - brightness) * background_color[3]);
    }
    else {
        color_part[0] =  (Uint8) (brightness * color_part[0]);
        color_part[1] =  (Uint8) (brightness * color_part[1]);
        color_part[2] =  (Uint8) (brightness * color_part[2]);
        color_part[3] =  (Uint8) (brightness * color_part[3]);
    }
    original_color = SDL_MapRGBA(surf->format, color_part[0], color_part[1],
                                 color_part[2], color_part[3]);
    return original_color;
}

static void
add_pixel_to_drawn_list(int x, int y, int *pts)
{
    if (x < pts[0]) {
        pts[0] = x;
    }
    if (y < pts[1]) {
        pts[1] = y;
    }
    if (x > pts[2]) {
        pts[2] = x;
    }
    if (y > pts[3]) {
        pts[3] = y;
    }
}

static int
clip_line(SDL_Surface *surf, int *x1, int *y1, int *x2, int *y2) {
    int p1 = *x1 - *x2;
    int p2 = -p1;
    int p3 = *y1 - *y2;
    int p4 = -p3;
    int q1 = *x1 - surf->clip_rect.x;
    int q2 = surf->clip_rect.w + surf->clip_rect.x - *x1;
    int q3 = *y1 - surf->clip_rect.y;
    int q4 = surf->clip_rect.h + surf->clip_rect.y - *y1;
    int old_x1 = *x1;
    int old_y1 = *y1;
    double nmax = 0;
    double pmin = 1;
    double r1, r2;
    if ((p1 == 0 && q1 < 0) || (p2 == 0 && q2 < 0) || (p3 == 0 && q3 < 0) || (p4 == 0 && q4 < 0))
        return 0;
    if (p1) {
        r1 = (double) q1 / p1;
        r2 = (double) q2 / p2;
        if (p1 < 0) {
            if (r1 > nmax)
                nmax = r1;
            if (r2 < pmin)
                pmin = r2;
        }
        else {
            if (r2 > nmax)
                nmax = r2;
            if (r1 < pmin)
                pmin = r1;
        }
    }
    if (p3) {
        r1 = (double) q3 / p3;
        r2 = (double) q4 / p4;
        if (p3 < 0) {
            if (r1 > nmax)
                nmax = r1;
            if (r2 < pmin)
                pmin = r2;
        }
        else {
            if (r2 > nmax)
                nmax = r2;
            if (r1 < pmin)
                pmin = r1;
        }
    }
    if (nmax > pmin)
        return 0;
    *x1 = old_x1 + (int) (p2 * nmax < 0 ? (p2 * nmax - 0.5) : (p2 * nmax + 0.5));
    *y1 = old_y1 + (int) (p4 * nmax < 0 ? (p4 * nmax - 0.5) : (p4 * nmax + 0.5));
    *x2 = old_x1 + (int) (p2 * pmin < 0 ? (p2 * pmin - 0.5) : (p2 * pmin + 0.5));
    *y2 = old_y1 + (int) (p4 * pmin < 0 ? (p4 * pmin - 0.5) : (p4 * pmin + 0.5));
    return 1;
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

static void
set_and_check_rect(SDL_Surface *surf, int x, int y, Uint32 color, int *drawn_area)
{
    if (set_at(surf, x, y, color))
        add_pixel_to_drawn_list(x, y, drawn_area);
}

static void
draw_aaline(SDL_Surface *surf, Uint32 color, float from_x, float from_y,
            float to_x, float to_y, int blend, int *drawn_area)
{
    float gradient, dx, dy, intersect_y, brightness;
    int x, x_pixel_start, x_pixel_end;
    Uint32 pixel_color;
    float x_gap, y_endpoint, clip_left, clip_right, clip_top, clip_bottom;
    int steep, y;

    dx = to_x - from_x;
    dy = to_y - from_y;

    /* Single point.
     * A line with length 0 is drawn as a single pixel at full brightness. */
    if (fabs(dx) < 0.0001 && fabs(dy) < 0.0001) {
        pixel_color = get_antialiased_color(surf, (int)floor(from_x + 0.5),
                                            (int)floor(from_y + 0.5), color,
                                            1, blend);
        set_and_check_rect(surf, (int)floor(from_x + 0.5),
                           (int)floor(from_y + 0.5), pixel_color, drawn_area);
        return;
    }

    /* To draw correctly the pixels at the border of the clipping area when
     * the line crosses it, we need to clip it one pixel wider in all four
     * directions: */
    clip_left = (float)surf->clip_rect.x - 1.0f;
    clip_right = (float)clip_left + surf->clip_rect.w + 1.0f;
    clip_top = (float)surf->clip_rect.y - 1.0f;
    clip_bottom = (float)clip_top + surf->clip_rect.h + 1.0f;

    steep = fabs(dx) < fabs(dy);
    if (steep) {
        swap(&from_x, &from_y);
        swap(&to_x, &to_y);
        swap(&dx, &dy);
        swap(&clip_left, &clip_top);
        swap(&clip_right, &clip_bottom);
    }
    if (dx < 0) {
        swap(&from_x, &to_x);
        swap(&from_y, &to_y);
        dx = -dx;
        dy = -dy;
    }

    if (to_x <= clip_left || from_x >= clip_right) {
        /* The line is completly to the side of the surface */
        return;
    }

    /* Note. There is no need to guard against a division by zero here. If dx
     * was zero then either we had a single point (and we've returned) or it
     * has been swapped with a non-zero dy. */
    gradient = dy/dx;

    /* No need to waste CPU cycles on pixels not on the surface. */
    if (from_x < clip_left) {
        from_y += gradient * (clip_left - from_x);
        from_x = clip_left;
    }
    if (to_x > clip_right) {
        to_y += gradient * (clip_right - to_x);
        to_x = clip_right;
    }

    if (gradient > 0.0f) {
        /* from_ is the topmost endpoint */
        if (to_y <= clip_top || from_y >= clip_bottom) {
            /* The line does not enter the surface */
            return;
        }
        if (from_y < clip_top) {
            from_x += (clip_top - from_y) / gradient;
            from_y = clip_top;
        }
        if (to_y > clip_bottom) {
            to_x += (clip_bottom - to_y) / gradient;
            to_y = clip_bottom;
        }
    }
    else {
        /* to_ is the topmost endpoint */
        if (from_y <= clip_top || to_y >= clip_bottom) {
            /* The line does not enter the surface */
            return;
        }
        if (to_y < clip_top) {
            to_x += (clip_top - to_y) / gradient;
            to_y = clip_top;
        }
        if (from_y > clip_bottom) {
            from_x += (clip_bottom - from_y) / gradient;
            from_y = clip_bottom;
        }
    }
    /* By moving the points one pixel down, we can assume y is never negative.
     * That permit us to use (int)y to round down intead of having to use
     * floor(y). We then draw the pixels one higher.*/
    from_y += 1.0f;
    to_y += 1.0f;

    /* Handle endpoints separatly.
     * The line is not a mathematical line of thickness zero. The same
     * goes for the endpoints. The have a height and width of one pixel. */
    /* First endpoint */
    x_pixel_start = (int)from_x;
    y_endpoint = intersect_y = from_y + gradient * (x_pixel_start - from_x);
    if (to_x > clip_left + 1.0f) {
        x_gap = 1 + x_pixel_start - from_x;
        brightness = y_endpoint - (int)y_endpoint;
        if (steep) {
            x = (int)y_endpoint;
            y = x_pixel_start;
        }
        else {
            x = x_pixel_start;
            y = (int)y_endpoint;
        }
        if ((int)y_endpoint < y_endpoint) {
            pixel_color = get_antialiased_color(surf, x, y, color,
                                                brightness * x_gap, blend);
            set_and_check_rect(surf, x, y, pixel_color, drawn_area);
        }
        if (steep) {
            x--;
        }
        else {
            y--;
        }
        brightness = 1 - brightness;
        pixel_color = get_antialiased_color(surf, x, y, color,
                                            brightness * x_gap, blend);
        set_and_check_rect(surf, x, y, pixel_color, drawn_area);
        intersect_y += gradient;
        x_pixel_start++;
    }
    /* Second endpoint */
    x_pixel_end = (int)ceil(to_x);
    if (from_x < clip_right - 1.0f) {
        y_endpoint = to_y + gradient * (x_pixel_end - to_x);
        x_gap = 1 - x_pixel_end + to_x;
        brightness = y_endpoint - (int)y_endpoint;
        if (steep) {
            x = (int)y_endpoint;
            y = x_pixel_end;
        }
        else {
            x = x_pixel_end;
            y = (int)y_endpoint;
        }
        if ((int)y_endpoint < y_endpoint) {
            pixel_color = get_antialiased_color(surf, x, y, color,
                                                brightness * x_gap, blend);
            set_and_check_rect(surf, x, y, pixel_color, drawn_area);
        }
        if (steep) {
            x--;
        }
        else {
            y--;
        }
        brightness = 1 - brightness;
        pixel_color = get_antialiased_color(surf, x, y, color,
                                            brightness * x_gap, blend);
        set_and_check_rect(surf, x, y, pixel_color, drawn_area);
    }

    /* main line drawing loop */
    for (x = x_pixel_start; x < x_pixel_end; x++) {
        y = (int)intersect_y;
        if (steep) {
            brightness = 1 - intersect_y + y;
            pixel_color = get_antialiased_color(surf, y - 1, x,
                                                color, brightness, blend);
            set_and_check_rect(surf, y - 1, x, pixel_color, drawn_area);
            if (y < intersect_y) {
                brightness = 1 - brightness;
                pixel_color = get_antialiased_color(surf, y, x,
                                                    color, brightness, blend);
                set_and_check_rect(surf, y, x, pixel_color, drawn_area);
            }
        }
        else {
            brightness = 1 - intersect_y + y;
            pixel_color = get_antialiased_color(surf, x, y - 1,
                                                color, brightness, blend);
            set_and_check_rect(surf, x, y - 1, pixel_color, drawn_area);
            if (y < intersect_y) {
                brightness = 1 - brightness;
                pixel_color = get_antialiased_color(surf, x, y,
                                                    color, brightness, blend);
                set_and_check_rect(surf, x, y, pixel_color, drawn_area);
            }
        }
        intersect_y += gradient;
    }
}

static void
drawhorzline(SDL_Surface *surf, Uint32 color, int x1, int y1, int x2)
{
    Uint8 *pixel, *end;
    Uint8 *colorptr;

    if (x1 == x2) {
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
drawhorzlineclip(SDL_Surface *surf, Uint32 color, int x1, int y1, int x2, int *pts)
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

    if (x1 == x2) {
        set_and_check_rect(surf, x1, y1, color, pts);
        return;
    }

    add_pixel_to_drawn_list(x1, y1, pts);
    add_pixel_to_drawn_list(x2, y1, pts);

    drawhorzline(surf, color, x1, y1, x2);
}

int inside_clip(SDL_Surface *surf, int x, int y) {
    if (x < surf->clip_rect.x || x >= surf->clip_rect.x + surf->clip_rect.w ||
        y < surf->clip_rect.y || y >= surf->clip_rect.y + surf->clip_rect.h)
        return 0;
    return 1;
}

static void
draw_line_width(SDL_Surface *surf, Uint32 color, int x1, int y1, int x2, int y2, int width,
                int *drawn_area)
{
    int dx, dy, err, e2, sx, sy, y;
    int left_top, right_bottom;
    int end_x = x2; int end_y = y2;
    int xinc = 0;
    /* Decide which direction to grow (width/thickness). */
    if (abs(x1 - x2) <= abs(y1 - y2)) {
        /* The line's thickness will be in the x direction. The top/bottom
         * ends of the line will be flat. */
        xinc = 1;
    }
    dx = abs(x2 - x1);
    sx = x1 < x2 ? 1 : -1;
    dy = abs(y2 - y1);
    sy = y1 < y2 ? 1 : -1;
    err = (dx > dy ? dx : -dy) / 2;
    if (clip_line(surf, &x1, &y1, &x2, &y2)) {
        if (width == 1)
            draw_line(surf, x1, y1, x2, y2, color, drawn_area);
        else {
            if (xinc) {
                left_top = x1 - (width - 1) / 2;
                right_bottom = x1 + width / 2;
            }
            else {
                left_top = y1 -(width - 1) / 2;
                right_bottom = y1 + width / 2;
            }
            while ((sign(x1 - x2, sx) != sx) || (sign(y1 - y2, sy) != sy)) {
                if (xinc)
                    drawhorzlineclip(surf, color, left_top, y1, right_bottom, drawn_area);
                else {
                    for (y = left_top; y <= right_bottom; y++)
                        set_and_check_rect(surf, x1, y, color, drawn_area);
                }
                e2 = err;
                if (e2 >-dx) {
                    err -= dy;
                    x1 += sx;
                    if (xinc) { left_top += sx; right_bottom += sx; }
                }
                if (e2 < dy) {
                    err += dx;
                    y1 += sy;
                    if (!xinc) { left_top += sy; right_bottom += sy; }
                }
            }
            if (xinc) {
                while (y1 != end_y && (inside_clip(surf, left_top, y1) || inside_clip(surf, right_bottom, y1))) {
                    drawhorzlineclip(surf, color, left_top, y1, right_bottom, drawn_area);
                    e2 = err;
                    if (e2 >-dx) { err -= dy; x1 += sx; left_top += sx; right_bottom += sx; }
                    if (e2 < dy) { err += dx; y1 += sy; }
                }
                drawhorzlineclip(surf, color, left_top, y1, right_bottom, drawn_area);
            }
            else {
                while (x1 != end_x && (inside_clip(surf, x1, left_top) || inside_clip(surf, x1, right_bottom))) {
                    for (y = left_top; y <= right_bottom; y++)
                        set_and_check_rect(surf, x1, y, color, drawn_area);
                    e2 = err;
                    if (e2 >-dx) { err -= dy; x1 += sx; }
                    if (e2 < dy) { err += dx; y1 += sy; left_top += sy; right_bottom += sy; }
                }
                for (y = left_top; y <= right_bottom; y++)
                    set_and_check_rect(surf, x1, y, color, drawn_area);
            }
        }
    }
}

/* Algorithm modified from
 * https://rosettacode.org/wiki/Bitmap/Bresenham%27s_line_algorithm
 */
static void
draw_line(SDL_Surface *surf, int x1, int y1, int x2, int y2, Uint32 color, int *drawn_area)
{
    int dx, dy, err, e2, sx, sy;
    if (x1 == x2 && y1 == y2) {  /* Single point */
        set_and_check_rect(surf, x1, y1, color, drawn_area);
        return;
    }
    if (y1 == y2) {  /* Horizontal line */
        dx = (x1 < x2) ? 1 : -1;
        for (sx = 0; sx <= abs(x1 - x2); sx++) {
            set_and_check_rect(surf, x1 + dx * sx, y1, color, drawn_area);
        }

        return;
    }
    if (x1 == x2) {  /* Vertical line */
        dy = (y1 < y2) ? 1 : -1;
        for (sy = 0; sy <= abs(y1 - y2); sy++)
            set_and_check_rect(surf, x1, y1 + dy * sy, color, drawn_area);
        return;
    }
    dx = abs(x2 - x1), sx = x1 < x2 ? 1 : -1;
    dy = abs(y2 - y1), sy = y1 < y2 ? 1 : -1;
    err = (dx > dy ? dx : -dy) / 2;
    while (x1 != x2 || y1 != y2) {
        set_and_check_rect(surf, x1, y1, color, drawn_area);
        e2 = err;
        if (e2 >-dx) { err -= dy; x1 += sx; }
        if (e2 < dy) { err += dx; y1 += sy; }
    }
    set_and_check_rect(surf, x2, y2, color, drawn_area);
}

static void
draw_arc(SDL_Surface *surf, int x, int y, int radius1, int radius2,
         double angle_start, double angle_stop, Uint32 color, int *drawn_area)
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
        draw_line(surf, points[0], points[1], points[2], points[3],
                  color, drawn_area);
        x_last = x_next;
        y_last = y_next;
    }
}

/* Bresenham Circle Algorithm
 * adapted from: https://de.wikipedia.org/wiki/Bresenham-Algorithmus
 * with additional line width parameter
 */
static void
draw_circle_bresenham(SDL_Surface *surf, int x0, int y0, int radius,
                      int thickness, Uint32 color, int *drawn_area)
{
    int f = 1 - radius;
    int ddF_x = 0;
    int ddF_y = -2 * radius;
    int x = 0;
    int y = radius;
    int y1;
    int i_y = radius - thickness;
    int thickness_inner = thickness;
    int i_f = 1 - i_y;
    int i_ddF_x = 0;
    int i_ddF_y = -2 * i_y;
    int i;

    while (x < y) {
        if (f >= 0) {
            y--;
            ddF_y += 2;
            f += ddF_y;
        }
        /* inner circle*/
        if (i_f >= 0) {
            i_y--;
            i_ddF_y += 2;
            i_f += i_ddF_y;
        }
        x++;
        ddF_x += 2;
        f += ddF_x + 1;

        /* inner circle*/
        i_ddF_x += 2;
        i_f += i_ddF_x + 1;

        if (x > i_y) {
            /* Distance between outer circle and 45-degree angle */
            /* plus one pixel so there's no gap */
            thickness_inner = y - x + 1;
        } else {
            /* Distance between outer and inner circle */
            thickness_inner = y - i_y;
        }

        /* Numbers represent parts of circle function draw in radians
           interval: [number - 1 * pi / 4, number * pi / 4] */
        for (i = 0; i < thickness_inner; i++) {
            y1 = y - i;
            set_and_check_rect(surf, x0 + x - 1, y0 + y1 - 1, color,
                   drawn_area);                                  /* 7 */
            set_and_check_rect(surf, x0 - x, y0 + y1 - 1, color, drawn_area); /* 6 */
            set_and_check_rect(surf, x0 + x - 1, y0 - y1, color, drawn_area); /* 2 */
            set_and_check_rect(surf, x0 - x, y0 - y1, color, drawn_area);     /* 3 */
            set_and_check_rect(surf, x0 + y1 - 1, y0 + x - 1, color,
                   drawn_area);                                  /* 8 */
            set_and_check_rect(surf, x0 + y1 - 1, y0 - x, color, drawn_area); /* 1 */
            set_and_check_rect(surf, x0 - y1, y0 + x - 1, color, drawn_area); /* 5 */
            set_and_check_rect(surf, x0 - y1, y0 - x, color, drawn_area);     /* 4 */
        }
    }
}

static void
draw_circle_bresenham_thin(SDL_Surface *surf, int x0, int y0, int radius,
                           Uint32 color, int *drawn_area)
{
    int f = 1 - radius;
    int ddF_x = 0;
    int ddF_y = -2 * radius;
    int x = 0;
    int y = radius;

    while (x < y) {
        if (f >= 0) {
            y--;
            ddF_y += 2;
            f += ddF_y;
        }
        x++;
        ddF_x += 2;
        f += ddF_x + 1;

        set_and_check_rect(surf, x0 + x - 1, y0 + y - 1, color, drawn_area); /* 7 */
        set_and_check_rect(surf, x0 - x,     y0 + y - 1, color, drawn_area); /* 6 */
        set_and_check_rect(surf, x0 + x - 1, y0 - y,     color, drawn_area); /* 2 */
        set_and_check_rect(surf, x0 - x,     y0 - y,     color, drawn_area); /* 3 */
        set_and_check_rect(surf, x0 + y - 1, y0 + x - 1, color, drawn_area); /* 8 */
        set_and_check_rect(surf, x0 + y - 1, y0 - x,     color, drawn_area); /* 1 */
        set_and_check_rect(surf, x0 - y,     y0 + x - 1, color, drawn_area); /* 5 */
        set_and_check_rect(surf, x0 - y,     y0 - x,     color, drawn_area); /* 4 */
    }
}

static void
draw_circle_quadrant(SDL_Surface *surf, int x0, int y0, int radius,
                     int thickness, Uint32 color, int top_right, int top_left,
                     int bottom_left, int bottom_right, int *drawn_area)
{
    int f = 1 - radius;
    int ddF_x = 0;
    int ddF_y = -2 * radius;
    int x = 0;
    int y = radius;
    int y1;
    int i_y = radius - thickness;
    int i_f = 1 - i_y;
    int i_ddF_x = 0;
    int i_ddF_y = -2 * i_y;
    int i;
    if (radius == 1) {
        if (top_right > 0)
            set_and_check_rect(surf, x0, y0 - 1, color, drawn_area);
        if (top_left > 0)
            set_and_check_rect(surf, x0 - 1, y0 - 1, color, drawn_area);
        if (bottom_left > 0)
            set_and_check_rect(surf, x0 - 1, y0, color, drawn_area);
        if (bottom_right > 0)
            set_and_check_rect(surf, x0, y0, color, drawn_area);
        return;
    }

    if (thickness != 0) {
        while (x < y) {
            if (f >= 0) {
                y--;
                ddF_y += 2;
                f += ddF_y;
            }
            if (i_f >= 0) {
                i_y--;
                i_ddF_y += 2;
                i_f += i_ddF_y;
            }
            x++;
            ddF_x += 2;
            f += ddF_x + 1;

            i_ddF_x += 2;
            i_f += i_ddF_x + 1;

            if (thickness > 1)
                thickness = y - i_y;

            /* Numbers represent parts of circle function draw in radians
            interval: [number - 1 * pi / 4, number * pi / 4] */
            if (top_right > 0) {
                for (i = 0; i < thickness; i++) {
                    y1 = y - i;
                    if ((y0 - y1) < (y0 - x))
                        set_and_check_rect(surf, x0 + x - 1, y0 - y1, color,
                               drawn_area); /* 2 */
                    if ((x0 + y1 - 1) >= (x0 + x - 1))
                        set_and_check_rect(surf, x0 + y1 - 1, y0 - x, color,
                               drawn_area); /* 1 */
                }
            }
            if (top_left > 0) {
                for (i = 0; i < thickness; i++) {
                    y1 = y - i;
                    if ((y0 - y1) <= (y0 - x))
                        set_and_check_rect(surf, x0 - x, y0 - y1, color,
                               drawn_area); /* 3 */
                    if ((x0 - y1) < (x0 - x))
                        set_and_check_rect(surf, x0 - y1, y0 - x, color,
                               drawn_area); /* 4 */
                }
            }
            if (bottom_left > 0) {
                for (i = 0; i < thickness; i++) {
                    y1 = y - i;
                    if ((x0 - y1) <= (x0 - x))
                        set_and_check_rect(surf, x0 - y1, y0 + x - 1, color,
                               drawn_area); /* 5 */
                    if ((y0 + y1 - 1) > (y0 + x - 1))
                        set_and_check_rect(surf, x0 - x, y0 + y1 - 1, color,
                               drawn_area); /* 6 */
                }
            }
            if (bottom_right > 0) {
                for (i = 0; i < thickness; i++) {
                    y1 = y - i;
                    if ((y0 + y1 - 1) >= (y0 + x - 1))
                        set_and_check_rect(surf, x0 + x - 1, y0 + y1 - 1, color,
                               drawn_area); /* 7 */
                    if ((x0 + y1 - 1) > (x0 + x - 1))
                        set_and_check_rect(surf, x0 + y1 - 1, y0 + x - 1, color,
                               drawn_area); /* 8 */
                }
            }
        }
    }
    else {
        while (x < y) {
            if (f >= 0) {
                y--;
                ddF_y += 2;
                f += ddF_y;
            }
            x++;
            ddF_x += 2;
            f += ddF_x + 1;
            if (top_right > 0) {
                for (y1 = y0 - x; y1 <= y0; y1++) {
                    set_and_check_rect(surf, x0 + y - 1, y1, color, drawn_area); /* 1 */
                }
                for (y1 = y0 - y; y1 <= y0; y1++) {
                    set_and_check_rect(surf, x0 + x - 1, y1, color, drawn_area); /* 2 */
                }
            }
            if (top_left > 0) {
                for (y1 = y0 - x; y1 <= y0; y1++) {
                    set_and_check_rect(surf, x0 - y, y1, color, drawn_area); /* 4 */
                }
                for (y1 = y0 - y; y1 <= y0; y1++) {
                    set_and_check_rect(surf, x0 - x, y1, color, drawn_area); /* 3 */
                }
            }
            if (bottom_left > 0) {
                for (y1 = y0; y1 < y0 + x; y1++) {
                    set_and_check_rect(surf, x0 - y, y1, color, drawn_area); /* 4 */
                }
                for (y1 = y0; y1 < y0 + y; y1++) {
                    set_and_check_rect(surf, x0 - x, y1, color, drawn_area); /* 3 */
                }
            }
            if (bottom_right > 0) {
                for (y1 = y0; y1 < y0 + x; y1++) {
                    set_and_check_rect(surf, x0 + y - 1, y1, color, drawn_area); /* 1 */
                }
                for (y1 = y0; y1 < y0 + y; y1++) {
                    set_and_check_rect(surf, x0 + x - 1, y1, color, drawn_area); /* 2 */
                }
            }
        }
    }
}

static void
draw_circle_filled(SDL_Surface *surf, int x0, int y0, int radius, Uint32 color,
                   int *drawn_area)
{
    int f = 1 - radius;
    int ddF_x = 0;
    int ddF_y = -2 * radius;
    int x = 0;
    int y = radius;

    while (x < y) {
        if (f >= 0) {
            y--;
            ddF_y += 2;
            f += ddF_y;
        }
        x++;
        ddF_x += 2;
        f += ddF_x + 1;

        /* optimisation to avoid overdrawing and repeated return rect checks:
           only draw a line if y-step is about to be decreased. */
        if (f >= 0) {
            drawhorzlineclip(surf, color, x0 - x, y0 + y - 1, x0 + x -1, drawn_area);
            drawhorzlineclip(surf, color, x0 - x, y0 - y, x0 + x -1, drawn_area);
        }
        drawhorzlineclip(surf, color, x0 - y, y0 + x - 1, x0 + y -1, drawn_area);
        drawhorzlineclip(surf, color, x0 - y, y0 - x, x0 + y -1, drawn_area);
    }
}

static void
draw_ellipse_filled(SDL_Surface *surf, int x0, int y0, int width, int height,
                    Uint32 color, int *drawn_area)
{
    int dx, dy, x, y, x_offset, y_offset;
    double d1, d2;
    if (width == 1) {
        draw_line(surf, x0, y0, x0, y0 + height - 1, color, drawn_area);
        return;
    }
    if (height == 1) {
        drawhorzlineclip(surf, color, x0, y0, x0 + width - 1, drawn_area);
        return;
    }
    x0 = x0 + width / 2;
    y0 = y0 + height / 2;
    x_offset = (width + 1) % 2;
    y_offset = (height + 1) % 2;
    width = width / 2;
    height = height / 2;
    x = 0;
    y = height;
    d1 = (height * height) - (width * width * height) + (0.25 * width * width);
    dx = 2 * height * height * x;
    dy = 2 * width * width * y;
    while (dx < dy) {
        drawhorzlineclip(surf, color, x0 - x, y0 - y, x0 + x - x_offset, drawn_area);
        drawhorzlineclip(surf, color, x0 - x, y0 + y - y_offset, x0 + x - x_offset, drawn_area);
        if (d1 < 0) {
            x++;
            dx = dx + (2 * height * height);
            d1 = d1 + dx + (height * height);
        }
        else {
            x++;
            y--;
            dx = dx + (2 * height * height);
            dy = dy - (2 * width * width);
            d1 = d1 + dx - dy + (height * height);
        }
    }
    d2 = (((double) height * height) * ((x + 0.5) * (x + 0.5))) +
         (((double) width * width) * ((y - 1) * (y - 1))) -
         ((double) width * width * height * height);
    while (y >= 0) {
        drawhorzlineclip(surf, color, x0 - x, y0 - y, x0 + x - x_offset, drawn_area);
        drawhorzlineclip(surf, color, x0 - x, y0 + y - y_offset, x0 + x - x_offset, drawn_area);
        if (d2 > 0) {
            y--;
            dy = dy - (2 * width * width);
            d2 = d2 + (width * width) - dy;
        }
        else {
            y--;
            x++;
            dx = dx + (2 * height * height);
            dy = dy - (2 * width * width);
            d2 = d2 + dx - dy + (width * width);
        }
    }
}

static void
draw_ellipse_thickness(SDL_Surface *surf, int x0, int y0, int width, int height,
                       int thickness, Uint32 color, int *drawn_area)
{
    int dx, dy, x, y, dx_inner, dy_inner, x_inner, y_inner, line, x_offset, y_offset;
    double d1, d2, d1_inner, d2_inner = 0;
    x0 = x0 + width / 2;
    y0 = y0 + height / 2;
    x_offset = (width + 1) % 2;
    y_offset = (height + 1) % 2;
    width = width / 2;
    height = height / 2;
    line = 1;
    x = 0;
    y = height;
    x_inner = 0;
    y_inner = height - thickness;
    d1 = (height * height) - (width * width * height) + (0.25 * width * width);
    d1_inner = ((height - thickness) * (height - thickness)) -
               ((width - thickness) * (width - thickness) * (height - thickness)) +
               (0.25 * (width - thickness) * (width - thickness));
    dx = 2 * height * height * x;
    dy = 2 * width * width * y;
    dx_inner = 2 * (height - thickness) * (height - thickness) * x_inner;
    dy_inner = 2 * (width - thickness) * (width - thickness) * y_inner;
    while (dx < dy) {
        if (line) {
            drawhorzlineclip(surf, color, x0 - x, y0 - y, x0 + x - x_offset, drawn_area);
            drawhorzlineclip(surf, color, x0 - x, y0 + y - y_offset, x0 + x - x_offset, drawn_area);
        }
        else {
            drawhorzlineclip(surf, color, x0 - x, y0 - y, x0 - x_inner, drawn_area);
            drawhorzlineclip(surf, color, x0 - x, y0 + y - y_offset, x0 - x_inner, drawn_area);
            drawhorzlineclip(surf, color, x0 + x - x_offset, y0 - y, x0 + x_inner - x_offset, drawn_area);
            drawhorzlineclip(surf, color, x0 + x - x_offset, y0 + y - y_offset, x0 + x_inner - x_offset, drawn_area);
        }
        if (d1 < 0) {
            x++;
            dx = dx + (2 * height * height);
            d1 = d1 + dx + (height * height);
        }
        else {
            x++;
            y--;
            dx = dx + (2 * height * height);
            dy = dy - (2 * width * width);
            d1 = d1 + dx - dy + (height * height);
            if (line && y < height - thickness) {
                line = 0;
            }
            if (!line) {
                if (dx_inner < dy_inner) {
                    while (d1_inner < 0) {
                        x_inner++;
                        dx_inner = dx_inner + (2 * (height - thickness) * (height - thickness));
                        d1_inner = d1_inner + dx_inner + ((height - thickness) * (height - thickness));
                    }
                    x_inner++;
                    y_inner--;
                    dx_inner = dx_inner + (2 * (height - thickness) * (height - thickness));
                    dy_inner = dy_inner - (2 * (width - thickness) * (width - thickness));
                    d1_inner = d1_inner + dx_inner - dy_inner + ((height - thickness) * (height - thickness));
                }
            }
        }
    }
    d2 = (((double) height * height) * ((x + 0.5) * (x + 0.5))) +
         (((double) width * width) * ((y - 1) * (y - 1))) -
         ((double) width * width * height * height);
    while (y >= 0) {
        if (line) {
            drawhorzlineclip(surf, color, x0 - x, y0 - y, x0 + x - x_offset, drawn_area);
            drawhorzlineclip(surf, color, x0 - x, y0 + y - y_offset, x0 + x - x_offset, drawn_area);
        }
        else {
            drawhorzlineclip(surf, color, x0 - x, y0 - y, x0 - x_inner, drawn_area);
            drawhorzlineclip(surf, color, x0 - x, y0 + y - y_offset, x0 - x_inner, drawn_area);
            drawhorzlineclip(surf, color, x0 + x - x_offset, y0 - y, x0 + x_inner - x_offset, drawn_area);
            drawhorzlineclip(surf, color, x0 + x - x_offset, y0 + y - y_offset, x0 + x_inner - x_offset, drawn_area);
        }
        if (d2 > 0) {
            y--;
            dy = dy - (2 * width * width);
            d2 = d2 + (width * width) - dy;
        }
        else {
            y--;
            x++;
            dx = dx + (2 * height * height);
            dy = dy - (2 * width * width);
            d2 = d2 + dx - dy + (width * width);
        }
        if (line && y < height - thickness) {
            line = 0;
        }
        if (!line) {
            if (dx_inner < dy_inner) {
                while (d1_inner < 0) {
                    x_inner++;
                    dx_inner = dx_inner + (2 * (height - thickness) * (height - thickness));
                    d1_inner = d1_inner + dx_inner + ((height - thickness) * (height - thickness));
                }
                x_inner++;
                y_inner--;
                dx_inner = dx_inner + (2 * (height - thickness) * (height - thickness));
                dy_inner = dy_inner - (2 * (width - thickness) * (width - thickness));
                d1_inner = d1_inner + dx_inner - dy_inner + ((height - thickness) * (height - thickness));
            }
            else if (y_inner >= 0) {
                if (d2_inner == 0) {
                    d2_inner = ((((double) height - thickness) * (height - thickness)) * ((x_inner + 0.5) * (x_inner + 0.5))) +
                               ((((double) width - thickness) * (width - thickness)) * ((y_inner - 1) * (y_inner - 1))) -
                               (((double) width - thickness) * (width - thickness) * (height - thickness) * (height - thickness));
                }
                if (d2_inner > 0) {
                    y_inner--;
                    dy_inner = dy_inner - (2 * (width - thickness) * (width - thickness));
                    d2_inner = d2_inner + ((width - thickness) * (width - thickness)) - dy_inner;
                }
                else {
                    y_inner--;
                    x_inner++;
                    dx_inner = dx_inner + (2 * (height - thickness) * (height - thickness));
                    dy_inner = dy_inner - (2 * (width - thickness) * (width - thickness));
                    d2_inner = d2_inner + dx_inner - dy_inner + ((width - thickness) * (width - thickness));
                }
            }
        }
    }
}

static void
draw_fillpoly(SDL_Surface *surf, int *point_x, int *point_y,
              Py_ssize_t num_points, Uint32 color, int *drawn_area)
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
        draw_line(surf, minx, miny, maxx, miny, color, drawn_area);
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
     *    (draw line for a pair of two such points)
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
            draw_line(surf, x_intersect[i], y, x_intersect[i + 1], y, color,
                     drawn_area);
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
            draw_line(surf, point_x[i], y, point_x[i_previous], y, color,
                     drawn_area);
        }
    }
    PyMem_Free(x_intersect);
}

static void
draw_round_rect(SDL_Surface *surf, int x1, int y1, int x2, int y2, int radius,
                int width, Uint32 color, int top_left, int top_right,
                int bottom_left, int bottom_right, int *drawn_area)
{
    int pts[16], i;
    float q_top, q_left, q_bottom, q_right, f;
    if (top_left < 0)
        top_left = radius;
    if (top_right < 0)
        top_right = radius;
    if (bottom_left < 0)
        bottom_left = radius;
    if (bottom_right < 0)
        bottom_right = radius;
    if ((top_left + top_right) > (x2 - x1 + 1) ||
        (bottom_left + bottom_right) > (x2 - x1 + 1) ||
        (top_left + bottom_left) > (y2 - y1 + 1) ||
        (top_right + bottom_right) > (y2 - y1 + 1)) {
        q_top = (x2 - x1 + 1) / (float)(top_left + top_right);
        q_left = (y2 - y1 + 1) / (float)(top_left + bottom_left);
        q_bottom = (x2 - x1 + 1) / (float)(bottom_left + bottom_right);
        q_right = (y2 - y1 + 1) / (float)(top_right + bottom_right);
        f = MIN(MIN(MIN(q_top, q_left), q_bottom), q_right);
        top_left = (int)(top_left * f);
        top_right = (int)(top_right * f);
        bottom_left = (int)(bottom_left * f);
        bottom_right = (int)(bottom_right * f);
    }
    if (width == 0) { /* Filled rect */
        pts[0] = x1;
        pts[1] = x1 + top_left;
        pts[2] = x2 - top_right;
        pts[3] = x2;
        pts[4] = x2;
        pts[5] = x2 - bottom_right;
        pts[6] = x1 + bottom_left;
        pts[7] = x1;
        pts[8] = y1 + top_left;
        pts[9] = y1;
        pts[10] = y1;
        pts[11] = y1 + top_right;
        pts[12] = y2 - bottom_right;
        pts[13] = y2;
        pts[14] = y2;
        pts[15] = y2 - bottom_left;
        draw_fillpoly(surf, pts, pts + 8, 8, color, drawn_area);
        draw_circle_quadrant(surf, x2 - top_right + 1, y1 + top_right,
                             top_right, 0, color, 1, 0, 0, 0, drawn_area);
        draw_circle_quadrant(surf, x1 + top_left, y1 + top_left, top_left, 0,
                             color, 0, 1, 0, 0, drawn_area);
        draw_circle_quadrant(surf, x1 + bottom_left, y2 - bottom_left + 1,
                             bottom_left, 0, color, 0, 0, 1, 0, drawn_area);
        draw_circle_quadrant(surf, x2 - bottom_right + 1, y2 - bottom_right + 1,
                             bottom_right, 0, color, 0, 0, 0, 1, drawn_area);
    }
    else {
        if (x2 - top_right == x1 + top_left) {
            for (i = 0; i < width; i++) {
                set_and_check_rect(surf, x1 + top_left, y1 + i, color,
                       drawn_area); /* Fill gap if reduced radius */
            }
        }
        else
            draw_line_width(surf, color, x1 + top_left, y1 + (int)(width / 2) - 1 + width % 2,
                            x2 - top_right, y1 + (int)(width / 2) - 1 + width % 2, width,
                            drawn_area); /* Top line */
        if (y2 - bottom_left == y1 + top_left) {
            for (i = 0; i < width; i++) {
                set_and_check_rect(surf, x1 + i, y1 + top_left, color,
                       drawn_area); /* Fill gap if reduced radius */
            }
        }
        else
            draw_line_width(surf, color, x1 + (int)(width / 2) - 1 + width % 2,
                            y1 + top_left, x1 + (int)(width / 2) - 1 + width % 2,
                            y2 - bottom_left, width, drawn_area); /* Left line */
        if (x2 - bottom_right == x1 + bottom_left) {
            for (i = 0; i < width; i++) {
                set_and_check_rect(surf, x1 + bottom_left, y2 - i, color,
                       drawn_area); /* Fill gap if reduced radius */
            }
        }
        else
            draw_line_width(surf, color, x1 + bottom_left, y2 - (int)(width / 2),
                            x2 - bottom_right, y2 - (int)(width / 2), width,
                            drawn_area); /* Bottom line */
        if (y2 - bottom_right == y1 + top_right) {
            for (i = 0; i < width; i++) {
                set_and_check_rect(surf, x2 - i, y1 + top_right, color,
                       drawn_area); /* Fill gap if reduced radius */
            }
        }
        else
            draw_line_width(surf, color, x2 - (int)(width / 2), y1 + top_right,
                            x2 - (int)(width / 2), y2 - bottom_right, width,
                            drawn_area); /* Right line */

        draw_circle_quadrant(surf, x2 - top_right + 1, y1 + top_right,
                             top_right, width, color, 1, 0, 0, 0, drawn_area);
        draw_circle_quadrant(surf, x1 + top_left, y1 + top_left, top_left,
                             width, color, 0, 1, 0, 0, drawn_area);
        draw_circle_quadrant(surf, x1 + bottom_left, y2 - bottom_left + 1,
                             bottom_left, width, color, 0, 0, 1, 0,
                             drawn_area);
        draw_circle_quadrant(surf, x2 - bottom_right + 1, y2 - bottom_right + 1,
                             bottom_right, width, color, 0, 0, 0, 1,
                             drawn_area);
    }
}

/* List of python functions */
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
    {"arc", (PyCFunction)arc, METH_VARARGS | METH_KEYWORDS,
     DOC_PYGAMEDRAWARC},
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
