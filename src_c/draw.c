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

#include <limits.h>  // for CHAR_BIT
#include <math.h>

#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Declaration of drawing algorithms */
static void
draw_line_width(SDL_Surface *surf, SDL_Rect surf_clip_rect, Uint32 color,
                int x1, int y1, int x2, int y2, int width, int *drawn_area);
static void
draw_line(SDL_Surface *surf, SDL_Rect surf_clip_rect, int x1, int y1, int x2,
          int y2, Uint32 color, int *drawn_area);
static void
draw_aaline(SDL_Surface *surf, SDL_Rect surf_clip_rect,
            PG_PixelFormat *surf_format, Uint32 color, float startx,
            float starty, float endx, float endy, int *drawn_area,
            int disable_first_endpoint, int disable_second_endpoint,
            int extra_pixel_for_aalines);
static void
draw_aaline_width(SDL_Surface *surf, SDL_Rect surf_clip_rect,
                  PG_PixelFormat *surf_format, Uint32 color, float from_x,
                  float from_y, float to_x, float to_y, int width,
                  int *drawn_area);
static void
draw_arc(SDL_Surface *surf, SDL_Rect surf_clip_rect, int x_center,
         int y_center, int radius1, int radius2, int width, double angle_start,
         double angle_stop, Uint32 color, int *drawn_area);
static void
draw_circle_bresenham(SDL_Surface *surf, SDL_Rect surf_clip_rect, int x0,
                      int y0, int radius, int thickness, Uint32 color,
                      int *drawn_area);
static void
draw_circle_bresenham_thin(SDL_Surface *surf, SDL_Rect surf_clip_rect, int x0,
                           int y0, int radius, Uint32 color, int *drawn_area);
static void
draw_circle_xiaolinwu(SDL_Surface *surf, SDL_Rect surf_clip_rect,
                      PG_PixelFormat *surf_format, int x0, int y0, int radius,
                      int thickness, Uint32 color, int top_right, int top_left,
                      int bottom_left, int bottom_right, int *drawn_area);
static void
draw_circle_xiaolinwu_thin(SDL_Surface *surf, SDL_Rect surf_clip_rect,
                           PG_PixelFormat *surf_format, int x0, int y0,
                           int radius, Uint32 color, int top_right,
                           int top_left, int bottom_left, int bottom_right,
                           int *drawn_area);
static void
draw_circle_filled(SDL_Surface *surf, SDL_Rect surf_clip_rect, int x0, int y0,
                   int radius, Uint32 color, int *drawn_area);
static void
draw_circle_quadrant(SDL_Surface *surf, SDL_Rect surf_clip_rect, int x0,
                     int y0, int radius, int thickness, Uint32 color,
                     int top_right, int top_left, int bottom_left,
                     int bottom_right, int *drawn_area);
static void
draw_ellipse_filled(SDL_Surface *surf, SDL_Rect surf_clip_rect, int x0, int y0,
                    int width, int height, Uint32 color, int *drawn_area);
static void
draw_ellipse_thickness(SDL_Surface *surf, SDL_Rect surf_clip_rect, int x0,
                       int y0, int width, int height, int thickness,
                       Uint32 color, int *drawn_area);
static void
draw_fillpoly(SDL_Surface *surf, SDL_Rect surf_clip_rect, int *vx, int *vy,
              Py_ssize_t n, Uint32 color, int *drawn_area);
static int
draw_filltri(SDL_Surface *surf, SDL_Rect surf_clip_rect, int *xlist,
             int *ylist, Uint32 color, int *drawn_area);
static void
draw_rect(SDL_Surface *surf, SDL_Rect surf_clip_rect, int x1, int y1, int x2,
          int y2, int width, Uint32 color);
static void
draw_round_rect(SDL_Surface *surf, SDL_Rect surf_clip_rect, int x1, int y1,
                int x2, int y2, int radius, int width, Uint32 color,
                int top_left, int top_right, int bottom_left, int bottom_right,
                int *drawn_area);

static int
flood_fill_inner(SDL_Surface *surf, int x1, int y1, Uint32 new_color,
                 SDL_Surface *pattern, int *drawn_area);

static void
unsafe_set_at(SDL_Surface *surf, int x, int y, Uint32 color);

// validation of a draw color
#define CHECK_LOAD_COLOR(colorobj)                       \
    if (!pg_MappedColorFromObj((colorobj), surf, &color, \
                               PG_COLOR_HANDLE_ALL)) {   \
        return NULL;                                     \
    }

/* Definition of functions that get called in Python */

/* Draws an antialiased line on the given surface.
 *
 * Returns a Rect bounding the drawn area.
 */
static PyObject *
aaline(PyObject *self, PyObject *arg, PyObject *kwargs)
{
    pgSurfaceObject *surfobj;
    PyObject *colorobj, *start, *end;
    SDL_Surface *surf = NULL;
    float startx, starty, endx, endy;
    int width = 1; /* Default width. */
    PyObject *blend = NULL;
    int drawn_area[4] = {INT_MAX, INT_MAX, INT_MIN,
                         INT_MIN}; /* Used to store bounding box values */
    Uint32 color;
    static char *keywords[] = {"surface", "color", "start_pos", "end_pos",
                               "width",   "blend", NULL};

    // blend argument is keyword only for backcompat.
    // if it is passed as a positional argument, it will be handled in width
    if (!PyArg_ParseTupleAndKeywords(arg, kwargs, "O!OOO|i$O", keywords,
                                     &pgSurface_Type, &surfobj, &colorobj,
                                     &start, &end, &width, &blend)) {
        return NULL; /* Exception already set. */
    }

    if (blend != NULL) {
        if (PyErr_WarnEx(
                PyExc_DeprecationWarning,
                "blend argument is deprecated and has no functionality and "
                "will be completely removed in a future version of pygame",
                1) == -1) {
            return NULL;
        }
    }

    surf = pgSurface_AsSurface(surfobj);
    SURF_INIT_CHECK(surf)

    if (PG_SURF_BytesPerPixel(surf) <= 0 || PG_SURF_BytesPerPixel(surf) > 4) {
        return PyErr_Format(PyExc_ValueError,
                            "unsupported surface bit depth (%d) for drawing",
                            PG_SURF_BytesPerPixel(surf));
    }

    SDL_Rect surf_clip_rect;
    if (!PG_GetSurfaceClipRect(surf, &surf_clip_rect)) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    PG_PixelFormat *surf_format = PG_GetSurfaceFormat(surf);
    if (surf_format == NULL) {
        return RAISE(pgExc_SDLError, SDL_GetError());
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

    if (width > 1) {
        draw_aaline_width(surf, surf_clip_rect, surf_format, color, startx,
                          starty, endx, endy, width, drawn_area);
    }
    else {
        // For all width <= 1 treat it as width == 1, this helps compat
        // with the old blend argument
        draw_aaline(surf, surf_clip_rect, surf_format, color, startx, starty,
                    endx, endy, drawn_area, 0, 0, 0);
    }

    if (!pgSurface_Unlock(surfobj)) {
        return RAISE(PyExc_RuntimeError, "error unlocking surface");
    }

    if (drawn_area[0] != INT_MAX && drawn_area[1] != INT_MAX &&
        drawn_area[2] != INT_MIN && drawn_area[3] != INT_MIN) {
        return pgRect_New4(drawn_area[0], drawn_area[1],
                           drawn_area[2] - drawn_area[0] + 1,
                           drawn_area[3] - drawn_area[1] + 1);
    }
    else {
        return pgRect_New4((int)startx, (int)starty, 0, 0);
    }
}

/* Draws a line on the given surface.
 *
 * Returns a Rect bounding the drawn area.
 */
static PyObject *
line(PyObject *self, PyObject *arg, PyObject *kwargs)
{
    pgSurfaceObject *surfobj;
    PyObject *colorobj, *start, *end;
    SDL_Surface *surf = NULL;
    int startx, starty, endx, endy;
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
    SURF_INIT_CHECK(surf)

    if (PG_SURF_BytesPerPixel(surf) <= 0 || PG_SURF_BytesPerPixel(surf) > 4) {
        return PyErr_Format(PyExc_ValueError,
                            "unsupported surface bit depth (%d) for drawing",
                            PG_SURF_BytesPerPixel(surf));
    }

    SDL_Rect surf_clip_rect;
    if (!PG_GetSurfaceClipRect(surf, &surf_clip_rect)) {
        return RAISE(pgExc_SDLError, SDL_GetError());
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

    draw_line_width(surf, surf_clip_rect, color, startx, starty, endx, endy,
                    width, drawn_area);

    if (!pgSurface_Unlock(surfobj)) {
        return RAISE(PyExc_RuntimeError, "error unlocking surface");
    }

    /* Compute return rect. */
    if (drawn_area[0] != INT_MAX && drawn_area[1] != INT_MAX &&
        drawn_area[2] != INT_MIN && drawn_area[3] != INT_MIN) {
        return pgRect_New4(drawn_area[0], drawn_area[1],
                           drawn_area[2] - drawn_area[0] + 1,
                           drawn_area[3] - drawn_area[1] + 1);
    }
    else {
        return pgRect_New4(startx, starty, 0, 0);
    }
}

/* Draws a series of antialiased lines on the given surface.
 *
 * Returns a Rect bounding the drawn area.
 */
static PyObject *
aalines(PyObject *self, PyObject *arg, PyObject *kwargs)
{
    pgSurfaceObject *surfobj;
    PyObject *colorobj;
    PyObject *points, *item = NULL;
    SDL_Surface *surf = NULL;
    Uint32 color;
    float pts[4];
    float pts_prev[4];
    float x, y;
    int l, t;
    int extra_px;
    int disable_endpoints;
    int steep_prev;
    int steep_curr;
    PyObject *blend = NULL;
    int drawn_area[4] = {INT_MAX, INT_MAX, INT_MIN,
                         INT_MIN}; /* Used to store bounding box values */
    int result, closed;
    Py_ssize_t loop, length;
    static char *keywords[] = {"surface", "color", "closed",
                               "points",  "blend", NULL};

    if (!PyArg_ParseTupleAndKeywords(arg, kwargs, "O!OpO|O", keywords,
                                     &pgSurface_Type, &surfobj, &colorobj,
                                     &closed, &points, &blend)) {
        return NULL; /* Exception already set. */
    }

    if (blend != NULL) {
        if (PyErr_WarnEx(
                PyExc_DeprecationWarning,
                "blend argument is deprecated and has no functionality and "
                "will be completely removed in a future version of pygame",
                1) == -1) {
            return NULL;
        }
    }

    surf = pgSurface_AsSurface(surfobj);
    SURF_INIT_CHECK(surf)

    if (PG_SURF_BytesPerPixel(surf) <= 0 || PG_SURF_BytesPerPixel(surf) > 4) {
        return PyErr_Format(PyExc_ValueError,
                            "unsupported surface bit depth (%d) for drawing",
                            PG_SURF_BytesPerPixel(surf));
    }

    SDL_Rect surf_clip_rect;
    if (!PG_GetSurfaceClipRect(surf, &surf_clip_rect)) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    PG_PixelFormat *surf_format = PG_GetSurfaceFormat(surf);
    if (surf_format == NULL) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    CHECK_LOAD_COLOR(colorobj)

    if (!PySequence_Check(points)) {
        return RAISE(PyExc_TypeError,
                     "points argument must be a sequence of number pairs");
    }

    length = PySequence_Length(points);

    if (length < 2) {
        return RAISE(PyExc_ValueError,
                     "points argument must contain 2 or more points");
    }

    // Allocate bytes for the xlist and ylist at once to reduce allocations.
    float *points_buf = PyMem_New(float, length * 2);
    float *xlist = points_buf;
    float *ylist = points_buf + length;

    if (points_buf == NULL) {
        return RAISE(PyExc_MemoryError,
                     "cannot allocate memory to draw aalines");
    }

    for (loop = 0; loop < length; ++loop) {
        item = PySequence_GetItem(points, loop);
        result = pg_TwoFloatsFromObj(item, &x, &y);
        if (loop == 0) {
            l = (int)x;
            t = (int)y;
        }
        Py_DECREF(item);

        if (!result) {
            PyMem_Free(points_buf);
            return RAISE(PyExc_TypeError, "points must be number pairs");
        }

        xlist[loop] = x;
        ylist[loop] = y;
    }

    if (!pgSurface_Lock(surfobj)) {
        PyMem_Free(points_buf);
        return RAISE(PyExc_RuntimeError, "error locking surface");
    }

    /* first line - if open, add endpoint pixels.*/
    pts[0] = xlist[0];
    pts[1] = ylist[0];
    pts[2] = xlist[1];
    pts[3] = ylist[1];

    /* Previous points.
     * Used to compare previous and current line.*/
    pts_prev[0] = pts[0];
    pts_prev[1] = pts[1];
    pts_prev[2] = pts[2];
    pts_prev[3] = pts[3];
    steep_prev =
        fabs(pts_prev[2] - pts_prev[0]) < fabs(pts_prev[3] - pts_prev[1]);
    steep_curr = fabs(xlist[2] - pts[2]) < fabs(ylist[2] - pts[1]);
    extra_px = steep_prev > steep_curr;
    disable_endpoints =
        !((roundf(pts[2]) == pts[2]) && (roundf(pts[3]) == pts[3]));
    if (closed) {
        draw_aaline(surf, surf_clip_rect, surf_format, color, pts[0], pts[1],
                    pts[2], pts[3], drawn_area, disable_endpoints,
                    disable_endpoints, extra_px);
    }
    else {
        draw_aaline(surf, surf_clip_rect, surf_format, color, pts[0], pts[1],
                    pts[2], pts[3], drawn_area, 0, disable_endpoints,
                    extra_px);
    }

    for (loop = 2; loop < length - 1; ++loop) {
        pts[0] = xlist[loop - 1];
        pts[1] = ylist[loop - 1];
        pts[2] = xlist[loop];
        pts[3] = ylist[loop];

        /* Comparing previous and current line.
         * If one is steep and other is not, extra pixel must be drawn.*/
        steep_prev =
            fabs(pts_prev[2] - pts_prev[0]) < fabs(pts_prev[3] - pts_prev[1]);
        steep_curr = fabs(pts[2] - pts[0]) < fabs(pts[3] - pts[1]);
        extra_px = steep_prev != steep_curr;
        disable_endpoints =
            !((roundf(pts[2]) == pts[2]) && (roundf(pts[3]) == pts[3]));
        pts_prev[0] = pts[0];
        pts_prev[1] = pts[1];
        pts_prev[2] = pts[2];
        pts_prev[3] = pts[3];
        draw_aaline(surf, surf_clip_rect, surf_format, color, pts[0], pts[1],
                    pts[2], pts[3], drawn_area, disable_endpoints,
                    disable_endpoints, extra_px);
    }

    /* Last line - if open, add endpoint pixels. */
    pts[0] = xlist[length - 2];
    pts[1] = ylist[length - 2];
    pts[2] = xlist[length - 1];
    pts[3] = ylist[length - 1];
    steep_prev =
        fabs(pts_prev[2] - pts_prev[0]) < fabs(pts_prev[3] - pts_prev[1]);
    steep_curr = fabs(pts[2] - pts[0]) < fabs(pts[3] - pts[1]);
    extra_px = steep_prev != steep_curr;
    disable_endpoints =
        !((roundf(pts[2]) == pts[2]) && (roundf(pts[3]) == pts[3]));
    pts_prev[0] = pts[0];
    pts_prev[1] = pts[1];
    pts_prev[2] = pts[2];
    pts_prev[3] = pts[3];
    if (closed) {
        draw_aaline(surf, surf_clip_rect, surf_format, color, pts[0], pts[1],
                    pts[2], pts[3], drawn_area, disable_endpoints,
                    disable_endpoints, extra_px);
    }
    else {
        draw_aaline(surf, surf_clip_rect, surf_format, color, pts[0], pts[1],
                    pts[2], pts[3], drawn_area, disable_endpoints, 0,
                    extra_px);
    }

    if (closed && length > 2) {
        pts[0] = xlist[length - 1];
        pts[1] = ylist[length - 1];
        pts[2] = xlist[0];
        pts[3] = ylist[0];
        steep_prev =
            fabs(pts_prev[2] - pts_prev[0]) < fabs(pts_prev[3] - pts_prev[1]);
        steep_curr = fabs(pts[2] - pts[0]) < fabs(pts[3] - pts[1]);
        extra_px = steep_prev != steep_curr;
        disable_endpoints =
            !((roundf(pts[2]) == pts[2]) && (roundf(pts[3]) == pts[3]));
        draw_aaline(surf, surf_clip_rect, surf_format, color, pts[0], pts[1],
                    pts[2], pts[3], drawn_area, disable_endpoints,
                    disable_endpoints, extra_px);
    }

    PyMem_Free(points_buf);

    if (!pgSurface_Unlock(surfobj)) {
        return RAISE(PyExc_RuntimeError, "error unlocking surface");
    }

    /* Compute return rect. */
    if (drawn_area[0] != INT_MAX && drawn_area[1] != INT_MAX &&
        drawn_area[2] != INT_MIN && drawn_area[3] != INT_MIN) {
        return pgRect_New4(drawn_area[0], drawn_area[1],
                           drawn_area[2] - drawn_area[0] + 1,
                           drawn_area[3] - drawn_area[1] + 1);
    }
    else {
        return pgRect_New4(l, t, 0, 0);
    }
}

/* Draws a series of lines on the given surface.
 *
 * Returns a Rect bounding the drawn area.
 */
static PyObject *
lines(PyObject *self, PyObject *arg, PyObject *kwargs)
{
    pgSurfaceObject *surfobj;
    PyObject *colorobj;
    PyObject *points, *item = NULL;
    SDL_Surface *surf = NULL;
    Uint32 color;
    int x, y, closed, result;
    int *xlist = NULL, *ylist = NULL;
    int width = 1; /* Default width. */
    Py_ssize_t loop, length;
    int drawn_area[4] = {INT_MAX, INT_MAX, INT_MIN,
                         INT_MIN}; /* Used to store bounding box values */
    static char *keywords[] = {"surface", "color", "closed",
                               "points",  "width", NULL};

    if (!PyArg_ParseTupleAndKeywords(arg, kwargs, "O!OpO|i", keywords,
                                     &pgSurface_Type, &surfobj, &colorobj,
                                     &closed, &points, &width)) {
        return NULL; /* Exception already set. */
    }

    surf = pgSurface_AsSurface(surfobj);
    SURF_INIT_CHECK(surf)

    if (PG_SURF_BytesPerPixel(surf) <= 0 || PG_SURF_BytesPerPixel(surf) > 4) {
        return PyErr_Format(PyExc_ValueError,
                            "unsupported surface bit depth (%d) for drawing",
                            PG_SURF_BytesPerPixel(surf));
    }

    SDL_Rect surf_clip_rect;
    if (!PG_GetSurfaceClipRect(surf, &surf_clip_rect)) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    CHECK_LOAD_COLOR(colorobj)

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
            PyMem_Free(xlist);
        }
        if (ylist) {
            PyMem_Free(ylist);
        }
        return RAISE(PyExc_MemoryError,
                     "cannot allocate memory to draw lines");
    }

    for (loop = 0; loop < length; ++loop) {
        item = PySequence_GetItem(points, loop);
        result = pg_TwoIntsFromObj(item, &x, &y);
        Py_DECREF(item);

        if (!result) {
            PyMem_Free(xlist);
            PyMem_Free(ylist);
            return RAISE(PyExc_TypeError, "points must be number pairs");
        }

        xlist[loop] = x;
        ylist[loop] = y;
    }

    x = xlist[0];
    y = ylist[0];

    if (width < 1) {
        PyMem_Free(xlist);
        PyMem_Free(ylist);
        return pgRect_New4(x, y, 0, 0);
    }

    if (!pgSurface_Lock(surfobj)) {
        PyMem_Free(xlist);
        PyMem_Free(ylist);
        return RAISE(PyExc_RuntimeError, "error locking surface");
    }

    for (loop = 1; loop < length; ++loop) {
        draw_line_width(surf, surf_clip_rect, color, xlist[loop - 1],
                        ylist[loop - 1], xlist[loop], ylist[loop], width,
                        drawn_area);
    }

    if (closed && length > 2) {
        draw_line_width(surf, surf_clip_rect, color, xlist[length - 1],
                        ylist[length - 1], xlist[0], ylist[0], width,
                        drawn_area);
    }

    PyMem_Free(xlist);
    PyMem_Free(ylist);

    if (!pgSurface_Unlock(surfobj)) {
        return RAISE(PyExc_RuntimeError, "error unlocking surface");
    }

    /* Compute return rect. */
    if (drawn_area[0] != INT_MAX && drawn_area[1] != INT_MAX &&
        drawn_area[2] != INT_MIN && drawn_area[3] != INT_MIN) {
        return pgRect_New4(drawn_area[0], drawn_area[1],
                           drawn_area[2] - drawn_area[0] + 1,
                           drawn_area[3] - drawn_area[1] + 1);
    }
    else {
        return pgRect_New4(x, y, 0, 0);
    }
}

static PyObject *
arc(PyObject *self, PyObject *arg, PyObject *kwargs)
{
    pgSurfaceObject *surfobj;
    PyObject *colorobj, *rectobj;
    SDL_Rect *rect = NULL, temp;
    SDL_Surface *surf = NULL;
    Uint32 color;
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
    SURF_INIT_CHECK(surf)

    if (PG_SURF_BytesPerPixel(surf) <= 0 || PG_SURF_BytesPerPixel(surf) > 4) {
        return PyErr_Format(PyExc_ValueError,
                            "unsupported surface bit depth (%d) for drawing",
                            PG_SURF_BytesPerPixel(surf));
    }

    SDL_Rect surf_clip_rect;
    if (!PG_GetSurfaceClipRect(surf, &surf_clip_rect)) {
        return RAISE(pgExc_SDLError, SDL_GetError());
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

    draw_arc(surf, surf_clip_rect, rect->x + rect->w / 2,
             rect->y + rect->h / 2, rect->w / 2, rect->h / 2, width,
             angle_start, angle_stop, color, drawn_area);

    if (!pgSurface_Unlock(surfobj)) {
        return RAISE(PyExc_RuntimeError, "error unlocking surface");
    }

    /* Compute return rect. */
    if (drawn_area[0] != INT_MAX && drawn_area[1] != INT_MAX &&
        drawn_area[2] != INT_MIN && drawn_area[3] != INT_MIN) {
        return pgRect_New4(drawn_area[0], drawn_area[1],
                           drawn_area[2] - drawn_area[0] + 1,
                           drawn_area[3] - drawn_area[1] + 1);
    }
    else {
        return pgRect_New4(rect->x, rect->y, 0, 0);
    }
}

static PyObject *
ellipse(PyObject *self, PyObject *arg, PyObject *kwargs)
{
    pgSurfaceObject *surfobj;
    PyObject *colorobj, *rectobj;
    SDL_Rect *rect = NULL, temp;
    SDL_Surface *surf = NULL;
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
    SURF_INIT_CHECK(surf)

    if (PG_SURF_BytesPerPixel(surf) <= 0 || PG_SURF_BytesPerPixel(surf) > 4) {
        return PyErr_Format(PyExc_ValueError,
                            "unsupported surface bit depth (%d) for drawing",
                            PG_SURF_BytesPerPixel(surf));
    }

    SDL_Rect surf_clip_rect;
    if (!PG_GetSurfaceClipRect(surf, &surf_clip_rect)) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    CHECK_LOAD_COLOR(colorobj)

    if (width < 0) {
        return pgRect_New4(rect->x, rect->y, 0, 0);
    }

    if (!pgSurface_Lock(surfobj)) {
        return RAISE(PyExc_RuntimeError, "error locking surface");
    }

    if (!width ||
        width >= MIN(rect->w / 2 + rect->w % 2, rect->h / 2 + rect->h % 2)) {
        draw_ellipse_filled(surf, surf_clip_rect, rect->x, rect->y, rect->w,
                            rect->h, color, drawn_area);
    }
    else {
        draw_ellipse_thickness(surf, surf_clip_rect, rect->x, rect->y, rect->w,
                               rect->h, width - 1, color, drawn_area);
    }

    if (!pgSurface_Unlock(surfobj)) {
        return RAISE(PyExc_RuntimeError, "error unlocking surface");
    }

    if (drawn_area[0] != INT_MAX && drawn_area[1] != INT_MAX &&
        drawn_area[2] != INT_MIN && drawn_area[3] != INT_MIN) {
        return pgRect_New4(drawn_area[0], drawn_area[1],
                           drawn_area[2] - drawn_area[0] + 1,
                           drawn_area[3] - drawn_area[1] + 1);
    }
    else {
        return pgRect_New4(rect->x, rect->y, 0, 0);
    }
}

static PyObject *
circle(PyObject *self, PyObject *args, PyObject *kwargs)
{
    pgSurfaceObject *surfobj;
    PyObject *colorobj;
    SDL_Surface *surf = NULL;
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
                                     &top_left, &bottom_left, &bottom_right)) {
        return NULL; /* Exception already set. */
    }

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
    SURF_INIT_CHECK(surf)

    if (PG_SURF_BytesPerPixel(surf) <= 0 || PG_SURF_BytesPerPixel(surf) > 4) {
        return PyErr_Format(PyExc_ValueError,
                            "unsupported surface bit depth (%d) for drawing",
                            PG_SURF_BytesPerPixel(surf));
    }

    SDL_Rect surf_clip_rect;
    if (!PG_GetSurfaceClipRect(surf, &surf_clip_rect)) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    CHECK_LOAD_COLOR(colorobj)

    if (radius < 1 || width < 0) {
        return pgRect_New4(posx, posy, 0, 0);
    }

    if (width > radius) {
        width = radius;
    }

    if (posx > surf_clip_rect.x + surf_clip_rect.w + radius ||
        posx < surf_clip_rect.x - radius ||
        posy > surf_clip_rect.y + surf_clip_rect.h + radius ||
        posy < surf_clip_rect.y - radius) {
        return pgRect_New4(posx, posy, 0, 0);
    }

    if (!pgSurface_Lock(surfobj)) {
        return RAISE(PyExc_RuntimeError, "error locking surface");
    }

    if ((top_right == 0 && top_left == 0 && bottom_left == 0 &&
         bottom_right == 0)) {
        if (!width || width == radius) {
            draw_circle_filled(surf, surf_clip_rect, posx, posy, radius, color,
                               drawn_area);
        }
        else if (width == 1) {
            draw_circle_bresenham_thin(surf, surf_clip_rect, posx, posy,
                                       radius, color, drawn_area);
        }
        else {
            draw_circle_bresenham(surf, surf_clip_rect, posx, posy, radius,
                                  width, color, drawn_area);
        }
    }
    else {
        draw_circle_quadrant(surf, surf_clip_rect, posx, posy, radius, width,
                             color, top_right, top_left, bottom_left,
                             bottom_right, drawn_area);
    }

    if (!pgSurface_Unlock(surfobj)) {
        return RAISE(PyExc_RuntimeError, "error unlocking surface");
    }
    if (drawn_area[0] != INT_MAX && drawn_area[1] != INT_MAX &&
        drawn_area[2] != INT_MIN && drawn_area[3] != INT_MIN) {
        return pgRect_New4(drawn_area[0], drawn_area[1],
                           drawn_area[2] - drawn_area[0] + 1,
                           drawn_area[3] - drawn_area[1] + 1);
    }
    else {
        return pgRect_New4(posx, posy, 0, 0);
    }
}

static PyObject *
aacircle(PyObject *self, PyObject *args, PyObject *kwargs)
{
    pgSurfaceObject *surfobj;
    PyObject *colorobj;
    SDL_Surface *surf = NULL;
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
                                     &top_left, &bottom_left, &bottom_right)) {
        return NULL; /* Exception already set. */
    }

    if (!pg_TwoIntsFromObj(posobj, &posx, &posy)) {
        return RAISE(PyExc_TypeError,
                     "center argument must be a pair of numbers");
    }

    if (!pg_IntFromObj(radiusobj, &radius)) {
        return RAISE(PyExc_TypeError, "radius argument must be a number");
    }

    surf = pgSurface_AsSurface(surfobj);
    SURF_INIT_CHECK(surf)

    if (PG_SURF_BytesPerPixel(surf) <= 0 || PG_SURF_BytesPerPixel(surf) > 4) {
        return PyErr_Format(PyExc_ValueError,
                            "unsupported surface bit depth (%d) for drawing",
                            PG_SURF_BytesPerPixel(surf));
    }

    SDL_Rect surf_clip_rect;
    if (!PG_GetSurfaceClipRect(surf, &surf_clip_rect)) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    PG_PixelFormat *surf_format = PG_GetSurfaceFormat(surf);
    if (surf_format == NULL) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    CHECK_LOAD_COLOR(colorobj)

    if (radius < 1 || width < 0) {
        return pgRect_New4(posx, posy, 0, 0);
    }

    if (width > radius) {
        width = radius;
    }

    if (posx > surf_clip_rect.x + surf_clip_rect.w + radius ||
        posx < surf_clip_rect.x - radius ||
        posy > surf_clip_rect.y + surf_clip_rect.h + radius ||
        posy < surf_clip_rect.y - radius) {
        return pgRect_New4(posx, posy, 0, 0);
    }

    if (!pgSurface_Lock(surfobj)) {
        return RAISE(PyExc_RuntimeError, "error locking surface");
    }

    if ((top_right == 0 && top_left == 0 && bottom_left == 0 &&
         bottom_right == 0)) {
        if (!width || width == radius) {
            draw_circle_filled(surf, surf_clip_rect, posx, posy, radius - 1,
                               color, drawn_area);
            draw_circle_xiaolinwu(surf, surf_clip_rect, surf_format, posx,
                                  posy, radius, 2, color, 1, 1, 1, 1,
                                  drawn_area);
        }
        else if (width == 1) {
            draw_circle_xiaolinwu_thin(surf, surf_clip_rect, surf_format, posx,
                                       posy, radius, color, 1, 1, 1, 1,
                                       drawn_area);
        }
        else {
            draw_circle_xiaolinwu(surf, surf_clip_rect, surf_format, posx,
                                  posy, radius, width, color, 1, 1, 1, 1,
                                  drawn_area);
        }
    }
    else {
        if (!width || width == radius) {
            draw_circle_xiaolinwu(surf, surf_clip_rect, surf_format, posx,
                                  posy, radius, radius, color, top_right,
                                  top_left, bottom_left, bottom_right,
                                  drawn_area);
        }
        else if (width == 1) {
            draw_circle_xiaolinwu_thin(
                surf, surf_clip_rect, surf_format, posx, posy, radius, color,
                top_right, top_left, bottom_left, bottom_right, drawn_area);
        }
        else {
            draw_circle_xiaolinwu(surf, surf_clip_rect, surf_format, posx,
                                  posy, radius, width, color, top_right,
                                  top_left, bottom_left, bottom_right,
                                  drawn_area);
        }
    }

    if (!pgSurface_Unlock(surfobj)) {
        return RAISE(PyExc_RuntimeError, "error unlocking surface");
    }
    if (drawn_area[0] != INT_MAX && drawn_area[1] != INT_MAX &&
        drawn_area[2] != INT_MIN && drawn_area[3] != INT_MIN) {
        return pgRect_New4(drawn_area[0], drawn_area[1],
                           drawn_area[2] - drawn_area[0] + 1,
                           drawn_area[3] - drawn_area[1] + 1);
    }
    else {
        return pgRect_New4(posx, posy, 0, 0);
    }
}

static PyObject *
polygon(PyObject *self, PyObject *arg, PyObject *kwargs)
{
    pgSurfaceObject *surfobj;
    PyObject *colorobj, *points, *item = NULL;
    SDL_Surface *surf = NULL;
    Uint32 color;
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
    SURF_INIT_CHECK(surf)

    if (PG_SURF_BytesPerPixel(surf) <= 0 || PG_SURF_BytesPerPixel(surf) > 4) {
        return PyErr_Format(PyExc_ValueError,
                            "unsupported surface bit depth (%d) for drawing",
                            PG_SURF_BytesPerPixel(surf));
    }

    SDL_Rect surf_clip_rect;
    if (!PG_GetSurfaceClipRect(surf, &surf_clip_rect)) {
        return RAISE(pgExc_SDLError, SDL_GetError());
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

    // Allocate bytes for the xlist and ylist at once to reduce allocations.
    int *points_buf = PyMem_New(int, length * 2);
    int *xlist = points_buf;
    int *ylist = points_buf + length;

    if (points_buf == NULL) {
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
            PyMem_Free(points_buf);
            return RAISE(PyExc_TypeError, "points must be number pairs");
        }

        xlist[loop] = x;
        ylist[loop] = y;
    }

    if (!pgSurface_Lock(surfobj)) {
        PyMem_Free(points_buf);
        return RAISE(PyExc_RuntimeError, "error locking surface");
    }

    if (length != 3) {
        draw_fillpoly(surf, surf_clip_rect, xlist, ylist, length, color,
                      drawn_area);
    }
    else {
        draw_filltri(surf, surf_clip_rect, xlist, ylist, color, drawn_area);
    }
    PyMem_Free(points_buf);

    if (!pgSurface_Unlock(surfobj)) {
        return RAISE(PyExc_RuntimeError, "error unlocking surface");
    }

    if (drawn_area[0] != INT_MAX && drawn_area[1] != INT_MAX &&
        drawn_area[2] != INT_MIN && drawn_area[3] != INT_MIN) {
        return pgRect_New4(drawn_area[0], drawn_area[1],
                           drawn_area[2] - drawn_area[0] + 1,
                           drawn_area[3] - drawn_area[1] + 1);
    }
    else {
        return pgRect_New4(l, t, 0, 0);
    }
}

static PyObject *
rect(PyObject *self, PyObject *args, PyObject *kwargs)
{
    pgSurfaceObject *surfobj;
    PyObject *colorobj, *rectobj;
    SDL_Rect *rect = NULL, temp;
    SDL_Surface *surf = NULL;
    Uint32 color;
    int width = 0, radius = 0; /* Default values. */
    int top_left_radius = -1, top_right_radius = -1, bottom_left_radius = -1,
        bottom_right_radius = -1;
    SDL_Rect sdlrect;
    SDL_Rect clipped;
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
    SURF_INIT_CHECK(surf)

    if (PG_SURF_BytesPerPixel(surf) <= 0 || PG_SURF_BytesPerPixel(surf) > 4) {
        return PyErr_Format(PyExc_ValueError,
                            "unsupported surface bit depth (%d) for drawing",
                            PG_SURF_BytesPerPixel(surf));
    }

    SDL_Rect surf_clip_rect;
    if (!PG_GetSurfaceClipRect(surf, &surf_clip_rect)) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    CHECK_LOAD_COLOR(colorobj)

    if (width < 0) {
        return pgRect_New4(rect->x, rect->y, 0, 0);
    }

    /* If there isn't any rounded rect-ness OR the rect is really thin in one
       direction. The "really thin in one direction" check is necessary because
       draw_round_rect fails (draws something bad) on rects with a dimension
       that is 0 or 1 pixels across.*/
    if ((radius <= 0 && top_left_radius <= 0 && top_right_radius <= 0 &&
         bottom_left_radius <= 0 && bottom_right_radius <= 0) ||
        abs(rect->w) < 2 || abs(rect->h) < 2) {
        sdlrect.x = rect->x;
        sdlrect.y = rect->y;
        sdlrect.w = rect->w;
        sdlrect.h = rect->h;
        /* SDL_FillRect respects the clip rect already, but in order to
            return the drawn area, we need to do this here, and keep the
            pointer to the result in clipped */
        if (!SDL_IntersectRect(&sdlrect, &surf_clip_rect, &clipped)) {
            return pgRect_New4(rect->x, rect->y, 0, 0);
        }
        if (width > 0 && (width * 2) < clipped.w && (width * 2) < clipped.h) {
            draw_rect(surf, surf_clip_rect, sdlrect.x, sdlrect.y,
                      sdlrect.x + sdlrect.w - 1, sdlrect.y + sdlrect.h - 1,
                      width, color);
        }
        else {
            pgSurface_Prep(surfobj);
            pgSurface_Lock(surfobj);
            bool success = PG_FillSurfaceRect(surf, &clipped, color);
            pgSurface_Unlock(surfobj);
            pgSurface_Unprep(surfobj);
            if (!success) {
                return RAISE(pgExc_SDLError, SDL_GetError());
            }
        }
        return pgRect_New(&clipped);
    }
    else {
        if (!pgSurface_Lock(surfobj)) {
            return RAISE(PyExc_RuntimeError, "error locking surface");
        }

        /* Little bit to normalize the rect: this matters for the rounded
           rects, despite not mattering for the normal rects. */
        if (rect->w < 0) {
            rect->x += rect->w;
            rect->w = -rect->w;
        }
        if (rect->h < 0) {
            rect->y += rect->h;
            rect->h = -rect->h;
        }

        if (width > rect->w / 2 || width > rect->h / 2) {
            width = MAX(rect->w / 2, rect->h / 2);
        }

        draw_round_rect(surf, surf_clip_rect, rect->x, rect->y,
                        rect->x + rect->w - 1, rect->y + rect->h - 1, radius,
                        width, color, top_left_radius, top_right_radius,
                        bottom_left_radius, bottom_right_radius, drawn_area);
        if (!pgSurface_Unlock(surfobj)) {
            return RAISE(PyExc_RuntimeError, "error unlocking surface");
        }
    }

    if (drawn_area[0] != INT_MAX && drawn_area[1] != INT_MAX &&
        drawn_area[2] != INT_MIN && drawn_area[3] != INT_MIN) {
        return pgRect_New4(drawn_area[0], drawn_area[1],
                           drawn_area[2] - drawn_area[0] + 1,
                           drawn_area[3] - drawn_area[1] + 1);
    }
    else {
        return pgRect_New4(rect->x, rect->y, 0, 0);
    }
}

static PyObject *
flood_fill(PyObject *self, PyObject *arg, PyObject *kwargs)
{
    pgSurfaceObject *surfobj;
    pgSurfaceObject *pat_surfobj = NULL;
    PyObject *colorobj, *start;
    SDL_Surface *surf = NULL;
    int startx, starty;
    Uint32 color;
    SDL_Surface *pattern = NULL;
    SDL_bool did_lock_surf = SDL_FALSE;
    SDL_bool did_lock_pat = SDL_FALSE;
    int flood_fill_result;

    int drawn_area[4] = {INT_MAX, INT_MAX, INT_MIN,
                         INT_MIN}; /* Used to store bounding box values */
    static char *keywords[] = {"surface", "color", "start_pos", NULL};

    if (!PyArg_ParseTupleAndKeywords(arg, kwargs, "O!OO", keywords,
                                     &pgSurface_Type, &surfobj, &colorobj,
                                     &start)) {
        return NULL; /* Exception already set. */
    }

    surf = pgSurface_AsSurface(surfobj);
    SURF_INIT_CHECK(surf)

    if (PG_SURF_BytesPerPixel(surf) <= 0 || PG_SURF_BytesPerPixel(surf) > 4) {
        return PyErr_Format(PyExc_ValueError,
                            "unsupported surface bit depth (%d) for drawing",
                            PG_SURF_BytesPerPixel(surf));
    }

    if (pgSurface_Check(colorobj)) {
        pat_surfobj = ((pgSurfaceObject *)colorobj);

        pattern = PG_ConvertSurface(pat_surfobj->surf, surf->format);

        if (pattern == NULL) {
            return RAISE(PyExc_RuntimeError, "error converting pattern surf");
        }

        SDL_SetSurfaceRLE(pattern, SDL_FALSE);

        color = 0;
    }
    else {
        CHECK_LOAD_COLOR(colorobj);
    }

    if (!pg_TwoIntsFromObj(start, &startx, &starty)) {
        return RAISE(PyExc_TypeError, "invalid start_pos argument");
    }

    if (SDL_MUSTLOCK(surf)) {
        did_lock_surf = SDL_TRUE;
        if (!pgSurface_Lock(surfobj)) {
            return RAISE(PyExc_RuntimeError, "error locking surface");
        }
    }
    if (pattern && SDL_MUSTLOCK(pattern)) {
        did_lock_pat = SDL_TRUE;
        if (!pgSurface_Lock(pat_surfobj)) {
            if (did_lock_surf) {
                pgSurface_Unlock(surfobj);
            }
            return RAISE(PyExc_RuntimeError, "error locking pattern surface");
        }
    }

    flood_fill_result =
        flood_fill_inner(surf, startx, starty, color, pattern, drawn_area);

    /* no allocated pattern surface to free */

    if (did_lock_pat) {
        if (!pgSurface_Unlock(pat_surfobj)) {
            if (did_lock_surf) {
                pgSurface_Unlock(surfobj);
            }
            return RAISE(PyExc_RuntimeError,
                         "error unlocking pattern surface");
        }
    }
    if (did_lock_surf) {
        if (!pgSurface_Unlock(surfobj)) {
            return RAISE(PyExc_RuntimeError, "error unlocking surface");
        }
    }

    if (flood_fill_result == -1) {
        return NULL; /* error already set by flood_fill_inner */
    }

    /* Compute return rect. */
    if (drawn_area[0] != INT_MAX && drawn_area[1] != INT_MAX &&
        drawn_area[2] != INT_MIN && drawn_area[3] != INT_MIN) {
        return pgRect_New4(drawn_area[0], drawn_area[1],
                           drawn_area[2] - drawn_area[0] + 1,
                           drawn_area[3] - drawn_area[1] + 1);
    }
    else {
        return pgRect_New4(startx, starty, 0, 0);
    }
}
/* Functions used in drawing algorithms */

static void
swap(float *a, float *b)
{
    float temp = *a;
    *a = *b;
    *b = temp;
}

#define WORD_BITS (CHAR_BIT * sizeof(unsigned int))

struct point2d {
    Uint32 x;
    Uint32 y;
};

static inline void
_bitarray_set(unsigned int *bitarray, size_t idx, SDL_bool value)
{
    const unsigned int mask = (1u << (idx % WORD_BITS));
    if (value) {
        bitarray[idx / WORD_BITS] |= mask;
    }
    else {
        bitarray[idx / WORD_BITS] &= ~mask;
    }
}

static inline SDL_bool
_bitarray_get(unsigned int *bitarray, size_t idx)
{
    if (bitarray[idx / WORD_BITS] & (1u << (idx % WORD_BITS))) {
        return SDL_TRUE;
    }
    else {
        return SDL_FALSE;
    }
}

static int
compare_int(const void *a, const void *b)
{
    return (*(const int *)a) - (*(const int *)b);
}

static Uint32
get_antialiased_color(SDL_Surface *surf, SDL_Rect surf_clip_rect,
                      PG_PixelFormat *surf_format, int x, int y,
                      Uint32 original_color, float brightness)
{
    Uint8 color_part[4], background_color[4];
    if (x < surf_clip_rect.x || x >= surf_clip_rect.x + surf_clip_rect.w ||
        y < surf_clip_rect.y || y >= surf_clip_rect.y + surf_clip_rect.h) {
        return original_color;
    }

    PG_GetRGBA(original_color, surf_format, PG_GetSurfacePalette(surf),
               &color_part[0], &color_part[1], &color_part[2], &color_part[3]);

    Uint32 pixel = 0;
    int bpp = PG_SURF_BytesPerPixel(surf);
    Uint8 *pixels = (Uint8 *)surf->pixels + y * surf->pitch + x * bpp;

    switch (bpp) {
        case 1:
            pixel = *pixels;
            break;

        case 2:
            pixel = *((Uint16 *)pixels);
            break;

        case 3:
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
            pixel = (pixels[0]) + (pixels[1] << 8) + (pixels[2] << 16);
#else  /* SDL_BIG_ENDIAN */
            pixel = (pixels[2]) + (pixels[1] << 8) + (pixels[0] << 16);
#endif /* SDL_BIG_ENDIAN */
            break;

        default: /* case 4: */
            pixel = *((Uint32 *)pixels);
            break;
    }

    PG_GetRGBA(pixel, surf_format, PG_GetSurfacePalette(surf),
               &background_color[0], &background_color[1],
               &background_color[2], &background_color[3]);

    color_part[0] = (Uint8)(brightness * color_part[0] +
                            (1 - brightness) * background_color[0]);
    color_part[1] = (Uint8)(brightness * color_part[1] +
                            (1 - brightness) * background_color[1]);
    color_part[2] = (Uint8)(brightness * color_part[2] +
                            (1 - brightness) * background_color[2]);
    color_part[3] = (Uint8)(brightness * color_part[3] +
                            (1 - brightness) * background_color[3]);
    original_color =
        PG_MapRGBA(surf_format, PG_GetSurfacePalette(surf), color_part[0],
                   color_part[1], color_part[2], color_part[3]);
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

static void
add_line_to_drawn_list(int x1, int y1, int x2, int y2, int *pts)
{
    if (x1 < pts[0]) {
        pts[0] = x1;
    }
    if (y1 < pts[1]) {
        pts[1] = y1;
    }
    if (x2 > pts[2]) {
        pts[2] = x2;
    }
    if (y2 > pts[3]) {
        pts[3] = y2;
    }
}

static int
clip_line(SDL_Surface *surf, SDL_Rect surf_clip_rect, int *x1, int *y1,
          int *x2, int *y2, int width, int xinc)
{
    int left, right, top, bottom;
    if (xinc) {
        left = MIN(*x1, *x2) - width;
        right = MAX(*x1, *x2) + width;
        top = MIN(*y1, *y2);
        bottom = MAX(*y1, *y2);
    }
    else {
        left = MIN(*x1, *x2);
        right = MAX(*x1, *x2);
        top = MIN(*y1, *y2) - width;
        bottom = MAX(*y1, *y2) + width;
    }
    if (surf_clip_rect.x > right || surf_clip_rect.y > bottom ||
        surf_clip_rect.x + surf_clip_rect.w <= left ||
        surf_clip_rect.y + surf_clip_rect.h <= top) {
        return 0;
    }

    return 1;
}

static int
set_at(SDL_Surface *surf, SDL_Rect surf_clip_rect, int x, int y, Uint32 color)
{
    Uint8 *pixels = (Uint8 *)surf->pixels;

    if (x < surf_clip_rect.x || x >= surf_clip_rect.x + surf_clip_rect.w ||
        y < surf_clip_rect.y || y >= surf_clip_rect.y + surf_clip_rect.h) {
        return 0;
    }

    switch (PG_SURF_BytesPerPixel(surf)) {
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
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
            color <<= 8;
#endif
            memcpy((pixels + y * surf->pitch) + x * 3, &color,
                   3 * sizeof(Uint8));
            break;
    }
    return 1;
}

static void
set_and_check_rect(SDL_Surface *surf, SDL_Rect surf_clip_rect, int x, int y,
                   Uint32 color, int *drawn_area)
{
    if (set_at(surf, surf_clip_rect, x, y, color)) {
        add_pixel_to_drawn_list(x, y, drawn_area);
    }
}

static void
draw_aaline(SDL_Surface *surf, SDL_Rect surf_clip_rect,
            PG_PixelFormat *surf_format, Uint32 color, float from_x,
            float from_y, float to_x, float to_y, int *drawn_area,
            int disable_first_endpoint, int disable_second_endpoint,
            int extra_pixel_for_aalines)
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
        pixel_color = get_antialiased_color(
            surf, surf_clip_rect, surf_format, (int)floor(from_x + 0.5),
            (int)floor(from_y + 0.5), color, 1);
        set_and_check_rect(surf, surf_clip_rect, (int)floor(from_x + 0.5),
                           (int)floor(from_y + 0.5), pixel_color, drawn_area);
        return;
    }

    /* To draw correctly the pixels at the border of the clipping area when
     * the line crosses it, we need to clip it one pixel wider in all four
     * directions: */
    clip_left = (float)surf_clip_rect.x - 1.0f;
    clip_right = (float)clip_left + surf_clip_rect.w + 1.0f;
    clip_top = (float)surf_clip_rect.y - 1.0f;
    clip_bottom = (float)clip_top + surf_clip_rect.h + 1.0f;

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
        /* The line is completely to the side of the surface */
        return;
    }

    /* Note. There is no need to guard against a division by zero here. If dx
     * was zero then either we had a single point (and we've returned) or it
     * has been swapped with a non-zero dy. */
    gradient = dy / dx;

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
     * That permit us to use (int)y to round down instead of having to use
     * floor(y). We then draw the pixels one higher.*/
    from_y += 1.0f;
    to_y += 1.0f;

    /* Handle endpoints separately.
     * The line is not a mathematical line of thickness zero. The same
     * goes for the endpoints. The have a height and width of one pixel.
     * Extra pixel drawing is requested externally from aalines.
     * It is drawn only when one line is steep and other is not.*/
    /* First endpoint */
    if (!disable_first_endpoint || extra_pixel_for_aalines) {
        x_pixel_start = (int)from_x;
        y_endpoint = intersect_y =
            from_y + gradient * (x_pixel_start - from_x);
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
                pixel_color =
                    get_antialiased_color(surf, surf_clip_rect, surf_format, x,
                                          y, color, brightness * x_gap);
                set_and_check_rect(surf, surf_clip_rect, x, y, pixel_color,
                                   drawn_area);
            }
            if (steep) {
                x--;
            }
            else {
                y--;
            }
            brightness = 1 - brightness;
            pixel_color =
                get_antialiased_color(surf, surf_clip_rect, surf_format, x, y,
                                      color, brightness * x_gap);
            set_and_check_rect(surf, surf_clip_rect, x, y, pixel_color,
                               drawn_area);
            intersect_y += gradient;
            x_pixel_start++;
        }
    }
    /* To be sure main loop skips first endpoint.*/
    if (disable_first_endpoint) {
        x_pixel_start = (int)ceil(from_x);
        intersect_y = from_y + gradient * (x_pixel_start - from_x);
    }
    /* Second endpoint */
    x_pixel_end = (int)ceil(to_x);
    if (!disable_second_endpoint || extra_pixel_for_aalines) {
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
                pixel_color =
                    get_antialiased_color(surf, surf_clip_rect, surf_format, x,
                                          y, color, brightness * x_gap);
                set_and_check_rect(surf, surf_clip_rect, x, y, pixel_color,
                                   drawn_area);
            }
            if (steep) {
                x--;
            }
            else {
                y--;
            }
            brightness = 1 - brightness;
            pixel_color =
                get_antialiased_color(surf, surf_clip_rect, surf_format, x, y,
                                      color, brightness * x_gap);
            set_and_check_rect(surf, surf_clip_rect, x, y, pixel_color,
                               drawn_area);
        }
    }

    /* main line drawing loop */
    for (x = x_pixel_start; x < x_pixel_end; x++) {
        y = (int)intersect_y;
        if (steep) {
            brightness = 1 - intersect_y + y;
            pixel_color =
                get_antialiased_color(surf, surf_clip_rect, surf_format, y - 1,
                                      x, color, brightness);
            set_and_check_rect(surf, surf_clip_rect, y - 1, x, pixel_color,
                               drawn_area);
            if (y < intersect_y) {
                brightness = 1 - brightness;
                pixel_color =
                    get_antialiased_color(surf, surf_clip_rect, surf_format, y,
                                          x, color, brightness);
                set_and_check_rect(surf, surf_clip_rect, y, x, pixel_color,
                                   drawn_area);
            }
        }
        else {
            brightness = 1 - intersect_y + y;
            pixel_color =
                get_antialiased_color(surf, surf_clip_rect, surf_format, x,
                                      y - 1, color, brightness);
            set_and_check_rect(surf, surf_clip_rect, x, y - 1, pixel_color,
                               drawn_area);
            if (y < intersect_y) {
                brightness = 1 - brightness;
                pixel_color =
                    get_antialiased_color(surf, surf_clip_rect, surf_format, x,
                                          y, color, brightness);
                set_and_check_rect(surf, surf_clip_rect, x, y, pixel_color,
                                   drawn_area);
            }
        }
        intersect_y += gradient;
    }
}

static void
drawhorzline(SDL_Surface *surf, Uint32 color, int x1, int y1, int x2)
{
    Uint8 *pixel, *end;

    pixel = ((Uint8 *)surf->pixels) + surf->pitch * y1;
    end = pixel + x2 * PG_SURF_BytesPerPixel(surf);
    pixel += x1 * PG_SURF_BytesPerPixel(surf);
    switch (PG_SURF_BytesPerPixel(surf)) {
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
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
            color <<= 8;
#endif
            for (; pixel <= end; pixel += 3) {
                memcpy(pixel, &color, 3 * sizeof(Uint8));
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
drawvertline(SDL_Surface *surf, Uint32 color, int y1, int x1, int y2)
{
    Uint8 *pixel, *end;

    pixel = ((Uint8 *)surf->pixels) + surf->pitch * y1;
    end = ((Uint8 *)surf->pixels) + surf->pitch * y2 +
          x1 * PG_SURF_BytesPerPixel(surf);
    pixel += x1 * PG_SURF_BytesPerPixel(surf);
    switch (PG_SURF_BytesPerPixel(surf)) {
        case 1:
            for (; pixel <= end; pixel += surf->pitch) {
                *pixel = (Uint8)color;
            }
            break;
        case 2:
            for (; pixel <= end; pixel += surf->pitch) {
                *(Uint16 *)pixel = (Uint16)color;
            }
            break;
        case 3:
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
            color <<= 8;
#endif
            for (; pixel <= end; pixel += surf->pitch) {
                memcpy(pixel, &color, 3 * sizeof(Uint8));
            }
            break;
        default: /*case 4*/
            for (; pixel <= end; pixel += surf->pitch) {
                *(Uint32 *)pixel = color;
            }
            break;
    }
}

static void
drawhorzlineclip(SDL_Surface *surf, SDL_Rect surf_clip_rect, Uint32 color,
                 int x1, int y1, int x2)
{
    if (y1 < surf_clip_rect.y || y1 >= surf_clip_rect.y + surf_clip_rect.h) {
        return;
    }

    if (x2 < x1) {
        int temp = x1;
        x1 = x2;
        x2 = temp;
    }

    x1 = MAX(x1, surf_clip_rect.x);
    x2 = MIN(x2, surf_clip_rect.x + surf_clip_rect.w - 1);

    if (x2 < surf_clip_rect.x || x1 >= surf_clip_rect.x + surf_clip_rect.w) {
        return;
    }

    if (x1 == x2) {
        set_at(surf, surf_clip_rect, x1, y1, color);
        return;
    }
    drawhorzline(surf, color, x1, y1, x2);
}

static void
drawhorzlineclipbounding(SDL_Surface *surf, SDL_Rect surf_clip_rect,
                         Uint32 color, int x1, int y1, int x2, int *pts)
{
    if (y1 < surf_clip_rect.y || y1 >= surf_clip_rect.y + surf_clip_rect.h) {
        return;
    }

    if (x2 < x1) {
        int temp = x1;
        x1 = x2;
        x2 = temp;
    }

    x1 = MAX(x1, surf_clip_rect.x);
    x2 = MIN(x2, surf_clip_rect.x + surf_clip_rect.w - 1);

    if (x2 < surf_clip_rect.x || x1 >= surf_clip_rect.x + surf_clip_rect.w) {
        return;
    }

    if (x1 == x2) {
        set_and_check_rect(surf, surf_clip_rect, x1, y1, color, pts);
        return;
    }

    add_line_to_drawn_list(x1, y1, x2, y1, pts);

    drawhorzline(surf, color, x1, y1, x2);
}

static void
drawvertlineclipbounding(SDL_Surface *surf, SDL_Rect surf_clip_rect,
                         Uint32 color, int y1, int x1, int y2, int *pts)
{
    if (x1 < surf_clip_rect.x || x1 >= surf_clip_rect.x + surf_clip_rect.w) {
        return;
    }

    if (y2 < y1) {
        int temp = y1;
        y1 = y2;
        y2 = temp;
    }

    y1 = MAX(y1, surf_clip_rect.y);
    y2 = MIN(y2, surf_clip_rect.y + surf_clip_rect.h - 1);

    if (y2 < surf_clip_rect.y || y1 >= surf_clip_rect.y + surf_clip_rect.h) {
        return;
    }

    if (y1 == y2) {
        set_and_check_rect(surf, surf_clip_rect, x1, y1, color, pts);
        return;
    }

    add_line_to_drawn_list(x1, y1, x1, y2, pts);

    drawvertline(surf, color, y1, x1, y2);
}

void
swap_coordinates(int *x1, int *y1, int *x2, int *y2)
{
    int temp = *x1;
    *x1 = *x2;
    *x2 = temp;

    temp = *y1;
    *y1 = *y2;
    *y2 = temp;
}

static int
draw_filltri(SDL_Surface *surf, SDL_Rect surf_clip_rect, int *xlist,
             int *ylist, Uint32 color, int *draw_area)
{
    int p0x, p0y, p1x, p1y, p2x, p2y;

    p0x = xlist[0];
    p1x = xlist[1];
    p2x = xlist[2];
    p0y = ylist[0];
    p1y = ylist[1];
    p2y = ylist[2];

    if (p1y < p0y) {
        swap_coordinates(&p1x, &p1y, &p0x, &p0y);
    }

    if (p2y < p1y) {
        swap_coordinates(&p1x, &p1y, &p2x, &p2y);

        if (p1y < p0y) {
            swap_coordinates(&p1x, &p1y, &p0x, &p0y);
        }
    }

    if ((p0y == p1y) && (p1y == p2y) && (p0x == p1x) && (p1x != p2x)) {
        swap_coordinates(&p1x, &p1y, &p2x, &p2y);
    }

    float d1 = (float)((p2x - p0x) / ((p2y - p0y) + 1e-17));
    float d2 = (float)((p1x - p0x) / ((p1y - p0y) + 1e-17));
    float d3 = (float)((p2x - p1x) / ((p2y - p1y) + 1e-17));
    int y;
    for (y = p0y; y <= p2y; y++) {
        int x1 = p0x + (int)((y - p0y) * d1);

        int x2;
        if (y < p1y) {
            x2 = p0x + (int)((y - p0y) * d2);
        }
        else {
            x2 = p1x + (int)((y - p1y) * d3);
        }

        drawhorzlineclipbounding(surf, surf_clip_rect, color, x1, y, x2,
                                 draw_area);
    }

    return 0;
}

static void
draw_line_width(SDL_Surface *surf, SDL_Rect surf_clip_rect, Uint32 color,
                int x1, int y1, int x2, int y2, int width, int *drawn_area)
{
    int dx, dy, err, e2, sx, sy, start_draw, end_draw;
    int end_x = surf_clip_rect.x + surf_clip_rect.w - 1;
    int end_y = surf_clip_rect.y + surf_clip_rect.h - 1;
    int xinc = 0;
    int extra_width = 1 - (width % 2);

    if (width < 1) {
        return;
    }
    if (width == 1) {
        draw_line(surf, surf_clip_rect, x1, y1, x2, y2, color, drawn_area);
        return;
    }

    width = (width / 2);

    /* Decide which direction to grow (width/thickness). */
    if (abs(x1 - x2) <= abs(y1 - y2)) {
        /* The line's thickness will be in the x direction. The top/bottom
         * ends of the line will be flat. */
        xinc = 1;
    }

    if (!clip_line(surf, surf_clip_rect, &x1, &y1, &x2, &y2, width, xinc)) {
        return;
    }

    if (x1 == x2 && y1 == y2) { /* Single point */
        start_draw = MAX((x1 - width) + extra_width, surf_clip_rect.x);
        end_draw = MIN(end_x, x1 + width);
        if (start_draw <= end_draw) {
            drawhorzline(surf, color, start_draw, y1, end_draw);
            add_line_to_drawn_list(start_draw, y1, end_draw, y1, drawn_area);
        }
        return;
    }
    // Bresenham's line algorithm
    dx = abs(x2 - x1);
    dy = abs(y2 - y1);
    sx = x2 > x1 ? 1 : -1;
    sy = y2 > y1 ? 1 : -1;
    err = (dx > dy ? dx : -dy) / 2;
    if (xinc) {
        while (y1 != (y2 + sy)) {
            if (surf_clip_rect.y <= y1 && y1 <= end_y) {
                start_draw = MAX((x1 - width) + extra_width, surf_clip_rect.x);
                end_draw = MIN(end_x, x1 + width);
                if (start_draw <= end_draw) {
                    drawhorzline(surf, color, start_draw, y1, end_draw);
                    add_line_to_drawn_list(start_draw, y1, end_draw, y1,
                                           drawn_area);
                }
            }
            e2 = err;
            if (e2 > -dx) {
                err -= dy;
                x1 += sx;
            }
            if (e2 < dy) {
                err += dx;
                y1 += sy;
            }
        }
    }
    else {
        while (x1 != (x2 + sx)) {
            if (surf_clip_rect.x <= x1 && x1 <= end_x) {
                start_draw = MAX((y1 - width) + extra_width, surf_clip_rect.y);
                end_draw = MIN(end_y, y1 + width);
                if (start_draw <= end_draw) {
                    drawvertline(surf, color, start_draw, x1, end_draw);
                    add_line_to_drawn_list(x1, start_draw, x1, end_draw,
                                           drawn_area);
                }
            }
            e2 = err;
            if (e2 > -dx) {
                err -= dy;
                x1 += sx;
            }
            if (e2 < dy) {
                err += dx;
                y1 += sy;
            }
        }
    }
}

static void
draw_aaline_width(SDL_Surface *surf, SDL_Rect surf_clip_rect,
                  PG_PixelFormat *surf_format, Uint32 color, float from_x,
                  float from_y, float to_x, float to_y, int width,
                  int *drawn_area)
{
    float gradient, dx, dy, intersect_y, brightness;
    int x, x_pixel_start, x_pixel_end, start_draw, end_draw;
    Uint32 pixel_color;
    float y_endpoint, clip_left, clip_right, clip_top, clip_bottom;
    int steep, y;
    int extra_width = 1 - (width % 2);

    width = (width / 2);

    dx = to_x - from_x;
    dy = to_y - from_y;
    steep = fabs(dx) < fabs(dy);

    /* Single point.
     * A line with length 0 is drawn as a single pixel at full brightness. */
    if (fabs(dx) < 0.0001 && fabs(dy) < 0.0001) {
        x = (int)floor(from_x + 0.5);
        y = (int)floor(from_y + 0.5);
        pixel_color = get_antialiased_color(surf, surf_clip_rect, surf_format,
                                            x, y, color, 1);
        set_and_check_rect(surf, surf_clip_rect, x, y, pixel_color,
                           drawn_area);
        if (dx != 0 && dy != 0) {
            if (steep) {
                start_draw = (int)(x - width + extra_width);
                end_draw = (int)(x + width) - 1;
                drawhorzlineclipbounding(surf, surf_clip_rect, color,
                                         start_draw, y, end_draw, drawn_area);
            }
            else {
                start_draw = (int)(y - width + extra_width);
                end_draw = (int)(y + width) - 1;
                drawvertlineclipbounding(surf, surf_clip_rect, color,
                                         start_draw, x, end_draw, drawn_area);
            }
        }
        return;
    }

    /* To draw correctly the pixels at the border of the clipping area when
     * the line crosses it, we need to clip it one pixel wider in all four
     * directions, and add width */
    clip_left = (float)surf_clip_rect.x - 1.0f;
    clip_right = (float)clip_left + surf_clip_rect.w + 1.0f;
    clip_top = (float)surf_clip_rect.y - 1.0f;
    clip_bottom = (float)clip_top + surf_clip_rect.h + 1.0f;

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
        /* The line is completely to the side of the surface */
        return;
    }

    /* Note. There is no need to guard against a division by zero here. If dx
     * was zero then either we had a single point (and we've returned) or it
     * has been swapped with a non-zero dy. */
    gradient = dy / dx;

    /* No need to waste CPU cycles on pixels not on the surface. */
    if (from_x < clip_left + 1) {
        from_y += gradient * (clip_left + 1 - from_x);
        from_x = clip_left + 1;
    }
    if (to_x > clip_right - 1) {
        to_y += gradient * (clip_right - 1 - to_x);
        to_x = clip_right - 1;
    }

    if (gradient > 0.0f) {
        if (from_x < clip_left + 1) {
            /* from_ is the topmost endpoint */
            if (to_y <= clip_top || from_y >= clip_bottom) {
                /* The line does not enter the surface */
                return;
            }
            if (from_y < clip_top - width) {
                from_x += (clip_top - width - from_y) / gradient;
                from_y = clip_top - width;
            }
            if (to_y > clip_bottom + width) {
                to_x += (clip_bottom + width - to_y) / gradient;
                to_y = clip_bottom + width;
            }
        }
    }
    else {
        if (to_x > clip_right - 1) {
            /* to_ is the topmost endpoint */
            if (from_y <= clip_top || to_y >= clip_bottom) {
                /* The line does not enter the surface */
                return;
            }
            if (to_y < clip_top - width) {
                to_x += (clip_top - width - to_y) / gradient;
                to_y = clip_top - width;
            }
            if (from_y > clip_bottom + width) {
                from_x += (clip_bottom + width - from_y) / gradient;
                from_y = clip_bottom + width;
            }
        }
    }

    /* By moving the points one pixel down, we can assume y is never negative.
     * That permit us to use (int)y to round down instead of having to use
     * floor(y). We then draw the pixels one higher.*/
    from_y += 1.0f;
    to_y += 1.0f;

    /* Handle endpoints separately */
    /* First endpoint */
    x_pixel_start = (int)from_x;
    y_endpoint = intersect_y = from_y + gradient * (x_pixel_start - from_x);
    if (to_x > clip_left + 1.0f) {
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
            if (steep) {
                pixel_color =
                    get_antialiased_color(surf, surf_clip_rect, surf_format,
                                          x + width, y, color, brightness);
                set_and_check_rect(surf, surf_clip_rect, x + width, y,
                                   pixel_color, drawn_area);
            }
            else {
                pixel_color =
                    get_antialiased_color(surf, surf_clip_rect, surf_format, x,
                                          y + width, color, brightness);
                set_and_check_rect(surf, surf_clip_rect, x, y + width,
                                   pixel_color, drawn_area);
            }
        }
        brightness = 1 - brightness;
        if (steep) {
            pixel_color =
                get_antialiased_color(surf, surf_clip_rect, surf_format,
                                      x - width, y, color, brightness);
            set_and_check_rect(surf, surf_clip_rect,
                               x - width + extra_width - 1, y, pixel_color,
                               drawn_area);
            start_draw = (int)(x - width + extra_width);
            end_draw = (int)(x + width) - 1;
            drawhorzlineclipbounding(surf, surf_clip_rect, color, start_draw,
                                     y, end_draw, drawn_area);
        }
        else {
            pixel_color = get_antialiased_color(
                surf, surf_clip_rect, surf_format, x,
                y - width + extra_width - 1, color, brightness);
            set_and_check_rect(surf, surf_clip_rect, x,
                               y - width + extra_width - 1, pixel_color,
                               drawn_area);
            start_draw = (int)(y - width + extra_width);
            end_draw = (int)(y + width) - 1;
            drawvertlineclipbounding(surf, surf_clip_rect, color, start_draw,
                                     x, end_draw, drawn_area);
        }
        intersect_y += gradient;
        x_pixel_start++;
    }

    /* Second endpoint */
    x_pixel_end = (int)ceil(to_x);
    if (from_x < clip_right - 1.0f) {
        y_endpoint = to_y + gradient * (x_pixel_end - to_x);
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
            if (steep) {
                pixel_color =
                    get_antialiased_color(surf, surf_clip_rect, surf_format,
                                          x + width, y, color, brightness);
                set_and_check_rect(surf, surf_clip_rect, x + width, y,
                                   pixel_color, drawn_area);
            }
            else {
                pixel_color =
                    get_antialiased_color(surf, surf_clip_rect, surf_format, x,
                                          y + width, color, brightness);
                set_and_check_rect(surf, surf_clip_rect, x, y + width,
                                   pixel_color, drawn_area);
            }
        }
        brightness = 1 - brightness;
        if (steep) {
            pixel_color = get_antialiased_color(
                surf, surf_clip_rect, surf_format, x - width + extra_width - 1,
                y, color, brightness);
            set_and_check_rect(surf, surf_clip_rect,
                               x - width + extra_width - 1, y, pixel_color,
                               drawn_area);
            start_draw = (int)(x - width);
            end_draw = (int)(x + width) - 1;
            drawhorzlineclipbounding(surf, surf_clip_rect, color, start_draw,
                                     y, end_draw, drawn_area);
        }
        else {
            pixel_color = get_antialiased_color(
                surf, surf_clip_rect, surf_format, x,
                y - width + extra_width - 1, color, brightness);
            set_and_check_rect(surf, surf_clip_rect, x,
                               y - width + extra_width - 1, pixel_color,
                               drawn_area);
            start_draw = (int)(y - width + extra_width);
            end_draw = (int)(y + width) - 1;
            drawvertlineclipbounding(surf, surf_clip_rect, color, start_draw,
                                     x, end_draw, drawn_area);
        }
    }

    /* main line drawing loop */
    for (x = x_pixel_start; x < x_pixel_end; x++) {
        y = (int)intersect_y;
        if (steep) {
            brightness = 1 - intersect_y + y;
            pixel_color = get_antialiased_color(
                surf, surf_clip_rect, surf_format, y - width + extra_width - 1,
                x, color, brightness);
            set_and_check_rect(surf, surf_clip_rect,
                               y - width + extra_width - 1, x, pixel_color,
                               drawn_area);
            if (y < intersect_y) {
                brightness = 1 - brightness;
                pixel_color =
                    get_antialiased_color(surf, surf_clip_rect, surf_format,
                                          y + width, x, color, brightness);
                set_and_check_rect(surf, surf_clip_rect, y + width, x,
                                   pixel_color, drawn_area);
            }
            start_draw = (int)(y - width + extra_width);
            end_draw = (int)(y + width) - 1;
            drawhorzlineclipbounding(surf, surf_clip_rect, color, start_draw,
                                     x, end_draw, drawn_area);
        }
        else {
            brightness = 1 - intersect_y + y;
            pixel_color = get_antialiased_color(
                surf, surf_clip_rect, surf_format, x,
                y - width + extra_width - 1, color, brightness);
            set_and_check_rect(surf, surf_clip_rect, x,
                               y - width + extra_width - 1, pixel_color,
                               drawn_area);
            if (y < intersect_y) {
                brightness = 1 - brightness;
                pixel_color =
                    get_antialiased_color(surf, surf_clip_rect, surf_format, x,
                                          y + width, color, brightness);
                set_and_check_rect(surf, surf_clip_rect, x, y + width,
                                   pixel_color, drawn_area);
            }
            start_draw = (int)(y - width + extra_width);
            end_draw = (int)(y + width) - 1;
            drawvertlineclipbounding(surf, surf_clip_rect, color, start_draw,
                                     x, end_draw, drawn_area);
        }
        intersect_y += gradient;
    }
}

/* Algorithm modified from
 * https://rosettacode.org/wiki/Bitmap/Bresenham%27s_line_algorithm
 */
static void
draw_line(SDL_Surface *surf, SDL_Rect surf_clip_rect, int x1, int y1, int x2,
          int y2, Uint32 color, int *drawn_area)
{
    int dx, dy, err, e2, sx, sy;
    if (x1 == x2 && y1 == y2) { /* Single point */
        set_and_check_rect(surf, surf_clip_rect, x1, y1, color, drawn_area);
        return;
    }
    if (y1 == y2) { /* Horizontal line */
        dx = (x1 < x2) ? 1 : -1;
        for (sx = 0; sx <= abs(x1 - x2); sx++) {
            set_and_check_rect(surf, surf_clip_rect, x1 + dx * sx, y1, color,
                               drawn_area);
        }

        return;
    }
    if (x1 == x2) { /* Vertical line */
        dy = (y1 < y2) ? 1 : -1;
        for (sy = 0; sy <= abs(y1 - y2); sy++) {
            set_and_check_rect(surf, surf_clip_rect, x1, y1 + dy * sy, color,
                               drawn_area);
        }
        return;
    }
    dx = abs(x2 - x1), sx = x1 < x2 ? 1 : -1;
    dy = abs(y2 - y1), sy = y1 < y2 ? 1 : -1;
    err = (dx > dy ? dx : -dy) / 2;
    while (x1 != x2 || y1 != y2) {
        set_and_check_rect(surf, surf_clip_rect, x1, y1, color, drawn_area);
        e2 = err;
        if (e2 > -dx) {
            err -= dy;
            x1 += sx;
        }
        if (e2 < dy) {
            err += dx;
            y1 += sy;
        }
    }
    set_and_check_rect(surf, surf_clip_rect, x2, y2, color, drawn_area);
}

#define SURF_GET_AT_FORMAT(p_color, p_surf, p_x, p_y, p_pixels, p_pix)       \
    switch (PG_SURF_BytesPerPixel(p_surf)) {                                 \
        case 1:                                                              \
            p_color = (Uint32) *                                             \
                      ((Uint8 *)(p_pixels) + (p_y) * p_surf->pitch + (p_x)); \
            break;                                                           \
        case 2:                                                              \
            p_color =                                                        \
                (Uint32) *                                                   \
                ((Uint16 *)((p_pixels) + (p_y) * p_surf->pitch) + (p_x));    \
            break;                                                           \
        case 3:                                                              \
            p_pix =                                                          \
                ((Uint8 *)(p_pixels + (p_y) * p_surf->pitch) + (p_x) * 3);   \
            p_color = (SDL_BYTEORDER == SDL_LIL_ENDIAN)                      \
                          ? (p_pix[0]) + (p_pix[1] << 8) + (p_pix[2] << 16)  \
                          : (p_pix[2]) + (p_pix[1] << 8) + (p_pix[0] << 16); \
            break;                                                           \
        default: /* case 4: */                                               \
            p_color =                                                        \
                *((Uint32 *)(p_pixels + (p_y) * p_surf->pitch) + (p_x));     \
            break;                                                           \
    }

static int
flood_fill_inner(SDL_Surface *surf, int x1, int y1, Uint32 new_color,
                 SDL_Surface *pattern, int *drawn_area)
{
    // breadth first flood fill, like graph search
    SDL_Rect cliprect;
    size_t mask_idx;
    if (!PG_GetSurfaceClipRect(surf, &cliprect)) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return -1;
    }
    size_t frontier_bufsize = 8, frontier_size = 1, next_frontier_size = 0;

    // Instead of a queue, we use two arrays and swap them between steps.
    // This makes implementation easier, especially memory management.
    struct point2d *frontier =
        malloc(frontier_bufsize * sizeof(struct point2d));
    if (frontier == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    struct point2d *frontier_next =
        malloc(frontier_bufsize * sizeof(struct point2d));

    if (frontier_next == NULL) {
        free(frontier);
        PyErr_NoMemory();
        return -1;
    }

    // 2D bitmask for queued nodes
    // we could check drawn color, but that doesnt work for patterns
    size_t mask_size = (size_t)cliprect.w * (size_t)cliprect.h;
    size_t mask_words = (mask_size + WORD_BITS - 1) / WORD_BITS;
    unsigned int *mask = calloc(mask_words, sizeof(unsigned int));

    if (mask == NULL) {
        free(frontier);
        free(frontier_next);
        PyErr_NoMemory();
        return -1;
    }
    Uint32 old_color = 0;
    Uint8 *pix;

    // Von Neumann neighbourhood
    int VN_X[] = {0, 0, 1, -1};
    int VN_Y[] = {1, -1, 0, 0};

    if (!(x1 >= cliprect.x && x1 < (cliprect.x + cliprect.w) &&
          y1 >= cliprect.y && y1 < (cliprect.y + cliprect.h))) {
        // not an error, but nothing to do here
        goto flood_fill_finished;
    }

    SURF_GET_AT_FORMAT(old_color, surf, x1, y1, (Uint8 *)surf->pixels, pix);

    if (pattern == NULL && old_color == new_color) {
        // not an error, but nothing to do here
        goto flood_fill_finished;
    }

    frontier[0].x = x1;
    frontier[0].y = y1;

    // mark starting point already queued
    mask_idx = (y1 - cliprect.y) * cliprect.w + (x1 - cliprect.x);
    _bitarray_set(mask, mask_idx, SDL_TRUE);

    while (frontier_size != 0) {
        next_frontier_size = 0;

        for (size_t i = 0; i < frontier_size; i++) {
            unsigned int x = frontier[i].x;
            unsigned int y = frontier[i].y;

            Uint32 current_color = 0;

            SURF_GET_AT_FORMAT(current_color, surf, x, y,
                               (Uint8 *)surf->pixels, pix);

            if (current_color != old_color) {
                continue;
            }

            if (pattern != NULL) {
                SURF_GET_AT_FORMAT(new_color, pattern, x % pattern->w,
                                   y % pattern->h, (Uint8 *)pattern->pixels,
                                   pix);
            }

            // clipping and color mapping have already happened here
            unsafe_set_at(surf, x, y, new_color);
            add_pixel_to_drawn_list(x, y, drawn_area);

            for (int n = 0; n < 4; n++) {
                long nx = x + VN_X[n];
                long ny = y + VN_Y[n];

                if (!(nx >= cliprect.x && nx < cliprect.x + cliprect.w &&
                      ny >= cliprect.y && ny < cliprect.y + cliprect.h)) {
                    continue;
                }

                mask_idx = (ny - cliprect.y) * cliprect.w + (nx - cliprect.x);
                if (_bitarray_get(mask, mask_idx)) {
                    continue;
                }

                // only queue node once
                _bitarray_set(mask, mask_idx, SDL_TRUE);

                if (next_frontier_size == frontier_bufsize) {
                    // grow frontier arrays
                    struct point2d *old_buf = frontier_next;

                    frontier_bufsize *= 4;

                    frontier_next =
                        realloc(frontier_next,
                                frontier_bufsize * sizeof(struct point2d));
                    if (frontier_next == NULL) {
                        free(mask);
                        free(frontier);
                        free(old_buf);
                        PyErr_NoMemory();
                        return -1;
                    }

                    old_buf = frontier;
                    frontier = realloc(
                        frontier, frontier_bufsize * sizeof(struct point2d));
                    if (frontier == NULL) {
                        free(old_buf);
                        free(mask);
                        free(frontier_next);
                        PyErr_NoMemory();
                        return -1;
                    }
                }

                frontier_next[next_frontier_size].x = nx;
                frontier_next[next_frontier_size].y = ny;
                next_frontier_size++;
            }
        }
        // swap buffers
        struct point2d *temp_buf;
        temp_buf = frontier;
        frontier = frontier_next;
        frontier_next = temp_buf;

        frontier_size = next_frontier_size;
    }

flood_fill_finished:
    free(frontier);
    free(mask);
    free(frontier_next);
    return 0;
}
static int
check_pixel_in_arc(int x, int y, double min_dotproduct, double invsqr_radius1,
                   double invsqr_radius2, double invsqr_inner_radius1,
                   double invsqr_inner_radius2, double x_middle,
                   double y_middle)
{
    // Check outer boundary
    const double x_adjusted = x * x * invsqr_radius1;
    const double y_adjusted = y * y * invsqr_radius2;
    if (x_adjusted + y_adjusted > 1) {
        return 0;
    }
    // Check inner boundary
    const double x_inner_adjusted = x * x * invsqr_inner_radius1;
    const double y_inner_adjusted = y * y * invsqr_inner_radius2;
    if (x_inner_adjusted + y_inner_adjusted < 1) {
        return 0;
    }

    // Return whether the angle of the point is within the accepted range
    return x * x_middle + y * y_middle >= min_dotproduct * sqrt(x * x + y * y);
}

// This function directly sets the pixel, without updating the clip boundary
// detection, hence it is unsafe. This has been removed for performance, as it
// is faster to calculate the clip boundary beforehand and thus improve pixel
// write performance
static void
unsafe_set_at(SDL_Surface *surf, int x, int y, Uint32 color)
{
    Uint8 *pixels = (Uint8 *)surf->pixels;

    switch (PG_SURF_BytesPerPixel(surf)) {
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
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
            color <<= 8;
#endif
            memcpy((pixels + y * surf->pitch) + x * 3, &color,
                   3 * sizeof(Uint8));
            break;
    }
}

static void
calc_arc_bounds(SDL_Surface *surf, SDL_Rect surf_clip_rect, double angle_start,
                double angle_stop, int radius1, int radius2, int inner_radius1,
                int inner_radius2, double invsqr_radius1,
                double invsqr_radius2, double invsqr_inner_radius1,
                double invsqr_inner_radius2, double min_dotproduct,
                double x_middle, double y_middle, int x_center, int y_center,
                int *minxbound, int *minybound, int *maxxbound, int *maxybound)
{
    // calculate bounding box
    // these values find the corners of the arc
    const double x_start = cos(angle_start);
    const double y_start = -sin(angle_start);
    const double x_stop = cos(angle_stop);
    const double y_stop = -sin(angle_stop);

    const int x_start_inner = (int)(x_start * inner_radius1 + 0.5);
    const int y_start_inner = (int)(y_start * inner_radius2 + 0.5);
    const int x_stop_inner = (int)(x_stop * inner_radius1 + 0.5);
    const int y_stop_inner = (int)(y_stop * inner_radius2 + 0.5);
    const int x_start_outer = (int)(x_start * radius1 + 0.5);
    const int y_start_outer = (int)(y_start * radius2 + 0.5);
    const int x_stop_outer = (int)(x_stop * radius1 + 0.5);
    const int y_stop_outer = (int)(y_stop * radius2 + 0.5);

    // calculate maximums, accounting for each quadrant
    // We can't just find the maximum and minimum points because the arc may
    // span multiple quadrants, resulting in a maxima at the edge of the circle
    // also account for the surface's clip rect. This allows us to bypass the
    // drawn area calculations
    int minx = -radius1;
    if (-x_middle < min_dotproduct) {
        minx = MIN(MIN(x_start_inner, x_stop_inner),
                   MIN(x_start_outer, x_stop_outer));
    }
    minx = MAX(minx, surf_clip_rect.x - x_center);

    int miny = -radius2;
    if (-y_middle < min_dotproduct) {
        miny = MIN(MIN(y_start_inner, y_stop_inner),
                   MIN(y_start_outer, y_stop_outer));
    }
    miny = MAX(miny, surf_clip_rect.y - y_center);

    int maxx = radius1;
    if (x_middle < min_dotproduct) {
        maxx = MAX(MAX(x_start_inner, x_stop_inner),
                   MAX(x_start_outer, x_stop_outer));
    }
    maxx = MIN(maxx, surf_clip_rect.x + surf_clip_rect.w - x_center - 1);

    int maxy = radius2;
    if (y_middle < min_dotproduct) {
        maxy = MAX(MAX(y_start_inner, y_stop_inner),
                   MAX(y_start_outer, y_stop_outer));
    }
    maxy = MIN(maxy, surf_clip_rect.y + surf_clip_rect.h - y_center - 1);

    // Early return to avoid setting drawn_area with possibly strange values
    if (minx >= maxx || miny >= maxy) {
        return;
    }

    // dynamically reduce bounds to handle special edge cases with clipping
    // I really hope you have code folding otherwise good luck I guess :)
    int exists = 0;
    // Reduce miny bound area
    while (!exists) {
        if (miny >= maxy) {
            return;
        }

        // Go through each pixel in the circle
        for (int x = minx; x <= maxx; ++x) {
            if (check_pixel_in_arc(x, miny, min_dotproduct, invsqr_radius1,
                                   invsqr_radius2, invsqr_inner_radius1,
                                   invsqr_inner_radius2, x_middle, y_middle)) {
                exists = 1;
                break;
            }
        }
        // Narrow bounds if no pixels found
        miny += !exists;
    }
    exists = 0;
    // Reduce maxy bound area
    while (!exists) {
        // Early return to avoid setting drawn_area with possibly strange
        // values
        if (maxy <= miny) {
            return;
        }

        // For every pixel in the row
        for (int x = minx; x <= maxx; ++x) {
            if (check_pixel_in_arc(x, maxy, min_dotproduct, invsqr_radius1,
                                   invsqr_radius2, invsqr_inner_radius1,
                                   invsqr_inner_radius2, x_middle, y_middle)) {
                exists = 1;
                break;
            }
        }
        // Decrement if no pixels found
        maxy -= !exists;
    }
    exists = 0;
    // Reduce minx bound area
    while (!exists) {
        // Early return to avoid setting drawn_area with possibly strange
        // values
        if (minx >= maxx) {
            return;
        }

        // For every pixel in the row
        for (int y = miny; y <= maxy; ++y) {
            if (check_pixel_in_arc(minx, y, min_dotproduct, invsqr_radius1,
                                   invsqr_radius2, invsqr_inner_radius1,
                                   invsqr_inner_radius2, x_middle, y_middle)) {
                exists = 1;
                break;
            }
        }
        // Decrement if no pixels found
        minx += !exists;
    }
    exists = 0;
    // Reduce maxx bound area
    while (!exists) {
        // Early return to avoid setting drawn_area with possibly strange
        // values
        if (minx >= maxx) {
            return;
        }

        // For every pixel in the row
        for (int y = miny; y <= maxy; ++y) {
            if (check_pixel_in_arc(maxx, y, min_dotproduct, invsqr_radius1,
                                   invsqr_radius2, invsqr_inner_radius1,
                                   invsqr_inner_radius2, x_middle, y_middle)) {
                exists = 1;
                break;
            }
        }
        // Decrement if no pixels found
        maxx -= !exists;
    }

    *minxbound = minx;
    *minybound = miny;
    *maxxbound = maxx;
    *maxybound = maxy;
}

static void
draw_arc(SDL_Surface *surf, SDL_Rect surf_clip_rect, int x_center,
         int y_center, int radius1, int radius2, int width, double angle_start,
         double angle_stop, Uint32 color, int *drawn_area)
{
    // handle cases from documentation
    if (width <= 0) {
        return;
    }
    if (angle_stop < angle_start) {
        angle_stop += 2 * M_PI;
    }
    // if angles are equal then don't draw anything either
    if (angle_stop <= angle_start) {
        return;
    }

    // Calculate the angle halfway from the start and stop. This is guaranteed
    // to be within the final arc.
    const double angle_middle = 0.5 * (angle_start + angle_stop);
    const double angle_distance = angle_middle - angle_start;

    // Calculate the unit vector for said angle from the center of the circle
    const double x_middle = cos(angle_middle);
    const double y_middle = -sin(angle_middle);

    // Calculate inverse square inner and outer radii
    const int inner_radius1 = radius1 - width;
    const int inner_radius2 = radius2 - width;
    const double invsqr_radius1 = 1 / (double)(radius1 * radius1);
    const double invsqr_radius2 = 1 / (double)(radius2 * radius2);
    const double invsqr_inner_radius1 =
        1 / (double)(inner_radius1 * inner_radius1);
    const double invsqr_inner_radius2 =
        1 / (double)(inner_radius2 * inner_radius2);

    // Calculate the minimum dot product which any point on the arc
    // can have with the middle angle, if you normalise the point as a vector
    // from the center of the circle
    const double min_dotproduct =
        (angle_distance < M_PI) ? cos(angle_middle - angle_start) : -1.0;

    // Calculate the bounding rect for the arc
    int minx = 0;
    int miny = 0;
    int maxx = -1;
    int maxy = -1;
    calc_arc_bounds(surf, surf_clip_rect, angle_start, angle_stop, radius1,
                    radius2, inner_radius1, inner_radius2, invsqr_radius1,
                    invsqr_radius2, invsqr_inner_radius1, invsqr_inner_radius2,
                    min_dotproduct, x_middle, y_middle, x_center, y_center,
                    &minx, &miny, &maxx, &maxy);

    // Early return to avoid weird bounding box issues
    if (minx >= maxx || miny >= maxy) {
        return;
    }

    // Iterate over every pixel within the circle and
    // check if it's in the arc
    const int max_required_y = MAX(maxy, -miny);
    for (int y = 0; y <= max_required_y; ++y) {
        // Check if positive y is within the bounds
        // and do the same for negative y
        const int pos_y = (y >= miny) && (y <= maxy);
        const int neg_y = (-y >= miny) && (-y <= maxy);

        // Precalculate y squared
        const int y2 = y * y;

        // Find the boundaries of the outer and inner circle radii
        // use 0 as the inner radius by default
        const int x_outer = (int)(radius1 * sqrt(1.0 - y2 * invsqr_radius2));
        int x_inner = 0;
        if (y < inner_radius2) {
            x_inner =
                (int)(inner_radius1 * sqrt(1.0 - y2 * invsqr_inner_radius2));
        }

        // Precalculate positive and negative y offsets
        const int py_offset = y_center + y;
        const int ny_offset = y_center - y;

        // Precalculate the y component of the dot product
        const double y_dot = y * y_middle;

        // Iterate over every x value within the bounds
        // of the circle
        for (int x = x_inner; x <= x_outer; x++) {
            // Check if positive x is within the bounds
            // and do the same for negative x
            const int pos_x = (x >= minx) && (x <= maxx);
            const int neg_x = (-x >= minx) && (-x <= maxx);
            // Skip coordinate to avoid unnecessary calculations if neither
            // positive nor negative x are within the allowed ranges
            if (!(pos_x || neg_x)) {
                continue;
            }

            // Precalculate offsets for positive and negative x
            const int px_offset = x_center + x;
            const int nx_offset = x_center - x;

            // Precalculate the minimum dot product require for the point to be
            // within the arc's angles
            const double cmp = min_dotproduct * sqrt(x * x + y2);

            // Precalculate the x component of the dot product
            const double x_dot = x * x_middle;

            // Check if the point is within the arc for each quadrant
            if (pos_y && pos_x && (x_dot + y_dot >= cmp)) {
                unsafe_set_at(surf, px_offset, py_offset, color);
            }
            if (pos_y && neg_x && (-x_dot + y_dot >= cmp)) {
                unsafe_set_at(surf, nx_offset, py_offset, color);
            }
            if (neg_y && pos_x && (x_dot - y_dot >= cmp)) {
                unsafe_set_at(surf, px_offset, ny_offset, color);
            }
            if (neg_y && neg_x && (-x_dot - y_dot >= cmp)) {
                unsafe_set_at(surf, nx_offset, ny_offset, color);
            }
        }
    }

    drawn_area[0] = minx + x_center;
    drawn_area[1] = miny + y_center;
    drawn_area[2] = maxx + x_center;
    drawn_area[3] = maxy + y_center;
}

/* Bresenham Circle Algorithm
 * adapted from: https://de.wikipedia.org/wiki/Bresenham-Algorithmus
 * with additional line width parameter
 */
static void
draw_circle_bresenham(SDL_Surface *surf, SDL_Rect surf_clip_rect, int x0,
                      int y0, int radius, int thickness, Uint32 color,
                      int *drawn_area)
{
    long long x = 0;
    long long y = radius;
    long long radius_squared = radius * radius;
    long long double_radius_squared = 2 * radius_squared;
    double d1 = radius_squared * (1.25 - radius);
    long long dx = 0;
    long long dy = double_radius_squared * y;

    int line = 1;
    long long radius_inner = radius - thickness + 1;
    long long x_inner = 0;
    long long y_inner = radius_inner;
    long long radius_inner_squared = radius_inner * radius_inner;
    long long double_radius_inner_squared = 2 * radius_inner_squared;
    double d1_inner = radius_inner_squared * (1.25 - radius_inner);
    double d2_inner = 0;
    long long dx_inner = 0;
    long long dy_inner = double_radius_inner_squared * y_inner;

    while (dx < dy) {
        while (d1 < 0) {
            x++;
            dx += double_radius_squared;
            d1 += dx + radius_squared;
        }
        if (line) {
            drawhorzlineclipbounding(surf, surf_clip_rect, color, x0 - (int)x,
                                     y0 - (int)y, x0 + (int)x - 1, drawn_area);
            drawhorzlineclipbounding(surf, surf_clip_rect, color, x0 - (int)x,
                                     y0 + (int)y - 1, x0 + (int)x - 1,
                                     drawn_area);
        }
        else {
            drawhorzlineclipbounding(surf, surf_clip_rect, color, x0 - (int)x,
                                     y0 - (int)y, x0 - (int)x_inner,
                                     drawn_area);
            drawhorzlineclipbounding(surf, surf_clip_rect, color, x0 - (int)x,
                                     y0 + (int)y - 1, x0 - (int)x_inner,
                                     drawn_area);
            drawhorzlineclipbounding(surf, surf_clip_rect, color,
                                     x0 + (int)x_inner - 1, y0 - (int)y,
                                     x0 + (int)x - 1, drawn_area);
            drawhorzlineclipbounding(surf, surf_clip_rect, color,
                                     x0 + (int)x_inner - 1, y0 + (int)y - 1,
                                     x0 + (int)x - 1, drawn_area);
        }
        x++;
        y--;
        dx += double_radius_squared;
        dy -= double_radius_squared;
        d1 += dx - dy + radius_squared;
        if (line && y < radius_inner) {
            line = 0;
        }
        if (!line) {
            while (d1_inner < 0) {
                x_inner += 1;
                dx_inner += double_radius_inner_squared;
                d1_inner += dx_inner + radius_inner_squared;
            }
            x_inner++;
            y_inner--;
            dx_inner += double_radius_inner_squared;
            dy_inner -= double_radius_inner_squared;
            d1_inner += dx_inner - dy_inner + radius_inner_squared;
        }
    }
    d1 = radius_squared *
         ((x + 0.5) * (x + 0.5) + (y - 1) * (y - 1) - radius_squared);
    while (y >= 0) {
        if (line) {
            drawhorzlineclipbounding(surf, surf_clip_rect, color, x0 - (int)x,
                                     y0 - (int)y, x0 + (int)x - 1, drawn_area);
            drawhorzlineclipbounding(surf, surf_clip_rect, color, x0 - (int)x,
                                     y0 + (int)y - 1, x0 + (int)x - 1,
                                     drawn_area);
        }
        else {
            drawhorzlineclipbounding(surf, surf_clip_rect, color, x0 - (int)x,
                                     y0 - (int)y, x0 - (int)x_inner,
                                     drawn_area);
            drawhorzlineclipbounding(surf, surf_clip_rect, color, x0 - (int)x,
                                     y0 + (int)y - 1, x0 - (int)x_inner,
                                     drawn_area);
            drawhorzlineclipbounding(surf, surf_clip_rect, color,
                                     x0 + (int)x_inner - 1, y0 - (int)y,
                                     x0 + (int)x - 1, drawn_area);
            drawhorzlineclipbounding(surf, surf_clip_rect, color,
                                     x0 + (int)x_inner - 1, y0 + (int)y - 1,
                                     x0 + (int)x - 1, drawn_area);
        }
        if (d1 > 0) {
            y--;
            dy -= double_radius_squared;
            d1 += radius_squared - dy;
        }
        else {
            y--;
            x++;
            dx += double_radius_squared;
            dy -= double_radius_squared;
            d1 += dx - dy + radius_squared;
        }
        if (line && y < radius_inner) {
            line = 0;
        }
        if (!line) {
            if (dx_inner < dy_inner) {
                while (d1_inner < 0) {
                    x_inner += 1;
                    dx_inner += double_radius_inner_squared;
                    d1_inner += dx_inner + radius_inner_squared;
                }
                x_inner++;
                y_inner--;
                dx_inner += double_radius_inner_squared;
                dy_inner -= double_radius_inner_squared;
                d1_inner += dx_inner - dy_inner + radius_inner_squared;
            }
            else {
                if (!d2_inner) {
                    d2_inner =
                        radius_inner_squared *
                        ((x_inner + 0.5) * (x_inner + 0.5) +
                         (y_inner - 1) * (y_inner - 1) - radius_inner_squared);
                }
                if (d2_inner > 0) {
                    y_inner--;
                    dy_inner -= double_radius_inner_squared;
                    d2_inner += radius_inner_squared - dy_inner;
                }
                else {
                    y_inner--;
                    x_inner++;
                    dx_inner += double_radius_inner_squared;
                    dy_inner -= double_radius_inner_squared;
                    d2_inner += dx_inner - dy_inner + radius_inner_squared;
                }
            }
        }
    }
}

static void
draw_circle_bresenham_thin(SDL_Surface *surf, SDL_Rect surf_clip_rect, int x0,
                           int y0, int radius, Uint32 color, int *drawn_area)
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

        set_and_check_rect(surf, surf_clip_rect, x0 + x - 1, y0 + y - 1, color,
                           drawn_area); /* 7 */
        set_and_check_rect(surf, surf_clip_rect, x0 - x, y0 + y - 1, color,
                           drawn_area); /* 6 */
        set_and_check_rect(surf, surf_clip_rect, x0 + x - 1, y0 - y, color,
                           drawn_area); /* 2 */
        set_and_check_rect(surf, surf_clip_rect, x0 - x, y0 - y, color,
                           drawn_area); /* 3 */
        set_and_check_rect(surf, surf_clip_rect, x0 + y - 1, y0 + x - 1, color,
                           drawn_area); /* 8 */
        set_and_check_rect(surf, surf_clip_rect, x0 + y - 1, y0 - x, color,
                           drawn_area); /* 1 */
        set_and_check_rect(surf, surf_clip_rect, x0 - y, y0 + x - 1, color,
                           drawn_area); /* 5 */
        set_and_check_rect(surf, surf_clip_rect, x0 - y, y0 - x, color,
                           drawn_area); /* 4 */
    }
}

static void
draw_circle_quadrant(SDL_Surface *surf, SDL_Rect surf_clip_rect, int x0,
                     int y0, int radius, int thickness, Uint32 color,
                     int top_right, int top_left, int bottom_left,
                     int bottom_right, int *drawn_area)
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
        if (top_right > 0) {
            set_and_check_rect(surf, surf_clip_rect, x0, y0 - 1, color,
                               drawn_area);
        }
        if (top_left > 0) {
            set_and_check_rect(surf, surf_clip_rect, x0 - 1, y0 - 1, color,
                               drawn_area);
        }
        if (bottom_left > 0) {
            set_and_check_rect(surf, surf_clip_rect, x0 - 1, y0, color,
                               drawn_area);
        }
        if (bottom_right > 0) {
            set_and_check_rect(surf, surf_clip_rect, x0, y0, color,
                               drawn_area);
        }
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

            if (thickness > 1) {
                thickness = y - i_y;
            }

            /* Numbers represent parts of circle function draw in radians
            interval: [number - 1 * pi / 4, number * pi / 4] */
            if (top_right > 0) {
                for (i = 0; i < thickness; i++) {
                    y1 = y - i;
                    if ((y0 - y1) < (y0 - x)) {
                        set_and_check_rect(surf, surf_clip_rect, x0 + x - 1,
                                           y0 - y1, color, drawn_area); /* 2 */
                    }
                    if ((x0 + y1 - 1) >= (x0 + x - 1)) {
                        set_and_check_rect(surf, surf_clip_rect, x0 + y1 - 1,
                                           y0 - x, color, drawn_area); /* 1 */
                    }
                }
            }
            if (top_left > 0) {
                for (i = 0; i < thickness; i++) {
                    y1 = y - i;
                    if ((y0 - y1) <= (y0 - x)) {
                        set_and_check_rect(surf, surf_clip_rect, x0 - x,
                                           y0 - y1, color, drawn_area); /* 3 */
                    }
                    if ((x0 - y1) < (x0 - x)) {
                        set_and_check_rect(surf, surf_clip_rect, x0 - y1,
                                           y0 - x, color, drawn_area); /* 4 */
                    }
                }
            }
            if (bottom_left > 0) {
                for (i = 0; i < thickness; i++) {
                    y1 = y - i;
                    if ((x0 - y1) <= (x0 - x)) {
                        set_and_check_rect(surf, surf_clip_rect, x0 - y1,
                                           y0 + x - 1, color,
                                           drawn_area); /* 5 */
                    }
                    if ((y0 + y1 - 1) > (y0 + x - 1)) {
                        set_and_check_rect(surf, surf_clip_rect, x0 - x,
                                           y0 + y1 - 1, color,
                                           drawn_area); /* 6 */
                    }
                }
            }
            if (bottom_right > 0) {
                for (i = 0; i < thickness; i++) {
                    y1 = y - i;
                    if ((y0 + y1 - 1) >= (y0 + x - 1)) {
                        set_and_check_rect(surf, surf_clip_rect, x0 + x - 1,
                                           y0 + y1 - 1, color,
                                           drawn_area); /* 7 */
                    }
                    if ((x0 + y1 - 1) > (x0 + x - 1)) {
                        set_and_check_rect(surf, surf_clip_rect, x0 + y1 - 1,
                                           y0 + x - 1, color,
                                           drawn_area); /* 8 */
                    }
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
                    set_and_check_rect(surf, surf_clip_rect, x0 + y - 1, y1,
                                       color, drawn_area); /* 1 */
                }
                for (y1 = y0 - y; y1 <= y0; y1++) {
                    set_and_check_rect(surf, surf_clip_rect, x0 + x - 1, y1,
                                       color, drawn_area); /* 2 */
                }
            }
            if (top_left > 0) {
                for (y1 = y0 - x; y1 <= y0; y1++) {
                    set_and_check_rect(surf, surf_clip_rect, x0 - y, y1, color,
                                       drawn_area); /* 4 */
                }
                for (y1 = y0 - y; y1 <= y0; y1++) {
                    set_and_check_rect(surf, surf_clip_rect, x0 - x, y1, color,
                                       drawn_area); /* 3 */
                }
            }
            if (bottom_left > 0) {
                for (y1 = y0; y1 < y0 + x; y1++) {
                    set_and_check_rect(surf, surf_clip_rect, x0 - y, y1, color,
                                       drawn_area); /* 4 */
                }
                for (y1 = y0; y1 < y0 + y; y1++) {
                    set_and_check_rect(surf, surf_clip_rect, x0 - x, y1, color,
                                       drawn_area); /* 3 */
                }
            }
            if (bottom_right > 0) {
                for (y1 = y0; y1 < y0 + x; y1++) {
                    set_and_check_rect(surf, surf_clip_rect, x0 + y - 1, y1,
                                       color, drawn_area); /* 1 */
                }
                for (y1 = y0; y1 < y0 + y; y1++) {
                    set_and_check_rect(surf, surf_clip_rect, x0 + x - 1, y1,
                                       color, drawn_area); /* 2 */
                }
            }
        }
    }
}

static void
draw_circle_filled(SDL_Surface *surf, SDL_Rect surf_clip_rect, int x0, int y0,
                   int radius, Uint32 color, int *drawn_area)
{
    int f = 1 - radius;
    int ddF_x = 0;
    int ddF_y = -2 * radius;
    int x = 0;
    int y = radius;
    int xmax = INT_MIN;

    if (x0 < 0) {
        xmax = x0 + INT_MAX + 1;
    }
    else {
        xmax = INT_MAX - x0;
    }

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
            drawhorzlineclipbounding(surf, surf_clip_rect, color, x0 - x,
                                     y0 + y - 1, x0 + MIN(x - 1, xmax),
                                     drawn_area);
            drawhorzlineclipbounding(surf, surf_clip_rect, color, x0 - x,
                                     y0 - y, x0 + MIN(x - 1, xmax),
                                     drawn_area);
        }
        drawhorzlineclipbounding(surf, surf_clip_rect, color, x0 - y,
                                 y0 + x - 1, x0 + MIN(y - 1, xmax),
                                 drawn_area);
        drawhorzlineclipbounding(surf, surf_clip_rect, color, x0 - y, y0 - x,
                                 x0 + MIN(y - 1, xmax), drawn_area);
    }
}

static void
draw_eight_symetric_pixels(SDL_Surface *surf, SDL_Rect surf_clip_rect,
                           PG_PixelFormat *surf_format, int x0, int y0,
                           Uint32 color, int x, int y, float opacity,
                           int top_right, int top_left, int bottom_left,
                           int bottom_right, int *drawn_area)
{
    opacity = opacity / 255.0f;
    Uint32 pixel_color;
    if (top_right == 1) {
        pixel_color = get_antialiased_color(surf, surf_clip_rect, surf_format,
                                            x0 + x, y0 - y, color, opacity);
        set_and_check_rect(surf, surf_clip_rect, x0 + x, y0 - y, pixel_color,
                           drawn_area);
        pixel_color = get_antialiased_color(surf, surf_clip_rect, surf_format,
                                            x0 + y, y0 - x, color, opacity);
        set_and_check_rect(surf, surf_clip_rect, x0 + y, y0 - x, pixel_color,
                           drawn_area);
    }
    if (top_left == 1) {
        pixel_color = get_antialiased_color(surf, surf_clip_rect, surf_format,
                                            x0 - x, y0 - y, color, opacity);
        set_and_check_rect(surf, surf_clip_rect, x0 - x, y0 - y, pixel_color,
                           drawn_area);
        pixel_color = get_antialiased_color(surf, surf_clip_rect, surf_format,
                                            x0 - y, y0 - x, color, opacity);
        set_and_check_rect(surf, surf_clip_rect, x0 - y, y0 - x, pixel_color,
                           drawn_area);
    }
    if (bottom_left == 1) {
        pixel_color = get_antialiased_color(surf, surf_clip_rect, surf_format,
                                            x0 - x, y0 + y, color, opacity);
        set_and_check_rect(surf, surf_clip_rect, x0 - x, y0 + y, pixel_color,
                           drawn_area);
        pixel_color = get_antialiased_color(surf, surf_clip_rect, surf_format,
                                            x0 - y, y0 + x, color, opacity);
        set_and_check_rect(surf, surf_clip_rect, x0 - y, y0 + x, pixel_color,
                           drawn_area);
    }
    if (bottom_right == 1) {
        pixel_color = get_antialiased_color(surf, surf_clip_rect, surf_format,
                                            x0 + x, y0 + y, color, opacity);
        set_and_check_rect(surf, surf_clip_rect, x0 + x, y0 + y, pixel_color,
                           drawn_area);
        pixel_color = get_antialiased_color(surf, surf_clip_rect, surf_format,
                                            x0 + y, y0 + x, color, opacity);
        set_and_check_rect(surf, surf_clip_rect, x0 + y, y0 + x, pixel_color,
                           drawn_area);
    }
}

/* Xiaolin Wu Circle Algorithm
 * adapted from: https://cgg.mff.cuni.cz/~pepca/ref/WU.pdf
 * with additional line width parameter and quadrants option
 */
static void
draw_circle_xiaolinwu(SDL_Surface *surf, SDL_Rect surf_clip_rect,
                      PG_PixelFormat *surf_format, int x0, int y0, int radius,
                      int thickness, Uint32 color, int top_right, int top_left,
                      int bottom_left, int bottom_right, int *drawn_area)
{
    for (int layer_radius = radius - thickness; layer_radius <= radius;
         layer_radius++) {
        int x = 0;
        int y = layer_radius;
        double pow_layer_r = pow(layer_radius, 2);
        double prev_opacity = 0.0;
        if (layer_radius == radius - thickness) {
            while (x < y) {
                double height = sqrt(pow_layer_r - pow(x, 2));
                double opacity = 255.0 * (ceil(height) - height);
                if (opacity < prev_opacity) {
                    --y;
                }
                prev_opacity = opacity;
                draw_eight_symetric_pixels(surf, surf_clip_rect, surf_format,
                                           x0, y0, color, x, y, 255.0f,
                                           top_right, top_left, bottom_left,
                                           bottom_right, drawn_area);
                draw_eight_symetric_pixels(
                    surf, surf_clip_rect, surf_format, x0, y0, color, x, y - 1,
                    (float)opacity, top_right, top_left, bottom_left,
                    bottom_right, drawn_area);
                ++x;
            }
        }
        else if (layer_radius == radius) {
            while (x < y) {
                double height = sqrt(pow_layer_r - pow(x, 2));
                double opacity = 255.0 * (ceil(height) - height);
                if (opacity < prev_opacity) {
                    --y;
                }
                prev_opacity = opacity;
                draw_eight_symetric_pixels(
                    surf, surf_clip_rect, surf_format, x0, y0, color, x, y,
                    255.0f - (float)opacity, top_right, top_left, bottom_left,
                    bottom_right, drawn_area);
                draw_eight_symetric_pixels(surf, surf_clip_rect, surf_format,
                                           x0, y0, color, x, y - 1, 255.0f,
                                           top_right, top_left, bottom_left,
                                           bottom_right, drawn_area);
                ++x;
            }
        }
        else {
            while (x < y) {
                double height = sqrt(pow_layer_r - pow(x, 2));
                double opacity = 255.0 * (ceil(height) - height);
                if (opacity < prev_opacity) {
                    --y;
                }
                prev_opacity = opacity;
                draw_eight_symetric_pixels(surf, surf_clip_rect, surf_format,
                                           x0, y0, color, x, y, 255.0f,
                                           top_right, top_left, bottom_left,
                                           bottom_right, drawn_area);
                draw_eight_symetric_pixels(surf, surf_clip_rect, surf_format,
                                           x0, y0, color, x, y - 1, 255.0f,
                                           top_right, top_left, bottom_left,
                                           bottom_right, drawn_area);
                ++x;
            }
        }
    }
}

static void
draw_circle_xiaolinwu_thin(SDL_Surface *surf, SDL_Rect surf_clip_rect,
                           PG_PixelFormat *surf_format, int x0, int y0,
                           int radius, Uint32 color, int top_right,
                           int top_left, int bottom_left, int bottom_right,
                           int *drawn_area)
{
    int x = 0;
    int y = radius;
    double pow_r = pow(radius, 2);
    double prev_opacity = 0.0;
    while (x < y) {
        double height = sqrt(pow_r - pow(x, 2));
        double opacity = 255.0 * (ceil(height) - height);
        if (opacity < prev_opacity) {
            --y;
        }
        prev_opacity = opacity;
        draw_eight_symetric_pixels(surf, surf_clip_rect, surf_format, x0, y0,
                                   color, x, y, 255.0f - (float)opacity,
                                   top_right, top_left, bottom_left,
                                   bottom_right, drawn_area);
        draw_eight_symetric_pixels(surf, surf_clip_rect, surf_format, x0, y0,
                                   color, x, y - 1, (float)opacity, top_right,
                                   top_left, bottom_left, bottom_right,
                                   drawn_area);
        ++x;
    }
}

static void
draw_ellipse_filled(SDL_Surface *surf, SDL_Rect surf_clip_rect, int x0, int y0,
                    int width, int height, Uint32 color, int *drawn_area)
{
    long long dx, dy, x, y;
    int x_offset, y_offset;
    double d1, d2;
    if (width == 1) {
        draw_line(surf, surf_clip_rect, x0, y0, x0, y0 + height - 1, color,
                  drawn_area);
        return;
    }
    if (height == 1) {
        drawhorzlineclipbounding(surf, surf_clip_rect, color, x0, y0,
                                 x0 + width - 1, drawn_area);
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
        drawhorzlineclipbounding(surf, surf_clip_rect, color, x0 - (int)x,
                                 y0 - (int)y, x0 + (int)x - x_offset,
                                 drawn_area);
        drawhorzlineclipbounding(surf, surf_clip_rect, color, x0 - (int)x,
                                 y0 + (int)y - y_offset,
                                 x0 + (int)x - x_offset, drawn_area);
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
    d2 = (((double)height * height) * ((x + 0.5) * (x + 0.5))) +
         (((double)width * width) * ((y - 1) * (y - 1))) -
         ((double)width * width * height * height);
    while (y >= 0) {
        drawhorzlineclipbounding(surf, surf_clip_rect, color, x0 - (int)x,
                                 y0 - (int)y, x0 + (int)x - x_offset,
                                 drawn_area);
        drawhorzlineclipbounding(surf, surf_clip_rect, color, x0 - (int)x,
                                 y0 + (int)y - y_offset,
                                 x0 + (int)x - x_offset, drawn_area);
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
draw_ellipse_thickness(SDL_Surface *surf, SDL_Rect surf_clip_rect, int x0,
                       int y0, int width, int height, int thickness,
                       Uint32 color, int *drawn_area)
{
    long long dx, dy, dx_inner, dy_inner, x, y, x_inner, y_inner;
    int line, x_offset, y_offset;
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
    d1_inner =
        ((height - thickness) * (height - thickness)) -
        ((width - thickness) * (width - thickness) * (height - thickness)) +
        (0.25 * (width - thickness) * (width - thickness));
    dx = 2 * height * height * x;
    dy = 2 * width * width * y;
    dx_inner = 2 * (height - thickness) * (height - thickness) * x_inner;
    dy_inner = 2 * (width - thickness) * (width - thickness) * y_inner;
    while (dx < dy) {
        if (line) {
            drawhorzlineclipbounding(surf, surf_clip_rect, color, x0 - (int)x,
                                     y0 - (int)y, x0 + (int)x - x_offset,
                                     drawn_area);
            drawhorzlineclipbounding(surf, surf_clip_rect, color, x0 - (int)x,
                                     y0 + (int)y - y_offset,
                                     x0 + (int)x - x_offset, drawn_area);
        }
        else {
            drawhorzlineclipbounding(surf, surf_clip_rect, color, x0 - (int)x,
                                     y0 - (int)y, x0 - (int)x_inner,
                                     drawn_area);
            drawhorzlineclipbounding(surf, surf_clip_rect, color, x0 - (int)x,
                                     y0 + (int)y - y_offset, x0 - (int)x_inner,
                                     drawn_area);
            drawhorzlineclipbounding(surf, surf_clip_rect, color,
                                     x0 + (int)x - x_offset, y0 - (int)y,
                                     x0 + (int)x_inner - x_offset, drawn_area);
            drawhorzlineclipbounding(surf, surf_clip_rect, color,
                                     x0 + (int)x - x_offset,
                                     y0 + (int)y - y_offset,
                                     x0 + (int)x_inner - x_offset, drawn_area);
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
                        dx_inner = dx_inner + (2 * (height - thickness) *
                                               (height - thickness));
                        d1_inner =
                            d1_inner + dx_inner +
                            ((height - thickness) * (height - thickness));
                    }
                    x_inner++;
                    y_inner--;
                    dx_inner = dx_inner + (2 * (height - thickness) *
                                           (height - thickness));
                    dy_inner = dy_inner -
                               (2 * (width - thickness) * (width - thickness));
                    d1_inner = d1_inner + dx_inner - dy_inner +
                               ((height - thickness) * (height - thickness));
                }
            }
        }
    }
    d2 = (((double)height * height) * ((x + 0.5) * (x + 0.5))) +
         (((double)width * width) * ((y - 1) * (y - 1))) -
         ((double)width * width * height * height);
    while (y >= 0) {
        if (line) {
            drawhorzlineclipbounding(surf, surf_clip_rect, color, x0 - (int)x,
                                     y0 - (int)y, x0 + (int)x - x_offset,
                                     drawn_area);
            drawhorzlineclipbounding(surf, surf_clip_rect, color, x0 - (int)x,
                                     y0 + (int)y - y_offset,
                                     x0 + (int)x - x_offset, drawn_area);
        }
        else {
            drawhorzlineclipbounding(surf, surf_clip_rect, color, x0 - (int)x,
                                     y0 - (int)y, x0 - (int)x_inner,
                                     drawn_area);
            drawhorzlineclipbounding(surf, surf_clip_rect, color, x0 - (int)x,
                                     y0 + (int)y - y_offset, x0 - (int)x_inner,
                                     drawn_area);
            drawhorzlineclipbounding(surf, surf_clip_rect, color,
                                     x0 + (int)x - x_offset, y0 - (int)y,
                                     x0 + (int)x_inner - x_offset, drawn_area);
            drawhorzlineclipbounding(surf, surf_clip_rect, color,
                                     x0 + (int)x - x_offset,
                                     y0 + (int)y - y_offset,
                                     x0 + (int)x_inner - x_offset, drawn_area);
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
                    dx_inner = dx_inner + (2 * (height - thickness) *
                                           (height - thickness));
                    d1_inner = d1_inner + dx_inner +
                               ((height - thickness) * (height - thickness));
                }
                x_inner++;
                y_inner--;
                dx_inner = dx_inner +
                           (2 * (height - thickness) * (height - thickness));
                dy_inner =
                    dy_inner - (2 * (width - thickness) * (width - thickness));
                d1_inner = d1_inner + dx_inner - dy_inner +
                           ((height - thickness) * (height - thickness));
            }
            else if (y_inner >= 0) {
                if (d2_inner == 0) {
                    d2_inner =
                        ((((double)height - thickness) *
                          (height - thickness)) *
                         ((x_inner + 0.5) * (x_inner + 0.5))) +
                        ((((double)width - thickness) * (width - thickness)) *
                         ((y_inner - 1) * (y_inner - 1))) -
                        (((double)width - thickness) * (width - thickness) *
                         (height - thickness) * (height - thickness));
                }
                if (d2_inner > 0) {
                    y_inner--;
                    dy_inner = dy_inner -
                               (2 * (width - thickness) * (width - thickness));
                    d2_inner = d2_inner +
                               ((width - thickness) * (width - thickness)) -
                               dy_inner;
                }
                else {
                    y_inner--;
                    x_inner++;
                    dx_inner = dx_inner + (2 * (height - thickness) *
                                           (height - thickness));
                    dy_inner = dy_inner -
                               (2 * (width - thickness) * (width - thickness));
                    d2_inner = d2_inner + dx_inner - dy_inner +
                               ((width - thickness) * (width - thickness));
                }
            }
        }
    }
}

static void
draw_fillpoly(SDL_Surface *surf, SDL_Rect surf_clip_rect, int *point_x,
              int *point_y, Py_ssize_t num_points, Uint32 color,
              int *drawn_area)
{
    /* point_x : x coordinates of the points
     * point-y : the y coordinates of the points
     * num_points : the number of points
     */
    Py_ssize_t i, i_previous;  // i_previous is the index of the point before i
    int y, miny, maxy;
    int x1, y1;
    int x2, y2;
    float intersect;
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
        drawhorzlineclipbounding(surf, surf_clip_rect, color, minx, miny, maxx,
                                 drawn_area);
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
                intersect = (y - y1) * (x2 - x1) / (float)(y2 - y1);
                if (n_intersections % 2 == 0) {
                    intersect = (float)floor(intersect);
                }
                else {
                    intersect = (float)ceil(intersect);
                }
                x_intersect[n_intersections++] = (int)intersect + x1;
            }
        }
        qsort(x_intersect, n_intersections, sizeof(int), compare_int);
        for (i = 0; (i < n_intersections); i += 2) {
            drawhorzlineclipbounding(surf, surf_clip_rect, color,
                                     x_intersect[i], y, x_intersect[i + 1],
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
            drawhorzlineclipbounding(surf, surf_clip_rect, color, point_x[i],
                                     y, point_x[i_previous], drawn_area);
        }
    }
    PyMem_Free(x_intersect);
}

static void
draw_rect(SDL_Surface *surf, SDL_Rect surf_clip_rect, int x1, int y1, int x2,
          int y2, int width, Uint32 color)
{
    int i;
    for (i = 0; i < width; i++) {
        drawhorzlineclip(surf, surf_clip_rect, color, x1, y1 + i, x2);
        drawhorzlineclip(surf, surf_clip_rect, color, x1, y2 - i, x2);
    }
    for (i = 0; i < (y2 - y1) - 2 * width + 1; i++) {
        drawhorzlineclip(surf, surf_clip_rect, color, x1, y1 + width + i,
                         x1 + width - 1);
        drawhorzlineclip(surf, surf_clip_rect, color, x2 - width + 1,
                         y1 + width + i, x2);
    }
}

static void
draw_round_rect(SDL_Surface *surf, SDL_Rect surf_clip_rect, int x1, int y1,
                int x2, int y2, int radius, int width, Uint32 color,
                int top_left, int top_right, int bottom_left, int bottom_right,
                int *drawn_area)
{
    int pts[16], i;
    float q_top, q_left, q_bottom, q_right, f;
    if (top_left < 0) {
        top_left = radius;
    }
    if (top_right < 0) {
        top_right = radius;
    }
    if (bottom_left < 0) {
        bottom_left = radius;
    }
    if (bottom_right < 0) {
        bottom_right = radius;
    }
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
        draw_fillpoly(surf, surf_clip_rect, pts, pts + 8, 8, color,
                      drawn_area);
        draw_circle_quadrant(surf, surf_clip_rect, x2 - top_right + 1,
                             y1 + top_right, top_right, 0, color, 1, 0, 0, 0,
                             drawn_area);
        draw_circle_quadrant(surf, surf_clip_rect, x1 + top_left,
                             y1 + top_left, top_left, 0, color, 0, 1, 0, 0,
                             drawn_area);
        draw_circle_quadrant(surf, surf_clip_rect, x1 + bottom_left,
                             y2 - bottom_left + 1, bottom_left, 0, color, 0, 0,
                             1, 0, drawn_area);
        draw_circle_quadrant(surf, surf_clip_rect, x2 - bottom_right + 1,
                             y2 - bottom_right + 1, bottom_right, 0, color, 0,
                             0, 0, 1, drawn_area);
    }
    else {
        if (x2 - top_right == x1 + top_left) {
            for (i = 0; i < width; i++) {
                set_and_check_rect(
                    surf, surf_clip_rect, x1 + top_left, y1 + i, color,
                    drawn_area); /* Fill gap if reduced radius */
            }
        }
        else {
            draw_line_width(surf, surf_clip_rect, color, x1 + top_left,
                            y1 + (int)(width / 2) - 1 + width % 2,
                            x2 - top_right,
                            y1 + (int)(width / 2) - 1 + width % 2, width,
                            drawn_area); /* Top line */
        }
        if (y2 - bottom_left == y1 + top_left) {
            for (i = 0; i < width; i++) {
                set_and_check_rect(
                    surf, surf_clip_rect, x1 + i, y1 + top_left, color,
                    drawn_area); /* Fill gap if reduced radius */
            }
        }
        else {
            draw_line_width(
                surf, surf_clip_rect, color,
                x1 + (int)(width / 2) - 1 + width % 2, y1 + top_left,
                x1 + (int)(width / 2) - 1 + width % 2, y2 - bottom_left, width,
                drawn_area); /* Left line */
        }
        if (x2 - bottom_right == x1 + bottom_left) {
            for (i = 0; i < width; i++) {
                set_and_check_rect(
                    surf, surf_clip_rect, x1 + bottom_left, y2 - i, color,
                    drawn_area); /* Fill gap if reduced radius */
            }
        }
        else {
            draw_line_width(surf, surf_clip_rect, color, x1 + bottom_left,
                            y2 - (int)(width / 2), x2 - bottom_right,
                            y2 - (int)(width / 2), width,
                            drawn_area); /* Bottom line */
        }
        if (y2 - bottom_right == y1 + top_right) {
            for (i = 0; i < width; i++) {
                set_and_check_rect(
                    surf, surf_clip_rect, x2 - i, y1 + top_right, color,
                    drawn_area); /* Fill gap if reduced radius */
            }
        }
        else {
            draw_line_width(surf, surf_clip_rect, color, x2 - (int)(width / 2),
                            y1 + top_right, x2 - (int)(width / 2),
                            y2 - bottom_right, width,
                            drawn_area); /* Right line */
        }

        draw_circle_quadrant(surf, surf_clip_rect, x2 - top_right + 1,
                             y1 + top_right, top_right, width, color, 1, 0, 0,
                             0, drawn_area);
        draw_circle_quadrant(surf, surf_clip_rect, x1 + top_left,
                             y1 + top_left, top_left, width, color, 0, 1, 0, 0,
                             drawn_area);
        draw_circle_quadrant(surf, surf_clip_rect, x1 + bottom_left,
                             y2 - bottom_left + 1, bottom_left, width, color,
                             0, 0, 1, 0, drawn_area);
        draw_circle_quadrant(surf, surf_clip_rect, x2 - bottom_right + 1,
                             y2 - bottom_right + 1, bottom_right, width, color,
                             0, 0, 0, 1, drawn_area);
    }
}

/* List of python functions */
static PyMethodDef _draw_methods[] = {
    {"aaline", (PyCFunction)aaline, METH_VARARGS | METH_KEYWORDS,
     DOC_DRAW_AALINE},
    {"line", (PyCFunction)line, METH_VARARGS | METH_KEYWORDS, DOC_DRAW_LINE},
    {"aalines", (PyCFunction)aalines, METH_VARARGS | METH_KEYWORDS,
     DOC_DRAW_AALINES},
    {"lines", (PyCFunction)lines, METH_VARARGS | METH_KEYWORDS,
     DOC_DRAW_LINES},
    {"ellipse", (PyCFunction)ellipse, METH_VARARGS | METH_KEYWORDS,
     DOC_DRAW_ELLIPSE},
    {"flood_fill", (PyCFunction)flood_fill, METH_VARARGS | METH_KEYWORDS,
     DOC_DRAW_FLOODFILL},
    {"arc", (PyCFunction)arc, METH_VARARGS | METH_KEYWORDS, DOC_DRAW_ARC},
    {"circle", (PyCFunction)circle, METH_VARARGS | METH_KEYWORDS,
     DOC_DRAW_CIRCLE},
    {"aacircle", (PyCFunction)aacircle, METH_VARARGS | METH_KEYWORDS,
     DOC_DRAW_AACIRCLE},
    {"polygon", (PyCFunction)polygon, METH_VARARGS | METH_KEYWORDS,
     DOC_DRAW_POLYGON},
    {"rect", (PyCFunction)rect, METH_VARARGS | METH_KEYWORDS, DOC_DRAW_RECT},

    {NULL, NULL, 0, NULL}};

MODINIT_DEFINE(draw)
{
    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "draw",
                                         DOC_DRAW,
                                         -1,
                                         _draw_methods,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL};

    /* imported needed apis; Do this first so if there is an error
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

    /* create the module */
    return PyModule_Create(&_module);
}
