/*
  Copyright (C) 2002-2007 Ulf Ekstrom except for the bitcount function.
  This wrapper code was originally written by Danny van Bruggen(?) for
  the SCAM library, it was then converted by Ulf Ekstrom to wrap the
  bitmask library, a spinoff from SCAM.

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

/* a couple of print debugging helpers */
/*
#define CALLLOG2(x,y) fprintf(stderr, (x), (y));
#define CALLLOG(x) fprintf(stderr, (x));
*/

#define PYGAMEAPI_MASK_INTERNAL 1
#include "mask.h"

#include "pygame.h"

#include "pgcompat.h"

#include "doc/mask_doc.h"

#include "structmember.h"

#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Macro to create mask objects. This will call the type's tp_new and tp_init.
 * Params:
 *     w: width of mask
 *     h: height of mask
 *     f: fill, 1 is used to set all the bits (to 1) and 0 is used to clear
 *        all the bits (to 0)
 */
#define CREATE_MASK_OBJ(w, h, f)                                             \
    (pgMaskObject *)PyObject_CallFunction((PyObject *)&pgMask_Type, "(ii)i", \
                                          (w), (h), (f))

/* Prototypes */
static PyTypeObject pgMask_Type;
static PG_INLINE pgMaskObject *
create_mask_using_bitmask(bitmask_t *bitmask);
static PG_INLINE pgMaskObject *
create_mask_using_bitmask_and_type(bitmask_t *bitmask, PyTypeObject *ob_type);

/********** mask helper functions **********/

/* Calculate the absolute difference between 2 Uint32s. */
static PG_INLINE Uint32
abs_diff_uint32(Uint32 a, Uint32 b)
{
    return (a > b) ? a - b : b - a;
}

/********** mask object methods **********/

/* Copies the given mask. */
static PyObject *
mask_copy(PyObject *self, PyObject *_null)
{
    bitmask_t *new_bitmask = bitmask_copy(pgMask_AsBitmap(self));

    if (NULL == new_bitmask) {
        return RAISE(PyExc_MemoryError, "cannot allocate memory for bitmask");
    }

    return (PyObject *)create_mask_using_bitmask_and_type(new_bitmask,
                                                          Py_TYPE(self));
}

/* Redirects mask.copy() to mask.__copy__(). This is done to allow
 * subclasses that override the __copy__() method to also override the copy()
 * method automatically. */
static PyObject *
mask_call_copy(PyObject *self, PyObject *_null)
{
    return PyObject_CallMethodObjArgs(self, PyUnicode_FromString("__copy__"),
                                      NULL);
}

static PyObject *
mask_get_size(PyObject *self, PyObject *_null)
{
    bitmask_t *mask = pgMask_AsBitmap(self);
    return Py_BuildValue("(ii)", mask->w, mask->h);
}

/* Creates a Rect object based on the given mask's size. The rect's
 * attributes can be altered via the kwargs.
 *
 * Returns:
 *     Rect object or NULL to indicate a fail
 *
 * Ref: src_c/surface.c surf_get_rect()
 */
static PyObject *
mask_get_rect(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *rect = NULL;
    bitmask_t *bitmask = pgMask_AsBitmap(self);

    if (0 != PyTuple_GET_SIZE(args)) {
        return RAISE(PyExc_TypeError,
                     "get_rect only supports keyword arguments");
    }

    rect = pgRect_New4(0, 0, bitmask->w, bitmask->h);

    if (NULL == rect) {
        return RAISE(PyExc_MemoryError, "cannot allocate memory for rect");
    }

    if (NULL != kwargs) {
        PyObject *key = NULL, *value = NULL;
        Py_ssize_t pos = 0;

        while (PyDict_Next(kwargs, &pos, &key, &value)) {
            if ((-1 == PyObject_SetAttr(rect, key, value))) {
                Py_DECREF(rect);
                return NULL;
            }
        }
    }

    return rect;
}

static PyObject *
mask_get_at(PyObject *self, PyObject *args, PyObject *kwargs)
{
    bitmask_t *mask = pgMask_AsBitmap(self);
    int x, y, val;
    PyObject *pos = NULL;
    static char *keywords[] = {"pos", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", keywords, &pos))
        return NULL;

    if (!pg_TwoIntsFromObj(pos, &x, &y)) {
        return RAISE(PyExc_TypeError, "pos must be two numbers");
    }

    if (x >= 0 && x < mask->w && y >= 0 && y < mask->h) {
        val = bitmask_getbit(mask, x, y);
    }
    else {
        PyErr_Format(PyExc_IndexError, "%d, %d is out of bounds", x, y);
        return NULL;
    }

    return PyLong_FromLong(val);
}

static PyObject *
mask_set_at(PyObject *self, PyObject *args, PyObject *kwargs)
{
    bitmask_t *mask = pgMask_AsBitmap(self);
    int x, y, value = 1;
    PyObject *pos = NULL;
    static char *keywords[] = {"pos", "value", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|i", keywords, &pos,
                                     &value))
        return NULL;

    if (!pg_TwoIntsFromObj(pos, &x, &y)) {
        return RAISE(PyExc_TypeError, "pos must be two numbers");
    }

    if (x >= 0 && x < mask->w && y >= 0 && y < mask->h) {
        if (value) {
            bitmask_setbit(mask, x, y);
        }
        else {
            bitmask_clearbit(mask, x, y);
        }
    }
    else {
        PyErr_Format(PyExc_IndexError, "%d, %d is out of bounds", x, y);
        return NULL;
    }
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *
mask_overlap(PyObject *self, PyObject *args, PyObject *kwargs)
{
    bitmask_t *mask = pgMask_AsBitmap(self);
    bitmask_t *othermask;
    PyObject *maskobj;
    int x, y, val;
    int xp, yp;
    PyObject *offset = NULL;
    static char *keywords[] = {"other", "offset", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O", keywords,
                                     &pgMask_Type, &maskobj, &offset))
        return NULL;

    othermask = pgMask_AsBitmap(maskobj);

    if (!pg_TwoIntsFromObj(offset, &x, &y)) {
        return RAISE(PyExc_TypeError, "offset must be two numbers");
    }

    val = bitmask_overlap_pos(mask, othermask, x, y, &xp, &yp);
    if (val) {
        return Py_BuildValue("(ii)", xp, yp);
    }
    else {
        Py_INCREF(Py_None);
        return Py_None;
    }
}

static PyObject *
mask_overlap_area(PyObject *self, PyObject *args, PyObject *kwargs)
{
    bitmask_t *mask = pgMask_AsBitmap(self);
    bitmask_t *othermask;
    PyObject *maskobj;
    int x, y, val;
    PyObject *offset = NULL;
    static char *keywords[] = {"other", "offset", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O", keywords,
                                     &pgMask_Type, &maskobj, &offset)) {
        return NULL;
    }

    othermask = pgMask_AsBitmap(maskobj);

    if (!pg_TwoIntsFromObj(offset, &x, &y)) {
        return RAISE(PyExc_TypeError, "offset must be two numbers");
    }

    val = bitmask_overlap_area(mask, othermask, x, y);
    return PyLong_FromLong(val);
}

static PyObject *
mask_overlap_mask(PyObject *self, PyObject *args, PyObject *kwargs)
{
    int x, y;
    bitmask_t *bitmask = pgMask_AsBitmap(self);
    PyObject *maskobj;
    pgMaskObject *output_maskobj;
    PyObject *offset = NULL;
    static char *keywords[] = {"other", "offset", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O", keywords,
                                     &pgMask_Type, &maskobj, &offset)) {
        return NULL; /* Exception already set. */
    }

    output_maskobj = CREATE_MASK_OBJ(bitmask->w, bitmask->h, 0);

    if (!pg_TwoIntsFromObj(offset, &x, &y)) {
        return RAISE(PyExc_TypeError, "offset must be two numbers");
    }

    if (NULL == output_maskobj) {
        return NULL; /* Exception already set. */
    }

    bitmask_overlap_mask(bitmask, pgMask_AsBitmap(maskobj),
                         output_maskobj->mask, x, y);

    return (PyObject *)output_maskobj;
}

static PyObject *
mask_fill(PyObject *self, PyObject *_null)
{
    bitmask_t *mask = pgMask_AsBitmap(self);

    bitmask_fill(mask);

    Py_RETURN_NONE;
}

static PyObject *
mask_clear(PyObject *self, PyObject *_null)
{
    bitmask_t *mask = pgMask_AsBitmap(self);

    bitmask_clear(mask);

    Py_RETURN_NONE;
}

static PyObject *
mask_invert(PyObject *self, PyObject *_null)
{
    bitmask_t *mask = pgMask_AsBitmap(self);

    bitmask_invert(mask);

    Py_RETURN_NONE;
}

static PyObject *
mask_scale(PyObject *self, PyObject *args, PyObject *kwargs)
{
    int x, y;
    bitmask_t *bitmask = NULL;
    PyObject *scale = NULL;
    static char *keywords[] = {"scale", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", keywords, &scale)) {
        return NULL; /* Exception already set. */
    }

    if (!pg_TwoIntsFromObj(scale, &x, &y)) {
        return RAISE(PyExc_TypeError, "scale must be two numbers");
    }

    if (x < 0 || y < 0) {
        return RAISE(PyExc_ValueError, "cannot scale mask to negative size");
    }

    bitmask = bitmask_scale(pgMask_AsBitmap(self), x, y);

    if (NULL == bitmask) {
        return RAISE(PyExc_MemoryError, "cannot allocate memory for bitmask");
    }

    return (PyObject *)create_mask_using_bitmask(bitmask);
}

static PyObject *
mask_draw(PyObject *self, PyObject *args, PyObject *kwargs)
{
    bitmask_t *mask = pgMask_AsBitmap(self);
    bitmask_t *othermask;
    PyObject *maskobj;
    int x, y;
    PyObject *offset = NULL;
    static char *keywords[] = {"other", "offset", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O", keywords,
                                     &pgMask_Type, &maskobj, &offset)) {
        return NULL;
    }

    if (!pg_TwoIntsFromObj(offset, &x, &y)) {
        return RAISE(PyExc_TypeError, "offset must be two numbers");
    }

    othermask = pgMask_AsBitmap(maskobj);

    bitmask_draw(mask, othermask, x, y);

    Py_RETURN_NONE;
}

static PyObject *
mask_erase(PyObject *self, PyObject *args, PyObject *kwargs)
{
    bitmask_t *mask = pgMask_AsBitmap(self);
    bitmask_t *othermask;
    PyObject *maskobj;
    int x, y;
    PyObject *offset = NULL;
    static char *keywords[] = {"other", "offset", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O", keywords,
                                     &pgMask_Type, &maskobj, &offset)) {
        return NULL;
    }

    if (!pg_TwoIntsFromObj(offset, &x, &y)) {
        return RAISE(PyExc_TypeError, "offset must be two numbers");
    }

    othermask = pgMask_AsBitmap(maskobj);

    bitmask_erase(mask, othermask, x, y);

    Py_RETURN_NONE;
}

static PyObject *
mask_count(PyObject *self, PyObject *_null)
{
    bitmask_t *m = pgMask_AsBitmap(self);

    return PyLong_FromLong(bitmask_count(m));
}

static PyObject *
mask_centroid(PyObject *self, PyObject *_null)
{
    bitmask_t *mask = pgMask_AsBitmap(self);
    int x, y;
    long int m10, m01, m00;
    PyObject *xobj, *yobj;

    m10 = m01 = m00 = 0;

    for (x = 0; x < mask->w; x++) {
        for (y = 0; y < mask->h; y++) {
            if (bitmask_getbit(mask, x, y)) {
                m10 += x;
                m01 += y;
                m00++;
            }
        }
    }

    if (m00) {
        xobj = PyLong_FromLong(m10 / m00);
        yobj = PyLong_FromLong(m01 / m00);
    }
    else {
        xobj = PyLong_FromLong(0);
        yobj = PyLong_FromLong(0);
    }

    return Py_BuildValue("(NN)", xobj, yobj);
}

static PyObject *
mask_angle(PyObject *self, PyObject *_null)
{
    bitmask_t *mask = pgMask_AsBitmap(self);
    int x, y;
    long int m10, m01, m00, m20, m02, m11;

    m10 = m01 = m00 = m20 = m02 = m11 = 0;

    for (x = 0; x < mask->w; x++) {
        for (y = 0; y < mask->h; y++) {
            if (bitmask_getbit(mask, x, y)) {
                m10 += x;
                m20 += (long)x * x;
                m11 += (long)x * y;
                m02 += (long)y * y;
                m01 += y;
                m00++;
            }
        }
    }

    if (m00) {
        int xc = m10 / m00;
        int yc = m01 / m00;
        double theta =
            -90.0 *
            atan2(2 * (m11 / m00 - (long)xc * yc),
                  (m20 / m00 - (long)xc * xc) - (m02 / m00 - (long)yc * yc)) /
            M_PI;
        return PyFloat_FromDouble(theta);
    }
    else {
        return PyFloat_FromDouble(0);
    }
}

static PyObject *
mask_outline(PyObject *self, PyObject *args, PyObject *kwargs)
{
    bitmask_t *c = pgMask_AsBitmap(self);
    bitmask_t *m = NULL;
    PyObject *plist = NULL;
    PyObject *value = NULL;
    int x, y, firstx, firsty, secx, secy, currx, curry, nextx, nexty, n;
    int e, every = 1;
    int a[] = {1, 1, 0, -1, -1, -1, 0, 1, 1, 1, 0, -1, -1, -1};
    int b[] = {0, 1, 1, 1, 0, -1, -1, -1, 0, 1, 1, 1, 0, -1};
    static char *keywords[] = {"every", NULL};

    firstx = firsty = secx = x = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|i", keywords, &every)) {
        return NULL;
    }

    plist = PyList_New(0);
    if (!plist) {
        return RAISE(PyExc_MemoryError,
                     "outline cannot allocate memory for list");
    }

    if (!c->w || !c->h) {
        return plist;
    }

    /* Copying to a larger mask to avoid border checking. */
    m = bitmask_create(c->w + 2, c->h + 2);
    if (!m) {
        Py_DECREF(plist);
        return RAISE(PyExc_MemoryError,
                     "outline cannot allocate memory for mask");
    }

    bitmask_draw(m, c, 1, 1);

    /* find the first set pixel in the mask */
    for (y = 1; y < m->h - 1; y++) {
        for (x = 1; x < m->w - 1; x++) {
            if (bitmask_getbit(m, x, y)) {
                firstx = x;
                firsty = y;
                value = Py_BuildValue("(ii)", x - 1, y - 1);

                if (NULL == value) {
                    Py_DECREF(plist);
                    bitmask_free(m);

                    return NULL; /* Exception already set. */
                }

                if (0 != PyList_Append(plist, value)) {
                    Py_DECREF(value);
                    Py_DECREF(plist);
                    bitmask_free(m);

                    return NULL; /* Exception already set. */
                }

                Py_DECREF(value);
                break;
            }
        }
        if (bitmask_getbit(m, x, y))
            break;
    }

    /* covers the mask having zero pixels set or only the final pixel */
    if ((x == m->w - 1) && (y == m->h - 1)) {
        bitmask_free(m);
        return plist;
    }

    e = every;

    /* check just the first pixel for neighbors */
    for (n = 0; n < 8; n++) {
        if (bitmask_getbit(m, x + a[n], y + b[n])) {
            currx = secx = x + a[n];
            curry = secy = y + b[n];
            e--;
            if (!e) {
                e = every;
                value = Py_BuildValue("(ii)", secx - 1, secy - 1);

                if (NULL == value) {
                    Py_DECREF(plist);
                    bitmask_free(m);

                    return NULL; /* Exception already set. */
                }

                if (0 != PyList_Append(plist, value)) {
                    Py_DECREF(value);
                    Py_DECREF(plist);
                    bitmask_free(m);

                    return NULL; /* Exception already set. */
                }

                Py_DECREF(value);
            }
            break;
        }
    }

    /* if there are no neighbors, return */
    if (!secx) {
        bitmask_free(m);
        return plist;
    }

    /* the outline tracing loop */
    for (;;) {
        /* look around the pixel, it has to have a neighbor */
        for (n = (n + 6) & 7;; n++) {
            if (bitmask_getbit(m, currx + a[n], curry + b[n])) {
                nextx = currx + a[n];
                nexty = curry + b[n];
                e--;
                if (!e) {
                    e = every;
                    if ((curry == firsty && currx == firstx) &&
                        (secx == nextx && secy == nexty)) {
                        break;
                    }

                    value = Py_BuildValue("(ii)", nextx - 1, nexty - 1);

                    if (NULL == value) {
                        Py_DECREF(plist);
                        bitmask_free(m);

                        return NULL; /* Exception already set. */
                    }

                    if (0 != PyList_Append(plist, value)) {
                        Py_DECREF(value);
                        Py_DECREF(plist);
                        bitmask_free(m);

                        return NULL; /* Exception already set. */
                    }

                    Py_DECREF(value);
                }
                break;
            }
        }
        /* if we are back at the first pixel, and the next one will be the
           second one we visited, we are done */
        if ((curry == firsty && currx == firstx) &&
            (secx == nextx && secy == nexty)) {
            break;
        }

        curry = nexty;
        currx = nextx;
    }

    bitmask_free(m);

    return plist;
}

static PyObject *
mask_convolve(PyObject *aobj, PyObject *args, PyObject *kwargs)
{
    PyObject *bobj;
    PyObject *oobj = Py_None;
    bitmask_t *a = NULL, *b = NULL;
    int xoffset = 0, yoffset = 0;
    PyObject *offset = NULL;
    static char *keywords[] = {"other", "output", "offset", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|OO", keywords,
                                     &pgMask_Type, &bobj, &oobj, &offset)) {
        return NULL; /* Exception already set. */
    }

    if (offset && !pg_TwoIntsFromObj(offset, &xoffset, &yoffset)) {
        return RAISE(PyExc_TypeError, "offset must be two numbers");
    }

    a = pgMask_AsBitmap(aobj);
    b = pgMask_AsBitmap(bobj);

    if (oobj != Py_None) {
        /* Use this mask for the output. */
        Py_INCREF(oobj);
    }
    else {
        pgMaskObject *maskobj = CREATE_MASK_OBJ(MAX(0, a->w + b->w - 1),
                                                MAX(0, a->h + b->h - 1), 0);

        if (NULL == maskobj) {
            return NULL; /* Exception already set. */
        }

        oobj = (PyObject *)maskobj;
    }

    bitmask_convolve(a, b, pgMask_AsBitmap(oobj), xoffset, yoffset);

    return oobj;
}

/* Gets the color of a given pixel.
 *
 * Params:
 *     pixel: pixel to get the color of
 *     bpp: bytes per pixel
 *
 * Returns:
 *     pixel color
 */
static PG_INLINE Uint32
get_pixel_color(Uint8 *pixel, Uint8 bpp)
{
    switch (bpp) {
        case 1:
            return *((Uint8 *)pixel);

        case 2:
            return *((Uint16 *)pixel);

        case 3:
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
            return (pixel[0]) + (pixel[1] << 8) + (pixel[2] << 16);
#else  /* SDL_BIG_ENDIAN */
            return (pixel[2]) + (pixel[1] << 8) + (pixel[0] << 16);
#endif /* SDL_BIG_ENDIAN */

        default: /* case 4: */
            return *((Uint32 *)pixel);
    }
}

/* Sets the color of a given pixel.
 *
 * Params:
 *     pixel: pixel to set the color of
 *     bpp: bytes per pixel
 *     color: color to set
 *
 * Ref: src_c/draw.c set_pixel_32()
 */
static void
set_pixel_color(Uint8 *pixel, Uint8 bpp, Uint32 color)
{
    switch (bpp) {
        case 1:
            *pixel = color;
            break;

        case 2:
            *(Uint16 *)pixel = color;
            break;

        case 3:
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
            *(Uint16 *)pixel = color;
            pixel[2] = color >> 16;
#else  /* != SDL_LIL_ENDIAN */
            pixel[2] = color;
            pixel[1] = color >> 8;
            pixel[0] = color >> 16;
#endif /* SDL_LIL_ENDIAN */
            break;

        default: /* case 4: */
            *(Uint32 *)pixel = color;
            break;
    }
}

/* For each surface pixel's alpha that is greater than the threshold,
 * the corresponding bitmask bit is set.
 *
 * Params:
 *     surf: surface
 *     bitmask: bitmask to alter
 *     threshold: threshold used check surface pixels (alpha) against
 *
 * Returns:
 *     void
 */
static void
set_from_threshold(SDL_Surface *surf, bitmask_t *bitmask, int threshold)
{
    SDL_PixelFormat *format = surf->format;
    Uint8 bpp = format->BytesPerPixel;
    Uint8 *pixel = NULL;
    Uint8 rgba[4];
    int x, y;

    for (y = 0; y < surf->h; ++y) {
        pixel = (Uint8 *)surf->pixels + y * surf->pitch;

        for (x = 0; x < surf->w; ++x, pixel += bpp) {
            SDL_GetRGBA(get_pixel_color(pixel, bpp), format, rgba, rgba + 1,
                        rgba + 2, rgba + 3);
            if (rgba[3] > threshold) {
                bitmask_setbit(bitmask, x, y);
            }
        }
    }
}

/* For each surface pixel's color that is not equal to the colorkey, the
 * corresponding bitmask bit is set.
 *
 * Params:
 *     surf: surface
 *     bitmask: bitmask to alter
 *     colorkey: color used to check surface pixels against
 *
 * Returns:
 *     void
 */
static void
set_from_colorkey(SDL_Surface *surf, bitmask_t *bitmask, Uint32 colorkey)
{
    Uint8 bpp = surf->format->BytesPerPixel;
    Uint8 *pixel = NULL;
    int x, y;

    for (y = 0; y < surf->h; ++y) {
        pixel = (Uint8 *)surf->pixels + y * surf->pitch;

        for (x = 0; x < surf->w; ++x, pixel += bpp) {
            if (get_pixel_color(pixel, bpp) != colorkey) {
                bitmask_setbit(bitmask, x, y);
            }
        }
    }
}

/* Creates a mask from a given surface.
 *
 * Returns:
 *     Mask object or NULL to indicate a fail
 */
static PyObject *
mask_from_surface(PyObject *self, PyObject *args, PyObject *kwargs)
{
    SDL_Surface *surf = NULL;
    pgSurfaceObject *surfobj;
    pgMaskObject *maskobj = NULL;
    Uint32 colorkey;
    int threshold = 127; /* default value */
    int use_thresh = 1;
    static char *keywords[] = {"surface", "threshold", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|i", keywords,
                                     &pgSurface_Type, &surfobj, &threshold)) {
        return NULL; /* Exception already set. */
    }

    surf = pgSurface_AsSurface(surfobj);

    if (!surf)
        return RAISE(pgExc_SDLError, "display Surface quit");

    if (surf->w < 0 || surf->h < 0) {
        return RAISE(PyExc_ValueError,
                     "cannot create mask with negative size");
    }

    maskobj = CREATE_MASK_OBJ(surf->w, surf->h, 0);

    if (NULL == maskobj) {
        return NULL; /* Exception already set. */
    }

    if (surf->w == 0 || surf->h == 0) {
        /* Nothing left to do for 0 sized surfaces. */
        return (PyObject *)maskobj;
    }

    if (!pgSurface_Lock(surfobj)) {
        Py_DECREF((PyObject *)maskobj);
        return RAISE(PyExc_RuntimeError, "cannot lock surface");
    }

    Py_BEGIN_ALLOW_THREADS; /* Release the GIL. */

    use_thresh = (SDL_GetColorKey(surf, &colorkey) == -1);

    if (use_thresh) {
        set_from_threshold(surf, maskobj->mask, threshold);
    }
    else {
        set_from_colorkey(surf, maskobj->mask, colorkey);
    }

    Py_END_ALLOW_THREADS; /* Obtain the GIL. */

    if (!pgSurface_Unlock(surfobj)) {
        Py_DECREF((PyObject *)maskobj);
        return RAISE(PyExc_RuntimeError, "cannot unlock surface");
    }

    return (PyObject *)maskobj;
}

/*

palette_colors - this only affects surfaces with a palette
    if true we look at the colors from the palette,
    otherwise we threshold the pixel values.  This is useful if
    the surface is actually greyscale colors, and not palette colors.

*/

void
bitmask_threshold(bitmask_t *m, SDL_Surface *surf, SDL_Surface *surf2,
                  Uint32 color, Uint32 threshold, int palette_colors)
{
    int x, y, rshift, gshift, bshift, rshift2, gshift2, bshift2;
    int rloss, gloss, bloss, rloss2, gloss2, bloss2;
    Uint8 *pixels, *pixels2;
    SDL_PixelFormat *format, *format2;
    Uint32 the_color, the_color2, rmask, gmask, bmask, rmask2, gmask2, bmask2;
    Uint8 *pix;
    Uint8 r, g, b, a;
    Uint8 tr, tg, tb, ta;
    int bpp1, bpp2;

    format = surf->format;
    rmask = format->Rmask;
    gmask = format->Gmask;
    bmask = format->Bmask;
    rshift = format->Rshift;
    gshift = format->Gshift;
    bshift = format->Bshift;
    rloss = format->Rloss;
    gloss = format->Gloss;
    bloss = format->Bloss;
    bpp1 = surf->format->BytesPerPixel;

    if (surf2) {
        format2 = surf2->format;
        rmask2 = format2->Rmask;
        gmask2 = format2->Gmask;
        bmask2 = format2->Bmask;
        rshift2 = format2->Rshift;
        gshift2 = format2->Gshift;
        bshift2 = format2->Bshift;
        rloss2 = format2->Rloss;
        gloss2 = format2->Gloss;
        bloss2 = format2->Bloss;
        pixels2 = (Uint8 *)surf2->pixels;
        bpp2 = surf->format->BytesPerPixel;
    }
    else { /* make gcc stop complaining */
        rmask2 = gmask2 = bmask2 = 0;
        rshift2 = gshift2 = bshift2 = 0;
        rloss2 = gloss2 = bloss2 = 0;
        format2 = NULL;
        pixels2 = NULL;
        bpp2 = 0;
    }

    SDL_GetRGBA(color, format, &r, &g, &b, &a);
    SDL_GetRGBA(threshold, format, &tr, &tg, &tb, &ta);

    for (y = 0; y < surf->h; y++) {
        pixels = (Uint8 *)surf->pixels + y * surf->pitch;
        if (surf2) {
            pixels2 = (Uint8 *)surf2->pixels + y * surf2->pitch;
        }
        for (x = 0; x < surf->w; x++) {
            /* the_color = surf->get_at(x,y) */
            switch (bpp1) {
                case 1:
                    the_color = (Uint32) * ((Uint8 *)pixels);
                    pixels++;
                    break;
                case 2:
                    the_color = (Uint32) * ((Uint16 *)pixels);
                    pixels += 2;
                    break;
                case 3:
                    pix = ((Uint8 *)pixels);
                    pixels += 3;
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                    the_color = (pix[0]) + (pix[1] << 8) + (pix[2] << 16);
#else
                    the_color = (pix[2]) + (pix[1] << 8) + (pix[0] << 16);
#endif
                    break;
                default: /* case 4: */
                    the_color = *((Uint32 *)pixels);
                    pixels += 4;
                    break;
            }

            if (surf2) {
                switch (bpp2) {
                    case 1:
                        the_color2 = (Uint32) * ((Uint8 *)pixels2);
                        pixels2++;
                        break;
                    case 2:
                        the_color2 = (Uint32) * ((Uint16 *)pixels2);
                        pixels2 += 2;
                        break;
                    case 3:
                        pix = ((Uint8 *)pixels2);
                        pixels2 += 3;
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                        the_color2 = (pix[0]) + (pix[1] << 8) + (pix[2] << 16);
#else
                        the_color2 = (pix[2]) + (pix[1] << 8) + (pix[0] << 16);
#endif
                        break;
                    default: /* case 4: */
                        the_color2 = *((Uint32 *)pixels2);
                        pixels2 += 4;
                        break;
                }
                /* TODO: will need to handle surfaces with palette colors.
                 */
                if ((bpp2 == 1) && (bpp1 == 1) && (!palette_colors)) {
                    /* Don't look at the color of the surface, just use the
                       value. This is useful for 8bit images that aren't
                       actually using the palette.
                    */
                    if (abs_diff_uint32(the_color2, the_color) < tr) {
                        /* this pixel is within the threshold of othersurface.
                         */
                        bitmask_setbit(m, x, y);
                    }
                }
                else if ((abs_diff_uint32(
                              (((the_color2 & rmask2) >> rshift2) << rloss2),
                              (((the_color & rmask) >> rshift) << rloss)) <
                          tr) &&
                         (abs_diff_uint32(
                              (((the_color2 & gmask2) >> gshift2) << gloss2),
                              (((the_color & gmask) >> gshift) << gloss)) <
                          tg) &&
                         (abs_diff_uint32(
                              (((the_color2 & bmask2) >> bshift2) << bloss2),
                              (((the_color & bmask) >> bshift) << bloss)) <
                          tb)) {
                    /* this pixel is within the threshold of othersurface. */
                    bitmask_setbit(m, x, y);
                }

                /* TODO: will need to handle surfaces with palette colors.
                   TODO: will need to handle the case where palette_colors == 0
                */
            }
            else if ((abs_diff_uint32(
                          (((the_color & rmask) >> rshift) << rloss), r) <
                      tr) &&
                     (abs_diff_uint32(
                          (((the_color & gmask) >> gshift) << gloss), g) <
                      tg) &&
                     (abs_diff_uint32(
                          (((the_color & bmask) >> bshift) << bloss), b) <
                      tb)) {
                /* this pixel is within the threshold of the color. */
                bitmask_setbit(m, x, y);
            }
        }
    }
}

static PyObject *
mask_from_threshold(PyObject *self, PyObject *args, PyObject *kwargs)
{
    pgSurfaceObject *surfobj;
    pgSurfaceObject *surfobj2 = NULL;
    pgMaskObject *maskobj = NULL;
    SDL_Surface *surf = NULL, *surf2 = NULL;
    PyObject *rgba_obj_color, *rgba_obj_threshold = NULL;
    Uint8 rgba_color[4];
    Uint8 rgba_threshold[4] = {0, 0, 0, 255};
    Uint32 color;
    Uint32 color_threshold;
    int palette_colors = 1;
    static char *keywords[] = {"surface",      "color",          "threshold",
                               "othersurface", "palette_colors", NULL};

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "O!O|OO!i", keywords, &pgSurface_Type, &surfobj,
            &rgba_obj_color, &rgba_obj_threshold, &pgSurface_Type, &surfobj2,
            &palette_colors))
        return NULL;

    surf = pgSurface_AsSurface(surfobj);
    if (!surf)
        return RAISE(pgExc_SDLError, "display Surface quit");
    if (surfobj2) {
        surf2 = pgSurface_AsSurface(surfobj2);
        if (!surf2)
            return RAISE(pgExc_SDLError, "display Surface quit");
    }

    if (PyLong_Check(rgba_obj_color)) {
        color = (Uint32)PyLong_AsLong(rgba_obj_color);
    }
    else if (PyLong_Check(rgba_obj_color)) {
        color = (Uint32)PyLong_AsUnsignedLong(rgba_obj_color);
    }
    else if (pg_RGBAFromColorObj(rgba_obj_color, rgba_color)) {
        color = SDL_MapRGBA(surf->format, rgba_color[0], rgba_color[1],
                            rgba_color[2], rgba_color[3]);
    }
    else {
        return RAISE(PyExc_TypeError, "invalid color argument");
    }

    if (rgba_obj_threshold) {
        if (PyLong_Check(rgba_obj_threshold))
            color_threshold = (Uint32)PyLong_AsLong(rgba_obj_threshold);
        else if (PyLong_Check(rgba_obj_threshold))
            color_threshold =
                (Uint32)PyLong_AsUnsignedLong(rgba_obj_threshold);
        else if (pg_RGBAFromColorObj(rgba_obj_threshold, rgba_threshold))
            color_threshold =
                SDL_MapRGBA(surf->format, rgba_threshold[0], rgba_threshold[1],
                            rgba_threshold[2], rgba_threshold[3]);
        else
            return RAISE(PyExc_TypeError, "invalid threshold argument");
    }
    else {
        color_threshold =
            SDL_MapRGBA(surf->format, rgba_threshold[0], rgba_threshold[1],
                        rgba_threshold[2], rgba_threshold[3]);
    }

    maskobj = CREATE_MASK_OBJ(surf->w, surf->h, 0);

    if (NULL == maskobj) {
        return NULL; /* Exception already set. */
    }

    pgSurface_Lock(surfobj);
    if (surfobj2) {
        pgSurface_Lock(surfobj2);
    }

    Py_BEGIN_ALLOW_THREADS;
    bitmask_threshold(maskobj->mask, surf, surf2, color, color_threshold,
                      palette_colors);
    Py_END_ALLOW_THREADS;

    pgSurface_Unlock(surfobj);
    if (surfobj2) {
        pgSurface_Unlock(surfobj2);
    }

    return (PyObject *)maskobj;
}

/* The initial labelling phase of the connected components algorithm.
 *
 * Connected component labeling based on the SAUF algorithm by Kesheng Wu,
 * Ekow Otoo, and Kenji Suzuki. The algorithm is best explained by their
 * paper, "Two Strategies to Speed up Connected Component Labeling Algorithms",
 * but in summary, it is a very efficient two pass method for 8-connected
 * components. It uses a decision tree to minimize the number of neighbors that
 * need to be checked. It stores equivalence information in an array based
 * union-find.
 *
 * Params:
 *     input - the input mask
 *     image - an array to store labelled pixels
 *     ufind - the union-find label equivalence array
 *     largest - an array to store the number of pixels for each label
 *
 * Returns:
 *     the highest label in the labelled image
 */
unsigned int
cc_label(bitmask_t *input, unsigned int *image, unsigned int *ufind,
         unsigned int *largest)
{
    unsigned int *buf;
    unsigned int x, y, w, h, root, aroot, croot, temp, label;

    label = 0;
    w = input->w;
    h = input->h;

    ufind[0] = 0;
    buf = image;

    /* special case for first pixel */
    if (bitmask_getbit(input, 0, 0)) { /* process for a new connected comp: */
        label++;                       /* create a new label */
        *buf = label;                  /* give the pixel the label */
        ufind[label] = label; /* put the label in the equivalence array */
        largest[label] = 1;   /* the label has 1 pixel associated with it */
    }
    else {
        *buf = 0;
    }
    buf++;

    /* special case for first row.
           Go over the first row except the first pixel.
    */
    for (x = 1; x < w; x++) {
        if (bitmask_getbit(input, x, 0)) {
            if (*(buf - 1)) { /* d label */
                *buf = *(buf - 1);
            }
            else { /* create label */
                label++;
                *buf = label;
                ufind[label] = label;
                largest[label] = 0;
            }
            largest[*buf]++;
        }
        else {
            *buf = 0;
        }
        buf++;
    }

    /* the rest of the image */
    for (y = 1; y < h; y++) {
        /* first pixel of the row */
        if (bitmask_getbit(input, 0, y)) {
            if (*(buf - w)) { /* b label */
                *buf = *(buf - w);
            }
            else if (*(buf - w + 1)) { /* c label */
                *buf = *(buf - w + 1);
            }
            else { /* create label */
                label++;
                *buf = label;
                ufind[label] = label;
                largest[label] = 0;
            }
            largest[*buf]++;
        }
        else {
            *buf = 0;
        }
        buf++;
        /* middle pixels of the row */
        for (x = 1; x < (w - 1); x++) {
            if (bitmask_getbit(input, x, y)) {
                if (*(buf - w)) { /* b label */
                    *buf = *(buf - w);
                }
                else if (*(buf - w + 1)) { /* c branch of tree */
                    if (*(buf - w - 1)) {  /* union(c, a) */
                        croot = root = *(buf - w + 1);
                        while (ufind[root] < root) { /* find root */
                            root = ufind[root];
                        }
                        if (croot != *(buf - w - 1)) {
                            temp = aroot = *(buf - w - 1);
                            while (ufind[aroot] < aroot) { /* find root */
                                aroot = ufind[aroot];
                            }
                            if (root > aroot) {
                                root = aroot;
                            }
                            while (ufind[temp] > root) { /* set root */
                                aroot = ufind[temp];
                                ufind[temp] = root;
                                temp = aroot;
                            }
                        }
                        while (ufind[croot] > root) { /* set root */
                            temp = ufind[croot];
                            ufind[croot] = root;
                            croot = temp;
                        }
                        *buf = root;
                    }
                    else if (*(buf - 1)) { /* union(c, d) */
                        croot = root = *(buf - w + 1);
                        while (ufind[root] < root) { /* find root */
                            root = ufind[root];
                        }
                        if (croot != *(buf - 1)) {
                            temp = aroot = *(buf - 1);
                            while (ufind[aroot] < aroot) { /* find root */
                                aroot = ufind[aroot];
                            }
                            if (root > aroot) {
                                root = aroot;
                            }
                            while (ufind[temp] > root) { /* set root */
                                aroot = ufind[temp];
                                ufind[temp] = root;
                                temp = aroot;
                            }
                        }
                        while (ufind[croot] > root) { /* set root */
                            temp = ufind[croot];
                            ufind[croot] = root;
                            croot = temp;
                        }
                        *buf = root;
                    }
                    else { /* c label */
                        *buf = *(buf - w + 1);
                    }
                }
                else if (*(buf - w - 1)) { /* a label */
                    *buf = *(buf - w - 1);
                }
                else if (*(buf - 1)) { /* d label */
                    *buf = *(buf - 1);
                }
                else { /* create label */
                    label++;
                    *buf = label;
                    ufind[label] = label;
                    largest[label] = 0;
                }
                largest[*buf]++;
            }
            else {
                *buf = 0;
            }
            buf++;
        }
        /* last pixel of the row, if its not also the first pixel of the row */
        if (w > 1) {
            if (bitmask_getbit(input, x, y)) {
                if (*(buf - w)) { /* b label */
                    *buf = *(buf - w);
                }
                else if (*(buf - w - 1)) { /* a label */
                    *buf = *(buf - w - 1);
                }
                else if (*(buf - 1)) { /* d label */
                    *buf = *(buf - 1);
                }
                else { /* create label */
                    label++;
                    *buf = label;
                    ufind[label] = label;
                    largest[label] = 0;
                }
                largest[*buf]++;
            }
            else {
                *buf = 0;
            }
            buf++;
        }
    }

    return label;
}

/* Creates a bounding rect for each connected component in the given mask.
 *
 * Allocates memory for rects.
 *
 * NOTE: Caller is responsible for freeing the "ret_rects" memory.
 *
 * Params:
 *     input - mask to search in for the connected components to bound
 *     num_bounding_boxes - passes back the number of bounding rects found
 *     ret_rects - passes back the bounding rects that are found with the first
 *         rect at index 1, memory is allocated
 *
 * Returns:
 *     0 on success
 *     -2 on memory allocation error
 */
static int
get_bounding_rects(bitmask_t *input, int *num_bounding_boxes,
                   SDL_Rect **ret_rects)
{
    unsigned int *image, *ufind, *largest, *buf;
    unsigned int x_uf, label = 0;
    int x, y, w, h, temp, relabel;
    SDL_Rect *rects;

    rects = NULL;

    w = input->w;
    h = input->h;

    if (!w || !h) {
        *ret_rects = rects;
        return 0;
    }
    /* a temporary image to assign labels to each bit of the mask */
    image = (unsigned int *)malloc(sizeof(unsigned int) * w * h);
    if (!image) {
        return -2;
    }

    /* allocate enough space for the maximum possible connected components */
    /* the union-find array. see wikipedia for info on union find */
    ufind = (unsigned int *)malloc(sizeof(unsigned int) * (w / 2 + 1) *
                                   (h / 2 + 1));
    if (!ufind) {
        free(image);
        return -2;
    }

    largest = (unsigned int *)malloc(sizeof(unsigned int) * (w / 2 + 1) *
                                     (h / 2 + 1));
    if (!largest) {
        free(image);
        free(ufind);
        return -2;
    }

    /* do the initial labelling */
    label = cc_label(input, image, ufind, largest);

    relabel = 0;
    /* flatten and relabel the union-find equivalence array.  Start at label 1
       because label 0 indicates an unset pixel.  For this reason, we also use
       <= label rather than < label. */
    for (x_uf = 1; x_uf <= label; ++x_uf) {
        if (ufind[x_uf] < x_uf) {             /* is it a union find root? */
            ufind[x_uf] = ufind[ufind[x_uf]]; /* relabel it to its root */
        }
        else { /* its a root */
            relabel++;
            ufind[x_uf] = relabel; /* assign the lowest label available */
        }
    }

    *num_bounding_boxes = relabel;

    if (relabel == 0) {
        /* early out, as we didn't find anything. */
        free(image);
        free(ufind);
        free(largest);
        *ret_rects = rects;
        return 0;
    }

    /* the bounding rects, need enough space for the number of labels */
    rects = (SDL_Rect *)malloc(sizeof(SDL_Rect) * (relabel + 1));
    if (!rects) {
        free(image);
        free(ufind);
        free(largest);
        return -2;
    }

    for (temp = 0; temp <= relabel; temp++) {
        rects[temp].h = 0; /* so we know if its a new rect or not */
    }

    /* find the bounding rect of each connected component */
    buf = image;
    for (y = 0; y < h; y++) {
        for (x = 0; x < w; x++) {
            if (ufind[*buf]) { /* if the pixel is part of a component */
                if (rects[ufind[*buf]].h) { /* the component has a rect */
                    temp = rects[ufind[*buf]].x;
                    rects[ufind[*buf]].x = MIN(x, temp);
                    rects[ufind[*buf]].y = MIN(y, rects[ufind[*buf]].y);
                    rects[ufind[*buf]].w =
                        MAX(rects[ufind[*buf]].w + temp, x + 1) -
                        rects[ufind[*buf]].x;
                    rects[ufind[*buf]].h = MAX(rects[ufind[*buf]].h,
                                               y - rects[ufind[*buf]].y + 1);
                }
                else { /* otherwise, start the rect */
                    rects[ufind[*buf]].x = x;
                    rects[ufind[*buf]].y = y;
                    rects[ufind[*buf]].w = 1;
                    rects[ufind[*buf]].h = 1;
                }
            }
            buf++;
        }
    }

    free(image);
    free(ufind);
    free(largest);
    *ret_rects = rects;

    return 0;
}

static PyObject *
mask_get_bounding_rects(PyObject *self, PyObject *_null)
{
    SDL_Rect *regions;
    SDL_Rect *aregion;
    int num_bounding_boxes, i, r;
    PyObject *rect_list;
    PyObject *rect;

    bitmask_t *mask = pgMask_AsBitmap(self);

    rect_list = NULL;
    regions = NULL;
    aregion = NULL;

    num_bounding_boxes = 0;

    Py_BEGIN_ALLOW_THREADS;

    r = get_bounding_rects(mask, &num_bounding_boxes, &regions);

    Py_END_ALLOW_THREADS;

    if (r == -2) {
        /* memory out failure */
        return RAISE(PyExc_MemoryError,
                     "Not enough memory to get bounding rects. \n");
    }

    rect_list = PyList_New(0);

    if (!rect_list) {
        free(regions);

        return NULL; /* Exception already set. */
    }

    /* build a list of rects to return.  Starts at 1 because we never use 0. */
    for (i = 1; i <= num_bounding_boxes; i++) {
        aregion = regions + i;
        rect = pgRect_New4(aregion->x, aregion->y, aregion->w, aregion->h);

        if (NULL == rect) {
            Py_DECREF(rect_list);
            free(regions);

            return RAISE(PyExc_MemoryError,
                         "cannot allocate memory for bounding rects");
        }

        if (0 != PyList_Append(rect_list, rect)) {
            Py_DECREF(rect);
            Py_DECREF(rect_list);
            free(regions);

            return NULL; /* Exception already set. */
        }

        Py_DECREF(rect);
    }

    free(regions);

    return rect_list;
}

/* Finds all the connected components in a given mask.
 *
 * Allocates memory for components.
 *
 * NOTE: Caller is responsible for freeing the "components" memory.
 *
 * Params:
 *     mask - mask to search in for the connected components
 *     components - passes back an array of connected component masks with the
 *         first component at index 1, memory is allocated
 *     min - minimum number of pixels for a component to be considered,
 *         defaults to 0 for negative values
 *
 * Returns:
 *     the number of connected components (>= 0)
 *     -2 on memory allocation error
 */
static int
get_connected_components(bitmask_t *mask, bitmask_t ***components, int min)
{
    unsigned int *image, *ufind, *largest, *buf;
    unsigned int x_uf, min_cc, label = 0;
    int x, y, w, h, relabel;
    bitmask_t **comps;

    w = mask->w;
    h = mask->h;

    if (!w || !h) {
        return 0;
    }

    /* a temporary image to assign labels to each bit of the mask */
    image = (unsigned int *)malloc(sizeof(unsigned int) * w * h);
    if (!image) {
        return -2;
    }

    /* allocate enough space for the maximum possible connected components */
    /* the union-find array. see wikipedia for info on union find */
    ufind = (unsigned int *)malloc(sizeof(unsigned int) * (w / 2 + 1) *
                                   (h / 2 + 1));
    if (!ufind) {
        free(image);
        return -2;
    }

    largest = (unsigned int *)malloc(sizeof(unsigned int) * (w / 2 + 1) *
                                     (h / 2 + 1));
    if (!largest) {
        free(image);
        free(ufind);
        return -2;
    }

    /* do the initial labelling */
    label = cc_label(mask, image, ufind, largest);

    for (x_uf = 1; x_uf <= label; ++x_uf) {
        if (ufind[x_uf] < x_uf) {
            largest[ufind[x_uf]] += largest[x_uf];
        }
    }

    relabel = 0;
    min_cc = (0 < min) ? (unsigned int)min : 0;

    /* flatten and relabel the union-find equivalence array.  Start at label 1
       because label 0 indicates an unset pixel.  For this reason, we also use
       <= label rather than < label. */
    for (x_uf = 1; x_uf <= label; ++x_uf) {
        if (ufind[x_uf] < x_uf) {             /* is it a union find root? */
            ufind[x_uf] = ufind[ufind[x_uf]]; /* relabel it to its root */
        }
        else { /* its a root */
            if (largest[x_uf] >= min_cc) {
                relabel++;
                ufind[x_uf] = relabel; /* assign the lowest label available */
            }
            else {
                ufind[x_uf] = 0;
            }
        }
    }

    if (relabel == 0) {
        /* early out, as we didn't find anything. */
        free(image);
        free(ufind);
        free(largest);
        return 0;
    }

    /* allocate space for the mask array */
    comps = (bitmask_t **)malloc(sizeof(bitmask_t *) * (relabel + 1));
    if (!comps) {
        free(image);
        free(ufind);
        free(largest);
        return -2;
    }

    /* create the empty masks */
    for (x = 1; x <= relabel; x++) {
        comps[x] = bitmask_create(w, h);
    }

    /* set the bits in each mask */
    buf = image;
    for (y = 0; y < h; y++) {
        for (x = 0; x < w; x++) {
            if (ufind[*buf]) { /* if the pixel is part of a component */
                bitmask_setbit(comps[ufind[*buf]], x, y);
            }
            buf++;
        }
    }

    free(image);
    free(ufind);
    free(largest);

    *components = comps;

    return relabel;
}

static PyObject *
mask_connected_components(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *mask_list = NULL;
    pgMaskObject *maskobj = NULL;
    bitmask_t **components = NULL;
    bitmask_t *mask = pgMask_AsBitmap(self);
    int i, m, num_components, min = 0; /* Default min value. */
    static char *keywords[] = {"minimum", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|i", keywords, &min)) {
        return NULL; /* Exception already set. */
    }

    Py_BEGIN_ALLOW_THREADS;
    num_components = get_connected_components(mask, &components, min);
    Py_END_ALLOW_THREADS;

    if (num_components == -2) {
        return RAISE(PyExc_MemoryError,
                     "cannot allocate memory for connected components");
    }

    mask_list = PyList_New(0);
    if (!mask_list) {
        /* Components were allocated starting at index 1. */
        for (i = 1; i <= num_components; ++i) {
            bitmask_free(components[i]);
        }

        free(components);
        return NULL; /* Exception already set. */
    }

    /* Components were allocated starting at index 1. */
    for (i = 1; i <= num_components; ++i) {
        maskobj = create_mask_using_bitmask(components[i]);

        if (NULL == maskobj) {
            /* Starting freeing with the current index. */
            for (m = i; m <= num_components; ++m) {
                bitmask_free(components[m]);
            }

            free(components);
            Py_DECREF(mask_list);
            return NULL; /* Exception already set. */
        }

        if (0 != PyList_Append(mask_list, (PyObject *)maskobj)) {
            /* Can't append to the list. Starting freeing with the next index
             * as maskobj contains the component from the current index. */
            for (m = i + 1; m <= num_components; ++m) {
                bitmask_free(components[m]);
            }

            free(components);
            Py_DECREF((PyObject *)maskobj);
            Py_DECREF(mask_list);
            return NULL; /* Exception already set. */
        }

        Py_DECREF((PyObject *)maskobj);
    }

    free(components);
    return mask_list;
}

/* Finds the largest connected component in a given mask.
 *
 * Tracks the number of pixels in each label, finding the biggest one while
 * flattening the union-find equivalence array. It then writes an output mask
 * containing only the largest connected component.
 *
 * Params:
 *     input - mask to search in for the largest connected component
 *     output - this mask is updated with the largest connected component
 *     ccx - x index, if < 0 then the largest connected component in the input
 *         mask is found and copied to the output mask, otherwise the connected
 *         component at (ccx, ccy) is copied to the output mask
 *     ccy - y index
 *
 * Returns:
 *     0 on success
 *     -2 on memory allocation error
 */
static int
largest_connected_comp(bitmask_t *input, bitmask_t *output, int ccx, int ccy)
{
    unsigned int *image, *ufind, *largest, *buf;
    unsigned int max, x, y, w, h, label;

    w = input->w;
    h = input->h;

    if (!w || !h) {
        return 0;
    }

    /* a temporary image to assign labels to each bit of the mask */
    image = (unsigned int *)malloc(sizeof(unsigned int) * w * h);
    if (!image) {
        return -2;
    }
    /* allocate enough space for the maximum possible connected components */
    /* the union-find array. see wikipedia for info on union find */
    ufind = (unsigned int *)malloc(sizeof(unsigned int) * (w / 2 + 1) *
                                   (h / 2 + 1));
    if (!ufind) {
        free(image);
        return -2;
    }
    /* an array to track the number of pixels associated with each label */
    largest = (unsigned int *)malloc(sizeof(unsigned int) * (w / 2 + 1) *
                                     (h / 2 + 1));
    if (!largest) {
        free(image);
        free(ufind);
        return -2;
    }

    /* do the initial labelling */
    label = cc_label(input, image, ufind, largest);

    max = 1;
    /* flatten the union-find equivalence array */
    for (x = 2; x <= label; x++) {
        if (ufind[x] != x) {                 /* is it a union find root? */
            largest[ufind[x]] += largest[x]; /* add its pixels to its root */
            ufind[x] = ufind[ufind[x]];      /* relabel it to its root */
        }
        if (largest[ufind[x]] > largest[max]) { /* is it the new biggest? */
            max = ufind[x];
        }
    }

    /* write out the final image */
    buf = image;
    if (ccx >= 0)
        max = ufind[*(buf + ccy * w + ccx)];
    for (y = 0; y < h; y++) {
        for (x = 0; x < w; x++) {
            if (ufind[*buf] == max) {         /* if the label is the max one */
                bitmask_setbit(output, x, y); /* set the bit in the mask */
            }
            buf++;
        }
    }

    free(image);
    free(ufind);
    free(largest);

    return 0;
}

static PyObject *
mask_connected_component(PyObject *self, PyObject *args, PyObject *kwargs)
{
    bitmask_t *input = pgMask_AsBitmap(self);
    pgMaskObject *output_maskobj = NULL;
    int x = -1, y = -1;
    Py_ssize_t args_exist = PyTuple_Size(args);
    PyObject *pos = NULL;
    static char *keywords[] = {"pos", NULL};

    if (kwargs)
        args_exist += PyDict_Size(kwargs);

    if (args_exist) {
        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O", keywords, &pos)) {
            return NULL; /* Exception already set. */
        }

        if (!pg_TwoIntsFromObj(pos, &x, &y)) {
            return RAISE(PyExc_TypeError, "pos must be two numbers");
        }

        if (x < 0 || x >= input->w || y < 0 || y >= input->h) {
            return PyErr_Format(PyExc_IndexError, "%d, %d is out of bounds", x,
                                y);
        }
    }

    output_maskobj = CREATE_MASK_OBJ(input->w, input->h, 0);

    if (NULL == output_maskobj) {
        return NULL; /* Exception already set. */
    }

    /* If a pixel index is provided and the indexed bit is not set, then the
     * returned mask is empty.
     */
    if (!args_exist || bitmask_getbit(input, x, y)) {
        if (largest_connected_comp(input, output_maskobj->mask, x, y) == -2) {
            Py_DECREF(output_maskobj);
            return RAISE(PyExc_MemoryError,
                         "cannot allocate memory for connected component");
        }
    }

    return (PyObject *)output_maskobj;
}

/* Extract the color data from a color object.
 *
 * Params:
 *     surf: surface that color will be mapped from
 *     color_obj: color object to extract color data from
 *     rbga_color: rbga array, contains default color if color_obj is NULL
 *     color: color value extracted from the color_obj (or from the default
 *         value of rbga_color)
 *
 * Returns:
 *     int: 1, means the color data extraction was successful and the color
 *             parameter contains a valid color value
 *          0, means the color data extraction has failed and an exception has
 *             been set
 */
static int
extract_color(SDL_Surface *surf, PyObject *color_obj, Uint8 rgba_color[],
              Uint32 *color)
{
    if (NULL == color_obj) {
        *color = SDL_MapRGBA(surf->format, rgba_color[0], rgba_color[1],
                             rgba_color[2], rgba_color[3]);
        return 1;
    }

    if (PyLong_Check(color_obj)) {
        long intval = PyLong_AsLong(color_obj);

        if ((-1 == intval && PyErr_Occurred()) || intval > (long)0xFFFFFFFF) {
            PyErr_SetString(PyExc_ValueError, "invalid color argument");
            return 0;
        }

        *color = (Uint32)intval;
        return 1;
    }

    if (PyLong_Check(color_obj)) {
        unsigned long longval = PyLong_AsUnsignedLong(color_obj);

        if (PyErr_Occurred() || longval > 0xFFFFFFFF) {
            PyErr_SetString(PyExc_ValueError, "invalid color argument");
            return 0;
        }

        *color = (Uint32)longval;
        return 1;
    }

    if (pg_RGBAFromFuzzyColorObj(color_obj, rgba_color)) {
        *color = SDL_MapRGBA(surf->format, rgba_color[0], rgba_color[1],
                             rgba_color[2], rgba_color[3]);
        return 1;
    }

    return 0; /* Exception already set. */
}

/* Draws a mask on a surface.
 *
 * Params:
 *     surf: surface to draw on
 *     bitmask: bitmask to draw
 *     x_dest: x position on surface of where to start drawing
 *     y_dest: y position on surface of where to start drawing
 *     draw_setbits: if non-zero then draw the set bits (bits==1)
 *     draw_unsetbits: if non-zero then draw the unset bits (bits==0)
 *     setsurf: use colors from this surface for set bits (bits==1)
 *     unsetsurf: use colors from this surface for unset bits (bits==0)
 *     setcolor: color for set bits, setsurf takes precedence (bits==1)
 *     unsetcolor: color for unset bits, unsetsurf takes precedence (bits==0)
 *
 * Assumptions:
 *     - surf and bitmask are non-NULL
 *     - all surfaces have the same pixel format
 *     - all surfaces that are non-NULL are locked
 */
static void
draw_to_surface(SDL_Surface *surf, bitmask_t *bitmask, int x_dest, int y_dest,
                int draw_setbits, int draw_unsetbits, SDL_Surface *setsurf,
                SDL_Surface *unsetsurf, Uint32 *setcolor, Uint32 *unsetcolor)
{
    Uint8 *pixel = NULL;
    Uint8 bpp;
    int x, y, x_end, y_end, x_start, y_start; /* surf indexing */
    int xm, ym, xm_start, ym_start; /* bitmask/setsurf/unsetsurf indexing */

    /* There is nothing to do when any of these conditions exist:
     * - surface has a width or height of <= 0
     * - mask has a width or height of <= 0
     * - draw_setbits and draw_unsetbits are both 0 */
    if ((surf->h <= 0) || (surf->w <= 0) || (bitmask->h <= 0) ||
        (bitmask->w <= 0) || (!draw_setbits && !draw_unsetbits)) {
        return;
    }

    /* There is also nothing to do when the destination position is such that
     * nothing will be drawn on the surface. */
    if ((x_dest >= surf->w) || (y_dest >= surf->h) || (-x_dest > bitmask->w) ||
        (-y_dest > bitmask->h)) {
        return;
    }

    bpp = surf->format->BytesPerPixel;

    xm_start = (x_dest < 0) ? -x_dest : 0;
    x_start = (x_dest > 0) ? x_dest : 0;
    x_end = MIN(surf->w, bitmask->w + x_dest);

    ym_start = (y_dest < 0) ? -y_dest : 0;
    y_start = (y_dest > 0) ? y_dest : 0;
    y_end = MIN(surf->h, bitmask->h + y_dest);

    if (NULL == setsurf && NULL == unsetsurf) {
        /* Draw just using color values. No surfaces. */
        draw_setbits = draw_setbits && NULL != setcolor;
        draw_unsetbits = draw_unsetbits && NULL != unsetcolor;

        for (y = y_start, ym = ym_start; y < y_end; ++y, ++ym) {
            pixel = (Uint8 *)surf->pixels + y * surf->pitch + x_start * bpp;

            for (x = x_start, xm = xm_start; x < x_end;
                 ++x, ++xm, pixel += bpp) {
                if (bitmask_getbit(bitmask, xm, ym)) {
                    if (draw_setbits) {
                        set_pixel_color(pixel, bpp, *setcolor);
                    }
                }
                else if (draw_unsetbits) {
                    set_pixel_color(pixel, bpp, *unsetcolor);
                }
            }
        }
    }
    else if (NULL == setcolor && NULL == unsetcolor && NULL != setsurf &&
             NULL != unsetsurf && setsurf->h + y_dest >= y_end &&
             setsurf->w + x_dest >= x_end && unsetsurf->h + y_dest >= y_end &&
             unsetsurf->w + x_dest >= x_end) {
        /* Draw using surfaces that are as big (or bigger) as what is being
         * drawn and no color values are being used. */
        Uint8 *setpixel = NULL, *unsetpixel = NULL;

        for (y = y_start, ym = ym_start; y < y_end; ++y, ++ym) {
            pixel = (Uint8 *)surf->pixels + y * surf->pitch + x_start * bpp;
            setpixel = (Uint8 *)setsurf->pixels + ym * setsurf->pitch +
                       xm_start * bpp;
            unsetpixel = (Uint8 *)unsetsurf->pixels + ym * unsetsurf->pitch +
                         xm_start * bpp;

            for (x = x_start, xm = xm_start; x < x_end;
                 ++x, ++xm, pixel += bpp, setpixel += bpp, unsetpixel += bpp) {
                if (bitmask_getbit(bitmask, xm, ym)) {
                    if (draw_setbits) {
                        set_pixel_color(pixel, bpp,
                                        get_pixel_color(setpixel, bpp));
                    }
                }
                else if (draw_unsetbits) {
                    set_pixel_color(pixel, bpp,
                                    get_pixel_color(unsetpixel, bpp));
                }
            }
        }
    }
    else {
        /* Draw using surfaces and color values. */
        Uint8 *setpixel = NULL, *unsetpixel = NULL;
        int use_setsurf, use_unsetsurf;

        /* Looping over each bit in the mask and deciding whether to use a
         * color from setsurf/unsetsurf or from setcolor/unsetcolor. */
        for (y = y_start, ym = ym_start; y < y_end; ++y, ++ym) {
            pixel = (Uint8 *)surf->pixels + y * surf->pitch + x_start * bpp;
            use_setsurf = draw_setbits && NULL != setsurf && setsurf->h > ym;
            use_unsetsurf =
                draw_unsetbits && NULL != unsetsurf && unsetsurf->h > ym;

            if (use_setsurf) {
                setpixel = (Uint8 *)setsurf->pixels + ym * setsurf->pitch +
                           xm_start * bpp;
            }

            if (use_unsetsurf) {
                unsetpixel = (Uint8 *)unsetsurf->pixels +
                             ym * unsetsurf->pitch + xm_start * bpp;
            }

            for (x = x_start, xm = xm_start; x < x_end;
                 ++x, ++xm, pixel += bpp) {
                if (bitmask_getbit(bitmask, xm, ym)) {
                    if (draw_setbits) {
                        if (use_setsurf && setsurf->w > xm) {
                            set_pixel_color(pixel, bpp,
                                            get_pixel_color(setpixel, bpp));
                        }
                        else if (NULL != setcolor) {
                            set_pixel_color(pixel, bpp, *setcolor);
                        }
                    }
                }
                else if (draw_unsetbits) {
                    if (use_unsetsurf && unsetsurf->w > xm) {
                        set_pixel_color(pixel, bpp,
                                        get_pixel_color(unsetpixel, bpp));
                    }
                    else if (NULL != unsetcolor) {
                        set_pixel_color(pixel, bpp, *unsetcolor);
                    }
                }

                if (use_setsurf) {
                    setpixel += bpp;
                }

                if (use_unsetsurf) {
                    unsetpixel += bpp;
                }
            }
        }
    }
}

/* Checks if the surfaces have the same pixel formats.
 *
 * Params:
 *     surf: surface to check against
 *     check_surf: surface to check
 *
 * Returns:
 *     int: 0 to indicate surfaces don't have the same format
 *          1 to indicate the surfaces have the same format
 *
 * Assumptions:
 *     - both parameters are non-NULL
 *     - these checks are enough to assume the pixel formats are the same
 */
static int
check_surface_pixel_format(SDL_Surface *surf, SDL_Surface *check_surf)
{
    if ((surf->format->BytesPerPixel != check_surf->format->BytesPerPixel) ||
        (surf->format->BitsPerPixel != check_surf->format->BitsPerPixel) ||
        (surf->format->format != check_surf->format->format)) {
        return 0;
    }

    return 1;
}

/* Draws a mask on a surface.
 *
 * Returns:
 *     Surface object or NULL to indicate a fail.
 */
static PyObject *
mask_to_surface(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *surfobj = Py_None, *setcolorobj = NULL, *unsetcolorobj = NULL;
    PyObject *setsurfobj = Py_None, *unsetsurfobj = Py_None;
    PyObject *destobj = NULL;
    SDL_Surface *surf = NULL, *setsurf = NULL, *unsetsurf = NULL;
    bitmask_t *bitmask = pgMask_AsBitmap(self);
    Uint32 *setcolor_ptr = NULL, *unsetcolor_ptr = NULL;
    Uint32 setcolor, unsetcolor;
    int draw_setbits = 0, draw_unsetbits = 0;
    int created_surfobj = 0; /* Set to 1 if this func creates the surfobj. */
    int x_dest = 0, y_dest = 0; /* Default destination coordinates. */
    Uint8 dflt_setcolor[] = {255, 255, 255, 255}; /* Default set color. */
    Uint8 dflt_unsetcolor[] = {0, 0, 0, 255};     /* Default unset color. */

    static char *keywords[] = {"surface",  "setsurface", "unsetsurface",
                               "setcolor", "unsetcolor", "dest",
                               NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OOOOOO", keywords,
                                     &surfobj, &setsurfobj, &unsetsurfobj,
                                     &setcolorobj, &unsetcolorobj, &destobj)) {
        return NULL; /* Exception already set. */
    }

    if (Py_None == surfobj) {
        surfobj =
            PyObject_CallFunction((PyObject *)&pgSurface_Type, "(ii)ii",
                                  bitmask->w, bitmask->h, PGS_SRCALPHA, 32);

        if (NULL == surfobj) {
            if (!PyErr_Occurred()) {
                return RAISE(PyExc_RuntimeError, "unable to create surface");
            }
            return NULL;
        }

        created_surfobj = 1;
    }
    else if (!pgSurface_Check(surfobj)) {
        return RAISE(PyExc_TypeError, "invalid surface argument");
    }

    surf = pgSurface_AsSurface(surfobj);

    if (!surf)
        return RAISE(pgExc_SDLError, "display Surface quit");

    if (Py_None != setsurfobj) {
        if (!pgSurface_Check(setsurfobj)) {
            PyErr_SetString(PyExc_TypeError, "invalid setsurface argument");
            goto to_surface_error;
        }

        setsurf = pgSurface_AsSurface(setsurfobj);

        if (!setsurf)
            return RAISE(pgExc_SDLError, "display Surface quit");

        if (0 == check_surface_pixel_format(surf, setsurf)) {
            /* Needs to have the same format settings as surface. */
            PyErr_SetString(PyExc_ValueError,
                            "setsurface needs to have same "
                            "bytesize/bitsize/alpha format as surface");
            goto to_surface_error;
        }
        else if ((setsurf->h <= 0) || (setsurf->w <= 0)) {
            /* Surface has no usable color positions, so ignore it. */
            setsurf = NULL;
        }
        else {
            draw_setbits = 1;
        }
    }

    if (Py_None != unsetsurfobj) {
        if (!pgSurface_Check(unsetsurfobj)) {
            PyErr_SetString(PyExc_TypeError, "invalid unsetsurface argument");
            goto to_surface_error;
        }

        unsetsurf = pgSurface_AsSurface(unsetsurfobj);

        if (!unsetsurf)
            return RAISE(pgExc_SDLError, "display Surface quit");

        if (0 == check_surface_pixel_format(surf, unsetsurf)) {
            /* Needs to have the same format settings as surface. */
            PyErr_SetString(PyExc_ValueError,
                            "unsetsurface needs to have same "
                            "bytesize/bitsize/alpha format as surface");
            goto to_surface_error;
        }
        else if ((unsetsurf->h <= 0) || (unsetsurf->w <= 0)) {
            /* Surface has no usable color positions, so ignore it. */
            unsetsurf = NULL;
        }
        else {
            draw_unsetbits = 1;
        }
    }

    if (Py_None != setcolorobj) {
        if (!extract_color(surf, setcolorobj, dflt_setcolor, &setcolor)) {
            goto to_surface_error; /* Exception already set. */
        }

        setcolor_ptr = &setcolor;
        draw_setbits = 1;
    }

    if (Py_None != unsetcolorobj) {
        if (!extract_color(surf, unsetcolorobj, dflt_unsetcolor,
                           &unsetcolor)) {
            goto to_surface_error; /* Exception already set. */
        }

        unsetcolor_ptr = &unsetcolor;
        draw_unsetbits = 1;
    }

    if (NULL != destobj) {
        int tempx = 0, tempy = 0;

        /* Destination coordinates can be extracted from:
         * - lists/tuples with 2 items
         * - Rect (or Rect like) objects (uses x, y values) */
        if (pg_TwoIntsFromObj(destobj, &tempx, &tempy)) {
            x_dest = tempx;
            y_dest = tempy;
        }
        else {
            SDL_Rect temp_rect;
            SDL_Rect *dest_rect = pgRect_FromObject(destobj, &temp_rect);

            if (NULL != dest_rect) {
                x_dest = dest_rect->x;
                y_dest = dest_rect->y;
            }
            else {
                PyErr_SetString(PyExc_TypeError, "invalid dest argument");
                goto to_surface_error;
            }
        }
    }

    if (!pgSurface_Lock((pgSurfaceObject *)surfobj)) {
        PyErr_SetString(PyExc_RuntimeError, "cannot lock surface");
        goto to_surface_error;
    }

    /* Only lock the setsurface if it is being used.
     * i.e. setsurf is non-NULL */
    if (NULL != setsurf && !pgSurface_Lock((pgSurfaceObject *)setsurfobj)) {
        PyErr_SetString(PyExc_RuntimeError, "cannot lock setsurface");
        goto to_surface_error;
    }

    /* Only lock the unsetsurface if it is being used.
     * i.e.. unsetsurf is non-NULL. */
    if (NULL != unsetsurf &&
        !pgSurface_Lock((pgSurfaceObject *)unsetsurfobj)) {
        PyErr_SetString(PyExc_RuntimeError, "cannot lock unsetsurface");
        goto to_surface_error;
    }

    Py_BEGIN_ALLOW_THREADS; /* Release the GIL. */

    draw_to_surface(surf, bitmask, x_dest, y_dest, draw_setbits,
                    draw_unsetbits, setsurf, unsetsurf, setcolor_ptr,
                    unsetcolor_ptr);

    Py_END_ALLOW_THREADS; /* Obtain the GIL. */

    if (NULL != unsetsurf &&
        !pgSurface_Unlock((pgSurfaceObject *)unsetsurfobj)) {
        PyErr_SetString(PyExc_RuntimeError, "cannot unlock unsetsurface");
        goto to_surface_error;
    }

    if (NULL != setsurf && !pgSurface_Unlock((pgSurfaceObject *)setsurfobj)) {
        PyErr_SetString(PyExc_RuntimeError, "cannot unlock setsurface");
        goto to_surface_error;
    }

    if (!pgSurface_Unlock((pgSurfaceObject *)surfobj)) {
        PyErr_SetString(PyExc_RuntimeError, "cannot unlock surface");
        goto to_surface_error;
    }

    if (!created_surfobj) {
        /* Only increase ref count if this func didn't create the surfobj. */
        Py_INCREF(surfobj);
    }

    return surfobj;

/* Handles the cleanup for fail cases. */
to_surface_error:
    if (created_surfobj) {
        /* Only decrease ref count if this func created the surfobj. */
        Py_DECREF(surfobj);
    }

    return NULL;
}

static PyMethodDef mask_methods[] = {
    {"__copy__", mask_copy, METH_NOARGS, DOC_MASKCOPY},
    {"copy", mask_call_copy, METH_NOARGS, DOC_MASKCOPY},
    {"get_size", mask_get_size, METH_NOARGS, DOC_MASKGETSIZE},
    {"get_rect", (PyCFunction)mask_get_rect, METH_VARARGS | METH_KEYWORDS,
     DOC_MASKGETRECT},
    {"get_at", (PyCFunction)mask_get_at, METH_VARARGS | METH_KEYWORDS,
     DOC_MASKGETAT},
    {"set_at", (PyCFunction)mask_set_at, METH_VARARGS | METH_KEYWORDS,
     DOC_MASKSETAT},
    {"overlap", (PyCFunction)mask_overlap, METH_VARARGS | METH_KEYWORDS,
     DOC_MASKOVERLAP},
    {"overlap_area", (PyCFunction)mask_overlap_area,
     METH_VARARGS | METH_KEYWORDS, DOC_MASKOVERLAPAREA},
    {"overlap_mask", (PyCFunction)mask_overlap_mask,
     METH_VARARGS | METH_KEYWORDS, DOC_MASKOVERLAPMASK},
    {"fill", mask_fill, METH_NOARGS, DOC_MASKFILL},
    {"clear", mask_clear, METH_NOARGS, DOC_MASKCLEAR},
    {"invert", mask_invert, METH_NOARGS, DOC_MASKINVERT},
    {"scale", (PyCFunction)mask_scale, METH_VARARGS | METH_KEYWORDS,
     DOC_MASKSCALE},
    {"draw", (PyCFunction)mask_draw, METH_VARARGS | METH_KEYWORDS,
     DOC_MASKDRAW},
    {"erase", (PyCFunction)mask_erase, METH_VARARGS | METH_KEYWORDS,
     DOC_MASKERASE},
    {"count", mask_count, METH_NOARGS, DOC_MASKCOUNT},
    {"centroid", mask_centroid, METH_NOARGS, DOC_MASKCENTROID},
    {"angle", mask_angle, METH_NOARGS, DOC_MASKANGLE},
    {"outline", (PyCFunction)mask_outline, METH_VARARGS | METH_KEYWORDS,
     DOC_MASKOUTLINE},
    {"convolve", (PyCFunction)mask_convolve, METH_VARARGS | METH_KEYWORDS,
     DOC_MASKCONVOLVE},
    {"connected_component", (PyCFunction)mask_connected_component,
     METH_VARARGS | METH_KEYWORDS, DOC_MASKCONNECTEDCOMPONENT},
    {"connected_components", (PyCFunction)mask_connected_components,
     METH_VARARGS | METH_KEYWORDS, DOC_MASKCONNECTEDCOMPONENTS},
    {"get_bounding_rects", mask_get_bounding_rects, METH_NOARGS,
     DOC_MASKGETBOUNDINGRECTS},
    {"to_surface", (PyCFunction)mask_to_surface, METH_VARARGS | METH_KEYWORDS,
     DOC_MASKTOSURFACE},

    {NULL, NULL, 0, NULL}};

/*mask object internals*/

/* This is a helper function for internal use only.
 * Creates a mask object using an existing bitmask.
 *
 * Params:
 *     bitmask: pointer to the bitmask to use
 *
 * Returns:
 *     Mask object or NULL to indicate a fail
 */
static PG_INLINE pgMaskObject *
create_mask_using_bitmask(bitmask_t *bitmask)
{
    return create_mask_using_bitmask_and_type(bitmask, &pgMask_Type);
}

/* This is a helper function for internal use only.
 * Creates a mask object using an existing bitmask and a type.
 *
 * Params:
 *     bitmask: pointer to the bitmask to use
 *     ob_type: pointer to the mask object type to create
 *
 * Returns:
 *     Mask object or NULL to indicate a fail
 */
static PG_INLINE pgMaskObject *
create_mask_using_bitmask_and_type(bitmask_t *bitmask, PyTypeObject *ob_type)
{
    /* tp_init is not needed as the bitmask has already been created. */
    pgMaskObject *maskobj =
        (pgMaskObject *)pgMask_Type.tp_new(ob_type, NULL, NULL);

    if (NULL == maskobj) {
        return (pgMaskObject *)RAISE(PyExc_MemoryError,
                                     "cannot allocate memory for mask");
    }

    maskobj->mask = bitmask;
    return maskobj;
}

static void
mask_dealloc(PyObject *self)
{
    bitmask_t *bitmask = pgMask_AsBitmap(self);

    if (NULL != bitmask) {
        /* Free up the bitmask. */
        bitmask_free(bitmask);
    }

    /* Free up the mask. */
    Py_TYPE(self)->tp_free(self);
}

static PyObject *
mask_repr(PyObject *self)
{
    bitmask_t *mask = pgMask_AsBitmap(self);
    return PyUnicode_FromFormat("<Mask(%dx%d)>", mask->w, mask->h);
}

static PyObject *
mask_new(PyTypeObject *subtype, PyObject *args, PyObject *kwargs)
{
    pgMaskObject *maskobj = (pgMaskObject *)subtype->tp_alloc(subtype, 0);

    if (NULL == maskobj) {
        return RAISE(PyExc_MemoryError, "cannot allocate memory for mask");
    }

    maskobj->mask = NULL;
    return (PyObject *)maskobj;
}

static int
mask_init(PyObject *self, PyObject *args, PyObject *kwargs)
{
    bitmask_t *bitmask = NULL;
    PyObject *size = NULL;
    int w, h;
    int fill = 0; /* Default is false. */
    char *keywords[] = {"size", "fill", NULL};
    const char *format = "O|p";

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, format, keywords, &size,
                                     &fill)) {
        return -1;
    }

    if (!pg_TwoIntsFromObj(size, &w, &h)) {
        PyErr_SetString(PyExc_TypeError, "size must be two numbers");
        return -1;
    }

    if (w < 0 || h < 0) {
        PyErr_SetString(PyExc_ValueError,
                        "cannot create mask with negative size");
        return -1;
    }

    bitmask = bitmask_create(w, h);

    if (NULL == bitmask) {
        PyErr_SetString(PyExc_MemoryError,
                        "cannot allocate memory for bitmask");
        return -1;
    }

    if (fill) {
        bitmask_fill(bitmask);
    }

    ((pgMaskObject *)self)->mask = bitmask;
    return 0;
}

typedef struct {
    int numbufs;
    Py_ssize_t shape[2];
    Py_ssize_t strides[2];
} mask_bufinfo;

static int
pgMask_GetBuffer(pgMaskObject *self, Py_buffer *view, int flags)
{
    bitmask_t *m = self->mask;
    mask_bufinfo *bufinfo = (mask_bufinfo *)self->bufdata;

    if (bufinfo == NULL) {
        bufinfo = PyMem_RawMalloc(sizeof(mask_bufinfo));
        if (bufinfo == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        bufinfo->numbufs = 1;

        bufinfo->shape[0] = (m->w - 1) / BITMASK_W_LEN + 1;
        bufinfo->shape[1] = m->h;

        bufinfo->strides[0] = m->h * sizeof(BITMASK_W);
        bufinfo->strides[1] = sizeof(BITMASK_W);

        self->bufdata = bufinfo;
    }
    else {
        bufinfo->numbufs++;
    }

    view->buf = m->bits;
    view->len = m->h * ((m->w - 1) / BITMASK_W_LEN + 1) * sizeof(BITMASK_W);
    view->readonly = 0;
    view->itemsize = sizeof(BITMASK_W);
    view->ndim = 2;
    view->internal = bufinfo;
    view->shape = (flags & PyBUF_ND) ? bufinfo->shape : NULL;
    view->strides = (flags & PyBUF_STRIDES) ? bufinfo->strides : NULL;
    if (flags & PyBUF_FORMAT) {
        view->format = "L"; /* L = unsigned long */
    }
    else {
        view->format = NULL;
    }
    view->suboffsets = NULL;

    Py_INCREF(self);
    view->obj = (PyObject *)self;

    return 0;
}

static void
pgMask_ReleaseBuffer(pgMaskObject *self, Py_buffer *view)
{
    mask_bufinfo *bufinfo = (mask_bufinfo *)view->internal;

    bufinfo->numbufs--;
    if (bufinfo->numbufs == 0) {
        PyMem_RawFree(bufinfo);
        self->bufdata = NULL;
    }
}

static PyBufferProcs pgMask_BufferProcs = {
    (getbufferproc)pgMask_GetBuffer, (releasebufferproc)pgMask_ReleaseBuffer};

static PyTypeObject pgMask_Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pygame.mask.Mask",
    .tp_basicsize = sizeof(pgMaskObject),
    .tp_dealloc = mask_dealloc,
    .tp_repr = (reprfunc)mask_repr,
    .tp_as_buffer = &pgMask_BufferProcs,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = DOC_PYGAMEMASKMASK,
    .tp_methods = mask_methods,
    .tp_init = mask_init,
    .tp_new = mask_new,
};

/*mask module methods*/
static PyMethodDef _mask_methods[] = {
    {"from_surface", (PyCFunction)mask_from_surface,
     METH_VARARGS | METH_KEYWORDS, DOC_PYGAMEMASKFROMSURFACE},
    {"from_threshold", (PyCFunction)mask_from_threshold,
     METH_VARARGS | METH_KEYWORDS, DOC_PYGAMEMASKFROMTHRESHOLD},
    {NULL, NULL, 0, NULL}};

MODINIT_DEFINE(mask)
{
    PyObject *module, *apiobj;
    static void *c_api[PYGAMEAPI_MASK_NUMSLOTS];

    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "mask",
                                         DOC_PYGAMEMASK,
                                         -1,
                                         _mask_methods,
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
    import_pygame_surface();
    if (PyErr_Occurred()) {
        return NULL;
    }
    import_pygame_rect();
    if (PyErr_Occurred()) {
        return NULL;
    }

    /* create the mask type */
    if (PyType_Ready(&pgMask_Type) < 0) {
        return NULL;
    }

    /* create the module */
    module = PyModule_Create(&_module);
    if (module == NULL) {
        return NULL;
    }
    Py_INCREF(&pgMask_Type);
    if (PyModule_AddObject(module, "MaskType", (PyObject *)&pgMask_Type)) {
        Py_DECREF(&pgMask_Type);
        Py_DECREF(module);
        return NULL;
    }

    Py_INCREF(&pgMask_Type);
    if (PyModule_AddObject(module, "Mask", (PyObject *)&pgMask_Type)) {
        Py_DECREF(&pgMask_Type);
        Py_DECREF(module);
        return NULL;
    }

    /* export the c api */
    c_api[0] = &pgMask_Type;
    apiobj = encapsulate_api(c_api, "mask");
    if (PyModule_AddObject(module, PYGAMEAPI_LOCAL_ENTRY, apiobj)) {
        Py_XDECREF(apiobj);
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
