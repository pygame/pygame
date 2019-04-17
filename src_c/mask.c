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

static PyTypeObject pgMask_Type;

/* mask object methods */

static PyObject *
mask_get_size(PyObject *self, PyObject *args)
{
    bitmask_t *mask = pgMask_AsBitmap(self);

    if (!PyArg_ParseTuple(args, ""))
        return NULL;

    return Py_BuildValue("(ii)", mask->w, mask->h);
}

static PyObject *
mask_get_at(PyObject *self, PyObject *args)
{
    bitmask_t *mask = pgMask_AsBitmap(self);
    int x, y, val;

    if (!PyArg_ParseTuple(args, "(ii)", &x, &y))
        return NULL;
    if (x >= 0 && x < mask->w && y >= 0 && y < mask->h) {
        val = bitmask_getbit(mask, x, y);
    }
    else {
        PyErr_Format(PyExc_IndexError, "%d, %d is out of bounds", x, y);
        return NULL;
    }

    return PyInt_FromLong(val);
}

static PyObject *
mask_set_at(PyObject *self, PyObject *args)
{
    bitmask_t *mask = pgMask_AsBitmap(self);
    int x, y, value = 1;

    if (!PyArg_ParseTuple(args, "(ii)|i", &x, &y, &value))
        return NULL;
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
mask_overlap(PyObject *self, PyObject *args)
{
    bitmask_t *mask = pgMask_AsBitmap(self);
    bitmask_t *othermask;
    PyObject *maskobj;
    int x, y, val;
    int xp, yp;

    if (!PyArg_ParseTuple(args, "O!(ii)", &pgMask_Type, &maskobj, &x, &y))
        return NULL;
    othermask = pgMask_AsBitmap(maskobj);

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
mask_overlap_area(PyObject *self, PyObject *args)
{
    bitmask_t *mask = pgMask_AsBitmap(self);
    bitmask_t *othermask;
    PyObject *maskobj;
    int x, y, val;

    if (!PyArg_ParseTuple(args, "O!(ii)", &pgMask_Type, &maskobj, &x, &y)) {
        return NULL;
    }
    othermask = pgMask_AsBitmap(maskobj);

    val = bitmask_overlap_area(mask, othermask, x, y);
    return PyInt_FromLong(val);
}

static PyObject *
mask_overlap_mask(PyObject *self, PyObject *args)
{
    int x, y;
    bitmask_t *mask = pgMask_AsBitmap(self);
    bitmask_t *output = bitmask_create(mask->w, mask->h);
    bitmask_t *othermask;
    PyObject *maskobj;
    pgMaskObject *maskobj2 = PyObject_New(pgMaskObject, &pgMask_Type);

    if (!PyArg_ParseTuple(args, "O!(ii)", &pgMask_Type, &maskobj, &x, &y)) {
        return NULL;
    }
    othermask = pgMask_AsBitmap(maskobj);

    bitmask_overlap_mask(mask, othermask, output, x, y);

    if (maskobj2)
        maskobj2->mask = output;

    return (PyObject *)maskobj2;
}

static PyObject *
mask_fill(PyObject *self, PyObject *args)
{
    bitmask_t *mask = pgMask_AsBitmap(self);

    bitmask_fill(mask);

    Py_RETURN_NONE;
}

static PyObject *
mask_clear(PyObject *self, PyObject *args)
{
    bitmask_t *mask = pgMask_AsBitmap(self);

    bitmask_clear(mask);

    Py_RETURN_NONE;
}

static PyObject *
mask_invert(PyObject *self, PyObject *args)
{
    bitmask_t *mask = pgMask_AsBitmap(self);

    bitmask_invert(mask);

    Py_RETURN_NONE;
}

static PyObject *
mask_scale(PyObject *self, PyObject *args)
{
    int x, y;
    bitmask_t *input = pgMask_AsBitmap(self);
    bitmask_t *output;
    pgMaskObject *maskobj = PyObject_New(pgMaskObject, &pgMask_Type);

    if (!PyArg_ParseTuple(args, "(ii)", &x, &y)) {
        return NULL;
    }

    if (x < 0 || y < 0) {
        return RAISE(PyExc_ValueError, "Cannot scale mask to negative size");
    }
    output = bitmask_scale(input, x, y);

    if (maskobj)
        maskobj->mask = output;

    return (PyObject *)maskobj;
}

static PyObject *
mask_draw(PyObject *self, PyObject *args)
{
    bitmask_t *mask = pgMask_AsBitmap(self);
    bitmask_t *othermask;
    PyObject *maskobj;
    int x, y;

    if (!PyArg_ParseTuple(args, "O!(ii)", &pgMask_Type, &maskobj, &x, &y)) {
        return NULL;
    }
    othermask = pgMask_AsBitmap(maskobj);

    bitmask_draw(mask, othermask, x, y);

    Py_RETURN_NONE;
}

static PyObject *
mask_erase(PyObject *self, PyObject *args)
{
    bitmask_t *mask = pgMask_AsBitmap(self);
    bitmask_t *othermask;
    PyObject *maskobj;
    int x, y;

    if (!PyArg_ParseTuple(args, "O!(ii)", &pgMask_Type, &maskobj, &x, &y)) {
        return NULL;
    }
    othermask = pgMask_AsBitmap(maskobj);

    bitmask_erase(mask, othermask, x, y);

    Py_RETURN_NONE;
}

static PyObject *
mask_count(PyObject *self, PyObject *args)
{
    bitmask_t *m = pgMask_AsBitmap(self);

    return PyInt_FromLong(bitmask_count(m));
}

static PyObject *
mask_centroid(PyObject *self, PyObject *args)
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
        xobj = PyInt_FromLong(m10 / m00);
        yobj = PyInt_FromLong(m01 / m00);
    }
    else {
        xobj = PyInt_FromLong(0);
        yobj = PyInt_FromLong(0);
    }

    return Py_BuildValue("(NN)", xobj, yobj);
}

static PyObject *
mask_angle(PyObject *self, PyObject *args)
{
    bitmask_t *mask = pgMask_AsBitmap(self);
    int x, y, xc, yc;
    long int m10, m01, m00, m20, m02, m11;
    double theta;

    m10 = m01 = m00 = m20 = m02 = m11 = 0;

    for (x = 0; x < mask->w; x++) {
        for (y = 0; y < mask->h; y++) {
            if (bitmask_getbit(mask, x, y)) {
                m10 += x;
                m20 += x * x;
                m11 += x * y;
                m02 += y * y;
                m01 += y;
                m00++;
            }
        }
    }

    if (m00) {
        xc = m10 / m00;
        yc = m01 / m00;
        theta = -90.0 *
                atan2(2 * (m11 / m00 - xc * yc),
                      (m20 / m00 - xc * xc) - (m02 / m00 - yc * yc)) /
                M_PI;
        return PyFloat_FromDouble(theta);
    }
    else {
        return PyFloat_FromDouble(0);
    }
}

static PyObject *
mask_outline(PyObject *self, PyObject *args)
{
    bitmask_t *c = pgMask_AsBitmap(self);
    bitmask_t *m = NULL;
    PyObject *plist = NULL;
    PyObject *value = NULL;
    int x, y, firstx, firsty, secx, secy, currx, curry, nextx, nexty, n;
    int e, every = 1;
    int a[] = {1, 1, 0, -1, -1, -1,  0,  1, 1, 1, 0, -1, -1, -1};
    int b[] = {0, 1, 1,  1,  0, -1, -1, -1, 0, 1, 1,  1,  0, -1};

    n = firstx = firsty = secx = x = 0;

    if (!PyArg_ParseTuple(args, "|i", &every)) {
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
                PyList_Append(plist, value);
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
                PyList_Append(plist, value);
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
                    PyList_Append(plist, value);
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
mask_convolve(PyObject *aobj, PyObject *args)
{
    PyObject *bobj = NULL;
    PyObject *oobj = Py_None;
    bitmask_t *a = NULL, *b = NULL, *output = NULL;
    int xoffset = 0, yoffset = 0;

    if (!PyArg_ParseTuple(args, "O!|O(ii)", &pgMask_Type, &bobj, &oobj,
                          &xoffset, &yoffset)) {
        return NULL;
    }

    a = pgMask_AsBitmap(aobj);
    b = pgMask_AsBitmap(bobj);

    if (oobj != Py_None) {
        /* Use this mask for the output. */
        Py_INCREF(oobj);
        output = pgMask_AsBitmap(oobj);
    }
    else {
        pgMaskObject *result = PyObject_New(pgMaskObject, &pgMask_Type);

        if (NULL == result) {
            return RAISE(PyExc_MemoryError, "cannot allocate memory for mask");
        }

        output =
            bitmask_create(MAX(0, a->w + b->w - 1), MAX(0, a->h + b->h - 1));

        if (NULL == output) {
            Py_DECREF(result);
            return RAISE(PyExc_MemoryError,
                         "cannot allocate memory for bitmask");
        }

        result->mask = output;
        oobj = (PyObject *)result;
    }

    bitmask_convolve(a, b, output, xoffset, yoffset);

    return oobj;
}

static PyObject *
mask_from_surface(PyObject *self, PyObject *args)
{
    bitmask_t *mask;
    SDL_Surface *surf;

    PyObject *surfobj;
    pgMaskObject *maskobj;

    int x, y, threshold, ashift, aloss, usethresh;
    Uint8 *pixels;

    SDL_PixelFormat *format;
    Uint32 color, amask;
#if IS_SDLv2
    Uint32 colorkey;
#endif /* IS_SDLv2 */
    Uint8 *pix;
    Uint8 a;

    /* set threshold as 127 default argument. */
    threshold = 127;

    /* get the surface from the passed in arguments.
     *   surface, threshold
     */

    if (!PyArg_ParseTuple(args, "O!|i", &pgSurface_Type, &surfobj,
                          &threshold)) {
        return NULL;
    }

    surf = pgSurface_AsSurface(surfobj);

    if (surf->w < 0 || surf->h < 0) {
        return RAISE(PyExc_ValueError,
                     "cannot create mask with negative size");
    }

    /* lock the surface, release the GIL. */
    pgSurface_Lock(surfobj);

    Py_BEGIN_ALLOW_THREADS;

    /* get the size from the surface, and create the mask. */
    mask = bitmask_create(surf->w, surf->h);

    if (!mask) {
        /* Py_END_ALLOW_THREADS;
         */
        return NULL; /*RAISE(PyExc_Error, "cannot create bitmask");*/
    }

    pixels = (Uint8 *)surf->pixels;
    format = surf->format;
    amask = format->Amask;
    ashift = format->Ashift;
    aloss = format->Aloss;
#if IS_SDLv1
    usethresh = !(surf->flags & SDL_SRCCOLORKEY);
#else  /* IS_SDLv2 */
    usethresh = (SDL_GetColorKey(surf, &colorkey) == -1);
#endif /* IS_SDLv2 */

    for (y = 0; y < surf->h; y++) {
        pixels = (Uint8 *)surf->pixels + y * surf->pitch;
        for (x = 0; x < surf->w; x++) {
            /* Get the color.  TODO: should use an inline helper
             *   function for this common function. */
            switch (format->BytesPerPixel) {
                case 1:
                    color = (Uint32) * ((Uint8 *)pixels);
                    pixels++;
                    break;
                case 2:
                    color = (Uint32) * ((Uint16 *)pixels);
                    pixels += 2;
                    break;
                case 3:
                    pix = ((Uint8 *)pixels);
                    pixels += 3;
#if IS_SDLv1
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                    color = (pix[0]) + (pix[1] << 8) + (pix[2] << 16);
#else
                    color = (pix[2]) + (pix[1] << 8) + (pix[0] << 16);
#endif
#else  /* IS_SDLv2 */
                    color = (pix[2]) + (pix[1] << 8) + (pix[0] << 16);
#endif /* IS_SDLv2 */
                    break;
                default: /* case 4: */
                    color = *((Uint32 *)pixels);
                    pixels += 4;
                    break;
            }

            if (usethresh) {
                a = ((color & amask) >> ashift) << aloss;
                /* no colorkey, so we check the threshold of the alpha */
                if (a > threshold) {
                    bitmask_setbit(mask, x, y);
                }
            }
            else {
                /*  test against the colour key. */
#if IS_SDLv1
                if (format->colorkey != color) {
#else  /* IS_SDLv2 */
                if (colorkey != color) {
#endif /* IS_SDLv2 */
                    bitmask_setbit(mask, x, y);
                }
            }
        }
    }

    Py_END_ALLOW_THREADS;

    /* unlock the surface, release the GIL.
     */
    pgSurface_Unlock(surfobj);

    /*create the new python object from mask*/
    maskobj = PyObject_New(pgMaskObject, &pgMask_Type);
    if (maskobj)
        maskobj->mask = mask;

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

    pixels = (Uint8 *)surf->pixels;
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
                    if ((abs((the_color2) - (the_color)) < tr)) {
                        /* this pixel is within the threshold of othersurface.
                         */
                        bitmask_setbit(m, x, y);
                    }
                }
                else if ((abs((((the_color2 & rmask2) >> rshift2) << rloss2) -
                              (((the_color & rmask) >> rshift) << rloss)) <
                          tr) &
                         (abs((((the_color2 & gmask2) >> gshift2) << gloss2) -
                              (((the_color & gmask) >> gshift) << gloss)) <
                          tg) &
                         (abs((((the_color2 & bmask2) >> bshift2) << bloss2) -
                              (((the_color & bmask) >> bshift) << bloss)) <
                          tb)) {
                    /* this pixel is within the threshold of othersurface. */
                    bitmask_setbit(m, x, y);
                }

                /* TODO: will need to handle surfaces with palette colors.
                   TODO: will need to handle the case where palette_colors == 0
                */
            }
            else if ((abs((((the_color & rmask) >> rshift) << rloss) - r) <
                      tr) &
                     (abs((((the_color & gmask) >> gshift) << gloss) - g) <
                      tg) &
                     (abs((((the_color & bmask) >> bshift) << bloss) - b) <
                      tb)) {
                /* this pixel is within the threshold of the color. */
                bitmask_setbit(m, x, y);
            }
        }
    }
}

static PyObject *
mask_from_threshold(PyObject *self, PyObject *args)
{
    PyObject *surfobj, *surfobj2 = NULL;
    pgMaskObject *maskobj;
    bitmask_t *m;
    SDL_Surface *surf = NULL, *surf2 = NULL;
    int bpp;
    PyObject *rgba_obj_color, *rgba_obj_threshold = NULL;
    Uint8 rgba_color[4];
    Uint8 rgba_threshold[4] = {0, 0, 0, 255};
    Uint32 color;
    Uint32 color_threshold;
    int palette_colors = 1;

    if (!PyArg_ParseTuple(args, "O!O|OO!i", &pgSurface_Type, &surfobj,
                          &rgba_obj_color, &rgba_obj_threshold,
                          &pgSurface_Type, &surfobj2, &palette_colors))
        return NULL;

    surf = pgSurface_AsSurface(surfobj);
    if (surfobj2) {
        surf2 = pgSurface_AsSurface(surfobj2);
    }

    if (PyInt_Check(rgba_obj_color)) {
        color = (Uint32)PyInt_AsLong(rgba_obj_color);
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
        if (PyInt_Check(rgba_obj_threshold))
            color_threshold = (Uint32)PyInt_AsLong(rgba_obj_threshold);
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

    bpp = surf->format->BytesPerPixel;
    m = bitmask_create(surf->w, surf->h);

    pgSurface_Lock(surfobj);
    if (surfobj2) {
        pgSurface_Lock(surfobj2);
    }

    Py_BEGIN_ALLOW_THREADS;
    bitmask_threshold(m, surf, surf2, color, color_threshold, palette_colors);
    Py_END_ALLOW_THREADS;

    pgSurface_Unlock(surfobj);
    if (surfobj2) {
        pgSurface_Unlock(surfobj2);
    }

    maskobj = PyObject_New(pgMaskObject, &pgMask_Type);
    if (maskobj)
        maskobj->mask = m;

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
 * Params:
 *     input - mask to search in for the connected components to bound
 *     num_bounding_boxes - passes back the number of bounding rects found
 *     rects - passes back the bounding rects that are found, memory is
 *         allocated
 *
 * Returns:
 *     0 on success
 *     -2 on memory allocation error
 */
static int
get_bounding_rects(bitmask_t *input, int *num_bounding_boxes,
                   GAME_Rect **ret_rects)
{
    unsigned int *image, *ufind, *largest, *buf;
    int x, y, w, h, temp, label, relabel;
    GAME_Rect *rects;

    rects = NULL;
    label = 0;

    w = input->w;
    h = input->h;

    if (!w || !h) {
        *ret_rects = rects;
        return 0;
    }
    /* a temporary image to assign labels to each bit of the mask */
    image = (unsigned int *)malloc(sizeof(int) * w * h);
    if (!image) {
        return -2;
    }

    /* allocate enough space for the maximum possible connected components */
    /* the union-find array. see wikipedia for info on union find */
    ufind = (unsigned int *)malloc(sizeof(int) * (w / 2 + 1) * (h / 2 + 1));
    if (!ufind) {
        return -2;
    }

    largest = (unsigned int *)malloc(sizeof(int) * (w / 2 + 1) * (h / 2 + 1));
    if (!largest) {
        return -2;
    }

    /* do the initial labelling */
    label = cc_label(input, image, ufind, largest);

    relabel = 0;
    /* flatten and relabel the union-find equivalence array.  Start at label 1
       because label 0 indicates an unset pixel.  For this reason, we also use
       <= label rather than < label. */
    for (x = 1; x <= label; x++) {
        if (ufind[x] < x) {             /* is it a union find root? */
            ufind[x] = ufind[ufind[x]]; /* relabel it to its root */
        }
        else { /* its a root */
            relabel++;
            ufind[x] = relabel; /* assign the lowest label available */
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
    rects = (GAME_Rect *)malloc(sizeof(GAME_Rect) * (relabel + 1));
    if (!rects) {
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
mask_get_bounding_rects(PyObject *self, PyObject *args)
{
    GAME_Rect *regions;
    GAME_Rect *aregion;
    int num_bounding_boxes, i, r;
    PyObject *ret;
    PyObject *rect;

    bitmask_t *mask = pgMask_AsBitmap(self);

    ret = NULL;
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

    ret = PyList_New(0);
    if (!ret)
        return NULL;

    /* build a list of rects to return.  Starts at 1 because we never use 0. */
    for (i = 1; i <= num_bounding_boxes; i++) {
        aregion = regions + i;
        rect = pgRect_New4(aregion->x, aregion->y, aregion->w, aregion->h);
        PyList_Append(ret, rect);
        Py_DECREF(rect);
    }

    free(regions);

    return ret;
}

/* Finds all the connected components in a given mask.
 *
 * Allocates memory for components.
 *
 * Params:
 *     mask - mask to search in for the connected components
 *     components - passes back an array of connected component masks,
 *         memory is allocated
 *     min - minimum number of pixels for a component to be considered
 *
 * Returns:
 *     the number of connected components (>= 0)
 *     -2 on memory allocation error
 */
static int
get_connected_components(bitmask_t *mask, bitmask_t ***components, int min)
{
    unsigned int *image, *ufind, *largest, *buf;
    int x, y, w, h, label, relabel;
    bitmask_t **comps;

    label = 0;

    w = mask->w;
    h = mask->h;

    if (!w || !h) {
        return 0;
    }

    /* a temporary image to assign labels to each bit of the mask */
    image = (unsigned int *)malloc(sizeof(int) * w * h);
    if (!image) {
        return -2;
    }

    /* allocate enough space for the maximum possible connected components */
    /* the union-find array. see wikipedia for info on union find */
    ufind = (unsigned int *)malloc(sizeof(int) * (w / 2 + 1) * (h / 2 + 1));
    if (!ufind) {
        free(image);
        return -2;
    }

    largest = (unsigned int *)malloc(sizeof(int) * (w / 2 + 1) * (h / 2 + 1));
    if (!largest) {
        free(image);
        free(ufind);
        return -2;
    }

    /* do the initial labelling */
    label = cc_label(mask, image, ufind, largest);

    for (x = 1; x <= label; x++) {
        if (ufind[x] < x) {
            largest[ufind[x]] += largest[x];
        }
    }

    relabel = 0;
    /* flatten and relabel the union-find equivalence array.  Start at label 1
       because label 0 indicates an unset pixel.  For this reason, we also use
       <= label rather than < label. */
    for (x = 1; x <= label; x++) {
        if (ufind[x] < x) {             /* is it a union find root? */
            ufind[x] = ufind[ufind[x]]; /* relabel it to its root */
        }
        else { /* its a root */
            if (largest[x] >= min) {
                relabel++;
                ufind[x] = relabel; /* assign the lowest label available */
            }
            else {
                ufind[x] = 0;
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
mask_connected_components(PyObject *self, PyObject *args)
{
    PyObject *ret;
    pgMaskObject *maskobj;
    bitmask_t **components;
    bitmask_t *mask = pgMask_AsBitmap(self);
    int i, num_components, min;

    min = 0;
    components = NULL;

    if (!PyArg_ParseTuple(args, "|i", &min)) {
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS;
    num_components = get_connected_components(mask, &components, min);
    Py_END_ALLOW_THREADS;

    if (num_components == -2)
        return RAISE(PyExc_MemoryError,
                     "Not enough memory to get components. \n");

    ret = PyList_New(0);
    if (!ret)
        return NULL;

    for (i = 1; i <= num_components; i++) {
        maskobj = PyObject_New(pgMaskObject, &pgMask_Type);
        if (maskobj) {
            maskobj->mask = components[i];
            PyList_Append(ret, (PyObject *)maskobj);
            Py_DECREF((PyObject *)maskobj);
        }
    }

    free(components);
    return ret;
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
    image = (unsigned int *)malloc(sizeof(int) * w * h);
    if (!image) {
        return -2;
    }
    /* allocate enough space for the maximum possible connected components */
    /* the union-find array. see wikipedia for info on union find */
    ufind = (unsigned int *)malloc(sizeof(int) * (w / 2 + 1) * (h / 2 + 1));
    if (!ufind) {
        free(image);
        return -2;
    }
    /* an array to track the number of pixels associated with each label */
    largest = (unsigned int *)malloc(sizeof(int) * (w / 2 + 1) * (h / 2 + 1));
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
mask_connected_component(PyObject *self, PyObject *args)
{
    bitmask_t *input = pgMask_AsBitmap(self);
    bitmask_t *output = NULL;
    pgMaskObject *maskobj = NULL;
    int x = -1, y = -1;
    Py_ssize_t args_exist = PyTuple_Size(args);

    if (args_exist) {
        if (!PyArg_ParseTuple(args, "|(ii)", &x, &y)) {
            return NULL;
        }

        if (x < 0 || x >= input->w || y < 0 || y >= input->h) {
            return PyErr_Format(PyExc_IndexError, "%d, %d is out of bounds", x,
                                y);
        }
    }

    output = bitmask_create(input->w, input->h);

    /* If a pixel index is provided and the indexed bit is not set, then the
     * returned mask is empty.
     */
    if (!args_exist || bitmask_getbit(input, x, y)) {
        if (largest_connected_comp(input, output, x, y) == -2) {
            bitmask_free(output);
            return RAISE(PyExc_MemoryError,
                         "cannot allocate memory for connected component");
        }
    }

    maskobj = PyObject_New(pgMaskObject, &pgMask_Type);
    if (maskobj) {
        maskobj->mask = output;
    }

    return (PyObject *)maskobj;
}

static PyMethodDef mask_methods[] = {
    {"get_size", mask_get_size, METH_VARARGS, DOC_MASKGETSIZE},
    {"get_at", mask_get_at, METH_VARARGS, DOC_MASKGETAT},
    {"set_at", mask_set_at, METH_VARARGS, DOC_MASKSETAT},
    {"overlap", mask_overlap, METH_VARARGS, DOC_MASKOVERLAP},
    {"overlap_area", mask_overlap_area, METH_VARARGS, DOC_MASKOVERLAPAREA},
    {"overlap_mask", mask_overlap_mask, METH_VARARGS, DOC_MASKOVERLAPMASK},
    {"fill", mask_fill, METH_NOARGS, DOC_MASKFILL},
    {"clear", mask_clear, METH_NOARGS, DOC_MASKCLEAR},
    {"invert", mask_invert, METH_NOARGS, DOC_MASKINVERT},
    {"scale", mask_scale, METH_VARARGS, DOC_MASKSCALE},
    {"draw", mask_draw, METH_VARARGS, DOC_MASKDRAW},
    {"erase", mask_erase, METH_VARARGS, DOC_MASKERASE},
    {"count", mask_count, METH_NOARGS, DOC_MASKCOUNT},
    {"centroid", mask_centroid, METH_NOARGS, DOC_MASKCENTROID},
    {"angle", mask_angle, METH_NOARGS, DOC_MASKANGLE},
    {"outline", mask_outline, METH_VARARGS, DOC_MASKOUTLINE},
    {"convolve", mask_convolve, METH_VARARGS, DOC_MASKCONVOLVE},
    {"connected_component", mask_connected_component, METH_VARARGS,
     DOC_MASKCONNECTEDCOMPONENT},
    {"connected_components", mask_connected_components, METH_VARARGS,
     DOC_MASKCONNECTEDCOMPONENTS},
    {"get_bounding_rects", mask_get_bounding_rects, METH_NOARGS,
     DOC_MASKGETBOUNDINGRECTS},

    {NULL, NULL, 0, NULL}};

/*mask object internals*/

static void
mask_dealloc(PyObject *self)
{
    bitmask_t *mask = pgMask_AsBitmap(self);
    bitmask_free(mask);
    PyObject_DEL(self);
}

static PyObject *
mask_repr(PyObject *self)
{
    bitmask_t *mask = pgMask_AsBitmap(self);
    return Text_FromFormat("<Mask(%dx%d)>", mask->w, mask->h);
}

static PyTypeObject pgMask_Type = {
    TYPE_HEAD(NULL, 0) "pygame.mask.Mask", /* tp_name */
    sizeof(pgMaskObject), /* tp_basicsize */
    0,                    /* tp_itemsize */
    mask_dealloc,         /* tp_dealloc */
    0,                    /* tp_print */
    0,                    /* tp_getattr */
    0,                    /* tp_setattr */
    0,                    /* tp_as_async (formerly tp_compare/tp_reserved) */
    (reprfunc)mask_repr,  /* tp_repr */
    0,                    /* tp_as_number */
    NULL,                 /* tp_as_sequence */
    0,                    /* tp_as_mapping */
    (hashfunc)NULL,       /* tp_hash */
    (ternaryfunc)NULL,    /* tp_call */
    (reprfunc)NULL,       /* tp_str */
    0L,                   /* tp_getattro */
    0L,                   /* tp_setattro */
    0L,                   /* tp_as_buffer */
    0L,                   /* tp_flags */
    DOC_PYGAMEMASKMASK, /* Documentation string */
    0,                  /* tp_traverse */
    0,                  /* tp_clear */
    0,                  /* tp_richcompare */
    0,                  /* tp_weaklistoffset */
    0,                  /* tp_iter */
    0,                  /* tp_iternext */
    mask_methods,       /* tp_methods */
    0,                  /* tp_members */
    0,                  /* tp_getset */
    0,                  /* tp_base */
    0,                  /* tp_dict */
    0,                  /* tp_descr_get */
    0,                  /* tp_descr_set */
    0,                  /* tp_dictoffset */
    0,                  /* tp_init */
    0,                  /* tp_alloc */
    0,                  /* tp_new */
};

/*mask module methods*/

static PyObject *
Mask(PyObject *self, PyObject *args, PyObject *kwargs)
{
    bitmask_t *mask;
    int w, h;
    int fill = 0; /* Default is false. */
    pgMaskObject *maskobj;
    char *keywords[] = {"size", "fill", NULL};
#if PY3
    const char *format = "(ii)|p";
#else
    const char *format = "(ii)|i";
#endif

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, format, keywords, &w, &h,
                                     &fill))
        return NULL;

    if (w < 0 || h < 0) {
        return RAISE(PyExc_ValueError,
                     "cannot create mask with negative size");
    }

    mask = bitmask_create(w, h);
    if (!mask)
        return RAISE(PyExc_MemoryError,
                     "cannot allocate enough memory for mask");

    if (fill)
        bitmask_fill(mask);

    /*create the new python object from mask*/
    maskobj = PyObject_New(pgMaskObject, &pgMask_Type);
    if (maskobj)
        maskobj->mask = mask;

    return (PyObject *)maskobj;
}

static PyMethodDef _mask_methods[] = {
    {"Mask", (PyCFunction)Mask, METH_VARARGS | METH_KEYWORDS,
     DOC_PYGAMEMASKMASK},
    {"from_surface", mask_from_surface, METH_VARARGS,
     DOC_PYGAMEMASKFROMSURFACE},
    {"from_threshold", mask_from_threshold, METH_VARARGS,
     DOC_PYGAMEMASKFROMTHRESHOLD},
    {NULL, NULL, 0, NULL}};

MODINIT_DEFINE(mask)
{
    PyObject *module, *dict, *apiobj;
    static void *c_api[PYGAMEAPI_MASK_NUMSLOTS];

#if PY3
    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "mask",
                                         DOC_PYGAMEMASK,
                                         -1,
                                         _mask_methods,
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
    import_pygame_surface();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }
    import_pygame_rect();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }

    /* create the mask type */
    if (PyType_Ready(&pgMask_Type) < 0) {
        MODINIT_ERROR;
    }

    /* create the module */
#if PY3
    module = PyModule_Create(&_module);
#else
    module = Py_InitModule3(MODPREFIX "mask", _mask_methods, DOC_PYGAMEMASK);
#endif
    if (module == NULL) {
        MODINIT_ERROR;
    }
    dict = PyModule_GetDict(module);
    if (PyDict_SetItemString(dict, "MaskType", (PyObject *)&pgMask_Type) ==
        -1) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    /* export the c api */
    c_api[0] = &pgMask_Type;
    apiobj = encapsulate_api(c_api, "mask");
    if (apiobj == NULL) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    if (PyModule_AddObject(module, PYGAMEAPI_LOCAL_ENTRY, apiobj) == -1) {
        Py_DECREF(apiobj);
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    MODINIT_RETURN(module);
}
