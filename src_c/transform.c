/*
  pygame - Python Game Library
  Copyright (C) 2000-2001  Pete Shinners
  Copyright (C) 2007  Rene Dudfield, Richard Goedeken

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
 *  surface transformations for pygame
 */
#include "pygame.h"

#include "pgcompat.h"

#include "doc/transform_doc.h"

#include <math.h>
#include <string.h>
#include <limits.h>

#include "simd_shared.h"
#include "simd_transform.h"
#include "scale.h"

typedef void (*SMOOTHSCALE_FILTER_P)(Uint8 *, Uint8 *, int, int, int, int,
                                     int);
struct _module_state {
    const char *filter_type;
    SMOOTHSCALE_FILTER_P filter_shrink_X;
    SMOOTHSCALE_FILTER_P filter_shrink_Y;
    SMOOTHSCALE_FILTER_P filter_expand_X;
    SMOOTHSCALE_FILTER_P filter_expand_Y;
};

#define GETSTATE(m) ((struct _module_state *)PyModule_GetState(m))

void
scale2x(SDL_Surface *src, SDL_Surface *dst);
extern SDL_Surface *
rotozoomSurface(SDL_Surface *src, double angle, double zoom, int smooth);

static int
_get_factor(PyObject *factorobj, float *x, float *y)
{
    Py_ssize_t len = PyObject_Length(factorobj);
    if (PyErr_Occurred()) {
        PyErr_Clear();
    }

    if (len > 2) {
        PyErr_Format(PyExc_TypeError,
                     "factor should be either one number or a sequence of two "
                     "numbers.");
        return 0;
    }
    if (len == 2) {
        if (!pg_TwoFloatsFromObj(factorobj, x, y)) {
            PyErr_Format(PyExc_TypeError,
                         "factor should be either one number or a sequence of "
                         "two numbers.");
            return 0;
        }
        return 1;
    }
    if (!pg_FloatFromObj(factorobj, x)) {
        PyErr_Format(PyExc_TypeError,
                     "factor should be either one number or a sequence of two "
                     "numbers.");
        return 0;
    }
    *y = *x;
    return 1;
}
#if defined(BUILD_STATIC)
extern int
_PgSurface_SrcAlpha(SDL_Surface *surf);
#else
static int
_PgSurface_SrcAlpha(SDL_Surface *surf)
{
    if (SDL_ISPIXELFORMAT_ALPHA(PG_SURF_FORMATENUM(surf))) {
        SDL_BlendMode mode;
        if (!PG_GetSurfaceBlendMode(surf, &mode)) {
            return -1;
        }
        if (mode == SDL_BLENDMODE_BLEND) {
            return 1;
        }
    }
    else {
        Uint8 color = SDL_ALPHA_OPAQUE;
        if (!PG_GetSurfaceAlphaMod(surf, &color)) {
            return -1;
        }
        if (color != SDL_ALPHA_OPAQUE) {
            return 1;
        }
    }
    return 0;
}
#endif

static SDL_Surface *
newsurf_fromsurf(SDL_Surface *surf, int width, int height)
{
    SDL_Surface *newsurf;
    Uint32 colorkey;
    Uint8 alpha;
    int isalpha;

    if (PG_SURF_BytesPerPixel(surf) == 0 || PG_SURF_BytesPerPixel(surf) > 4) {
        return (SDL_Surface *)(RAISE(
            PyExc_ValueError, "unsupported Surface bit depth for transform"));
    }

    newsurf = PG_CreateSurface(width, height, PG_SURF_FORMATENUM(surf));
    if (!newsurf) {
        return (SDL_Surface *)(RAISE(pgExc_SDLError, SDL_GetError()));
    }

    /* Copy palette, colorkey, etc info */
    if (SDL_ISPIXELFORMAT_INDEXED(PG_SURF_FORMATENUM(surf))) {
        SDL_Palette *newsurf_palette = PG_GetSurfacePalette(newsurf);
        SDL_Palette *surf_palette = PG_GetSurfacePalette(surf);

        if (newsurf_palette == NULL) {
            PyErr_SetString(
                pgExc_SDLError,
                "Palette expected (newsurf) but no palette found.");
            SDL_FreeSurface(newsurf);
            return NULL;
        }

        if (surf_palette == NULL) {
            PyErr_SetString(pgExc_SDLError,
                            "Palette expected (surf) but no palette found.");
            SDL_FreeSurface(newsurf);
            return NULL;
        }

        if (!PG_SetPaletteColors(newsurf_palette, surf_palette->colors, 0,
                                 surf_palette->ncolors)) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            SDL_FreeSurface(newsurf);
            return NULL;
        }
    }

    if (!PG_GetSurfaceAlphaMod(surf, &alpha)) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        SDL_FreeSurface(newsurf);
        return NULL;
    }
    if (alpha != 255) {
        if (!PG_SetSurfaceAlphaMod(newsurf, alpha)) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            SDL_FreeSurface(newsurf);
            return NULL;
        }
    }

    isalpha = _PgSurface_SrcAlpha(surf);
    if (isalpha == 1) {
        if (!PG_SetSurfaceBlendMode(newsurf, SDL_BLENDMODE_BLEND)) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            SDL_FreeSurface(newsurf);
            return NULL;
        }
    }
    else if (isalpha == -1) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        SDL_FreeSurface(newsurf);
        return NULL;
    }
    else {
        if (!PG_SetSurfaceBlendMode(newsurf, SDL_BLENDMODE_NONE)) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            SDL_FreeSurface(newsurf);
            return NULL;
        }
    }

    if (SDL_HasColorKey(surf)) {
        SDL_GetColorKey(surf, &colorkey);
        if (!PG_SetSurfaceColorKey(newsurf, SDL_TRUE, colorkey)) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            SDL_FreeSurface(newsurf);
            return NULL;
        }
        if (PG_SurfaceHasRLE(surf) && !PG_SetSurfaceRLE(newsurf, SDL_TRUE)) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            SDL_FreeSurface(newsurf);
            return NULL;
        }
    }

    return newsurf;
}

static SDL_Surface *
rotate90(SDL_Surface *src, int angle)
{
    int numturns = (angle / 90) % 4;
    int dstwidth, dstheight;
    SDL_Surface *dst;
    char *srcpix, *dstpix, *srcrow, *dstrow;
    int srcstepx, srcstepy, dststepx, dststepy;
    int loopx, loopy;

    if (numturns < 0) {
        numturns = 4 + numturns;
    }
    if (!(numturns % 2)) {
        dstwidth = src->w;
        dstheight = src->h;
    }
    else {
        dstwidth = src->h;
        dstheight = src->w;
    }

    dst = newsurf_fromsurf(src, dstwidth, dstheight);
    if (!dst) {
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS;
    SDL_LockSurface(dst);
    srcrow = (char *)src->pixels;
    dstrow = (char *)dst->pixels;
    srcstepx = dststepx = PG_SURF_BytesPerPixel(src);
    srcstepy = src->pitch;
    dststepy = dst->pitch;

    switch (numturns) {
            /*case 0: we don't need to change anything*/
        case 1:
            srcrow += ((src->w - 1) * srcstepx);
            srcstepy = -srcstepx;
            srcstepx = src->pitch;
            break;
        case 2:
            srcrow += ((src->h - 1) * srcstepy) + ((src->w - 1) * srcstepx);
            srcstepx = -srcstepx;
            srcstepy = -srcstepy;
            break;
        case 3:
            srcrow += ((src->h - 1) * srcstepy);
            srcstepx = -srcstepy;
            srcstepy = PG_SURF_BytesPerPixel(src);
            break;
    }

    switch (PG_SURF_BytesPerPixel(src)) {
        case 1:
            for (loopy = 0; loopy < dstheight; ++loopy) {
                dstpix = dstrow;
                srcpix = srcrow;
                for (loopx = 0; loopx < dstwidth; ++loopx) {
                    *dstpix = *srcpix;
                    srcpix += srcstepx;
                    dstpix += dststepx;
                }
                dstrow += dststepy;
                srcrow += srcstepy;
            }
            break;
        case 2:
            for (loopy = 0; loopy < dstheight; ++loopy) {
                dstpix = dstrow;
                srcpix = srcrow;
                for (loopx = 0; loopx < dstwidth; ++loopx) {
                    *(Uint16 *)dstpix = *(Uint16 *)srcpix;
                    srcpix += srcstepx;
                    dstpix += dststepx;
                }
                dstrow += dststepy;
                srcrow += srcstepy;
            }
            break;
        case 3:
            for (loopy = 0; loopy < dstheight; ++loopy) {
                dstpix = dstrow;
                srcpix = srcrow;
                for (loopx = 0; loopx < dstwidth; ++loopx) {
                    memcpy(dstpix, srcpix, 3);
                    srcpix += srcstepx;
                    dstpix += dststepx;
                }
                dstrow += dststepy;
                srcrow += srcstepy;
            }
            break;
        case 4:
            for (loopy = 0; loopy < dstheight; ++loopy) {
                dstpix = dstrow;
                srcpix = srcrow;
                for (loopx = 0; loopx < dstwidth; ++loopx) {
                    *(Uint32 *)dstpix = *(Uint32 *)srcpix;
                    srcpix += srcstepx;
                    dstpix += dststepx;
                }
                dstrow += dststepy;
                srcrow += srcstepy;
            }
            break;
    }
    SDL_UnlockSurface(dst);
    Py_END_ALLOW_THREADS;
    return dst;
}

static void
rotate(SDL_Surface *src, SDL_Surface *dst, Uint32 bgcolor, double sangle,
       double cangle)
{
    int x, y, dx, dy;

    Uint8 *srcpix = (Uint8 *)src->pixels;
    Uint8 *dstrow = (Uint8 *)dst->pixels;
    int srcpitch = src->pitch;
    int dstpitch = dst->pitch;

    int cy = dst->h / 2;
    int xd = ((src->w - dst->w) << 15);
    int yd = ((src->h - dst->h) << 15);

    int isin = (int)(sangle * 65536);
    int icos = (int)(cangle * 65536);

    int ax =
        ((dst->w) << 15) - (int)(cangle * (((long long)dst->w - 1) << 15));
    int ay =
        ((dst->h) << 15) - (int)(sangle * (((long long)dst->w - 1) << 15));

    int xmaxval = ((src->w) << 16) - 1;
    int ymaxval = ((src->h) << 16) - 1;

    switch (PG_SURF_BytesPerPixel(src)) {
        case 1:
            for (y = 0; y < dst->h; y++) {
                Uint8 *dstpos = (Uint8 *)dstrow;
                dx = (ax + (isin * (cy - y))) + xd;
                dy = (ay - (icos * (cy - y))) + yd;
                for (x = 0; x < dst->w; x++) {
                    if (dx < 0 || dy < 0 || dx > xmaxval || dy > ymaxval) {
                        *dstpos++ = bgcolor;
                    }
                    else {
                        *dstpos++ =
                            *(Uint8 *)(srcpix + ((dy >> 16) * srcpitch) +
                                       (dx >> 16));
                    }
                    dx += icos;
                    dy += isin;
                }
                dstrow += dstpitch;
            }
            break;
        case 2:
            for (y = 0; y < dst->h; y++) {
                Uint16 *dstpos = (Uint16 *)dstrow;
                dx = (ax + (isin * (cy - y))) + xd;
                dy = (ay - (icos * (cy - y))) + yd;
                for (x = 0; x < dst->w; x++) {
                    if (dx < 0 || dy < 0 || dx > xmaxval || dy > ymaxval) {
                        *dstpos++ = bgcolor;
                    }
                    else {
                        *dstpos++ =
                            *(Uint16 *)(srcpix + ((dy >> 16) * srcpitch) +
                                        ((long long)dx >> 16 << 1));
                    }
                    dx += icos;
                    dy += isin;
                }
                dstrow += dstpitch;
            }
            break;
        case 4:
            for (y = 0; y < dst->h; y++) {
                Uint32 *dstpos = (Uint32 *)dstrow;
                dx = (ax + (isin * (cy - y))) + xd;
                dy = (ay - (icos * (cy - y))) + yd;
                for (x = 0; x < dst->w; x++) {
                    if (dx < 0 || dy < 0 || dx > xmaxval || dy > ymaxval) {
                        *dstpos++ = bgcolor;
                    }
                    else {
                        *dstpos++ =
                            *(Uint32 *)(srcpix + ((dy >> 16) * srcpitch) +
                                        ((long long)dx >> 16 << 2));
                    }
                    dx += icos;
                    dy += isin;
                }
                dstrow += dstpitch;
            }
            break;
        default: /*case 3:*/
            for (y = 0; y < dst->h; y++) {
                Uint8 *dstpos = (Uint8 *)dstrow;
                dx = (ax + (isin * (cy - y))) + xd;
                dy = (ay - (icos * (cy - y))) + yd;
                for (x = 0; x < dst->w; x++) {
                    if (dx < 0 || dy < 0 || dx > xmaxval || dy > ymaxval) {
                        memcpy(dstpos, &bgcolor, 3 * sizeof(Uint8));
                        dstpos += 3;
                    }
                    else {
                        Uint8 *srcpos =
                            (Uint8 *)(srcpix + ((dy >> 16) * srcpitch) +
                                      ((dx >> 16) * 3));
                        memcpy(dstpos, srcpos, 3 * sizeof(Uint8));
                        dstpos += 3;
                    }
                    dx += icos;
                    dy += isin;
                }
                dstrow += dstpitch;
            }
            break;
    }
}

static SDL_Surface *
scale_to(pgSurfaceObject *srcobj, pgSurfaceObject *dstobj, int width,
         int height)
{
    SDL_Surface *src = NULL;
    SDL_Surface *retsurf = NULL;
    SDL_Surface *modsurf = NULL;
    int stretch_result_num = 0;

    if (width < 0 || height < 0) {
        return RAISE(PyExc_ValueError, "Cannot scale to negative size");
    }

    src = pgSurface_AsSurface(srcobj);

    if (!dstobj) {
        modsurf = retsurf = newsurf_fromsurf(src, width, height);
        if (!retsurf) {
            return NULL;
        }
    }
    else {
        modsurf = retsurf = pgSurface_AsSurface(dstobj);

        PG_PixelFormat *ret_format = PG_GetSurfaceFormat(retsurf);
        PG_PixelFormat *src_format = PG_GetSurfaceFormat(src);
        if (ret_format == NULL || src_format == NULL) {
            return RAISE(pgExc_SDLError, SDL_GetError());
        }

        if (PG_FORMAT_BytesPerPixel(ret_format) !=
                PG_FORMAT_BytesPerPixel(src_format) ||
            ret_format->Rmask != src_format->Rmask ||
            ret_format->Gmask != src_format->Gmask ||
            ret_format->Bmask != src_format->Bmask) {
            return RAISE(PyExc_ValueError,
                         "Source and destination surfaces need to be "
                         "compatible formats.");
        }

        /* If the surface formats are otherwise compatible but the alpha is
         * not the same, use a proxy surface to modify the pixels of the
         * existing dstobj return surface. Otherwise SDL_SoftStretch
         * rejects the input.
         * For example, RGBA and RGBX surfaces are compatible in this way. */
        if (ret_format->Amask != src_format->Amask) {
            modsurf = PG_CreateSurfaceFrom(retsurf->w, retsurf->h,
                                           PG_SURF_FORMATENUM(src),
                                           retsurf->pixels, retsurf->pitch);
        }
    }

    if (retsurf->w != width || retsurf->h != height) {
        return (SDL_Surface *)(RAISE(
            PyExc_ValueError,
            "Destination surface not the given width or height."));
    }

    /* Testing width and height of src and dest, because pygame supports
     * surfaces with "0" as the width or height, and for those nothing should
     * happen here. */
    if ((width && height) && (src->w && src->h)) {
        pgSurface_Lock(srcobj);
        Py_BEGIN_ALLOW_THREADS;

        stretch_result_num = PG_SoftStretchNearest(src, NULL, modsurf, NULL);

        Py_END_ALLOW_THREADS;
        pgSurface_Unlock(srcobj);

        if (modsurf != retsurf) {
            SDL_FreeSurface(modsurf);
        }

        if (stretch_result_num < 0) {
            return (SDL_Surface *)(RAISE(pgExc_SDLError, SDL_GetError()));
        }
    }

    return retsurf;
}

static PyObject *
surf_scale(PyObject *self, PyObject *args, PyObject *kwargs)
{
    pgSurfaceObject *surfobj;
    pgSurfaceObject *surfobj2 = NULL;
    PyObject *size;
    SDL_Surface *newsurf, *surf;
    int width, height;
    static char *keywords[] = {"surface", "size", "dest_surface", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O|O!", keywords,
                                     &pgSurface_Type, &surfobj, &size,
                                     &pgSurface_Type, &surfobj2)) {
        return NULL;
    }

    surf = pgSurface_AsSurface(surfobj);
    SURF_INIT_CHECK(surf)

    if (!pg_TwoIntsFromObj(size, &width, &height)) {
        return RAISE(PyExc_TypeError, "size must be two numbers");
    }

    newsurf = scale_to(surfobj, surfobj2, width, height);
    if (!newsurf) {
        return NULL;
    }

    if (surfobj2) {
        Py_INCREF(surfobj2);
        return (PyObject *)surfobj2;
    }
    else {
        return (PyObject *)pgSurface_New(newsurf);
    }
}

static PyObject *
surf_scale_by(PyObject *self, PyObject *args, PyObject *kwargs)
{
    pgSurfaceObject *surfobj;
    pgSurfaceObject *surfobj2 = NULL;
    PyObject *factorobj = NULL;
    float scalex, scaley;
    SDL_Surface *surf, *newsurf;
    static char *keywords[] = {"surface", "factor", "dest_surface", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O|O!", keywords,
                                     &pgSurface_Type, &surfobj, &factorobj,
                                     &pgSurface_Type, &surfobj2)) {
        return NULL;
    }

    if (!_get_factor(factorobj, &scalex, &scaley)) {
        return NULL;
    }

    surf = pgSurface_AsSurface(surfobj);
    SURF_INIT_CHECK(surf)

    newsurf = scale_to(surfobj, surfobj2, (int)(surf->w * scalex),
                       (int)(surf->h * scaley));
    if (!newsurf) {
        return NULL;
    }

    if (surfobj2) {
        Py_INCREF(surfobj2);
        return (PyObject *)surfobj2;
    }
    else {
        return (PyObject *)pgSurface_New(newsurf);
    }
}

static PyObject *
surf_scale2x(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *surfobj, *surfobj2 = NULL;
    SDL_Surface *surf;
    SDL_Surface *newsurf;
    static char *keywords[] = {"surface", "dest_surface", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|O!", keywords,
                                     &pgSurface_Type, &surfobj,
                                     &pgSurface_Type, &surfobj2)) {
        return NULL;
    }

    surf = pgSurface_AsSurface(surfobj);
    SURF_INIT_CHECK(surf)

    /* if the second surface is not there, then make a new one. */

    if (!surfobj2) {
        int width = surf->w * 2;
        int height = surf->h * 2;

        newsurf = newsurf_fromsurf(surf, width, height);

        if (!newsurf) {
            return NULL;
        }
    }
    else {
        newsurf = pgSurface_AsSurface(surfobj2);
    }

    /* check to see if the size is twice as big. */
    if (newsurf->w != (surf->w * 2) || newsurf->h != (surf->h * 2)) {
        return RAISE(PyExc_ValueError, "Destination surface not 2x bigger.");
    }

    /* check to see if the format of the surface is the same. */
    if (PG_SURF_BytesPerPixel(surf) != PG_SURF_BytesPerPixel(newsurf)) {
        return RAISE(PyExc_ValueError,
                     "Source and destination surfaces need the same format.");
    }

    SDL_LockSurface(newsurf);
    SDL_LockSurface(surf);

    Py_BEGIN_ALLOW_THREADS;
    scale2x(surf, newsurf);
    Py_END_ALLOW_THREADS;

    SDL_UnlockSurface(surf);
    SDL_UnlockSurface(newsurf);

    if (surfobj2) {
        Py_INCREF(surfobj2);
        return surfobj2;
    }
    else {
        return (PyObject *)pgSurface_New(newsurf);
    }
}

static PyObject *
surf_rotate(PyObject *self, PyObject *args, PyObject *kwargs)
{
    pgSurfaceObject *surfobj;
    SDL_Surface *surf, *newsurf;
    float angle;

    double radangle, sangle, cangle;
    double x, y, cx, cy, sx, sy;
    int nxmax, nymax;
    Uint32 bgcolor;
    static char *keywords[] = {"surface", "angle", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!f", keywords,
                                     &pgSurface_Type, &surfobj, &angle)) {
        return NULL;
    }
    surf = pgSurface_AsSurface(surfobj);
    SURF_INIT_CHECK(surf)

    if (surf->w < 1 || surf->h < 1) {
        Py_INCREF(surfobj);
        return (PyObject *)surfobj;
    }

    if (PG_SURF_BytesPerPixel(surf) == 0 || PG_SURF_BytesPerPixel(surf) > 4) {
        return RAISE(PyExc_ValueError,
                     "unsupported Surface bit depth for transform");
    }

    if (!(fmod((double)angle, (double)90.0f))) {
        pgSurface_Lock(surfobj);

        /* The function releases GIL internally, don't release here */
        newsurf = rotate90(surf, (int)angle);

        pgSurface_Unlock(surfobj);
        if (!newsurf) {
            return NULL;
        }
        return (PyObject *)pgSurface_New(newsurf);
    }

    radangle = angle * .01745329251994329;
    sangle = sin(radangle);
    cangle = cos(radangle);

    x = surf->w;
    y = surf->h;
    cx = cangle * x;
    cy = cangle * y;
    sx = sangle * x;
    sy = sangle * y;
    nxmax = (int)(MAX(MAX(MAX(fabs(cx + sy), fabs(cx - sy)), fabs(-cx + sy)),
                      fabs(-cx - sy)));
    nymax = (int)(MAX(MAX(MAX(fabs(sx + cy), fabs(sx - cy)), fabs(-sx + cy)),
                      fabs(-sx - cy)));

    newsurf = newsurf_fromsurf(surf, nxmax, nymax);
    if (!newsurf) {
        return NULL;
    }

    /* get the background color */
    if (!SDL_HasColorKey(surf)) {
        SDL_LockSurface(surf);
        switch (PG_SURF_BytesPerPixel(surf)) {
            case 1:
                bgcolor = *(Uint8 *)surf->pixels;
                break;
            case 2:
                bgcolor = *(Uint16 *)surf->pixels;
                break;
            case 4:
                bgcolor = *(Uint32 *)surf->pixels;
                break;
            default: /*case 3:*/
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                bgcolor = (((Uint8 *)surf->pixels)[0]) +
                          (((Uint8 *)surf->pixels)[1] << 8) +
                          (((Uint8 *)surf->pixels)[2] << 16);
#else
                bgcolor = (((Uint8 *)surf->pixels)[2]) +
                          (((Uint8 *)surf->pixels)[1] << 8) +
                          (((Uint8 *)surf->pixels)[0] << 16);
#endif
        }
        SDL_UnlockSurface(surf);
        PG_PixelFormat *surf_format = PG_GetSurfaceFormat(surf);
        if (surf_format == NULL) {
            return RAISE(pgExc_SDLError, SDL_GetError());
        }
        bgcolor &= ~surf_format->Amask;
    }
    else {
        SDL_GetColorKey(surf, &bgcolor);
    }

    SDL_LockSurface(newsurf);
    pgSurface_Lock(surfobj);

    Py_BEGIN_ALLOW_THREADS;
    rotate(surf, newsurf, bgcolor, sangle, cangle);
    Py_END_ALLOW_THREADS;

    pgSurface_Unlock(surfobj);
    SDL_UnlockSurface(newsurf);

    return (PyObject *)pgSurface_New(newsurf);
}

static PyObject *
surf_flip(PyObject *self, PyObject *args, PyObject *kwargs)
{
    pgSurfaceObject *surfobj;
    SDL_Surface *surf, *newsurf;
    int xaxis, yaxis;
    int loopx, loopy;
    int srcpitch, dstpitch;
    Uint8 *srcpix, *dstpix;
    static char *keywords[] = {"surface", "flip_x", "flip_y", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!ii", keywords,
                                     &pgSurface_Type, &surfobj, &xaxis,
                                     &yaxis)) {
        return NULL;
    }
    surf = pgSurface_AsSurface(surfobj);
    SURF_INIT_CHECK(surf)

    newsurf = newsurf_fromsurf(surf, surf->w, surf->h);
    if (!newsurf) {
        return NULL;
    }

    srcpitch = surf->pitch;
    dstpitch = newsurf->pitch;

    SDL_LockSurface(newsurf);
    pgSurface_Lock(surfobj);

    srcpix = (Uint8 *)surf->pixels;
    dstpix = (Uint8 *)newsurf->pixels;

    Py_BEGIN_ALLOW_THREADS;

    if (!xaxis) {
        if (!yaxis) {
            for (loopy = 0; loopy < surf->h; ++loopy) {
                memcpy(dstpix + loopy * dstpitch, srcpix + loopy * srcpitch,
                       surf->w * PG_SURF_BytesPerPixel(surf));
            }
        }
        else {
            for (loopy = 0; loopy < surf->h; ++loopy) {
                memcpy(dstpix + loopy * dstpitch,
                       srcpix + (surf->h - 1 - loopy) * srcpitch,
                       surf->w * PG_SURF_BytesPerPixel(surf));
            }
        }
    }
    else /*if (xaxis)*/
    {
        if (yaxis) {
            switch (PG_SURF_BytesPerPixel(surf)) {
                case 1:
                    for (loopy = 0; loopy < surf->h; ++loopy) {
                        Uint8 *dst = (Uint8 *)(dstpix + loopy * dstpitch);
                        Uint8 *src =
                            ((Uint8 *)(srcpix +
                                       (surf->h - 1 - loopy) * srcpitch)) +
                            surf->w - 1;
                        for (loopx = 0; loopx < surf->w; ++loopx) {
                            *dst++ = *src--;
                        }
                    }
                    break;
                case 2:
                    for (loopy = 0; loopy < surf->h; ++loopy) {
                        Uint16 *dst = (Uint16 *)(dstpix + loopy * dstpitch);
                        Uint16 *src =
                            ((Uint16 *)(srcpix +
                                        (surf->h - 1 - loopy) * srcpitch)) +
                            surf->w - 1;
                        for (loopx = 0; loopx < surf->w; ++loopx) {
                            *dst++ = *src--;
                        }
                    }
                    break;
                case 4:
                    for (loopy = 0; loopy < surf->h; ++loopy) {
                        Uint32 *dst = (Uint32 *)(dstpix + loopy * dstpitch);
                        Uint32 *src =
                            ((Uint32 *)(srcpix +
                                        (surf->h - 1 - loopy) * srcpitch)) +
                            surf->w - 1;
                        for (loopx = 0; loopx < surf->w; ++loopx) {
                            *dst++ = *src--;
                        }
                    }
                    break;
                case 3:
                    for (loopy = 0; loopy < surf->h; ++loopy) {
                        Uint8 *dst = (Uint8 *)(dstpix + loopy * dstpitch);
                        Uint8 *src =
                            ((Uint8 *)(srcpix +
                                       (surf->h - 1 - loopy) * srcpitch)) +
                            surf->w * 3 - 3;
                        for (loopx = 0; loopx < surf->w; ++loopx) {
                            memcpy(dst, src, 3);
                            dst += 3;
                            src -= 3;
                        }
                    }
                    break;
            }
        }
        else {
            switch (PG_SURF_BytesPerPixel(surf)) {
                case 1:
                    for (loopy = 0; loopy < surf->h; ++loopy) {
                        Uint8 *dst = (Uint8 *)(dstpix + loopy * dstpitch);
                        Uint8 *src = ((Uint8 *)(srcpix + loopy * srcpitch)) +
                                     surf->w - 1;
                        for (loopx = 0; loopx < surf->w; ++loopx) {
                            *dst++ = *src--;
                        }
                    }
                    break;
                case 2:
                    for (loopy = 0; loopy < surf->h; ++loopy) {
                        Uint16 *dst = (Uint16 *)(dstpix + loopy * dstpitch);
                        Uint16 *src = ((Uint16 *)(srcpix + loopy * srcpitch)) +
                                      surf->w - 1;
                        for (loopx = 0; loopx < surf->w; ++loopx) {
                            *dst++ = *src--;
                        }
                    }
                    break;
                case 4:
                    for (loopy = 0; loopy < surf->h; ++loopy) {
                        Uint32 *dst = (Uint32 *)(dstpix + loopy * dstpitch);
                        Uint32 *src = ((Uint32 *)(srcpix + loopy * srcpitch)) +
                                      surf->w - 1;
                        for (loopx = 0; loopx < surf->w; ++loopx) {
                            *dst++ = *src--;
                        }
                    }
                    break;
                case 3:
                    for (loopy = 0; loopy < surf->h; ++loopy) {
                        Uint8 *dst = (Uint8 *)(dstpix + loopy * dstpitch);
                        Uint8 *src = ((Uint8 *)(srcpix + loopy * srcpitch)) +
                                     surf->w * 3 - 3;
                        for (loopx = 0; loopx < surf->w; ++loopx) {
                            memcpy(dst, src, 3);
                            dst += 3;
                            src -= 3;
                        }
                    }
                    break;
            }
        }
    }
    Py_END_ALLOW_THREADS;

    pgSurface_Unlock(surfobj);
    SDL_UnlockSurface(newsurf);
    return (PyObject *)pgSurface_New(newsurf);
}

static PyObject *
surf_rotozoom(PyObject *self, PyObject *args, PyObject *kwargs)
{
    pgSurfaceObject *surfobj;
    SDL_Surface *surf, *newsurf, *surf32;
    float scale, angle;
    static char *keywords[] = {"surface", "angle", "scale", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!ff", keywords,
                                     &pgSurface_Type, &surfobj, &angle,
                                     &scale)) {
        return NULL;
    }
    surf = pgSurface_AsSurface(surfobj);
    SURF_INIT_CHECK(surf)

    if (scale == 0.0 || surf->w == 0 || surf->h == 0) {
        newsurf = newsurf_fromsurf(surf, 0, 0);
        return (PyObject *)pgSurface_New(newsurf);
    }

    if (PG_SURF_BitsPerPixel(surf) == 32) {
        surf32 = surf;
        pgSurface_Lock(surfobj);
    }
    else {
        Py_BEGIN_ALLOW_THREADS;
        surf32 = PG_CreateSurface(surf->w, surf->h, SDL_PIXELFORMAT_ABGR8888);
        SDL_BlitSurface(surf, NULL, surf32, NULL);
        Py_END_ALLOW_THREADS;
    }

    Py_BEGIN_ALLOW_THREADS;
    newsurf = rotozoomSurface(surf32, angle, scale, 1);
    Py_END_ALLOW_THREADS;
    if (newsurf == NULL) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return NULL;
    }

    if (surf32 == surf) {
        pgSurface_Unlock(surfobj);
    }
    else {
        SDL_FreeSurface(surf32);
    }
    return (PyObject *)pgSurface_New(newsurf);
}

static SDL_Surface *
chop(SDL_Surface *src, int x, int y, int width, int height)
{
    SDL_Surface *dst;
    int dstwidth, dstheight;
    char *srcpix, *dstpix, *srcrow, *dstrow;
    int srcstepx, srcstepy, dststepx, dststepy;
    int loopx, loopy;

    if ((x + width) > src->w) {
        width = MIN(MAX(src->w - x, 0), src->w);
    }
    if ((y + height) > src->h) {
        height = MIN(MAX(src->h - y, 0), src->h);
    }
    if (x < 0) {
        width -= (-x);
        x = 0;
    }
    if (y < 0) {
        height -= (-y);
        y = 0;
    }

    dstwidth = src->w - width;
    dstheight = src->h - height;

    dst = newsurf_fromsurf(src, dstwidth, dstheight);
    if (!dst) {
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS;
    SDL_LockSurface(dst);
    srcrow = (char *)src->pixels;
    dstrow = (char *)dst->pixels;
    srcstepx = dststepx = PG_SURF_BytesPerPixel(src);
    srcstepy = src->pitch;
    dststepy = dst->pitch;

    for (loopy = 0; loopy < src->h; loopy++) {
        if ((loopy < y) || (loopy >= (y + height))) {
            dstpix = dstrow;
            srcpix = srcrow;
            for (loopx = 0; loopx < src->w; loopx++) {
                if ((loopx < x) || (loopx >= (x + width))) {
                    switch (PG_SURF_BytesPerPixel(src)) {
                        case 1:
                            *dstpix = *srcpix;
                            break;
                        case 2:
                            *(Uint16 *)dstpix = *(Uint16 *)srcpix;
                            break;
                        case 3:
                            memcpy(dstpix, srcpix, 3);
                            break;
                        case 4:
                            *(Uint32 *)dstpix = *(Uint32 *)srcpix;
                            break;
                    }
                    dstpix += dststepx;
                }
                srcpix += srcstepx;
            }
            dstrow += dststepy;
        }
        srcrow += srcstepy;
    }
    SDL_UnlockSurface(dst);
    Py_END_ALLOW_THREADS;

    return dst;
}

static PyObject *
surf_chop(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *surfobj, *rectobj;
    SDL_Surface *surf, *newsurf;
    SDL_Rect *rect, temp;
    static char *keywords[] = {"surface", "rect", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O", keywords,
                                     &pgSurface_Type, &surfobj, &rectobj)) {
        return NULL;
    }

    if (!(rect = pgRect_FromObject(rectobj, &temp))) {
        return RAISE(PyExc_TypeError, "Rect argument is invalid");
    }

    surf = pgSurface_AsSurface(surfobj);
    SURF_INIT_CHECK(surf)

    /* The function releases GIL internally, don't release here */
    newsurf = chop(surf, rect->x, rect->y, rect->w, rect->h);

    return (PyObject *)pgSurface_New(newsurf);
}

/*
 * smooth scale functions.
 */

/* this function implements an area-averaging shrinking filter in the
 * X-dimension */
static void
filter_shrink_X_ONLYC(Uint8 *srcpix, Uint8 *dstpix, int height, int srcpitch,
                      int dstpitch, int srcwidth, int dstwidth)
{
    int srcdiff = srcpitch - (srcwidth * 4);
    int dstdiff = dstpitch - (dstwidth * 4);
    int x, y;

    int xspace = 0x10000 * srcwidth / dstwidth; /* must be > 1 */
    int xrecip = (int)(0x100000000LL / xspace);
    for (y = 0; y < height; y++) {
        Uint16 accumulate[4] = {0, 0, 0, 0};
        int xcounter = xspace;
        for (x = 0; x < srcwidth; x++) {
            if (xcounter > 0x10000) {
                accumulate[0] += (Uint16)*srcpix++;
                accumulate[1] += (Uint16)*srcpix++;
                accumulate[2] += (Uint16)*srcpix++;
                accumulate[3] += (Uint16)*srcpix++;
                xcounter -= 0x10000;
            }
            else {
                int xfrac = 0x10000 - xcounter;
                /* write out a destination pixel */
                *dstpix++ =
                    (Uint8)(((accumulate[0] + ((srcpix[0] * xcounter) >> 16)) *
                             xrecip) >>
                            16);
                *dstpix++ =
                    (Uint8)(((accumulate[1] + ((srcpix[1] * xcounter) >> 16)) *
                             xrecip) >>
                            16);
                *dstpix++ =
                    (Uint8)(((accumulate[2] + ((srcpix[2] * xcounter) >> 16)) *
                             xrecip) >>
                            16);
                *dstpix++ =
                    (Uint8)(((accumulate[3] + ((srcpix[3] * xcounter) >> 16)) *
                             xrecip) >>
                            16);
                /* reload the accumulator with the remainder of this pixel */
                accumulate[0] = (Uint16)((*srcpix++ * xfrac) >> 16);
                accumulate[1] = (Uint16)((*srcpix++ * xfrac) >> 16);
                accumulate[2] = (Uint16)((*srcpix++ * xfrac) >> 16);
                accumulate[3] = (Uint16)((*srcpix++ * xfrac) >> 16);
                xcounter = xspace - xfrac;
            }
        }
        srcpix += srcdiff;
        dstpix += dstdiff;
    }
}

/* this function implements an area-averaging shrinking filter in the
 * Y-dimension */
static void
filter_shrink_Y_ONLYC(Uint8 *srcpix, Uint8 *dstpix, int width, int srcpitch,
                      int dstpitch, int srcheight, int dstheight)
{
    Uint16 *templine;
    int srcdiff = srcpitch - (width * 4);
    int dstdiff = dstpitch - (width * 4);
    int x, y;
    int yspace = 0x10000 * srcheight / dstheight; /* must be > 1 */
    int yrecip = (int)(0x100000000LL / yspace);
    int ycounter = yspace;

    /* allocate and clear a memory area for storing the accumulator line */
    templine = (Uint16 *)malloc(dstpitch * 2);
    if (templine == NULL) {
        return;
    }
    memset(templine, 0, dstpitch * 2);

    for (y = 0; y < srcheight; y++) {
        Uint16 *accumulate = templine;
        if (ycounter > 0x10000) {
            for (x = 0; x < width; x++) {
                *accumulate++ += (Uint16)*srcpix++;
                *accumulate++ += (Uint16)*srcpix++;
                *accumulate++ += (Uint16)*srcpix++;
                *accumulate++ += (Uint16)*srcpix++;
            }
            ycounter -= 0x10000;
        }
        else {
            int yfrac = 0x10000 - ycounter;
            /* write out a destination line */
            for (x = 0; x < width; x++) {
                *dstpix++ =
                    (Uint8)(((*accumulate++ + ((*srcpix++ * ycounter) >> 16)) *
                             yrecip) >>
                            16);
                *dstpix++ =
                    (Uint8)(((*accumulate++ + ((*srcpix++ * ycounter) >> 16)) *
                             yrecip) >>
                            16);
                *dstpix++ =
                    (Uint8)(((*accumulate++ + ((*srcpix++ * ycounter) >> 16)) *
                             yrecip) >>
                            16);
                *dstpix++ =
                    (Uint8)(((*accumulate++ + ((*srcpix++ * ycounter) >> 16)) *
                             yrecip) >>
                            16);
            }
            dstpix += dstdiff;
            /* reload the accumulator with the remainder of this line */
            accumulate = templine;
            srcpix -= 4 * width;
            for (x = 0; x < width; x++) {
                *accumulate++ = (Uint16)((*srcpix++ * yfrac) >> 16);
                *accumulate++ = (Uint16)((*srcpix++ * yfrac) >> 16);
                *accumulate++ = (Uint16)((*srcpix++ * yfrac) >> 16);
                *accumulate++ = (Uint16)((*srcpix++ * yfrac) >> 16);
            }
            ycounter = yspace - yfrac;
        }
        srcpix += srcdiff;
    } /* for (int y = 0; y < srcheight; y++) */

    /* free the temporary memory */
    free(templine);
}

/* this function implements a bilinear filter in the X-dimension */
static void
filter_expand_X_ONLYC(Uint8 *srcpix, Uint8 *dstpix, int height, int srcpitch,
                      int dstpitch, int srcwidth, int dstwidth)
{
    int dstdiff = dstpitch - (dstwidth * 4);
    int *xidx0, *xmult0, *xmult1;
    int x, y;
    const int factorwidth = 4;

#ifdef _MSC_VER
    /* Make MSVC static analyzer happy by assuring dstwidth >= 2 to suppress
     * a false analyzer report */
    __analysis_assume(dstwidth >= 2);
#endif

    /* Allocate memory for factors */
    xidx0 = malloc(dstwidth * 4);
    if (xidx0 == NULL) {
        return;
    }
    xmult0 = (int *)malloc(dstwidth * factorwidth);
    xmult1 = (int *)malloc(dstwidth * factorwidth);
    if (xmult0 == NULL || xmult1 == NULL) {
        free(xidx0);
        if (xmult0) {
            free(xmult0);
        }
        if (xmult1) {
            free(xmult1);
        }

        return;
    }

    /* Create multiplier factors and starting indices and put them in arrays */
    for (x = 0; x < dstwidth; x++) {
        xidx0[x] = x * (srcwidth - 1) / dstwidth;
        xmult1[x] = 0x10000 * ((x * (srcwidth - 1)) % dstwidth) / dstwidth;
        xmult0[x] = 0x10000 - xmult1[x];
    }

    /* Do the scaling in raster order so we don't trash the cache */
    for (y = 0; y < height; y++) {
        Uint8 *srcrow0 = srcpix + y * srcpitch;
        for (x = 0; x < dstwidth; x++) {
            Uint8 *src = srcrow0 + xidx0[x] * 4;
            int xm0 = xmult0[x];
            int xm1 = xmult1[x];
            *dstpix++ = (Uint8)(((src[0] * xm0) + (src[4] * xm1)) >> 16);
            *dstpix++ = (Uint8)(((src[1] * xm0) + (src[5] * xm1)) >> 16);
            *dstpix++ = (Uint8)(((src[2] * xm0) + (src[6] * xm1)) >> 16);
            *dstpix++ = (Uint8)(((src[3] * xm0) + (src[7] * xm1)) >> 16);
        }
        dstpix += dstdiff;
    }

    /* free memory */
    free(xidx0);
    free(xmult0);
    free(xmult1);
}

/* this function implements a bilinear filter in the Y-dimension */
static void
filter_expand_Y_ONLYC(Uint8 *srcpix, Uint8 *dstpix, int width, int srcpitch,
                      int dstpitch, int srcheight, int dstheight)
{
    int x, y;

    for (y = 0; y < dstheight; y++) {
        int yidx0 = y * (srcheight - 1) / dstheight;
        Uint8 *srcrow0 = srcpix + yidx0 * srcpitch;
        Uint8 *srcrow1 = srcrow0 + srcpitch;
        int ymult1 = 0x10000 * ((y * (srcheight - 1)) % dstheight) / dstheight;
        int ymult0 = 0x10000 - ymult1;
        for (x = 0; x < width; x++) {
            *dstpix++ =
                (Uint8)(((*srcrow0++ * ymult0) + (*srcrow1++ * ymult1)) >> 16);
            *dstpix++ =
                (Uint8)(((*srcrow0++ * ymult0) + (*srcrow1++ * ymult1)) >> 16);
            *dstpix++ =
                (Uint8)(((*srcrow0++ * ymult0) + (*srcrow1++ * ymult1)) >> 16);
            *dstpix++ =
                (Uint8)(((*srcrow0++ * ymult0) + (*srcrow1++ * ymult1)) >> 16);
        }
    }
}

static void
smoothscale_init(struct _module_state *st)
{
    if (st->filter_shrink_X) {
        return;
    }

#if !defined(__EMSCRIPTEN__)
#if PG_ENABLE_SSE_NEON
    if (SDL_HasSSE2()) {
        st->filter_type = "SSE2";
        st->filter_shrink_X = filter_shrink_X_SSE2;
        st->filter_shrink_Y = filter_shrink_Y_SSE2;
        st->filter_expand_X = filter_expand_X_SSE2;
        st->filter_expand_Y = filter_expand_Y_SSE2;
        return;
    }
    if (SDL_HasNEON()) {
        st->filter_type = "NEON";
        st->filter_shrink_X = filter_shrink_X_SSE2;
        st->filter_shrink_Y = filter_shrink_Y_SSE2;
        st->filter_expand_X = filter_expand_X_SSE2;
        st->filter_expand_Y = filter_expand_Y_SSE2;
        return;
    }
#endif /* PG_ENABLE_SSE_NEON */
#endif /* !__EMSCRIPTEN__ */
#ifdef SCALE_MMX_SUPPORT
    if (SDL_HasSSE()) {
        st->filter_type = "SSE";
        st->filter_shrink_X = filter_shrink_X_SSE;
        st->filter_shrink_Y = filter_shrink_Y_SSE;
        st->filter_expand_X = filter_expand_X_SSE;
        st->filter_expand_Y = filter_expand_Y_SSE;
        return;
    }
    if (SDL_HasMMX()) {
        st->filter_type = "MMX";
        st->filter_shrink_X = filter_shrink_X_MMX;
        st->filter_shrink_Y = filter_shrink_Y_MMX;
        st->filter_expand_X = filter_expand_X_MMX;
        st->filter_expand_Y = filter_expand_Y_MMX;
        return;
    }
#endif /* ~SCALE_MMX_SUPPORT */

    /* If no accelerated options were selected, falls through to generic */
    st->filter_type = "GENERIC";
    st->filter_shrink_X = filter_shrink_X_ONLYC;
    st->filter_shrink_Y = filter_shrink_Y_ONLYC;
    st->filter_expand_X = filter_expand_X_ONLYC;
    st->filter_expand_Y = filter_expand_Y_ONLYC;
}

static void
convert_24_32(Uint8 *srcpix, int srcpitch, Uint8 *dstpix, int dstpitch,
              int width, int height)
{
    int srcdiff = srcpitch - (width * 3);
    int dstdiff = dstpitch - (width * 4);
    int x, y;

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            *dstpix++ = *srcpix++;
            *dstpix++ = *srcpix++;
            *dstpix++ = *srcpix++;
            *dstpix++ = 0xff;
        }
        srcpix += srcdiff;
        dstpix += dstdiff;
    }
}

static void
convert_32_24(Uint8 *srcpix, int srcpitch, Uint8 *dstpix, int dstpitch,
              int width, int height)
{
    int srcdiff = srcpitch - (width * 4);
    int dstdiff = dstpitch - (width * 3);
    int x, y;

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            *dstpix++ = *srcpix++;
            *dstpix++ = *srcpix++;
            *dstpix++ = *srcpix++;
            srcpix++;
        }
        srcpix += srcdiff;
        dstpix += dstdiff;
    }
}

static void
scalesmooth(SDL_Surface *src, SDL_Surface *dst, struct _module_state *st)
{
    Uint8 *srcpix = (Uint8 *)src->pixels;
    Uint8 *dstpix = (Uint8 *)dst->pixels;
    Uint8 *dst32 = NULL;
    int srcpitch = src->pitch;
    int dstpitch = dst->pitch;

    int srcwidth = src->w;
    int srcheight = src->h;
    int dstwidth = dst->w;
    int dstheight = dst->h;

    int bpp = PG_SURF_BytesPerPixel(src);

    Uint8 *temppix = NULL;
    int tempwidth = 0, temppitch = 0;

    /* convert to 32-bit if necessary */
    if (bpp == 3) {
        int newpitch = srcwidth * 4;
        Uint8 *newsrc = (Uint8 *)malloc((size_t)newpitch * srcheight);
        if (!newsrc) {
            return;
        }
        convert_24_32(srcpix, srcpitch, newsrc, newpitch, srcwidth, srcheight);
        srcpix = newsrc;
        srcpitch = newpitch;
        /* create a destination buffer for the 32-bit result */
        dstpitch = dstwidth << 2;
        dst32 = (Uint8 *)malloc((size_t)dstpitch * dstheight);
        if (dst32 == NULL) {
            free(srcpix);
            return;
        }
        dstpix = dst32;
    }

    /* Create a temporary processing buffer if we will be scaling both X and Y
     */
    if (srcwidth != dstwidth && srcheight != dstheight) {
        tempwidth = dstwidth;
        temppitch = tempwidth << 2;
        temppix = (Uint8 *)malloc((size_t)temppitch * srcheight);
        if (temppix == NULL) {
            if (bpp == 3) {
                free(srcpix);
                free(dstpix);
            }
            return;
        }
    }

    /* Start the filter by doing X-scaling */
    if (dstwidth < srcwidth) /* shrink */
    {
        if (srcheight != dstheight) {
            st->filter_shrink_X(srcpix, temppix, srcheight, srcpitch,
                                temppitch, srcwidth, dstwidth);
        }
        else {
            st->filter_shrink_X(srcpix, dstpix, srcheight, srcpitch, dstpitch,
                                srcwidth, dstwidth);
        }
    }
    else if (dstwidth > srcwidth) /* expand */
    {
        if (srcheight != dstheight) {
            st->filter_expand_X(srcpix, temppix, srcheight, srcpitch,
                                temppitch, srcwidth, dstwidth);
        }
        else {
            st->filter_expand_X(srcpix, dstpix, srcheight, srcpitch, dstpitch,
                                srcwidth, dstwidth);
        }
    }
    /* Now do the Y scale */
    if (dstheight < srcheight) /* shrink */
    {
        if (srcwidth != dstwidth) {
            st->filter_shrink_Y(temppix, dstpix, tempwidth, temppitch,
                                dstpitch, srcheight, dstheight);
        }
        else {
            st->filter_shrink_Y(srcpix, dstpix, srcwidth, srcpitch, dstpitch,
                                srcheight, dstheight);
        }
    }
    else if (dstheight > srcheight) /* expand */
    {
        if (srcwidth != dstwidth) {
            st->filter_expand_Y(temppix, dstpix, tempwidth, temppitch,
                                dstpitch, srcheight, dstheight);
        }
        else {
            st->filter_expand_Y(srcpix, dstpix, srcwidth, srcpitch, dstpitch,
                                srcheight, dstheight);
        }
    }

    /* Convert back to 24-bit if necessary */
    if (bpp == 3) {
        convert_32_24(dst32, dstpitch, (Uint8 *)dst->pixels, dst->pitch,
                      dstwidth, dstheight);
        free(dst32);
        dst32 = NULL;
        free(srcpix);
        srcpix = NULL;
    }
    /* free temporary buffer if necessary */
    if (temppix != NULL) {
        free(temppix);
    }
}

static SDL_Surface *
smoothscale_to(PyObject *self, pgSurfaceObject *srcobj,
               pgSurfaceObject *dstobj, int width, int height)
{
    SDL_Surface *src = NULL;
    SDL_Surface *retsurf = NULL;
    int bpp;
    if (width < 0 || height < 0) {
        return (SDL_Surface *)(RAISE(PyExc_ValueError,
                                     "Cannot scale to negative size"));
    }

    src = pgSurface_AsSurface(srcobj);

    bpp = PG_SURF_BytesPerPixel(src);
    if (bpp < 3 || bpp > 4) {
        return (SDL_Surface *)(RAISE(
            PyExc_ValueError,
            "Only 24-bit or 32-bit surfaces can be smoothly scaled"));
    }

    if (!dstobj) {
        retsurf = newsurf_fromsurf(src, width, height);
        if (!retsurf) {
            return NULL;
        }
    }
    else {
        retsurf = pgSurface_AsSurface(dstobj);
    }

    if (retsurf->w != width || retsurf->h != height) {
        return (SDL_Surface *)(RAISE(
            PyExc_ValueError,
            "Destination surface not the given width or height."));
    }

    if (((width * bpp + 3) >> 2) > retsurf->pitch) {
        return (SDL_Surface *)(RAISE(
            PyExc_ValueError,
            "SDL Error: destination surface pitch not 4-byte aligned."));
    }

    if (width && height) {
        SDL_LockSurface(retsurf);
        pgSurface_Lock(srcobj);

        /* handle trivial case */
        if (src->w == width && src->h == height) {
            int y;
            Py_BEGIN_ALLOW_THREADS;
            for (y = 0; y < height; y++) {
                memcpy((Uint8 *)retsurf->pixels + y * retsurf->pitch,
                       (Uint8 *)src->pixels + y * src->pitch, width * bpp);
            }
            Py_END_ALLOW_THREADS;
        }
        else {
            struct _module_state *st = GETSTATE(self);
            Py_BEGIN_ALLOW_THREADS;
            scalesmooth(src, retsurf, st);
            Py_END_ALLOW_THREADS;
        }

        pgSurface_Unlock(srcobj);
        SDL_UnlockSurface(retsurf);
    }

    return retsurf;
}

static PyObject *
surf_scalesmooth(PyObject *self, PyObject *args, PyObject *kwargs)
{
    pgSurfaceObject *surfobj;
    pgSurfaceObject *surfobj2 = NULL;
    SDL_Surface *surf;
    PyObject *size;
    SDL_Surface *newsurf;
    int width, height;
    static char *keywords[] = {"surface", "size", "dest_surface", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O|O!", keywords,
                                     &pgSurface_Type, &surfobj, &size,
                                     &pgSurface_Type, &surfobj2)) {
        return NULL;
    }

    surf = pgSurface_AsSurface(surfobj);
    SURF_INIT_CHECK(surf)

    if (!pg_TwoIntsFromObj(size, &width, &height)) {
        return RAISE(PyExc_TypeError, "size must be two numbers");
    }

    newsurf = smoothscale_to(self, surfobj, surfobj2, width, height);
    if (!newsurf) {
        return NULL;
    }

    if (surfobj2) {
        Py_INCREF(surfobj2);
        return (PyObject *)surfobj2;
    }
    else {
        return (PyObject *)pgSurface_New(newsurf);
    }
}

static PyObject *
surf_scalesmooth_by(PyObject *self, PyObject *args, PyObject *kwargs)
{
    pgSurfaceObject *surfobj;
    pgSurfaceObject *surfobj2 = NULL;
    PyObject *factorobj = NULL;
    float scale, scaley;
    SDL_Surface *surf, *newsurf;
    static char *keywords[] = {"surface", "factor", "dest_surface", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O|O!", keywords,
                                     &pgSurface_Type, &surfobj, &factorobj,
                                     &pgSurface_Type, &surfobj2)) {
        return NULL;
    }

    if (!_get_factor(factorobj, &scale, &scaley)) {
        return NULL;
    }

    surf = pgSurface_AsSurface(surfobj);
    SURF_INIT_CHECK(surf)

    newsurf = smoothscale_to(self, surfobj, surfobj2, (int)(surf->w * scale),
                             (int)(surf->h * scaley));
    if (!newsurf) {
        return NULL;
    }

    if (surfobj2) {
        Py_INCREF(surfobj2);
        return (PyObject *)surfobj2;
    }
    else {
        return (PyObject *)pgSurface_New(newsurf);
    }
}

static PyObject *
surf_get_smoothscale_backend(PyObject *self, PyObject *_null)
{
    return PyUnicode_FromString(GETSTATE(self)->filter_type);
}

static PyObject *
surf_set_smoothscale_backend(PyObject *self, PyObject *args, PyObject *kwargs)
{
    struct _module_state *st = GETSTATE(self);
    char *keywords[] = {"backend", NULL};
    const char *type;

#ifdef _MSC_VER
    /* MSVC static analyzer false alarm: assure type is NULL-terminated by
     * making analyzer assume it was initialised */
    __analysis_assume(type = "inited");
#endif

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s", keywords, &type)) {
        return NULL;
    }

    if (strcmp(type, "GENERIC") == 0) {
        st->filter_type = "GENERIC";
        st->filter_shrink_X = filter_shrink_X_ONLYC;
        st->filter_shrink_Y = filter_shrink_Y_ONLYC;
        st->filter_expand_X = filter_expand_X_ONLYC;
        st->filter_expand_Y = filter_expand_Y_ONLYC;
    }
#if defined(SCALE_MMX_SUPPORT)
    else if (strcmp(type, "MMX") == 0) {
        if (!SDL_HasMMX()) {
            return RAISE(PyExc_ValueError,
                         "MMX not supported on this machine");
        }
        if (PyErr_WarnEx(
                PyExc_DeprecationWarning,
                "MMX backend is deprecated in favor of new SSE2 backend",
                1) == -1) {
            return NULL;
        }
        st->filter_type = "MMX";
        st->filter_shrink_X = filter_shrink_X_MMX;
        st->filter_shrink_Y = filter_shrink_Y_MMX;
        st->filter_expand_X = filter_expand_X_MMX;
        st->filter_expand_Y = filter_expand_Y_MMX;
    }
    else if (strcmp(type, "SSE") == 0) {
        if (!SDL_HasSSE()) {
            return RAISE(PyExc_ValueError,
                         "SSE not supported on this machine");
        }
        if (PyErr_WarnEx(
                PyExc_DeprecationWarning,
                "SSE backend is deprecated in favor of new SSE2 backend",
                1) == -1) {
            return NULL;
        }
        st->filter_type = "SSE";
        st->filter_shrink_X = filter_shrink_X_SSE;
        st->filter_shrink_Y = filter_shrink_Y_SSE;
        st->filter_expand_X = filter_expand_X_SSE;
        st->filter_expand_Y = filter_expand_Y_SSE;
    }
#else
    else if (strcmp(st->filter_type, "MMX") == 0 ||
             strcmp(st->filter_type, "SSE") == 0) {
        return PyErr_Format(PyExc_ValueError,
                            "%s not supported on this machine", type);
    }
#endif /* ~defined(SCALE_MMX_SUPPORT) */
#if !defined(__EMSCRIPTEN__)
#if PG_ENABLE_SSE_NEON
    else if (strcmp(type, "SSE2") == 0) {
        if (!SDL_HasSSE2()) {
            return RAISE(PyExc_ValueError,
                         "SSE2 not supported on this machine");
        }
        st->filter_type = "SSE2";
        st->filter_shrink_X = filter_shrink_X_SSE2;
        st->filter_shrink_Y = filter_shrink_Y_SSE2;
        st->filter_expand_X = filter_expand_X_SSE2;
        st->filter_expand_Y = filter_expand_Y_SSE2;
    }

    else if (strcmp(type, "NEON") == 0) {
        if (!SDL_HasNEON()) {
            return RAISE(PyExc_ValueError,
                         "NEON not supported on this machine");
        }
        st->filter_type = "NEON";
        st->filter_shrink_X = filter_shrink_X_SSE2;
        st->filter_shrink_Y = filter_shrink_Y_SSE2;
        st->filter_expand_X = filter_expand_X_SSE2;
        st->filter_expand_Y = filter_expand_Y_SSE2;
    }
#endif /* PG_ENABLE_SSE_NEON */
#endif /* !__EMSCRIPTEN__ */
    else {
        return PyErr_Format(PyExc_ValueError, "Unknown backend type %s", type);
    }
    Py_RETURN_NONE;
}

/* _get_color_move_pixels is for iterating over pixels in a Surface.

    bpp - bytes per pixel
    the_color - is set for that pixel
    pixels - pointer is advanced by one pixel.
 */
static PG_INLINE Uint8 *
_get_color_move_pixels(Uint8 bpp, Uint8 *pixels, Uint32 *the_color)
{
    Uint8 *pix;
    // printf("bpp:%i, pixels:%p\n", bpp, pixels);

    switch (bpp) {
        case 1:
            *the_color = (Uint32) * ((Uint8 *)pixels);
            return pixels + 1;
        case 2:
            *the_color = (Uint32) * ((Uint16 *)pixels);
            return pixels + 2;
        case 3:
            pix = ((Uint8 *)pixels);
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
            *the_color = (pix[0]) + (pix[1] << 8) + (pix[2] << 16);
#else
            *the_color = (pix[2]) + (pix[1] << 8) + (pix[0] << 16);
#endif
            return pixels + 3;
        default: /* case 4: */
            *the_color = *((Uint32 *)pixels);
            return pixels + 4;
    }
    // printf("---bpp:%i, pixels:%p\n", bpp, pixels);
}

/* _set_at_pixels sets the pixel to the_color.

    x - x pos in the SDL_Surface pixels.
    y - y pos in the SDL_Surface pixels.
    format - of the SDL_Surface pixels.
    pitch - of the SDL_Surface.
    the_color - to set in the pixels at this position.
*/
static PG_INLINE void
_set_at_pixels(int x, int y, Uint8 *pixels, PG_PixelFormat *format,
               int surf_pitch, Uint32 the_color)
{
    Uint8 *byte_buf;

    switch (PG_FORMAT_BytesPerPixel(format)) {
        case 1:
            *((Uint8 *)pixels + y * surf_pitch + x) = (Uint8)the_color;
            break;
        case 2:
            *((Uint16 *)(pixels + y * surf_pitch) + x) = (Uint16)the_color;
            break;
        case 3:
            byte_buf = (Uint8 *)(pixels + y * surf_pitch) + x * 3;
#if (SDL_BYTEORDER == SDL_LIL_ENDIAN)
            *(byte_buf + (format->Rshift >> 3)) = (Uint8)(the_color >> 16);
            *(byte_buf + (format->Gshift >> 3)) = (Uint8)(the_color >> 8);
            *(byte_buf + (format->Bshift >> 3)) = (Uint8)the_color;
#else
            *(byte_buf + 2 - (format->Rshift >> 3)) = (Uint8)(the_color >> 16);
            *(byte_buf + 2 - (format->Gshift >> 3)) = (Uint8)(the_color >> 8);
            *(byte_buf + 2 - (format->Bshift >> 3)) = (Uint8)the_color;
#endif
            break;
        default: /* case 4: */
            *((Uint32 *)(pixels + y * surf_pitch) + x) = the_color;
            break;
    }
}

static int
get_threshold(SDL_Surface *dest_surf, PG_PixelFormat *dest_fmt,
              SDL_Surface *surf, PG_PixelFormat *surf_fmt,
              SDL_Palette *surf_palette, Uint32 color_search_color,
              Uint32 color_threshold, Uint32 color_set_color, int set_behavior,
              SDL_Surface *search_surf, PG_PixelFormat *search_surf_fmt,
              SDL_Palette *search_surf_palette, int inverse_set)
{
    int x, y, similar;
    Uint8 *pixels, *destpixels = NULL, *pixels2 = NULL;
    Uint32 the_color, the_color2, dest_set_color;
    Uint8 search_color_r, search_color_g, search_color_b;
    Uint8 surf_r, surf_g, surf_b;
    Uint8 threshold_r, threshold_g, threshold_b;
    Uint8 search_surf_r, search_surf_g, search_surf_b;

    int within_threshold;

    similar = 0;

    if (set_behavior) {
        destpixels = (Uint8 *)dest_surf->pixels;
    }
    if (search_surf) {
        pixels2 = (Uint8 *)search_surf->pixels;
    }

    PG_GetRGB(color_search_color, surf_fmt, surf_palette, &search_color_r,
              &search_color_g, &search_color_b);
    PG_GetRGB(color_threshold, surf_fmt, surf_palette, &threshold_r,
              &threshold_g, &threshold_b);

    for (y = 0; y < surf->h; y++) {
        pixels = (Uint8 *)surf->pixels + y * surf->pitch;
        if (search_surf) {
            pixels2 = (Uint8 *)search_surf->pixels + y * search_surf->pitch;
        }

        for (x = 0; x < surf->w; x++) {
            pixels = _get_color_move_pixels(PG_SURF_BytesPerPixel(surf),
                                            pixels, &the_color);
            PG_GetRGB(the_color, surf_fmt, surf_palette, &surf_r, &surf_g,
                      &surf_b);

            if (search_surf) {
                /* Get search_surf.color */
                pixels2 = _get_color_move_pixels(
                    PG_SURF_BytesPerPixel(search_surf), pixels2, &the_color2);
                PG_GetRGB(the_color2, search_surf_fmt, search_surf_palette,
                          &search_surf_r, &search_surf_g, &search_surf_b);

                /* search_surf(the_color2) is within threshold of
                 * surf(the_color) */
                within_threshold =
                    ((abs((int)search_surf_r - (int)surf_r) <= threshold_r) &&
                     (abs((int)search_surf_g - (int)surf_g) <= threshold_g) &&
                     (abs((int)search_surf_b - (int)surf_b) <= threshold_b));
                dest_set_color =
                    ((set_behavior == 2) ? the_color2 : color_set_color);
            }
            else {
                /* search_color within threshold of surf.the_color */
                // printf("rgb: %i,%i,%i\n", surf_r, surf_g, surf_b);
                within_threshold =
                    ((abs((int)search_color_r - (int)surf_r) <= threshold_r) &&
                     (abs((int)search_color_g - (int)surf_g) <= threshold_g) &&
                     (abs((int)search_color_b - (int)surf_b) <= threshold_b));
                dest_set_color =
                    ((set_behavior == 2) ? the_color : color_set_color);
            }

            if (within_threshold) {
                similar++;
            }
            if (set_behavior && ((within_threshold && inverse_set) ||
                                 (!within_threshold && !inverse_set))) {
                _set_at_pixels(x, y, destpixels, dest_fmt, dest_surf->pitch,
                               dest_set_color);
            }
        }
    }
    return similar;
}

/* _color_from_obj gets a color from a python object.

Returns 0 if ok, and sets color to the color.
   -1 means error.
   If color_obj is NULL, use rgba_default.
   If rgba_default is NULL, do not use a default color, return -1.
*/
int
_color_from_obj(PyObject *color_obj, SDL_Surface *surf, Uint8 rgba_default[4],
                Uint32 *color)
{
    if (color_obj) {
        if (!pg_MappedColorFromObj(color_obj, surf, color,
                                   PG_COLOR_HANDLE_ALL)) {
            return -1;
        }
    }
    else {
        if (!rgba_default) {
            return -1;
        }

        PG_PixelFormat *surf_format;
        SDL_Palette *surf_palette;
        if (!PG_GetSurfaceDetails(surf, &surf_format, &surf_palette)) {
            return -1;
        }
        *color = PG_MapRGBA(surf_format, surf_palette, rgba_default[0],
                            rgba_default[1], rgba_default[2], rgba_default[3]);
    }
    return 0;
}

static PyObject *
surf_threshold(PyObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *dest_surf_obj;
    SDL_Surface *dest_surf = NULL;

    pgSurfaceObject *surf_obj;
    SDL_Surface *surf = NULL;

    PyObject *search_color_obj;
    PyObject *threshold_obj = NULL;
    PyObject *set_color_obj = NULL;
    int set_behavior = 1;
    int inverse_set = 0;
    PyObject *search_surf_obj = NULL;
    SDL_Surface *search_surf = NULL;

    Uint8 rgba_threshold_default[4] = {0, 0, 0, 255};
    Uint8 rgba_set_color_default[4] = {0, 0, 0, 255};

    Uint32 color_search_color = 0;
    Uint32 color_threshold = 0;
    Uint32 color_set_color = 0;

    int num_threshold_pixels = 0;

    /*
    https://pyga.me/docs/ref/transform.html#pygame.transform.threshold

    Returns the number of pixels within the threshold.
    */
    static char *kwlist[] = {
        "dest_surface", /* Surface we are changing. See 'set_behavior'.
                          None - if counting (set_behavior is 0),
                                 don't need 'dest_surf'. */
        "surface",      /* Surface we are looking at. */
        "search_color", /* Color we are searching for. */
        "threshold",    /* =(0,0,0,0)  Within this distance from
                                       search_color (or search_surf). */
        "set_color",    /* =(0,0,0,0)  Color we set. */
        "set_behavior", /* =1 What and where we set pixels (if at all)
                             1 - pixels in dest_surface will be changed
                                 to 'set_color'.
                             0 - we do not change 'dest_surf', just count.
                                 Make dest_surf=None.
                             2 - pixels set in 'dest_surf' will be from
                           'surface'.
                        */
        "search_surf",  /* =None If set, compare to this surface.
                             None - search against 'search_color' instead.
                             Surface - look at the color in here rather
                                       than 'search_color'.
                        */
        "inverse_set",  /* =False.
                             False - pixels outside of threshold are changed.
                             True - pixels within threshold are changed.
                        */
        0};

    /* Get all arguments into our variables.

    https://docs.python.org/3/c-api/arg.html#c.PyArg_ParseTupleAndKeywords
    https://docs.python.org/3/c-api/arg.html#parsing-arguments
    */

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "OO!O|OOiOi", kwlist,
            /* required */
            &dest_surf_obj,             /* O python object from c type  */
            &pgSurface_Type, &surf_obj, /* O! python object from c type */
            &search_color_obj, /* O| python object. All after | optional. */
            /* optional */
            &threshold_obj,   /* O  python object. */
            &set_color_obj,   /* O  python object. */
            &set_behavior,    /* i  plain python int. */
            &search_surf_obj, /* O python object. */
            &inverse_set)) {  /* i  plain python int. */
        return NULL;
    }

    if (set_behavior == 0 &&
        !(set_color_obj == NULL || set_color_obj == Py_None)) {
        return RAISE(PyExc_TypeError,
                     "if set_behavior==0 set_color should be None");
    }
    if (set_behavior == 0 && dest_surf_obj != Py_None) {
        return RAISE(PyExc_TypeError,
                     "if set_behavior==0 dest_surf_obj should be None");
    }

    if (dest_surf_obj && dest_surf_obj != Py_None &&
        pgSurface_Check(dest_surf_obj)) {
        dest_surf = pgSurface_AsSurface(dest_surf_obj);
    }
    else if (set_behavior != 0) {
        return RAISE(
            PyExc_TypeError,
            "argument 1 must be pygame.Surface, or None with set_behavior=1");
    }

    surf = pgSurface_AsSurface(surf_obj);
    if (NULL == surf) {
        return RAISE(PyExc_TypeError, "invalid surf argument");
    }

    if (search_surf_obj && pgSurface_Check(search_surf_obj)) {
        search_surf = pgSurface_AsSurface(search_surf_obj);
    }

    if (search_surf && search_color_obj != Py_None) {
        return RAISE(PyExc_TypeError,
                     "if search_surf is used, search_color should be None");
    }

    if (set_behavior == 2 && set_color_obj != Py_None) {
        return RAISE(PyExc_TypeError,
                     "if set_behavior==2 set_color should be None");
    }

    if (search_color_obj != Py_None) {
        if (_color_from_obj(search_color_obj, surf, NULL,
                            &color_search_color)) {
            return RAISE(PyExc_TypeError, "invalid search_color argument");
        }
    }
    if (_color_from_obj(threshold_obj, surf, rgba_threshold_default,
                        &color_threshold)) {
        return RAISE(PyExc_TypeError, "invalid threshold argument");
    }

    if (set_color_obj != Py_None) {
        if (_color_from_obj(set_color_obj, surf, rgba_set_color_default,
                            &color_set_color)) {
            return RAISE(PyExc_TypeError, "invalid set_color argument");
        }
    }

    if (dest_surf && (surf->h != dest_surf->h || surf->w != dest_surf->w)) {
        return RAISE(PyExc_TypeError, "surf and dest_surf not the same size");
    }

    if (search_surf &&
        (surf->h != search_surf->h || surf->w != search_surf->w)) {
        return RAISE(PyExc_TypeError,
                     "surf and search_surf not the same size");
    }

    PG_PixelFormat *dest_fmt = NULL;
    if (dest_surf) {
        dest_fmt = PG_GetSurfaceFormat(dest_surf);
        if (dest_fmt == NULL) {
            return RAISE(pgExc_SDLError, SDL_GetError());
        }
    }

    PG_PixelFormat *surf_format;
    SDL_Palette *surf_palette;
    if (!PG_GetSurfaceDetails(surf, &surf_format, &surf_palette)) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    PG_PixelFormat *search_surf_format = NULL;
    SDL_Palette *search_surf_palette = NULL;
    if (search_surf) {
        if (!PG_GetSurfaceDetails(search_surf, &search_surf_format,
                                  &search_surf_palette)) {
            return RAISE(pgExc_SDLError, SDL_GetError());
        }
    }

    if (dest_surf) {
        pgSurface_Lock((pgSurfaceObject *)dest_surf_obj);
    }
    pgSurface_Lock(surf_obj);
    if (search_surf) {
        pgSurface_Lock((pgSurfaceObject *)search_surf_obj);
    }

    Py_BEGIN_ALLOW_THREADS;
    num_threshold_pixels = get_threshold(
        dest_surf, dest_fmt, surf, surf_format, surf_palette,
        color_search_color, color_threshold, color_set_color, set_behavior,
        search_surf, search_surf_format, search_surf_palette, inverse_set);
    Py_END_ALLOW_THREADS;

    if (dest_surf) {
        pgSurface_Unlock((pgSurfaceObject *)dest_surf_obj);
    }
    pgSurface_Unlock(surf_obj);
    if (search_surf) {
        pgSurface_Unlock((pgSurfaceObject *)search_surf_obj);
    }

    return PyLong_FromLong(num_threshold_pixels);
}

/*

TODO:
add_4
sub_4
mul_4
clamp_4

*/

#define SURF_GET_AT(p_color, p_surf, p_x, p_y, p_pixels, p_format, p_pix)    \
    switch (PG_FORMAT_BytesPerPixel(p_format)) {                             \
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

#if (SDL_BYTEORDER == SDL_LIL_ENDIAN)

#define SURF_SET_AT(p_color, p_surf, p_x, p_y, p_pixels, p_format,       \
                    p_byte_buf)                                          \
    switch (PG_FORMAT_BytesPerPixel(p_format)) {                         \
        case 1:                                                          \
            *((Uint8 *)p_pixels + (p_y) * p_surf->pitch + (p_x)) =       \
                (Uint8)p_color;                                          \
            break;                                                       \
        case 2:                                                          \
            *((Uint16 *)(p_pixels + (p_y) * p_surf->pitch) + (p_x)) =    \
                (Uint16)p_color;                                         \
            break;                                                       \
        case 3:                                                          \
            p_byte_buf =                                                 \
                (Uint8 *)(p_pixels + (p_y) * p_surf->pitch) + (p_x) * 3; \
            *(p_byte_buf + (p_format->Rshift >> 3)) =                    \
                (Uint8)(p_color >> p_format->Rshift);                    \
            *(p_byte_buf + (p_format->Gshift >> 3)) =                    \
                (Uint8)(p_color >> p_format->Gshift);                    \
            *(p_byte_buf + (p_format->Bshift >> 3)) =                    \
                (Uint8)(p_color >> p_format->Bshift);                    \
            break;                                                       \
        default:                                                         \
            *((Uint32 *)(p_pixels + (p_y) * p_surf->pitch) + (p_x)) =    \
                p_color;                                                 \
            break;                                                       \
    }

#else

#define SURF_SET_AT(p_color, p_surf, p_x, p_y, p_pixels, p_format,       \
                    p_byte_buf)                                          \
    switch (PG_FORMAT_BytesPerPixel(p_format)) {                         \
        case 1:                                                          \
            *((Uint8 *)p_pixels + (p_y) * p_surf->pitch + (p_x)) =       \
                (Uint8)p_color;                                          \
            break;                                                       \
        case 2:                                                          \
            *((Uint16 *)(p_pixels + (p_y) * p_surf->pitch) + (p_x)) =    \
                (Uint16)p_color;                                         \
            break;                                                       \
        case 3:                                                          \
            p_byte_buf =                                                 \
                (Uint8 *)(p_pixels + (p_y) * p_surf->pitch) + (p_x) * 3; \
            *(p_byte_buf + 2 - (p_format->Rshift >> 3)) =                \
                (Uint8)(p_color >> p_format->Rshift);                    \
            *(p_byte_buf + 2 - (p_format->Gshift >> 3)) =                \
                (Uint8)(p_color >> p_format->Gshift);                    \
            *(p_byte_buf + 2 - (p_format->Bshift >> 3)) =                \
                (Uint8)(p_color >> p_format->Bshift);                    \
            break;                                                       \
        default:                                                         \
            *((Uint32 *)(p_pixels + (p_y) * p_surf->pitch) + (p_x)) =    \
                p_color;                                                 \
            break;                                                       \
    }

#endif

void
grayscale_non_simd(SDL_Surface *src, PG_PixelFormat *src_format,
                   SDL_Surface *newsurf, PG_PixelFormat *newsurf_format)
{
    SDL_Palette *src_palette = PG_GetSurfacePalette(src);
    SDL_Palette *newsurf_palette = PG_GetSurfacePalette(newsurf);

    for (int y = 0; y < src->h; y++) {
        for (int x = 0; x < src->w; x++) {
            Uint32 pixel;
            Uint8 *pix;
            SURF_GET_AT(pixel, src, x, y, (Uint8 *)src->pixels, src_format,
                        pix);
            Uint8 r, g, b, a;
            PG_GetRGBA(pixel, src_format, src_palette, &r, &g, &b, &a);

            /* RGBA to GRAY formula used by OpenCV
             * We are using a bitshift and integer addition to align the
             * calculation with what is fastest for SIMD operations.
             * Results are almost identical to floating point multiplication.
             */
            Uint8 grayscale_pixel =
                (Uint8)((((76 * r) + 255) >> 8) + (((150 * g) + 255) >> 8) +
                        (((29 * b) + 255) >> 8));
            Uint32 new_pixel =
                PG_MapRGBA(newsurf_format, newsurf_palette, grayscale_pixel,
                           grayscale_pixel, grayscale_pixel, a);
            SURF_SET_AT(new_pixel, newsurf, x, y, (Uint8 *)newsurf->pixels,
                        newsurf_format, pix);
        }
    }
}

SDL_Surface *
grayscale(pgSurfaceObject *srcobj, pgSurfaceObject *dstobj)
{
    SDL_Surface *src = pgSurface_AsSurface(srcobj);
    SDL_Surface *newsurf;

    if (!dstobj) {
        newsurf = newsurf_fromsurf(src, srcobj->surf->w, srcobj->surf->h);
        if (!newsurf) {
            return NULL;
        }
    }
    else {
        newsurf = pgSurface_AsSurface(dstobj);
    }

    if (newsurf->w != src->w || newsurf->h != src->h) {
        return (SDL_Surface *)(RAISE(
            PyExc_ValueError,
            "Destination surface must be the same size as source surface."));
    }

    if (PG_SURF_BytesPerPixel(src) != PG_SURF_BytesPerPixel(newsurf)) {
        return (SDL_Surface *)(RAISE(
            PyExc_ValueError,
            "Source and destination surfaces need the same format."));
    }

    PG_PixelFormat *src_format = PG_GetSurfaceFormat(src);
    PG_PixelFormat *newsurf_format = PG_GetSurfaceFormat(newsurf);
    if (src_format == NULL || newsurf_format == NULL) {
        return (SDL_Surface *)(RAISE(pgExc_SDLError, SDL_GetError()));
    }

#if defined(__EMSCRIPTEN__)
    grayscale_non_simd(src, src_format, newsurf, newsurf_format);
#else  // !defined(__EMSCRIPTEN__)
    if (PG_FORMAT_BytesPerPixel(src_format) == 4 &&
        src_format->Rmask == newsurf_format->Rmask &&
        src_format->Gmask == newsurf_format->Gmask &&
        src_format->Bmask == newsurf_format->Bmask && (src->pitch % 4 == 0) &&
        (newsurf->pitch == (newsurf->w * 4))) {
        if (pg_has_avx2()) {
            grayscale_avx2(src, src_format, newsurf);
        }
#if defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON)
        else if (pg_HasSSE_NEON()) {
            grayscale_sse2(src, src_format, newsurf);
        }
#endif  // defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON)
        else {
            grayscale_non_simd(src, src_format, newsurf, newsurf_format);
        }
    }
    else {
        grayscale_non_simd(src, src_format, newsurf, newsurf_format);
    }
#endif  // !defined(__EMSCRIPTEN__)

    SDL_UnlockSurface(newsurf);

    return newsurf;
}

static PyObject *
surf_grayscale(PyObject *self, PyObject *args, PyObject *kwargs)
{
    pgSurfaceObject *surfobj;
    pgSurfaceObject *surfobj2 = NULL;
    SDL_Surface *newsurf;

    static char *keywords[] = {"surface", "dest_surface", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|O!", keywords,
                                     &pgSurface_Type, &surfobj,
                                     &pgSurface_Type, &surfobj2)) {
        return NULL;
    }

    newsurf = grayscale(surfobj, surfobj2);

    if (!newsurf) {
        return NULL;
    }

    if (surfobj2) {
        Py_INCREF(surfobj2);
        return (PyObject *)surfobj2;
    }
    else {
        return (PyObject *)pgSurface_New(newsurf);
    }
}

SDL_Surface *
solid_overlay(pgSurfaceObject *srcobj, Uint32 color, pgSurfaceObject *dstobj,
              const int keep_alpha)
{
    SDL_Surface *src = pgSurface_AsSurface(srcobj);
    SDL_Surface *newsurf;
    Uint8 a;

    if (!dstobj) {
        newsurf = newsurf_fromsurf(src, srcobj->surf->w, srcobj->surf->h);
        if (!newsurf) {
            return NULL;
        }
    }
    else {
        newsurf = pgSurface_AsSurface(dstobj);
    }

    if (newsurf->w != src->w || newsurf->h != src->h) {
        return (SDL_Surface *)(RAISE(
            PyExc_ValueError,
            "Destination surface must be the same size as source surface."));
    }

    PG_PixelFormat *fmt, *newsurf_format;
    SDL_Palette *src_palette, *newsurf_palette;
    if (!PG_GetSurfaceDetails(src, &fmt, &src_palette) ||
        !PG_GetSurfaceDetails(newsurf, &newsurf_format, &newsurf_palette)) {
        return (SDL_Surface *)(RAISE(pgExc_SDLError, SDL_GetError()));
    }

    if (fmt->format != newsurf_format->format) {
        return (SDL_Surface *)(RAISE(
            PyExc_ValueError,
            "Source and destination surfaces need the same format."));
    }

    /* If the source surface has no alpha channel, we can't overlay with alpha
     * blending. */
    if (!SDL_ISPIXELFORMAT_ALPHA(fmt->format)) {
        return newsurf;
    }

    /* If we are keeping the src alpha, then we need to remove the alpha from
     * the color so it's easier to add the base pixel alpha back in */
    if (keep_alpha) {
        color &= ~fmt->Amask;
    }

    int src_lock = SDL_MUSTLOCK(src);
    int dst_lock = src != newsurf && SDL_MUSTLOCK(newsurf);

    if (src_lock && !PG_LockSurface(src)) {
        return NULL;
    }
    if (dst_lock && !PG_LockSurface(newsurf)) {
        if (src_lock) {
            SDL_UnlockSurface(src);
        }
        return NULL;
    }

    /* optimized path for 32 bit surfaces */
    if (PG_FORMAT_BytesPerPixel(fmt) == 4) {
        /* This algorithm iterates over each pixel's alpha channel. If it's not
         * zero, the pixel is set to the desired color. If the keep_alpha flag
         * is set, the original alpha value is retained, allowing the overlay
         * color to inherit the surface pixel's alpha value. */
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
        const int dst_ashift = fmt->Ashift;
        const char _a_off = fmt->Ashift >> 3;
#else
        const int dst_ashift = 24 - newsurf->format->Ashift;
        const char _a_off = 3 - (newsurf->format->Ashift >> 3);
#endif

        Uint8 *srcp = (Uint8 *)src->pixels + _a_off;
        Uint32 *dstp = (Uint32 *)newsurf->pixels;

        const int src_skip = src->pitch - src->w * 4;
        const int dst_skip = newsurf->pitch / 4 - newsurf->w;
        int n, height = src->h;

        if (srcobj == dstobj) {
            if (!keep_alpha) {
                while (height--) {
                    LOOP_UNROLLED4(
                        {
                            if (*srcp) {
                                *dstp = color;
                            }
                            srcp += 4;
                            dstp++;
                        },
                        n, src->w);
                    srcp += src_skip;
                    dstp += dst_skip;
                }
            }
            else {
                while (height--) {
                    LOOP_UNROLLED4(
                        {
                            if ((a = *srcp)) {
                                *dstp = color | (a << dst_ashift);
                            }
                            srcp += 4;
                            dstp++;
                        },
                        n, src->w);
                    srcp += src_skip;
                    dstp += dst_skip;
                }
            }
        }
        else {
            if (!keep_alpha) {
                while (height--) {
                    LOOP_UNROLLED4(
                        {
                            if (*srcp) {
                                *dstp = color;
                            }
                            srcp += 4;
                            dstp++;
                        },
                        n, src->w);
                    srcp += src_skip;
                    dstp += dst_skip;
                }
            }
            else {
                while (height--) {
                    LOOP_UNROLLED4(
                        {
                            if ((a = *srcp)) {
                                *dstp = color | (a << dst_ashift);
                            }
                            srcp += 4;
                            dstp++;
                        },
                        n, src->w);
                    srcp += src_skip;
                    dstp += dst_skip;
                }
            }
        }
    }
    else /* path for 16 bit surfaces */
    {
        int x, y;
        Uint8 r, g, b;
        Uint16 *src_row = (Uint16 *)src->pixels;
        Uint16 *dst_row = (Uint16 *)newsurf->pixels;
        const int src_skip = src->pitch / 2 - src->w;
        const int dst_skip = newsurf->pitch / 2 - newsurf->w;

        Uint8 Cr, Cg, Cb, Ca;
        PG_GetRGBA(color, fmt, src_palette, &Cr, &Cg, &Cb, &Ca);
        const Uint16 color16 = (Uint16)PG_MapRGBA(
            newsurf_format, newsurf_palette, Cr, Cg, Cb, Ca);

        for (y = 0; y < src->h; y++) {
            for (x = 0; x < src->w; x++) {
                PG_GetRGBA((Uint32)*src_row, fmt, src_palette, &r, &g, &b, &a);

                if (a) {
                    if (keep_alpha) {
                        *dst_row = (Uint16)PG_MapRGBA(
                            newsurf_format, newsurf_palette, Cr, Cg, Cb, a);
                    }
                    else {
                        *dst_row = color16;
                    }
                }

                src_row++;
                dst_row++;
            }
            src_row += src_skip;
            dst_row += dst_skip;
        }
    }

    if (src_lock) {
        SDL_UnlockSurface(src);
    }
    if (dst_lock) {
        SDL_UnlockSurface(newsurf);
    }

    return newsurf;
}

static PyObject *
surf_solid_overlay(PyObject *self, PyObject *args, PyObject *kwargs)
{
    pgSurfaceObject *surfobj;
    PyObject *colorobj;
    Uint32 color;

    pgSurfaceObject *surfobj2 = NULL;
    SDL_Surface *newsurf;
    SDL_Surface *surf;
    int keep_alpha = 0;

    static char *keywords[] = {"surface", "color", "dest_surface",
                               "keep_alpha", NULL};

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "O!O|O!i", keywords, &pgSurface_Type, &surfobj,
            &colorobj, &pgSurface_Type, &surfobj2, &keep_alpha)) {
        return NULL;
    }

    surf = pgSurface_AsSurface(surfobj);

    if (!pg_MappedColorFromObj(colorobj, surf, &color, PG_COLOR_HANDLE_ALL)) {
        return RAISE(PyExc_TypeError, "invalid color argument");
    }

    newsurf = solid_overlay(surfobj, color, surfobj2, keep_alpha);

    if (!newsurf) {
        return NULL;
    }

    if (surfobj2) {
        Py_INCREF(surfobj2);
        return (PyObject *)surfobj2;
    }
    else {
        return (PyObject *)pgSurface_New(newsurf);
    }
}

#define MIN3(a, b, c) MIN(MIN(a, b), c)
#define MAX3(a, b, c) MAX(MAX(a, b), c)

/* Following hsl code is based on code from
 * https://gist.github.com/ciembor/1494530
 * and
 * http://en.wikipedia.org/wiki/HSL_color_space
 */
static PG_FORCEINLINE void
RGB_to_HSL(Uint8 r, Uint8 g, Uint8 b, float *h, float *s, float *l)
{
    float r_1 = r / 255.0f;
    float g_1 = g / 255.0f;
    float b_1 = b / 255.0f;

    float min = MIN3(r_1, g_1, b_1);
    float max = MAX3(r_1, g_1, b_1);
    float delta = max - min;

    *l = (max + min) / 2.0f;

    if (delta == 0) {
        *h = *s = 0;  // achromatic
    }
    else {
        *s = *l > 0.5f ? delta / (2.0f - max - min) : delta / (max + min);

        if (max == r_1) {
            *h = (g_1 - b_1) / delta + (g_1 < b_1 ? 6 : 0);
        }
        else if (max == g_1) {
            *h = (b_1 - r_1) / delta + 2;
        }
        else {
            *h = (r_1 - g_1) / delta + 4;
        }
        *h /= 6;
    }
}

static PG_FORCEINLINE float
hue_to_rgb(float p, float q, float t)
{
    if (t < 0) {
        t += 1;
    }
    if (t > 1) {
        t -= 1;
    }
    if (t < 1 / 6.0f) {
        return p + (q - p) * 6 * t;
    }
    if (t < 1 / 2.0f) {
        return q;
    }
    if (t < 2 / 3.0f) {
        return p + (q - p) * (2 / 3.0f - t) * 6;
    }
    return p;
}

static PG_FORCEINLINE void
HSL_to_RGB(float h, float s, float l, Uint8 *r, Uint8 *g, Uint8 *b)
{
    if (s == 0) {
        *r = *g = *b = (Uint8)(l * 255);
    }
    else {
        float q = l < 0.5f ? l * (1 + s) : l + s - l * s;
        float p = 2 * l - q;
        *r = (Uint8)(hue_to_rgb(p, q, h + 1 / 3.0f) * 255);
        *g = (Uint8)(hue_to_rgb(p, q, h) * 255);
        *b = (Uint8)(hue_to_rgb(p, q, h - 1 / 3.0f) * 255);
    }
}

static void
modify_hsl(SDL_Surface *surf, PG_PixelFormat *fmt, SDL_Surface *dst,
           PG_PixelFormat *dst_format, float h, float s, float l)
{
    int surf_locked = 0;
    if (SDL_MUSTLOCK(surf)) {
        if (PG_LockSurface(surf)) {
            surf_locked = 1;
        }
    }
    int dst_locked = 0;
    if (SDL_MUSTLOCK(dst)) {
        if (PG_LockSurface(dst)) {
            dst_locked = 1;
        }
    }

    int x, y;
    Uint8 r, g, b, a;
    float s_h = 0, s_s = 0, s_l = 0;
    Uint8 *srcp8 = (Uint8 *)surf->pixels;
    Uint8 *dstp8 = (Uint8 *)dst->pixels;
    SDL_Palette *surf_palette = PG_GetSurfacePalette(surf);
    SDL_Palette *dst_palette = PG_GetSurfacePalette(dst);

    if (PG_FORMAT_BytesPerPixel(fmt) == 4 ||
        PG_FORMAT_BytesPerPixel(fmt) == 3) {
        const int src_skip =
            surf->pitch - surf->w * PG_FORMAT_BytesPerPixel(fmt);
        const int dst_skip =
            dst->pitch - dst->w * PG_FORMAT_BytesPerPixel(fmt);

#if SDL_BYTEORDER == SDL_LIL_ENDIAN
        const int Ridx = fmt->Rshift >> 3;
        const int Gidx = fmt->Gshift >> 3;
        const int Bidx = fmt->Bshift >> 3;
        const int Aidx = fmt->Ashift >> 3;
#else
        const int Ridx = 3 - (fmt->Rshift >> 3);
        const int Gidx = 3 - (fmt->Gshift >> 3);
        const int Bidx = 3 - (fmt->Bshift >> 3);
        const int Aidx = 3 - (fmt->Ashift >> 3);
#endif

        int height = surf->h;

        while (height--) {
            for (x = 0; x < surf->w; x++) {
                RGB_to_HSL(srcp8[Ridx], srcp8[Gidx], srcp8[Bidx], &s_h, &s_s,
                           &s_l);

                if (h) {
                    s_h += h;
                    if (s_h > 1) {
                        s_h -= 1;
                    }
                    else if (s_h < 0) {
                        s_h += 1;
                    }
                }
                if (s) {
                    s_s = s_s * (1 + s);
                    s_s = s_s > 1 ? 1 : s_s < 0 ? 0 : s_s;
                }
                if (l) {
                    s_l = l < 0 ? s_l * (1 + l) : s_l * (1 - l) + l;
                    s_l = s_l > 1 ? 1 : s_l < 0 ? 0 : s_l;
                }

                HSL_to_RGB(s_h, s_s, s_l, &r, &g, &b);
                dstp8[Ridx] = r;
                dstp8[Gidx] = g;
                dstp8[Bidx] = b;
                if (fmt->Amask) {
                    dstp8[Aidx] = srcp8[Aidx];
                }

                srcp8 += PG_FORMAT_BytesPerPixel(fmt);
                dstp8 += PG_FORMAT_BytesPerPixel(fmt);
            }
            srcp8 += src_skip;
            dstp8 += dst_skip;
        }
    }
    else {
        Uint8 *pix;
        Uint32 pixel;
        for (y = 0; y < surf->h; y++) {
            for (x = 0; x < surf->w; x++) {
                SURF_GET_AT(pixel, surf, x, y, srcp8, fmt, pix);
                PG_GetRGBA(pixel, fmt, surf_palette, &r, &g, &b, &a);
                RGB_to_HSL(r, g, b, &s_h, &s_s, &s_l);

                if (h) {
                    s_h += h;
                    if (s_h > 1) {
                        s_h -= 1;
                    }
                    else if (s_h < 0) {
                        s_h += 1;
                    }
                }
                if (s) {
                    s_s = s_s * (1 + s);
                    s_s = s_s > 1 ? 1 : s_s < 0 ? 0 : s_s;
                }
                if (l) {
                    s_l = l < 0 ? s_l * (1 + l) : s_l * (1 - l) + l;
                    s_l = s_l > 1 ? 1 : s_l < 0 ? 0 : s_l;
                }

                HSL_to_RGB(s_h, s_s, s_l, &r, &g, &b);
                pixel = PG_MapRGBA(dst_format, dst_palette, r, g, b, a);
                SURF_SET_AT(pixel, dst, x, y, dstp8, fmt, pix);
            }
        }
    }

    if (surf_locked) {
        SDL_UnlockSurface(surf);
    }
    if (dst_locked) {
        SDL_UnlockSurface(dst);
    }
}

static PyObject *
surf_hsl(PyObject *self, PyObject *args, PyObject *kwargs)
{
    pgSurfaceObject *surfobj;
    pgSurfaceObject *surfobj2 = NULL;
    SDL_Surface *dst, *src;
    float h = 0, s = 0, l = 0;

    static char *keywords[] = {"surface",   "hue",          "saturation",
                               "lightness", "dest_surface", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|fffO!", keywords,
                                     &pgSurface_Type, &surfobj, &h, &s, &l,
                                     &pgSurface_Type, &surfobj2)) {
        return NULL;
    }

    if (s < -1 || s > 1) {
        PyObject *value = PyFloat_FromDouble((double)s);
        if (!value) {
            return NULL;
        }

        PyErr_Format(PyExc_ValueError,
                     "saturation value must be between -1 and 1, got %R",
                     value);
        Py_DECREF(value);
        return NULL;
    }
    if (l < -1 || l > 1) {
        PyObject *value = PyFloat_FromDouble((double)l);
        if (!value) {
            return NULL;
        }
        PyErr_Format(PyExc_ValueError,
                     "lightness value must be between -1 and 1, got %R",
                     value);
        Py_DECREF(value);
        return NULL;
    }

    h = (float)(fmodf(h, 360.0) / 360.0);

    src = pgSurface_AsSurface(surfobj);
    SURF_INIT_CHECK(src);

    if (!surfobj2) {
        dst = newsurf_fromsurf(src, src->w, src->h);
        if (!dst) {
            return NULL;
        }
    }
    else {
        dst = pgSurface_AsSurface(surfobj2);
    }

    if (dst->w != src->w || dst->h != src->h) {
        return RAISE(
            PyExc_ValueError,
            "Destination surface must be the same size as source surface.");
    }
    if (PG_SURF_FORMATENUM(src) != PG_SURF_FORMATENUM(dst)) {
        return RAISE(PyExc_ValueError,
                     "Source and destination surfaces need the same format.");
    }

    PG_PixelFormat *src_format = PG_GetSurfaceFormat(src);
    PG_PixelFormat *dst_format = PG_GetSurfaceFormat(dst);
    if (src_format == NULL || dst_format == NULL) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    Py_BEGIN_ALLOW_THREADS;
    modify_hsl(src, src_format, dst, dst_format, h, s, l);
    Py_END_ALLOW_THREADS;

    if (surfobj2) {
        Py_INCREF(surfobj2);
        return (PyObject *)surfobj2;
    }
    else {
        return (PyObject *)pgSurface_New(dst);
    }
}

/*
number to use for missing samples
*/
#define LAPLACIAN_NUM 0xFFFFFFFF

void
laplacian(SDL_Surface *surf, PG_PixelFormat *format, SDL_Surface *destsurf,
          PG_PixelFormat *destformat)
{
    int ii;
    int x, y, height, width;

    Uint32 sample[9];
    // Uint32 total[4];
    int total[4];

    Uint8 c1r, c1g, c1b, c1a;
    // Uint32 c1r, c1g, c1b, c1a;
    Uint8 acolor[4];

    Uint32 the_color;

    int atmp0;
    int atmp1;
    int atmp2;
    int atmp3;

    Uint8 *pixels, *destpixels;
    Uint8 *pix;

    Uint8 *byte_buf;

    height = surf->h;
    width = surf->w;

    pixels = (Uint8 *)surf->pixels;
    destpixels = (Uint8 *)destsurf->pixels;
    SDL_Palette *surf_palette = PG_GetSurfacePalette(surf);

    /*
        -1 -1 -1
        -1  8 -1
        -1 -1 -1

        col = (sample[4] * 8) - (sample[0] + sample[1] + sample[2] +
                                 sample[3] +             sample[5] +
                                 sample[6] + sample[7] + sample[8])

        [(-1,-1), (0,-1), (1,-1),     (-1,0), (0,0), (1,0),     (-1,1), (0,1),
       (1,1)]

    */

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            // Need to bounds check these accesses.

            if (y > 0) {
                if (x > 0) {
                    SURF_GET_AT(sample[0], surf, x + -1, y + -1, pixels,
                                format, pix);
                }

                SURF_GET_AT(sample[1], surf, x + 0, y + -1, pixels, format,
                            pix);

                if (x + 1 < width) {
                    SURF_GET_AT(sample[2], surf, x + 1, y + -1, pixels, format,
                                pix);
                }
            }
            else {
                sample[0] = LAPLACIAN_NUM;
                sample[1] = LAPLACIAN_NUM;
                sample[2] = LAPLACIAN_NUM;
            }

            if (x > 0) {
                SURF_GET_AT(sample[3], surf, x + -1, y + 0, pixels, format,
                            pix);
            }
            else {
                sample[3] = LAPLACIAN_NUM;
            }

            // SURF_GET_AT(sample[4], surf, x+0 , y+0);
            sample[4] = 0;

            if (x + 1 < width) {
                SURF_GET_AT(sample[5], surf, x + 1, y + 0, pixels, format,
                            pix);
            }
            else {
                sample[5] = LAPLACIAN_NUM;
            }

            if (y + 1 < height) {
                if (x > 0) {
                    SURF_GET_AT(sample[6], surf, x + -1, y + 1, pixels, format,
                                pix);
                }

                SURF_GET_AT(sample[7], surf, x + 0, y + 1, pixels, format,
                            pix);

                if (x + 1 < width) {
                    SURF_GET_AT(sample[8], surf, x + 1, y + 1, pixels, format,
                                pix);
                }
            }
            else {
                sample[6] = LAPLACIAN_NUM;
                sample[7] = LAPLACIAN_NUM;
                sample[8] = LAPLACIAN_NUM;
            }

            memset(total, 0, 4 * sizeof(int));

            for (ii = 0; ii < 9; ii++) {
                PG_GetRGBA(sample[ii], format, surf_palette, &c1r, &c1g, &c1b,
                           &c1a);
                total[0] += c1r;
                total[1] += c1g;
                total[2] += c1b;
                total[3] += c1a;
            }

            SURF_GET_AT(sample[4], surf, x, y, pixels, format, pix);

            PG_GetRGBA(sample[4], format, surf_palette, &c1r, &c1g, &c1b,
                       &c1a);

            // cast on the right to a signed int, and then clamp to 0-255.

            // atmp = c1r * 8

            atmp0 = c1r * 8;
            acolor[0] = MIN(MAX(atmp0 - total[0], 0), 255);
            atmp1 = c1g * 8;
            acolor[1] = MIN(MAX(atmp1 - total[1], 0), 255);
            atmp2 = c1b * 8;
            acolor[2] = MIN(MAX(atmp2 - total[2], 0), 255);
            atmp3 = c1a * 8;
            acolor[3] = MIN(MAX(atmp3 - total[3], 0), 255);

            // printf("%d;;%d;;%d;;  ", atmp0, acolor[0],total[0]);

            // printf("%d,%d,%d,%d;;  \n", acolor[0], acolor[1], acolor[2],
            // acolor[3]);

            // the_color = (Uint32)acolor;
            // the_color = 0x00000000;

            // cast on the right to Uint32, and then clamp to 255.

            the_color = PG_MapRGBA(format, surf_palette, acolor[0], acolor[1],
                                   acolor[2], acolor[3]);

            // set_at(destsurf, color, x,y);

            switch (PG_FORMAT_BytesPerPixel(destformat)) {
                case 1:
                    *((Uint8 *)destpixels + y * destsurf->pitch + x) =
                        (Uint8)the_color;
                    break;
                case 2:
                    *((Uint16 *)(destpixels + y * destsurf->pitch) + x) =
                        (Uint16)the_color;
                    break;
                case 3:
                    byte_buf =
                        (Uint8 *)(destpixels + y * destsurf->pitch) + x * 3;
#if (SDL_BYTEORDER == SDL_LIL_ENDIAN)
                    *(byte_buf + (destformat->Rshift >> 3)) =
                        (Uint8)(the_color >> format->Rshift);
                    *(byte_buf + (destformat->Gshift >> 3)) =
                        (Uint8)(the_color >> format->Gshift);
                    *(byte_buf + (destformat->Bshift >> 3)) =
                        (Uint8)(the_color >> format->Bshift);
#else
                    *(byte_buf + 2 - (destformat->Rshift >> 3)) =
                        (Uint8)(the_color >> format->Rshift);
                    *(byte_buf + 2 - (destformat->Gshift >> 3)) =
                        (Uint8)(the_color >> format->Gshift);
                    *(byte_buf + 2 - (destformat->Bshift >> 3)) =
                        (Uint8)(the_color >> format->Bshift);
#endif
                    break;
                default:
                    *((Uint32 *)(destpixels + y * destsurf->pitch) + x) =
                        the_color;
                    break;
            }
        }
    }
}

static PyObject *
surf_laplacian(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *surfobj, *surfobj2 = NULL;
    SDL_Surface *surf;
    SDL_Surface *newsurf;
    static char *keywords[] = {"surface", "dest_surface", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|O!", keywords,
                                     &pgSurface_Type, &surfobj,
                                     &pgSurface_Type, &surfobj2)) {
        return NULL;
    }

    surf = pgSurface_AsSurface(surfobj);
    SURF_INIT_CHECK(surf)

    /* if the second surface is not there, then make a new one. */

    if (!surfobj2) {
        int width = surf->w;
        int height = surf->h;

        newsurf = newsurf_fromsurf(surf, width, height);

        if (!newsurf) {
            return NULL;
        }
    }
    else {
        newsurf = pgSurface_AsSurface(surfobj2);
    }

    /* check to see if the size is the correct size. */
    if (newsurf->w != (surf->w) || newsurf->h != (surf->h)) {
        return RAISE(PyExc_ValueError,
                     "Destination surface not the same size.");
    }

    /* check to see if the format of the surface is the same. */
    if (PG_SURF_BytesPerPixel(surf) != PG_SURF_BytesPerPixel(newsurf)) {
        return RAISE(PyExc_ValueError,
                     "Source and destination surfaces need the same format.");
    }

    PG_PixelFormat *surf_format = PG_GetSurfaceFormat(surf);
    PG_PixelFormat *newsurf_format = PG_GetSurfaceFormat(newsurf);
    if (surf_format == NULL || newsurf_format == NULL) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    SDL_LockSurface(newsurf);
    SDL_LockSurface(surf);

    Py_BEGIN_ALLOW_THREADS;
    laplacian(surf, surf_format, newsurf, newsurf_format);
    Py_END_ALLOW_THREADS;

    SDL_UnlockSurface(surf);
    SDL_UnlockSurface(newsurf);

    if (surfobj2) {
        Py_INCREF(surfobj2);
        return surfobj2;
    }
    else {
        return (PyObject *)pgSurface_New(newsurf);
    }
}

int
average_surfaces(SDL_Surface **surfaces, size_t num_surfaces,
                 SDL_Surface *destsurf, int palette_colors)
{
    /*
        returns the average surface from the ones given.

        All surfaces need to be the same size.

        palette_colors - if true we average the colors in palette, otherwise we
            average the pixel values.  This is useful if the surface is
            actually grayscale colors, and not palette colors.

    */

    Uint32 *accumulate;
    Uint32 *the_idx;
    Uint32 the_color;
    SDL_Surface *surf;
    size_t surf_idx;
    int height, width, x, y;

    float div_inv;

    PG_PixelFormat *format, *destformat;
    SDL_Palette *destpalette;
    Uint8 *pixels, *destpixels;
    Uint8 *pix;
    Uint8 *byte_buf;

    Uint32 rmask, gmask, bmask;
    int rshift, gshift, bshift, rloss, gloss, bloss;
    int num_elements;

    if (!num_surfaces) {
        return 0;
    }

    height = surfaces[0]->h;
    width = surfaces[0]->w;

    destpixels = (Uint8 *)destsurf->pixels;
    if (!PG_GetSurfaceDetails(destsurf, &destformat, &destpalette)) {
        return -1;
    }

    /* allocate an array to accumulate them all.

    If we're using 1 byte per pixel, then only need to average on that much.
    */

    if ((PG_FORMAT_BytesPerPixel(destformat) == 1) && destpalette &&
        (!palette_colors)) {
        num_elements = 1;
    }
    else {
        num_elements = 3;
    }

    accumulate =
        (Uint32 *)calloc(1, sizeof(Uint32) * height * width * num_elements);

    if (!accumulate) {
        return -1;
    }

    /* add up the r,g,b from all the surfaces. */

    for (surf_idx = 0; surf_idx < num_surfaces; surf_idx++) {
        surf = surfaces[surf_idx];

        pixels = (Uint8 *)surf->pixels;
        format = PG_GetSurfaceFormat(surf);
        if (!format) {
            return -1;
        }
        rmask = format->Rmask;
        gmask = format->Gmask;
        bmask = format->Bmask;
        rshift = format->Rshift;
        gshift = format->Gshift;
        bshift = format->Bshift;
        rloss = PG_FORMAT_R_LOSS(format);
        gloss = PG_FORMAT_G_LOSS(format);
        bloss = PG_FORMAT_B_LOSS(format);

        the_idx = accumulate;
        /* If palette surface, we use a different code path... */

        if ((PG_FORMAT_BytesPerPixel(format) == 1 &&
             PG_FORMAT_BytesPerPixel(destformat) == 1) &&
            PG_GetSurfacePalette(surf) && (destpalette) && (!palette_colors)) {
            /*
            This is useful if the surface is actually grayscale colors,
            and not palette colors.
            */
            for (y = 0; y < height; y++) {
                for (x = 0; x < width; x++) {
                    SURF_GET_AT(the_color, surf, x, y, pixels, format, pix);
                    *(the_idx) += the_color;
                    the_idx++;
                }
            }
        }
        else {
            /* TODO: This doesn't work correctly for palette surfaces yet, when
               the source is paletted.  Probably need to use something like
               GET_PIXELVALS_1 from surface.h
            */

            /* for non palette surfaces, we do this... */
            for (y = 0; y < height; y++) {
                for (x = 0; x < width; x++) {
                    SURF_GET_AT(the_color, surf, x, y, pixels, format, pix);

                    *(the_idx) += ((the_color & rmask) >> rshift) << rloss;
                    *(the_idx + 1) += ((the_color & gmask) >> gshift) << gloss;
                    *(the_idx + 2) += ((the_color & bmask) >> bshift) << bloss;
                    the_idx += 3;
                }
            }
        }
    }

    /* blit the accumulated array back to the destination surface. */

    div_inv = (float)(1.0L / (num_surfaces));

    the_idx = accumulate;

    if (num_elements == 1 && (!palette_colors)) {
        /* this is where we are using the palette surface without using its
        colors from the palette.
        */
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x++) {
                the_color = (Uint32)(*(the_idx)*div_inv + .5f);
                SURF_SET_AT(the_color, destsurf, x, y, destpixels, destformat,
                            byte_buf);
                the_idx++;
            }
        }
        /* TODO: will need to handle palette colors.
         */
    }
    else if (num_elements == 3) {
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x++) {
                the_color = PG_MapRGB(destformat, destpalette,
                                      (Uint8)(*(the_idx)*div_inv + .5f),
                                      (Uint8)(*(the_idx + 1) * div_inv + .5f),
                                      (Uint8)(*(the_idx + 2) * div_inv + .5f));

                /* TODO: should it take into consideration the output
                    shifts/masks/losses?  Or does SDL_MapRGB do that correctly?

                    *(the_idx) += ((the_color & rmask) >> rshift) << rloss;
                    *(the_idx + 1) += ((the_color & gmask) >> gshift) << gloss;
                    *(the_idx + 2) += ((the_color & bmask) >> bshift) << bloss;
                */

                SURF_SET_AT(the_color, destsurf, x, y, destpixels, destformat,
                            byte_buf);

                the_idx += 3;
            }
        }
    }
    else {
        free(accumulate);
        return -4;
    }

    free(accumulate);

    return 1;
}

/*
    returns the average surface from the ones given.

    All surfaces need to be the same size.

    palette_colors - if true we average the colors in palette, otherwise we
        average the pixel values.  This is useful if the surface is
        actually grayscale colors, and not palette colors.

*/
static PyObject *
surf_average_surfaces(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *surfobj2 = NULL;
    SDL_Surface *surf;
    SDL_Surface *newsurf = NULL;
    SDL_Surface **surfaces;
    int width, height;
    int an_error = 0;
    size_t size, loop, loop_up_to;
    int palette_colors = 1;
    PyObject *list, *obj;
    PyObject *ret = NULL;
    static char *keywords[] = {"surfaces", "dest_surface", "palette_colors",
                               NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O!i", keywords, &list,
                                     &pgSurface_Type, &surfobj2,
                                     &palette_colors)) {
        return NULL;
    }

    if (!PySequence_Check(list)) {
        return RAISE(PyExc_TypeError,
                     "Argument must be a sequence of surface objects.");
    }

    size = PySequence_Length(list); /*warning, size could be -1 on error?*/

    if (size < 1) {
        return RAISE(PyExc_TypeError,
                     "Needs to be given at least one surface.");
    }

    /* Allocate an array of surface pointers. */

    surfaces = (SDL_Surface **)calloc(1, sizeof(SDL_Surface *) * size);

    if (!surfaces) {
        return RAISE(PyExc_MemoryError,
                     "Not enough memory to store surfaces.\n");
    }

    /* Iterate over 'surfaces' passed in. */

    /* need to get the first surface to see how big it is */

    for (loop = 0; loop < size; ++loop) {
        obj = PySequence_GetItem(list, loop);

        if (!obj) {
            Py_XDECREF(obj);
            ret = RAISE(PyExc_TypeError, "Needs to be a surface object.");
            an_error = 1;
            break;
        }

        if (!pgSurface_Check(obj)) {
            Py_XDECREF(obj);
            ret = RAISE(PyExc_TypeError, "Needs to be a surface object.");
            an_error = 1;
            break;
        }

        surf = pgSurface_AsSurface(obj);

        if (!surf) {
            Py_XDECREF(obj);
            ret = RAISE(PyExc_TypeError, "Needs to be a surface object.");
            an_error = 1;
            break;
        }

        if (loop == 0) {
            /* if the second surface is not there, then make a new one. */
            if (!surfobj2) {
                width = surf->w;
                height = surf->h;

                newsurf = newsurf_fromsurf(surf, width, height);

                if (!newsurf) {
                    Py_XDECREF(obj);
                    ret = RAISE(PyExc_ValueError,
                                "Could not create new surface.");
                    an_error = 1;
                    break;
                }
            }
            else {
                newsurf = pgSurface_AsSurface(surfobj2);
            }

            /* check to see if the size is the correct size. */
            if (newsurf->w != (surf->w) || newsurf->h != (surf->h)) {
                Py_XDECREF(obj);
                ret = RAISE(PyExc_ValueError,
                            "Destination surface not the same size.");
                an_error = 1;
                break;
            }

            /* check to see if the format of the surface is the same. */
            if (PG_SURF_BytesPerPixel(surf) !=
                PG_SURF_BytesPerPixel(newsurf)) {
                Py_XDECREF(obj);
                ret = RAISE(
                    PyExc_ValueError,
                    "Source and destination surfaces need the same format.");
                an_error = 1;
                break;
            }
        }

        /* Copy surface pointer, and also lock surface. */
        SDL_LockSurface(surf);
        surfaces[loop] = surf;

        Py_DECREF(obj);
    }

    loop_up_to = loop;

    if (!an_error) {
        /* Process images, get average surface. */

        SDL_LockSurface(newsurf);

        Py_BEGIN_ALLOW_THREADS;
        average_surfaces(surfaces, size, newsurf, palette_colors);
        Py_END_ALLOW_THREADS;

        SDL_UnlockSurface(newsurf);

        if (surfobj2) {
            Py_INCREF(surfobj2);
            ret = surfobj2;
        }
        else {
            ret = (PyObject *)pgSurface_New(newsurf);
        }
    }

    /* cleanup */

    /* unlock the surfaces we got up to. */

    for (loop = 0; loop < loop_up_to; loop++) {
        if (surfaces[loop]) {
            SDL_UnlockSurface(surfaces[loop]);
        }
    }

    free(surfaces);

    return ret;
}

/* VS 2015 crashes when compiling this function, turning off optimisations to
 try to fix it */
#if defined(_MSC_VER) && (_MSC_VER == 1900)
#pragma optimize("", off)
#endif

/* When GCC compiles the following function with -O3 on PPC64 little endian,
 * the function gives incorrect output with 24-bit surfaces. This is most
 * likely a compiler bug, see #2876 for related issue.
 * So turn optimisations off here */
#if defined(__GNUC__) && defined(__PPC64__)
#pragma GCC push_options
#pragma GCC optimize("O0")
#endif

void
average_color(SDL_Surface *surf, PG_PixelFormat *format, int x, int y,
              int width, int height, Uint8 *r, Uint8 *g, Uint8 *b, Uint8 *a,
              SDL_bool consider_alpha)
{
    Uint32 color, rmask, gmask, bmask, amask;
    Uint8 *pixels;
    unsigned int rtot, gtot, btot, atot, size, rshift, gshift, bshift, ashift,
        alpha;
    unsigned int rloss, gloss, bloss, aloss;
    int row, col, width_and_x, height_and_y;

    rmask = format->Rmask;
    gmask = format->Gmask;
    bmask = format->Bmask;
    amask = format->Amask;
    rshift = format->Rshift;
    gshift = format->Gshift;
    bshift = format->Bshift;
    ashift = format->Ashift;
    rloss = PG_FORMAT_R_LOSS(format);
    gloss = PG_FORMAT_G_LOSS(format);
    bloss = PG_FORMAT_B_LOSS(format);
    aloss = PG_FORMAT_A_LOSS(format);
    rtot = gtot = btot = atot = 0;

    /* make sure the area specified is within the Surface */
    if ((x + width) > surf->w) {
        width = surf->w - x;
    }
    if ((y + height) > surf->h) {
        height = surf->h - y;
    }
    if (x < 0) {
        width -= (-x);
        x = 0;
    }
    if (y < 0) {
        height -= (-y);
        y = 0;
    }

    size = width * height;
    width_and_x = width + x;
    height_and_y = height + y;

    if (consider_alpha) {
        switch (PG_FORMAT_BytesPerPixel(format)) {
            case 1: {
                Uint8 color8;
                for (row = y; row < height_and_y; row++) {
                    pixels = (Uint8 *)surf->pixels + row * surf->pitch + x;
                    for (col = x; col < width_and_x; col++) {
                        color8 = *(Uint8 *)pixels;
                        alpha = ((color8 & amask) >> ashift) << aloss;
                        atot += alpha;
                        rtot +=
                            ((((color8 & rmask) >> rshift) << rloss) * alpha);
                        gtot +=
                            ((((color8 & gmask) >> gshift) << gloss) * alpha);
                        btot +=
                            ((((color8 & bmask) >> bshift) << bloss) * alpha);
                        pixels++;
                    }
                }
            } break;
            case 2:
                for (row = y; row < height_and_y; row++) {
                    pixels = (Uint8 *)surf->pixels + row * surf->pitch + x * 2;
                    for (col = x; col < width_and_x; col++) {
                        color = (Uint32) * ((Uint16 *)pixels);
                        alpha = ((color & amask) >> ashift) << aloss;
                        atot += alpha;
                        rtot +=
                            ((((color & rmask) >> rshift) << rloss) * alpha);
                        gtot +=
                            ((((color & gmask) >> gshift) << gloss) * alpha);
                        btot +=
                            ((((color & bmask) >> bshift) << bloss) * alpha);
                        pixels += 2;
                    }
                }
                break;
            case 3:
                for (row = y; row < height_and_y; row++) {
                    pixels = (Uint8 *)surf->pixels + row * surf->pitch + x * 3;
                    for (col = x; col < width_and_x; col++) {
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                        color =
                            (pixels[0]) + (pixels[1] << 8) + (pixels[2] << 16);
#else
                        color =
                            (pixels[2]) + (pixels[1] << 8) + (pixels[0] << 16);
#endif
                        alpha = ((color & amask) >> ashift) << aloss;
                        atot += alpha;
                        rtot +=
                            ((((color & rmask) >> rshift) << rloss) * alpha);
                        gtot +=
                            ((((color & gmask) >> gshift) << gloss) * alpha);
                        btot +=
                            ((((color & bmask) >> bshift) << bloss) * alpha);
                        pixels += 3;
                    }
                }
                break;
            default: /* case 4: */
                for (row = y; row < height_and_y; row++) {
                    pixels = (Uint8 *)surf->pixels + row * surf->pitch + x * 4;
                    for (col = x; col < width_and_x; col++) {
                        color = *(Uint32 *)pixels;
                        alpha = ((color & amask) >> ashift) << aloss;
                        atot += alpha;
                        rtot +=
                            ((((color & rmask) >> rshift) << rloss) * alpha);
                        gtot +=
                            ((((color & gmask) >> gshift) << gloss) * alpha);
                        btot +=
                            ((((color & bmask) >> bshift) << bloss) * alpha);
                        pixels += 4;
                    }
                }
                break;
        }
        *a = atot / size;
        size = (atot == 0 ? size : atot);
        *r = rtot / size;
        *g = gtot / size;
        *b = btot / size;
    }
    else {
        switch (PG_FORMAT_BytesPerPixel(format)) {
            case 1: {
                Uint8 color8;
                for (row = y; row < height_and_y; row++) {
                    pixels = (Uint8 *)surf->pixels + row * surf->pitch + x;
                    for (col = x; col < width_and_x; col++) {
                        color8 = *(Uint8 *)pixels;
                        rtot += ((color8 & rmask) >> rshift) << rloss;
                        gtot += ((color8 & gmask) >> gshift) << gloss;
                        btot += ((color8 & bmask) >> bshift) << bloss;
                        atot += ((color8 & amask) >> ashift) << aloss;
                        pixels++;
                    }
                }
            } break;
            case 2:
                for (row = y; row < height_and_y; row++) {
                    pixels = (Uint8 *)surf->pixels + row * surf->pitch + x * 2;
                    for (col = x; col < width_and_x; col++) {
                        color = (Uint32) * ((Uint16 *)pixels);
                        rtot += ((color & rmask) >> rshift) << rloss;
                        gtot += ((color & gmask) >> gshift) << gloss;
                        btot += ((color & bmask) >> bshift) << bloss;
                        atot += ((color & amask) >> ashift) << aloss;
                        pixels += 2;
                    }
                }
                break;
            case 3:
                for (row = y; row < height_and_y; row++) {
                    pixels = (Uint8 *)surf->pixels + row * surf->pitch + x * 3;
                    for (col = x; col < width_and_x; col++) {
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                        color =
                            (pixels[0]) + (pixels[1] << 8) + (pixels[2] << 16);
#else
                        color =
                            (pixels[2]) + (pixels[1] << 8) + (pixels[0] << 16);
#endif
                        rtot += ((color & rmask) >> rshift) << rloss;
                        gtot += ((color & gmask) >> gshift) << gloss;
                        btot += ((color & bmask) >> bshift) << bloss;
                        atot += ((color & amask) >> ashift) << aloss;
                        pixels += 3;
                    }
                }
                break;
            default: /* case 4: */
                for (row = y; row < height_and_y; row++) {
                    pixels = (Uint8 *)surf->pixels + row * surf->pitch + x * 4;
                    for (col = x; col < width_and_x; col++) {
                        color = *(Uint32 *)pixels;
                        rtot += ((color & rmask) >> rshift) << rloss;
                        gtot += ((color & gmask) >> gshift) << gloss;
                        btot += ((color & bmask) >> bshift) << bloss;
                        atot += ((color & amask) >> ashift) << aloss;
                        pixels += 4;
                    }
                }
                break;
        }
        *r = rtot / size;
        *g = gtot / size;
        *b = btot / size;
        *a = atot / size;
    }
}

/* Optimisation was only disabled for one function - see above */
#if defined(_MSC_VER) && (_MSC_VER == 1900)
#pragma optimize("", on)
#endif

/* Optimisation was only disabled for one function on GCC+PPC64 - see above */
#if defined(__GNUC__) && defined(__PPC64__)
#pragma GCC pop_options
#endif

static PyObject *
surf_average_color(PyObject *self, PyObject *args, PyObject *kwargs)
{
    pgSurfaceObject *surfobj;
    PyObject *rectobj = NULL;
    SDL_Surface *surf;
    SDL_Rect *rect, temp;
    Uint8 r, g, b, a;
    int x, y, w, h;
    static char *keywords[] = {"surface", "rect", "consider_alpha", NULL};
    int consider_alpha = SDL_FALSE;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|Op", keywords,
                                     &pgSurface_Type, &surfobj, &rectobj,
                                     &consider_alpha)) {
        return NULL;
    }

    surf = pgSurface_AsSurface(surfobj);
    pgSurface_Lock(surfobj);

    if (!rectobj) {
        x = 0;
        y = 0;
        w = surf->w;
        h = surf->h;
    }
    else {
        if (!(rect = pgRect_FromObject(rectobj, &temp))) {
            return RAISE(PyExc_TypeError, "Rect argument is invalid");
        }
        x = rect->x;
        y = rect->y;
        w = rect->w;
        h = rect->h;
    }

    PG_PixelFormat *surf_format = PG_GetSurfaceFormat(surf);
    if (surf_format == NULL) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    Py_BEGIN_ALLOW_THREADS;
    average_color(surf, surf_format, x, y, w, h, &r, &g, &b, &a,
                  consider_alpha);
    Py_END_ALLOW_THREADS;

    pgSurface_Unlock(surfobj);
    return Py_BuildValue("(bbbb)", r, g, b, a);
}

static int
box_blur(SDL_Surface *src, SDL_Surface *dst, int radius, SDL_bool repeat)
{
    // Reference : https://blog.csdn.net/blogshinelee/article/details/80997324

    Uint8 *srcpx = (Uint8 *)src->pixels;
    Uint8 *dstpx = (Uint8 *)dst->pixels;
    Uint8 nb = PG_SURF_BytesPerPixel(src);
    int w = dst->w, h = dst->h;
    int dst_pitch = dst->pitch;
    int src_pitch = src->pitch;
    int i, x, y, color;

    // Allocate bytes for buf, sum_v, and sum_h at once to reduce allocations.
    Uint32 *overall_buf = malloc(sizeof(Uint32) * (dst_pitch * 2 + nb));
    Uint32 *buf = overall_buf;
    Uint32 *sum_v = overall_buf + dst_pitch;
    Uint32 *sum_h = overall_buf + dst_pitch * 2;

    if (overall_buf == NULL) {
        return -1;
    }

    memset(sum_v, 0, dst_pitch * sizeof(Uint32));
    for (y = 0; y <= radius; y++) {  // y-pre
        for (i = 0; i < dst_pitch; i++) {
            sum_v[i] += srcpx[src_pitch * y + i];
        }
    }
    if (repeat) {
        for (i = 0; i < dst_pitch; i++) {
            sum_v[i] += srcpx[i] * radius;
        }
    }
    for (y = 0; y < h; y++) {  // y
        for (i = 0; i < dst_pitch; i++) {
            buf[i] = sum_v[i] / (radius * 2 + 1);

            // update vertical sum
            if (y - radius >= 0) {
                sum_v[i] -= srcpx[src_pitch * (y - radius) + i];
            }
            else if (repeat) {
                sum_v[i] -= srcpx[i];
            }
            if (y + radius + 1 < h) {
                sum_v[i] += srcpx[src_pitch * (y + radius + 1) + i];
            }
            else if (repeat) {
                sum_v[i] += srcpx[src_pitch * (h - 1) + i];
            }
        }

        memset(sum_h, 0, nb * sizeof(Uint32));
        for (x = 0; x <= radius; x++) {  // x-pre
            for (color = 0; color < nb; color++) {
                sum_h[color] += buf[x * nb + color];
            }
        }
        if (repeat) {
            for (color = 0; color < nb; color++) {
                sum_h[color] += buf[color] * radius;
            }
        }
        for (x = 0; x < w; x++) {  // x
            for (color = 0; color < nb; color++) {
                dstpx[dst_pitch * y + nb * x + color] =
                    sum_h[color] / (radius * 2 + 1);

                // update horizontal sum
                if (x - radius >= 0) {
                    sum_h[color] -= buf[(x - radius) * nb + color];
                }
                else if (repeat) {
                    sum_h[color] -= buf[color];
                }
                if (x + radius + 1 < w) {
                    sum_h[color] += buf[(x + radius + 1) * nb + color];
                }
                else if (repeat) {
                    sum_h[color] += buf[(w - 1) * nb + color];
                }
            }
        }
    }

    free(overall_buf);
    return 0;
}

static int
gaussian_blur(SDL_Surface *src, SDL_Surface *dst, int sigma, SDL_bool repeat)
{
    Uint8 *srcpx = (Uint8 *)src->pixels;
    Uint8 *dstpx = (Uint8 *)dst->pixels;
    Uint8 nb = PG_SURF_BytesPerPixel(src);
    int w = dst->w, h = dst->h;
    int dst_pitch = dst->pitch;
    int src_pitch = src->pitch;
    int i, j, x, y, color;
    int kernel_radius = sigma * 2;
    float lut_sum = 0.0;

    // Allocate bytes for buf, buf2, and lut at once to reduce allocations.
    float *overall_buf =
        malloc(sizeof(float) * (dst_pitch * 2 + kernel_radius + 1));
    float *buf = overall_buf;
    float *buf2 = overall_buf + dst_pitch;
    float *lut = overall_buf + dst_pitch * 2;

    if (overall_buf == NULL) {
        return -1;
    }

    for (i = 0; i <= kernel_radius; i++) {  // init gaussian lut
        // Gaussian function
        lut[i] =
            expf(-powf((float)i, 2.0f) / (2.0f * powf((float)sigma, 2.0f)));
        lut_sum += lut[i] * 2;
    }
    lut_sum -= lut[0];
    for (i = 0; i <= kernel_radius; i++) {
        lut[i] /= lut_sum;
    }

    for (i = 0; i < dst_pitch; i++) {
        buf[i] = 0.0;
        buf2[i] = 0.0;
    }

    for (y = 0; y < h; y++) {
        for (j = -kernel_radius; j <= kernel_radius; j++) {
            for (i = 0; i < dst_pitch; i++) {
                if (y + j >= 0 && y + j < h) {
                    buf[i] +=
                        (float)srcpx[src_pitch * (y + j) + i] * lut[abs(j)];
                }
                else if (repeat) {
                    if (y + j < 0) {
                        buf[i] += (float)srcpx[i] * lut[abs(j)];
                    }
                    else {
                        buf[i] += (float)srcpx[src_pitch * (h - 1) + i] *
                                  lut[abs(j)];
                    }
                }
            }
        }

        for (x = 0; x < w; x++) {
            for (j = -kernel_radius; j <= kernel_radius; j++) {
                for (color = 0; color < nb; color++) {
                    if (x + j >= 0 && x + j < w) {
                        buf2[nb * x + color] +=
                            buf[nb * (x + j) + color] * lut[abs(j)];
                    }
                    else if (repeat) {
                        if (x + j < 0) {
                            buf2[nb * x + color] += buf[color] * lut[abs(j)];
                        }
                        else {
                            buf2[nb * x + color] +=
                                buf[nb * (w - 1) + color] * lut[abs(j)];
                        }
                    }
                }
            }
        }
        for (i = 0; i < dst_pitch; i++) {
            dstpx[dst_pitch * y + i] = (Uint8)buf2[i];
            buf[i] = 0.0;
            buf2[i] = 0.0;
        }
    }

    free(overall_buf);
    return 0;
}

static SDL_Surface *
blur(pgSurfaceObject *srcobj, pgSurfaceObject *dstobj, int radius,
     SDL_bool repeat, char algorithm)
{
    SDL_Surface *src = NULL;
    SDL_Surface *retsurf = NULL;
    int result = 0;

    if (radius < 0) {
        return RAISE(PyExc_ValueError,
                     "The radius should not be less than zero.");
    }

    src = pgSurface_AsSurface(srcobj);

    if (PG_GetSurfacePalette(src)) {
        return RAISE(PyExc_ValueError, "Indexed surfaces cannot be blurred.");
    }

    if (!dstobj) {
        retsurf = newsurf_fromsurf(src, src->w, src->h);
        if (!retsurf) {
            return NULL;
        }
    }
    else {
        retsurf = pgSurface_AsSurface(dstobj);
    }

    if ((retsurf->w) != (src->w) || (retsurf->h) != (src->h)) {
        return RAISE(PyExc_ValueError,
                     "Destination surface not the same size.");
    }

    if (retsurf->w == 0 || retsurf->h == 0) {
        return retsurf;
    }

    Uint8 *ret_start = retsurf->pixels;
    Uint8 *ret_end = ret_start + retsurf->h * retsurf->pitch;
    Uint8 *src_start = src->pixels;
    Uint8 *src_end = src_start + src->h * src->pitch;
    if ((ret_start <= src_start && ret_end >= src_start) ||
        (src_start <= ret_start && src_end >= ret_start)) {
        return RAISE(
            PyExc_ValueError,
            "Blur routines do not support dest_surfaces that share pixels "
            "with the source surface. Likely the surfaces are the same, one "
            "of them is a subsurface, or they are sharing the same buffer.");
    }

    if (PG_SURF_BytesPerPixel(src) != PG_SURF_BytesPerPixel(retsurf)) {
        return (SDL_Surface *)(RAISE(
            PyExc_ValueError,
            "Source and destination surfaces need the same format."));
    }

    if (radius > MIN(src->w, src->h)) {
        radius = MIN(src->w, src->h) - 1;
    }

    SDL_LockSurface(retsurf);
    pgSurface_Lock(srcobj);

    Py_BEGIN_ALLOW_THREADS;

    if (algorithm == 'b') {
        result = box_blur(src, retsurf, radius, repeat);
    }
    else if (algorithm == 'g') {
        result = gaussian_blur(src, retsurf, radius, repeat);
    }

    Py_END_ALLOW_THREADS;

    pgSurface_Unlock(srcobj);
    SDL_UnlockSurface(retsurf);

    // Routines only set error flag if memory allocation failed
    // Setting Python exception here outside of Py_ THREADS block to be safe
    if (result) {
        PyErr_NoMemory();
        return NULL;
    }

    return retsurf;
}

static PyObject *
surf_box_blur(PyObject *self, PyObject *args, PyObject *kwargs)
{
    pgSurfaceObject *dst_surf_obj = NULL;
    pgSurfaceObject *src_surf_obj;
    SDL_Surface *new_surf = NULL;
    int repeat_edge_pixels = SDL_TRUE;

    int radius;

    static char *kwlist[] = {"surface", "radius", "repeat_edge_pixels",
                             "dest_surface", 0};

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "O!i|pO!", kwlist, &pgSurface_Type, &src_surf_obj,
            &radius, &repeat_edge_pixels, &pgSurface_Type, &dst_surf_obj)) {
        return NULL;
    }

    new_surf =
        blur(src_surf_obj, dst_surf_obj, radius, repeat_edge_pixels, 'b');
    if (!new_surf) {
        return NULL;
    }

    if (dst_surf_obj) {
        Py_INCREF(dst_surf_obj);
        return (PyObject *)dst_surf_obj;
    }

    return (PyObject *)pgSurface_New(new_surf);
}

static PyObject *
surf_gaussian_blur(PyObject *self, PyObject *args, PyObject *kwargs)
{
    pgSurfaceObject *dst_surf_obj = NULL;
    pgSurfaceObject *src_surf_obj;
    SDL_Surface *new_surf = NULL;
    int repeat_edge_pixels = SDL_TRUE;

    int radius;

    static char *kwlist[] = {"surface", "radius", "repeat_edge_pixels",
                             "dest_surface", 0};

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "O!i|pO!", kwlist, &pgSurface_Type, &src_surf_obj,
            &radius, &repeat_edge_pixels, &pgSurface_Type, &dst_surf_obj)) {
        return NULL;
    }

    new_surf =
        blur(src_surf_obj, dst_surf_obj, radius, repeat_edge_pixels, 'g');
    if (!new_surf) {
        return NULL;
    }

    if (dst_surf_obj) {
        Py_INCREF(dst_surf_obj);
        return (PyObject *)dst_surf_obj;
    }

    return (PyObject *)pgSurface_New(new_surf);
}

void
invert_non_simd(SDL_Surface *src, PG_PixelFormat *src_format,
                SDL_Surface *newsurf, PG_PixelFormat *newsurf_format)
{
    SDL_Palette *src_palette = PG_GetSurfacePalette(src);
    SDL_Palette *newsurf_palette = PG_GetSurfacePalette(newsurf);
    for (int y = 0; y < src->h; y++) {
        for (int x = 0; x < src->w; x++) {
            Uint32 pixel;
            Uint8 *pix;
            SURF_GET_AT(pixel, src, x, y, (Uint8 *)src->pixels, src_format,
                        pix);
            unsigned char r, g, b, a;
            PG_GetRGBA(pixel, src_format, src_palette, &r, &g, &b, &a);
            Uint32 new_pixel =
                PG_MapRGBA(newsurf_format, newsurf_palette, ~r, ~g, ~b, a);
            SURF_SET_AT(new_pixel, newsurf, x, y, (Uint8 *)newsurf->pixels,
                        newsurf_format, pix);
        }
    }
}

SDL_Surface *
invert(pgSurfaceObject *srcobj, pgSurfaceObject *dstobj)
{
    SDL_Surface *src = pgSurface_AsSurface(srcobj);
    SDL_Surface *newsurf;

    if (!dstobj) {
        newsurf = newsurf_fromsurf(src, srcobj->surf->w, srcobj->surf->h);
        if (!newsurf) {
            return NULL;
        }
    }
    else {
        newsurf = pgSurface_AsSurface(dstobj);
    }

    if (newsurf->w != src->w || newsurf->h != src->h) {
        return (SDL_Surface *)(RAISE(
            PyExc_ValueError,
            "Destination surface must be the same size as source surface."));
    }

    if (PG_SURF_BytesPerPixel(src) != PG_SURF_BytesPerPixel(newsurf)) {
        return (SDL_Surface *)(RAISE(
            PyExc_ValueError,
            "Source and destination surfaces need the same format."));
    }

    PG_PixelFormat *src_format = PG_GetSurfaceFormat(src);
    PG_PixelFormat *newsurf_format = PG_GetSurfaceFormat(newsurf);
    if (src_format == NULL || newsurf_format == NULL) {
        return (SDL_Surface *)(RAISE(pgExc_SDLError, SDL_GetError()));
    }

#if defined(__EMSCRIPTEN__)
    invert_non_simd(src, src_format, newsurf, newsurf_format);
#else  // !defined(__EMSCRIPTEN__)
    if (PG_FORMAT_BytesPerPixel(src_format) == 4 &&
        src_format->Rmask == newsurf_format->Rmask &&
        src_format->Gmask == newsurf_format->Gmask &&
        src_format->Bmask == newsurf_format->Bmask && (src->pitch % 4 == 0) &&
        (newsurf->pitch == (newsurf->w * 4))) {
        if (pg_has_avx2()) {
            invert_avx2(src, src_format, newsurf);
        }
#if defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON)
        else if (pg_HasSSE_NEON()) {
            invert_sse2(src, src_format, newsurf);
        }
#endif  // defined(__SSE2__) || defined(PG_ENABLE_ARM_NEON)
        else {
            invert_non_simd(src, src_format, newsurf, newsurf_format);
        }
    }
    else {
        invert_non_simd(src, src_format, newsurf, newsurf_format);
    }
#endif  // !defined(__EMSCRIPTEN__)

    SDL_UnlockSurface(newsurf);

    return newsurf;
}

static PyObject *
surf_invert(PyObject *self, PyObject *args, PyObject *kwargs)
{
    pgSurfaceObject *surfobj;
    pgSurfaceObject *surfobj2 = NULL;
    SDL_Surface *newsurf;

    static char *keywords[] = {"surface", "dest_surface", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|O!", keywords,
                                     &pgSurface_Type, &surfobj,
                                     &pgSurface_Type, &surfobj2)) {
        return NULL;
    }

    newsurf = invert(surfobj, surfobj2);

    if (!newsurf) {
        return NULL;
    }

    if (surfobj2) {
        Py_INCREF(surfobj2);
        return (PyObject *)surfobj2;
    }
    return (PyObject *)pgSurface_New(newsurf);
}

static PyObject *
surf_pixelate(PyObject *self, PyObject *args, PyObject *kwargs)
{
    pgSurfaceObject *src;
    pgSurfaceObject *dst = NULL;
    int pixel_size;
    SDL_Surface *new_surf;
    pgSurfaceObject *intermediate;

    static char *kwds[] = {"surface", "pixel_size", "dest_surface", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!i|O!", kwds,
                                     &pgSurface_Type, &src, &pixel_size,
                                     &pgSurface_Type, &dst)) {
        return NULL;
    }

    if (pixel_size < 1) {
        PyErr_SetString(PyExc_ValueError, "pixel_size must be greater than 0");
        return NULL;
    }

    SDL_Surface *src_surf = pgSurface_AsSurface(src);
    SURF_INIT_CHECK(src_surf);

    int width = (int)round((double)src_surf->w / pixel_size);
    int height = (int)round((double)src_surf->h / pixel_size);
    if (width < 1) {
        width = 1;
    }
    if (height < 1) {
        height = 1;
    }

    SDL_Surface *temp = scale_to(src, NULL, width, height);
    if (!temp) {
        return NULL; /* Exception already set in scale_to */
    }
    intermediate = pgSurface_New(temp);
    if (intermediate == NULL) {
        SDL_FreeSurface(temp);
        return NULL; /* Exception already set in scale_to */
    }

    new_surf = scale_to(intermediate, dst, src_surf->w, src_surf->h);
    Py_DECREF(intermediate);

    if (new_surf == NULL) {
        return NULL; /* Exception already set in scale_to */
    }

    if (dst) {
        Py_INCREF(dst);
        return (PyObject *)dst;
    }

    return (PyObject *)pgSurface_New(new_surf);
}

static PyMethodDef _transform_methods[] = {
    {"scale", (PyCFunction)surf_scale, METH_VARARGS | METH_KEYWORDS,
     DOC_TRANSFORM_SCALE},
    {"scale_by", (PyCFunction)surf_scale_by, METH_VARARGS | METH_KEYWORDS,
     DOC_TRANSFORM_SCALEBY},
    {"rotate", (PyCFunction)surf_rotate, METH_VARARGS | METH_KEYWORDS,
     DOC_TRANSFORM_ROTATE},
    {"flip", (PyCFunction)surf_flip, METH_VARARGS | METH_KEYWORDS,
     DOC_TRANSFORM_FLIP},
    {"rotozoom", (PyCFunction)surf_rotozoom, METH_VARARGS | METH_KEYWORDS,
     DOC_TRANSFORM_ROTOZOOM},
    {"chop", (PyCFunction)surf_chop, METH_VARARGS | METH_KEYWORDS,
     DOC_TRANSFORM_CHOP},
    {"scale2x", (PyCFunction)surf_scale2x, METH_VARARGS | METH_KEYWORDS,
     DOC_TRANSFORM_SCALE2X},
    {"smoothscale", (PyCFunction)surf_scalesmooth,
     METH_VARARGS | METH_KEYWORDS, DOC_TRANSFORM_SMOOTHSCALE},
    {"smoothscale_by", (PyCFunction)surf_scalesmooth_by,
     METH_VARARGS | METH_KEYWORDS, DOC_TRANSFORM_SMOOTHSCALEBY},
    {"get_smoothscale_backend", surf_get_smoothscale_backend, METH_NOARGS,
     DOC_TRANSFORM_GETSMOOTHSCALEBACKEND},
    {"set_smoothscale_backend", (PyCFunction)surf_set_smoothscale_backend,
     METH_VARARGS | METH_KEYWORDS, DOC_TRANSFORM_SETSMOOTHSCALEBACKEND},
    {"threshold", (PyCFunction)surf_threshold, METH_VARARGS | METH_KEYWORDS,
     DOC_TRANSFORM_THRESHOLD},
    {"laplacian", (PyCFunction)surf_laplacian, METH_VARARGS | METH_KEYWORDS,
     DOC_TRANSFORM_THRESHOLD},
    {"average_surfaces", (PyCFunction)surf_average_surfaces,
     METH_VARARGS | METH_KEYWORDS, DOC_TRANSFORM_AVERAGESURFACES},
    {"average_color", (PyCFunction)surf_average_color,
     METH_VARARGS | METH_KEYWORDS, DOC_TRANSFORM_AVERAGECOLOR},
    {"box_blur", (PyCFunction)surf_box_blur, METH_VARARGS | METH_KEYWORDS,
     DOC_TRANSFORM_BOXBLUR},
    {"gaussian_blur", (PyCFunction)surf_gaussian_blur,
     METH_VARARGS | METH_KEYWORDS, DOC_TRANSFORM_GAUSSIANBLUR},
    {"invert", (PyCFunction)surf_invert, METH_VARARGS | METH_KEYWORDS,
     DOC_TRANSFORM_INVERT},
    {"grayscale", (PyCFunction)surf_grayscale, METH_VARARGS | METH_KEYWORDS,
     DOC_TRANSFORM_GRAYSCALE},
    {"solid_overlay", (PyCFunction)surf_solid_overlay,
     METH_VARARGS | METH_KEYWORDS, DOC_TRANSFORM_SOLIDOVERLAY},
    {"hsl", (PyCFunction)surf_hsl, METH_VARARGS | METH_KEYWORDS,
     DOC_TRANSFORM_HSL},
    {"pixelate", (PyCFunction)surf_pixelate, METH_VARARGS | METH_KEYWORDS,
     DOC_TRANSFORM_PIXELATE},
    {NULL, NULL, 0, NULL}};

MODINIT_DEFINE(transform)
{
    PyObject *module;
    struct _module_state *st;

    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "transform",
                                         DOC_TRANSFORM,
                                         sizeof(struct _module_state),
                                         _transform_methods,
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
    module = PyModule_Create(&_module);

    if (module == 0) {
        return NULL;
    }

    st = GETSTATE(module);
    if (st->filter_type == 0) {
        smoothscale_init(st);
    }
    return module;
}
