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

#if defined(SCALE_MMX_SUPPORT)

#include <SDL_cpuinfo.h>

#if PY3
#define GETSTATE(m) PY3_GETSTATE(_module_state, m)
#else
static struct _module_state _state = {0, 0, 0, 0, 0};
#define GETSTATE(m) PY2_GETSTATE(_state)
#endif

#else /* if defined(SCALE_MMX_SUPPORT) */

static void
filter_shrink_X_ONLYC(Uint8 *, Uint8 *, int, int, int, int, int);
static void
filter_shrink_Y_ONLYC(Uint8 *, Uint8 *, int, int, int, int, int);
static void
filter_expand_X_ONLYC(Uint8 *, Uint8 *, int, int, int, int, int);
static void
filter_expand_Y_ONLYC(Uint8 *, Uint8 *, int, int, int, int, int);

static struct _module_state _state = {
    "GENERIC", filter_shrink_X_ONLYC, filter_shrink_Y_ONLYC,
    filter_expand_X_ONLYC, filter_expand_Y_ONLYC};
#define GETSTATE(m) PY2_GETSTATE(_state)
#define smoothscale_init(st)

#endif /* if defined(SCALE_MMX_SUPPORT) */

void
scale2x(SDL_Surface *src, SDL_Surface *dst);
void
scale2xraw(SDL_Surface *src, SDL_Surface *dst);
extern SDL_Surface *
rotozoomSurface(SDL_Surface *src, double angle, double zoom, int smooth);


#if IS_SDLv2
static int
_PgSurface_SrcAlpha(SDL_Surface *surf)
{
    if (SDL_ISPIXELFORMAT_ALPHA(surf->format->format)) {
        SDL_BlendMode mode;
        if (SDL_GetSurfaceBlendMode(surf, &mode) < 0) {
            return -1;
        }
        if (mode == SDL_BLENDMODE_BLEND)
            return 1;
    }
    else {
        Uint8 color = SDL_ALPHA_OPAQUE;
        if (SDL_GetSurfaceAlphaMod(surf, &color) != 0) {
            return -1;
        }
        if (color != SDL_ALPHA_OPAQUE)
            return 1;
    }
    return 0;
}
#endif /* IS_SDLv2 */



static SDL_Surface *
newsurf_fromsurf(SDL_Surface *surf, int width, int height)
{
    SDL_Surface *newsurf;
#if IS_SDLv2
    Uint32 colorkey;
    Uint8 alpha;
    int isalpha;
#endif /* IS_SDLv2 */
    int result;

    if (surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4)
        return (SDL_Surface *)(RAISE(
            PyExc_ValueError, "unsupport Surface bit depth for transform"));

    newsurf = SDL_CreateRGBSurface(surf->flags, width, height,
                                   surf->format->BitsPerPixel,
                                   surf->format->Rmask, surf->format->Gmask,
                                   surf->format->Bmask, surf->format->Amask);
    if (!newsurf)
        return (SDL_Surface *)(RAISE(pgExc_SDLError, SDL_GetError()));

        /* Copy palette, colorkey, etc info */
#if IS_SDLv1
    if (surf->format->BytesPerPixel == 1 && surf->format->palette)
        SDL_SetColors(newsurf, surf->format->palette->colors, 0,
                      surf->format->palette->ncolors);
    if (surf->flags & SDL_SRCCOLORKEY)
        SDL_SetColorKey(newsurf,
                        (surf->flags & SDL_RLEACCEL) | SDL_SRCCOLORKEY,
                        surf->format->colorkey);

    if (surf->flags & SDL_SRCALPHA) {
        result = SDL_SetAlpha(newsurf, surf->flags, surf->format->alpha);
        if (result == -1)
            return (SDL_Surface *)(RAISE(pgExc_SDLError, SDL_GetError()));
    }
#else  /* IS_SDLv2 */
    if (SDL_ISPIXELFORMAT_INDEXED(surf->format->format)) {
        if (SDL_SetPaletteColors(newsurf->format->palette,
                                 surf->format->palette->colors, 0,
                                 surf->format->palette->ncolors) != 0) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            SDL_FreeSurface(newsurf);
            return NULL;
        }
    }

    if (SDL_GetSurfaceAlphaMod(surf, &alpha) != 0) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        SDL_FreeSurface(newsurf);
        return NULL;
    }
    if (alpha != 255) {
        if (SDL_SetSurfaceAlphaMod(newsurf, alpha) != 0) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            SDL_FreeSurface(newsurf);
            return NULL;
        }
    }

    isalpha = _PgSurface_SrcAlpha(surf);
    if (isalpha == 1) {
        if (SDL_SetSurfaceBlendMode(newsurf, SDL_BLENDMODE_BLEND) != 0) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            SDL_FreeSurface(newsurf);
            return NULL;
        }
    } else if (isalpha == -1) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        SDL_FreeSurface(newsurf);
        return NULL;
    } else {
        if (SDL_SetSurfaceBlendMode(newsurf, SDL_BLENDMODE_NONE) != 0){
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            SDL_FreeSurface(newsurf);
            return NULL;
        }
    }

    if (SDL_GetColorKey(surf, &colorkey) == 0) {
        if (SDL_SetColorKey(newsurf, SDL_TRUE, colorkey) != 0 ||
            SDL_SetSurfaceRLE(newsurf, SDL_TRUE) != 0) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            SDL_FreeSurface(newsurf);
            return NULL;
        }
    }

#endif /* IS_SDLv2 */
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

    if (numturns < 0)
        numturns = 4 + numturns;
    if (!(numturns % 2)) {
        dstwidth = src->w;
        dstheight = src->h;
    }
    else {
        dstwidth = src->h;
        dstheight = src->w;
    }

    dst = newsurf_fromsurf(src, dstwidth, dstheight);
    if (!dst)
        return NULL;
    SDL_LockSurface(dst);
    srcrow = (char *)src->pixels;
    dstrow = (char *)dst->pixels;
    srcstepx = dststepx = src->format->BytesPerPixel;
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
            srcstepy = src->format->BytesPerPixel;
            break;
    }

    switch (src->format->BytesPerPixel) {
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
                    dstpix[0] = srcpix[0];
                    dstpix[1] = srcpix[1];
                    dstpix[2] = srcpix[2];
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

    int ax = ((dst->w) << 15) - (int)(cangle * ((dst->w - 1) << 15));
    int ay = ((dst->h) << 15) - (int)(sangle * ((dst->w - 1) << 15));

    int xmaxval = ((src->w) << 16) - 1;
    int ymaxval = ((src->h) << 16) - 1;

    switch (src->format->BytesPerPixel) {
        case 1:
            for (y = 0; y < dst->h; y++) {
                Uint8 *dstpos = (Uint8 *)dstrow;
                dx = (ax + (isin * (cy - y))) + xd;
                dy = (ay - (icos * (cy - y))) + yd;
                for (x = 0; x < dst->w; x++) {
                    if (dx < 0 || dy < 0 || dx > xmaxval || dy > ymaxval)
                        *dstpos++ = bgcolor;
                    else
                        *dstpos++ =
                            *(Uint8 *)(srcpix + ((dy >> 16) * srcpitch) +
                                       (dx >> 16));
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
                    if (dx < 0 || dy < 0 || dx > xmaxval || dy > ymaxval)
                        *dstpos++ = bgcolor;
                    else
                        *dstpos++ =
                            *(Uint16 *)(srcpix + ((dy >> 16) * srcpitch) +
                                        (dx >> 16 << 1));
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
                    if (dx < 0 || dy < 0 || dx > xmaxval || dy > ymaxval)
                        *dstpos++ = bgcolor;
                    else
                        *dstpos++ =
                            *(Uint32 *)(srcpix + ((dy >> 16) * srcpitch) +
                                        (dx >> 16 << 2));
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
                        dstpos[0] = ((Uint8 *)&bgcolor)[0];
                        dstpos[1] = ((Uint8 *)&bgcolor)[1];
                        dstpos[2] = ((Uint8 *)&bgcolor)[2];
                        dstpos += 3;
                    }
                    else {
                        Uint8 *srcpos =
                            (Uint8 *)(srcpix + ((dy >> 16) * srcpitch) +
                                      ((dx >> 16) * 3));
                        dstpos[0] = srcpos[0];
                        dstpos[1] = srcpos[1];
                        dstpos[2] = srcpos[2];
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

static void
stretch(SDL_Surface *src, SDL_Surface *dst)
{
    int looph, loopw;

    Uint8 *srcrow = (Uint8 *)src->pixels;
    Uint8 *dstrow = (Uint8 *)dst->pixels;

    int srcpitch = src->pitch;
    int dstpitch = dst->pitch;

    int dstwidth = dst->w;
    int dstheight = dst->h;
    int dstwidth2 = dst->w << 1;
    int dstheight2 = dst->h << 1;

    int srcwidth2 = src->w << 1;
    int srcheight2 = src->h << 1;

    int w_err, h_err = srcheight2 - dstheight2;

    switch (src->format->BytesPerPixel) {
        case 1:
            for (looph = 0; looph < dstheight; ++looph) {
                Uint8 *srcpix = (Uint8 *)srcrow, *dstpix = (Uint8 *)dstrow;
                w_err = srcwidth2 - dstwidth2;
                for (loopw = 0; loopw < dstwidth; ++loopw) {
                    *dstpix++ = *srcpix;
                    while (w_err >= 0) {
                        ++srcpix;
                        w_err -= dstwidth2;
                    }
                    w_err += srcwidth2;
                }
                while (h_err >= 0) {
                    srcrow += srcpitch;
                    h_err -= dstheight2;
                }
                dstrow += dstpitch;
                h_err += srcheight2;
            }
            break;
        case 2:
            for (looph = 0; looph < dstheight; ++looph) {
                Uint16 *srcpix = (Uint16 *)srcrow, *dstpix = (Uint16 *)dstrow;
                w_err = srcwidth2 - dstwidth2;
                for (loopw = 0; loopw < dstwidth; ++loopw) {
                    *dstpix++ = *srcpix;
                    while (w_err >= 0) {
                        ++srcpix;
                        w_err -= dstwidth2;
                    }
                    w_err += srcwidth2;
                }
                while (h_err >= 0) {
                    srcrow += srcpitch;
                    h_err -= dstheight2;
                }
                dstrow += dstpitch;
                h_err += srcheight2;
            }
            break;
        case 3:
            for (looph = 0; looph < dstheight; ++looph) {
                Uint8 *srcpix = (Uint8 *)srcrow, *dstpix = (Uint8 *)dstrow;
                w_err = srcwidth2 - dstwidth2;
                for (loopw = 0; loopw < dstwidth; ++loopw) {
                    dstpix[0] = srcpix[0];
                    dstpix[1] = srcpix[1];
                    dstpix[2] = srcpix[2];
                    dstpix += 3;
                    while (w_err >= 0) {
                        srcpix += 3;
                        w_err -= dstwidth2;
                    }
                    w_err += srcwidth2;
                }
                while (h_err >= 0) {
                    srcrow += srcpitch;
                    h_err -= dstheight2;
                }
                dstrow += dstpitch;
                h_err += srcheight2;
            }
            break;
        default: /*case 4:*/
            for (looph = 0; looph < dstheight; ++looph) {
                Uint32 *srcpix = (Uint32 *)srcrow, *dstpix = (Uint32 *)dstrow;
                w_err = srcwidth2 - dstwidth2;
                for (loopw = 0; loopw < dstwidth; ++loopw) {
                    *dstpix++ = *srcpix;
                    while (w_err >= 0) {
                        ++srcpix;
                        w_err -= dstwidth2;
                    }
                    w_err += srcwidth2;
                }
                while (h_err >= 0) {
                    srcrow += srcpitch;
                    h_err -= dstheight2;
                }
                dstrow += dstpitch;
                h_err += srcheight2;
            }
            break;
    }
}

static PyObject *
surf_scale(PyObject *self, PyObject *arg)
{
    PyObject *surfobj, *surfobj2;
    SDL_Surface *surf, *newsurf;
    int width, height;
    surfobj2 = NULL;

    /*get all the arguments*/
    if (!PyArg_ParseTuple(arg, "O!(ii)|O!", &pgSurface_Type, &surfobj, &width,
                          &height, &pgSurface_Type, &surfobj2))
        return NULL;

    if (width < 0 || height < 0)
        return RAISE(PyExc_ValueError, "Cannot scale to negative size");

    surf = pgSurface_AsSurface(surfobj);

    if (!surfobj2) {
        newsurf = newsurf_fromsurf(surf, width, height);
        if (!newsurf)
            return NULL;
    }
    else
        newsurf = pgSurface_AsSurface(surfobj2);

    /* check to see if the size is twice as big. */
    if (newsurf->w != width || newsurf->h != height)
        return RAISE(PyExc_ValueError,
                     "Destination surface not the given width or height.");

    /* check to see if the format of the surface is the same. */
    if (surf->format->BytesPerPixel != newsurf->format->BytesPerPixel)
        return RAISE(PyExc_ValueError,
                     "Source and destination surfaces need the same format.");

    if ((width && height) && (surf->w && surf->h)) {
        SDL_LockSurface(newsurf);
        pgSurface_Lock(surfobj);

        Py_BEGIN_ALLOW_THREADS;
        stretch(surf, newsurf);
        Py_END_ALLOW_THREADS;

        pgSurface_Unlock(surfobj);
        SDL_UnlockSurface(newsurf);
    }

    if (surfobj2) {
        Py_INCREF(surfobj2);
        return surfobj2;
    }
    else
        return pgSurface_New(newsurf);
}

static PyObject *
surf_scale2x(PyObject *self, PyObject *arg)
{
    PyObject *surfobj, *surfobj2;
    SDL_Surface *surf;
    SDL_Surface *newsurf;
    int width, height;
    surfobj2 = NULL;

    /*get all the arguments*/
    if (!PyArg_ParseTuple(arg, "O!|O!", &pgSurface_Type, &surfobj,
                          &pgSurface_Type, &surfobj2))
        return NULL;

    surf = pgSurface_AsSurface(surfobj);

    /* if the second surface is not there, then make a new one. */

    if (!surfobj2) {
        width = surf->w * 2;
        height = surf->h * 2;

        newsurf = newsurf_fromsurf(surf, width, height);

        if (!newsurf)
            return NULL;
    }
    else
        newsurf = pgSurface_AsSurface(surfobj2);

    /* check to see if the size is twice as big. */
    if (newsurf->w != (surf->w * 2) || newsurf->h != (surf->h * 2))
        return RAISE(PyExc_ValueError, "Destination surface not 2x bigger.");

    /* check to see if the format of the surface is the same. */
    if (surf->format->BytesPerPixel != newsurf->format->BytesPerPixel)
        return RAISE(PyExc_ValueError,
                     "Source and destination surfaces need the same format.");

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
    else
        return pgSurface_New(newsurf);
}

static PyObject *
surf_scale2xraw(PyObject *self, PyObject *arg)
{
    PyObject *surfobj, *surfobj2;
    SDL_Surface *surf;
    SDL_Surface *newsurf;
    int width, height;
    surfobj2 = NULL;

    /*get all the arguments*/
    if (!PyArg_ParseTuple(arg, "O!|O!", &pgSurface_Type, &surfobj,
                          &pgSurface_Type, &surfobj2))
        return NULL;

    surf = pgSurface_AsSurface(surfobj);

    /* if the second surface is not there, then make a new one. */

    if (!surfobj2) {
        width = surf->w * 2;
        height = surf->h * 2;

        newsurf = newsurf_fromsurf(surf, width, height);

        if (!newsurf)
            return NULL;
    }
    else
        newsurf = pgSurface_AsSurface(surfobj2);

    /* check to see if the size is twice as big. */
    if (newsurf->w != (surf->w * 2) || newsurf->h != (surf->h * 2))
        return RAISE(PyExc_ValueError, "Destination surface not 2x bigger.");

    /* check to see if the format of the surface is the same. */
    if (surf->format->BytesPerPixel != newsurf->format->BytesPerPixel)
        return RAISE(PyExc_ValueError,
                     "Source and destination surfaces need the same format.");

    SDL_LockSurface(newsurf);
    SDL_LockSurface(surf);

    Py_BEGIN_ALLOW_THREADS;
    scale2xraw(surf, newsurf);
    Py_END_ALLOW_THREADS;

    SDL_UnlockSurface(surf);
    SDL_UnlockSurface(newsurf);

    if (surfobj2) {
        Py_INCREF(surfobj2);
        return surfobj2;
    }
    else
        return pgSurface_New(newsurf);
}

static PyObject *
surf_rotate(PyObject *self, PyObject *arg)
{
    PyObject *surfobj;
    SDL_Surface *surf, *newsurf;
    float angle;

    double radangle, sangle, cangle;
    double x, y, cx, cy, sx, sy;
    int nxmax, nymax;
    Uint32 bgcolor;

    /*get all the arguments*/
    if (!PyArg_ParseTuple(arg, "O!f", &pgSurface_Type, &surfobj, &angle))
        return NULL;
    surf = pgSurface_AsSurface(surfobj);

    if (surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4)
        return RAISE(PyExc_ValueError,
                     "unsupport Surface bit depth for transform");

    if (!(fmod((double)angle, (double)90.0f))) {
        pgSurface_Lock(surfobj);

        Py_BEGIN_ALLOW_THREADS;
        newsurf = rotate90(surf, (int)angle);
        Py_END_ALLOW_THREADS;

        pgSurface_Unlock(surfobj);
        if (!newsurf)
            return NULL;
        return pgSurface_New(newsurf);
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
    if (!newsurf)
        return NULL;

        /* get the background color */
#if IS_SDLv1
    if (surf->flags & SDL_SRCCOLORKEY)
        bgcolor = surf->format->colorkey;
    else
#else  /* IS_SDLv2 */
    if (SDL_GetColorKey(surf, &bgcolor) != 0)
#endif /* IS_SDLv2 */
    {
        SDL_LockSurface(surf);
        switch (surf->format->BytesPerPixel) {
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
        bgcolor &= ~surf->format->Amask;
    }

    SDL_LockSurface(newsurf);
    pgSurface_Lock(surfobj);

    Py_BEGIN_ALLOW_THREADS;
    rotate(surf, newsurf, bgcolor, sangle, cangle);
    Py_END_ALLOW_THREADS;

    pgSurface_Unlock(surfobj);
    SDL_UnlockSurface(newsurf);

    return pgSurface_New(newsurf);
}

static PyObject *
surf_flip(PyObject *self, PyObject *arg)
{
    PyObject *surfobj;
    SDL_Surface *surf, *newsurf;
    int xaxis, yaxis;
    int loopx, loopy;
    int pixsize, srcpitch, dstpitch;
    Uint8 *srcpix, *dstpix;

    /*get all the arguments*/
    if (!PyArg_ParseTuple(arg, "O!ii", &pgSurface_Type, &surfobj, &xaxis,
                          &yaxis))
        return NULL;
    surf = pgSurface_AsSurface(surfobj);

    newsurf = newsurf_fromsurf(surf, surf->w, surf->h);
    if (!newsurf)
        return NULL;

    pixsize = surf->format->BytesPerPixel;
    srcpitch = surf->pitch;
    dstpitch = newsurf->pitch;

    SDL_LockSurface(newsurf);
    pgSurface_Lock(surfobj);

    srcpix = (Uint8 *)surf->pixels;
    dstpix = (Uint8 *)newsurf->pixels;

    Py_BEGIN_ALLOW_THREADS;

    if (!xaxis) {
        if (!yaxis) {
            for (loopy = 0; loopy < surf->h; ++loopy)
                memcpy(dstpix + loopy * dstpitch, srcpix + loopy * srcpitch,
                       surf->w * surf->format->BytesPerPixel);
        }
        else {
            for (loopy = 0; loopy < surf->h; ++loopy)
                memcpy(dstpix + loopy * dstpitch,
                       srcpix + (surf->h - 1 - loopy) * srcpitch,
                       surf->w * surf->format->BytesPerPixel);
        }
    }
    else /*if (xaxis)*/
    {
        if (yaxis) {
            switch (surf->format->BytesPerPixel) {
                case 1:
                    for (loopy = 0; loopy < surf->h; ++loopy) {
                        Uint8 *dst = (Uint8 *)(dstpix + loopy * dstpitch);
                        Uint8 *src =
                            ((Uint8 *)(srcpix +
                                       (surf->h - 1 - loopy) * srcpitch)) +
                            surf->w - 1;
                        for (loopx = 0; loopx < surf->w; ++loopx)
                            *dst++ = *src--;
                    }
                    break;
                case 2:
                    for (loopy = 0; loopy < surf->h; ++loopy) {
                        Uint16 *dst = (Uint16 *)(dstpix + loopy * dstpitch);
                        Uint16 *src =
                            ((Uint16 *)(srcpix +
                                        (surf->h - 1 - loopy) * srcpitch)) +
                            surf->w - 1;
                        for (loopx = 0; loopx < surf->w; ++loopx)
                            *dst++ = *src--;
                    }
                    break;
                case 4:
                    for (loopy = 0; loopy < surf->h; ++loopy) {
                        Uint32 *dst = (Uint32 *)(dstpix + loopy * dstpitch);
                        Uint32 *src =
                            ((Uint32 *)(srcpix +
                                        (surf->h - 1 - loopy) * srcpitch)) +
                            surf->w - 1;
                        for (loopx = 0; loopx < surf->w; ++loopx)
                            *dst++ = *src--;
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
                            dst[0] = src[0];
                            dst[1] = src[1];
                            dst[2] = src[2];
                            dst += 3;
                            src -= 3;
                        }
                    }
                    break;
            }
        }
        else {
            switch (surf->format->BytesPerPixel) {
                case 1:
                    for (loopy = 0; loopy < surf->h; ++loopy) {
                        Uint8 *dst = (Uint8 *)(dstpix + loopy * dstpitch);
                        Uint8 *src = ((Uint8 *)(srcpix + loopy * srcpitch)) +
                                     surf->w - 1;
                        for (loopx = 0; loopx < surf->w; ++loopx)
                            *dst++ = *src--;
                    }
                    break;
                case 2:
                    for (loopy = 0; loopy < surf->h; ++loopy) {
                        Uint16 *dst = (Uint16 *)(dstpix + loopy * dstpitch);
                        Uint16 *src = ((Uint16 *)(srcpix + loopy * srcpitch)) +
                                      surf->w - 1;
                        for (loopx = 0; loopx < surf->w; ++loopx)
                            *dst++ = *src--;
                    }
                    break;
                case 4:
                    for (loopy = 0; loopy < surf->h; ++loopy) {
                        Uint32 *dst = (Uint32 *)(dstpix + loopy * dstpitch);
                        Uint32 *src = ((Uint32 *)(srcpix + loopy * srcpitch)) +
                                      surf->w - 1;
                        for (loopx = 0; loopx < surf->w; ++loopx)
                            *dst++ = *src--;
                    }
                    break;
                case 3:
                    for (loopy = 0; loopy < surf->h; ++loopy) {
                        Uint8 *dst = (Uint8 *)(dstpix + loopy * dstpitch);
                        Uint8 *src = ((Uint8 *)(srcpix + loopy * srcpitch)) +
                                     surf->w * 3 - 3;
                        for (loopx = 0; loopx < surf->w; ++loopx) {
                            dst[0] = src[0];
                            dst[1] = src[1];
                            dst[2] = src[2];
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
    return pgSurface_New(newsurf);
}

static PyObject *
surf_rotozoom(PyObject *self, PyObject *arg)
{
    PyObject *surfobj;
    SDL_Surface *surf, *newsurf, *surf32;
    float scale, angle;

    /*get all the arguments*/
    if (!PyArg_ParseTuple(arg, "O!ff", &pgSurface_Type, &surfobj, &angle,
                          &scale))
        return NULL;
    surf = pgSurface_AsSurface(surfobj);
    if (scale == 0.0) {
        newsurf = newsurf_fromsurf(surf, surf->w, surf->h);
        return pgSurface_New(newsurf);
    }

    if (surf->format->BitsPerPixel == 32) {
        surf32 = surf;
        pgSurface_Lock(surfobj);
    }
    else {
        Py_BEGIN_ALLOW_THREADS;
        surf32 = SDL_CreateRGBSurface(SDL_SWSURFACE, surf->w, surf->h, 32,
                                      0x000000ff, 0x0000ff00, 0x00ff0000,
                                      0xff000000);
        SDL_BlitSurface(surf, NULL, surf32, NULL);
        Py_END_ALLOW_THREADS;
    }

    Py_BEGIN_ALLOW_THREADS;
    newsurf = rotozoomSurface(surf32, angle, scale, 1);
    Py_END_ALLOW_THREADS;

    if (surf32 == surf)
        pgSurface_Unlock(surfobj);
    else
        SDL_FreeSurface(surf32);
    return pgSurface_New(newsurf);
}

static SDL_Surface *
chop(SDL_Surface *src, int x, int y, int width, int height)
{
    SDL_Surface *dst;
    int dstwidth, dstheight;
    char *srcpix, *dstpix, *srcrow, *dstrow;
    int srcstepx, srcstepy, dststepx, dststepy;
    int loopx, loopy;

    if ((x + width) > src->w)
        width = src->w - x;
    if ((y + height) > src->h)
        height = src->h - y;
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
    if (!dst)
        return NULL;
    SDL_LockSurface(dst);
    srcrow = (char *)src->pixels;
    dstrow = (char *)dst->pixels;
    srcstepx = dststepx = src->format->BytesPerPixel;
    srcstepy = src->pitch;
    dststepy = dst->pitch;

    for (loopy = 0; loopy < src->h; loopy++) {
        if ((loopy < y) || (loopy >= (y + height))) {
            dstpix = dstrow;
            srcpix = srcrow;
            for (loopx = 0; loopx < src->w; loopx++) {
                if ((loopx < x) || (loopx >= (x + width))) {
                    switch (src->format->BytesPerPixel) {
                        case 1:
                            *dstpix = *srcpix;
                            break;
                        case 2:
                            *(Uint16 *)dstpix = *(Uint16 *)srcpix;
                            break;
                        case 3:
                            dstpix[0] = srcpix[0];
                            dstpix[1] = srcpix[1];
                            dstpix[2] = srcpix[2];
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
    return dst;
}

static PyObject *
surf_chop(PyObject *self, PyObject *arg)
{
    PyObject *surfobj, *rectobj;
    SDL_Surface *surf, *newsurf;
    GAME_Rect *rect, temp;

    if (!PyArg_ParseTuple(arg, "O!O", &pgSurface_Type, &surfobj, &rectobj))
        return NULL;
    if (!(rect = pgRect_FromObject(rectobj, &temp)))
        return RAISE(PyExc_TypeError, "Rect argument is invalid");

    surf = pgSurface_AsSurface(surfobj);
    Py_BEGIN_ALLOW_THREADS;
    newsurf = chop(surf, rect->x, rect->y, rect->w, rect->h);
    Py_END_ALLOW_THREADS;

    return pgSurface_New(newsurf);
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
    if (templine == NULL)
        return;
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
    int factorwidth = 4;

    /* Allocate memory for factors */
    xidx0 = malloc(dstwidth * 4);
    if (xidx0 == NULL)
        return;
    xmult0 = (int *)malloc(dstwidth * factorwidth);
    xmult1 = (int *)malloc(dstwidth * factorwidth);
    if (xmult0 == NULL || xmult1 == NULL) {
        free(xidx0);
        if (xmult0)
            free(xmult0);
        if (xmult1)
            free(xmult1);
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

#if defined(SCALE_MMX_SUPPORT)
static void
smoothscale_init(struct _module_state *st)
{
    if (st->filter_shrink_X == 0) {
        if (SDL_HasSSE()) {
            st->filter_type = "SSE";
            st->filter_shrink_X = filter_shrink_X_SSE;
            st->filter_shrink_Y = filter_shrink_Y_SSE;
            st->filter_expand_X = filter_expand_X_SSE;
            st->filter_expand_Y = filter_expand_Y_SSE;
        }
        else if (SDL_HasMMX()) {
            st->filter_type = "MMX";
            st->filter_shrink_X = filter_shrink_X_MMX;
            st->filter_shrink_Y = filter_shrink_Y_MMX;
            st->filter_expand_X = filter_expand_X_MMX;
            st->filter_expand_Y = filter_expand_Y_MMX;
        }
        else {
            st->filter_type = "GENERIC";
            st->filter_shrink_X = filter_shrink_X_ONLYC;
            st->filter_shrink_Y = filter_shrink_Y_ONLYC;
            st->filter_expand_X = filter_expand_X_ONLYC;
            st->filter_expand_Y = filter_expand_Y_ONLYC;
        }
    }
}
#endif

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

    int bpp = src->format->BytesPerPixel;

    Uint8 *temppix = NULL;
    int tempwidth = 0, temppitch = 0, tempheight = 0;

    /* convert to 32-bit if necessary */
    if (bpp == 3) {
        int newpitch = srcwidth * 4;
        Uint8 *newsrc = (Uint8 *)malloc(newpitch * srcheight);
        if (!newsrc)
            return;
        convert_24_32(srcpix, srcpitch, newsrc, newpitch, srcwidth, srcheight);
        srcpix = newsrc;
        srcpitch = newpitch;
        /* create a destination buffer for the 32-bit result */
        dstpitch = dstwidth << 2;
        dst32 = (Uint8 *)malloc(dstpitch * dstheight);
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
        tempheight = srcheight;
        temppix = (Uint8 *)malloc(temppitch * tempheight);
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
        if (srcheight != dstheight)
            st->filter_shrink_X(srcpix, temppix, srcheight, srcpitch,
                                temppitch, srcwidth, dstwidth);
        else
            st->filter_shrink_X(srcpix, dstpix, srcheight, srcpitch, dstpitch,
                                srcwidth, dstwidth);
    }
    else if (dstwidth > srcwidth) /* expand */
    {
        if (srcheight != dstheight)
            st->filter_expand_X(srcpix, temppix, srcheight, srcpitch,
                                temppitch, srcwidth, dstwidth);
        else
            st->filter_expand_X(srcpix, dstpix, srcheight, srcpitch, dstpitch,
                                srcwidth, dstwidth);
    }
    /* Now do the Y scale */
    if (dstheight < srcheight) /* shrink */
    {
        if (srcwidth != dstwidth)
            st->filter_shrink_Y(temppix, dstpix, tempwidth, temppitch,
                                dstpitch, srcheight, dstheight);
        else
            st->filter_shrink_Y(srcpix, dstpix, srcwidth, srcpitch, dstpitch,
                                srcheight, dstheight);
    }
    else if (dstheight > srcheight) /* expand */
    {
        if (srcwidth != dstwidth)
            st->filter_expand_Y(temppix, dstpix, tempwidth, temppitch,
                                dstpitch, srcheight, dstheight);
        else
            st->filter_expand_Y(srcpix, dstpix, srcwidth, srcpitch, dstpitch,
                                srcheight, dstheight);
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
    if (temppix != NULL)
        free(temppix);
}

static PyObject *
surf_scalesmooth(PyObject *self, PyObject *arg)
{
    PyObject *surfobj, *surfobj2;
    SDL_Surface *surf, *newsurf;
    int width, height, bpp;
    surfobj2 = NULL;

    /*get all the arguments*/
    if (!PyArg_ParseTuple(arg, "O!(ii)|O!", &pgSurface_Type, &surfobj, &width,
                          &height, &pgSurface_Type, &surfobj2))
        return NULL;

    if (width < 0 || height < 0)
        return RAISE(PyExc_ValueError, "Cannot scale to negative size");

    surf = pgSurface_AsSurface(surfobj);

    bpp = surf->format->BytesPerPixel;
    if (bpp < 3 || bpp > 4)
        return RAISE(PyExc_ValueError,
                     "Only 24-bit or 32-bit surfaces can be smoothly scaled");

    if (!surfobj2) {
        newsurf = newsurf_fromsurf(surf, width, height);
        if (!newsurf)
            return NULL;
    }
    else
        newsurf = pgSurface_AsSurface(surfobj2);

    /* check to see if the size is twice as big. */
    if (newsurf->w != width || newsurf->h != height)
        return RAISE(PyExc_ValueError,
                     "Destination surface not the given width or height.");

    if (((width * bpp + 3) >> 2) > newsurf->pitch)
        return RAISE(
            PyExc_ValueError,
            "SDL Error: destination surface pitch not 4-byte aligned.");

    if (width && height) {
        SDL_LockSurface(newsurf);
        pgSurface_Lock(surfobj);
        Py_BEGIN_ALLOW_THREADS;

        /* handle trivial case */
        if (surf->w == width && surf->h == height) {
            int y;
            for (y = 0; y < height; y++) {
                memcpy((Uint8 *)newsurf->pixels + y * newsurf->pitch,
                       (Uint8 *)surf->pixels + y * surf->pitch, width * bpp);
            }
        }
        else {
            scalesmooth(surf, newsurf, GETSTATE(self));
        }
        Py_END_ALLOW_THREADS;

        pgSurface_Unlock(surfobj);
        SDL_UnlockSurface(newsurf);
    }

    if (surfobj2) {
        Py_INCREF(surfobj2);
        return surfobj2;
    }
    else
        return pgSurface_New(newsurf);
}

static PyObject *
surf_get_smoothscale_backend(PyObject *self)
{
    return Text_FromUTF8(GETSTATE(self)->filter_type);
}

static PyObject *
surf_set_smoothscale_backend(PyObject *self, PyObject *args, PyObject *kwds)
{
    struct _module_state *st = GETSTATE(self);
    char *keywords[] = {"type", NULL};
    const char *type;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s:set_smoothscale_backend",
                                     keywords, &type)) {
        return NULL;
    }

#if defined(SCALE_MMX_SUPPORT)
    if (strcmp(type, "GENERIC") == 0) {
        st->filter_type = "GENERIC";
        st->filter_shrink_X = filter_shrink_X_ONLYC;
        st->filter_shrink_Y = filter_shrink_Y_ONLYC;
        st->filter_expand_X = filter_expand_X_ONLYC;
        st->filter_expand_Y = filter_expand_Y_ONLYC;
    }
    else if (strcmp(type, "MMX") == 0) {
        if (!SDL_HasMMX()) {
            return RAISE(PyExc_ValueError,
                         "MMX not supported on this machine");
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
        st->filter_type = "SSE";
        st->filter_shrink_X = filter_shrink_X_SSE;
        st->filter_shrink_Y = filter_shrink_Y_SSE;
        st->filter_expand_X = filter_expand_X_SSE;
        st->filter_expand_Y = filter_expand_Y_SSE;
    }
    else {
        return PyErr_Format(PyExc_ValueError, "Unknown backend type %s", type);
    }
    Py_RETURN_NONE;
#else  /* Not an x86 processor */
    if (strcmp(type, "GENERIC") != 0) {
        if (strcmp(st->filter_type, "MMX") == 0 ||
            strcmp(st->filter_type, "SSE") == 0) {
            return PyErr_Format(PyExc_ValueError,
                                "%s not supported on this machine", type);
        }
        return PyErr_Format(PyExc_ValueError, "Unknown backend type %s", type);
    }
    Py_RETURN_NONE;
#endif /* defined(SCALE_MMX_SUPPORT) */
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
_set_at_pixels(int x, int y, Uint8 *pixels, SDL_PixelFormat *format,
               int surf_pitch, Uint32 the_color)
{
    Uint8 *byte_buf;

    switch (format->BytesPerPixel) {
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
get_threshold(SDL_Surface *dest_surf, SDL_Surface *surf,
              Uint32 color_search_color, Uint32 color_threshold,
              Uint32 color_set_color, int set_behavior,
              SDL_Surface *search_surf, int inverse_set)
{
    int x, y, result, similar;
    Uint8 *pixels, *destpixels = NULL, *pixels2 = NULL;
    SDL_Rect sdlrect;
    SDL_PixelFormat *format, *destformat = NULL;
    Uint32 the_color, the_color2, dest_set_color;
    Uint8 search_color_r, search_color_g, search_color_b;
    Uint8 surf_r, surf_g, surf_b;
    Uint8 threshold_r, threshold_g, threshold_b;
    Uint8 search_surf_r, search_surf_g, search_surf_b;

    int within_threshold;

    similar = 0;
    pixels = (Uint8 *)surf->pixels;
    format = surf->format;

    if (set_behavior) {
        destpixels = (Uint8 *)dest_surf->pixels;
        destformat = dest_surf->format;
    }
    if (search_surf) {
        pixels2 = (Uint8 *)search_surf->pixels;
    }

    SDL_GetRGB(color_search_color, format, &search_color_r, &search_color_g,
               &search_color_b);
    SDL_GetRGB(color_threshold, format, &threshold_r, &threshold_g,
               &threshold_b);

    for (y = 0; y < surf->h; y++) {
        pixels = (Uint8 *)surf->pixels + y * surf->pitch;
        if (search_surf)
            pixels2 = (Uint8 *)search_surf->pixels + y * search_surf->pitch;

        for (x = 0; x < surf->w; x++) {
            pixels = _get_color_move_pixels(surf->format->BytesPerPixel,
                                            pixels, &the_color);
            SDL_GetRGB(the_color, surf->format, &surf_r, &surf_g, &surf_b);

            if (search_surf) {
                /* Get search_surf.color */
                pixels2 = _get_color_move_pixels(
                    search_surf->format->BytesPerPixel, pixels2, &the_color2);
                SDL_GetRGB(the_color2, search_surf->format, &search_surf_r,
                           &search_surf_g, &search_surf_b);

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

            if (within_threshold)
                similar++;
            if (set_behavior && ((within_threshold && inverse_set) ||
                                 (!within_threshold && !inverse_set))) {
                _set_at_pixels(x, y, destpixels, dest_surf->format,
                               dest_surf->pitch, dest_set_color);
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
_color_from_obj(PyObject *color_obj, SDL_PixelFormat *format,
                Uint8 rgba_default[4], Uint32 *color)
{
    Uint8 rgba_color[4];
    if (color_obj) {
        if (PyInt_Check(color_obj))
            *color = (Uint32)PyInt_AsLong(color_obj);
        else if (PyLong_Check(color_obj))
            *color = (Uint32)PyLong_AsUnsignedLong(color_obj);
        else if (pg_RGBAFromColorObj(color_obj, rgba_color))
            *color = SDL_MapRGBA(format, rgba_color[0], rgba_color[1],
                                 rgba_color[2], rgba_color[3]);
        else
            return -1;
    }
    else {
        if (!rgba_default)
            return -1;
        *color = SDL_MapRGBA(format, rgba_default[0], rgba_default[1],
                             rgba_default[2], rgba_default[3]);
    }
    return 0;
}

static PyObject *
surf_threshold(PyObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *dest_surf_obj;
    SDL_Surface *dest_surf = NULL;

    PyObject *surf_obj = NULL;
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
    https://www.pygame.org/docs/ref/transform.html#pygame.transform.threshold

    Returns the number of pixels within the threshold.
    */
    static char *kwlist[] = {
        "dest_surf",    /* Surface we are changing. See 'set_behavior'.
                             None - if counting (set_behavior is 0),
                                    don't need 'dest_surf'. */
        "surf",         /* Surface we are looking at. */
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
            &inverse_set))    /* i  plain python int. */
        return NULL;

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
    if (search_surf_obj && pgSurface_Check(search_surf_obj))
        search_surf = pgSurface_AsSurface(search_surf_obj);

    if (search_surf && search_color_obj != Py_None) {
        return RAISE(PyExc_TypeError,
                     "if search_surf is used, search_color should be None");
    }

    if (set_behavior == 2 && set_color_obj != Py_None) {
        return RAISE(PyExc_TypeError,
                     "if set_behavior==2 set_color should be None");
    }

    if (search_color_obj != Py_None) {
        if (_color_from_obj(search_color_obj, surf->format, NULL,
                            &color_search_color))
            return RAISE(PyExc_TypeError, "invalid search_color argument");
    }
    if (_color_from_obj(threshold_obj, surf->format, rgba_threshold_default,
                        &color_threshold))
        return RAISE(PyExc_TypeError, "invalid threshold argument");

    if (set_color_obj != Py_None) {
        if (_color_from_obj(set_color_obj, surf->format,
                            rgba_set_color_default, &color_set_color))
            return RAISE(PyExc_TypeError, "invalid set_color argument");
    }

    if (dest_surf && surf &&
        (surf->h != dest_surf->h || surf->w != dest_surf->w)) {
        return RAISE(PyExc_TypeError, "surf and dest_surf not the same size");
    }
    if (search_surf && surf &&
        (surf->h != search_surf->h || surf->w != search_surf->w)) {
        return RAISE(PyExc_TypeError,
                     "surf and search_surf not the same size");
    }

    if (dest_surf)
        pgSurface_Lock(dest_surf_obj);
    pgSurface_Lock(surf_obj);
    if (search_surf)
        pgSurface_Lock(search_surf_obj);

    Py_BEGIN_ALLOW_THREADS;
    num_threshold_pixels =
        get_threshold(dest_surf, surf, color_search_color, color_threshold,
                      color_set_color, set_behavior, search_surf, inverse_set);
    Py_END_ALLOW_THREADS;

    if (dest_surf)
        pgSurface_Unlock(dest_surf_obj);
    pgSurface_Unlock(surf_obj);
    if (search_surf)
        pgSurface_Unlock(search_surf_obj);

    return PyInt_FromLong(num_threshold_pixels);
}

/*

TODO:
add_4
sub_4
mul_4
clamp_4

*/

#define SURF_GET_AT(p_color, p_surf, p_x, p_y, p_pixels, p_format, p_pix)     \
    switch (p_format->BytesPerPixel) {                                        \
        case 1:                                                               \
            p_color = (Uint32) *                                              \
                      ((Uint8 *)(p_pixels) + (p_y)*p_surf->pitch + (p_x));    \
            break;                                                            \
        case 2:                                                               \
            p_color = (Uint32) *                                              \
                      ((Uint16 *)((p_pixels) + (p_y)*p_surf->pitch) + (p_x)); \
            break;                                                            \
        case 3:                                                               \
            p_pix = ((Uint8 *)(p_pixels + (p_y)*p_surf->pitch) + (p_x)*3);    \
            p_color = (SDL_BYTEORDER == SDL_LIL_ENDIAN)                       \
                          ? (p_pix[0]) + (p_pix[1] << 8) + (p_pix[2] << 16)   \
                          : (p_pix[2]) + (p_pix[1] << 8) + (p_pix[0] << 16);  \
            break;                                                            \
        default: /* case 4: */                                                \
            p_color = *((Uint32 *)(p_pixels + (p_y)*p_surf->pitch) + (p_x));  \
            break;                                                            \
    }

#if (SDL_BYTEORDER == SDL_LIL_ENDIAN)

#define SURF_SET_AT(p_color, p_surf, p_x, p_y, p_pixels, p_format,            \
                    p_byte_buf)                                               \
    switch (p_format->BytesPerPixel) {                                        \
        case 1:                                                               \
            *((Uint8 *)p_pixels + (p_y)*p_surf->pitch + (p_x)) =              \
                (Uint8)p_color;                                               \
            break;                                                            \
        case 2:                                                               \
            *((Uint16 *)(p_pixels + (p_y)*p_surf->pitch) + (p_x)) =           \
                (Uint16)p_color;                                              \
            break;                                                            \
        case 3:                                                               \
            p_byte_buf = (Uint8 *)(p_pixels + (p_y)*p_surf->pitch) + (p_x)*3; \
            *(p_byte_buf + (p_format->Rshift >> 3)) = (Uint8)(p_color >> 16); \
            *(p_byte_buf + (p_format->Gshift >> 3)) = (Uint8)(p_color >> 8);  \
            *(p_byte_buf + (p_format->Bshift >> 3)) = (Uint8)p_color;         \
            break;                                                            \
        default:                                                              \
            *((Uint32 *)(p_pixels + (p_y)*p_surf->pitch) + (p_x)) = p_color;  \
            break;                                                            \
    }

#else

#define SURF_SET_AT(p_color, p_surf, p_x, p_y, p_pixels, p_format,            \
                    p_byte_buf)                                               \
    switch (p_format->BytesPerPixel) {                                        \
        case 1:                                                               \
            *((Uint8 *)p_pixels + (p_y)*p_surf->pitch + (p_x)) =              \
                (Uint8)p_color;                                               \
            break;                                                            \
        case 2:                                                               \
            *((Uint16 *)(p_pixels + (p_y)*p_surf->pitch) + (p_x)) =           \
                (Uint16)p_color;                                              \
            break;                                                            \
        case 3:                                                               \
            p_byte_buf = (Uint8 *)(p_pixels + (p_y)*p_surf->pitch) + (p_x)*3; \
            *(p_byte_buf + 2 - (p_format->Rshift >> 3)) =                     \
                (Uint8)(p_color >> 16);                                       \
            *(p_byte_buf + 2 - (p_format->Gshift >> 3)) =                     \
                (Uint8)(p_color >> 8);                                        \
            *(p_byte_buf + 2 - (p_format->Bshift >> 3)) = (Uint8)p_color;     \
            break;                                                            \
        default:                                                              \
            *((Uint32 *)(p_pixels + (p_y)*p_surf->pitch) + (p_x)) = p_color;  \
            break;                                                            \
    }

#endif

/*
number to use for missing samples
*/
#define LAPLACIAN_NUM 0xFFFFFFFF

void
laplacian(SDL_Surface *surf, SDL_Surface *destsurf)
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

    SDL_PixelFormat *format, *destformat;
    Uint8 *pixels, *destpixels;
    Uint8 *pix;

    Uint8 *byte_buf;

    height = surf->h;
    width = surf->w;

    pixels = (Uint8 *)surf->pixels;
    format = surf->format;

    destpixels = (Uint8 *)destsurf->pixels;
    destformat = destsurf->format;

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

            total[0] = 0;
            total[1] = 0;
            total[2] = 0;
            total[3] = 0;

            for (ii = 0; ii < 9; ii++) {
                SDL_GetRGBA(sample[ii], format, &c1r, &c1g, &c1b, &c1a);
                total[0] += c1r;
                total[1] += c1g;
                total[2] += c1b;
                total[3] += c1a;
            }

            SURF_GET_AT(sample[4], surf, x, y, pixels, format, pix);

            SDL_GetRGBA(sample[4], format, &c1r, &c1g, &c1b, &c1a);

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

            the_color = SDL_MapRGBA(surf->format, acolor[0], acolor[1],
                                    acolor[2], acolor[3]);

            // set_at(destsurf, color, x,y);

            switch (destformat->BytesPerPixel) {
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
                        (Uint8)(the_color >> 16);
                    *(byte_buf + (destformat->Gshift >> 3)) =
                        (Uint8)(the_color >> 8);
                    *(byte_buf + (destformat->Bshift >> 3)) = (Uint8)the_color;
#else
                    *(byte_buf + 2 - (destformat->Rshift >> 3)) =
                        (Uint8)(the_color >> 16);
                    *(byte_buf + 2 - (destformat->Gshift >> 3)) =
                        (Uint8)(the_color >> 8);
                    *(byte_buf + 2 - (destformat->Bshift >> 3)) =
                        (Uint8)the_color;
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
surf_laplacian(PyObject *self, PyObject *arg)
{
    PyObject *surfobj, *surfobj2;
    SDL_Surface *surf;
    SDL_Surface *newsurf;
    int width, height;
    surfobj2 = NULL;

    /*get all the arguments*/
    if (!PyArg_ParseTuple(arg, "O!|O!", &pgSurface_Type, &surfobj,
                          &pgSurface_Type, &surfobj2))
        return NULL;

    surf = pgSurface_AsSurface(surfobj);

    /* if the second surface is not there, then make a new one. */

    if (!surfobj2) {
        width = surf->w;
        height = surf->h;

        newsurf = newsurf_fromsurf(surf, width, height);

        if (!newsurf)
            return NULL;
    }
    else
        newsurf = pgSurface_AsSurface(surfobj2);

    /* check to see if the size is the correct size. */
    if (newsurf->w != (surf->w) || newsurf->h != (surf->h))
        return RAISE(PyExc_ValueError,
                     "Destination surface not the same size.");

    /* check to see if the format of the surface is the same. */
    if (surf->format->BytesPerPixel != newsurf->format->BytesPerPixel)
        return RAISE(PyExc_ValueError,
                     "Source and destination surfaces need the same format.");

    SDL_LockSurface(newsurf);
    SDL_LockSurface(surf);

    Py_BEGIN_ALLOW_THREADS;
    laplacian(surf, newsurf);
    Py_END_ALLOW_THREADS;

    SDL_UnlockSurface(surf);
    SDL_UnlockSurface(newsurf);

    if (surfobj2) {
        Py_INCREF(surfobj2);
        return surfobj2;
    }
    else
        return pgSurface_New(newsurf);
}

int
average_surfaces(SDL_Surface **surfaces, int num_surfaces,
                 SDL_Surface *destsurf, int palette_colors)
{
    /*
        returns the average surface from the ones given.

        All surfaces need to be the same size.

        palette_colors - if true we average the colors in palette, otherwise we
            average the pixel values.  This is useful if the surface is
            actually greyscale colors, and not palette colors.

    */

    Uint32 *accumulate;
    Uint32 *the_idx;
    Uint32 the_color;
    SDL_Surface *surf;
    int height, width, x, y, surf_idx;

    float div_inv;

    SDL_PixelFormat *format, *destformat;
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
    destformat = destsurf->format;

    /* allocate an array to accumulate them all.

    If we're using 1 byte per pixel, then only need to average on that much.
    */

    if ((destformat->BytesPerPixel == 1) && (destformat->palette) &&
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

        the_idx = accumulate;
        /* If palette surface, we use a different code path... */

        if ((format->BytesPerPixel == 1 && destformat->BytesPerPixel == 1) &&
            (format->palette) && (destformat->palette) && (!palette_colors)) {
            /*
            This is useful if the surface is actually greyscale colors,
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
                the_color = (*(the_idx)*div_inv + .5f);
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
                the_color =
                    SDL_MapRGB(destformat, (Uint8)(*(the_idx)*div_inv + .5f),
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
        actually greyscale colors, and not palette colors.

*/
static PyObject *
surf_average_surfaces(PyObject *self, PyObject *arg)
{
    PyObject *surfobj2;
    SDL_Surface *surf;
    SDL_Surface *newsurf;
    SDL_Surface **surfaces;
    int width, height;
    int an_error;
    size_t size, loop, loop_up_to;
    int palette_colors = 1;

    PyObject *list, *obj;
    PyObject *ret = NULL;

    an_error = 0;

    surfobj2 = NULL;
    newsurf = NULL;

    if (!PyArg_ParseTuple(arg, "O|O!i", &list, &pgSurface_Type, &surfobj2,
                          &palette_colors))
        return NULL;

    if (!PySequence_Check(list))
        return RAISE(PyExc_TypeError,
                     "Argument must be a sequence of surface objects.");

    size = PySequence_Length(list); /*warning, size could be -1 on error?*/

    if (size < 1)
        return RAISE(PyExc_TypeError,
                     "Needs to be given at least one surface.");

    /* Allocate an array of surface pointers. */

    surfaces = (SDL_Surface **)calloc(1, sizeof(SDL_Surface *) * size);

    if (!surfaces) {
        return RAISE(PyExc_MemoryError,
                     "Not enough memory to store surfaces.\n");
    }

    /* Iterate over 'surfaces' passed in. */

    /* need to get the first surface to see how big it is */

    loop = 0;

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
            else
                newsurf = pgSurface_AsSurface(surfobj2);

            /* check to see if the size is the correct size. */
            if (newsurf->w != (surf->w) || newsurf->h != (surf->h)) {
                Py_XDECREF(obj);
                ret = RAISE(PyExc_ValueError,
                            "Destination surface not the same size.");
                an_error = 1;
                break;
            }

            /* check to see if the format of the surface is the same. */
            if (surf->format->BytesPerPixel !=
                newsurf->format->BytesPerPixel) {
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
            ret = pgSurface_New(newsurf);
        }
    }
    else {
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

void
average_color(SDL_Surface *surf, int x, int y, int width, int height, Uint8 *r,
              Uint8 *g, Uint8 *b, Uint8 *a)
{
    Uint32 color, rmask, gmask, bmask, amask;
    Uint8 *pixels, *pix;
    unsigned int rtot, gtot, btot, atot, size, rshift, gshift, bshift, ashift;
    unsigned int rloss, gloss, bloss, aloss;
    int row, col, width_and_x, height_and_y;

    SDL_PixelFormat *format;

    format = surf->format;
    rmask = format->Rmask;
    gmask = format->Gmask;
    bmask = format->Bmask;
    amask = format->Amask;
    rshift = format->Rshift;
    gshift = format->Gshift;
    bshift = format->Bshift;
    ashift = format->Ashift;
    rloss = format->Rloss;
    gloss = format->Gloss;
    bloss = format->Bloss;
    aloss = format->Aloss;
    rtot = gtot = btot = atot = 0;

    /* make sure the area specified is within the Surface */
    if ((x + width) > surf->w)
        width = surf->w - x;
    if ((y + height) > surf->h)
        height = surf->h - y;
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

    switch (format->BytesPerPixel) {
        case 1:
            for (row = y; row < height_and_y; row++) {
                pixels = (Uint8 *)surf->pixels + row * surf->pitch + x;
                for (col = x; col < width_and_x; col++) {
                    color = (Uint32) * ((Uint8 *)pixels);
                    rtot += ((color & rmask) >> rshift) << rloss;
                    gtot += ((color & gmask) >> gshift) << gloss;
                    btot += ((color & bmask) >> bshift) << bloss;
                    atot += ((color & amask) >> ashift) << aloss;
                    pixels++;
                }
            }
            break;
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
                    pix = pixels;
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                    color = (pix[0]) + (pix[1] << 8) + (pix[2] << 16);
#else
                    color = (pix[2]) + (pix[1] << 8) + (pix[0] << 16);
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

/* Optimisation was only disabled for one function - see above */
#if defined(_MSC_VER) && (_MSC_VER == 1900)
#pragma optimize("", on)
#endif

static PyObject *
surf_average_color(PyObject *self, PyObject *arg)
{
    PyObject *surfobj, *rectobj = NULL;
    SDL_Surface *surf;
    GAME_Rect *rect, temp;
    Uint8 r, g, b, a;
    int x, y, w, h;

    if (!PyArg_ParseTuple(arg, "O!|O", &pgSurface_Type, &surfobj, &rectobj))
        return NULL;

    surf = pgSurface_AsSurface(surfobj);
    pgSurface_Lock(surfobj);

    if (!rectobj) {
        x = 0;
        y = 0;
        w = surf->w;
        h = surf->h;
    }
    else {
        if (!(rect = pgRect_FromObject(rectobj, &temp)))
            return RAISE(PyExc_TypeError, "Rect argument is invalid");
        x = rect->x;
        y = rect->y;
        w = rect->w;
        h = rect->h;
    }

    Py_BEGIN_ALLOW_THREADS;
    average_color(surf, x, y, w, h, &r, &g, &b, &a);
    Py_END_ALLOW_THREADS;

    pgSurface_Unlock(surfobj);
    return Py_BuildValue("(bbbb)", r, g, b, a);
}

static PyMethodDef _transform_methods[] = {
    {"scale", surf_scale, METH_VARARGS, DOC_PYGAMETRANSFORMSCALE},
    {"rotate", surf_rotate, METH_VARARGS, DOC_PYGAMETRANSFORMROTATE},
    {"flip", surf_flip, METH_VARARGS, DOC_PYGAMETRANSFORMFLIP},
    {"rotozoom", surf_rotozoom, METH_VARARGS, DOC_PYGAMETRANSFORMROTOZOOM},
    {"chop", surf_chop, METH_VARARGS, DOC_PYGAMETRANSFORMCHOP},
    {"scale2x", surf_scale2x, METH_VARARGS, DOC_PYGAMETRANSFORMSCALE2X},
    {"scale2xraw", surf_scale2xraw, METH_VARARGS, DOC_PYGAMETRANSFORMSCALE2XRAW},
    {"smoothscale", surf_scalesmooth, METH_VARARGS,
     DOC_PYGAMETRANSFORMSMOOTHSCALE},
    {"get_smoothscale_backend", (PyCFunction)surf_get_smoothscale_backend,
     METH_NOARGS, DOC_PYGAMETRANSFORMGETSMOOTHSCALEBACKEND},
    {"set_smoothscale_backend", (PyCFunction)surf_set_smoothscale_backend,
     METH_VARARGS | METH_KEYWORDS, DOC_PYGAMETRANSFORMSETSMOOTHSCALEBACKEND},
    {"threshold", (PyCFunction)surf_threshold, METH_VARARGS | METH_KEYWORDS,
     DOC_PYGAMETRANSFORMTHRESHOLD},
    {"laplacian", surf_laplacian, METH_VARARGS, DOC_PYGAMETRANSFORMTHRESHOLD},
    {"average_surfaces", surf_average_surfaces, METH_VARARGS,
     DOC_PYGAMETRANSFORMAVERAGESURFACES},
    {"average_color", surf_average_color, METH_VARARGS,
     DOC_PYGAMETRANSFORMAVERAGECOLOR},

    {NULL, NULL, 0, NULL}};

MODINIT_DEFINE(transform)
{
    PyObject *module;
    struct _module_state *st;

#if PY3
    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "transform",
                                         DOC_PYGAMETRANSFORM,
                                         sizeof(struct _module_state),
                                         _transform_methods,
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
    module = PyModule_Create(&_module);
#else
    module = Py_InitModule3(MODPREFIX "transform", _transform_methods,
                            DOC_PYGAMETRANSFORM);
#endif

    if (module == 0) {
        MODINIT_ERROR;
    }

    st = GETSTATE(module);
    if (st->filter_type == 0) {
        smoothscale_init(st);
    }
    MODINIT_RETURN(module);
}
