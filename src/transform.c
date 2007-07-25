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
#include <SDL_cpuinfo.h>
#include "pygamedocs.h"
#include <math.h>

void scale2x (SDL_Surface *src, SDL_Surface *dst);
extern SDL_Surface* rotozoomSurface (SDL_Surface *src, double angle,
                                     double zoom, int smooth);

static SDL_Surface*
newsurf_fromsurf (SDL_Surface* surf, int width, int height)
{
    SDL_Surface* newsurf;
    int result;

    if (surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4)
        return (SDL_Surface*)
            (RAISE (PyExc_ValueError,
                    "unsupport Surface bit depth for transform"));

    newsurf = SDL_CreateRGBSurface (surf->flags, width, height,
                                    surf->format->BitsPerPixel,
                                    surf->format->Rmask, surf->format->Gmask,
                                    surf->format->Bmask, surf->format->Amask);
    if (!newsurf)
        return (SDL_Surface*) (RAISE (PyExc_SDLError, SDL_GetError ()));

    /* Copy palette, colorkey, etc info */
    if (surf->format->BytesPerPixel==1 && surf->format->palette)
        SDL_SetColors (newsurf, surf->format->palette->colors, 0,
                       surf->format->palette->ncolors);
    if (surf->flags & SDL_SRCCOLORKEY)
        SDL_SetColorKey (newsurf, (surf->flags&SDL_RLEACCEL) | SDL_SRCCOLORKEY,
                         surf->format->colorkey);

    if (surf->flags&SDL_SRCALPHA)
    {
        result = SDL_SetAlpha (newsurf, surf->flags, surf->format->alpha);
        if (result == -1)
            return (SDL_Surface*) (RAISE (PyExc_SDLError, SDL_GetError ()));
    }
    return newsurf;
}

static SDL_Surface*
rotate90 (SDL_Surface *src, int angle)
{
    int numturns = (angle / 90) % 4;
    int dstwidth, dstheight;
    SDL_Surface* dst;
    char *srcpix, *dstpix, *srcrow, *dstrow;
    int srcstepx, srcstepy, dststepx, dststepy;
    int loopx, loopy;

    if (numturns < 0)
        numturns = 4 + numturns;
    if (!(numturns % 2))
    {
        dstwidth = src->w;
        dstheight = src->h;
    }
    else
    {
        dstwidth = src->h;
        dstheight = src->w;
    }
    
    dst = newsurf_fromsurf (src, dstwidth, dstheight);
    if (!dst)
        return NULL;
    SDL_LockSurface (dst);
    srcrow = (char*) src->pixels;
    dstrow = (char*) dst->pixels;
    srcstepx = dststepx = src->format->BytesPerPixel;
    srcstepy = src->pitch;
    dststepy = dst->pitch;
    
    switch (numturns)
    {
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

    switch (src->format->BytesPerPixel)
    {
    case 1:
        for (loopy = 0; loopy < dstheight; ++loopy)
        {
            dstpix = dstrow;
            srcpix = srcrow;
            for (loopx = 0; loopx < dstwidth; ++loopx)
            {
                *dstpix = *srcpix;
                srcpix += srcstepx;
                dstpix += dststepx;
            }
            dstrow += dststepy;
            srcrow += srcstepy;
        }
        break;
    case 2:
        for (loopy = 0; loopy < dstheight; ++loopy)
        {
            dstpix = dstrow;
            srcpix = srcrow;
            for (loopx = 0; loopx < dstwidth; ++loopx)
            {
                *(Uint16*)dstpix = *(Uint16*)srcpix;
                srcpix += srcstepx;
                dstpix += dststepx;
            }
            dstrow += dststepy;
            srcrow += srcstepy;
        }
        break;
    case 3:
        for (loopy = 0; loopy < dstheight; ++loopy)
        {
            dstpix = dstrow;
            srcpix = srcrow;
            for (loopx = 0; loopx < dstwidth; ++loopx)
            {
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
        for (loopy = 0; loopy < dstheight; ++loopy)
        {
            dstpix = dstrow;
            srcpix = srcrow;
            for (loopx = 0; loopx < dstwidth; ++loopx)
            {
                *(Uint32*)dstpix = *(Uint32*)srcpix;
                srcpix += srcstepx;
                dstpix += dststepx;
            }
            dstrow += dststepy;
            srcrow += srcstepy;
        }
        break;
    }
    SDL_UnlockSurface (dst);
    return dst;
}


static void
rotate (SDL_Surface *src, SDL_Surface *dst, Uint32 bgcolor, double sangle,
        double cangle)
{
    int x, y, dx, dy;
    
    Uint8 *srcpix = (Uint8*) src->pixels;
    Uint8 *dstrow = (Uint8*) dst->pixels;
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
    
    switch (src->format->BytesPerPixel)
    {
    case 1:
        for (y = 0; y < dst->h; y++)
        {
            Uint8 *dstpos = (Uint8*)dstrow;
            dx = (ax + (isin * (cy - y))) + xd;
            dy = (ay - (icos * (cy - y))) + yd;
            for (x = 0; x < dst->w; x++)
            {
                if(dx < 0 || dy < 0 || dx > xmaxval || dy > ymaxval)
                    *dstpos++ = bgcolor;
                else
                    *dstpos++ = *(Uint8*)
                        (srcpix + ((dy >> 16) * srcpitch) + (dx >> 16));
                dx += icos;
                dy += isin;
            }
            dstrow += dstpitch;
        }
        break;
    case 2:
        for (y = 0; y < dst->h; y++)
        {
            Uint16 *dstpos = (Uint16*)dstrow;
            dx = (ax + (isin * (cy - y))) + xd;
            dy = (ay - (icos * (cy - y))) + yd;
            for (x = 0; x < dst->w; x++)
            {
                if (dx < 0 || dy < 0 || dx > xmaxval || dy > ymaxval)
                    *dstpos++ = bgcolor;
                else
                    *dstpos++ = *(Uint16*)
                        (srcpix + ((dy >> 16) * srcpitch) + (dx >> 16 << 1));
                dx += icos;
                dy += isin;
            }
            dstrow += dstpitch;
        }
        break;
    case 4:
        for (y = 0; y < dst->h; y++)
        {
            Uint32 *dstpos = (Uint32*)dstrow;
            dx = (ax + (isin * (cy - y))) + xd;
            dy = (ay - (icos * (cy - y))) + yd;
            for (x = 0; x < dst->w; x++)
            {
                if (dx < 0 || dy < 0 || dx > xmaxval || dy > ymaxval)
                    *dstpos++ = bgcolor;
                else
                    *dstpos++ = *(Uint32*)
                        (srcpix + ((dy >> 16) * srcpitch) + (dx >> 16 << 2));
                dx += icos;
                dy += isin;
            }
            dstrow += dstpitch;
        }
        break;
    default: /*case 3:*/
        for (y = 0; y < dst->h; y++)
        {
            Uint8 *dstpos = (Uint8*)dstrow;
            dx = (ax + (isin * (cy - y))) + xd;
            dy = (ay - (icos * (cy - y))) + yd;
            for (x = 0; x < dst->w; x++)
            {
                if (dx < 0 || dy < 0 || dx > xmaxval || dy > ymaxval)
                {
                    dstpos[0] = ((Uint8*) &bgcolor)[0];
                    dstpos[1] = ((Uint8*) &bgcolor)[1];
                    dstpos[2] = ((Uint8*) &bgcolor)[2];
                    dstpos += 3;
                }
                else
                {
                    Uint8* srcpos = (Uint8*)
                        (srcpix + ((dy >> 16) * srcpitch) + ((dx >> 16) * 3));
                    dstpos[0] = srcpos[0];
                    dstpos[1] = srcpos[1];
                    dstpos[2] = srcpos[2];
                    dstpos += 3;
                }
                dx += icos; dy += isin;
            }
            dstrow += dstpitch;
        }
        break;
    }
}

static void
stretch (SDL_Surface *src, SDL_Surface *dst)
{
    int looph, loopw;
	
    Uint8* srcrow = (Uint8*) src->pixels;
    Uint8* dstrow = (Uint8*) dst->pixels;

    int srcpitch = src->pitch;
    int dstpitch = dst->pitch;

    int dstwidth = dst->w;
    int dstheight = dst->h;
    int dstwidth2 = dst->w << 1;
    int dstheight2 = dst->h << 1;

    int srcwidth2 = src->w << 1;
    int srcheight2 = src->h << 1;

    int w_err, h_err = srcheight2 - dstheight2;


    switch (src->format->BytesPerPixel)
    {
    case 1:
        for (looph = 0; looph < dstheight; ++looph)
        {
            Uint8 *srcpix = (Uint8*)srcrow, *dstpix = (Uint8*)dstrow;
            w_err = srcwidth2 - dstwidth2;
            for (loopw = 0; loopw < dstwidth; ++ loopw)
            {
                *dstpix++ = *srcpix;
                while (w_err >= 0)
                {
                    ++srcpix;
                    w_err -= dstwidth2;
                }
                w_err += srcwidth2;
            }
            while (h_err >= 0)
            {
                srcrow += srcpitch;
                h_err -= dstheight2;
            }
            dstrow += dstpitch;
            h_err += srcheight2;
        }
        break;
    case 2:
        for (looph = 0; looph < dstheight; ++looph)
        {
            Uint16 *srcpix = (Uint16*)srcrow, *dstpix = (Uint16*)dstrow;
            w_err = srcwidth2 - dstwidth2;
            for (loopw = 0; loopw < dstwidth; ++ loopw)
            {
                *dstpix++ = *srcpix;
                while (w_err >= 0)
                {
                    ++srcpix;
                    w_err -= dstwidth2;
                }
                w_err += srcwidth2;
            }
            while (h_err >= 0)
            {
                srcrow += srcpitch;
                h_err -= dstheight2;
            }
            dstrow += dstpitch;
            h_err += srcheight2;
        }
        break;
    case 3:
        for (looph = 0; looph < dstheight; ++looph)
        {
            Uint8 *srcpix = (Uint8*)srcrow, *dstpix = (Uint8*)dstrow;
            w_err = srcwidth2 - dstwidth2;
            for (loopw = 0; loopw < dstwidth; ++ loopw)
            {
                dstpix[0] = srcpix[0];
                dstpix[1] = srcpix[1];
                dstpix[2] = srcpix[2];
                dstpix += 3;
                while (w_err >= 0)
                {
                    srcpix+=3;
                    w_err -= dstwidth2;
                }
                w_err += srcwidth2;
            }
            while (h_err >= 0)
            {
                srcrow += srcpitch;
                h_err -= dstheight2;
            }
            dstrow += dstpitch;
            h_err += srcheight2;
        }
        break;
    default: /*case 4:*/
        for (looph = 0; looph < dstheight; ++looph)
        {
            Uint32 *srcpix = (Uint32*)srcrow, *dstpix = (Uint32*)dstrow;
            w_err = srcwidth2 - dstwidth2;
            for (loopw = 0; loopw < dstwidth; ++ loopw)
            {
                *dstpix++ = *srcpix;
                while (w_err >= 0)
                {
                    ++srcpix;
                    w_err -= dstwidth2;
                }
                w_err += srcwidth2;
            }
            while (h_err >= 0)
            {
                srcrow += srcpitch;
                h_err -= dstheight2;
            }
            dstrow += dstpitch;
            h_err += srcheight2;
        }
        break;
    }
}

static PyObject*
surf_scale (PyObject* self, PyObject* arg)
{
    PyObject *surfobj, *surfobj2;
    SDL_Surface* surf, *newsurf;
    int width, height;
    surfobj2 = NULL;

    /*get all the arguments*/
    if (!PyArg_ParseTuple (arg, "O!(ii)|O!", &PySurface_Type, &surfobj, 
                           &width, &height, &PySurface_Type, &surfobj2))
        return NULL;

    if (width < 0 || height < 0)
        return RAISE (PyExc_ValueError, "Cannot scale to negative size");

    surf = PySurface_AsSurface (surfobj);
	
    if (!surfobj2)
    {
        newsurf = newsurf_fromsurf (surf, width, height);
        if (!newsurf)
            return NULL;
    }
    else
        newsurf = PySurface_AsSurface (surfobj2);

    /* check to see if the size is twice as big. */
    if (newsurf->w != width || newsurf->h != height)
        return RAISE (PyExc_ValueError, 
                      "Destination surface not the given width or height.");

    /* check to see if the format of the surface is the same. */
    if (surf->format->BytesPerPixel != newsurf->format->BytesPerPixel)
        return RAISE (PyExc_ValueError, 
                      "Source and destination surfaces need the same format.");

    if (width && height)
    {
        SDL_LockSurface (newsurf);
        PySurface_Lock (surfobj);
	
        Py_BEGIN_ALLOW_THREADS;
        stretch (surf, newsurf);
        Py_END_ALLOW_THREADS;

        PySurface_Unlock (surfobj);
        SDL_UnlockSurface (newsurf);
    }

    if (surfobj2)
    {
        Py_INCREF (surfobj2);
        return surfobj2;
    }
    else
        return PySurface_New (newsurf);
}

static PyObject*
surf_scale2x (PyObject* self, PyObject* arg)
{
    PyObject *surfobj, *surfobj2;
    SDL_Surface *surf;
    SDL_Surface *newsurf;
    int width, height;
    surfobj2 = NULL;
        
    /*get all the arguments*/
    if (!PyArg_ParseTuple (arg, "O!|O!", &PySurface_Type, &surfobj, 
                           &PySurface_Type, &surfobj2))
        return NULL;

    surf = PySurface_AsSurface (surfobj);

    /* if the second surface is not there, then make a new one. */

    if (!surfobj2)
    {
        width = surf->w * 2;
        height = surf->h * 2;

        newsurf = newsurf_fromsurf (surf, width, height);

        if (!newsurf)
            return NULL;
    }
    else
        newsurf = PySurface_AsSurface (surfobj2);

    /* check to see if the size is twice as big. */
    if (newsurf->w != (surf->w * 2) || newsurf->h != (surf->h * 2))
        return RAISE (PyExc_ValueError, "Destination surface not 2x bigger.");

    /* check to see if the format of the surface is the same. */
    if (surf->format->BytesPerPixel != newsurf->format->BytesPerPixel)
        return RAISE (PyExc_ValueError, 
                      "Source and destination surfaces need the same format.");

    SDL_LockSurface (newsurf);
    SDL_LockSurface (surf);

    Py_BEGIN_ALLOW_THREADS;
    scale2x (surf, newsurf);
    Py_END_ALLOW_THREADS;

    SDL_UnlockSurface (surf);
    SDL_UnlockSurface (newsurf);

    if (surfobj2)
    {
        Py_INCREF (surfobj2);
        return surfobj2;
    }
    else
        return PySurface_New (newsurf);
}

static PyObject*
surf_rotate (PyObject* self, PyObject* arg)
{
    PyObject *surfobj;
    SDL_Surface* surf, *newsurf;
    float angle;

    double radangle, sangle, cangle;
    double x, y, cx, cy, sx, sy;
    int nxmax,nymax;
    Uint32 bgcolor;

    /*get all the arguments*/
    if (!PyArg_ParseTuple (arg, "O!f", &PySurface_Type, &surfobj, &angle))
        return NULL;
    surf = PySurface_AsSurface (surfobj);

    if (surf->format->BytesPerPixel <= 0 || surf->format->BytesPerPixel > 4)
        return RAISE (PyExc_ValueError,
                      "unsupport Surface bit depth for transform");

    if (!(((int) angle) % 90))
    {
        PySurface_Lock (surfobj);

        Py_BEGIN_ALLOW_THREADS;
        newsurf = rotate90 (surf, (int) angle);
        Py_END_ALLOW_THREADS;

        PySurface_Unlock (surfobj);
        if (!newsurf)
            return NULL;
        return PySurface_New (newsurf);
    }

    radangle = angle*.01745329251994329;
    sangle = sin (radangle);
    cangle = cos (radangle);

    x = surf->w;
    y = surf->h;
    cx = cangle*x;
    cy = cangle*y;
    sx = sangle*x;
    sy = sangle*y;
    nxmax = (int) (MAX (MAX (MAX (fabs (cx + sy), fabs (cx - sy)),
                             fabs (-cx + sy)), fabs (-cx - sy)));
    nymax = (int) (MAX (MAX (MAX (fabs (sx + cy), fabs (sx - cy)),
                             fabs (-sx + cy)), fabs (-sx - cy)));

    newsurf = newsurf_fromsurf (surf, nxmax, nymax);
    if (!newsurf)
        return NULL;

    /* get the background color */
    if (surf->flags & SDL_SRCCOLORKEY)
        bgcolor = surf->format->colorkey;
    else
    {
        SDL_LockSurface (surf);
        switch (surf->format->BytesPerPixel)
        {
        case 1:
            bgcolor = *(Uint8*) surf->pixels;
            break;
        case 2:
            bgcolor = *(Uint16*) surf->pixels;
            break;
        case 4:
            bgcolor = *(Uint32*) surf->pixels;
            break;
        default: /*case 3:*/
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
            bgcolor = (((Uint8*) surf->pixels)[0]) +
                (((Uint8*) surf->pixels)[1] << 8) +
                (((Uint8*) surf->pixels)[2] << 16);
#else
            bgcolor = (((Uint8*) surf->pixels)[2]) +
                (((Uint8*) surf->pixels)[1] << 8) +
                (((Uint8*) surf->pixels)[0] << 16);
#endif
        }
        SDL_UnlockSurface (surf);
        bgcolor &= ~surf->format->Amask;
    }

    SDL_LockSurface (newsurf);
    PySurface_Lock (surfobj);

    Py_BEGIN_ALLOW_THREADS;
    rotate (surf, newsurf, bgcolor, sangle, cangle);
    Py_END_ALLOW_THREADS;

    PySurface_Unlock (surfobj);
    SDL_UnlockSurface (newsurf);

    return PySurface_New (newsurf);
}

static PyObject*
surf_flip (PyObject* self, PyObject* arg)
{
    PyObject *surfobj;
    SDL_Surface* surf, *newsurf;
    int xaxis, yaxis;
    int loopx, loopy;
    int pixsize, srcpitch, dstpitch;
    Uint8 *srcpix, *dstpix;

    /*get all the arguments*/
    if (!PyArg_ParseTuple (arg, "O!ii", &PySurface_Type, &surfobj,
                           &xaxis, &yaxis))
        return NULL;
    surf = PySurface_AsSurface (surfobj);

    newsurf = newsurf_fromsurf (surf, surf->w, surf->h);
    if (!newsurf)
        return NULL;

    pixsize = surf->format->BytesPerPixel;
    srcpitch = surf->pitch;
    dstpitch = newsurf->pitch;

    SDL_LockSurface (newsurf);
    PySurface_Lock (surfobj);

    srcpix = (Uint8*) surf->pixels;
    dstpix = (Uint8*) newsurf->pixels;


    Py_BEGIN_ALLOW_THREADS;

    if (!xaxis)
    {
        if (!yaxis)
        {
            for (loopy = 0; loopy < surf->h; ++loopy)
                memcpy (dstpix + loopy * dstpitch, srcpix + loopy * srcpitch,
                        surf->w * surf->format->BytesPerPixel);
            }
            else
            {
                for (loopy = 0; loopy < surf->h; ++loopy)
                    memcpy (dstpix + loopy * dstpitch,
                            srcpix + (surf->h - 1 - loopy) * srcpitch,
                            surf->w * surf->format->BytesPerPixel);
            }
	}
	else /*if (xaxis)*/
	{
            if (yaxis)
            {
                switch (surf->format->BytesPerPixel)
                {
                case 1:
                    for (loopy = 0; loopy < surf->h; ++loopy)
                    {
                        Uint8* dst = (Uint8*) (dstpix + loopy * dstpitch);
                        Uint8* src = ((Uint8*) (srcpix + (surf->h - 1 - loopy)
                                                * srcpitch)) + surf->w - 1;
                        for (loopx = 0; loopx < surf->w; ++loopx)
                            *dst++ = *src--;
                    }
                    break;
                case 2:
                    for (loopy = 0; loopy < surf->h; ++loopy)
                    {
                        Uint16* dst = (Uint16*) (dstpix + loopy * dstpitch);
                        Uint16* src = ((Uint16*)
                                       (srcpix + (surf->h - 1 - loopy)
                                        * srcpitch)) + surf->w - 1;
                        for (loopx = 0; loopx < surf->w; ++loopx)
                            *dst++ = *src--;
                    }
                    break;
                case 4:
                    for (loopy = 0; loopy < surf->h; ++loopy)
                    {
                        Uint32* dst = (Uint32*) (dstpix + loopy * dstpitch);
                        Uint32* src = ((Uint32*)
                                       (srcpix + (surf->h - 1 - loopy)
                                        * srcpitch)) + surf->w - 1;
                        for (loopx = 0; loopx < surf->w; ++loopx)
                            *dst++ = *src--;
                    }
                    break;
                case 3:
                    for (loopy = 0; loopy < surf->h; ++loopy)
                    {
                        Uint8* dst = (Uint8*) (dstpix + loopy * dstpitch);
                        Uint8* src = ((Uint8*) (srcpix + (surf->h - 1 - loopy)
                                                * srcpitch)) + surf->w * 3 - 3;
                        for (loopx = 0; loopx < surf->w; ++loopx)
                        {
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
            else
            {
                switch (surf->format->BytesPerPixel)
                {
                case 1:
                    for (loopy = 0; loopy < surf->h; ++loopy)
                    {
                        Uint8* dst = (Uint8*) (dstpix + loopy * dstpitch);
                        Uint8* src = ((Uint8*) (srcpix + loopy * srcpitch)) +
                            surf->w - 1;
                        for (loopx = 0; loopx < surf->w; ++loopx)
                            *dst++ = *src--;
                    }
                    break;
                case 2:
                    for (loopy = 0; loopy < surf->h; ++loopy)
                    {
                        Uint16* dst = (Uint16*) (dstpix + loopy * dstpitch);
                        Uint16* src = ((Uint16*) (srcpix + loopy * srcpitch))
                            + surf->w - 1;
                        for (loopx = 0; loopx < surf->w; ++loopx)
                            *dst++ = *src--;
                    }
                    break;
                case 4:
                    for (loopy = 0; loopy < surf->h; ++loopy)
                    {
                        Uint32* dst = (Uint32*) (dstpix + loopy * dstpitch);
                        Uint32* src = ((Uint32*) (srcpix + loopy * srcpitch))
                            + surf->w - 1;
                        for (loopx = 0; loopx < surf->w; ++loopx)
                            *dst++ = *src--;
                    }
                    break;
                case 3:
                    for (loopy = 0; loopy < surf->h; ++loopy)
                    {
                        Uint8* dst = (Uint8*) (dstpix + loopy * dstpitch);
                        Uint8* src = ((Uint8*) (srcpix + loopy * srcpitch))
                            + surf->w * 3 - 3;
                        for (loopx = 0; loopx < surf->w; ++loopx)
                        {
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
    
    PySurface_Unlock (surfobj);
    SDL_UnlockSurface (newsurf);
    return PySurface_New (newsurf);
}

static PyObject*
surf_rotozoom (PyObject* self, PyObject* arg)
{
    PyObject *surfobj;
    SDL_Surface *surf, *newsurf, *surf32;
    float scale, angle;

    /*get all the arguments*/
    if (!PyArg_ParseTuple (arg, "O!ff", &PySurface_Type, &surfobj, &angle,
                           &scale))
        return NULL;
    surf = PySurface_AsSurface (surfobj);
    if (scale == 0.0)
    {
        newsurf = newsurf_fromsurf (surf, surf->w, surf->h);
        return PySurface_New (newsurf);
    }

    if (surf->format->BitsPerPixel == 32)
    {
        surf32 = surf;
        PySurface_Lock (surfobj);
    }
    else
    {
        Py_BEGIN_ALLOW_THREADS;
        surf32 = SDL_CreateRGBSurface (SDL_SWSURFACE, surf->w, surf->h, 32,
                                       0x000000ff, 0x0000ff00, 0x00ff0000,
                                       0xff000000);
        SDL_BlitSurface (surf, NULL, surf32, NULL);
        Py_END_ALLOW_THREADS;
    }

    Py_BEGIN_ALLOW_THREADS;
    newsurf = rotozoomSurface (surf32, angle, scale, 1);
    Py_END_ALLOW_THREADS;

    if (surf32 == surf)
        PySurface_Unlock (surfobj);
    else
        SDL_FreeSurface (surf32);
    return PySurface_New (newsurf);
}

static SDL_Surface*
chop (SDL_Surface *src, int x, int y, int width, int height)
{
    SDL_Surface* dst;
    int dstwidth,dstheight;
    char *srcpix, *dstpix, *srcrow, *dstrow;
    int srcstepx, srcstepy, dststepx, dststepy;
    int loopx,loopy;

    if ((x + width) > src->w)
        width = src->w - x;
    if ((y + height) > src->h)
        height=src->h - y;
    if (x < 0)
    {
        width -= (-x);
        x = 0;
    }
    if (y < 0)
    {
        height -= (-y);
        y = 0;
    }

    dstwidth = src->w-width;
    dstheight = src->h-height;

    dst = newsurf_fromsurf (src,dstwidth,dstheight);
    if (!dst)
        return NULL;
    SDL_LockSurface (dst);
    srcrow = (char*) src->pixels;
    dstrow = (char*) dst->pixels;
    srcstepx = dststepx = src->format->BytesPerPixel;
    srcstepy = src->pitch;
    dststepy = dst->pitch;

    for (loopy = 0; loopy < src->h; loopy++)
    {
        if ((loopy < y) || (loopy >= (y + height)))
	{
            dstpix = dstrow;
            srcpix = srcrow;
            for (loopx = 0; loopx < src->w; loopx++)
	    {
                if ((loopx < x) || (loopx >= (x + width)))
		{
                    switch (src->format->BytesPerPixel)
		    {
		    case 1:
                        *dstpix = *srcpix;
                        break;
		    case 2:
                        *(Uint16*) dstpix = *(Uint16*) srcpix;
                        break;
		    case 3:
                        dstpix[0] = srcpix[0];
                        dstpix[1] = srcpix[1];
                        dstpix[2] = srcpix[2];    
                        break;
		    case 4:
                        *(Uint32*) dstpix = *(Uint32*) srcpix;
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
    SDL_UnlockSurface (dst);
    return dst;
}

static PyObject*
surf_chop (PyObject* self, PyObject* arg)
{
    PyObject *surfobj, *rectobj;
    SDL_Surface* surf, *newsurf;
    GAME_Rect* rect, temp;
	
    if (!PyArg_ParseTuple (arg, "O!O", &PySurface_Type, &surfobj, &rectobj))
        return NULL;
    if (!(rect = GameRect_FromObject (rectobj, &temp)))
        return RAISE (PyExc_TypeError, "Rect argument is invalid");

    surf=PySurface_AsSurface (surfobj);
    Py_BEGIN_ALLOW_THREADS;
    newsurf = chop (surf, rect->x, rect->y, rect->w, rect->h);
    Py_END_ALLOW_THREADS;

    return PySurface_New (newsurf);
}




/*
 * smooth scale functions.
 */


/* this function implements an area-averaging shrinking filter in the X-dimension */
static void filter_shrink_X_ONLYC(Uint8 *srcpix, Uint8 *dstpix, int height, int srcpitch, int dstpitch, int srcwidth, int dstwidth)
{
    int srcdiff = srcpitch - (srcwidth * 4);
    int dstdiff = dstpitch - (dstwidth * 4);
    int x, y;

    int xspace = 0x10000 * srcwidth / dstwidth; /* must be > 1 */
    int xrecip = (int) ((long long) 0x100000000 / xspace);
    for (y = 0; y < height; y++)
    {
        Uint16 accumulate[4] = {0,0,0,0};
        int xcounter = xspace;
        for (x = 0; x < srcwidth; x++)
        {
            if (xcounter > 0x10000)
            {
                accumulate[0] += (Uint16) *srcpix++;
                accumulate[1] += (Uint16) *srcpix++;
                accumulate[2] += (Uint16) *srcpix++;
                accumulate[3] += (Uint16) *srcpix++;
                xcounter -= 0x10000;
            }
            else
            {
                /* write out a destination pixel */
                *dstpix++ = (Uint8) (((accumulate[0] + ((srcpix[0] * xcounter) >> 16)) * xrecip) >> 16);
                *dstpix++ = (Uint8) (((accumulate[1] + ((srcpix[1] * xcounter) >> 16)) * xrecip) >> 16);
                *dstpix++ = (Uint8) (((accumulate[2] + ((srcpix[2] * xcounter) >> 16)) * xrecip) >> 16);
                *dstpix++ = (Uint8) (((accumulate[3] + ((srcpix[3] * xcounter) >> 16)) * xrecip) >> 16);
                /* reload the accumulator with the remainder of this pixel */
                int xfrac = 0x10000 - xcounter;
                accumulate[0] = (Uint16) ((*srcpix++ * xfrac) >> 16);
                accumulate[1] = (Uint16) ((*srcpix++ * xfrac) >> 16);
                accumulate[2] = (Uint16) ((*srcpix++ * xfrac) >> 16);
                accumulate[3] = (Uint16) ((*srcpix++ * xfrac) >> 16);
                xcounter = xspace - xfrac;
            }
        }
        srcpix += srcdiff;
        dstpix += dstdiff;
    }
}

/* this function implements an area-averaging shrinking filter in the X-dimension */
static void filter_shrink_X_MMX(Uint8 *srcpix, Uint8 *dstpix, int height, int srcpitch, int dstpitch, int srcwidth, int dstwidth)
{
    int srcdiff = srcpitch - (srcwidth * 4);
    int dstdiff = dstpitch - (dstwidth * 4);
    int x, y;

    int xspace = 0x04000 * srcwidth / dstwidth; /* must be > 1 */
    int xrecip = (int) ((long long) 0x040000000 / xspace);
    long long One64 = 0x4000400040004000;
#if defined(__GNUC__) && defined(__x86_64__)
    long long srcdiff64 = srcdiff;
    long long dstdiff64 = dstdiff;
    asm __volatile__(" /* MMX code for X-shrink area average filter */ "
        " pxor          %%mm0,      %%mm0;           "
        " movd             %6,      %%mm7;           " /* mm7 == xrecipmmx */
        " movq             %2,      %%mm6;           " /* mm6 = 2^14  */
        " pshufw    $0, %%mm7,      %%mm7;           "
        "1:                                          " /* outer Y-loop */
        " movl             %5,      %%ecx;           " /* ecx == xcounter */
        " pxor          %%mm1,      %%mm1;           " /* mm1 == accumulator */
        " movl             %4,      %%edx;           " /* edx == width */
        "2:                                          " /* inner X-loop */
        " cmpl        $0x4000,      %%ecx;           "
        " jbe              3f;                       "
        " movd           (%0),      %%mm2;           " /* mm2 = srcpix */
        " add              $4,         %0;           "
        " punpcklbw     %%mm0,      %%mm2;           "
        " paddw         %%mm2,      %%mm1;           " /* accumulator += srcpix */
        " subl        $0x4000,      %%ecx;           "
        " jmp              4f;                       "
        "3:                                          " /* prepare to output a pixel */
        " movd          %%ecx,      %%mm2;           "
        " movq          %%mm6,      %%mm3;           " /* mm3 = 2^14  */
        " pshufw    $0, %%mm2,      %%mm2;           "
        " movd           (%0),      %%mm4;           " /* mm4 = srcpix */
        " add              $4,         %0;           "
        " punpcklbw     %%mm0,      %%mm4;           "
        " psubw         %%mm2,      %%mm3;           " /* mm3 = xfrac */
        " psllw            $2,      %%mm4;           "
        " pmulhuw       %%mm4,      %%mm2;           " /* mm2 = (srcpix * xcounter >> 16) */
        " pmulhuw       %%mm4,      %%mm3;           " /* mm3 = (srcpix * xfrac) >> 16 */
        " paddw         %%mm1,      %%mm2;           "
        " movq          %%mm3,      %%mm1;           " /* accumulator = (srcpix * xfrac) >> 16 */
        " pmulhuw       %%mm7,      %%mm2;           "
        " packuswb      %%mm0,      %%mm2;           "
        " movd          %%mm2,       (%1);           "
        " add              %5,      %%ecx;           "
        " add              $4,         %1;           "
        " subl        $0x4000,      %%ecx;           "
        "4:                                          " /* tail of inner X-loop */
        " decl          %%edx;                       "
        " jne              2b;                       "
        " add              %7,         %0;           " /* srcpix += srcdiff */
        " add              %8,         %1;           " /* dstpix += dstdiff */
        " decl             %3;                       "
        " jne              1b;                       "
        " emms;                                      "
        : "+r"(srcpix), "+r"(dstpix)  /* outputs */
        : "m"(One64),   "m"(height), "m"(srcwidth),
          "m"(xspace),  "m"(xrecip), "m"(srcdiff64), "m"(dstdiff64)     /* inputs */
        : "%ecx","%edx"               /* clobbered */
        );
#elif defined(__GNUC__) && defined(__i386__)
    asm __volatile__(" /* MMX code for X-shrink area average filter */ "
        " pxor          %%mm0,      %%mm0;           "
        " movd             %6,      %%mm7;           " /* mm7 == xrecipmmx */
        " movq             %2,      %%mm6;           " /* mm6 = 2^14  */
        " pshufw    $0, %%mm7,      %%mm7;           "
        "1:                                          " /* outer Y-loop */
        " movl             %5,      %%ecx;           " /* ecx == xcounter */
        " pxor          %%mm1,      %%mm1;           " /* mm1 == accumulator */
        " movl             %4,      %%edx;           " /* edx == width */
        "2:                                          " /* inner X-loop */
        " cmpl        $0x4000,      %%ecx;           "
        " jbe              3f;                       "
        " movd           (%0),      %%mm2;           " /* mm2 = srcpix */
        " add              $4,         %0;           "
        " punpcklbw     %%mm0,      %%mm2;           "
        " paddw         %%mm2,      %%mm1;           " /* accumulator += srcpix */
        " subl        $0x4000,      %%ecx;           "
        " jmp              4f;                       "
        "3:                                          " /* prepare to output a pixel */
        " movd          %%ecx,      %%mm2;           "
        " movq          %%mm6,      %%mm3;           " /* mm3 = 2^14  */
        " pshufw    $0, %%mm2,      %%mm2;           "
        " movd           (%0),      %%mm4;           " /* mm4 = srcpix */
        " add              $4,         %0;           "
        " punpcklbw     %%mm0,      %%mm4;           "
        " psubw         %%mm2,      %%mm3;           " /* mm3 = xfrac */
        " psllw            $2,      %%mm4;           "
        " pmulhuw       %%mm4,      %%mm2;           " /* mm2 = (srcpix * xcounter >> 16) */
        " pmulhuw       %%mm4,      %%mm3;           " /* mm3 = (srcpix * xfrac) >> 16 */
        " paddw         %%mm1,      %%mm2;           "
        " movq          %%mm3,      %%mm1;           " /* accumulator = (srcpix * xfrac) >> 16 */
        " pmulhuw       %%mm7,      %%mm2;           "
        " packuswb      %%mm0,      %%mm2;           "
        " movd          %%mm2,       (%1);           "
        " add              %5,      %%ecx;           "
        " add              $4,         %1;           "
        " subl        $0x4000,      %%ecx;           "
        "4:                                          " /* tail of inner X-loop */
        " decl          %%edx;                       "
        " jne              2b;                       "
        " add              %7,         %0;           " /* srcpix += srcdiff */
        " add              %8,         %1;           " /* dstpix += dstdiff */
        " decl             %3;                       "
        " jne              1b;                       "
        " emms;                                      "
        : "+r"(srcpix), "+r"(dstpix)                   /* outputs */
        : "m"(One64),   "m"(height), "m"(srcwidth),
          "m"(xspace),  "m"(xrecip), "m"(srcdiff),  "m"(dstdiff)  /* input */
        : "%ecx","%edx"     /* clobbered */
        );
#endif
}

/* this function implements an area-averaging shrinking filter in the Y-dimension */
static void filter_shrink_Y_ONLYC(Uint8 *srcpix, Uint8 *dstpix, int width, int srcpitch, int dstpitch, int srcheight, int dstheight)
{
    Uint16 *templine;
    int srcdiff = srcpitch - (width * 4);
    int dstdiff = dstpitch - (width * 4);
    int x, y;

    /* allocate and clear a memory area for storing the accumulator line */
    templine = (Uint16 *) malloc(dstpitch * 2);
    if (templine == NULL) return;
    memset(templine, 0, dstpitch * 2);

    int yspace = 0x10000 * srcheight / dstheight; /* must be > 1 */
    int yrecip = (int) ((long long) 0x100000000 / yspace);
    int ycounter = yspace;
    for (y = 0; y < srcheight; y++)
    {
        Uint16 *accumulate = templine;
        if (ycounter > 0x10000)
        {
            for (x = 0; x < width; x++)
            {
                *accumulate++ += (Uint16) *srcpix++;
                *accumulate++ += (Uint16) *srcpix++;
                *accumulate++ += (Uint16) *srcpix++;
                *accumulate++ += (Uint16) *srcpix++;
            }
            ycounter -= 0x10000;
        }
        else
        {
            /* write out a destination line */
            for (x = 0; x < width; x++)
            {
                *dstpix++ = (Uint8) (((*accumulate++ + ((*srcpix++ * ycounter) >> 16)) * yrecip) >> 16);
                *dstpix++ = (Uint8) (((*accumulate++ + ((*srcpix++ * ycounter) >> 16)) * yrecip) >> 16);
                *dstpix++ = (Uint8) (((*accumulate++ + ((*srcpix++ * ycounter) >> 16)) * yrecip) >> 16);
                *dstpix++ = (Uint8) (((*accumulate++ + ((*srcpix++ * ycounter) >> 16)) * yrecip) >> 16);
            }
            dstpix += dstdiff;
            /* reload the accumulator with the remainder of this line */
            accumulate = templine;
            srcpix -= 4 * width;
            int yfrac = 0x10000 - ycounter;
            for (x = 0; x < width; x++)
            {
                *accumulate++ = (Uint16) ((*srcpix++ * yfrac) >> 16);
                *accumulate++ = (Uint16) ((*srcpix++ * yfrac) >> 16);
                *accumulate++ = (Uint16) ((*srcpix++ * yfrac) >> 16);
                *accumulate++ = (Uint16) ((*srcpix++ * yfrac) >> 16);
            }
            ycounter = yspace - yfrac;
        }
        srcpix += srcdiff;
    } /* for (int y = 0; y < srcheight; y++) */

    /* free the temporary memory */
    free(templine);
}

/* this function implements an area-averaging shrinking filter in the Y-dimension */
static void filter_shrink_Y_MMX(Uint8 *srcpix, Uint8 *dstpix, int width, int srcpitch, int dstpitch, int srcheight, int dstheight)
{
    Uint16 *templine;
    int srcdiff = srcpitch - (width * 4);
    int dstdiff = dstpitch - (width * 4);
    int x, y;

    /* allocate and clear a memory area for storing the accumulator line */
    templine = (Uint16 *) malloc(dstpitch * 2);
    if (templine == NULL) return;
    memset(templine, 0, dstpitch * 2);

    int yspace = 0x4000 * srcheight / dstheight; /* must be > 1 */
    int yrecip = (int) ((long long) 0x040000000 / yspace);
    long long One64 = 0x4000400040004000;
#if defined(__GNUC__) && defined(__x86_64__)
    long long srcdiff64 = srcdiff;
    long long dstdiff64 = dstdiff;
    asm __volatile__(" /* MMX code for Y-shrink area average filter */ "
        " movl             %5,      %%ecx;           " /* ecx == ycounter */
        " pxor          %%mm0,      %%mm0;           "
        " movd             %6,      %%mm7;           " /* mm7 == yrecipmmx */
        " pshufw    $0, %%mm7,      %%mm7;           "
        "1:                                          " /* outer Y-loop */
        " mov              %2,      %%rax;           " /* rax == accumulate */
        " cmpl        $0x4000,      %%ecx;           "
        " jbe              3f;                       "
        " movl             %4,      %%edx;           " /* edx == width */
        "2:                                          "
        " movd           (%0),      %%mm1;           "
        " add              $4,         %0;           "
        " movq        (%%rax),      %%mm2;           "
        " punpcklbw     %%mm0,      %%mm1;           "
        " paddw         %%mm1,      %%mm2;           "
        " movq          %%mm2,    (%%rax);           "
        " add              $8,      %%rax;           "
        " decl          %%edx;                       "
        " jne              2b;                       "
        " subl        $0x4000,      %%ecx;           "
        " jmp              6f;                       "
        "3:                                          " /* prepare to output a line */
        " movd          %%ecx,      %%mm1;           "
        " movl             %4,      %%edx;           " /* edx = width */
        " movq             %9,      %%mm6;           " /* mm6 = 2^14  */
        " pshufw    $0, %%mm1,      %%mm1;           "
        " psubw         %%mm1,      %%mm6;           " /* mm6 = yfrac */
        "4:                                          "
        " movd           (%0),      %%mm4;           " /* mm4 = srcpix */
        " add              $4,         %0;           "
        " punpcklbw     %%mm0,      %%mm4;           "
        " movq        (%%rax),      %%mm5;           " /* mm5 = accumulate */
        " movq          %%mm6,      %%mm3;           "
        " psllw            $2,      %%mm4;           "
        " pmulhuw       %%mm4,      %%mm3;           " /* mm3 = (srcpix * yfrac) >> 16 */
        " pmulhuw       %%mm1,      %%mm4;           " /* mm4 = (srcpix * ycounter >> 16) */
        " movq          %%mm3,    (%%rax);           "
        " paddw         %%mm5,      %%mm4;           "
        " add              $8,      %%rax;           "
        " pmulhuw       %%mm7,      %%mm4;           "
        " packuswb      %%mm0,      %%mm4;           "
        " movd          %%mm4,       (%1);           "
        " add              $4,         %1;           "
        " decl          %%edx;                       "
        " jne              4b;                       "
        " add              %8,         %1;           " /* dstpix += dstdiff */
        " addl             %5,      %%ecx;           "
        " subl        $0x4000,      %%ecx;           "
        "6:                                          " /* tail of outer Y-loop */
        " add              %7,         %0;           " /* srcpix += srcdiff */
        " decl             %3;                       "
        " jne              1b;                       "
        " emms;                                      "
        : "+r"(srcpix), "+r"(dstpix)    /* outputs */
        : "m"(templine),"m"(srcheight), "m"(width),     "m"(yspace),  
          "m"(yrecip),  "m"(srcdiff64), "m"(dstdiff64), "m"(One64)  /* input */
        : "%ecx","%edx","%rax"          /* clobbered */
        );
#elif defined(__GNUC__) && defined(__i386__)
    asm __volatile__(" /* MMX code for Y-shrink area average filter */ "
        " movl             %5,      %%ecx;           " /* ecx == ycounter */
        " pxor          %%mm0,      %%mm0;           "
        " movd             %6,      %%mm7;           " /* mm7 == yrecipmmx */
        " pshufw    $0, %%mm7,      %%mm7;           "
        "1:                                          " /* outer Y-loop */
        " movl             %2,      %%eax;           " /* rax == accumulate */
        " cmpl        $0x4000,      %%ecx;           "
        " jbe              3f;                       "
        " movl             %4,      %%edx;           " /* edx == width */
        "2:                                          "
        " movd           (%0),      %%mm1;           "
        " add              $4,         %0;           "
        " movq        (%%eax),      %%mm2;           "
        " punpcklbw     %%mm0,      %%mm1;           "
        " paddw         %%mm1,      %%mm2;           "
        " movq          %%mm2,    (%%eax);           "
        " add              $8,      %%eax;           "
        " decl          %%edx;                       "
        " jne              2b;                       "
        " subl        $0x4000,      %%ecx;           "
        " jmp              6f;                       "
        "3:                                          " /* prepare to output a line */
        " movd          %%ecx,      %%mm1;           "
        " movl             %4,      %%edx;           " /* edx = width */
        " movq             %9,      %%mm6;           " /* mm6 = 2^14  */
        " pshufw    $0, %%mm1,      %%mm1;           "
        " psubw         %%mm1,      %%mm6;           " /* mm6 = yfrac */
        "4:                                          "
        " movd           (%0),      %%mm4;           " /* mm4 = srcpix */
        " add              $4,         %0;           "
        " punpcklbw     %%mm0,      %%mm4;           "
        " movq        (%%eax),      %%mm5;           " /* mm5 = accumulate */
        " movq          %%mm6,      %%mm3;           "
        " psllw            $2,      %%mm4;           "
        " pmulhuw       %%mm4,      %%mm3;           " /* mm3 = (srcpix * yfrac) >> 16 */
        " pmulhuw       %%mm1,      %%mm4;           " /* mm4 = (srcpix * ycounter >> 16) */
        " movq          %%mm3,    (%%eax);           "
        " paddw         %%mm5,      %%mm4;           "
        " add              $8,      %%eax;           "
        " pmulhuw       %%mm7,      %%mm4;           "
        " packuswb      %%mm0,      %%mm4;           "
        " movd          %%mm4,       (%1);           "
        " add              $4,         %1;           "
        " decl          %%edx;                       "
        " jne              4b;                       "
        " add              %8,         %1;           " /* dstpix += dstdiff */
        " addl             %5,      %%ecx;           "
        " subl        $0x4000,      %%ecx;           "
        "6:                                          " /* tail of outer Y-loop */
        " add              %7,         %0;           " /* srcpix += srcdiff */
        " decl             %3;                       "
        " jne              1b;                       "
        " emms;                                      "
        : "+r"(srcpix),  "+r"(dstpix)     /* outputs */
        : "m"(templine), "m"(srcheight), "m"(width),  "m"(yspace),
          "m"(yrecip),   "m"(srcdiff),   "m"(dstdiff),"m"(One64)  /* input */
        : "%ecx","%edx","%eax"           /* clobbered */
        );

#endif

    /* free the temporary memory */
    free(templine);
}

/* this function implements a bilinear filter in the X-dimension */
static void filter_expand_X_ONLYC(Uint8 *srcpix, Uint8 *dstpix, int height, int srcpitch, int dstpitch, int srcwidth, int dstwidth)
{
    int dstdiff = dstpitch - (dstwidth * 4);
    int *xidx0, *xmult0, *xmult1;
    int x, y;

    /* Allocate memory for factors */
    xidx0 = malloc(dstwidth * 4);
    if (xidx0 == NULL) return;
    int factorwidth = 4;
    xmult0 = (int *) malloc(dstwidth * factorwidth);
    xmult1 = (int *) malloc(dstwidth * factorwidth);
    if (xmult0 == NULL || xmult1 == NULL)
    {
        free(xidx0);
        if (xmult0) free(xmult0);
        if (xmult1) free(xmult1);
    }

    /* Create multiplier factors and starting indices and put them in arrays */
    for (x = 0; x < dstwidth; x++)
    {
        xidx0[x] = x * (srcwidth - 1) / dstwidth;
        xmult1[x] = 0x10000 * ((x * (srcwidth - 1)) % dstwidth) / dstwidth;
        xmult0[x] = 0x10000 - xmult1[x];
    }

    /* Do the scaling in raster order so we don't trash the cache */
    for (y = 0; y < height; y++)
    {
        Uint8 *srcrow0 = srcpix + y * srcpitch;
        for (x = 0; x < dstwidth; x++)
        {
            Uint8 *src = srcrow0 + xidx0[x] * 4;
            int xm0 = xmult0[x];
            int xm1 = xmult1[x];
            *dstpix++ = (Uint8) (((src[0] * xm0) + (src[4] * xm1)) >> 16);
            *dstpix++ = (Uint8) (((src[1] * xm0) + (src[5] * xm1)) >> 16);
            *dstpix++ = (Uint8) (((src[2] * xm0) + (src[6] * xm1)) >> 16);
            *dstpix++ = (Uint8) (((src[3] * xm0) + (src[7] * xm1)) >> 16);
        }
        dstpix += dstdiff;
    }

    /* free memory */
    free(xidx0);
    free(xmult0);
    free(xmult1);
}

/* this function implements a bilinear filter in the X-dimension */
static void filter_expand_X_MMX(Uint8 *srcpix, Uint8 *dstpix, int height, int srcpitch, int dstpitch, int srcwidth, int dstwidth)
{
    int dstdiff = dstpitch - (dstwidth * 4);
    int *xidx0, *xmult0, *xmult1;
    int x, y;

    /* Allocate memory for factors */
    xidx0 = malloc(dstwidth * 4);
    if (xidx0 == NULL) return;
    int factorwidth = 8;
    xmult0 = (int *) malloc(dstwidth * factorwidth);
    xmult1 = (int *) malloc(dstwidth * factorwidth);
    if (xmult0 == NULL || xmult1 == NULL)
    {
        free(xidx0);
        if (xmult0) free(xmult0);
        if (xmult1) free(xmult1);
    }

    /* Create multiplier factors and starting indices and put them in arrays */
    for (x = 0; x < dstwidth; x++)
    {
        xidx0[x] = x * (srcwidth - 1) / dstwidth;
        int xm1 = 0x100 * ((x * (srcwidth - 1)) % dstwidth) / dstwidth;
        int xm0 = 0x100 - xm1;
        xmult1[x*2]   = xm1 | (xm1 << 16);
        xmult1[x*2+1] = xm1 | (xm1 << 16);
        xmult0[x*2]   = xm0 | (xm0 << 16);
        xmult0[x*2+1] = xm0 | (xm0 << 16);
    }

    /* Do the scaling in raster order so we don't trash the cache */
    for (y = 0; y < height; y++)
    {
        Uint8 *srcrow0 = srcpix + y * srcpitch;
        Uint8 *dstrow = dstpix + y * dstpitch;
        int *xm0 = xmult0;
        int *xm1 = xmult1;
        int *x0 = xidx0;
#if defined(__GNUC__) && defined(__x86_64__)
        asm __volatile__( " /* MMX code for inner loop of X bilinear filter */ "
             " movl             %5,      %%ecx;           "
             " pxor          %%mm0,      %%mm0;           "
             "1:                                          "
             " movsxl         (%3),      %%rax;           " /* get xidx0[x] */
             " add              $4,         %3;           "
             " movq           (%0),      %%mm1;           " /* load mult0 */
             " add              $8,         %0;           "
             " movq           (%1),      %%mm2;           " /* load mult1 */
             " add              $8,         %1;           "
             " movd   (%4,%%rax,4),      %%mm4;           "
             " movd  4(%4,%%rax,4),      %%mm5;           "
             " punpcklbw     %%mm0,      %%mm4;           "
             " punpcklbw     %%mm0,      %%mm5;           "
             " pmullw        %%mm1,      %%mm4;           "
             " pmullw        %%mm2,      %%mm5;           "
             " paddw         %%mm4,      %%mm5;           "
             " psrlw            $8,      %%mm5;           "
             " packuswb      %%mm0,      %%mm5;           "
             " movd          %%mm5,       (%2);           "
             " add              $4,         %2;           "
             " decl          %%ecx;                       "
             " jne              1b;                       "
             " emms;                                      "
             : "+r"(xm0),   "+r"(xm1), "+r"(dstrow), "+r"(x0) /* outputs */
             : "r"(srcrow0),"m"(dstwidth)  /* input */
             : "%ecx","%rax"                /* clobbered */
             );
#elif defined(__GNUC__) && defined(__i386__)
    	int width = dstwidth;
    	long long One64 = 0x0100010001000100;
        asm __volatile__( " /* MMX code for inner loop of X bilinear filter */ "
             " pxor          %%mm0,      %%mm0;           "
             " movq             %5,      %%mm7;           "
             "1:                                          "
             " movl           (%2),      %%eax;           " /* get xidx0[x] */
             " add              $4,         %2;           "
             " movq          %%mm7,      %%mm2;           "
             " movq           (%0),      %%mm1;           " /* load mult0 */
             " add              $8,         %0;           "
             " psubw         %%mm1,      %%mm2;           " /* load mult1 */
             " movd   (%4,%%eax,4),      %%mm4;           "
             " movd  4(%4,%%eax,4),      %%mm5;           "
             " punpcklbw     %%mm0,      %%mm4;           "
             " punpcklbw     %%mm0,      %%mm5;           "
             " pmullw        %%mm1,      %%mm4;           "
             " pmullw        %%mm2,      %%mm5;           "
             " paddw         %%mm4,      %%mm5;           "
             " psrlw            $8,      %%mm5;           "
             " packuswb      %%mm0,      %%mm5;           "
             " movd          %%mm5,       (%1);           "
             " add              $4,         %1;           "
             " decl             %3;                       "
             " jne              1b;                       "
             " emms;                                      "
             : "+r"(xm0),    "+r"(dstrow), "+r"(x0), "+m"(width)  /* outputs */
             : "S"(srcrow0), "m"(One64)    /* input */
             : "%eax"            /* clobbered */
             );
#endif
    }

    /* free memory */
    free(xidx0);
    free(xmult0);
    free(xmult1);
}

/* this function implements a bilinear filter in the Y-dimension */
static void filter_expand_Y_ONLYC(Uint8 *srcpix, Uint8 *dstpix, int width, int srcpitch, int dstpitch, int srcheight, int dstheight)
{
    int dstdiff = dstpitch - (width * 4);
    int x, y;

    for (y = 0; y < dstheight; y++)
    {
        int yidx0 = y * (srcheight - 1) / dstheight;
        Uint8 *srcrow0 = srcpix + yidx0 * srcpitch;
        Uint8 *srcrow1 = srcrow0 + srcpitch;
        int ymult1 = 0x10000 * ((y * (srcheight - 1)) % dstheight) / dstheight;
        int ymult0 = 0x10000 - ymult1;
        for (x = 0; x < width; x++)
        {
            *dstpix++ = (Uint8) (((*srcrow0++ * ymult0) + (*srcrow1++ * ymult1)) >> 16);
            *dstpix++ = (Uint8) (((*srcrow0++ * ymult0) + (*srcrow1++ * ymult1)) >> 16);
            *dstpix++ = (Uint8) (((*srcrow0++ * ymult0) + (*srcrow1++ * ymult1)) >> 16);
            *dstpix++ = (Uint8) (((*srcrow0++ * ymult0) + (*srcrow1++ * ymult1)) >> 16);
        }
    }
}

/* this function implements a bilinear filter in the Y-dimension */
static void filter_expand_Y_MMX(Uint8 *srcpix, Uint8 *dstpix, int width, int srcpitch, int dstpitch, int srcheight, int dstheight)
{
    int dstdiff = dstpitch - (width * 4);
    int x, y;

    for (y = 0; y < dstheight; y++)
    {
        int yidx0 = y * (srcheight - 1) / dstheight;
        Uint8 *srcrow0 = srcpix + yidx0 * srcpitch;
        Uint8 *srcrow1 = srcrow0 + srcpitch;
        int ymult1 = 0x0100 * ((y * (srcheight - 1)) % dstheight) / dstheight;
        int ymult0 = 0x0100 - ymult1;
        Uint8 *dstrow = dstpix + y * dstpitch;
#if defined(__GNUC__) && defined(__x86_64__)
        asm __volatile__( " /* MMX code for inner loop of Y bilinear filter */ "
             " movl          %5,      %%ecx;                      "
             " movd          %3,      %%mm1;                      "
             " movd          %4,      %%mm2;                      "
             " pxor       %%mm0,      %%mm0;                      "
             " pshufw      $0, %%mm1, %%mm1;                      "
             " pshufw      $0, %%mm2, %%mm2;                      "
             "1:                                                  "
             " movd        (%0),      %%mm4;                      "
             " add           $4,         %0;                      "
             " movd        (%1),      %%mm5;                      "
             " add           $4,         %1;                      "
             " punpcklbw  %%mm0,      %%mm4;                      "
             " punpcklbw  %%mm0,      %%mm5;                      "
             " pmullw     %%mm1,      %%mm4;                      "
             " pmullw     %%mm2,      %%mm5;                      "
             " paddw      %%mm4,      %%mm5;                      "
             " psrlw         $8,      %%mm5;                      "
             " packuswb   %%mm0,      %%mm5;                      "
             " movd       %%mm5,       (%2);                      "
             " add           $4,         %2;                      "
             " decl       %%ecx;                                  "
             " jne           1b;                                  "
             " emms;                                              "
             : "+r"(srcrow0), "+r"(srcrow1), "+r"(dstrow)   /* outputs */
             : "m"(ymult0),   "m"(ymult1),   "m"(width)    /* input */
             : "%ecx"         /* clobbered */
             );
#elif defined(__GNUC__) && defined(__i386__)
        asm __volatile__( " /* MMX code for inner loop of Y bilinear filter */ "
             " movl          %5,      %%eax;                      "
             " movd          %3,      %%mm1;                      "
             " movd          %4,      %%mm2;                      "
             " pxor       %%mm0,      %%mm0;                      "
             " pshufw      $0, %%mm1, %%mm1;                      "
             " pshufw      $0, %%mm2, %%mm2;                      "
             "1:                                                  "
             " movd        (%0),      %%mm4;                      "
             " add           $4,         %0;                      "
             " movd        (%1),      %%mm5;                      "
             " add           $4,         %1;                      "
             " punpcklbw  %%mm0,     %%mm4;                       "
             " punpcklbw  %%mm0,     %%mm5;                       "
             " pmullw     %%mm1,     %%mm4;                       "
             " pmullw     %%mm2,     %%mm5;                       "
             " paddw      %%mm4,     %%mm5;                       "
             " psrlw         $8,     %%mm5;                       "
             " packuswb   %%mm0,     %%mm5;                       "
             " movd       %%mm5,      (%2);                       "
             " add           $4,        %2;                       "
             " decl       %%eax;                                  "
             " jne           1b;                                  "
             " emms;                                              "
             : "+r"(srcrow0), "+r"(srcrow1),"+r"(dstrow)   /* no outputs */
             : "m"(ymult0),   "m"(ymult1),  "m"(width)    /* input */
             : "%eax"        /* clobbered */
             );
#endif
    }
}

static void convert_24_32(Uint8 *srcpix, int srcpitch, Uint8 *dstpix, int dstpitch, int width, int height)
{
    int srcdiff = srcpitch - (width * 3);
    int dstdiff = dstpitch - (width * 4);
    int x, y;

    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            *dstpix++ = *srcpix++;
            *dstpix++ = *srcpix++;
            *dstpix++ = *srcpix++;
            *dstpix++ = 0xff;
        }
        srcpix += srcdiff;
        dstpix += dstdiff;
    }
}

static void convert_32_24(Uint8 *srcpix, int srcpitch, Uint8 *dstpix, int dstpitch, int width, int height)
{
    int srcdiff = srcpitch - (width * 4);
    int dstdiff = dstpitch - (width * 3);
    int x, y;

    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            *dstpix++ = *srcpix++;
            *dstpix++ = *srcpix++;
            *dstpix++ = *srcpix++;
            srcpix++;
        }
        srcpix += srcdiff;
        dstpix += dstdiff;
    }
}

static void scalesmooth(SDL_Surface *src, SDL_Surface *dst)
{
	Uint8* srcpix = (Uint8*)src->pixels;
	Uint8* dstpix = (Uint8*)dst->pixels;
    Uint8* dst32 = NULL;
	int srcpitch = src->pitch;
	int dstpitch = dst->pitch;

	int srcwidth = src->w;
	int srcheight = src->h;
	int dstwidth = dst->w;
    int dstheight = dst->h;

    int bpp = src->format->BytesPerPixel;

    /* convert to 32-bit if necessary */
    if (bpp == 3)
    {
        int newpitch = srcwidth * 4;
        Uint8 *newsrc = (Uint8 *) malloc(newpitch * srcheight);
        if (!newsrc) return;
        convert_24_32(srcpix, srcpitch, newsrc, newpitch, srcwidth, srcheight);
        srcpix = newsrc;
        srcpitch = newpitch;
        /* create a destination buffer for the 32-bit result */
        dstpitch = dstwidth << 2;
        dst32 = (Uint8 *) malloc(dstpitch * dstheight);
        if (dst32 == NULL)
        {
            free(dst32);
            return;
        }
        dstpix = dst32;
    }

    /* Create a temporary processing buffer if we will be scaling both X and Y */
    Uint8 *temppix = NULL;
    int tempwidth=0, temppitch=0, tempheight=0;
    if (srcwidth != dstwidth && srcheight != dstheight)
    {
        tempwidth = dstwidth;
        temppitch = tempwidth << 2;
        tempheight = srcheight;
        temppix = (Uint8 *) malloc(temppitch * tempheight);
        if (temppix == NULL)
        {
            if (bpp == 3)
            {
                free(srcpix);
                free(dstpix);
            }
            return;
        }
    }

#if defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__)) /* MMX routines will only compile in GCC */
    if (SDL_HasMMX())
    {
        /* Start the filter by doing X-scaling */
        if (dstwidth < srcwidth) /* shrink */
        {
            if (srcheight != dstheight)
                filter_shrink_X_MMX(srcpix, temppix, srcheight, srcpitch, temppitch, srcwidth, dstwidth);
            else
                filter_shrink_X_MMX(srcpix, dstpix, srcheight, srcpitch, dstpitch, srcwidth, dstwidth);
        }
        else if (dstwidth > srcwidth) /* expand */
        {
            if (srcheight != dstheight)
                filter_expand_X_MMX(srcpix, temppix, srcheight, srcpitch, temppitch, srcwidth, dstwidth);
            else
                filter_expand_X_MMX(srcpix, dstpix, srcheight, srcpitch, dstpitch, srcwidth, dstwidth);
        }
        /* Now do the Y scale */
        if (dstheight < srcheight) /* shrink */
        {
            if (srcwidth != dstwidth)
                filter_shrink_Y_MMX(temppix, dstpix, tempwidth, temppitch, dstpitch, srcheight, dstheight);
            else
                filter_shrink_Y_MMX(srcpix, dstpix, srcwidth, srcpitch, dstpitch, srcheight, dstheight);
        }
        else if (dstheight > srcheight)  /* expand */
        {
            if (srcwidth != dstwidth)
                filter_expand_Y_MMX(temppix, dstpix, tempwidth, temppitch, dstpitch, srcheight, dstheight);
            else
                filter_expand_Y_MMX(srcpix, dstpix, srcwidth, srcpitch, dstpitch, srcheight, dstheight);
        }
    }
    else
#endif
    { /* No MMX -- use the C versions */
        /* Start the filter by doing X-scaling */
        if (dstwidth < srcwidth) /* shrink */
        {
            if (srcheight != dstheight)
                filter_shrink_X_ONLYC(srcpix, temppix, srcheight, srcpitch, temppitch, srcwidth, dstwidth);
            else
                filter_shrink_X_ONLYC(srcpix, dstpix, srcheight, srcpitch, dstpitch, srcwidth, dstwidth);
        }
        else if (dstwidth > srcwidth) /* expand */
        {
            if (srcheight != dstheight)
                filter_expand_X_ONLYC(srcpix, temppix, srcheight, srcpitch, temppitch, srcwidth, dstwidth);
            else
                filter_expand_X_ONLYC(srcpix, dstpix, srcheight, srcpitch, dstpitch, srcwidth, dstwidth);
        }
        /* Now do the Y scale */
        if (dstheight < srcheight) /* shrink */
        {
            if (srcwidth != dstwidth)
                filter_shrink_Y_ONLYC(temppix, dstpix, tempwidth, temppitch, dstpitch, srcheight, dstheight);
            else
                filter_shrink_Y_ONLYC(srcpix, dstpix, srcwidth, srcpitch, dstpitch, srcheight, dstheight);
        }
        else if (dstheight > srcheight)  /* expand */
        {
            if (srcwidth != dstwidth)
                filter_expand_Y_ONLYC(temppix, dstpix, tempwidth, temppitch, dstpitch, srcheight, dstheight);
            else
                filter_expand_Y_ONLYC(srcpix, dstpix, srcwidth, srcpitch, dstpitch, srcheight, dstheight);
        }
    }

    /* Convert back to 24-bit if necessary */
    if (bpp == 3)
    {
        convert_32_24(dst32, dstpitch, (Uint8*)dst->pixels, dst->pitch, dstwidth, dstheight);
        free(dst32);
        dst32 = NULL;
        free(srcpix);
        srcpix = NULL;
    }
    /* free temporary buffer if necessary */
    if (temppix != NULL) free(temppix);

}


static PyObject* surf_scalesmooth(PyObject* self, PyObject* arg)
{
	PyObject *surfobj, *surfobj2;
	SDL_Surface* surf, *newsurf;
	int width, height, bpp;
    surfobj2 = NULL;

    /*get all the arguments*/
    if (!PyArg_ParseTuple (arg, "O!(ii)|O!", &PySurface_Type, &surfobj, 
                           &width, &height, &PySurface_Type, &surfobj2))
        return NULL;

    if (width < 0 || height < 0)
        return RAISE (PyExc_ValueError, "Cannot scale to negative size");

    surf = PySurface_AsSurface (surfobj);

    bpp = surf->format->BytesPerPixel;
    if(bpp < 3 || bpp > 4)
		return RAISE(PyExc_ValueError, "Only 24-bit or 32-bit surfaces can be smoothly scaled");

	
    if (!surfobj2)
    {
        newsurf = newsurf_fromsurf (surf, width, height);
        if (!newsurf)
            return NULL;
    }
    else
        newsurf = PySurface_AsSurface (surfobj2);

    /* check to see if the size is twice as big. */
    if (newsurf->w != width || newsurf->h != height)
        return RAISE (PyExc_ValueError, 
                      "Destination surface not the given width or height.");


    if(((width * bpp + 3) >> 2) > newsurf->pitch)
        return RAISE(PyExc_ValueError, "SDL Error: destination surface pitch not 4-byte aligned.");

	
    if(width && height)
    {
        SDL_LockSurface(newsurf);
        PySurface_Lock(surfobj);
        Py_BEGIN_ALLOW_THREADS;

        /* handle trivial case */
        if (surf->w == width && surf->h == height) {
            int y;
            for (y = 0; y < height; y++) {
                memcpy(newsurf->pixels + y * newsurf->pitch, 
                       surf->pixels + y * surf->pitch, width * bpp);
            }
        }
        else {
            scalesmooth(surf, newsurf);
        }
        Py_END_ALLOW_THREADS;

        PySurface_Unlock(surfobj);
        SDL_UnlockSurface(newsurf);
    }

    if (surfobj2)
    {
        Py_INCREF (surfobj2);
        return surfobj2;
    }
    else
        return PySurface_New (newsurf);

}








static PyMethodDef transform_builtins[] =
{
    { "scale", surf_scale, METH_VARARGS, DOC_PYGAMETRANSFORMSCALE },
    { "rotate", surf_rotate, METH_VARARGS, DOC_PYGAMETRANSFORMROTATE },
    { "flip", surf_flip, METH_VARARGS, DOC_PYGAMETRANSFORMFLIP },
    { "rotozoom", surf_rotozoom, METH_VARARGS, DOC_PYGAMETRANSFORMROTOZOOM},
    { "chop", surf_chop, METH_VARARGS, DOC_PYGAMETRANSFORMCHOP },
    { "scale2x", surf_scale2x, METH_VARARGS, DOC_PYGAMETRANSFORMSCALE2X },
    { "smoothscale", surf_scalesmooth, METH_VARARGS, DOC_PYGAMETRANSFORMSMOOTHSCALE },
    
    { NULL, NULL, 0, NULL }
};

PYGAME_EXPORT
void inittransform (void)
{
    PyObject *module;
    module = Py_InitModule3 ("transform", transform_builtins,
                             DOC_PYGAMETRANSFORM);

    /*imported needed apis*/
    import_pygame_base ();
    import_pygame_rect ();
    import_pygame_surface ();
}
