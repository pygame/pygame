/*
  pygame - Python Game Library
  Copyright (C) 2000-2001  Pete Shinners, 2007 Rene Dudfield, Richard Goedeken

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
#include <math.h>
#include "pgdefines.h"
#include "transform.h"
#include "filters.h"
#include "surface.h"

#define LAPLACIAN_NUM 0xFFFFFFFF
#define READINT24(x) ((x)[0]<<16 | (x)[1]<<8 | (x)[2]) 
#define WRITEINT24(x, i)  {(x)[0]=i>>16; (x)[1]=(i>>8)&0xff; x[2]=i&0xff; }

#define COLOR_R(color,format) \
    (((color & format->Rmask) >> format->Rshift) << format->Rloss)

#define COLOR_G(color,format) \
    (((color & format->Gmask) >> format->Gshift) << format->Gloss)

#define COLOR_B(color,format) \
    (((color & format->Bmask) >> format->Bshift) << format->Bloss)

#define DIFF_COLOR_R(color1,format1,color2,format2)                     \
    abs(COLOR_R(color1, format1) - COLOR_R(color2,format2))

#define DIFF_COLOR_G(color1,format1,color2,format2)                     \
    abs(COLOR_G(color1, format1) - COLOR_G(color2,format2))

#define DIFF_COLOR_B(color1,format1,color2,format2)                     \
    abs(COLOR_B(color1, format1) - COLOR_B(color2,format2))

static void
_convert_24_32 (Uint8 *srcpix, int srcpitch, Uint8 *dstpix, int dstpitch,
    int width, int height);
static void
_convert_32_24 (Uint8 *srcpix, int srcpitch, Uint8 *dstpix, int dstpitch,
    int width, int height);
static SDL_Surface* _surface_fromsurface (SDL_Surface *surface, int width,
    int height);

static void
_convert_24_32 (Uint8 *srcpix, int srcpitch, Uint8 *dstpix, int dstpitch,
    int width, int height)
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

static void
_convert_32_24 (Uint8 *srcpix, int srcpitch, Uint8 *dstpix, int dstpitch,
    int width, int height)
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

static SDL_Surface*
_surface_fromsurface (SDL_Surface *surface, int width, int height)
{
    SDL_Surface *newsurface;
    int retval;

    if (!surface)
    {
        SDL_SetError ("surface argument NULL");
        return NULL;
    }

    newsurface = SDL_CreateRGBSurface (surface->flags, width, height,
        surface->format->BitsPerPixel, surface->format->Rmask,
        surface->format->Gmask, surface->format->Bmask, surface->format->Amask);
    if (!newsurface)
        return NULL;

    /* Copy palette, colorkey, etc info */
    if (surface->format->BytesPerPixel == 1 && surface->format->palette)
        SDL_SetColors (newsurface, surface->format->palette->colors, 0,
            surface->format->palette->ncolors);

    if (surface->flags & SDL_SRCCOLORKEY)
    {
        retval = SDL_SetColorKey (newsurface,
            (surface->flags & SDL_RLEACCEL) | SDL_SRCCOLORKEY,
            surface->format->colorkey);
        if (retval == -1)
        {
            SDL_FreeSurface (newsurface);
            return NULL;
        }
    }

    if (surface->flags & SDL_SRCALPHA)
    {
        retval = SDL_SetAlpha (newsurface, surface->flags,
            surface->format->alpha);
        if (retval == -1)
        {
            SDL_FreeSurface (newsurface);
            return NULL;
        }
    }

    return newsurface;
}

SDL_Surface*
pyg_transform_rotate90 (SDL_Surface *surface, int angle)
{
    int dstwidth, dstheight;
    SDL_Surface* dstsurface;
    Uint8 *srcpix, *dstpix, *srcrow, *dstrow;
    int srcstepx, srcstepy, dststepx, dststepy;
    int loopx, loopy;
    int turns;

    if (!surface)
    {
        SDL_SetError ("surface argument NULL");
        return NULL;
    }

    turns = (angle / 90) % 4;
    if (turns == 0) /* No changes, return a plain copy. */
        return SDL_ConvertSurface (surface, surface->format, surface->flags);
    
    if (turns < 0)
        turns = 4 + turns;
    if ((turns % 2) == 0)
    {
        dstwidth = surface->w;
        dstheight = surface->h;
    }
    else
    {
        dstwidth = surface->h;
        dstheight = surface->w;
    }

    dstsurface = _surface_fromsurface (surface, dstwidth, dstheight);
    if (!dstsurface)
        return NULL;
    
    if (SDL_MUSTLOCK (surface) && SDL_LockSurface (surface) == -1)
    {
        SDL_FreeSurface (dstsurface);
        return NULL;
    }
    if (SDL_MUSTLOCK (dstsurface) && SDL_LockSurface (dstsurface) == -1)
    {
        SDL_FreeSurface (dstsurface);
        UNLOCK_SURFACE (surface);
        return NULL;
    }

    srcrow = (Uint8*) surface->pixels;
    dstrow = (Uint8*) dstsurface->pixels;
    srcstepx = dststepx = surface->format->BytesPerPixel;
    srcstepy = surface->pitch;
    dststepy = dstsurface->pitch;

    switch (turns)
    {
        /*case 0: we don't need to change anything*/
    case 1:
        srcrow += ((surface->w - 1) * srcstepx);
        srcstepy = -srcstepx;
        srcstepx = surface->pitch;
        break;
    case 2:
        srcrow += ((surface->h - 1) * srcstepy) + ((surface->w - 1) * srcstepx);
        srcstepx = -srcstepx;
        srcstepy = -srcstepy;
        break;
    case 3:
        srcrow += ((surface->h - 1) * srcstepy);
        srcstepx = -srcstepy;
        srcstepy = surface->format->BytesPerPixel;
        break;
    }

    switch (surface->format->BytesPerPixel)
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
    UNLOCK_SURFACE (surface);
    UNLOCK_SURFACE (dstsurface);
    return dstsurface;
}

SDL_Surface*
pyg_transform_rotate (SDL_Surface *surface, double angle)
{
    SDL_Surface *newsurface;
    double sangle, cangle;
    double x, y, cx, cy, sx, sy;
    Uint32 bgcolor;
    Uint8 *srcpix, *dstrow;
    int srcpitch, dstpitch;
    int nxmax, nymax, xmaxval, ymaxval;
    int ax, ay, xd, yd, isin, icos, icy, dx, dy;

    if (!surface)
    {
        SDL_SetError ("surface argument NULL");
        return NULL;
    }

    sangle = sin (angle);
    cangle = cos (angle);

    x = surface->w;
    y = surface->h;
    cx = cangle * x;
    cy = cangle * y;
    sx = sangle * x;
    sy = sangle * y;
    nxmax = (int) (MAX (MAX (MAX (fabs (cx + sy), fabs (cx - sy)),
                fabs (-cx + sy)), fabs (-cx - sy)));
    nymax = (int) (MAX (MAX (MAX (fabs (sx + cy), fabs (sx - cy)),
                fabs (-sx + cy)), fabs (-sx - cy)));

    newsurface = _surface_fromsurface (surface, nxmax, nymax);
    if (!newsurface)
        return NULL;

    if (SDL_MUSTLOCK (surface) && SDL_LockSurface (surface) == -1)
    {
        SDL_FreeSurface (newsurface);
        return NULL;
    }
    if (SDL_MUSTLOCK (newsurface) && SDL_LockSurface (newsurface) == -1)
    {
        SDL_FreeSurface (newsurface);
        UNLOCK_SURFACE (surface);
        return NULL;
    }

    srcpix = (Uint8*) surface->pixels;
    dstrow = (Uint8*) newsurface->pixels;
    srcpitch = surface->pitch;
    dstpitch = newsurface->pitch;
    icy = newsurface->h / 2;
    xd = ((surface->w - newsurface->w) << 15);
    yd = ((surface->h - newsurface->h) << 15);
    isin = (int)(sangle * 65536);
    icos = (int)(cangle * 65536);
    ax = ((newsurface->w) << 15) - (int)(cangle * ((newsurface->w - 1) << 15));
    ay = ((newsurface->h) << 15) - (int)(sangle * ((newsurface->w - 1) << 15));
    xmaxval = ((surface->w) << 16) - 1;
    ymaxval = ((surface->h) << 16) - 1;
    
    /* Get the background color. */
    if (surface->flags & SDL_SRCCOLORKEY)
        bgcolor = surface->format->colorkey;
    else
    {
        switch (surface->format->BytesPerPixel)
        {
        case 1:
            bgcolor = *(Uint8*) surface->pixels;
            break;
        case 2:
            bgcolor = *(Uint16*) surface->pixels;
            break;
        case 4:
            bgcolor = *(Uint32*) surface->pixels;
            break;
        default: /*case 3:*/
            bgcolor = GET_PIXEL24 ((Uint8*) surface->pixels);
            break;
        }
        bgcolor &= ~surface->format->Amask;
    }

    switch (surface->format->BytesPerPixel)
    {
    case 1:
        for (y = 0; y < newsurface->h; y++)
        {
            Uint8 *dstpos = (Uint8*)dstrow;
            dx = (ax + (isin * (icy - y))) + xd;
            dy = (ay - (icos * (icy - y))) + yd;
            for (x = 0; x < newsurface->w; x++)
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
        for (y = 0; y < newsurface->h; y++)
        {
            Uint16 *dstpos = (Uint16*)dstrow;
            dx = (ax + (isin * (icy - y))) + xd;
            dy = (ay - (icos * (icy - y))) + yd;
            for (x = 0; x < newsurface->w; x++)
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
    case 3:
        for (y = 0; y < newsurface->h; y++)
        {
            Uint8 *dstpos = (Uint8*)dstrow;
            dx = (ax + (isin * (icy - y))) + xd;
            dy = (ay - (icos * (icy - y))) + yd;
            for (x = 0; x < newsurface->w; x++)
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
    default:
        for (y = 0; y < newsurface->h; y++)
        {
            Uint32 *dstpos = (Uint32*)dstrow;
            dx = (ax + (isin * (icy - y))) + xd;
            dy = (ay - (icos * (icy - y))) + yd;
            for (x = 0; x < newsurface->w; x++)
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
    }

    UNLOCK_SURFACE (surface);
    UNLOCK_SURFACE (newsurface);
    
    return newsurface;
}

SDL_Surface*
pyg_transform_scale (SDL_Surface *srcsurface, SDL_Surface *dstsurface,
    int width, int height)
{
    int looph, loopw;
    Uint8 *srcrow, *dstrow;
    int srcpitch, dstpitch;
    int dstwidth, dstheight, dstwidth2, dstheight2;
    int srcwidth, srcheight;
    int w_err, h_err;

    if (!srcsurface)
    {
        SDL_SetError ("srsurface argument NULL");
        return 0;
    }

    if (!dstsurface)
    {
        /* Duplicate source */
        dstsurface = _surface_fromsurface (srcsurface, width, height);
        if (!dstsurface)
            return NULL;
    }

    if (SDL_MUSTLOCK (srcsurface) && SDL_LockSurface (srcsurface) == -1)
    {
        SDL_FreeSurface (dstsurface);
        return NULL;
    }
    if (SDL_MUSTLOCK (dstsurface) && SDL_LockSurface (dstsurface) == -1)
    {
        SDL_FreeSurface (dstsurface);
        UNLOCK_SURFACE (srcsurface);
        return NULL;
    }

    srcrow = (Uint8*) srcsurface->pixels;
    dstrow = (Uint8*) dstsurface->pixels;
    srcpitch = srcsurface->pitch;
    dstpitch = dstsurface->pitch;
    dstwidth = dstsurface->w;
    dstheight = dstsurface->h;
    dstwidth2 = dstsurface->w << 1;
    dstheight2 = dstsurface->h << 1;
    srcwidth = srcsurface->w << 1;
    srcheight = srcsurface->h << 1;
    h_err = srcheight - dstheight2;
    
    switch (srcsurface->format->BytesPerPixel)
    {
    case 1:
        for (looph = 0; looph < dstheight; ++looph)
        {
            Uint8 *srcpix = (Uint8*)srcrow;
            Uint8 *dstpix = (Uint8*)dstrow;
            w_err = srcwidth - dstwidth2;
            for (loopw = 0; loopw < dstwidth; ++ loopw)
            {
                *dstpix++ = *srcpix;
                while (w_err >= 0)
                {
                    ++srcpix;
                    w_err -= dstwidth2;
                }
                w_err += srcwidth;
            }
            while (h_err >= 0)
            {
                srcrow += srcpitch;
                h_err -= dstheight2;
            }
            dstrow += dstpitch;
            h_err += srcheight;
        }
        break;
    case 2:
        for (looph = 0; looph < dstheight; ++looph)
        {
            Uint16 *srcpix = (Uint16*)srcrow;
            Uint16 *dstpix = (Uint16*)dstrow;
            w_err = srcwidth - dstwidth2;
            for (loopw = 0; loopw < dstwidth; ++ loopw)
            {
                *dstpix++ = *srcpix;
                while (w_err >= 0)
                {
                    ++srcpix;
                    w_err -= dstwidth2;
                }
                w_err += srcwidth;
            }
            while (h_err >= 0)
            {
                srcrow += srcpitch;
                h_err -= dstheight2;
            }
            dstrow += dstpitch;
            h_err += srcheight;
        }
        break;
    case 3:
        for (looph = 0; looph < dstheight; ++looph)
        {
            Uint8 *srcpix = (Uint8*)srcrow;
            Uint8 *dstpix = (Uint8*)dstrow;
            w_err = srcwidth - dstwidth2;
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
                w_err += srcwidth;
            }
            while (h_err >= 0)
            {
                srcrow += srcpitch;
                h_err -= dstheight2;
            }
            dstrow += dstpitch;
            h_err += srcheight;
        }
        break;
    default: /*case 4:*/
        for (looph = 0; looph < dstheight; ++looph)
        {
            Uint32 *srcpix = (Uint32*)srcrow;
            Uint32 *dstpix = (Uint32*)dstrow;
            w_err = srcwidth - dstwidth2;
            for (loopw = 0; loopw < dstwidth; ++ loopw)
            {
                *dstpix++ = *srcpix;
                while (w_err >= 0)
                {
                    ++srcpix;
                    w_err -= dstwidth2;
                }
                w_err += srcwidth;
            }
            while (h_err >= 0)
            {
                srcrow += srcpitch;
                h_err -= dstheight2;
            }
            dstrow += dstpitch;
            h_err += srcheight;
        }
        break;
    }

    UNLOCK_SURFACE (srcsurface);
    UNLOCK_SURFACE (dstsurface);

    return dstsurface;
}

SDL_Surface*
pyg_transform_flip (SDL_Surface *surface, int xaxis, int yaxis)
{
    int loopx, loopy;
    int srcpitch, dstpitch;
    Uint8 *srcpix, *dstpix;
    SDL_Surface *newsurface;

    if (!surface)
    {
        SDL_SetError ("surface argument NULL");
        return 0;
    }

    newsurface = _surface_fromsurface (surface, surface->w, surface->h);
    if (!newsurface)
        return NULL;

    if (SDL_MUSTLOCK (surface) && SDL_LockSurface (surface) == -1)
    {
        SDL_FreeSurface (newsurface);
        return NULL;
    }
    if (SDL_MUSTLOCK (newsurface) && SDL_LockSurface (newsurface) == -1)
    {
        SDL_FreeSurface (newsurface);
        UNLOCK_SURFACE (surface);
        return NULL;
    }

    srcpitch = surface->pitch;
    dstpitch = newsurface->pitch;
    srcpix = (Uint8*) surface->pixels;
    dstpix = (Uint8*) newsurface->pixels;

    if (!xaxis)
    {
        if (!yaxis)
        {
            /* Nothing happened. */
            UNLOCK_SURFACE (surface);
            UNLOCK_SURFACE (newsurface);
            return newsurface;
        }
        for (loopy = 0; loopy < surface->h; ++loopy)
            memcpy (dstpix + loopy * dstpitch,
                srcpix + (surface->h - 1 - loopy) * srcpitch,
                (size_t)(surface->w * surface->format->BytesPerPixel));
    }
    else
    {
        if (yaxis)
        {
            switch (surface->format->BytesPerPixel)
            {
            case 1:
                for (loopy = 0; loopy < surface->h; ++loopy)
                {
                    Uint8* dst = (Uint8*) (dstpix + loopy * dstpitch);
                    Uint8* src = ((Uint8*) (srcpix + (surface->h - 1 - loopy)
                            * srcpitch)) + surface->w - 1;
                    for (loopx = 0; loopx < surface->w; ++loopx)
                        *dst++ = *src--;
                }
                break;
            case 2:
                for (loopy = 0; loopy < surface->h; ++loopy)
                {
                    Uint16* dst = (Uint16*) (dstpix + loopy * dstpitch);
                    Uint16* src = ((Uint16*) (srcpix + (surface->h - 1 - loopy)
                            * srcpitch)) + surface->w - 1;
                    for (loopx = 0; loopx < surface->w; ++loopx)
                        *dst++ = *src--;
                }
                break;
            case 3:
                for (loopy = 0; loopy < surface->h; ++loopy)
                {
                    Uint8* dst = (Uint8*) (dstpix + loopy * dstpitch);
                    Uint8* src = ((Uint8*) (srcpix + (surface->h - 1 - loopy)
                            * srcpitch)) + surface->w * 3 - 3;
                    for (loopx = 0; loopx < surface->w; ++loopx)
                    {
                        dst[0] = src[0];
                        dst[1] = src[1];
                        dst[2] = src[2];
                        dst += 3;
                        src -= 3;
                    }
                }
                break;
            default:
                for (loopy = 0; loopy < surface->h; ++loopy)
                {
                    Uint32* dst = (Uint32*) (dstpix + loopy * dstpitch);
                    Uint32* src = ((Uint32*) (srcpix + (surface->h - 1 - loopy)
                            * srcpitch)) + surface->w - 1;
                    for (loopx = 0; loopx < surface->w; ++loopx)
                        *dst++ = *src--;
                }
                break;
            }
        } /* !yaxis */
        else
        {
            switch (surface->format->BytesPerPixel)
            {
            case 1:
                for (loopy = 0; loopy < surface->h; ++loopy)
                {
                    Uint8* dst = (Uint8*) (dstpix + loopy * dstpitch);
                    Uint8* src = ((Uint8*) (srcpix + loopy * srcpitch)) +
                        surface->w - 1;
                    for (loopx = 0; loopx < surface->w; ++loopx)
                        *dst++ = *src--;
                }
                break;
            case 2:
                for (loopy = 0; loopy < surface->h; ++loopy)
                {
                    Uint16* dst = (Uint16*) (dstpix + loopy * dstpitch);
                    Uint16* src = ((Uint16*) (srcpix + loopy * srcpitch))
                        + surface->w - 1;
                    for (loopx = 0; loopx < surface->w; ++loopx)
                        *dst++ = *src--;
                }
                break;
            case 3:
                for (loopy = 0; loopy < surface->h; ++loopy)
                {
                    Uint8* dst = (Uint8*) (dstpix + loopy * dstpitch);
                    Uint8* src = ((Uint8*) (srcpix + loopy * srcpitch))
                        + surface->w * 3 - 3;
                    for (loopx = 0; loopx < surface->w; ++loopx)
                    {
                        dst[0] = src[0];
                        dst[1] = src[1];
                        dst[2] = src[2];
                        dst += 3;
                        src -= 3;
                    }
                }
                break;
            default:
                for (loopy = 0; loopy < surface->h; ++loopy)
                {
                    Uint32* dst = (Uint32*) (dstpix + loopy * dstpitch);
                    Uint32* src = ((Uint32*) (srcpix + loopy * srcpitch))
                        + surface->w - 1;
                    for (loopx = 0; loopx < surface->w; ++loopx)
                        *dst++ = *src--;
                }
                break;
            }
        }
    }
    UNLOCK_SURFACE (surface);
    UNLOCK_SURFACE (newsurface);
    return newsurface;

}

SDL_Surface*
pyg_transform_chop (SDL_Surface *surface, SDL_Rect *rect)
{
    SDL_Surface *newsurface;
    int dstwidth,dstheight;
    Uint8 *srcpix, *dstpix, *srcrow, *dstrow;
    int srcstepx, srcstepy, dststepx, dststepy;
    int loopx, loopy;

    if (!surface)
    {
        SDL_SetError ("surface argument NULL");
        return NULL;
    }
    if (!rect)
    {
        SDL_SetError ("rect argument NULL");
        return NULL;
    }
    
    if ((rect->x + rect->w) > surface->w)
        rect->w = surface->w - rect->x;
    if ((rect->y + rect->h) > surface->h)
        rect->h = surface->h - rect->y;
    if (rect->x < 0)
    {
        rect->w -= (-rect->x);
        rect->x = 0;
    }
    if (rect->y < 0)
    {
        rect->h -= (-rect->y);
        rect->y = 0;
    }

    dstwidth = surface->w - rect->w;
    dstheight = surface->h - rect->h;

    newsurface = _surface_fromsurface (surface, dstwidth, dstheight);
    if (!newsurface)
        return NULL;

    if (SDL_MUSTLOCK (surface) && SDL_LockSurface (surface) == -1)
    {
        SDL_FreeSurface (newsurface);
        return NULL;
    }
    if (SDL_MUSTLOCK (newsurface) && SDL_LockSurface (newsurface) == -1)
    {
        SDL_FreeSurface (newsurface);
        UNLOCK_SURFACE (surface);
        return NULL;
    }
    
    srcrow = (Uint8*) surface->pixels;
    dstrow = (Uint8*) newsurface->pixels;
    srcstepx = dststepx = surface->format->BytesPerPixel;
    srcstepy = surface->pitch;
    dststepy = newsurface->pitch;

    for (loopy = 0; loopy < surface->h; loopy++)
    {
        if ((loopy < rect->y) || (loopy >= (rect->y + rect->h)))
        {
            dstpix = dstrow;
            srcpix = srcrow;
            for (loopx = 0; loopx < surface->w; loopx++)
            {
                if ((loopx < rect->x) || (loopx >= (rect->x + rect->w)))
                {
                    switch (surface->format->BytesPerPixel)
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

    UNLOCK_SURFACE (surface);
    UNLOCK_SURFACE (newsurface);
    return newsurface;
}

/*
   This implements the AdvanceMAME Scale2x feature found on this page,
   http://advancemame.sourceforge.net/scale2x.html
   
   It is an incredibly simple and powerful image doubling routine that does
   an astonishing job of doubling game graphic data while interpolating out
   the jaggies. Congrats to the AdvanceMAME team, I'm very impressed and
   surprised with this code!
*/
SDL_Surface*
pyg_transform_scale2x (SDL_Surface *srcsurface, SDL_Surface *dstsurface)
{
    int allocated = 0;
    int looph, loopw;
    Uint8* srcpix, *dstpix;
    int srcpitch, dstpitch;
    int width, height;
    
    if (!srcsurface)
    {
        SDL_SetError ("srcsurface argument NULL");
        return NULL;
    }

    if (!dstsurface)
    {
        dstsurface = _surface_fromsurface (srcsurface, srcsurface->w * 2, 
            srcsurface->h * 2);
        if (!dstsurface)
            return NULL;
        allocated = 1;
    }

    if (dstsurface->w != (srcsurface->w * 2) ||
        dstsurface->h != (srcsurface->h * 2))
    {
        SDL_SetError ("dstsurface must be twice as big as srcsurface");
        if (allocated)
            SDL_FreeSurface (dstsurface);
        return NULL;
    }
    if (srcsurface->format->BytesPerPixel != dstsurface->format->BytesPerPixel)
    {
        SDL_SetError ("srcsurface and dstsurface must have the same format");
        if (allocated)
            SDL_FreeSurface (dstsurface);
        return NULL;
    }

    if (SDL_MUSTLOCK (srcsurface) && SDL_LockSurface (srcsurface) == -1)
    {
        if (allocated)
            SDL_FreeSurface (dstsurface);
        return NULL;
    }
    if (SDL_MUSTLOCK (dstsurface) && SDL_LockSurface (dstsurface) == -1)
    {
        if (allocated)
            SDL_FreeSurface (dstsurface);
        UNLOCK_SURFACE (srcsurface);
        return NULL;
    }

    srcpix = (Uint8*) srcsurface->pixels;
    dstpix = (Uint8*) dstsurface->pixels;
    srcpitch = srcsurface->pitch;
    dstpitch = dstsurface->pitch;
    width = srcsurface->w;
    height = srcsurface->h;

    switch(srcsurface->format->BytesPerPixel)
    {
    case 1:
    { 
        Uint8 E0, E1, E2, E3, B, D, E, F, H;
        for (looph = 0; looph < height; ++looph)
        {
            for (loopw = 0; loopw < width; ++ loopw)
            {
                B = *(Uint8*)(srcpix + (MAX (0, looph-1)*srcpitch) + (1*loopw));
                D = *(Uint8*)(srcpix + (looph*srcpitch) + (1*MAX(0,loopw-1)));
                E = *(Uint8*)(srcpix + (looph*srcpitch) + (1*loopw));
                F = *(Uint8*)(srcpix + (looph*srcpitch) +
                    (1*MIN(width-1,loopw+1)));
                H = *(Uint8*)(srcpix + (MIN(height-1,looph+1)*srcpitch) +
                    (1*loopw));
                                
                E0 = D == B && B != F && D != H ? D : E;
                E1 = B == F && B != D && F != H ? F : E;
                E2 = D == H && D != B && H != F ? D : E;
                E3 = H == F && D != H && B != F ? F : E;
                
                *(Uint8*)(dstpix + looph*2*dstpitch + loopw*2*1) = E0;
                *(Uint8*)(dstpix + looph*2*dstpitch + (loopw*2+1)*1) = E1;
                *(Uint8*)(dstpix + (looph*2+1)*dstpitch + loopw*2*1) = E2;
                *(Uint8*)(dstpix + (looph*2+1)*dstpitch + (loopw*2+1)*1) = E3;
            }
        }
        break;
    }
    case 2:
    { 
        Uint16 E0, E1, E2, E3, B, D, E, F, H;
        for (looph = 0; looph < height; ++looph)
        {
            for (loopw = 0; loopw < width; ++ loopw)
            {
                B = *(Uint16*)(srcpix + (MAX(0,looph-1)*srcpitch) + (2*loopw));
                D = *(Uint16*)(srcpix + (looph*srcpitch) + (2*MAX(0,loopw-1)));
                E = *(Uint16*)(srcpix + (looph*srcpitch) + (2*loopw));
                F = *(Uint16*)(srcpix + (looph*srcpitch) +
                    (2*MIN(width-1,loopw+1)));
                H = *(Uint16*)(srcpix + (MIN(height-1,looph+1)*srcpitch) +
                    (2*loopw));
                                
                E0 = D == B && B != F && D != H ? D : E;
                E1 = B == F && B != D && F != H ? F : E;
                E2 = D == H && D != B && H != F ? D : E;
                E3 = H == F && D != H && B != F ? F : E;
                
                *(Uint16*)(dstpix + looph*2*dstpitch + loopw*2*2) = E0;
                *(Uint16*)(dstpix + looph*2*dstpitch + (loopw*2+1)*2) = E1;
                *(Uint16*)(dstpix + (looph*2+1)*dstpitch + loopw*2*2) = E2;
                *(Uint16*)(dstpix + (looph*2+1)*dstpitch + (loopw*2+1)*2) = E3;
            }
        }
        break;
    }
    case 3:
    {
        int E0, E1, E2, E3, B, D, E, F, H;
        for (looph = 0; looph < height; ++looph)
        {
            for (loopw = 0; loopw < width; ++ loopw)
            {
                B = READINT24(srcpix + (MAX(0,looph-1)*srcpitch) + (3*loopw));
                D = READINT24(srcpix + (looph*srcpitch) + (3*MAX(0,loopw-1)));
                E = READINT24(srcpix + (looph*srcpitch) + (3*loopw));
                F = READINT24(srcpix + (looph*srcpitch) +
                    (3*MIN(width-1,loopw+1)));
                H = READINT24(srcpix + (MIN(height-1,looph+1)*srcpitch) +
                    (3*loopw));
                
                E0 = D == B && B != F && D != H ? D : E;
                E1 = B == F && B != D && F != H ? F : E;
                E2 = D == H && D != B && H != F ? D : E;
                E3 = H == F && D != H && B != F ? F : E;
                
                WRITEINT24((dstpix + looph*2*dstpitch + loopw*2*3), E0);
                WRITEINT24((dstpix + looph*2*dstpitch + (loopw*2+1)*3), E1);
                WRITEINT24((dstpix + (looph*2+1)*dstpitch + loopw*2*3), E2);
                WRITEINT24((dstpix + (looph*2+1)*dstpitch + (loopw*2+1)*3), E3);
            }
        }
        break;
    }
    default:
    {
        Uint32 E0, E1, E2, E3, B, D, E, F, H;
        for (looph = 0; looph < height; ++looph)
        {
            for (loopw = 0; loopw < width; ++ loopw)
            {
                B = *(Uint32*)(srcpix + (MAX(0,looph-1)*srcpitch) + (4*loopw));
                D = *(Uint32*)(srcpix + (looph*srcpitch) + (4*MAX(0,loopw-1)));
                E = *(Uint32*)(srcpix + (looph*srcpitch) + (4*loopw));
                F = *(Uint32*)(srcpix + (looph*srcpitch) +
                    (4*MIN(width-1,loopw+1)));
                H = *(Uint32*)(srcpix + (MIN(height-1,looph+1)*srcpitch) +
                    (4*loopw));
                                
                E0 = D == B && B != F && D != H ? D : E;
                E1 = B == F && B != D && F != H ? F : E;
                E2 = D == H && D != B && H != F ? D : E;
                E3 = H == F && D != H && B != F ? F : E;
                
                *(Uint32*)(dstpix + looph*2*dstpitch + loopw*2*4) = E0;
                *(Uint32*)(dstpix + looph*2*dstpitch + (loopw*2+1)*4) = E1;
                *(Uint32*)(dstpix + (looph*2+1)*dstpitch + loopw*2*4) = E2;
                *(Uint32*)(dstpix + (looph*2+1)*dstpitch + (loopw*2+1)*4) = E3;
            }
        }
        break;
    }
    }

    UNLOCK_SURFACE (srcsurface);
    UNLOCK_SURFACE (dstsurface);
    return dstsurface;
}

SDL_Surface*
pyg_transform_smoothscale (SDL_Surface *srcsurface, SDL_Surface *dstsurface,
    int width, int height, FilterFuncs *filters)
{
    Uint8* srcpix, *dstpix, *dst32 = NULL;
    int srcpitch, dstpitch;
    int srcwidth, srcheight, dstwidth, dstheight;
    int bpp, allocated = 0;
    Uint8 *temppix = NULL;
    int tempwidth = 0, temppitch = 0, tempheight = 0;

    if (!srcsurface)
    {
        SDL_SetError ("srcsurface argument NULL");
        return NULL;
    }

    if (!dstsurface)
    {
        dstsurface = _surface_fromsurface (srcsurface, width, height);
        if (!dstsurface)
            return NULL;
        allocated = 1;
    }

    if (dstsurface->w != width || dstsurface->h != height)
    {
        SDL_SetError ("dstsurface does not match the required size");
        if (allocated)
            SDL_FreeSurface (dstsurface);
        return NULL;
    }

    bpp = srcsurface->format->BytesPerPixel;
    if (bpp != dstsurface->format->BytesPerPixel)
    {
        SDL_SetError ("srcsurface and dstsurface must have the same format");
        if (allocated)
            SDL_FreeSurface (dstsurface);
        return NULL;
    }

    
    if (bpp != 3 && bpp != 4)
    {
        SDL_SetError ("only 24 bit as 32 bit surface can be smoothly scaled");
        if (allocated)
            SDL_FreeSurface (dstsurface);
        return NULL;
    }

    if (((width * bpp + 3) >> 2) > dstsurface->pitch)
    {
        SDL_SetError ("dstsurface pitch not 4-byte aligned");
        if (allocated)
            SDL_FreeSurface (dstsurface);
        return NULL;
    }

    if (SDL_MUSTLOCK (srcsurface) && SDL_LockSurface (srcsurface) == -1)
    {
        if (allocated)
            SDL_FreeSurface (dstsurface);
        return NULL;
    }
    if (SDL_MUSTLOCK (dstsurface) && SDL_LockSurface (dstsurface) == -1)
    {
        if (allocated)
            SDL_FreeSurface (dstsurface);
        UNLOCK_SURFACE (srcsurface);
        return NULL;
    }

    if (srcsurface->w == width && srcsurface->h == height)
    {
        /* Handle trivial case. */
        int y;
        for (y = 0; y < height; y++)
        {
            memcpy((Uint8*)dstsurface->pixels + y * dstsurface->pitch, 
                (Uint8*)srcsurface->pixels + y * srcsurface->pitch,
                (size_t) (width * bpp));
        }
        UNLOCK_SURFACE (srcsurface);
        UNLOCK_SURFACE (dstsurface);
        return dstsurface;
    }

    srcpix = (Uint8*)srcsurface->pixels;
    dstpix = (Uint8*)dstsurface->pixels;
    srcpitch = srcsurface->pitch;
    dstpitch = dstsurface->pitch;

    srcwidth = srcsurface->w;
    srcheight = srcsurface->h;
    dstwidth = dstsurface->w;
    dstheight = dstsurface->h;

    /* convert to 32-bit if necessary */
    if (bpp == 3)
    {
        int newpitch = srcwidth * 4;
        Uint8 *newsrc = malloc((size_t) (newpitch * srcheight));
        if (!newsrc)
        {
            UNLOCK_SURFACE (srcsurface);
            UNLOCK_SURFACE (dstsurface);
            if (allocated)
                SDL_FreeSurface (dstsurface);
            SDL_SetError ("could not allocate memory");
            return NULL;
        }

        _convert_24_32 (srcpix, srcpitch, newsrc, newpitch, srcwidth,
            srcheight);
        srcpix = newsrc;
        srcpitch = newpitch;
        /* create a destination buffer for the 32-bit result */
        dstpitch = dstwidth << 2;
        dst32 = malloc ((size_t) (dstpitch * dstheight));
        if (dst32 == NULL)
        {
            free(srcpix);
            UNLOCK_SURFACE (srcsurface);
            UNLOCK_SURFACE (dstsurface);
            if (allocated)
                SDL_FreeSurface (dstsurface);
            SDL_SetError ("could not allocate memory");
            return NULL;
        }
        dstpix = dst32;
    }
    
    /* Create a temporary processing buffer if we will be scaling both X
     * and Y */
    if (srcwidth != dstwidth && srcheight != dstheight)
    {
        tempwidth = dstwidth;
        temppitch = tempwidth << 2;
        tempheight = srcheight;
        temppix = (Uint8 *) malloc((size_t) (temppitch * tempheight));
        if (temppix == NULL)
        {
            if (bpp == 3)
            {
                free(srcpix);
                free(dstpix);
            }
            UNLOCK_SURFACE (srcsurface);
            UNLOCK_SURFACE (dstsurface);
            if (allocated)
                SDL_FreeSurface (dstsurface);
            SDL_SetError ("could not allocate memory");
            return NULL;
        }
    }
    
    /* Start the filter by doing X-scaling */
    if (dstwidth < srcwidth) /* shrink */
    {
        if (srcheight != dstheight)
            filters->shrink_X (srcpix, temppix, srcheight, srcpitch,
                temppitch, srcwidth, dstwidth);
        else
            filters->shrink_X (srcpix, dstpix, srcheight, srcpitch,
                dstpitch, srcwidth, dstwidth);
    }
    else if (dstwidth > srcwidth) /* expand */
    {
        if (srcheight != dstheight)
            filters->expand_X (srcpix, temppix, srcheight, srcpitch,
                temppitch, srcwidth, dstwidth);
        else
            filters->expand_X (srcpix, dstpix, srcheight, srcpitch,
                dstpitch, srcwidth, dstwidth);
    }
      
    /* Now do the Y scale */
    if (dstheight < srcheight) /* shrink */
    {
        if (srcwidth != dstwidth)
            filters->shrink_Y (temppix, dstpix, tempwidth, temppitch,
                dstpitch, srcheight, dstheight);
        else
            filters->shrink_Y (srcpix, dstpix, srcwidth, srcpitch,
                dstpitch, srcheight, dstheight);
    }
    else if (dstheight > srcheight)  /* expand */
    {
        if (srcwidth != dstwidth)
            filters->expand_Y (temppix, dstpix, tempwidth, temppitch,
                dstpitch, srcheight, dstheight);
        else
            filters->expand_Y (srcpix, dstpix, srcwidth, srcpitch,
                dstpitch, srcheight, dstheight);
    }

    /* Convert back to 24-bit if necessary */
    if (bpp == 3)
    {
        _convert_32_24 (dst32, dstpitch, (Uint8*)dstsurface->pixels,
            dstsurface->pitch, dstwidth, dstheight);
        free (dst32);
        dst32 = NULL;
        free (srcpix);
        srcpix = NULL;
    }

    /* free temporary buffer if necessary */
    if (temppix != NULL)
        free (temppix);

    UNLOCK_SURFACE (srcsurface);
    UNLOCK_SURFACE (dstsurface);
    return dstsurface;
}

SDL_Surface*
pyg_transform_laplacian (SDL_Surface *srcsurface, SDL_Surface *dstsurface)
{
    int allocated = 0;
    int ii, x, y, height, width, bpp;
    Uint32 sample[9];
    int total[4];
    Uint8 c1r, c1g, c1b, c1a;
    Uint8 acolor[4];
    Uint32 color;
    int atmp0, atmp1, atmp2, atmp3;
    SDL_PixelFormat *format;

    if (!srcsurface)
    {
        SDL_SetError ("srcsurface argument NULL");
        return NULL;
    }

    if (!dstsurface)
    {
        dstsurface = _surface_fromsurface (srcsurface, srcsurface->w,
            srcsurface->h);
        if (!dstsurface)
            return NULL;
        allocated = 1;
    }

    bpp = srcsurface->format->BytesPerPixel;

    if (dstsurface->w != srcsurface->w || dstsurface->h != srcsurface->h)
    {
        SDL_SetError ("dstsurface must be of the same size as srcsurface");
        if (allocated)
            SDL_FreeSurface (dstsurface);
        return NULL;
    }
    if (bpp != dstsurface->format->BytesPerPixel)
    {
        SDL_SetError ("srcsurface and dstsurface must have the same format");
        if (allocated)
            SDL_FreeSurface (dstsurface);
        return NULL;
    }

    if (SDL_MUSTLOCK (srcsurface) && SDL_LockSurface (srcsurface) == -1)
    {
        if (allocated)
            SDL_FreeSurface (dstsurface);
        return NULL;
    }
    if (SDL_MUSTLOCK (dstsurface) && SDL_LockSurface (dstsurface) == -1)
    {
        if (allocated)
            SDL_FreeSurface (dstsurface);
        UNLOCK_SURFACE (srcsurface);
        return NULL;
    }
    
    format = srcsurface->format;
    height = srcsurface->h;
    width = srcsurface->w;

/*
    -1 -1 -1
    -1  8 -1
    -1 -1 -1

    col = (sample[4] * 8) - (sample[0] + sample[1] + sample[2] +
                             sample[3] +             sample[5] + 
                             sample[6] + sample[7] + sample[8])

    [(-1,-1), (0,-1), (1,-1),     (-1,0), (0,0), (1,0),     (-1,1), (0,1), (1,1)]

*/

    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            /* Need to bounds check these accesses. */

            if (y > 0)
            {
                if (x > 0)
                {
                    GET_PIXEL_AT (sample[0], srcsurface, bpp, x - 1, y - 1);
                }
                GET_PIXEL_AT (sample[1], srcsurface, bpp, x, y - 1);
                if (x + 1 < width)
                {
                    GET_PIXEL_AT(sample[2], srcsurface, bpp, x + 1, y - 1);
                }
            }
            else
            {
                sample[0] = LAPLACIAN_NUM;
                sample[1] = LAPLACIAN_NUM;
                sample[2] = LAPLACIAN_NUM;
            }
            if (x > 0)
            {
                GET_PIXEL_AT (sample[3], srcsurface, bpp, x - 1, y);
            }
            else
            {
                sample[3] = LAPLACIAN_NUM;
            }

            sample[4] = 0;

            if (x + 1 < width)
            {
                GET_PIXEL_AT(sample[5], srcsurface, bpp, x + 1, y);
            }
            else
            {
                sample[5] = LAPLACIAN_NUM;
            }

            if (y + 1 < height)
            {
                if (x > 0)
                {
                    GET_PIXEL_AT (sample[6], srcsurface, bpp, x - 1, y + 1);
                }

                GET_PIXEL_AT (sample[7], srcsurface, bpp, x, y + 1);

                if (x + 1 < width)
                {
                    GET_PIXEL_AT (sample[8], srcsurface, bpp, x + 1, y + 1);
                }
            }
            else
            {
                sample[6] = LAPLACIAN_NUM;
                sample[7] = LAPLACIAN_NUM;
                sample[8] = LAPLACIAN_NUM;
            }
            total[0] = total[1] = total[2] = total[3] = 0;
            
            for (ii = 0; ii < 9; ii++)
            {
                SDL_GetRGBA (sample[ii], format, &c1r, &c1g, &c1b, &c1a);
                total[0] += c1r;
                total[1] += c1g;
                total[2] += c1b;
                total[3] += c1a;
            }
            
            GET_PIXEL_AT (sample[4], srcsurface, bpp, x, y);
            SDL_GetRGBA (sample[4], format, &c1r, &c1g, &c1b, &c1a);
            
            /* cast on the right to a signed int, and then clamp to 0-255. */
            /* atmp = c1r * 8 */
            
            atmp0 = c1r * 8;
            acolor[0] = MIN (MAX (atmp0 - total[0], 0), 255);
            atmp1 = c1g * 8;
            acolor[1] = MIN (MAX (atmp1 - total[1], 0), 255);
            atmp2 = c1b * 8;
            acolor[2] = MIN (MAX (atmp2 - total[2], 0), 255);
            atmp3 = c1a * 8;
            acolor[3] = MIN (MAX (atmp3 - total[3], 0), 255);

            /* cast on the right to Uint32, and then clamp to 255. */
            color = SDL_MapRGBA (format, acolor[0], acolor[1], acolor[2],
                acolor[3]);

            SET_PIXEL_AT (dstsurface, format, x, y, color);
        }
    }

    UNLOCK_SURFACE (srcsurface);
    UNLOCK_SURFACE (dstsurface);
    return dstsurface;
}

SDL_Surface*
pyg_transform_average_surfaces (SDL_Surface **surfaces, int count,
    SDL_Surface *dstsurface)
{
    Uint32 *accumulate, *the_idx;
    Uint32 color;
    SDL_Surface *surface;
    int maxw, maxh, x, y, i, allocated = 0;
    float div_inv;
    SDL_PixelFormat *format, *dstformat;
    Uint32 rmask, gmask, bmask;
    int rshift, gshift, bshift, rloss, gloss, bloss;

    if (count == 0)
    {
        SDL_SetError ("count argument 0");
        return NULL;
    }

    if (!surfaces)
    {
        SDL_SetError ("surface argument NULL");
        return 0;
    }

    maxw = maxh = 0;
    for (i = 0; i < count; i++)
    {
        maxw = MAX (surfaces[i]->w, maxw);
        maxh = MAX (surfaces[i]->h, maxh);
    }

    if (!dstsurface)
    {
        /* We use a 32 bit access, as we do not know the format of the
         * surfaces within the list.
         */
        dstsurface = SDL_CreateRGBSurface (SDL_SWSURFACE, maxw, maxh, 32,
            0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000);
        if (!dstsurface)
            return NULL;
        allocated = 1;
    }

    if (maxw != dstsurface->w || maxh != dstsurface->h)
    {
        SDL_SetError
            ("destination surface must match the biggest surface size");
        if (allocated)
            SDL_FreeSurface (dstsurface);
        return NULL;
    }
    
    accumulate = (Uint32 *) calloc (1, sizeof(Uint32) * maxw * maxh * 3);
    if (!accumulate)
    {
        if (allocated)
            SDL_FreeSurface (dstsurface);
        SDL_SetError ("could not allocate memory");
        return NULL;
    }

    for (i = 0; i < count; i++)
    {
        surface = surfaces[i];
        if (SDL_MUSTLOCK (surface) && SDL_LockSurface (surface) == -1)
        {
            free (accumulate);
            if (allocated)
                SDL_FreeSurface (dstsurface);
            return NULL;
        }

        format = surface->format;
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
        
        for (y = 0; y < surface->h; y++)
        {
            for (x = 0; x < surface->w; x++)
            {
                GET_PIXEL_AT (color, surface, format->BytesPerPixel, x, y);
        
                *(the_idx) += ((color & rmask) >> rshift) << rloss;
                *(the_idx + 1) += ((color & gmask) >> gshift) << gloss;
                *(the_idx + 2) += ((color & bmask) >> bshift) << bloss;
                the_idx += 3;
            }
        }
        UNLOCK_SURFACE (surface);
    }

    /*  blit the accumulated array back to the destination surface. */
    div_inv = (float) (1.0L / count);
    the_idx = accumulate;

    if (SDL_MUSTLOCK (dstsurface) && SDL_LockSurface (dstsurface) == -1)
    {
        free (accumulate);
        if (allocated)
            SDL_FreeSurface (dstsurface);
        return NULL;
    }
    dstformat = dstsurface->format;

    for (y = 0;y < maxh; y++)
    {
        for (x = 0; x < maxw; x++)
        {
            color = SDL_MapRGB (dstformat, 
                (Uint8) (*(the_idx) * div_inv + .5f),
                (Uint8) (*(the_idx + 1) * div_inv + .5f),
                (Uint8) (*(the_idx + 2) * div_inv + .5f));
            SET_PIXEL_AT (dstsurface, dstformat, x, y, color);
            the_idx += 3;
        }
    }
    UNLOCK_SURFACE (dstsurface);
    free (accumulate);
    return dstsurface;
}

int
pyg_transform_average_color (SDL_Surface *surface, SDL_Rect *rect,
    Uint8 *r, Uint8 *g, Uint8 *b, Uint8 *a)
{
    Uint32 color, rmask, gmask, bmask, amask;
    Sint16 x, y, w, h;
    Uint8 *pixels, *pix;
    unsigned int rtot, gtot, btot, atot, size, rshift, gshift, bshift, ashift;
    unsigned int rloss, gloss, bloss, aloss;
    int row, col;
    SDL_PixelFormat *format;

    if (!surface)
    {
        SDL_SetError ("surface argument NULL");
        return 0;
    }

    if (!rect)
    {
        x = y = 0;
        w = surface->w;
        h = surface->w;
    }
    else
    {
        x = rect->x;
        y = rect->y;
        w = rect->w;
        h = rect->h;
    }

    format = surface->format;
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

    /* Make sure the area specified is within the Surface */
    if ((x + w) > surface->w)
        w = surface->w - x;
    if ((y + h) > surface->h)
        h = surface->h - y;
    if (x < 0)
    {
        w -= (-x);
        x = 0;
    }
    if (y < 0)
    {
        h -= (-y);
        y = 0;
    }

    size = w * h;

    LOCK_SURFACE (surface, 0);

    switch (format->BytesPerPixel)
    {
    case 1:
        for (row = y; row < y + h; row++)
        {
            pixels = (Uint8 *) surface->pixels + row * surface->pitch + x;
            for (col = x; col < x + w; col++)
            {
                color = (Uint32)*((Uint8 *) pixels);
                rtot += ((color & rmask) >> rshift) << rloss;
                gtot += ((color & gmask) >> gshift) << gloss;
                btot += ((color & bmask) >> bshift) << bloss;
                atot += ((color & amask) >> ashift) << aloss;
                pixels++;
            }
        }
        break;    
    case 2:
        for (row = y; row < y + h; row++)
        {
            pixels = (Uint8 *) surface->pixels + row * surface->pitch + x * 2;
            for (col = x; col < x + w; col++)
            {
                color = (Uint32)*((Uint16 *) pixels);
                rtot += ((color & rmask) >> rshift) << rloss;
                gtot += ((color & gmask) >> gshift) << gloss;
                btot += ((color & bmask) >> bshift) << bloss;
                atot += ((color & amask) >> ashift) << aloss;
                pixels += 2;
            }
        }
        break;    
    case 3:
        for (row = y; row < y + h; row++)
        {
            pixels = (Uint8 *) surface->pixels + row * surface->pitch + x * 3;
            for (col = x; col < x + w; col++)
            {
                pix = pixels;
                color = GET_PIXEL24 (pix);
                rtot += ((color & rmask) >> rshift) << rloss;
                gtot += ((color & gmask) >> gshift) << gloss;
                btot += ((color & bmask) >> bshift) << bloss;
                atot += ((color & amask) >> ashift) << aloss;
                pixels += 3;
            }
        }                    
        break;
    default:
        for (row = y; row < y + h; row++)
        {
            pixels = (Uint8 *) surface->pixels + row * surface->pitch + x * 4;
            for (col = x; col < x + w; col++)
            {
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

    UNLOCK_SURFACE (surface);

    *r = rtot / size;
    *g = gtot / size;
    *b = btot / size;
    *a = atot / size;
    return 1;
}

int
pyg_transform_threshold_color (SDL_Surface *srcsurface, Uint32 diffcolor,
    Uint32 threscolor, SDL_Surface *dstsurface)
{
    SDL_PixelFormat *format, *format2 = NULL;
    int x, y;
    int similar = 0;
    Uint8 *srcpixels;
    Uint8 dr, dg, db, da;
    Uint8 tr, tg, tb, ta;
    Uint32 color;

    if (!srcsurface)
    {
        SDL_SetError ("srcsurface argument NULL");
        return -1;
    }

    LOCK_SURFACE (srcsurface, -1);

    if (dstsurface)
    {
        if (srcsurface->w != dstsurface->w || srcsurface->h != dstsurface->h)
        {
            UNLOCK_SURFACE (srcsurface);
            SDL_SetError ("srcsurface and dstsurface must have the same size");
            return -1;
        }

        if (SDL_MUSTLOCK (dstsurface) && SDL_LockSurface (dstsurface) == -1)
        {
            UNLOCK_SURFACE (srcsurface);
            return -1;
        }
    }

    format = srcsurface->format;
    if (dstsurface)
        format2 = dstsurface->format;

    /* TODO: set filler for the clipping area on dstsruface. */

    SDL_GetRGBA (threscolor, format, &tr, &tg, &tb, &ta);
    SDL_GetRGBA (diffcolor, format, &dr, &dg, &db, &da);

    for (y = 0; y < srcsurface->h; y++)
    {
        srcpixels = (Uint8 *) srcsurface->pixels + y * srcsurface->pitch;
        for (x = 0; x < srcsurface->w; x++)
        {
            GET_PIXEL (color, format->BytesPerPixel, srcpixels);
            srcpixels += format->BytesPerPixel;

            if (abs(COLOR_R(color, format) - dr) <= tr &&
                abs(COLOR_G(color, format) - dg) <= tg &&
                abs(COLOR_B(color, format) - db) <= tb)
            {
                similar++;
                if (dstsurface)
                {
                    SET_PIXEL_AT (dstsurface, format2, x, y, color);
                }
            }
        }
    }

    UNLOCK_SURFACE (srcsurface);
    if (dstsurface)
        UNLOCK_SURFACE (dstsurface);
    return similar;
}

int
pyg_transform_threshold_surface (SDL_Surface *srcsurface,
    SDL_Surface *diffsurface, Uint32 threscolor, SDL_Surface *dstsurface)
{
    SDL_PixelFormat *format, *diffformat, *format2 = NULL;
    int x, y;
    int similar = 0;
    Uint8 *srcpixels, *diffpixels = NULL;
    Uint8 tr, tg, tb, ta;
    Uint32 color, color2;

    if (!srcsurface)
    {
        SDL_SetError ("srcsurface argument NULL");
        return -1;
    }
    if (!diffsurface)
    {
        SDL_SetError ("diffsurface argument NULL");
        return -1;
    }

    if (srcsurface->w != diffsurface->w || srcsurface->h != diffsurface->h)
    {
        SDL_SetError ("srcsurface and diffsurface must have the same size");
        return -1;
    }
    
    LOCK_SURFACE (srcsurface, -1);
    if (SDL_MUSTLOCK (diffsurface) && SDL_LockSurface (diffsurface) == -1)
    {
        UNLOCK_SURFACE (srcsurface);
        return -1;
    }

    if (dstsurface)
    {
        if (srcsurface->w != dstsurface->w || srcsurface->h != dstsurface->h)
        {
            UNLOCK_SURFACE (srcsurface);
            UNLOCK_SURFACE (diffsurface);
            SDL_SetError ("srcsurface and dstsurface must have the same size");
            return -1;
        }
        if (SDL_MUSTLOCK (dstsurface) && SDL_LockSurface (dstsurface) == -1)
        {
            UNLOCK_SURFACE (srcsurface);
            UNLOCK_SURFACE (diffsurface);
            return -1;
        }
    }

    format = srcsurface->format;
    diffformat = diffsurface->format;
    if (dstsurface)
        format2 = dstsurface->format;

    /* TODO: set filler for the clipping area on dstsruface. */
    SDL_GetRGBA (threscolor, format, &tr, &tg, &tb, &ta);

    for (y = 0; y < srcsurface->h; y++)
    {
        srcpixels = (Uint8 *) srcsurface->pixels + y * srcsurface->pitch;
        diffpixels = (Uint8 *) diffsurface->pixels + y * diffsurface->pitch;

        for (x = 0; x < srcsurface->w; x++)
        {
            GET_PIXEL (color, format->BytesPerPixel, srcpixels);
            GET_PIXEL (color2, diffformat->BytesPerPixel, diffpixels);

            srcpixels += format->BytesPerPixel;
            diffpixels += diffformat->BytesPerPixel;

            if (DIFF_COLOR_R(color, format, color2, diffformat) <= tr &&
                DIFF_COLOR_G(color, format, color2, diffformat) <= tg &&
                DIFF_COLOR_B(color, format, color2, diffformat) <= tb)
            {
                similar++;
                if (dstsurface)
                {
                    SET_PIXEL_AT (dstsurface, format2, x, y, color);
                }
            }
        }
    }

    UNLOCK_SURFACE (srcsurface);
    UNLOCK_SURFACE (diffsurface);
    if (dstsurface)
        UNLOCK_SURFACE (dstsurface);
    return similar;
}
