/*
  pygame - Python Game Library
  Copyright (C) 2000-2001 Pete Shinners

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
#include "draw.h"
#include "surface.h"

#define FRAC(z) ((z) - trunc(z))
#define INVFRAC(z) (1 - FRAC(z))

#define DRAWPIX32(pixel,colorptr,br,blend,hasalpha)                     \
    if (blend)                                                          \
    {                                                                   \
        int _x;                                                         \
        float nbr = 1.0 - br;                                           \
        _x = colorptr[0] * br + pixel[0] * nbr;                         \
        pixel[0]= (_x > 254) ? 255: _x;                                 \
        _x = colorptr[1] * br + pixel[1] * nbr;                         \
        pixel[1]= (_x > 254) ? 255: _x;                                 \
        _x = colorptr[2] * br + pixel[2] * nbr;                         \
        pixel[2]= (_x > 254) ? 255: _x;                                 \
        if (hasalpha)                                                   \
            pixel[3] = pixel[0] + (br * 255) - (pixel[3] * br);         \
    }                                                                   \
    else                                                                \
    {                                                                   \
        pixel[0] = (Uint8)(colorptr[0] * br);                           \
        pixel[1] = (Uint8)(colorptr[1] * br);                           \
        pixel[2] = (Uint8)(colorptr[2] *br);                            \
        if (hasalpha)                                                   \
            pixel[3] = br * 255;                                        \
    }

/* the line clipping based heavily off of code from
 * http://www.ncsa.uiuc.edu/Vis/Graphics/src/clipCohSuth.c
 */
#define LEFT_EDGE   0x1
#define RIGHT_EDGE  0x2
#define BOTTOM_EDGE 0x4
#define TOP_EDGE    0x8
#define INSIDE(a)   (!a)
#define REJECT(a,b) (a&b)
#define ACCEPT(a,b) (!(a|b))

static int _compare_int (const void *a, const void *b);
static int _encode (int x, int y, int left, int top, int right, int bottom);
static int _fencode (float x, float y, int left, int top, int right,
    int bottom);
static int _clipline (SDL_Rect *clip, int x1, int _y1, int x2, int y2,
    int *outpts);
static int _clipaaline (SDL_Rect *clip, float x1, float _y1, float x2, float y2,
    float *outpts);
static void _drawline (SDL_Surface* surface, Uint32 color, int x1, int _y1,
    int x2, int y2);
static void _drawhorzline (SDL_Surface* surface, Uint32 color, int startx,
    int y, int endx);
static void _drawvertline (SDL_Surface* surface, Uint32 color, int x,
    int starty, int endy);
static int _drawlinewidth (SDL_Surface* surface, SDL_Rect *cliprect,
    Uint32 color, int *pts, int width);
static void _drawaaline (SDL_Surface* surface, Uint32 color, int x1, int _y1,
    int x2, int y2, int blend);

static int
_compare_int (const void *a, const void *b)
{
    return (*(const int *)a) - (*(const int *)b);
}

static int
_encode (int x, int y, int left, int top, int right, int bottom)
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
_fencode (float x, float y, int left, int top, int right, int bottom)
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
_clipline (SDL_Rect *clip, int x1, int _y1, int x2, int y2, int *outpts)
{
    int left = clip->x;
    int right = clip->x + clip->w - 1;
    int top = clip->y;
    int bottom = clip->y + clip->h - 1;
    int code1, code2;
    int draw = 0;
    int swaptmp;
    float m; /*slope*/

    if (!outpts)
    {
        SDL_SetError ("outpts argument NULL");
        return 0;
    }

    while (1)
    {
        code1 = _encode (x1, _y1, left, top, right, bottom);
        code2 = _encode (x2, y2, left, top, right, bottom);
        if (ACCEPT (code1, code2))
        {
            draw = 1;
            break;
        }
        else if (REJECT (code1, code2))
            break;
        else
        {
            if (INSIDE (code1))
            {
                swaptmp = x2;
                x2 = x1;
                x1 = swaptmp;
                swaptmp = y2;
                y2 = _y1;
                _y1 = swaptmp;
                swaptmp = code2;
                code2 = code1;
                code1 = swaptmp;
            }
            if (x2 != x1)
                m = (y2 - _y1) / (float)(x2 - x1);
            else
                m = 1.0f;
            if (code1 & LEFT_EDGE)
            {
                _y1 += (int)((left - x1) * m);
                x1 = left;
            }
            else if (code1 & RIGHT_EDGE)
            {
                _y1 += (int)((right - x1) * m);
                x1 = right;
            }
            else if (code1 & BOTTOM_EDGE)
            {
                if (x2 != x1)
                    x1 += (int)((bottom - _y1) / m);
                _y1 = bottom;
            }
            else if (code1 & TOP_EDGE)
            {
                if(x2 != x1)
                    x1 += (int)((top - _y1) / m);
                _y1 = top;
            }
        }
    }
    
    if (draw)
    {
        outpts[0] = x1;
        outpts[1] = _y1;
        outpts[2] = x2;
        outpts[3] = y2;
    }
    return draw;
}

static int
_clipaaline (SDL_Rect *clip, float x1, float _y1, float x2, float y2,
    float *outpts)
{
    int left = clip->x + 1;
    int right = clip->x + clip->w - 2;
    int top = clip->y + 1;
    int bottom = clip->y + clip->h - 2;
    int code1, code2;
    int draw = 0;
    float swaptmp;
    int intswaptmp;
    float m; /*slope*/

    if (!outpts)
    {
        SDL_SetError ("outpts argument NULL");
        return 0;
    }

    while (1)
    {
        code1 = _fencode (x1, _y1, left, top, right, bottom);
        code2 = _fencode (x2, y2, left, top, right, bottom);
        if (ACCEPT (code1, code2))
        {
            draw = 1;
            break;
        }
        else if (REJECT (code1, code2))
            break;
        else
        {
            if (INSIDE (code1))
            {
                swaptmp = x2;
                x2 = x1;
                x1 = swaptmp;
                swaptmp = y2;
                y2 = _y1;
                _y1 = swaptmp;
                intswaptmp = code2;
                code2 = code1;
                code1 = intswaptmp;
            }
            if (x2 != x1)
                m = (y2 - _y1) / (x2 - x1);
            else
                m = 1.0f;
            if (code1 & LEFT_EDGE)
            {
                _y1 += ((float)left - x1) * m;
                x1 = (float)left;
            }
            else if (code1 & RIGHT_EDGE)
            {
                _y1 += ((float)right - x1) * m;
                x1 = (float)right;
            }
            else if (code1 & BOTTOM_EDGE)
            {
                if (x2 != x1)
                    x1 += ((float)bottom - _y1) / m;
                _y1 = (float)bottom;
            }
            else if (code1 & TOP_EDGE)
            {
                if(x2 != x1)
                    x1 += ((float)top - _y1) / m;
                _y1 = (float)top;
            }
        }
    }
    
    if (draw)
    {
        outpts[0] = x1;
        outpts[1] = _y1;
        outpts[2] = x2;
        outpts[3] = y2;
    }
    return draw;
}

static void
_drawline (SDL_Surface* surface, Uint32 color, int x1, int _y1, int x2, int y2)
{
    int deltax, deltay, signx, signy;
    int pixx, pixy;
    int x = 0, y = 0;
    int swaptmp;
    Uint8 *pixel;
    Uint8 *colorptr;
    
    deltax = x2 - x1;
    deltay = y2 - _y1;
    signx = (deltax < 0) ? -1 : 1;
    signy = (deltay < 0) ? -1 : 1;
    deltax = signx * deltax + 1;
    deltay = signy * deltay + 1;

    pixx = surface->format->BytesPerPixel;
    pixy = surface->pitch;
    pixel = ((Uint8*)surface->pixels) + pixx * x1 + pixy * _y1;
    
    pixx *= signx;
    pixy *= signy;

    if (deltax < deltay) /*swap axis if rise > run*/
    {
        swaptmp = deltax;
        deltax = deltay;
        deltay = swaptmp;
        swaptmp = pixx;
        pixx = pixy;
        pixy = swaptmp;
    }

    switch(surface->format->BytesPerPixel)
    {
    case 1:
        for (; x < deltax; x++, pixel += pixx)
        {
            *pixel = (Uint8) color;
            y += deltay;
            if (y >= deltax)
            {
                y -= deltax;
                pixel += pixy;
            }
        }
        break;
    case 2:
        for (; x < deltax; x++, pixel += pixx)
        {
            *(Uint16*)pixel = (Uint16)color;
            y += deltay;
            if (y >= deltax)
            {
                y -= deltax;
                pixel += pixy;
            }
        }
        break;
    case 3:
        if (SDL_BYTEORDER == SDL_BIG_ENDIAN)
            color <<= 8;
        colorptr = (Uint8*)&color;
        for (; x < deltax; x++, pixel += pixx)
        {
            pixel[0] = colorptr[0];
            pixel[1] = colorptr[1];
            pixel[2] = colorptr[2];
            y += deltay;
            if (y >= deltax)
            {
                y -= deltax;
                pixel += pixy;
            }
        }
        break;
    default:
        for (; x < deltax; x++, pixel += pixx)
        {
            *(Uint32*)pixel = (Uint32)color;
            y += deltay;
            if (y >= deltax)
            {
                y -= deltax;
                pixel += pixy;
            }
        }
        break;
    }
}

static void
_drawhorzline (SDL_Surface* surface, Uint32 color, int startx, int y, int endx)
{
    Uint8 *pixel, *end;
    Uint8 *colorptr;

    if(startx == endx)
    {
        SET_PIXEL_AT (surface, surface->format, startx, y, color);
        return;
    }

    pixel = ((Uint8*)surface->pixels) + surface->pitch * y;
    if(startx < endx)
    {
        end = pixel + endx * surface->format->BytesPerPixel;
        pixel += startx * surface->format->BytesPerPixel;
    }
    else
    {
        end = pixel + startx * surface->format->BytesPerPixel;
        pixel += endx * surface->format->BytesPerPixel;
    }

    switch (surface->format->BytesPerPixel)
    {
    case 1:
        for (; pixel <= end; ++pixel)
        {
            *pixel = (Uint8)color;
        }
        break;
    case 2:
        for (; pixel <= end; pixel += 2)
        {
            *(Uint16*)pixel = (Uint16)color;
        }
        break;
    case 3:
        if (SDL_BYTEORDER == SDL_BIG_ENDIAN)
            color <<= 8;
        colorptr = (Uint8*)&color;
        for (; pixel <= end; pixel += 3)
        {
            pixel[0] = colorptr[0];
            pixel[1] = colorptr[1];
            pixel[2] = colorptr[2];
        }
        break;
    default: /*case 4*/
        for (; pixel <= end; pixel += 4)
        {
            *(Uint32*)pixel = color;
        }
        break;
    }
}

static void
_drawvertline (SDL_Surface* surface, Uint32 color, int x, int starty, int endy)
{
    Uint8 *pixel, *end;
    Uint8 *colorptr;
    Uint32 pitch = surface->pitch;

    if (starty == endy)
    {
        SET_PIXEL_AT (surface, surface->format, x, starty, color);
        return;
    }

    pixel = ((Uint8*)surface->pixels) + x * surface->format->BytesPerPixel;
    if (starty < endy)
    {
        end = pixel + surface->pitch * endy;
        pixel += surface->pitch * starty;
    }
    else
    {
        end = pixel + surface->pitch * starty;
        pixel += surface->pitch * endy;
    }

    switch (surface->format->BytesPerPixel)
    {
    case 1:
        for (; pixel <= end; pixel += pitch)
        {
            *pixel = (Uint8)color;
        }
        break;
    case 2:
        for (; pixel <= end; pixel += pitch)
        {
            *(Uint16*)pixel = (Uint16)color;
        }
        break;
    case 3:
        if (SDL_BYTEORDER == SDL_BIG_ENDIAN)
            color <<= 8;
        colorptr = (Uint8*)&color;
        for (; pixel <= end; pixel += pitch)
        {
            pixel[0] = colorptr[0];
            pixel[1] = colorptr[1];
            pixel[2] = colorptr[2];
        }
        break;
    default: /*case 4*/
        for (; pixel <= end; pixel += pitch)
        {
            *(Uint32*)pixel = color;
        }
        break;
    }
}

static int
_drawlinewidth (SDL_Surface* surface, SDL_Rect *cliprect, Uint32 color,
    int *pts, int width)
{
    int loop;
    int xinc = 0, yinc = 0;
    int newpts[4], tmp[4], range[4];
    int anydrawn = 0;
    
    if (ABS (pts[0] - pts[2]) > ABS (pts[1] - pts[3]))
        yinc = 1;
    else
        xinc = 1;
    
    if (_clipline (cliprect, pts[0], pts[1], pts[2], pts[3], newpts))
    {
        if (newpts[1] == newpts[3])
            _drawhorzline (surface, color, newpts[0], newpts[1], newpts[2]);
        else if (pts[0] == pts[2])
            _drawvertline (surface, color, newpts[0], newpts[1], newpts[3]);
        else
            _drawline (surface, color, newpts[0], newpts[1], newpts[2],
                newpts[3]);
        
        anydrawn = 1;
        memcpy (range, newpts, sizeof(int) * 4);
    }
    else
    {
        range[0] = range[1] = 10000;
        range[2] = range[3] = -10000;
    }
    
    for (loop = 1; loop < width; loop += 2)
    {
        newpts[0] = pts[0] + xinc*(loop/2+1);
        newpts[1] = pts[1] + yinc*(loop/2+1);
        newpts[2] = pts[2] + xinc*(loop/2+1);
        newpts[3] = pts[3] + yinc*(loop/2+1);
        
        if(_clipline (cliprect, newpts[0], newpts[1], newpts[2], newpts[3],
                tmp))
        {
            memcpy (newpts, tmp, sizeof(int)*4);
            if (newpts[1] == newpts[3])
                _drawhorzline (surface, color, newpts[0], newpts[1], newpts[2]);
            else if (pts[0] == pts[2])
                _drawvertline (surface, color, newpts[0], newpts[1], newpts[3]);
            else
                _drawline (surface, color, newpts[0], newpts[1], newpts[2],
                    newpts[3]);
            
            anydrawn = 1;
            range[0] = MIN(newpts[0], range[0]);
            range[1] = MIN(newpts[1], range[1]);
            range[2] = MAX(newpts[2], range[2]);
            range[3] = MAX(newpts[3], range[3]);
        }

        if(loop + 1 < width)
        {
            newpts[0] = pts[0] - xinc*(loop/2+1);
            newpts[1] = pts[1] - yinc*(loop/2+1);
            newpts[2] = pts[2] - xinc*(loop/2+1);
            newpts[3] = pts[3] - yinc*(loop/2+1);
            if (_clipline (cliprect, newpts[0], newpts[1], newpts[2], newpts[3],
                    tmp))
            {
                memcpy (newpts, tmp, sizeof(int)*4);
                if (newpts[1] == newpts[3])
                    _drawhorzline (surface, color, newpts[0], newpts[1],
                        newpts[2]);
                else if (pts[0] == pts[2])
                    _drawvertline (surface, color, newpts[0], newpts[1],
                        newpts[3]);
                else
                    _drawline (surface, color, newpts[0], newpts[1], newpts[2],
                        newpts[3]);

                anydrawn = 1;
                range[0] = MIN(newpts[0], range[0]);
                range[1] = MIN(newpts[1], range[1]);
                range[2] = MAX(newpts[2], range[2]);
                range[3] = MAX(newpts[3], range[3]);
            }
        }
    }

    if (anydrawn)
        memcpy (pts, range, sizeof(int)*4);
    return anydrawn;
}

/* Adapted from
 * http://freespace.virgin.net/hugo.elias/graphics/x_wuline.htm
 */
static void
_drawaaline (SDL_Surface* surface, Uint32 color, int x1, int _y1, int x2,
    int y2, int blend)
{
    float grad, xd, yd;
    float xgap, ygap, xend, yend, xf, yf;
    float brightness1, brightness2;
    float swaptmp;
    int x, y, ix1, ix2, iy1, iy2;
    int pixx, pixy;
    Uint8* pixel;
    Uint8* pm = (Uint8*)surface->pixels;
    Uint8* colorptr = (Uint8*)&color;
    const int hasalpha = surface->format->Amask;

    pixx = surface->format->BytesPerPixel;
    pixy = surface->pitch;

    xd = x2 - x1;
    yd = y2 - _y1;

    if (xd == 0 && yd == 0)
    {
        /* Single point. Due to the nature of the aaline clipping, this
         * is less exact than the normal line. */
        SET_PIXEL_AT (surface, surface->format, x1, _y1, color);
        return;
    }

    if (SDL_BYTEORDER == SDL_BIG_ENDIAN)
        color <<= 8;

    if (fabs (xd) > fabs (yd))
    {
        if (x1 > x2)
        {
            swaptmp = x1;
            x1 = x2;
            x2 = swaptmp;
            swaptmp = _y1;
            _y1 = y2;
            y2 = swaptmp;
            xd = x2 - x1;
            yd = y2 - _y1;
        }
        grad = yd / xd;
        /* This makes more sense than trunc(x1+0.5) */
        xend = trunc ((float)x1) + 0.5;
        yend = _y1 + grad * (xend - x1);
        xgap = INVFRAC ((float)x1);
        ix1 = (int)xend;
        iy1 = (int)yend;
        yf = yend + grad;
        brightness1 = INVFRAC (yend) * xgap;
        brightness2 = FRAC (yend) * xgap;
        pixel = pm + pixx * ix1 + pixy * iy1;
        DRAWPIX32 (pixel, colorptr, brightness1, blend, hasalpha);
        pixel += pixy;
        DRAWPIX32 (pixel, colorptr, brightness2, blend, hasalpha);
        xend = trunc ((float)x2) + 0.5;
        yend = y2 + grad * (xend - x2);
        /* this also differs from Hugo's description. */
        xgap = FRAC ((float)x2);
        ix2 = (int)xend;
        iy2 = (int)yend;
        brightness1 = INVFRAC (yend) * xgap;
        brightness2 = FRAC (yend) * xgap;
        pixel = pm + pixx * ix2 + pixy * iy2;
        DRAWPIX32 (pixel, colorptr, brightness1, blend, hasalpha);
        pixel += pixy;
        DRAWPIX32 (pixel, colorptr, brightness2, blend, hasalpha);
        for (x = ix1 + 1; x < ix2; ++x)
        {
            brightness1 = INVFRAC (yf);
            brightness2 = FRAC (yf);
            pixel = pm + pixx * x + pixy * (int)yf;
            DRAWPIX32 (pixel, colorptr, brightness1, blend, hasalpha);
            pixel += pixy;
            DRAWPIX32 (pixel, colorptr, brightness2, blend, hasalpha);
            yf += grad;
        }
    }
    else
    {
        if (_y1 > y2)
        {
            swaptmp = _y1;
            _y1 = y2;
            y2 = swaptmp;
            swaptmp = x1;
            x1 = x2;
            x2 = swaptmp;
            yd = y2 - _y1;
            xd = x2 - x1;
        }
        grad = xd / yd;
        /* This makes more sense than trunc(x1+0.5) */
        yend = trunc ((float)_y1) + .5;
        xend = x1 + grad * (yend - _y1);
        ygap = INVFRAC ((float)_y1);
        iy1 = (int)yend;
        ix1 = (int)xend;
        xf = xend + grad;
        brightness1 = INVFRAC (xend) * ygap;
        brightness2 = FRAC (xend) * ygap;
        pixel = pm + pixx * ix1 + pixy * iy1;
        DRAWPIX32 (pixel, colorptr, brightness1, blend, hasalpha);
        pixel += pixx;
        DRAWPIX32 (pixel, colorptr, brightness2, blend, hasalpha);
        yend = trunc ((float)y2) + 0.5;
        xend = x2 + grad * (yend - y2);
        ygap = FRAC ((float)y2);
        iy2 = (int)yend;
        ix2 = (int)xend;
        brightness1 = INVFRAC (xend) * ygap;
        brightness2 = FRAC (xend) * ygap;
        pixel = pm + pixx * ix2 + pixy * iy2;
        DRAWPIX32 (pixel, colorptr, brightness1, blend, hasalpha);
        pixel += pixx;
        DRAWPIX32 (pixel, colorptr, brightness2, blend, hasalpha);
        for (y = iy1 + 1; y < iy2; ++y)
        {
            brightness1 = INVFRAC (xf);
            brightness2 = FRAC (xf);
            pixel = pm + pixx * (int)xf + pixy * y;
            DRAWPIX32 (pixel, colorptr, brightness1, blend, hasalpha);
            pixel += pixx;
            DRAWPIX32 (pixel, colorptr, brightness2, blend, hasalpha);
            xf += grad;
        }
    }
}

int
pyg_draw_aaline (SDL_Surface *surface, SDL_Rect *cliprect, Uint32 color,
    int x1, int _y1, int x2, int y2, int blendargs, SDL_Rect *area)
{
    float pts[4] = { 0 };

    if (!surface)
    {
        SDL_SetError ("surface argument NULL");
        return 0;
    }

    LOCK_SURFACE (surface, 0);
    if (!cliprect)
        cliprect = &surface->clip_rect;

    if (!_clipaaline (cliprect, (float)x1, (float)_y1, (float)x2, (float)y2,
            pts))
    {
        UNLOCK_SURFACE (surface);
        return 0;
    }
    _drawaaline (surface, color, pts[0], pts[1], pts[2], pts[3], blendargs);
    UNLOCK_SURFACE (surface);

    if (area)
    {
        area->x = (int) floor (pts[0]);
        area->y = (int) floor (pts[1]);
        area->w = (int) ceil (pts[2] - pts[0]);
        area->h = (int) ceil (pts[3] - pts[1]);
    }
    
    return 1;
}

int
pyg_draw_line (SDL_Surface *surface, SDL_Rect *cliprect, Uint32 color,
    int x1, int _y1, int x2, int y2, int width, SDL_Rect *area)
{
    int pts[4] = { 0 };

    if (!surface)
    {
        SDL_SetError ("surface argument NULL");
        return 0;
    }
    
    if (width < 1)
    {
        SDL_SetError ("width must not be < 1");
        return 0;
    }

    LOCK_SURFACE (surface, 0);
    if (!cliprect)
        cliprect = &surface->clip_rect;

    if (width == 1)
    {
        if (!_clipline (cliprect, x1, _y1, x2, y2, pts))
        {
            UNLOCK_SURFACE (surface);
            return 0;
        }

        if (pts[1] == pts[3])
            _drawhorzline (surface, color, pts[0], pts[1], pts[2]);
        else if (pts[0] == pts[2])
            _drawvertline (surface, color, pts[0], pts[1], pts[3]);
        else
            _drawline (surface, color, pts[0], pts[1], pts[2], pts[3]);
    }
    else
    {
        pts[0] = x1;
        pts[1] = _y1;
        pts[2] = x2;
        pts[3] = y2;
        if (!_drawlinewidth (surface, cliprect, color, pts, width))
        {
            UNLOCK_SURFACE (surface);
            return 0;
        }
    }

    if (area)
    {
        area->x = pts[0];
        area->y = pts[1];
        area->w = pts[2] - pts[0];
        area->h = pts[3] - pts[1];
    }
    UNLOCK_SURFACE (surface);
    return 1;
}

int
pyg_draw_aalines (SDL_Surface *surface, SDL_Rect *cliprect, Uint32 color,
    int *xpts, int *ypts, unsigned int count, int blendargs, SDL_Rect *area)
{
    unsigned int i;
    float xa, ya, xb, yb;
    float pts[4] = { 0 };
    int drawn = 0;
    int rpts[4] = { 0 };

    if (!surface)
    {
        SDL_SetError ("surface argument NULL");
        return 0;
    }
    if (!xpts || !ypts)
    {
        if (!xpts)
            SDL_SetError ("xpts argument NULL");
        else if (!ypts)
            SDL_SetError ("ypts argument NULL");
        return 0;
    }
    if (count == 0)
        return 0;

    LOCK_SURFACE (surface, 0);
    if (!cliprect)
        cliprect = &surface->clip_rect;

    for (i = 0; i < count - 1; i++)
    {
        xa = xpts[i];
        ya = ypts[i];
        xb = xpts[i + 1];
        yb = ypts[i + 1];

        rpts[0] = MIN (xa, rpts[0]);
        rpts[1] = MIN (ya, rpts[1]);
        rpts[2] = MAX (xb, rpts[2]);
        rpts[3] = MAX (yb, rpts[3]);
        
        if (_clipaaline (cliprect, xa, ya, xb, yb, pts))
        {
            drawn = 1;
            _drawaaline (surface, color, (int)pts[0], (int)pts[1], (int)pts[2],
                (int)pts[3], blendargs);
        }
    }

    if (area)
    {
        area->x = rpts[0];
        area->y = rpts[1];
        area->w = rpts[2] - rpts[0];
        area->h = rpts[3] - rpts[1];
    }

    UNLOCK_SURFACE (surface);
    return drawn;
}

int
pyg_draw_lines (SDL_Surface *surface, SDL_Rect *cliprect, Uint32 color,
    int *xpts, int *ypts, unsigned int count, int width, SDL_Rect *area)
{
    unsigned int i;
    int xa, ya, xb, yb;
    int pts[4] = { 0 };
    int drawn = 0;
    int rpts[4] = { 0 };

    if (!surface)
    {
        SDL_SetError ("surface argument NULL");
        return 0;
    }
    if (width < 1)
    {
        SDL_SetError ("width must not be < 1");
        return 0;
    }
    if (!xpts || !ypts)
    {
        if (!xpts)
            SDL_SetError ("xpts argument NULL");
        else if (!ypts)
            SDL_SetError ("ypts argument NULL");
        return 0;
    }
    if (count == 0)
        return 0;

    LOCK_SURFACE (surface, 0);
    if (!cliprect)
        cliprect = &surface->clip_rect;

    for (i = 0; i < count - 1; i++)
    {
        xa = xpts[i];
        ya = ypts[i];
        xb = xpts[i + 1];
        yb = ypts[i + 1];

        rpts[0] = MIN (xa, rpts[0]);
        rpts[1] = MIN (ya, rpts[1]);
        rpts[2] = MAX (xb, rpts[2]);
        rpts[3] = MAX (yb, rpts[3]);
        
        if (width == 1)
        {
            if (_clipline (cliprect, xa, ya, xb, yb, pts))
            {
                drawn = 1;
                if (pts[1] == pts[3])
                    _drawhorzline (surface, color, pts[0], pts[1], pts[2]);
                else if (pts[0] == pts[2])
                    _drawvertline (surface, color, pts[0], pts[1], pts[3]);
                else
                    _drawline (surface, color, pts[0], pts[1], pts[2], pts[3]);
            }
        }
        else
        {
            pts[0] = xa;
            pts[1] = ya;
            pts[2] = xb;
            pts[3] = yb;
            if (_drawlinewidth (surface, cliprect, color, pts, width))
                drawn = 1;
        }
    }

    if (area)
    {
        area->x = rpts[0];
        area->y = rpts[1];
        area->w = rpts[2] - rpts[0];
        area->h = rpts[3] - rpts[1];
    }

    UNLOCK_SURFACE (surface);
    return 1;
}

int
pyg_draw_filled_ellipse (SDL_Surface *surface, SDL_Rect *cliprect,
    Uint32 color, int x, int y, int radx, int rady, SDL_Rect *area)
{
    int pts[4];
    int ix, iy;
    int h, i, j, k;
    int oh, oi, oj, ok;
    
    if (!surface)
    {
        SDL_SetError ("surface argument NULL");
        return 0;
    }
    if (!cliprect)
        cliprect = &surface->clip_rect;

    if (!surface)
    {
        SDL_SetError ("surface argument NULL");
        return 0;
    }
    if (!cliprect)
        cliprect = &surface->clip_rect;

    LOCK_SURFACE (surface, 0);

    if (radx == 0 && rady == 0)
    {
        /* Special case - draw a single pixel */
        SET_PIXEL_AT (surface, surface->format, x, y, color);
        if (area)
        {
            area->x = x;
            area->y = y;
            area->w = 1;
            area->h = 1;
        }
        UNLOCK_SURFACE (surface);
        return 1;
    }
    if (radx == 0)
    {
        /* Special case for radx = 0 - draw a vline */
        if (_clipline (cliprect, x, y - rady, x, y + rady, pts))
        {
            _drawvertline (surface, color, pts[0], pts[1], pts[3]);
            if (area)
            {
                area->x = pts[0];
                area->y = pts[1];
                area->w = 1;
                area->h = pts[3];
            }
            UNLOCK_SURFACE (surface);
            return 1;
        }
        UNLOCK_SURFACE (surface);
        return 0;
    }
    if (rady == 0)
    {
        /* Special case for rady = 0 - draw a hline */
        if (_clipline (cliprect, x - radx, y, x + radx, y, pts))
        {
            _drawhorzline (surface, color, pts[0], pts[1], pts[2]);
            if (area)
            {
                area->x = pts[0];
                area->y = pts[1];
                area->w = pts[2];
                area->h = 1;
            }
            UNLOCK_SURFACE (surface);
            return 1;
        }
        UNLOCK_SURFACE (surface);
        return 0;
    }


    oh = oi = oj = ok = 0xFFFF;
    if (radx >= rady)
    {
        ix = 0;
        iy = radx * 64;
        
        do
        {
            h = (ix + 8) >> 6;
            i = (iy + 8) >> 6;
            j = (h * rady) / radx;
            k = (i * rady) / radx;
            if ((ok != k) && (oj != k) && (k < rady))
            {
                if (_clipline (cliprect, x-h, y-k-1, x+h-1, y-k-1, pts))
                    _drawhorzline (surface, color, pts[0], pts[1], pts[2]);
                if (_clipline (cliprect, x-h, y+k, x+h-1, y+k, pts))
                    _drawhorzline (surface, color, pts[0], pts[1], pts[2]);
                ok = k;
            }
            if ((oj != j) && (ok != j) && (k != j))
            {
                if (_clipline (cliprect, x-i, y+j, x+i-1, y+j, pts))
                    _drawhorzline (surface, color, pts[0], pts[1], pts[2]);
                if (_clipline (cliprect, x-i, y-j-1, x+i-1, y-j-1, pts))
                    _drawhorzline (surface, color, pts[0], pts[1], pts[2]);
                oj = j;
            }
            ix = ix + iy / radx;
            iy = iy - ix / radx;
        }
        while (i > h);
    }
    else
    {
        ix = 0;
        iy = rady * 64;

        do
        {
            h = (ix + 8) >> 6;
            i = (iy + 8) >> 6;
            j = (h * radx) / rady;
            k = (i * radx) / rady;

            if ((oi != i) && (oh != i) && (i < rady))
            {
                if (_clipline (cliprect, x-j, y+i, x+j-1, y+i, pts))
                    _drawhorzline (surface, color, pts[0], pts[1], pts[2]);
                if (_clipline (cliprect, x-j, y-i-1, x+j-1, y-i-i, pts))
                    _drawhorzline (surface, color, pts[0], pts[1], pts[2]);
                oi = i;
            }
            if ((oh != h) && (oi != h) && (i != h))
            {
                if (_clipline (cliprect, x-k, y+h, x+k-1, y+h, pts))
                    _drawhorzline (surface, color, pts[0], pts[1], pts[2]);
                if (_clipline (cliprect, x-k, y-h-1, x+k-1, y-h-1, pts))
                    _drawhorzline (surface, color, pts[0], pts[1], pts[2]);
                oh = h;
            }

            ix = ix + iy / rady;
            iy = iy - ix / rady;

        } while(i > h);
    }

    if (area)
    {
        area->x = x - radx;
        area->y = y - rady;
        area->w = 2 * radx;
        area->h = 2 * rady;
    }
    return 1;
}

int
pyg_draw_ellipse (SDL_Surface *surface, SDL_Rect *cliprect, Uint32 color,
    int x, int y, int radx, int rady, SDL_Rect *area)
{
    int ix, iy;
    int h, i, j, k;
    int oh, oi, oj, ok;
    int xmh, xph, ypk, ymk;
    int xmi, xpi, ymj, ypj;
    int xmj, xpj, ymi, ypi;
    int xmk, xpk, ymh, yph;

    if (!surface)
    {
        SDL_SetError ("surface argument NULL");
        return 0;
    }
    if (!cliprect)
        cliprect = &surface->clip_rect;

    LOCK_SURFACE (surface, 0);

    if (radx == 0 && rady == 0)
    {
        /* Special case - draw a single pixel */
        SET_PIXEL_AT (surface, surface->format, x, y, color);
        if (area)
        {
            area->x = x;
            area->y = y;
            area->w = 1;
            area->h = 1;
        }
        UNLOCK_SURFACE (surface);
        return 1;
    }
    if (radx == 0)
    {
        /* Special case for radx = 0 - draw a vline */
        int pts[4];
        
        if (_clipline (cliprect, x, y - rady, x, y + rady, pts))
        {
            _drawvertline (surface, color, pts[0], pts[1], pts[3]);
            if (area)
            {
                area->x = pts[0];
                area->y = pts[1];
                area->w = 1;
                area->h = pts[3];
            }
            UNLOCK_SURFACE (surface);
            return 1;
        }
        UNLOCK_SURFACE (surface);
        return 0;
    }
    if (rady == 0)
    {
        /* Special case for rady = 0 - draw a hline */
        int pts[4];
        if (_clipline (cliprect, x - radx, y, x + radx, y, pts))
        {
            _drawhorzline (surface, color, pts[0], pts[1], pts[2]);
            if (area)
            {
                area->x = pts[0];
                area->y = pts[1];
                area->w = pts[2];
                area->h = 1;
            }
            UNLOCK_SURFACE (surface);
            return 1;
        }
        UNLOCK_SURFACE (surface);
        return 0;
    }
    
    oh = oi = oj = ok = 0xFFFF;
    if (radx > rady)
    {
        ix = 0;
        iy = radx * 64;
        do
        {
            h = (ix + 16) >> 6;
            i = (iy + 16) >> 6;
            j = (h * rady) / radx;
            k = (i * rady) / radx;
            
            if (((ok!=k) && (oj!=k)) || ((oj != j) && (ok != j)) || (k != j))
            {
                xph = x + h - 1;
                xmh = x - h;
                if (k > 0)
                {
                    ypk = y + k - 1;
                    ymk = y - k;
                    if (h > 0)
                    {
                        SET_PIXEL_AT(surface, surface->format, xmh, ypk, color);
                        SET_PIXEL_AT(surface, surface->format, xmh, ymk, color);
                    }
                    SET_PIXEL_AT (surface, surface->format, xph, ypk, color);
                    SET_PIXEL_AT (surface, surface->format, xph, ymk, color);
                }
                ok = k;
                xpi = x + i - 1;
                xmi = x - i;
                if (j > 0)
                {
                    ypj = y + j - 1;
                    ymj = y - j;
                    SET_PIXEL_AT (surface, surface->format, xmi, ypj, color);
                    SET_PIXEL_AT (surface, surface->format, xpi, ypj, color);
                    SET_PIXEL_AT (surface, surface->format, xmi, ymj, color);
                    SET_PIXEL_AT (surface, surface->format, xpi, ymj, color);
                }
                oj = j;
            }
            ix = ix + iy / radx;
            iy = iy - ix / radx;
                        
        }
        while (i > h);
    }
    else
    {
        ix = 0;
        iy = rady * 64;
        do
        {
            h = (ix + 32) >> 6;
            i = (iy + 32) >> 6;
            j = (h * radx) / rady;
            k = (i * radx) / rady;

            if (((oi!=i) && (oh!=i)) || ((oh!=h) && (oi!=h) && (i!=h)))
            {
                xmj = x -j;
                xpj = x + j - 1;
                if (i > 0)
                {
                    ypi = y + i - 1;
                    ymi = y - i;
                    if (j > 0)
                    {
                        SET_PIXEL_AT(surface, surface->format, xmj, ypi, color);
                        SET_PIXEL_AT(surface, surface->format, xmj, ymi, color);
                    }
                    SET_PIXEL_AT (surface, surface->format, xpj, ypi, color);
                    SET_PIXEL_AT (surface, surface->format, xpj, ymi, color);
                }
                oi = i;
                xmk = x - k;
                xpk = x + k - 1;
                if (h > 0)
                {
                    yph = y + h - 1;
                    ymh = y - h;
                    SET_PIXEL_AT (surface, surface->format, xmk, yph, color);
                    SET_PIXEL_AT (surface, surface->format, xpk, yph, color);
                    SET_PIXEL_AT (surface, surface->format, xmk, ymh, color);
                    SET_PIXEL_AT (surface, surface->format, xpk, ymh, color);
                }
                oh = h;
            }
            ix = ix + iy / rady;
            iy = iy - ix / rady;
        }
        while (i > h);
    }
    UNLOCK_SURFACE (surface);

    if (area)
    {
        area->x = x - radx;
        area->y = y - rady;
        area->w = 2 * radx;
        area->h = 2 * rady;
    }

    return 1;
}

int
pyg_draw_arc (SDL_Surface *surface, SDL_Rect *cliprect, Uint32 color, int x,
    int y, int rad1, int rad2, double anglestart, double anglestop,
    SDL_Rect *area)
{
    double a, step;
    int xlast, xnext, ylast, ynext;
    int rpts[4] = { 0 };
    int drawn = 0;

    if (!surface)
    {
        SDL_SetError ("surface argument NULL");
        return 0;
    }

    if (rad1 < rad2)
    {
        if (rad1 < 1.0e-4)
            step = 1.0;
        else
            step = asin (2.0 / rad1);
    }
    else
    {
        if (rad2 < 1.0e-4)
            step = 1.0;
        else
            step = asin (2.0 / rad2);
    }

    if (step < 0.05)
        step = 0.05;

    LOCK_SURFACE (surface, 0);

    xlast = x + cos (anglestart) * rad1;
    ylast = y - sin (anglestart) * rad2;
    for (a = anglestart + step; a <= anglestop; a += step)
    {
        int pts[4];
        xnext = x + cos (a) * rad1;
        ynext = y - sin (a) * rad2;
        pts[0] = xlast;
        pts[1] = ylast;
        pts[2] = xnext;
        pts[3] = ynext;

        rpts[0] = MIN (xlast, rpts[0]);
        rpts[1] = MIN (ylast, rpts[1]);
        rpts[2] = MAX (xnext, rpts[2]);
        rpts[3] = MAX (ynext, rpts[3]);

        if (_clipline (cliprect, xlast, ylast, xnext, ynext, pts))
        {
            drawn = 1;
            if (pts[1] == pts[3])
                _drawhorzline (surface, color, pts[0], pts[1], pts[2]);
            else if (pts[0] == pts[2])
                _drawvertline (surface, color, pts[0], pts[1], pts[3]);
            else
                _drawline (surface, color, pts[0], pts[1], pts[2], pts[3]);
        }
        xlast = xnext;
        ylast = ynext;
    }

    if (area)
    {
        area->x = rpts[0];
        area->y = rpts[1];
        area->w = rpts[2] - rpts[0];
        area->h = rpts[3] - rpts[1];
    }
    UNLOCK_SURFACE (surface);
    return 1;
}

int
pyg_draw_aapolygon (SDL_Surface *surface, SDL_Rect *cliprect, Uint32 color,
    int *xpts, int *ypts, unsigned int count, int blendargs, SDL_Rect *area)
{
    unsigned int i;
    int anydrawn = 0;
    float minx, maxx, miny, maxy;
    float xa, ya, xb, yb, pts[4];

    if (!surface)
    {
        SDL_SetError ("surface argument NULL");
        return 0;
    }
    if (!xpts || !ypts)
    {
        if (!xpts)
            SDL_SetError ("xpts argument NULL");
        else if (!ypts)
            SDL_SetError ("ypts argument NULL");
        return 0;
    }
    if (count == 0)
        return 0;

    LOCK_SURFACE (surface, 0);
    if (!cliprect)
        cliprect = &surface->clip_rect;

    minx = maxx = miny = maxy = 0;
    for (i = 1; i < count; i++)
    {
        xa = xpts[i - 1];
        xb = xpts[i];
        ya = ypts[i - 1];
        yb = ypts[i];

        minx = MIN (xa, MIN (xb, minx));
        miny = MIN (ya, MIN (yb, miny));
        maxx = MAX (xa, MAX (xb, maxx));
        maxy = MAX (ya, MAX (yb, maxy));

        if (_clipaaline (cliprect, xa, ya, xb, yb, pts))
        {
            _drawaaline (surface, color, (int)pts[0], (int)pts[1], (int)pts[2],
                (int)pts[3], blendargs);
            anydrawn = 1;
        }
    }

    /* Connect the first and last line */
    xa = xpts[0];
    ya = ypts[0];
    xb = xpts[count - 1];
    yb = ypts[count - 1];
    if (_clipaaline (cliprect, xa, ya, xb, yb, pts))
    {
        _drawaaline (surface, color, (int)pts[0], (int)pts[1], (int)pts[2],
            (int)pts[3], blendargs);
        anydrawn = 1;
    }
    
    UNLOCK_SURFACE (surface);

    if (area)
    {
        area->x = (int) floor (minx);
        area->y = (int) floor (miny);
        area->w = (int) ceil (maxx - minx);
        area->h = (int) ceil (maxy - miny);
    }
    return anydrawn;
}

int
pyg_draw_polygon (SDL_Surface *surface, SDL_Rect *cliprect, Uint32 color,
    int *xpts, int *ypts, unsigned int count, int width, SDL_Rect *area)
{
    int anydrawn = 0;
    int minx, maxx, miny, maxy;
    int pts[4];
    unsigned int i;

    if (!surface)
    {
        SDL_SetError ("surface argument NULL");
        return 0;
    }
    if (width < 1)
    {
        SDL_SetError ("width must not be < 1");
        return 0;
    }
    if (!xpts || !ypts)
    {
        if (!xpts)
            SDL_SetError ("xpts argument NULL");
        else if (!ypts)
            SDL_SetError ("ypts argument NULL");
        return 0;
    }
    if (count == 0)
        return 0;

    LOCK_SURFACE (surface, 0);
    if (!cliprect)
        cliprect = &surface->clip_rect;

    minx = maxx = miny = maxy = 0;
    for (i = 1; i < count; i++)
    {
        pts[0] = xpts[i - 1];
        pts[2] = xpts[i];
        pts[1] = ypts[i - 1];
        pts[3] = ypts[i];

        minx = MIN (pts[0], MIN (pts[2], minx));
        miny = MIN (pts[1], MIN (pts[3], miny));
        maxx = MAX (pts[0], MAX (pts[2], maxx));
        maxy = MAX (pts[1], MAX (pts[3], maxy));

        if (_drawlinewidth (surface, cliprect, color, pts, width))
            anydrawn = 1;

    }

    /* Connect the first and last line */
    pts[0] = xpts[0];
    pts[1] = ypts[0];
    pts[2] = xpts[count - 1];
    pts[3] = ypts[count - 1];
    if (_drawlinewidth (surface, cliprect, color, pts, width))
        anydrawn = 1;
    
    UNLOCK_SURFACE (surface);

    if (area)
    {
        area->x = minx;
        area->y = miny;
        area->w = maxx - minx;
        area->h = maxy - miny;
    }
    return anydrawn;
}

int
pyg_draw_filled_polygon (SDL_Surface *surface, SDL_Rect *cliprect, Uint32 color,
    int *xpts, int *ypts, unsigned int count, SDL_Rect *area)
{
    unsigned int i;
    int y;
    int miny, maxy, minx, maxx;
    int x1, _y1, x2, y2;
    unsigned int ind1, ind2;
    int ints, pts[4];
    int *polyints;

    if (!surface)
    {
        SDL_SetError ("surface argument NULL");
        return 0;
    }
    if (!xpts || !ypts)
    {
        if (!xpts)
            SDL_SetError ("xpts argument NULL");
        else if (!ypts)
            SDL_SetError ("ypts argument NULL");
        return 0;
    }
    if (count == 0)
        return 0;

    if (!cliprect)
        cliprect = &surface->clip_rect;

    polyints = malloc (sizeof (int) * count);
    if (!polyints)
    {
        SDL_SetError ("could not allocate memory");
        return 0;
    }

    LOCK_SURFACE (surface, 0);

    /* Determine X and Y maxima */
    miny = maxy = ypts[0];
    minx = maxx = xpts[0];
    for (i = 1; i < count; i++)
    {
        minx = MIN (minx, xpts[i]);
        maxx = MAX (maxx, xpts[i]);
        miny = MIN (miny, ypts[i]);
        maxy = MAX (maxy, ypts[i]);
    }

    /* Draw, scanning y */
    for (y = miny; y <= maxy; y++)
    {
        ints = 0;
        for (i = 0; i < count; i++)
        {
            if (i == 0)
            {
                ind1 = count - 1;
                ind2 = 0;
            }
            else
            {
                ind1 = i - 1;
                ind2 = i;
            }
            
            _y1 = ypts[ind1];
            y2 = ypts[ind2];
            if (_y1 < y2)
            {
                x1 = xpts[ind1];
                x2 = xpts[ind2];
            }
            else if (_y1 > y2)
            {
                y2 = ypts[ind1];
                _y1 = ypts[ind2];
                x2 = xpts[ind1];
                x1 = xpts[ind2];
            }
            else
            {
                continue;
            }
            
            if ((y >= _y1) && (y < y2))
            {
                polyints[ints++] = (y - _y1) * (x2 - x1) / (y2 - _y1) + x1;
            }
            else if ((y == maxy) && (y > _y1) && (y <= y2))
            {
                polyints[ints++] = (y - _y1) * (x2 - x1) / (y2 - _y1) + x1;
            }
        }

        qsort (polyints, (size_t) ints, sizeof(int), _compare_int);

        for (i = 0; i < ((unsigned int)ints); i += 2)
        {
            if (_clipline (cliprect, polyints[i], y, polyints[i + 1], y, pts))
                _drawhorzline (surface, color, pts[0], pts[1], pts[2]);
        }
    }

    free (polyints);
    UNLOCK_SURFACE (surface);

    if (area)
    {
        area->x = minx;
        area->y = miny;
        area->w = maxx - minx;
        area->h = maxy - miny;
    }
    return 1;
}
