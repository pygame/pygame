/*
  pygame - Python Game Library
  Copyright (C) 2000-2001 Pete Shinners, 2007-2009 Marcus von Appen

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
#ifndef _PYGAME_SDLSURFACE_H_
#define _PYGAME_SDLSURFACE_H_

#include <SDL.h>
#include "pgdefines.h"

typedef enum
{
    BLEND_RGB_ADD = 1,
    BLEND_RGB_SUB,
    BLEND_RGB_MULT,
    BLEND_RGB_MIN,
    BLEND_RGB_MAX,
    BLEND_RGB_AND,
    BLEND_RGB_OR,
    BLEND_RGB_XOR,
    BLEND_RGB_DIFF,
    BLEND_RGB_SCREEN,
    BLEND_RGB_AVG,
    
    BLEND_RGBA_ADD,
    BLEND_RGBA_SUB,
    BLEND_RGBA_MULT,
    BLEND_RGBA_MIN,
    BLEND_RGBA_MAX,
    BLEND_RGBA_AND,
    BLEND_RGBA_OR,
    BLEND_RGBA_XOR,
    BLEND_RGBA_DIFF,
    BLEND_RGBA_SCREEN,
    BLEND_RGBA_AVG
} BlendMode;

#define RGB2FORMAT(rgb,format)                                          \
    if (format->palette == NULL)                                        \
    {                                                                   \
        Uint8 _r,_g,_b;                                                 \
        _r = (Uint8) ((rgb & 0xff0000) >> 16);                          \
        _g = (Uint8) ((rgb & 0x00ff00) >> 8);                           \
        _b = (Uint8)  (rgb & 0x0000ff);                                 \
        rgb = (_r >> format->Rloss) << format->Rshift |                 \
            (_g >> format->Gloss) << format->Gshift |                   \
            (_b >> format->Bloss) << format->Bshift | format->Amask;    \
    }                                                                   \
    else                                                                \
    {                                                                   \
        rgb = SDL_MapRGB (format,                                       \
            ((Uint8)((rgb & 0xff0000) >> 16)),                          \
            ((Uint8)((rgb & 0x00ff00) >>  8)),                          \
            ((Uint8)(rgb & 0x0000ff)));                                 \
    }

#define ARGB2FORMAT(argb,format)                                        \
    if (format->palette == NULL)                                        \
    {                                                                   \
        Uint8 _r,_g,_b, _a;                                             \
        _a = (Uint8) ((argb & 0xff000000) >> 24);                       \
        _r = (Uint8) ((argb & 0x00ff0000) >> 16);                       \
        _g = (Uint8) ((argb & 0x0000ff00) >> 8);                        \
        _b = (Uint8)  (argb & 0x000000ff);                              \
        argb = (_r >> format->Rloss) << format->Rshift |                \
            (_g >> format->Gloss) << format->Gshift |                   \
            (_b >> format->Bloss) << format->Bshift |                   \
            ((_a >> format->Aloss) << format->Ashift & format->Amask);  \
    }                                                                   \
    else                                                                \
    {                                                                   \
        argb = SDL_MapRGBA (format,                                     \
            ((Uint8)((argb & 0x00ff0000) >> 16)),                       \
            ((Uint8)((argb & 0x0000ff00) >>  8)),                       \
            ((Uint8) (argb & 0x000000ff)),                              \
            ((Uint8)((argb & 0xff000000) >> 24)));                      \
    }

#define GET_RGB_VALS(pixel, fmt, r, g, b, a)                            \
        r = (pixel & fmt->Rmask) >> fmt->Rshift;                        \
        r = (r << fmt->Rloss) + (r >> (8 - (fmt->Rloss << 1)));         \
        g = (pixel & fmt->Gmask) >> fmt->Gshift;                        \
        g = (g << fmt->Gloss) + (g >> (8 - (fmt->Gloss << 1)));         \
        b = (pixel & fmt->Bmask) >> fmt->Bshift;                        \
        b = (b << fmt->Bloss) + (b >> (8 - (fmt->Bloss << 1)));         \
        if (fmt->Amask)                                                 \
        {                                                               \
            a = (pixel & fmt->Amask) >> fmt->Ashift;                    \
            a = (a << fmt->Aloss) + (a >> (8 - (fmt->Aloss << 1)));     \
        }                                                               \
        else                                                            \
            a = 255;

#define GET_PALETTE_VALS(pixel, fmt, sr, sg, sb, sa)       \
    sr = fmt->palette->colors[*((Uint8 *) (pixel))].r;     \
    sg = fmt->palette->colors[*((Uint8 *) (pixel))].g;     \
    sb = fmt->palette->colors[*((Uint8 *) (pixel))].b;     \
    sa = 255;

#if SDL_BYTEORDER == SDL_LIL_ENDIAN
#define GET_PIXEL24(b) (b[0] + (b[1] << 8) + (b[2] << 16))
#define SET_PIXEL24_RGB(buf,format,r,g,b)                               \
    *((buf) + ((format)->Rshift >> 3)) = r;                             \
    *((buf) + ((format)->Gshift >> 3)) = g;                             \
    *((buf) + ((format)->Bshift >> 3)) = b;
#define SET_PIXEL24(buf,format,rgb)                                     \
    *((buf) + ((format)->Rshift >> 3)) = (rgb)[0];                      \
    *((buf) + ((format)->Gshift >> 3)) = (rgb)[1];                      \
    *((buf) + ((format)->Bshift >> 3)) = (rgb)[2];
#else
#define GET_PIXEL24(b) (b[2] + (b[1] << 8) + (b[0] << 16))
#define SET_PIXEL24_RGB(buf,format,r,g,b)                               \
    *((buf) + 2 - ((format)->Rshift >> 3)) = r;                         \
    *((buf) + 2 - ((format)->Gshift >> 3)) = g;                         \
    *((buf) + 2 - ((format)->Bshift >> 3)) = b;
#define SET_PIXEL24(buf,format,rgb)                                     \
    *((buf) + 2 - ((format)->Rshift >> 3)) = (rgb)[0];                  \
    *((buf) + 2 - ((format)->Gshift >> 3)) = (rgb)[1];                  \
    *((buf) + 2 - ((format)->Bshift >> 3)) = (rgb)[2];
#endif

#define SET_PIXEL_AT(surface,format,_x,_y,color)                        \
    if ((_x) >= (surface)->clip_rect.x &&                               \
        (_x) <= (surface)->clip_rect.x + (surface)->clip_rect.w &&      \
        (_y) >= (surface)->clip_rect.y &&                               \
        (_y) <= (surface)->clip_rect.y + (surface)->clip_rect.h)        \
    {                                                                   \
        switch ((format)->BytesPerPixel)                                \
        {                                                               \
        case 1:                                                         \
            *((Uint8*) ((Uint8*)(surface)->pixels) + (_y) *             \
                (surface)->pitch + (_x)) = (Uint8)(color);              \
            break;                                                      \
        case 2:                                                         \
            *((Uint16*)(((Uint8*)(surface)->pixels) + (_y) *            \
                    (surface)->pitch) + (_x)) = (Uint16)(color);        \
            break;                                                      \
        case 4:                                                         \
            *((Uint32*)(((Uint8*)(surface)->pixels) + (_y) *            \
                    (surface)->pitch) + (_x)) = (color);                \
            break;                                                      \
        default:                                                        \
        {                                                               \
            Uint8* _buf, _rgb[3];                                       \
            SDL_GetRGB ((color), (format), _rgb, _rgb+1, _rgb+2);       \
            _buf = (Uint8*)(((Uint8*)(surface)->pixels) + (_y) *        \
                (surface)->pitch) + (_x) * 3;                           \
            SET_PIXEL24(_buf, format, _rgb);                            \
            break;                                                      \
        }                                                               \
        }                                                               \
    }

#define GET_PIXEL_AT(pxl,surface,bpp,_x,_y)                             \
    switch ((bpp))                                                      \
    {                                                                   \
    case 1:                                                             \
        pxl = *((Uint8*) ((Uint8*)(surface)->pixels) + (_y) *           \
            (surface)->pitch + (_x));                                   \
        break;                                                          \
    case 2:                                                             \
        pxl = *((Uint16*)(((Uint8*)(surface)->pixels) + (_y) *          \
                (surface)->pitch) + (_x));                              \
        break;                                                          \
    case 4:                                                             \
        pxl = *((Uint32*)(((Uint8*)(surface)->pixels) + (_y) *          \
                (surface)->pitch) + (_x));                              \
        break;                                                          \
    default:                                                            \
    {                                                                   \
        Uint8* buf = ((Uint8 *) (((Uint8*)(surface)->pixels) + (_y) *   \
                (surface)->pitch) + (_x) * 3);                          \
        pxl = GET_PIXEL24(buf);                                         \
        break;                                                          \
    }                                                                   \
    }

#define GET_PIXEL(pxl, bpp, source)               \
    switch (bpp)                                  \
    {                                             \
    case 1:                                       \
        pxl = *((Uint8 *)(source));               \
        break;                                    \
    case 2:                                       \
        pxl = *((Uint16 *) (source));             \
        break;                                    \
    case 4:                                       \
        pxl = *((Uint32 *) (source));             \
        break;                                    \
    default:                                      \
    {                                             \
        Uint8 *b = (Uint8 *) source;              \
        pxl = GET_PIXEL24(b);                     \
        break;                                    \
    }                                             \
    }

#define CREATE_PIXEL(buf, r, g, b, a, bp, ft)                   \
    switch (bp)                                                 \
    {                                                           \
    case 1:                                                     \
        *((Uint8 *)buf) = (Uint8) SDL_MapRGB (ft, r, g, b);     \
        break;                                                  \
    case 2:                                                     \
        *((Uint16 *) (buf)) =                                   \
            ((r >> ft->Rloss) << ft->Rshift) |                  \
            ((g >> ft->Gloss) << ft->Gshift) |                  \
            ((b >> ft->Bloss) << ft->Bshift) |                  \
            ((a >> ft->Aloss) << ft->Ashift & ft->Amask);       \
        break;                                                  \
    case 4:                                                     \
        *((Uint32 *) (buf)) =                                   \
            ((r >> ft->Rloss) << ft->Rshift) |                  \
            ((g >> ft->Gloss) << ft->Gshift) |                  \
            ((b >> ft->Bloss) << ft->Bshift) |                  \
            ((a >> ft->Aloss) << ft->Ashift & ft->Amask);       \
        break;                                                  \
    default:                                                    \
    {                                                           \
        SET_PIXEL24_RGB(buf, ft, r,g,b);                        \
        break;                                                  \
    }                                                           \
    }

#define LOOP_UNROLLED4(code, n, width) \
    n = (width + 3) / 4;               \
    switch (width & 3)                 \
    {                                  \
    case 0: do { code;                 \
        case 3: code;                  \
        case 2: code;                  \
        case 1: code;                  \
        } while (--n > 0);             \
    }

#define REPEAT_4(code) \
    code;              \
    code;              \
    code;              \
    code;

#define REPEAT_3(code) \
    code;              \
    code;              \
    code;

#define D_BLEND_RGB_ADD(tmp, sR, sG, sB, dR, dG, dB)    \
    tmp = dR + sR; dR = (tmp <= 255 ? tmp : 255);       \
    tmp = dG + sG; dG = (tmp <= 255 ? tmp : 255);       \
    tmp = dB + sB; dB = (tmp <= 255 ? tmp : 255);

#define D_BLEND_RGB_SUB(tmp, sR, sG, sB, dR, dG, dB)    \
    tmp = dR - sR; dR = (tmp > 0 ? tmp : 0);            \
    tmp = dG - sG; dG = (tmp > 0 ? tmp : 0);            \
    tmp = dB - sB; dB = (tmp > 0 ? tmp : 0);

#define D_BLEND_RGB_MULT(sR, sG, sB, dR, dG, dB)    \
    dR = (dR && sR) ? (dR * sR) >> 8 : 0;           \
    dG = (dG && sG) ? (dG * sG) >> 8 : 0;           \
    dB = (dB && sB) ? (dB * sB) >> 8 : 0;

#define D_BLEND_RGB_MIN(sR, sG, sB, dR, dG, dB)     \
    if(sR < dR) { dR = sR; }                        \
    if(sG < dG) { dG = sG; }                        \
    if(sB < dB) { dB = sB; }

#define D_BLEND_RGB_MAX(sR, sG, sB, dR, dG, dB)     \
    if(sR > dR) { dR = sR; }                        \
    if(sG > dG) { dG = sG; }                        \
    if(sB > dB) { dB = sB; }

#define D_BLEND_RGB_XOR(sR, sG, sB, dR, dG, dB)     \
    dR = MIN (255, MAX(sR ^ dR, 0));                \
    dG = MIN (255, MAX(sG ^ dG, 0));                \
    dB = MIN (255, MAX(sB ^ dB, 0));

#define D_BLEND_RGB_AND(sR, sG, sB, dR, dG, dB)     \
    dR = MIN (255, MAX(sR & dR, 0));                \
    dG = MIN (255, MAX(sG & dG, 0));                \
    dB = MIN (255, MAX(sB & dB, 0));

#define D_BLEND_RGB_OR(sR, sG, sB, dR, dG, dB)      \
    dR = MIN (255, MAX(sR | dR, 0));                \
    dG = MIN (255, MAX(sG | dG, 0));                \
    dB = MIN (255, MAX(sB | dB, 0));

#define D_BLEND_RGB_DIFF(sR, sG, sB, dR, dG, dB)    \
    dR = ABS((int)dR - (int)sR);                    \
    dG = ABS((int)dG - (int)sG);                    \
    dB = ABS((int)dB - (int)sB);

#define D_BLEND_RGB_SCREEN(sR, sG, sB, dR, dG, dB)  \
    dR = 255 - ((255 - sR) * (255 - dR) >> 8);      \
    dG = 255 - ((255 - sG) * (255 - dG) >> 8);      \
    dB = 255 - ((255 - sB) * (255 - dB) >> 8);

#define D_BLEND_RGB_AVG(sR, sG, sB, dR, dG, dB) \
    dR = (sR + dR) >> 1;                        \
    dG = (sG + dG) >> 1;                        \
    dB = (sB + dB) >> 1;
    
#define D_BLEND_RGBA_ADD(tmp, sR, sG, sB, sA, dR, dG, dB, dA)       \
    tmp = dR + sR; dR = (tmp <= 255 ? tmp : 255);                   \
    tmp = dG + sG; dG = (tmp <= 255 ? tmp : 255);                   \
    tmp = dB + sB; dB = (tmp <= 255 ? tmp : 255);                   \
    tmp = dA + sA; dA = (tmp <= 255 ? tmp : 255);

#define D_BLEND_RGBA_SUB(tmp, sR, sG, sB, sA, dR, dG, dB, dA)       \
    tmp = dR - sR; dR = (tmp > 0 ? tmp : 0);                        \
    tmp = dG - sG; dG = (tmp > 0 ? tmp : 0);                        \
    tmp = dB - sB; dB = (tmp > 0 ? tmp : 0);                        \
    tmp = dA - sA; dA = (tmp > 0 ? tmp : 0);

#define D_BLEND_RGBA_MULT(sR, sG, sB, sA, dR, dG, dB, dA)   \
    dR = (dR && sR) ? (dR * sR) >> 8 : 0;                   \
    dG = (dG && sG) ? (dG * sG) >> 8 : 0;                   \
    dB = (dB && sB) ? (dB * sB) >> 8 : 0;                   \
    dA = (dA && sA) ? (dA * sA) >> 8 : 0;

#define D_BLEND_RGBA_MIN(sR, sG, sB, sA, dR, dG, dB, dA)    \
    if(sR < dR) { dR = sR; }                                \
    if(sG < dG) { dG = sG; }                                \
    if(sB < dB) { dB = sB; }                                \
    if(sA < dA) { dA = sA; }

#define D_BLEND_RGBA_MAX(sR, sG, sB, sA, dR, dG, dB, dA)    \
    if(sR > dR) { dR = sR; }                                \
    if(sG > dG) { dG = sG; }                                \
    if(sB > dB) { dB = sB; }                                \
    if(sA > dA) { dA = sA; }

#define D_BLEND_RGBA_XOR(sR, sG, sB, sA, dR, dG, dB, dA) \
    dR = MIN (255, MAX(sR ^ dR, 0));                     \
    dG = MIN (255, MAX(sG ^ dG, 0));                     \
    dB = MIN (255, MAX(sB ^ dB, 0));                     \
    dA = MIN (255, MAX(sA ^ dA, 0));

#define D_BLEND_RGBA_AND(sR, sG, sB, sA, dR, dG, dB, dA) \
    dR = MIN (255, MAX(sR & dR, 0));                     \
    dG = MIN (255, MAX(sG & dG, 0));                     \
    dB = MIN (255, MAX(sB & dB, 0));                     \
    dA = MIN (255, MAX(sA & dA, 0));

#define D_BLEND_RGBA_OR(sR, sG, sB, sA, dR, dG, dB, dA) \
    dR = MIN (255, MAX(sR | dR, 0));                    \
    dG = MIN (255, MAX(sG | dG, 0));                    \
    dB = MIN (255, MAX(sB | dB, 0));                    \
    dA = MIN (255, MAX(sA | dA, 0));

#define D_BLEND_RGBA_DIFF(sR, sG, sB, sA, dR, dG, dB, dA)        \
    dR = ABS((int)dR - (int)sR);                                 \
    dG = ABS((int)dG - (int)sG);                                 \
    dB = ABS((int)dB - (int)sB);                                 \
    dA = ABS((int)dA - (int)sA);

#define D_BLEND_RGBA_SCREEN(sR, sG, sB, sA, dR, dG, dB, dA) \
    dR = 255 - ((255 - sR) * (255 - dR) >> 8);              \
    dG = 255 - ((255 - sG) * (255 - dG) >> 8);              \
    dB = 255 - ((255 - sB) * (255 - dB) >> 8);              \
    dA = 255 - ((255 - sA) * (255 - dA) >> 8);

#define D_BLEND_RGBA_AVG(sR, sG, sB, sA, dR, dG, dB, dA)   \
    dR = (sR + dR) >> 1;                                   \
    dG = (sG + dG) >> 1;                                   \
    dB = (sB + dB) >> 1;                                   \
    dA = (sA + dA) >> 1;

#if 1
/* Choose an alpha blend equation. If the sign is preserved on a right shift
 * then use a specialized, faster, equation. Otherwise a more general form,
 * where all additions are done before the shift, is needed.
*/
#if (-1 >> 1) < 0
#define ALPHA_BLEND_COMP(sC, dC, sA) ((((sC - dC) * sA + sC) >> 8) + dC)
#else
#define ALPHA_BLEND_COMP(sC, dC, sA) (((dC << 8) + (sC - dC) * sA + sC) >> 8)
#endif

#define ALPHA_BLEND(sR, sG, sB, sA, dR, dG, dB, dA) \
    do {                                            \
        if (dA)                                     \
        {                                           \
            dR = ALPHA_BLEND_COMP(sR, dR, sA);      \
            dG = ALPHA_BLEND_COMP(sG, dG, sA);      \
            dB = ALPHA_BLEND_COMP(sB, dB, sA);      \
            dA = sA + dA - ((sA * dA) / 255);       \
        }                                           \
        else                                        \
        {                                           \
            dR = sR;                                \
            dG = sG;                                \
            dB = sB;                                \
            dA = sA;                                \
        }                                           \
    } while(0)
#elif 0
#define ALPHA_BLEND(sR, sG, sB, sA, dR, dG, dB, dA)    \
    do {                                               \
        if(sA){                                        \
            if(dA && sA < 255){                        \
                int dContrib = dA*(255 - sA)/255;      \
                dA = sA+dA - ((sA*dA)/255);            \
                dR = (dR*dContrib + sR*sA)/dA;         \
                dG = (dG*dContrib + sG*sA)/dA;         \
                dB = (dB*dContrib + sB*sA)/dA;         \
            }else{                                     \
                dR = sR;                               \
                dG = sG;                               \
                dB = sB;                               \
                dA = sA;                               \
            }                                          \
        }                                              \
    } while(0)
#endif


#define CLIP_RECT_TO_SURFACE(sf,r)                                      \
    if ((r)->x > (sf)->w || (r)->y > (sf)->h ||                         \
        ((r)->x + (r)->w) <= 0 || ((r)->y + (r)->h) <= 0)               \
    {                                                                   \
        (r)->x = (r)->y = (r)->w = (r)->h = 0;                          \
    }                                                                   \
    else                                                                \
    {                                                                   \
        (r)->x = MAX((r)->x, 0);                                        \
        (r)->y = MAX((r)->y, 0);                                        \
        (r)->w = (Uint16) MIN((r)->x + (r)->w, (sf)->w) - (r)->x;       \
        (r)->h = (Uint16) MIN((r)->y + (r)->h, (sf)->h) - (r)->y;       \
    }

int
pyg_sdlsurface_fill_blend (SDL_Surface *surface, SDL_Rect *rect, Uint32 color,
    int blendargs);

int pyg_sdlsurface_save (SDL_Surface *surface, char *filename, char *type);
int pyg_sdlsurface_save_rw (SDL_Surface *surface, SDL_RWops *rw, char *type,
    int freerw);

int
pyg_sdlsoftware_blit (SDL_Surface *src, SDL_Rect *srcrect, SDL_Surface *dst,
    SDL_Rect *dstrect, BlendMode blitargs);

int pyg_sdlsurface_scroll (SDL_Surface *surface, int dx, int dy);

#endif /* _PYGAME_SDLSURFACE_H_ */
