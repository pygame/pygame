/*
  pygame - Python Game Library
  Copyright (C) 2000-2001 Pete Shinners, 2007-2008 Marcus von Appen

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

#define PYGAME_BLEND_RGB_ADD  0x1
#define PYGAME_BLEND_RGB_SUB  0x2
#define PYGAME_BLEND_RGB_MULT 0x3
#define PYGAME_BLEND_RGB_MIN  0x4
#define PYGAME_BLEND_RGB_MAX  0x5

#define PYGAME_BLEND_ADD  PYGAME_BLEND_RGB_ADD
#define PYGAME_BLEND_SUB  PYGAME_BLEND_RGB_SUB
#define PYGAME_BLEND_MULT PYGAME_BLEND_RGB_MULT
#define PYGAME_BLEND_MIN  PYGAME_BLEND_RGB_MIN
#define PYGAME_BLEND_MAX  PYGAME_BLEND_RGB_MAX

#define PYGAME_BLEND_RGBA_ADD  0x6
#define PYGAME_BLEND_RGBA_SUB  0x7
#define PYGAME_BLEND_RGBA_MULT 0x8
#define PYGAME_BLEND_RGBA_MIN  0x9
#define PYGAME_BLEND_RGBA_MAX  0x10

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

#if SDL_BYTEORDER == SDL_LIL_ENDIAN
#define SET_PIXEL24(buf,format,rgb)                                     \
    *((buf) + ((format)->Rshift >> 3)) = (rgb)[0];                      \
    *((buf) + ((format)->Gshift >> 3)) = (rgb)[1];                      \
    *((buf) + ((format)->Bshift >> 3)) = (rgb)[2];
#else
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
            SET_PIXEL24(_buf, format, rgb);                             \
            break;                                                      \
        }                                                               \
        }                                                               \
    }

#if SDL_BYTEORDER == SDL_LIL_ENDIAN
#define GET_PIXEL_24(b) (b[0] + (b[1] << 8) + (b[2] << 16))
#else
#define GET_PIXEL_24(b) (b[2] + (b[1] << 8) + (b[0] << 16))
#endif

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
        pxl = GET_PIXEL_24(b);                                          \
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
        pxl = GET_PIXEL_24(b);                    \
    }                                             \
    break;                                        \
    }

#define CREATE_PIXEL(buf, r, g, b, a, bp, ft)     \
    switch (bp)                                   \
    {                                             \
    case 2:                                       \
        *((Uint16 *) (buf)) =                     \
            ((r >> ft->Rloss) << ft->Rshift) |    \
            ((g >> ft->Gloss) << ft->Gshift) |    \
            ((b >> ft->Bloss) << ft->Bshift) |    \
            ((a >> ft->Aloss) << ft->Ashift);     \
        break;                                    \
    case 4:                                       \
        *((Uint32 *) (buf)) =                     \
            ((r >> ft->Rloss) << ft->Rshift) |    \
            ((g >> ft->Gloss) << ft->Gshift) |    \
            ((b >> ft->Bloss) << ft->Bshift) |    \
            ((a >> ft->Aloss) << ft->Ashift);     \
        break;                                    \
    }

#define GET_RGB_VALS(pixel, fmt, r, g, b, a)                            \
    r = (((pixel & fmt->Rmask) >> fmt->Rshift) << fmt->Rloss);          \
    g = (((pixel & fmt->Gmask) >> fmt->Gshift) << fmt->Gloss);          \
    b = (((pixel & fmt->Bmask) >> fmt->Bshift) << fmt->Bloss);          \
    if (fmt->Amask)                                                     \
        a = (((pixel & fmt->Amask) >> fmt->Ashift) << fmt->Aloss);      \
    else                                                                \
        a =  255;

#define GET_PALETTE_VALS(pixel, fmt, sr, sg, sb, sa)       \
    sr = fmt->palette->colors[*((Uint8 *) (pixel))].r;     \
    sg = fmt->palette->colors[*((Uint8 *) (pixel))].g;     \
    sb = fmt->palette->colors[*((Uint8 *) (pixel))].b;     \
    sa = 255;

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

#define BLEND_ADD(tmp, sR, sG, sB, dR, dG, dB)          \
    tmp = dR + sR; dR = (tmp <= 255 ? tmp : 255);       \
    tmp = dG + sG; dG = (tmp <= 255 ? tmp : 255);       \
    tmp = dB + sB; dB = (tmp <= 255 ? tmp : 255);

#define BLEND_SUB(tmp, sR, sG, sB, dR, dG, dB)         \
    tmp = dR - sR; dR = (tmp > 0 ? tmp : 0);           \
    tmp = dG - sG; dG = (tmp > 0 ? tmp : 0);           \
    tmp = dB - sB; dB = (tmp > 0 ? tmp : 0);

#define BLEND_MULT(sR, sG, sB, dR, dG, dB)         \
    dR = (dR && sR) ? (dR * sR) >> 8 : 0;          \
    dG = (dG && sG) ? (dG * sG) >> 8 : 0;          \
    dB = (dB && sB) ? (dB * sB) >> 8 : 0;

#define BLEND_MIN(sR, sG, sB, dR, dG, dB)         \
    if(sR < dR) { dR = sR; }                      \
    if(sG < dG) { dG = sG; }                      \
    if(sB < dB) { dB = sB; }

#define BLEND_MAX(sR, sG, sB, dR, dG, dB)         \
    if(sR > dR) { dR = sR; }                      \
    if(sG > dG) { dG = sG; }                      \
    if(sB > dB) { dB = sB; }

#define BLEND_RGBA_ADD(tmp, sR, sG, sB, sA, dR, dG, dB, dA)     \
    tmp = dR + sR; dR = (tmp <= 255 ? tmp : 255);               \
    tmp = dG + sG; dG = (tmp <= 255 ? tmp : 255);               \
    tmp = dB + sB; dB = (tmp <= 255 ? tmp : 255);               \
    tmp = dA + sA; dA = (tmp <= 255 ? tmp : 255);

#define BLEND_RGBA_SUB(tmp, sR, sG, sB, sA, dR, dG, dB, dA)     \
    tmp = dR - sR; dR = (tmp > 0 ? tmp : 0);                    \
    tmp = dG - sG; dG = (tmp > 0 ? tmp : 0);                    \
    tmp = dB - sB; dB = (tmp > 0 ? tmp : 0);                    \
    tmp = dA - sA; dA = (tmp > 0 ? tmp : 0);

#define BLEND_RGBA_MULT(sR, sG, sB, sA, dR, dG, dB, dA) \
    dR = (dR && sR) ? (dR * sR) >> 8 : 0;               \
    dG = (dG && sG) ? (dG * sG) >> 8 : 0;               \
    dB = (dB && sB) ? (dB * sB) >> 8 : 0;               \
    dA = (dA && sA) ? (dA * sA) >> 8 : 0;

#define BLEND_RGBA_MIN(sR, sG, sB, sA, dR, dG, dB, dA) \
    if(sR < dR) { dR = sR; }                           \
    if(sG < dG) { dG = sG; }                           \
    if(sB < dB) { dB = sB; }                           \
    if(sA < dA) { dA = sA; }

#define BLEND_RGBA_MAX(sR, sG, sB, sA, dR, dG, dB, dA) \
    if(sR > dR) { dR = sR; }                           \
    if(sG > dG) { dG = sG; }                           \
    if(sB > dB) { dB = sB; }                           \
    if(sA > dA) { dA = sA; }

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

int
pyg_sdlsurface_fill_blend (SDL_Surface *surface, SDL_Rect *rect, Uint32 color,
    int blendargs);

int pyg_sdlsurface_save (SDL_Surface *surface, char *filename, char *type);
int pyg_sdlsurface_save_rw (SDL_Surface *surface, SDL_RWops *rw, char *type,
    int freerw);

int
pyg_sdlsoftware_blit (SDL_Surface *src, SDL_Rect *srcrect, SDL_Surface *dst,
    SDL_Rect *dstrect, int blitargs);


#endif /* _PYGAME_SDLSURFACE_H_ */
