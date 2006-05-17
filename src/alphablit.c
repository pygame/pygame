/*
    pygame - Python Game Library
    Copyright (C) 2000-2001  Pete Shinners
    Copyright (C) 2006  Rene Dudfield

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

#include <SDL.h>
#include <SDL.h>


#define PYGAME_BLEND_ADD  0x1
#define PYGAME_BLEND_SUB  0x2
#define PYGAME_BLEND_MULT 0x3
#define PYGAME_BLEND_MIN  0x4
#define PYGAME_BLEND_MAX  0x5



/* The structure passed to the low level blit functions */
typedef struct {
        Uint8 *s_pixels;
        int s_width;
        int s_height;
        int s_skip;
        Uint8 *d_pixels;
        int d_width;
        int d_height;
        int d_skip;
        void *aux_data;
        SDL_PixelFormat *src;
        Uint8 *table;
        SDL_PixelFormat *dst;
} SDL_BlitInfo;
static void alphablit_alpha(SDL_BlitInfo *info);
static void alphablit_colorkey(SDL_BlitInfo *info);
static void alphablit_solid(SDL_BlitInfo *info);
static void blit_blend_THEM(SDL_BlitInfo *info, int the_args);



static int SoftBlitPyGame(SDL_Surface *src, SDL_Rect *srcrect,
                          SDL_Surface *dst, SDL_Rect *dstrect, int the_args);
extern int SDL_RLESurface(SDL_Surface *surface);
extern void SDL_UnRLESurface(SDL_Surface *surface, int recode);












static int SoftBlitPyGame(SDL_Surface *src, SDL_Rect *srcrect,
                        SDL_Surface *dst, SDL_Rect *dstrect, int the_args)
{
        int okay;
        int src_locked;
        int dst_locked;

    /* Everything is okay at the beginning...  */
        okay = 1;

        /* Lock the destination if it's in hardware */
        dst_locked = 0;
        if ( SDL_MUSTLOCK(dst) ) {
                if ( SDL_LockSurface(dst) < 0 )
                        okay = 0;
                else
                        dst_locked = 1;
        }
        /* Lock the source if it's in hardware */
        src_locked = 0;
        if ( SDL_MUSTLOCK(src) ) {
                if ( SDL_LockSurface(src) < 0 )
                        okay = 0;
                else
                        src_locked = 1;
        }

        /* Set up source and destination buffer pointers, and BLIT! */
        if ( okay  && srcrect->w && srcrect->h ) {
                SDL_BlitInfo info;

                /* Set up the blit information */
                info.s_pixels = (Uint8 *)src->pixels + src->offset +
                                (Uint16)srcrect->y*src->pitch +
                                (Uint16)srcrect->x*src->format->BytesPerPixel;
                info.s_width = srcrect->w;
                info.s_height = srcrect->h;
                info.s_skip=src->pitch-info.s_width*src->format->BytesPerPixel;
                info.d_pixels = (Uint8 *)dst->pixels + dst->offset +
                                (Uint16)dstrect->y*dst->pitch +
                                (Uint16)dstrect->x*dst->format->BytesPerPixel;
                info.d_width = dstrect->w;
                info.d_height = dstrect->h;
                info.d_skip=dst->pitch-info.d_width*dst->format->BytesPerPixel;
                info.src = src->format;
                info.dst = dst->format;

                switch(the_args) {
                    case 0:
                    {
                        if(src->flags&SDL_SRCALPHA && src->format->Amask)
                            alphablit_alpha(&info);
                        else if(src->flags & SDL_SRCCOLORKEY)
                            alphablit_colorkey(&info);
                        else
                            alphablit_solid(&info);
                        break;
                    }
                    case PYGAME_BLEND_ADD:
                    case PYGAME_BLEND_SUB:
                    case PYGAME_BLEND_MULT:
                    case PYGAME_BLEND_MIN:
                    case PYGAME_BLEND_MAX: {
                        blit_blend_THEM(&info, the_args); 
                        break;
                    }
                    default:
                    {
                        SDL_SetError("Invalid argument passed to blit.");
                        okay = 0;
                        break;
                    }
                }

        }

        /* We need to unlock the surfaces if they're locked */
        if ( dst_locked )
                SDL_UnlockSurface(dst);
        if ( src_locked )
                SDL_UnlockSurface(src);
        /* Blit is done! */
        return(okay ? 0 : -1);
}



/*
 * Macros below are for the blit blending functions.
 */


#define GET_PIXEL(buf, bpp, fmt, pixel)                    \
do {                                                                       \
        switch (bpp) {                                                           \
                case 1:                                                           \
                        pixel = *((Uint8 *)(buf));                           \
                break;                                                       \
                case 2:                                                           \
                        pixel = *((Uint16 *)(buf));                           \
                break;                                                           \
                case 4:                                                           \
                        pixel = *((Uint32 *)(buf));                           \
                break;                                                           \
                default:        {/* case 3: FIXME: broken code (no alpha) */                   \
                        Uint8 *b = (Uint8 *)buf;                           \
                        if(SDL_BYTEORDER == SDL_LIL_ENDIAN) {                   \
                                pixel = b[0] + (b[1] << 8) + (b[2] << 16); \
                        } else {                                           \
                                pixel = (b[0] << 16) + (b[1] << 8) + b[2]; \
                        }                                                   \
                }                                                           \
                break;                                                           \
        }                                                                   \
} while(0)


#define DISEMBLE_RGBA(buf, bpp, fmt, pixel, R, G, B, A)                    \
do {                                                                       \
        if(bpp==1){\
            pixel = *((Uint8 *)(buf));                           \
            R = fmt->palette->colors[pixel].r; \
            G = fmt->palette->colors[pixel].g; \
            B = fmt->palette->colors[pixel].b; \
            A = 255; \
        } else { \
        switch (bpp) {                                                           \
                case 2:                                                           \
                        pixel = *((Uint16 *)(buf));                           \
                break;                                                           \
                case 4:                                                           \
                        pixel = *((Uint32 *)(buf));                           \
                break;                                                           \
                default:        {/* case 3: FIXME: broken code (no alpha) */                   \
                        Uint8 *b = (Uint8 *)buf;                           \
                        if(SDL_BYTEORDER == SDL_LIL_ENDIAN) {                   \
                                pixel = b[0] + (b[1] << 8) + (b[2] << 16); \
                        } else {                                           \
                                pixel = (b[0] << 16) + (b[1] << 8) + b[2]; \
                        }                                                   \
                }                                                           \
                break;                                                           \
            }                                                                   \
            R = ((pixel&fmt->Rmask)>>fmt->Rshift)<<fmt->Rloss;                 \
            G = ((pixel&fmt->Gmask)>>fmt->Gshift)<<fmt->Gloss;                 \
            B = ((pixel&fmt->Bmask)>>fmt->Bshift)<<fmt->Bloss;                 \
            A = ((pixel&fmt->Amask)>>fmt->Ashift)<<fmt->Aloss;                 \
        }\
} while(0)


#define DISEMBLE_RGBA4(buf, bpp, fmt, pixel, R, G, B, A)                    \
                        pixel = *((Uint32 *)(buf));                           \
            R = ((pixel&fmt->Rmask)>>fmt->Rshift)<<fmt->Rloss;                 \
            G = ((pixel&fmt->Gmask)>>fmt->Gshift)<<fmt->Gloss;                 \
            B = ((pixel&fmt->Bmask)>>fmt->Bshift)<<fmt->Bloss;                 \
            A = ((pixel&fmt->Amask)>>fmt->Ashift)<<fmt->Aloss;                 \


#define PIXEL_FROM_RGBA(pixel, fmt, r, g, b, a)                         \
{                                                                       \
        pixel = ((r>>fmt->Rloss)<<fmt->Rshift)|                                \
                ((g>>fmt->Gloss)<<fmt->Gshift)|                                \
                ((b>>fmt->Bloss)<<fmt->Bshift)|                                \
                ((a<<fmt->Aloss)<<fmt->Ashift);                                \
}
#define ASSEMBLE_RGBA(buf, bpp, fmt, r, g, b, a)                        \
{                                                                       \
        switch (bpp) {                                                        \
                case 2: {                                                \
                        Uint16 pixel;                                        \
                        PIXEL_FROM_RGBA(pixel, fmt, r, g, b, a);        \
                        *((Uint16 *)(buf)) = pixel;                        \
                }                                                        \
                break;                                                        \
                case 4: {                                                \
                        Uint32 pixel;                                        \
                        PIXEL_FROM_RGBA(pixel, fmt, r, g, b, a);        \
                        *((Uint32 *)(buf)) = pixel;                        \
                }                                                        \
                break;                                                        \
        }                                                                \
}


#define ASSEMBLE_RGBA4(buf, bpp, fmt, r, g, b, a)                        \
                        Uint32 pixel;                                        \
                        PIXEL_FROM_RGBA(pixel, fmt, r, g, b, a);        \
                        *((Uint32 *)(buf)) = pixel;                        \



#if 0
#define ALPHA_BLEND(sR, sG, sB, sA, dR, dG, dB, dA)  \
do {                                            \
        dR = (((sR-dR)*(sA))>>8)+dR;                \
        dG = (((sG-dG)*(sA))>>8)+dG;                \
        dB = (((sB-dB)*(sA))>>8)+dB;                \
        dA = sA+dA - ((sA*dA)/255);                \
} while(0)
#else
#define ALPHA_BLEND(sR, sG, sB, sA, dR, dG, dB, dA)  \
do {   if(dA){\
        dR = ((dR<<8) + (sR-dR)*sA + sR) >> 8;	   \
        dG = ((dG<<8) + (sG-dG)*sA + sG) >> 8;     \
        dB = ((dB<<8) + (sB-dB)*sA + sB) >> 8;	   \
        dA = sA+dA - ((sA*dA)/255);                \
    }else{\
        dR = sR;                \
        dG = sG;                \
        dB = sB;                \
        dA = sA;               \
    }\
} while(0)
#endif




#define BLEND_TOP_VARS \
        int n,ii; \
        int width = info->d_width; \
        int height = info->d_height; \
        Uint8 *src = info->s_pixels; \
        int srcskip = info->s_skip; \
        Uint8 *dst = info->d_pixels; \
        int dstskip = info->d_skip; \
        SDL_PixelFormat *srcfmt = info->src; \
        SDL_PixelFormat *dstfmt = info->dst; \
        int srcbpp = srcfmt->BytesPerPixel; \
        int dstbpp = dstfmt->BytesPerPixel; \
        Uint8 dR, dG, dB, dA, sR, sG, sB, sA; \
        Uint32 pixel; \
        Uint32 tmp; \
        Sint32 tmp2; \
        ii = tmp = tmp2 = 0 ; \



#define BLEND_TOP \
        while ( height-- ) \
        { \
            for(n=width; n>0; --n) \
            { \


#define BLEND_TOP_GENERIC \
        BLEND_TOP; \
        DISEMBLE_RGBA(src, srcbpp, srcfmt, pixel, sR, sG, sB, sA); \
        DISEMBLE_RGBA(dst, dstbpp, dstfmt, pixel, dR, dG, dB, dA); \


#define BLEND_BOTTOM \
            } \
            src += srcskip; \
            dst += dstskip; \
        } \

#define BLEND_BOTTOM_GENERIC \
                ASSEMBLE_RGBA(dst, dstbpp, dstfmt, dR, dG, dB, dA); \
                src += srcbpp; \
                dst += dstbpp; \
                BLEND_BOTTOM; \


#define BLEND_TOP_4 \
    if(srcfmt->BytesPerPixel == 4 && dstfmt->BytesPerPixel == 4) { \
        BLEND_TOP;  \
            for(ii=0;ii < 3; ii++){ \

#define BLEND_START_GENERIC \
                src++;dst++; \
            } \
            src++;dst++; \
        BLEND_BOTTOM;  \
    } else { \
        BLEND_TOP_GENERIC;  \
// NOTE: we don't touch alpha.


#define BLEND_END_GENERIC \
        BLEND_BOTTOM_GENERIC; \
    } \


/*
 * Blending functions for the 32bit routines.
 */

#define BLEND_ADD4(S,D)  \
    tmp = (D) + (S);  (D) = (tmp <= 255 ? tmp: 255); \

#define BLEND_SUB4(S,D)  \
    tmp2 = (D)-(S); (D) = (tmp2 >= 0 ? tmp2 : 0); \

#define BLEND_MULT4(S,D)  \
    tmp = ((D) * (S)) >> 8;  (D) = (tmp <= 255 ? tmp: 255); \

#define BLEND_MIN4(S,D)  \
    if ((S) < (D)) { (D) = (S); } \

#define BLEND_MAX4(S,D)  \
    if ((S) > (D)) { (D) = (S); } \


/*
 * These are the dissasembled blending functions.
 */

#define BLEND_ADD(sR, sG, sB, sA, dR, dG, dB, dA)  \
    dR = (dR+sR <= 255 ? dR+sR: 255); \
    dG = (dG+sG <= 255 ? dG+sG : 255); \
    dB = (dB+sB <= 255 ? dB+sB : 255); \


#define BLEND_SUB(sR, sG, sB, sA, dR, dG, dB, dA)  \
    tmp2 = dR - sR; dR = (tmp2 >= 0 ? tmp2 : 0); \
    tmp2 = dG - sG; dG = (tmp2 >= 0 ? tmp2 : 0); \
    tmp2 = dB - sB; dB = (tmp2 >= 0 ? tmp2 : 0); \


#define BLEND_MULT(sR, sG, sB, sA, dR, dG, dB, dA)  \
    dR = (dR * sR) >> 8; \
    dG = (dG * sG) >> 8; \
    dB = (dB * sB) >> 8; \

#define BLEND_MIN(sR, sG, sB, sA, dR, dG, dB, dA)  \
    if(sR < dR) { dR = sR; } \
    if(sG < dG) { dG = sG; } \
    if(sB < dB) { dB = sB; } \

#define BLEND_MAX(sR, sG, sB, sA, dR, dG, dB, dA)  \
    if(sR > dR) { dR = sR; } \
    if(sG > dG) { dG = sG; } \
    if(sB > dB) { dB = sB; } \


/*
 * blit_blend takes the blending args, and then uses that to select the 
 *  correct code for blending with.
 */
static void blit_blend_THEM(SDL_BlitInfo *info, int the_args) {
    BLEND_TOP_VARS;

    switch(the_args) {
        /*
         * We use macros to keep the code shorter.
         * First we see if it is a 32bit RGBA surface.  If so we have some 
         *  special case code for that.  Otherwise we use the generic code.
         */
        case PYGAME_BLEND_ADD: {
            BLEND_TOP_4;
            BLEND_ADD4(*src,*dst); 
            BLEND_START_GENERIC; 
            BLEND_ADD(sR, sG, sB, sA, dR, dG, dB, dA); 
            BLEND_END_GENERIC;
            break;
        }
        case PYGAME_BLEND_SUB: {
            BLEND_TOP_4;
            BLEND_SUB4(*src,*dst); 
            BLEND_START_GENERIC; 
            BLEND_SUB(sR, sG, sB, sA, dR, dG, dB, dA); 
            BLEND_END_GENERIC;
            break;
        }
        case PYGAME_BLEND_MULT: {
            BLEND_TOP_4;
            BLEND_MULT4(*src,*dst); 
            BLEND_START_GENERIC; 
            BLEND_MULT(sR, sG, sB, sA, dR, dG, dB, dA); 
            BLEND_END_GENERIC;
            break;
        }
        case PYGAME_BLEND_MIN: {
            BLEND_TOP_4;
            BLEND_MIN4(*src,*dst); 
            BLEND_START_GENERIC; 
            BLEND_MIN(sR, sG, sB, sA, dR, dG, dB, dA); 
            BLEND_END_GENERIC;
            break;
        }
        case PYGAME_BLEND_MAX: {
            BLEND_TOP_4;
            BLEND_MAX4(*src,*dst); 
            BLEND_START_GENERIC; 
            BLEND_MAX(sR, sG, sB, sA, dR, dG, dB, dA); 
            BLEND_END_GENERIC;
            break;
        }

    }
}





static void alphablit_alpha(SDL_BlitInfo *info)
{
        int n;
        int width = info->d_width;
        int height = info->d_height;
        Uint8 *src = info->s_pixels;
        int srcskip = info->s_skip;
        Uint8 *dst = info->d_pixels;
        int dstskip = info->d_skip;
        SDL_PixelFormat *srcfmt = info->src;
        SDL_PixelFormat *dstfmt = info->dst;
        int srcbpp = srcfmt->BytesPerPixel;
        int dstbpp = dstfmt->BytesPerPixel;
        int dR, dG, dB, dA, sR, sG, sB, sA;

        while ( height-- )
        {
            for(n=width; n>0; --n)
            {
                Uint32 pixel;
                DISEMBLE_RGBA(src, srcbpp, srcfmt, pixel, sR, sG, sB, sA);
                DISEMBLE_RGBA(dst, dstbpp, dstfmt, pixel, dR, dG, dB, dA);
                ALPHA_BLEND(sR, sG, sB, sA, dR, dG, dB, dA);
                ASSEMBLE_RGBA(dst, dstbpp, dstfmt, dR, dG, dB, dA);
                src += srcbpp;
                dst += dstbpp;
            }
            src += srcskip;
            dst += dstskip;
        }
}

static void alphablit_colorkey(SDL_BlitInfo *info)
{
        int n;
        int width = info->d_width;
        int height = info->d_height;
        Uint8 *src = info->s_pixels;
        int srcskip = info->s_skip;
        Uint8 *dst = info->d_pixels;
        int dstskip = info->d_skip;
        SDL_PixelFormat *srcfmt = info->src;
        SDL_PixelFormat *dstfmt = info->dst;
        int srcbpp = srcfmt->BytesPerPixel;
        int dstbpp = dstfmt->BytesPerPixel;
        int dR, dG, dB, dA, sR, sG, sB, sA;
        int alpha = srcfmt->alpha;
        Uint32 colorkey = srcfmt->colorkey;

        while ( height-- )
        {
            for(n=width; n>0; --n)
            {
                Uint32 pixel;
                DISEMBLE_RGBA(dst, dstbpp, dstfmt, pixel, dR, dG, dB, dA);
                DISEMBLE_RGBA(src, srcbpp, srcfmt, pixel, sR, sG, sB, sA);
                sA = (pixel == colorkey) ? 0 : alpha;
                ALPHA_BLEND(sR, sG, sB, sA, dR, dG, dB, dA);
                ASSEMBLE_RGBA(dst, dstbpp, dstfmt, dR, dG, dB, dA);
                src += srcbpp;
                dst += dstbpp;
            }
            src += srcskip;
            dst += dstskip;
        }
}


static void alphablit_solid(SDL_BlitInfo *info)
{
        int n;
        int width = info->d_width;
        int height = info->d_height;
        Uint8 *src = info->s_pixels;
        int srcskip = info->s_skip;
        Uint8 *dst = info->d_pixels;
        int dstskip = info->d_skip;
        SDL_PixelFormat *srcfmt = info->src;
        SDL_PixelFormat *dstfmt = info->dst;
        int srcbpp = srcfmt->BytesPerPixel;
        int dstbpp = dstfmt->BytesPerPixel;
        int dR, dG, dB, dA, sR, sG, sB, sA;
        int alpha = srcfmt->alpha;

        while ( height-- )
        {
            for(n=width; n>0; --n)
            {
                int pixel;
                DISEMBLE_RGBA(dst, dstbpp, dstfmt, pixel, dR, dG, dB, dA);
                DISEMBLE_RGBA(src, srcbpp, srcfmt, pixel, sR, sG, sB, sA);
                ALPHA_BLEND(sR, sG, sB, alpha, dR, dG, dB, dA);
                ASSEMBLE_RGBA(dst, dstbpp, dstfmt, dR, dG, dB, dA);
                src += srcbpp;
                dst += dstbpp;
            }
            src += srcskip;
            dst += dstskip;
        }
}



/*we assume the "dst" has pixel alpha*/
int pygame_Blit(SDL_Surface *src, SDL_Rect *srcrect,
                   SDL_Surface *dst, SDL_Rect *dstrect, int the_args)
{
        SDL_Rect fulldst;
        int srcx, srcy, w, h;

        /* Make sure the surfaces aren't locked */
        if ( ! src || ! dst ) {
                SDL_SetError("SDL_UpperBlit: passed a NULL surface");
                return(-1);
        }
        if ( src->locked || dst->locked ) {
                SDL_SetError("Surfaces must not be locked during blit");
                return(-1);
        }

        /* If the destination rectangle is NULL, use the entire dest surface */
        if ( dstrect == NULL ) {
                fulldst.x = fulldst.y = 0;
                dstrect = &fulldst;
        }

        /* clip the source rectangle to the source surface */
        if(srcrect) {
                int maxw, maxh;

                srcx = srcrect->x;
                w = srcrect->w;
                if(srcx < 0) {
                        w += srcx;
                        dstrect->x -= srcx;
                        srcx = 0;
                }
                maxw = src->w - srcx;
                if(maxw < w)
                        w = maxw;

                srcy = srcrect->y;
                h = srcrect->h;
                if(srcy < 0) {
                        h += srcy;
                        dstrect->y -= srcy;
                        srcy = 0;
                }
                maxh = src->h - srcy;
                if(maxh < h)
                        h = maxh;

        } else {
                srcx = srcy = 0;
                w = src->w;
                h = src->h;
        }

        /* clip the destination rectangle against the clip rectangle */
        {
                SDL_Rect *clip = &dst->clip_rect;
                int dx, dy;

                dx = clip->x - dstrect->x;
                if(dx > 0) {
                        w -= dx;
                        dstrect->x += dx;
                        srcx += dx;
                }
                dx = dstrect->x + w - clip->x - clip->w;
                if(dx > 0)
                        w -= dx;

                dy = clip->y - dstrect->y;
                if(dy > 0) {
                        h -= dy;
                        dstrect->y += dy;
                        srcy += dy;
                }
                dy = dstrect->y + h - clip->y - clip->h;
                if(dy > 0)
                        h -= dy;
        }

        if(w > 0 && h > 0) {
                SDL_Rect sr;
                sr.x = srcx;
                sr.y = srcy;
                sr.w = dstrect->w = w;
                sr.h = dstrect->h = h;
                return SoftBlitPyGame(src, &sr, dst, dstrect, the_args);
        }
        dstrect->w = dstrect->h = 0;
        return 0;
}


int pygame_AlphaBlit (SDL_Surface *src, SDL_Rect *srcrect,
                   SDL_Surface *dst, SDL_Rect *dstrect)
{
    return pygame_Blit(src, srcrect, dst, dstrect, 0);
}


