/*
  pygame - Python Game Library
  Copyright (C) 2000-2001  Pete Shinners, 2006 Rene Dudfield,
                2007-2010 Marcus von Appen

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

#include "surface_blit.h"

static void alphablit_alpha (SDL_BlitInfo* info);
static void alphablit_colorkey (SDL_BlitInfo* info);
static void alphablit_solid (SDL_BlitInfo* info);

static void 
alphablit_alpha (SDL_BlitInfo * info)
{
    int             n;
    int             width = info->d_width;
    int             height = info->d_height;
    Uint8          *src = info->s_pixels;
    int             srcskip = info->s_skip;
    Uint8          *dst = info->d_pixels;
    int             dstskip = info->d_skip;
    SDL_PixelFormat *srcfmt = info->src;
    SDL_PixelFormat *dstfmt = info->dst;
    int             srcbpp = srcfmt->BytesPerPixel;
    int             dstbpp = dstfmt->BytesPerPixel;
    int             dR, dG, dB, dA, sR, sG, sB, sA;
    Uint32          pixel;

/*    
    printf ("Alpha blit with %d and %d\n", srcbpp, dstbpp);
*/

    if (srcbpp == 1)
    {
        if (dstbpp == 1)
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PALETTE_VALS(src, srcfmt, sR, sG, sB, sA);
                    GET_PALETTE_VALS(dst, dstfmt, dR, dG, dB, dA);
                    ALPHA_BLEND (sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else /* dstbpp > 1 */
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PALETTE_VALS(src, srcfmt, sR, sG, sB, sA);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_RGB_VALS (pixel, dstfmt, dR, dG, dB, dA);
                    ALPHA_BLEND (sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
    else /* srcbpp > 1 */
    {
        if (dstbpp == 1)
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_RGB_VALS (pixel, srcfmt, sR, sG, sB, sA);
                    GET_PALETTE_VALS(dst, dstfmt, dR, dG, dB, dA);
                    ALPHA_BLEND (sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }

        }
        else /* dstbpp > 1 */
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_RGB_VALS (pixel, srcfmt, sR, sG, sB, sA);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_RGB_VALS (pixel, dstfmt, dR, dG, dB, dA);
                    ALPHA_BLEND (sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
}

static void 
alphablit_colorkey (SDL_BlitInfo * info)
{
    int             n;
    int             width = info->d_width;
    int             height = info->d_height;
    Uint8          *src = info->s_pixels;
    int             srcskip = info->s_skip;
    Uint8          *dst = info->d_pixels;
    int             dstskip = info->d_skip;
    SDL_PixelFormat *srcfmt = info->src;
    SDL_PixelFormat *dstfmt = info->dst;
    int             srcbpp = srcfmt->BytesPerPixel;
    int             dstbpp = dstfmt->BytesPerPixel;
    int             dR, dG, dB, dA, sR, sG, sB, sA;
    int             alpha = srcfmt->alpha;
    Uint32          colorkey = srcfmt->colorkey;
    Uint32          pixel;

/*    
    printf ("Colorkey blit with %d and %d\n", srcbpp, dstbpp);
*/  

    if (srcbpp == 1)
    {
        if (dstbpp == 1)
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PALETTE_VALS(src, srcfmt, sR, sG, sB, sA);
                    sA = (*src == colorkey) ? 0 : alpha;
                    GET_PALETTE_VALS(dst, dstfmt, dR, dG, dB, dA);
                    ALPHA_BLEND (sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else /* dstbpp > 1 */
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PALETTE_VALS(src, srcfmt, sR, sG, sB, sA);
                    sA = (*src == colorkey) ? 0 : alpha;
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_RGB_VALS (pixel, dstfmt, dR, dG, dB, dA);
                    ALPHA_BLEND (sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
    else /* srcbpp > 1 */
    {
        if (dstbpp == 1)
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_RGB_VALS (pixel, srcfmt, sR, sG, sB, sA);
                    sA = (pixel == colorkey) ? 0 : alpha;
                    GET_PALETTE_VALS(dst, dstfmt, dR, dG, dB, dA);
                    ALPHA_BLEND (sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }

        }
        else /* dstbpp > 1 */
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_RGB_VALS (pixel, srcfmt, sR, sG, sB, sA);
                    sA = (pixel == colorkey) ? 0 : alpha;
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_RGB_VALS (pixel, dstfmt, dR, dG, dB, dA);
                    ALPHA_BLEND (sR, sG, sB, sA, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
}

static void 
alphablit_solid (SDL_BlitInfo * info)
{
    int             n;
    int             width = info->d_width;
    int             height = info->d_height;
    Uint8          *src = info->s_pixels;
    int             srcskip = info->s_skip;
    Uint8          *dst = info->d_pixels;
    int             dstskip = info->d_skip;
    SDL_PixelFormat *srcfmt = info->src;
    SDL_PixelFormat *dstfmt = info->dst;
    int             srcbpp = srcfmt->BytesPerPixel;
    int             dstbpp = dstfmt->BytesPerPixel;
    int             dR, dG, dB, dA, sR, sG, sB, sA;
    int             alpha = srcfmt->alpha;
    int             pixel;

/*    
    printf ("Solid blit with %d and %d\n", srcbpp, dstbpp);
*/    

    if (srcbpp == 1)
    {
        if (dstbpp == 1)
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PALETTE_VALS(src, srcfmt, sR, sG, sB, sA);
                    GET_PALETTE_VALS(dst, dstfmt, dR, dG, dB, dA);
                    ALPHA_BLEND (sR, sG, sB, alpha, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
        else /* dstbpp > 1 */
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PALETTE_VALS(src, srcfmt, sR, sG, sB, sA);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_RGB_VALS (pixel, dstfmt, dR, dG, dB, dA);
                    ALPHA_BLEND (sR, sG, sB, alpha, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
    else /* srcbpp > 1 */
    {
        if (dstbpp == 1)
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_RGB_VALS (pixel, srcfmt, sR, sG, sB, sA);
                    GET_PALETTE_VALS(dst, dstfmt, dR, dG, dB, dA);
                    ALPHA_BLEND (sR, sG, sB, alpha, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n, width);
                src += srcskip;
                dst += dstskip;
            }

        }
        else /* dstbpp > 1 */
        {
            while (height--)
            {
                LOOP_UNROLLED4(
                {
                    GET_PIXEL(pixel, srcbpp, src);
                    GET_RGB_VALS (pixel, srcfmt, sR, sG, sB, sA);
                    GET_PIXEL (pixel, dstbpp, dst);
                    GET_RGB_VALS (pixel, dstfmt, dR, dG, dB, dA);
                    ALPHA_BLEND (sR, sG, sB, alpha, dR, dG, dB, dA);
                    CREATE_PIXEL(dst, dR, dG, dB, dA, dstbpp, dstfmt);
                    src += srcbpp;
                    dst += dstbpp;
                }, n ,width);
                src += srcskip;
                dst += dstskip;
            }
        }
    }
}

int
pyg_sdlsoftware_blit (SDL_Surface *src, SDL_Rect *srcrect, SDL_Surface *dst,
    SDL_Rect *dstrect, BlendMode blitargs)
{
    int okay;
    int src_locked;
    int dst_locked;
    SDL_Rect sr, dr;
    int srcx, srcy, w, h;

    /* Everything is okay at the beginning...  */
    okay = 1;

    /* Make sure the surfaces aren't locked */
    if (!src || !dst)
    {
        SDL_SetError ("passed a NULL surface");
        return 0;
    }
    if (src->locked || dst->locked)
    {
        SDL_SetError ("surfaces must not be locked during blit");
        return 0;
    }

    /* If the destination rectangle is NULL, use the entire dest surface */
    if (!dstrect)
    {
        dr.x = dr.y = 0;
        dstrect = &dr;
    }

    /* clip the source rectangle to the source surface */
    if (srcrect)
    {
        int maxw, maxh;

        srcx = srcrect->x;
        w = srcrect->w;
        if (srcx < 0)
        {
            w += srcx;
            dstrect->x -= srcx;
            srcx = 0;
        }
        maxw = src->w - srcx;
        if (maxw < w)
            w = maxw;

        srcy = srcrect->y;
        h = srcrect->h;
        if (srcy < 0)
        {
            h += srcy;
            dstrect->y -= srcy;
            srcy = 0;
        }
        maxh = src->h - srcy;
        if (maxh < h)
            h = maxh;

    }
    else
    {
        srcx = srcy = 0;
        w = src->w;
        h = src->h;
    }

    /* clip the destination rectangle against the clip rectangle */
    {
        SDL_Rect *clip = &dst->clip_rect;
        int dx, dy;

        dx = clip->x - dstrect->x;
        if (dx > 0)
        {
            w -= dx;
            dstrect->x += dx;
            srcx += dx;
        }
        dx = dstrect->x + w - clip->x - clip->w;
        if (dx > 0)
            w -= dx;

        dy = clip->y - dstrect->y;
        if (dy > 0)
        {
            h -= dy;
            dstrect->y += dy;
            srcy += dy;
        }
        dy = dstrect->y + h - clip->y - clip->h;
        if (dy > 0)
            h -= dy;
    }

    if (w > 0 && h > 0)
    {
        sr.x = srcx;
        sr.y = srcy;
        sr.w = dstrect->w = w;
        sr.h = dstrect->h = h;
        srcrect = &sr;
    }

    /* Lock the destination if it's in hardware */
    dst_locked = 0;
    if (SDL_MUSTLOCK (dst))
    {
        if (SDL_LockSurface (dst) < 0)
            okay = 0;
        else
            dst_locked = 1;
    }
    /* Lock the source if it's in hardware */
    src_locked = 0;
    if (SDL_MUSTLOCK (src))
    {
        if (SDL_LockSurface (src) < 0)
            okay = 0;
        else
            src_locked = 1;
    }

    /* Set up source and destination buffer pointers, and BLIT! */
    if (okay && srcrect->w && srcrect->h)
    {
        SDL_BlitInfo info;

        /* Set up the blit information */
        info.s_pixels = (Uint8 *) src->pixels + src->offset +
            (Uint16) srcrect->y * src->pitch +
            (Uint16) srcrect->x * src->format->BytesPerPixel;
        info.s_width = srcrect->w;
        info.s_height = srcrect->h;
        info.s_skip = src->pitch - info.s_width * src->format->BytesPerPixel;
        info.d_pixels = (Uint8 *) dst->pixels + dst->offset +
            (Uint16) dstrect->y * dst->pitch +
            (Uint16) dstrect->x * dst->format->BytesPerPixel;
        info.d_width = dstrect->w;
        info.d_height = dstrect->h;
        info.d_skip = dst->pitch - info.d_width * dst->format->BytesPerPixel;
        info.s_pitch = src->pitch;
        info.d_pitch = dst->pitch;
        info.src = src->format;
        info.dst = dst->format;

        switch (blitargs)
        {
        case 0:
        {
            if (src->flags & SDL_SRCALPHA && src->format->Amask)
                alphablit_alpha (&info);
            else if (src->flags & SDL_SRCCOLORKEY)
                alphablit_colorkey (&info);
            else
                alphablit_solid (&info);
            break;
        }
        case BLEND_RGB_ADD:
        {
            blit_blend_rgb_add (&info);
            break;
        }
        case BLEND_RGB_SUB:
        {
            blit_blend_rgb_sub (&info);
            break;
        }
        case BLEND_RGB_MULT:
        {
            blit_blend_rgb_mul (&info);
            break;
        }
        case BLEND_RGB_MIN:
        {
            blit_blend_rgb_min (&info);
            break;
        }
        case BLEND_RGB_MAX:
        {
            blit_blend_rgb_max (&info);
            break;
        }
        case BLEND_RGB_AND:
        {
            blit_blend_rgb_and (&info);
            break;
        }
        case BLEND_RGB_OR:
        {
            blit_blend_rgb_or (&info);
            break;
        }
        case BLEND_RGB_XOR:
        {
            blit_blend_rgb_xor (&info);
            break;
        }
        case BLEND_RGB_DIFF:
        {
            blit_blend_rgb_diff (&info);
            break;
        }
        case BLEND_RGB_SCREEN:
        {
            blit_blend_rgb_screen (&info);
            break;
        }
        case BLEND_RGB_AVG:
        {
            blit_blend_rgb_avg (&info);
            break;
        }

        case BLEND_RGBA_ADD:
        {
            blit_blend_rgba_add (&info);
            break;
        }
        case BLEND_RGBA_SUB:
        {
            blit_blend_rgba_sub (&info);
            break;
        }
        case BLEND_RGBA_MULT:
        {
            blit_blend_rgba_mul (&info);
            break;
        }
        case BLEND_RGBA_MIN:
        {
            blit_blend_rgba_min (&info);
            break;
        }
        case BLEND_RGBA_MAX:
        {
            blit_blend_rgba_max (&info);
            break;
        }
        case BLEND_RGBA_AND:
        {
            blit_blend_rgba_and (&info);
            break;
        }
        case BLEND_RGBA_OR:
        {
            blit_blend_rgba_or (&info);
            break;
        }
        case BLEND_RGBA_XOR:
        {
            blit_blend_rgba_xor (&info);
            break;
        }
        case BLEND_RGBA_DIFF:
        {
            blit_blend_rgba_diff (&info);
            break;
        }
        case BLEND_RGBA_SCREEN:
        {
            blit_blend_rgba_screen (&info);
            break;
        }
        case BLEND_RGBA_AVG:
        {
            blit_blend_rgba_avg (&info);
            break;
        }

        default:
        {
            SDL_SetError ("invalid blit argument");
            okay = 0;
            break;
        }
        }
    }
    
    /* We need to unlock the surfaces if they're locked */
    if (dst_locked)
        SDL_UnlockSurface (dst);
    if (src_locked)
        SDL_UnlockSurface (src);

    /* Blit is done! */
    return (okay ? 1 : 0);
}

int
pyg_sdlsurface_scroll (SDL_Surface *surface, int dx, int dy)
{
    int bpp, pitch, w, h;
    int locked = 0;
    SDL_Rect *cliprect;
    Uint8 *src, *dst;

    if (!surface)
    {
        SDL_SetError ("passed a NULL surface");
        return 0;
    }
    if (surface->locked)
    {
        SDL_SetError ("surface must not be locked during scroll");
        return 0;
    }

    if ((surface->flags & SDL_OPENGL) &&
        !(surface->flags & (SDL_OPENGLBLIT & ~SDL_OPENGL)))
    {
        SDL_SetError ("cannot scroll an OPENGL Surface (OPENGLBLIT is ok)");
        return 0;
    }

    if (dx == 0 && dy == 0)
        return 1;

    cliprect = &surface->clip_rect;
    w = cliprect->w;
    h = cliprect->h;
    if (dx >= w || dx <= -w || dy >= h || dy <= -h)
        return 1;

    if (SDL_MUSTLOCK (surface))
    {
        if (SDL_LockSurface (surface) < 0)
            return 0;
        else
            locked = 1;
    }

    bpp = surface->format->BytesPerPixel;
    pitch = surface->pitch;
    src = dst = (Uint8 *) surface->pixels +
                cliprect->y * pitch + cliprect->x * bpp;
    if (dx >= 0)
    {
        w -= dx;
        if (dy > 0)
        {
            h -= dy;
            dst += dy * pitch + dx * bpp;
        }
        else
        {
            h += dy;
            src -= dy * pitch;
            dst += dx * bpp;
        }
    }
    else
    {
        w += dx;
        if (dy > 0)
        {
            h -= dy;
            src -= dx * bpp;
            dst += dy * pitch;
        }
        else
        {
            h += dy;
            src -= dy * pitch + dx * bpp;
        }
    }

    if (src < dst)
    {
        src += (h - 1) * pitch;
        dst += (h - 1) * pitch;
        pitch = -pitch;
    }
    while (h--)
    {
        memmove (dst, src, (size_t) (w * bpp));
        src += pitch;
        dst += pitch;
    }

    if (locked)
        SDL_UnlockSurface (surface);

    return 1;
}
