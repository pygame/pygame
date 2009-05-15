/*
  pygame - Python Game Library
  Copyright (C) 2000-2001 Pete Shinners, 2006 Rene Dudfield

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
#include "pgdefines.h"
#include "tga.h"

/*******************************************************/
/* tga code by Mattias Engdegård, in the public domain */
/*******************************************************/
struct TGAheader
{
    Uint8 infolen;		/* length of info field */
    Uint8 has_cmap;		/* 1 if image has colormap, 0 otherwise */
    Uint8 type;

    Uint8 cmap_start[2];	/* index of first colormap entry */
    Uint8 cmap_len[2];		/* number of entries in colormap */
    Uint8 cmap_bits;		/* bits per colormap entry */

    Uint8 yorigin[2];		/* image origin (ignored here) */
    Uint8 xorigin[2];
    Uint8 width[2];		/* image size */
    Uint8 height[2];
    Uint8 pixel_bits;		/* bits/pixel */
    Uint8 flags;
};

enum tga_type
{
    TGA_TYPE_INDEXED = 1,
    TGA_TYPE_RGB = 2,
    TGA_TYPE_BW = 3,
    TGA_TYPE_RLE = 8		/* additive */
};

#define TGA_INTERLEAVE_MASK 0xc0
#define TGA_INTERLEAVE_NONE 0x00
#define TGA_INTERLEAVE_2WAY 0x40
#define TGA_INTERLEAVE_4WAY 0x80

#define TGA_ORIGIN_MASK  0x30
#define TGA_ORIGIN_LEFT  0x00
#define TGA_ORIGIN_RIGHT 0x10
#define TGA_ORIGIN_LOWER 0x00
#define TGA_ORIGIN_UPPER 0x20

/* read/write unaligned little-endian 16-bit ints */
#define LE16(p) ((p)[0] + ((p)[1] << 8))
#define SETLE16(p, v) ((p)[0] = (v), (p)[1] = (v) >> 8)

#define TGA_RLE_MAX 128		/* max length of a TGA RLE chunk */

static int _rle_line (Uint8 *src, Uint8 *dst, int w, int bpp);

/* return the number of bytes in the resulting buffer after RLE-encoding
   a line of TGA data */
static int
_rle_line (Uint8 *src, Uint8 *dst, int w, int bpp)
{
    int x = 0;
    int out = 0;
    int raw = 0;
    while (x < w)
    {
	Uint32 pix;
	int x0 = x;
	memcpy (&pix, src + x * bpp, (size_t) bpp);
	x++;
	while (x < w && memcmp (&pix, src + x * bpp, (size_t)bpp) == 0
            && x - x0 < TGA_RLE_MAX)
	    x++;
	/* use a repetition chunk iff the repeated pixels would consume
	   two bytes or more */
	if ((x - x0 - 1) * bpp >= 2 || x == w)
        {
	    /* output previous raw chunks */
	    while (raw < x0)
            {
		int n = MIN (TGA_RLE_MAX, x0 - raw);
		dst[out++] = n - 1;
		memcpy (dst + out, src + raw * bpp, (size_t)(n * bpp));
		out += n * bpp;
		raw += n;
	    }

	    if (x - x0 > 0)
            {
		/* output new repetition chunk */
		dst[out++] = 0x7f + x - x0;
		memcpy (dst + out, &pix, (size_t)bpp);
		out += bpp;
	    }
	    raw = x;
	}
    }
    return out;
}

int
pyg_save_tga_rw (SDL_Surface *surface, SDL_RWops *out, int rle, int freerw)
{
    SDL_Surface *linebuf = NULL;
    int retval = 0;
    int alpha = 0;
    int ckey = -1;
    struct TGAheader h;
    int srcbpp;
    unsigned surf_flags;
    unsigned surf_alpha;
    Uint32 rmask, gmask, bmask, amask;
    SDL_Rect r;
    int bpp;
    Uint8 *rlebuf = NULL;

    if (!surface)
    {
        SDL_SetError ("surface argument NULL");
        return 0;
    }
    if (!out)
    {
        SDL_SetError ("out argument NULL");
        return 0;
    }

    h.infolen = 0;
    SETLE16 (h.cmap_start, 0);

    srcbpp = surface->format->BitsPerPixel;
    if (srcbpp < 8)
    {
        SDL_SetError ("cannot save <8bpp images as TGA");
        return 0;
    }

    if (srcbpp == 8)
    {
        h.has_cmap = 1;
        h.type = TGA_TYPE_INDEXED;
        if (surface->flags & SDL_SRCCOLORKEY)
        {
            ckey = surface->format->colorkey;
            h.cmap_bits = 32;
        }
        else
            h.cmap_bits = 24;
        SETLE16 (h.cmap_len, surface->format->palette->ncolors);
        h.pixel_bits = 8;
        rmask = gmask = bmask = amask = 0;
    }
    else
    {
        h.has_cmap = 0;
        h.type = TGA_TYPE_RGB;
        h.cmap_bits = 0;
        SETLE16 (h.cmap_len, 0);
        if (surface->format->Amask)
        {
            alpha = 1;
            h.pixel_bits = 32;
        }
        else
            h.pixel_bits = 24;
        if (SDL_BYTEORDER == SDL_BIG_ENDIAN)
        {
            int s = alpha ? 0 : 8;
            amask = 0x000000ff >> s;
            rmask = 0x0000ff00 >> s;
            gmask = 0x00ff0000 >> s;
            bmask = 0xff000000 >> s;
        }
        else
        {
            amask = alpha ? 0xff000000 : 0;
            rmask = 0x00ff0000;
            gmask = 0x0000ff00;
            bmask = 0x000000ff;
        }
    }
    bpp = h.pixel_bits >> 3;
    if (rle)
        h.type += TGA_TYPE_RLE;

    SETLE16 (h.yorigin, 0);
    SETLE16 (h.xorigin, 0);
    SETLE16 (h.width, surface->w);
    SETLE16 (h.height, surface->h);
    h.flags = TGA_ORIGIN_UPPER | (alpha ? 8 : 0);

    if (!SDL_RWwrite (out, &h, sizeof (h), 1))
        return 0;

    if (h.has_cmap)
    {
        int i;
        SDL_Palette *pal = surface->format->palette;
        Uint8 entry[4];
        for (i = 0; i < pal->ncolors; i++)
        {
            entry[0] = pal->colors[i].b;
            entry[1] = pal->colors[i].g;
            entry[2] = pal->colors[i].r;
            entry[3] = (i == ckey) ? 0 : 0xff;
            if (!SDL_RWwrite (out, entry, h.cmap_bits >> 3, 1))
                return 0;
        }
    }

    linebuf = SDL_CreateRGBSurface (SDL_SWSURFACE, surface->w, 1, h.pixel_bits,
        rmask, gmask, bmask, amask);
    if (!linebuf)
        return 0;
    if (h.has_cmap)
        SDL_SetColors (linebuf, surface->format->palette->colors, 0,
            surface->format->palette->ncolors);
    if (rle)
    {
        rlebuf = malloc
            ((size_t)(bpp * surface->w + 1 + surface->w / TGA_RLE_MAX));
        if (!rlebuf)
        {
            SDL_SetError ("out of memory");
            goto error;
        }
    }

    /* Temporarily remove colourkey and alpha from surface so copies are
       opaque */
    surf_flags = surface->flags & (SDL_SRCALPHA | SDL_SRCCOLORKEY);
    surf_alpha = surface->format->alpha;
    if (surf_flags & SDL_SRCALPHA)
        SDL_SetAlpha (surface, 0, 255);
    if (surf_flags & SDL_SRCCOLORKEY)
        SDL_SetColorKey (surface, 0, surface->format->colorkey);

    r.x = 0;
    r.w = surface->w;
    r.h = 1;
    for (r.y = 0; r.y < surface->h; r.y++)
    {
        int n;
        void *buf;
        if (SDL_BlitSurface (surface, &r, linebuf, NULL) < 0)
            break;
        if (rle)
        {
            buf = rlebuf;
            n = _rle_line (linebuf->pixels, rlebuf, surface->w, bpp);
        }
        else
        {
            buf = linebuf->pixels;
            n = surface->w * bpp;
        }
        if (!SDL_RWwrite (out, buf, n, 1))
            break;
    }

    /* restore flags */
    if (surf_flags & SDL_SRCALPHA)
        SDL_SetAlpha (surface, SDL_SRCALPHA, (Uint8)surf_alpha);
    if (surf_flags & SDL_SRCCOLORKEY)
        SDL_SetColorKey (surface, SDL_SRCCOLORKEY, surface->format->colorkey);

    if (freerw)
        SDL_RWclose (out);
    retval = 1;

error:
    if (rlebuf)
        free (rlebuf);
    SDL_FreeSurface (linebuf);
    
    return retval;
}

int
pyg_save_tga (SDL_Surface *surface, char *file, int rle)
{
    int ret;
    SDL_RWops *out;

    if (!surface)
    {
        SDL_SetError ("surface argument NULL");
        return 0;
    }
    if (!file)
    {
        SDL_SetError ("file argument NULL");
        return 0;
    }

    out = SDL_RWFromFile (file, "wb");
    if (!out)
        return 0;

    return pyg_save_tga_rw (surface, out, rle, 1);
}
