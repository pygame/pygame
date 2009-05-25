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

*/

#include "filters.h"

#if defined(__GNUC__)
#if defined(__x86_64__)
#include "filters_64.c"
#elif defined(__i386__)
#include "filters_32.c"
#endif
#endif /* __GNUC__ */

FilterType
pyg_filter_init_filterfuncs (FilterFuncs *filters, FilterType type)
{
    if (!filters)
        return 0;
    
    filters->type = FILTER_C;
    filters->shrink_X = pyg_filter_shrink_X_C;
    filters->shrink_Y = pyg_filter_shrink_Y_C;
    filters->expand_X = pyg_filter_expand_X_C;
    filters->expand_Y = pyg_filter_expand_Y_C;
    
#if defined(FILTERS_SUPPORT_MMX)
    if (type == FILTER_MMX && SDL_HasMMX ())
    {
        filters->type = FILTER_MMX;
        filters->shrink_X = pyg_filter_shrink_X_MMX;
        filters->shrink_Y = pyg_filter_shrink_Y_MMX;
        filters->expand_X = pyg_filter_expand_X_MMX;
        filters->expand_Y = pyg_filter_expand_Y_MMX;
    }
#endif /* FILTERS_SUPPORT_MMX */

#if defined(FILTERS_SUPPORT_SSE)
    if (type == FILTER_SSE && SDL_HasSSE ())
    {
        filters->type = FILTER_SSE;
        filters->shrink_X = pyg_filter_shrink_X_SSE;
        filters->shrink_Y = pyg_filter_shrink_Y_SSE;
        filters->expand_X = pyg_filter_expand_X_SSE;
        filters->expand_Y = pyg_filter_expand_Y_SSE;
    }
#endif /* FILTERS_SUPPORT_SSE */
    return filters->type;
}

/* this function implements an area-averaging shrinking filter in the
 * X-dimension */
void
pyg_filter_shrink_X_C (Uint8 *srcpix, Uint8 *dstpix, int height, int srcpitch,
    int dstpitch, int srcwidth, int dstwidth)
{
    int srcdiff = srcpitch - (srcwidth * 4);
    int dstdiff = dstpitch - (dstwidth * 4);
    int x, y;

    int xspace = 0x10000 * srcwidth / dstwidth; /* must be > 1 */
    int xrecip = (int) (0x100000000LL / xspace);
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
                int xfrac = 0x10000 - xcounter;
                /* write out a destination pixel */
                *dstpix++ = (Uint8) (((accumulate[0] +
                            ((srcpix[0] * xcounter) >> 16)) * xrecip) >> 16);
                *dstpix++ = (Uint8) (((accumulate[1] +
                            ((srcpix[1] * xcounter) >> 16)) * xrecip) >> 16);
                *dstpix++ = (Uint8) (((accumulate[2] +
                            ((srcpix[2] * xcounter) >> 16)) * xrecip) >> 16);
                *dstpix++ = (Uint8) (((accumulate[3] +
                            ((srcpix[3] * xcounter) >> 16)) * xrecip) >> 16);
                /* reload the accumulator with the remainder of this pixel */
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

/* this function implements an area-averaging shrinking filter in the
 * Y-dimension */
void
pyg_filter_shrink_Y_C (Uint8 *srcpix, Uint8 *dstpix, int width, int srcpitch,
    int dstpitch, int srcheight, int dstheight)
{
    Uint16 *templine;
    int srcdiff = srcpitch - (width * 4);
    int dstdiff = dstpitch - (width * 4);
    int x, y;
    int yspace = 0x10000 * srcheight / dstheight; /* must be > 1 */
    int yrecip = (int) (0x100000000LL / yspace);
    int ycounter = yspace;

    /* allocate and clear a memory area for storing the accumulator line */
    templine = (Uint16 *) malloc((size_t)(dstpitch * 2));
    if (templine == NULL)
        return;
    memset (templine, 0, (size_t) (dstpitch * 2));

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
            int yfrac = 0x10000 - ycounter;
            /* write out a destination line */
            for (x = 0; x < width; x++)
            {
                *dstpix++ = (Uint8) (((*accumulate++ +
                            ((*srcpix++ * ycounter) >> 16)) * yrecip) >> 16);
                *dstpix++ = (Uint8) (((*accumulate++ +
                            ((*srcpix++ * ycounter) >> 16)) * yrecip) >> 16);
                *dstpix++ = (Uint8) (((*accumulate++ +
                            ((*srcpix++ * ycounter) >> 16)) * yrecip) >> 16);
                *dstpix++ = (Uint8) (((*accumulate++ +
                            ((*srcpix++ * ycounter) >> 16)) * yrecip) >> 16);
            }
            dstpix += dstdiff;
            /* reload the accumulator with the remainder of this line */
            accumulate = templine;
            srcpix -= 4 * width;
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
    free (templine);
}

/* this function implements a bilinear filter in the X-dimension */
void
pyg_filter_expand_X_C (Uint8 *srcpix, Uint8 *dstpix, int height, int srcpitch,
    int dstpitch, int srcwidth, int dstwidth)
{
    int dstdiff = dstpitch - (dstwidth * 4);
    int *xidx0, *xmult0, *xmult1;
    int x, y;
    int factorwidth = 4;

    /* Allocate memory for factors */
    xidx0 = malloc((size_t) (dstwidth * 4));
    if (xidx0 == NULL)
        return;
    xmult0 = (int *) malloc((size_t) (dstwidth * factorwidth));
    xmult1 = (int *) malloc((size_t) (dstwidth * factorwidth));
    if (xmult0 == NULL || xmult1 == NULL)
    {
        free (xidx0);
        if (xmult0)
            free (xmult0);
        if (xmult1)
            free (xmult1);
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
    free (xidx0);
    free (xmult0);
    free (xmult1);
}

/* this function implements a bilinear filter in the Y-dimension */
void
pyg_filter_expand_Y_C (Uint8 *srcpix, Uint8 *dstpix, int width, int srcpitch,
    int dstpitch, int srcheight, int dstheight)
{
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
            *dstpix++ = (Uint8) (((*srcrow0++ * ymult0) +
                    (*srcrow1++ * ymult1)) >> 16);
            *dstpix++ = (Uint8) (((*srcrow0++ * ymult0) +
                    (*srcrow1++ * ymult1)) >> 16);
            *dstpix++ = (Uint8) (((*srcrow0++ * ymult0) +
                    (*srcrow1++ * ymult1)) >> 16);
            *dstpix++ = (Uint8) (((*srcrow0++ * ymult0) +
                    (*srcrow1++ * ymult1)) >> 16);
        }
    }
}
