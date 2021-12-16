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

/* Pentium 64 bit SSE/MMX smoothscale routines
 * These are written for compilation with MSVC only.
 *
 * This file should not depend on anything but the C standard library.
 */

#include <stdint.h>
typedef uint8_t Uint8;   /* SDL convention */
typedef uint16_t Uint16; /* SDL convention */
#include <stdlib.h>
#include <memory.h>
#include "scale.h"

/* These functions implement an area-averaging shrinking filter in the
 * Y-dimension.
 */
void
filter_shrink_Y_MMX(Uint8 *srcpix, Uint8 *dstpix, int width, int srcpitch,
                    int dstpitch, int srcheight, int dstheight)
{
    Uint16 *templine;

    /* allocate and clear a memory area for storing the accumulator line */
    templine = (Uint16 *)malloc(dstpitch * 2);
    if (templine == 0)
        return;
    memset(templine, 0, dstpitch * 2);

    filter_shrink_Y_MMX_gcc(srcpix, dstpix, templine, width, srcpitch,
                            dstpitch, srcheight, dstheight);

    /* free the temporary memory */
    free(templine);
}

void
filter_shrink_Y_SSE(Uint8 *srcpix, Uint8 *dstpix, int width, int srcpitch,
                    int dstpitch, int srcheight, int dstheight)
{
    Uint16 *templine;

    /* allocate and clear a memory area for storing the accumulator line */
    templine = (Uint16 *)malloc(dstpitch * 2);
    if (templine == 0)
        return;
    memset(templine, 0, dstpitch * 2);

    filter_shrink_Y_SSE_gcc(srcpix, dstpix, templine, width, srcpitch,
                            dstpitch, srcheight, dstheight);

    /* free the temporary memory */
    free(templine);
}

/* These functions implement a bilinear filter in the X-dimension.
 */
void
filter_expand_X_MMX(Uint8 *srcpix, Uint8 *dstpix, int height, int srcpitch,
                    int dstpitch, int srcwidth, int dstwidth)
{
    int *xidx0, *xmult0, *xmult1;
    int factorwidth = 8;

    /* Allocate memory for factors */
    xidx0 = malloc(dstwidth * 4);
    if (xidx0 == 0)
        return;
    xmult0 = (int *)malloc(dstwidth * factorwidth);
    xmult1 = (int *)malloc(dstwidth * factorwidth);
    if (xmult0 == 0 || xmult1 == 0) {
        free(xidx0);
        if (xmult0)
            free(xmult0);
        if (xmult1)
            free(xmult1);
        return;
    }

    filter_expand_X_MMX_gcc(srcpix, dstpix, xidx0, xmult0, xmult1, height,
                            srcpitch, dstpitch, srcwidth, dstwidth);

    /* free memory */
    free(xidx0);
    free(xmult0);
    free(xmult1);
}

void
filter_expand_X_SSE(Uint8 *srcpix, Uint8 *dstpix, int height, int srcpitch,
                    int dstpitch, int srcwidth, int dstwidth)
{
    int *xidx0, *xmult0, *xmult1;
    int factorwidth = 8;

    /* Allocate memory for factors */
    xidx0 = malloc(dstwidth * 4);
    if (xidx0 == 0)
        return;
    xmult0 = (int *)malloc(dstwidth * factorwidth);
    xmult1 = (int *)malloc(dstwidth * factorwidth);
    if (xmult0 == 0 || xmult1 == 0) {
        free(xidx0);
        if (xmult0)
            free(xmult0);
        if (xmult1)
            free(xmult1);
        return;
    }

    filter_expand_X_SSE_gcc(srcpix, dstpix, xidx0, xmult0, xmult1, height,
                            srcpitch, dstpitch, srcwidth, dstwidth);

    /* free memory */
    free(xidx0);
    free(xmult0);
    free(xmult1);
}
