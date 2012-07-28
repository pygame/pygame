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
 * These are written for compilation with GCC only.
 *
 * This file should not depend on anything but the C standard library.
 */

#if !defined(__GNUC__) || !defined(__x86_64__)
#error "Pygame build bug: should not be compiling this file!"
#endif

#include <stdint.h>
typedef uint8_t Uint8;    /* SDL convension */
typedef uint16_t Uint16;  /* SDL convension */
#include <stdlib.h>
#include <memory.h>
#include "scale.h"

/* These functions implement an area-averaging shrinking filter in the X-dimension.
 */
void
filter_shrink_X_MMX(Uint8 *srcpix, Uint8 *dstpix, int height, int srcpitch, int dstpitch, int srcwidth, int dstwidth)
{
    int srcdiff = srcpitch - (srcwidth * 4);
    int dstdiff = dstpitch - (dstwidth * 4);

    int xspace = 0x04000 * srcwidth / dstwidth; /* must be > 1 */
    int xrecip = 0x40000000 / xspace;
    long long One64 = 0x4000400040004000ULL;
    long long srcdiff64 = srcdiff;
    long long dstdiff64 = dstdiff;
    asm __volatile__(" /* MMX code for X-shrink area average filter */ "
        " pxor          %%mm0,      %%mm0;           "
        " movd             %6,      %%mm7;           " /* mm7 == xrecipmmx */
        " punpcklwd     %%mm7,      %%mm7;           "
        " punpckldq     %%mm7,      %%mm7;           "
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
        " movq             %2,      %%mm3;           " /* mm3 = 2^14  */
        " punpcklwd     %%mm2,      %%mm2;           "
        " punpckldq     %%mm2,      %%mm2;           "
        " movd           (%0),      %%mm4;           " /* mm4 = srcpix */
        " add              $4,         %0;           "
        " punpcklbw     %%mm0,      %%mm4;           "
        " psubw         %%mm2,      %%mm3;           " /* mm3 = xfrac */
        " psllw            $2,      %%mm4;           "
        " movq          %%mm4,      %%mm5;           " /* mm2 = (srcpix * xcounter >> 16) */
        " psraw           $15,      %%mm5;           "
        " pand          %%mm2,      %%mm5;           "
        " movq          %%mm2,      %%mm6;           "
        " psraw           $15,      %%mm6;           "
        " pand          %%mm4,      %%mm6;           "
        " pmulhw        %%mm4,      %%mm2;           "
        " paddw         %%mm5,      %%mm2;           "
        " paddw         %%mm6,      %%mm2;           "
        " movq          %%mm4,      %%mm5;           " /* mm3 = (srcpix * xfrac) >> 16) */
        " psraw           $15,      %%mm5;           "
        " pand          %%mm3,      %%mm5;           "
        " movq          %%mm3,      %%mm6;           "
        " psraw           $15,      %%mm6;           "
        " pand          %%mm4,      %%mm6;           "
        " pmulhw        %%mm4,      %%mm3;           "
        " paddw         %%mm5,      %%mm3;           "
        " paddw         %%mm6,      %%mm3;           "
        " paddw         %%mm1,      %%mm2;           "
        " movq          %%mm3,      %%mm1;           " /* accumulator = (srcpix * xfrac) >> 16 */
        " movq          %%mm7,      %%mm5;           "
        " psraw           $15,      %%mm5;           "
        " pand          %%mm2,      %%mm5;           "
        " movq          %%mm2,      %%mm6;           "
        " psraw           $15,      %%mm6;           "
        " pand          %%mm7,      %%mm6;           "
        " pmulhw        %%mm7,      %%mm2;           "
        " paddw         %%mm5,      %%mm2;           "
        " paddw         %%mm6,      %%mm2;           "
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
}

void
filter_shrink_X_SSE(Uint8 *srcpix, Uint8 *dstpix, int height, int srcpitch, int dstpitch, int srcwidth, int dstwidth)
{
    int srcdiff = srcpitch - (srcwidth * 4);
    int dstdiff = dstpitch - (dstwidth * 4);

    int xspace = 0x04000 * srcwidth / dstwidth; /* must be > 1 */
    int xrecip = 0x40000000 / xspace;
    long long One64 = 0x4000400040004000ULL;
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
}

/* These functions implement an area-averaging shrinking filter in the Y-dimension.
 */
void
filter_shrink_Y_MMX(Uint8 *srcpix, Uint8 *dstpix, int width, int srcpitch, int dstpitch, int srcheight, int dstheight)
{
    Uint16 *templine;
    int srcdiff = srcpitch - (width * 4);
    int dstdiff = dstpitch - (width * 4);
    int yspace = 0x4000 * srcheight / dstheight; /* must be > 1 */
    int yrecip = 0x40000000 / yspace;
    long long One64 = 0x4000400040004000ULL;

    /* allocate and clear a memory area for storing the accumulator line */
    templine = (Uint16 *) malloc(dstpitch * 2);
    if (templine == 0) return;
    memset(templine, 0, dstpitch * 2);
    long long srcdiff64 = srcdiff;
    long long dstdiff64 = dstdiff;
    asm __volatile__(" /* MMX code for Y-shrink area average filter */ "
        " movl             %5,      %%ecx;           " /* ecx == ycounter */
        " pxor          %%mm0,      %%mm0;           "
        " movd             %6,      %%mm7;           " /* mm7 == yrecipmmx */
        " punpcklwd     %%mm7,      %%mm7;           "
        " punpckldq     %%mm7,      %%mm7;           "
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
        " punpcklwd     %%mm1,      %%mm1;           "
        " punpckldq     %%mm1,      %%mm1;           "
        " psubw         %%mm1,      %%mm6;           " /* mm6 = yfrac */
        "4:                                          "
        " movd           (%0),      %%mm4;           " /* mm4 = srcpix */
        " add              $4,         %0;           "
        " punpcklbw     %%mm0,      %%mm4;           "
        " movq        (%%rax),      %%mm5;           " /* mm5 = accumulate */
        " movq          %%mm6,      %%mm3;           "
        " psllw            $2,      %%mm4;           "
        " movq          %%mm4,      %%mm0;           " /* mm3 = (srcpix * yfrac) >> 16) */
        " psraw           $15,      %%mm0;           "
        " pand          %%mm3,      %%mm0;           "
        " movq          %%mm3,      %%mm2;           "
        " psraw           $15,      %%mm2;           "
        " pand          %%mm4,      %%mm2;           "
        " pmulhw        %%mm4,      %%mm3;           "
        " paddw         %%mm0,      %%mm3;           "
        " paddw         %%mm2,      %%mm3;           "
        " movq          %%mm1,      %%mm0;           " /* mm4 = (srcpix * ycounter >> 16) */
        " psraw           $15,      %%mm0;           "
        " pand          %%mm4,      %%mm0;           "
        " movq          %%mm4,      %%mm2;           "
        " psraw           $15,      %%mm2;           "
        " pand          %%mm1,      %%mm2;           "
        " pmulhw        %%mm1,      %%mm4;           "
        " paddw         %%mm0,      %%mm4;           "
        " paddw         %%mm2,      %%mm4;           "
        " movq          %%mm3,    (%%rax);           "
        " paddw         %%mm5,      %%mm4;           "
        " add              $8,      %%rax;           "
        " movq          %%mm7,      %%mm0;           "
        " psraw           $15,      %%mm0;           "
        " pand          %%mm4,      %%mm0;           "
        " movq          %%mm4,      %%mm2;           "
        " psraw           $15,      %%mm2;           "
        " pand          %%mm7,      %%mm2;           "
        " pmulhw        %%mm7,      %%mm4;           "
        " paddw         %%mm0,      %%mm4;           "
        " paddw         %%mm2,      %%mm4;           "
        " pxor          %%mm0,      %%mm0;           "
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

    /* free the temporary memory */
    free(templine);
}

void
filter_shrink_Y_SSE(Uint8 *srcpix, Uint8 *dstpix, int width, int srcpitch, int dstpitch, int srcheight, int dstheight)
{
    Uint16 *templine;
    int srcdiff = srcpitch - (width * 4);
    int dstdiff = dstpitch - (width * 4);
    int yspace = 0x4000 * srcheight / dstheight; /* must be > 1 */
    int yrecip = 0x40000000 / yspace;
    long long One64 = 0x4000400040004000ULL;

    /* allocate and clear a memory area for storing the accumulator line */
    templine = (Uint16 *) malloc(dstpitch * 2);
    if (templine == 0) return;
    memset(templine, 0, dstpitch * 2);
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

    /* free the temporary memory */
    free(templine);
}

/* These functions implement a bilinear filter in the X-dimension.
 */
void
filter_expand_X_MMX(Uint8 *srcpix, Uint8 *dstpix, int height, int srcpitch, int dstpitch, int srcwidth, int dstwidth)
{
    int *xidx0, *xmult0, *xmult1;
    int x, y;
    int factorwidth = 8;

    /* Allocate memory for factors */
    xidx0 = malloc(dstwidth * 4);
    if (xidx0 == 0) return;
    xmult0 = (int *) malloc(dstwidth * factorwidth);
    xmult1 = (int *) malloc(dstwidth * factorwidth);
    if (xmult0 == 0 || xmult1 == 0)
    {
        free(xidx0);
        if (xmult0) free(xmult0);
        if (xmult1) free(xmult1);
    }

    /* Create multiplier factors and starting indices and put them in arrays */
    for (x = 0; x < dstwidth; x++)
    {
        int xm1 = 0x100 * ((x * (srcwidth - 1)) % dstwidth) / dstwidth;
        int xm0 = 0x100 - xm1;
        xidx0[x] = x * (srcwidth - 1) / dstwidth;
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
        asm __volatile__( " /* MMX code for inner loop of X bilinear filter */ "
             " movl             %5,      %%ecx;           "
             " pxor          %%mm0,      %%mm0;           "
             "1:                                          "
             " movslq         (%3),      %%rax;           " /* get xidx0[x] */
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
    }

    /* free memory */
    free(xidx0);
    free(xmult0);
    free(xmult1);
}

void
filter_expand_X_SSE(Uint8 *srcpix, Uint8 *dstpix, int height, int srcpitch, int dstpitch, int srcwidth, int dstwidth)
{
    int *xidx0, *xmult0, *xmult1;
    int x, y;
    int factorwidth = 8;

    /* Allocate memory for factors */
    xidx0 = malloc(dstwidth * 4);
    if (xidx0 == 0) return;
    xmult0 = (int *) malloc(dstwidth * factorwidth);
    xmult1 = (int *) malloc(dstwidth * factorwidth);
    if (xmult0 == 0 || xmult1 == 0)
    {
        free(xidx0);
        if (xmult0) free(xmult0);
        if (xmult1) free(xmult1);
    }

    /* Create multiplier factors and starting indices and put them in arrays */
    for (x = 0; x < dstwidth; x++)
    {
        int xm1 = 0x100 * ((x * (srcwidth - 1)) % dstwidth) / dstwidth;
        int xm0 = 0x100 - xm1;
        xidx0[x] = x * (srcwidth - 1) / dstwidth;
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
        asm __volatile__( " /* MMX code for inner loop of X bilinear filter */ "
             " movl             %5,      %%ecx;           "
             " pxor          %%mm0,      %%mm0;           "
             "1:                                          "
             " movslq         (%3),      %%rax;           " /* get xidx0[x] */
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
    }

    /* free memory */
    free(xidx0);
    free(xmult0);
    free(xmult1);
}

/* These functions implement a bilinear filter in the Y-dimension
 */
void
filter_expand_Y_MMX(Uint8 *srcpix, Uint8 *dstpix, int width, int srcpitch, int dstpitch, int srcheight, int dstheight)
{
    int y;

    for (y = 0; y < dstheight; y++)
    {
        int yidx0 = y * (srcheight - 1) / dstheight;
        Uint8 *srcrow0 = srcpix + yidx0 * srcpitch;
        Uint8 *srcrow1 = srcrow0 + srcpitch;
        int ymult1 = 0x0100 * ((y * (srcheight - 1)) % dstheight) / dstheight;
        int ymult0 = 0x0100 - ymult1;
        Uint8 *dstrow = dstpix + y * dstpitch;
        asm __volatile__( " /* MMX code for inner loop of Y bilinear filter */ "
             " movl          %5,      %%ecx;                      "
             " movd          %3,      %%mm1;                      "
             " movd          %4,      %%mm2;                      "
             " pxor       %%mm0,      %%mm0;                      "
             " punpcklwd  %%mm1,      %%mm1;                      "
             " punpckldq  %%mm1,      %%mm1;                      "
             " punpcklwd  %%mm2,      %%mm2;                      "
             " punpckldq  %%mm2,      %%mm2;                      "
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
    }
}

void
filter_expand_Y_SSE(Uint8 *srcpix, Uint8 *dstpix, int width, int srcpitch, int dstpitch, int srcheight, int dstheight)
{
    int y;

    for (y = 0; y < dstheight; y++)
    {
        int yidx0 = y * (srcheight - 1) / dstheight;
        Uint8 *srcrow0 = srcpix + yidx0 * srcpitch;
        Uint8 *srcrow1 = srcrow0 + srcpitch;
        int ymult1 = 0x0100 * ((y * (srcheight - 1)) % dstheight) / dstheight;
        int ymult0 = 0x0100 - ymult1;
        Uint8 *dstrow = dstpix + y * dstpitch;
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
    }
}

