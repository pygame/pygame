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

/* Pentium MMX/SSE smoothscale routines
 * Available on Win32 or GCC on a Pentium.
 * Sorry, no Win64 support yet for Visual C builds, but it can be added.
 */

#if !defined(SCALE_HEADER)
#define SCALE_HEADER

#if (defined(__GNUC__) && ((defined(__x86_64__) && !defined(_NO_MMX_FOR_X86_64)) || defined(__i386__))) || (defined(MS_WIN32) && !(defined(_M_X64) && defined(_NO_MMX_FOR_X86_64)))
#define SCALE_MMX_SUPPORT

/* These functions implement an area-averaging shrinking filter in the X-dimension.
 */
void filter_shrink_X_MMX(Uint8 *srcpix, Uint8 *dstpix, int height, int srcpitch, int dstpitch, int srcwidth, int dstwidth);

void filter_shrink_X_SSE(Uint8 *srcpix, Uint8 *dstpix, int height, int srcpitch, int dstpitch, int srcwidth, int dstwidth);

/* These functions implement an area-averaging shrinking filter in the Y-dimension.
 */
void filter_shrink_Y_MMX(Uint8 *srcpix, Uint8 *dstpix, int width, int srcpitch, int dstpitch, int srcheight, int dstheight);

void filter_shrink_Y_SSE(Uint8 *srcpix, Uint8 *dstpix, int width, int srcpitch, int dstpitch, int srcheight, int dstheight);

/* These functions implement a bilinear filter in the X-dimension.
 */
void filter_expand_X_MMX(Uint8 *srcpix, Uint8 *dstpix, int height, int srcpitch, int dstpitch, int srcwidth, int dstwidth);

void filter_expand_X_SSE(Uint8 *srcpix, Uint8 *dstpix, int height, int srcpitch, int dstpitch, int srcwidth, int dstwidth);

/* These functions implement a bilinear filter in the Y-dimension.
 */
void filter_expand_Y_MMX(Uint8 *srcpix, Uint8 *dstpix, int width, int srcpitch, int dstpitch, int srcheight, int dstheight);

void filter_expand_Y_SSE(Uint8 *srcpix, Uint8 *dstpix, int width, int srcpitch, int dstpitch, int srcheight, int dstheight);

#if defined(_M_X64)

void filter_shrink_Y_MMX_gcc(Uint8 *srcpix, Uint8 *dstpix, Uint16 *templine, int width, int srcpitch, int dstpitch, int srcheight, int dstheight);

void filter_shrink_Y_SSE_gcc(Uint8 *srcpix, Uint8 *dstpix, Uint16 *templine, int width, int srcpitch, int dstpitch, int srcheight, int dstheight);

void filter_expand_X_MMX_gcc(Uint8 *srcpix, Uint8 *dstpix, int *xidx0, int *xmult0, int *xmult1, int height, int srcpitch, int dstpitch, int srcwidth, int dstwidth);

void filter_expand_X_SSE_gcc(Uint8 *srcpix, Uint8 *dstpix, int *xidx0, int *xmult0, int *xmult1, int height, int srcpitch, int dstpitch, int srcwidth, int dstwidth);

#endif /* #if defined(_M_X64) */

#endif /* #if (defined(__GNUC__) && .....) */

#endif /* #if !defined(SCALE_HEADER) */
