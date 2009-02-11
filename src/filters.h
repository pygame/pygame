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
#ifndef _PYGAME_FILTERS_H_
#define _PYGAME_FILTERS_H_

#include <SDL.h>

void
pyg_filter_shrink_X_C (Uint8 *srcpix, Uint8 *dstpix, int height, int srcpitch,
    int dstpitch, int srcwidth, int dstwidth);

void
pyg_filter_shrink_X_MMX (Uint8 *srcpix, Uint8 *dstpix, int height,
    int srcpitch, int dstpitch, int srcwidth, int dstwidth);

void
pyg_filter_shrink_Y_C (Uint8 *srcpix, Uint8 *dstpix, int width, int srcpitch,
    int dstpitch, int srcheight, int dstheight);

void
pyg_filter_shrink_Y_MMX (Uint8 *srcpix, Uint8 *dstpix, int width, int srcpitch,
    int dstpitch, int srcheight, int dstheight);

void
pyg_filter_expand_X_C (Uint8 *srcpix, Uint8 *dstpix, int height, int srcpitch,
    int dstpitch, int srcwidth, int dstwidth);

void
pyg_filter_expand_X_MMX (Uint8 *srcpix, Uint8 *dstpix, int height,
    int srcpitch, int dstpitch, int srcwidth, int dstwidth);

void
pyg_filter_expand_Y_C (Uint8 *srcpix, Uint8 *dstpix, int width, int srcpitch,
    int dstpitch, int srcheight, int dstheight);

void
pyg_filter_expand_Y_MMX (Uint8 *srcpix, Uint8 *dstpix, int width, int srcpitch,
    int dstpitch, int srcheight, int dstheight);

#endif /* _PYGAME_FILTERS_H_ */
