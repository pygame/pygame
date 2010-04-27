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

CREATE_BLITTER(blend_rgb_add, D_BLEND_RGB_ADD(tmp,sR,sG,sB,dR,dG,dB),,,)
CREATE_BLITTER(blend_rgb_sub, D_BLEND_RGB_SUB(tmp2,sR,sG,sB,dR,dG,dB),,,)
CREATE_BLITTER(blend_rgb_mul, D_BLEND_RGB_MULT(sR,sG,sB,dR,dG,dB),,,)
CREATE_BLITTER(blend_rgb_min, D_BLEND_RGB_MIN(sR,sG,sB,dR,dG,dB),,,)
CREATE_BLITTER(blend_rgb_max, D_BLEND_RGB_MAX(sR,sG,sB,dR,dG,dB),,,)
CREATE_BLITTER(blend_rgb_xor, D_BLEND_RGB_XOR(sR,sG,sB,dR,dG,dB),,,)
CREATE_BLITTER(blend_rgb_and, D_BLEND_RGB_AND(sR,sG,sB,dR,dG,dB),,,)
CREATE_BLITTER(blend_rgb_or, D_BLEND_RGB_OR(sR,sG,sB,dR,dG,dB),,,)
CREATE_BLITTER(blend_rgb_diff, D_BLEND_RGB_DIFF(sR,sG,sB,dR,dG,dB),,,)
CREATE_BLITTER(blend_rgb_screen, D_BLEND_RGB_SCREEN(sR,sG,sB,dR,dG,dB),,,)
CREATE_BLITTER(blend_rgb_avg, D_BLEND_RGB_AVG(sR,sG,sB,dR,dG,dB),,,)
