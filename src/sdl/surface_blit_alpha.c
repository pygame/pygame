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

CREATE_BLITTER(alpha_alpha, ALPHA_BLEND(sR,sG,sB,sA,dR,dG,dB,dA),,,)
CREATE_BLITTER(alpha_solid, ALPHA_BLEND(sR,sG,sB,alpha,dR,dG,dB,dA),,,)
#ifdef HAVE_OPENMP
/* src values are reachable via sppx in OpenMP macro */
CREATE_BLITTER(alpha_colorkey, ALPHA_BLEND(sR,sG,sB,sA,dR,dG,dB,dA),
    (sA = (*sppx == colorkey) ? 0 : alpha),
    (sA = (pixel == colorkey) ? 0 : alpha),
    (sA = ((*(Uint32*)sppx) == colorkey) ? 0 : alpha))
#else
CREATE_BLITTER(alpha_colorkey, ALPHA_BLEND(sR,sG,sB,sA,dR,dG,dB,dA),
    (sA = (*src == colorkey) ? 0 : alpha),
    (sA = (pixel == colorkey) ? 0 : alpha),
    (sA = ((*(Uint32*)src) == colorkey) ? 0 : alpha))
#endif
