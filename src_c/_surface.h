/*
  pygame - Python Game Library
  Copyright (C) 2000-2001  Pete Shinners
  Copyright (C) 2007 Marcus von Appen

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

#ifndef _SURFACE_H
#define _SURFACE_H

#include "_pygame.h"
#include "surface.h"

// Some simd compat stuff going here for now.
#if PG_SDL3
// SDL3 no longer includes intrinsics by default, we need to do it explicitly
#include <SDL3/SDL_intrin.h>

/* If SDL_AVX2_INTRINSICS is defined by SDL3, we need to set macros that our
 * code checks for avx2 build time support */
#ifdef SDL_AVX2_INTRINSICS
#ifndef HAVE_IMMINTRIN_H
#define HAVE_IMMINTRIN_H 1
#endif /* HAVE_IMMINTRIN_H*/
#ifndef __AVX2__
#define __AVX2__
#endif /* __AVX2__*/
#endif /* SDL_AVX2_INTRINSICS*/

// TODO reenable this to test best
#ifdef SDL_SSE2_INTRINSICS
#ifndef __SSE2__
#define __SSE2__
#endif /* __SSE2__*/
#endif /* SDL_SSE2_INTRINSICS*/
#endif /* PG_SDL3 */

#endif
