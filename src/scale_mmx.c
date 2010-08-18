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
 * These are only compiled with GCC.
 */
#if defined(__GNUC__)
/* Choose between the 32 bit and 64 bit versions.
 * Including source code like this may be frowned upon by some,
 * but the alternative is ungainly conditionally compiled code.
 */
#   if defined(__x86_64__)
#       include "scale_mmx64.c"
#   elif defined(__i386__)
#       include "scale_mmx32.c"
#   endif
#endif
