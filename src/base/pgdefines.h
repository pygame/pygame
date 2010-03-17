/*
  pygame - Python Game Library
  Copyright (C) 2008 Marcus von Appen

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
#ifndef _PYGAME_DEFINES_H_
#define _PYGAME_DEFINES_H_

#include <math.h>
#include <float.h>
#include <limits.h>
#include "pgtypes.h"

#if defined(IS_WIN32) && defined(WIN32)
#define PRAGMA(x) __pragma(x)
#elif defined (__GNUC__)
#define PRAGMA(x) _Pragma(#x)
#else
#define PRAGMA(x)
#endif

#ifndef MIN
#define MIN(x,y) (((x) < (y)) ? (x) : (y))
#endif

#ifndef MAX
#define MAX(x,y) (((x) > (y)) ? (x) : (y))
#endif

#ifndef ABS
#define ABS(x) (((x) < 0) ? -(x) : (x))
#endif

#ifndef trunc
#define trunc(d) (((d) >= 0.0) ? (floor(d)) : (ceil(d)))
#endif

#ifndef round
#define round(d) (((d) > 0.0) ? (floor ((d) + .5)) : (ceil (d - .5)))
#endif

#ifndef CLAMP
#define CLAMP(x,low,high)                                               \
    (((x) < (high)) ? (((x) > (low)) ? (x) : (low)) :                   \
        (((high) > (low)) ? (high) : (low)))
#endif

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif
#ifndef DEG2RAD
#define DEG2RAD(x) ((x) * M_PI / 180.0f)
#endif
#ifndef RAD2DEG
#define RAD2DEG(x) (180.0f / M_PI * (x))
#endif

#define ADD_LIMIT(x,y,lower,upper)                                    \
    ((((y)) < 0) ?                                                      \
        (((x) < ((lower) - (y))) ? (lower) : ((x) + (y))) :             \
        (((x) > ((upper) - (y))) ? (upper) : ((x) + (y))))

#define SUB_LIMIT(x,y,lower,upper) (ADD_LIMIT(x,(-(y)),lower,upper))

#define INT_ADD_LIMIT(x,y) (ADD_LIMIT(x,y,INT_MIN,INT_MAX))
#define INT_SUB_LIMIT(x,y) (SUB_LIMIT(x,y,INT_MIN,INT_MAX))
#define INT16_ADD_LIMIT INT_ADD_LIMIT
#define INT16_SUB_LIMIT INT_SUB_LIMIT

#define UINT_ADD_LIMIT(x,y) (ADD_LIMIT(x,y,0,UINT_MAX))
#define UINT_SUB_LIMIT(x,y) (SUB_LIMIT(x,y,0,UINT_MAX))
#define UINT16_ADD_LIMIT UINT_ADD_LIMIT
#define UINT16_SUB_LIMIT UINT_SUB_LIMIT

#define LONG_ADD_LIMIT(x,y) (ADD_LIMIT(x,y,LONG_MIN,LONG_MAX))
#define LONG_SUB_LIMIT(x,y) (SUB_LIMIT(x,y,LONG_MIN,LONG_MAX))
#define INT32_ADD_LIMIT LONG_ADD_LIMIT
#define INT32_SUB_LIMIT LONG_SUB_LIMIT

#define ULONG_ADD_LIMIT(x,y) (ADD_LIMIT(x,y,0,ULONG_MAX))
#define ULONG_SUB_LIMIT(x,y) (SUB_LIMIT(x,y,0,ULONG_MAX))
#define UINT32_ADD_LIMIT ULONG_ADD_LIMIT
#define UINT32_SUB_LIMIT ULONG_SUB_LIMIT

#define DBL_ADD_LIMIT(x,y) (ADD_LIMIT(x,y,-DBL_MAX,DBL_MAX))
#define DBL_SUB_LIMIT(x,y) (SUB_LIMIT(x,y,-DBL_MAX,DBL_MAX))

/**
 * Using a ? : conditional leads to weird optimisations for some GCC
 * versions. signedness flags and such are removed, leading to wrong
 * results.
 */
#define INT_ADD_UINT_LIMIT(x,y,z)                                       \
    if ((x) >= 0)                                                       \
        z = MIN ((x) + (y), INT_MAX);                                   \
    else                                                                \
    {                                                                   \
        if ((y) <= INT_MAX || (y) < ((unsigned int)(INT_MAX + (x))))    \
            z = (x) + (int)(y);                                         \
        else                                                            \
            z = INT_MAX;                                                \
    }
#define INT16_ADD_UINT16_LIMIT INT_ADD_UINT_LIMIT

/**
 * Using a ? : conditional leads to weird optimisations for some GCC
 * versions. signedness flags and such are removed, leading to wrong
 * results.
 */
#define INT_SUB_UINT_LIMIT(x,y,z)                       \
    if ((x) < 0)                                        \
    {                                                   \
        if ((y) > ((unsigned int)(INT_MAX + (x))))      \
            z = INT_MIN;                                \
        else                                            \
            z = (x) - (y);                              \
    }                                                   \
    else                                                \
    {                                                   \
        if ((y) > ((pguint16)(INT_MAX - (x))))          \
            z = INT_MIN;                                \
        else                                            \
            z = (x) - (y);                              \
    }

#define INT16_SUB_UINT16_LIMIT INT_SUB_UINT_LIMIT

#endif /* _PYGAME_DEFINES_H_ */
