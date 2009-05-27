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
#ifndef _PYGAME_TYPES_H_
#define _PYGAME_TYPES_H_

/**
 * Guarantee at least 8 bit usage.
 */
typedef unsigned char pgbyte;

#define PGBYTE_MASKED(x) ((x) & 0xFF)

/**
 * Guarantee at least 16 bit usage.
 */
typedef unsigned int pguint16;
typedef int pgint16;

#define PGINT16_MASKED(x) ((x) & 0xFFFF)
#define PGUINT16_MASKED(x) PGINT16_MASKED(x)

/**
 * Guarantee at least 32 bit usage.
 */
typedef unsigned long int pguint32;
typedef long int pgint32;

#define PGINT32_MASKED(x) ((x) & 0xFFFFFFFF)
#define PGUINT32_MASKED(x) PGINT32_MASKED(x)

/**
 * Simple rectangle structure similiar to PyRect - but usable within pure C
 * code for released GIL purposes.
 */
typedef struct
{
    pgint16 x;
    pgint16 y;
    pguint16 w;
    pguint16 h;
} CRect;

#endif /* _PYGAME_TYPES_H_ */
