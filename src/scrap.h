/*
    pygame - Python Game Library
    Copyright (C) 2006 Rene Dudfield

    Originally put in the public domain by Sam Lantinga.

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

/* Handle clipboard text and data in arbitrary formats */

/* Miscellaneous defines */
#define T(A, B, C, D)	(int)((A<<24)|(B<<16)|(C<<8)|(D<<0))

extern int init_scrap(void);
extern int lost_scrap(void);
extern void put_scrap(int type, int srclen, char *src);
extern void get_scrap(int type, int *dstlen, char **dst);
