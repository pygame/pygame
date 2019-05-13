/*
    Bitmask Collision Detection Library 1.5
    Copyright (C) 2002-2005 Ulf Ekstrom except for the bitcount function which
    is copyright (C) Donald W. Gillies, 1992, and the other bitcount function
    which was taken from Jorg Arndt's excellent "Algorithms for Programmers"
    text.

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

#include "include/bitmask.h"
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#ifndef INLINE
#warning No INLINE definition in bitmask.h, performance may suffer.
#endif

#define MIN(a, b) ((a) <= (b) ? (a) : (b))
#define MAX(a, b) ((a) >= (b) ? (a) : (b))

/* The code by Gillies is slightly (1-3%) faster than the more
   readable code below */
#define GILLIES

static INLINE unsigned int
bitcount(BITMASK_W n)
{
    if (BITMASK_W_LEN == 32) {
#ifdef GILLIES
        /* (C) Donald W. Gillies, 1992.  All rights reserved.  You may reuse
           this bitcount() function anywhere you please as long as you retain
           this Copyright Notice. */
        register unsigned long tmp;
        return (tmp = (n) - (((n) >> 1) & 033333333333) -
                      (((n) >> 2) & 011111111111),
                tmp = ((tmp + (tmp >> 3)) & 030707070707),
                tmp = (tmp + (tmp >> 6)),
                tmp = (tmp + (tmp >> 12) + (tmp >> 24)) & 077);
/* End of Donald W. Gillies bitcount code */
#else
        /* This piece taken from Jorg Arndt's "Algorithms for Programmers" */
        n = ((n >> 1) & 0x55555555) + (n & 0x55555555);  // 0-2 in 2 bits
        n = ((n >> 2) & 0x33333333) + (n & 0x33333333);  // 0-4 in 4 bits
        n = ((n >> 4) + n) & 0x0f0f0f0f;                 // 0-8 in 4 bits
        n += n >> 8;                                     // 0-16 in 8 bits
        n += n >> 16;                                    // 0-32 in 8 bits
        return n & 0xff;
#endif
    }
    else if (BITMASK_W_LEN == 64) {
        n = ((n >> 1) & 0x5555555555555555) + (n & 0x5555555555555555);
        n = ((n >> 2) & 0x3333333333333333) + (n & 0x3333333333333333);
        n = ((n >> 4) + n) & 0x0f0f0f0f0f0f0f0f;
        n += n >> 8;
        n += n >> 16;
        n += n >> 32;
        return n & 0xff;
    }
    else {
        /* Handle non-32 or 64 bit case the slow way */
        unsigned int nbits = 0;
        while (n) {
            if (n & 1)
                nbits++;
            n = n >> 1;
        }
        return nbits;
    }
}

bitmask_t *
bitmask_create(int w, int h)
{
    bitmask_t *temp;
    size_t size;

    /* Guard against negative parameters. */
    if (w < 0 || h < 0) {
        return 0;
    }

    size = offsetof(bitmask_t, bits);

    if (w && h) {
        size += h * ((w - 1) / BITMASK_W_LEN + 1) * sizeof(BITMASK_W);
    }

    temp = malloc(size);

    if (!temp) {
        return 0;
    }

    temp->w = w;
    temp->h = h;
    bitmask_clear(temp);

    return temp;
}

void
bitmask_free(bitmask_t *m)
{
    free(m);
}

void
bitmask_clear(bitmask_t *m)
{
    if (!m->h || !m->w)
        return;

    memset(m->bits, 0,
           m->h * ((m->w - 1) / BITMASK_W_LEN + 1) * sizeof(BITMASK_W));
}

void
bitmask_fill(bitmask_t *m)
{
    int len, shift;
    BITMASK_W *pixels, cmask, full;

    if (!m->h || !m->w) {
        return;
    }

    len = m->h * ((m->w - 1) / BITMASK_W_LEN);

    shift = BITMASK_W_LEN - (m->w % BITMASK_W_LEN);
    full = ~(BITMASK_W)0;
    cmask = (~(BITMASK_W)0) >> shift;
    /* fill all the pixels that aren't in the rightmost BITMASK_Ws */
    for (pixels = m->bits; pixels < (m->bits + len); pixels++) {
        *pixels = full;
    }
    /* for the rightmost BITMASK_Ws, use cmask to ensure we aren't setting
       bits that are outside of the mask */
    for (pixels = m->bits + len; pixels < (m->bits + len + m->h); pixels++) {
        *pixels = cmask;
    }
}

void
bitmask_invert(bitmask_t *m)
{
    int len, shift;
    BITMASK_W *pixels, cmask;

    if (!m->h || !m->w) {
        return;
    }

    len = m->h * ((m->w - 1) / BITMASK_W_LEN);

    shift = BITMASK_W_LEN - (m->w % BITMASK_W_LEN);
    cmask = (~(BITMASK_W)0) >> shift;
    /* flip all the pixels that aren't in the rightmost BITMASK_Ws */
    for (pixels = m->bits; pixels < (m->bits + len); pixels++) {
        *pixels = ~(*pixels);
    }
    /* for the rightmost BITMASK_Ws, & with cmask to ensure we aren't setting
       bits that are outside of the mask */
    for (pixels = m->bits + len; pixels < (m->bits + len + m->h); pixels++) {
        *pixels = cmask & ~(*pixels);
    }
}

unsigned int
bitmask_count(bitmask_t *m)
{
    BITMASK_W *pixels;
    unsigned int tot = 0;

    if (!m->w || !m->h) {
        return tot;
    }

    for (pixels = m->bits;
         pixels < (m->bits + m->h * ((m->w - 1) / BITMASK_W_LEN + 1));
         pixels++) {
        tot += bitcount(*pixels);
    }

    return tot;
}

int
bitmask_overlap(const bitmask_t *a, const bitmask_t *b, int xoffset,
                int yoffset)
{
    const BITMASK_W *a_entry, *a_end;
    const BITMASK_W *b_entry;
    const BITMASK_W *ap, *app, *bp;
    unsigned int shift, rshift, i, astripes, bstripes;

    /* Return if no overlap or one mask has a width/height of 0. */
    if ((xoffset >= a->w) || (yoffset >= a->h) || (yoffset <= -b->h) ||
        (xoffset <= -b->w) || (!a->h) || (!a->w) || (!b->h) || (!b->w)) {
        return 0;
    }

    if (xoffset >= 0) {
    swapentry:
        if (yoffset >= 0) {
            a_entry = a->bits +
                      a->h * ((unsigned int)xoffset / BITMASK_W_LEN) + yoffset;
            a_end = a_entry + MIN(b->h, a->h - yoffset);
            b_entry = b->bits;
        }
        else {
            a_entry = a->bits + a->h * ((unsigned int)xoffset / BITMASK_W_LEN);
            a_end = a_entry + MIN(b->h + yoffset, a->h);
            b_entry = b->bits - yoffset;
        }
        shift = xoffset & BITMASK_W_MASK;
        if (shift) {
            rshift = BITMASK_W_LEN - shift;
            astripes = ((unsigned int)(a->w - 1)) / BITMASK_W_LEN -
                       (unsigned int)xoffset / BITMASK_W_LEN;
            bstripes = ((unsigned int)(b->w - 1)) / BITMASK_W_LEN + 1;
            if (bstripes > astripes) /* zig-zag .. zig*/
            {
                for (i = 0; i < astripes; i++) {
                    for (ap = a_entry, app = ap + a->h, bp = b_entry;
                         ap < a_end;)
                        if ((*ap++ >> shift) & *bp ||
                            (*app++ << rshift) & *bp++)
                            return 1;
                    a_entry += a->h;
                    a_end += a->h;
                    b_entry += b->h;
                }
                for (ap = a_entry, bp = b_entry; ap < a_end;)
                    if ((*ap++ >> shift) & *bp++)
                        return 1;
                return 0;
            }
            else /* zig-zag */
            {
                for (i = 0; i < bstripes; i++) {
                    for (ap = a_entry, app = ap + a->h, bp = b_entry;
                         ap < a_end;)
                        if ((*ap++ >> shift) & *bp ||
                            (*app++ << rshift) & *bp++)
                            return 1;
                    a_entry += a->h;
                    a_end += a->h;
                    b_entry += b->h;
                }
                return 0;
            }
        }
        else /* xoffset is a multiple of the stripe width, and the above
                routines wont work */
        {
            astripes = (MIN(b->w, a->w - xoffset) - 1) / BITMASK_W_LEN + 1;
            for (i = 0; i < astripes; i++) {
                for (ap = a_entry, bp = b_entry; ap < a_end;)
                    if (*ap++ & *bp++)
                        return 1;
                a_entry += a->h;
                a_end += a->h;
                b_entry += b->h;
            }
            return 0;
        }
    }
    else {
        const bitmask_t *c = a;
        a = b;
        b = c;
        xoffset *= -1;
        yoffset *= -1;
        goto swapentry;
    }
}

/* Will hang if there are no bits set in w! */
static INLINE int
firstsetbit(BITMASK_W w)
{
    int i = 0;
    while ((w & 1) == 0) {
        i++;
        w /= 2;
    }
    return i;
}

/* x and y are given in the coordinates of mask a, and are untouched if there
 * is no overlap */
int
bitmask_overlap_pos(const bitmask_t *a, const bitmask_t *b, int xoffset,
                    int yoffset, int *x, int *y)
{
    const BITMASK_W *a_entry, *a_end, *b_entry, *ap, *bp;
    unsigned int shift, rshift, i, astripes, bstripes, xbase;

    /* Return if no overlap or one mask has a width/height of 0. */
    if ((xoffset >= a->w) || (yoffset >= a->h) || (yoffset <= -b->h) ||
        (xoffset <= -b->w) || (!a->h) || (!a->w) || (!b->h) || (!b->w)) {
        return 0;
    }

    if (xoffset >= 0) {
        xbase = xoffset / BITMASK_W_LEN; /* first stripe from mask a */
        if (yoffset >= 0) {
            a_entry = a->bits + a->h * xbase + yoffset;
            a_end = a_entry + MIN(b->h, a->h - yoffset);
            b_entry = b->bits;
        }
        else {
            a_entry = a->bits + a->h * xbase;
            a_end = a_entry + MIN(b->h + yoffset, a->h);
            b_entry = b->bits - yoffset;
            yoffset = 0; /* relied on below */
        }
        shift = xoffset & BITMASK_W_MASK;
        if (shift) {
            rshift = BITMASK_W_LEN - shift;
            astripes = (a->w - 1) / BITMASK_W_LEN - xoffset / BITMASK_W_LEN;
            bstripes = (b->w - 1) / BITMASK_W_LEN + 1;
            if (bstripes > astripes) /* zig-zag .. zig*/
            {
                for (i = 0; i < astripes; i++) {
                    for (ap = a_entry, bp = b_entry; ap < a_end; ap++, bp++)
                        if (*ap & (*bp << shift)) {
                            *y = ap - a_entry + yoffset;
                            *x = (xbase + i) * BITMASK_W_LEN +
                                 firstsetbit(*ap & (*bp << shift));
                            return 1;
                        }
                    a_entry += a->h;
                    a_end += a->h;
                    for (ap = a_entry, bp = b_entry; ap < a_end; ap++, bp++)
                        if (*ap & (*bp >> rshift)) {
                            *y = ap - a_entry + yoffset;
                            *x = (xbase + i + 1) * BITMASK_W_LEN +
                                 firstsetbit(*ap & (*bp >> rshift));
                            return 1;
                        }
                    b_entry += b->h;
                }
                for (ap = a_entry, bp = b_entry; ap < a_end; ap++, bp++)
                    if (*ap & (*bp << shift)) {
                        *y = ap - a_entry + yoffset;
                        *x = (xbase + astripes) * BITMASK_W_LEN +
                             firstsetbit(*ap & (*bp << shift));
                        return 1;
                    }
                return 0;
            }
            else /* zig-zag */
            {
                for (i = 0; i < bstripes; i++) {
                    for (ap = a_entry, bp = b_entry; ap < a_end; ap++, bp++)
                        if (*ap & (*bp << shift)) {
                            *y = ap - a_entry + yoffset;
                            *x = (xbase + i) * BITMASK_W_LEN +
                                 firstsetbit(*ap & (*bp << shift));
                            return 1;
                        }
                    a_entry += a->h;
                    a_end += a->h;
                    for (ap = a_entry, bp = b_entry; ap < a_end; ap++, bp++)
                        if (*ap & (*bp >> rshift)) {
                            *y = ap - a_entry + yoffset;
                            *x = (xbase + i + 1) * BITMASK_W_LEN +
                                 firstsetbit(*ap & (*bp >> rshift));
                            return 1;
                        }
                    b_entry += b->h;
                }
                return 0;
            }
        }
        else
        /* xoffset is a multiple of the stripe width, and the above routines
         won't work. This way is also slightly faster. */
        {
            astripes = (MIN(b->w, a->w - xoffset) - 1) / BITMASK_W_LEN + 1;
            for (i = 0; i < astripes; i++) {
                for (ap = a_entry, bp = b_entry; ap < a_end; ap++, bp++) {
                    if (*ap & *bp) {
                        *y = ap - a_entry + yoffset;
                        *x = (xbase + i) * BITMASK_W_LEN +
                             firstsetbit(*ap & *bp);
                        return 1;
                    }
                }
                a_entry += a->h;
                a_end += a->h;
                b_entry += b->h;
            }
            return 0;
        }
    }
    else {
        if (bitmask_overlap_pos(b, a, -xoffset, -yoffset, x, y)) {
            *x += xoffset;
            *y += yoffset;
            return 1;
        }
        else
            return 0;
    }
}

int
bitmask_overlap_area(const bitmask_t *a, const bitmask_t *b, int xoffset,
                     int yoffset)
{
    const BITMASK_W *a_entry, *a_end, *b_entry, *ap, *bp;
    unsigned int shift, rshift, i, astripes, bstripes;
    unsigned int count = 0;

    /* Return if no overlap or one mask has a width/height of 0. */
    if ((xoffset >= a->w) || (yoffset >= a->h) || (yoffset <= -b->h) ||
        (xoffset <= -b->w) || (!a->h) || (!a->w) || (!b->h) || (!b->w)) {
        return 0;
    }

    if (xoffset >= 0) {
    swapentry:
        if (yoffset >= 0) {
            a_entry = a->bits + a->h * (xoffset / BITMASK_W_LEN) + yoffset;
            a_end = a_entry + MIN(b->h, a->h - yoffset);
            b_entry = b->bits;
        }
        else {
            a_entry = a->bits + a->h * (xoffset / BITMASK_W_LEN);
            a_end = a_entry + MIN(b->h + yoffset, a->h);
            b_entry = b->bits - yoffset;
        }
        shift = xoffset & BITMASK_W_MASK;
        if (shift) {
            rshift = BITMASK_W_LEN - shift;
            astripes = (a->w - 1) / BITMASK_W_LEN - xoffset / BITMASK_W_LEN;
            bstripes = (b->w - 1) / BITMASK_W_LEN + 1;
            if (bstripes > astripes) /* zig-zag .. zig*/
            {
                for (i = 0; i < astripes; i++) {
                    for (ap = a_entry, bp = b_entry; ap < a_end; ap++, bp++)
                        count += bitcount(
                            ((*ap >> shift) | (*(ap + a->h) << rshift)) & *bp);
                    a_entry += a->h;
                    a_end += a->h;
                    b_entry += b->h;
                }
                for (ap = a_entry, bp = b_entry; ap < a_end;)
                    count += bitcount((*ap++ >> shift) & *bp++);
                return count;
            }
            else /* zig-zag */
            {
                for (i = 0; i < bstripes; i++) {
                    for (ap = a_entry, bp = b_entry; ap < a_end; ap++, bp++)
                        count += bitcount(
                            ((*ap >> shift) | (*(ap + a->h) << rshift)) & *bp);
                    a_entry += a->h;
                    a_end += a->h;
                    b_entry += b->h;
                }
                return count;
            }
        }
        else /* xoffset is a multiple of the stripe width, and the above
                routines wont work */
        {
            astripes = (MIN(b->w, a->w - xoffset) - 1) / BITMASK_W_LEN + 1;
            for (i = 0; i < astripes; i++) {
                for (ap = a_entry, bp = b_entry; ap < a_end;)
                    count += bitcount(*ap++ & *bp++);

                a_entry += a->h;
                a_end += a->h;
                b_entry += b->h;
            }
            return count;
        }
    }
    else {
        const bitmask_t *c = a;
        a = b;
        b = c;
        xoffset *= -1;
        yoffset *= -1;
        goto swapentry;
    }
}

/* Makes a mask of the overlap of two other masks */
void
bitmask_overlap_mask(const bitmask_t *a, const bitmask_t *b, bitmask_t *c,
                     int xoffset, int yoffset)
{
    const BITMASK_W *a_entry, *a_end, *ap;
    const BITMASK_W *b_entry, *b_end, *bp;
    BITMASK_W *c_entry, *c_end, *cp;
    int shift, rshift, i, astripes, bstripes;

    /* Return if no overlap or one mask has a width/height of 0. */
    if ((xoffset >= a->w) || (yoffset >= a->h) || (yoffset <= -b->h) ||
        (xoffset <= -b->w) || (!a->h) || (!a->w) || (!b->h) || (!b->w)) {
        return;
    }

    if (xoffset >= 0) {
        if (yoffset >= 0) {
            a_entry = a->bits + a->h * (xoffset / BITMASK_W_LEN) + yoffset;
            c_entry = c->bits + c->h * (xoffset / BITMASK_W_LEN) + yoffset;
            a_end = a_entry + MIN(b->h, a->h - yoffset);
            b_entry = b->bits;
        }
        else {
            a_entry = a->bits + a->h * (xoffset / BITMASK_W_LEN);
            c_entry = c->bits + c->h * (xoffset / BITMASK_W_LEN);
            a_end = a_entry + MIN(b->h + yoffset, a->h);
            b_entry = b->bits - yoffset;
        }
        shift = xoffset & BITMASK_W_MASK;
        if (shift) {
            rshift = BITMASK_W_LEN - shift;
            astripes = (a->w - 1) / BITMASK_W_LEN - xoffset / BITMASK_W_LEN;
            bstripes = (b->w - 1) / BITMASK_W_LEN + 1;
            if (bstripes > astripes) /* zig-zag .. zig*/
            {
                for (i = 0; i < astripes; i++) {
                    for (ap = a_entry, bp = b_entry, cp = c_entry; ap < a_end;
                         ap++, bp++, cp++)
                        *cp = *ap & (*bp << shift);
                    a_entry += a->h;
                    a_end += a->h;
                    for (ap = a_entry, bp = b_entry, cp = c_entry; ap < a_end;
                         ap++, bp++, cp++)
                        *cp |= *ap & (*bp >> rshift);
                    b_entry += b->h;
                    c_entry += c->h;
                }
                for (ap = a_entry, bp = b_entry, cp = c_entry; ap < a_end;
                     ap++, bp++, cp++)
                    *cp = *ap & (*bp << shift);
            }
            else /* zig-zag */
            {
                for (i = 0; i < bstripes; i++) {
                    for (ap = a_entry, bp = b_entry, cp = c_entry; ap < a_end;
                         ap++, bp++, cp++)
                        *cp = *ap & (*bp << shift);
                    a_entry += a->h;
                    a_end += a->h;
                    for (ap = a_entry, bp = b_entry, cp = c_entry; ap < a_end;
                         ap++, bp++, cp++)
                        *cp |= *ap & (*bp >> rshift);
                    b_entry += b->h;
                    c_entry += c->h;
                }
            }
        }
        else /* xoffset is a multiple of the stripe width,
                and the above routines won't work. */
        {
            astripes = (MIN(b->w, a->w - xoffset) - 1) / BITMASK_W_LEN + 1;
            for (i = 0; i < astripes; i++) {
                for (ap = a_entry, bp = b_entry, cp = c_entry; ap < a_end;
                     ap++, bp++, cp++) {
                    *cp = *ap & *bp;
                }
                a_entry += a->h;
                c_entry += c->h;
                a_end += a->h;
                b_entry += b->h;
            }
        }
    }
    else {
        xoffset *= -1;
        yoffset *= -1;

        if (yoffset >= 0) {
            b_entry = b->bits + b->h * (xoffset / BITMASK_W_LEN) + yoffset;
            b_end = b_entry + MIN(a->h, b->h - yoffset);
            a_entry = a->bits;
            c_entry = c->bits;
        }
        else {
            b_entry = b->bits + b->h * (xoffset / BITMASK_W_LEN);
            b_end = b_entry + MIN(a->h + yoffset, b->h);
            a_entry = a->bits - yoffset;
            c_entry = c->bits - yoffset;
        }
        shift = xoffset & BITMASK_W_MASK;
        if (shift) {
            rshift = BITMASK_W_LEN - shift;
            astripes = (b->w - 1) / BITMASK_W_LEN - xoffset / BITMASK_W_LEN;
            bstripes = (a->w - 1) / BITMASK_W_LEN + 1;
            if (bstripes > astripes) /* zig-zag .. zig*/
            {
                for (i = 0; i < astripes; i++) {
                    for (bp = b_entry, ap = a_entry, cp = c_entry; bp < b_end;
                         bp++, ap++, cp++)
                        *cp = *ap & (*bp >> shift);
                    b_entry += b->h;
                    b_end += b->h;
                    for (bp = b_entry, ap = a_entry, cp = c_entry; bp < b_end;
                         bp++, ap++, cp++)
                        *cp |= *ap & (*bp << rshift);
                    a_entry += a->h;
                    c_entry += c->h;
                }
                for (bp = b_entry, ap = a_entry, cp = c_entry; bp < b_end;
                     bp++, ap++, cp++)
                    *cp = *ap & (*bp >> shift);
            }
            else /* zig-zag */
            {
                for (i = 0; i < bstripes; i++) {
                    for (bp = b_entry, ap = a_entry, cp = c_entry; bp < b_end;
                         bp++, ap++, cp++)
                        *cp = *ap & (*bp >> shift);
                    b_entry += b->h;
                    b_end += b->h;
                    for (bp = b_entry, ap = a_entry, cp = c_entry; bp < b_end;
                         bp++, ap++, cp++)
                        *cp |= *ap & (*bp << rshift);
                    a_entry += a->h;
                    c_entry += c->h;
                }
            }
        }
        else /* xoffset is a multiple of the stripe width, and the above
                routines won't work. */
        {
            astripes = (MIN(a->w, b->w - xoffset) - 1) / BITMASK_W_LEN + 1;
            for (i = 0; i < astripes; i++) {
                for (bp = b_entry, ap = a_entry, cp = c_entry; bp < b_end;
                     bp++, ap++, cp++) {
                    *cp = *ap & *bp;
                }
                b_entry += b->h;
                b_end += b->h;
                a_entry += a->h;
                c_entry += c->h;
            }
        }
        xoffset *= -1;
        yoffset *= -1;
    }
    /* Zero out bits outside the mask rectangle (to the right), if there
     is a chance we were drawing there. */
    if (xoffset + b->w > c->w) {
        BITMASK_W edgemask;
        int n = c->w / BITMASK_W_LEN;
        shift = (n + 1) * BITMASK_W_LEN - c->w;
        edgemask = (~(BITMASK_W)0) >> shift;
        c_end = c->bits + n * c->h + MIN(c->h, b->h + yoffset);
        for (cp = c->bits + n * c->h + MAX(yoffset, 0); cp < c_end; cp++)
            *cp &= edgemask;
    }
}

/* Draws mask b onto mask a (bitwise OR) */
void
bitmask_draw(bitmask_t *a, const bitmask_t *b, int xoffset, int yoffset)
{
    BITMASK_W *a_entry, *a_end, *ap;
    const BITMASK_W *b_entry, *b_end, *bp;
    int shift, rshift, i, astripes, bstripes;

    /* Return if no overlap or one mask has a width/height of 0. */
    if ((xoffset >= a->w) || (yoffset >= a->h) || (yoffset <= -b->h) ||
        (xoffset <= -b->w) || (!a->h) || (!a->w) || (!b->h) || (!b->w)) {
        return;
    }

    if (xoffset >= 0) {
        if (yoffset >= 0) {
            a_entry = a->bits + a->h * (xoffset / BITMASK_W_LEN) + yoffset;
            a_end = a_entry + MIN(b->h, a->h - yoffset);
            b_entry = b->bits;
        }
        else {
            a_entry = a->bits + a->h * (xoffset / BITMASK_W_LEN);
            a_end = a_entry + MIN(b->h + yoffset, a->h);
            b_entry = b->bits - yoffset;
        }
        shift = xoffset & BITMASK_W_MASK;
        if (shift) {
            rshift = BITMASK_W_LEN - shift;
            astripes = (a->w - 1) / BITMASK_W_LEN - xoffset / BITMASK_W_LEN;
            bstripes = (b->w - 1) / BITMASK_W_LEN + 1;
            if (bstripes > astripes) /* zig-zag .. zig*/
            {
                for (i = 0; i < astripes; i++) {
                    for (ap = a_entry, bp = b_entry; ap < a_end; ap++, bp++)
                        *ap |= (*bp << shift);
                    a_entry += a->h;
                    a_end += a->h;
                    for (ap = a_entry, bp = b_entry; ap < a_end; ap++, bp++)
                        *ap |= (*bp >> rshift);
                    b_entry += b->h;
                }
                for (ap = a_entry, bp = b_entry; ap < a_end; ap++, bp++)
                    *ap |= (*bp << shift);
            }
            else /* zig-zag */
            {
                for (i = 0; i < bstripes; i++) {
                    for (ap = a_entry, bp = b_entry; ap < a_end; ap++, bp++)
                        *ap |= (*bp << shift);
                    a_entry += a->h;
                    a_end += a->h;
                    for (ap = a_entry, bp = b_entry; ap < a_end; ap++, bp++)
                        *ap |= (*bp >> rshift);
                    b_entry += b->h;
                }
            }
        }
        else /* xoffset is a multiple of the stripe width,
                and the above routines won't work. */
        {
            astripes = (MIN(b->w, a->w - xoffset) - 1) / BITMASK_W_LEN + 1;
            for (i = 0; i < astripes; i++) {
                for (ap = a_entry, bp = b_entry; ap < a_end; ap++, bp++) {
                    *ap |= *bp;
                }
                a_entry += a->h;
                a_end += a->h;
                b_entry += b->h;
            }
        }
    }
    else {
        xoffset *= -1;
        yoffset *= -1;

        if (yoffset >= 0) {
            b_entry = b->bits + b->h * (xoffset / BITMASK_W_LEN) + yoffset;
            b_end = b_entry + MIN(a->h, b->h - yoffset);
            a_entry = a->bits;
        }
        else {
            b_entry = b->bits + b->h * (xoffset / BITMASK_W_LEN);
            b_end = b_entry + MIN(a->h + yoffset, b->h);
            a_entry = a->bits - yoffset;
        }
        shift = xoffset & BITMASK_W_MASK;
        if (shift) {
            rshift = BITMASK_W_LEN - shift;
            astripes = (b->w - 1) / BITMASK_W_LEN - xoffset / BITMASK_W_LEN;
            bstripes = (a->w - 1) / BITMASK_W_LEN + 1;
            if (bstripes > astripes) /* zig-zag .. zig*/
            {
                for (i = 0; i < astripes; i++) {
                    for (bp = b_entry, ap = a_entry; bp < b_end; bp++, ap++)
                        *ap |= (*bp >> shift);
                    b_entry += b->h;
                    b_end += b->h;
                    for (bp = b_entry, ap = a_entry; bp < b_end; bp++, ap++)
                        *ap |= (*bp << rshift);
                    a_entry += a->h;
                }
                for (bp = b_entry, ap = a_entry; bp < b_end; bp++, ap++)
                    *ap |= (*bp >> shift);
            }
            else /* zig-zag */
            {
                for (i = 0; i < bstripes; i++) {
                    for (bp = b_entry, ap = a_entry; bp < b_end; bp++, ap++)
                        *ap |= (*bp >> shift);
                    b_entry += b->h;
                    b_end += b->h;
                    for (bp = b_entry, ap = a_entry; bp < b_end; bp++, ap++)
                        *ap |= (*bp << rshift);
                    a_entry += a->h;
                }
            }
        }
        else /* xoffset is a multiple of the stripe width, and the above
                routines won't work. */
        {
            astripes = (MIN(a->w, b->w - xoffset) - 1) / BITMASK_W_LEN + 1;
            for (i = 0; i < astripes; i++) {
                for (bp = b_entry, ap = a_entry; bp < b_end; bp++, ap++) {
                    *ap |= *bp;
                }
                b_entry += b->h;
                b_end += b->h;
                a_entry += a->h;
            }
        }
        xoffset *= -1;
        yoffset *= -1;
    }
    /* Zero out bits outside the mask rectangle (to the right), if there
     is a chance we were drawing there. */
    if (xoffset + b->w > a->w) {
        BITMASK_W edgemask;
        int n = a->w / BITMASK_W_LEN;
        shift = (n + 1) * BITMASK_W_LEN - a->w;
        edgemask = (~(BITMASK_W)0) >> shift;
        a_end = a->bits + n * a->h + MIN(a->h, b->h + yoffset);
        for (ap = a->bits + n * a->h + MAX(yoffset, 0); ap < a_end; ap++)
            *ap &= edgemask;
    }
}

/* Erases mask b from mask a (a &= ~b) */
void
bitmask_erase(bitmask_t *a, const bitmask_t *b, int xoffset, int yoffset)
{
    BITMASK_W *a_entry, *a_end, *ap;
    const BITMASK_W *b_entry, *b_end, *bp;
    int shift, rshift, i, astripes, bstripes;

    /* Return if no overlap or one mask has a width/height of 0. */
    if ((xoffset >= a->w) || (yoffset >= a->h) || (yoffset <= -b->h) ||
        (xoffset <= -b->w) || (!a->h) || (!a->w) || (!b->h) || (!b->w)) {
        return;
    }

    if (xoffset >= 0) {
        if (yoffset >= 0) {
            a_entry = a->bits + a->h * (xoffset / BITMASK_W_LEN) + yoffset;
            a_end = a_entry + MIN(b->h, a->h - yoffset);
            b_entry = b->bits;
        }
        else {
            a_entry = a->bits + a->h * (xoffset / BITMASK_W_LEN);
            a_end = a_entry + MIN(b->h + yoffset, a->h);
            b_entry = b->bits - yoffset;
        }
        shift = xoffset & BITMASK_W_MASK;
        if (shift) {
            rshift = BITMASK_W_LEN - shift;
            astripes = (a->w - 1) / BITMASK_W_LEN - xoffset / BITMASK_W_LEN;
            bstripes = (b->w - 1) / BITMASK_W_LEN + 1;
            if (bstripes > astripes) /* zig-zag .. zig*/
            {
                for (i = 0; i < astripes; i++) {
                    for (ap = a_entry, bp = b_entry; ap < a_end; ap++, bp++)
                        *ap &= ~(*bp << shift);
                    a_entry += a->h;
                    a_end += a->h;
                    for (ap = a_entry, bp = b_entry; ap < a_end; ap++, bp++)
                        *ap &= ~(*bp >> rshift);
                    b_entry += b->h;
                }
                for (ap = a_entry, bp = b_entry; ap < a_end; ap++, bp++)
                    *ap &= ~(*bp << shift);
            }
            else /* zig-zag */
            {
                for (i = 0; i < bstripes; i++) {
                    for (ap = a_entry, bp = b_entry; ap < a_end; ap++, bp++)
                        *ap &= ~(*bp << shift);
                    a_entry += a->h;
                    a_end += a->h;
                    for (ap = a_entry, bp = b_entry; ap < a_end; ap++, bp++)
                        *ap &= ~(*bp >> rshift);
                    b_entry += b->h;
                }
            }
        }
        else /* xoffset is a multiple of the stripe width,
              and the above routines won't work. */
        {
            astripes = (MIN(b->w, a->w - xoffset) - 1) / BITMASK_W_LEN + 1;
            for (i = 0; i < astripes; i++) {
                for (ap = a_entry, bp = b_entry; ap < a_end; ap++, bp++) {
                    *ap &= ~*bp;
                }
                a_entry += a->h;
                a_end += a->h;
                b_entry += b->h;
            }
        }
    }
    else {
        xoffset *= -1;
        yoffset *= -1;

        if (yoffset >= 0) {
            b_entry = b->bits + b->h * (xoffset / BITMASK_W_LEN) + yoffset;
            b_end = b_entry + MIN(a->h, b->h - yoffset);
            a_entry = a->bits;
        }
        else {
            b_entry = b->bits + b->h * (xoffset / BITMASK_W_LEN);
            b_end = b_entry + MIN(a->h + yoffset, b->h);
            a_entry = a->bits - yoffset;
        }
        shift = xoffset & BITMASK_W_MASK;
        if (shift) {
            rshift = BITMASK_W_LEN - shift;
            astripes = (b->w - 1) / BITMASK_W_LEN - xoffset / BITMASK_W_LEN;
            bstripes = (a->w - 1) / BITMASK_W_LEN + 1;
            if (bstripes > astripes) /* zig-zag .. zig*/
            {
                for (i = 0; i < astripes; i++) {
                    for (bp = b_entry, ap = a_entry; bp < b_end; bp++, ap++)
                        *ap &= ~(*bp >> shift);
                    b_entry += b->h;
                    b_end += b->h;
                    for (bp = b_entry, ap = a_entry; bp < b_end; bp++, ap++)
                        *ap &= ~(*bp << rshift);
                    a_entry += a->h;
                }
                for (bp = b_entry, ap = a_entry; bp < b_end; bp++, ap++)
                    *ap &= ~(*bp >> shift);
            }
            else /* zig-zag */
            {
                for (i = 0; i < bstripes; i++) {
                    for (bp = b_entry, ap = a_entry; bp < b_end; bp++, ap++)
                        *ap &= ~(*bp >> shift);
                    b_entry += b->h;
                    b_end += b->h;
                    for (bp = b_entry, ap = a_entry; bp < b_end; bp++, ap++)
                        *ap &= ~(*bp << rshift);
                    a_entry += a->h;
                }
            }
        }
        else /* xoffset is a multiple of the stripe width, and the above
                routines won't work. */
        {
            astripes = (MIN(a->w, b->w - xoffset) - 1) / BITMASK_W_LEN + 1;
            for (i = 0; i < astripes; i++) {
                for (bp = b_entry, ap = a_entry; bp < b_end; bp++, ap++)
                    *ap &= ~*bp;
                b_entry += b->h;
                b_end += b->h;
                a_entry += a->h;
            }
        }
    }
}

bitmask_t *
bitmask_scale(const bitmask_t *m, int w, int h)
{
    bitmask_t *nm;
    int x, y, nx, ny, dx, dy, dnx, dny;

    if (m->w < 0 || m->h < 0 || w < 0 || h < 0) {
        return 0;
    }

    nm = bitmask_create(w, h);

    if (!nm)
        return NULL;

    ny = dny = 0;
    for (y = 0, dy = h; y < m->h; y++, dy += h) {
        while (dny < dy) {
            nx = dnx = 0;
            for (x = 0, dx = w; x < m->w; x++, dx += w) {
                if (bitmask_getbit(m, x, y)) {
                    while (dnx < dx) {
                        bitmask_setbit(nm, nx, ny);
                        nx++;
                        dnx += m->w;
                    }
                }
                else {
                    while (dnx < dx) {
                        nx++;
                        dnx += m->w;
                    }
                }
            }
            ny++;
            dny += m->h;
        }
    }
    return nm;
}

void
bitmask_convolve(const bitmask_t *a, const bitmask_t *b, bitmask_t *output,
                 int xoffset, int yoffset)
{
    int x, y;

    if (!a->h || !a->w || !b->h || !b->w || !output->h || !output->w) {
        return;
    }

    xoffset += b->w - 1;
    yoffset += b->h - 1;

    for (y = 0; y < b->h; y++)
        for (x = 0; x < b->w; x++)
            if (bitmask_getbit(b, x, y))
                bitmask_draw(output, a, xoffset - x, yoffset - y);
}
