/*
    pygame - Python Game Library
    Copyright (C) 2000-2001  Pete Shinners

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

/*
   This implements the AdvanceMAME Scale2x feature found on this page,
   http://advancemame.sourceforge.net/scale2x.html

   It is an incredibly simple and powerful image doubling routine that does
   an astonishing job of doubling game graphic data while interpolating out
   the jaggies. Congrats to the AdvanceMAME team, I'm very impressed and
   surprised with this code!
*/

#ifdef PG_SDL3
#include <SDL3/SDL.h>
#else
#include <SDL.h>
#endif

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

static inline int
read_int24(const Uint8 *x)
{
    return (x[0] << 16 | x[1] << 8 | x[2]);
}

static inline void
store_int24(Uint8 *x, int i)
{
    x[0] = i >> 16;
    x[1] = (i >> 8) & 0xff;
    x[2] = i & 0xff;
}

/*
  this requires a destination surface already setup to be twice as
  large as the source. oh, and formats must match too. this will just
  blindly assume you didn't flounder.
*/

void
scale2x(SDL_Surface *src, SDL_Surface *dst)
{
    int looph, loopw;

    Uint8 *srcpix = (Uint8 *)src->pixels;
    Uint8 *dstpix = (Uint8 *)dst->pixels;

    const int srcpitch = src->pitch;
    const int dstpitch = dst->pitch;
    const int width = src->w;
    const int height = src->h;

#if SDL_VERSION_ATLEAST(3, 0, 0)
    const Uint8 Bpp = SDL_BYTESPERPIXEL(src->format);
#else
    const Uint8 Bpp = src->format->BytesPerPixel;
#endif

    switch (Bpp) {
        case 1: {
            Uint8 E0, E1, E2, E3, B, D, E, F, H;
            for (looph = 0; looph < height; ++looph) {
                Uint8 *src_row = srcpix + looph * srcpitch;
                Uint8 *dst_row0 = dstpix + looph * 2 * dstpitch;
                Uint8 *dst_row1 = dstpix + (looph * 2 + 1) * dstpitch;

                Uint8 *src_row_prev = srcpix + MAX(0, looph - 1) * srcpitch;
                Uint8 *src_row_next =
                    srcpix + MIN(height - 1, looph + 1) * srcpitch;

                for (loopw = 0; loopw < width; ++loopw) {
                    B = *(Uint8 *)(src_row_prev + loopw);
                    D = *(Uint8 *)(src_row + MAX(0, loopw - 1));
                    E = *(Uint8 *)(src_row + loopw);
                    F = *(Uint8 *)(src_row + MIN(width - 1, loopw + 1));
                    H = *(Uint8 *)(src_row_next + loopw);

                    if (B != H && D != F) {
                        E0 = (D == B) ? D : E;
                        E1 = (B == F) ? F : E;
                        E2 = (D == H) ? D : E;
                        E3 = (H == F) ? F : E;
                    }
                    else {
                        E0 = E;
                        E1 = E;
                        E2 = E;
                        E3 = E;
                    }

                    *(Uint8 *)(dst_row0 + loopw * 2) = E0;
                    *(Uint8 *)(dst_row0 + loopw * 2 + 1) = E1;
                    *(Uint8 *)(dst_row1 + loopw * 2) = E2;
                    *(Uint8 *)(dst_row1 + loopw * 2 + 1) = E3;
                }
            }
            break;
        }
        case 2: {
            Uint16 E0, E1, E2, E3, B, D, E, F, H;
            for (looph = 0; looph < height; ++looph) {
                Uint8 *src_row = srcpix + looph * srcpitch;
                Uint8 *dst_row0 = dstpix + looph * 2 * dstpitch;
                Uint8 *dst_row1 = dstpix + (looph * 2 + 1) * dstpitch;

                Uint8 *src_row_prev = srcpix + MAX(0, looph - 1) * srcpitch;
                Uint8 *src_row_next =
                    srcpix + MIN(height - 1, looph + 1) * srcpitch;

                for (loopw = 0; loopw < width; ++loopw) {
                    B = *(Uint16 *)(src_row_prev + 2 * loopw);
                    D = *(Uint16 *)(src_row + 2 * MAX(0, loopw - 1));
                    E = *(Uint16 *)(src_row + 2 * loopw);
                    F = *(Uint16 *)(src_row + 2 * MIN(width - 1, loopw + 1));
                    H = *(Uint16 *)(src_row_next + 2 * loopw);

                    if (B != H && D != F) {
                        E0 = (D == B) ? D : E;
                        E1 = (B == F) ? F : E;
                        E2 = (D == H) ? D : E;
                        E3 = (H == F) ? F : E;
                    }
                    else {
                        E0 = E;
                        E1 = E;
                        E2 = E;
                        E3 = E;
                    }

                    *(Uint16 *)(dst_row0 + loopw * 2 * 2) = E0;
                    *(Uint16 *)(dst_row0 + (loopw * 2 + 1) * 2) = E1;
                    *(Uint16 *)(dst_row1 + loopw * 2 * 2) = E2;
                    *(Uint16 *)(dst_row1 + (loopw * 2 + 1) * 2) = E3;
                }
            }
            break;
        }
        case 3: {
            int E0, E1, E2, E3, B, D, E, F, H;
            for (looph = 0; looph < height; ++looph) {
                Uint8 *src_row = srcpix + looph * srcpitch;
                Uint8 *dst_row0 = dstpix + looph * 2 * dstpitch;
                Uint8 *dst_row1 = dstpix + (looph * 2 + 1) * dstpitch;

                Uint8 *src_row_prev = srcpix + MAX(0, looph - 1) * srcpitch;
                Uint8 *src_row_next =
                    srcpix + MIN(height - 1, looph + 1) * srcpitch;

                for (loopw = 0; loopw < width; ++loopw) {
                    B = read_int24(src_row_prev + (3 * loopw));
                    D = read_int24(src_row + (3 * MAX(0, loopw - 1)));
                    E = read_int24(src_row + (3 * loopw));
                    F = read_int24(src_row + (3 * MIN(width - 1, loopw + 1)));
                    H = read_int24(src_row_next + (3 * loopw));

                    if (B != H && D != F) {
                        E0 = (D == B) ? D : E;
                        E1 = (B == F) ? F : E;
                        E2 = (D == H) ? D : E;
                        E3 = (H == F) ? F : E;
                    }
                    else {
                        E0 = E;
                        E1 = E;
                        E2 = E;
                        E3 = E;
                    }

                    store_int24(dst_row0 + loopw * 2 * 3, E0);
                    store_int24(dst_row0 + (loopw * 2 + 1) * 3, E1);
                    store_int24(dst_row1 + loopw * 2 * 3, E2);
                    store_int24(dst_row1 + (loopw * 2 + 1) * 3, E3);
                }
            }
            break;
        }
        default: {
            Uint32 E0, E1, E2, E3, B, D, E, F, H;

            for (looph = 0; looph < height; ++looph) {
                Uint8 *src_row = srcpix + looph * srcpitch;
                Uint8 *dst_row0 = dstpix + looph * 2 * dstpitch;
                Uint8 *dst_row1 = dstpix + (looph * 2 + 1) * dstpitch;

                Uint8 *src_row_prev = srcpix + MAX(0, looph - 1) * srcpitch;
                Uint8 *src_row_next =
                    srcpix + MIN(height - 1, looph + 1) * srcpitch;

                for (loopw = 0; loopw < width; ++loopw) {
                    B = *(Uint32 *)(src_row_prev + 4 * loopw);
                    D = *(Uint32 *)(src_row + 4 * MAX(0, loopw - 1));
                    E = *(Uint32 *)(src_row + 4 * loopw);
                    F = *(Uint32 *)(src_row + 4 * MIN(width - 1, loopw + 1));
                    H = *(Uint32 *)(src_row_next + 4 * loopw);

                    if (B != H && D != F) {
                        E0 = (D == B) ? D : E;
                        E1 = (B == F) ? F : E;
                        E2 = (D == H) ? D : E;
                        E3 = (H == F) ? F : E;
                    }
                    else {
                        E0 = E;
                        E1 = E;
                        E2 = E;
                        E3 = E;
                    }

                    *(Uint32 *)(dst_row0 + loopw * 2 * 4) = E0;
                    *(Uint32 *)(dst_row0 + (loopw * 2 + 1) * 4) = E1;
                    *(Uint32 *)(dst_row1 + loopw * 2 * 4) = E2;
                    *(Uint32 *)(dst_row1 + (loopw * 2 + 1) * 4) = E3;
                }
            }
            break;
        }
    }
}
