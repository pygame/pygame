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

#include <SDL.h>
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

#define READINT24(x) ((x)[0] << 16 | (x)[1] << 8 | (x)[2])
#define WRITEINT24(x, i)          \
    {                             \
        (x)[0] = i >> 16;         \
        (x)[1] = (i >> 8) & 0xff; \
        x[2] = i & 0xff;          \
    }

/*
  this requires a destination surface already setup to be twice as
  large as the source. oh, and formats must match too. this will just
  blindly assume you didn't flounder.
*/

void scale2x(SDL_Surface* src, SDL_Surface* dst)
{

    Uint8 *srcpix = (Uint8 *)src->pixels;
    Uint8 *dstpix = (Uint8 *)dst->pixels;

    const int srcpitch = src->pitch;
    const int dstpitch = dst->pitch;
    const int width = src->w;
    const int height = src->h;

    int bytes_ppixel = src->format->BytesPerPixel; 
    Uint32 E0, E1, E2, E3, B, D, E, F, H;
    // I had to sacrifice Uint16/8. the compiler shall fill the remaining area with zeroes
    for(int looph = 0; looph < height; ++looph)
        for (int loopw = 0; loopw < width; ++loopw)
        {
            // multiply by bytes_ppixel rather than doing a case for each different scenario
            B = *(Uint32 *)(srcpix + (MAX(0, looph - 1) * srcpitch) + (bytes_ppixel * loopw));

            D = *(Uint32 *)(srcpix + (looph * srcpitch) + (bytes_ppixel * MAX(0, loopw - 1)));

            E = *(Uint32 *)(srcpix + (looph * srcpitch) + (bytes_ppixel * loopw));

            F = *(Uint32 *)(srcpix + (looph * srcpitch) + (bytes_ppixel * MIN(width - 1, loopw + 1)));

            H = *(Uint32 *)(srcpix + (MIN(height - 1, looph + 1) * srcpitch) + (bytes_ppixel * loopw));
            
            E0 = D == B && B != F && D != H ? D : E;
            E1 = B == F && B != D && F != H ? F : E;
            E2 = D == H && D != B && H != F ? D : E;
            E3 = H == F && D != H && B != F ? F : E;            

            *(Uint32 *)(dstpix + looph * 2 * dstpitch + loopw * 2 * bytes_ppixel) = E0;

            *(Uint32 *)(dstpix + looph * 2 * dstpitch + (loopw * 2 + 1) * bytes_ppixel) = E1;
                    
            *(Uint32 *)(dstpix + (looph * 2 + 1) * dstpitch + loopw * 2 * bytes_ppixel) = E2;
            
            *(Uint32 *)(dstpix + (looph * 2 + 1) * dstpitch + (loopw * 2 + 1) * bytes_ppixel) = E3;
        }
}

void
scale2xraw(SDL_Surface *src, SDL_Surface *dst)
{
    int looph, loopw;

    Uint8 *srcpix = (Uint8 *)src->pixels;
    Uint8 *dstpix = (Uint8 *)dst->pixels;

    const int srcpitch = src->pitch;
    const int dstpitch = dst->pitch;
    const int width = src->w;
    const int height = src->h;

    switch (src->format->BytesPerPixel) {
        case 1: {
            Uint8 E0, E1, E2, E3, E;
            for (looph = 0; looph < height; ++looph) {
                for (loopw = 0; loopw < width; ++loopw) {
                    E = *(Uint8 *)(srcpix + (looph * srcpitch) + (1 * loopw));

                    E0 = E;
                    E1 = E;
                    E2 = E;
                    E3 = E;

                    *(Uint8 *)(dstpix + looph * 2 * dstpitch + loopw * 2 * 1) =
                        E0;
                    *(Uint8 *)(dstpix + looph * 2 * dstpitch +
                               (loopw * 2 + 1) * 1) = E1;
                    *(Uint8 *)(dstpix + (looph * 2 + 1) * dstpitch +
                               loopw * 2 * 1) = E2;
                    *(Uint8 *)(dstpix + (looph * 2 + 1) * dstpitch +
                               (loopw * 2 + 1) * 1) = E3;
                }
            }
            break;
        }
        case 2: {
            Uint16 E0, E1, E2, E3, E;
            for (looph = 0; looph < height; ++looph) {
                for (loopw = 0; loopw < width; ++loopw) {
                    E = *(Uint16 *)(srcpix + (looph * srcpitch) + (2 * loopw));

                    E0 = E;
                    E1 = E;
                    E2 = E;
                    E3 = E;

                    *(Uint16 *)(dstpix + looph * 2 * dstpitch +
                                loopw * 2 * 2) = E0;
                    *(Uint16 *)(dstpix + looph * 2 * dstpitch +
                                (loopw * 2 + 1) * 2) = E1;
                    *(Uint16 *)(dstpix + (looph * 2 + 1) * dstpitch +
                                loopw * 2 * 2) = E2;
                    *(Uint16 *)(dstpix + (looph * 2 + 1) * dstpitch +
                                (loopw * 2 + 1) * 2) = E3;
                }
            }
            break;
        }
        case 3: {
            int E0, E1, E2, E3, E;
            for (looph = 0; looph < height; ++looph) {
                for (loopw = 0; loopw < width; ++loopw) {
                    E = READINT24(srcpix + (looph * srcpitch) + (3 * loopw));

                    E0 = E;
                    E1 = E;
                    E2 = E;
                    E3 = E;

                    WRITEINT24((dstpix + looph * 2 * dstpitch + loopw * 2 * 3),
                               E0);
                    WRITEINT24(
                        (dstpix + looph * 2 * dstpitch + (loopw * 2 + 1) * 3),
                        E1);
                    WRITEINT24(
                        (dstpix + (looph * 2 + 1) * dstpitch + loopw * 2 * 3),
                        E2);
                    WRITEINT24((dstpix + (looph * 2 + 1) * dstpitch +
                                (loopw * 2 + 1) * 3),
                               E3);
                }
            }
            break;
        }
        default: { /*case 4:*/
            Uint32 E0, E1, E2, E3, E;
            for (looph = 0; looph < height; ++looph) {
                for (loopw = 0; loopw < width; ++loopw) {
                    E = *(Uint32 *)(srcpix + (looph * srcpitch) + (4 * loopw));

                    E0 = E;
                    E1 = E;
                    E2 = E;
                    E3 = E;

                    *(Uint32 *)(dstpix + looph * 2 * dstpitch +
                                loopw * 2 * 4) = E0;
                    *(Uint32 *)(dstpix + looph * 2 * dstpitch +
                                (loopw * 2 + 1) * 4) = E1;
                    *(Uint32 *)(dstpix + (looph * 2 + 1) * dstpitch +
                                loopw * 2 * 4) = E2;
                    *(Uint32 *)(dstpix + (looph * 2 + 1) * dstpitch +
                                (loopw * 2 + 1) * 4) = E3;
                }
            }
            break;
        }
    }
}
