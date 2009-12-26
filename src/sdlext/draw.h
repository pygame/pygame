/*
  pygame - Python Game Library
  Copyright (C) 2000-2001 Pete Shinners

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
#ifndef _PYGAME_DRAW_H_
#define _PYGAME_DRAW_H_

#include <SDL.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LOCK_SURFACE(x,ret)                                             \
    if (SDL_MUSTLOCK (x))                                               \
    {                                                                   \
        if (SDL_LockSurface (x) == -1)                                  \
            return (ret);                                               \
    }

#define UNLOCK_SURFACE(x)                       \
    if (SDL_MUSTLOCK (x))                       \
        SDL_UnlockSurface (x);

int
pyg_draw_aaline (SDL_Surface *surface, SDL_Rect *cliprect, Uint32 color,
    int x1, int _y1, int x2, int y2, int blendargs, SDL_Rect *area);

int
pyg_draw_line (SDL_Surface *surface, SDL_Rect *cliprect, Uint32 color,
    int x1, int _y1, int x2, int y2, int width, SDL_Rect *area);

int
pyg_draw_aalines (SDL_Surface *surface, SDL_Rect *cliprect, Uint32 color,
    int *xpts, int *ypts, unsigned int count, int blendargs, SDL_Rect *area);

int
pyg_draw_lines (SDL_Surface *surface, SDL_Rect *cliprect, Uint32 color,
    int *xpts, int *ypts, unsigned int count, int width, SDL_Rect *area);

int
pyg_draw_filled_ellipse (SDL_Surface *surface, SDL_Rect *cliprect,
    Uint32 color, int x, int y, int radx, int rady, SDL_Rect *area);

int
pyg_draw_ellipse (SDL_Surface *surface, SDL_Rect *cliprect, Uint32 color,
    int x, int y, int radx, int rady, SDL_Rect *area);

int
pyg_draw_arc (SDL_Surface *surface, SDL_Rect *cliprect, Uint32 color, int x,
    int y, int rad1, int rad2, double anglestart, double anglestop,
    SDL_Rect *area);

int
pyg_draw_aapolygon (SDL_Surface *surface, SDL_Rect *cliprect, Uint32 color,
    int *xpts, int *ypts, unsigned int count, int blendargs, SDL_Rect *area);

int
pyg_draw_polygon (SDL_Surface *surface, SDL_Rect *cliprect, Uint32 color,
    int *xpts, int *ypts, unsigned int count, int width, SDL_Rect *area);

int
pyg_draw_filled_polygon (SDL_Surface *surface, SDL_Rect *cliprect, Uint32 color,
    int *xpts, int *ypts, unsigned int count, SDL_Rect *area);

#ifdef __cplusplus
}
#endif

#endif /* _PYGAME_DRAW_H_ */
