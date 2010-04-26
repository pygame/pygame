/*
  pygame - Python Game Library
  Copyright (C) 2000-2001  Pete Shinners, 2007 Rene Dudfied, Richard Goedeken

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
#ifndef _PYGAME_TRANSFORM_H_
#define _PYGAME_TRANSFORM_H_

#include <math.h>
#include <SDL.h>
#include "pgdefines.h"
#include "filters.h"

SDL_Surface*
pyg_transform_scale (SDL_Surface *srcsurface, SDL_Surface *dstsurface,
    int width, int height);

SDL_Surface*
pyg_transform_rotate90 (SDL_Surface *surface, int angle);

SDL_Surface*
pyg_transform_rotate (SDL_Surface *surface, double angle);

SDL_Surface*
pyg_transform_flip (SDL_Surface *surface, int xaxis, int yaxis);

SDL_Surface*
pyg_transform_chop (SDL_Surface *surface, SDL_Rect *rect);

SDL_Surface*
pyg_transform_scale2x (SDL_Surface *srcsurface, SDL_Surface *dstsurface);

SDL_Surface*
pyg_transform_smoothscale (SDL_Surface *srcsurface, SDL_Surface *dstsurface,
    int width, int height, FilterFuncs *filters);

SDL_Surface*
pyg_transform_laplacian (SDL_Surface *srcsurface, SDL_Surface *dstsurface);

SDL_Surface*
pyg_transform_average_surfaces (SDL_Surface **surfaces, int count,
    SDL_Surface *dstsurface);

int
pyg_transform_average_color (SDL_Surface *surface, SDL_Rect *rect,
    Uint8 *r, Uint8 *g, Uint8 *b, Uint8 *a);

int
pyg_transform_threshold_color (SDL_Surface *srcsurface, Uint32 diffcolor,
    Uint32 threscolor, SDL_Surface *dstsurface);

int
pyg_transform_threshold_surface (SDL_Surface *srcsurface,
    SDL_Surface *diffsurface, Uint32 threscolor, SDL_Surface *dstsurface);

#endif /* _PYGAME_TRANSFORM_H_ */
