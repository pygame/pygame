/*
  pygame - Python Game Library
  Copyright (C) 2000-2001 Pete Shinners, 2006 Rene Dudfield

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
#ifndef _PYGAME_JPG_H_
#define _PYGAME_JPG_H_

#include <SDL.h>

/**
 * \brief Saves a SDL_Surface as JPG image.
 *
 * @param surface The SDL_Surface to save.
 * @param file The filename to save the surface to.
 * @return 0 on failure, !0 on success.
 */
int
pyg_save_jpeg (SDL_Surface *surface, char *file);

/**
 * \brief Saves a SDL_Surface as JPG image.
 *
 * @param surface The SDL_Surface to save.
 * @param rw The SDL_RWops to save the surface to.
 * @param freerw Indicates, whether the SDL_RWops shall be freed.
 * @return 0 on failure, !0 on success.
 */
int
pyg_save_jpeg_rw (SDL_Surface *surface, SDL_RWops *rw, int freerw);

#endif /* _PYGAME_JPG_H_ */
