/*
  pygame - Python Game Library
  Copyright (C) 2006, 2007 Rene Dudfield, Marcus von Appen

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

#ifndef _PYGAME_SCRAPX11_H_
#define _PYGAME_SCRAPX11_H_

#include "scrap.h"

#ifdef SDL_VIDEO_DRIVER_X11
int
scrap_init_x11 (void);

void
scrap_quit_x11 (void);

int
scrap_contains_x11 (char *type);

int
scrap_lost_x11 (void);

ScrapType
scrap_get_mode_x11 (void);

ScrapType
scrap_set_mode_x11 (ScrapType mode);

int
scrap_get_x11 (char *type, char **data, unsigned int *size);

int
scrap_put_x11 (char *type, char *data, unsigned int size);

int
scrap_get_types_x11 (char** types);

#endif /* SDL_VIDEO_DRIVER_X11 */

#endif /* _PYGAME_SCRAPX11_H_ */
