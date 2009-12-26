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

#ifndef _PYGAME_SCRAPWIN_H_
#define _PYGAME_SCRAPWIN_H_

#include "scrap.h"

#if defined(SDL_VIDEO_DRIVER_WINDIB) || defined(SDL_VIDEO_DRIVER_DDRAW) || defined(SDL_VIDEO_DRIVER_GAPI)

int
scrap_init_win (void);

void
scrap_quit_win (void);

int
scrap_contains_win (char *type);

int
scrap_lost_win (void);

ScrapType
scrap_get_mode_win (void);

ScrapType
scrap_set_mode_win (ScrapType mode);

int
scrap_get_win (char *type, char **data, unsigned int *size);

int
scrap_put_win (char *type, char *data, unsigned int size);

int
scrap_get_types_win (char** types);

#endif /* SDL_VIDEO_DRIVER_WINDIB || SDL_VIDEO_DRIVER_DDRAW || SDL_VIDEO_DRIVER_GAPI */

#endif /* _PYGAME_SCRAPWIN_H_ */
