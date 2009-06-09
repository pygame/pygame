/*
  pygame - Python Game Library
  Copyright (C) 2000-2001 Pete Shinners, 2008 Marcus von Appen

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
#ifndef _PYGAME_SDLTTFMOD_H_
#define _PYGAME_SDLTTFMOD_H_

#include "pgcompat.h"
#include <SDL.h>

#define PYGAME_SDLTTF_INTERNAL
#define PYGAME_SDLTTFFONT_INTERNAL

#define RENDER_SOLID 0
#define RENDER_SHADED 1
#define RENDER_BLENDED 2

extern PyTypeObject PySDLFont_TTF_Type;
#define PySDLFont_TTF_Check(x) (PyObject_TypeCheck (x, &PySDLFont_TTF_Type))
PyObject* PySDLFont_TTF_New (char *file, int ptsize);
void font_export_capi (void **capi);

#endif /* _PYGAME_SDLTTFMOD_H_ */
