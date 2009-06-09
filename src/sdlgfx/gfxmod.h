/*
  pygame - Python Game Library
  Copyright (C) 2008 Marcus von Appen

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
#ifndef _PYGAME_SDLGFXMOD_H_
#define _PYGAME_SDLGFXMOD_H_

#include "pgcompat.h"
#include <SDL.h>

#define PYGAME_SDLGFXFPS_INTERNAL

extern PyTypeObject PyFPSmanager_Type;
#define PyFPSmanager_Check(x) (PyObject_TypeCheck (x, &PyFPSmanager_Type))
PyObject* PyFPSmanager_New (void);

void fps_export_capi (void **capi);

#endif /* _PYGAME_SDLGFXMOD_H_ */
