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
#ifndef _PYGAME_SDLVIDEO_H_
#define _PYGAME_SDLVIDEO_H_

#include "pgcompat.h"
#include <SDL.h>

#define PYGAME_SDLSURFACE_INTERNAL
#define PYGAME_SDLOVERLAY_INTERNAL
#define PYGAME_SDLPXFMT_INTERNAL
#define PYGAME_SDLVIDEO_INTERNAL

extern PyTypeObject PySDLSurface_Type;
#define PySDLSurface_Check(x) (PyObject_TypeCheck (x, &PySDLSurface_Type))
PyObject* PySDLSurface_New (int w, int h);
PyObject* PySDLSurface_NewFromSDLSurface (SDL_Surface *sf);
PyObject* PySDLSurface_Copy (PyObject *source);
int PySDLSurface_AddRefLock (PyObject *surface, PyObject *lock);
int PySDLSurface_RemoveRefLock (PyObject *surface, PyObject *lock);
PyObject* PySDLSurface_AcquireLockObj (PyObject *surface, PyObject *lock);

extern PyTypeObject PyOverlay_Type;
#define PyOverlay_Check(x) (PyObject_TypeCheck (x, &PyOverlay_Type))
PyObject* PyOverlay_New  (PyObject *surface, int width, int height,
    Uint32 format);
int PyOverlay_AddRefLock (PyObject *overlay, PyObject *lock);
int PyOverlay_RemoveRefLock (PyObject *overlay, PyObject *lock);

extern PyTypeObject PyPixelFormat_Type;
#define PyPixelFormat_Check(x) (PyObject_TypeCheck (x, &PyPixelFormat_Type))
PyObject* PyPixelFormat_New (void);
PyObject* PyPixelFormat_NewFromSDLPixelFormat (SDL_PixelFormat *fmt);

int SDLColorFromObj (PyObject *value, SDL_PixelFormat *format, Uint32 *color);

void surface_export_capi (void **capi);
void overlay_export_capi (void **capi);
void pixelformat_export_capi (void **capi);

#endif /* _PYGAME_SDLVIDEO_H_ */
