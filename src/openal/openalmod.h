/*
  pygame - Python Game Library
  Copyright (C) 2010 Marcus von Appen

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
#ifndef _PYGAME_OPENALMOD_H_
#define _PYGAME_OPENALMOD_H_

#define PYGAME_OPENAL_INTERNAL
#define PYGAME_OPENALDEVICE_INTERNAL
#define PYGAME_OPENALCONTEXT_INTERNAL
#define PYGAME_OPENALBUFFERS_INTERNAL

#if defined(IS_MSYS) || defined(IS_WIN32)
#include <al.h>
#include <alc.h>
#else
#include <AL/al.h>
#include <AL/alc.h>
#endif

#include "pgcompat.h"

extern PyTypeObject PyDevice_Type;
#define PyDevice_Check(x) (PyObject_TypeCheck (x, &PyDevice_Type))
PyObject* PyDevice_New (const char *name);

extern PyTypeObject PyContext_Type;
#define PyContext_Check(x) (PyObject_TypeCheck (x, &PyContext_Type))

extern PyTypeObject PyBuffers_Type;
#define PyBuffers_Check(x) (PyObject_TypeCheck (x, &PyBuffers_Type))
PyObject* PyBuffers_New (ALsizei count);

int SetALErrorException (ALenum error);
int SetALCErrorException (ALCenum error);
#define CLEAR_ERROR_STATE() alGetError()

void device_export_capi (void **capi);
void context_export_capi (void **capi);
void buffers_export_capi (void **capi);

#endif /* _PYGAME_OPENALMOD_H_ */
