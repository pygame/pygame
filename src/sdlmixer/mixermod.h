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
#ifndef _PYGAME_SDLMIXERMOD_H_
#define _PYGAME_SDLMIXERMOD_H_

#include "pgcompat.h"
#include <SDL_mixer.h>

#define PYGAME_SDLMIXER_INTERNAL
#define PYGAME_SDLMIXERCHUNK_INTERNAL
#define PYGAME_SDLMIXERCHANNEL_INTERNAL
#define PYGAME_SDLMIXERMUSIC_INTERNAL

extern PyTypeObject PyChunk_Type;
#define PyChunk_Check(x) (PyObject_TypeCheck (x, &PyChunk_Type))
PyObject* PyChunk_New (char *filename);
PyObject* PyChunk_NewFromMixerChunk (Mix_Chunk *sample);

extern PyTypeObject PyChannel_Type;
#define PyChannel_Check(x) (PyObject_TypeCheck (x, &PyChannel_Type))
PyObject* PyChannel_New (void);
PyObject* PyChannel_NewFromIndex (int index);

extern PyTypeObject PyMusic_Type;
#define PyMusic_Check(x) (PyObject_TypeCheck (x, &PyMusic_Type))
PyObject* PyMusic_New (char *filename);

void chunk_export_capi (void **capi);
void channel_export_capi (void **capi);
void music_export_capi (void **capi);

#endif /* _PYGAME_SDLMIXERMOD_H_ */
