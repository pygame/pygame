/*
    pygame - Python Game Library
    Copyright (C) 2000-2001  Pete Shinners

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

    Pete Shinners
    pete@shinners.org
*/

#ifndef PGMIXER_H
#define PGMIXER_H

#include <Python.h>
#include <SDL_mixer.h>
#include <structmember.h>


/* test mixer initializations */
#define MIXER_INIT_CHECK() \
    if(!SDL_WasInit(SDL_INIT_AUDIO)) \
        return RAISE(pgExc_SDLError, "mixer not initialized")

#include "pgimport.h"

#define PYGAMEAPI_MIXER_FIRSTSLOT 0
#define PYGAMEAPI_MIXER_NUMSLOTS 7
typedef struct {
  PyObject_HEAD
  Mix_Chunk *chunk;
  Uint8 *mem;
  PyObject *weakreflist;
} pgSoundObject;
typedef struct {
  PyObject_HEAD
  int chan;
} pgChannelObject;
#define pgSound_AsChunk(x) (((pgSoundObject*)x)->chunk)
#define pgChannel_AsInt(x) (((pgChannelObject*)x)->chan)

#ifndef PYGAMEAPI_MIXER_INTERNAL

PYGAMEAPI_DEFINE_SLOTS(pgMIXER_C_API, PYGAMEAPI_MIXER_NUMSLOTS);

#define pgSound_Check(x) ((x)->ob_type == (PyTypeObject*)pgMIXER_C_API[0])
#define pgSound_Type (*(PyTypeObject*)pgMIXER_C_API[0])
#define pgSound_New (*(PyObject*(*)(Mix_Chunk*))pgMIXER_C_API[1])
#define pgSound_Play (*(PyObject*(*)(PyObject*, PyObject*))pgMIXER_C_API[2])
#define pgChannel_Check(x) ((x)->ob_type == (PyTypeObject*)pgMIXER_C_API[3])
#define pgChannel_Type (*(PyTypeObject*)pgMIXER_C_API[3])
#define pgChannel_New (*(PyObject*(*)(int))pgMIXER_C_API[4])
#define pgMixer_AutoInit (*(PyObject*(*)(PyObject*, PyObject*))pgMIXER_C_API[5])
#define pgMixer_AutoQuit (*(void(*)(void))pgMIXER_C_API[6])

#define import_pygame_mixer() \
    _IMPORT_PYGAME_MODULE(mixer, MIXER, pgMIXER_C_API)

#endif /* PYGAMEAPI_MIXER_INTERNAL */

#endif /* ~PGMIXER_H */
