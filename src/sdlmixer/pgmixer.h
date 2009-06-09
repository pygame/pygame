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
#ifndef _PYGAME_SDLMIXER_H_
#define _PYGAME_SDLMIXER_H_

#include <SDL.h>
#include <SDL_mixer.h>

#include "pgbase.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ASSERT_MIXER_INIT(x)                                            \
    if (!SDL_WasInit(SDL_INIT_AUDIO))                                   \
    {                                                                   \
        PyErr_SetString(PyExc_PyGameError, "mixer subsystem not initialized"); \
        return (x);                                                     \
    }

#define ASSERT_MIXER_OPEN(x)                                        \
    ASSERT_MIXER_INIT(x);                                           \
    if (!Mix_QuerySpec(NULL, NULL, NULL))                           \
    {                                                               \
        PyErr_SetString(PyExc_PyGameError, "mixer is not open");    \
        return (x);                                                 \
    }


#define PYGAME_SDLMIXER_FIRSTSLOT 0
#define PYGAME_SDLMIXER_NUMSLOTS 0
#ifndef PYGAME_SDLMIXER_INTERNAL
#endif /* PYGAME_SDLMIXER_INTERNAL */

typedef struct
{
    PyObject_HEAD
    Mix_Chunk *chunk;
    int        playchannel;
} PyChunk;
#define PyChunk_AsChunk(x) (((PyChunk*)x)->chunk)
#define PYGAME_SDLMIXERCHUNK_FIRSTSLOT \
    (PYGAME_SDLMIXER_FIRSTSLOT + PYGAME_SDLMIXER_NUMSLOTS)
#define PYGAME_SDLMIXERCHUNK_NUMSLOTS 1
#ifndef PYGAME_SDLMIXERCHUNK_INTERNAL
#define PyChunk_Type \
    (*(PyTypeObject*)PyGameSDLMixer_C_API[PYGAME_SDLMIXERCHUNK_FIRSTSLOT+0])
#define PyChunk_Check(x)                                                   \
    (PyObject_TypeCheck(x,                                              \
        (PyTypeObject*)PyGameSDLMixer_C_API[PYGAME_SDLMIXERCHUNK_FIRSTSLOT+0]))
#define PyChunk_New                                                   \
    (*(PyObject*(*)(char*))PyGameSDLMixer_C_API[PYGAME_SDLMIXERCHUNK_FIRSTSLOT+1])
#define PyChunk_NewFromMixerChunk                                       \
    (*(PyObject*(*)(Mix_Chunk*))PyGameSDLMixer_C_API[PYGAME_SDLMIXERCHUNK_FIRSTSLOT+2])
#endif /* PYGAME_SDLMIXERCHUNK_INTERNAL */

typedef struct
{
    PyObject_HEAD
    int        channel;
    PyObject  *playchunk;
} PyChannel;
#define PyChannel_AsChannel(x) (((PyChannel*)x)->channel)
#define PYGAME_SDLMIXERCHANNEL_FIRSTSLOT \
    (PYGAME_SDLMIXERCHUNK_FIRSTSLOT + PYGAME_SDLMIXERCHUNK_NUMSLOTS)
#define PYGAME_SDLMIXERCHANNEL_NUMSLOTS 3
#ifndef PYGAME_SDLMIXERCHANNEL_INTERNAL
#define PyChannel_Type \
    (*(PyTypeObject*)PyGameSDLMixer_C_API[PYGAME_SDLMIXERCHANNEL_FIRSTSLOT+0])
#define PyChannel_Check(x)                                                   \
    (PyObject_TypeCheck(x,                                              \
        (PyTypeObject*)PyGameSDLMixer_C_API[PYGAME_SDLMIXERCHANNEL_FIRSTSLOT+0]))
#define PyChannel_New                                                   \
    (*(PyObject*(*)(void))PyGameSDLMixer_C_API[PYGAME_SDLMIXERCHANNEL_FIRSTSLOT+1])
#define PyChannel_NewFromIndex                                          \
    (*(PyObject*(*)(int))PyGameSDLMixer_C_API[PYGAME_SDLMIXERCHANNEL_FIRSTSLOT+2])
#endif /* PYGAME_SDLMIXERCHANNEL_INTERNAL */

typedef struct
{
    PyObject_HEAD
    Mix_Music *music;
} PyMusic;
#define PyMusic_AsMusic(x) (((PyMusic*)x)->music)
#define PYGAME_SDLMIXERMUSIC_FIRSTSLOT \
    (PYGAME_SDLMIXERCHANNEL_FIRSTSLOT + PYGAME_SDLMIXERCHANNEL_NUMSLOTS)
#define PYGAME_SDLMIXERMUSIC_NUMSLOTS 2
#ifndef PYGAME_SDLMIXERMUSIC_INTERNAL
#define PyMusic_Type \
    (*(PyTypeObject*)PyGameSDLMixer_C_API[PYGAME_SDLMIXERMUSIC_FIRSTSLOT+0])
#define PyMusic_Check(x)                                                   \
    (PyObject_TypeCheck(x,                                              \
        (PyTypeObject*)PyGameSDLMixer_C_API[PYGAME_SDLMIXERMUSIC_FIRSTSLOT+0]))
#define PyMusic_New                                                   \
    (*(PyObject*(*)(char*))PyGameSDLMixer_C_API[PYGAME_SDLMIXERMUSIC_FIRSTSLOT+1])
#endif /* PYGAME_SDLMIXERMUSIC_INTERNAL */

/**
 * C API export.
 */
#ifdef PYGAME_INTERNAL
void **PyGameSDLMixer_C_API;
#else
static void **PyGameSDLMixer_C_API;
#endif

#define PYGAME_SDLMIXER_SLOTS                                    \
    (PYGAME_SDLMIXERMUSIC_FIRSTSLOT + PYGAME_SDLMIXERMUSIC_NUMSLOTS)
#define PYGAME_SDLMIXER_ENTRY "_PYGAME_SDLMIXER_CAPI"

static int
import_pygame2_sdlmixer_base (void)
{
    PyObject *_module = PyImport_ImportModule ("pygame2.sdlmixer.base");
    if (_module != NULL)
    {
        PyObject *_capi = PyObject_GetAttrString(_module, PYGAME_SDLMIXER_ENTRY);
        if (!PyCObject_Check (_capi))
        {
            Py_DECREF (_module);
            return -1;
        }
        PyGameSDLMixer_C_API = (void**) PyCObject_AsVoidPtr (_capi);
        Py_DECREF (_capi);
        return 0;
    }
    return -1;
}

#ifdef __cplusplus
}
#endif

#endif /* _PYGAME_SDLMIXER_H_ */
