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

*/

#define PYGAME_SDLMIXERCONSTANTS_INTERNAL

#include "pgmixer.h"

/* macros used to create each constant */
#define DEC_CONSTMIX(x)  PyModule_AddIntConstant(module, #x, (int) MIX_##x)
#define DEC_CONST(x)  PyModule_AddIntConstant(module, #x, (int) SDL_##x)
#define DEC_CONSTK(x) PyModule_AddIntConstant(module, #x, (int) SDL##x)
#define DEC_CONSTN(x) PyModule_AddIntConstant(module, #x, (int) x)
#define DEC_CONSTS(x,y) PyModule_AddIntConstant(module, #x, (int) y)
#define ADD_STRING_CONST(x) PyModule_AddStringConstant(module, #x, x)

#ifdef IS_PYTHON_3
PyMODINIT_FUNC PyInit_constants (void)
#else
PyMODINIT_FUNC initconstants (void)
#endif
{
    PyObject *module;

#ifdef IS_PYTHON_3
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "constants",
        "Pygame SDL Mixer constants",
        -1,
        NULL,
        NULL, NULL, NULL, NULL
    };
    module = PyModule_Create (&_module);
#else
    module = Py_InitModule3 ("constants", NULL, "Pygame SDL Mixer constants");
#endif
    if (!module)
        goto fail;

    DEC_CONSTN(AUDIO_U8);
    DEC_CONSTN(AUDIO_S8);
    DEC_CONSTN(AUDIO_U16LSB);
    DEC_CONSTN(AUDIO_S16LSB);
    DEC_CONSTN(AUDIO_U16MSB);
    DEC_CONSTN(AUDIO_S16MSB);
    DEC_CONSTN(AUDIO_U16);
    DEC_CONSTN(AUDIO_S16);
    DEC_CONSTN(AUDIO_U16SYS);
    DEC_CONSTN(AUDIO_S16SYS);
    
    DEC_CONSTMIX(FADING_IN);
    DEC_CONSTMIX(FADING_OUT);
    DEC_CONSTMIX(NO_FADING);

    DEC_CONSTMIX(INIT_FLAC);
    DEC_CONSTMIX(INIT_MOD);
    DEC_CONSTMIX(INIT_MP3);
    DEC_CONSTMIX(INIT_OGG);

    DEC_CONSTN(MUS_NONE);
    DEC_CONSTN(MUS_CMD);
    DEC_CONSTN(MUS_WAV);
    DEC_CONSTN(MUS_MOD);
    DEC_CONSTN(MUS_MID);
    DEC_CONSTN(MUS_OGG);
    DEC_CONSTN(MUS_MP3);

#ifdef MUS_MP3_MAD /* Not all SDL_mixer versions might use libmad */
    DEC_CONSTN(MUS_MP3_MAD);
#endif

    DEC_CONSTMIX(CHANNELS);
    DEC_CONSTMIX(DEFAULT_FREQUENCY);
    DEC_CONSTMIX(DEFAULT_FORMAT);
    DEC_CONSTMIX(DEFAULT_CHANNELS);
    DEC_CONSTMIX(MAX_VOLUME);
    
    MODINIT_RETURN(module);
fail:
    Py_XDECREF (module);
    MODINIT_RETURN (NULL);
}
