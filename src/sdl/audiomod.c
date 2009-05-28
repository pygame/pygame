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
#define PYGAME_SDLAUDIO_INTERNAL

#include "pgsdl.h"
#include "sdlaudio_doc.h"

static PyObject* _sdl_audioinit (PyObject *self);
static PyObject* _sdl_audiowasinit (PyObject *self);
static PyObject* _sdl_audioquit (PyObject *self);

static PyMethodDef _audio_methods[] = {
    { "init", (PyCFunction) _sdl_audioinit, METH_NOARGS, DOC_AUDIO_INIT },
    { "was_init", (PyCFunction) _sdl_audiowasinit, METH_NOARGS,
      DOC_AUDIO_WAS_INIT },
    { "quit", (PyCFunction) _sdl_audioquit, METH_NOARGS, DOC_AUDIO_QUIT },
    { NULL, NULL, 0, NULL }
};

static PyObject*
_sdl_audioinit (PyObject *self)
{
    if (SDL_WasInit (SDL_INIT_AUDIO))
        Py_RETURN_NONE;
        
    if (SDL_InitSubSystem (SDL_INIT_AUDIO) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_sdl_audiowasinit (PyObject *self)
{
    if (SDL_WasInit (SDL_INIT_AUDIO))
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject*
_sdl_audioquit (PyObject *self)
{
    if (SDL_WasInit (SDL_INIT_AUDIO))
        SDL_QuitSubSystem (SDL_INIT_AUDIO);
    Py_RETURN_NONE;
}

#ifdef IS_PYTHON_3
PyMODINIT_FUNC PyInit_audio (void)
#else
PyMODINIT_FUNC initaudio (void)
#endif
{
    PyObject *mod;

#ifdef IS_PYTHON_3
    static struct PyModuleDef _audiomodule = {
        PyModuleDef_HEAD_INIT, "audio", DOC_AUDIO, -1, _audio_methods,
        NULL, NULL, NULL, NULL
    };
    mod = PyModule_Create (&_audiomodule);
#else
    mod = Py_InitModule3 ("audio", _audio_methods, DOC_AUDIO);
#endif
    if (!mod)
        goto fail;

    if (import_pygame2_base () < 0)
        goto fail;
    if (import_pygame2_sdl_base () < 0)
        goto fail;
    
    MODINIT_RETURN(mod);
fail:
    Py_XDECREF (mod);
    MODINIT_RETURN (NULL);
}
