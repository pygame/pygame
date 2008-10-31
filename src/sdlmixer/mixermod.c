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
#define PYGAME_SDLMIXER_INTERNAL

#include "mixermod.h"
#include "pgmixer.h"
#include "pgsdl.h"

static void _quit (void);

static PyObject* _mixer_init (PyObject *self);
static PyObject* _mixer_wasinit (PyObject *self);
static PyObject* _mixer_quit (PyObject *self);
static PyObject* _mixer_getcompiledversion (PyObject *self);
static PyObject* _mixer_getversion (PyObject *self);
static PyObject* _mixer_geterror (PyObject *self);
static PyObject* _mixer_openaudio (PyObject *self, PyObject *args);
static PyObject* _mixer_closeaudio (PyObject *self);
static PyObject* _mixer_queryspec (PyObject *self);

static PyMethodDef _mixer_methods[] = {
    { "init", (PyCFunction) _mixer_init, METH_NOARGS, "" },
    { "was_init", (PyCFunction) _mixer_wasinit, METH_NOARGS, "" },
    { "quit", (PyCFunction) _mixer_quit, METH_NOARGS, "" },
    { "get_compiled_version", (PyCFunction) _mixer_getcompiledversion,
      METH_NOARGS, "" },
    { "get_version", (PyCFunction) _mixer_getversion, METH_NOARGS, "" },
    { "get_error", (PyCFunction) _mixer_geterror, METH_NOARGS, "" },
    { "open_audio", _mixer_openaudio, METH_VARARGS, "" },
    { "close_audio", (PyCFunction) _mixer_closeaudio, METH_NOARGS, "" },
    { "query_spec", (PyCFunction) _mixer_queryspec, METH_NOARGS, "" },
    { NULL, NULL, 0, NULL },
};

static void
_quit (void)
{
    if (SDL_WasInit (SDL_INIT_AUDIO))
        SDL_QuitSubSystem (SDL_INIT_AUDIO);
}

static PyObject*
_mixer_init (PyObject *self)
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
_mixer_wasinit (PyObject *self)
{
    if (SDL_WasInit (SDL_INIT_AUDIO))
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject*
_mixer_quit (PyObject *self)
{
    _quit ();
    Py_RETURN_NONE;
}

static PyObject*
_mixer_getcompiledversion (PyObject *self)
{
    SDL_version compiled;
    MIX_VERSION (&compiled);
    return Py_BuildValue ("(iii)", compiled.major, compiled.minor,
        compiled.patch);
}

static PyObject*
_mixer_getversion (PyObject *self)
{
    const SDL_version *linked = Mix_Linked_Version ();
    return Py_BuildValue ("(iii)", linked->major, linked->minor, linked->patch);
}

static PyObject*
_mixer_geterror (PyObject *self)
{
    char *err = Mix_GetError ();
    if (!err)
        Py_RETURN_NONE;
    return PyString_FromString (err);
}

static PyObject*
_mixer_openaudio (PyObject *self, PyObject *args)
{
    int freq, chans, chunks;
    Uint16 format;

    if (!PyArg_ParseTuple (args, "iiii:open_audio", &freq, &format, &chans,
            &chunks))
        return NULL;

    if (Mix_OpenAudio (freq, format, chans, chunks) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, Mix_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_mixer_closeaudio (PyObject *self)
{
    Mix_CloseAudio ();
    Py_RETURN_NONE;
}

static PyObject*
_mixer_queryspec (PyObject *self)
{
    int freq, chans, times;
    Uint16 format;

    times = Mix_QuerySpec (&freq, &format, &chans);
    if (!times)
    {
        PyErr_SetString (PyExc_PyGameError, Mix_GetError ());
        return NULL;
    }
    return Py_BuildValue ("(iiii)", times, freq, format, chans);
}

#if PY_VERSION_HEX >= 0x03000000
PyMODINIT_FUNC PyInit_base (void)
#else
PyMODINIT_FUNC initbase (void)
#endif
{
    PyObject *mod = NULL;
    PyObject *c_api_obj;
    static void *c_api[PYGAME_SDLMIXER_SLOTS];
    
#if PY_VERSION_HEX >= 0x03000000
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "base",
        "",
        -1,
        _mixer_methods,
        NULL, NULL, NULL, NULL
    };
#endif
    PyChunk_Type.tp_new = &PyType_GenericNew;
    if (PyType_Ready (&PyChunk_Type) < 0)
        goto fail;
    PyChannel_Type.tp_new = &PyType_GenericNew;
    if (PyType_Ready (&PyChannel_Type) < 0)
        goto fail;
    PyMusic_Type.tp_new = &PyType_GenericNew;
    if (PyType_Ready (&PyMusic_Type) < 0)
        goto fail;

    Py_INCREF (&PyChunk_Type);
    Py_INCREF (&PyChannel_Type);
    Py_INCREF (&PyMusic_Type);


#if PY_VERSION_HEX < 0x03000000
    mod = Py_InitModule3 ("base", _mixer_methods, "");
#else
    mod = PyModule_Create (&_module);
#endif
    if (!mod)
        goto fail;
        
    PyModule_AddObject (mod, "Chunk", (PyObject *) &PyChunk_Type);
    PyModule_AddObject (mod, "Channel", (PyObject *) &PyChannel_Type);
    PyModule_AddObject (mod, "Music", (PyObject *) &PyMusic_Type);

    /* Export C API */
    chunk_export_capi (c_api);

    c_api_obj = PyCObject_FromVoidPtr ((void *) c_api, NULL);
    if (c_api_obj)
        PyModule_AddObject (mod, PYGAME_SDLMIXER_ENTRY, c_api_obj);    

    if (import_pygame2_base () < 0)
        goto fail;
    if (import_pygame2_sdl_base () < 0)
        goto fail;
    RegisterQuitCallback (_quit);
    MODINIT_RETURN(mod);
fail:
    Py_XDECREF (mod);
    MODINIT_RETURN (NULL);
}
