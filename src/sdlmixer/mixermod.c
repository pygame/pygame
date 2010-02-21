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

#include "pymacros.h"
#include "mixermod.h"
#include "pgmixer.h"
#include "pgsdl.h"
#include "sdlmixerbase_doc.h"

static PyObject* _mixer_init (PyObject *self, PyObject *args);
static PyObject* _mixer_wasinit (PyObject *self);
static PyObject* _mixer_quit (PyObject *self);
static PyObject* _mixer_getcompiledversion (PyObject *self);
static PyObject* _mixer_getversion (PyObject *self);
static PyObject* _mixer_geterror (PyObject *self);
static PyObject* _mixer_openaudio (PyObject *self, PyObject *args);
static PyObject* _mixer_closeaudio (PyObject *self);
static PyObject* _mixer_queryspec (PyObject *self);

static PyMethodDef _mixer_methods[] = {
    { "init", _mixer_init, METH_VARARGS, DOC_BASE_INIT },
    { "was_init", (PyCFunction) _mixer_wasinit, METH_NOARGS,
      DOC_BASE_WAS_INIT },
    { "quit", (PyCFunction) _mixer_quit, METH_NOARGS, DOC_BASE_QUIT },
    { "get_compiled_version", (PyCFunction) _mixer_getcompiledversion,
      METH_NOARGS, DOC_BASE_GET_COMPILED_VERSION },
    { "get_version", (PyCFunction) _mixer_getversion, METH_NOARGS,
      DOC_BASE_GET_VERSION },
    { "get_error", (PyCFunction) _mixer_geterror, METH_NOARGS,
      DOC_BASE_GET_ERROR },
    { "open_audio", _mixer_openaudio, METH_VARARGS, DOC_BASE_OPEN_AUDIO },
    { "close_audio", (PyCFunction) _mixer_closeaudio, METH_NOARGS,
      DOC_BASE_CLOSE_AUDIO },
    { "query_spec", (PyCFunction) _mixer_queryspec, METH_NOARGS,
      DOC_BASE_QUERY_SPEC },
    { NULL, NULL, 0, NULL },
};

static PyObject*
_mixer_init (PyObject *self, PyObject *args)
{
    long flags = 0;
    long retval = 0;

    if (!PyArg_ParseTuple (args, "|l", &flags))
        return NULL;

    if (SDL_WasInit (SDL_INIT_AUDIO))
    {
        retval = Mix_Init (flags);
        return PyLong_FromLong (retval);
    }

    if (SDL_InitSubSystem (SDL_INIT_AUDIO) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    retval = Mix_Init (flags);
    return PyLong_FromLong (retval);
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
    if (SDL_WasInit (SDL_INIT_AUDIO))
        SDL_QuitSubSystem (SDL_INIT_AUDIO);
    Mix_Quit ();
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
    return Text_FromUTF8 (err);
}

static PyObject*
_mixer_openaudio (PyObject *self, PyObject *args)
{
    int freq, chans, chunks, retval;
    Uint16 format;

    if (!PyArg_ParseTuple (args, "iiii:open_audio", &freq, &format, &chans,
            &chunks))
        return NULL;
    
    Py_BEGIN_ALLOW_THREADS;
    retval = Mix_OpenAudio (freq, format, chans, chunks);
    Py_END_ALLOW_THREADS;

    if (retval == -1)
    {
        PyErr_SetString (PyExc_PyGameError, Mix_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_mixer_closeaudio (PyObject *self)
{
    Py_BEGIN_ALLOW_THREADS;
    Mix_CloseAudio ();
    Py_END_ALLOW_THREADS;
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

#ifdef IS_PYTHON_3
PyMODINIT_FUNC PyInit_base (void)
#else
PyMODINIT_FUNC initbase (void)
#endif
{
    PyObject *mod = NULL;
    PyObject *c_api_obj;
    static void *c_api[PYGAME_SDLMIXER_SLOTS];
    
#ifdef IS_PYTHON_3
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "base",
        DOC_BASE,
        -1,
        _mixer_methods,
        NULL, NULL, NULL, NULL
    };
    mod = PyModule_Create (&_module);
#else
    mod = Py_InitModule3 ("base", _mixer_methods, DOC_BASE);
#endif
    if (!mod)
        goto fail;
        
    if (PyType_Ready (&PyChunk_Type) < 0)
        goto fail;
    if (PyType_Ready (&PyChannel_Type) < 0)
        goto fail;
    if (PyType_Ready (&PyMusic_Type) < 0)
        goto fail;
    ADD_OBJ_OR_FAIL (mod, "Chunk", PyChunk_Type, fail);
    ADD_OBJ_OR_FAIL (mod, "Channel", PyChannel_Type, fail);
    ADD_OBJ_OR_FAIL (mod, "Music", PyMusic_Type, fail);

    /* Export C API */
    chunk_export_capi (c_api);
    channel_export_capi (c_api);
    music_export_capi (c_api);

    c_api_obj = PyCObject_FromVoidPtr ((void *) c_api, NULL);
    if (c_api_obj)
    {
        if (PyModule_AddObject (mod, PYGAME_SDLMIXER_ENTRY, c_api_obj) == -1)
        {
            Py_DECREF (c_api_obj);
            goto fail;
        }
    }

    if (import_pygame2_base () < 0)
        goto fail;
    if (import_pygame2_sdl_base () < 0)
        goto fail;
    if (import_pygame2_sdl_rwops () < 0)
        goto fail;
    MODINIT_RETURN(mod);
fail:
    Py_XDECREF (mod);
    MODINIT_RETURN (NULL);
}
