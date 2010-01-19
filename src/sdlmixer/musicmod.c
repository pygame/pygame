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

#include "pgmixer.h"
#include "pgsdl.h"
#include "sdlmixermusic_doc.h"

static PyObject* _music_setvolume (PyObject *self, PyObject *args);
static PyObject* _music_getvolume (PyObject *self);
static PyObject* _music_pause (PyObject *self);
static PyObject* _music_resume (PyObject *self);
static PyObject* _music_halt (PyObject *self);
static PyObject* _music_rewind (PyObject *self);
static PyObject* _music_fadeout (PyObject *self, PyObject *args);
static PyObject* _music_playing (PyObject *self);
static PyObject* _music_paused (PyObject *self);
static PyObject* _music_fading (PyObject *self);
static PyObject* _music_setposition (PyObject *self, PyObject *args);

static PyMethodDef _music_methods[] = {
    { "set_volume", _music_setvolume, METH_O, DOC_MUSIC_SET_VOLUME },
    { "get_volume", (PyCFunction) _music_getvolume, METH_NOARGS,
      DOC_MUSIC_GET_VOLUME },
    { "pause", (PyCFunction) _music_pause, METH_NOARGS, DOC_MUSIC_PAUSE },
    { "resume", (PyCFunction) _music_resume, METH_NOARGS, DOC_MUSIC_RESUME },
    { "halt", (PyCFunction) _music_halt, METH_NOARGS, DOC_MUSIC_HALT },
    { "rewind", (PyCFunction) _music_rewind, METH_NOARGS, DOC_MUSIC_REWIND},
    { "fade_out", _music_fadeout, METH_O, DOC_MUSIC_FADE_OUT },
    { "playing", (PyCFunction) _music_playing, METH_NOARGS, DOC_MUSIC_PLAYING },
    { "paused", (PyCFunction) _music_paused, METH_NOARGS, DOC_MUSIC_PAUSED },
    { "fading", (PyCFunction) _music_fading, METH_NOARGS, DOC_MUSIC_FADING },
    { "set_position", _music_setposition, METH_O, DOC_MUSIC_SET_POSITION },
    { NULL, NULL, 0, NULL },
};

static PyObject*
_music_setvolume (PyObject *self, PyObject *args)
{
    int volume;
    
    ASSERT_MIXER_OPEN (NULL);
    
    if (!IntFromObj (args, &volume))
        return NULL;
    if (volume < 0 || volume > MIX_MAX_VOLUME)
    {
        PyErr_SetString (PyExc_ValueError, "volume must be in the range 0-128");
        return NULL;
    }
    return PyInt_FromLong (Mix_VolumeMusic (volume));
}

static PyObject*
_music_getvolume (PyObject *self)
{
    ASSERT_MIXER_OPEN (NULL);
    return PyInt_FromLong (Mix_VolumeMusic (-1));
}

static PyObject*
_music_pause (PyObject *self)
{
    ASSERT_MIXER_OPEN (NULL);

    if (Mix_PlayingMusic ())
        Mix_PauseMusic ();
    Py_RETURN_NONE;
}

static PyObject*
_music_resume (PyObject *self)
{
    ASSERT_MIXER_OPEN (NULL);

    Mix_ResumeMusic ();
    Py_RETURN_NONE;
}

static PyObject*
_music_halt (PyObject *self)
{
    ASSERT_MIXER_OPEN (NULL);

    Mix_HaltMusic ();
    Py_RETURN_NONE;
}

static PyObject*
_music_rewind (PyObject *self)
{
    ASSERT_MIXER_OPEN (NULL);

    Mix_RewindMusic ();
    Py_RETURN_NONE;
}

static PyObject*
_music_fadeout (PyObject *self, PyObject *args)
{
    int ms;

    ASSERT_MIXER_OPEN (NULL);

    if (!IntFromObj (args, &ms))
        return NULL;

    if (ms < 0)
    {
        PyErr_SetString (PyExc_ValueError, "ms must not be negative");
        return NULL;
    }

    if (!Mix_FadeOutMusic (ms))
    {
        PyErr_SetString (PyExc_PyGameError, Mix_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_music_playing (PyObject *self)
{
    ASSERT_MIXER_OPEN (NULL);
    return PyBool_FromLong (Mix_PlayingMusic ());
}

static PyObject*
_music_paused (PyObject *self)
{
    ASSERT_MIXER_OPEN (NULL);
    return PyBool_FromLong (Mix_PausedMusic ());
}

static PyObject*
_music_fading (PyObject *self)
{
    ASSERT_MIXER_OPEN (NULL);
    return PyLong_FromUnsignedLong (Mix_FadingMusic ());
}

static PyObject*
_music_setposition (PyObject *self, PyObject *args)
{
    double pos;
    
    ASSERT_MIXER_OPEN (NULL);
    
    if (!DoubleFromObj (args, &pos))
        return NULL;
    if (pos < 0)
    {
        PyErr_SetString (PyExc_ValueError, "pos must not be negative");
        return NULL;
    }

    if (Mix_SetMusicPosition (pos) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, Mix_GetError ());
        return NULL;
    }

    Py_RETURN_NONE;
}

#ifdef IS_PYTHON_3
PyMODINIT_FUNC PyInit_music (void)
#else
PyMODINIT_FUNC initmusic (void)
#endif
{
    PyObject *mod;

#ifdef IS_PYTHON_3
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "music",
        DOC_MUSIC,
        -1,
        _music_methods,
         NULL, NULL, NULL, NULL
   };
    mod = PyModule_Create (&_module);
#else
    mod = Py_InitModule3 ("music", _music_methods, DOC_MUSIC);
#endif
    if (!mod)
        goto fail;

    if (import_pygame2_base () < 0)
        goto fail;
    if (import_pygame2_sdl_base () < 0)
        goto fail;
    if (import_pygame2_sdl_rwops () < 0)
        goto fail;
    if (import_pygame2_sdlmixer_base () < 0)
        goto fail;
    MODINIT_RETURN(mod);
fail:
    Py_XDECREF (mod);
    MODINIT_RETURN (NULL);
}
