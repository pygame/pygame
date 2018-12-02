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

/*
 *  music module for pygame
 */
#define PYGAMEAPI_MUSIC_INTERNAL
#include "pygame.h"

#include "pgcompat.h"

#include "doc/music_doc.h"

#include "mixer.h"

static Mix_Music *current_music = NULL;
static Mix_Music *queue_music = NULL;
static int endmusic_event = SDL_NOEVENT;
static Uint64 music_pos = 0;
static long music_pos_time = -1;
static int music_frequency = 0;
static Uint16 music_format = 0;
static int music_channels = 0;

static void
mixmusic_callback(void *udata, Uint8 *stream, int len)
{
    if (!Mix_PausedMusic()) {
        music_pos += len;
        music_pos_time = SDL_GetTicks();
    }
}

static void
endmusic_callback(void)
{
    if (endmusic_event && SDL_WasInit(SDL_INIT_VIDEO)) {
        SDL_Event e;
        memset(&e, 0, sizeof(e));
        e.type = endmusic_event;
        SDL_PushEvent(&e);
    }
    if (queue_music) {
        if (current_music)
            Mix_FreeMusic(current_music);
        current_music = queue_music;
        queue_music = NULL;
        Mix_HookMusicFinished(endmusic_callback);
        music_pos = 0;
        Mix_PlayMusic(current_music, 0);
    }
    else {
        music_pos_time = -1;
        Mix_SetPostMix(NULL, NULL);
    }
}

/*music module methods*/
static PyObject *
music_play(PyObject *self, PyObject *args, PyObject *keywds)
{
    int loops = 0;
    float startpos = 0.0;
    int val, volume = 0;

    static char *kwids[] = {"loops", "start", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "|if", kwids, &loops,
                                     &startpos))
        return NULL;

    MIXER_INIT_CHECK();
    if (!current_music)
        return RAISE(pgExc_SDLError, "music not loaded");

    Py_BEGIN_ALLOW_THREADS
    Mix_HookMusicFinished(endmusic_callback);
    Mix_SetPostMix(mixmusic_callback, NULL);
    Mix_QuerySpec(&music_frequency, &music_format, &music_channels);
    music_pos = 0;
    music_pos_time = SDL_GetTicks();

    volume = Mix_VolumeMusic(-1);
    val = Mix_FadeInMusicPos(current_music, loops, 0, startpos);
    Mix_VolumeMusic(volume);
    Py_END_ALLOW_THREADS
        if (val == -1) return RAISE(pgExc_SDLError, SDL_GetError());

    Py_RETURN_NONE;
}

static PyObject *
music_get_busy(PyObject *self)
{
    MIXER_INIT_CHECK();
    return PyInt_FromLong(Mix_PlayingMusic());
}

static PyObject *
music_fadeout(PyObject *self, PyObject *args)
{
    int _time;
    if (!PyArg_ParseTuple(args, "i", &_time))
        return NULL;

    MIXER_INIT_CHECK();

    Py_BEGIN_ALLOW_THREADS
    Mix_FadeOutMusic(_time);
    if (queue_music) {
        Mix_FreeMusic(queue_music);
        queue_music = NULL;
    }
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

static PyObject *
music_stop(PyObject *self)
{
    MIXER_INIT_CHECK();

    Py_BEGIN_ALLOW_THREADS
    Mix_HaltMusic();
    if (queue_music) {
        Mix_FreeMusic(queue_music);
        queue_music = NULL;
    }
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

static PyObject *
music_pause(PyObject *self)
{
    MIXER_INIT_CHECK();

    Mix_PauseMusic();
    Py_RETURN_NONE;
}

static PyObject *
music_unpause(PyObject *self)
{
    MIXER_INIT_CHECK();

    Mix_ResumeMusic();
    Py_RETURN_NONE;
}

static PyObject *
music_rewind(PyObject *self)
{
    MIXER_INIT_CHECK();

    Mix_RewindMusic();
    Py_RETURN_NONE;
}

static PyObject *
music_set_volume(PyObject *self, PyObject *args)
{
    float volume;

    if (!PyArg_ParseTuple(args, "f", &volume))
        return NULL;

    MIXER_INIT_CHECK();

    Mix_VolumeMusic((int)(volume * 128));
    Py_RETURN_NONE;
}

static PyObject *
music_get_volume(PyObject *self)
{
    int volume;
    MIXER_INIT_CHECK();

    volume = Mix_VolumeMusic(-1);
    return PyFloat_FromDouble(volume / 128.0);
}

static PyObject *
music_set_pos(PyObject *self, PyObject *arg)
{
    double pos = PyFloat_AsDouble(arg);
    if (pos == -1 && PyErr_Occurred()) {
        PyErr_Clear();
        return RAISE(PyExc_TypeError, "set_pos expects 1 float argument");
    }

    MIXER_INIT_CHECK();

    if (Mix_SetMusicPosition(pos) == -1) {
        return RAISE(pgExc_SDLError, "set_pos unsupported for this codec");
    }
    Py_RETURN_NONE;
}

static PyObject *
music_get_pos(PyObject *self)
{
    long ticks;

    MIXER_INIT_CHECK();

    if (music_pos_time < 0)
        return PyLong_FromLong(-1);

    ticks = (long)(1000 * music_pos /
                   (music_channels * music_frequency *
                    ((music_format & 0xff) >> 3)));
    if (!Mix_PausedMusic())
        ticks += SDL_GetTicks() - music_pos_time;

    return PyInt_FromLong((long)ticks);
}

static PyObject *
music_set_endevent(PyObject *self, PyObject *args)
{
    int eventid = SDL_NOEVENT;

    if (!PyArg_ParseTuple(args, "|i", &eventid))
        return NULL;
    endmusic_event = eventid;
    Py_RETURN_NONE;
}

static PyObject *
music_get_endevent(PyObject *self)
{
    return PyInt_FromLong(endmusic_event);
}

static PyObject *
music_load(PyObject *self, PyObject *args)
{
    PyObject *obj;
    PyObject *oencoded;
    Mix_Music *new_music = NULL;
    const char *name;
    SDL_RWops *rw;

    if (!PyArg_ParseTuple(args, "O", &obj)) {
        return NULL;
    }

    MIXER_INIT_CHECK();

    oencoded = pgRWopsEncodeString(obj, "UTF-8", NULL, pgExc_SDLError);
    if (oencoded == Py_None) {
        Py_DECREF(oencoded);
        rw = pgRWopsFromFileObjectThreaded(obj);
        if (rw == NULL) {
            return NULL;
        }
        Py_BEGIN_ALLOW_THREADS
#if IS_SDLv1
            new_music = Mix_LoadMUS_RW(rw);
#else  /* IS_SDLv2 */
            new_music = Mix_LoadMUS_RW(rw, SDL_TRUE);
#endif /* IS_SDLv2 */
        Py_END_ALLOW_THREADS
    }
    else if (oencoded != NULL) {
        name = Bytes_AS_STRING(oencoded);
        Py_BEGIN_ALLOW_THREADS new_music = Mix_LoadMUS(name);
        Py_END_ALLOW_THREADS Py_DECREF(oencoded);
    }
    else {
        return NULL;
    }

    if (new_music == NULL) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    Py_BEGIN_ALLOW_THREADS if (current_music != NULL)
    {
        Mix_FreeMusic(current_music);
        current_music = NULL;
    }
    if (queue_music != NULL) {
        Mix_FreeMusic(queue_music);
        queue_music = NULL;
    }
    Py_END_ALLOW_THREADS

        current_music = new_music;
    Py_RETURN_NONE;
}

static PyObject *
music_queue(PyObject *self, PyObject *args)
{
    PyObject *obj;
    PyObject *oencoded;
    Mix_Music *queue_music = NULL;
    const char *name;
    SDL_RWops *rw;

    if (!PyArg_ParseTuple(args, "O", &obj)) {
        return NULL;
    }

    MIXER_INIT_CHECK();

    oencoded = pgRWopsEncodeString(obj, "UTF-8", NULL, pgExc_SDLError);
    if (oencoded == Py_None) {
        Py_DECREF(oencoded);
        rw = pgRWopsFromFileObjectThreaded(obj);
        if (rw == NULL) {
            return NULL;
        }
        Py_BEGIN_ALLOW_THREADS
#if IS_SDLv1
            queue_music = Mix_LoadMUS_RW(rw);
#else  /* IS_SDLv2 */
            queue_music = Mix_LoadMUS_RW(rw, SDL_TRUE);
#endif /* IS_SDLv2 */
        Py_END_ALLOW_THREADS
    }
    else if (oencoded != NULL) {
        name = Bytes_AS_STRING(oencoded);
        Py_BEGIN_ALLOW_THREADS queue_music = Mix_LoadMUS(name);
        Py_END_ALLOW_THREADS Py_DECREF(oencoded);
    }
    else {
        return NULL;
    }

    if (queue_music == NULL) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    Py_BEGIN_ALLOW_THREADS if (queue_music == NULL)
    {
        Mix_FreeMusic(queue_music);
        queue_music = NULL;
    }
    Py_END_ALLOW_THREADS

        queue_music = queue_music;
    Py_RETURN_NONE;
}

static PyMethodDef _music_methods[] = {
    {"set_endevent", music_set_endevent, METH_VARARGS,
     DOC_PYGAMEMIXERMUSICSETENDEVENT},
    {"get_endevent", (PyCFunction)music_get_endevent, METH_NOARGS,
     DOC_PYGAMEMIXERMUSICGETENDEVENT},

    {"play", (PyCFunction)music_play, METH_VARARGS | METH_KEYWORDS,
     DOC_PYGAMEMIXERMUSICPLAY},
    {"get_busy", (PyCFunction)music_get_busy, METH_NOARGS,
     DOC_PYGAMEMIXERMUSICGETBUSY},
    {"fadeout", music_fadeout, METH_VARARGS, DOC_PYGAMEMIXERMUSICFADEOUT},
    {"stop", (PyCFunction)music_stop, METH_NOARGS, DOC_PYGAMEMIXERMUSICSTOP},
    {"pause", (PyCFunction)music_pause, METH_NOARGS,
     DOC_PYGAMEMIXERMUSICPAUSE},
    {"unpause", (PyCFunction)music_unpause, METH_NOARGS,
     DOC_PYGAMEMIXERMUSICUNPAUSE},
    {"rewind", (PyCFunction)music_rewind, METH_NOARGS,
     DOC_PYGAMEMIXERMUSICREWIND},
    {"set_volume", music_set_volume, METH_VARARGS,
     DOC_PYGAMEMIXERMUSICSETVOLUME},
    {"get_volume", (PyCFunction)music_get_volume, METH_NOARGS,
     DOC_PYGAMEMIXERMUSICGETVOLUME},
    {"set_pos", (PyCFunction)music_set_pos, METH_O,
     DOC_PYGAMEMIXERMUSICSETPOS},
    {"get_pos", (PyCFunction)music_get_pos, METH_NOARGS,
     DOC_PYGAMEMIXERMUSICGETPOS},

    {"load", music_load, METH_VARARGS, DOC_PYGAMEMIXERMUSICLOAD},
    {"queue", music_queue, METH_VARARGS, DOC_PYGAMEMIXERMUSICQUEUE},

    {NULL, NULL, 0, NULL}};

MODINIT_DEFINE(mixer_music)
{
    PyObject *module;
    PyObject *cobj;

#if PY3
    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "mixer_music",
                                         DOC_PYGAMEMIXERMUSIC,
                                         -1,
                                         _music_methods,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL};
#endif

    pgMIXER_C_API[0] = pgMIXER_C_API[0]; /*clean an unused warning*/

    /* imported needed apis; Do this first so if there is an error
       the module is not loaded.
    */
    import_pygame_base();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }
    import_pygame_rwobject();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }

    /* create the module */
#if PY3
    module = PyModule_Create(&_module);
#else
    module = Py_InitModule3(MODPREFIX "mixer_music", _music_methods,
                            DOC_PYGAMEMIXERMUSIC);
#endif
    if (module == NULL) {
        MODINIT_ERROR;
    }
    cobj = PyCapsule_New(&current_music, "pygame.music_mixer._MUSIC_POINTER",
                         NULL);
    if (cobj == NULL) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    if (PyModule_AddObject(module, "_MUSIC_POINTER", cobj) < 0) {
        Py_DECREF(cobj);
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    cobj =
        PyCapsule_New(&queue_music, "pygame.music_mixer._QUEUE_POINTER", NULL);
    if (cobj == NULL) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    if (PyModule_AddObject(module, "_QUEUE_POINTER", cobj) < 0) {
        Py_DECREF(cobj);
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    MODINIT_RETURN(module);
}
