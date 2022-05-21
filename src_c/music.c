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
static int queue_music_loops = 0;
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
_pg_push_music_event(int type)
{
    pgEventObject *e;
    SDL_Event event;
    PyGILState_STATE gstate = PyGILState_Ensure();

    e = (pgEventObject *)pgEvent_New2(type, NULL);
    if (e) {
        pgEvent_FillUserEvent(e, &event);
        if (SDL_PushEvent(&event) <= 0)
            Py_DECREF(e->dict);
        Py_DECREF(e);
    }
    PyGILState_Release(gstate);
}

static void
endmusic_callback(void)
{
    if (endmusic_event && SDL_WasInit(SDL_INIT_VIDEO))
        _pg_push_music_event(endmusic_event);

    if (queue_music) {
        if (current_music)
            Mix_FreeMusic(current_music);
        current_music = queue_music;
        queue_music = NULL;
        Mix_HookMusicFinished(endmusic_callback);
        music_pos = 0;
        Mix_PlayMusic(current_music, queue_music_loops);
        queue_music_loops = 0;
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
    int val, volume = 0, fade_ms = 0;

    static char *kwids[] = {"loops", "start", "fade_ms", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "|ifi", kwids, &loops,
                                     &startpos, &fade_ms))
        return NULL;

    MIXER_INIT_CHECK();
    if (!current_music)
        return RAISE(pgExc_SDLError, "music not loaded");

    Py_BEGIN_ALLOW_THREADS;
    Mix_HookMusicFinished(endmusic_callback);
    Mix_SetPostMix(mixmusic_callback, NULL);
    Mix_QuerySpec(&music_frequency, &music_format, &music_channels);
    music_pos = 0;
    music_pos_time = SDL_GetTicks();

    volume = Mix_VolumeMusic(-1);
    val = Mix_FadeInMusicPos(current_music, loops, fade_ms, startpos);
    Mix_VolumeMusic(volume);
    Py_END_ALLOW_THREADS;
    if (val == -1)
        return RAISE(pgExc_SDLError, SDL_GetError());

    Py_RETURN_NONE;
}

static PyObject *
music_get_busy(PyObject *self, PyObject *_null)
{
    int playing;

    MIXER_INIT_CHECK();

    Py_BEGIN_ALLOW_THREADS;
    playing = (Mix_PlayingMusic() && !Mix_PausedMusic());
    Py_END_ALLOW_THREADS;

    return PyBool_FromLong(playing);
}

static PyObject *
music_fadeout(PyObject *self, PyObject *args)
{
    int _time;
    if (!PyArg_ParseTuple(args, "i", &_time))
        return NULL;

    MIXER_INIT_CHECK();

    Py_BEGIN_ALLOW_THREADS;
    /* To prevent the queue_music from playing, free it before fading. */
    if (queue_music) {
        Mix_FreeMusic(queue_music);
        queue_music = NULL;
        queue_music_loops = 0;
    }

    Mix_FadeOutMusic(_time);

    Py_END_ALLOW_THREADS;
    Py_RETURN_NONE;
}

static PyObject *
music_stop(PyObject *self, PyObject *_null)
{
    MIXER_INIT_CHECK();

    Py_BEGIN_ALLOW_THREADS;
    /* To prevent the queue_music from playing, free it before stopping. */
    if (queue_music) {
        Mix_FreeMusic(queue_music);
        queue_music = NULL;
        queue_music_loops = 0;
    }

    Mix_HaltMusic();

    Py_END_ALLOW_THREADS;
    Py_RETURN_NONE;
}

static PyObject *
music_pause(PyObject *self, PyObject *_null)
{
    MIXER_INIT_CHECK();

    Mix_PauseMusic();
    Py_RETURN_NONE;
}

static PyObject *
music_unpause(PyObject *self, PyObject *_null)
{
    MIXER_INIT_CHECK();

    Mix_ResumeMusic();
    /* need to set pos_time for the adjusted time spent paused*/
    music_pos_time = SDL_GetTicks();
    Py_RETURN_NONE;
}

static PyObject *
music_rewind(PyObject *self, PyObject *_null)
{
    MIXER_INIT_CHECK();

    Py_BEGIN_ALLOW_THREADS;
    Mix_RewindMusic();
    Py_END_ALLOW_THREADS;

    Py_RETURN_NONE;
}

static PyObject *
music_set_volume(PyObject *self, PyObject *args)
{
    float volume;

    if (!PyArg_ParseTuple(args, "f", &volume))
        return NULL;

    MIXER_INIT_CHECK();

    Py_BEGIN_ALLOW_THREADS;
    Mix_VolumeMusic((int)(volume * 128));
    Py_END_ALLOW_THREADS;

    Py_RETURN_NONE;
}

static PyObject *
music_get_volume(PyObject *self, PyObject *_null)
{
    int volume;
    MIXER_INIT_CHECK();

    volume = Mix_VolumeMusic(-1);
    return PyFloat_FromDouble(volume / 128.0);
}

static PyObject *
music_set_pos(PyObject *self, PyObject *arg)
{
    int position_set;
    double pos = PyFloat_AsDouble(arg);
    if (pos == -1 && PyErr_Occurred()) {
        PyErr_Clear();
        return RAISE(PyExc_TypeError, "set_pos expects 1 float argument");
    }

    MIXER_INIT_CHECK();

    Py_BEGIN_ALLOW_THREADS;
    position_set = Mix_SetMusicPosition(pos);
    Py_END_ALLOW_THREADS;

    if (position_set == -1)
        return RAISE(pgExc_SDLError, "set_pos unsupported for this codec");

    Py_RETURN_NONE;
}

static PyObject *
music_get_pos(PyObject *self, PyObject *_null)
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

    return PyLong_FromLong((long)ticks);
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
music_get_endevent(PyObject *self, PyObject *_null)
{
    return PyLong_FromLong(endmusic_event);
}

Mix_MusicType
_get_type_from_hint(char *namehint)
{
    Mix_MusicType type = MUS_NONE;
    char *dot;

    // Adjusts namehint into a mere file extension component
    if (namehint != NULL) {
        dot = strrchr(namehint, '.');
        if (dot != NULL) {
            namehint = dot + 1;
        }
    }
    else {
        return type;
    }

    /* Copied almost directly from SDL_mixer. Originally meant to check file
     * extensions to get a hint of what music type it should be.
     * https://github.com/libsdl-org/SDL_mixer/blob/master/src/music.c#L586-L631
     */
    if (SDL_strcasecmp(namehint, "WAV") == 0) {
        type = MUS_WAV;
    }
    else if (SDL_strcasecmp(namehint, "MID") == 0 ||
             SDL_strcasecmp(namehint, "MIDI") == 0 ||
             SDL_strcasecmp(namehint, "KAR") == 0) {
        type = MUS_MID;
    }
    else if (SDL_strcasecmp(namehint, "OGG") == 0) {
        type = MUS_OGG;
    }
#ifdef MUS_OPUS
    else if (SDL_strcasecmp(namehint, "OPUS") == 0) {
        type = MUS_OPUS;
    }
#endif
    else if (SDL_strcasecmp(namehint, "FLAC") == 0) {
        type = MUS_FLAC;
    }
    else if (SDL_strcasecmp(namehint, "MPG") == 0 ||
             SDL_strcasecmp(namehint, "MPEG") == 0 ||
             SDL_strcasecmp(namehint, "MP3") == 0 ||
             SDL_strcasecmp(namehint, "MAD") == 0) {
        type = MUS_MP3;
    }
    else if (SDL_strcasecmp(namehint, "669") == 0 ||
             SDL_strcasecmp(namehint, "AMF") == 0 ||
             SDL_strcasecmp(namehint, "AMS") == 0 ||
             SDL_strcasecmp(namehint, "DBM") == 0 ||
             SDL_strcasecmp(namehint, "DSM") == 0 ||
             SDL_strcasecmp(namehint, "FAR") == 0 ||
             SDL_strcasecmp(namehint, "IT") == 0 ||
             SDL_strcasecmp(namehint, "MED") == 0 ||
             SDL_strcasecmp(namehint, "MDL") == 0 ||
             SDL_strcasecmp(namehint, "MOD") == 0 ||
             SDL_strcasecmp(namehint, "MOL") == 0 ||
             SDL_strcasecmp(namehint, "MTM") == 0 ||
             SDL_strcasecmp(namehint, "NST") == 0 ||
             SDL_strcasecmp(namehint, "OKT") == 0 ||
             SDL_strcasecmp(namehint, "PTM") == 0 ||
             SDL_strcasecmp(namehint, "S3M") == 0 ||
             SDL_strcasecmp(namehint, "STM") == 0 ||
             SDL_strcasecmp(namehint, "ULT") == 0 ||
             SDL_strcasecmp(namehint, "UMX") == 0 ||
             SDL_strcasecmp(namehint, "WOW") == 0 ||
             SDL_strcasecmp(namehint, "XM") == 0) {
        type = MUS_MOD;
    }
    return type;
}

Mix_Music *
_load_music(PyObject *obj, char *namehint)
{
    Mix_Music *new_music = NULL;
    char *ext = NULL;
    SDL_RWops *rw = NULL;
    PyObject *_type = NULL;
    PyObject *error = NULL;
    PyObject *_traceback = NULL;

    MIXER_INIT_CHECK();

    rw = pgRWops_FromObject(obj);
    if (rw ==
        NULL) { /* stop on NULL, error already set is what we SHOULD do */
        PyErr_Fetch(&_type, &error, &_traceback);
        PyErr_SetObject(pgExc_SDLError, error);
        Py_XDECREF(_type);
        Py_XDECREF(_traceback);
        return NULL;
    }
    if (namehint) {
        ext = namehint;
    }
    else {
        ext = pgRWops_GetFileExtension(rw);
    }

    Py_BEGIN_ALLOW_THREADS;
    new_music = Mix_LoadMUSType_RW(rw, _get_type_from_hint(ext), SDL_TRUE);
    Py_END_ALLOW_THREADS;

    if (!new_music) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return NULL;
    }

    return new_music;
}

static PyObject *
music_load(PyObject *self, PyObject *args, PyObject *keywds)
{
    Mix_Music *new_music = NULL;
    PyObject *obj;
    char *namehint = NULL;
    static char *kwids[] = {"filename", "namehint", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|s", kwids, &obj,
                                     &namehint))
        return NULL;

    MIXER_INIT_CHECK();

    new_music = _load_music(obj, namehint);
    if (new_music == NULL)  // meaning it has an error to return
        return NULL;

    Py_BEGIN_ALLOW_THREADS;
    if (current_music != NULL) {
        Mix_FreeMusic(current_music);
        current_music = NULL;
    }
    if (queue_music != NULL) {
        Mix_FreeMusic(queue_music);
        queue_music = NULL;
        queue_music_loops = 0;
    }
    Py_END_ALLOW_THREADS;

    current_music = new_music;
    Py_RETURN_NONE;
}

static PyObject *
music_unload(PyObject *self, PyObject *_null)
{
    MIXER_INIT_CHECK();

    Py_BEGIN_ALLOW_THREADS;
    if (current_music) {
        Mix_FreeMusic(current_music);
        current_music = NULL;
    }
    if (queue_music) {
        Mix_FreeMusic(queue_music);
        queue_music = NULL;
        queue_music_loops = 0;
    }
    Py_END_ALLOW_THREADS;

    Py_RETURN_NONE;
}

static PyObject *
music_queue(PyObject *self, PyObject *args, PyObject *keywds)
{
    Mix_Music *local_queue_music = NULL;
    PyObject *obj;
    int loops = 0;
    char *namehint = NULL;
    static char *kwids[] = {"filename", "namehint", "loops", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|si", kwids, &obj,
                                     &namehint, &loops))
        return NULL;

    MIXER_INIT_CHECK();

    queue_music_loops = loops;

    local_queue_music = _load_music(obj, namehint);
    if (local_queue_music == NULL)  // meaning it has an error to return
        return NULL;

    Py_BEGIN_ALLOW_THREADS;
    /* Free any existing queued music. */
    if (queue_music != NULL) {
        Mix_FreeMusic(queue_music);
    }
    Py_END_ALLOW_THREADS;

    queue_music = local_queue_music;

    Py_RETURN_NONE;
}

static PyMethodDef _music_methods[] = {
    {"set_endevent", music_set_endevent, METH_VARARGS,
     DOC_PYGAMEMIXERMUSICSETENDEVENT},
    {"get_endevent", music_get_endevent, METH_NOARGS,
     DOC_PYGAMEMIXERMUSICGETENDEVENT},

    {"play", (PyCFunction)music_play, METH_VARARGS | METH_KEYWORDS,
     DOC_PYGAMEMIXERMUSICPLAY},
    {"get_busy", music_get_busy, METH_NOARGS, DOC_PYGAMEMIXERMUSICGETBUSY},
    {"fadeout", music_fadeout, METH_VARARGS, DOC_PYGAMEMIXERMUSICFADEOUT},
    {"stop", music_stop, METH_NOARGS, DOC_PYGAMEMIXERMUSICSTOP},
    {"pause", music_pause, METH_NOARGS, DOC_PYGAMEMIXERMUSICPAUSE},
    {"unpause", music_unpause, METH_NOARGS, DOC_PYGAMEMIXERMUSICUNPAUSE},
    {"rewind", music_rewind, METH_NOARGS, DOC_PYGAMEMIXERMUSICREWIND},
    {"set_volume", music_set_volume, METH_VARARGS,
     DOC_PYGAMEMIXERMUSICSETVOLUME},
    {"get_volume", music_get_volume, METH_NOARGS,
     DOC_PYGAMEMIXERMUSICGETVOLUME},
    {"set_pos", music_set_pos, METH_O, DOC_PYGAMEMIXERMUSICSETPOS},
    {"get_pos", music_get_pos, METH_NOARGS, DOC_PYGAMEMIXERMUSICGETPOS},

    {"load", (PyCFunction)music_load, METH_VARARGS | METH_KEYWORDS,
     DOC_PYGAMEMIXERMUSICLOAD},
    {"unload", music_unload, METH_NOARGS, DOC_PYGAMEMIXERMUSICUNLOAD},
    {"queue", (PyCFunction)music_queue, METH_VARARGS | METH_KEYWORDS,
     DOC_PYGAMEMIXERMUSICQUEUE},

    {NULL, NULL, 0, NULL}};

MODINIT_DEFINE(mixer_music)
{
    PyObject *module;
    PyObject *cobj;

    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "mixer_music",
                                         DOC_PYGAMEMIXERMUSIC,
                                         -1,
                                         _music_methods,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL};

    /* imported needed apis; Do this first so if there is an error
       the module is not loaded.
    */
    import_pygame_base();
    if (PyErr_Occurred()) {
        return NULL;
    }
    import_pygame_rwobject();
    if (PyErr_Occurred()) {
        return NULL;
    }
    import_pygame_event();
    if (PyErr_Occurred()) {
        return NULL;
    }

    /* create the module */
    module = PyModule_Create(&_module);
    if (module == NULL) {
        return NULL;
    }
    cobj = PyCapsule_New(&current_music, "pygame.music_mixer._MUSIC_POINTER",
                         NULL);
    if (PyModule_AddObject(module, "_MUSIC_POINTER", cobj)) {
        Py_XDECREF(cobj);
        Py_DECREF(module);
        return NULL;
    }
    cobj =
        PyCapsule_New(&queue_music, "pygame.music_mixer._QUEUE_POINTER", NULL);
    if (PyModule_AddObject(module, "_QUEUE_POINTER", cobj)) {
        Py_XDECREF(cobj);
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
