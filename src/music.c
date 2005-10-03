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
#include "pygamedocs.h"
#include "mixer.h"



static Mix_Music* current_music = NULL;
static Mix_Music* queue_music = NULL;
static int endmusic_event = SDL_NOEVENT;
static Uint64 music_pos = 0;
static long music_pos_time = -1;
static int music_frequency = 0;
static Uint16 music_format = 0;
static int music_channels = 0;


static void mixmusic_callback(void *udata, Uint8 *stream, int len)
{
        if(!Mix_PausedMusic()) {
            music_pos += len;
            music_pos_time = SDL_GetTicks();
        }
}

static void endmusic_callback(void)
{
	if(endmusic_event && SDL_WasInit(SDL_INIT_VIDEO))
	{
		SDL_Event e;
		memset(&e, 0, sizeof(e));
		e.type = endmusic_event;
		SDL_PushEvent(&e);
	}
	if(queue_music)
	{
	    	if(current_music)
		    Mix_FreeMusic(current_music);
	    	current_music = queue_music;
		queue_music = NULL;
	    	Mix_HookMusicFinished(endmusic_callback);
		music_pos = 0;
	    	Mix_PlayMusic(current_music, 0);
	}
	else
	{
	    music_pos_time = -1;
	    Mix_SetPostMix(NULL, NULL);
	}
}






/*music module methods*/


static PyObject* music_play(PyObject* self, PyObject* args)
{
	int loops=0;
        float startpos=0.0;
	int val, volume=0;

	if(!PyArg_ParseTuple(args, "|if", &loops, &startpos))
		return NULL;

	MIXER_INIT_CHECK();
	if(!current_music)
		return RAISE(PyExc_SDLError, "music not loaded");

	Mix_HookMusicFinished(endmusic_callback);
	Mix_SetPostMix(mixmusic_callback, NULL);
	Mix_QuerySpec(&music_frequency, &music_format, &music_channels);
	music_pos = 0;
	music_pos_time = SDL_GetTicks();

#if MIX_MAJOR_VERSION>=1 && MIX_MINOR_VERSION>=2 && MIX_PATCHLEVEL>=3
	volume = Mix_VolumeMusic(-1);
	val = Mix_FadeInMusicPos(current_music, loops, 0, startpos);
	Mix_VolumeMusic(volume);
#else
	if(startpos)
		return RAISE(PyExc_NotImplementedError, "music start position requires SDL_mixer-1.2.4");
	val = Mix_PlayMusic(current_music, loops);
#endif
	if(val == -1)
		return RAISE(PyExc_SDLError, SDL_GetError());

	RETURN_NONE
}


static PyObject* music_get_busy(PyObject* self, PyObject* args)
{
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	MIXER_INIT_CHECK();

	return PyInt_FromLong(Mix_PlayingMusic());
}


static PyObject* music_fadeout(PyObject* self, PyObject* args)
{
	int time;
	if(!PyArg_ParseTuple(args, "i", &time))
		return NULL;

	MIXER_INIT_CHECK();

	Mix_FadeOutMusic(time);
	if(queue_music)
	{
		Mix_FreeMusic(queue_music);
		queue_music = NULL;
	}
	RETURN_NONE
}


static PyObject* music_stop(PyObject* self, PyObject* args)
{
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	MIXER_INIT_CHECK();

	Mix_HaltMusic();
	if(queue_music)
	{
		Mix_FreeMusic(queue_music);
		queue_music = NULL;
	}
	RETURN_NONE
}


static PyObject* music_pause(PyObject* self, PyObject* args)
{
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	MIXER_INIT_CHECK();

	Mix_PauseMusic();
	RETURN_NONE
}


static PyObject* music_unpause(PyObject* self, PyObject* args)
{
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	MIXER_INIT_CHECK();

	Mix_ResumeMusic();
	RETURN_NONE
}


static PyObject* music_rewind(PyObject* self, PyObject* args)
{
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	MIXER_INIT_CHECK();

	Mix_RewindMusic();
	RETURN_NONE
}


static PyObject* music_set_volume(PyObject* self, PyObject* args)
{
	float volume;

	if(!PyArg_ParseTuple(args, "f", &volume))
		return NULL;

	MIXER_INIT_CHECK();

	Mix_VolumeMusic((int)(volume*128));
	RETURN_NONE
}


static PyObject* music_get_volume(PyObject* self, PyObject* args)
{
	int volume;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	MIXER_INIT_CHECK();

	volume = Mix_VolumeMusic(-1);
	return PyFloat_FromDouble(volume / 128.0);
}


static PyObject* music_get_pos(PyObject* self, PyObject* args)
{
	long ticks;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	MIXER_INIT_CHECK();

	if (music_pos_time < 0)
		return PyLong_FromLong(-1);

	ticks = (long)(1000 * music_pos /
		(music_channels * music_frequency * ((music_format & 0xff) >> 3)));
        if(!Mix_PausedMusic())
            ticks += SDL_GetTicks() - music_pos_time;

	return PyInt_FromLong((long)ticks);
}


static PyObject* music_set_endevent(PyObject* self, PyObject* args)
{
	int eventid = SDL_NOEVENT;

	if(!PyArg_ParseTuple(args, "|i", &eventid))
		return NULL;
	endmusic_event = eventid;
	RETURN_NONE;
}


static PyObject* music_get_endevent(PyObject* self, PyObject* args)
{
	if(!PyArg_ParseTuple(args, ""))
		return NULL;
	return PyInt_FromLong(endmusic_event);
}


static PyObject* music_load(PyObject* self, PyObject* args)
{
#if 0
/*argh, can't do it this way, SDL_mixer doesn't support music RWops*/
	char* name = NULL;
        PyObject* file;
	SDL_RWops *rw;
	if(!PyArg_ParseTuple(args, "O", &file))
		return NULL;
	MIXER_INIT_CHECK();
	if(current_music)
	{
		Mix_FreeMusic(current_music);
		current_music = NULL;
	}

	if(PyString_Check(file) || PyUnicode_Check(file))
	{
		if(!PyArg_ParseTuple(args, "s", &name))
			return NULL;
		Py_BEGIN_ALLOW_THREADS
		current_music = Mix_LoadMUS(name);
		Py_END_ALLOW_THREADS
	}
	else
	{
		if(!(rw = RWopsFromPythonThreaded(file)))
			return NULL;
		if(RWopsCheckPythonThreaded(rw))
			current_music = Mix_LoadMUS_RW(rw, 1);
		else
		{
			Py_BEGIN_ALLOW_THREADS
			current_music = Mix_LoadMUS_RW(rw, 1);
			Py_END_ALLOW_THREADS
		}
	}
#else
	char* filename;
	if(!PyArg_ParseTuple(args, "s", &filename))
		return NULL;

	MIXER_INIT_CHECK();

	if(current_music)
	{
		Mix_FreeMusic(current_music);
		current_music = NULL;
	}
	Py_BEGIN_ALLOW_THREADS
	current_music = Mix_LoadMUS(filename);
	Py_END_ALLOW_THREADS
#endif
	if(!current_music)
		return RAISE(PyExc_SDLError, SDL_GetError());
	if(queue_music)
	{
		Mix_FreeMusic(queue_music);
		queue_music = NULL;
	}


	RETURN_NONE;
}


static PyObject* music_queue(PyObject* self, PyObject* args)
{
	char* filename;
	Mix_Music* new_music;
	if(!PyArg_ParseTuple(args, "s", &filename))
		return NULL;

	MIXER_INIT_CHECK();

	Py_BEGIN_ALLOW_THREADS
	new_music = Mix_LoadMUS(filename);
	Py_END_ALLOW_THREADS

	if(!new_music)
		return RAISE(PyExc_SDLError, SDL_GetError());

	if(queue_music)
	{
		Mix_FreeMusic(queue_music);
		queue_music = NULL;
	}
	queue_music = new_music;
    	RETURN_NONE
}



static PyMethodDef music_builtins[] =
{
	{ "set_endevent", music_set_endevent, 1, DOC_PYGAMEMIXERMUSICSETENDEVENT },
	{ "get_endevent", music_get_endevent, 1, DOC_PYGAMEMIXERMUSICGETENDEVENT },

	{ "play", music_play, 1, DOC_PYGAMEMIXERMUSICPLAY },
	{ "get_busy", music_get_busy, 1, DOC_PYGAMEMIXERMUSICGETBUSY },
	{ "fadeout", music_fadeout, 1, DOC_PYGAMEMIXERMUSICFADEOUT },
	{ "stop", music_stop, 1, DOC_PYGAMEMIXERMUSICSTOP },
	{ "pause", music_pause, 1, DOC_PYGAMEMIXERMUSICPAUSE },
	{ "unpause", music_unpause, 1, DOC_PYGAMEMIXERMUSICUNPAUSE },
	{ "rewind", music_rewind, 1, DOC_PYGAMEMIXERMUSICREWIND },
	{ "set_volume", music_set_volume, 1, DOC_PYGAMEMIXERMUSICSETVOLUME },
	{ "get_volume", music_get_volume, 1, DOC_PYGAMEMIXERMUSICGETVOLUME },
	{ "get_pos", music_get_pos, 1, DOC_PYGAMEMIXERMUSICGETPOS },

	{ "load", music_load, 1, DOC_PYGAMEMIXERMUSICLOAD },
	{ "queue", music_queue, 1, DOC_PYGAMEMIXERMUSICQUEUE },

	{ NULL, NULL }
};



PYGAME_EXPORT
void initmixer_music(void)
{
	PyObject *module;

	PyMIXER_C_API[0] = PyMIXER_C_API[0]; /*clean an unused warning*/
        /* create the module */
	module = Py_InitModule3("mixer_music", music_builtins, DOC_PYGAMEMIXERMUSIC);
	PyModule_AddObject(module, "_MUSIC_POINTER", PyCObject_FromVoidPtr(&current_music, NULL));
	PyModule_AddObject(module, "_QUEUE_POINTER", PyCObject_FromVoidPtr(&queue_music, NULL));

	/*imported needed apis*/
	import_pygame_base();
}

