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
	music_pos += len;
	music_pos_time = SDL_GetTicks();
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

    /*DOC*/ static char doc_play[] =
    /*DOC*/    "pygame.mixer.music.play(loops=0, startpos=0.0) -> None\n"
    /*DOC*/    "play the current loaded music\n"
    /*DOC*/    "\n"
    /*DOC*/    "Starts playing the current loaded music. This will restart the\n"
    /*DOC*/    "sound if it is playing. Loops controls how many extra time the\n"
    /*DOC*/    "sound will play, a negative loop will play indefinitely, it\n"
    /*DOC*/    "defaults to 0.\n"
    /*DOC*/    "\n"
    /*DOC*/    "The starting position argument controls where in the music the\n"
    /*DOC*/    "song starts playing. The starting position is dependent on the\n"
    /*DOC*/    "format of music playing. MP3 and OGG use the position as time\n"
    /*DOC*/    "(in seconds). MOD music it is the pattern order number. Passing\n"
    /*DOC*/    "a startpos will raise a NotImplementedError if it cannot set the\n"
    /*DOC*/    "start position (or your version of SDL_mixer is too old)\n"
    /*DOC*/ ;

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



    /*DOC*/ static char doc_get_busy[] =
    /*DOC*/    "pygame.mixer.music.get_busy() -> bool\n"
    /*DOC*/    "query state of the music\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns true if music is currently playing\n"
    /*DOC*/ ;

static PyObject* music_get_busy(PyObject* self, PyObject* args)
{
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	MIXER_INIT_CHECK();

	return PyInt_FromLong(Mix_PlayingMusic());
}


    /*DOC*/ static char doc_fadeout[] =
    /*DOC*/    "pygame.mixer.music.fadeout(millisec) -> None\n"
    /*DOC*/    "fadeout current music\n"
    /*DOC*/    "\n"
    /*DOC*/    "Fades out the current playing music and stops it over the given\n"
    /*DOC*/    "milliseconds.\n"
    /*DOC*/ ;

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


    /*DOC*/ static char doc_stop[] =
    /*DOC*/    "pygame.mixer.music.stop() -> None\n"
    /*DOC*/    "stop the playing music\n"
    /*DOC*/    "\n"
    /*DOC*/    "Stops playback of the current music.\n"
    /*DOC*/ ;

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


    /*DOC*/ static char doc_pause[] =
    /*DOC*/    "pygame.mixer.music.pause() -> None\n"
    /*DOC*/    "pause the playing music\n"
    /*DOC*/    "\n"
    /*DOC*/    "Temporarily stops the current music.\n"
    /*DOC*/ ;

static PyObject* music_pause(PyObject* self, PyObject* args)
{
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	MIXER_INIT_CHECK();

	Mix_PauseMusic();
	RETURN_NONE
}


    /*DOC*/ static char doc_unpause[] =
    /*DOC*/    "pygame.mixer.music.unpause() -> None\n"
    /*DOC*/    "restarts the paused music\n"
    /*DOC*/    "\n"
    /*DOC*/    "Restarts playback of the current music object when paused.\n"
    /*DOC*/ ;

static PyObject* music_unpause(PyObject* self, PyObject* args)
{
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	MIXER_INIT_CHECK();

	Mix_ResumeMusic();
	RETURN_NONE
}


    /*DOC*/ static char doc_rewind[] =
    /*DOC*/    "pygame.mixer.music.rewind() -> None\n"
    /*DOC*/    "restarts music\n"
    /*DOC*/    "\n"
    /*DOC*/    "Restarts playback of the current music.\n"
    /*DOC*/ ;

static PyObject* music_rewind(PyObject* self, PyObject* args)
{
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	MIXER_INIT_CHECK();

	Mix_RewindMusic();
	RETURN_NONE
}


    /*DOC*/ static char doc_set_volume[] =
    /*DOC*/    "pygame.mixer.music.set_volume(val) -> None\n"
    /*DOC*/    "set music volume\n"
    /*DOC*/    "\n"
    /*DOC*/    "Sets the current volume for the music. Value is\n"
    /*DOC*/    "between 0.0 and 1.0.\n"
    /*DOC*/ ;

static PyObject* music_set_volume(PyObject* self, PyObject* args)
{
	float volume;

	if(!PyArg_ParseTuple(args, "f", &volume))
		return NULL;

	MIXER_INIT_CHECK();

	Mix_VolumeMusic((int)(volume*128));
	RETURN_NONE
}


    /*DOC*/ static char doc_get_volume[] =
    /*DOC*/    "pygame.mixer.music.get_volume() -> val\n"
    /*DOC*/    "query music volume\n"
    /*DOC*/    "\n"
    /*DOC*/    "Get the current volume for the music. Value is\n"
    /*DOC*/    "between 0.0 and 1.0.\n"
    /*DOC*/ ;

static PyObject* music_get_volume(PyObject* self, PyObject* args)
{
	int volume;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	MIXER_INIT_CHECK();

	volume = Mix_VolumeMusic(-1);
	return PyFloat_FromDouble(volume / 128.0);
}



    /*DOC*/ static char doc_get_pos[] =
    /*DOC*/    "pygame.mixer.music.get_pos() -> val\n"
    /*DOC*/    "query music position\n"
    /*DOC*/    "\n"
    /*DOC*/    "Get the current (interpolated) time position of the music.\n"
    /*DOC*/    "Value is in ms, just like get_ticks().\n"
    /*DOC*/    "\n"
    /*DOC*/    "The returned time is only tracking the amount of music\n"
    /*DOC*/    "played. It will not reflect the result of starting the\n"
    /*DOC*/    "music at an offset.\n"
    /*DOC*/ ;

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
	ticks += SDL_GetTicks() - music_pos_time;

	return PyInt_FromLong((long)ticks);
}



    /*DOC*/ static char doc_set_endevent[] =
    /*DOC*/    "pygame.mixer.music.set_endevent([eventid]) -> None\n"
    /*DOC*/    "sets music finished event\n"
    /*DOC*/    "\n"
    /*DOC*/    "When the music has finished playing, you can optionally have\n"
    /*DOC*/    "pygame place a user defined message on the event queue. If the\n"
    /*DOC*/    "eventid field is omittied or NOEVENT, no messages will be sent\n"
    /*DOC*/    "when the music finishes playing. Once the endevent is set, it\n"
    /*DOC*/    "will be called every time the music finished playing.\n"
    /*DOC*/ ;

static PyObject* music_set_endevent(PyObject* self, PyObject* args)
{
	int eventid = SDL_NOEVENT;

	if(!PyArg_ParseTuple(args, "|i", &eventid))
		return NULL;
	endmusic_event = eventid;
	RETURN_NONE;
}



    /*DOC*/ static char doc_get_endevent[] =
    /*DOC*/    "pygame.mixer.music.get_endevent([eventid]) -> int\n"
    /*DOC*/    "query the current music finished event\n"
    /*DOC*/    "\n"
    /*DOC*/    "When the music has finished playing, you can optionally have\n"
    /*DOC*/    "pygame place a user defined message on the event queue. If there\n"
    /*DOC*/    "is no callback event set, NOEVENT will be returned. Otherwise it\n"
    /*DOC*/    "will return the id of the current music finishe event.\n"
    /*DOC*/ ;

static PyObject* music_get_endevent(PyObject* self, PyObject* args)
{
	if(!PyArg_ParseTuple(args, ""))
		return NULL;
	return PyInt_FromLong(endmusic_event);
}



    /*DOC*/ static char doc_load[] =
    /*DOC*/    "pygame.mixer.music.load(filename) -> None\n"
    /*DOC*/    "load current music\n"
    /*DOC*/    "\n"
    /*DOC*/    "Load a music object as the current music. The music only handles\n"
    /*DOC*/    "one music as the current. If music is currently playing, it will\n"
    /*DOC*/    "be stopped and replaced with the given one. Loading music only\n"
    /*DOC*/    "supports filenames, not file-like objects.\n"
    /*DOC*/ ;

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




    /*DOC*/ static char doc_queue[] =
    /*DOC*/    "pygame.mixer.music.queue(soundfile) -> None\n"
    /*DOC*/    "preload and queue a music file\n"
    /*DOC*/    "\n"
    /*DOC*/    "This will load a music file and queue it. A queued music file\n"
    /*DOC*/    "will begin as soon as the current music naturally ends. If the\n"
    /*DOC*/    "current music is ever stopped or changed, the queued song will\n"
    /*DOC*/    "be lost.\n"
    /*DOC*/ ;

static PyObject* music_queue(PyObject* self, PyObject* args)
{
	char* filename;
	if(!PyArg_ParseTuple(args, "s", &filename))
		return NULL;

	MIXER_INIT_CHECK();

	if(queue_music)
	{
		Mix_FreeMusic(queue_music);
		queue_music = NULL;
	}
	Py_BEGIN_ALLOW_THREADS
	queue_music = Mix_LoadMUS(filename);
	Py_END_ALLOW_THREADS

    	RETURN_NONE
}



static PyMethodDef music_builtins[] =
{
	{ "set_endevent", music_set_endevent, 1, doc_set_endevent },
	{ "get_endevent", music_get_endevent, 1, doc_get_endevent },

	{ "play", music_play, 1, doc_play },
	{ "get_busy", music_get_busy, 1, doc_get_busy },
	{ "fadeout", music_fadeout, 1, doc_fadeout },
	{ "stop", music_stop, 1, doc_stop },
	{ "pause", music_pause, 1, doc_pause },
	{ "unpause", music_unpause, 1, doc_unpause },
	{ "rewind", music_rewind, 1, doc_rewind },
	{ "set_volume", music_set_volume, 1, doc_set_volume },
	{ "get_volume", music_get_volume, 1, doc_get_volume },
	{ "get_pos", music_get_pos, 1, doc_get_pos },

	{ "load", music_load, 1, doc_load },
	{ "queue", music_queue, 1, doc_queue },

	{ NULL, NULL }
};





    /*DOC*/ static char doc_pygame_mixer_music_MODULE[] =
    /*DOC*/    "The music module is tied closely to the pygame.mixer module. It\n"
    /*DOC*/    "is an optional module since it depends on the SDL_mixer library.\n"
    /*DOC*/    "You should not manually import the music library. Instead it is\n"
    /*DOC*/    "automatically included as a part of the mixer library. The default\n"
    /*DOC*/    "module path to music is pygame.mixer.music.\n"
    /*DOC*/    "\n"
    /*DOC*/    "The difference between playback of music and playback of sounds\n"
    /*DOC*/    "is that the music object is not loaded and decoded all at once,\n"
    /*DOC*/    "instead the music data is streamed and decoded during playback.\n"
    /*DOC*/    "There can only be one music file loaded at a single time. Loading\n"
    /*DOC*/    "a new music file will replace any currently loaded music.\n"
    /*DOC*/    "\n"
    /*DOC*/    "The music module has many of the same types of functions as the\n"
    /*DOC*/    "Sound objects. The main difference is only one music object can\n"
    /*DOC*/    "be loaded at a time, with the load() function. Music\n"
    /*DOC*/    "must be stored in an individual file on the system, it cannot be\n"
    /*DOC*/    "loaded from special file-like objects through python.\n"
    /*DOC*/ ;

PYGAME_EXPORT
void initmixer_music(void)
{
	PyObject *module;

	PyMIXER_C_API[0] = PyMIXER_C_API[0]; /*clean an unused warning*/
        /* create the module */
	module = Py_InitModule3("mixer_music", music_builtins, doc_pygame_mixer_music_MODULE);
	PyModule_AddObject(module, "_MUSIC_POINTER", PyCObject_FromVoidPtr(&current_music, NULL));
	PyModule_AddObject(module, "_QUEUE_POINTER", PyCObject_FromVoidPtr(&queue_music, NULL));

	/*imported needed apis*/
	import_pygame_base();
}

