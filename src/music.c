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
static int endmusic_event = SDL_NOEVENT;


static void endmusic_callback(void)
{
	if(endmusic_event && SDL_WasInit(SDL_INIT_VIDEO))
	{
		SDL_Event e;
		memset(&e, 0, sizeof(e));
		e.type = endmusic_event;
		SDL_PushEvent(&e);
	}
}






/*music module methods*/

    /*DOC*/ static char doc_play[] =
    /*DOC*/    "pygame.mixer.music.play([loops]) -> None\n"
    /*DOC*/    "play the current loaded music\n"
    /*DOC*/    "\n"
    /*DOC*/    "Starts playing the current loaded music. This will restart the\n"
    /*DOC*/    "sound if it is playing. Loops controls how many extra time the\n"
    /*DOC*/    "sound will play, a negative loop will play indefinitely, it\n"
    /*DOC*/    "defaults to 0.\n"
    /*DOC*/ ;

static PyObject* play(PyObject* self, PyObject* args)
{
	int loops = 0;
	int val;

	if(!PyArg_ParseTuple(args, "|i", &loops))
		return NULL;

	MIXER_INIT_CHECK();
	if(!current_music)
		return RAISE(PyExc_SDLError, "music not loaded");

	Mix_HookMusicFinished(endmusic_callback);

	val = Mix_PlayMusic(current_music, loops);
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

static PyObject* get_busy(PyObject* self, PyObject* args)
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

static PyObject* fadeout(PyObject* self, PyObject* args)
{
	int time;
	if(!PyArg_ParseTuple(args, "i", &time))
		return NULL;

	MIXER_INIT_CHECK();

	Mix_FadeOutMusic(time);
	RETURN_NONE
}


    /*DOC*/ static char doc_stop[] =
    /*DOC*/    "pygame.mixer.music.stop() -> None\n"
    /*DOC*/    "stop the playing music\n"
    /*DOC*/    "\n"
    /*DOC*/    "Stops playback of the current music.\n"
    /*DOC*/ ;

static PyObject* stop(PyObject* self, PyObject* args)
{
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	MIXER_INIT_CHECK();

	Mix_HaltMusic();
	RETURN_NONE
}


    /*DOC*/ static char doc_pause[] =
    /*DOC*/    "pygame.mixer.music.pause() -> None\n"
    /*DOC*/    "pause the playing music\n"
    /*DOC*/    "\n"
    /*DOC*/    "Temporarily stops the current music.\n"
    /*DOC*/ ;

static PyObject* pause(PyObject* self, PyObject* args)
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

static PyObject* unpause(PyObject* self, PyObject* args)
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

static PyObject* mus_rewind(PyObject* self, PyObject* args)
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

static PyObject* set_volume(PyObject* self, PyObject* args)
{
	float volume;

	if(!PyArg_ParseTuple(args, "f", &volume))
		return NULL;

	MIXER_INIT_CHECK();

	Mix_VolumeMusic((int)(volume*127));
	RETURN_NONE
}


    /*DOC*/ static char doc_get_volume[] =
    /*DOC*/    "pygame.mixer.music.get_volume() -> val\n"
    /*DOC*/    "query music volume\n"
    /*DOC*/    "\n"
    /*DOC*/    "Get the current volume for the music. Value is\n"
    /*DOC*/    "between 0.0 and 1.0.\n"
    /*DOC*/ ;

static PyObject* get_volume(PyObject* self, PyObject* args)
{
	int volume;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	MIXER_INIT_CHECK();

	volume = Mix_VolumeMusic(-1);
	return PyFloat_FromDouble(volume / 127.0);
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

static PyObject* set_endevent(PyObject* self, PyObject* args)
{
	int eventid = SDL_NOEVENT;

	if(!PyArg_ParseTuple(args, "i", &eventid))
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

static PyObject* get_endevent(PyObject* self, PyObject* args)
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

static PyObject* load(PyObject* self, PyObject* args)
{
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
	
	if(!current_music)
		return RAISE(PyExc_SDLError, SDL_GetError());

	RETURN_NONE;
}



static PyMethodDef music_builtins[] =
{
	{ "set_endevent", set_endevent, 1, doc_set_endevent },
	{ "get_endevent", get_endevent, 1, doc_get_endevent },

	{ "play", play, 1, doc_play },
	{ "get_busy", get_busy, 1, doc_get_busy },
	{ "fadeout", fadeout, 1, doc_fadeout },
	{ "stop", stop, 1, doc_stop },
	{ "pause", pause, 1, doc_pause },
	{ "unpause", unpause, 1, doc_unpause },
	{ "rewind", mus_rewind, 1, doc_rewind },
	{ "set_volume", set_volume, 1, doc_set_volume },
	{ "get_volume", get_volume, 1, doc_get_volume },

	{ "load", load, 1, doc_load },

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

	/*imported needed apis*/
	import_pygame_base();
}

