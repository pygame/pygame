/*
    PyGame - Python Game Library
    Copyright (C) 2000  Pete Shinners

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
 *  mixer module for PyGAME
 */
#define PYGAMEAPI_MUSIC_INTERNAL
#include "pygame.h"
#include "mixer.h"



static Mix_Music* current_music = NULL;



static void autoquit()
{
	if(SDL_WasInit(SDL_INIT_AUDIO))
	{
		if(current_music)
		{
			Mix_FreeMusic(current_music);
			current_music = NULL;
		}
	}
	PyMixer_AutoQuit();
}


static PyObject* autoinit(PyObject* self, PyObject* arg)
{
	return PyMixer_AutoInit(self, arg);
}


    /*DOC*/ static char doc_quit[] =
    /*DOC*/    "pygame.music.quit() -> None\n"
    /*DOC*/    "uninitialize music module\n"
    /*DOC*/    "\n"
    /*DOC*/    "Stops playback of any music and uninitializes the\n"
    /*DOC*/    "module.\n"
    /*DOC*/ ;

static PyObject* quit(PyObject* self, PyObject* arg)
{
	if(!PyArg_ParseTuple(arg, ""))
		return NULL;

	autoquit();

	RETURN_NONE
}


    /*DOC*/ static char doc_init[] =
    /*DOC*/    "pygame.music.init([freq, [size, [stereo]]]) -> None\n"
    /*DOC*/    "initialize the music module\n"
    /*DOC*/    "\n"
    /*DOC*/    "Initializes the music module. Since the music\n"
    /*DOC*/    "requires use of the mixer, the mixer module will\n"
    /*DOC*/    "also be initialized with this call. See the mixer\n"
    /*DOC*/    "init function for more details on the arguments.\n"
    /*DOC*/    "Don't be fooled though, just because the mixer\n"
    /*DOC*/    "module is initialized, does not mean the music is\n"
    /*DOC*/    "initialized.\n"
    /*DOC*/ ;

static PyObject* init(PyObject* self, PyObject* arg)
{
	PyObject* result;
	int value;

	result = autoinit(self, arg);
	value = PyObject_IsTrue(result);
	Py_DECREF(result);
	if(!value)
		return RAISE(PyExc_SDLError, SDL_GetError());

	RETURN_NONE
}


    /*DOC*/ static char doc_get_init[] =
    /*DOC*/    "pygame.music.get_init() -> bool\n"
    /*DOC*/    "query the music module\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns true if the music module is initialized.\n"
    /*DOC*/ ;

static PyObject* get_init(PyObject* self, PyObject* arg)
{
	if(!PyArg_ParseTuple(arg, ""))
		return NULL;

	return PyInt_FromLong(SDL_WasInit(SDL_INIT_AUDIO));
}






/*music module methods*/

    /*DOC*/ static char doc_play[] =
    /*DOC*/    "pygame.music.play([loops]) -> None\n"
    /*DOC*/    "play the current loaded music\n"
    /*DOC*/    "\n"
    /*DOC*/    "Starts playing the current loaded music. This will\n"
    /*DOC*/    "restart the sound if it is playing. Loops controls\n"
    /*DOC*/    "how many extra time the sound will play, a\n"
    /*DOC*/    "negative loop will play indefinitely, it defaults\n"
    /*DOC*/    "to 0.\n"
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

	val = Mix_PlayMusic(current_music, loops);
	if(val == -1)
		return RAISE(PyExc_SDLError, SDL_GetError());

	RETURN_NONE
}


    /*DOC*/ static char doc_get_busy[] =
    /*DOC*/    "pygame.music.get_busy() -> bool\n"
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
    /*DOC*/    "pygame.music.fadeout(millisec) -> None\n"
    /*DOC*/    "fadeout current music\n"
    /*DOC*/    "\n"
    /*DOC*/    "Fades out the current playing music and stops it\n"
    /*DOC*/    "over the given milliseconds.\n"
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
    /*DOC*/    "pygame.music.stop() -> None\n"
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
    /*DOC*/    "pygame.music.pause() -> None\n"
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
    /*DOC*/    "pygame.music.unpause() -> None\n"
    /*DOC*/    "restarts the paused music\n"
    /*DOC*/    "\n"
    /*DOC*/    "Restarts playback of the current music object when\n"
    /*DOC*/    "paused.\n"
    /*DOC*/ ;

static PyObject* unpause(PyObject* self, PyObject* args)
{
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	MIXER_INIT_CHECK();

	Mix_RewindMusic();
	RETURN_NONE
}


    /*DOC*/ static char doc_rewind[] =
    /*DOC*/    "pygame.music.rewind() -> None\n"
    /*DOC*/    "restarts music\n"
    /*DOC*/    "\n"
    /*DOC*/    "Restarts playback of the current music.\n"
    /*DOC*/ ;

static PyObject* mus_rewind(PyObject* self, PyObject* args)
{
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	MIXER_INIT_CHECK();

	Mix_ResumeMusic();
	RETURN_NONE
}


    /*DOC*/ static char doc_set_volume[] =
    /*DOC*/    "pygame.music.set_volume(val) -> None\n"
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
    /*DOC*/    "pygame.music.get_volume() -> val\n"
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




    /*DOC*/ static char doc_load[] =
    /*DOC*/    "pygame.music.load(filename) -> None\n"
    /*DOC*/    "load current music\n"
    /*DOC*/    "\n"
    /*DOC*/    "Load a music object as the current music. The\n"
    /*DOC*/    "music only handles one music as the current. If\n"
    /*DOC*/    "music is currently playing, it will be stopped and\n"
    /*DOC*/    "replaced with the given one. Loading music only\n"
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

	current_music = Mix_LoadMUS(filename);
	if(!current_music)
		return RAISE(PyExc_SDLError, SDL_GetError());

	RETURN_NONE;
}



static PyMethodDef music_builtins[] =
{
	{ "__PYGAMEinit__", autoinit, 1, doc_init },
	{ "init", init, 1, doc_init },
	{ "quit", quit, 1, doc_quit },
	{ "get_init", get_init, 1, doc_get_init },

/*	{ "music_endevent", music_endevent, 1, doc_music_endevent },*/

	{ "play", play, 1, doc_play },
	{ "get_busy", get_busy, 1, doc_get_busy },
	{ "fadeout", fadeout, 1, doc_fadeout },
	{ "stop", stop, 1, doc_stop },
	{ "pause", pause, 1, doc_pause },
	{ "unpause", unpause, 1, doc_unpause },
	{ "rewind", mus_rewind, 1, doc_rewind },
	{ "set", set_volume, 1, doc_set_volume },
	{ "get", get_volume, 1, doc_get_volume },

	{ "load", load, 1, doc_load },

	{ NULL, NULL }
};





    /*DOC*/ static char doc_pygame_music_MODULE[] =
    /*DOC*/    "Contains music routines\n"
    /*DOC*/ ;

void initmusic()
{
	PyObject *module, *dict;

    /* create the module */
	module = Py_InitModule3("music", music_builtins, doc_pygame_music_MODULE);
	dict = PyModule_GetDict(module);

	/*imported needed apis*/
	import_pygame_base();
	import_pygame_mixer();
}

