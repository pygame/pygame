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

#define PYGAMEAPI_CDROM_INTERNAL
#include "pygame.h"


#define CDROM_MAXDRIVES 32
static SDL_CD* cdrom_drivedata[CDROM_MAXDRIVES] = {NULL};


staticforward PyTypeObject PyCD_Type;
static PyObject* PyCD_New(int id);
#define PyCD_Check(x) ((x)->ob_type == &PyCD_Type)



static void cdrom_autoquit(void)
{
	int loop;
	for(loop = 0; loop < CDROM_MAXDRIVES; ++loop)
	{
		if(cdrom_drivedata[loop])
		{
			SDL_CDClose(cdrom_drivedata[loop]);
			cdrom_drivedata[loop] = NULL;
		}
	}

	if(SDL_WasInit(SDL_INIT_CDROM))
		SDL_QuitSubSystem(SDL_INIT_CDROM);
}

static PyObject* cdrom_autoinit(PyObject* self, PyObject* arg)
{
	if(!PyArg_ParseTuple(arg, ""))
		return NULL;

	if(!SDL_WasInit(SDL_INIT_CDROM))
	{
		if(SDL_InitSubSystem(SDL_INIT_CDROM))
			return PyInt_FromLong(0);
		PyGame_RegisterQuit(cdrom_autoquit);
	}
	return PyInt_FromLong(1);
}


    /*DOC*/ static char doc_cdrom_quit[] =
    /*DOC*/    "pygame.cdrom.quit() -> None\n"
    /*DOC*/    "uninitialize the cdrom subsystem\n"
    /*DOC*/    "\n"
    /*DOC*/    "Uninitialize the CDROM module manually\n"
    /*DOC*/ ;

static PyObject* cdrom_quit(PyObject* self, PyObject* arg)
{
	if(!PyArg_ParseTuple(arg, ""))
		return NULL;

	cdrom_autoquit();

	RETURN_NONE
}


    /*DOC*/ static char doc_cdrom_init[] =
    /*DOC*/    "pygame.cdrom.init() -> None\n"
    /*DOC*/    "initialize the cdrom subsystem\n"
    /*DOC*/    "\n"
    /*DOC*/    "Initialize the CDROM module manually\n"
    /*DOC*/ ;

static PyObject* cdrom_init(PyObject* self, PyObject* arg)
{
	PyObject* result;
	int istrue;

	if(!PyArg_ParseTuple(arg, ""))
		return NULL;

	result = cdrom_autoinit(self, arg);
	istrue = PyObject_IsTrue(result);
	Py_DECREF(result);
	if(!istrue)
		return RAISE(PyExc_SDLError, SDL_GetError());

	RETURN_NONE
}



    /*DOC*/ static char doc_get_init[] =
    /*DOC*/    "pygame.cdrom.get_init() -> bool\n"
    /*DOC*/    "query init of cdrom module\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns true of the cdrom module is initialized\n"
    /*DOC*/ ;

static PyObject* get_init(PyObject* self, PyObject* arg)
{
	if(!PyArg_ParseTuple(arg, ""))
		return NULL;

	return PyInt_FromLong(SDL_WasInit(SDL_INIT_CDROM)!=0);
}


static void cd_dealloc(PyObject* self)
{
	PyObject_DEL(self);
}



    /*DOC*/ static char doc_CD[] =
    /*DOC*/    "pygame.cdrom.CD(id) -> CD\n"
    /*DOC*/    "create new CD object\n"
    /*DOC*/    "\n"
    /*DOC*/    "Creates a new CD object for the given CDROM id. The given id\n"
    /*DOC*/    "must be less than the value from pygame.cdrom.get_count().\n"
    /*DOC*/ ;

static PyObject* CD(PyObject* self, PyObject* args)
{
	int id;
	if(!PyArg_ParseTuple(args, "i", &id))
		return NULL;

	CDROM_INIT_CHECK();

	return PyCD_New(id);
}



    /*DOC*/ static char doc_get_count[] =
    /*DOC*/    "pygame.cdrom.get_count() -> int\n"
    /*DOC*/    "query number of cdroms on system\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the number of CDROM drives available on\n"
    /*DOC*/    "the system.\n"
    /*DOC*/ ;

static PyObject* get_count(PyObject* self, PyObject* args)
{
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	CDROM_INIT_CHECK();

	return PyInt_FromLong(SDL_CDNumDrives());
}




    /*DOC*/ static char doc_cd_init[] =
    /*DOC*/    "CD.init() -> None\n"
    /*DOC*/    "initialize a cdrom device for use\n"
    /*DOC*/    "\n"
    /*DOC*/    "In order to call most members in the CD object, the\n"
    /*DOC*/    "CD must be initialized. You can initialzie the CD object\n"
    /*DOC*/    "at anytime, and it is ok to initialize more than once.\n"
    /*DOC*/ ;

static PyObject* cd_init(PyObject* self, PyObject* args)
{
	int cd_id = PyCD_AsID(self);

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	CDROM_INIT_CHECK();
	if(!cdrom_drivedata[cd_id])
	{
		cdrom_drivedata[cd_id] = SDL_CDOpen(cd_id);
                if(!cdrom_drivedata[cd_id])
			return RAISE(PyExc_SDLError, "Cannot initialize device");
	}
	RETURN_NONE
}


    /*DOC*/ static char doc_cd_quit[] =
    /*DOC*/    "CD.quit() -> None\n"
    /*DOC*/    "uninitialize a cdrom device for use\n"
    /*DOC*/    "\n"
    /*DOC*/    "After you are completely finished with a cdrom device, you\n"
    /*DOC*/    "can use this quit() function to free access to the drive.\n"
    /*DOC*/    "This will be cleaned up automatically when the cdrom module is.\n"
    /*DOC*/    "uninitialized. It is safe to call this function on an uninitialized CD.\n"
    /*DOC*/ ;

static PyObject* cd_quit(PyObject* self, PyObject* args)
{
	int cd_id = PyCD_AsID(self);

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	CDROM_INIT_CHECK();

	if(cdrom_drivedata[cd_id])
	{
		SDL_CDClose(cdrom_drivedata[cd_id]);
		cdrom_drivedata[cd_id] = NULL;
	}
	RETURN_NONE
}



    /*DOC*/ static char doc_cd_get_init[] =
    /*DOC*/    "CD.get_init() -> bool\n"
    /*DOC*/    "check if cd is initialized\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns a true value if the CD is initialized.\n"
    /*DOC*/ ;

static PyObject* cd_get_init(PyObject* self, PyObject* args)
{
	int cd_id = PyCD_AsID(self);

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	return PyInt_FromLong(cdrom_drivedata[cd_id] != NULL);
}




    /*DOC*/ static char doc_cd_play[] =
    /*DOC*/    "CD.play(track, [start, end]) -> None\n"
    /*DOC*/    "play music from cdrom\n"
    /*DOC*/    "\n"
    /*DOC*/    "Play an audio track on a cdrom disk.\n"
    /*DOC*/    "You may also optionally pass a starting and ending\n"
    /*DOC*/    "time to play of the song. If you pass the start and\n"
    /*DOC*/    "end time in seconds, only that portion of the audio\n"
    /*DOC*/    "track will be played. If you only provide a start time\n"
    /*DOC*/    "and no end time, this will play to the end of the track.\n"
    /*DOC*/    "You can also pass 'None' as the ending time, and it will\n"
    /*DOC*/    "play to the end of the cd.\n"
    /*DOC*/ ;

static PyObject* cd_play(PyObject* self, PyObject* args)
{
	int cd_id = PyCD_AsID(self);
	SDL_CD* cdrom = cdrom_drivedata[cd_id];
	int result, track, startframe, numframes, playforever=0;
	float start=0.0f, end=0.0f;
	PyObject *endobject=NULL;

	if(!PyArg_ParseTuple(args, "i|fO", &track, &start, &endobject))
	    return NULL;
	if(endobject == Py_None)
	    playforever = 1;
	else if(!PyArg_ParseTuple(args, "i|ff", &track, &start, &end))
	    return NULL;

	CDROM_INIT_CHECK();
	if(!cdrom)
		return RAISE(PyExc_SDLError, "CD drive not initialized");
	SDL_CDStatus(cdrom);
	if(track < 0 || track >= cdrom->numtracks)
		return RAISE(PyExc_IndexError, "Invalid track number");
	if(cdrom->track[track].type != SDL_AUDIO_TRACK)
		return RAISE(PyExc_SDLError, "CD track type is not audio");
	
	/*validate times*/
	if(playforever)
	    end = start;
	else if(start == end && start != 0.0f)
	    RETURN_NONE;
	
	startframe = (int)(start * CD_FPS);
	numframes = 0;
	if(startframe < 0)
		startframe = 0;
	if(end)
		numframes = (int)((end-start) * CD_FPS);
	else
		numframes = cdrom->track[track].length - startframe;
	if(numframes < 0 || startframe > (int)(cdrom->track[track].length * CD_FPS))
		RETURN_NONE;

	result = SDL_CDPlayTracks(cdrom, track, startframe, 0, numframes);
	if(result == -1)
		return RAISE(PyExc_SDLError, SDL_GetError());

	RETURN_NONE
}



    /*DOC*/ static char doc_cd_pause[] =
    /*DOC*/    "CD.pause() -> None\n"
    /*DOC*/    "pause playing cdrom\n"
    /*DOC*/    "\n"
    /*DOC*/    "Pauses the playing CD. If the CD is not playing, this will\n"
    /*DOC*/    "do nothing.\n"
    /*DOC*/ ;

static PyObject* cd_pause(PyObject* self, PyObject* args)
{
	int cd_id = PyCD_AsID(self);
	SDL_CD* cdrom = cdrom_drivedata[cd_id];
	int result;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	CDROM_INIT_CHECK();
	if(!cdrom)
		return RAISE(PyExc_SDLError, "CD drive not initialized");

	result = SDL_CDPause(cdrom);
	if(result == -1)
		return RAISE(PyExc_SDLError, SDL_GetError());

	RETURN_NONE
}



    /*DOC*/ static char doc_cd_resume[] =
    /*DOC*/    "CD.resume() -> int\n"
    /*DOC*/    "resume paused cdrom\n"
    /*DOC*/    "\n"
    /*DOC*/    "Resumes playback of a paused CD. If the CD has not been\n"
    /*DOC*/    "pause, this will do nothing.\n"
    /*DOC*/ ;

static PyObject* cd_resume(PyObject* self, PyObject* args)
{
	int cd_id = PyCD_AsID(self);
	SDL_CD* cdrom = cdrom_drivedata[cd_id];
	int result;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	CDROM_INIT_CHECK();
	if(!cdrom)
		return RAISE(PyExc_SDLError, "CD drive not initialized");

	result = SDL_CDResume(cdrom);
	if(result == -1)
		return RAISE(PyExc_SDLError, SDL_GetError());

	RETURN_NONE
}



    /*DOC*/ static char doc_cd_stop[] =
    /*DOC*/    "CD.stop() -> int\n"
    /*DOC*/    "stops playing cdrom\n"
    /*DOC*/    "\n"
    /*DOC*/    "Stops the playing CD. If the CD is not playing, this will\n"
    /*DOC*/    "do nothing.\n"
    /*DOC*/ ;

static PyObject* cd_stop(PyObject* self, PyObject* args)
{
	int cd_id = PyCD_AsID(self);
	SDL_CD* cdrom = cdrom_drivedata[cd_id];
	int result;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	CDROM_INIT_CHECK();
	if(!cdrom)
		return RAISE(PyExc_SDLError, "CD drive not initialized");

	result = SDL_CDStop(cdrom);
	if(result == -1)
		return RAISE(PyExc_SDLError, SDL_GetError());

	RETURN_NONE
}



    /*DOC*/ static char doc_cd_eject[] =
    /*DOC*/    "CD.eject() -> None\n"
    /*DOC*/    "ejects cdrom drive\n"
    /*DOC*/    "\n"
    /*DOC*/    "Ejects the media from the CDROM drive. If the drive is empty, this\n"
    /*DOC*/    "will open the CDROM drive.\n"
    /*DOC*/ ;

static PyObject* cd_eject(PyObject* self, PyObject* args)
{
	int cd_id = PyCD_AsID(self);
	SDL_CD* cdrom = cdrom_drivedata[cd_id];
	int result;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	CDROM_INIT_CHECK();
	if(!cdrom)
		return RAISE(PyExc_SDLError, "CD drive not initialized");

	result = SDL_CDEject(cdrom);
	if(result == -1)
		return RAISE(PyExc_SDLError, SDL_GetError());

	RETURN_NONE
}


    /*DOC*/ static char doc_cd_get_empty[] =
    /*DOC*/    "CD.get_empty() -> bool\n"
    /*DOC*/    "checks for a cd in the drive\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns a true value if the cd drive is empty.\n"
    /*DOC*/ ;

static PyObject* cd_get_empty(PyObject* self, PyObject* args)
{
	int cd_id = PyCD_AsID(self);
	SDL_CD* cdrom = cdrom_drivedata[cd_id];
	int status;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	CDROM_INIT_CHECK();
	if(!cdrom)
		return RAISE(PyExc_SDLError, "CD drive not initialized");

	status = SDL_CDStatus(cdrom);
	return PyInt_FromLong(status == CD_TRAYEMPTY);
}


    /*DOC*/ static char doc_cd_get_busy[] =
    /*DOC*/    "CD.get_busy() -> bool\n"
    /*DOC*/    "checks if the cd is currently playing\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns a true value if the cd drive is currently playing. If\n"
    /*DOC*/    "the drive is paused, this will return false.\n"
    /*DOC*/ ;

static PyObject* cd_get_busy(PyObject* self, PyObject* args)
{
	int cd_id = PyCD_AsID(self);
	SDL_CD* cdrom = cdrom_drivedata[cd_id];
	int status;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	CDROM_INIT_CHECK();
	if(!cdrom)
		return RAISE(PyExc_SDLError, "CD drive not initialized");

	status = SDL_CDStatus(cdrom);
	return PyInt_FromLong(status == CD_PLAYING);
}


    /*DOC*/ static char doc_cd_get_paused[] =
    /*DOC*/    "CD.get_paused() -> bool\n"
    /*DOC*/    "checks if the cd is currently paused\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns a true value if the cd drive is currently paused.\n"
    /*DOC*/ ;

static PyObject* cd_get_paused(PyObject* self, PyObject* args)
{
	int cd_id = PyCD_AsID(self);
	SDL_CD* cdrom = cdrom_drivedata[cd_id];
	int status;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	CDROM_INIT_CHECK();
	if(!cdrom)
		return RAISE(PyExc_SDLError, "CD drive not initialized");

	status = SDL_CDStatus(cdrom);
	return PyInt_FromLong(status == CD_PAUSED);
}


    /*DOC*/ static char doc_cd_get_current[] =
    /*DOC*/    "CD.get_current() -> track, seconds\n"
    /*DOC*/    "get current position of the cdrom\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the current track on the cdrom and the number of\n"
    /*DOC*/    "seconds into that track.\n"
    /*DOC*/ ;

static PyObject* cd_get_current(PyObject* self, PyObject* args)
{
	int cd_id = PyCD_AsID(self);
	SDL_CD* cdrom = cdrom_drivedata[cd_id];
	int track;
	float seconds;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	CDROM_INIT_CHECK();
	if(!cdrom)
		return RAISE(PyExc_SDLError, "CD drive not initialized");

	SDL_CDStatus(cdrom);
	track = cdrom->cur_track;
	seconds = cdrom->cur_frame / (float)CD_FPS;

	return Py_BuildValue("(if)", track, seconds);
}


    /*DOC*/ static char doc_cd_get_numtracks[] =
    /*DOC*/    "CD.get_numtracks() -> numtracks\n"
    /*DOC*/    "get number of tracks on cd\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the number of available tracks on the CD. Note that not\n"
    /*DOC*/    "all of these tracks contain audio data. Use CD.get_track_audio() to check\n"
    /*DOC*/    "the track type before playing.\n"
    /*DOC*/ ;

static PyObject* cd_get_numtracks(PyObject* self, PyObject* args)
{
	int cd_id = PyCD_AsID(self);
	SDL_CD* cdrom = cdrom_drivedata[cd_id];

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	CDROM_INIT_CHECK();
	if(!cdrom)
		return RAISE(PyExc_SDLError, "CD drive not initialized");

	SDL_CDStatus(cdrom);
	return PyInt_FromLong(cdrom->numtracks);
}


    /*DOC*/ static char doc_cd_get_id[] =
    /*DOC*/    "CD.get_id() -> idnum\n"
    /*DOC*/    "get device id number for drive\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the device id number for this cdrom drive. This is the\n"
    /*DOC*/    "same number used in the call to pygame.cdrom.CD() to create this\n"
    /*DOC*/    "cd device. The CD object does not need to be initialized for this\n"
    /*DOC*/    "function to work.\n"
    /*DOC*/ ;

static PyObject* cd_get_id(PyObject* self, PyObject* args)
{
	int cd_id = PyCD_AsID(self);

	if(!PyArg_ParseTuple(args, ""))
		return NULL;
	return PyInt_FromLong(cd_id);
}


    /*DOC*/ static char doc_cd_get_name[] =
    /*DOC*/    "CD.get_name(id) -> string\n"
    /*DOC*/    "query name of cdrom drive\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the name of the CDROM device, given by the\n"
    /*DOC*/    "system. This function can be called before the drive\n"
    /*DOC*/    "is initialized.\n"
    /*DOC*/ ;

static PyObject* cd_get_name(PyObject* self, PyObject* args)
{
	int cd_id = PyCD_AsID(self);

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	CDROM_INIT_CHECK();

	return PyString_FromString(SDL_CDName(cd_id));
}


    /*DOC*/ static char doc_cd_get_track_audio[] =
    /*DOC*/    "CD.get_track_audio(track) -> bool\n"
    /*DOC*/    "check if a track has audio data\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns true if the cdrom track contains audio data.\n"
    /*DOC*/ ;

static PyObject* cd_get_track_audio(PyObject* self, PyObject* args)
{
	int cd_id = PyCD_AsID(self);
	SDL_CD* cdrom = cdrom_drivedata[cd_id];
	int track;

	if(!PyArg_ParseTuple(args, "i", &track))
		return NULL;

	CDROM_INIT_CHECK();
	if(!cdrom)
		return RAISE(PyExc_SDLError, "CD drive not initialized");
	SDL_CDStatus(cdrom);
	if(track < 0 || track >= cdrom->numtracks)
		return RAISE(PyExc_IndexError, "Invalid track number");

	return PyInt_FromLong(cdrom->track[track].type == SDL_AUDIO_TRACK);
}


    /*DOC*/ static char doc_cd_get_track_length[] =
    /*DOC*/    "CD.get_track_length(track) -> seconds\n"
    /*DOC*/    "check the length of an audio track\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the number of seconds in an audio track. If the\n"
    /*DOC*/    "track does not contain audio data, returns 0.0.\n"
    /*DOC*/ ;

static PyObject* cd_get_track_length(PyObject* self, PyObject* args)
{
	int cd_id = PyCD_AsID(self);
	SDL_CD* cdrom = cdrom_drivedata[cd_id];
	int track;

	if(!PyArg_ParseTuple(args, "i", &track))
		return NULL;

	CDROM_INIT_CHECK();
	if(!cdrom)
		return RAISE(PyExc_SDLError, "CD drive not initialized");
	SDL_CDStatus(cdrom);
	if(track < 0 || track >= cdrom->numtracks)
		return RAISE(PyExc_IndexError, "Invalid track number");
	if(cdrom->track[track].type != SDL_AUDIO_TRACK)
		return PyFloat_FromDouble(0.0);

	return PyFloat_FromDouble(cdrom->track[track].length / (double)CD_FPS);
}

    /*DOC*/ static char doc_cd_get_track_start[] =
    /*DOC*/    "CD.get_track_start(track) -> seconds\n"
    /*DOC*/    "check the start of an audio track\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the number of seconds an audio track starts\n"
    /*DOC*/    "on the cd.\n"
    /*DOC*/ ;

static PyObject* cd_get_track_start(PyObject* self, PyObject* args)
{
	int cd_id = PyCD_AsID(self);
	SDL_CD* cdrom = cdrom_drivedata[cd_id];
	int track;

	if(!PyArg_ParseTuple(args, "i", &track))
		return NULL;

	CDROM_INIT_CHECK();
	if(!cdrom)
		return RAISE(PyExc_SDLError, "CD drive not initialized");
	SDL_CDStatus(cdrom);
	if(track < 0 || track >= cdrom->numtracks)
		return RAISE(PyExc_IndexError, "Invalid track number");

	return PyFloat_FromDouble(cdrom->track[track].offset / (double)CD_FPS);
}


    /*DOC*/ static char doc_cd_get_all[] =
    /*DOC*/    "CD.get_all() -> tuple\n"
    /*DOC*/    "get all track information for the cd\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns a tuple with values for each track on the CD.\n"
    /*DOC*/    "Each item in the tuple is a tuple with 4 values for each\n"
    /*DOC*/    "track. First is a boolean set to true if this is an audio\n"
    /*DOC*/    "track. The next 3 values are the start time, end time, and\n"
    /*DOC*/    "length of the track.\n"
    /*DOC*/ ;

static PyObject* cd_get_all(PyObject* self, PyObject* args)
{
	int cd_id = PyCD_AsID(self);
	SDL_CD* cdrom = cdrom_drivedata[cd_id];
	int track;
	PyObject *tuple, *item;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	CDROM_INIT_CHECK();
	if(!cdrom)
		return RAISE(PyExc_SDLError, "CD drive not initialized");

	SDL_CDStatus(cdrom);
	tuple = PyTuple_New(cdrom->numtracks);
	if(!tuple)
		return NULL;
	for(track=0; track < cdrom->numtracks; track++)
	{
		int audio = cdrom->track[track].type == SDL_AUDIO_TRACK;
		double start = cdrom->track[track].offset / (double)CD_FPS;
		double length = cdrom->track[track].length / (double)CD_FPS;
		double end = start + length;
		item = PyTuple_New(4);
		if(!item)
		{
			Py_DECREF(tuple);
			return NULL;
		}
		PyTuple_SET_ITEM(item, 0, PyInt_FromLong(audio));
		PyTuple_SET_ITEM(item, 1, PyFloat_FromDouble(start));
		PyTuple_SET_ITEM(item, 2, PyFloat_FromDouble(end));
		PyTuple_SET_ITEM(item, 3, PyFloat_FromDouble(length));
		PyTuple_SET_ITEM(tuple, track, item);
	}

	return tuple;
}




static PyMethodDef cd_builtins[] =
{
	{ "init", cd_init, 1, doc_cd_init },
	{ "quit", cd_quit, 1, doc_cd_quit },
	{ "get_init", cd_get_init, 1, doc_cd_get_init },

	{ "play", cd_play, 1, doc_cd_play },
	{ "pause", cd_pause, 1, doc_cd_pause },
	{ "resume", cd_resume, 1, doc_cd_resume },
	{ "stop", cd_stop, 1, doc_cd_stop },
	{ "eject", cd_eject, 1, doc_cd_eject },

	{ "get_empty", cd_get_empty, 1, doc_cd_get_empty },
	{ "get_busy", cd_get_busy, 1, doc_cd_get_busy },
	{ "get_paused", cd_get_paused, 1, doc_cd_get_paused },
	{ "get_current", cd_get_current, 1, doc_cd_get_current },
	{ "get_numtracks", cd_get_numtracks, 1, doc_cd_get_numtracks },
	{ "get_id", cd_get_id, 1, doc_cd_get_id },
	{ "get_name", cd_get_name, 1, doc_cd_get_name },
	{ "get_all", cd_get_all, 1, doc_cd_get_all },

	{ "get_track_audio", cd_get_track_audio, 1, doc_cd_get_track_audio },
	{ "get_track_length", cd_get_track_length, 1, doc_cd_get_track_length },
	{ "get_track_start", cd_get_track_start, 1, doc_cd_get_track_start },

	{ NULL, NULL }
};

static PyObject* cd_getattr(PyObject* self, char* attrname)
{
	return Py_FindMethod(cd_builtins, self, attrname);
}


    /*DOC*/ static char doc_CD_MODULE[] =
    /*DOC*/    "The CD object represents a CDROM drive and allows you to\n"
    /*DOC*/    "access the CD inside that drive. All functions (except get_name() and get_id())\n"
    /*DOC*/    "require the CD object to be initialized. This is done with the\n"
    /*DOC*/    "CD.init() function.\n"
    /*DOC*/    "\n"
    /*DOC*/    "Be sure to understand there is a difference between the cdrom module\n"
    /*DOC*/    "and the CD objects.\n"
    /*DOC*/ ;


static PyTypeObject PyCD_Type =
{
	PyObject_HEAD_INIT(NULL)
	0,
	"CD",
	sizeof(PyCDObject),
	0,
	cd_dealloc,
	0,
	cd_getattr,
	0,
	0,
	0,
	0,
	NULL,
	0,
	(hashfunc)NULL,
	(ternaryfunc)NULL,
	(reprfunc)NULL,
	0L,0L,0L,0L,
	doc_CD_MODULE /* Documentation string */
};



static PyObject* PyCD_New(int id)
{
	PyCDObject* cd;

	if(id < 0 || id >= CDROM_MAXDRIVES || id >= SDL_CDNumDrives())
		return RAISE(PyExc_SDLError, "Invalid cdrom device number");

	cd = PyObject_NEW(PyCDObject, &PyCD_Type);
	if(!cd) return NULL;

	cd->id = id;

	return (PyObject*)cd;
}





static PyMethodDef cdrom_builtins[] =
{
	{ "__PYGAMEinit__", cdrom_autoinit, 1, doc_cdrom_init },
	{ "init", cdrom_init, 1, doc_cdrom_init },
	{ "quit", cdrom_quit, 1, doc_cdrom_quit },
	{ "get_init", get_init, 1, doc_get_init },
	{ "get_count", get_count, 1, doc_get_count },
	{ "CD", CD, 1, doc_CD },
	{ NULL, NULL }
};



    /*DOC*/ static char doc_pygame_cdrom_MODULE[] =
    /*DOC*/    "The cdrom module provides a few functions to initialize\n"
    /*DOC*/    "the CDROM subsystem and to manage the CD objects. The CD\n"
    /*DOC*/    "objects are created with the pygame.cdrom.CD() function.\n"
    /*DOC*/    "This function needs a cdrom device number to work on. All\n"
    /*DOC*/    "cdrom drives on the system are enumerated for use as a CD\n"
    /*DOC*/    "object. To access most of the CD functions, you'll need to\n"
    /*DOC*/    "init() the CD. (note that the cdrom module will already\n"
    /*DOC*/    "be initialized). When multiple CD objects are created for the\n"
    /*DOC*/    "same CDROM device, the state and values for those CD objects\n"
    /*DOC*/    "will be shared.\n"
    /*DOC*/    "\n"
    /*DOC*/    "You can call the CD.get_name() and CD.get_id() functions\n"
    /*DOC*/    "without initializing the CD object.\n"
    /*DOC*/    "\n"
    /*DOC*/    "Be sure to understand there is a difference between the cdrom module\n"
    /*DOC*/    "and the CD objects.\n"
    /*DOC*/ ;

PYGAME_EXPORT
void initcdrom(void)
{
	PyObject *module, *dict, *apiobj;
	static void* c_api[PYGAMEAPI_CDROM_NUMSLOTS];

	PyType_Init(PyCD_Type);


    /* create the module */
	module = Py_InitModule3("cdrom", cdrom_builtins, doc_pygame_cdrom_MODULE);
	dict = PyModule_GetDict(module);

	PyDict_SetItemString(dict, "CDType", (PyObject *)&PyCD_Type);

	/* export the c api */
	c_api[0] = &PyCD_Type;
	c_api[1] = PyCD_New;
	apiobj = PyCObject_FromVoidPtr(c_api, NULL);
	PyDict_SetItemString(dict, PYGAMEAPI_LOCAL_ENTRY, apiobj);
	Py_DECREF(apiobj);

	/*imported needed apis*/
	import_pygame_base();
}

