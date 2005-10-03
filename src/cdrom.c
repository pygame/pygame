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
#include "pygamedocs.h"


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


static PyObject* cdrom_quit(PyObject* self, PyObject* arg)
{
	if(!PyArg_ParseTuple(arg, ""))
		return NULL;

	cdrom_autoquit();

	RETURN_NONE
}


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


static PyObject* CD(PyObject* self, PyObject* args)
{
	int id;
	if(!PyArg_ParseTuple(args, "i", &id))
		return NULL;

	CDROM_INIT_CHECK();

	return PyCD_New(id);
}


static PyObject* get_count(PyObject* self, PyObject* args)
{
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	CDROM_INIT_CHECK();

	return PyInt_FromLong(SDL_CDNumDrives());
}


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


static PyObject* cd_get_init(PyObject* self, PyObject* args)
{
	int cd_id = PyCD_AsID(self);

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	return PyInt_FromLong(cdrom_drivedata[cd_id] != NULL);
}


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


static PyObject* cd_get_id(PyObject* self, PyObject* args)
{
	int cd_id = PyCD_AsID(self);

	if(!PyArg_ParseTuple(args, ""))
		return NULL;
	return PyInt_FromLong(cd_id);
}


static PyObject* cd_get_name(PyObject* self, PyObject* args)
{
	int cd_id = PyCD_AsID(self);

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	CDROM_INIT_CHECK();

	return PyString_FromString(SDL_CDName(cd_id));
}


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
	{ "init", cd_init, 1, DOC_CDINIT },
	{ "quit", cd_quit, 1, DOC_CDQUIT },
	{ "get_init", cd_get_init, 1, DOC_CDGETINIT },

	{ "play", cd_play, 1, DOC_CDINIT },
	{ "pause", cd_pause, 1, DOC_CDPAUSE },
	{ "resume", cd_resume, 1, DOC_CDRESUME },
	{ "stop", cd_stop, 1, DOC_CDSTOP },
	{ "eject", cd_eject, 1, DOC_CDEJECT },

	{ "get_empty", cd_get_empty, 1, DOC_CDGETEMPTY },
	{ "get_busy", cd_get_busy, 1, DOC_CDGETBUSY },
	{ "get_paused", cd_get_paused, 1, DOC_CDGETPAUSED },
	{ "get_current", cd_get_current, 1, DOC_CDGETCURRENT },
	{ "get_numtracks", cd_get_numtracks, 1, DOC_CDGETNUMTRACKS },
	{ "get_id", cd_get_id, 1, DOC_CDGETINIT },
	{ "get_name", cd_get_name, 1, DOC_CDGETNAME },
	{ "get_all", cd_get_all, 1, DOC_CDGETALL },

	{ "get_track_audio", cd_get_track_audio, 1, DOC_CDGETTRACKAUDIO },
	{ "get_track_length", cd_get_track_length, 1, DOC_CDGETTRACKLENGTH },
	{ "get_track_start", cd_get_track_start, 1, DOC_CDGETTRACKSTART },

	{ NULL, NULL }
};

static PyObject* cd_getattr(PyObject* self, char* attrname)
{
	return Py_FindMethod(cd_builtins, self, attrname);
}



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
	DOC_PYGAMECDROMCD /* Documentation string */
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
	{ "__PYGAMEinit__", cdrom_autoinit, 1, "auto initialize function" },
	{ "init", cdrom_init, 1, DOC_PYGAMECDROMINIT },
	{ "quit", cdrom_quit, 1, DOC_PYGAMECDROMQUIT },
	{ "get_init", get_init, 1, DOC_PYGAMECDROMGETINIT },
	{ "get_count", get_count, 1, DOC_PYGAMECDROMGETCOUNT },
	{ "CD", CD, 1, DOC_PYGAMECDROMCD },
	{ NULL, NULL }
};



PYGAME_EXPORT
void initcdrom(void)
{
	PyObject *module, *dict, *apiobj;
	static void* c_api[PYGAMEAPI_CDROM_NUMSLOTS];

	PyType_Init(PyCD_Type);


    /* create the module */
	module = Py_InitModule3("cdrom", cdrom_builtins, DOC_PYGAMECDROM);
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

