/*
    pygame - Python Game Library
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

#define PYGAMEAPI_CDROM_INTERNAL
#include "pygame.h"




staticforward PyTypeObject PyCD_Type;
static PyObject* PyCD_New(SDL_CD* cdrom);
#define PyCD_Check(x) ((x)->ob_type == &PyCD_Type)



static void cdrom_autoquit()
{
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
	PyCDObject* cd_ref = (PyCDObject*)self;

	if(SDL_WasInit(SDL_INIT_CDROM))
		SDL_CDClose(cd_ref->cd);
	PyMem_DEL(self);	
}



    /*DOC*/ static char doc_cdrom_open[] =
    /*DOC*/    "pygame.cdrom.open(id) -> CD\n"
    /*DOC*/    "open cd device\n"
    /*DOC*/    "\n"
    /*DOC*/    "Creates a new CD object for the given CDROM id.\n"
    /*DOC*/ ;

static PyObject* cdrom_open(PyObject* self, PyObject* args)
{
	int id;
	
	if(!PyArg_ParseTuple(args, "i", &id))
		return NULL;

	CDROM_INIT_CHECK();

	return PyCD_New(SDL_CDOpen(id));
}



    /*DOC*/ static char doc_cdrom_count[] =
    /*DOC*/    "pygame.cdrom.count() -> int\n"
    /*DOC*/    "query number of cdroms on system\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the number of CDROM drives available on\n"
    /*DOC*/    "the system\n"
    /*DOC*/ ;

static PyObject* cdrom_count(PyObject* self, PyObject* args)
{
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	CDROM_INIT_CHECK();

	return PyInt_FromLong(SDL_CDNumDrives());
}



    /*DOC*/ static char doc_cdrom_name[] =
    /*DOC*/    "pygame.cdrom.name(id) -> string\n"
    /*DOC*/    "query name of cdrom drive\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the name of the CDROM device, given by the\n"
    /*DOC*/    "system.\n"
    /*DOC*/ ;

static PyObject* cdrom_name(PyObject* self, PyObject* args)
{
	int drive;
	
	if(!PyArg_ParseTuple(args, "i", &drive))
		return NULL;

	CDROM_INIT_CHECK();

	return PyString_FromString(SDL_CDName(drive));
}



    /*DOC*/ static char doc_cd_play_tracks[] =
    /*DOC*/    "CD.play_tracks(start_track, start_frame, ntracks, nframes) -> int\n"
    /*DOC*/    "play music from cdrom\n"
    /*DOC*/    "\n"
    /*DOC*/    "Start playing from start_track, for ntracks and\n"
    /*DOC*/    "nframes. If ntracks and nframes are 0, it will\n"
    /*DOC*/    "play until the end of the cdrom\n"
    /*DOC*/ ;

static PyObject* cd_play_tracks(PyObject* self, PyObject* args)
{
	int start_track, start_frame, ntracks, nframes;
	PyCDObject* cd_ref = (PyCDObject*)self;

	if(!PyArg_ParseTuple(args, "iiii", &start_track, &start_frame, &ntracks,
		&nframes	))
		return NULL;
		
	SDL_CDStatus(cd_ref->cd);
		
	return PyInt_FromLong(
		SDL_CDPlayTracks(cd_ref->cd, start_track, start_frame, ntracks, nframes));
}



    /*DOC*/ static char doc_cd_play[] =
    /*DOC*/    "CD.play(start_frame, nframes) -> int\n"
    /*DOC*/    "play music from cdrom\n"
    /*DOC*/    "\n"
    /*DOC*/    "Start playing from start_frame, for nframes. If\n"
    /*DOC*/    "nframes is 0, it will play until the end of the\n"
    /*DOC*/    "cdrom\n"
    /*DOC*/ ;

static PyObject* cd_play(PyObject* self, PyObject* args)
{
	int start_frame, nframes;
	PyCDObject* cd_ref = (PyCDObject*)self;
	
	if(!PyArg_ParseTuple(args, "ii", &start_frame, &nframes))
		return NULL;

	SDL_CDStatus(cd_ref->cd);

	return PyInt_FromLong(SDL_CDPlay(cd_ref->cd, start_frame, nframes));
}



    /*DOC*/ static char doc_cd_pause[] =
    /*DOC*/    "CD.pause() -> int\n"
    /*DOC*/    "pause playing cdrom\n"
    /*DOC*/    "\n"
    /*DOC*/    "Pauses the playing CD.\n"
    /*DOC*/ ;

static PyObject* cd_pause(PyObject* self, PyObject* args)
{
	PyCDObject* cd_ref = (PyCDObject*)self;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	if(SDL_CDStatus(cd_ref->cd) != CD_PLAYING)
		return PyInt_FromLong(-1);

	return PyInt_FromLong(SDL_CDPause(cd_ref->cd));
}



    /*DOC*/ static char doc_cd_resume[] =
    /*DOC*/    "CD.resume() -> int\n"
    /*DOC*/    "resume paused cdrom\n"
    /*DOC*/    "\n"
    /*DOC*/    "Resumes playback of a paused CD.\n"
    /*DOC*/ ;

static PyObject* cd_resume(PyObject* self, PyObject* args)
{
	PyCDObject* cd_ref = (PyCDObject*)self;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	if(SDL_CDStatus(cd_ref->cd) != CD_PAUSED)
		return PyInt_FromLong(-1);

	return PyInt_FromLong(SDL_CDResume(cd_ref->cd));
}



    /*DOC*/ static char doc_cd_stop[] =
    /*DOC*/    "CD.stop() -> int\n"
    /*DOC*/    "stops playing cdrom\n"
    /*DOC*/    "\n"
    /*DOC*/    "Stops the playing CD.\n"
    /*DOC*/ ;

static PyObject* cd_stop(PyObject* self, PyObject* args)
{
	PyCDObject* cd_ref = (PyCDObject*)self;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	if(SDL_CDStatus(cd_ref->cd) < CD_PLAYING)
		return PyInt_FromLong(-1);

	return PyInt_FromLong(SDL_CDStop(cd_ref->cd));
}



    /*DOC*/ static char doc_cd_eject[] =
    /*DOC*/    "CD.eject() -> int\n"
    /*DOC*/    "ejects cdrom drive\n"
    /*DOC*/    "\n"
    /*DOC*/    "Ejects the media from the CDROM drive.\n"
    /*DOC*/ ;

static PyObject* cd_eject(PyObject* self, PyObject* args)
{
	PyCDObject* cd_ref = (PyCDObject*)self;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	if(SDL_CDStatus(cd_ref->cd) <= 0 )
		return PyInt_FromLong(-1);

	return PyInt_FromLong(SDL_CDEject(cd_ref->cd));
}



    /*DOC*/ static char doc_cd_status[] =
    /*DOC*/    "CD.get_status() -> int\n"
    /*DOC*/    "query drive status\n"
    /*DOC*/    "\n"
    /*DOC*/    "Get the status of the CDROM drive.\n"
    /*DOC*/ ;

static PyObject* cd_status(PyObject* self, PyObject* args)
{
	PyCDObject* cd_ref = (PyCDObject*)self;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	return PyInt_FromLong(SDL_CDStatus(cd_ref->cd));
}



    /*DOC*/ static char doc_cd_cur_track[] =
    /*DOC*/    "CD.get_cur_track() -> int\n"
    /*DOC*/    "query current track\n"
    /*DOC*/    "\n"
    /*DOC*/    "Get the current track of a playing CD.\n"
    /*DOC*/ ;

static PyObject* cd_cur_track(PyObject* self, PyObject* args)
{
	PyCDObject* cd_ref = (PyCDObject*)self;
	SDL_CD* s_cd_ref = cd_ref->cd;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	if(SDL_CDStatus(s_cd_ref) <= 1)
		return PyInt_FromLong(-1);

	return PyInt_FromLong(s_cd_ref->cur_track);
}



    /*DOC*/ static char doc_cd_cur_frame[] =
    /*DOC*/    "CD.get_cur_frame() -> int\n"
    /*DOC*/    "query current frame\n"
    /*DOC*/    "\n"
    /*DOC*/    "Get the current frame of a playing CD.\n"
    /*DOC*/ ;

static PyObject* cd_cur_frame(PyObject* self, PyObject* args)
{
	PyCDObject* cd_ref = (PyCDObject*)self;
	SDL_CD* s_cd_ref = cd_ref->cd;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	if(SDL_CDStatus(s_cd_ref) <= 1)
		return PyInt_FromLong(-1);

	return PyInt_FromLong(s_cd_ref->cur_frame);
}



    /*DOC*/ static char doc_cd_get_empty[] =
    /*DOC*/    "CD.get_empty() -> bool\n"
    /*DOC*/    "query status of drive\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns true when there is no disk in the CD\n"
    /*DOC*/    "drive.\n"
    /*DOC*/ ;

static PyObject* cd_get_empty(PyObject* self, PyObject* args)
{
	PyCDObject* cd_ref = (PyCDObject*)self;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	return PyInt_FromLong(CD_INDRIVE(SDL_CDStatus(cd_ref->cd)));
}





static int cd_seq_len(PyObject* self)
{
	PyCDObject* cd_ref = (PyCDObject*)self;

	if(!SDL_WasInit(SDL_INIT_CDROM))
	{
		RAISE(PyExc_RuntimeError, "SDL has shut down.");
		return -1;
	}
	
	if(!CD_INDRIVE(SDL_CDStatus(cd_ref->cd)))
	{
		RAISE(PyExc_RuntimeError, "No CD present");
		return -1;
	}

	return cd_ref->cd->numtracks;
}



static PyObject* cd_seq_get(PyObject* self, int index)
{
	PyCDObject* cd_ref = (PyCDObject*)self;
	SDL_CD* s_cd_ref = cd_ref->cd;
	SDL_CDtrack* s_track_ref;

	if(!SDL_WasInit(SDL_INIT_CDROM))
		return RAISE(PyExc_RuntimeError, "SDL has shut down.");

	if(!CD_INDRIVE(SDL_CDStatus(s_cd_ref)) || index >= s_cd_ref->numtracks
		|| index < 0)
	{
		return PyInt_FromLong(-1);
	}

	s_track_ref = &s_cd_ref->track[index];
																			 
	return Py_BuildValue("(iii)", s_track_ref->type, s_track_ref->length,
		s_track_ref->offset);	
}



static PyObject* cd_seq_get_slice(PyObject* self, int start, int end)
{
	PyCDObject* cd_ref = (PyCDObject*)self;
	SDL_CD* s_cd_ref = cd_ref->cd;
	PyObject* track_tuple;
	int count;
	int i;

	if(!SDL_WasInit(SDL_INIT_CDROM))
		return RAISE(PyExc_RuntimeError, "SDL has shut down.");
	
	i = min(start,end);
	end = max(start,end);
	start = i;

	if(start < 0)
		return RAISE(PyExc_IndexError, "Index out of range.");
	
	if(CD_INDRIVE(SDL_CDStatus(s_cd_ref)) <= 0)
		return RAISE(PyExc_RuntimeError, "CD unavailable");

	/* If end is greater than the real number of tracks, trunctate it.
	 * This is done because of how Python handles slicing
	*/
	end = min(end, s_cd_ref->numtracks);
	count = end - start;

	track_tuple = PyTuple_New(count);
	if(!track_tuple)
		return NULL;

	for(i = 0;i < count;i++)
	{
		PyObject* elem_tuple;
		SDL_CDtrack* s_track_ref = &s_cd_ref->track[i + start];

		elem_tuple = Py_BuildValue("(iii)",
			s_track_ref->type, s_track_ref->length, s_track_ref->offset);
		if(!elem_tuple)
		{
			Py_DECREF(track_tuple);
			return NULL;
		}

		PyTuple_SET_ITEM(track_tuple, i, elem_tuple);
	}
	return track_tuple; 	
}



static PyMethodDef cd_builtins[] =
{
	{ "play_tracks", cd_play_tracks, 1, doc_cd_play_tracks },
	{ "play", cd_play, 1, doc_cd_play },
	{ "pause", cd_pause, 1, doc_cd_pause },
	{ "resume", cd_resume, 1, doc_cd_resume },
	{ "stop", cd_stop, 1, doc_cd_stop },
	{ "eject", cd_eject, 1, doc_cd_eject },
	{ "get_status", cd_status, 1, doc_cd_status },
	{ "get_track", cd_cur_track, 1, doc_cd_cur_track },
	{ "get_frame", cd_cur_frame, 1, doc_cd_cur_frame },
	{ "get_empty", cd_get_empty, 1, doc_cd_get_empty },
	{ NULL, NULL }
};

static PyObject* cd_getattr(PyObject* self, char* attrname)
{
	if(SDL_WasInit(SDL_INIT_CDROM))
		return Py_FindMethod(cd_builtins, self, attrname);

	return RAISE(PyExc_NameError, attrname);
}



static PySequenceMethods cd_as_sequence = 
{
	cd_seq_len,
	0,
	0,
	cd_seq_get,
	cd_seq_get_slice,
	0,
	0
};


    /*DOC*/ static char doc_CD_MODULE[] =
    /*DOC*/    "thin wrapper around the SDL CDROM api, likely to\n"
    /*DOC*/    "change\n"
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
	&cd_as_sequence,
	0
};



static PyObject* PyCD_New(SDL_CD* cdrom)
{
	PyCDObject* cd;

	if(!cdrom)
		return RAISE(PyExc_SDLError, SDL_GetError());
	
	cd = PyObject_NEW(PyCDObject, &PyCD_Type);
	if(!cd) return NULL;

	cd->cd = cdrom;

	return (PyObject*)cd;
}





static PyMethodDef cdrom_builtins[] =
{
	{ "__PYGAMEinit__", cdrom_autoinit, 1, doc_cdrom_init },
	{ "init", cdrom_init, 1, doc_cdrom_init },
	{ "quit", cdrom_quit, 1, doc_cdrom_quit },
	{ "get_init", get_init, 1, doc_get_init },
	{ "get_count", cdrom_count, 1, doc_cdrom_count },
	{ "get_name", cdrom_name, 1, doc_cdrom_name },
	{ "open", cdrom_open, 1, doc_cdrom_open },
	{ NULL, NULL }
};



    /*DOC*/ static char doc_pygame_cdrom_MODULE[] =
    /*DOC*/    "thin wrapper around the SDL CDROM api, likely to\n"
    /*DOC*/    "change\n"
    /*DOC*/ ;


void initcdrom()
{
	PyObject *module, *dict, *apiobj;
	static void* c_api[PYGAMEAPI_CDROM_NUMSLOTS];

	PyType_Init(PyCD_Type);


    /* create the module */
	module = Py_InitModule3("cdrom", cdrom_builtins, doc_pygame_cdrom_MODULE);
	dict = PyModule_GetDict(module);

	/* export the c api */
	c_api[0] = &PyCD_Type;
	c_api[1] = PyCD_New;
	apiobj = PyCObject_FromVoidPtr(c_api, NULL);
	PyDict_SetItemString(dict, PYGAMEAPI_LOCAL_ENTRY, apiobj);

	/*imported needed apis*/
	import_pygame_base();
}

