#include "gmovie.h"


void _movie_init_internal(PyMovie *self, const char *filename, SDL_Surface *surf)
{
	Py_INCREF(self);
	//already malloced memory for PyMovie.
	self->_backend="FFMPEG_WRAPPER";
	if(!surf)
	{
		//set overlay to true
		self->overlay = 1;
	}
	else
	{
		self->overlay = 0;
		self->canon_surf=surf;
	}
	self->start_time = AV_NOPTS_VALUE;
	stream_open(self, filename, NULL, 0);
	if(!self)
	{
		PyErr_SetString(PyExc_IOError, "stream_open failed");
        Py_DECREF(self);
        return;
    }
	//PySys_WriteStdout("Movie->filename: %s\n", self->filename);
	Py_DECREF(self);
	return;
}


int _movie_init(PyObject *self, PyObject *args, PyObject *kwds)
{
	Py_INCREF(self);
	const char *c;
	PyObject *surf;
	if (!PyArg_ParseTuple (args, "s|O", &c, &surf))
    {
        PyErr_SetString(PyExc_TypeError, "No valid arguments");
    	return -1;
    }
   
    if(surf && PySurface_Check(surf))
    {
    	SDL_Surface *target = PySurface_AsSurface(surf);
    	 _movie_init_internal((PyMovie *)self, c, target);	
    }
    else
    {
		_movie_init_internal((PyMovie *)self, c, NULL);
    }
	PyObject *er;
    er = PyErr_Occurred();
    Py_XINCREF(er);
    if(er)
    {
        PyErr_Print();
    }
    Py_XDECREF(er);
    if(!self)
    {
        PyErr_SetString(PyExc_IOError, "No movie object created.");
        PyErr_Print();
        Py_DECREF(self);
        return -1;
    }
    Py_DECREF(self);
    //PySys_WriteStdout("Returning from _movie_init\n");
    return 0;
}   

void _movie_dealloc(PyMovie *movie)
{
    stream_close(movie);
    movie->ob_type->tp_free((PyObject *) movie);
}

char *format_timestamp(double pts, char *buf)
{
	int h, m;
	double s;
	if(pts/60 > 60)
	{
		h = (int)pts/3600;
		pts = pts-h*3600;
	}
	else
	{
		h=0;
	}
	m=(int)pts/60;
	if(m>0)
	{
		s = pts-m*60;
	}
	else
	{
		s = pts;
		m=0;
	}
	//char buf[50];
	if(h>0 && m>0)
	{
		PyOS_snprintf(buf, sizeof(buf), "%i:%i:%fs", h, m, s);
	}	
	else if (m>0)
	{
		PyOS_snprintf(buf, sizeof(buf), "%i:%fs", m, s);
	}
	else if(s>0)
	{
		PyOS_snprintf(buf, sizeof(buf), "%fs", s);
	}
	else
	{
		PyOS_snprintf(buf, sizeof(buf), "0.0s");
	}
	return buf;
}

PyObject* _movie_repr (PyMovie *movie)
{
    /*Eventually add a time-code call */
    Py_INCREF(movie);
    char buf[150];
    char buf2[50];
    PyOS_snprintf(buf, sizeof(buf), "(Movie: %s, %s)", movie->filename, format_timestamp(movie->pts, buf2));
    PyObject *buffer = PyString_FromString(buf);
    return buffer;
}
PyObject* _movie_play(PyMovie *movie, PyObject* args)
{
	PyEval_InitThreads();
	if(movie->playing)
	{
		if(movie->paused)
		{
			//if we've called this after we paused n times where n is odd, then we unpause the movie
			_movie_pause(movie);
		}
		//indicating we've called play again before its finished		
	}
	else
	{
		PyInterpreterState *interp;
		PyThreadState *thread;
		thread=PyThreadState_Get();
		interp = thread->interp;
		movie->_tstate = PyThreadState_New(interp);
	}
	Py_INCREF(movie);
    //PySys_WriteStdout("Inside .play\n");
    int loops;
    if(!PyArg_ParseTuple(args, "i", &loops))
    {
    	loops = 0;
    }
	//movie has been stopped, need to close down streams and that.
    if(movie->stop)
    {
    	if(movie->video_st)
			stream_component_close(movie, movie->video_st->index);
		if(movie->audio_st)
			stream_component_close(movie, movie->audio_st->index);
		movie->stop = 0;
    }
    
    SDL_LockMutex(movie->dest_mutex);
    movie->loops =loops;
    movie->paused = 0;
    movie->playing = 1;
    SDL_UnlockMutex(movie->dest_mutex);
	Py_BEGIN_ALLOW_THREADS	
	movie->parse_tid = SDL_CreateThread(decoder_wrapper, movie);
	Py_END_ALLOW_THREADS
    Py_DECREF(movie);
    Py_RETURN_NONE;
}

PyObject* _movie_stop(PyMovie *movie)
{
    Py_INCREF(movie);
    Py_BEGIN_ALLOW_THREADS
    SDL_LockMutex(movie->dest_mutex);
    stream_pause(movie);
    movie->seek_req = 1;
    movie->seek_pos = 0;
    movie->seek_flags =AVSEEK_FLAG_BACKWARD;
    movie->stop = 1;
    Py_END_ALLOW_THREADS
    SDL_UnlockMutex(movie->dest_mutex);  
    Py_DECREF(movie);
    Py_RETURN_NONE;
}  

PyObject* _movie_pause(PyMovie *movie)
{
	
	Py_BEGIN_ALLOW_THREADS
    stream_pause(movie); 
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

PyObject* _movie_rewind(PyMovie *movie, PyObject* args)
{
    /* For now, just alias rewind to stop */
    return _movie_stop(movie);
}


PyObject* _movie_resize       (PyMovie *movie, PyObject* args)
{
	int w, h;
	if(PyArg_ParseTuple(args, "ii", &w, &h)<0)
	{
		return NULL;
	}
	if(w<0 || h<0)
	{
		return RAISE (PyExc_SDLError, "Cannot set negative sized display mode");
	}
	movie->height = h;
	movie->width  = w;
	movie->resize_w =  movie->resize_h= 1;	
	Py_RETURN_NONE;
	
}
PyObject* _movie_get_paused (PyMovie *movie, void *closure)
{
    return PyBool_FromLong((long)movie->paused);
}
PyObject* _movie_get_playing (PyMovie *movie, void *closure)
{
    PyObject *pyo;
    pyo= PyBool_FromLong((long)movie->playing);
    return pyo;
}

PyObject* _movie_get_width (PyMovie *movie, void *closure)
{
    PyObject *pyo;
    if(movie->video_st)
    {
    	if(movie->width)
    		{pyo = PyInt_FromLong((long)movie->width);}
    	else
    	{
    		if(movie->video_st->codec)
	    		{pyo= PyInt_FromLong((long)movie->video_st->codec->width);}
    		else
    			{pyo = PyInt_FromLong((long)0);}
    	}
    }
    else
    {
		pyo = PyInt_FromLong((long)0);    	
    }
    return pyo;
}

int _movie_set_width (PyMovie *movie, PyObject *width, void *closure)
{
	int w;
	if(PyInt_Check(width))
	{
		w = (int)PyInt_AsLong(width);
		movie->resize_w=1;
		movie->width=w;
		return 0;
	}
	else
	{
		return -1;
	}
	
}

PyObject* _movie_get_height (PyMovie *movie, void *closure)
{
    PyObject *pyo;
    if(movie->video_st)
    {
    	if(movie->height)
    		{pyo=PyInt_FromLong((long)movie->height);}
    	else
    		{pyo= PyInt_FromLong((long)movie->video_st->codec->height);}
    }
    else
    {
		pyo = PyInt_FromLong((long)0);    	
    }
    return pyo;
}
int _movie_set_height (PyMovie *movie, PyObject *height, void *closure)
{
	int h;
	if(PyInt_Check(height))
	{
		h = (int)PyInt_AsLong(height);
		movie->resize_h=1;
		movie->height=h;
		return 0;
	}
	else
	{
		return -1;
	}
	
}


PyObject *_movie_get_surface(PyMovie *movie, void *closure)
{
	if(movie->canon_surf)
	{
		return (PyObject *)PySurface_New(movie->canon_surf);
	}
	Py_RETURN_NONE;
}

int _movie_set_surface(PyObject *mov, PyObject *surface, void *closure)
{
	PyMovie *movie = (PyMovie *)mov;
	if(movie->canon_surf)
	{
		SDL_FreeSurface(movie->canon_surf);	
	}
	//PySurface_Check doesn't really work right for some reason... so we skip it for now.
	if(PySurface_Check(surface))
	{
		movie->canon_surf=PySurface_AsSurface(surface);
		movie->overlay=0;
		return 0;
	}
	return -1;
}

static PyMethodDef _movie_methods[] = {
   { "play",   (PyCFunction) _movie_play,   METH_VARARGS, DOC_GMOVIEMOVIEPLAY},
   { "stop",   (PyCFunction) _movie_stop,   METH_NOARGS,  DOC_GMOVIEMOVIESTOP},
   { "pause",  (PyCFunction) _movie_pause,  METH_NOARGS,  DOC_GMOVIEMOVIEPAUSE},
   { "rewind", (PyCFunction) _movie_rewind, METH_VARARGS, DOC_GMOVIEMOVIEREWIND},
   { "resize", (PyCFunction) _movie_resize, METH_VARARGS, DOC_GMOVIEMOVIERESIZE},
   { NULL,     NULL,                        0,            NULL }
};

static PyMemberDef _movie_members[] = {
	{ "_backend", T_STRING, offsetof(struct PyMovie, _backend), 0,    "Lists which backend this movie object belongs to."},
	{ NULL}
};
	
static PyGetSetDef _movie_getsets[] =
{
    { "paused",  (getter) _movie_get_paused,  NULL,                        DOC_GMOVIEMOVIEPAUSE,   NULL },
    { "playing", (getter) _movie_get_playing, NULL,                        DOC_GMOVIEMOVIEPLAYING, NULL },
    { "height",  (getter) _movie_get_height,  (setter) _movie_set_height,  DOC_GMOVIEMOVIEHEIGHT,  NULL },
    { "width",   (getter) _movie_get_width,   (setter) _movie_set_width,   DOC_GMOVIEMOVIEWIDTH,   NULL },
    { "surface", (getter) _movie_get_surface, (setter) _movie_set_surface, DOC_GMOVIEMOVIESURFACE, NULL },
    { NULL,      NULL,                        NULL,                        NULL,                   NULL }
};

static PyTypeObject PyMovie_Type =
{
    PyObject_HEAD_INIT(NULL)
    0, 
    "pygame._movie.Movie",          /* tp_name */
    sizeof (PyMovie),           /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor) _movie_dealloc,/* tp_dealloc */
    0,                          /* tp_print */
    0,                          /* tp_getattr */
    0,                          /* tp_setattr */
    0,                          /* tp_compare */
    (reprfunc) _movie_repr,     /* tp_repr */
    0,                          /* tp_as_number */
    0,                          /* tp_as_sequence */
    0,                          /* tp_as_mapping */
    0,                          /* tp_hash */
    0,                          /* tp_call */
    0,      /* tp_str */
    0,                          /* tp_getattro */
    0,                          /* tp_setattro */
    0,                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    DOC_GMOVIE,                 /* tp_doc */
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    _movie_methods,             /* tp_methods */
    _movie_members,             /* tp_members */
    _movie_getsets,             /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc) _movie_init,                          /* tp_init */
    0,                          /* tp_alloc */
    0,                 /* tp_new */
    0,                          /* tp_free */
    0,                          /* tp_is_gc */
    0,                          /* tp_bases */
    0,                          /* tp_mro */
    0,                          /* tp_cache */
    0,                          /* tp_subclasses */
    0,                          /* tp_weaklist */
    0                           /* tp_del */
};
/*DOC*/ static char _movie_doc[] =
/*DOC*/    "ffmpeg wrapper module for pygame";

PyMODINIT_FUNC
init_movie(void)
{
    PyObject* module;

    /* imported needed apis; Do this first so if there is an error
       the module is not loaded.
    */
    import_pygame_base ();
    if (PyErr_Occurred ()) {
        MODINIT_ERROR;
    }

    // Fill in some slots in the type, and make it ready
   PyMovie_Type.tp_new = PyType_GenericNew;
   if (PyType_Ready(&PyMovie_Type) < 0) {
      MODINIT_ERROR;
   }
   /*
   PyAudioStream_Type.tp_new = PyType_GenericNew;
   if (PyType_Ready(&PyAudioStream_Type) < 0) {
      MODINIT_ERROR;
   }
   PyVideoStream_Type.tp_new = PyType_GenericNew;
   if (PyType_Ready(&PyVideoStream_Type) < 0) {
      MODINIT_ERROR;
   }*/
   // Create the module
   
   module = Py_InitModule3 ("_movie", NULL, _movie_doc); //movie doc needed

   if (module == NULL) {
      return;
   }

	
   
   //Register all the fun stuff for movies.
   avcodec_register_all();
   av_register_all();
   //initialize lookup tables for YUV-to-RGB conversion
   initializeLookupTables();
   //import stuff we need
   import_pygame_surface();
   //initialize our flush marker for the queues.
   av_init_packet(&flush_pkt);
   uint8_t *s = (uint8_t *)"FLUSH";
   flush_pkt.data= s;
   
   // Add the type to the module.
   Py_INCREF(&PyMovie_Type);
   PyModule_AddObject(module, "Movie", (PyObject*)&PyMovie_Type);
}

