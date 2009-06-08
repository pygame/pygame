#include "gmovie.h"


 PyMovie*  _movie_init_internal(PyMovie *self, const char *filename, SDL_Surface *surf)
{
	Py_INCREF(self);
	//already malloced memory for PyMovie.
	if(!surf)
	{
		//set overlay to true
		self->overlay = 1;
	}
	else
	{
		self->overlay = 0;
		self->dest_surface=surf;
	}
	self->start_time = AV_NOPTS_VALUE;
	AVInputFormat *iformat;
	self=stream_open(self, filename, iformat);
	if(!self)
	{
		PyErr_SetString(PyExc_IOError, "stream_open failed");
        Py_DECREF(self);
        Py_RETURN_NONE;
    }	
	PySys_WriteStdout("Movie->filename: %s\n", self->filename);
	Py_DECREF(self);
	return self;
}

 int _movie_init(PyObject *self, PyObject *args, PyObject *kwds)
{
	Py_INCREF(self);
	const char *c;
	if (!PyArg_ParseTuple (args, "s", &c))
    {
        PyErr_SetString(PyExc_TypeError, "No valid arguments");
    	return -1;
    }	
	self = _movie_init_internal(self, c, NULL);
	PyObject *er;
    er = PyErr_Occurred();
    if(er)
    {
        PyErr_Print();
    }
    if(!self)
    {
        PyErr_SetString(PyExc_IOError, "No movie object created.");
        PyErr_Print();
        Py_DECREF(self);
        return -1;
    }
    Py_DECREF(self);
    PySys_WriteStdout("Returning from _movie_init\n");
    return 0;
}   

 void _movie_dealloc(PyMovie *movie)
{
    stream_close(movie);
    movie->ob_type->tp_free((PyObject *) movie);
}

 PyObject* _movie_repr (PyMovie *movie)
{
    /*Eventually add a time-code call */
    Py_INCREF(movie);
    char buf[100];
    //PySys_WriteStdout("_movie_repr: %10s\n", movie->filename); 
    PyOS_snprintf(buf, sizeof(buf), "(Movie: %s)", movie->filename);
    Py_DECREF(movie);
    return PyString_FromString(buf);
}
 PyObject* _movie_play(PyMovie *movie, PyObject* args)
{
	Py_INCREF(movie);
    int loops;
    if(!PyArg_ParseTuple(args, "i", &loops))
    {
    	loops = 0;
    }
    SDL_LockMutex(movie->dest_mutex);
    movie->loops =loops;
    movie->paused = 0;
    movie->playing = 1;
    SDL_UnlockMutex(movie->dest_mutex);
    while((loops-1)!=-1)
    {
    	decoder(movie);
    	PySys_WriteStdout("Loops: %i\n", loops);
    	loops--;
    	if(loops==1)
    	{
    		PySys_WriteStdout("Second Loop Around\n");
    	}
    	movie=stream_open(movie, movie->filename, NULL);
    	movie->paused=0;
    }
    Py_DECREF(movie);
    Py_RETURN_NONE;
}

 PyObject* _movie_stop(PyMovie *movie)
{
    Py_INCREF(movie);
    SDL_LockMutex(movie->dest_mutex);
    stream_pause(movie);
    movie->seek_req = 1;
    movie->seek_pos = 0;
    movie->seek_flags =AVSEEK_FLAG_BACKWARD;
    SDL_UnlockMutex(movie->dest_mutex);  
    Py_DECREF(movie);
    Py_RETURN_NONE;
}  

 PyObject* _movie_pause(PyMovie *movie)
{
    stream_pause(movie); 
    Py_RETURN_NONE;
}

 PyObject* _movie_rewind(PyMovie *movie, PyObject* args)
{
    /* For now, just alias rewind to stop */
    return _movie_stop(movie);
}

 PyObject* _movie_get_paused (PyMovie *movie, void *closure)
{
    return PyInt_FromLong((long)movie->paused);
}
PyObject* _movie_get_playing (PyMovie *movie, void *closure)
{
    PyObject *pyo;
    pyo= PyInt_FromLong((long)movie->playing);
    return pyo;
}

PyObject* _movie_get_width (PyMovie *movie, void *closure)
{
    PyObject *pyo;
    pyo= PyInt_FromLong((long)movie->width);
    return pyo;
}

PyObject* _movie_get_height (PyMovie *movie, void *closure)
{
    PyObject *pyo;
    pyo= PyInt_FromLong((long)movie->height);
    return pyo;
}

 static PyMethodDef _movie_methods[] = {
   { "play",    (PyCFunction) _movie_play, METH_VARARGS,
               "Play the movie file from current time-mark. If loop<0, then it will loop infinitely. If there is no loop value, then it will play once." },
   { "stop", (PyCFunction) _movie_stop, METH_NOARGS,
                "Stop the movie, and set time-mark to 0:0"},
   { "pause", (PyCFunction) _movie_pause, METH_NOARGS,
                "Pause movie."},
   { "rewind", (PyCFunction) _movie_rewind, METH_VARARGS,
                "Rewind movie to time_pos. If there is no time_pos, same as stop."},
   { NULL, NULL, 0, NULL }
};

 static PyGetSetDef _movie_getsets[] =
{
    { "paused", (getter) _movie_get_paused, NULL, NULL, NULL },
    { "playing", (getter) _movie_get_playing, NULL, NULL, NULL },
    { "height", (getter) _movie_get_height, NULL, NULL, NULL },
    { "width", (getter) _movie_get_width, NULL, NULL, NULL },
    { NULL, NULL, NULL, NULL, NULL }
};

 static PyTypeObject PyMovie_Type =
{
    PyObject_HEAD_INIT(NULL)
    0, 
    "pygame.gmovie.Movie",          /* tp_name */
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
    0,                          /* tp_doc */
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    _movie_methods,             /* tp_methods */
    0,                          /* tp_members */
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


PyMODINIT_FUNC
initgmovie(void)
{
    PyObject* module;

    /* imported needed apis; Do this first so if there is an error
       the module is not loaded.
    */
    //import_pygame_base ();
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
   
   module = Py_InitModule3 ("gmovie", NULL, "pygame.gmovie plays movies and streams."); //movie doc needed

   if (module == NULL) {
      return;
   }

	
   
   //Register all the fun stuff for movies.
   avcodec_register_all();
   avdevice_register_all();
   av_register_all();

   av_init_packet(&flush_pkt);
   uint8_t *s = (uint8_t *)"FLUSH";
   flush_pkt.data= s;
   
   

   // Add the type to the module.
   Py_INCREF(&PyMovie_Type);
   //Py_INCREF(&PyAudioStream_Type);
   //Py_INCREF(&PyVideoStream_Type);
   PyModule_AddObject(module, "Movie", (PyObject*)&PyMovie_Type);
   //PyModule_AddObject(module, "AudioStream", (PyObject *)&PyAudioStream_Type);
   //PyModule_AddObject(module, "VideoStream", (PyObject *)&PyVideoStream_Type);
}

