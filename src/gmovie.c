#include "gmovie.h"

void _movie_init_internal(PyMovie *self, const char *filename, SDL_Surface *surf)
{
    Py_INCREF(self);
    //already malloced memory for PyMovie.
    self->_backend="FFMPEG_WRAPPER";
    self->commands = (CommandQueue *)PyMem_Malloc(sizeof(CommandQueue));
    self->commands->q_mutex = SDL_CreateMutex();
	self->commands->size=0;
	self->commands->reg_ix=0;
	registerCommands(self);
    self->commands->size=0;
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
    /* filename checking... */
    PyObject *path = PyImport_ImportModule("os.path");
    Py_INCREF(path);
    PyObject *dict = PyModule_GetDict(path);
    PyObject *key = Py_BuildValue("s", "exists");
    Py_INCREF(key);
	PyObject *existsFunc  = PyDict_GetItem(dict, key);
	PyObject *boolean = PyObject_CallFunction(existsFunc, "s", filename);
	Py_INCREF(boolean);
	if(boolean==Py_False)
	{
		Py_DECREF(boolean);
		Py_DECREF(key);
		Py_DECREF(path);
		RAISE(PyExc_ValueError, "This filename does not exist.");
		return;
		//Py_RETURN_NONE;
	}
	Py_DECREF(boolean);
	Py_DECREF(key);
	Py_DECREF(path);    
    
    self->start_time = AV_NOPTS_VALUE;
    stream_open(self, filename, NULL, 0);
    //PySys_WriteStdout("Time-base-a: %f\n", av_q2d(self->audio_st->codec->time_base));
    if(!self)
    {
        PyErr_SetString(PyExc_IOError, "stream_open failed");
        Py_DECREF(self);
        return;
    }
	if(self->canon_surf)
	{
		/*Here we check if the surface's dimensions match the aspect ratio of the video. If not, 
		 * we throw an error. 
		 */
		int width = self->canon_surf->w;
		int height = self->canon_surf->h;
		double aspect_ratio = (double)self->video_st->codec->width/(double)self->video_st->codec->height;
		double surf_ratio = (double)width/(double)height;
		if (surf_ratio!=aspect_ratio)
		{
			PyErr_SetString(PyExc_ValueError, "surface does not have the same aspect ratio as the video. This would cause surface corruption.");
			Py_DECREF(self);
			return;
		}
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
	//PySys_WriteStdout("Deleting\n");
	if(movie->_tstate)
	{
		PyThreadState_Clear(movie->_tstate);
		//PyThreadState_Delete(movie->_tstate);
	}
	#ifdef PROFILE
		TimeSampleNode *cur = movie->istats->first;
		TimeSampleNode *prev;
		while(cur!=NULL) 
    	{
    		prev=cur;
    		cur=cur->next;
			PyMem_Free(prev);
    	}
    	PyMem_Free(movie->istats);
    #endif
    SDL_DestroyMutex(movie->commands->q_mutex);
    PyMem_Free(movie->commands);
    movie->commands=NULL;
	stream_close(movie, 0);
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
    Py_DECREF(movie);
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
    else if(movie->_tstate==NULL)
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
    	//first we release the GIL, then we release all the resources associated with the streams, if they exist.
    	PyEval_ReleaseLock();
		_movie_stop(movie);
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
    //stream_pause(movie);
    ALLOC_COMMAND(stopCommand, stop);
    addCommand(movie->commands, (Command *)stop);
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
    if(movie->canon_surf)
    {
    	Py_RETURN_NONE;
    }
    ALLOC_COMMAND(resizeCommand, resize);
    
    resize->h = h;
    resize->w  = w;
    addCommand(movie->commands, (Command *)resize);
    Py_RETURN_NONE;

}

PyObject *_movie_shift(PyMovie *movie, PyObject*args)
{
	int x=0;
	int y=0;
	if(PyArg_ParseTuple(args, "ii", &x, &y)<0)
	{
		return NULL;	
	}
	ALLOC_COMMAND(shiftCommand, shift);
	shift->xleft=x;
	shift->ytop=y;
	addCommand(movie->commands, (Command *)shift);
	Py_RETURN_NONE;
}

PyObject* _movie_easy_seek (PyMovie *movie, PyObject* args)
{
	int64_t pos=0;
	int hour=0;
	int minute=0;
	int second=0;
	int reverse=0;
	//int relative=0;
	//char *keywords[4] = {"second", "minute", "hour", "reverse"};
	if(!PyArg_ParseTuple(args, "iiii", &second, &minute, &hour, &reverse))
	{
		Py_RETURN_NONE;
	}
	if(second)
	{
		pos+=(int64_t)second;
	}
	if(minute)
	{
		pos+=60*(int64_t)minute;
	}
	if(hour)
	{
		pos+=3600*(int64_t)hour;
	}
	if(reverse)
	{
		reverse=-1;
	}
	else
	{
		reverse=1;
	}
	pos *= AV_TIME_BASE;
	stream_seek(movie, pos, reverse);
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

PyObject* _movie_get_finished(PyMovie *movie,  void *closure)
{
	return PyBool_FromLong((long)movie->finished);	
}

PyObject* _movie_get_width (PyMovie *movie, void *closure)
{
    PyObject *pyo;
    if(movie->video_st)
    {
        if(movie->resize_w)
        {
            pyo = PyInt_FromLong((long)movie->width);
        }
        else
        {
            if(movie->video_st->codec)
            {
                pyo= PyInt_FromLong((long)movie->video_st->codec->width);
            }
            else
            {
                pyo = PyInt_FromLong((long)0);
            }
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
    if(PyInt_Check(width) && !movie->canon_surf)
    {
        w = (int)PyInt_AsLong(width);
        ALLOC_COMMAND(resizeCommand, resize);
    
    	resize->h = 0;
    	resize->w  = w;
		addCommand(movie->commands, (Command *)resize);
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
        if(movie->resize_h)
        {
            pyo=PyInt_FromLong((long)movie->height);
        }
        else
        {
            pyo= PyInt_FromLong((long)movie->video_st->codec->height);
        }
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
    if(PyInt_Check(height)  && !movie->canon_surf)
    {
        h = (int)PyInt_AsLong(height);
    	
        ALLOC_COMMAND(resizeCommand, resize);
    	resize->h = h;
    	resize->w  = 0;
		addCommand(movie->commands, (Command *)resize);
        return 0;
    }
    else
    {
        return -1;
    }

}

PyObject* _movie_get_ytop (PyMovie *movie, void *closure)
{
    PyObject *pyo;
    if(movie->video_st)
    {
        pyo= PyInt_FromLong((long)movie->ytop);
    }
    else
    {
        pyo = PyInt_FromLong((long)0);
    }
    return pyo;
}
int _movie_set_ytop (PyMovie *movie, PyObject *ytop, void *closure)
{
    int y;
    if(PyInt_Check(ytop))
    {
        y = (int)PyInt_AsLong(ytop);
        ALLOC_COMMAND(shiftCommand, shift);
        shift->ytop = y;
        shift->xleft = 0;
        addCommand(movie->commands, (Command *)shift);
        return 0;
    }
    else
    {
        return -1;
    }
}

PyObject* _movie_get_xleft (PyMovie *movie, void *closure)
{
    PyObject *pyo;
    if(movie->video_st)
    {
        pyo= PyInt_FromLong((long)movie->xleft);
    }
    else
    {
        pyo = PyInt_FromLong((long)0);
    }
    return pyo;
}
int _movie_set_xleft (PyMovie *movie, PyObject *xleft, void *closure)
{
    int x;
    if(PyInt_Check(xleft))
    {
        x = (int)PyInt_AsLong(xleft);
        ALLOC_COMMAND(shiftCommand, shift);
        shift->ytop = 0;
        shift->xleft = x;
        addCommand(movie->commands, (Command *)shift);
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
    surfaceCommand *surf;
    if(PySurface_Check(surface))
    {
		SDL_Surface *surfa = PySurface_AsSurface(surface);
        int width = surfa->w;
		int height = surfa->h;
		double aspect_ratio = (double)movie->video_st->codec->width/(double)movie->video_st->codec->height;
		double surf_ratio = (double)width/(double)height;
		if (surf_ratio!=aspect_ratio)
		{
			RAISE(PyExc_ValueError, "surface does not have the same aspect ratio as the video. This would cause surface corruption.");
			return -1;
		}
		/*movie->canon_surf=PySurface_AsSurface(surface);
        movie->overlay=0;*/
        surf = (surfaceCommand *)PyMem_Malloc(sizeof(surfaceCommand));
        surf->type = movie->surfaceCommandType;
        surf->surface = surfa;
        addCommand(movie->commands, (Command *)surf);
        return 0;
    }
    else if(surface == Py_None)
    {
    	surf = (surfaceCommand *)PyMem_Malloc(sizeof(surfaceCommand));
        surf->type = movie->surfaceCommandType;
        surf->surface = NULL;
        addCommand(movie->commands, (Command *)surf);
    	return 0;
    }
    return -1;
}

static PyMethodDef _movie_methods[] = {
                                          { "play",      (PyCFunction) _movie_play,      METH_VARARGS, DOC_GMOVIEMOVIEPLAY},
                                          { "stop",      (PyCFunction) _movie_stop,      METH_NOARGS,  DOC_GMOVIEMOVIESTOP},
                                          { "pause",     (PyCFunction) _movie_pause,     METH_NOARGS,  DOC_GMOVIEMOVIEPAUSE},
                                          { "rewind",    (PyCFunction) _movie_rewind,    METH_VARARGS, DOC_GMOVIEMOVIEREWIND},
                                          { "resize",    (PyCFunction) _movie_resize,    METH_VARARGS, DOC_GMOVIEMOVIERESIZE},
                                          { "easy_seek", (PyCFunction) _movie_easy_seek, METH_VARARGS, DOC_GMOVIEMOVIEEASY_SEEK},
                                          { "shift",     (PyCFunction) _movie_shift,     METH_VARARGS, DOC_GMOVIEMOVIESHIFT},
                                          { NULL,     NULL,                        0,            NULL }
                                      };

static PyMemberDef _movie_members[] = {
                                          { "_backend", T_STRING, offsetof(struct PyMovie, _backend), 0,    "Lists which backend this movie object belongs to."},
                                          { NULL}
                                      };

static PyGetSetDef _movie_getsets[] =
    {
        { "paused",   (getter) _movie_get_paused,   NULL,                       DOC_GMOVIEMOVIEPAUSE,    NULL },
        { "playing",  (getter) _movie_get_playing,  NULL,                       DOC_GMOVIEMOVIEPLAYING,  NULL },
        { "finished", (getter) _movie_get_finished, NULL,                       DOC_GMOVIEMOVIEFINISHED, NULL },
        { "height",   (getter) _movie_get_height,  (setter) _movie_set_height,  DOC_GMOVIEMOVIEHEIGHT,   NULL },
        { "width",    (getter) _movie_get_width,   (setter) _movie_set_width,   DOC_GMOVIEMOVIEWIDTH,    NULL },
        { "surface",  (getter) _movie_get_surface, (setter) _movie_set_surface, DOC_GMOVIEMOVIESURFACE,  NULL },
        { "ytop",     (getter) _movie_get_ytop,    (setter) _movie_set_ytop,    DOC_GMOVIEMOVIEYTOP,     NULL },
        { "xleft",    (getter) _movie_get_xleft,   (setter) _movie_set_xleft,   DOC_GMOVIEMOVIEXLEFT,    NULL },
        { NULL,       NULL,                        NULL,                        NULL,                    NULL }
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
/*DOC*/
static char _movie_doc[] =
    /*DOC*/    "ffmpeg wrapper module for pygame";

PyMODINIT_FUNC
init_movie(void)
{
    PyObject* module;

    /* imported needed apis; Do this first so if there is an error
       the module is not loaded.
    */
    import_pygame_base ();
    if (PyErr_Occurred ())
    {
        MODINIT_ERROR;
    }

    // Fill in some slots in the type, and make it ready
    PyMovie_Type.tp_new = PyType_GenericNew;
    if (PyType_Ready(&PyMovie_Type) < 0)
    {
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

    if (module == NULL)
    {
        return;
    }



    //Register all the fun stuff for movies.
    avcodec_register_all();
    av_register_all();
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

