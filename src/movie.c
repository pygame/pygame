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
 *  movie playback for pygame
 */
#include "pygame.h"
#include "pygamedocs.h"
#include "smpeg.h"


typedef struct {
  PyObject_HEAD
  SMPEG* movie;
  PyObject* surftarget;
  PyObject* filesource;
} PyMovieObject;
#define PyMovie_AsSMPEG(x) (((PyMovieObject*)x)->movie)


staticforward PyTypeObject PyMovie_Type;
static PyObject* PyMovie_New(SMPEG*);
#define PyMovie_Check(x) ((x)->ob_type == &PyMovie_Type)





/* movie object methods */

static PyObject* movie_play(PyObject* self, PyObject* args)
{
	SMPEG* movie = PyMovie_AsSMPEG(self);
	int loops=0;
	if(!PyArg_ParseTuple(args, "|i", &loops))
		return NULL;
        Py_BEGIN_ALLOW_THREADS
	SMPEG_loop(movie, loops);
	SMPEG_play(movie);
        Py_END_ALLOW_THREADS
	RETURN_NONE
}


static PyObject* movie_stop(PyObject* self, PyObject* args)
{
	SMPEG* movie = PyMovie_AsSMPEG(self);
	if(!PyArg_ParseTuple(args, ""))
		return NULL;
        Py_BEGIN_ALLOW_THREADS
	SMPEG_stop(movie);
        Py_END_ALLOW_THREADS
	RETURN_NONE
}


static PyObject* movie_pause(PyObject* self, PyObject* args)
{
	SMPEG* movie = PyMovie_AsSMPEG(self);
	if(!PyArg_ParseTuple(args, ""))
		return NULL;
        Py_BEGIN_ALLOW_THREADS
	SMPEG_pause(movie);
        Py_END_ALLOW_THREADS
	RETURN_NONE
}


static PyObject* movie_rewind(PyObject* self, PyObject* args)
{
	SMPEG* movie = PyMovie_AsSMPEG(self);
	if(!PyArg_ParseTuple(args, ""))
		return NULL;
        Py_BEGIN_ALLOW_THREADS
	SMPEG_rewind(movie);
        Py_END_ALLOW_THREADS
	RETURN_NONE
}


static PyObject* movie_skip(PyObject* self, PyObject* args)
{
	SMPEG* movie = PyMovie_AsSMPEG(self);
	float seconds;
	if(!PyArg_ParseTuple(args, "f", &seconds))
		return NULL;
        Py_BEGIN_ALLOW_THREADS
	SMPEG_skip(movie, seconds);
        Py_END_ALLOW_THREADS
	RETURN_NONE
}


static PyObject* movie_set_volume(PyObject* self, PyObject* args)
{
	SMPEG* movie = PyMovie_AsSMPEG(self);
	float value;
	int volume;
	if(!PyArg_ParseTuple(args, "f", &value))
		return NULL;

        Py_BEGIN_ALLOW_THREADS
	volume = (int)(value * 100);
	if(volume<0) volume = 0;
	if(volume>100) volume = 100;
	SMPEG_setvolume(movie, volume);
        Py_END_ALLOW_THREADS

	RETURN_NONE
}


static PyObject* movie_set_display(PyObject* self, PyObject* args)
{
	SMPEG* movie = PyMovie_AsSMPEG(self);
	PyObject* surfobj, *posobj=NULL;
	GAME_Rect *rect, temp;
	int x=0, y=0;
	if(!PyArg_ParseTuple(args, "O|O", &surfobj, &posobj))
		return NULL;

	Py_XDECREF(((PyMovieObject*)self)->surftarget);
	((PyMovieObject*)self)->surftarget = NULL;

	if(PySurface_Check(surfobj))
	{
	    SMPEG_Info info;
	    SDL_Surface* surf;

		if(posobj == NULL)
		{
			SMPEG_Info info;
			SMPEG_getinfo(movie, &info);
			SMPEG_scaleXY(movie, info.width, info.height);
			x = y = 0;
		}
		else if(TwoIntsFromObj(posobj, &x, &y))
		{
			SMPEG_Info info;
			SMPEG_getinfo(movie, &info);
			SMPEG_scaleXY(movie, info.width, info.height);
		}
		else if((rect = GameRect_FromObject(posobj, &temp)))
		{
			x = rect->x;
			y = rect->y;
			SMPEG_scaleXY(movie, rect->w, rect->h);
		}
		else
			return RAISE(PyExc_TypeError, "Invalid position argument");

	    surf = PySurface_AsSurface(surfobj);

            SMPEG_getinfo(movie, &info);
	    SMPEG_enablevideo(movie, 1);
	    SMPEG_setdisplay(movie, surf, NULL, NULL);
	    SMPEG_move(movie, x, y);
	}
	else
	{
            Py_BEGIN_ALLOW_THREADS
	    SMPEG_enablevideo(movie, 0);
            Py_END_ALLOW_THREADS
	    if(surfobj != Py_None)
		       return RAISE(PyExc_TypeError, "destination must be a Surface");
	}

	RETURN_NONE;
}


static PyObject* movie_has_video(PyObject* self, PyObject* args)
{
	SMPEG* movie = PyMovie_AsSMPEG(self);
	SMPEG_Info info;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

        Py_BEGIN_ALLOW_THREADS
	SMPEG_getinfo(movie, &info);
        Py_END_ALLOW_THREADS
	return PyInt_FromLong(info.has_video);
}


static PyObject* movie_has_audio(PyObject* self, PyObject* args)
{
	SMPEG* movie = PyMovie_AsSMPEG(self);
	SMPEG_Info info;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

        Py_BEGIN_ALLOW_THREADS
	SMPEG_getinfo(movie, &info);
        Py_END_ALLOW_THREADS
	return PyInt_FromLong(info.has_audio);
}


static PyObject* movie_get_size(PyObject* self, PyObject* args)
{
	SMPEG* movie = PyMovie_AsSMPEG(self);
	SMPEG_Info info;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

        Py_BEGIN_ALLOW_THREADS
	SMPEG_getinfo(movie, &info);
        Py_END_ALLOW_THREADS
	return Py_BuildValue("(ii)", info.width, info.height);
}


static PyObject* movie_get_frame(PyObject* self, PyObject* args)
{
	SMPEG* movie = PyMovie_AsSMPEG(self);
	SMPEG_Info info;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

        Py_BEGIN_ALLOW_THREADS
	SMPEG_getinfo(movie, &info);
        Py_END_ALLOW_THREADS
	return PyInt_FromLong(info.current_frame);
}


static PyObject* movie_get_time(PyObject* self, PyObject* args)
{
	SMPEG* movie = PyMovie_AsSMPEG(self);
	SMPEG_Info info;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

        Py_BEGIN_ALLOW_THREADS
	SMPEG_getinfo(movie, &info);
        Py_END_ALLOW_THREADS
	return PyFloat_FromDouble(info.current_time);
}


static PyObject* movie_get_length(PyObject* self, PyObject* args)
{
	SMPEG* movie = PyMovie_AsSMPEG(self);
	SMPEG_Info info;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

        Py_BEGIN_ALLOW_THREADS
	SMPEG_getinfo(movie, &info);
        Py_END_ALLOW_THREADS
	return PyFloat_FromDouble(info.total_time);
}


static PyObject* movie_get_busy(PyObject* self, PyObject* args)
{
	SMPEG* movie = PyMovie_AsSMPEG(self);

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	return PyInt_FromLong(SMPEG_status(movie) == SMPEG_PLAYING);
}


static PyObject* movie_render_frame(PyObject* self, PyObject* args)
{
        SMPEG* movie = PyMovie_AsSMPEG(self);
        SMPEG_Info info;
        int framenum;

        if(!PyArg_ParseTuple(args, "i", &framenum))
                return NULL;
        Py_BEGIN_ALLOW_THREADS
        SMPEG_renderFrame(movie, framenum);
        SMPEG_getinfo(movie, &info);
        Py_END_ALLOW_THREADS
        return PyInt_FromLong(info.current_frame);
}


static PyMethodDef movie_builtins[] =
{
	{ "play", movie_play, 1, DOC_MOVIEPLAY },
	{ "stop", movie_stop, 1, DOC_MOVIESTOP },
	{ "pause", movie_pause, 1, DOC_MOVIEPAUSE },
	{ "rewind", movie_rewind, 1, DOC_MOVIEREWIND },
	{ "skip", movie_skip, 1, DOC_MOVIESKIP },

	{ "set_volume", movie_set_volume, 1, DOC_MOVIESETVOLUME },
	{ "set_display", movie_set_display, 1, DOC_MOVIESETDISPLAY },

	{ "has_video", movie_has_video, 1, DOC_MOVIEHASVIDEO },
	{ "has_audio", movie_has_audio, 1, DOC_MOVIEHASAUDIO },
	{ "get_size", movie_get_size, 1, DOC_MOVIEGETSIZE },
	{ "get_frame", movie_get_frame, 1, DOC_MOVIEGETFRAME },
	{ "get_time", movie_get_time, 1, DOC_MOVIEGETTIME },
	{ "get_length", movie_get_length, 1, DOC_MOVIEGETLENGTH },
	{ "get_busy", movie_get_busy, 1, DOC_MOVIEGETBUSY },
    { "render_frame", movie_render_frame, 1, DOC_MOVIERENDERFRAME },

	{ NULL, NULL }
};


/*sound object internals*/

static void movie_dealloc(PyObject* self)
{
	SMPEG* movie = PyMovie_AsSMPEG(self);
        Py_BEGIN_ALLOW_THREADS
	SMPEG_delete(movie);
        Py_END_ALLOW_THREADS
	Py_XDECREF(((PyMovieObject*)self)->surftarget);
	Py_XDECREF(((PyMovieObject*)self)->filesource);
	PyObject_DEL(self);
}


static PyObject* movie_getattr(PyObject* self, char* attrname)
{
	return Py_FindMethod(movie_builtins, self, attrname);
}


static PyTypeObject PyMovie_Type =
{
	PyObject_HEAD_INIT(NULL)
	0,
	"Movie",
	sizeof(PyMovieObject),
	0,
	movie_dealloc,
	0,
	movie_getattr,
	NULL,					/*setattr*/
	NULL,					/*compare*/
	NULL,					/*repr*/
	NULL,					/*as_number*/
	NULL,					/*as_sequence*/
	NULL,					/*as_mapping*/
	(hashfunc)NULL, 		/*hash*/
	(ternaryfunc)NULL,		/*call*/
	(reprfunc)NULL, 		/*str*/
	0L,0L,0L,0L,
	DOC_PYGAMEMOVIEMOVIE /* Documentation string */
};




/*movie module methods*/

static PyObject* Movie(PyObject* self, PyObject* arg)
{
	PyObject* file, *final, *filesource=NULL;
	char* name = NULL;
	SMPEG* movie=NULL;
	SMPEG_Info info;
	SDL_Surface* screen;
	char* error;
	int audioavail = 0;
	if(!PyArg_ParseTuple(arg, "O", &file))
		return NULL;

	if(!SDL_WasInit(SDL_INIT_AUDIO))
		audioavail = 1;

	if(PyString_Check(file) || PyUnicode_Check(file))
	{
		if(!PyArg_ParseTuple(arg, "s", &name))
			return NULL;
		movie = SMPEG_new(name, &info, audioavail);
	}
	else if(PyFile_Check(file))
	{
		SDL_RWops *rw = SDL_RWFromFP(PyFile_AsFile(file), 0);
		movie = SMPEG_new_rwops(rw, &info, audioavail);
		filesource = file;
		Py_INCREF(file);
	}
	else
	{
		SDL_RWops *rw;
                if(!(rw = RWopsFromPythonThreaded(file)))
			return NULL;
                Py_BEGIN_ALLOW_THREADS
		movie = SMPEG_new_rwops(rw, &info, audioavail);
                Py_END_ALLOW_THREADS
	}

	if(!movie)
		return RAISE(PyExc_SDLError, "Cannot create Movie object");

	error = SMPEG_error(movie);
	if(error)
	{
/* while this would seem correct, it causes a crash, so don't delete */
/*	    SMPEG_delete(movie);*/
	    return RAISE(PyExc_SDLError, error);
	}

        Py_BEGIN_ALLOW_THREADS
	SMPEG_enableaudio(movie, audioavail);

	screen = SDL_GetVideoSurface();
	if(screen)
		SMPEG_setdisplay(movie, screen, NULL, NULL);

	SMPEG_scaleXY(movie, info.width, info.height);
        Py_END_ALLOW_THREADS

	final = PyMovie_New(movie);
	if(!final)
		SMPEG_delete(movie);
	((PyMovieObject*)final)->filesource = filesource;

	return final;
}



static PyMethodDef mixer_builtins[] =
{
	{ "Movie", Movie, 1, DOC_PYGAMEMOVIEMOVIE },

	{ NULL, NULL }
};


static PyObject* PyMovie_New(SMPEG* movie)
{
	PyMovieObject* movieobj;

	if(!movie)
		return RAISE(PyExc_RuntimeError, "unable to create movie.");

	movieobj = PyObject_NEW(PyMovieObject, &PyMovie_Type);
	if(movieobj)
		movieobj->movie = movie;

	movieobj->surftarget = NULL;
	movieobj->filesource = NULL;

	return (PyObject*)movieobj;
}


PYGAME_EXPORT
void initmovie(void)
{
	PyObject *module, *dict;

	PyType_Init(PyMovie_Type);

	/* create the module */
	module = Py_InitModule3("movie", mixer_builtins, DOC_PYGAMEMOVIE);
	dict = PyModule_GetDict(module);

	PyDict_SetItemString(dict, "MovieType", (PyObject *)&PyMovie_Type);

	/*imported needed apis*/
	import_pygame_base();
	import_pygame_surface();
	import_pygame_rwobject();
	import_pygame_rect();
}

