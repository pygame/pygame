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

    /*DOC*/ static char doc_movie_play[] =
    /*DOC*/    "Movie.play(loops=0) -> None\n"
    /*DOC*/    "start movie playback\n"
    /*DOC*/    "\n"
    /*DOC*/    "Starts playback of a movie. If audio or video is enabled\n"
    /*DOC*/    "for the Movie, those outputs will be created. \n"
    /*DOC*/    "\n"
    /*DOC*/    "You can specify an optional argument which will be the\n"
    /*DOC*/    "number of times the movie loops while playing.\n"
    /*DOC*/ ;

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



    /*DOC*/ static char doc_movie_stop[] =
    /*DOC*/    "Movie.stop() -> None\n"
    /*DOC*/    "stop movie playback\n"
    /*DOC*/    "\n"
    /*DOC*/    "Stops playback of a movie. If sound and video are being\n"
    /*DOC*/    "rendered, both will be stopped at their current position.\n"
    /*DOC*/ ;

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


    /*DOC*/ static char doc_movie_pause[] =
    /*DOC*/    "Movie.pause() -> None\n"
    /*DOC*/    "pause/resume movie playback\n"
    /*DOC*/    "\n"
    /*DOC*/    "This will temporarily stop playback of the movie. When called\n"
    /*DOC*/    "a second time, playback will resume where it left off.\n"
    /*DOC*/ ;

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


    /*DOC*/ static char doc_movie_rewind[] =
    /*DOC*/    "Movie.rewind() -> None\n"
    /*DOC*/    "set playback position to the beginning of the movie\n"
    /*DOC*/    "\n"
    /*DOC*/    "Sets the movie playback position to the start of\n"
    /*DOC*/    "the movie.\n"
    /*DOC*/ ;

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

    /*DOC*/ static char doc_movie_skip[] =
    /*DOC*/    "Movie.skip(seconds) -> None\n"
    /*DOC*/    "skip the movie playback position forward\n"
    /*DOC*/    "\n"
    /*DOC*/    "Sets the movie playback position ahead by the given\n"
    /*DOC*/    "amount of seconds. the seconds value is a floating\n"
    /*DOC*/    "point value\n"
    /*DOC*/ ;

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


    /*DOC*/ static char doc_movie_set_volume[] =
    /*DOC*/    "Movie.set_volume(val) -> None\n"
    /*DOC*/    "change volume for sound\n"
    /*DOC*/    "\n"
    /*DOC*/    "Set the play volume for this Movie. The volume value is between\n"
    /*DOC*/    "0.0 and 1.0.\n"
    /*DOC*/ ;

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


    /*DOC*/ static char doc_movie_set_display[] =
    /*DOC*/    "Movie.set_display(Surface, [pos]) -> None\n"
    /*DOC*/    "change the video output surface\n"
    /*DOC*/    "\n"
    /*DOC*/    "Set the output surface for the Movie's video. You may\n"
    /*DOC*/    "also specify a position for the topleft corner of the\n"
    /*DOC*/    "video. The position defaults to (0,0) if not given.\n"
    /*DOC*/    "\n"
    /*DOC*/    "The position argument can optionally be a rectangle,\n"
    /*DOC*/    "in which case the video will be stretched to fill the\n"
    /*DOC*/    "rectangular area.\n"
    /*DOC*/    "\n"
    /*DOC*/    "You may also pass None as the destination Surface, and\n"
    /*DOC*/    "no video will be rendered for the movie playback.\n"
    /*DOC*/ ;

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


    /*DOC*/ static char doc_movie_has_video[] =
    /*DOC*/    "Movie.has_video() -> bool\n"
    /*DOC*/    "query if movie stream has video\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns a true value when the Movie object has a valid\n"
    /*DOC*/    "video stream.\n"
    /*DOC*/ ;


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

    /*DOC*/ static char doc_movie_has_audio[] =
    /*DOC*/    "Movie.has_audio() -> bool\n"
    /*DOC*/    "query if movie stream has audio\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns a true value when the Movie object has a valid\n"
    /*DOC*/    "audio stream.\n"
    /*DOC*/ ;

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

    /*DOC*/ static char doc_movie_get_size[] =
    /*DOC*/    "Movie.get_size() -> width,height\n"
    /*DOC*/    "query the size of the video image\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the size of the video image the mpeg provides.\n"
    /*DOC*/ ;

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

    /*DOC*/ static char doc_movie_get_frame[] =
    /*DOC*/    "Movie.get_frame() -> int\n"
    /*DOC*/    "query the current frame in the movie\n"
    /*DOC*/    "\n"
    /*DOC*/    "Gets the current video frame number for the movie.\n"
    /*DOC*/ ;

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

    /*DOC*/ static char doc_movie_get_time[] =
    /*DOC*/    "Movie.get_time() -> float\n"
    /*DOC*/    "query the current time in the movie\n"
    /*DOC*/    "\n"
    /*DOC*/    "Gets the current time (in seconds) for the movie.\n"
    /*DOC*/    "(currently not working? SMPEG always reports 0)\n"
    /*DOC*/ ;

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

    /*DOC*/ static char doc_movie_get_length[] =
    /*DOC*/    "Movie.get_length() -> float\n"
    /*DOC*/    "query playback time of the movie\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the total time (in seconds) of the movie.\n"
    /*DOC*/ ;

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

    /*DOC*/ static char doc_movie_get_busy[] =
    /*DOC*/    "Movie.get_busy() -> bool\n"
    /*DOC*/    "query the playback state\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns true if the movie is currently playing.\n"
    /*DOC*/ ;

static PyObject* movie_get_busy(PyObject* self, PyObject* args)
{
	SMPEG* movie = PyMovie_AsSMPEG(self);

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	return PyInt_FromLong(SMPEG_status(movie) == SMPEG_PLAYING);
}



    /*DOC*/ static char doc_movie_render_frame[] =
    /*DOC*/    "Movie.render_frame(framenum) -> int\n"
    /*DOC*/    "Render a specfic numbered frame.\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the current frame number.\n"
    /*DOC*/ ;

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
	{ "play", movie_play, 1, doc_movie_play },
	{ "stop", movie_stop, 1, doc_movie_stop },
	{ "pause", movie_pause, 1, doc_movie_pause },
	{ "rewind", movie_rewind, 1, doc_movie_rewind },
	{ "skip", movie_skip, 1, doc_movie_skip },

	{ "set_volume", movie_set_volume, 1, doc_movie_set_volume },
	{ "set_display", movie_set_display, 1, doc_movie_set_display },

	{ "has_video", movie_has_video, 1, doc_movie_has_video },
	{ "has_audio", movie_has_audio, 1, doc_movie_has_audio },
	{ "get_size", movie_get_size, 1, doc_movie_get_size },
	{ "get_frame", movie_get_frame, 1, doc_movie_get_frame },
	{ "get_time", movie_get_time, 1, doc_movie_get_time },
	{ "get_length", movie_get_length, 1, doc_movie_get_length },
	{ "get_busy", movie_get_busy, 1, doc_movie_get_busy },
        { "render_frame", movie_render_frame, 1, doc_movie_render_frame},

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


    /*DOC*/ static char doc_Movie_MODULE[] =
    /*DOC*/    "The Movie object represents an opened MPEG file.\n"
    /*DOC*/    "You control playback similar to a Sound object.\n"
    /*DOC*/    "\n"
    /*DOC*/    "Movie objects have a target display Surface.\n"
    /*DOC*/    "The movie is rendered to this Surface in a background\n"
    /*DOC*/    "thread. If the Surface is the display surface, and\n"
    /*DOC*/    "the system supports it, the movie will render into a\n"
    /*DOC*/    "Hardware YUV overlay plane. If you don't set a display\n"
    /*DOC*/    "Surface, it will default to the display Surface.\n"
    /*DOC*/    "\n"
    /*DOC*/    "Movies are played back in background threads, so there\n"
    /*DOC*/    "is very little management needed on the user end. Just\n"
    /*DOC*/    "load the Movie, set the destination, and Movie.play()\n"
    /*DOC*/    "\n"
    /*DOC*/    "Movies will only playback audio if the pygame.mixer\n"
    /*DOC*/    "module is not initialized. It is easy to temporarily\n"
    /*DOC*/    "call pygame.mixer.quit() to disable audio, then create\n"
    /*DOC*/    "and play your movie. Finally calling pygame.mixer.init()\n"
    /*DOC*/    "again when finished with the Movie.\n"
    /*DOC*/    "\n"
    /*DOC*/    "NOTE: When disabling the mixer so a movie may play audio,\n"
    /*DOC*/    "you must disable the audio before calling pygame.movie.Movie\n"
    /*DOC*/    "or the movie will not realise that it may access the audio.\n"
    /*DOC*/    "Before reinitialising the mixer, You must remove all\n"
    /*DOC*/    "references to the movie before calling pygame.mixer.init()\n"
    /*DOC*/    "or the init will fail, leading to errors when you attempt to\n"
    /*DOC*/    "use the mixer.\n"
    /*DOC*/    "\n"
    /*DOC*/    "eg.\n"
    /*DOC*/    "pygame.mixer.quit()\n"
    /*DOC*/    "movie=pygame.movie.Movie(\"my.mpg\")\n"
    /*DOC*/    "movie.play()\n"
    /*DOC*/    "# process events until movie finished here\n"
    /*DOC*/    "movie.stop()\n"
    /*DOC*/    "movie=None # if you don't do this bit the init will fail\n"
    /*DOC*/    "pygame.mixer.init()\n"
    /*DOC*/ ;

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
	doc_Movie_MODULE /* Documentation string */
};




/*movie module methods*/


    /*DOC*/ static char doc_Movie[] =
    /*DOC*/    "pygame.movie.Movie(file) -> Movie\n"
    /*DOC*/    "load a new MPEG stream\n"
    /*DOC*/    "\n"
    /*DOC*/    "Loads a new movie stream from a MPEG file. The file\n"
    /*DOC*/    "argument is either a filename, or any python file-like object\n"
    /*DOC*/ ;

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
	{ "Movie", Movie, 1, doc_Movie },

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



    /*DOC*/ static char doc_pygame_movie_MODULE[] =
    /*DOC*/    "The movie module is an optional pygame module that\n"
    /*DOC*/    "allows for decoding and playback of MPEG movie files.\n"
    /*DOC*/    "The module only contains a single function, Movie()\n"
    /*DOC*/    "which creates a new Movie object.\n"
    /*DOC*/    "\n"
    /*DOC*/    "Movies are played back in background threads, so there\n"
    /*DOC*/    "is very little management needed on the user end. Just\n"
    /*DOC*/    "load the Movie, set the destination, and Movie.play()\n"
    /*DOC*/    "\n"
    /*DOC*/    "Movies will only playback audio if the pygame.mixer\n"
    /*DOC*/    "module is not initialized. It is easy to temporarily\n"
    /*DOC*/    "call pygame.mixer.quit() to disable audio, then create\n"
    /*DOC*/    "and play your movie. Finally calling pygame.mixer.init()\n"
    /*DOC*/    "again when finished with the Movie.\n"
    /*DOC*/ ;

PYGAME_EXPORT
void initmovie(void)
{
	PyObject *module, *dict;

	PyType_Init(PyMovie_Type);

	/* create the module */
	module = Py_InitModule3("movie", mixer_builtins, doc_pygame_movie_MODULE);
	dict = PyModule_GetDict(module);

	PyDict_SetItemString(dict, "MovieType", (PyObject *)&PyMovie_Type);

	/*imported needed apis*/
	import_pygame_base();
	import_pygame_surface();
	import_pygame_rwobject();
	import_pygame_rect();
}

