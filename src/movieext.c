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
#include "ffmovie.h"



typedef struct {
  PyObject_HEAD
  FFMovie* movie;
  PyObject* surftarget;
} PyMovieObject;
#define PyMovie_AsFFMovie(x) (((PyMovieObject*)x)->movie)


staticforward PyTypeObject PyMovie_Type;
static PyObject* PyMovie_New(FFMovie*);
#define PyMovie_Check(x) ((x)->ob_type == &PyMovie_Type)



static void autoquit(void)
{
    ffmovie_abortall();
}


/* movie object methods */

    /*DOC*/ static char doc_movie_play[] =
    /*DOC*/    "Movie.play() -> None\n"
    /*DOC*/    "start movie playback\n"
    /*DOC*/    "\n"
    /*DOC*/    "Starts playback of a movie. If audio or video is enabled\n"
    /*DOC*/    "for the Movie, those outputs will be created. \n"
    /*DOC*/ ;

static PyObject* movie_play(PyObject* self, PyObject* args)
{
	FFMovie* movie = PyMovie_AsFFMovie(self);
	int loops=0;

        if(!PyArg_ParseTuple(args, "|i", &loops))
		return NULL;
        if(!movie)
            RAISE(PyExc_SDLError, "Movie object invalid");

        Py_BEGIN_ALLOW_THREADS
	ffmovie_play(movie);
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
	FFMovie* movie = PyMovie_AsFFMovie(self);
	if(!PyArg_ParseTuple(args, ""))
		return NULL;
        if(!movie)
            RAISE(PyExc_SDLError, "Movie object invalid");
        Py_BEGIN_ALLOW_THREADS
	ffmovie_stop(movie);
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
	FFMovie* movie = PyMovie_AsFFMovie(self);
	if(!PyArg_ParseTuple(args, ""))
		return NULL;
        if(!movie)
            RAISE(PyExc_SDLError, "Movie object invalid");
        Py_BEGIN_ALLOW_THREADS
	ffmovie_pause(movie);
        Py_END_ALLOW_THREADS
	RETURN_NONE
}


    /*DOC*/ static char doc_movie_rewind[] =
    /*DOC*/    "Movie.rewind() -> None\n"
    /*DOC*/    "set playback position to the beginning of the movie\n"
    /*DOC*/    "\n"
    /*DOC*/    "Sets the movie playback position to the start of\n"
    /*DOC*/    "the movie. This can raise a ValueError if the movie\n"
    /*DOC*/    "cannot be rewound.\n"
    /*DOC*/    "\n"
    /*DOC*/    "The movie will automatically start playing if the movie\n"
    /*DOC*/    "is currently playing.\n"
    /*DOC*/    "\n"
    /*DOC*/    "If rewinding the movie fails, it will invalidate this Movie\n"
    /*DOC*/    "object.\n"
    /*DOC*/ ;
/*need something for this, if anything reinit a new movie and start*/

static PyObject* movie_rewind(PyObject* self, PyObject* args)
{
        PyMovieObject* movieobj = (PyMovieObject*)self;
        if(!movieobj->movie)
            RAISE(PyExc_SDLError, "Movie object invalid");

        Py_BEGIN_ALLOW_THREADS
        movieobj->movie = ffmovie_reopen(movieobj->movie);        
        Py_END_ALLOW_THREADS
        
        if(!movieobj->movie)
            RAISE(PyExc_SDLError, "Error handling movie source for rewind");
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
	FFMovie* movie = PyMovie_AsFFMovie(self);
	float value;
    
	if(!PyArg_ParseTuple(args, "f", &value))
		return NULL;
        if(!movie)
            RAISE(PyExc_SDLError, "Movie object invalid");

        Py_BEGIN_ALLOW_THREADS
        ffmovie_setvolume(movie, (int)(value*128));
        Py_END_ALLOW_THREADS

	RETURN_NONE
}


    /*DOC*/ static char doc_movie_set_display[] =
    /*DOC*/    "Movie.set_display(Surface, [rect]) -> None\n"
    /*DOC*/    "change the video output surface\n"
    /*DOC*/    "\n"
    /*DOC*/    "Set the output surface for the Movie's video. You may\n"
    /*DOC*/    "also specify a position for the topleft corner of the\n"
    /*DOC*/    "video.\n"
    /*DOC*/    "\n"
    /*DOC*/    "The position argument must be a rectstyle if given.\n"
    /*DOC*/    "The video will be stretched to fill the rectangular area.\n"
    /*DOC*/    "\n"
    /*DOC*/    "You may also pass None as the destination Surface, and\n"
    /*DOC*/    "no video will be rendered for the movie playback.\n"
    /*DOC*/ ;

static PyObject* movie_set_display(PyObject* self, PyObject* args)
{
	FFMovie* movie = PyMovie_AsFFMovie(self);
	PyObject* surfobj, *posobj=NULL;
        SDL_Rect* sdlrect, sdltemp;
	GAME_Rect *rect, temp;
    
	if(!PyArg_ParseTuple(args, "O|O", &surfobj, &posobj))
		return NULL;
        if(!movie)
            RAISE(PyExc_SDLError, "Movie object invalid");

	Py_XDECREF(((PyMovieObject*)self)->surftarget);
	((PyMovieObject*)self)->surftarget = NULL;

	if(PySurface_Check(surfobj))
	{
       SDL_Surface* surf = PySurface_AsSurface(surfobj);

	   ((PyMovieObject*)self)->surftarget = surfobj;
       Py_INCREF(surfobj);

		if(posobj == NULL)
		{
			sdlrect = NULL;
		}
		else if((rect = GameRect_FromObject(posobj, &temp)))
                {
                        sdlrect = &sdltemp;
                        sdltemp.x = rect->x;
                        sdltemp.y = rect->y;
                        sdltemp.w = rect->w;
                        sdltemp.h = rect->h;
                }
                else
			return RAISE(PyExc_TypeError, "Invalid position argument");

            Py_BEGIN_ALLOW_THREADS            
            ffmovie_setdisplay(movie, surf, sdlrect);
            Py_END_ALLOW_THREADS
	}
	else
	{
            Py_BEGIN_ALLOW_THREADS
	    ffmovie_setdisplay(movie, NULL, NULL);
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
	FFMovie* movie = PyMovie_AsFFMovie(self);

	if(!PyArg_ParseTuple(args, ""))
		return NULL;
        if(!movie)
            RAISE(PyExc_SDLError, "Movie object invalid");

	return PyInt_FromLong(movie->video_st != NULL);
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
	FFMovie* movie = PyMovie_AsFFMovie(self);

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	return PyInt_FromLong(movie->audio_st != NULL);
}

    /*DOC*/ static char doc_movie_get_size[] =
    /*DOC*/    "Movie.get_size() -> width,height\n"
    /*DOC*/    "query the size of the video image\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the size of the video image the mpeg provides.\n"
    /*DOC*/ ;

static PyObject* movie_get_size(PyObject* self, PyObject* args)
{
	FFMovie* movie = PyMovie_AsFFMovie(self);
        int w=0, h=0;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;
        if(!movie)
            RAISE(PyExc_SDLError, "Movie object invalid");

	if(movie->video_st != NULL) {
                w = movie->video_st->codec.width;
                h = movie->video_st->codec.height;
        }
	return Py_BuildValue("(ii)", w, h);
}

    /*DOC*/ static char doc_movie_get_frame[] =
    /*DOC*/    "Movie.get_frame() -> int\n"
    /*DOC*/    "query the current frame in the movie\n"
    /*DOC*/    "\n"
    /*DOC*/    "Gets the current video frame number for the movie.\n"
    /*DOC*/ ;

static PyObject* movie_get_frame(PyObject* self, PyObject* args)
{
	FFMovie* movie = PyMovie_AsFFMovie(self);

	if(!PyArg_ParseTuple(args, ""))
		return NULL;
        if(!movie)
            RAISE(PyExc_SDLError, "Movie object invalid");

	return PyInt_FromLong(movie->frame_count);
}

    /*DOC*/ static char doc_movie_get_time[] =
    /*DOC*/    "Movie.get_time() -> float\n"
    /*DOC*/    "query the current time in the movie\n"
    /*DOC*/    "\n"
    /*DOC*/    "Gets the current time (in seconds) for the movie.\n"
    /*DOC*/ ;

static PyObject* movie_get_time(PyObject* self, PyObject* args)
{
	FFMovie* movie = PyMovie_AsFFMovie(self);

	if(!PyArg_ParseTuple(args, ""))
		return NULL;
        if(!movie)
            RAISE(PyExc_SDLError, "Movie object invalid");

	return PyFloat_FromDouble(movie->frame_last_pts);
}

    /*DOC*/ static char doc_movie_get_length[] =
    /*DOC*/    "Movie.get_length() -> float\n"
    /*DOC*/    "query playback time of the movie\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns the total time (in seconds) of the movie.\n"
    /*DOC*/ ;
/*FIX, currently a no op*/
static PyObject* movie_get_length(PyObject* self, PyObject* args)
{
/*	FFMovie* movie = PyMovie_AsFFMovie(self);*/

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	return PyFloat_FromDouble(0.0);
}


    /*DOC*/ static char doc_movie_get_busy[] =
    /*DOC*/    "Movie.get_busy() -> bool\n"
    /*DOC*/    "query the playback state\n"
    /*DOC*/    "\n"
    /*DOC*/    "Returns true if the movie is currently playing.\n"
    /*DOC*/ ;

static PyObject* movie_get_busy(PyObject* self, PyObject* args)
{
	FFMovie* movie = PyMovie_AsFFMovie(self);

	if(!PyArg_ParseTuple(args, ""))
		return NULL;
        if(!movie)
            RAISE(PyExc_SDLError, "Movie object invalid");

	return PyInt_FromLong(movie->context != NULL);
}

    /*DOC*/ static char doc_movie_skip[] =
    /*DOC*/    "Movie.skip(seconds) -> None\n"
    /*DOC*/    "skip ahead a given amount of time\n"
    /*DOC*/    "\n"
    /*DOC*/    "Skips ahead in the movie a given number of seconds.\n"
    /*DOC*/    "Totally experimental, and doesn't work very foos.\n"
    /*DOC*/ ;

static PyObject* movie_skip(PyObject* self, PyObject* args)
{
	FFMovie* movie = PyMovie_AsFFMovie(self);
        float seconds;

	if(!PyArg_ParseTuple(args, "f", &seconds))
		return NULL;
        if(!movie)
            RAISE(PyExc_SDLError, "Movie object invalid");

        movie->time_offset -= seconds;
        
	return PyInt_FromLong(movie->context != NULL);
}


static PyObject* movie_noop(PyObject* self, PyObject* args)
{
	return PyInt_FromLong(0);
}



static PyMethodDef movie_builtins[] =
{
	{ "play", movie_play, 1, doc_movie_play },
	{ "stop", movie_stop, 1, doc_movie_stop },
	{ "pause", movie_pause, 1, doc_movie_pause },
	{ "rewind", movie_rewind, 1, doc_movie_rewind },

	{ "set_volume", movie_set_volume, 1, doc_movie_set_volume },
	{ "set_display", movie_set_display, 1, doc_movie_set_display },

	{ "has_video", movie_has_video, 1, doc_movie_has_video },
	{ "has_audio", movie_has_audio, 1, doc_movie_has_audio },
	{ "get_size", movie_get_size, 1, doc_movie_get_size },
	{ "get_frame", movie_get_frame, 1, doc_movie_get_frame },
	{ "get_time", movie_get_time, 1, doc_movie_get_time },
	{ "get_length", movie_get_length, 1, doc_movie_get_length },
	{ "get_busy", movie_get_busy, 1, doc_movie_get_busy },

	{ "skip", movie_skip, 1, doc_movie_skip },
        { "render_frame", movie_noop, 1, "obsolete, does nothing"},

	{ NULL, NULL }
};


/*sound object internals*/

static void movie_dealloc(PyObject* self)
{
	FFMovie* movie = PyMovie_AsFFMovie(self);
 
        if(movie) {
            Py_BEGIN_ALLOW_THREADS
            ffmovie_close(movie);
            Py_END_ALLOW_THREADS
        }
	Py_XDECREF(((PyMovieObject*)self)->surftarget);
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
	PyObject* file, *final;
	char* name = NULL;
	FFMovie* movie=NULL;
	SDL_Surface* screen;
	if(!PyArg_ParseTuple(arg, "O", &file))
		return NULL;

	if(PyString_Check(file) || PyUnicode_Check(file))
	{
		if(!PyArg_ParseTuple(arg, "s", &name))
			return NULL;

		movie = ffmovie_open(name);
	}
        
	if(!movie)
		return RAISE(PyExc_SDLError, "Cannot create Movie object");

      	screen = SDL_GetVideoSurface();


	final = PyMovie_New(movie);
	if(!final) {
		ffmovie_close(movie);
        }

	return final;
}





static PyMethodDef moviemod_builtins[] =
{
	{ "Movie", Movie, 1, doc_Movie },

	{ NULL, NULL }
};



static PyObject* PyMovie_New(FFMovie* movie)
{
	PyMovieObject* movieobj;

	if(!movie)
		return RAISE(PyExc_RuntimeError, "unable to create movie.");

	movieobj = PyObject_NEW(PyMovieObject, &PyMovie_Type);
	if(movieobj)
		movieobj->movie = movie;

	movieobj->surftarget = NULL;

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
void initmovieext(void)
{
	PyObject *module, *dict;

	PyType_Init(PyMovie_Type);

	/* create the module */
	module = Py_InitModule3("movieext", moviemod_builtins, doc_pygame_movie_MODULE);
	dict = PyModule_GetDict(module);

	PyDict_SetItemString(dict, "MovieType", (PyObject *)&PyMovie_Type);

        /*imported needed apis*/
	import_pygame_base();
	import_pygame_surface();
	import_pygame_rwobject();
	import_pygame_rect();

    PyGame_RegisterQuit(autoquit);
}

