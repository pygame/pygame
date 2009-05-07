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
#include "pgcompat.h"
#include "pygamedocs.h"
#include "smpeg.h"

typedef struct
{
    PyObject_HEAD
    SMPEG* movie;
    PyObject* surftarget;
    PyObject* filesource;
} PyMovieObject;
#define PyMovie_AsSMPEG(x) (((PyMovieObject*)x)->movie)

static PyTypeObject PyMovie_Type;
static PyObject* PyMovie_New (SMPEG*);
#define PyMovie_Check(x) ((x)->ob_type == &PyMovie_Type)

/* movie object methods */
static PyObject*
movie_play (PyObject* self)
{
    SMPEG* movie = PyMovie_AsSMPEG (self);
    int loops = 0;

	if (!SDL_WasInit (SDL_INIT_VIDEO))
        return RAISE (PyExc_SDLError,
                      "cannot convert without pygame.display initialized");

    Py_BEGIN_ALLOW_THREADS;
    SMPEG_loop (movie, loops);
    SMPEG_play (movie);
    Py_END_ALLOW_THREADS;
    Py_RETURN_NONE;
}

static PyObject*
movie_stop (PyObject* self)
{
    SMPEG* movie = PyMovie_AsSMPEG (self);

	if (!SDL_WasInit (SDL_INIT_VIDEO))
        return RAISE (PyExc_SDLError,
                      "cannot convert without pygame.display initialized");

    Py_BEGIN_ALLOW_THREADS;
    SMPEG_stop (movie);
    Py_END_ALLOW_THREADS;
    Py_RETURN_NONE;
}

static PyObject*
movie_pause (PyObject* self) {
    SMPEG* movie = PyMovie_AsSMPEG (self);

	if (!SDL_WasInit (SDL_INIT_VIDEO))
        return RAISE (PyExc_SDLError,
                      "cannot convert without pygame.display initialized");


    Py_BEGIN_ALLOW_THREADS;
    SMPEG_pause (movie);
    Py_END_ALLOW_THREADS;
    Py_RETURN_NONE;
}

static PyObject*
movie_rewind (PyObject* self)
{
    SMPEG* movie = PyMovie_AsSMPEG (self);

	if (!SDL_WasInit (SDL_INIT_VIDEO))
        return RAISE (PyExc_SDLError,
                      "cannot convert without pygame.display initialized");

    Py_BEGIN_ALLOW_THREADS;
    SMPEG_rewind (movie);
    Py_END_ALLOW_THREADS;
    Py_RETURN_NONE;
}

static PyObject*
movie_skip (PyObject* self, PyObject* args)
{
    SMPEG* movie = PyMovie_AsSMPEG (self);
    float seconds;

	if (!SDL_WasInit (SDL_INIT_VIDEO))
	return RAISE (PyExc_SDLError,
				  "cannot convert without pygame.display initialized");

	
	if (!PyArg_ParseTuple (args, "f", &seconds))
        return NULL;
    Py_BEGIN_ALLOW_THREADS;
    SMPEG_skip (movie, seconds);
    Py_END_ALLOW_THREADS;
    Py_RETURN_NONE;
}

static PyObject*
movie_set_volume (PyObject* self, PyObject* args)
{
    SMPEG* movie = PyMovie_AsSMPEG (self);
    float value;
    int volume;
    if (!PyArg_ParseTuple (args, "f", &value))
        return NULL;

	if (!SDL_WasInit (SDL_INIT_VIDEO))
        return RAISE (PyExc_SDLError,
                      "cannot convert without pygame.display initialized");


    Py_BEGIN_ALLOW_THREADS;
    volume = (int) (value * 100);
    if (volume < 0)
        volume = 0;
    if (volume > 100)
        volume = 100;
    SMPEG_setvolume (movie, volume);
    Py_END_ALLOW_THREADS;
    Py_RETURN_NONE;
}

static PyObject*
movie_set_display (PyObject* self, PyObject* args)
{
    SMPEG* movie = PyMovie_AsSMPEG (self);
    PyObject* surfobj, *posobj=NULL;
    GAME_Rect *rect, temp;
    int x=0, y=0;
    if (!PyArg_ParseTuple (args, "O|O", &surfobj, &posobj))
        return NULL;


	if (!SDL_WasInit (SDL_INIT_VIDEO))
        return RAISE (PyExc_SDLError,
                      "cannot convert without pygame.display initialized");


    Py_XDECREF (((PyMovieObject*) self)->surftarget);
    ((PyMovieObject*) self)->surftarget = NULL;

    if (PySurface_Check (surfobj))
    {
        SMPEG_Info info;
        SDL_Surface* surf;

        if (posobj == NULL)
        {
            Py_BEGIN_ALLOW_THREADS;
            SMPEG_getinfo (movie, &info);
            SMPEG_scaleXY (movie, info.width, info.height);
            Py_END_ALLOW_THREADS;
            x = y = 0;
        }
        else if (TwoIntsFromObj (posobj, &x, &y))
        {
            Py_BEGIN_ALLOW_THREADS;
            SMPEG_getinfo (movie, &info);
            SMPEG_scaleXY (movie, info.width, info.height);
            Py_END_ALLOW_THREADS;
        }
        else if ((rect = GameRect_FromObject (posobj, &temp)))
        {
            x = rect->x;
            y = rect->y;
            Py_BEGIN_ALLOW_THREADS;
            SMPEG_scaleXY (movie, rect->w, rect->h);
            Py_END_ALLOW_THREADS;
        }
        else
            return RAISE (PyExc_TypeError, "Invalid position argument");

        surf = PySurface_AsSurface (surfobj);

        Py_BEGIN_ALLOW_THREADS;
        SMPEG_getinfo (movie, &info);
        SMPEG_enablevideo (movie, 1);
        SMPEG_setdisplay (movie, surf, NULL, NULL);
        SMPEG_move (movie, x, y);
        Py_END_ALLOW_THREADS;
    }
    else
    {
        Py_BEGIN_ALLOW_THREADS;
        SMPEG_enablevideo (movie, 0);
        Py_END_ALLOW_THREADS;
        if (surfobj != Py_None)
            return RAISE (PyExc_TypeError, "destination must be a Surface");
    }

    Py_RETURN_NONE;
}

static PyObject*
movie_has_video (PyObject* self)
{
    SMPEG* movie = PyMovie_AsSMPEG (self);
    SMPEG_Info info;

	if (!SDL_WasInit (SDL_INIT_VIDEO))
        return RAISE (PyExc_SDLError,
                      "cannot convert without pygame.display initialized");

    Py_BEGIN_ALLOW_THREADS;
    SMPEG_getinfo (movie, &info);
    Py_END_ALLOW_THREADS;
    return PyInt_FromLong (info.has_video);
}

static PyObject*
movie_has_audio (PyObject* self)
{
    SMPEG* movie = PyMovie_AsSMPEG (self);
    SMPEG_Info info;

	if (!SDL_WasInit (SDL_INIT_VIDEO))
        return RAISE (PyExc_SDLError,
                      "cannot convert without pygame.display initialized");


    Py_BEGIN_ALLOW_THREADS;
    SMPEG_getinfo (movie, &info);
    Py_END_ALLOW_THREADS;
    return PyInt_FromLong (info.has_audio);
}

static PyObject*
movie_get_size (PyObject* self)
{
    SMPEG* movie = PyMovie_AsSMPEG (self);
    SMPEG_Info info;

	if (!SDL_WasInit (SDL_INIT_VIDEO))
        return RAISE (PyExc_SDLError,
                      "cannot convert without pygame.display initialized");

    Py_BEGIN_ALLOW_THREADS;
    SMPEG_getinfo (movie, &info);
    Py_END_ALLOW_THREADS;
    return Py_BuildValue ("(ii)", info.width, info.height);
}

static PyObject*
movie_get_frame (PyObject* self)
{
    SMPEG* movie = PyMovie_AsSMPEG (self);
    SMPEG_Info info;

	if (!SDL_WasInit (SDL_INIT_VIDEO))
        return RAISE (PyExc_SDLError,
                      "cannot convert without pygame.display initialized");

    Py_BEGIN_ALLOW_THREADS;
    SMPEG_getinfo (movie, &info);
    Py_END_ALLOW_THREADS;
    return PyInt_FromLong (info.current_frame);
}

static PyObject*
movie_get_time (PyObject* self)
{
    SMPEG* movie = PyMovie_AsSMPEG (self);
    SMPEG_Info info;

	if (!SDL_WasInit (SDL_INIT_VIDEO))
        return RAISE (PyExc_SDLError,
                      "cannot convert without pygame.display initialized");

    Py_BEGIN_ALLOW_THREADS;
    SMPEG_getinfo (movie, &info);
    Py_END_ALLOW_THREADS;
    return PyFloat_FromDouble (info.current_time);
}

static PyObject*
movie_get_length (PyObject* self)
{
	SMPEG* movie;
    SMPEG_Info info;

	if (!SDL_WasInit (SDL_INIT_VIDEO))
        return RAISE (PyExc_SDLError,
                      "cannot convert without pygame.display initialized");

	movie = PyMovie_AsSMPEG (self);

    Py_BEGIN_ALLOW_THREADS;
    SMPEG_getinfo (movie, &info);
    Py_END_ALLOW_THREADS;
    return PyFloat_FromDouble (info.total_time);
}

static PyObject*
movie_get_busy (PyObject* self)
{
    SMPEG* movie;

	if (!SDL_WasInit (SDL_INIT_VIDEO))
        return RAISE (PyExc_SDLError,
                      "cannot convert without pygame.display initialized");

    movie = PyMovie_AsSMPEG (self);

    return PyInt_FromLong (SMPEG_status (movie) == SMPEG_PLAYING);
}

static PyObject*
movie_render_frame (PyObject* self, PyObject* args)
{
    SMPEG* movie = PyMovie_AsSMPEG (self);
    SMPEG_Info info;
    int framenum;

    if (!SDL_WasInit (SDL_INIT_VIDEO))
        return RAISE (PyExc_SDLError,
                      "cannot convert without pygame.display initialized");

    if (!PyArg_ParseTuple (args, "i", &framenum))
        return NULL;
    Py_BEGIN_ALLOW_THREADS;
    SMPEG_renderFrame (movie, framenum);
    SMPEG_getinfo (movie, &info);
    Py_END_ALLOW_THREADS;
    return PyInt_FromLong (info.current_frame);
}

static PyMethodDef movie_methods[] =
{
    { "play", (PyCFunction)movie_play, METH_NOARGS, DOC_MOVIEPLAY },
    { "stop", (PyCFunction) movie_stop, METH_NOARGS, DOC_MOVIESTOP },
    { "pause", (PyCFunction) movie_pause, METH_NOARGS, DOC_MOVIEPAUSE },
    { "rewind", (PyCFunction) movie_rewind, METH_NOARGS, DOC_MOVIEREWIND },
    { "skip", movie_skip, METH_VARARGS, DOC_MOVIESKIP },
    
    { "set_volume", movie_set_volume, METH_VARARGS, DOC_MOVIESETVOLUME },
    { "set_display", movie_set_display, METH_VARARGS, DOC_MOVIESETDISPLAY },
    
    { "has_video", (PyCFunction) movie_has_video, METH_NOARGS,
      DOC_MOVIEHASVIDEO },
    { "has_audio", (PyCFunction) movie_has_audio, METH_NOARGS,
      DOC_MOVIEHASAUDIO },
    { "get_size", (PyCFunction) movie_get_size, METH_NOARGS, DOC_MOVIEGETSIZE },
    { "get_frame", (PyCFunction) movie_get_frame, METH_NOARGS,
      DOC_MOVIEGETFRAME },
    { "get_time", (PyCFunction) movie_get_time, METH_NOARGS, DOC_MOVIEGETTIME },
    { "get_length", (PyCFunction) movie_get_length, METH_NOARGS,
      DOC_MOVIEGETLENGTH },
    { "get_busy", (PyCFunction) movie_get_busy, METH_NOARGS, DOC_MOVIEGETBUSY },
    { "render_frame", movie_render_frame, METH_VARARGS, DOC_MOVIERENDERFRAME },

    { NULL, NULL, 0, NULL }
};


/*sound object internals*/
static void
movie_dealloc (PyObject* self)
{
    SMPEG* movie = PyMovie_AsSMPEG (self);
    Py_BEGIN_ALLOW_THREADS;
    SMPEG_delete (movie);
    Py_END_ALLOW_THREADS;
    Py_XDECREF (((PyMovieObject*) self)->surftarget);
    Py_XDECREF (((PyMovieObject*) self)->filesource);
    PyObject_DEL (self);
}

static PyTypeObject PyMovie_Type =
{
    TYPE_HEAD (NULL, 0)
    "movie",                    /* name */
    sizeof(PyMovieObject),      /* basic size */
    0,                          /* itemsize */
    movie_dealloc,              /* dealloc */
    0,                          /* print */
    0,                          /* getattr */
    0,                          /* setattr */
    0,                          /* compare */
    0,                          /* repr */
    0,                          /* as_number */
    0,                          /* as_sequence */
    0,                          /* as_mapping */
    0,                          /* hash */
    0,                          /* call */
    0,                          /* str */
    0,                          /* tp_getattro */
    0,                          /* tp_setattro */
    0,                          /* tp_as_buffer */
    0,                          /* flags */
    DOC_PYGAMEMOVIEMOVIE,       /* Documentation string */
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,	                        /* tp_iter */
    0,                          /* tp_iternext */
    movie_methods,              /* tp_methods */
    0,                          /* tp_members */
    0,                          /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    0,                          /* tp_init */
    0,				/* tp_alloc */
    0,			        /* tp_new */
};

/*movie module methods*/
static PyObject*
Movie (PyObject* self, PyObject* arg)
{
    PyObject* file, *final, *filesource=NULL;
    char* name = NULL;
    SMPEG* movie=NULL;
    SMPEG_Info info;
    SDL_Surface* screen;
    char* error;
    int audioavail = 0;

    if (!SDL_WasInit (SDL_INIT_VIDEO))
        return RAISE (PyExc_SDLError,
                      "cannot convert without pygame.display initialized");

    if (!PyArg_ParseTuple (arg, "O", &file))
        return NULL;

    if (!SDL_WasInit (SDL_INIT_AUDIO))
        audioavail = 1;

    if (Bytes_Check (file) || PyUnicode_Check (file))
    {
        if (!PyArg_ParseTuple (arg, "s", &name))
            return NULL;
        movie = SMPEG_new (name, &info, audioavail);
    }
#if !PY3
    else if (PyFile_Check (file))
    {
        SDL_RWops *rw = SDL_RWFromFP (PyFile_AsFile (file), 0);
        movie = SMPEG_new_rwops (rw, &info, audioavail);
        filesource = file;
        Py_INCREF (file);
    }
#endif
    else
    {
        SDL_RWops *rw;
        if (!(rw = RWopsFromPythonThreaded (file)))
            return NULL;
        Py_BEGIN_ALLOW_THREADS;
        movie = SMPEG_new_rwops (rw, &info, audioavail);
        Py_END_ALLOW_THREADS;
    }

    if (!movie)
        return RAISE (PyExc_SDLError, "Cannot create Movie object");

    error = SMPEG_error (movie);
    if (error)
    {
        /* while this would seem correct, it causes a crash, so don't
         * delete SMPEG_delete(movie);*/
        return RAISE (PyExc_SDLError, error);
    }

    Py_BEGIN_ALLOW_THREADS;
    SMPEG_enableaudio (movie, audioavail);
    screen = SDL_GetVideoSurface ();
    if (screen)
        SMPEG_setdisplay (movie, screen, NULL, NULL);

    SMPEG_scaleXY (movie, info.width, info.height);
    Py_END_ALLOW_THREADS;

    final = PyMovie_New (movie);
    if (!final)
        SMPEG_delete (movie);
    ((PyMovieObject*) final)->filesource = filesource;

    return final;
}

static PyMethodDef _movie_methods[] =
{
    { "Movie", Movie, METH_VARARGS, DOC_PYGAMEMOVIEMOVIE },
    { NULL, NULL, 0, NULL }
};

static PyObject*
PyMovie_New (SMPEG* movie)
{
    PyMovieObject* movieobj;

    if (!movie)
        return RAISE (PyExc_RuntimeError, "unable to create movie.");

    movieobj = PyObject_NEW (PyMovieObject, &PyMovie_Type);
    if (movieobj)
        movieobj->movie = movie;

    movieobj->surftarget = NULL;
    movieobj->filesource = NULL;

    return (PyObject*)movieobj;
}

MODINIT_DEFINE (movie)
{
    PyObject *module, *dict;

#if PY3
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "movie",
        DOC_PYGAMEMOVIE,
        -1,
        _movie_methods,
        NULL, NULL, NULL, NULL
    };
#endif

    /* imported needed apis; Do this first so if there is an error
       the module is not loaded.
    */
    import_pygame_base ();
    if (PyErr_Occurred ()) {
	MODINIT_ERROR;
    }
    import_pygame_surface ();
    if (PyErr_Occurred ()) {
	MODINIT_ERROR;
    }
    import_pygame_rwobject ();
    if (PyErr_Occurred ()) {
	MODINIT_ERROR;
    }
    import_pygame_rect ();
    if (PyErr_Occurred ()) {
	MODINIT_ERROR;
    }

    /* type preparation */
    if (PyType_Ready (&PyMovie_Type) == -1) {
        MODINIT_ERROR;
    }

    /* create the module */
#if PY3
    module = PyModule_Create (&_module);
#else
    module = Py_InitModule3 ("movie", _movie_methods, DOC_PYGAMEMOVIE);
#endif
    if (module == NULL) {
        MODINIT_ERROR;
    }
    dict = PyModule_GetDict (module);

    if (PyDict_SetItemString (dict, "MovieType",
                              (PyObject *)&PyMovie_Type) == -1) {
        DECREF_MOD (module);
        MODINIT_ERROR;
    }

    MODINIT_RETURN (module);
}
