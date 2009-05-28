#ifndef _FFMOVIE_VID_H_
#define _FFMOVIE_VID_H_

#include <libavformat/avformat.h>
#include <Python.h>
#include "pygame.h"
#include "_ff_all.h"

//Struct to display the video image. 
typedef struct {
    double pts;         //presentation time stamp for this picture, used for syncing
    int width, height;
    int allocated;      //if structure has been allocated. null if not.
} VideoPicture;

#define VIDEO_PICTURE_QUEUE_SIZE 1

typedef struct PyVideoStream
{
    PyObject_HEAD

    SDL_Surface *out_surf; /*surface to output video to. If surface is the display surface, 
                         * then we can use overlay code. Otherwise, we use the python interface.
                         */
    SDL_Overlay *bmp;
    SDL_Thread *video_tid;  //thread id for the video thread
    int rgb;                //if true, must convert image data to rgb before writing to it. 
    //int no_background;    //Not needed or relevant when we're working with pygame. ;)
    
    //state values for pausing and seeking
    int paused;          
    int playing;
    int last_paused;
    int seek_req;
    int seek_flags;
    int64_t seek_pos;

    /*time-keeping values
     *By default, all operating streams work on the same external clock
     *It is only when streams are controlled individually that an individual clock is needed.
     */
    int av_sync_type;      /* Normally external. */
    double external_clock; /* external clock base */
    int64_t external_clock_time;
    int64_t offset;        /*Offset for when the individual clock is used. This way if all the streams are playing, we can keep them synced up, but shifted. */

    /* Frame-tracker values 
     * Needed for syncing, time delay
     */
    double frame_timer;
    double frame_last_pts;
    double frame_last_delay;
    double frame_offset;
    double video_clock;                          ///<pts of last decoded frame / predicted pts of next decoded frame
    
    //Video stream struct.
    AVStream *video_st;
    //video queue, with video packets
    PacketQueue videoq;
    double video_current_pts;                    ///<current displayed pts (different from video_clock if frame fifos are used)
    int64_t video_current_pts_time;              ///<time (av_gettime) at which we updated video_current_pts - used to have running video pts
    
    VideoPicture pictq[VIDEO_PICTURE_QUEUE_SIZE]; //queue of VideoPicture objects, ring-buffer structure. Normally size of 1.
    int pictq_size, pictq_rindex, pictq_windex;
    SDL_mutex *pictq_mutex;
    SDL_cond *pictq_cond;

    //    QETimer *video_timer;
    int width, height, xleft, ytop;

    int pts;
    
    int overlay;
    
    int loops;

} PyVideoStream;


// stream python stuff 

static PyObject* _vid_stream_init_internal(PyObject *self,  PyObject* surface); //expects file to have been opened in _vid_stream_new
static PyObject* _vid_stream_init (PyObject *self, PyObject *args, PyObject *kwds);
//static void _vid_stream_dealloc (PyVideoStream *video);
static void _dealloc_vid_stream(PyVideoStream *pvs);
static PyObject* _vid_stream_repr (PyVideoStream *video);
//static PyObject* _vid_stream_str (PyVideoStream *video);
static PyObject* _vid_stream_play(PyVideoStream *video, PyObject* args);
static PyObject* _vid_stream_stop(PyVideoStream *video);
static PyObject* _vid_stream_pause(PyVideoStream *video);
static PyObject* _vid_stream_rewind(PyVideoStream *video, PyObject* args);

/* Getters/setters */
static PyObject* _vid_stream_get_paused (PyVideoStream *pvs, void *closure);
static PyObject* _vid_stream_get_playing (PyVideoStream *pvs, void *closure);

static PyMethodDef _video_methods[] = {
   { "play",    (PyCFunction) _vid_stream_play, METH_VARARGS,
               "Play the movie file from current time-mark. If loop<0, then it will loop infinitely. If there is no loop value, then it will play once." },
   { "stop", (PyCFunction) _vid_stream_stop, METH_NOARGS,
                "Stop the movie, and set time-mark to 0:0"},
   { "pause", (PyCFunction) _vid_stream_pause, METH_NOARGS,
                "Pause movie."},
   { "rewind", (PyCFunction) _vid_stream_rewind, METH_VARARGS,
                "Rewind movie to time_pos. If there is no time_pos, same as stop."},
   { NULL, NULL, 0, NULL }
};

static PyGetSetDef _video_getsets[] =
{
    { "paused", (getter) _vid_stream_get_paused, NULL, NULL, NULL },
    { "playing", (getter) _vid_stream_get_playing, NULL, NULL, NULL },
    { NULL, NULL, NULL, NULL, NULL }
};

static PyTypeObject PyVideoStream_Type =
{
    PyObject_HEAD_INIT(NULL)
    0, 
    "pygame.gmovie.VideoStream",          /* tp_name */
    sizeof (PyVideoStream),           /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor) _dealloc_vid_stream,/* tp_dealloc */
    0,                          /* tp_print */
    0,                          /* tp_getattr */
    0,                          /* tp_setattr */
    0,                          /* tp_compare */
    (reprfunc) _vid_stream_repr,     /* tp_repr */
    0,                          /* tp_as_number */
    0,                          /* tp_as_sequence */
    0,                          /* tp_as_mapping */
    0,                          /* tp_hash */
    0,                          /* tp_call */
    0,                          /* tp_str */
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
    _video_methods,             /* tp_methods */
    0,                          /* tp_members */
    _video_getsets,             /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc)_vid_stream_init,                          /* tp_init */
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


static PyObject* _vid_stream_init_internal(PyObject *self, PyObject *surface)
{
	    /*Expects filename. If surface is null, then it sets overlay to >0. */
    PySys_WriteStdout("Within _vid_stream_init_internal\n");    
    Py_INCREF(self);
	PyVideoStream *pvs;
	pvs=(PyVideoStream *)self;
	Py_INCREF(pvs);
    if(!surface)
    {
        PySys_WriteStdout("_vid_stream_init_internal: Overlay=True\n");
        pvs->overlay=1;
    }
    else
    {
        PySys_WriteStdout("_vid_stream_init_internal: Overlay=False\n");
        SDL_Surface *surf;
        surf = PySurface_AsSurface(surface);
        pvs->out_surf=surf;
        pvs->overlay=0;
    }
    
    //Py_DECREF((PyObject *) movie);
    Py_DECREF(pvs);
    Py_DECREF(self);
    PySys_WriteStdout("_vid_stream_init_internal: Returning from _vid_stream_init_internal\n");
    return (PyObject *)pvs;
}

static PyObject* _vid_stream_init(PyObject *self, PyObject *args, PyObject *kwds)
{
	Py_INCREF(self);
    PyObject *obj1;
    PySys_WriteStdout("Within _vid_stream_init\n");
    if (!PyArg_ParseTuple (args, "O", &obj1))
    {
        PyErr_SetString(PyExc_TypeError, "No valid arguments");
        //Py_RETURN_NONE;
    	return 0;
    }
    PySys_WriteStdout("_vid_stream_init: after PyArg_ParseTuple\n"); 

    PySys_WriteStdout("_vid_stream_init: Before _vid_stream_init_internal\n");
    self = _vid_stream_init_internal(self, obj1);
    PySys_WriteStdout("_vid_stream_init: After _vid_stream_init_internal\n");
    PyObject *er;
    er = PyErr_Occurred();
    if(er)
    {
        PyErr_Print();
    }
    if(!self)
    {
        PyErr_SetString(PyExc_IOError, "No video stream created.");
        PyErr_Print();
    }
    PySys_WriteStdout("Returning from _vid_stream_init\n");
    return self;
}

static PyObject* _vid_stream_repr (PyVideoStream *video)
{
    /*Eventually add a time-code call */
    char buf[100];
    //PySys_WriteStdout("_vid_stream_repr: %10s\n", video->filename); 
    PyOS_snprintf(buf, sizeof(buf), "(Video Stream: %p)", &video);
    return PyString_FromString(buf);
}


static PyObject* _vid_stream_play(PyVideoStream *video, PyObject* args)
{
    PySys_WriteStdout("In _vid_stream_play\n");
    int loops;
    if(!PyArg_ParseTuple(args, "i", &loops))
    {
        PyErr_SetString(PyExc_TypeError, "Not a valid argument.");
        Py_RETURN_NONE;
    }
    PySys_WriteStdout("_vid_stream_play: loops set to: %i\n", loops);
    video->loops =loops;
    video->paused = 0;
    video->playing = 1;
    Py_RETURN_NONE;
}

static PyObject* _vid_stream_stop(PyVideoStream *video)
{
    //stream_pause(video);
    video->seek_req = 1;
    video->seek_pos = 0;
    video->seek_flags =AVSEEK_FLAG_BACKWARD;
    Py_RETURN_NONE;
}  

static PyObject* _vid_stream_pause(PyVideoStream *video)
{
    //stream_pause(video); 
    Py_RETURN_NONE;
}

static PyObject* _vid_stream_rewind(PyVideoStream *video, PyObject* args)
{
    /* For now, just alias rewind to stop */
    return _vid_stream_stop(video);
}

static PyObject* _vid_stream_get_paused (PyVideoStream *video, void *closure)
{
    return PyInt_FromLong((long)video->paused);
}
static PyObject* _vid_stream_get_playing (PyVideoStream *video, void *closure)
{
    PyObject *pyo;
    pyo= PyInt_FromLong((long)video->playing);
    return pyo;
}

	



#endif /*_FFMOVIE_VID_H_*/
