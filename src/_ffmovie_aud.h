#ifndef _FFMOVIE_AUD_H_
#define _FFMOVIE_AUD_H_

#include <libavformat/avformat.h>
#include <Python.h>
#include "pygame.h"
#include "_ff_all.h"


typedef struct PyAudioStream
{
    PyObject_HEAD

    //state control variables for pausing/seeking.
    int paused;
    int last_paused;
    int seek_req;
    int seek_flags;
    int64_t seek_pos;
    
    //time-keeping values
    double audio_clock;
    double audio_diff_cum; /* used for AV difference average computation */
    double audio_diff_avg_coef;
    double audio_diff_threshold;
    int audio_diff_avg_count;
    
    
    AVStream *audio_st;     //audio stream
    PacketQueue audioq;     //packet queue for audio packets
    int audio_hw_buf_size;  //the size of the audio hardware buffer
    /* samples output by the codec. we reserve more space for avsync
       compensation */
    DECLARE_ALIGNED(16,uint8_t,audio_buf1[(AVCODEC_MAX_AUDIO_FRAME_SIZE * 3) / 2]);
    DECLARE_ALIGNED(16,uint8_t,audio_buf2[(AVCODEC_MAX_AUDIO_FRAME_SIZE * 3) / 2]);
    uint8_t *audio_buf;

    unsigned int audio_buf_size; /* in bytes */
    int audio_buf_index; /* in bytes */
    AVPacket audio_pkt;
    uint8_t *audio_pkt_data;
    int audio_pkt_size;
    enum SampleFormat audio_src_fmt;
    AVAudioConvert *reformat_ctx;

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
    int pts;
    
    int playing;
    int loops;

} PyAudioStream;


// stream python stuff 

static PyObject* _aud_stream_init_internal(PyObject *self); //expects file to have been opened in _aud_stream_new
static PyObject* _aud_stream_init (PyObject *self, PyObject *args, PyObject *kwds);
//static void _aud_stream_dealloc (PyAudioStream *audio);
static void _dealloc_aud_stream(PyAudioStream *pas);
static PyObject* _aud_stream_repr (PyAudioStream *audio);
//static PyObject* _aud_stream_str (PyAudioStream *audio);
static PyObject* _aud_stream_play(PyAudioStream *audio, PyObject* args);
static PyObject* _aud_stream_stop(PyAudioStream *audio);
static PyObject* _aud_stream_pause(PyAudioStream *audio);
static PyObject* _aud_stream_rewind(PyAudioStream *audio, PyObject* args);

/* Getters/setters */
static PyObject* _aud_stream_get_paused (PyAudioStream *pvs, void *closure);
static PyObject* _aud_stream_get_playing (PyAudioStream *pvs, void *closure);

static PyMethodDef _audio_methods[] = {
   { "play",    (PyCFunction) _aud_stream_play, METH_VARARGS,
               "Play the movie file from current time-mark. If loop<0, then it will loop infinitely. If there is no loop value, then it will play once." },
   { "stop", (PyCFunction) _aud_stream_stop, METH_NOARGS,
                "Stop the movie, and set time-mark to 0:0"},
   { "pause", (PyCFunction) _aud_stream_pause, METH_NOARGS,
                "Pause movie."},
   { "rewind", (PyCFunction) _aud_stream_rewind, METH_VARARGS,
                "Rewind movie to time_pos. If there is no time_pos, same as stop."},
   { NULL, NULL, 0, NULL }
};

static PyGetSetDef _audio_getsets[] =
{
    { "paused", (getter) _aud_stream_get_paused, NULL, NULL, NULL },
    { "playing", (getter) _aud_stream_get_playing, NULL, NULL, NULL },
    { NULL, NULL, NULL, NULL, NULL }
};

static PyTypeObject PyAudioStream_Type =
{
    PyObject_HEAD_INIT(NULL)
    0, 
    "pygame.gmovie.AudioStream",          /* tp_name */
    sizeof (PyAudioStream),           /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor) _dealloc_aud_stream,/* tp_dealloc */
    0,                          /* tp_print */
    0,                          /* tp_getattr */
    0,                          /* tp_setattr */
    0,                          /* tp_compare */
    (reprfunc) _aud_stream_repr,     /* tp_repr */
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
    _audio_methods,             /* tp_methods */
    0,                          /* tp_members */
    _audio_getsets,             /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc) _aud_stream_init,                          /* tp_init */
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


static PyObject* _aud_stream_init_internal(PyObject *self)
{
	    /*Expects filename. If surface is null, then it sets overlay to >0. */
    PySys_WriteStdout("Within _aud_stream_init_internal\n");    
    
    Py_INCREF(self);
	//do stuff
    
    
    //Py_DECREF((PyObject *) movie);
    PySys_WriteStdout("_aud_stream_init_internal: Returning from _aud_stream_init_internal\n");
    Py_DECREF(self);
    return self;
}

static PyObject* _aud_stream_init(PyObject *self, PyObject *args, PyObject *kwds)
{
	Py_INCREF(self);
    PySys_WriteStdout("Within _aud_stream_init\n");

    PySys_WriteStdout("_aud_stream_init: Before _aud_stream_init_internal\n");
    self = _aud_stream_init_internal(self);
    PySys_WriteStdout("_aud_stream_init: After _aud_stream_init_internal\n");
    PyObject *er;
    er = PyErr_Occurred();
    if(er)
    {
        PyErr_Print();
    }
    if(!self)
    {
        PyErr_SetString(PyExc_IOError, "No audio stream created.");
        PyErr_Print();
        Py_XDECREF(self);
        Py_RETURN_NONE;
    }
    PySys_WriteStdout("Returning from _aud_stream_init\n");
    return self;
}

static PyObject* _aud_stream_repr (PyAudioStream *audio)
{
    /*Eventually add a time-code call */
    char buf[100];
    //PySys_WriteStdout("_aud_stream_repr: %10s\n", audio->filename); 
    PyOS_snprintf(buf, sizeof(buf), "(Video Stream: %p)", &audio);
    return PyString_FromString(buf);
}


static PyObject* _aud_stream_play(PyAudioStream *audio, PyObject* args)
{
    PySys_WriteStdout("In _aud_stream_play\n");
    int loops;
    if(!PyArg_ParseTuple(args, "i", &loops))
    {
        PyErr_SetString(PyExc_TypeError, "Not a valid argument.");
        Py_RETURN_NONE;
    }
    PySys_WriteStdout("_aud_stream_play: loops set to: %i\n", loops);
    audio->loops =loops;
    audio->paused = 0;
    audio->playing = 1;
    Py_RETURN_NONE;
}

static PyObject* _aud_stream_stop(PyAudioStream *audio)
{
    //stream_pause(audio);
    audio->seek_req = 1;
    audio->seek_pos = 0;
    audio->seek_flags =AVSEEK_FLAG_BACKWARD;
    Py_RETURN_NONE;
}  

static PyObject* _aud_stream_pause(PyAudioStream *audio)
{
    //stream_pause(audio); 
    Py_RETURN_NONE;
}

static PyObject* _aud_stream_rewind(PyAudioStream *audio, PyObject* args)
{
    /* For now, just alias rewind to stop */
    return _aud_stream_stop(audio);
}

static PyObject* _aud_stream_get_paused (PyAudioStream *audio, void *closure)
{
    return PyInt_FromLong((long)audio->paused);
}
static PyObject* _aud_stream_get_playing (PyAudioStream *audio, void *closure)
{
    PyObject *pyo;
    pyo= PyInt_FromLong((long)audio->playing);
    return pyo;
}



#endif /*_FFMOVIE_AUD_H_*/
