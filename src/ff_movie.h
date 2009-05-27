#include "pygamedocs.h"
#include "pygame.h"
#include "pgcompat.h"
#include "audioconvert.h"
#include "surface.h"
#include "_ffmovie_vid.h"
#include "_ffmovie_aud.h"

#include <Python.h>
#include <SDL.h>
#include <SDL_thread.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/random.h>
#include <libavutil/avstring.h>
#include <libswscale/swscale.h>
#include <libavdevice/avdevice.h>

#define MAX_VIDEOQ_SIZE (5 * 256 * 1024)
#define MAX_AUDIOQ_SIZE (5 * 16 * 1024)
#define MAX_SUBTITLEQ_SIZE (5 * 16 * 1024)

/* SDL audio buffer size, in samples. Should be small to have precise
   A/V sync as SDL does not have hardware buffer fullness info. */
#define SDL_AUDIO_BUFFER_SIZE 1024

/* no AV sync correction is done if below the AV sync threshold */
#define AV_SYNC_THRESHOLD 0.01
/* no AV correction is done if too big error */
#define AV_NOSYNC_THRESHOLD 10.0

/* maximum audio speed change to get correct sync */
#define SAMPLE_CORRECTION_PERCENT_MAX 10

/* we use about AUDIO_DIFF_AVG_NB A-V differences to make the average */
#define AUDIO_DIFF_AVG_NB   20

/* NOTE: the size must be big enough to compensate the hardware audio buffersize size */
#define SAMPLE_ARRAY_SIZE (2*65536)

#define SCALEBITS 10
#define ONE_HALF  (1 << (SCALEBITS - 1))
#define FIX(x)    ((int) ((x) * (1<<SCALEBITS) + 0.5))

#define RGB_TO_Y_CCIR(r, g, b) \
((FIX(0.29900*219.0/255.0) * (r) + FIX(0.58700*219.0/255.0) * (g) + \
  FIX(0.11400*219.0/255.0) * (b) + (ONE_HALF + (16 << SCALEBITS))) >> SCALEBITS)

#define RGB_TO_U_CCIR(r1, g1, b1, shift)\
(((- FIX(0.16874*224.0/255.0) * r1 - FIX(0.33126*224.0/255.0) * g1 +         \
     FIX(0.50000*224.0/255.0) * b1 + (ONE_HALF << shift) - 1) >> (SCALEBITS + shift)) + 128)

#define RGB_TO_V_CCIR(r1, g1, b1, shift)\
(((FIX(0.50000*224.0/255.0) * r1 - FIX(0.41869*224.0/255.0) * g1 -           \
   FIX(0.08131*224.0/255.0) * b1 + (ONE_HALF << shift) - 1) >> (SCALEBITS + shift)) + 128)

#define ALPHA_BLEND(a, oldp, newp, s)\
((((oldp << s) * (255 - (a))) + (newp * (a))) / (255 << s))

#define RGBA_IN(r, g, b, a, s)\
{\
    unsigned int v = ((const uint32_t *)(s))[0];\
    a = (v >> 24) & 0xff;\
    r = (v >> 16) & 0xff;\
    g = (v >> 8) & 0xff;\
    b = v & 0xff;\
}

#define YUVA_IN(y, u, v, a, s, pal)\
{\
    unsigned int val = ((const uint32_t *)(pal))[*(const uint8_t*)(s)];\
    a = (val >> 24) & 0xff;\
    y = (val >> 16) & 0xff;\
    u = (val >> 8) & 0xff;\
    v = val & 0xff;\
}

#define YUVA_OUT(d, y, u, v, a)\
{\
    ((uint32_t *)(d))[0] = (a << 24) | (y << 16) | (u << 8) | v;\
}


#define BPP 1



static int sws_flags = SWS_BICUBIC;

#define SUBPICTURE_QUEUE_SIZE 4


typedef struct SubPicture {
    double pts;         /* presentation time stamp for this picture */
    AVSubtitle sub;     //contains relevant info about subtitles    
} SubPicture;

enum {
    AV_SYNC_AUDIO_MASTER, /* default choice */
    AV_SYNC_VIDEO_MASTER, 
    AV_SYNC_SUB_MASTER, 
    AV_SYNC_EXTERNAL_CLOCK, /* synchronize to an external clock */
};



typedef struct PySubtitleStream
{
    PyObject_HEAD

	/*not yet currently used */ 
    SDL_Surface *out_surf; /*surface to output subtitle to. If surface is the display surface, 
                         * then we can use overlay code. Otherwise, we use the python interface.
                         * Can be different from PyVideoStream's output object.
                         */

    int same_as_vid;      //true if the surface is the same as the video stream   //state variables for pause/seek
    int paused;
    int last_paused;
    int seek_req;
    int seek_flags;
    int64_t seek_pos;
    
    
    SDL_Thread *subtitle_tid;                    //thread id for subtitle decode thread
    int subtitle_stream;                         //which subtitle thread we want
    int subtitle_stream_changed;                 //if the subtitle-stream has changed
    AVStream *subtitle_st;                       //subtitle stream
    PacketQueue subtitleq;                       //packet queue for decoded subtitle packets
    SubPicture subpq[SUBPICTURE_QUEUE_SIZE];     //Picture objects for displaying the subtitle info
    int subpq_size, subpq_rindex, subpq_windex;  
    SDL_mutex *subpq_mutex;
    SDL_cond *subpq_cond;

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

} PySubtitleStream;

typedef struct PyMovie
{
    PyObject_HEAD
    PyObject *streams;      /* lists object for all the streams of the video file. */
    SDL_Thread *parse_tid;  /* top-level thread id for parsing file */
    AVInputFormat *iformat; /* contains information about the file */
    int abort_request;      /* lets other threads know to stop. */
    AVFormatContext *ic;    /* context information about the format of the video file */

    SDL_Surface *out_surf;

    /* General seek/pause state variables */
    int paused;
    int last_paused;
    int seek_req;
    int seek_flags;
    int64_t seek_pos;

    /* General clock-keeping for all playing streams */
    int av_sync_type;
    double external_clock; /* external clock base */
    int64_t external_clock_time;

    /* General frame variables for all playing streams */
    double frame_timer;
    double frame_last_pts;
    double frame_last_delay;

    char filename[1024];

    /* Index values of the playing streams */
    int audio_stream;
    int video_stream;
    int subtitle_stream;

    int aud_stream_ix; //for reference to ic, not the streams member of Movie
    int vid_stream_ix;
    int sub_stream_ix;

    int playing;

    int overlay; //>0 if we are to use the overlay, otherwise <=0

    int ytop, xleft;
    int loops;

    SDL_mutex *general_mutex;
} PyMovie;

/*class methods and internals */
static PyObject* _movie_init_internal(PyTypeObject *type,const char *filename, PyObject* surface); 
static int _movie_init (PyTypeObject *type, PyObject *args);
static void _movie_dealloc (PyMovie *movie);
static PyObject* _movie_repr (PyMovie *movie);
static PyObject* _movie_str (PyMovie *movie);
static PyObject* _movie_play(PyMovie *movie, PyObject* args);
static PyObject* _movie_stop(PyMovie *movie);
static PyObject* _movie_pause(PyMovie *movie);
static PyObject* _movie_rewind(PyMovie *movie, PyObject* args);

/* Getters/setters */
static PyObject* _movie_get_paused (PyMovie *movie, void *closure);
static PyObject* _movie_get_playing (PyMovie *movie, void *closure);

/* C API interfaces */
static PyObject* PyMovie_New (char *fname, SDL_Surface *surf);

/*internals */

static void _dealloc_aud_stream(PyAudioStream *pas);
static void _dealloc_vid_stream(PyVideoStream *pvs);
static void _dealloc_sub_stream(PySubtitleStream *pss);