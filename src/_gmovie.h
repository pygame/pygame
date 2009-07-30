#ifndef _GMOVIE_H_
#define _GMOVIE_H_

/* local includes */
#include "pygamedocs.h"
#include "pygame.h"
#include "pgcompat.h"
//#include "audioconvert.h"
#include "surface.h"
#include "_gsound.h"
#include "structmember.h"

/* Library includes */
/* This is as small as I could make it... */
#include <Python.h>
#include <SDL.h>
#include <SDL_thread.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>

/*constant definitions */
#define MAX_VIDEOQ_SIZE (5 * 256 * 1024)
#define MAX_AUDIOQ_SIZE (5 * 16 * 1024)
#define MAX_SUBTITLEQ_SIZE (5 * 16 * 1024)

/* no AV sync correction is done if below the AV sync threshold */
#define AV_SYNC_THRESHOLD 0.01
/* no AV correction is done if too big error */
#define AV_NOSYNC_THRESHOLD 10.0

//sets the module to single-thread mode.
#define THREADFREE 0

/*Thread management macros to save me a load of typing...*/
#if THREADFREE!=1
	#define DECLAREGIL PyThreadState *_oldtstate;
	#define GRABGIL    PyEval_AcquireLock();_oldtstate = PyThreadState_Swap(movie->_tstate);
	#define RELEASEGIL PyThreadState_Swap(_oldtstate); PyEval_ReleaseLock();
#else
	#define DECLAREGIL
	#define GRABGIL
	#define RELEASEGIL
#endif

/* Used to indicate when to flush the queues for seeking */
AVPacket flush_pkt;

/* Queues for already-loaded pictures, for rapid display */
#define VIDEO_PICTURE_QUEUE_SIZE 8
#define SUBPICTURE_QUEUE_SIZE    4
/* RGB24 or RGBA... */
/* In this case I've chosen RGB24 because its smaller */
#define RGB24 1
#define RGBA  0

#if RGB24
	#define RGBSTEP 3
#else
	#define RGBSTEP 4
#endif

#define BPP 1

//enables profiling info to be gathered
//#define PROFILE 1

#ifdef PROFILE
#include <math.h>
	typedef struct TimeSampleNode
	{
		int64_t sample;
		struct TimeSampleNode *next;
	}TimeSampleNode;
	typedef struct ImageScaleStats
	{
		double mean;
		int64_t max;
		int64_t min;
		double stdev;
		double median;
		double firstqu;
		double thirdqu;
		double serror;
		int64_t n_samples;
		struct TimeSampleNode *first;
		struct TimeSampleNode *last;
	}ImageScaleStats;
#endif
//included from ffmpeg header files, as the header file is not publically available.
#if defined(__ICC) || defined(__SUNPRO_C)
    #define DECLARE_ALIGNED(n,t,v)      t v __attribute__ ((aligned (n)))
    #define DECLARE_ASM_CONST(n,t,v)    const t __attribute__ ((aligned (n))) v
#elif defined(__GNUC__)
    #define DECLARE_ALIGNED(n,t,v)      t v __attribute__ ((aligned (n)))
    #define DECLARE_ASM_CONST(n,t,v)    static const t v attribute_used __attribute__ ((aligned (n)))
#elif defined(_MSC_VER)
    #define DECLARE_ALIGNED(n,t,v)      __declspec(align(n)) t v
    #define DECLARE_ASM_CONST(n,t,v)    __declspec(align(n)) static const t v
#elif HAVE_INLINE_ASM
    #error The asm code needs alignment, but we do not know how to do it for this compiler.
#else
    #define DECLARE_ALIGNED(n,t,v)      t v
    #define DECLARE_ASM_CONST(n,t,v)    static const t v
#endif

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

#define _ALPHA_BLEND(a, oldp, newp, s)\
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


/* structure definitions */
/* PacketQueue to hold incoming ffmpeg packets from the stream */
typedef struct PacketQueue
{
    AVPacketList *first_pkt, *last_pkt;
    int nb_packets;
    int size;
    int abort_request;
    SDL_mutex *mutex;
}
PacketQueue;

/* Holds already loaded pictures, so that decoding, and writing to a overlay/surface can happen while waiting
 * the <strong> very </strong> long time(in computer terms) to show the next frame. 
 */
typedef struct VidPicture
{
    SDL_Overlay *dest_overlay; /* Overlay for fast speedy yuv-rendering of the video */
    SDL_Surface *dest_surface; /* Surface for other desires, for example, rendering a video in a small portion of the screen */
    SDL_Rect    dest_rect;	   /* Dest-rect, which tells where to locate the video */
    int         width;         /* Width and height */
    int         height;
    int         xleft;		   /* Where left border of video is located */
    int         ytop;		   /* Where top border of video is located */
    int         overlay;	   /* Whether or not to use the overlay */
    int         ready; 		   /* Boolean to indicate this picture is ready to be used. After displaying the contents, it changes to False */
    double      pts;		   /* presentation time-stamp of the picture */
}
VidPicture;

#if 0
typedef struct SubPicture
{
	double pts;
	AVSubtitle sub;
} SubPicture;
#endif
enum {
    AV_SYNC_AUDIO_MASTER, /* default choice */
    AV_SYNC_VIDEO_MASTER,
    AV_SYNC_SUB_MASTER,
    AV_SYNC_EXTERNAL_CLOCK, /* synchronize to an external clock */
};

typedef struct PyMovie
{
    PyObject_HEAD
    /* General purpose members */
    SDL_Thread      *parse_tid;      /* Thread id for the decode_thread call */
    int              abort_request;  /* Tells whether or not to stop playing and return */
    int              paused; 		 /* Boolean for communicating to the threads to pause playback */
    int              last_paused;    /* For comparing the state of paused to what it was last time around. */
    char             filename[1024];
    char            *_backend;       //"FFMPEG_WRAPPER";
    int              overlay;        //>0 if we are to use the overlay, otherwise <=0
    int              playing;
    int              height;
    int              width;
    int              ytop;           //For use with SDL_Rect
    int              xleft;
    int              loops;          //Number of times to play the video
    int              resize_h;       //Indicator values that we have resized the video screen from the default
    int              resize_w;
    int 		     replay;         //we've played this once before, we're playing it again.
    AVInputFormat   *iformat;        //Format of the file
    SDL_mutex       *dest_mutex;     //mutex to control access to important info
    int              av_sync_type;   //determines the clock type we use
    AVFormatContext *ic;             /* context information about the format of the video file */
    int              stop;           //whether we're in a stop state...
    SDL_Surface     *canon_surf;     //pointer to the surface given by the programmer. We do NOT free this... it is not ours. We just write to it.
    PyThreadState   *_tstate;        //really do not touch this unless you have to. This is used for threading control and primitives.
	int finished;
    int diff_co; //counter

    /* Seek-info */
    int      seek_req;
    int      seek_flags; 
    int64_t  seek_pos;
	int64_t  start_time;     //used for seeking
    
    /* external clock members */
    double  external_clock; /* external clock base */
    int64_t external_clock_time;

    /* Audio stream members */
    double      audio_clock; 
    AVStream   *audio_st;    //audio stream struct
    PacketQueue audioq;      //audio packets
    /* samples output by the codec. we reserve more space for avsync compensation */
    DECLARE_ALIGNED(16,uint8_t,audio_buf1[(AVCODEC_MAX_AUDIO_FRAME_SIZE * 3) / 2]);
    uint8_t *audio_buf;      //pointer to a buffer
    AVPacket audio_pkt;      //current packet
    uint8_t *audio_pkt_data; //current packet data
    int      audio_pkt_size; //current packet size
    int64_t  audio_pts;      
    //int audio_volume; /*must self implement*/
    enum SampleFormat audio_src_fmt;
    //AVAudioConvert   *reformat_ctx;
    int               audio_stream;
    int               audio_disable;
    SDL_mutex        *audio_mutex;
    SDL_Thread       *audio_tid;
    int               channel;
    int               audio_paused;

    /* Frame/Video Management members */
    double     frame_timer;
    double     frame_last_pts;
    double     frame_last_delay;
    double     frame_delay; /*display time of each frame, based on fps*/
    double     video_clock; /*seconds of video frame decoded*/
    AVStream  *video_st;
    double     video_current_pts; /* current displayed pts (different from
                                     video_clock if frame fifos are used) */
    double     video_current_pts_time;
    double     timing;
    double     last_showtime;
    double     pts;
    int        video_stream;
    int        video_disable;
    /* simple ring_buffer queue for holding VidPicture structs */
    VidPicture pictq[VIDEO_PICTURE_QUEUE_SIZE];
    int pictq_size, pictq_windex, pictq_rindex;

    /* Thread id for the video_thread, when used in threaded mode */
    SDL_Thread *video_tid;

    PacketQueue videoq;
    SDL_mutex  *videoq_mutex;
    SDL_cond   *videoq_cond;
    struct SwsContext *img_convert_ctx;
#if 0
	/*subtitle */
	int sub_stream;
	int sub_stream_changed;
	AVStream *sub_st;
	PacketQueue subq;
	SubPicture subpq[SUBPICTURE_QUEUE_SIZE];
	int subpq_rindex, subpq_windex, subpq_size;
	SDL_mutex *subpq_mutex;
	int subtitle_disable;
#endif
#ifdef PROFILE
	ImageScaleStats *istats;
#endif

}
PyMovie;
/* end of struct definitions */
/* function definitions */

/* 		PacketQueue Management */
void packet_queue_init  (PacketQueue *q);
void packet_queue_flush (PacketQueue *q);
void packet_queue_end   (PacketQueue *q, int end);
int  packet_queue_put   (PacketQueue *q, AVPacket *pkt);
void packet_queue_abort (PacketQueue *q);
int  packet_queue_get   (PacketQueue *q, AVPacket *pkt, int block);

/* 		Misc*/
int  initialize_context     (PyMovie *movie, int threaded);
int  initialize_codec       (PyMovie *movie, int stream_index, int threaded);
/* 		Video Management */
int  video_open          (PyMovie *is, int index);
void video_image_display (PyMovie *is);
int  video_display       (PyMovie *is);
int  video_render        (PyMovie *movie);
int  queue_picture       (PyMovie *is, AVFrame *src_frame);
void update_video_clock  (PyMovie *movie, AVFrame* frame, double pts);
void video_refresh_timer (PyMovie *movie); //unlike in ffplay, this does the job of compute_frame_delay

/* 		Audio management */
int  synchronize_audio        (PyMovie *is, short *samples, int samples_size1, double pts);
int  audio_decode_frame       (PyMovie *is, double *pts_ptr);

/* 		General Movie Management */
void stream_seek            (PyMovie *is, int64_t pos, int rel);
void stream_pause           (PyMovie *is);
int  stream_component_open  (PyMovie *is, int stream_index, int threaded); //TODO: break down into separate functions
int  stream_component_start (PyMovie *is, int stream_index, int threaded);
void stream_component_end   (PyMovie *is, int stream_index, int threaded);
void stream_component_close (PyMovie *is, int stream_index, int threaded);
int  decoder                (void *arg);
void stream_open            (PyMovie *is, const char *filename, AVInputFormat *iformat, int threaded);
void stream_close           (PyMovie *is, int threaded);
void stream_cycle_channel   (PyMovie *is, int codec_type);
int  decoder_wrapper        (void *arg);

/* 		Clock Management */
double get_audio_clock    (PyMovie *is);
double get_video_clock    (PyMovie *is);
double get_external_clock (PyMovie *is);
double get_master_clock   (PyMovie *is);

#if 0
/*		Subtitle Management*/
int subtitle_render(void *arg);
void blend_subrect(AVPicture *dst, const AVSubtitleRect *rect, int imgw, int imgh);
void free_subpicture(SubPicture *sp);
#endif
#endif /*_GMOVIE_H_*/
