#ifndef _GMOVIE_H_
#define _GMOVIE_H_

/* local includes */
#include "pygamedocs.h"
#include "pygame.h"
#include "pgcompat.h"
#include "audioconvert.h"
#include "surface.h"
#include "_gsound.h"
#include "structmember.h"

/* Library includes */
#include <Python.h>
#include <SDL.h>
#include <SDL_thread.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>

/*constant definitions */
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

//sets the module to single-thread mode.
#define THREADFREE 0

#if THREADFREE!=1
	#define DECLAREGIL PyThreadState *_oldtstate;
	#define GRABGIL    PyEval_AcquireLock();_oldtstate = PyThreadState_Swap(movie->_tstate);
	#define RELEASEGIL PyThreadState_Swap(_oldtstate); PyEval_ReleaseLock(); 
#else
	#define DECLAREGIL 
	#define GRABGIL   
	#define RELEASEGIL
#endif
//backwards compatibility with blend_subrect
#define BPP 1

AVPacket flush_pkt;

/* Queues for already-loaded pictures, for rapid display */
#define VIDEO_PICTURE_QUEUE_SIZE 16

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

/* structure definitions */
/* PacketQueue to hold incoming ffmpeg packets from the stream */
typedef struct PacketQueue {
    AVPacketList *first_pkt, *last_pkt;
    int nb_packets;
    int size;
    int abort_request;
    SDL_mutex *mutex;
    SDL_cond *cond;
} PacketQueue;

/* Holds already loaded pictures, so that decoding, and writing to a overlay/surface can happen while waiting
 * the <strong> very </strong> long time(in computer terms) to show the next frame. 
 */
typedef struct VidPicture{
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
} VidPicture;


enum {
    AV_SYNC_AUDIO_MASTER, /* default choice */
    AV_SYNC_VIDEO_MASTER, 
    AV_SYNC_SUB_MASTER, 
    AV_SYNC_EXTERNAL_CLOCK, /* synchronize to an external clock */
};

typedef struct PyMovie {
	PyObject_HEAD
    /* General purpose members */
    SDL_Thread      *parse_tid; /* Thread id for the decode_thread call */
    int              abort_request;     /* Tells whether or not to stop playing and return */
    int              paused; 		   /* Boolean for communicating to the threads to pause playback */
	int              last_paused;       /* For comparing the state of paused to what it was last time around. */
    char             filename[1024];
    char            *_backend;  //"FFMPEG_WRAPPER";
    int              overlay; //>0 if we are to use the overlay, otherwise <=0
    int              playing;
    int              height;
    int              width;
    int              ytop;
    int              xleft;
    int              loops;
    int              resize_h;
    int              resize_w;
	int 		     replay;
	int64_t          start_time;
	AVInputFormat   *iformat;
	SDL_mutex       *dest_mutex;
	int              av_sync_type;
	AVFormatContext *ic;    /* context information about the format of the video file */
	int              stop;
	SDL_Surface     *canon_surf;
	PyThreadState   *_tstate; //really do not touch this unless you have to.
		
	
	/* Seek-info */
    int     seek_req;
    int     seek_flags;
    int64_t seek_pos;

	/* external clock members */
	double  external_clock; /* external clock base */
    int64_t external_clock_time;

	/* Audio stream members */
    double      audio_clock;
    AVStream   *audio_st;
    PacketQueue audioq;
    /* samples output by the codec. we reserve more space for avsync compensation */
    DECLARE_ALIGNED(16,uint8_t,audio_buf1[(AVCODEC_MAX_AUDIO_FRAME_SIZE * 3) / 2]);
    DECLARE_ALIGNED(16,uint8_t,audio_buf2[(AVCODEC_MAX_AUDIO_FRAME_SIZE * 3) / 2]);
    uint8_t *audio_buf;
    int      audio_buf_size; /* in bytes */
    int      audio_buf_index; /* in bytes */
    AVPacket audio_pkt;
    uint8_t *audio_pkt_data;
    int      audio_pkt_size;
    //int audio_volume; /*must self implement*/
	enum SampleFormat audio_src_fmt;
    AVAudioConvert *reformat_ctx;
    int             audio_stream;
	int             audio_disable;
	SDL_mutex      *audio_mutex;
	SDL_Thread     *audio_tid;
	int             channel;
	int             audio_paused;
	
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

} PyMovie;
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
void ConvertYUV420PtoRGBA   (AVPicture *YUV420P, SDL_Surface *OUTPUT, int interlaced );
void initializeLookupTables (void);
int  initialize_context(PyMovie *movie, int threaded);
/* 		Video Management */
int  video_open          (PyMovie *is, int index);
void video_image_display (PyMovie *is);
int  video_display       (PyMovie *is);
int  video_render        (PyMovie *movie);
int  queue_picture       (PyMovie *is, AVFrame *src_frame);
void update_video_clock  (PyMovie *movie, AVFrame* frame, double pts);
void video_refresh_timer (PyMovie *movie); //unlike in ffplay, this does the job of compute_frame_delay

/* 		Audio management */
int  audio_write_get_buf_size (PyMovie *is);
int  synchronize_audio        (PyMovie *is, short *samples, int samples_size1, double pts);
int  audio_decode_frame       (PyMovie *is, double *pts_ptr);

/* 		General Movie Management */
void stream_seek            (PyMovie *is, int64_t pos, int rel);
void stream_pause           (PyMovie *is);
int  stream_component_open  (PyMovie *is, int stream_index, int threaded); //TODO: break down into separate functions
int  stream_component_start (PyMovie *is, int stream_index, int threaded);
void stream_component_end   (PyMovie *is, int stream_index);
void stream_component_close (PyMovie *is, int stream_index);
int  decoder                (void *arg);
void stream_open            (PyMovie *is, const char *filename, AVInputFormat *iformat, int threaded);
void stream_close           (PyMovie *is);
void stream_cycle_channel   (PyMovie *is, int codec_type);
int  decoder_wrapper        (void *arg);

/* 		Clock Management */
double get_audio_clock    (PyMovie *is);
double get_video_clock    (PyMovie *is);
double get_external_clock (PyMovie *is);
double get_master_clock   (PyMovie *is);


#endif /*_GMOVIE_H_*/
