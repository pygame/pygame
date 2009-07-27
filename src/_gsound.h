#ifndef _GSOUND_H_
#define _GSOUND_H_
#include <Python.h>
#include <SDL/SDL_mixer.h>
#include <SDL.h>
#include <SDL_thread.h>
#include <libavformat/avformat.h>
typedef struct BufferNode
{
    uint8_t *buf;
    int      len;
    int64_t  pts;
    struct BufferNode
                *next;
}
BufferNode;


typedef struct BufferQueue
{
    BufferNode *first, *last;
    int size;
    SDL_mutex *mutex;
}
BufferQueue;

typedef struct AudioInfo
{
    double      audio_clock;        //keeps track of our PTS, in seconds
    double      old_clock;          //for when the video is paused
    int         channels;           //data for keeping track of the fraction of a second the current frame will take
    int         sample_rate;        //''
    int         current_frame_size; //''
    int         pts;                //current pts
    int         playing;            //if we've started playing any buffers
    int         channel;            //what channel the last buffer played on
    int         ended;              //whether or not we've "ended", so we know to output silence.
	int			paused;
    BufferQueue queue;              //queue of our buffers
    SDL_mutex   *mutex;
	//PyThreadState *_tstate;
	int restart;
	int holder;
	char *error;
}
AudioInfo;

AudioInfo ainfo;

int soundInit     (int freq, int size, int channels, int chunksize);
int soundQuit     (void);
int soundStart    (void);
int soundEnd      (void);
int playBuffer    (uint8_t *buf, uint32_t len, int channel, int64_t pts);
int stopBuffer    (int channel);
int pauseBuffer   (int channel);
int getPaused     (int channel);
double getAudioClock (void);
int getBufferQueueSize(void);
int seekBuffer    (uint8_t *buf, uint32_t len, int channel );
int setCallback   (void (*callback) (int channel));
int resetAudioInfo(void);
#endif /*_GSOUND_H_*/
