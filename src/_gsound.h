/*
  pygame - Python Game Library

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

*/

/*
 * _movie - movie support for pygame with ffmpeg
 * Author: Tyler Laing
 *
 * This module allows for the loading of, playing, pausing, stopping, and so on
 *  of a video file. Any format supported by ffmpeg is supported by this
 *  video player. Any bugs, please email trinioler@gmail.com :)
 */


#ifndef _GSOUND_H_
#define _GSOUND_H_
#include <Python.h>
#include <SDL/SDL_mixer.h>
#include <SDL.h>
#include <SDL_thread.h>
#include <libavformat/avformat.h>


/* Buffer Node for a simple FIFO queue of sound samples. */
typedef struct BufferNode
{
    uint8_t *buf;
    int      len;
    int64_t  pts;
    struct BufferNode
                *next;
}
BufferNode;

/* Queue Struct for handling sound samples. This enables one thread pushing
 * samples onto the queue with minimal interruption of grabbing a sample off the queue
 */
typedef struct BufferQueue
{
    BufferNode *first, *last;
    int size;
    SDL_mutex *mutex;
}
BufferQueue;

/* Audio Info Struct: holds various important info, like the clock, etc. */
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
    int            paused;
    BufferQueue queue;              //queue of our buffers
    SDL_mutex   *mutex;
    //PyThreadState *_tstate;
    int restart;
    double time_base;
}
AudioInfo;

/*Global pointer to audio info struct... means only one video at a time :/*/
AudioInfo *ainfo;

int soundInit     (int freq, int size, int channels, int chunksize, double time_base);
int soundQuit     (void);
int soundStart    (void);
int soundEnd      (void);
int playBuffer    (uint8_t *buf, uint32_t len, int channel, int64_t pts);
int stopBuffer    (int channel);
int pauseBuffer   (int channel);
int getPaused     (int channel);
double getAudioClock (void);
int getBufferQueueSize(void);
int seekBuffer    (double pts);
int setCallback   (void (*callback) (int channel));
int resetAudioInfo(void);
void playBufferQueue(int channel);
#endif /*_GSOUND_H_*/
