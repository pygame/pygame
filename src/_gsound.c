#include "_gsound.h"


#define _MIXER_DEFAULT_FREQUENCY 22050
#define _MIXER_DEFAULT_SIZE -16
#define _MIXER_DEFAULT_CHANNELS 2
#define _MIXER_DEFAULT_CHUNKSIZE 4096

int queue_get(BufferQueue *q, BufferNode **pkt1)
{
    int ret;
    SDL_mutexP(q->mutex); //lock
    for(;;)
    {
        *pkt1 = q->first;
        if (*pkt1)
        {
            q->first = (*pkt1)->next;
            if (!q->first)
                q->last = NULL;
            q->size--;
            ret = 1;
            break;
        }
        else
        {
            ret=0;
            break;
        }
    }
    SDL_mutexV(q->mutex);//unlock
    return ret;
}

int queue_put(BufferQueue *q, BufferNode *pkt)
{
    if (!pkt)
        return -1;
    SDL_mutexP(q->mutex); //lock
    pkt->next = NULL;
    if (!q->last)
        q->first = pkt;
    else
        q->last->next = pkt;
    q->last = pkt;
    q->size++;
    SDL_mutexV(q->mutex);//unlock
    return 0;
}
void queue_flush(BufferQueue *q)
{
    while(q->size>0)
    {
        BufferNode *node;
        queue_get(q, &node);
        PyMem_Free(node->buf);
        node->buf=NULL; //safety
        PyMem_Free(node);
        node=NULL;//safety
    }
}

void cb_mixer(int channel)
{
    playBuffer(NULL, (uint32_t) 0, channel, 0);
}

//initialize the mixer audio subsystem, code cribbed from mixer.c
int soundInit  (int freq, int size, int channels, int chunksize, PyThreadState *_tstate)
{
    Uint16 fmt = 0;
    int i;

    if (!freq)
    {
        freq = _MIXER_DEFAULT_FREQUENCY;
    }
    if (!size)
    {
        size = _MIXER_DEFAULT_SIZE;
    }
    if (!channels)
    {
        channels = _MIXER_DEFAULT_CHANNELS;
    }
    if (!chunksize)
    {
        chunksize = _MIXER_DEFAULT_CHUNKSIZE;
    }
    if (channels >= 2)
        channels = 2;
    else
        channels = 1;

    /* printf("size:%d:\n", size); */

    switch (size)
    {
    case 8:
        fmt = AUDIO_U8;
        break;
    case -8:
        fmt = AUDIO_S8;
        break;
    case 16:
        fmt = AUDIO_U16SYS;
        break;
    case -16:
        fmt = AUDIO_S16SYS;
        break;
    default:
        PyErr_Format(PyExc_ValueError, "unsupported size %i", size);
        return -1;
    }

    /* printf("size:%d:\n", size); */

    /*make chunk a power of 2*/
    for (i = 0; 1 << i < chunksize; ++i)
        ; //yes, semicolon on for loop
    if(1<<i >= 256)
        chunksize = 1<<i;
    else
    {
        chunksize=256;
    }
    if (!SDL_WasInit (SDL_INIT_AUDIO))
    {

        if (SDL_InitSubSystem (SDL_INIT_AUDIO) == -1)
            return -1;

        if (Mix_OpenAudio (freq, fmt, channels, chunksize) == -1)
        {
            SDL_QuitSubSystem (SDL_INIT_AUDIO);
            return -1;
        }

        /* A bug in sdl_mixer where the stereo is reversed for 8 bit.
           So we use this CPU hogging effect to reverse it for us.
           Hopefully this bug is fixed in SDL_mixer 1.2.9
        printf("MIX_MAJOR_VERSION :%d: MIX_MINOR_VERSION :%d: MIX_PATCHLEVEL :%d: \n", 
               MIX_MAJOR_VERSION, MIX_MINOR_VERSION, MIX_PATCHLEVEL);
        */

#if MIX_MAJOR_VERSION>=1 && MIX_MINOR_VERSION>=2 && MIX_PATCHLEVEL<=8
        if(fmt == AUDIO_U8)
        {
            if(!Mix_SetReverseStereo(MIX_CHANNEL_POST, 1))
            {
                /* We do nothing... because might as well just let it go ahead. */
                /* return RAISE (PyExc_SDLError, Mix_GetError());
                */
            }
        }
#endif

    }
    ainfo.channel = 0;
    ainfo.channels = channels;
    ainfo.audio_clock=0.0;
    ainfo.queue.size=0;
    ainfo.queue.first=ainfo.queue.last=NULL;
    ainfo.sample_rate=freq;
    ainfo.mutex = SDL_CreateMutex();
    ainfo.queue.mutex = SDL_CreateMutex();
    ainfo.ended=1;
    ainfo._tstate = _tstate;
    return 0;
}

int soundQuit(void)
{
    if (SDL_WasInit (SDL_INIT_AUDIO))
    {
        Mix_HaltMusic ();
        queue_flush(&ainfo.queue);
        Mix_ChannelFinished(NULL);
        Mix_CloseAudio ();
        SDL_QuitSubSystem (SDL_INIT_AUDIO);
    }
    return 0;
}

int soundStart (void)
{
    Mix_VolumeMusic (127);
    Mix_ChannelFinished(&cb_mixer);
    ainfo.ended=0;
    ainfo.audio_clock =0.0;
    ainfo.playing=0;
    ainfo.channel=0;
    ainfo.current_frame_size=1;
    //playBuffer(NULL, 0, 0, 0);
    return 0;
}

int soundEnd   (void)
{
    ainfo.ended = 1;
    //temp
    ainfo.restart=1;
    queue_flush(&ainfo.queue);
    return 0;
}
//	#define DECLAREGIL PyThreadState *_oldtstate;
//	#define GRABGIL    PyEval_AcquireLock();_oldtstate = PyThreadState_Swap(movie->_tstate);
//	#define RELEASEGIL PyThreadState_Swap(_oldtstate); PyEval_ReleaseLock();
/* Play a sound buffer, with a given length */
int playBuffer (uint8_t *buf, uint32_t len, int channel, int64_t pts)
{
	
	//temp
	AudioInfo *info = &ainfo;
	
	if(channel>info->channels-1)
	{
		channel=0;
	}
	if(info->channel<0 || info->channel >info->channels-1)
	{
		info->channel=0;
	}
	//SDL_mutexP(info->mutex);
    Mix_Chunk *mix;
    int false=0;
    int allocated=0;
    
    if(!info->ended && (info->queue.size>0||info->playing))
    {
        if(buf)
        {   
        	//not a callback call, so we copy the buffer into a buffernode and add it to the queue.
            BufferNode *node;
            node = (BufferNode *)PyMem_Malloc(sizeof(BufferNode));
            node->buf = (uint8_t *)PyMem_Malloc((size_t)len);
            memcpy(node->buf, buf, (size_t)len);
            node->len = len;
            node->next =NULL;
            node->pts = pts;
            queue_put(&ainfo.queue, node);
            //SDL_mutexV(info->mutex);
            return info->channel;
        }
        else if(!buf && info->queue.size==0)
        {  
            
            //callback call but when the queue is empty, so we just load a short empty sound.
            buf = (uint8_t *) PyMem_Malloc((size_t)1024);
            memset(buf, 0, (size_t)1024);
            info->current_frame_size=1;
            len=1024;
            allocated =1;
        	false=1;
        }
        else
        {
            //callback call, and convenienty enough, the queue has a buffer ready to go, so we copy it into buf
            BufferNode *new;
            queue_get(&ainfo.queue, &new);
            if(!new)
            {
                //SDL_mutexV(info->mutex);
                return -1;
            }
            info->current_frame_size=new->len;
            /*if (new->pts != AV_NOPTS_VALUE)
            {
                ainfo.audio_clock = (1.0/(double)ainfo.sample_rate) *new->pts;
            }*/
            buf = (uint8_t *)PyMem_Malloc((size_t)new->len);
            memcpy(buf, new->buf, new->len);
            len=new->len;
            PyMem_Free(new->buf);
            new->buf=NULL;
            PyMem_Free(new);
            new=NULL;
            allocated=1;
        }
    }
    
    //we assume that if stopped is true, then
    if(info->ended && !buf)
    {
        //callback call but when the queue is empty, so we just load a short empty sound.
        buf = (uint8_t *) PyMem_Malloc((size_t)1024);
        memset(buf, 0, (size_t)1024);
        info->current_frame_size=1;
        len=1024;
        allocated =1;
        false=1;
    }
    
    //regardless of 1st call, or a callback, we load the data from buf into a newly allocated block.
    mix= (Mix_Chunk *)PyMem_Malloc(sizeof(Mix_Chunk));
    mix->allocated=0;
	mix->abuf = (Uint8 *)PyMem_Malloc((size_t)len);
    memcpy(mix->abuf, buf, len);
    mix->alen = (Uint32 )len;
    mix->volume = 127;
    ainfo.playing = 1;
 	if(!info->ended)
	{
    	int bytes_per_sec = info->channels*info->sample_rate*2;
    	info->audio_clock+= (double) len/(double) bytes_per_sec;
	}
    ainfo.current_frame_size =len;
    int chan = ainfo.channel;
    
    //SDL_mutexV(ainfo.mutex);
    
    int playing = Mix_Playing(chan);
    if(playing && allocated &&false)
    {
    	return chan;
    }
    int ret = Mix_PlayChannel(chan, mix, 0);

    ainfo.channel = ret;
    //if buffer was allocated, we gotta clean it up.
    if(allocated)
    {
        PyMem_Free(buf);
    }
    return ret;
}
int stopBuffer (int channel)
{
    if(!channel)
        return 0;
    //Mix_HaltChannel(channel);
    return 0;
}

int pauseBuffer(int channel)
{
	AudioInfo *info = &ainfo;
    if(channel<=-1)
        return 0;
    int paused = Mix_Paused(channel);
    if(paused)
    {
    	info->audio_clock=info->old_clock;
    	info->ended=0;
        Mix_Resume(-1);
    }
    else
    {
    	ainfo.old_clock = ainfo.audio_clock;
    	ainfo.ended=1;
        Mix_Pause(-1);
    }
    return 0;
}

int getPaused (int channel)
{
    if(channel<=-1)
        return 0;
    return Mix_Paused(channel);
}
int seekBuffer (uint8_t *buf, uint32_t len, int channel)
{
    stopBuffer(channel);
    return playBuffer(buf, len, channel, 0);
}

int setCallback(void (*callback)(int channel))
{
    Mix_ChannelFinished(callback);
    return 1;
}

double getAudioClock(void)
{
    SDL_mutexP(ainfo.mutex);//lock
    int bytes_per_sec = ainfo.channels*ainfo.sample_rate*2;
    double pts = ainfo.audio_clock;
    //if(ainfo.current_frame_size!=1)
	//{
    	pts -= (double) ainfo.current_frame_size/(double) bytes_per_sec;
	//}
    SDL_mutexV(ainfo.mutex);
    return pts;
}

