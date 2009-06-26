#include "_gsound.h"
#include <Python.h>

#define _MIXER_DEFAULT_FREQUENCY 22050
#define _MIXER_DEFAULT_SIZE -16
#define _MIXER_DEFAULT_CHANNELS 2
#define _MIXER_DEFAULT_CHUNKSIZE 4096

BufferQueue queue;
int playing =0;
int s_channel;
int queue_get(BufferQueue *q, BufferNode **pkt1)
{
	//BufferNode *pkt1;
	int ret;
 
	for(;;) {
        *pkt1 = q->first;
        if (*pkt1) {
            q->first = (*pkt1)->next;
            if (!q->first)
                q->last = NULL;
            q->size--;
            //node = pkt1;
            //PyMem_Free(pkt1);
            ret = 1;
            break;
        } else {
            ret=0;
            break;
        }
    }
    return ret;
}	

int queue_put(BufferQueue *q, BufferNode *pkt)
{
    if (!pkt)
        return -1;
    pkt->next = NULL;
    if (!q->last)
        q->first = pkt;
    else
        q->last->next = pkt;
    q->last = pkt;
    q->size++;
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
	//PyGILState_STATE gstate;
	//gstate=PyGILState_Ensure();
	Mix_Chunk *mix = Mix_GetChunk(channel);
	if(mix==NULL)
	{
		playBuffer(NULL, (uint32_t) 0, channel);
		return;
	}
	if(mix->abuf==NULL)
	{
		PyMem_Free(mix);
		mix=NULL;
		playBuffer(NULL, (uint32_t) 0, channel);
		return;
	}
	//PyMem_Free(mix->abuf);
	//mix->abuf=NULL;
	//PyMem_Free(mix);
	//mix=NULL;
	//PySys_WriteStdout("Callback called.\n");
	playBuffer(NULL, (uint32_t) 0, channel);
	//PySys_WriteStdout("Callback finished.\n");
	//PyGILState_Release(gstate);
}

//initialize the mixer audio subsystem, code cribbed from mixer.c
int soundInit  (int freq, int size, int channels, int chunksize)
{
	Uint16 fmt = 0;
    int i;

    if (!freq) {
	freq = _MIXER_DEFAULT_FREQUENCY;
    }
    if (!size) {
	size = _MIXER_DEFAULT_SIZE;
    }
    if (!channels) {
	channels = _MIXER_DEFAULT_CHANNELS;
    }
    if (!chunksize) {
	chunksize = _MIXER_DEFAULT_CHUNKSIZE;
    }
    if (channels >= 2)
        channels = 2;
    else
        channels = 1;

    /* printf("size:%d:\n", size); */

    switch (size) {
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
    for (i = 0; 1 << i < chunksize; ++i); //yes, semicolon on for loop
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
		s_channel = 0;
        /* A bug in sdl_mixer where the stereo is reversed for 8 bit.
           So we use this CPU hogging effect to reverse it for us.
           Hopefully this bug is fixed in SDL_mixer 1.2.9
        printf("MIX_MAJOR_VERSION :%d: MIX_MINOR_VERSION :%d: MIX_PATCHLEVEL :%d: \n", 
               MIX_MAJOR_VERSION, MIX_MINOR_VERSION, MIX_PATCHLEVEL);
        */

#if MIX_MAJOR_VERSION>=1 && MIX_MINOR_VERSION>=2 && MIX_PATCHLEVEL<=8
        if(fmt == AUDIO_U8) {
            if(!Mix_SetReverseStereo(MIX_CHANNEL_POST, 1)) {
                /* We do nothing... because might as well just let it go ahead. */
                /* return RAISE (PyExc_SDLError, Mix_GetError());
                */
            }
        }
#endif
       
    }
    
	
	queue.size=0;
	queue.first=queue.last=NULL;	
    return 0;
}

int soundQuit(void)
{
	if (SDL_WasInit (SDL_INIT_AUDIO))
    {
        Mix_HaltMusic ();
		queue_flush(&queue);
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
	return 0;
}
	
int soundEnd   (void)
{
	Mix_Volume(-1, 0);
	Mix_ChannelFinished(NULL);
	Mix_HaltChannel(-1);
	return 0;
}
	
/* Play a sound buffer, with a given length */
int playBuffer (uint8_t *buf, uint32_t len, int channel)
{
	Mix_Chunk *mix;
	int allocated=0;
	if(queue.size>0||playing)
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
			queue_put(&queue, node);
			return channel;
		}
		else if(!buf && queue.size==0)
		{
			//callback call but when the queue is empty, so we just load a short empty sound.
			buf = (uint8_t *) PyMem_Malloc((size_t)128);
			memset(buf, 0, (size_t)128);
			len=128;
			allocated =1;	
		}
		else
		{
			//callback call, and convenienty enough, the queue has a buffer ready to go, so we copy it into buf
			BufferNode *new;
			queue_get(&queue, &new);
			if(!new)
				return -1;
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
	//regardless of 1st call, or a callback, we load the data from buf into a newly allocated block.
	mix= (Mix_Chunk *)PyMem_Malloc(sizeof(Mix_Chunk));
	mix->allocated=0;
	mix->abuf = (Uint8 *)PyMem_Malloc((size_t)len);
	memcpy(mix->abuf, buf, len);
	mix->alen = (Uint32 )len;
	mix->volume = 127;
	playing = 1;
	int ret = Mix_PlayChannel(s_channel, mix, 0);
	s_channel = ret;
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
	Mix_Chunk *mix;
	mix = Mix_GetChunk(channel);
	Mix_HaltChannel(channel);
	if(mix)
	{
		mix->alen = -1;
		mix->abuf=NULL; //this data still exists, we just make the pointer to it be NULL
		mix->allocated=0;
		mix->volume = 0;
		Mix_FreeChunk(mix);
		mix=NULL; //safety
	}
	return 0;
}
int pauseBuffer(int channel)
{
	if(channel<=-1)
		return 0;
	int paused = Mix_Paused(channel);
	if(paused)
	{
		Mix_Resume(channel);
	}
	else
	{
		Mix_Pause(channel);
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
	return playBuffer(buf, len, channel);
}

int setCallback(void (*callback)(int channel))
{
	Mix_ChannelFinished(callback);
	return 1;
}

