#include "_gsound.h"
#include <Python.h>

#define _MIXER_DEFAULT_FREQUENCY 22050
#define _MIXER_DEFAULT_SIZE -16
#define _MIXER_DEFAULT_CHANNELS 2
#define _MIXER_DEFAULT_CHUNKSIZE 4096

SDL_cond *audio_sig;
BufferQueue *queue;
int playing =0;
int queue_get(BufferQueue *q, BufferNode *node)
{
	BufferNode *pkt1;
	int ret;
 
	for(;;) {
        pkt1 = q->first;
        if (pkt1) {
            q->first = pkt1->next;
            if (!q->first)
                q->last = NULL;
            q->size--;
            node = pkt1;
            PyMem_Free(pkt1);
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

void cb_mixer(int channel)
{
	PyGILState_STATE gstate;
	gstate=PyGILState_Ensure();
	playBuffer(NULL, (uint32_t) 0);
	PyGILState_Release(gstate);
}

//initialize the mixer audio subsystem, code cribbed from mixer.c
int soundInit  (int freq, int size, int channels, int chunksize, SDL_cond *cond)
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
        Mix_VolumeMusic (127);
        Mix_ChannelFinished(&cb_mixer);
    }
    audio_sig=cond;
    if(audio_sig)
	{
		SDL_CondSignal(audio_sig);
	}
	if(!queue)
	{
		queue = (BufferQueue *)PyMem_Malloc(sizeof(BufferQueue));
		queue->size=0;
		queue->first=queue->last=NULL;	
	}
    return 0;
}

int soundQuit(void)
{
	Mix_CloseAudio();
	return 0;
}
	
/* Play a sound buffer, with a given length */
int playBuffer (uint8_t *buf, uint32_t len)
{
	Mix_Chunk *mix;
	if(queue->size>0)
	{
		if(buf)
		{
			BufferNode *node;
			node = (BufferNode *)PyMem_Malloc(sizeof(BufferNode));
			node->buf = (uint8_t *)PyMem_Malloc((size_t)len);
			memcpy(&node->buf, &buf, (size_t)len);
			node->len = len;
			node->next =NULL;
			queue_put(queue, node);
		}
		BufferNode *new;
		queue_get(queue, new);
		memcpy(&buf, &new->buf, new->len);
		len=new->len;
		PyMem_Free(new->buf);
		PyMem_Free(new);
	}
	else if(playing)
	{
		if(buf)
		{
			BufferNode *node;
			node = (BufferNode *)PyMem_Malloc(sizeof(BufferNode));
			node->buf = (uint8_t *)PyMem_Malloc((size_t)len);
			memcpy(&node->buf, &buf, (size_t)len);
			node->len = len;
			node->next =NULL;
			queue_put(queue, node);
		}
		return 0;
	}
	mix= (Mix_Chunk *)PyMem_Malloc(sizeof(Mix_Chunk));
	mix->allocated=1;
	mix->abuf = (Uint8 *)buf;
	mix->alen = (Uint32 )len;
	mix->volume = 127;
	playing = 1;
	int ret = Mix_PlayChannel(-1, mix, 0);
	return ret;
}
int stopBuffer (int channel)
{
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
		
int seekBuffer (uint8_t *buf, uint32_t len, int channel)
{
	stopBuffer(channel);
	return playBuffer(buf, len);
}

int setCallback(void (*callback)(int channel))
{
	Mix_ChannelFinished(callback);
	return 1;
}

