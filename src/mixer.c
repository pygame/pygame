/*
    pygame - Python Game Library
    Copyright (C) 2000-2001  Pete Shinners

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

    Pete Shinners
    pete@shinners.org
*/

/*
 *  mixer module for pygame
 */
#define PYGAMEAPI_MIXER_INTERNAL
#include "pygame.h"
#include "pygamedocs.h"
#include "mixer.h"

#define MIX_DEFAULT_CHUNKSIZE	1024



staticforward PyTypeObject PySound_Type;
staticforward PyTypeObject PyChannel_Type;
static PyObject* PySound_New(Mix_Chunk*);
static PyObject* PyChannel_New(int);
#define PySound_Check(x) ((x)->ob_type == &PySound_Type)
#define PyChannel_Check(x) ((x)->ob_type == &PyChannel_Type)


static int request_frequency = MIX_DEFAULT_FREQUENCY;
static int request_size = MIX_DEFAULT_FORMAT;
static int request_stereo = MIX_DEFAULT_CHANNELS;
static int request_chunksize = MIX_DEFAULT_CHUNKSIZE;


static int sound_init(PyObject* self, PyObject* arg, PyObject* kwarg);


struct ChannelData
{
    PyObject* sound;
    PyObject* queue;
    int endevent;
};
static struct ChannelData *channeldata = NULL;
static int numchanneldata = 0;

Mix_Music** current_music;
Mix_Music** queue_music;


static void endsound_callback(int channel)
{
    if(channeldata)
    {
	if(channeldata[channel].endevent && SDL_WasInit(SDL_INIT_VIDEO))
	{
	    SDL_Event e;
	    memset(&e, 0, sizeof(e));
	    e.type = channeldata[channel].endevent;
            if(e.type >= SDL_USEREVENT && e.type < SDL_NUMEVENTS) {
                e.user.code = channel;
            }

	    SDL_PushEvent(&e);
	}
	if(channeldata[channel].queue)
	{
	    int channelnum;
	    Mix_Chunk* sound = PySound_AsChunk(channeldata[channel].queue);
	    Py_XDECREF(channeldata[channel].sound);
	    channeldata[channel].sound = channeldata[channel].queue;
	    channeldata[channel].queue = NULL;
	    channelnum = Mix_PlayChannelTimed(channel, sound, 0, -1);
	    if(channelnum != -1)
	    	Mix_GroupChannel(channelnum, (intptr_t)sound);
	}
	else
	{
	    Py_XDECREF(channeldata[channel].sound);
	    channeldata[channel].sound = NULL;
	}
    }
}


static void autoquit(void)
{
        int i;
	if(SDL_WasInit(SDL_INIT_AUDIO))
	{
		Mix_HaltMusic();

                if(channeldata)
                {
                    for(i=0; i<numchanneldata; ++i)
		    {
                        Py_XDECREF(channeldata[i].sound);
			Py_XDECREF(channeldata[i].queue);
		    }
                    free(channeldata);
                    channeldata = NULL;
                    numchanneldata = 0;
                }

		if(current_music)
		{
			if(*current_music)
			{
				Mix_FreeMusic(*current_music);
				*current_music = NULL;
			}
			current_music = NULL;
		}
		if(queue_music)
		{
			if(*queue_music)
			{
				Mix_FreeMusic(*queue_music);
				*queue_music = NULL;
			}
			queue_music = NULL;
		}

		Mix_CloseAudio();
		SDL_QuitSubSystem(SDL_INIT_AUDIO);
	}
}


static PyObject* autoinit(PyObject* self, PyObject* arg)
{
	int freq, size, stereo, chunk;
	int i;
	freq = request_frequency;
	size = request_size;
	stereo = request_stereo;
	chunk = request_chunksize;

	if(!PyArg_ParseTuple(arg, "|iiii", &freq, &size, &stereo, &chunk))
		return NULL;
	if(stereo>=2)
		stereo = 2;
	else
		stereo = 1;

        if(size == 8) size = AUDIO_U8;
        else if(size == -8) size = AUDIO_S8;
        else if(size == 16) size = AUDIO_U16SYS;
        else if(size == -16) size = AUDIO_S16SYS;

	/*make chunk a power of 2*/
	for(i=0; 1<<i < chunk; ++i); //yes, semicolon on for loop
	chunk = max(1<<i, 256);

	if(!SDL_WasInit(SDL_INIT_AUDIO))
	{
		PyGame_RegisterQuit(autoquit);

                if(!channeldata) /*should always be null*/
                {
                    numchanneldata = MIX_CHANNELS;
                    channeldata = (struct ChannelData*)malloc(
			    sizeof(struct ChannelData)*numchanneldata);
                    for(i=0; i < numchanneldata; ++i)
		    {
                        channeldata[i].sound = NULL;
			channeldata[i].queue = NULL;
			channeldata[i].endevent = 0;
		    }
                }

		if(SDL_InitSubSystem(SDL_INIT_AUDIO) == -1)
			return PyInt_FromLong(0);

		if(Mix_OpenAudio(freq, (Uint16)size, stereo, chunk) == -1)
		{
			SDL_QuitSubSystem(SDL_INIT_AUDIO);
			return PyInt_FromLong(0);
		}
#if MIX_MAJOR_VERSION>=1 && MIX_MINOR_VERSION>=2 && MIX_PATCHLEVEL>=3
                Mix_ChannelFinished(endsound_callback);
#endif

              	Mix_VolumeMusic(127);
	}
	return PyInt_FromLong(1);
}


static PyObject* quit(PyObject* self, PyObject* arg)
{
	if(!PyArg_ParseTuple(arg, ""))
		return NULL;

	autoquit();

	RETURN_NONE
}


static PyObject* init(PyObject* self, PyObject* arg)
{
	PyObject* result;
	int value;

	result = autoinit(self, arg);
	if(!result)
		return NULL;
	value = PyObject_IsTrue(result);
	Py_DECREF(result);
	if(!value)
		return RAISE(PyExc_SDLError, SDL_GetError());

	RETURN_NONE
}


static PyObject* get_init(PyObject* self, PyObject* arg)
{
	int freq, channels, realform;
	Uint16 format;

	if(!PyArg_ParseTuple(arg, ""))
		return NULL;

	if(!SDL_WasInit(SDL_INIT_AUDIO))
		RETURN_NONE

	if(!Mix_QuerySpec(&freq, &format, &channels))
		RETURN_NONE

	//create a signed or unsigned number of bits per sample
	realform = format&~0xff ? -(format&0xff) : format&0xff;
	return Py_BuildValue("(iii)", freq, realform, channels>1);
}


static PyObject* pre_init(PyObject* self, PyObject* arg)
{
	request_frequency = MIX_DEFAULT_FREQUENCY;
	request_size = MIX_DEFAULT_FORMAT;
	request_stereo = MIX_DEFAULT_CHANNELS;
	request_chunksize = MIX_DEFAULT_CHUNKSIZE;

	if(!PyArg_ParseTuple(arg, "|iiii", &request_frequency, &request_size,
				&request_stereo, &request_chunksize))
		return NULL;

	RETURN_NONE
}




/* sound object methods */

static PyObject* snd_play(PyObject* self, PyObject* args)
{
	Mix_Chunk* chunk = PySound_AsChunk(self);
	int channelnum = -1;
	int loops = 0, playtime = -1;

	if(!PyArg_ParseTuple(args, "|ii", &loops, &playtime))
		return NULL;

	channelnum = Mix_PlayChannelTimed(-1, chunk, loops, playtime);
	if(channelnum == -1)
		RETURN_NONE

        Py_XDECREF(channeldata[channelnum].sound);
	Py_XDECREF(channeldata[channelnum].queue);
	channeldata[channelnum].queue = NULL;
        channeldata[channelnum].sound = self;
        Py_INCREF(self);

	//make sure volume on this arbitrary channel is set to full
	Mix_Volume(channelnum, 128);

	Mix_GroupChannel(channelnum, (intptr_t)chunk);
	return PyChannel_New(channelnum);
}


static PyObject* snd_get_num_channels(PyObject* self, PyObject* args)
{
	Mix_Chunk* chunk = PySound_AsChunk(self);
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	MIXER_INIT_CHECK();

	return PyInt_FromLong(Mix_GroupCount((intptr_t)chunk));
}


#if 0
static char docXX_snd_get_busy[] =
	"Sound.get_busy() -> int\n"
	"Returns the number of channels this sound is actively playing on";

static PyObject* snd_get_busy(PyObject* self, PyObject* args)
{
	Mix_Chunk* chunk = PySound_AsChunk(self);
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	MIXER_INIT_CHECK();

	/*there is no call for this in sdl_mixer, yargh */
	return PyInt_FromLong(Mix_PlayingGroup((int)chunk));
}
#endif


#if 0
static char docXX_snd_get_channel[] =
	"Sound.get_channel(int) -> Channel\n"
	"Retrieves a channel index for this sound";

static PyObject* snd_get_channel(PyObject* self, PyObject* args)
{
	Mix_Chunk* chunk = PySound_AsChunk(self);
	int chan;
	if(!PyArg_ParseTuple(args, "i", &chan))
		return NULL;

	MIXER_INIT_CHECK();

	return PyChannel_New(chan);
}
#endif


static PyObject* snd_fadeout(PyObject* self, PyObject* args)
{
	Mix_Chunk* chunk = PySound_AsChunk(self);
	int time;
	if(!PyArg_ParseTuple(args, "i", &time))
		return NULL;

	MIXER_INIT_CHECK();

	Mix_FadeOutGroup((intptr_t)chunk, time);
	RETURN_NONE
}


static PyObject* snd_stop(PyObject* self, PyObject* args)
{
	Mix_Chunk* chunk = PySound_AsChunk(self);
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	MIXER_INIT_CHECK();

	Mix_HaltGroup((intptr_t)chunk);
	RETURN_NONE
}


static PyObject* snd_set_volume(PyObject* self, PyObject* args)
{
	Mix_Chunk* chunk = PySound_AsChunk(self);
	float volume;

	if(!PyArg_ParseTuple(args, "f", &volume))
		return NULL;

	MIXER_INIT_CHECK();

	Mix_VolumeChunk(chunk, (int)(volume*128));
	RETURN_NONE
}


static PyObject* snd_get_volume(PyObject* self, PyObject* args)
{
	Mix_Chunk* chunk = PySound_AsChunk(self);
	int volume;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	MIXER_INIT_CHECK();

	volume = Mix_VolumeChunk(chunk, -1);
	return PyFloat_FromDouble(volume / 128.0);
}


static PyObject* snd_get_length(PyObject* self, PyObject* args)
{
	Mix_Chunk* chunk = PySound_AsChunk(self);
	int freq, channels, mixerbytes, numsamples;
	Uint16 format;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	MIXER_INIT_CHECK();

	Mix_QuerySpec(&freq, &format, &channels);
        if(format==AUDIO_S8 || format==AUDIO_U8)
            mixerbytes = 1;
        else
            mixerbytes = 2;
        numsamples = chunk->alen / mixerbytes / channels;

	return PyFloat_FromDouble((float)numsamples / (float)freq);
}



static PyMethodDef sound_methods[] =
{
	{ "play", snd_play, 1, DOC_SOUNDPLAY },
	{ "get_num_channels", snd_get_num_channels, 1, DOC_SOUNDGETNUMCHANNELS },

/*	{ "get_busy", snd_get_busy, doc_snd_get_busy }, */
/*	{ "get_channel", snd_get_channel doc_snd_get_channel }, */
	{ "fadeout", snd_fadeout, 1, DOC_SOUNDFADEOUT },
	{ "stop", snd_stop, 1, DOC_SOUNDSTOP },
	{ "set_volume", snd_set_volume, 1, DOC_SOUNDSETVOLUME },
	{ "get_volume", snd_get_volume, 1, DOC_SOUNDGETVOLUME },

        { "get_length", snd_get_length, 1, DOC_SOUNDGETLENGTH },
        
	{ NULL, NULL }
};


/*sound object internals*/

static void sound_dealloc(PySoundObject* self)
{
    	Mix_Chunk* chunk = PySound_AsChunk((PyObject*)self);
        if(chunk)
            Mix_FreeChunk(chunk);
        if(self->weakreflist)
            PyObject_ClearWeakRefs((PyObject*)self);
	self->ob_type->tp_free((PyObject*)self);
}


static PyTypeObject PySound_Type =
{
	PyObject_HEAD_INIT(NULL)
	0,
	"pygame.mixer.Sound",
	sizeof(PySoundObject),
	0,
	(destructor)sound_dealloc,
	0,
	NULL,
	NULL,					/*setattr*/
	NULL,					/*compare*/
	NULL,					/*repr*/
	NULL,					/*as_number*/
	NULL,					/*as_sequence*/
	NULL,					/*as_mapping*/
	(hashfunc)NULL, 		/*hash*/
	(ternaryfunc)NULL,		/*call*/
	(reprfunc)NULL, 		/*str*/
	0L,0L,0L,
    	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
	DOC_PYGAMEMIXERSOUND, /* Documentation string */
	0,					/* tp_traverse */
	0,					/* tp_clear */
	0,					/* tp_richcompare */
	offsetof(PySoundObject, weakreflist),    /* tp_weaklistoffset */
	0,					/* tp_iter */
	0,					/* tp_iternext */
	sound_methods,			        /* tp_methods */
	0,				        /* tp_members */
	0,				        /* tp_getset */
	0,					/* tp_base */
	0,					/* tp_dict */
	0,					/* tp_descr_get */
	0,					/* tp_descr_set */
	0,					/* tp_dictoffset */
	sound_init,			/* tp_init */
	0,					/* tp_alloc */
	0,	                /* tp_new */
};

	//PyType_GenericNew,	                /* tp_new */



/* channel object methods */


static PyObject* chan_play(PyObject* self, PyObject* args)
{
	int channelnum = PyChannel_AsInt(self);
	PyObject* sound;
	Mix_Chunk* chunk;
	int loops = 0, playtime = -1;

	if(!PyArg_ParseTuple(args, "O!|ii", &PySound_Type, &sound, &loops, &playtime))
		return NULL;
	chunk = PySound_AsChunk(sound);

	channelnum = Mix_PlayChannelTimed(channelnum, chunk, loops, playtime);
	if(channelnum != -1)
		Mix_GroupChannel(channelnum, (intptr_t)chunk);

        Py_XDECREF(channeldata[channelnum].sound);
	Py_XDECREF(channeldata[channelnum].queue);
        channeldata[channelnum].sound = sound;
	channeldata[channelnum].queue = NULL;
        Py_INCREF(sound);


	RETURN_NONE
}


static PyObject* chan_queue(PyObject* self, PyObject* args)
{
	int channelnum = PyChannel_AsInt(self);
	PyObject* sound;
	Mix_Chunk* chunk;

	if(!PyArg_ParseTuple(args, "O!", &PySound_Type, &sound))
		return NULL;
	chunk = PySound_AsChunk(sound);

	if(!channeldata[channelnum].sound) /*nothing playing*/
	{
	    channelnum = Mix_PlayChannelTimed(channelnum, chunk, 0, -1);
	    if(channelnum != -1)
		    Mix_GroupChannel(channelnum, (intptr_t)chunk);

            channeldata[channelnum].sound = sound;
            Py_INCREF(sound);
	}
	else
	{
	    Py_XDECREF(channeldata[channelnum].queue);
	    channeldata[channelnum].queue = sound;
	    Py_INCREF(sound);
	}

	RETURN_NONE
}


static PyObject* chan_get_busy(PyObject* self, PyObject* args)
{
	int channelnum = PyChannel_AsInt(self);
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	MIXER_INIT_CHECK();

	return PyInt_FromLong(Mix_Playing(channelnum));
}


static PyObject* chan_fadeout(PyObject* self, PyObject* args)
{
	int channelnum = PyChannel_AsInt(self);
	int time;
	if(!PyArg_ParseTuple(args, "i", &time))
		return NULL;

	MIXER_INIT_CHECK();

	Mix_FadeOutChannel(channelnum, time);
	RETURN_NONE
}


static PyObject* chan_stop(PyObject* self, PyObject* args)
{
	int channelnum = PyChannel_AsInt(self);
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	MIXER_INIT_CHECK();

	Mix_HaltChannel(channelnum);
	RETURN_NONE
}


static PyObject* chan_pause(PyObject* self, PyObject* args)
{
	int channelnum = PyChannel_AsInt(self);
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	MIXER_INIT_CHECK();

	Mix_Pause(channelnum);
	RETURN_NONE
}


static PyObject* chan_unpause(PyObject* self, PyObject* args)
{
	int channelnum = PyChannel_AsInt(self);
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	MIXER_INIT_CHECK();

	Mix_Resume(channelnum);
	RETURN_NONE
}


static PyObject* chan_set_volume(PyObject* self, PyObject* args)
{
	int channelnum = PyChannel_AsInt(self);
	float volume, stereovolume=-1.11f;
        int result;

	if(!PyArg_ParseTuple(args, "f|f", &volume, &stereovolume))
		return NULL;

	MIXER_INIT_CHECK();
#if MIX_MAJOR_VERSION>=1 && MIX_MINOR_VERSION>=2 && MIX_PATCHLEVEL>=1
        if((stereovolume <= -1.10f) && (stereovolume >= -1.12f)) {
            // The normal volume will be used.  No panning.  so panning is set to full.
            //   this is incase it was set previously to something else.
            // NOTE: there is no way to GetPanning variables.
            result = Mix_SetPanning(channelnum, (Uint8)255, (Uint8)255);
            
        } else {
            // NOTE: here the volume will be set to 1.0 and the panning will be used.
            result = Mix_SetPanning(channelnum, (Uint8)(volume*255), (Uint8)(stereovolume*255));
            volume = 1.0f;
        }
#else
        if(! ((stereovolume <= -1.10f) && (stereovolume >= -1.12f)))
            volume = (volume + stereovolume) * 0.5f;
#endif

	result = Mix_Volume(channelnum, (int)(volume*128));

	RETURN_NONE
}


static PyObject* chan_get_volume(PyObject* self, PyObject* args)
{
	int channelnum = PyChannel_AsInt(self);
	int volume;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	MIXER_INIT_CHECK();

	volume = Mix_Volume(channelnum, -1);

	return PyFloat_FromDouble(volume / 128.0);
}


static PyObject* chan_get_sound(PyObject* self, PyObject* args)
{
	int channelnum = PyChannel_AsInt(self);
	PyObject* sound;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	sound = channeldata[channelnum].sound;
	if(!sound)
	    RETURN_NONE

	Py_INCREF(sound);
	return sound;
}


static PyObject* chan_get_queue(PyObject* self, PyObject* args)
{
	int channelnum = PyChannel_AsInt(self);
	PyObject* sound;

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	sound = channeldata[channelnum].queue;
	if(!sound)
	    RETURN_NONE

	Py_INCREF(sound);
	return sound;
}


static PyObject* chan_set_endevent(PyObject* self, PyObject* args)
{
	int channelnum = PyChannel_AsInt(self);
	int event = SDL_NOEVENT;

	if(!PyArg_ParseTuple(args, "|i", &event))
		return NULL;

	channeldata[channelnum].endevent = event;
    	RETURN_NONE
}


static PyObject* chan_get_endevent(PyObject* self, PyObject* args)
{
	int channelnum = PyChannel_AsInt(self);

	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	return PyInt_FromLong(channeldata[channelnum].endevent);
}




static PyMethodDef channel_builtins[] =
{
	{ "play", chan_play, 1, DOC_CHANNELPLAY },
	{ "queue", chan_queue, 1, DOC_CHANNELQUEUE },
	{ "get_busy", chan_get_busy, 1, DOC_CHANNELGETBUSY },
	{ "fadeout", chan_fadeout, 1, DOC_CHANNELFADEOUT },
	{ "stop", chan_stop, 1, DOC_CHANNELSTOP },
	{ "pause", chan_pause, 1, DOC_CHANNELPAUSE },
	{ "unpause", chan_unpause, 1, DOC_CHANNELUNPAUSE },
	{ "set_volume", chan_set_volume, 1, DOC_CHANNELSETVOLUME },
	{ "get_volume", chan_get_volume, 1, DOC_CHANNELGETVOLUME },

	{ "get_sound", chan_get_sound, 1, DOC_CHANNELGETSOUND },
	{ "get_queue", chan_get_queue, 1, DOC_CHANNELGETQUEUE },

	{ "set_endevent", chan_set_endevent, 1, DOC_CHANNELSETENDEVENT },
	{ "get_endevent", chan_get_endevent, 1, DOC_CHANNELGETENDEVENT },

	{ NULL, NULL }
};


/* channel object internals */

static void channel_dealloc(PyObject* self)
{
	PyObject_DEL(self);
}


static PyObject* channel_getattr(PyObject* self, char* attrname)
{
	return Py_FindMethod(channel_builtins, self, attrname);
}


static PyTypeObject PyChannel_Type =
{
	PyObject_HEAD_INIT(NULL)
	0,
	"Channel",
	sizeof(PyChannelObject),
	0,
	channel_dealloc,
	0,
	channel_getattr,
	NULL,					/*setattr*/
	NULL,					/*compare*/
	NULL,					/*repr*/
	NULL,					/*as_number*/
	NULL,					/*as_sequence*/
	NULL,					/*as_mapping*/
	(hashfunc)NULL, 		/*hash*/
	(ternaryfunc)NULL,		/*call*/
	(reprfunc)NULL, 		/*str*/
	0L,0L,0L,0L,
	DOC_PYGAMEMIXERCHANNEL /* Documentation string */
};



/*mixer module methods*/

static PyObject* get_num_channels(PyObject* self, PyObject* args)
{
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	MIXER_INIT_CHECK();

	return PyInt_FromLong(Mix_GroupCount(-1));
}


static PyObject* set_num_channels(PyObject* self, PyObject* args)
{
	int numchans, i;
	if(!PyArg_ParseTuple(args, "i", &numchans))
		return NULL;

	MIXER_INIT_CHECK();
        if(numchans > numchanneldata)
        {
            channeldata= (struct ChannelData*)realloc(channeldata,
		    sizeof(struct ChannelData) * numchans);
            for(i = numchanneldata; i < numchans; ++i)
	    {
		channeldata[i].sound = NULL;
		channeldata[i].queue = NULL;
	    }
            numchanneldata = numchans;
        }

	Mix_AllocateChannels(numchans);
	RETURN_NONE
}


static PyObject* set_reserved(PyObject* self, PyObject* args)
{
	int numchans;
	if(!PyArg_ParseTuple(args, "i", &numchans))
		return NULL;

	MIXER_INIT_CHECK();

	Mix_ReserveChannels(numchans);
	RETURN_NONE
}


static PyObject* get_busy(PyObject* self, PyObject* args)
{
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	if(!SDL_WasInit(SDL_INIT_AUDIO))
		return PyInt_FromLong(0);

	return PyInt_FromLong(Mix_Playing(-1));
}


static PyObject* Channel(PyObject* self, PyObject* args)
{
	int chan;
	if(!PyArg_ParseTuple(args, "i", &chan))
		return NULL;

	MIXER_INIT_CHECK();

	return PyChannel_New(chan);
}


static PyObject* mixer_find_channel(PyObject* self, PyObject* args)
{
	int chan, force = 0;
	if(!PyArg_ParseTuple(args, "|i", &force))
		return NULL;

	MIXER_INIT_CHECK();

	chan = Mix_GroupAvailable(-1);
	if(chan == -1)
	{
		if(!force)
			RETURN_NONE
		chan = Mix_GroupOldest(-1);
	}
	return PyChannel_New(chan);
}


static PyObject* mixer_fadeout(PyObject* self, PyObject* args)
{
	int time;
	if(!PyArg_ParseTuple(args, "i", &time))
		return NULL;

	MIXER_INIT_CHECK();

	Mix_FadeOutChannel(-1, time);
	RETURN_NONE
}


static PyObject* mixer_stop(PyObject* self, PyObject* args)
{
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	MIXER_INIT_CHECK();

	Mix_HaltChannel(-1);
	RETURN_NONE
}


static PyObject* mixer_pause(PyObject* self, PyObject* args)
{
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	MIXER_INIT_CHECK();

	Mix_Pause(-1);
	RETURN_NONE
}


static PyObject* mixer_unpause(PyObject* self, PyObject* args)
{
	if(!PyArg_ParseTuple(args, ""))
		return NULL;

	MIXER_INIT_CHECK();

	Mix_Resume(-1);
	RETURN_NONE
}


#if 0
    /*DOC*/ static char doc_Sound[] =
    /*DOC*/    "pygame.mixer.Sound(file) -> Sound\n"
    /*DOC*/    "load a new soundfile\n"
    /*DOC*/    "\n"
    /*DOC*/    "Loads a new sound object from a WAV file. File can be a filename\n"
    /*DOC*/    "or a file-like object. The sound will be converted to match the\n"
    /*DOC*/    "current mode of the mixer.\n"
    /*DOC*/ ;
#endif
static int sound_init(PyObject* self, PyObject* arg, PyObject* kwarg)
{
	PyObject* file;
	char* name = NULL;
	Mix_Chunk* chunk;
    
        ((PySoundObject*)self)->chunk = NULL;

        if(!PyArg_ParseTuple(arg, "O", &file))
		return -1;

        
	if(!SDL_WasInit(SDL_INIT_AUDIO)) 
        {
		RAISE(PyExc_SDLError, "mixer system not initialized");
                return -1;
        }
        
	if(PyString_Check(file) || PyUnicode_Check(file))
	{
		if(!PyArg_ParseTuple(arg, "s", &name))
			return -1;
		Py_BEGIN_ALLOW_THREADS
		chunk = Mix_LoadWAV(name);
		Py_END_ALLOW_THREADS
	}
	else
	{
                SDL_RWops *rw;
		if(!(rw = RWopsFromPython(file)))
			return -1;
		if(RWopsCheckPython(rw))
			chunk = Mix_LoadWAV_RW(rw, 1);
		else
		{
			Py_BEGIN_ALLOW_THREADS
			chunk = Mix_LoadWAV_RW(rw, 1);
			Py_END_ALLOW_THREADS
		}
	}

	if(!chunk)
        {
		RAISE(PyExc_SDLError, SDL_GetError());
                return -1;
        }
        
        ((PySoundObject*)self)->chunk = chunk;
	return 0;
}




static PyMethodDef mixer_builtins[] =
{
	{ "__PYGAMEinit__", autoinit, 1, "auto initialize for mixer" },
	{ "init", init, 1, DOC_PYGAMEMIXERINIT },
	{ "quit", quit, 1, DOC_PYGAMEMIXERQUIT },
	{ "get_init", get_init, 1, DOC_PYGAMEMIXERGETINIT },
	{ "pre_init", pre_init, 1, DOC_PYGAMEMIXERPREINIT },

	{ "get_num_channels", get_num_channels, 1, DOC_PYGAMEMIXERGETNUMCHANNELS },
	{ "set_num_channels", set_num_channels, 1, DOC_PYGAMEMIXERSETNUMCHANNELS },
	{ "set_reserved", set_reserved, 1, DOC_PYGAMEMIXERSETRESERVED },

	{ "get_busy", get_busy, 1, DOC_PYGAMEMIXERGETBUSY },
	{ "Channel", Channel, 1, DOC_PYGAMEMIXERCHANNEL },
	{ "find_channel", mixer_find_channel, 1, DOC_PYGAMEMIXERFINDCHANNEL },
	{ "fadeout", mixer_fadeout, 1, DOC_PYGAMEMIXERFADEOUT },
	{ "stop", mixer_stop, 1, DOC_PYGAMEMIXERSTOP },
	{ "pause", mixer_pause, 1, DOC_PYGAMEMIXERPAUSE },
	{ "unpause", mixer_unpause, 1, DOC_PYGAMEMIXERUNPAUSE },
/*	{ "lookup_frequency", lookup_frequency, 1, doc_lookup_frequency },*/

	{ NULL, NULL }
};



static PyObject* PySound_New(Mix_Chunk* chunk)
{
	PySoundObject* soundobj;

	if(!chunk)
		return RAISE(PyExc_RuntimeError, "unable to create sound.");

	soundobj = (PySoundObject *)PySound_Type.tp_new(&PySound_Type, NULL, NULL);
	if(soundobj)
		soundobj->chunk = chunk;

	return (PyObject*)soundobj;
}



static PyObject* PyChannel_New(int channelnum)
{
	PyChannelObject* chanobj;

	if(channelnum < 0 || channelnum >= Mix_GroupCount(-1))
		return RAISE(PyExc_IndexError, "invalid channel index");

	chanobj = PyObject_NEW(PyChannelObject, &PyChannel_Type);
	if(!chanobj)
		return NULL;

	chanobj->chan = channelnum;
	return (PyObject*)chanobj;
}


PYGAME_EXPORT
void initmixer(void)
{
	PyObject *module, *dict, *apiobj, *music=NULL;
	static void* c_api[PYGAMEAPI_MIXER_NUMSLOTS];

	PyMIXER_C_API[0] = PyMIXER_C_API[0]; /*this cleans an unused warning*/

        if (PyType_Ready(&PySound_Type) < 0)
            return;
	PyType_Init(PyChannel_Type);

    /* create the module */
        PySound_Type.tp_new = &PyType_GenericNew;
	module = Py_InitModule3("mixer", mixer_builtins, DOC_PYGAMEMIXER);
	dict = PyModule_GetDict(module);

	PyDict_SetItemString(dict, "Sound", (PyObject *)&PySound_Type);
	PyDict_SetItemString(dict, "SoundType", (PyObject *)&PySound_Type);
	PyDict_SetItemString(dict, "ChannelType", (PyObject *)&PyChannel_Type);

	/* export the c api */
	c_api[0] = &PySound_Type;
	c_api[1] = PySound_New;
	c_api[2] = snd_play;
	c_api[3] = &PyChannel_Type;
	c_api[4] = PyChannel_New;
	c_api[5] = autoinit;
	c_api[6] = autoquit;
	apiobj = PyCObject_FromVoidPtr(c_api, NULL);
	PyDict_SetItemString(dict, PYGAMEAPI_LOCAL_ENTRY, apiobj);

	/*imported needed apis*/
	import_pygame_base();
	import_pygame_rwobject();

	music = PyImport_ImportModule("pygame.mixer_music");
        if(music)
	{
		PyObject* ptr, *dict;
		PyModule_AddObject(module, "music", music);
		dict = PyModule_GetDict(music);
		ptr = PyDict_GetItemString(dict, "_MUSIC_POINTER");
		current_music = (Mix_Music**)PyCObject_AsVoidPtr(ptr);
		ptr = PyDict_GetItemString(dict, "_QUEUE_POINTER");
		queue_music = (Mix_Music**)PyCObject_AsVoidPtr(ptr);
	}
	else /*music module not compiled? cleanly ignore*/
	{
            current_music = NULL;
           PyErr_Clear();
	}
}
