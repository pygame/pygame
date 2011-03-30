/* Auto generated file: with makeref.py .  Docs go in src/ *.doc . */
#define DOC_PYGAMEMIXER "pygame module for loading and playing sounds"

#define DOC_PYGAMEMIXERINIT "pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=4096): return None\ninitialize the mixer module"

#define DOC_PYGAMEMIXERPREINIT "pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffersize=4096): return None\npreset the mixer init arguments"

#define DOC_PYGAMEMIXERQUIT "pygame.mixer.quit(): return None\nuninitialize the mixer"

#define DOC_PYGAMEMIXERGETINIT "pygame.mixer.get_init(): return (frequency, format, channels)\ntest if the mixer is initialized"

#define DOC_PYGAMEMIXERSTOP "pygame.mixer.stop(): return None\nstop playback of all sound channels"

#define DOC_PYGAMEMIXERPAUSE "pygame.mixer.pause(): return None\ntemporarily stop playback of all sound channels"

#define DOC_PYGAMEMIXERUNPAUSE "pygame.mixer.unpause(): return None\nresume paused playback of sound channels"

#define DOC_PYGAMEMIXERFADEOUT "pygame.mixer.fadeout(time): return None\nfade out the volume on all sounds before stopping"

#define DOC_PYGAMEMIXERSETNUMCHANNELS "pygame.mixer.set_num_channels(count): return None\nset the total number of playback channels"

#define DOC_PYGAMEMIXERGETNUMCHANNELS "pygame.mixer.get_num_channels(): return count\nget the total number of playback channels"

#define DOC_PYGAMEMIXERSETRESERVED "pygame.mixer.set_reserved(count): return None\nreserve channels from being automatically used"

#define DOC_PYGAMEMIXERFINDCHANNEL "pygame.mixer.find_channel(force=False): return Channel\nfind an unused channel"

#define DOC_PYGAMEMIXERGETBUSY "pygame.mixer.get_busy(): return bool\ntest if any sound is being mixed"

#define DOC_PYGAMEMIXERSOUND "pygame.mixer.Sound(filename): return Sound\npygame.mixer.Sound(file=filename): return Sound\npygame.mixer.Sound(buffer): return Sound\npygame.mixer.Sound(buffer=buffer): return Sound\npygame.mixer.Sound(object): return Sound\npygame.mixer.Sound(file=object): return Sound\npygame.mixer.Sound(array=object): return Sound\nCreate a new Sound object from a file or buffer object"

#define DOC_SOUNDPLAY "Sound.play(loops=0, maxtime=0, fade_ms=0): return Channel\nbegin sound playback"

#define DOC_SOUNDSTOP "Sound.stop(): return None\nstop sound playback"

#define DOC_SOUNDFADEOUT "Sound.fadeout(time): return None\nstop sound playback after fading out"

#define DOC_SOUNDSETVOLUME "Sound.set_volume(value): return None\nset the playback volume for this Sound"

#define DOC_SOUNDGETVOLUME "Sound.get_volume(): return value\nget the playback volume"

#define DOC_SOUNDGETNUMCHANNELS "Sound.get_num_channels(): return count\ncount how many times this Sound is playing"

#define DOC_SOUNDGETLENGTH "Sound.get_length(): return seconds\nget the length of the Sound"

#define DOC_SOUNDGETBUFFER "Sound.get_buffer(): return BufferProxy\nacquires a buffer object for the samples of the Sound."

#define DOC_PYGAMEMIXERCHANNEL "pygame.mixer.Channel(id): return Channel\nCreate a Channel object for controlling playback"

#define DOC_CHANNELPLAY "Channel.play(Sound, loops=0, maxtime=0, fade_ms=0): return None\nplay a Sound on a specific Channel"

#define DOC_CHANNELSTOP "Channel.stop(): return None\nstop playback on a Channel"

#define DOC_CHANNELPAUSE "Channel.pause(): return None\ntemporarily stop playback of a channel"

#define DOC_CHANNELUNPAUSE "Channel.unpause(): return None\nresume pause playback of a channel"

#define DOC_CHANNELFADEOUT "Channel.fadeout(time): return None\nstop playback after fading channel out"

#define DOC_CHANNELSETVOLUME "Channel.set_volume(value): return None\nChannel.set_volume(left, right): return None\nset the volume of a playing channel"

#define DOC_CHANNELGETVOLUME "Channel.get_volume(): return value\nget the volume of the playing channel"

#define DOC_CHANNELGETBUSY "Channel.get_busy(): return bool\ncheck if the channel is active"

#define DOC_CHANNELGETSOUND "Channel.get_sound(): return Sound\nget the currently playing Sound"

#define DOC_CHANNELQUEUE "Channel.queue(Sound): return None\nqueue a Sound object to follow the current"

#define DOC_CHANNELGETQUEUE "Channel.get_queue(): return Sound\nreturn any Sound that is queued"

#define DOC_CHANNELSETENDEVENT "Channel.set_endevent(): return None\nChannel.set_endevent(type): return None\nhave the channel send an event when playback stops"

#define DOC_CHANNELGETENDEVENT "Channel.get_endevent(): return type\nget the event a channel sends when playback stops"



/* Docs in a comments... slightly easier to read. */


/*

pygame.mixer
 pygame module for loading and playing sounds



pygame.mixer.init
 pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=4096): return None
initialize the mixer module



pygame.mixer.pre_init
 pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffersize=4096): return None
preset the mixer init arguments



pygame.mixer.quit
 pygame.mixer.quit(): return None
uninitialize the mixer



pygame.mixer.get_init
 pygame.mixer.get_init(): return (frequency, format, channels)
test if the mixer is initialized



pygame.mixer.stop
 pygame.mixer.stop(): return None
stop playback of all sound channels



pygame.mixer.pause
 pygame.mixer.pause(): return None
temporarily stop playback of all sound channels



pygame.mixer.unpause
 pygame.mixer.unpause(): return None
resume paused playback of sound channels



pygame.mixer.fadeout
 pygame.mixer.fadeout(time): return None
fade out the volume on all sounds before stopping



pygame.mixer.set_num_channels
 pygame.mixer.set_num_channels(count): return None
set the total number of playback channels



pygame.mixer.get_num_channels
 pygame.mixer.get_num_channels(): return count
get the total number of playback channels



pygame.mixer.set_reserved
 pygame.mixer.set_reserved(count): return None
reserve channels from being automatically used



pygame.mixer.find_channel
 pygame.mixer.find_channel(force=False): return Channel
find an unused channel



pygame.mixer.get_busy
 pygame.mixer.get_busy(): return bool
test if any sound is being mixed



pygame.mixer.Sound
 pygame.mixer.Sound(filename): return Sound
pygame.mixer.Sound(file=filename): return Sound
pygame.mixer.Sound(buffer): return Sound
pygame.mixer.Sound(buffer=buffer): return Sound
pygame.mixer.Sound(object): return Sound
pygame.mixer.Sound(file=object): return Sound
pygame.mixer.Sound(array=object): return Sound
Create a new Sound object from a file or buffer object



Sound.play
 Sound.play(loops=0, maxtime=0, fade_ms=0): return Channel
begin sound playback



Sound.stop
 Sound.stop(): return None
stop sound playback



Sound.fadeout
 Sound.fadeout(time): return None
stop sound playback after fading out



Sound.set_volume
 Sound.set_volume(value): return None
set the playback volume for this Sound



Sound.get_volume
 Sound.get_volume(): return value
get the playback volume



Sound.get_num_channels
 Sound.get_num_channels(): return count
count how many times this Sound is playing



Sound.get_length
 Sound.get_length(): return seconds
get the length of the Sound



Sound.get_buffer
 Sound.get_buffer(): return BufferProxy
acquires a buffer object for the samples of the Sound.



pygame.mixer.Channel
 pygame.mixer.Channel(id): return Channel
Create a Channel object for controlling playback



Channel.play
 Channel.play(Sound, loops=0, maxtime=0, fade_ms=0): return None
play a Sound on a specific Channel



Channel.stop
 Channel.stop(): return None
stop playback on a Channel



Channel.pause
 Channel.pause(): return None
temporarily stop playback of a channel



Channel.unpause
 Channel.unpause(): return None
resume pause playback of a channel



Channel.fadeout
 Channel.fadeout(time): return None
stop playback after fading channel out



Channel.set_volume
 Channel.set_volume(value): return None
Channel.set_volume(left, right): return None
set the volume of a playing channel



Channel.get_volume
 Channel.get_volume(): return value
get the volume of the playing channel



Channel.get_busy
 Channel.get_busy(): return bool
check if the channel is active



Channel.get_sound
 Channel.get_sound(): return Sound
get the currently playing Sound



Channel.queue
 Channel.queue(Sound): return None
queue a Sound object to follow the current



Channel.get_queue
 Channel.get_queue(): return Sound
return any Sound that is queued



Channel.set_endevent
 Channel.set_endevent(): return None
Channel.set_endevent(type): return None
have the channel send an event when playback stops



Channel.get_endevent
 Channel.get_endevent(): return type
get the event a channel sends when playback stops



*/

