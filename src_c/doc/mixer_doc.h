/* Auto generated file: with makeref.py .  Docs go in src/ *.doc . */
#define DOC_PYGAMEMIXER "pygame module for loading and playing sounds"

#define DOC_PYGAMEMIXERINIT "init(frequency=22050, size=-16, channels=2, buffer=4096, devicename=None) -> None\ninitialize the mixer module"

#define DOC_PYGAMEMIXERPREINIT "pre_init(frequency=22050, size=-16, channels=2, buffersize=4096, devicename=None) -> None\npreset the mixer init arguments"

#define DOC_PYGAMEMIXERQUIT "quit() -> None\nuninitialize the mixer"

#define DOC_PYGAMEMIXERGETINIT "get_init() -> (frequency, format, channels)\ntest if the mixer is initialized"

#define DOC_PYGAMEMIXERSTOP "stop() -> None\nstop playback of all sound channels"

#define DOC_PYGAMEMIXERPAUSE "pause() -> None\ntemporarily stop playback of all sound channels"

#define DOC_PYGAMEMIXERUNPAUSE "unpause() -> None\nresume paused playback of sound channels"

#define DOC_PYGAMEMIXERFADEOUT "fadeout(time) -> None\nfade out the volume on all sounds before stopping"

#define DOC_PYGAMEMIXERSETNUMCHANNELS "set_num_channels(count) -> None\nset the total number of playback channels"

#define DOC_PYGAMEMIXERGETNUMCHANNELS "get_num_channels() -> count\nget the total number of playback channels"

#define DOC_PYGAMEMIXERSETRESERVED "set_reserved(count) -> None\nreserve channels from being automatically used"

#define DOC_PYGAMEMIXERFINDCHANNEL "find_channel(force=False) -> Channel\nfind an unused channel"

#define DOC_PYGAMEMIXERGETBUSY "get_busy() -> bool\ntest if any sound is being mixed"

#define DOC_PYGAMEMIXERSOUND "Sound(filename) -> Sound\nSound(file=filename) -> Sound\nSound(buffer) -> Sound\nSound(buffer=buffer) -> Sound\nSound(object) -> Sound\nSound(file=object) -> Sound\nSound(array=object) -> Sound\nCreate a new Sound object from a file or buffer object"

#define DOC_SOUNDPLAY "play(loops=0, maxtime=0, fade_ms=0) -> Channel\nbegin sound playback"

#define DOC_SOUNDSTOP "stop() -> None\nstop sound playback"

#define DOC_SOUNDFADEOUT "fadeout(time) -> None\nstop sound playback after fading out"

#define DOC_SOUNDSETVOLUME "set_volume(value) -> None\nset the playback volume for this Sound"

#define DOC_SOUNDGETVOLUME "get_volume() -> value\nget the playback volume"

#define DOC_SOUNDGETNUMCHANNELS "get_num_channels() -> count\ncount how many times this Sound is playing"

#define DOC_SOUNDGETLENGTH "get_length() -> seconds\nget the length of the Sound"

#define DOC_SOUNDGETRAW "get_raw() -> bytes\nreturn a bytestring copy of the Sound samples."

#define DOC_PYGAMEMIXERCHANNEL "Channel(id) -> Channel\nCreate a Channel object for controlling playback"

#define DOC_CHANNELPLAY "play(Sound, loops=0, maxtime=0, fade_ms=0) -> None\nplay a Sound on a specific Channel"

#define DOC_CHANNELSTOP "stop() -> None\nstop playback on a Channel"

#define DOC_CHANNELPAUSE "pause() -> None\ntemporarily stop playback of a channel"

#define DOC_CHANNELUNPAUSE "unpause() -> None\nresume pause playback of a channel"

#define DOC_CHANNELFADEOUT "fadeout(time) -> None\nstop playback after fading channel out"

#define DOC_CHANNELSETVOLUME "set_volume(value) -> None\nset_volume(left, right) -> None\nset the volume of a playing channel"

#define DOC_CHANNELGETVOLUME "get_volume() -> value\nget the volume of the playing channel"

#define DOC_CHANNELGETBUSY "get_busy() -> bool\ncheck if the channel is active"

#define DOC_CHANNELGETSOUND "get_sound() -> Sound\nget the currently playing Sound"

#define DOC_CHANNELQUEUE "queue(Sound) -> None\nqueue a Sound object to follow the current"

#define DOC_CHANNELGETQUEUE "get_queue() -> Sound\nreturn any Sound that is queued"

#define DOC_CHANNELSETENDEVENT "set_endevent() -> None\nset_endevent(type) -> None\nhave the channel send an event when playback stops"

#define DOC_CHANNELGETENDEVENT "get_endevent() -> type\nget the event a channel sends when playback stops"



/* Docs in a comment... slightly easier to read. */

/*

pygame.mixer
pygame module for loading and playing sounds

pygame.mixer.init
 init(frequency=22050, size=-16, channels=2, buffer=4096, devicename=None) -> None
initialize the mixer module

pygame.mixer.pre_init
 pre_init(frequency=22050, size=-16, channels=2, buffersize=4096, devicename=None) -> None
preset the mixer init arguments

pygame.mixer.quit
 quit() -> None
uninitialize the mixer

pygame.mixer.get_init
 get_init() -> (frequency, format, channels)
test if the mixer is initialized

pygame.mixer.stop
 stop() -> None
stop playback of all sound channels

pygame.mixer.pause
 pause() -> None
temporarily stop playback of all sound channels

pygame.mixer.unpause
 unpause() -> None
resume paused playback of sound channels

pygame.mixer.fadeout
 fadeout(time) -> None
fade out the volume on all sounds before stopping

pygame.mixer.set_num_channels
 set_num_channels(count) -> None
set the total number of playback channels

pygame.mixer.get_num_channels
 get_num_channels() -> count
get the total number of playback channels

pygame.mixer.set_reserved
 set_reserved(count) -> None
reserve channels from being automatically used

pygame.mixer.find_channel
 find_channel(force=False) -> Channel
find an unused channel

pygame.mixer.get_busy
 get_busy() -> bool
test if any sound is being mixed

pygame.mixer.Sound
 Sound(filename) -> Sound
 Sound(file=filename) -> Sound
 Sound(buffer) -> Sound
 Sound(buffer=buffer) -> Sound
 Sound(object) -> Sound
 Sound(file=object) -> Sound
 Sound(array=object) -> Sound
Create a new Sound object from a file or buffer object

pygame.mixer.Sound.play
 play(loops=0, maxtime=0, fade_ms=0) -> Channel
begin sound playback

pygame.mixer.Sound.stop
 stop() -> None
stop sound playback

pygame.mixer.Sound.fadeout
 fadeout(time) -> None
stop sound playback after fading out

pygame.mixer.Sound.set_volume
 set_volume(value) -> None
set the playback volume for this Sound

pygame.mixer.Sound.get_volume
 get_volume() -> value
get the playback volume

pygame.mixer.Sound.get_num_channels
 get_num_channels() -> count
count how many times this Sound is playing

pygame.mixer.Sound.get_length
 get_length() -> seconds
get the length of the Sound

pygame.mixer.Sound.get_raw
 get_raw() -> bytes
return a bytestring copy of the Sound samples.

pygame.mixer.Channel
 Channel(id) -> Channel
Create a Channel object for controlling playback

pygame.mixer.Channel.play
 play(Sound, loops=0, maxtime=0, fade_ms=0) -> None
play a Sound on a specific Channel

pygame.mixer.Channel.stop
 stop() -> None
stop playback on a Channel

pygame.mixer.Channel.pause
 pause() -> None
temporarily stop playback of a channel

pygame.mixer.Channel.unpause
 unpause() -> None
resume pause playback of a channel

pygame.mixer.Channel.fadeout
 fadeout(time) -> None
stop playback after fading channel out

pygame.mixer.Channel.set_volume
 set_volume(value) -> None
 set_volume(left, right) -> None
set the volume of a playing channel

pygame.mixer.Channel.get_volume
 get_volume() -> value
get the volume of the playing channel

pygame.mixer.Channel.get_busy
 get_busy() -> bool
check if the channel is active

pygame.mixer.Channel.get_sound
 get_sound() -> Sound
get the currently playing Sound

pygame.mixer.Channel.queue
 queue(Sound) -> None
queue a Sound object to follow the current

pygame.mixer.Channel.get_queue
 get_queue() -> Sound
return any Sound that is queued

pygame.mixer.Channel.set_endevent
 set_endevent() -> None
 set_endevent(type) -> None
have the channel send an event when playback stops

pygame.mixer.Channel.get_endevent
 get_endevent() -> type
get the event a channel sends when playback stops

*/