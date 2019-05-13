/* Auto generated file: with makeref.py .  Docs go in docs/reST/ref/ . */
#define DOC_PYGAMECDROM "pygame module for audio cdrom control"
#define DOC_PYGAMECDROMINIT "init() -> None\ninitialize the cdrom module"
#define DOC_PYGAMECDROMQUIT "quit() -> None\nuninitialize the cdrom module"
#define DOC_PYGAMECDROMGETINIT "get_init() -> bool\ntrue if the cdrom module is initialized"
#define DOC_PYGAMECDROMGETCOUNT "get_count() -> count\nnumber of cd drives on the system"
#define DOC_PYGAMECDROMCD "CD(id) -> CD\nclass to manage a cdrom drive"
#define DOC_CDINIT "init() -> None\ninitialize a cdrom drive for use"
#define DOC_CDQUIT "quit() -> None\nuninitialize a cdrom drive for use"
#define DOC_CDGETINIT "get_init() -> bool\ntrue if this cd device initialized"
#define DOC_CDPLAY "play(track, start=None, end=None) -> None\nstart playing audio"
#define DOC_CDSTOP "stop() -> None\nstop audio playback"
#define DOC_CDPAUSE "pause() -> None\ntemporarily stop audio playback"
#define DOC_CDRESUME "resume() -> None\nunpause audio playback"
#define DOC_CDEJECT "eject() -> None\neject or open the cdrom drive"
#define DOC_CDGETID "get_id() -> id\nthe index of the cdrom drive"
#define DOC_CDGETNAME "get_name() -> name\nthe system name of the cdrom drive"
#define DOC_CDGETBUSY "get_busy() -> bool\ntrue if the drive is playing audio"
#define DOC_CDGETPAUSED "get_paused() -> bool\ntrue if the drive is paused"
#define DOC_CDGETCURRENT "get_current() -> track, seconds\nthe current audio playback position"
#define DOC_CDGETEMPTY "get_empty() -> bool\nFalse if a cdrom is in the drive"
#define DOC_CDGETNUMTRACKS "get_numtracks() -> count\nthe number of tracks on the cdrom"
#define DOC_CDGETTRACKAUDIO "get_track_audio(track) -> bool\ntrue if the cdrom track has audio data"
#define DOC_CDGETALL "get_all() -> [(audio, start, end, length), ...]\nget all track information"
#define DOC_CDGETTRACKSTART "get_track_start(track) -> seconds\nstart time of a cdrom track"
#define DOC_CDGETTRACKLENGTH "get_track_length(track) -> seconds\nlength of a cdrom track"


/* Docs in a comment... slightly easier to read. */

/*

pygame.cdrom
pygame module for audio cdrom control

pygame.cdrom.init
 init() -> None
initialize the cdrom module

pygame.cdrom.quit
 quit() -> None
uninitialize the cdrom module

pygame.cdrom.get_init
 get_init() -> bool
true if the cdrom module is initialized

pygame.cdrom.get_count
 get_count() -> count
number of cd drives on the system

pygame.cdrom.CD
 CD(id) -> CD
class to manage a cdrom drive

pygame.cdrom.CD.init
 init() -> None
initialize a cdrom drive for use

pygame.cdrom.CD.quit
 quit() -> None
uninitialize a cdrom drive for use

pygame.cdrom.CD.get_init
 get_init() -> bool
true if this cd device initialized

pygame.cdrom.CD.play
 play(track, start=None, end=None) -> None
start playing audio

pygame.cdrom.CD.stop
 stop() -> None
stop audio playback

pygame.cdrom.CD.pause
 pause() -> None
temporarily stop audio playback

pygame.cdrom.CD.resume
 resume() -> None
unpause audio playback

pygame.cdrom.CD.eject
 eject() -> None
eject or open the cdrom drive

pygame.cdrom.CD.get_id
 get_id() -> id
the index of the cdrom drive

pygame.cdrom.CD.get_name
 get_name() -> name
the system name of the cdrom drive

pygame.cdrom.CD.get_busy
 get_busy() -> bool
true if the drive is playing audio

pygame.cdrom.CD.get_paused
 get_paused() -> bool
true if the drive is paused

pygame.cdrom.CD.get_current
 get_current() -> track, seconds
the current audio playback position

pygame.cdrom.CD.get_empty
 get_empty() -> bool
False if a cdrom is in the drive

pygame.cdrom.CD.get_numtracks
 get_numtracks() -> count
the number of tracks on the cdrom

pygame.cdrom.CD.get_track_audio
 get_track_audio(track) -> bool
true if the cdrom track has audio data

pygame.cdrom.CD.get_all
 get_all() -> [(audio, start, end, length), ...]
get all track information

pygame.cdrom.CD.get_track_start
 get_track_start(track) -> seconds
start time of a cdrom track

pygame.cdrom.CD.get_track_length
 get_track_length(track) -> seconds
length of a cdrom track

*/