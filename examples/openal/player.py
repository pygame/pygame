import sys, wave
import pygame2
import pygame2.examples

try:
    import pygame2.openal as openal
    import pygame2.openal.constants as const
except ImportError:
    print ("No pygame2.openal support")
    sys.exit ()

# Format mappings, used to determine the sample byte size and channels
# used.
formatmap = {
    (1, 8) : const.AL_FORMAT_MONO8,
    (2, 8) : const.AL_FORMAT_STEREO8,
    (1, 16): const.AL_FORMAT_MONO16,
    (2, 16) : const.AL_FORMAT_STEREO16,
    }

class SimplePlayer (object):
    """A simple OpenAL-based audio player."""
    def __init__ (self, format, data, samplerate):
        if format not in formatmap:
            raise ValueError ("format must be a valid AL_FORMAT_* constant")
        self.channels, self.samplesize = format
        self.alformat = formatmap[format]

        # Create the device and default context.
        self.device = openal.Device ()
        self.context = openal.Context (self.device)
        self.context.make_current ()

        # Create a playback source and set the default gain and pitch.
        # Then set the position of the sound to be centered and static
        # (no velocity).
        self.sources = self.context.create_sources (1)

        self.sources.set_prop (self.sources.sources[0], const.AL_PITCH, 1., 'f')
        self.sources.set_prop (self.sources.sources[0], const.AL_GAIN, 1., 'f')
        self.sources.set_prop(self.sources.sources[0], const.AL_POSITION,
                              (0., 0., 0.), 'fa')
        self.sources.set_prop(self.sources.sources[0], const.AL_VELOCITY,
                              (0., 0., 0.), 'fa')

        # Create a buffer for holding the WAV data, load the data and
        # queue the buffer for playback.
        buffers = self.context.create_buffers (1)
        buffers.buffer_data (buffers.buffers[0],self.alformat, data, samplerate)
        self.sources.queue_buffers (self.sources.sources[0], buffers)

    def play (self):
        if not self.isplaying ():
            self.sources.play (self.sources.sources[0])

    def isplaying (self):
        state = self.sources.get_prop (self.sources.sources[0],
                                       const.AL_SOURCE_STATE, 'i')
        return state == const.AL_PLAYING

    def pause (self):
        if not self.isplaying ():
            return
        self.sources.pause (self.sources.sources[0])

def run ():
    if len (sys.argv) < 2:
        print ("Usage: player.py wavefile")
        print ("    Using an example wav file...")
        wavefp = wave.open (pygame2.examples.RESOURCES.get ("house_lo.wav"),
                            "rb")
    else:
        wavefp = wave.open (sys.argv[1], "rb")
        
    channels = wavefp.getnchannels ()
    bitrate = wavefp.getsampwidth () * 8
    samplerate = wavefp.getframerate ()

    player = SimplePlayer ((channels, bitrate),
                           wavefp.readframes (wavefp.getnframes ()),
                           samplerate)
    player.play ()

    # Wait until the playback is done
    while player.isplaying ():
        pass 
    
if __name__ == "__main__":
    run ()
