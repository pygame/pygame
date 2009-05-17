"""
TODO:
    - write all docs as inline python docs.
    - rewrite docs for pygame doc style.
        - follow the format, and style of current docs.
    - export docs from .py to .doc.
    - create a background thread version for input threads.
        - that can automatically inject input into the event queue
          once the input object is running.  Like joysticks.
    - generate test stubs (probably after docs are written)
        - $ cd test/util/
          $ python gen_stubs.py sprite.Sprite
    - start writing tests.

Uses portmidi library for putting midi into and out of pygame.

This uses pyportmidi for now, but may use its own bindings at some
point.
"""




import pygame
import pygame.locals

import atexit


#
MIDIIN = pygame.locals.USEREVENT + 10
MIDIOUT = pygame.locals.USEREVENT + 11

_init = False



def init():
    """ Call the initialisation function before using the midi module.
    """
    global _init, pypm
    if not _init:
        import pygame.pypm
        pypm = pygame.pypm

        pypm.Initialize()
        _init = True
        atexit.register(quit)

def quit():
    """ Call this to quit the midi module.

        Called automatically atexit if you don't call it.
    """
    global _init, pypm
    if _init:
        # TODO: find all Input and Output classes and close them first?
        pypm.Terminate()
        _init = False
        del pypm
        del pygame.pypm




def get_count():
    """ gets the number of devices.
        Device ids range from 0 to get_count() -1
    """
    return pypm.CountDevices()




def get_default_input_device_id():
    """gets the device number of the default input device.

    Return the default device ID or -1 if there are no devices.
    The result can be passed to the Input()/Ouput() class.

    On the PC, the user can specify a default device by
    setting an environment variable. For example, to use device #1.

        set PM_RECOMMENDED_OUTPUT_DEVICE=1

    The user should first determine the available device ID by using
    the supplied application "testin" or "testout".

    In general, the registry is a better place for this kind of info,
    and with USB devices that can come and go, using integers is not
    very reliable for device identification. Under Windows, if
    PM_RECOMMENDED_OUTPUT_DEVICE (or PM_RECOMMENDED_INPUT_DEVICE) is
    *NOT* found in the environment, then the default device is obtained
    by looking for a string in the registry under:
        HKEY_LOCAL_MACHINE/SOFTWARE/PortMidi/Recommended_Input_Device
    and HKEY_LOCAL_MACHINE/SOFTWARE/PortMidi/Recommended_Output_Device
    for a string. The number of the first device with a substring that
    matches the string exactly is returned. For example, if the string
    in the registry is "USB", and device 1 is named
    "In USB MidiSport 1x1", then that will be the default
    input because it contains the string "USB".

    In addition to the name, get_device_info() returns "interf", which
    is the interface name. (The "interface" is the underlying software
	system or API used by PortMidi to access devices. Examples are
	MMSystem, DirectX (not implemented), ALSA, OSS (not implemented), etc.)
	At present, the only Win32 interface is "MMSystem", the only Linux
	interface is "ALSA", and the only Max OS X interface is "CoreMIDI".
    To specify both the interface and the device name in the registry,
    separate the two with a comma and a space, e.g.:
        MMSystem, In USB MidiSport 1x1
    In this case, the string before the comma must be a substring of
    the "interf" string, and the string after the space must be a
    substring of the "name" name string in order to match the device.

    Note: in the current release, the default is simply the first device
	(the input or output device with the lowest PmDeviceID).
    """
    return pypm.GetDefaultInputDeviceID()




def get_default_output_device_id():
    """get the device number of the default output device.

    TODO: rewrite the doc from get_default_input_device_id for here too.
           TODO: once those docs have been rewritten.  
    """
    return pypm.GetDefaultOutputDeviceID()


def get_device_info(an_id):
    """ returns (interf, name, input, output, opened)

    If the id is out of range, the function returns None.
    """
    return pypm.GetDeviceInfo(an_id) 


class MidiException(Exception):
    def __init__(self, value):
        self.parameter = value
    def __str__(self):
        return repr(self.parameter)


class Input(object):


    def __init__(self, device_id, buffer_size=4096):
        """
        The buffer_size specifies the number of input events to be buffered 
        waiting to be read using Input.read().
        """
        self._input = pypm.Input(device_id, buffer_size)
        self.device_id = device_id


    def read(self, length):
        """ [[status,data1,data2,data3],timestamp]
        """
        return self._input.Read(length)

    def poll(self):
        """ returns true if there's data, or false if not.
            Otherwise it raises a MidiException.
        """
        r = self._input.Poll()
        if r == pypm.TRUE:
            return True
        elif r == pypm.FALSE:
            return False
        else:
            err_text = GetErrorText(r)
            raise MidiException( (r, err_text) )




class Output(object):
    def __init__(self, device_id, latency = 0, buffer_size = 4096):
        """
        The buffer_size specifies the number of output events to be 
        buffered waiting for output.  (In some cases -- see below -- 
        PortMidi does not buffer output at all and merely passes data 
        to a lower-level API, in which case buffersize is ignored.)

        latency is the delay in milliseconds applied to timestamps to determine
        when the output should actually occur. (If latency is < 0, 0 is 
        assumed.)

        If latency is zero, timestamps are ignored and all output is delivered
        immediately. If latency is greater than zero, output is delayed until
        the message timestamp plus the latency. (NOTE: time is measured 
        relative to the time source indicated by time_proc. Timestamps are 
        absolute, not relative delays or offsets.) In some cases, PortMidi 
        can obtain better timing than your application by passing timestamps 
        along to the device driver or hardware. Latency may also help you 
        to synchronize midi data to audio data by matching midi latency to 
        the audio buffer latency.
        """
        self._output = pypm.Output(device_id, latency)
        self.device_id = device_id

    def write(self, data):
        """

        output a series of MIDI information in the form of a list:
             write([[[status <,data1><,data2><,data3>],timestamp],
                    [[status <,data1><,data2><,data3>],timestamp],...])
        <data> fields are optional
        example: choose program change 1 at time 20000 and
        send note 65 with velocity 100 500 ms later.
             write([[[0xc0,0,0],20000],[[0x90,60,100],20500]])
        notes:
          1. timestamps will be ignored if latency = 0.
          2. To get a note to play immediately, send MIDI info with
             timestamp read from function Time.
          3. understanding optional data fields:
               write([[[0xc0,0,0],20000]]) is equivalent to
               write([[[0xc0],20000]])

        Can send up to 1024 elements in your data list, otherwise an 
         IndexError exception is raised.
        """
        self._output.Write(data)


    def write_short(self, status, data1 = 0, data2 = 0):
        """ write_short(status <, data1><, data2>)
             output MIDI information of 3 bytes or less.
             data fields are optional
             status byte could be:
                  0xc0 = program change
                  0x90 = note on
                  etc.
                  data bytes are optional and assumed 0 if omitted
             example: note 65 on with velocity 100
                  WriteShort(0x90,65,100)
        """
        self._output.WriteShort(status, data1, data2)


    def write_sys_ex(self, when, msg):
        """ write_sys_ex(<timestamp>,<msg>)
        writes a timestamped system-exclusive midi message.
        <msg> can be a *list* or a *string*
        example:
            (assuming y is an input MIDI stream)
            y.write_sys_ex(0,'\\xF0\\x7D\\x10\\x11\\x12\\x13\\xF7')
                              is equivalent to
            y.write_sys_ex(pygame.midi.Time,
            [0xF0, 0x7D, 0x10, 0x11, 0x12, 0x13, 0xF7])
        """
        self._output.WriteSysEx(when, msg)


    def note_on(self, note, velocity=None, channel = 0):
        """ note_on(note, velocity=None, channel = 0)
        Turn a note on in the output stream.
        """
        if velocity is None:
            velocity = 0
        self.write_short(0x90+channel, note, velocity)

    def note_off(self, note, velocity=None, channel = 0):
        """ note_off(note, velocity=None, channel = 0)
        Turn a note off in the output stream.
        """
        if velocity is None:
            velocity = 0

        self.write_short(0x80 + channel, note, velocity)


    def set_instrument(self, instrument_id, channel = 0):
        """ set_instrument(instrument_id, channel = 0)
        Select an instrument, with a value between 0 and 127.
        """
        if not (0 <= instrument_id <= 127):
            raise ValueError("Undefined instrument id: %d" % instrument_id)
        self.write_short(0xc0+channel, instrument_id)


def time():
    """ Returns the current time in ms of the PortMidi timer.
    """
    return pypm.Time()



def midis2events(midis, device_id):
    """ takes a sequence of midi events and returns a list of pygame events.
    """
    evs = []
    for midi in midis:

        ((status,data1,data2,data3),timestamp) = midi

        e = pygame.event.Event(MIDIIN,
                               status=status,
                               data1=data1,
                               data2=data2,
                               data3=data3,
                               timestamp=timestamp,
                               vice_id = device_id)
        evs.append( e )


    return evs






