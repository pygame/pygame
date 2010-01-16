##    pygame - Python Game Library
##    Copyright (C) 2008  Rene Dudfield
##
##    This library is free software; you can redistribute it and/or
##    modify it under the terms of the GNU Library General Public
##    License as published by the Free Software Foundation; either
##    version 2 of the License, or (at your option) any later version.
##
##    This library is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
##    Library General Public License for more details.
##
##    You should have received a copy of the GNU Library General Public
##    License along with this library; if not, write to the Free
##    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
##

"""
PyPortMidi interface methods to ease interaction with MIDI devices.
"""

import pygame2
import atexit

# Necessary globals
_pypm = None    # The PyPortMidi module binding
_init = False   # Initialization flag.

def _check_init ():
    if not _init:
        raise pygame2.Error ("pygame2.midi is not initialized")

def init ():
    """init () -> None
    
    Initializes the midi module.

    Raises a pygame2.Error on failure.
    """
    global _pypm, _init
    if not _init:
        try:
            import pygame2.midi.pypm
            _pypm = pygame2.midi.pypm
        except ImportError:
            raise pygame2.Error ("pygame2.midi.pypm not usable")
        _pypm.Initialize ()
        _init = True
        atexit.register (quit)

def quit ():
    """quit () -> None
    
    Uninitializes the midi module and releases all hold resources.
    """
    global _pypm, _init
    if _pypm:
        _pypm.Terminate ()
        _pypm = None
        _init = False

def was_init ():
    """was_init () -> bool
    
    Gets, whether the midi module was already initialized.
    """
    return _init

def time ():
    """time () -> int
    
    Gets the time in milliseconds since the midi module was initialized.
    """
    return _pypm.Time ()

def get_count ():
    """get_count () -> int
    
    Gets the number of available midi devices.
    
    Raises a pygame2.Error, if the midi module is not initialized..
    """
    _check_init ()
    return _pypm.CountDevices ()


def get_default_input_id ():
    """get_default_input_id () -> int
    
    Returns the default device ID or -1 if there are no devices.
    The result can be passed to the Input()/Ouput() class.
    
    On the PC, the user can specify a default device by
    setting an environment variable. For example, to use device #1.
    
        set PM_RECOMMENDED_INPUT_DEVICE=1
    
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

    Raises a pygame2.Error, if the midi module is not initialized.
    """
    _check_init ()
    return _pypm.GetDefaultInputDeviceID ()

def get_default_output_id ():
    """get_default_output_id () -> int
    
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

    Raises a pygame2.Error, if the midi module is not initialized.
    """
    _check_init ()
    return _pypm.GetDefaultOutputDeviceID()

def get_device_info (id):
    """get_device_info (id) -> string, string, bool, bool, bool
    
    Gets information about a midi device.
    
    Gets enhanced information about a midi device. The return values are
    
    * the name of the device, e.g. 'ALSA'
    * the enhanced description of the device, e.g. 'Midi Through Port-0'
    * a boolean indicating, whether the device is an input device
    * a boolean indicating, whether the device is an output device
    * a boolean indicating, whether the device is opened
    
    in this order.
    
    Raises a TypeError, if the *id* is not a integer value.
    Raises a ValueError, if the *id* is not within the range of available
    devices.
    Raises a pygame2.Error, if the midi module is not initialized.
    """
    _check_init ()
    _id = int (id)
    if _id < 0 or _id >= get_count ():
        raise ValueError ("id must be in the range of available devices")
    name, desc, input, output, opened = _pypm.GetDeviceInfo (_id)
    return name, desc, input == 1, output == 1, opened == 1

class Input (object):
    """Input (id, bufsize=4096) -> Input
    
    Creates a new Input instance for a specific device.
    
    The Input class gives read access to a specific midi device, which allows
    input, with buffering support.
    
    Raises a ValueError, if the *id* is not within the range of available
    devices.
    Raises a pygame2.Error, if the midi module is not initialized.
    """
    def __init__ (self, id, bufsize=4096):
        _check_init ()
        interf, name, input, output, opened = get_device_info (id)
        if not input:
            raise pygame2.Error ("device is not an input device")
        
        self._input = _pypm.Input(device_id, buffer_size)
        self._id = id

    def close (self):
        """I.close () -> None
        
        Closes the Input device.
        Raises a pygame2.Error, if the midi module is not initialized.
        """
        _check_init ()
        if self._input:
            self._input.Close ()
        self._input = None
    
    def read (self, amount):
        """I.read (amount) -> list

        Reads a certain *amount* of midi events from the buffer.
        
        Reads from the Input buffer and gives back midi events in the form
        
        [ [[status,data1,data2,data3],timestamp],
          [[status,data1,data2,data3],timestamp], ...]
        
        Raises a pygame2.Error, if the midi module is not initialized.
        """
        _check_init ()
        if not self._input:
            raise pygame2.Error ("device is not opened")
        return self._input.Read (amount)
    
    def poll (self):
        """I.poll () -> bool

        Gets, whether data is available on the buffer or not.
        
        Raises a pygame2.Error, if the midi module is not initialized.
        """
        _check_init ()
        if not self._input:
            raise pygame2.Error ("device is not opened")

        r = self._input.Poll ()
        if r == _pypm.TRUE:
            return True
        elif r == _pypm.FALSE:
            return False
        else:
            err_text = _pypm.GetErrorText (r)
            raise pygame2.Error (err_text)

class Output (object):
    """Output (id, latency=0, bufsize=4096) -> Output
    
    Creates a new Output instance for a specific device.
    
    The Output class gives write access to a specific midi device, which allows
    output.

    *latency* is the delay in milliseconds applied to timestamps to determine
    when the output should actually occur. (If *latency* is < 0, 0 is 
    assumed.)

    If *latency* is zero, timestamps are ignored and all output is delivered
    immediately. If *latency* is greater than zero, output is delayed until
    the message timestamp plus the latency. (NOTE: time is measured 
    relative to the time source indicated by time_proc. Timestamps are 
    absolute, not relative delays or offsets.) In some cases, PortMidi 
    can obtain better timing than your application by passing timestamps 
    along to the device driver or hardware. Latency may also help you 
    to synchronize midi data to audio data by matching midi latency to 
    the audio buffer latency.
    
    Raises a ValueError, if the *id* is not within the range of available
    devices.
    Raises a pygame2.Error, if the midi module is not initialized.
    """
    def __init__ (self, id, latency=0):
        _check_init ()
        interf, name, input, output, opened = get_device_info (id)
        if not output:
            raise pygame2.Error ("device is not an output device")
        
        self.output = _pypm.Output (id, latency)
        self._id = id
    
    def close (self):
        """O.close () -> None
        
        Closes the Output device.
        
        Raises a pygame2.Error, if the midi module is not initialized.
        """
        _check_init ()
        if self._input:
            self._input.Close ()
        self._input = None
    
    def abort (self):
        """O.abort () -> None
        
        Aborts outgoing messages immediately.
        
        The caller should immediately close the output port;
        this call may result in transmission of a partial midi message.
        
        Raises a pygame2.Error, if the midi module is not initialized.
        """
        _check_init()
        if self._output:
            self._output.Abort ()
    
    def write (self, data):
        """O.write (data) -> None
        
        Writes midi data to the output device.
        
        Writes series of MIDI information in the form of a list:
        
             write([[[status <,data1><,data2><,data3>],timestamp],
                    [[status <,data1><,data2><,data3>],timestamp],...])
        
        <data> fields are optional.
        
        Example: choose program change 1 at time 20000 and
        send note 65 with velocity 100 500 ms later.
             
             write([[[0xc0,0,0],20000],[[0x90,60,100],20500]])
        
        Notes:
          1. timestamps will be ignored if latency = 0.
          2. To get a note to play immediately, send MIDI info with
             timestamp read from function time().
          3. understanding optional data fields:
               
               write([[[0xc0,0,0],20000]])
             
             is equivalent to
               
               write([[[0xc0],20000]])

        This can send up to 1024 elements in your data list, otherwise an 
        IndexError is raised.
        Raises a pygame2.Error, if the midi module is not initialized.
        """
        _check_init ()
        if not self._output:
            raise pygame2.Error ("device is not opened")
        self._output.Write (data)
    
    def write_short (self, status, data1=0, data2=0):
        """O.write_short (status <, data1><, data2>) -> None

        Writes MIDI information of 3 bytes or less.

        Writes a short MIDI information to the device. The data fields are
        optional and assumed 0, if omitted. The *status* byte could be:

            0xc0 = program change
            0x90 = note on
            ...
            
        Example: note 65 on with velocity 100
            
            write_short(0x90,65,100)
        
        Raises a pygame2.Error, if the midi module is not initialized.
        """
        _check_init ()
        if not self._output:
            raise pygame2.Error ("device is not opened")
        self._output.WriteShort (status, data1, data2)

    def write_sys_ex (self, timestamp, msg):
        """O.write_sys_ex (timestamp, msg) -> None
        
        Writes a timestamped, system-exclusive message.
        
        Writes a system-exclusive message *msg*, which can be either a byte
        buffer or string  - or - list of bytes.
        
        Example:
        
            write_sys_ex (0, '\\xF0\\x7D\\x10\\x11\\x12\\x13\\xF7')
        
        is equivalent to
        
            write_sys_ex (pygame2.midi.time (),
                          [0xF0,0x7D,0x10,0x11,0x12,0x13,0xF7])
        
        Raises a pygame2.Error, if the midi module is not initialized.
        """
        _check_init ()
        if not self._output:
            raise pygame2.Error ("device is not opened")
        self._output.WriteSysExc (timestamp, msg)

    def note_on (self, note, velocity=None, channel=0):
        """O.note_on (note, velocity=None, channel=0) -> None
        
        Turn a note on in the output stream.

        Turn a note on in the output stream. The note must already be off for
        this to work correctly.
        
        Raises a ValueError, if *channel* is not in the range [0, 15].
        Raises a pygame2.Error, if the midi module is not initialized.
        """
        if velocity is None:
            velocity = 0

        if not (0 <= channel <= 15):
            raise ValueError ("Channel not between 0 and 15")

        self.write_short (0x90 + channel, note, velocity)
    
    def note_off (self, note, velocity=None, channel=0):
        """O.note_off (note, velocity=None, channel = 0) -> None

        Turn a note off in the output stream.

        Turn a note off in the output stream. The note must already
        be on for this to work correctly.
        
        Raises a ValueError, if *channel* is not in the range [0, 15].
        Raises a pygame2.Error, if the midi module is not initialized.
        """
        if velocity is None:
            velocity = 0

        if not (0 <= channel <= 15):
            raise ValueError ("Channel not between 0 and 15")

        self.write_short (0x80 + channel, note, velocity)

    def set_instrument (self, id, channel=0):
        """O.set_instrument (id, channel=0) -> None
        
        Select an instrument, with a value between 0 and 127.

        Selects an instrument, where the *id* is in a range of [0, 127].
        
        Raises a ValueError, if *id* is not in the range [0, 127].
        Raises a ValueError, if *channel* is not in the range [0, 15].
        Raises a pygame2.Error, if the midi module is not initialized.
        """
        if not (0 <= id <= 127):
            raise ValueError ("instrument id not between 0 and 127")
        if not (0 <= channel <= 15):
            raise ValueError ("Channel not between 0 and 15")
        self.write_short (0xc0 + channel, id)
