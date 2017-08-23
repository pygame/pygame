# pyPortMidi
# Python bindings for PortMidi
# John Harrison
# http://sound.media.mit.edu/~harrison
# harrison@media.mit.edu
# written in Pyrex

__version__ = "0.0.6"

import array

# CHANGES:

# 0.0.6: (Feb 25, 2011) christopher arndt <chris@chrisarndt.de>
#   Do not try to close device in Input/Output.__dealloc__ if not open
#   Major code layout clean up

# 0.0.5: (June 1st, 2009)
#   Output no longer calls abort when it deallocates.
#   Added abort and close methods.
#   Need to call Abort() explicityly if you want that to happen.


#
# 0.0.3: (March 15, 2005)
#   changed everything from tuples to lists
#   return 4 values for PmRead instead of 3 (for SysEx)
#   minor fixes for flexibility and error checking
#   flushed out DistUtils package and added Mac and Linux compile support
#   Markus Pfaff: added ability for WriteSysEx to accept lists as well
#                 as strings

# 0.0.2:
#   fixed pointer to function calls to avoid necessity of pyport library

# 0.0.1:
#   initial release


FILT_ACTIVE = 0x1
FILT_SYSEX = 0x2
FILT_CLOCK = 0x4
FILT_PLAY = 0x8
FILT_F9 = 0x10
FILT_TICK = 0x10
FILT_FD = 0x20
FILT_UNDEFINED = 0x30
FILT_RESET = 0x40
FILT_REALTIME = 0x7F
FILT_NOTE = 0x80
FILT_CHANNEL_AFTERTOUCH = 0x100
FILT_POLY_AFTERTOUCH = 0x200
FILT_AFTERTOUCH = 0x300
FILT_PROGRAM = 0x400
FILT_CONTROL = 0x800
FILT_PITCHBEND = 0x1000
FILT_MTC = 0x2000
FILT_SONG_POSITION = 0x4000
FILT_SONG_SELECT = 0x8000
FILT_TUNE = 0x10000
FALSE = 0
TRUE = 1

cdef extern from "portmidi.h":
    ctypedef enum PmError:
        pmNoError = 0,
        pmHostError = -10000,
        pmInvalidDeviceId, # out of range or output device when input is requested or vice versa
        pmInsufficientMemory,
        pmBufferTooSmall,
        pmBufferOverflow,
        pmBadPtr,
        pmBadData, # illegal midi data, e.g. missing EOX
        pmInternalError,
        pmBufferMaxSize, # buffer is already as large as it can be

    PmError Pm_Initialize()
    PmError Pm_Terminate()
    ctypedef void PortMidiStream
    ctypedef PortMidiStream PmStream # CHECK THIS!
    ctypedef int PmDeviceID
    int Pm_HasHostError(PortMidiStream * stream)
    char *Pm_GetErrorText(PmError errnum)
    Pm_GetHostErrorText(char * msg, unsigned int len)

    ctypedef struct PmDeviceInfo:
        int structVersion
        char *interf # underlying MIDI API, e.g. MMSystem or DirectX
        char *name   # device name, e.g. USB MidiSport 1x1
        int input    # true iff input is available
        int output   # true iff output is available
        int opened   # used by generic PortMidi code to do error checking on arguments

    int Pm_CountDevices()
    PmDeviceID Pm_GetDefaultInputDeviceID()
    PmDeviceID Pm_GetDefaultOutputDeviceID()
    ctypedef long PmTimestamp
    ctypedef PmTimestamp(*PmTimeProcPtr)(void *time_info)
    # PmBefore is not defined...
    PmDeviceInfo* Pm_GetDeviceInfo(PmDeviceID id)

    PmError Pm_OpenInput(PortMidiStream** stream,
                         PmDeviceID inputDevice,
                         void *inputDriverInfo,
                         long bufferSize,
                         long (*PmPtr) (), # long = PtTimestamp
                         void *time_info)

    PmError Pm_OpenOutput(PortMidiStream** stream,
                          PmDeviceID outputDevice,
                          void *outputDriverInfo,
                          long bufferSize,
                          #long (*PmPtr) (), # long = PtTimestamp
                          PmTimeProcPtr time_proc, # long = PtTimestamp
                          void *time_info,
                          long latency)

    PmError Pm_SetFilter(PortMidiStream* stream, long filters)
    PmError Pm_Abort(PortMidiStream* stream)
    PmError Pm_Close(PortMidiStream* stream)
    ctypedef long PmMessage

    ctypedef struct PmEvent:
        PmMessage message
        PmTimestamp timestamp

    PmError Pm_Read(PortMidiStream *stream, PmEvent *buffer, long length)
    PmError Pm_Poll(PortMidiStream *stream)
    int Pm_Channel(int channel)
    PmError Pm_SetChannelMask(PortMidiStream *stream, int mask)
    PmError Pm_Write(PortMidiStream *stream, PmEvent *buffer, long length)
    PmError Pm_WriteSysEx(PortMidiStream *stream, PmTimestamp when,
                          unsigned char *msg)

cdef extern from "porttime.h":
    ctypedef enum PtError:
        ptNoError = 0,
        ptHostError = -10000,
        ptAlreadyStarted,
        ptAlreadyStopped,
        ptInsufficientMemory

    ctypedef long PtTimestamp
    ctypedef void (* PtCallback)(PtTimestamp timestamp, void *userData)
    PtError Pt_Start(int resolution, PtCallback *callback, void *userData)
    PtTimestamp Pt_Time()


def Initialize():
    """Initialize PortMidi library.

    This function must be called once before any other function or class from
    this module can be used.

    """
    Pm_Initialize()
    # equiv to TIME_START: start timer w/ ms accuracy
    Pt_Start(1, NULL, NULL)

def Terminate():
    """Terminate use of PortMidi library.

    Call this to clean up Midi streams when done.

    If you do not call this on Windows machines when you are done with MIDI,
    your system may crash.

    """
    Pm_Terminate()

def GetDefaultInputDeviceID():
    """Return the number of the default MIDI input device.

    See the PortMidi documentation on how the default device is set and
    determined.

    """
    return Pm_GetDefaultInputDeviceID()

def GetDefaultOutputDeviceID():
    """Return the number of the default MIDI output device.

    See the PortMidi documentation on how the default device is set and
    determined.

    """
    return Pm_GetDefaultOutputDeviceID()

def CountDevices():
    """Return number of available MIDI (input and output) devices."""

    return Pm_CountDevices()

def GetDeviceInfo(device_no):
    """Return device info tuple for MIDI device given by device_no.

    The returned tuple has the following five items:

    * underlying MIDI API (string)
    * device name (string)
    * whether device can be opened as input (1) or not (0)
    * whether device can be opened as output (1) or not (0)
    * whether device is currently opened (1) or not (0)

    """
    cdef PmDeviceInfo *info

    # disregarding the constness from Pm_GetDeviceInfo,
    # since pyrex doesn't do const.
    info = <PmDeviceInfo *>Pm_GetDeviceInfo(device_no)

    if info != NULL:
        return info.interf, info.name, info.input, info.output, info.opened
    # return None

def Time():
    """Return the current time in ms of the PortMidi timer."""

    return Pt_Time()

def GetErrorText(err):
    """Return human-readable error message translated from error number."""

    return Pm_GetErrorText(err)

def Channel(chan):
    """Return Channel object for given MIDI channel number 1 - 16.

    Channel(<chan>) is used with ChannelMask on input MIDI streams.

    Example:

    To receive input on channels 1 and 10 on a MIDI stream called
    MidiIn::

        MidiIn.SetChannelMask(pypm.Channel(1) | pypm.Channel(10))

    .. note::
        PyPortMidi Channel function has been altered from
        the original PortMidi c call to correct for what
        seems to be a bug --- i.e. channel filters were
        all numbered from 0 to 15 instead of 1 to 16.

    """
    return Pm_Channel(chan - 1)


cdef class Output:
    """Represents an output MIDI stream device.

    Takes the form::

        output = pypm.Output(output_device, latency)

    latency is in ms. If latency == 0 then timestamps for output are ignored.

    """
    cdef int device
    cdef PmStream *midi
    cdef int debug
    cdef int _aborted

    def __init__(self, output_device, latency=0):
        """Instantiate MIDI output stream object."""

        cdef PmError err
        #cdef PtTimestamp (*PmPtr) ()
        cdef PmTimeProcPtr PmPtr

        self.device = output_device
        self.debug = 0
        self._aborted = 0

        if latency == 0:
            PmPtr = NULL
        else:
            PmPtr = <PmTimeProcPtr>&Pt_Time

        if self.debug:
            print "Opening Midi Output"

        # Why is buffer size 0 here?
        err = Pm_OpenOutput(&(self.midi), output_device, NULL, 0, PmPtr, NULL,
                            latency)
        if err < 0:
            errmsg = Pm_GetErrorText(err)
            # Something's amiss here - if we try to throw an Exception
            # here, we crash.
            if not err == -10000:
                raise Exception(errmsg)
            else:
                print "Unable to open Midi OutputDevice=%i: %s" % (
                    output_device, errmsg)

    def __dealloc__(self):
        """Close midi device if still open when the instance is destroyed."""

        cdef PmError err

        if self.debug:
            print "Closing MIDI output stream and destroying instance."

        if self.midi:
            err = Pm_Close(self.midi)
            if err < 0:
                raise Exception(Pm_GetErrorText(err))

    def _check_open(self):
        """Check whether midi device is open, and if not, raises an error.

        Internal method, should be used only by other methods of this class.

        """
        if self.midi == NULL:
            raise Exception("midi Output not open.")

        if self._aborted:
            raise Exception(
                "midi Output aborted. Need to call Close after Abort.")

    def Close(self):
        """Close the midi output device, flushing any pending buffers.

        PortMidi attempts to close open streams when the application exits --
        this is particularly difficult under Windows, so it is best to take
        care to close all devices explicitly.

        """
        cdef PmError err

        if not self.midi:
            return

        err = Pm_Close(self.midi)
        if err < 0:
            raise Exception(Pm_GetErrorText(err))

        self.midi = NULL

    def Abort(self):
        """Terminate outgoing messages immediately.

        The caller should immediately close the output port after calling this
        method. This call may result in transmission of a partial midi message.
        There is no abort for Midi input because the user can simply ignore
        messages in the buffer and close an input device at any time.

        """
        cdef PmError err

        if not self.midi:
            return

        err = Pm_Abort(self.midi)
        if err < 0:
            raise Exception(Pm_GetErrorText(err))

        self._aborted = 1

    def Write(self, data):
        """Output a series of MIDI events given by data list n this device.

        Usage::

            Write([
                [[status, data1, data2, data3], timestamp],
                [[status, data1, data2, data3], timestamp],
                ...
            ])

        The data1/2/3 items in each event are optional::

           Write([[[0xc0, 0, 0], 20000]])

        is equivalent to::

           Write([[[0xc0], 20000]])

        Example:

        Send program change 1 at time 20000 and send note 65 with velocity 100
        at 500 ms later::

             Write([[[0xc0, 0, 0], 20000], [[0x90, 60, 100], 20500]])

        .. notes::
            1. Timestamps will be ignored if latency == 0.

            2. To get a note to play immediately, send the note on event with
               the result from the Time() function as the timestamp.

        """
        cdef PmEvent buffer[1024]
        cdef PmError err
        cdef int item
        cdef int ev_no

        self._check_open()

        if len(data) > 1024:
            raise IndexError('Maximum event list length is 1024.')
        else:
            for ev_no, event in enumerate(data):
                if not event[0]:
                    raise ValueError('No data in event no. %i.' % ev_no)
                if len(event[0]) > 4:
                    raise ValueError('Too many data bytes (%i) in event no. %i.'
                        % (len(event[0]), ev_no))

                buffer[ev_no].message = 0

                for item in range(len(event[0])):
                    buffer[ev_no].message += (
                        (event[0][item] & 0xFF) << (8 * item))

                buffer[ev_no].timestamp = event[1]

                if self.debug:
                    print "%i : %r : %s" % (
                        ev_no, buffer[ev_no].message, buffer[ev_no].timestamp)

        if self.debug:
            print "Writing to midi buffer."
        err = Pm_Write(self.midi, buffer, len(data))
        if err < 0:
            raise Exception(Pm_GetErrorText(err))

    def WriteShort(self, status, data1=0, data2=0):
        """Output MIDI event of three bytes or less immediately on this device.

        Usage::

            WriteShort(status, data1, data2)

        status must be a valid MIDI status byte, for example:

        0xCx = Program Change
        0xBx = Controller Change
        0x9x = Note On

        where x is the MIDI channel number 0 - 0xF.

        The data1 and data2 arguments are optional and assumed to be 0 if
        omitted.

        Example:

        Send note 65 on with velocity 100::

             WriteShort(0x90, 65, 100)

        """
        cdef PmEvent buffer[1]
        cdef PmError err

        self._check_open()

        buffer[0].timestamp = Pt_Time()
        buffer[0].message = (((data2 << 16) & 0xFF0000) |
            ((data1 << 8) & 0xFF00) | (status & 0xFF))

        if self.debug:
            print "Writing to MIDI buffer."
        err = Pm_Write(self.midi, buffer, 1) # stream, buffer, length
        if err < 0:
            raise Exception(Pm_GetErrorText(err))

    def WriteSysEx(self, when, msg):
        """Output a timestamped system-exclusive MIDI message on this device.

        Usage::

            WriteSysEx(<timestamp>, <msg>)

        <msg> can be a *list* or a *string*

        Example (assuming 'out' is an output MIDI stream):

            out.WriteSysEx(0, '\\xF0\\x7D\\x10\\x11\\x12\\x13\\xF7')

        This is equivalent to::

            out.WriteSysEx(pypm.Time(),
                [0xF0, 0x7D, 0x10, 0x11, 0x12, 0x13, 0xF7])

        """
        cdef PmError err
        cdef char *cmsg
        cdef PtTimestamp cur_time

        self._check_open()

        if type(msg) is list:
             # Markus Pfaff contribution
            msg = array.array('B', msg).tostring()
        cmsg = msg

        cur_time = Pt_Time()
        err = Pm_WriteSysEx(self.midi, when, <unsigned char *> cmsg)
        if err < 0:
            raise Exception(Pm_GetErrorText(err))

        # wait for SysEx to go thru or...
        # my win32 machine crashes w/ multiple SysEx
        while Pt_Time() == cur_time:
            pass


cdef class Input:
    """Represents an input MIDI stream device.

    Takes the form::

        input = pypm.Input(input_device)

    """
    cdef int device
    cdef PmStream *midi
    cdef int debug

    def __init__(self, input_device, buffersize=4096):
        """Instantiate MIDI input stream object."""

        cdef PmError err
        self.device = input_device
        self.debug = 0

        err = Pm_OpenInput(&(self.midi), input_device, NULL, buffersize,
                           &Pt_Time, NULL)
        if err < 0:
            raise Exception(Pm_GetErrorText(err))

        if self.debug:
            print "MIDI input opened."

    def __dealloc__(self):
        """Close midi device if still open when the instance is destroyed."""

        cdef PmError err

        if self.debug:
            print "Closing MIDI input stream and destroying instance"

        if self.midi:
            err = Pm_Close(self.midi)
            if err < 0:
                raise Exception(Pm_GetErrorText(err))

    def _check_open(self):
        """Check whether midi device is open, and if not, raises an error.

        Internal method, should be used only by other methods of this class.

        """
        if self.midi == NULL:
            raise Exception("midi Input not open.")

    def Close(self):
        """Close the midi input device.

        PortMidi attempts to close open streams when the application exits --
        this is particularly difficult under Windows, so it is best to take
        care to close all devices explicitly.

        """
        cdef PmError err

        if not self.midi:
            return

        if self.midi:
            err = Pm_Close(self.midi)
            if err < 0:
                raise Exception(Pm_GetErrorText(err))

        self.midi = NULL


    def SetFilter(self, filters):
        """Set filters on an open input stream.

        Usage::

            input.SetFilter(filters)

        Filters are used to drop selected input event types. By default, only
        active sensing messages are filtered. To prohibit, say, active sensing
        and sysex messages, call

        ::

            input.SetFilter(FILT_ACTIVE | FILT_SYSEX);

        Filtering is useful when midi routing or midi thru functionality is
        being provided by the user application. For example, you may want to
        exclude timing messages (clock, MTC, start/stop/continue), while
        allowing note-related messages to pass. Or you may be using a sequencer
        or drum-machine for MIDI clock information but want to exclude any
        notes it may play.

        .. note::
            SetFilter empties the buffer after setting the filter,
            just in case anything got through.

        """
        cdef PmEvent buffer[1]
        cdef PmError err

        self._check_open()

        err = Pm_SetFilter(self.midi, filters)

        if err < 0:
            raise Exception(Pm_GetErrorText(err))

        while(Pm_Poll(self.midi) != pmNoError):
            err = Pm_Read(self.midi, buffer, 1)
            if err < 0:
                raise Exception(Pm_GetErrorText(err))

    def SetChannelMask(self, mask):
        """Set channel mask to filter incoming messages based on channel.

        The mask is a 16-bit bitfield corresponding to appropriate channels
        Channel(<channel>) can assist in calling this function, i.e. to
        receive only input on channel 1, call this method like this::

            SetChannelMask(Channel(1))

        Multiple channels should be OR'd together::

            SetChannelMask(Channel(10) | Channel(11))

        .. note::
            The PyPortMidi Channel function has been altered from the original
            PortMidi C call to correct for what seems to be a bug --- i.e.
            channel filters were all numbered from 0 to 15 instead of 1 to 16.

        """
        cdef PmError err

        self._check_open()

        err = Pm_SetChannelMask(self.midi, mask)
        if err < 0:
            raise Exception(Pm_GetErrorText(err))

    def Poll(self):
        """Test whether input is available.

        Returns TRUE if input can be read, FALSE otherwise, or an error value.

        """
        cdef PmError err

        self._check_open()

        err = Pm_Poll(self.midi)
        if err < 0:
            raise Exception(Pm_GetErrorText(err))

        return err

    def Read(self, max_events):
        """Read and return up to max_events events from input.

        Reads up to max_events midi events stored in the input buffer and
        returns them as a list in the following form::

            [
                [[status, data1, data2, data3], timestamp],
                [[status, data1, data2, data3], timestamp],
                ...
            ]

        """
        cdef PmEvent buffer[1024]
        cdef PmError num_events

        self._check_open()

        if max_events > 1024:
            raise ValueError('Maximum buffer length is 1024.')
        if not max_events:
            raise ValueError('Minimum buffer length is 1.')

        num_events = Pm_Read(self.midi, buffer, max_events)
        if num_events < 0:
            raise Exception(Pm_GetErrorText(num_events))

        events = []
        if num_events >= 1:
            for ev_no in range(<int>num_events):
                events.append(
                    [
                        [
                            buffer[ev_no].message & 0xFF,
                            (buffer[ev_no].message >> 8) & 0xFF,
                            (buffer[ev_no].message >> 16) & 0xFF,
                            (buffer[ev_no].message >> 24) & 0xFF
                        ],
                        buffer[ev_no].timestamp
                    ]
                )

        return events
