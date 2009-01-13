"""
uses portmidi for putting midi into pygame.

This uses pyportmidi for now, but will probably use own bindings at some
point.

"""


import pygame.pypm
pypm = pygame.pypm


import pygame
import pygame.locals

import atexit


#
MIDIIN = pygame.locals.USEREVENT + 10
MIDIOUT = pygame.locals.USEREVENT + 11

_init = False



def init():
    global _init
    if not _init:
        pypm.Initialize()
        _init = True
        atexit.register(quit)

def quit():
    global _init
    if _init:
        # TODO: find all Input and Output classes and close them first?
        pypm.Terminate()
        _init = False




def get_count():
    """ gets the count of devices.
    """
    return pypm.CountDevices()




def get_default_input_device_id():
    """gets the device number of the default input device.
    """
    return pypm.GetDefaultInputDeviceID()




def get_default_output_device_id():
    """get the device number of the default output device.
    """
    return pypm.GetDefaultOutputDeviceID()


def get_device_info(an_id):
    """ returns (interf, name, input, output, opened)
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
    def __init__(self, device_id, latency = 0):
        """
        """
        self._output = pypm.Output(device_id, latency)
        self.device_id = device_id

    def write(self, data):
        """
        """
        self._output.Write(data)

    def write_short(self, status, data1 = 0, data2 = 0):
        """
        """
        self._output.WriteShort(status, data1, data2)

    def write_sys_ex(self, when, msg):
        """
        """
        self._output.WriteSysEx(when, msg)



    def note_on(self, note, velocity=None):
        """
        """
        if velocity is None:
            velocity = 0
        self.write_short(0x90, note, velocity)

    def note_off(self, note, velocity=None):
        """
        """
        if velocity is None:
            velocity = 0
        self.write_short(0x80, note, velocity)

    def set_instrument(self, instrument_id):
        """
        """
        if not (0 <= instrument_id <= 127):
            raise ValueError("Undefined instrument id: %d" % instrument_id)
        self.write_short(0xc0, instrument_id)






def midis2events(midis, device_id):
    """ takes a sequence of midi events and returns pygame events.
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






