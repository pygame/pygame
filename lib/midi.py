"""
uses portmidi for putting midi into pygame.

This uses pyportmidi for now, but will probably use own bindings at some
point.

"""


import pygame.pypm
pypm = pygame.pypm


import pygame
import pygame.locals
#
MIDIIN = pygame.locals.USEREVENT + 10
MIDIOUT = pygame.locals.USEREVENT + 11





def init():
    pypm.Initialize()
    return 1

def quit():
    pypm.Terminate()

def get_count():
    """ gets the count of devices.
    """
    return pypm.CountDevices()




class MidiException(Exception):
    def __init__(self, value):
        self.parameter = value
    def __str__(self):
        return repr(self.parameter)


class Input(object):
    def __init__(self, device_id):
        """
        """
        self._input = pypm.Input(device_id)
        self.device_id = device_id


    def read(self, length):
        """ [[status,data1,data2,data3],timestamp]
        """
        return self._input.Read(length)

    def poll(self):
        """ returns true if there's data, or false if not.
        """
        r = self._input.Poll()
        if r == pypm.TRUE:
            return True
        elif r == pypm.FALSE:
            return False
        else:
            err_text = GetErrorText(r)
            raise MidiException( (err_text, r) )






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
                         device_id = device_id)
        evs.append( e )


    return evs



if __name__ == "__main__":
    import pygame
    from pygame.locals import *
    pygame.init()
    pygame.fastevent.init()
    event_get = pygame.fastevent.get
    event_post = pygame.fastevent.post

    init()

    i = Input(1)



    going = True
    while going:
        events = event_get()
        for e in events:
            if e.type in [QUIT]:
                going = False
            if e.type in [KEYDOWN]:
                pass
            if e.type in [MIDIIN, MIDIOUT]:
                print e

        if i.poll():
            midi_events = i.read(10)
            # convert them into pygame events.
            midi_evs = midis2events(midi_events, i.device_id)

            for m_e in midi_evs:
                event_post( m_e )


    quit()




