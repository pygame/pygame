import unittest
import os
import sys
import time

import pygame
import pygame.midi
import pygame.compat
from pygame.locals import *


class MidiInputTest(unittest.TestCase):

    def setUp(self):
        pygame.midi.init()
        in_id = pygame.midi.get_default_input_id()
        if in_id != -1:
            self.midi_input = pygame.midi.Input(in_id)
        else:
            self.midi_input = None

    def tearDown(self):
        if self.midi_input:
            self.midi_input.close()
        pygame.midi.quit()

    def test_Input(self):
        """|tags: interactive|
        """

        i = pygame.midi.get_default_input_id()
        if self.midi_input:
            self.assertEqual(self.midi_input.device_id, i)

        # try feeding it an input id.
        i = pygame.midi.get_default_output_id()

        # can handle some invalid input too.
        self.assertRaises(pygame.midi.MidiException, pygame.midi.Input, i)
        self.assertRaises(pygame.midi.MidiException, pygame.midi.Input, 9009)
        self.assertRaises(pygame.midi.MidiException, pygame.midi.Input, -1)
        self.assertRaises(TypeError, pygame.midi.Input, "1234")
        self.assertRaises(OverflowError, pygame.midi.Input, pow(2, 99))

    def test_poll(self):

        # __doc__ (as of 2009-05-19) for pygame.midi.Input.poll:

          # returns true if there's data, or false if not.
          # Input.poll(): return Bool
          #
          # raises a MidiException on error.

        if not self.midi_input:
           self.skipTest('No midi Input device')

        self.assertFalse(self.midi_input.poll())
        # TODO fake some incoming data

        pygame.midi.quit()
        self.assertRaises(RuntimeError, self.midi_input.poll)
        # set midi_input to None to avoid error in tearDown
        self.midi_input = None

    def test_read(self):

        # __doc__ (as of 2009-05-19) for pygame.midi.Input.read:

          # reads num_events midi events from the buffer.
          # Input.read(num_events): return midi_event_list
          #
          # Reads from the Input buffer and gives back midi events.
          # [[[status,data1,data2,data3],timestamp],
          #  [[status,data1,data2,data3],timestamp],...]

        if not self.midi_input:
           self.skipTest('No midi Input device')

        read = self.midi_input.read(5)
        self.assertEqual(read, [])
        # TODO fake some  incoming data

        pygame.midi.quit()
        self.assertRaises(RuntimeError, self.midi_input.read, 52)
        # set midi_input to None to avoid error in tearDown
        self.midi_input = None

    def test_close(self):
        if not self.midi_input:
           self.skipTest('No midi Input device')

        self.assertIsNotNone(self.midi_input._input)
        self.midi_input.close()
        self.assertIsNone(self.midi_input._input)


class MidiOutputTest(unittest.TestCase):

    def setUp(self):
        pygame.midi.init()
        m_out_id = pygame.midi.get_default_output_id()
        if m_out_id != -1:
            self.midi_output = pygame.midi.Output(m_out_id)
        else:
            self.midi_output = None

    def tearDown(self):
        if self.midi_output:
            self.midi_output.close()
        pygame.midi.quit()

    def test_Output(self):
        """|tags: interactive|
        """
        i = pygame.midi.get_default_output_id()
        if self.midi_output:
            self.assertEqual(self.midi_output.device_id, i)

        # try feeding it an input id.
        i = pygame.midi.get_default_input_id()

        # can handle some invalid input too.
        self.assertRaises(pygame.midi.MidiException, pygame.midi.Output, i)
        self.assertRaises(pygame.midi.MidiException, pygame.midi.Output, 9009)
        self.assertRaises(pygame.midi.MidiException, pygame.midi.Output, -1)
        self.assertRaises(TypeError, pygame.midi.Output,"1234")
        self.assertRaises(OverflowError, pygame.midi.Output, pow(2,99))

    def test_note_off(self):
        """|tags: interactive|
        """

        # __doc__ (as of 2009-05-19) for pygame.midi.Output.note_off:

          # turns a midi note off.  Note must be on.
          # Output.note_off(note, velocity=None, channel = 0)
          #
          # Turn a note off in the output stream.  The note must already
          # be on for this to work correctly.

        if self.midi_output:
            out = self.midi_output
            out.note_on(5, 30, 0)
            out.note_off(5, 30, 0)
            with self.assertRaises(ValueError) as cm:
                out.note_off(5, 30, 25)
            self.assertEqual(str(cm.exception), "Channel not between 0 and 15.")
            with self.assertRaises(ValueError) as cm:
                out.note_off(5, 30, -1)
            self.assertEqual(str(cm.exception), "Channel not between 0 and 15.")

    def test_note_on(self):
        """|tags: interactive|
        """
        # __doc__ (as of 2009-05-19) for pygame.midi.Output.note_on:

          # turns a midi note on.  Note must be off.
          # Output.note_on(note, velocity=None, channel = 0)
          #
          # Turn a note on in the output stream.  The note must already
          # be off for this to work correctly.

        if self.midi_output:
            out = self.midi_output
            out.note_on(5, 30, 0)
            out.note_on(5, 42, 10)
            with self.assertRaises(ValueError) as cm:
                out.note_on(5, 30, 25)
            self.assertEqual(str(cm.exception), "Channel not between 0 and 15.")
            with self.assertRaises(ValueError) as cm:
                out.note_on(5, 30, -1)
            self.assertEqual(str(cm.exception), "Channel not between 0 and 15.")

    def test_set_instrument(self):

        # __doc__ (as of 2009-05-19) for pygame.midi.Output.set_instrument:

          # Select an instrument, with a value between 0 and 127.
          # Output.set_instrument(instrument_id, channel = 0)

        if not self.midi_output:
           self.skipTest('No midi device')
        out = self.midi_output
        out.set_instrument(5)
        out.set_instrument(42, channel=2)
        with self.assertRaises(ValueError) as cm:
            out.set_instrument(-6)
        self.assertEqual(str(cm.exception), "Undefined instrument id: -6")
        with self.assertRaises(ValueError) as cm:
            out.set_instrument(156)
        self.assertEqual(str(cm.exception), "Undefined instrument id: 156")
        with self.assertRaises(ValueError) as cm:
            out.set_instrument(5, -1)
        self.assertEqual(str(cm.exception), "Channel not between 0 and 15.")
        with self.assertRaises(ValueError) as cm:
            out.set_instrument(5, 16)
        self.assertEqual(str(cm.exception), "Channel not between 0 and 15.")

    def test_write(self):
        if not self.midi_output:
           self.skipTest('No midi device')

        out = self.midi_output
        out.write([[[0xc0, 0, 0], 20000]])
        # is equivalent to
        out.write([[[0xc0], 20000]])
        # example from the docstring :
        # 1. choose program change 1 at time 20000 and
        # 2. send note 65 with velocity 100 500 ms later
        out.write([
            [[0xc0, 0, 0], 20000],
            [[0x90, 60, 100], 20500]
        ])

        out.write([])
        verrry_long = [[[0x90, 60, i % 100], 20000 + 100 * i] for i in range(1024)]
        out.write(verrry_long)

        too_long = [[[0x90, 60, i % 100], 20000 + 100 * i] for i in range(1025)]
        self.assertRaises(IndexError, out.write, too_long)
        # test wrong data
        with self.assertRaises(TypeError) as cm:
            out.write('Non sens ?')
        error_msg = "unsupported operand type(s) for &: 'str' and 'int'"
        self.assertEqual(str(cm.exception), error_msg)

        with self.assertRaises(TypeError) as cm:
            out.write(["Hey what's that?"])
        self.assertEqual(str(cm.exception), error_msg)

    def test_write_short(self):
        """|tags: interactive|
        """
        # __doc__ (as of 2009-05-19) for pygame.midi.Output.write_short:

          # write_short(status <, data1><, data2>)
          # Output.write_short(status)
          # Output.write_short(status, data1 = 0, data2 = 0)
          #
          # output MIDI information of 3 bytes or less.
          # data fields are optional
          # status byte could be:
          #      0xc0 = program change
          #      0x90 = note on
          #      etc.
          #      data bytes are optional and assumed 0 if omitted
          # example: note 65 on with velocity 100
          #      write_short(0x90,65,100)

        if not self.midi_output:
           self.skipTest('No midi device')

        out = self.midi_output
        # program change
        out.write_short(0xc0)
        # put a note on, then off.
        out.write_short(0x90, 65, 100)
        out.write_short(0x80, 65, 100)
        out.write_short(0x90)

    def test_write_sys_ex(self):
        if not self.midi_output:
           self.skipTest('No midi device')

        out = self.midi_output
        out.write_sys_ex(pygame.midi.time(),
                         [0xF0, 0x7D, 0x10, 0x11, 0x12, 0x13, 0xF7])

    def test_pitch_bend(self):
        # FIXME : pitch_bend in the code, but not in documentation
        if not self.midi_output:
           self.skipTest('No midi device')

        out = self.midi_output
        with self.assertRaises(ValueError) as cm:
            out.pitch_bend(5, channel=-1)
        self.assertEqual(str(cm.exception), "Channel not between 0 and 15.")
        with self.assertRaises(ValueError) as cm:
            out.pitch_bend(5, channel=16)
        with self.assertRaises(ValueError) as cm:
            out.pitch_bend(-10001, 1)
        self.assertEqual(str(cm.exception), "Pitch bend value must be between "
                                            "-8192 and +8191, not -10001.")
        with self.assertRaises(ValueError) as cm:
            out.pitch_bend(10665, 2)

    def test_close(self):
        if not self.midi_output:
           self.skipTest('No midi device')
        self.assertIsNotNone(self.midi_output._output)
        self.midi_output.close()
        self.assertIsNone(self.midi_output._output)

    def test_abort(self):
        if not self.midi_output:
           self.skipTest('No midi device')
        self.assertEqual(self.midi_output._aborted, 0)
        self.midi_output.abort()
        self.assertEqual(self.midi_output._aborted, 1)


class MidiModuleTest(unittest.TestCase):

    def setUp(self):
        pygame.midi.init()

    def tearDown(self):
        pygame.midi.quit()

    def test_MidiException(self):

        def raiseit():
            raise pygame.midi.MidiException('Hello Midi param')

        with self.assertRaises(pygame.midi.MidiException) as cm:
            raiseit()
        self.assertEqual(cm.exception.parameter, 'Hello Midi param')

    def test_get_count(self):
        c = pygame.midi.get_count()
        self.assertIsInstance(c, int)
        self.assertTrue(c >= 0)

    def test_get_default_input_id(self):

        # __doc__ (as of 2009-05-19) for pygame.midi.get_default_input_device_id:

          # gets the device number of the default input device.
          # pygame.midi.get_default_input_device_id(): return default_id
          #
          #
          # Return the default device ID or -1 if there are no devices.
          # The result can be passed to the Input()/Ouput() class.
          #
          # On the PC, the user can specify a default device by
          # setting an environment variable. For example, to use device #1.
          #
          #     set PM_RECOMMENDED_INPUT_DEVICE=1
          #
          # The user should first determine the available device ID by using
          # the supplied application "testin" or "testout".
          #
          # In general, the registry is a better place for this kind of info,
          # and with USB devices that can come and go, using integers is not
          # very reliable for device identification. Under Windows, if
          # PM_RECOMMENDED_OUTPUT_DEVICE (or PM_RECOMMENDED_INPUT_DEVICE) is
          # *NOT* found in the environment, then the default device is obtained
          # by looking for a string in the registry under:
          #     HKEY_LOCAL_MACHINE/SOFTWARE/PortMidi/Recommended_Input_Device
          # and HKEY_LOCAL_MACHINE/SOFTWARE/PortMidi/Recommended_Output_Device
          # for a string. The number of the first device with a substring that
          # matches the string exactly is returned. For example, if the string
          # in the registry is "USB", and device 1 is named
          # "In USB MidiSport 1x1", then that will be the default
          # input because it contains the string "USB".
          #
          # In addition to the name, get_device_info() returns "interf", which
          # is the interface name. (The "interface" is the underlying software
          #     system or API used by PortMidi to access devices. Examples are
          #     MMSystem, DirectX (not implemented), ALSA, OSS (not implemented), etc.)
          #     At present, the only Win32 interface is "MMSystem", the only Linux
          #     interface is "ALSA", and the only Max OS X interface is "CoreMIDI".
          # To specify both the interface and the device name in the registry,
          # separate the two with a comma and a space, e.g.:
          #     MMSystem, In USB MidiSport 1x1
          # In this case, the string before the comma must be a substring of
          # the "interf" string, and the string after the space must be a
          # substring of the "name" name string in order to match the device.
          #
          # Note: in the current release, the default is simply the first device
          #     (the input or output device with the lowest PmDeviceID).

        midin_id = pygame.midi.get_default_input_id()
        # if there is a not None return make sure it is an int.
        self.assertIsInstance(midin_id, int)
        self.assertTrue(midin_id >= -1)
        pygame.midi.quit()
        self.assertRaises(RuntimeError, pygame.midi.get_default_output_id)

    def test_get_default_output_id(self):

        # __doc__ (as of 2009-05-19) for pygame.midi.get_default_output_device_id:

          # get the device number of the default output device.
          # pygame.midi.get_default_output_device_id(): return default_id
          #
          #
          # Return the default device ID or -1 if there are no devices.
          # The result can be passed to the Input()/Ouput() class.
          #
          # On the PC, the user can specify a default device by
          # setting an environment variable. For example, to use device #1.
          #
          #     set PM_RECOMMENDED_OUTPUT_DEVICE=1
          #
          # The user should first determine the available device ID by using
          # the supplied application "testin" or "testout".
          #
          # In general, the registry is a better place for this kind of info,
          # and with USB devices that can come and go, using integers is not
          # very reliable for device identification. Under Windows, if
          # PM_RECOMMENDED_OUTPUT_DEVICE (or PM_RECOMMENDED_INPUT_DEVICE) is
          # *NOT* found in the environment, then the default device is obtained
          # by looking for a string in the registry under:
          #     HKEY_LOCAL_MACHINE/SOFTWARE/PortMidi/Recommended_Input_Device
          # and HKEY_LOCAL_MACHINE/SOFTWARE/PortMidi/Recommended_Output_Device
          # for a string. The number of the first device with a substring that
          # matches the string exactly is returned. For example, if the string
          # in the registry is "USB", and device 1 is named
          # "In USB MidiSport 1x1", then that will be the default
          # input because it contains the string "USB".
          #
          # In addition to the name, get_device_info() returns "interf", which
          # is the interface name. (The "interface" is the underlying software
          #     system or API used by PortMidi to access devices. Examples are
          #     MMSystem, DirectX (not implemented), ALSA, OSS (not implemented), etc.)
          #     At present, the only Win32 interface is "MMSystem", the only Linux
          #     interface is "ALSA", and the only Max OS X interface is "CoreMIDI".
          # To specify both the interface and the device name in the registry,
          # separate the two with a comma and a space, e.g.:
          #     MMSystem, In USB MidiSport 1x1
          # In this case, the string before the comma must be a substring of
          # the "interf" string, and the string after the space must be a
          # substring of the "name" name string in order to match the device.
          #
          # Note: in the current release, the default is simply the first device
          #     (the input or output device with the lowest PmDeviceID).

        c = pygame.midi.get_default_output_id()
        self.assertIsInstance(c, int)
        self.assertTrue(c >= -1)
        pygame.midi.quit()
        self.assertRaises(RuntimeError, pygame.midi.get_default_output_id)

    def test_get_device_info(self):

        # __doc__ (as of 2009-05-19) for pygame.midi.get_device_info:

          # returns (interf, name, input, output, opened)
          # pygame.midi.get_device_info(an_id): return (interf, name, input,
          # output, opened)
          #
          #
          # If the id is out of range, the function returns None.

        an_id = pygame.midi.get_default_output_id()
        if an_id != -1:
            interf, name, input, output, opened = pygame.midi.get_device_info(an_id)
            self.assertEqual(output, 1)
            self.assertEqual(input, 0)
            self.assertEqual(opened, 0)

        an_in_id = pygame.midi.get_default_input_id()
        if an_in_id != -1:
            r = pygame.midi.get_device_info(an_in_id)
            # if r is None, it means that the id is out of range.
            interf, name, input, output, opened = r

            self.assertEqual(output, 0)
            self.assertEqual(input, 1)
            self.assertEqual(opened, 0)
        out_of_range = pygame.midi.get_count()
        for num in range(out_of_range):
            self.assertIsNotNone(pygame.midi.get_device_info(num))
        info = pygame.midi.get_device_info(out_of_range)
        self.assertIsNone(info)

    def test_init(self):

        # __doc__ (as of 2009-05-19) for pygame.midi.init:

          # initialize the midi module
          # pygame.midi.init(): return None
          #
          # Call the initialisation function before using the midi module.
          #
          # It is safe to call this more than once.
        pygame.midi.quit()
        self.assertRaises(RuntimeError, pygame.midi.get_count)
        # initialising many times should be fine.
        pygame.midi.init()
        pygame.midi.init()
        pygame.midi.init()
        pygame.midi.init()

    def test_midis2events(self):

        # __doc__ (as of 2009-05-19) for pygame.midi.midis2events:

          # converts midi events to pygame events
          # pygame.midi.midis2events(midis, device_id): return [Event, ...]
          #
          # Takes a sequence of midi events and returns list of pygame events.

        midi_data = ([[0xc0, 0, 1, 2], 20000],
                     [[0x90, 60, 100, 'blablabla'], 20000]
                    )
        events = pygame.midi.midis2events(midi_data, 2)
        self.assertEqual(len(events), 2)

        for eve in events:
            print(eve, type(eve))
            # pygame.event.Event is a function, but ...
            self.assertEqual(eve.__class__.__name__, 'Event')
            self.assertEqual(eve.vice_id, 2)
            # FIXME I don't know what we want for the Event.timestamp
            # For now it accepts  it accepts int as is:
            self.assertIsInstance(eve.timestamp, int)
            self.assertEqual(eve.timestamp, 20000)
        self.assertEqual(events[1].data3, 'blablabla')

    def test_quit(self):

        # __doc__ (as of 2009-05-19) for pygame.midi.quit:

          # uninitialize the midi module
          # pygame.midi.quit(): return None
          #
          #
          # Called automatically atexit if you don't call it.
          #
          # It is safe to call this function more than once.


         # It is safe to call this more than once.
        pygame.midi.quit()
        pygame.midi.init()
        pygame.midi.quit()
        pygame.midi.quit()
        pygame.midi.init()
        pygame.midi.init()
        pygame.midi.quit()

    def test_time(self):

        # __doc__ (as of 2009-05-19) for pygame.midi.time:

          # returns the current time in ms of the PortMidi timer
          # pygame.midi.time(): return time

        mtime = pygame.midi.time()
        self.assertIsInstance(mtime, int)
        # should be close to 2-3... since the timer is just init'd.
        self.assertTrue(0 <= mtime < 100)


if __name__ == '__main__':
    unittest.main()
