import unittest
import os
import sys
import time

import pygame
import pygame.midi
import pygame.compat
from pygame.locals import *


class MidiTest(unittest.TestCase):

    def setUp(self):
        pygame.midi.init()

    def tearDown(self):
        pygame.midi.quit()

    def todo_test_poll(self):

        # __doc__ (as of 2009-05-19) for pygame.midi.Input.poll:

          # returns true if there's data, or false if not.
          # Input.poll(): return Bool
          #
          # raises a MidiException on error.

        self.fail()

    def todo_test_read(self):

        # __doc__ (as of 2009-05-19) for pygame.midi.Input.read:

          # reads num_events midi events from the buffer.
          # Input.read(num_events): return midi_event_list
          #
          # Reads from the Input buffer and gives back midi events.
          # [[[status,data1,data2,data3],timestamp],
          #  [[status,data1,data2,data3],timestamp],...]

        self.fail()

    def test_MidiException(self):

        def raiseit():
            raise pygame.midi.MidiException('Hello Midi param')

        with self.assertRaises(pygame.midi.MidiException) as cm:
            raiseit()
        self.assertEqual(cm.exception.parameter, 'Hello Midi param')

    def test_note_off(self):
        """|tags: interactive|
        """

        # __doc__ (as of 2009-05-19) for pygame.midi.Output.note_off:

          # turns a midi note off.  Note must be on.
          # Output.note_off(note, velocity=None, channel = 0)
          #
          # Turn a note off in the output stream.  The note must already
          # be on for this to work correctly.


        m_id = pygame.midi.get_default_output_id()
        if m_id != -1:
            out = pygame.midi.Output(m_id)
            out.note_on(5, 30, 0)
            out.note_off(5, 30, 0)

    def test_note_on(self):
        """|tags: interactive|
        """
        # __doc__ (as of 2009-05-19) for pygame.midi.Output.note_on:

          # turns a midi note on.  Note must be off.
          # Output.note_on(note, velocity=None, channel = 0)
          #
          # Turn a note on in the output stream.  The note must already
          # be off for this to work correctly.

        m_id = pygame.midi.get_default_output_id()
        if m_id != -1:
            out = pygame.midi.Output(m_id)
            out.note_on(5, 30, 0)

    def todo_test_set_instrument(self):

        # __doc__ (as of 2009-05-19) for pygame.midi.Output.set_instrument:

          # Select an instrument, with a value between 0 and 127.
          # Output.set_instrument(instrument_id, channel = 0)

        self.fail()

    def test_write(self):

        m_id = pygame.midi.get_default_output_id()
        if m_id == -1:
            self.skipTest('No midi device')
        out = pygame.midi.Output(m_id)
        data = []
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
        verrry_long = [[[0x90, 60, i // 100], 20000 + 100 * i] for i in range(1024)]
        out.write(verrry_long)

        too_long = [[[0x90, 60, i // 100], 20000 + 100 * i] for i in range(1025)]
        self.assertRaises(IndexError, out.write, too_long)
        # test wrong data
        with self.assertRaises(TypeError) as cm:
            out.write('Non sens ?')
        error_msg = "unsupported operand type(s) for &: 'str' and 'int'"
        self.assertEqual(cm.exception.message, error_msg)

        with self.assertRaises(TypeError) as cm:
            out.write(["Hey what's that?"])
        self.assertEqual(cm.exception.message, error_msg)

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

        i = pygame.midi.get_default_output_id()
        if i != -1:
            out = pygame.midi.Output(i)
            # program change
            out.write_short(0xc0)
            # put a note on, then off.
            out.write_short(0x90, 65, 100)
            out.write_short(0x80, 65, 100)
            out.write_short(0x90)

    def test_Input(self):
        """|tags: interactive|
        """

        i = pygame.midi.get_default_input_id()
        if i != -1:
            o = pygame.midi.Input(i)
            del o

        # try feeding it an input id.
        i = pygame.midi.get_default_output_id()

        # can handle some invalid input too.
        self.assertRaises(pygame.midi.MidiException, pygame.midi.Input, i)
        self.assertRaises(pygame.midi.MidiException, pygame.midi.Input, 9009)
        self.assertRaises(pygame.midi.MidiException, pygame.midi.Input, -1)
        self.assertRaises(TypeError, pygame.midi.Input,"1234")
        self.assertRaises(OverflowError, pygame.midi.Input, pow(2,99))


    def test_Output(self):
        """|tags: interactive|
        """
        i = pygame.midi.get_default_output_id()
        if i != -1:
            o = pygame.midi.Output(i)
            del o

        # try feeding it an input id.
        i = pygame.midi.get_default_input_id()

        # can handle some invalid input too.
        self.assertRaises(pygame.midi.MidiException, pygame.midi.Output, i)
        self.assertRaises(pygame.midi.MidiException, pygame.midi.Output, 9009)
        self.assertRaises(pygame.midi.MidiException, pygame.midi.Output, -1)
        self.assertRaises(TypeError, pygame.midi.Output,"1234")
        self.assertRaises(OverflowError, pygame.midi.Output, pow(2,99))


    def todo_test_write_sys_ex(self):

        # __doc__ (as of 2009-05-19) for pygame.midi.Output.write_sys_ex:

          # writes a timestamped system-exclusive midi message.
          # Output.write_sys_ex(when, msg)
          #
          # write_sys_ex(<timestamp>,<msg>)
          #
          # msg - can be a *list* or a *string*
          # example:
          #   (assuming o is an onput MIDI stream)
          #     o.write_sys_ex(0,'\xF0\x7D\x10\x11\x12\x13\xF7')
          #   is equivalent to
          #     o.write_sys_ex(pygame.midi.Time,
          #                    [0xF0,0x7D,0x10,0x11,0x12,0x13,0xF7])

        self.fail()


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

        c = pygame.midi.get_default_input_id()
        # if there is a not None return make sure it is an int.
        self.assertIsInstance(c, int)
        self.assertTrue(c >= -1)
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

    def todo_test_midis2events(self):

        # __doc__ (as of 2009-05-19) for pygame.midi.midis2events:

          # converts midi events to pygame events
          # pygame.midi.midis2events(midis, device_id): return [Event, ...]
          #
          # Takes a sequence of midi events and returns list of pygame events.

        self.fail()

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
