.. include:: common.txt

:mod:`pygame.midi`
==================

.. module:: pygame.midi
   :synopsis: pygame module for interacting with midi input and output.

| :sl:`pygame module for interacting with midi input and output.`

The midi module can send output to midi devices, and get input from midi
devices. It can also list midi devices on the system.

Including real midi devices, and virtual ones.

It uses the portmidi library. Is portable to which ever platforms portmidi
supports (currently Windows, ``OSX``, and Linux).

This uses pyportmidi for now, but may use its own bindings at some point in the
future. The pyportmidi bindings are included with pygame.

New in pygame 1.9.0.

.. class:: Input

   | :sl:`Input is used to get midi input from midi devices.`
   | :sg:`Input(device_id) -> None`
   | :sg:`Input(device_id, buffer_size) -> None`

   buffer_size -the number of input events to be buffered waiting to

   ::

     be read using Input.read()

   .. method:: close

      | :sl:`closes a midi stream, flushing any pending buffers.`
      | :sg:`close() -> None`

      PortMidi attempts to close open streams when the application exits --
      this is particularly difficult under Windows.

      .. ## Input.close ##

   .. method:: poll

      | :sl:`returns true if there's data, or false if not.`
      | :sg:`poll() -> Bool`

      raises a MidiException on error.

      .. ## Input.poll ##

   .. method:: read

      | :sl:`reads num_events midi events from the buffer.`
      | :sg:`read(num_events) -> midi_event_list`

      Reads from the Input buffer and gives back midi events.
      [[[status,data1,data2,data3],timestamp],

      ::

       [[status,data1,data2,data3],timestamp],...]

      .. ## Input.read ##

   .. ## pygame.midi.Input ##

.. function:: MidiException

   | :sl:`exception that pygame.midi functions and classes can raise`
   | :sg:`MidiException(errno) -> None`

   .. ## pygame.midi.MidiException ##

.. class:: Output

   | :sl:`Output is used to send midi to an output device`
   | :sg:`Output(device_id) -> None`
   | :sg:`Output(device_id, latency = 0) -> None`
   | :sg:`Output(device_id, buffer_size = 4096) -> None`
   | :sg:`Output(device_id, latency, buffer_size) -> None`

   The buffer_size specifies the number of output events to be buffered waiting
   for output. (In some cases -- see below -- PortMidi does not buffer output
   at all and merely passes data to a lower-level ``API``, in which case
   buffersize is ignored.)

   latency is the delay in milliseconds applied to timestamps to determine when
   the output should actually occur. (If latency is <<0, 0 is assumed.)

   If latency is zero, timestamps are ignored and all output is delivered
   immediately. If latency is greater than zero, output is delayed until the
   message timestamp plus the latency. (NOTE: time is measured relative to the
   time source indicated by time_proc. Timestamps are absolute, not relative
   delays or offsets.) In some cases, PortMidi can obtain better timing than
   your application by passing timestamps along to the device driver or
   hardware. Latency may also help you to synchronize midi data to audio data
   by matching midi latency to the audio buffer latency.

   .. method:: abort

      | :sl:`terminates outgoing messages immediately`
      | :sg:`abort() -> None`

      The caller should immediately close the output port; this call may result
      in transmission of a partial midi message. There is no abort for Midi
      input because the user can simply ignore messages in the buffer and close
      an input device at any time.

      .. ## Output.abort ##

   .. method:: close

      | :sl:`closes a midi stream, flushing any pending buffers.`
      | :sg:`close() -> None`

      PortMidi attempts to close open streams when the application exits --
      this is particularly difficult under Windows.

      .. ## Output.close ##

   .. method:: note_off

      | :sl:`turns a midi note off.  Note must be on.`
      | :sg:`note_off(note, velocity=None, channel = 0) -> None`

      Turn a note off in the output stream. The note must already be on for
      this to work correctly.

      .. ## Output.note_off ##

   .. method:: note_on

      | :sl:`turns a midi note on.  Note must be off.`
      | :sg:`note_on(note, velocity=None, channel = 0) -> None`

      Turn a note on in the output stream. The note must already be off for
      this to work correctly.

      .. ## Output.note_on ##

   .. method:: set_instrument

      | :sl:`select an instrument, with a value between 0 and 127`
      | :sg:`set_instrument(instrument_id, channel = 0) -> None`

      .. ## Output.set_instrument ##

   .. method:: pitch_bend

      | :sl:`modify the pitch of a channel.`
      | :sg:`set_instrument(value = 0, channel = 0) -> None`

      Adjust the pitch of a channel.  The value is a signed integer
      from -8192 to +8191.  For example, 0 means "no change", +4096 is
      typically a semitone higher, and -8192 is 1 whole tone lower (though
      the musical range corresponding to the pitch bend range can also be
      changed in some synthesizers).

      If no value is given, the pitch bend is returned to "no change".
      New in pygame 1.9.4.

   .. method:: write

      | :sl:`writes a list of midi data to the Output`
      | :sg:`write(data) -> None`

      writes series of ``MIDI`` information in the form of a list:

      ::

           write([[[status <,data1><,data2><,data3>],timestamp],
                  [[status <,data1><,data2><,data3>],timestamp],...])

      <data> fields are optional example: choose program change 1 at time 20000
      and send note 65 with velocity 100 500 ms later.

      ::

           write([[[0xc0,0,0],20000],[[0x90,60,100],20500]])

      notes:

      ::

        1. timestamps will be ignored if latency = 0.
        2. To get a note to play immediately, send MIDI info with
           timestamp read from function Time.
        3. understanding optional data fields:
             write([[[0xc0,0,0],20000]]) is equivalent to
             write([[[0xc0],20000]])

      Can send up to 1024 elements in your data list, otherwise an

      ::

       IndexError exception is raised.

      .. ## Output.write ##

   .. method:: write_short

      | :sl:`write_short(status <, data1><, data2>)`
      | :sg:`write_short(status) -> None`
      | :sg:`write_short(status, data1 = 0, data2 = 0) -> None`

      output ``MIDI`` information of 3 bytes or less. data fields are optional
      status byte could be:

      ::

           0xc0 = program change
           0x90 = note on
           etc.
           data bytes are optional and assumed 0 if omitted

      example: note 65 on with velocity 100

      ::

           write_short(0x90,65,100)

      .. ## Output.write_short ##

   .. method:: write_sys_ex

      | :sl:`writes a timestamped system-exclusive midi message.`
      | :sg:`write_sys_ex(when, msg) -> None`

      msg - can be a \*list* or a \*string* when - a timestamp in milliseconds
      example:

      ::

        (assuming o is an output MIDI stream)
          o.write_sys_ex(0,'\xF0\x7D\x10\x11\x12\x13\xF7')
        is equivalent to
          o.write_sys_ex(pygame.midi.time(),
                         [0xF0,0x7D,0x10,0x11,0x12,0x13,0xF7])

      .. ## Output.write_sys_ex ##

   .. ## pygame.midi.Output ##

.. function:: get_count

   | :sl:`gets the number of devices.`
   | :sg:`get_count() -> num_devices`

   Device ids range from 0 to ``get_count()`` -1

   .. ## pygame.midi.get_count ##

.. function:: get_default_input_id

   | :sl:`gets default input device number`
   | :sg:`get_default_input_id() -> default_id`

   Return the default device ``ID`` or -1 if there are no devices. The result
   can be passed to the Input()/Output() class.

   On the ``PC``, the user can specify a default device by setting an
   environment variable. For example, to use device #1.

   ::

       set PM_RECOMMENDED_INPUT_DEVICE=1

   The user should first determine the available device ``ID`` by using the
   supplied application "testin" or "testout".

   In general, the registry is a better place for this kind of info, and with
   ``USB`` devices that can come and go, using integers is not very reliable
   for device identification. Under Windows, if
   ``PM_RECOMMENDED_OUTPUT_DEVICE`` (or ``PM_RECOMMENDED_INPUT_DEVICE``) is
   \*NOT* found in the environment, then the default device is obtained by
   looking for a string in the registry under:

   ::

       HKEY_LOCAL_MACHINE/SOFTWARE/PortMidi/Recommended_Input_Device

   and HKEY_LOCAL_MACHINE/SOFTWARE/PortMidi/Recommended_Output_Device for a
   string. The number of the first device with a substring that matches the
   string exactly is returned. For example, if the string in the registry is
   "USB", and device 1 is named "In ``USB`` MidiSport 1x1", then that will be
   the default input because it contains the string "USB".

   In addition to the name, ``get_device_info()`` returns "interf", which is
   the interface name. (The "interface" is the underlying software system or
   ``API`` used by PortMidi to access devices. Examples are MMSystem, DirectX
   (not implemented), ``ALSA``, ``OSS`` (not implemented), etc.) At present,
   the only Win32 interface is "MMSystem", the only Linux interface is "ALSA",
   and the only Max ``OS`` ``X`` interface is "CoreMIDI". To specify both the
   interface and the device name in the registry, separate the two with a comma
   and a space, ``e.g.``:

   ::

       MMSystem, In USB MidiSport 1x1

   In this case, the string before the comma must be a substring of the
   "interf" string, and the string after the space must be a substring of the
   "name" name string in order to match the device.

   Note: in the current release, the default is simply the first device (the
   input or output device with the lowest PmDeviceID).

   .. ## pygame.midi.get_default_input_id ##

.. function:: get_default_output_id

   | :sl:`gets default output device number`
   | :sg:`get_default_output_id() -> default_id`

   Return the default device ``ID`` or -1 if there are no devices. The result
   can be passed to the Input()/Output() class.

   On the ``PC``, the user can specify a default device by setting an
   environment variable. For example, to use device #1.

   ::

       set PM_RECOMMENDED_OUTPUT_DEVICE=1

   The user should first determine the available device ``ID`` by using the
   supplied application "testin" or "testout".

   In general, the registry is a better place for this kind of info, and with
   ``USB`` devices that can come and go, using integers is not very reliable
   for device identification. Under Windows, if
   ``PM_RECOMMENDED_OUTPUT_DEVICE`` (or ``PM_RECOMMENDED_INPUT_DEVICE``) is
   \*NOT* found in the environment, then the default device is obtained by
   looking for a string in the registry under:

   ::

       HKEY_LOCAL_MACHINE/SOFTWARE/PortMidi/Recommended_Input_Device

   and HKEY_LOCAL_MACHINE/SOFTWARE/PortMidi/Recommended_Output_Device for a
   string. The number of the first device with a substring that matches the
   string exactly is returned. For example, if the string in the registry is
   "USB", and device 1 is named "In ``USB`` MidiSport 1x1", then that will be
   the default input because it contains the string "USB".

   In addition to the name, ``get_device_info()`` returns "interf", which is
   the interface name. (The "interface" is the underlying software system or
   ``API`` used by PortMidi to access devices. Examples are MMSystem, DirectX
   (not implemented), ``ALSA``, ``OSS`` (not implemented), etc.) At present,
   the only Win32 interface is "MMSystem", the only Linux interface is "ALSA",
   and the only Max ``OS`` ``X`` interface is "CoreMIDI". To specify both the
   interface and the device name in the registry, separate the two with a comma
   and a space, ``e.g.``:

   ::

       MMSystem, In USB MidiSport 1x1

   In this case, the string before the comma must be a substring of the
   "interf" string, and the string after the space must be a substring of the
   "name" name string in order to match the device.

   Note: in the current release, the default is simply the first device (the
   input or output device with the lowest PmDeviceID).

   .. ## pygame.midi.get_default_output_id ##

.. function:: get_device_info

   | :sl:`returns information about a midi device`
   | :sg:`get_device_info(an_id) -> (interf, name, input, output, opened)`

   interf - a text string describing the device interface, eg 'ALSA'. name - a
   text string for the name of the device, eg 'Midi Through Port-0' input - 0,
   or 1 if the device is an input device. output - 0, or 1 if the device is an
   output device. opened - 0, or 1 if the device is opened.

   If the id is out of range, the function returns None.

   .. ## pygame.midi.get_device_info ##

.. function:: init

   | :sl:`initialize the midi module`
   | :sg:`init() -> None`

   Call the initialisation function before using the midi module.

   It is safe to call this more than once.

   .. ## pygame.midi.init ##

.. function:: midis2events

   | :sl:`converts midi events to pygame events`
   | :sg:`midis2events(midis, device_id) -> [Event, ...]`

   Takes a sequence of midi events and returns list of pygame events.

   .. ## pygame.midi.midis2events ##

.. function:: quit

   | :sl:`uninitialize the midi module`
   | :sg:`quit() -> None`

   Called automatically atexit if you don't call it.

   It is safe to call this function more than once.

   .. ## pygame.midi.quit ##

.. function:: time

   | :sl:`returns the current time in ms of the PortMidi timer`
   | :sg:`time() -> time`

   The time is reset to 0, when the module is inited.

   .. ## pygame.midi.time ##

.. ## pygame.midi ##
