/* Auto generated file: with makeref.py .  Docs go in src/ *.doc . */
#define DOC_PYGAMEMIDI "pygame module for interacting with midi input and output."

#define DOC_PYGAMEMIDIINPUT "Input(device_id) -> None\nInput(device_id, buffer_size) -> None\nInput is used to get midi input from midi devices."

#define DOC_INPUTCLOSE "close() -> None\ncloses a midi stream, flushing any pending buffers."

#define DOC_INPUTPOLL "poll() -> Bool\nreturns true if there's data, or false if not."

#define DOC_INPUTREAD "read(num_events) -> midi_event_list\nreads num_events midi events from the buffer."

#define DOC_PYGAMEMIDIMIDIEXCEPTION "MidiException(errno) -> None\nexception that pygame.midi functions and classes can raise"

#define DOC_PYGAMEMIDIOUTPUT "Output(device_id) -> None\nOutput(device_id, latency = 0) -> None\nOutput(device_id, buffer_size = 4096) -> None\nOutput(device_id, latency, buffer_size) -> None\nOutput is used to send midi to an output device"

#define DOC_OUTPUTABORT "abort() -> None\nterminates outgoing messages immediately"

#define DOC_OUTPUTCLOSE "close() -> None\ncloses a midi stream, flushing any pending buffers."

#define DOC_OUTPUTNOTEOFF "note_off(note, velocity=None, channel = 0) -> None\nturns a midi note off.  Note must be on."

#define DOC_OUTPUTNOTEON "note_on(note, velocity=None, channel = 0) -> None\nturns a midi note on.  Note must be off."

#define DOC_OUTPUTSETINSTRUMENT "set_instrument(instrument_id, channel = 0) -> None\nselect an instrument, with a value between 0 and 127"

#define DOC_OUTPUTWRITE "write(data) -> None\nwrites a list of midi data to the Output"

#define DOC_OUTPUTWRITESHORT "write_short(status) -> None\nwrite_short(status, data1 = 0, data2 = 0) -> None\nwrite_short(status <, data1><, data2>)"

#define DOC_OUTPUTWRITESYSEX "write_sys_ex(when, msg) -> None\nwrites a timestamped system-exclusive midi message."

#define DOC_PYGAMEMIDIGETCOUNT "get_count() -> num_devices\ngets the number of devices."

#define DOC_PYGAMEMIDIGETDEFAULTINPUTID "get_default_input_id() -> default_id\ngets default input device number"

#define DOC_PYGAMEMIDIGETDEFAULTOUTPUTID "get_default_output_id() -> default_id\ngets default output device number"

#define DOC_PYGAMEMIDIGETDEVICEINFO "get_device_info(an_id) -> (interf, name, input, output, opened)\nreturns information about a midi device"

#define DOC_PYGAMEMIDIINIT "init() -> None\ninitialize the midi module"

#define DOC_PYGAMEMIDIMIDIS2EVENTS "midis2events(midis, device_id) -> [Event, ...]\nconverts midi events to pygame events"

#define DOC_PYGAMEMIDIQUIT "quit() -> None\nuninitialize the midi module"

#define DOC_PYGAMEMIDITIME "time() -> time\nreturns the current time in ms of the PortMidi timer"



/* Docs in a comment... slightly easier to read. */

/*

pygame.midi
pygame module for interacting with midi input and output.

pygame.midi.Input
 Input(device_id) -> None
 Input(device_id, buffer_size) -> None
Input is used to get midi input from midi devices.

pygame.midi.Input.close
 close() -> None
closes a midi stream, flushing any pending buffers.

pygame.midi.Input.poll
 poll() -> Bool
returns true if there's data, or false if not.

pygame.midi.Input.read
 read(num_events) -> midi_event_list
reads num_events midi events from the buffer.

pygame.midi.MidiException
 MidiException(errno) -> None
exception that pygame.midi functions and classes can raise

pygame.midi.Output
 Output(device_id) -> None
 Output(device_id, latency = 0) -> None
 Output(device_id, buffer_size = 4096) -> None
 Output(device_id, latency, buffer_size) -> None
Output is used to send midi to an output device

pygame.midi.Output.abort
 abort() -> None
terminates outgoing messages immediately

pygame.midi.Output.close
 close() -> None
closes a midi stream, flushing any pending buffers.

pygame.midi.Output.note_off
 note_off(note, velocity=None, channel = 0) -> None
turns a midi note off.  Note must be on.

pygame.midi.Output.note_on
 note_on(note, velocity=None, channel = 0) -> None
turns a midi note on.  Note must be off.

pygame.midi.Output.set_instrument
 set_instrument(instrument_id, channel = 0) -> None
select an instrument, with a value between 0 and 127

pygame.midi.Output.write
 write(data) -> None
writes a list of midi data to the Output

pygame.midi.Output.write_short
 write_short(status) -> None
 write_short(status, data1 = 0, data2 = 0) -> None
write_short(status <, data1><, data2>)

pygame.midi.Output.write_sys_ex
 write_sys_ex(when, msg) -> None
writes a timestamped system-exclusive midi message.

pygame.midi.get_count
 get_count() -> num_devices
gets the number of devices.

pygame.midi.get_default_input_id
 get_default_input_id() -> default_id
gets default input device number

pygame.midi.get_default_output_id
 get_default_output_id() -> default_id
gets default output device number

pygame.midi.get_device_info
 get_device_info(an_id) -> (interf, name, input, output, opened)
returns information about a midi device

pygame.midi.init
 init() -> None
initialize the midi module

pygame.midi.midis2events
 midis2events(midis, device_id) -> [Event, ...]
converts midi events to pygame events

pygame.midi.quit
 quit() -> None
uninitialize the midi module

pygame.midi.time
 time() -> time
returns the current time in ms of the PortMidi timer

*/