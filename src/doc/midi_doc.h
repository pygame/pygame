/* Auto generated file: with makeref.py .  Docs go in src/ *.doc . */
#define DOC_PYGAMEMIDI "pygame module for interacting with midi input and output."

#define DOC_PYGAMEMIDIINPUT "Input(device_id)\nInput(device_id, buffer_size)\nInput is used to get midi input from midi devices."

#define DOC_INPUTCLOSE "Input.close(): return None\n closes a midi stream, flushing any pending buffers."

#define DOC_INPUTPOLL "Input.poll(): return Bool\nreturns true if there's data, or false if not."

#define DOC_INPUTREAD "Input.read(num_events): return midi_event_list\nreads num_events midi events from the buffer."

#define DOC_PYGAMEMIDIMIDIEXCEPTION "MidiException(errno)\nexception that pygame.midi functions and classes can raise"

#define DOC_PYGAMEMIDIOUTPUT "Output(device_id)\nOutput(device_id, latency = 0)\nOutput(device_id, buffer_size = 4096)\nOutput(device_id, latency, buffer_size)\nOutput is used to send midi to an output device"

#define DOC_OUTPUTABORT "Output.abort(): return None\n terminates outgoing messages immediately"

#define DOC_OUTPUTCLOSE "Output.close(): return None\n closes a midi stream, flushing any pending buffers."

#define DOC_OUTPUTNOTEOFF "Output.note_off(note, velocity=None, channel = 0)\nturns a midi note off.  Note must be on."

#define DOC_OUTPUTNOTEON "Output.note_on(note, velocity=None, channel = 0)\nturns a midi note on.  Note must be off."

#define DOC_OUTPUTSETINSTRUMENT "Output.set_instrument(instrument_id, channel = 0)\nselect an instrument, with a value between 0 and 127"

#define DOC_OUTPUTWRITE "Output.write(data)\nwrites a list of midi data to the Output"

#define DOC_OUTPUTWRITESHORT "Output.write_short(status)\nOutput.write_short(status, data1 = 0, data2 = 0)\nwrite_short(status <, data1><, data2>)"

#define DOC_OUTPUTWRITESYSEX "Output.write_sys_ex(when, msg)\nwrites a timestamped system-exclusive midi message."

#define DOC_PYGAMEMIDIGETCOUNT "pygame.midi.get_count(): return num_devices\ngets the number of devices."

#define DOC_PYGAMEMIDIGETDEFAULTINPUTID "pygame.midi.get_default_input_id(): return default_id\ngets default input device number"

#define DOC_PYGAMEMIDIGETDEFAULTOUTPUTID "pygame.midi.get_default_output_id(): return default_id\ngets default output device number"

#define DOC_PYGAMEMIDIGETDEVICEINFO "pygame.midi.get_device_info(an_id): return (interf, name, input, output, opened)\n returns information about a midi device"

#define DOC_PYGAMEMIDIINIT "pygame.midi.init(): return None\ninitialize the midi module"

#define DOC_PYGAMEMIDIMIDIS2EVENTS "pygame.midi.midis2events(midis, device_id): return [Event, ...]\nconverts midi events to pygame events"

#define DOC_PYGAMEMIDIQUIT "pygame.midi.quit(): return None\nuninitialize the midi module"

#define DOC_PYGAMEMIDITIME "pygame.midi.time(): return time\nreturns the current time in ms of the PortMidi timer"



/* Docs in a comments... slightly easier to read. */


/*

pygame.midi
 pygame module for interacting with midi input and output.



pygame.midi.Input
 Input(device_id)
Input(device_id, buffer_size)
Input is used to get midi input from midi devices.



Input.close
 Input.close(): return None
 closes a midi stream, flushing any pending buffers.



Input.poll
 Input.poll(): return Bool
returns true if there's data, or false if not.



Input.read
 Input.read(num_events): return midi_event_list
reads num_events midi events from the buffer.



pygame.midi.MidiException
 MidiException(errno)
exception that pygame.midi functions and classes can raise



pygame.midi.Output
 Output(device_id)
Output(device_id, latency = 0)
Output(device_id, buffer_size = 4096)
Output(device_id, latency, buffer_size)
Output is used to send midi to an output device



Output.abort
 Output.abort(): return None
 terminates outgoing messages immediately



Output.close
 Output.close(): return None
 closes a midi stream, flushing any pending buffers.



Output.note_off
 Output.note_off(note, velocity=None, channel = 0)
turns a midi note off.  Note must be on.



Output.note_on
 Output.note_on(note, velocity=None, channel = 0)
turns a midi note on.  Note must be off.



Output.set_instrument
 Output.set_instrument(instrument_id, channel = 0)
select an instrument, with a value between 0 and 127



Output.write
 Output.write(data)
writes a list of midi data to the Output



Output.write_short
 Output.write_short(status)
Output.write_short(status, data1 = 0, data2 = 0)
write_short(status <, data1><, data2>)



Output.write_sys_ex
 Output.write_sys_ex(when, msg)
writes a timestamped system-exclusive midi message.



pygame.midi.get_count
 pygame.midi.get_count(): return num_devices
gets the number of devices.



pygame.midi.get_default_input_id
 pygame.midi.get_default_input_id(): return default_id
gets default input device number



pygame.midi.get_default_output_id
 pygame.midi.get_default_output_id(): return default_id
gets default output device number



pygame.midi.get_device_info
 pygame.midi.get_device_info(an_id): return (interf, name, input, output, opened)
 returns information about a midi device



pygame.midi.init
 pygame.midi.init(): return None
initialize the midi module



pygame.midi.midis2events
 pygame.midi.midis2events(midis, device_id): return [Event, ...]
converts midi events to pygame events



pygame.midi.quit
 pygame.midi.quit(): return None
uninitialize the midi module



pygame.midi.time
 pygame.midi.time(): return time
returns the current time in ms of the PortMidi timer



*/

