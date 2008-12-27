"""Contains an example of midi input, and a separate example of midi output.

By default it runs the output example.
python midi.py --output
python midi.py --input

"""

import sys

import pygame
import pygame.midi
from pygame.locals import *



def input_main():
    pygame.init()
    pygame.fastevent.init()
    event_get = pygame.fastevent.get
    event_post = pygame.fastevent.post

    pygame.midi.init()

    i = pygame.midi.Input(1)

    pygame.display.set_mode((1,1))



    going = True
    while going:
        events = event_get()
        for e in events:
            if e.type in [QUIT]:
                going = False
            if e.type in [KEYDOWN]:
                going = False
            if e.type in [pygame.midi.MIDIIN]:
                print e

        if i.poll():
            midi_events = i.read(10)
            # convert them into pygame events.
            midi_evs = pygame.midi.midis2events(midi_events, i.device_id)

            for m_e in midi_evs:
                event_post( m_e )

    del i
    pygame.midi.quit()



def output_main():
    """Execute a musical keyboard example for the Church Organ instrument

    This is a simple piano keyboard example, with a two octave keyboard,
    starting at note F3. Left mouse down over a key starts a note, left up
    stops it. The notes are also mapped to the computer keyboard keys,
    assuming an American English keyboard (sorry everyone else, but I don't
    know if I can map to absolute key position instead of value.) The white
    keys are on the second row, TAB to BACKSLASH, starting with note F3.
    The black keys map to the top row, '1' to BACKSPACE, starting with
    F#3. Close the window or pressing ESCAPE to quit the program.

    Midi output is to the default midi output device for the computer.
    
    """

    CHURCH_ORGAN = 19

    instrument = CHURCH_ORGAN
    start_note = 33  # F3 (white key note)
    n_notes = 24  # Two octaves (14 white keys)

    port = pygame.midi.get_default_output_device_id()
    pygame.init()
    pygame.midi.init()
    midi_out = pygame.midi.Output(port, 0)
    try:
        # The window is sized to the keyboard image. Surface 'regions'
        # translates mouse position to musical instrument key. Each pixel
        # represents a (note, velocity, widget id) triplet. Dictionary
        # 'key_mapping' translates a computer keyboard key to a musical note.
        # The main event loop simply uses 'regions' for mouse down events,
        # 'key_mapping' for key down abd up events. 'mouse_note' records the
        # last note selected with a left mouse down for the later note-off
        # midi message when the mouse button is released. It is zero when no
        # note was selected.
        keyboard_id = 1
        midi_out.set_instrument(instrument)
        image, regions = keyboard(start_note, n_notes, keyboard_id)

        screen = pygame.display.set_mode(image.get_size())
        image = image.convert()
        screen.blit(image, (0, 0))
        pygame.display.flip()

        key_mapping = {pygame.K_TAB: (33, 127),
                       pygame.K_1: (34, 127),
                       pygame.K_q: (35, 127),
                       pygame.K_2: (36, 127),
                       pygame.K_w: (37, 127),
                       pygame.K_3: (38, 127),
                       pygame.K_e: (39, 127),
                       pygame.K_r: (40, 127),
                       pygame.K_5: (41, 127),
                       pygame.K_t: (42, 127),
                       pygame.K_6: (43, 127),
                       pygame.K_y: (44, 127),
                       pygame.K_u: (45, 127),
                       pygame.K_8: (46, 127),
                       pygame.K_i: (47, 127),
                       pygame.K_9: (48, 127),
                       pygame.K_o: (49, 127),
                       pygame.K_0: (50, 127),
                       pygame.K_p: (51, 127),
                       pygame.K_LEFTBRACKET: (52, 127),
                       pygame.K_EQUALS: (53, 127),
                       pygame.K_RIGHTBRACKET: (54, 127),
                       pygame.K_BACKSPACE: (55, 127),
                       pygame.K_BACKSLASH: (56, 127),
                      }

        pygame.event.set_blocked(MOUSEMOTION)
        repeat = 1
        mouse_note = 0
        while repeat:
            for e in pygame.event.get():
                if e.type == pygame.MOUSEBUTTONDOWN:
                    mouse_note, velocity, id, __ = regions.get_at(e.pos)
                    if id == keyboard_id:
                        midi_out.note_on(mouse_note, velocity)
                    else:
                        mouse_note = 0
                elif e.type == pygame.MOUSEBUTTONUP:
                    if mouse_note:
                        midi_out.note_off(mouse_note)
                elif e.type == pygame.QUIT:
                    repeat = 0
                    break
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        repeat = 0
                        break
                    try:
                        note, velocity = key_mapping[e.key]
                    except KeyError:
                        pass
                    else:
                        midi_out.note_on(note, velocity)
                elif e.type == pygame.KEYUP:
                    try:
                        note, __ = key_mapping[e.key]
                    except KeyError:
                        pass
                    else:
                        midi_out.note_off(note, 0)
    
            pygame.display.flip()
    finally:
        del midi_out
        pygame.midi.quit()

def keyboard(start_note, n_notes, id):
    """Return a keyboard image, region map pair

    The keyboard is a stylized n_notes note piano keyboard, with the number and
    position of black and white keys determined by start_note. The region map
    is a 3 byte surface, where, for each pixel, the red part is the note number,
    the green part the midi velocity (127 for now), and the blue part a widget
    identifier of value id.
    
    """
    
    white_key_width = 40
    white_key_height = 4 * white_key_width
    black_key_width = white_key_width // 2
    black_key_height = int(0.6 * white_key_height + 0.5)

    white_key_notes = [note for note in range(start_note, start_note + n_notes)
                            if is_white_key(note)]
    top = 0
    left = 0
    width = len(white_key_notes) * white_key_width
    height = white_key_height
    if not is_white_key(start_note):
        shift = black_key_width // 2
        left += shift
        width += shift
    if not is_white_key(start_note):
        width += (black_key_width + 1) // 2

    image = pygame.Surface((width, height), 0, 24)
    image.fill((255, 255, 255))

    regions = pygame.Surface((width, height), 0, 24)

    # Add white keys
    for x in range(len(white_key_notes)):
        rect = pygame.Rect(x * white_key_width + left, top,
                           white_key_width, white_key_height)
        pygame.draw.line(image,
                         (127, 127, 127),
                         rect.topleft,
                         rect.bottomleft,
                         1)
        pygame.draw.line(image,
                         (127, 127, 127),
                         rect.topright,
                         rect.bottomright,
                         1)
        regions.fill((white_key_notes[x], 127, id), rect)

    # Add black keys
    x = 0
    for note in range(start_note, start_note + n_notes):
        if is_white_key(note):
            x += 1
        else:
            key_left = white_key_width * x - black_key_width // 2 + left
            rect = pygame.Rect(key_left, top,
                               black_key_width, black_key_height)
            image.fill((0, 0, 0), rect)
            regions.fill((note, 127, id), rect)

    return image, regions

def is_white_key(note):
    """True if note is represented by a white key"""
    
    key_pattern = [True, False, True, True, False, True,
                   False, True, True, False, True, False]
    return key_pattern[(note - 1) % len(key_pattern)]



if __name__ == '__main__':
    if "--input" in sys.argv or "-i" in sys.argv:
        input_main()

    #elif "--output" in sys.argv or "-o" in sys.argv:
    # output example is run by default.
    else:
        output_main()




