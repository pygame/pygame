"""
Contains an example of midi input, and a separate example of midi output.

By default it runs the output example.
python midi.py --output
python midi.py --input
"""


import pygame
import pygame.midi
from pygame.locals import *



def input_main():
    import pygame
    from pygame.locals import *
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
    GRAND_PIANO = 0
    CHURCH_ORGAN = 19


    pygame.init()
    pygame.midi.init()


    try:
        midi_out = pygame.midi.Output(0, 0)
        midi_out.set_instrument(CHURCH_ORGAN)
        screen = pygame.display.set_mode((200, 200))
        screen.fill((0, 0, 255))
        pygame.display.flip()

        regions = pygame.Surface((200, 200), 0, 24)
        regions.fill((60, 127, 0), (0, 0, 50, 200))
        regions.fill((61, 127, 0), (50, 0, 50, 200))
        regions.fill((62, 127, 0), (100, 0, 50, 200))
        regions.fill((63, 127, 0), (150, 0, 50, 200))

        key_mapping = {pygame.K_q: (60, 127),
                       pygame.K_w: (61, 127),
                       pygame.K_e: (62, 127),
                       pygame.K_r: (63, 127)}

        pygame.event.set_blocked(MOUSEMOTION)
        repeat = 1
        mouse_note = 0
        while repeat:
            for e in pygame.event.get():
                if e.type == pygame.MOUSEBUTTONDOWN:
                    mouse_note, velocity, __, __ = regions.get_at(e.pos)
                    midi_out.note_on(mouse_note, velocity)
                elif e.type == pygame.MOUSEBUTTONUP:
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

if __name__ == '__main__':
    if "--input" in sys.argv or "-i" in sys.argv:
        input_main()

    #elif "--output" in sys.argv or "-o" in sys.argv:
    # output example is run by default.
    else:
        output_main()




