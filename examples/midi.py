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

    This is a piano keyboard example, with a two octave keyboard, starting at
    note F3. Left mouse down over a key starts a note, left up stops it. The
    notes are also mapped to the computer keyboard keys, assuming an American
    English keyboard (sorry everyone else, but I don't know if I can map to
    absolute key position instead of value.) The white keys are on the second
    row, TAB to BACKSLASH, starting with note F3. The black keys map to the top
    row, '1' to BACKSPACE, starting with F#3. Close the window or pressing
    ESCAPE to quit the program.

    Midi output is to the default midi output device for the computer.
    
    """

    CHURCH_ORGAN = 19

    instrument = CHURCH_ORGAN
    start_note = 33  # F3 (white key note)
    n_notes = 24  # Two octaves (14 white keys)

    bg_color = Color('steelblue1')

    port = pygame.midi.get_default_output_device_id()
    pygame.init()
    pygame.midi.init()
    midi_out = pygame.midi.Output(port, 0)
    try:
        keyboard_id = 1
        midi_out.set_instrument(instrument)
        keyboard = Keyboard(start_note, n_notes, keyboard_id)

        screen = pygame.display.set_mode(keyboard.rect.size)
        screen.fill(bg_color)
        pygame.display.flip()

        background = pygame.Surface(screen.get_size())
        background.fill(bg_color)
        pygame.display.update(keyboard.draw(screen, background, []))

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
            update_rects = None
            for e in pygame.event.get():
                if e.type == pygame.MOUSEBUTTONDOWN:
                    mouse_note, id, velocity = keyboard.mouse_down(e.pos)
                    if id == keyboard_id:
                        midi_out.note_on(mouse_note, velocity)
                    else:
                        mouse_note = 0
                elif e.type == pygame.MOUSEBUTTONUP:
                    if mouse_note:
                        midi_out.note_off(mouse_note)
                        keyboard.key_up(mouse_note)
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
                        keyboard.key_down(note)
                        midi_out.note_on(note, velocity)
                elif e.type == pygame.KEYUP:
                    try:
                        note, __ = key_mapping[e.key]
                    except KeyError:
                        pass
                    else:
                        keyboard.key_up(note)
                        midi_out.note_off(note, 0)

            pygame.display.update(keyboard.draw(screen, background, []))
    finally:
        del midi_out
        pygame.midi.quit()

class NullKey(object):
    def notify(self, event):
        pass

null_key = NullKey()

class Key(object):
    WHITE_KEY_WIDTH = 42
    WHITE_KEY_HEIGHT = 160

    DOWN_STATE_NONE = 0
    DOWN_STATE_SELF = 1
    DOWN_STATE_BLACK = DOWN_STATE_SELF << 1
    DOWN_STATE_SELF_BLACK = DOWN_STATE_SELF | DOWN_STATE_BLACK
    DOWN_STATE_WHITE = DOWN_STATE_BLACK << 1
    DOWN_STATE_SELF_WHITE = DOWN_STATE_SELF | DOWN_STATE_WHITE
    DOWN_STATE_BLACK_WHITE = DOWN_STATE_BLACK | DOWN_STATE_WHITE
    DOWN_STATE_ALL = DOWN_STATE_SELF | DOWN_STATE_BLACK_WHITE

    EVENT_DOWN = 0
    EVENT_UP = 1
    EVENT_BLACK_DOWN = 2
    EVENT_BLACK_UP = 3
    EVENT_WHITE_DOWN = 4
    EVENT_WHITE_UP = 5

    _state_table = None  # Replace with a mapping in subclasses.
                    
    _strip = pygame.image.load('data/midikeys.png')
    _image_names = ['black 1', 'black 2', 'white 1', 'white 2', 'white 3',
                    'white left 1', 'white left 2', 'white left 3',
                    'white left 4', 'white left 5', 'white left 6',
                    'white center 1', 'white center 2', 'white center 3',
                    'white center 4', 'white center 5', 'white center 6',
                    'white right 1', 'white right 2', 'white right 3']
    _image_offsets = {}
    for i in range(len(_image_names)):
        _image_offsets[_image_names[i]] = i * WHITE_KEY_WIDTH

    try:
        _update = set()
    except NameError:
        import sets
        _update = sets.Set()

    def __init__(self, ident, rect, source_rect, key_left = None):
        if key_left is None:
            key_left = null_key
        self.rect = Rect(rect)
        self._state = Key.DOWN_STATE_NONE
        self._source_rect = source_rect
        self._ident = ident
        self._hash = hash(self._ident)
        self._key_left = key_left
        self._background_rect = Rect(rect.left, rect.bottom - 10,
                                     rect.width, 10)
        self._update.add(self)

    def notify(self, event):
        self._state, source_rect = self._state_table[self._state][event]
        if source_rect is not None:
            self._source_rect = source_rect
            Key._update.add(self)
            self.notify_left()

    def notify_left(self):
        raise NotImplementedError("notify_left method not overridden")

    def __eq__(self, other):
        return self._ident == other._ident

    def __hash__(self):
        return self._hash

    def __str__(self):
        return ("<Key %s at (%d, %d)>" %
                (self._ident, self.rect.top, self.rect.left))

    def _draw(self, surf, background, dirty_rects):
        surf.blit(background, self._background_rect, self._background_rect)
        surf.blit(self._strip, self.rect, self._source_rect)
        dirty_rects.append(self.rect)
        return dirty_rects

    def _add_state(cls, table, state, *pairs):
        mapping = {cls.EVENT_UP: (state, None),
                   cls.EVENT_DOWN: (state, None),
                   cls.EVENT_BLACK_DOWN: (state, None),
                   cls.EVENT_BLACK_UP: (state, None),
                   cls.EVENT_WHITE_DOWN: (state, None),
                   cls.EVENT_WHITE_UP: (state, None)}
        mapping.update(pairs)
        table[state] = mapping
    _add_state = classmethod(_add_state)

class BlackKey(Key):
    BLACK_KEY_WIDTH = 22
    BLACK_KEY_HEIGHT = 94

    _up_rect = (Key._image_offsets['black 1'], 0,
                BLACK_KEY_WIDTH, BLACK_KEY_HEIGHT)
    _down_rect = (Key._image_offsets['black 2'], 0,
                  BLACK_KEY_WIDTH, BLACK_KEY_HEIGHT)

    _state_table = {}
    Key._add_state(_state_table, Key.DOWN_STATE_NONE,
                   (Key.EVENT_DOWN, (Key.DOWN_STATE_SELF, _down_rect)))
    Key._add_state(_state_table, Key.DOWN_STATE_SELF,
                   (Key.EVENT_UP, (Key.DOWN_STATE_NONE, _up_rect)))

    def __init__(self, ident, posn, key_left = None):
        rect =  Rect(posn[0], posn[1],
                     self.BLACK_KEY_WIDTH, self.BLACK_KEY_HEIGHT)
        Key.__init__(self, ident, rect, self._up_rect, key_left)

    def notify_left(self):
        if self._state & self.DOWN_STATE_SELF:
            self._key_left.notify(Key.EVENT_BLACK_DOWN)
        else:
            self._key_left.notify(Key.EVENT_BLACK_UP)

class WhiteKey(Key):
    _up_rect = (Key._image_offsets['white 1'], 0,
                Key.WHITE_KEY_WIDTH, Key.WHITE_KEY_HEIGHT)
    _down_rect_1 = (Key._image_offsets['white 2'], 0,
                    Key.WHITE_KEY_WIDTH, Key.WHITE_KEY_HEIGHT)
    _down_rect_2 = (Key._image_offsets['white 3'], 0,
                    Key.WHITE_KEY_WIDTH, Key.WHITE_KEY_HEIGHT)

    _state_table = {}
    Key._add_state(_state_table, Key.DOWN_STATE_NONE,
        (Key.EVENT_DOWN, (Key.DOWN_STATE_SELF, _down_rect_1)),
        (Key.EVENT_WHITE_DOWN, (Key.DOWN_STATE_WHITE, None)))
    Key._add_state(_state_table, Key.DOWN_STATE_SELF,
        (Key.EVENT_UP, (Key.DOWN_STATE_NONE, _up_rect)),
        (Key.EVENT_WHITE_DOWN, (Key.DOWN_STATE_SELF_WHITE, _down_rect_2)))
    Key._add_state(_state_table, Key.DOWN_STATE_WHITE,
        (Key.EVENT_DOWN, (Key.DOWN_STATE_SELF_WHITE, _down_rect_2)),
        (Key.EVENT_WHITE_UP, (Key.DOWN_STATE_NONE, None)))
    Key._add_state(_state_table, Key.DOWN_STATE_SELF_WHITE,
        (Key.EVENT_UP, (Key.DOWN_STATE_WHITE, _up_rect)),
        (Key.EVENT_WHITE_UP, (Key.DOWN_STATE_SELF, _down_rect_1)))
    
    def __init__(self, ident, posn, key_left = None):
        rect = Rect(posn[0], posn[1],
                    self.WHITE_KEY_WIDTH, self.WHITE_KEY_HEIGHT)
        Key.__init__(self, ident, rect, self._up_rect, key_left)

    def notify_left(self):
        if self._state & self.DOWN_STATE_SELF:
            self._key_left.notify(self.EVENT_WHITE_DOWN)
        else:
            self._key_left.notify(self.EVENT_WHITE_UP)

class WhiteKeyLeft(Key):
    _up_rect_1 = (Key._image_offsets['white left 1'], 0,
                  Key.WHITE_KEY_WIDTH, Key.WHITE_KEY_HEIGHT)
    _up_rect_2 = (Key._image_offsets['white left 3'], 0,
                  Key.WHITE_KEY_WIDTH, Key.WHITE_KEY_HEIGHT)                  
    _down_rect_1 = (Key._image_offsets['white left 2'], 0,
                    Key.WHITE_KEY_WIDTH, Key.WHITE_KEY_HEIGHT)
    _down_rect_2 = (Key._image_offsets['white left 4'], 0,
                    Key.WHITE_KEY_WIDTH, Key.WHITE_KEY_HEIGHT)
    _down_rect_3 = (Key._image_offsets['white left 5'], 0,
                    Key.WHITE_KEY_WIDTH, Key.WHITE_KEY_HEIGHT)
    _down_rect_4 = (Key._image_offsets['white left 6'], 0,
                    Key.WHITE_KEY_WIDTH, Key.WHITE_KEY_HEIGHT)

    _state_table = {}
    Key._add_state(_state_table, Key.DOWN_STATE_NONE,
        (Key.EVENT_DOWN, (Key.DOWN_STATE_SELF, _down_rect_1)),
        (Key.EVENT_BLACK_DOWN, (Key.DOWN_STATE_BLACK, _up_rect_2)),
        (Key.EVENT_WHITE_DOWN, (Key.DOWN_STATE_WHITE, None)))
    Key._add_state(_state_table, Key.DOWN_STATE_SELF,
        (Key.EVENT_UP, (Key.DOWN_STATE_NONE, _up_rect_1)),
        (Key.EVENT_BLACK_DOWN, (Key.DOWN_STATE_SELF_BLACK, _down_rect_2)),
        (Key.EVENT_WHITE_DOWN, (Key.DOWN_STATE_SELF_WHITE, _down_rect_3)))
    Key._add_state(_state_table, Key.DOWN_STATE_BLACK,
        (Key.EVENT_DOWN, (Key.DOWN_STATE_SELF_BLACK, _down_rect_2)),
        (Key.EVENT_BLACK_UP, (Key.DOWN_STATE_NONE, _up_rect_1)),
        (Key.EVENT_WHITE_DOWN, (Key.DOWN_STATE_BLACK_WHITE, None)))
    Key._add_state(_state_table, Key.DOWN_STATE_SELF_BLACK,
        (Key.EVENT_UP, (Key.DOWN_STATE_BLACK, _up_rect_2)),
        (Key.EVENT_BLACK_UP, (Key.DOWN_STATE_SELF, _down_rect_1)),
        (Key.EVENT_WHITE_DOWN, (Key.DOWN_STATE_ALL, _down_rect_4)))
    Key._add_state(_state_table, Key.DOWN_STATE_WHITE,
        (Key.EVENT_DOWN, (Key.DOWN_STATE_SELF_WHITE, _down_rect_3)),
        (Key.EVENT_BLACK_DOWN, (Key.DOWN_STATE_BLACK_WHITE, _up_rect_2)),
        (Key.EVENT_WHITE_UP, (Key.DOWN_STATE_NONE, None)))
    Key._add_state(_state_table, Key.DOWN_STATE_SELF_WHITE,
        (Key.EVENT_UP, (Key.DOWN_STATE_WHITE, _up_rect_1)),
        (Key.EVENT_BLACK_DOWN, (Key.DOWN_STATE_ALL, _down_rect_4)),
        (Key.EVENT_WHITE_UP, (Key.DOWN_STATE_SELF, _down_rect_1)))
    Key._add_state(_state_table, Key.DOWN_STATE_BLACK_WHITE,
        (Key.EVENT_DOWN, (Key.DOWN_STATE_ALL, _down_rect_4)),
        (Key.EVENT_BLACK_UP, (Key.DOWN_STATE_WHITE, _up_rect_1)),
        (Key.EVENT_WHITE_UP, (Key.DOWN_STATE_BLACK, None)))
    Key._add_state(_state_table, Key.DOWN_STATE_ALL,
        (Key.EVENT_UP, (Key.DOWN_STATE_BLACK_WHITE, _up_rect_2)),
        (Key.EVENT_BLACK_UP, (Key.DOWN_STATE_SELF_WHITE, _down_rect_3)),
        (Key.EVENT_WHITE_UP, (Key.DOWN_STATE_SELF_BLACK, _down_rect_2)))
    
    def __init__(self, ident, posn, key_left=None):
        rect = Rect(posn[0], posn[1],
                    self.WHITE_KEY_WIDTH, self.WHITE_KEY_HEIGHT)
        Key.__init__(self, ident, rect, self._up_rect_1, key_left)

    def notify_left(self):
        if self._state & self.DOWN_STATE_SELF:
            self._key_left.notify(self.EVENT_WHITE_DOWN)
        else:
            self._key_left.notify(self.EVENT_WHITE_UP)

class WhiteKeyCenter(Key):
    _up_rect_1 = (Key._image_offsets['white center 1'], 0,
                  Key.WHITE_KEY_WIDTH, Key.WHITE_KEY_HEIGHT)
    _up_rect_2 = (Key._image_offsets['white center 3'], 0,
                  Key.WHITE_KEY_WIDTH, Key.WHITE_KEY_HEIGHT)                  
    _down_rect_1 = (Key._image_offsets['white center 2'], 0,
                    Key.WHITE_KEY_WIDTH, Key.WHITE_KEY_HEIGHT)
    _down_rect_2 = (Key._image_offsets['white center 4'], 0,
                    Key.WHITE_KEY_WIDTH, Key.WHITE_KEY_HEIGHT)
    _down_rect_3 = (Key._image_offsets['white center 5'], 0,
                    Key.WHITE_KEY_WIDTH, Key.WHITE_KEY_HEIGHT)
    _down_rect_4 = (Key._image_offsets['white center 6'], 0,
                    Key.WHITE_KEY_WIDTH, Key.WHITE_KEY_HEIGHT)

    _state_table = {}
    Key._add_state(_state_table, Key.DOWN_STATE_NONE,
        (Key.EVENT_DOWN, (Key.DOWN_STATE_SELF, _down_rect_1)),
        (Key.EVENT_BLACK_DOWN, (Key.DOWN_STATE_BLACK, _up_rect_2)),
        (Key.EVENT_WHITE_DOWN, (Key.DOWN_STATE_WHITE, None)))
    Key._add_state(_state_table, Key.DOWN_STATE_SELF,
        (Key.EVENT_UP, (Key.DOWN_STATE_NONE, _up_rect_1)),
        (Key.EVENT_BLACK_DOWN, (Key.DOWN_STATE_SELF_BLACK, _down_rect_2)),
        (Key.EVENT_WHITE_DOWN, (Key.DOWN_STATE_SELF_WHITE, _down_rect_3)))
    Key._add_state(_state_table, Key.DOWN_STATE_BLACK,
        (Key.EVENT_DOWN, (Key.DOWN_STATE_SELF_BLACK, _down_rect_2)),
        (Key.EVENT_BLACK_UP, (Key.DOWN_STATE_NONE, _up_rect_1)),
        (Key.EVENT_WHITE_DOWN, (Key.DOWN_STATE_BLACK_WHITE, None)))
    Key._add_state(_state_table, Key.DOWN_STATE_SELF_BLACK,
        (Key.EVENT_UP, (Key.DOWN_STATE_BLACK, _up_rect_2)),
        (Key.EVENT_BLACK_UP, (Key.DOWN_STATE_SELF, _down_rect_1)),
        (Key.EVENT_WHITE_DOWN, (Key.DOWN_STATE_ALL, _down_rect_4)))
    Key._add_state(_state_table, Key.DOWN_STATE_WHITE,
        (Key.EVENT_DOWN, (Key.DOWN_STATE_SELF_WHITE, _down_rect_3)),
        (Key.EVENT_BLACK_DOWN, (Key.DOWN_STATE_BLACK_WHITE, _up_rect_2)),
        (Key.EVENT_WHITE_UP, (Key.DOWN_STATE_NONE, None)))
    Key._add_state(_state_table, Key.DOWN_STATE_SELF_WHITE,
        (Key.EVENT_UP, (Key.DOWN_STATE_WHITE, _up_rect_1)),
        (Key.EVENT_BLACK_DOWN, (Key.DOWN_STATE_ALL, _down_rect_4)),
        (Key.EVENT_WHITE_UP, (Key.DOWN_STATE_SELF, _down_rect_1)))
    Key._add_state(_state_table, Key.DOWN_STATE_BLACK_WHITE,
        (Key.EVENT_DOWN, (Key.DOWN_STATE_ALL, _down_rect_4)),
        (Key.EVENT_BLACK_UP, (Key.DOWN_STATE_WHITE, _up_rect_1)),
        (Key.EVENT_WHITE_UP, (Key.DOWN_STATE_BLACK, None)))
    Key._add_state(_state_table, Key.DOWN_STATE_ALL,
        (Key.EVENT_UP, (Key.DOWN_STATE_BLACK_WHITE, _up_rect_2)),
        (Key.EVENT_BLACK_UP, (Key.DOWN_STATE_SELF_WHITE, _down_rect_3)),
        (Key.EVENT_WHITE_UP, (Key.DOWN_STATE_SELF_BLACK, _down_rect_2)))
    
    def __init__(self, ident, posn, key_left=None):
        rect = Rect(posn[0], posn[1],
                    self.WHITE_KEY_WIDTH, self.WHITE_KEY_HEIGHT)
        Key.__init__(self, ident, rect, self._up_rect_1, key_left)

    def notify_left(self):
        if self._state & self.DOWN_STATE_SELF:
            self._key_left.notify(self.EVENT_WHITE_DOWN)
        else:
            self._key_left.notify(self.EVENT_WHITE_UP)

class WhiteKeyRight(Key):
    _up_rect = (Key._image_offsets['white right 1'], 0,
                Key.WHITE_KEY_WIDTH, Key.WHITE_KEY_HEIGHT)
    _down_rect_1 = (Key._image_offsets['white right 2'], 0,
                    Key.WHITE_KEY_WIDTH, Key.WHITE_KEY_HEIGHT)
    _down_rect_2 = (Key._image_offsets['white right 3'], 0,
                    Key.WHITE_KEY_WIDTH, Key.WHITE_KEY_HEIGHT)

    _state_table = {}
    Key._add_state(_state_table, Key.DOWN_STATE_NONE,
        (Key.EVENT_DOWN, (Key.DOWN_STATE_SELF, _down_rect_1)),
        (Key.EVENT_WHITE_DOWN, (Key.DOWN_STATE_WHITE, None)))
    Key._add_state(_state_table, Key.DOWN_STATE_SELF,
        (Key.EVENT_UP, (Key.DOWN_STATE_NONE, _up_rect)),
        (Key.EVENT_WHITE_DOWN, (Key.DOWN_STATE_SELF_WHITE, _down_rect_2)))
    Key._add_state(_state_table, Key.DOWN_STATE_WHITE,
        (Key.EVENT_DOWN, (Key.DOWN_STATE_SELF_WHITE, _down_rect_2)),
        (Key.EVENT_WHITE_UP, (Key.DOWN_STATE_NONE, None)))
    Key._add_state(_state_table, Key.DOWN_STATE_SELF_WHITE,
        (Key.EVENT_UP, (Key.DOWN_STATE_WHITE, _up_rect)),
        (Key.EVENT_WHITE_UP, (Key.DOWN_STATE_SELF, _down_rect_1)))
    
    def __init__(self, ident, posn, key_left = None):
        rect = Rect(posn[0], posn[1],
                    self.WHITE_KEY_WIDTH, self.WHITE_KEY_HEIGHT)
        Key.__init__(self, ident, rect, self._up_rect, key_left)

    def notify_left(self):
        if self._state & self.DOWN_STATE_SELF:
            self._key_left.notify(self.EVENT_WHITE_DOWN)
        else:
            self._key_left.notify(self.EVENT_WHITE_UP)

class Keyboard(object):
    def __init__(self, start_note, n_notes, id):
        self._start_note = start_note
        self._end_note = start_note + n_notes - 1
        self._id = id
        self._set_rect()
        self._add_keys()
        self._map_regions()

    def _set_rect(self):
        n_white_notes = 0
        for note in range(self._start_note, self._end_note + 1):
            if is_white_key(note):
                n_white_notes += 1
        top = 0
        left = 0
        width = n_white_notes * Key.WHITE_KEY_WIDTH
        height = Key.WHITE_KEY_HEIGHT
        if not is_white_key(self._start_note):
            shift = BlackKey.BLACK_KEY_WIDTH // 2
            left += shift
            width += shift
        if not is_white_key(self._end_note):
            width += BlackKey.BLACK_KEY_WIDTH // 2

        self._n_white_notes = n_white_notes
        self.rect = Rect(0, 0, width, height)
        
    def _add_keys(self):
        key_map = {}

        start_note = self._start_note
        end_note = self._end_note
        black_offset = BlackKey.BLACK_KEY_WIDTH // 2
        prev_white_key = None
        if is_white_key(start_note):
            x = 0
            is_prev_white = True
        else:
            x = black_offset
            is_prev_white = False
        for note in range(start_note, end_note + 1):
            ident = note  # For now notes uniquely identify keyboard keys.
            if is_white_key(note):
                if is_prev_white:
                    if note == end_note or is_white_key(note + 1):
                        key = WhiteKey(ident, (x, 0), prev_white_key)
                    else:
                        key = WhiteKeyLeft(ident, (x, 0), prev_white_key)
                else:
                    if note == end_note or is_white_key(note + 1):
                        key = WhiteKeyRight(ident, (x, 0), prev_white_key)
                    else:
                        key = WhiteKeyCenter(ident, (x, 0), prev_white_key)
                is_prev_white = True
                x += Key.WHITE_KEY_WIDTH
                prev_white_key = key
            else:
                key = BlackKey(ident, (x - black_offset, 0), prev_white_key)
                is_prev_white = False
            key_map[note] = key

        self._keys = key_map

    def _map_regions(self):
        id = self._id
        regions = pygame.Surface(self.rect.size, 0, 24)
        black_keys = []
        for note, key in self._keys.iteritems():
            if is_white_key(note):
                regions.fill((note, id, 127), key.rect)
            else:
                black_keys.append((note, key))
        for note, key in black_keys:
            regions.fill((note, id, 127), key.rect)
        self._regions = regions

    def mouse_down(self, posn):
        message = self._regions.get_at(posn)
        self.key_down(message[0])
        return message[:3]

    def mouse_up(self, posn):
        note, id, __ = self._regions.get_at(posn)
        self.key_up(note)
        return note

    def draw(self, surf, background, dirty_rects):
        changed_keys = Key._update
        while changed_keys:
            dirty_rects = Key._update.pop()._draw(surf,
                                                  background,
                                                  dirty_rects)
        return dirty_rects

    def key_down(self, note):
        self._keys[note].notify(Key.EVENT_DOWN)

    def key_up(self, note):
        self._keys[note].notify(Key.EVENT_UP)

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
