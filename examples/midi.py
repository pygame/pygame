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

    # A note to new pygamers:
    #
    # All the midi module stuff is in this function. It is unnecessary to
    # understand how the keyboard display works to appreciate how midi
    # messages are sent.

    # The keyboard is drawn by a Keyboard instance. This instance maps window
    # position and Midi note to musical keyboard keys. A separate key_mapping
    # dictionary maps computer keyboard keys to (Midi note, velocity) pairs.
    # Midi sound is controlled with direct method calls to a pygame.midi.Output
    # instance.
    #
    # Things to consider when using pygame.midi:
    #
    # 1) Initialize the midi module with an explicitly call to
    #    pygame.midi.init().
    # 2) Create a midi.Output instance for the desired output device port.
    # 3) Select instruments with set_instrument() method calls.
    # 4) Play notes with note_on() and note_off() method calls.
    # 5) Call pygame.midi.Quit() when finished. Though the midi module tries
    #    to ensure that midi is properly shut down, it is best to do it
    #    explicitly. A try/finally statement is the safest way to do this.
    #
    GRAND_PIANO = 0
    CHURCH_ORGAN = 19

    instrument = CHURCH_ORGAN
    #instrument = GRAND_PIANO
    start_note = 33  # F3 (white key note)
    n_notes = 24  # Two octaves (14 white keys)

    bg_color = Color('slategray')

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
        dirty_rects = []
        keyboard.draw(screen, background, dirty_rects)
        pygame.display.update(dirty_rects)

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

            dirty_rects = []
            keyboard.draw(screen, background, dirty_rects)
            pygame.display.update(dirty_rects)
    finally:
        del midi_out
        pygame.midi.quit()

class NullKey(object):
    """A dummy key that ignores events passed to it by other keys

    A NullKey instance is the left key instance used by default
    for the left most keyboard key.

    """
    
    def _right_white_down(self):
        pass

    def _right_white_up(self):
        pass

    def _right_black_down(self):
        pass

    def _right_black_up(self):
        pass

null_key = NullKey()

def key_class(updates, image_strip, image_rects, is_white=True):
    """Return a keyboard key widget class

    Arguments:
    updates - a set into which a key instance adds itself if it needs
        redrawing.
    image_strip - The surface containing the images of all key states.
    image_rects - A list of Rects giving the regions within image_strip that
        are relevant to this key class.
    is_white (default True) - Set false if this is a black key.

    This function automates the creation of a key widget class for the
    three basic key types. A key has two basic states, up or down (
    depressed). Corresponding up and down images are drawn for each
    of these two states. But to give the illusion of depth, a key
    may have shadows cast upon it by the adjacent keys to its right.
    These shadows change depending on the up/down state of the key and
    its neighbors. So a key may support multiple images and states
    depending on the shadows. A key type is determined by the length
    of image_rects and the value of is_white.

    """

    # Naming convention: Variables used by the Key class as part of a
    # closure start with 'c_'.

    # State logic and shadows:
    #
    # A key may cast a shadow upon the key to its left. A black key casts a
    # shadow on an adjacent white key. The shadow changes depending of whether
    # the black or white key is depressed. A white key casts a shadow on the
    # white key to its left if it is up and the left key is down. Therefore
    # a keys state, and image it will draw, is determined entirely by its itself
    # and the key immediately adjacent to it on the right. A white key is always
    # assumed to have an adjacent white key.
    #
    # There can be up to eight key states, representing all permutations
    # of the three fundamental states of self up/down, adjacent white
    # right up/down, adjacent black up/down.
    # 
    down_state_none = 0
    down_state_self = 1
    down_state_white = down_state_self << 1
    down_state_self_white = down_state_self | down_state_white
    down_state_black = down_state_white << 1
    down_state_self_black = down_state_self | down_state_black
    down_state_white_black = down_state_white | down_state_black
    down_state_all = down_state_self | down_state_white_black

    # Some values used in the class.
    #
    c_down_state_initial = down_state_none
    c_down_state_rect_initial = image_rects[0]
    c_down_state_self = down_state_self
    c_updates = updates
    c_image_strip = image_strip
    c_width, c_height = image_rects[0].size

    # A key propogates its up/down state change to the adjacent white key on
    # the left by calling the adjacent key's _right_black_down or
    #_right_white_down method. 
    #
    if is_white:
        key_color = 'white'
    else:
        key_color = 'black'
    c_notify_down_method = "_right_%s_down" % key_color
    c_notify_up_method = "_right_%s_up" % key_color

    # Images:
    #
    # A black key only needs two images, for the up and down states. Its
    # appearance is unaffected by the adjacent keys to its right, which cast no
    # shadows upon it.
    #
    # A white key with a no adjacent black to its right only needs three
    # images, for self up, self down, and both self and adjacent white down.
    #
    # A white key with both a black and white key to its right needs six
    # images: self up, self up and adjacent black down, self down, self and
    # adjacent white down, self and adjacent black down, and all three down.
    #
    # Each 'c_event' dictionary maps the current key state to a new key state,
    # along with corresponding image, for the related event. If no redrawing
    # is required for the state change then the image rect is simply None.
    # 
    c_event_down = {down_state_none: (down_state_self, image_rects[1])}
    c_event_up = {down_state_self: (down_state_none, image_rects[0])}
    c_event_right_white_down = {
        down_state_none: (down_state_none, None),
        down_state_self: (down_state_self, None)}
    c_event_right_white_up = c_event_right_white_down.copy()
    c_event_right_black_down = c_event_right_white_down.copy()
    c_event_right_black_up = c_event_right_white_down.copy()
    if len(image_rects) > 2:
        c_event_down[down_state_white] = (down_state_self_white, image_rects[2])
        c_event_up[down_state_self_white] = (down_state_white, image_rects[0])
        c_event_right_white_down[down_state_none] = (down_state_white, None)
        c_event_right_white_down[down_state_self] = (
            down_state_self_white, image_rects[2])
        c_event_right_white_up[down_state_white] = (down_state_none, None)
        c_event_right_white_up[down_state_self_white] = (
            down_state_self, image_rects[1])
        c_event_right_black_down[down_state_white] = (
            down_state_white, None)
        c_event_right_black_down[down_state_self_white] = (
            down_state_self_white, None)
        c_event_right_black_up[down_state_white] = (
            down_state_white, None)
        c_event_right_black_up[down_state_self_white] = (
            down_state_self_white, None)
    if len(image_rects) > 3:
        c_event_down[down_state_black] = (down_state_self_black, image_rects[4])
        c_event_down[down_state_white_black] = (down_state_all, image_rects[5])
        c_event_up[down_state_self_black] = (down_state_black, image_rects[3])
        c_event_up[down_state_all] = (down_state_white_black, image_rects[3])
        c_event_right_white_down[down_state_black] = (
            down_state_white_black, None)
        c_event_right_white_down[down_state_self_black] = (
            down_state_all, image_rects[5])
        c_event_right_white_up[down_state_white_black] = (
            down_state_black, None)
        c_event_right_white_up[down_state_all] = (
            down_state_self_black, image_rects[4])
        c_event_right_black_down[down_state_none] = (
            down_state_black, image_rects[3])
        c_event_right_black_down[down_state_self] = (
            down_state_self_black, image_rects[4])
        c_event_right_black_down[down_state_white] = (
            down_state_white_black, image_rects[3])
        c_event_right_black_down[down_state_self_white] = (
            down_state_all, image_rects[5])
        c_event_right_black_up[down_state_black] = (
            down_state_none, image_rects[0])
        c_event_right_black_up[down_state_self_black] = (
            down_state_self, image_rects[1])
        c_event_right_black_up[down_state_white_black] = (
            down_state_white, image_rects[0])
        c_event_right_black_up[down_state_all] = (
            down_state_self_white, image_rects[2])


    class Key(object):
        """A key widget, maintains key state and draws the key's image

        Constructor arguments:
        ident - A unique key identifier. Any immutable type suitable as a key.
        posn - The location of the key on the display surface.
        key_left - Optional, the adjacent white key to the left. Changes in
            up and down state are propogated to that key.

        A key has an associated position and state. Related to state is the
        image drawn. State changes are managed with method calls, one method
        per event type. The up and down event methods are public. Other
        internal methods are for passing on state changes to the key_left
        key instance. The key will evaluate True if it is down, False
        otherwise.

        """

        def __init__(self, ident, posn, key_left = None):
            """Return a new Key instance

            The initial state is up, with all adjacent keys to the right also
            up.

            """
            if key_left is None:
                key_left = null_key
            rect = Rect(posn[0], posn[1], c_width, c_height)
            self.rect = rect
            self._state = c_down_state_initial
            self._source_rect = c_down_state_rect_initial
            self._ident = ident
            self._hash = hash(ident)
            self._notify_down = getattr(key_left, c_notify_down_method)
            self._notify_up = getattr(key_left, c_notify_up_method)
            self._key_left = key_left
            self._background_rect = Rect(rect.left, rect.bottom - 10,
                                         c_width, 10)
            c_updates.add(self)

        def down(self):
            """Signal that this key has been depressed (is down)"""
            
            self._state, source_rect = c_event_down[self._state]
            if source_rect is not None:
                self._source_rect = source_rect
                c_updates.add(self)
                self._notify_down()

        def up(self):
            """Signal that this key has been released (is up)"""
            
            self._state, source_rect = c_event_up[self._state]
            if source_rect is not None:
                self._source_rect = source_rect
                c_updates.add(self)
                self._notify_up()

        def _right_white_down(self):
            """Signal that the adjacent white key has been depressed

            This method is for internal propogation of events between
            key instances.

            """
            
            self._state, source_rect = c_event_right_white_down[self._state]
            if source_rect is not None:
                self._source_rect = source_rect
                c_updates.add(self)

        def _right_white_up(self):
            """Signal that the adjacent white key has been released

            This method is for internal propogation of events between
            key instances.

            """
            
            self._state, source_rect = c_event_right_white_up[self._state]
            if source_rect is not None:
                self._source_rect = source_rect
                c_updates.add(self)

        def _right_black_down(self):
            """Signal that the adjacent black key has been depressed

            This method is for internal propogation of events between
            key instances.

            """

            self._state, source_rect = c_event_right_black_down[self._state]
            if source_rect is not None:
                self._source_rect = source_rect
                c_updates.add(self)

        def _right_black_up(self):
            """Signal that the adjacent black key has been released

            This method is for internal propogation of events between
            key instances.

            """

            self._state, source_rect = c_event_right_black_up[self._state]
            if source_rect is not None:
                self._source_rect = source_rect
                c_updates.add(self)

        def __eq__(self, other):
            """True if same identifiers"""
            
            return self._ident == other._ident

        def __hash__(self):
            """Return the immutable hash value"""
            
            return self._hash

        def __str__(self):
            """Return the key's identifier and position as a string"""
            
            return ("<Key %s at (%d, %d)>" %
                    (self._ident, self.rect.top, self.rect.left))

        def __nonzero__(self):
            """True if the key is down"""
            
            return bool(self._state & c_down_state_self)

        def draw(self, surf, background, dirty_rects):
            """Redraw the key on the surface surf

            The background is redrawn. The altered rectangle is added
            to the dirty_rects list.

            """
            
            surf.blit(background, self._background_rect, self._background_rect)
            surf.blit(c_image_strip, self.rect, self._source_rect)
            dirty_rects.append(self.rect)

    return Key

def key_images():
    """Return a keyboard keys image strip and a mapping of image locations

    The return tuple is a surface and a dictionary of rects mapped to key
    type.

    This function encapsolates the constants relevant to the keyboard image
    file. There are five key types. One is the black key. The other four
    white keys are determined by the proximity of the black keys. The plain
    white key has no black key adjacent to it. A white-left and white-right
    key has a black key to the left or right of it respectively. A white-center
    key has a black key on both sides. A key may have up to six related
    images depending on the state of adjacent keys to its right.

    """
    strip_file = 'data/midikeys.png'
    white_key_width = 42
    white_key_height = 160
    black_key_width = 22
    black_key_height = 94
    strip = pygame.image.load(strip_file)
    names = [
        'black none', 'black self',
        'white none', 'white self', 'white self-white',
        'white-left none', 'white-left self', 'white-left black',
        'white-left self-black', 'white-left self-white', 'white-left all',
        'white-center none', 'white-center self',
        'white-center black', 'white-center self-black',
        'white-center self-white', 'white-center all',
        'white-right none', 'white-right self', 'white-right self-white']
    rects = {}
    for i in range(2):
        rects[names[i]] = Rect(i * white_key_width, 0,
                               black_key_width, black_key_height)
    for i in range(2, len(names)):
        rects[names[i]] = Rect(i * white_key_width, 0,
                               white_key_width, white_key_height)
    return strip, rects

class Keyboard(object):
    """Musical keyboard widget

    Constructor arguments:
    start_note: midi note value of the starting note on the keyboard.
    n_notes: number of notes (keys) on the keyboard.
    id: The widget identifier, and integer between 0 and 255 inclusive.

    A Keyboard instance draws the musical keyboard and maintains the state of
    all the keyboard keys. Individual keys can be in a down (depresssed) or
    up (released) state.

    """

    _image_strip, _rects = key_images()

    white_key_width, white_key_height = _rects['white none'].size
    black_key_width, black_key_height = _rects['black none'].size

    try:
        _updates = set()
    except:
        import sets
        _updates = sets.Set()

    # There are five key classes, representing key shape:
    # black key (BlackKey), plain white key (WhiteKey), white key to the left
    # of a black key (WhiteKeyLeft), white key between two black keys
    # (WhiteKeyCenter), and white key to the right of a black key
    # (WhiteKeyRight).
    BlackKey = key_class(_updates,
                         _image_strip,
                         [_rects['black none'], _rects['black self']],
                         False)
    WhiteKey = key_class(_updates,
                         _image_strip,
                         [_rects['white none'],
                          _rects['white self'],
                          _rects['white self-white']])
    WhiteKeyLeft = key_class(_updates,
                             _image_strip,
                             [_rects['white-left none'],
                              _rects['white-left self'],
                              _rects['white-left self-white'],
                              _rects['white-left black'],
                              _rects['white-left self-black'],
                              _rects['white-left all']])
    WhiteKeyCenter = key_class(_updates,
                               _image_strip,
                               [_rects['white-center none'],
                                _rects['white-center self'],
                                _rects['white-center self-white'],
                                _rects['white-center black'],
                                _rects['white-center self-black'],
                                _rects['white-center all']])
    WhiteKeyRight = key_class(_updates,
                              _image_strip,
                              [_rects['white-right none'],
                               _rects['white-right self'],
                               _rects['white-right self-white']])

    def __init__(self, start_note, n_notes, id):
        """Return a new Keyboard instance with n_note keys"""

        self._start_note = start_note
        self._end_note = start_note + n_notes - 1
        self._id = id
        self._set_rect()
        self._add_keys()
        self._map_regions()

    def _set_rect(self):
        """Calculate the image rectangle"""

        # The width of the keyboard is determined by the number of white keys
        # with allowance added if a black key is on either end.
        #
        n_white_notes = 0
        for note in range(self._start_note, self._end_note + 1):
            if is_white_key(note):
                n_white_notes += 1
        top = 0
        left = 0
        width = n_white_notes * self.white_key_width
        height = self.white_key_height
        if not is_white_key(self._start_note):
            shift = self.black_key_width // 2
            left += shift
            width += shift
        if not is_white_key(self._end_note):
            width += self.black_key_width // 2

        self._n_white_notes = n_white_notes
        self.rect = Rect(0, 0, width, height)
        
    def _add_keys(self):
        """Populate the keyboard with key instances

        Set the _keys attribute.
        
        """

        # Keys are entered in a dictionary keyed by note.
        #
        key_map = {}

        start_note = self._start_note
        end_note = self._end_note
        black_offset = self.black_key_width // 2
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
                        key = self.WhiteKey(ident, (x, 0), prev_white_key)
                    else:
                        key = self.WhiteKeyLeft(ident, (x, 0), prev_white_key)
                else:
                    if note == end_note or is_white_key(note + 1):
                        key = self.WhiteKeyRight(ident, (x, 0), prev_white_key)
                    else:
                        key = self.WhiteKeyCenter(ident, (x, 0), prev_white_key)
                is_prev_white = True
                x += self.white_key_width
                prev_white_key = key
            else:
                key = self.BlackKey(ident, (x - black_offset, 0), prev_white_key)
                is_prev_white = False
            key_map[note] = key

        self._keys = key_map

    def _map_regions(self):
        """Create an internal surface that is colored by key positions

        Sets the _regions attribute.
        
        """

        # The pixels of the _regions surface map a position to a
        # (note, id, velocity) tuple. It is a quick way to detemine which key
        # was clicked.
        #
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
        """Signal a key down event for the key at position posn"""
        
        message = self._regions.get_at(posn)
        self.key_down(message[0])
        return message[:3]

    def mouse_up(self, posn):
        """Signal a key up event for the key at position posn"""
        
        note, id, __ = self._regions.get_at(posn)
        self.key_up(note)
        return note

    def draw(self, surf, background, dirty_rects):
        """Redraw all altered keyboard keys"""
        
        changed_keys = self._updates
        while changed_keys:
            changed_keys.pop().draw(surf, background, dirty_rects)

    def key_down(self, note):
        """Signal a key down event for the Midi note"""
        
        self._keys[note].down()

    def key_up(self, note):
        """Signal a key up event for the Midi note"""
        
        self._keys[note].up()

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
