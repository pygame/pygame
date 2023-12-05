#    pygame - Python Game Library
#    Copyright (C) 2000-2003  Pete Shinners
#
#    This library is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Library General Public
#    License as published by the Free Software Foundation; either
#    version 2 of the License, or (at your option) any later version.
#
#    This library is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Library General Public License for more details.
#
#    You should have received a copy of the GNU Library General Public
#    License along with this library; if not, write to the Free
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
#    Pete Shinners
#    pete@shinners.org

"""Set of cursor resources available for use. These cursors come
in a sequence of values that are needed as the arguments for
pygame.mouse.set_cursor(). To dereference the sequence in place
and create the cursor in one step, call like this:
    pygame.mouse.set_cursor(*pygame.cursors.arrow).

Here is a list of available cursors:
    arrow, diamond, ball, broken_x, tri_left, tri_right

There is also a sample string cursor named 'thickarrow_strings'.
The compile() function can convert these string cursors into cursor 
byte data that can be used to create Cursor objects.

Alternately, you can also create Cursor objects using surfaces or 
cursors constants, such as pygame.SYSTEM_CURSOR_ARROW.
"""

import pygame

_cursor_id_table = {
    pygame.SYSTEM_CURSOR_ARROW: "SYSTEM_CURSOR_ARROW",
    pygame.SYSTEM_CURSOR_IBEAM: "SYSTEM_CURSOR_IBEAM",
    pygame.SYSTEM_CURSOR_WAIT: "SYSTEM_CURSOR_WAIT",
    pygame.SYSTEM_CURSOR_CROSSHAIR: "SYSTEM_CURSOR_CROSSHAIR",
    pygame.SYSTEM_CURSOR_WAITARROW: "SYSTEM_CURSOR_WAITARROW",
    pygame.SYSTEM_CURSOR_SIZENWSE: "SYSTEM_CURSOR_SIZENWSE",
    pygame.SYSTEM_CURSOR_SIZENESW: "SYSTEM_CURSOR_SIZENESW",
    pygame.SYSTEM_CURSOR_SIZEWE: "SYSTEM_CURSOR_SIZEWE",
    pygame.SYSTEM_CURSOR_SIZENS: "SYSTEM_CURSOR_SIZENS",
    pygame.SYSTEM_CURSOR_SIZEALL: "SYSTEM_CURSOR_SIZEALL",
    pygame.SYSTEM_CURSOR_NO: "SYSTEM_CURSOR_NO",
    pygame.SYSTEM_CURSOR_HAND: "SYSTEM_CURSOR_HAND",
}


class Cursor:
    """Base class for representing cursors."""

    def __init__(self):
        self.data = None
        raise NotImplementedError("Base Cursor class should not be instantiated directly.")

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.data == other.data

    def __ne__(self, other):
        return not self.__eq__(other)

    def __copy__(self):
        return self.__class__(self.data)
    
    def __hash__(self):
        return hash(self.data)

    def __repr__(self):
        raise NotImplementedError("__repr__ must be used with subclasses.")


class SystemCursor(Cursor):
    """Cursor representing a system cursor."""

    def __init__(self, cursor_id):
        super().__init__()
        if cursor_id not in _cursor_id_table:
            raise ValueError("Invalid system cursor id.")
        self.data = (cursor_id,)

    def __repr__(self):
        id_string = _cursor_id_table.get(self.data[0], "constant lookup error")
        return f"<SystemCursor(id: {id_string})>"


class BitmapCursor(Cursor):
    """Cursor representing a bitmap cursor."""

    def __init__(self, size, hotspot, xormasks, andmasks):
        super().__init__()
        self.data = (size, hotspot, xormasks, andmasks)

    def __repr__(self):
        return f"<BitmapCursor(size: {self.data[0]}, hotspot: {self.data[1]})>"


class ColorCursor(Cursor):
    """Cursor representing a color cursor."""

    def __init__(self, hotspot, surface):
        super().__init__()
        if not isinstance(surface, pygame.Surface):
            raise TypeError("Surface argument must be a pygame.Surface.")
        self.data = (hotspot, surface)

    def __repr__(self):
        return f"<ColorCursor(hotspot: {self.data[0]}, surface: {repr(self.data[1])})>"


# Python side of the set_cursor function: C side in mouse.c
def set_cursor(*args):
    """set_cursor(pygame.cursors.Cursor OR args for a pygame.cursors.Cursor) -> None
    Set the mouse cursor to a new cursor"""

    if len(args) == 1 and isinstance(args[0], Cursor):
        # If a Cursor instance is passed, use it directly
        cursor = args[0]
    else:
        # Determine which subclass to instantiate based on args
        if len(args) == 1 and args[0] in _cursor_id_table:
            cursor = SystemCursor(args[0])
        elif len(args) == 2 and isinstance(args[1], pygame.Surface):
            cursor = ColorCursor(*args)
        elif len(args) == 4 and all(isinstance(arg, tuple) and len(arg) == 2 for arg in args[:2]):
            cursor = BitmapCursor(*args)
        else:
            raise TypeError("Arguments do not match any cursor specification.")

    pygame.mouse.set_cursor(cursor)


pygame.mouse.set_cursor = set_cursor


# Python side of the get_cursor function: C side in mouse.c
def get_cursor():
    """get_cursor() -> pygame.cursors.Cursor
    Get the current mouse cursor"""

    cursor_data = pygame.mouse.get_cursor()

    # Determine the type of cursor based on the structure of cursor_data
    if isinstance(cursor_data, tuple):
        if len(cursor_data) == 1 and cursor_data[0] in _cursor_id_table:
            return SystemCursor(cursor_data[0])
        elif len(cursor_data) == 2 and isinstance(cursor_data[1], pygame.Surface):
            return ColorCursor(*cursor_data)
        elif len(cursor_data) == 4 and all(isinstance(arg, tuple) and len(arg) == 2 for arg in cursor_data[:2]):
            return BitmapCursor(*cursor_data)
        else:
            raise TypeError("Unknown cursor format returned by pygame.mouse.get_cursor().")

    # If cursor_data doesn't match any known format, raise an error
    raise TypeError("Unknown cursor format returned by pygame.mouse.get_cursor().")


pygame.mouse.get_cursor = get_cursor


arrow = BitmapCursor(
    size=(16, 16),
    hotspot=(0, 0),
    xormasks=(
        0x00, 0x00, 0x40, 0x00, 0x60, 0x00, 0x70, 0x00, 0x78, 0x00,
        0x7C, 0x00, 0x7E, 0x00, 0x7F, 0x00, 0x7F, 0x80, 0x7C, 0x00,
        0x6C, 0x00, 0x46, 0x00, 0x06, 0x00, 0x03, 0x00, 0x03, 0x00,
        0x00, 0x00,
    ),
    andmasks=(
        0x40, 0x00, 0xE0, 0x00, 0xF0, 0x00, 0xF8, 0x00, 0xFC, 0x00,
        0xFE, 0x00, 0xFF, 0x00, 0xFF, 0x80, 0xFF, 0xC0, 0xFF, 0x80,
        0xFE, 0x00, 0xEF, 0x00, 0x4F, 0x00, 0x07, 0x80, 0x07, 0x80,
        0x03, 0x00,
    ),
)


diamond = BitmapCursor(
    size=(16, 16),
    hotspot=(7, 7),
    xormasks=(
        0, 0, 1, 0, 3, 128, 7, 192, 14, 224,
        28, 112, 56, 56, 112, 28, 56, 56, 28, 112,
        14, 224, 7, 192, 3, 128, 1, 0, 0, 0, 0, 0
    ),
    andmasks=(
        1, 0, 3, 128, 7, 192, 15, 224, 31, 240, 62, 248,
        124, 124, 248, 62, 124, 124, 62, 248, 31, 240,
        15, 224, 7, 192, 3, 128, 1, 0, 0, 0
    ),
)

ball = BitmapCursor(
    size=(16, 16),
    hotspot=(7, 7),
    xormasks=(
        0, 0, 3, 192, 15, 240, 24, 248, 51, 252,
        55, 252, 127, 254, 127, 254, 127, 254, 127, 254,
        63, 252, 63, 252, 31, 248, 15, 240, 3, 192, 0, 0
    ),
    andmasks=(
        3, 192, 15, 240, 31, 248, 63, 252, 127, 254,
        127, 254, 255, 255, 255, 255, 255, 255, 255, 255,
        127, 254, 127, 254, 63, 252, 31, 248, 15, 240, 3, 192
    ),
)


broken_x = BitmapCursor(
    size=(16, 16),
    hotspot=(7, 7),
    xormasks=(
        0, 0, 96, 6, 112, 14, 56, 28, 28, 56,
        12, 48, 0, 0, 0, 0, 0, 0, 0, 0,
        12, 48, 28, 56, 56, 28, 112, 14, 96, 6, 0, 0
    ),
    andmasks=(
        224, 7, 240, 15, 248, 31, 124, 62, 62, 124,
        30, 120, 14, 112, 0, 0, 0, 0, 14, 112,
        30, 120, 62, 124, 124, 62, 248, 31, 240, 15, 224, 7
    ),
)


tri_left = BitmapCursor(
    size=(16, 16),
    hotspot=(1, 1),
    xormasks=(
        0, 0, 96, 0, 120, 0, 62, 0, 63, 128,
        31, 224, 31, 248, 15, 254, 15, 254, 7, 128,
        7, 128, 3, 128, 3, 128, 1, 128, 1, 128, 0, 0
    ),
    andmasks=(
        224, 0, 248, 0, 254, 0, 127, 128, 127, 224,
        63, 248, 63, 254, 31, 255, 31, 255, 15, 254,
        15, 192, 7, 192, 7, 192, 3, 192, 3, 192, 1, 128
    ),
)

tri_right = BitmapCursor(
    size=(16, 16),
    hotspot=(14, 1),
    xormasks=(
        0, 0, 0, 6, 0, 30, 0, 124, 1, 252,
        7, 248, 31, 248, 127, 240, 127, 240, 1, 224,
        1, 224, 1, 192, 1, 192, 1, 128, 1, 128, 0, 0
    ),
    andmasks=(
        0, 7, 0, 31, 0, 127, 1, 254, 7, 254,
        31, 252, 127, 252, 255, 248, 255, 248, 127, 240,
        3, 240, 3, 224, 3, 224, 3, 192, 3, 192, 1, 128
    ),
)


# Here is an example string resource cursor. To use this:
#    curs, mask = pygame.cursors.compile_cursor(pygame.cursors.thickarrow_strings, 'X', '.')
#    pygame.mouse.set_cursor((24, 24), (0, 0), curs, mask)
# Be warned, though, that cursors created from compiled strings do not support colors.

# sized 24x24
thickarrow_cursor = (
    "XX                      ",
    "XXX                     ",
    "XXXX                    ",
    "XX.XX                   ",
    "XX..XX                  ",
    "XX...XX                 ",
    "XX....XX                ",
    "XX.....XX               ",
    "XX......XX              ",
    "XX.......XX             ",
    "XX........XX            ",
    "XX........XXX           ",
    "XX......XXXXX           ",
    "XX.XXX..XX              ",
    "XXXX XX..XX             ",
    "XX   XX..XX             ",
    "     XX..XX             ",
    "      XX..XX            ",
    "      XX..XX            ",
    "       XXXX             ",
    "       XX               ",
    "                        ",
    "                        ",
    "                        ",
)

# sized 24x16
double_arrow_horizontal_cursor = (
    "     X      X           ",
    "    XX      XX          ",
    "   X.X      X.X         ",
    "  X..X      X..X        ",
    " X...XXXXXXXX...X       ",
    "X................X      ",
    " X...XXXXXXXX...X       ",
    "  X..X      X..X        ",
    "   X.X      X.X         ",
    "    XX      XX          ",
    "     X      X           ",
    "                        ",
    "                        ",
    "                        ",
    "                        ",
    "                        ",
)

# sized 16x24
double_arrow_vertical_cursor = (
    "     X          ",
    "    X.X         ",
    "   X...X        ",
    "  X.....X       ",
    " X.......X      ",
    "XXXXX.XXXXX     ",
    "    X.X         ",
    "    X.X         ",
    "    X.X         ",
    "    X.X         ",
    "    X.X         ",
    "    X.X         ",
    "    X.X         ",
    "XXXXX.XXXXX     ",
    " X.......X      ",
    "  X.....X       ",
    "   X...X        ",
    "    X.X         ",
    "     X          ",
    "                ",
    "                ",
    "                ",
    "                ",
    "                ",
)

# sized 24x16
double_arrow_diagonal_cursor = (
    "XXXXXXXX                ",
    "X.....X                 ",
    "X....X                  ",
    "X...X                   ",
    "X..X.X                  ",
    "X.X X.X                 ",
    "XX   X.X    X           ",
    "X     X.X  XX           ",
    "       X.XX.X           ",
    "        X...X           ",
    "        X...X           ",
    "       X....X           ",
    "      X.....X           ",
    "     XXXXXXXX           ",
    "                        ",
    "                        ",
)

# sized 8x16
textmarker_cursor = (
    "ooo ooo ",
    "   o    ",
    "   o    ",
    "   o    ",
    "   o    ",
    "   o    ",
    "   o    ",
    "   o    ",
    "   o    ",
    "   o    ",
    "   o    ",
    "ooo ooo ",
    "        ",
    "        ",
    "        ",
    "        ",
)


def compile(strings, black="X", white=".", xor="o"):
    """pygame.cursors.compile(strings, black, white, xor) -> data, mask
    compile cursor strings into cursor data

    This takes a set of strings with equal length and computes
    the binary data for that cursor. The string widths must be
    divisible by 8.

    The black and white arguments are single letter strings that
    tells which characters will represent black pixels, and which
    characters represent white pixels. All other characters are
    considered clear.

    Some systems allow you to set a special toggle color for the
    system color, this is also called the xor color. If the system
    does not support xor cursors, that color will simply be black.

    This returns a tuple containing the cursor data and cursor mask
    data. Both these arguments are used when setting a cursor with
    pygame.mouse.set_cursor().
    """
    # first check for consistent lengths
    size = len(strings[0]), len(strings)
    if size[0] % 8 or size[1] % 8:
        raise ValueError(f"cursor string sizes must be divisible by 8 {size}")

    for s in strings[1:]:
        if len(s) != size[0]:
            raise ValueError("Cursor strings are inconsistent lengths")

    # create the data arrays.
    # this could stand a little optimizing
    mask_data = []
    fill_data = []
    mask_item = fill_item = 0
    step = 8
    for s in strings:
        for c in s:
            mask_item = mask_item << 1
            fill_item = fill_item << 1
            step = step - 1
            if c == black:
                mask_item = mask_item | 1
                fill_item = fill_item | 1
            elif c == white:
                mask_item = mask_item | 1
            elif c == xor:
                fill_item = fill_item | 1

            if not step:
                mask_data.append(mask_item)
                fill_data.append(fill_item)
                mask_item = fill_item = 0
                step = 8

    return tuple(fill_data), tuple(mask_data)


def load_xbm(curs, mask):
    """pygame.cursors.load_xbm(cursorfile, maskfile) -> cursor_args
    Reads a pair of XBM files into set_cursor arguments.

    Arguments can either be filenames or filelike objects
    with the readlines method. Not largely tested, but
    should work with typical XBM files.
    """

    def bitswap(num):
        """Swap the bit order of an 8-bit number."""
        val = 0
        for x in range(8):
            b = num & (1 << x) != 0
            val = val << 1 | b
        return val

    def read_xbm_data(source):
        """Read XBM data from a file or file-like object."""
        if hasattr(source, "readlines"):
            return source.readlines()
        else:
            with open(source, encoding="ascii") as f:
                return f.readlines()

    def extract_info(lines):
        """Extracts width, height, and hotspot info from XBM lines."""
        for i, line in enumerate(lines):
            if line.startswith("#define"):
                lines = lines[i:]
                break

        width = int(lines[0].split()[-1])
        height = int(lines[1].split()[-1])
        if lines[2].startswith("#define"):
            hotx = int(lines[2].split()[-1])
            hoty = int(lines[3].split()[-1])
        else:
            hotx = hoty = 0

        return width, height, hotx, hoty, lines

    def parse_data(lines):
        """Parses cursor or mask data from XBM lines."""
        possible_starts = ("static char", "static unsigned char")
        for i, line in enumerate(lines):
            if line.startswith(possible_starts):
                break
        data = " ".join(lines[i + 1 :]).replace("};", "").replace(",", " ")
        return tuple(bitswap(int(x, 16)) for x in data.split())

    curs_lines = read_xbm_data(curs)
    mask_lines = read_xbm_data(mask)

    width, height, hotx, hoty, curs_lines = extract_info(curs_lines)
    _, _, _, _, mask_lines = extract_info(mask_lines)

    cursor_data = parse_data(curs_lines)
    mask_data = parse_data(mask_lines)

    return (width, height), (hotx, hoty), cursor_data, mask_data
