##    pygame - Python Game Library
##    Copyright (C) 2000  Pete Shinners
##
##    This library is free software; you can redistribute it and/or
##    modify it under the terms of the GNU Library General Public
##    License as published by the Free Software Foundation; either
##    version 2 of the License, or (at your option) any later version.
##
##    This library is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
##    Library General Public License for more details.
##
##    You should have received a copy of the GNU Library General Public
##    License along with this library; if not, write to the Free
##    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
##
##    Pete Shinners
##    pete@shinners.org

"""Set of cursor resources available for use. These cursors come
in a sequence of values that are needed as the arguments for
pygame.mouse.set_cursor(). to dereference the sequence in place
and create the cursor in one step, call like this;
pygame.mouse.set_cursor(*pygame.cursors.arrow).

Here is a list of available cursors; arrow, diamond, ball,
        broken_x, tri_left, tri_right

Also includes a compile() function which will compile a
set of strings representing bit data into sequence of cursor
data numbers. The function read_xbm() will read a pair of XBM
cursor files."""

#default pygame black arrow
arrow = ((16, 16), (0, 0),
    (0x00,0x00,0x40,0x00,0x60,0x00,0x70,0x00,0x78,0x00,0x7C,0x00,0x7E,0x00,0x7F,0x00,
     0x7F,0x80,0x7C,0x00,0x6C,0x00,0x46,0x00,0x06,0x00,0x03,0x00,0x03,0x00,0x00,0x00),
    (0x40,0x00,0xE0,0x00,0xF0,0x00,0xF8,0x00,0xFC,0x00,0xFE,0x00,0xFF,0x00,0xFF,0x80,
     0xFF,0xC0,0xFF,0x80,0xFE,0x00,0xEF,0x00,0x4F,0x00,0x07,0x80,0x07,0x80,0x03,0x00))

diamond = ((16, 16), (7, 7),
    (0, 0, 1, 0, 3, 128, 7, 192, 14, 224, 28, 112, 56, 56, 112, 28, 56,
     56, 28, 112, 14, 224, 7, 192, 3, 128, 1, 0, 0, 0, 0, 0),
    (1, 0, 3, 128, 7, 192, 15, 224, 31, 240, 62, 248, 124, 124, 248, 62,
     124, 124, 62, 248, 31, 240, 15, 224, 7, 192, 3, 128, 1, 0, 0, 0))

ball = ((16, 16), (7, 7),
    (0, 0, 3, 192, 15, 240, 24, 248, 51, 252, 55, 252, 127, 254, 127, 254,
     127, 254, 127, 254, 63, 252, 63, 252, 31, 248, 15, 240, 3, 192, 0, 0),
    (3, 192, 15, 240, 31, 248, 63, 252, 127, 254, 127, 254, 255, 255, 255,
     255, 255, 255, 255, 255, 127, 254, 127, 254, 63, 252, 31, 248, 15, 240,
     3, 192))

broken_x = ((16, 16), (7, 7),
    (0, 0, 96, 6, 112, 14, 56, 28, 28, 56, 12, 48, 0, 0, 0, 0, 0, 0, 0, 0,
     12, 48, 28, 56, 56, 28, 112, 14, 96, 6, 0, 0),
    (224, 7, 240, 15, 248, 31, 124, 62, 62, 124, 30, 120, 14, 112, 0, 0, 0,
     0, 14, 112, 30, 120, 62, 124, 124, 62, 248, 31, 240, 15, 224, 7))


tri_left = ((16, 16), (1, 1),
    (0, 0, 96, 0, 120, 0, 62, 0, 63, 128, 31, 224, 31, 248, 15, 254, 15, 254,
     7, 128, 7, 128, 3, 128, 3, 128, 1, 128, 1, 128, 0, 0),
    (224, 0, 248, 0, 254, 0, 127, 128, 127, 224, 63, 248, 63, 254, 31, 255,
     31, 255, 15, 254, 15, 192, 7, 192, 7, 192, 3, 192, 3, 192, 1, 128))

tri_right = ((16, 16), (14, 1),
    (0, 0, 0, 6, 0, 30, 0, 124, 1, 252, 7, 248, 31, 248, 127, 240, 127, 240,
     1, 224, 1, 224, 1, 192, 1, 192, 1, 128, 1, 128, 0, 0),
    (0, 7, 0, 31, 0, 127, 1, 254, 7, 254, 31, 252, 127, 252, 255, 248, 255,
     248, 127, 240, 3, 240, 3, 224, 3, 224, 3, 192, 3, 192, 1, 128))



#here is an example string resource cursor. to use this;
#  curs, mask = pygame.cursors.compile_cursor(pygame.cursors.arrow_strings)
#  pygame.mouse.set_cursor((16, 16), (0, 0), curs, mask)

thickarrow_strings = (               #sized 24x24
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


def compile(strings, black, white):
    """compile(strings, black, white) -> data
compile cursor strings into cursor data

This takes a set of strings with equal length and computes
the binary data for that cursor. The string widths must be
divisible by 6. The data returned are suitable for passing
to pygame.mouse.set_cursor()
"""
    
    #first check for consistent lengths
    size = len(strings[0]), len(strings)
    if size[0] % 8 or size[1] % 8:
        raise ValueError, "cursor string sizes must be divisible by 8 "+`size`
    for s in strings[1:]:
        if len(s) != size[0]:
            raise ValueError, "Cursor strings are inconsistent lengths"

    #create the data arrays.
    #this could stand a little optimizing
    maskdata = []
    filldata = []
    maskitem = fillitem = 0
    step = 8
    for c in [x for s in strings for x in s]:
        maskitem <<= 1
        fillitem <<= 1
        step -= 1
        if c == black:
            maskitem |= 1
        elif c == white:
            maskitem |= 1
            fillitem |= 1
        if not step:
            maskdata.append(maskitem)
            filldata.append(fillitem)
            maskitem = fillitem = 0
            step = 8
    return tuple(filldata), tuple(maskdata)



def read_xbm(curs, mask):
    """readxbm(cursorfile, maskfile) -> cursor_args
reads a pair of XBM files into set_cursor arguments

Arguments can either be strings or filelike objects
with the readlines method. Not largely tested, but
should work with typical XBM files.
(for the paranoid) Note there is a security
risk in here. Any malicious xbm files with evil python
code in the data could do damage.
"""
    def bitswap(num):
        val = 0
        for b in [num&(1<<x) != 0 for x in range(8)]:
            val = val<<1 | b
        return val
    if type(curs) is type(''): curs = open(curs)
    if type(mask) is type(''): mask = open(mask)
    curs = curs.readlines()
    mask = mask.readlines()
    info = tuple([int(curs[x].split()[-1]) for x in range(4)])
    data = ' '.join(curs[5:]).replace('};', '').replace(',', '')
    cursdata = tuple([bitswap(int(x, 16)) for x in data.split()])
    data = ' '.join(mask[5:]).replace('};', '').replace(',', '')
    maskdata = tuple([bitswap(int(x, 16)) for x in data.split()])
    return info[:2], info[2:], cursdata, maskdata
