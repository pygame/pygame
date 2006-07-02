#!/usr/bin/env python

'''
Usage: make_constants.py source_file include_dir

for example:
    python support/make_constants.py SDL/constants.py /usr/include/SDL
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

import os
import os.path
import re
import sys

BEGIN_TAG = '#BEGIN GENERATED CONSTANTS'
END_TAG = '#END GENERATED CONSTANTS'

define_pattern = re.compile('#define[ \t]+([^ \t]+)[ \t]+((0x)?[0-9a-fA-F]+)')
def get_file_defines(include_file):
    defines = []
    for match in define_pattern.finditer(open(include_file).read(), re.M):
        num = match.groups()[1]
        try:
            if num[:2] == '0x':
                num = int(num, 16)
            else:
                num = int(num)
            name = match.groups()[0]
            defines.append('%s = 0x%08x\n' % (name, num))
        except ValueError:
            pass
    return defines 

def make_constants(source_file, include_dir):
    lines = pre_lines = []
    post_lines = []
    for line in open(source_file).readlines():
        if line[:len(END_TAG)] == END_TAG:
            lines = post_lines
        if lines != None:
            lines.append(line)
        if line[:len(BEGIN_TAG)] == BEGIN_TAG:
            lines = None

    if lines == pre_lines:
        raise Exception, '%s does not have begin tag' % source_file
    elif lines == None:
        raise Exception, '%s does not have end tag' % source_file
        
    for file in os.listdir(include_dir):
        if file[-2:] == '.h':
            if file[:10] == 'SDL_config' or \
               file in ('SDL_platform.h','SDL_opengl.h'):
                continue
            defines = get_file_defines(os.path.join(include_dir, file))
            if defines:
                pre_lines.append('\n#Constants from %s:\n' % file)
                pre_lines += defines

    file = open(source_file, 'w')
    for line in pre_lines + post_lines:
        file.write(line)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print __doc__
    else:
        make_constants(*sys.argv[1:])
