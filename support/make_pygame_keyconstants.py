#!/usr/bin/env python

'''
Usage: python support/make_pygame_keyconstants.py

'''

__docformat__ = 'restructuredtext'
__version__ = '$Id$'

import sys

import SDL.constants

BEGIN_TAG = '#BEGIN GENERATED CONSTANTS'
END_TAG = '#END GENERATED CONSTANTS'

def make_pygame_keyconstants(source_file):
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
        
    for constant in dir(SDL.constants):
        if constant[:5] == 'SDLK_':
            line = '%-24s= SDL.constants.%s\n' % (constant[3:], constant)
            pre_lines.append(line)

    file = open(source_file, 'w')
    for line in pre_lines + post_lines:
        file.write(line)

if __name__ == '__main__':
    if len(sys.argv) != 1:
        print __doc__
    else:
        make_pygame_keyconstants('pygame/constants.py')
