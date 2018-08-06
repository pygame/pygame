# -*- encoding: utf-8 -*-

# module setup_win_common.py

"""A module for reading the information common to all Windows setups.

Exports read and get_definitions.
"""

import os
PATH = os.path.join('buildconfig', 'Setup_Win_Common.in')

class Definition(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value

def read():
    """Return the contents of the Windows Common Setup as a string"""

    setup_in = open(PATH)
    try:
        return setup_in.read()
    finally:
        setup_in.close()

def get_definitions():
    """Return a list of definitions in the Windows Common Setup

    Each macro definition object has a 'name' and 'value' attribute.
    """
    import re

    setup_in = open(PATH)
    try:
        deps = []
        match = re.compile(r'([a-zA-Z0-9_]+) += +(.+)$').match
        for line in setup_in:
            m = match(line)
            if m is not None:
                deps.append(Definition(m.group(1), m.group(2)))
        return deps
    finally:
        setup_in.close()

__all__= ['read', 'get_dependencies']
