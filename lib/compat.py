# coding: ascii
"""Python 2.x/3.x compatibility tools"""

import sys

__all__ = ['geterror', 'long_', 'xrange_', 'ord_', 'unichr_',
           'unicode_', 'raw_input_', 'as_bytes', 'as_unicode',
           'bytes_', 'next_', 'imap_', 'PY_MAJOR_VERSION', 'PY_MINOR_VERSION']

PY_MAJOR_VERSION, PY_MINOR_VERSION = sys.version_info[0:2]


def geterror():
    return sys.exc_info()[1]

# Python 3
if PY_MAJOR_VERSION >= 3:
    long_ = int
    xrange_ = range
    from io import StringIO
    from io import BytesIO
    unichr_ = chr
    unicode_ = str
    bytes_ = bytes
    raw_input_ = input
    imap_ = map

    # Represent escaped bytes and strings in a portable way.
    #
    # as_bytes: Allow a Python 3.x string to represent a bytes object.
    #   e.g.: as_bytes("a\x01\b") == b"a\x01b" # Python 3.x
    #         as_bytes("a\x01\b") == "a\x01b"  # Python 2.x
    # as_unicode: Allow a Python "r" string to represent a unicode string.
    #   e.g.: as_unicode(r"Bo\u00F6tes") == u"Bo\u00F6tes" # Python 2.x
    #         as_unicode(r"Bo\u00F6tes") == "Bo\u00F6tes"  # Python 3.x
    def as_bytes(string):
        """ '<binary literal>' => b'<binary literal>' """
        return string.encode('latin-1', 'strict')

    def as_unicode(rstring):
        """ r'<Unicode literal>' => '<Unicode literal>' """
        return rstring.encode('ascii', 'strict').decode('unicode_escape',
                                                        'strict')

# Python 2
else:
    long_ = long
    xrange_ = xrange
    from cStringIO import StringIO
    BytesIO = StringIO
    unichr_ = unichr
    unicode_ = unicode
    bytes_ = str
    raw_input_ = raw_input
    from itertools import imap as imap_

    # Represent escaped bytes and strings in a portable way.
    #
    # as_bytes: Allow a Python 3.x string to represent a bytes object.
    #   e.g.: as_bytes("a\x01\b") == b"a\x01b" # Python 3.x
    #         as_bytes("a\x01\b") == "a\x01b"  # Python 2.x
    # as_unicode: Allow a Python "r" string to represent a unicode string.
    #   e.g.: as_unicode(r"Bo\u00F6tes") == u"Bo\u00F6tes" # Python 2.x
    #         as_unicode(r"Bo\u00F6tes") == "Bo\u00F6tes"  # Python 3.x
    def as_bytes(string):
        """ '<binary literal>' => '<binary literal>' """
        return string

    def as_unicode(rstring):
        """ r'<Unicode literal>' => u'<Unicode literal>' """
        return rstring.decode('unicode_escape', 'strict')


def get_BytesIO():
    return BytesIO


def get_StringIO():
    return StringIO


def ord_(o):
    try:
        return ord(o)
    except TypeError:
        return o

if sys.platform == 'win32':
    filesystem_errors = "replace"
elif PY_MAJOR_VERSION >= 3:
    filesystem_errors = "surrogateescape"
else:
    filesystem_errors = "strict"


def filesystem_encode(u):
    fsencoding = sys.getfilesystemencoding()
    if fsencoding.lower() in ['ascii', 'ANSI_X3.4-1968'] and sys.platform.startswith('linux'):
        # Don't believe Linux systems claiming ASCII-only filesystems. In
        # practice, arbitrary bytes are allowed, and most things expect UTF-8.
        fsencoding = 'utf-8'
    return u.encode(fsencoding, filesystem_errors)

# Include a next compatible function for Python versions < 2.6
if (PY_MAJOR_VERSION, PY_MINOR_VERSION) >= (2, 6):
    next_ = next
else:
    def next_(i, *args):
        try:
            return i.next()
        except StopIteration:
            if args:
                return args[0]
            raise
