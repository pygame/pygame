"""Python 2.x/3.x compatibility tools"""

import sys

__all__ = ['geterror', 'long_', 'xrange_', 'ord_', 'unichr_',
           'unicode_', 'raw_input_', 'as_bytes', 'as_unicode']

def geterror ():
    return sys.exc_info()[1]

try:
    long_ = long
except NameError:
    long_ = int

try:
    xrange_ = xrange
except NameError:
    xrange_ = range

def get_BytesIO():
    try:
        from cStringIO import StringIO as BytesIO
    except ImportError:
        from io import BytesIO
    return BytesIO

def get_StringIO():
    try:
        from cStringIO import StringIO
    except ImportError:
        from io import StringIO
    return StringIO

def ord_(o):
    try:
        return ord(o)
    except TypeError:
        return o

try:
    unichr_ = unichr
except NameError:
    unichr_ = chr

try:
    unicode_ = unicode
except NameError:
    unicode_ = str

try:
    bytes_ = bytes
except NameError:
    bytes_ = str

try:
    raw_input_ = raw_input
except NameError:
    raw_input_ = input

if sys.platform == 'win32':
    filesystem_errors = "replace"
elif sys.version_info >= (3, 0, 0):
    filesystem_errors = "surrogateescape"
else:
    filesystem_errors = "strict"
    
def filesystem_encode(u):
    fsencoding = sys.getfilesystemencoding()
    if (fsencoding.lower() == 'ascii') and sys.platform.startswith('linux'):
        # Don't believe Linux systems claiming ASCII-only filesystems. In
        # practice, arbitrary bytes are allowed, and most things expect UTF-8.
        fsencoding = 'utf-8'
    return u.encode(fsencoding, filesystem_errors)

# Represent escaped bytes and strings in a portable way.
#
# as_bytes: Allow a Python 3.x string to represent a bytes object.
#   e.g.: as_bytes("a\x01\b") == b"a\x01b" # Python 3.x
#         as_bytes("a\x01\b") == "a\x01b"  # Python 2.x
# as_unicode: Allow a Python "r" string to represent a unicode string.
#   e.g.: as_unicode(r"Bo\u00F6tes") == u"Bo\u00F6tes" # Python 2.x
#         as_unicode(r"Bo\u00F6tes") == "Bo\u00F6tes"  # Python 3.x
try:
    unicode
    def as_bytes(string):
        """ '<binary literal>' => '<binary literal>' """
        return string
        
    def as_unicode(rstring):
        """ r'<Unicode literal>' => u'<Unicode literal>' """
        return rstring.decode('unicode_escape', 'strict')
except NameError:
    def as_bytes(string):
        """ '<binary literal>' => b'<binary literal>' """
        return string.encode('latin-1', 'strict')
        
    def as_unicode(rstring):
        """ r'<Unicode literal>' => '<Unicode literal>' """
        return rstring.encode('ascii', 'strict').decode('unicode_escape',
                                                        'stict')
# Include a next compatible function for Python versions < 2.6
try:
    next_ = next
except NameError:
    def next_(i, *args):
        try:
            return i.next()
        except StopIteration:
            if args:
                return args[0]
            raise

# itertools.imap is missing in 3.x
try:
    import itertools.imap as imap_
except ImportError:
    imap_ = map
