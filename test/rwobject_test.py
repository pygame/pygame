import sys
if __name__ == '__main__':
    import os
    pkg_dir = os.path.split(os.path.abspath(__file__))[0]
    parent_dir, pkg_name = os.path.split(pkg_dir)
    is_pygame_pkg = (pkg_name == 'tests' and
                     os.path.split(parent_dir)[1] == 'pygame')
    if not is_pygame_pkg:
        sys.path.insert(0, parent_dir)
else:
    is_pygame_pkg = __name__.startswith('pygame.tests.')

import unittest

from pygame import encode_string, encode_file_path
from pygame.compat import bytes_, as_bytes, as_unicode


class RWopsEncodeStringTest(unittest.TestCase):
    global getrefcount

    def test_obj_None(self):
        self.assert_(encode_string(None) is None)
    
    def test_returns_bytes(self):
        u = as_unicode(r"Hello")
        self.assert_(isinstance(encode_string(u), bytes_))
    
    def test_obj_bytes(self):
        b = as_bytes("encyclop\xE6dia")
        self.assert_(encode_string(b, 'ascii', 'strict') is b)
        
    def test_encode_unicode(self):
        u = as_unicode(r"\u00DEe Olde Komp\u00FCter Shoppe")
        b = u.encode('utf-8')
        self.assertEqual(encode_string(u, 'utf-8'), b)
        
    def test_error_fowarding(self):
        self.assertRaises(SyntaxError, encode_string)
        
    def test_errors(self):
        s = r"abc\u0109defg\u011Dh\u0125ij\u0135klmnoprs\u015Dtu\u016Dvz"
        u = as_unicode(s)
        b = u.encode('ascii', 'ignore')
        self.assertEqual(encode_string(u, 'ascii', 'ignore'), b)

    def test_encoding_error(self):
        u = as_unicode(r"a\x80b")
        self.assert_(encode_string(u, 'ascii', 'strict') is None)

    def test_check_defaults(self):
        u = as_unicode(r"a\u01F7b")
        b = u.encode("unicode_escape", "backslashreplace") 
        self.assert_(encode_string(u) == b)

    def test_etype(self):
        u = as_unicode(r"a\x80b")
        self.assertRaises(SyntaxError, encode_string,
                          u, 'ascii', 'strict', SyntaxError)

    def test_string_with_null_bytes(self):
        b = as_bytes("a\x00b\x00c")
        self.assert_(encode_string(b, etype=SyntaxError) is b)
        u = b.decode()
        self.assert_(encode_string(u, 'ascii', 'strict') == b)

    try:
        from sys import getrefcount as _g
        getrefcount = _g                   # This nonsense is for Python 3.x
    except ImportError:
        pass
    else:
        def test_refcount(self):
            bpath = as_bytes(" This is a string that is not cached.")[1:]
            upath = bpath.decode('ascii')
            before = getrefcount(bpath)
            bpath = encode_string(bpath)
            self.assertEqual(getrefcount(bpath), before)
            bpath = encode_string(upath)
            self.assertEqual(getrefcount(bpath), before)
            
    def test_smp(self):
        utf_8 = as_bytes("a\xF0\x93\x82\xA7b")
        u = as_unicode(r"a\U000130A7b")
        b = encode_string(u, 'utf-8', 'strict', AssertionError)
        self.assertEqual(b, utf_8)
        #  For Python 3.1, surrogate pair handling depends on whether the
        #  interpreter was built with UCS-2 or USC-4 unicode strings.
        ##u = as_unicode(r"a\uD80C\uDCA7b")
        ##b = encode_string(u, 'utf-8', 'strict', AssertionError)
        ##self.assertEqual(b, utf_8)

class RWopsEncodeFilePathTest(unittest.TestCase):
    # Most tests can be skipped since RWopsEncodeFilePath wraps
    # RWopsEncodeString
    def test_encoding(self):
        u = as_unicode(r"Hello")
        self.assert_(isinstance(encode_file_path(u), bytes_))
    
    def test_error_fowarding(self):
        self.assertRaises(SyntaxError, encode_file_path)

    def test_path_with_null_bytes(self):
        b = as_bytes("a\x00b\x00c")
        self.assert_(encode_file_path(b) is None)

    def test_etype(self):
        b = as_bytes("a\x00b\x00c")
        self.assertRaises(TypeError, encode_file_path, b, TypeError)
                                   
if __name__ == '__main__':
    unittest.main()
