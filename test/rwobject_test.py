import sys
import unittest

from pygame import encode_string, encode_file_path
from pygame.compat import bytes_, as_bytes, as_unicode


class RWopsEncodeStringTest(unittest.TestCase):
    global getrefcount

    def test_obj_None(self):
        encoded_string = encode_string(None)

        self.assertIsNone(encoded_string)

    def test_returns_bytes(self):
        u = as_unicode(r"Hello")
        encoded_string = encode_string(u)

        self.assertIsInstance(encoded_string, bytes_)

    def test_obj_bytes(self):
        b = as_bytes("encyclop\xE6dia")
        encoded_string = encode_string(b, 'ascii', 'strict')

        self.assertIs(encoded_string, b)

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
        encoded_string = encode_string(u, 'ascii', 'strict')

        self.assertIsNone(encoded_string)

    def test_check_defaults(self):
        u = as_unicode(r"a\u01F7b")
        b = u.encode("unicode_escape", "backslashreplace")
        encoded_string = encode_string(u)

        self.assertEqual(encoded_string, b)

    def test_etype(self):
        u = as_unicode(r"a\x80b")
        self.assertRaises(SyntaxError, encode_string,
                          u, 'ascii', 'strict', SyntaxError)

    def test_string_with_null_bytes(self):
        b = as_bytes("a\x00b\x00c")
        encoded_string = encode_string(b, etype=SyntaxError)
        encoded_decode_string = encode_string(b.decode(), 'ascii', 'strict')

        self.assertIs(encoded_string, b)
        self.assertEqual(encoded_decode_string, b)

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
        encoded_file_path = encode_file_path(u)

        self.assertIsInstance(encoded_file_path, bytes_)

    def test_error_fowarding(self):
        self.assertRaises(SyntaxError, encode_file_path)

    def test_path_with_null_bytes(self):
        b = as_bytes("a\x00b\x00c")
        encoded_file_path = encode_file_path(b)

        self.assertIsNone(encoded_file_path)

    def test_etype(self):
        b = as_bytes("a\x00b\x00c")
        self.assertRaises(TypeError, encode_file_path, b, TypeError)

if __name__ == '__main__':
    unittest.main()
