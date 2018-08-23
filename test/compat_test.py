import sys

import unittest
from pygame import compat
encode_file_path = sys.modules['pygame.rwobject'].encode_file_path

class CompatModuleTest(unittest.TestCase):
    def test_as_unicode(self):
        r = r'Bo\u00F6tes'
        ords = [ord('B'), ord('o'), 0xF6, ord('t'), ord('e'), ord('s')]
        self.assertEqual(len(r), 11)
        u = compat.as_unicode(r)
        self.failUnless(isinstance(u, compat.unicode_))
        self.assertEqual([ord(c) for c in u], ords)

    def test_as_bytes(self):
        ords = [0, 1, 0x7F, 0x80, 0xC3, 0x20, 0xC3, 0xB6, 0xFF]
        s = ''.join([chr(i) for i in ords])
        self.assertEqual(len(s), len(ords))
        b = compat.as_bytes(s)
        self.failUnless(isinstance(b, compat.bytes_))
        self.assertEqual([compat.ord_(i) for i in b], ords)

    def test_ord_(self):
        self.failUnless(isinstance(compat.ord_(compat.bytes_(1)[0]), int))

    def test_bytes_(self):
        self.assertFalse(compat.bytes_ is compat.unicode_)
        self.failUnless(hasattr(compat.bytes_, 'capitalize'))
        self.assertFalse(hasattr(compat.bytes_, 'isdecimal'))

    def test_unicode_(self):
        self.failUnless(hasattr(compat.unicode_(), 'isdecimal'))

    def test_long_(self):
        self.failUnless(isinstance(int('99999999999999999999'), compat.long_))

    def test_geterror(self):
        msg = 'Success'
        try:
            raise TypeError(msg)
        except TypeError:
            e = compat.geterror()
            self.failUnless(isinstance(e, TypeError))
            self.assertEqual(str(e), msg)

    def test_xrange_(self):
        self.assertFalse(isinstance(compat.xrange_(2), list))

    def test_unichr_(self):
        ordval = 86
        c = compat.unichr_(ordval)
        self.failUnless(isinstance(c, compat.unicode_))
        self.assertEqual(ord(c), ordval)

    def test_get_BytesIO(self):
        BytesIO = compat.get_BytesIO()
        b1 = compat.as_bytes("\x00\xffabc")
        b2 = BytesIO(b1).read()
        self.failUnless(isinstance(b2, compat.bytes_))
        self.assertEqual(b2, b1)

    def test_get_StringIO(self):
        StringIO = compat.get_StringIO()
        b1 = "abcde"
        b2 = StringIO(b1).read()
        self.failUnless(isinstance(b2, str))
        self.assertEqual(b2, b1)

    def test_raw_input_(self):
        StringIO = compat.get_StringIO()
        msg = 'success'
        tmp = sys.stdin
        sys.stdin = StringIO(msg + '\n')
        try:
            s = compat.raw_input_()
            self.assertEqual(s, msg)
        finally:
            sys.stdin = tmp

    def test_filesystem_encode(self):
        upath = compat.as_unicode(r"ab\u212Acd")
        self.assertEqual(compat.filesystem_encode(upath),
                         encode_file_path(upath))

if __name__ == '__main__':
    unittest.main()
