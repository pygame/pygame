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

if is_pygame_pkg:
    from pygame.tests.test_utils import test_not_implemented, unittest
else:
    from test.test_utils import test_not_implemented, unittest
import pygame
rwobject = sys.modules['pygame.rwobject']
encode_file_path = rwobject.encode_file_path
from pygame.compat import bytes_, as_bytes, as_unicode


class RWopsEncodeFilePathTest(unittest.TestCase):
    def test_obj_None(self):
        self.assert_(encode_file_path(None) is None)
    
    def test_returns_bytes(self):
        path = as_unicode("Hello")
        self.assert_(isinstance(encode_file_path(path), bytes_))
    
    def test_obj_bytes(self):
        path = as_bytes("encyclop\xE6dia")
        result = encode_file_path(path, encoding='ascii', errors='strict')
        self.assert_(result == path)
        
    def test_encode_unicode(self):
        path = as_unicode(r"\u00DEe Olde Komputer Shoppe")
        epath = path.encode('utf-8')
        self.assert_(encode_file_path(path, encoding='utf-8') == epath)
        
    def test_error_fowarding(self):
        self.assertRaises(SyntaxError, encode_file_path)
        
    def test_etype(self):
        self.assertRaises(OverflowError, encode_file_path,
                          as_unicode(r"\u00DE"), OverflowError,
                          'ascii', 'strict')

    def test_errors(self):
        s = r"abc\u0109defg\u011Dh\u0125ij\u0135klmnoprs\u015Dtu\u016Dvz"
        path = as_unicode(s)
        epath = path.encode('ascii', 'ignore')
        result = encode_file_path(path, encoding='ascii', errors='ignore')
        self.assert_(result == epath)

    def test_default_etype(self):
        self.assertRaises(UnicodeError, encode_file_path,
                          as_unicode(r"\u00DE"),
                          encoding='ascii', errors='strict')

    def test_path_with_null_bytes(self):
        path = as_bytes("a\x00b\x00c")
        self.assertRaises(TypeError, encode_file_path, path)

if __name__ == '__main__':
    unittest.main()
