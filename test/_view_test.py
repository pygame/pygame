import sys
import re
import weakref
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
from pygame._view import View

class ViewTest(unittest.TestCase):
    def test___array_struct___property(self):
        c = []
        v = View(c)
        self.assert_(v.__array_struct__ is c)

    def test_parent_property(self):
        c = []
        p = []
        v = View(c, p)
        self.assert_(v.parent is p)
        
    def test_destructor(self):
        def d(capsule, parent):
            success.append(capsule is c)
            success.append(parent is None)
        
        success = []
        c = []
        view_id = id(View(c, destructor=d))
        self.assert_(success)
        self.assert_(success[0])
        self.assert_(success[1])

    def test_weakref(self):
        v = View([])
        weak_v = weakref.ref(v)
        self.assert_(weak_v() is v)
        del v
        self.assert_(weak_v() is None)

    def test_gc(self):
        """refcount agnostic check that contained objects are freed"""
        def d(capsule, parent):
            return r[0]
        class Obj(object):
            pass
        c = Obj()
        p = Obj()
        r = [Obj()]
        weak_c = weakref.ref(c)
        weak_p = weakref.ref(p)
        weak_r0 = weakref.ref(r[0])
        weak_d = weakref.ref(d)
        v = View(c, p, d)
        weak_v = weakref.ref(v)
        del c, p, d
        self.assert_(weak_c() is not None)
        self.assert_(weak_p() is not None)
        self.assert_(weak_d() is not None)
        del v
        self.assert_(weak_v() is None)
        self.assert_(weak_c() is None)
        self.assert_(weak_p() is None)
        self.assert_(weak_d() is None)
        self.assert_(weak_r0() is not None)
        del r[0]
        self.assert_(weak_r0() is None)
        
    def test_c_api(self):
        api = pygame._view._PYGAME_C_API
        self.assert_(isinstance(api, type(pygame.base._PYGAME_C_API)))

    def test_repr(self):
        z = 0
        cname = re.findall(r"'([^']+)'", repr(View))[0]
        oname, ovalue = re.findall(r"<([^)]+)\(([^)]+)\)>", repr(View(0)))[0]
        self.assertEqual(oname, cname)
        self.assertEqual(id(z), int(ovalue, 16))

    def test_subclassing(self):
        class MyView(View):
            def __repr__(self):
                return "*%s*" % (View.__repr__(self),)
        v = MyView(0)
        self.assertEqual(v.__array_struct__, 0)
        r = repr(v)
        self.assertEqual(r[:2], '*<')
        self.assertEqual(r[-2:], '>*')


class GetArrayInterfaceTest(unittest.TestCase):

    def test_get_array_interface(self):
        surf = pygame.Surface((7, 11), 0, 32)
        d = pygame._view.get_array_interface(surf.get_view("2"))
        self.assertEqual(len(d), 5)
        self.assertEqual(d['version'], 3)
        if pygame.get_sdl_byteorder() == pygame.LIL_ENDIAN:
            byteorder = '<'
        else:
            byteorder = '>'
        self.assertEqual(d['typestr'], byteorder + 'u4')
        self.assertEqual(d['shape'], (7, 11))
        self.assertEqual(d['strides'], (4, 28))
        self.assertEqual(d['data'], (surf._pixels_address, False))

if __name__ == '__main__':
    unittest.main()
