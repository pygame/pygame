import sys
import re
import weakref
import gc
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
    view_keywords = {'shape': (5, 4, 3),
                     'typestr': '|u1',
                     'data': (0, True),
                     'strides': (4, 20, 1)}

    def test___array_struct___property(self):
        kwds = self.view_keywords
        v = View(**kwds)
        d = pygame._view.get_array_interface(v)
        self.assertEqual(len(d), 5)
        self.assertEqual(d['version'], 3)
        self.assertEqual(d['shape'], kwds['shape'])
        self.assertEqual(d['typestr'], kwds['typestr'])
        self.assertEqual(d['data'], kwds['data'])
        self.assertEqual(d['strides'], kwds['strides'])

    def test___array_interface___property(self):
        kwds = self.view_keywords
        v = View(**kwds)
        d = v.__array_interface__
        self.assertEqual(len(d), 5)
        self.assertEqual(d['version'], 3)
        self.assertEqual(d['shape'], kwds['shape'])
        self.assertEqual(d['typestr'], kwds['typestr'])
        self.assertEqual(d['data'], kwds['data'])
        self.assertEqual(d['strides'], kwds['strides'])

    def test_parent_property(self):
        p = []
        v = View(parent=p, **self.view_keywords)
        self.assert_(v.parent is p)
    
    def test_prelude(self):
        def callback(parent):
            success.append(parent is p)

        class MyException(Exception):
            pass

        def raise_exception(parent):
            raise MyException("Just a test.")

        p = []
        
        # For array interface
        success = []
        v = View(parent=p, prelude=callback, **self.view_keywords)
        self.assertEqual(len(success), 0)
        d = v.__array_interface__
        self.assertEqual(len(success), 1)
        self.assertTrue(success[0])
        d = v.__array_interface__
        self.assertEqual(len(success), 1)
        d = v = None
        gc.collect()
        self.assertEqual(len(success), 1)

        # For array struct
        success = []
        v = View(parent=p, prelude=callback, **self.view_keywords)
        self.assertEqual(len(success), 0)
        c = v.__array_struct__
        self.assertEqual(len(success), 1)
        self.assertTrue(success[0])
        c = v.__array_struct__
        self.assertEqual(len(success), 1)
        c = v = None
        gc.collect()
        self.assertEqual(len(success), 1)
        
        # Callback raises an exception
        v = View(prelude=raise_exception, **self.view_keywords)
        self.assertRaises(MyException, lambda : v.__array_struct__)

    def test_postscript(self):
        def callback(parent):
            success.append(parent is p)

        p = []
        
        # For array interface
        success = []
        v = View(parent=p, postscript=callback, **self.view_keywords)
        self.assertEqual(len(success), 0)
        d = v.__array_interface__
        self.assertEqual(len(success), 0)
        d = v.__array_interface__
        self.assertEqual(len(success), 0)
        d = v = None
        gc.collect()
        self.assertEqual(len(success), 1)
        self.assertTrue(success[0])

        # For array struct
        success = []
        v = View(parent=p, postscript=callback, **self.view_keywords)
        self.assertEqual(len(success), 0)
        c = v.__array_struct__
        self.assertEqual(len(success), 0)
        c = v.__array_struct__
        self.assertEqual(len(success), 0)
        c = v = None
        gc.collect()
        self.assertEqual(len(success), 1)
        self.assertTrue(success[0])

    def test_weakref(self):
        v = View(**self.view_keywords)
        weak_v = weakref.ref(v)
        self.assert_(weak_v() is v)
        v = None
        gc.collect()
        self.assert_(weak_v() is None)

    def test_gc(self):
        """refcount agnostic check that contained objects are freed"""
        def prelude_callback(parent):
            return r[0]
        def postscript_callback(parent):
            return r[1]
        class Obj(object):
            pass
        p = Obj()
        r = [Obj(), Obj()]
        weak_p = weakref.ref(p)
        weak_r0 = weakref.ref(r[0])
        weak_r1 = weakref.ref(r[1])
        weak_prelude = weakref.ref(prelude_callback)
        weak_postscript = weakref.ref(postscript_callback)
        v = View(parent=p,
                 prelude=prelude_callback,
                 postscript=postscript_callback,
                 **self.view_keywords)
        weak_v = weakref.ref(v)
        p = prelude_callback = postscript_callback = None
        gc.collect()
        self.assertTrue(weak_p() is not None)
        self.assertTrue(weak_prelude() is not None)
        self.assertTrue(weak_postscript() is not None)
        v = None
        gc.collect()
        self.assertTrue(weak_v() is None)
        self.assertTrue(weak_p() is None)
        self.assertTrue(weak_prelude() is None)
        self.assertTrue(weak_postscript() is None)
        self.assertTrue(weak_r0() is not None)
        self.assertTrue(weak_r1() is not None)
        r = None
        gc.collect()
        self.assertTrue(weak_r0() is None)
        self.assertTrue(weak_r1() is None)
        
    def test_c_api(self):
        api = pygame._view._PYGAME_C_API
        self.assert_(isinstance(api, type(pygame.base._PYGAME_C_API)))

    def test_repr(self):
        v = View(**self.view_keywords)
        cname = re.findall(r"'([^']+)'", repr(View))[0]
        oname, ovalue = re.findall(r"<([^)]+)\(([^)]+)\)>", repr(v))[0]
        self.assertEqual(oname, cname)
        self.assertEqual(id(v), int(ovalue, 16))

    def test_subclassing(self):
        class MyView(View):
            def __repr__(self):
                return "*%s*" % (View.__repr__(self),)
        v = MyView(parent=0, **self.view_keywords)
        self.assertEqual(v.parent, 0)
        r = repr(v)
        self.assertEqual(r[:2], '*<')
        self.assertEqual(r[-2:], '>*')

if __name__ == '__main__':
    unittest.main()
