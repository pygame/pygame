import sys
import re
import weakref
import gc
import ctypes
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
from pygame.bufferproxy import BufferProxy
from pygame.compat import as_bytes

class BufferProxyTest(unittest.TestCase):
    view_keywords = {'shape': (5, 4, 3),
                     'typestr': '|u1',
                     'data': (0, True),
                     'strides': (4, 20, 1)}

    def test___array_struct___property(self):
        kwds = self.view_keywords
        v = BufferProxy(**kwds)
        d = pygame.get_array_interface(v)
        self.assertEqual(len(d), 5)
        self.assertEqual(d['version'], 3)
        self.assertEqual(d['shape'], kwds['shape'])
        self.assertEqual(d['typestr'], kwds['typestr'])
        self.assertEqual(d['data'], kwds['data'])
        self.assertEqual(d['strides'], kwds['strides'])

    def test___array_interface___property(self):
        kwds = self.view_keywords
        v = BufferProxy(**kwds)
        d = v.__array_interface__
        self.assertEqual(len(d), 5)
        self.assertEqual(d['version'], 3)
        self.assertEqual(d['shape'], kwds['shape'])
        self.assertEqual(d['typestr'], kwds['typestr'])
        self.assertEqual(d['data'], kwds['data'])
        self.assertEqual(d['strides'], kwds['strides'])

    def test_parent_property(self):
        p = []
        v = BufferProxy(parent=p, **self.view_keywords)
        self.assert_(v.parent is p)

    def test_before(self):
        def callback(parent):
            success.append(parent is p)

        class MyException(Exception):
            pass

        def raise_exception(parent):
            raise MyException("Just a test.")

        p = []

        # For array interface
        success = []
        v = BufferProxy(parent=p, before=callback, **self.view_keywords)
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
        v = BufferProxy(parent=p, before=callback, **self.view_keywords)
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
        v = BufferProxy(before=raise_exception, **self.view_keywords)
        self.assertRaises(MyException, lambda : v.__array_struct__)

    def test_after(self):
        def callback(parent):
            success.append(parent is p)

        p = []

        # For array interface
        success = []
        v = BufferProxy(parent=p, after=callback, **self.view_keywords)
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
        v = BufferProxy(parent=p, after=callback, **self.view_keywords)
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
        v = BufferProxy(**self.view_keywords)
        weak_v = weakref.ref(v)
        self.assert_(weak_v() is v)
        v = None
        gc.collect()
        self.assert_(weak_v() is None)

    def test_gc(self):
        """refcount agnostic check that contained objects are freed"""
        def before_callback(parent):
            return r[0]
        def after_callback(parent):
            return r[1]
        class Obj(object):
            pass
        p = Obj()
        r = [Obj(), Obj()]
        weak_p = weakref.ref(p)
        weak_r0 = weakref.ref(r[0])
        weak_r1 = weakref.ref(r[1])
        weak_before = weakref.ref(before_callback)
        weak_after = weakref.ref(after_callback)
        v = BufferProxy(parent=p,
                 before=before_callback,
                 after=after_callback,
                 **self.view_keywords)
        weak_v = weakref.ref(v)
        p = before_callback = after_callback = None
        gc.collect()
        self.assertTrue(weak_p() is not None)
        self.assertTrue(weak_before() is not None)
        self.assertTrue(weak_after() is not None)
        v = None
        gc.collect()
        self.assertTrue(weak_v() is None)
        self.assertTrue(weak_p() is None)
        self.assertTrue(weak_before() is None)
        self.assertTrue(weak_after() is None)
        self.assertTrue(weak_r0() is not None)
        self.assertTrue(weak_r1() is not None)
        r = None
        gc.collect()
        self.assertTrue(weak_r0() is None)
        self.assertTrue(weak_r1() is None)

    def test_c_api(self):
        api = pygame.bufferproxy._PYGAME_C_API
        self.assert_(isinstance(api, type(pygame.base._PYGAME_C_API)))

    def test_repr(self):
        v = BufferProxy(**self.view_keywords)
        cname = re.findall(r"'([^']+)'", repr(BufferProxy))[0]
        oname, ovalue = re.findall(r"<([^)]+)\(([^)]+)\)>", repr(v))[0]
        self.assertEqual(oname, cname)
        self.assertEqual(id(v), int(ovalue, 16))

    def test_subclassing(self):
        class MyBufferProxy(BufferProxy):
            def __repr__(self):
                return "*%s*" % (BufferProxy.__repr__(self),)
        v = MyBufferProxy(parent=0, **self.view_keywords)
        self.assertEqual(v.parent, 0)
        r = repr(v)
        self.assertEqual(r[:2], '*<')
        self.assertEqual(r[-2:], '>*')

class BufferProxyLegacyTest(unittest.TestCase):
    content = as_bytes('\x01\x00\x00\x02') * 12
    buffer = ctypes.create_string_buffer(content)
    data = (ctypes.addressof(buffer), True)

    def test_length(self):

        # __doc__ (as of 2008-08-02) for pygame.bufferproxy.BufferProxy.length:

          # The size of the buffer data in bytes.
        bf = BufferProxy(shape=(3, 4),
                  typestr='|u4',
                  data=self.data,
                  strides=(12, 4))
        self.assertEqual(bf.length, len(self.content))
        bf = BufferProxy(shape=(3, 3),
                  typestr='|u4',
                  data=self.data,
                  strides=(12, 4))
        self.assertEqual(bf.length, 3*3*4)

    def todo_test_raw(self):

        # __doc__ (as of 2008-08-02) for pygame.bufferproxy.BufferProxy.raw:

          # The raw buffer data as string. The string may contain NUL bytes.

        self.fail()

    def todo_test_write(self):

        # __doc__ (as of 2008-08-02) for pygame.bufferproxy.BufferProxy.write:

          # B.write (bufferproxy, buffer, offset) -> None
          #
          # Writes raw data to the bufferproxy.
          #
          # Writes the raw data from buffer to the BufferProxy object, starting
          # at the specified offset within the BufferProxy.
          # If the length of the passed buffer exceeds the length of the
          # BufferProxy (reduced by the offset), an IndexError will be raised.

        self.fail()


if __name__ == '__main__':
    unittest.main()
