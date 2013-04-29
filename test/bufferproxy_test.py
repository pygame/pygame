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

    def test_module_name(self):
        self.assertEqual(pygame.bufferproxy.__name__,
                         "pygame.bufferproxy")

    def test_class_name(self):
        self.assertEqual(BufferProxy.__name__, "BufferProxy")

    def test___array_struct___property(self):
        kwds = self.view_keywords
        v = BufferProxy(kwds)
        d = pygame.get_array_interface(v)
        self.assertEqual(len(d), 5)
        self.assertEqual(d['version'], 3)
        self.assertEqual(d['shape'], kwds['shape'])
        self.assertEqual(d['typestr'], kwds['typestr'])
        self.assertEqual(d['data'], kwds['data'])
        self.assertEqual(d['strides'], kwds['strides'])

    def test___array_interface___property(self):
        kwds = self.view_keywords
        v = BufferProxy(kwds)
        d = v.__array_interface__
        self.assertEqual(len(d), 5)
        self.assertEqual(d['version'], 3)
        self.assertEqual(d['shape'], kwds['shape'])
        self.assertEqual(d['typestr'], kwds['typestr'])
        self.assertEqual(d['data'], kwds['data'])
        self.assertEqual(d['strides'], kwds['strides'])

    def test_parent_property(self):
        kwds = dict(self.view_keywords)
        p = []
        kwds['parent'] = p
        v = BufferProxy(kwds)
        self.assert_(v.parent is p)

    def test_before(self):
        def callback(parent):
            success.append(parent is p)

        class MyException(Exception):
            pass

        def raise_exception(parent):
            raise MyException("Just a test.")

        kwds = dict(self.view_keywords)
        p = []
        kwds['parent'] = p

        # For array interface
        success = []
        kwds['before'] = callback
        v = BufferProxy(kwds)
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
        kwds['before'] = callback
        v = BufferProxy(kwds)
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
        kwds['before'] = raise_exception
        v = BufferProxy(kwds)
        self.assertRaises(MyException, lambda : v.__array_struct__)

    def test_after(self):
        def callback(parent):
            success.append(parent is p)

        kwds = dict(self.view_keywords)
        p = []
        kwds['parent'] = p

        # For array interface
        success = []
        kwds['after'] = callback
        v = BufferProxy(kwds)
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
        kwds['after'] = callback
        v = BufferProxy(kwds)
        self.assertEqual(len(success), 0)
        c = v.__array_struct__
        self.assertEqual(len(success), 0)
        c = v.__array_struct__
        self.assertEqual(len(success), 0)
        c = v = None
        gc.collect()
        self.assertEqual(len(success), 1)
        self.assertTrue(success[0])

    def test_attribute(self):
        v = BufferProxy(self.view_keywords)
        self.assertRaises(AttributeError, getattr, v, 'undefined')
        v.undefined = 12;
        self.assertEqual(v.undefined, 12)
        del v.undefined
        self.assertRaises(AttributeError, getattr, v, 'undefined')

    def test_weakref(self):
        v = BufferProxy(self.view_keywords)
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
        a = Obj()
        r = [Obj(), Obj()]
        weak_p = weakref.ref(p)
        weak_a = weakref.ref(a)
        weak_r0 = weakref.ref(r[0])
        weak_r1 = weakref.ref(r[1])
        weak_before = weakref.ref(before_callback)
        weak_after = weakref.ref(after_callback)
        kwds = dict(self.view_keywords)
        kwds['parent'] = p
        kwds['before'] = before_callback
        kwds['after'] = after_callback
        v = BufferProxy(kwds)
        v.some_attribute = a
        weak_v = weakref.ref(v)
        kwds = p = a = before_callback = after_callback = None
        gc.collect()
        self.assertTrue(weak_p() is not None)
        self.assertTrue(weak_a() is not None)
        self.assertTrue(weak_before() is not None)
        self.assertTrue(weak_after() is not None)
        v = None
        gc.collect()
        self.assertTrue(weak_v() is None)
        self.assertTrue(weak_p() is None)
        self.assertTrue(weak_a() is None)
        self.assertTrue(weak_before() is None)
        self.assertTrue(weak_after() is None)
        self.assertTrue(weak_r0() is not None)
        self.assertTrue(weak_r1() is not None)
        r = None
        gc.collect()
        self.assertTrue(weak_r0() is None)
        self.assertTrue(weak_r1() is None)

        # Cycle removal
        kwds = dict(self.view_keywords)
        kwds['parent'] = []
        v = BufferProxy(kwds)
        v.some_attribute = v
        tracked = True
        for o in gc.get_objects():
            if o is v:
                break
        else:
            tracked = False
        self.assertTrue(tracked)
        kwds['parent'].append(v)
        kwds = None
        gc.collect()
        n1 = len(gc.garbage)
        v = None
        gc.collect()
        n2 = len(gc.garbage)
        self.assertEqual(n2, n1)

    def test_c_api(self):
        api = pygame.bufferproxy._PYGAME_C_API
        self.assert_(isinstance(api, type(pygame.base._PYGAME_C_API)))

    def test_repr(self):
        v = BufferProxy(self.view_keywords)
        cname = re.findall(r"'([^']+)'", repr(BufferProxy))[0]
        oname, ovalue = re.findall(r"<([^)]+)\(([^)]+)\)>", repr(v))[0]
        self.assertEqual(oname, cname)
        self.assertEqual(id(v), int(ovalue, 16))

    def test_subclassing(self):
        class MyBufferProxy(BufferProxy):
            def __repr__(self):
                return "*%s*" % (BufferProxy.__repr__(self),)
        kwds = dict(self.view_keywords)
        kwds['parent'] = 0
        v = MyBufferProxy(kwds)
        self.assertEqual(v.parent, 0)
        r = repr(v)
        self.assertEqual(r[:2], '*<')
        self.assertEqual(r[-2:], '>*')

    if pygame.HAVE_NEWBUF:
        def test_newbuf_arg(self):
            self.NEWBUF_test_newbuf_arg()

    def NEWBUF_test_newbuf_arg(self):
        from pygame.newbuffer import BufferMixin
        from ctypes import (c_char, c_short, c_ssize_t, addressof, sizeof,
                            create_string_buffer, cast, POINTER)
        
        class Array(BufferMixin):
            def __init__(self, format, init):
                if format == 'B':
                    c_itemtype = c_char
                elif format == 'h':
                    c_itemtype = c_short
                else:
                    raise ValueError("Unknown item format '" + format + "'")
                self._format = create_string_buffer(format.encode('ascii'))
                self.itemsize = sizeof(c_itemtype)
                self._nitems = len(init)
                self._len = self._nitems * self.itemsize
                self.buffer = (c_itemtype * self._nitems)(*init)
                self._len = sizeof(self.buffer)
                self.ndim = 1
                self.shape = (c_ssize_t * 1)(self._nitems)
                self.strides = (c_ssize_t * 1)(self.itemsize)
            def buffer_info(self):
                return (addressof(self.buffer), self._nitems)
            def tobytes(self):
                return cast(self.buffer, POINTER(c_char))[0:self._len]
            def __len__(self):
                return self._nitems
            def _get_buffer(self, view, flags):
                view.buf = addressof(self.buffer)
                view.len = self._len
                view.readonly = False
                view.format = addressof(self._format)
                view.itemsize = self.itemsize
                view.ndim = self.ndim
                view.shape = addressof(self.shape)
                view.strides = addressof(self.strides)
                view.suboffsets = None
                view.internal = None
                view.obj = self

        raw = as_bytes('abc\x00def')
        b = Array('B', raw)
        b_address, b_nitems = b.buffer_info()
        v = BufferProxy(b)
        self.assertEqual(v.length, len(b))
        self.assertEqual(v.raw, raw)
        d = v.__array_interface__
        try:
            self.assertEqual(d['typestr'], '|u1')
            self.assertEqual(d['shape'], (b_nitems,))
            self.assertEqual(d['strides'], (b.itemsize,))
            self.assertEqual(d['data'], (b_address, False))
        finally:
            d = None
        b = Array('h', [-1, 0, 2])
        v = BufferProxy(b)
        b_address, b_nitems = b.buffer_info()
        b_nbytes = b.itemsize * b_nitems
        self.assertEqual(v.length, b_nbytes)
        try:
            s = b.tostring()
        except AttributeError:
            s = b.tobytes()
        self.assertEqual(v.raw, s)
        d = v.__array_interface__
        try:
            lil_endian = pygame.get_sdl_byteorder() == pygame.LIL_ENDIAN
            f = '{}i{}'.format('<' if lil_endian else '>', b.itemsize)
            self.assertEqual(d['typestr'], f)
            self.assertEqual(d['shape'], (b_nitems,))
            self.assertEqual(d['strides'], (b.itemsize,))
            self.assertEqual(d['data'], (b_address, False))
        finally:
            d = None

    try:
        pygame.bufferproxy.get_segcount
    except AttributeError:
        pass
    else:
        def test_oldbuf_arg(self):
            self.OLDBUF_test_oldbuf_arg()

    def OLDBUF_test_oldbuf_arg(self):
        from pygame.bufferproxy import (get_segcount, get_read_buffer,
                                        get_write_buffer)

        content = as_bytes('\x01\x00\x00\x02') * 12
        memory = ctypes.create_string_buffer(content)
        memaddr = ctypes.addressof(memory)
        def raise_exception(o):
            raise ValueError("An exception")

        bf = BufferProxy({'shape': (len(content),),
                          'typestr': '|u1',
                          'data': (memaddr, False),
                          'strides': (1,)})
        seglen, segaddr = get_read_buffer(bf, 0)
        self.assertEqual(segaddr, 0)
        self.assertEqual(seglen, 0)
        seglen, segaddr = get_write_buffer(bf, 0)
        self.assertEqual(segaddr, 0)
        self.assertEqual(seglen, 0)
        segcount, buflen = get_segcount(bf)
        self.assertEqual(segcount, 1)
        self.assertEqual(buflen, len(content))
        seglen, segaddr = get_read_buffer(bf, 0)
        self.assertEqual(segaddr, memaddr)
        self.assertEqual(seglen, len(content))
        seglen, segaddr = get_write_buffer(bf, 0)
        self.assertEqual(segaddr, memaddr)
        self.assertEqual(seglen, len(content))
        
        bf = BufferProxy({'shape': (len(content),),
                          'typestr': '|u1',
                          'data': (memaddr, True),
                          'strides': (1,)})
        segcount, buflen = get_segcount(bf)
        self.assertEqual(segcount, 1)
        self.assertEqual(buflen, len(content))
        seglen, segaddr = get_read_buffer(bf, 0)
        self.assertEqual(segaddr, memaddr)
        self.assertEqual(seglen, len(content))
        self.assertRaises(ValueError, get_write_buffer, bf, 0)

        bf = BufferProxy({'shape': (len(content),),
                          'typestr': '|u1',
                          'data': (memaddr, True),
                          'strides': (1,),
                          'before': raise_exception})
        segcount, buflen = get_segcount(bf)
        self.assertEqual(segcount, 0)
        self.assertEqual(buflen, 0)

        bf = BufferProxy({'shape': (3, 4),
                          'typestr': '|u4',
                          'data': (memaddr, True),
                          'strides': (12, 4)})
        segcount, buflen = get_segcount(bf)
        self.assertEqual(segcount, 3 * 4)
        self.assertEqual(buflen, 3 * 4 * 4)
        for i in range(0, 4):
            seglen, segaddr = get_read_buffer(bf, i)
            self.assertEqual(segaddr, memaddr + i * 4)
            self.assertEqual(seglen, 4)

class BufferProxyLegacyTest(unittest.TestCase):
    content = as_bytes('\x01\x00\x00\x02') * 12
    buffer = ctypes.create_string_buffer(content)
    data = (ctypes.addressof(buffer), True)

    def test_length(self):

        # __doc__ (as of 2008-08-02) for pygame.bufferproxy.BufferProxy.length:

          # The size of the buffer data in bytes.
        bf = BufferProxy({'shape': (3, 4),
                          'typestr': '|u4',
                          'data': self.data,
                          'strides': (12, 4)})
        self.assertEqual(bf.length, len(self.content))
        bf = BufferProxy({'shape': (3, 3),
                          'typestr': '|u4',
                          'data': self.data,
                          'strides': (12, 4)})
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
