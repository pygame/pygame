if __name__ == '__main__':
    import sys
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
    from pygame.tests.test_utils import test_not_implemented, unittest, arrinter
else:
    from test.test_utils import test_not_implemented, unittest, arrinter
import pygame
import sys

init_called = quit_called = 0
def __PYGAMEinit__(): #called automatically by pygame.init()
    global init_called
    init_called = init_called + 1
    pygame.register_quit(pygame_quit)
def pygame_quit():
    global quit_called
    quit_called = quit_called + 1


quit_hook_ran = 0
def quit_hook():
    global quit_hook_ran
    quit_hook_ran = 1

class BaseModuleTest(unittest.TestCase):
    def testAutoInit(self):
        pygame.init()
        pygame.quit()
        self.assertEqual(init_called, 1)
        self.assertEqual(quit_called, 1)

    def test_get_sdl_byteorder(self):

        # __doc__ (as of 2008-06-25) for pygame.base.get_sdl_byteorder:

          # pygame.get_sdl_byteorder(): return int
          # get the byte order of SDL

        self.assert_(pygame.get_sdl_byteorder() + 1)

    def test_get_sdl_version(self):

        # __doc__ (as of 2008-06-25) for pygame.base.get_sdl_version:

          # pygame.get_sdl_version(): return major, minor, patch
          # get the version number of SDL

        self.assert_( len(pygame.get_sdl_version()) == 3) 

    class ExporterBase(object):
        def __init__(self, shape, typechar, itemsize):
            import ctypes

            ndim = len(shape)
            self.ndim = ndim
            self.shape = tuple(shape)
            array_len = 1
            for d in shape:
                array_len *= d
            self.size = itemsize * array_len
            self.parent = ctypes.create_string_buffer(self.size)
            self.itemsize = itemsize
            strides = [itemsize] * ndim
            for i in range(ndim - 1, 0, -1):
                strides[i - 1] = strides[i] * shape[i]
            self.strides = tuple(strides)
            self.data = ctypes.addressof(self.parent), False
            if self.itemsize == 1:
                byteorder = '|'
            elif sys.byteorder == 'big':
                byteorder = '>'
            else:
                byteorder = '<'
            self.typestr = byteorder + typechar + str(self.itemsize)

    def assertSame(self, proxy, obj):
        self.assertEqual(proxy.length, obj.size)
        d = proxy.__array_interface__
        try:
            self.assertEqual(d['typestr'], obj.typestr)
            self.assertEqual(d['shape'], obj.shape)
            self.assertEqual(d['strides'], obj.strides)
            self.assertEqual(d['data'], obj.data)
        finally:
            d = None

    def test_PgObject_GetBuffer_array_interface(self):
        from pygame.bufferproxy import BufferProxy

        class Exporter(self.ExporterBase):
            def get__array_interface__(self):
                return {'version': 3,
                        'typestr': self.typestr,
                        'shape': self.shape,
                        'strides': self.strides,
                        'data': self.data}
            __array_interface__ = property(get__array_interface__)
            # Should be ignored by PgObject_GetBuffer
            __array_struct__ = property(lambda self: None)

        _shape = [2, 3, 5, 7, 11]  # Some prime numbers
        for ndim in range(1, len(_shape)):
            o = Exporter(_shape[0:ndim], 'i', 2)
            v = BufferProxy(o)
            self.assertSame(v, o)
        ndim = 2
        shape = _shape[0:ndim]
        for typechar in ('i', 'u'):
            for itemsize in (1, 2, 4, 8):
                o = Exporter(shape, typechar, itemsize)
                v = BufferProxy(o)
                self.assertSame(v, o)
        for itemsize in (4, 8):
            o = Exporter(shape, 'f', itemsize)
            v = BufferProxy(o)
            self.assertSame(v, o)
        
    def test_GetView_array_struct(self):
        from pygame.bufferproxy import BufferProxy

        class Exporter(self.ExporterBase):
            def __init__(self, shape, typechar, itemsize):
                super(Exporter, self).__init__(shape, typechar, itemsize)
                self.view = BufferProxy(self.__dict__)

            def get__array_struct__(self):
                return self.view.__array_struct__
            __array_struct__ = property(get__array_struct__)
            # Should not cause PgObject_GetBuffer to fail
            __array_interface__ = property(lambda self: None)

        _shape = [2, 3, 5, 7, 11]  # Some prime numbers
        for ndim in range(1, len(_shape)):
            o = Exporter(_shape[0:ndim], 'i', 2)
            v = BufferProxy(o)
            self.assertSame(v, o)
        ndim = 2
        shape = _shape[0:ndim]
        for typechar in ('i', 'u'):
            for itemsize in (1, 2, 4, 8):
                o = Exporter(shape, typechar, itemsize)
                v = BufferProxy(o)
                self.assertSame(v, o)
        for itemsize in (4, 8):
            o = Exporter(shape, 'f', itemsize)
            v = BufferProxy(o)
            self.assertSame(v, o)

    if (pygame.HAVE_NEWBUF):
        def test_newbuf(self):
            self.NEWBUF_test_newbuf()
        def test_PgDict_AsBuffer_PyBUF_flags(self):
            self.NEWBUF_test_PgDict_AsBuffer_PyBUF_flags()
        def test_PgObject_AsBuffer_PyBUF_flags(self):
            self.NEWBUF_test_PgObject_AsBuffer_PyBUF_flags()
        if is_pygame_pkg:
            from pygame.tests.test_utils import buftools
        else:
            from test.test_utils import buftools

    def NEWBUF_assertSame(self, proxy, exp):
        buftools = self.buftools
        self.assertEqual(proxy.length, exp.len)
        imp = buftools.BufferImporter(proxy, buftools.PyBUF_RECORDS_RO)
        try:
            self.assertEqual(imp.readonly, exp.readonly)
            self.assertEqual(imp.format, exp.format)
            self.assertEqual(imp.itemsize, exp.itemsize)
            self.assertEqual(imp.ndim, exp.ndim)
            self.assertEqual(imp.shape, exp.shape)
            self.assertEqual(imp.strides, exp.strides)
            self.assertTrue(imp.suboffsets is None)
        finally:
            imp = None

    def NEWBUF_test_newbuf(self):
        from pygame.bufferproxy import BufferProxy

        Exporter = self.buftools.BufferExporter
        _shape = [2, 3, 5, 7, 11]  # Some prime numbers
        for ndim in range(1, len(_shape)):
            o = Exporter(_shape[0:ndim], '=h')
            v = BufferProxy(o)
            self.NEWBUF_assertSame(v, o)
        ndim = 2
        shape = _shape[0:ndim]
        for format in ['b', 'B', '=h', '=H', '=i', '=I', '=q', '=Q', 'f', 'd']:
            o = Exporter(shape, format)
            v = BufferProxy(o)
            self.NEWBUF_assertSame(v, o)

    def NEWBUF_test_PgDict_AsBuffer_PyBUF_flags(self):
        from pygame.bufferproxy import BufferProxy

        is_lil_endian = pygame.get_sdl_byteorder() == pygame.LIL_ENDIAN
        fsys, frev = ('<', '>') if is_lil_endian else ('>', '<')
        buftools = self.buftools
        BufferImporter = buftools.BufferImporter
        a = BufferProxy({'typestr': '|u4',
                         'shape': (10, 2),
                         'data': (9, False)}) # 9? No data accesses.
        b = BufferImporter(a, buftools.PyBUF_SIMPLE)
        self.assertEqual(b.ndim, 0)
        self.assertTrue(b.format is None)
        self.assertEqual(b.len, a.length)
        self.assertEqual(b.itemsize, 4)
        self.assertTrue(b.shape is None)
        self.assertTrue(b.strides is None)
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, 9)
        b = BufferImporter(a, buftools.PyBUF_WRITABLE)
        self.assertEqual(b.ndim, 0)
        self.assertTrue(b.format is None)
        self.assertEqual(b.len, a.length)
        self.assertEqual(b.itemsize, 4)
        self.assertTrue(b.shape is None)
        self.assertTrue(b.strides is None)
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, 9)
        b = BufferImporter(a, buftools.PyBUF_ND)
        self.assertEqual(b.ndim, 2)
        self.assertTrue(b.format is None)
        self.assertEqual(b.len, a.length)
        self.assertEqual(b.itemsize, 4)
        self.assertEqual(b.shape, (10, 2))
        self.assertTrue(b.strides is None)
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, 9)
        a = BufferProxy({'typestr': fsys + 'i2',
                         'shape': (5, 10),
                         'strides': (24, 2),
                         'data': (42, False)}) # 42? No data accesses.
        b = BufferImporter(a, buftools.PyBUF_STRIDES)
        self.assertEqual(b.ndim, 2)
        self.assertTrue(b.format is None)
        self.assertEqual(b.len, 100)
        self.assertEqual(b.itemsize, 2)
        self.assertEqual(b.shape, (5, 10))
        self.assertEqual(b.strides, (24, 2))
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, 42)
        b = BufferImporter(a, buftools.PyBUF_FULL_RO)
        self.assertEqual(b.ndim, 2)
        self.assertEqual(b.format, '=h')
        self.assertEqual(b.len, 100)
        self.assertEqual(b.itemsize, 2)
        self.assertEqual(b.shape, (5, 10))
        self.assertEqual(b.strides, (24, 2))
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, 42)
        self.assertRaises(BufferError, BufferImporter, a, buftools.PyBUF_SIMPLE)
        self.assertRaises(BufferError, BufferImporter, a, buftools.PyBUF_ND)
        self.assertRaises(BufferError, BufferImporter, a,
                          buftools.PyBUF_C_CONTIGUOUS)
        self.assertRaises(BufferError, BufferImporter, a,
                          buftools.PyBUF_F_CONTIGUOUS)
        self.assertRaises(BufferError, BufferImporter, a,
                          buftools.PyBUF_ANY_CONTIGUOUS)
        self.assertRaises(BufferError, BufferImporter, a, buftools.PyBUF_CONTIG)
        self.assertRaises(BufferError, BufferImporter, a, buftools.PyBUF_SIMPLE)
        self.assertRaises(BufferError, BufferImporter, a, buftools.PyBUF_ND)
        self.assertRaises(BufferError, BufferImporter, a,
                          buftools.PyBUF_C_CONTIGUOUS)
        self.assertRaises(BufferError, BufferImporter, a,
                          buftools.PyBUF_F_CONTIGUOUS)
        self.assertRaises(BufferError, BufferImporter, a,
                          buftools.PyBUF_ANY_CONTIGUOUS)
        self.assertRaises(BufferError, BufferImporter, a, buftools.PyBUF_CONTIG)
        a = BufferProxy({'typestr': frev + 'i2',
                         'shape': (3, 5, 10),
                         'strides': (120, 24, 2),
                         'data': (1000000, True)}) # 1000000? No data accesses.
        b = BufferImporter(a, buftools.PyBUF_FULL_RO)
        self.assertEqual(b.ndim, 3)
        self.assertEqual(b.format, frev + 'h')
        self.assertEqual(b.len, 300)
        self.assertEqual(b.itemsize, 2)
        self.assertEqual(b.shape, (3, 5, 10))
        self.assertEqual(b.strides, (120, 24, 2))
        self.assertTrue(b.suboffsets is None)
        self.assertTrue(b.readonly)
        self.assertEqual(b.buf, 1000000)
        self.assertRaises(BufferError, BufferImporter, a, buftools.PyBUF_FULL)

    def NEWBUF_test_PgObject_AsBuffer_PyBUF_flags(self):
        from pygame.bufferproxy import BufferProxy
        import ctypes

        is_lil_endian = pygame.get_sdl_byteorder() == pygame.LIL_ENDIAN
        fsys, frev = ('<', '>') if is_lil_endian else ('>', '<')
        buftools = self.buftools
        BufferImporter = buftools.BufferImporter
        e = arrinter.Exporter((10, 2), typekind='f',
                              itemsize=ctypes.sizeof(ctypes.c_double))
        a = BufferProxy(e)
        b = BufferImporter(a, buftools.PyBUF_SIMPLE)
        self.assertEqual(b.ndim, 0)
        self.assertTrue(b.format is None)
        self.assertEqual(b.len, e.len)
        self.assertEqual(b.itemsize, e.itemsize)
        self.assertTrue(b.shape is None)
        self.assertTrue(b.strides is None)
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, e.data)
        b = BufferImporter(a, buftools.PyBUF_WRITABLE)
        self.assertEqual(b.ndim, 0)
        self.assertTrue(b.format is None)
        self.assertEqual(b.len, e.len)
        self.assertEqual(b.itemsize, e.itemsize)
        self.assertTrue(b.shape is None)
        self.assertTrue(b.strides is None)
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, e.data)
        b = BufferImporter(a, buftools.PyBUF_ND)
        self.assertEqual(b.ndim, e.nd)
        self.assertTrue(b.format is None)
        self.assertEqual(b.len, a.length)
        self.assertEqual(b.itemsize, e.itemsize)
        self.assertEqual(b.shape, e.shape)
        self.assertTrue(b.strides is None)
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, e.data)
        e = arrinter.Exporter((5, 10), typekind='i', itemsize=2,
                              strides=(24, 2))
        a = BufferProxy(e)
        b = BufferImporter(a, buftools.PyBUF_STRIDES)
        self.assertEqual(b.ndim, e.nd)
        self.assertTrue(b.format is None)
        self.assertEqual(b.len, e.len)
        self.assertEqual(b.itemsize, e.itemsize)
        self.assertEqual(b.shape, e.shape)
        self.assertEqual(b.strides, e.strides)
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, e.data)
        b = BufferImporter(a, buftools.PyBUF_FULL_RO)
        self.assertEqual(b.ndim, e.nd)
        self.assertEqual(b.format, '=h')
        self.assertEqual(b.len, e.len)
        self.assertEqual(b.itemsize, e.itemsize)
        self.assertEqual(b.shape, e.shape)
        self.assertEqual(b.strides, e.strides)
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, e.data)
        self.assertRaises(BufferError, BufferImporter, a, buftools.PyBUF_SIMPLE)
        self.assertRaises(BufferError, BufferImporter, a,
                          buftools.PyBUF_WRITABLE)
        self.assertRaises(BufferError, BufferImporter, a,
                          buftools.PyBUF_WRITABLE)
        self.assertRaises(BufferError, BufferImporter, a, buftools.PyBUF_ND)
        self.assertRaises(BufferError, BufferImporter, a,
                          buftools.PyBUF_C_CONTIGUOUS)
        self.assertRaises(BufferError, BufferImporter, a,
                          buftools.PyBUF_F_CONTIGUOUS)
        self.assertRaises(BufferError, BufferImporter, a,
                          buftools.PyBUF_ANY_CONTIGUOUS)
        self.assertRaises(BufferError, BufferImporter, a, buftools.PyBUF_CONTIG)
        self.assertRaises(BufferError, BufferImporter, a, buftools.PyBUF_SIMPLE)
        self.assertRaises(BufferError, BufferImporter, a, buftools.PyBUF_ND)
        self.assertRaises(BufferError, BufferImporter, a,
                          buftools.PyBUF_C_CONTIGUOUS)
        self.assertRaises(BufferError, BufferImporter, a,
                          buftools.PyBUF_F_CONTIGUOUS)
        self.assertRaises(BufferError, BufferImporter, a,
                          buftools.PyBUF_ANY_CONTIGUOUS)
        self.assertRaises(BufferError, BufferImporter, a, buftools.PyBUF_CONTIG)
        e = arrinter.Exporter((3, 5, 10), typekind='i', itemsize=2,
                              strides=(120, 24, 2),
                              flags=arrinter.PAI_ALIGNED)
        a = BufferProxy(e)
        b = BufferImporter(a, buftools.PyBUF_FULL_RO)
        self.assertEqual(b.ndim, e.nd)
        self.assertEqual(b.format, frev + 'h')
        self.assertEqual(b.len, e.len)
        self.assertEqual(b.itemsize, e.itemsize)
        self.assertEqual(b.shape, e.shape)
        self.assertEqual(b.strides, e.strides)
        self.assertTrue(b.suboffsets is None)
        self.assertTrue(b.readonly)
        self.assertEqual(b.buf, e.data)
        self.assertRaises(BufferError, BufferImporter, a, buftools.PyBUF_FULL)

    def test_PgObject_GetBuffer_exception(self):
        # For consistency with surfarray
        from pygame.bufferproxy import BufferProxy

        bp = BufferProxy(1)
        self.assertRaises(ValueError, getattr, bp, 'length')

    def not_init_assertions(self):
        self.assert_(not pygame.display.get_init(),
                     "display shouldn't be initialized" )
        if 'pygame.mixer' in sys.modules:
            self.assert_(not pygame.mixer.get_init(),
                         "mixer shouldn't be initialized" )
        if 'pygame.font' in sys.modules:
            self.assert_(not pygame.font.get_init(),
                         "init shouldn't be initialized" )

        ## !!! TODO : Remove when scrap works for OS X
        import platform
        if platform.system().startswith('Darwin'):
            return

        try:
            self.assertRaises(pygame.error, pygame.scrap.get)
        except NotImplementedError:
            # Scrap is optional.
            pass
        
        # pygame.cdrom
        # pygame.joystick

    def init_assertions(self):
        self.assert_(pygame.display.get_init())
        if 'pygame.mixer' in sys.modules:
            self.assert_(pygame.mixer.get_init())
        if 'pygame.font' in sys.modules:
            self.assert_(pygame.font.get_init())

    def test_quit__and_init(self):
        # __doc__ (as of 2008-06-25) for pygame.base.quit:

          # pygame.quit(): return None
          # uninitialize all pygame modules
        
        # Make sure everything is not init
        self.not_init_assertions()
    
        # Initiate it
        pygame.init()
        
        # Check
        self.init_assertions()

        # Quit
        pygame.quit()
        
        # All modules have quit
        self.not_init_assertions()

    def test_register_quit(self):

        # __doc__ (as of 2008-06-25) for pygame.base.register_quit:

          # register_quit(callable): return None
          # register a function to be called when pygame quits
        
        self.assert_(not quit_hook_ran)

        pygame.init()
        pygame.register_quit(quit_hook)
        pygame.quit()

        self.assert_(quit_hook_ran)

    def test_get_error(self):

        # __doc__ (as of 2008-08-02) for pygame.base.get_error:

          # pygame.get_error(): return errorstr
          # get the current error message
          # 
          # SDL maintains an internal error message. This message will usually
          # be given to you when pygame.error is raised. You will rarely need to
          # call this function.
          # 

        self.assertEqual(pygame.get_error(), "")
        pygame.set_error("hi")
        self.assertEqual(pygame.get_error(), "hi")
        pygame.set_error("")
        self.assertEqual(pygame.get_error(), "")



    def test_set_error(self):

        self.assertEqual(pygame.get_error(), "")
        pygame.set_error("hi")
        self.assertEqual(pygame.get_error(), "hi")
        pygame.set_error("")
        self.assertEqual(pygame.get_error(), "")



    def test_init(self):

        # __doc__ (as of 2008-08-02) for pygame.base.init:

        # pygame.init(): return (numpass, numfail)
        # initialize all imported pygame modules
        # 
        # Initialize all imported Pygame modules. No exceptions will be raised
        # if a module fails, but the total number if successful and failed
        # inits will be returned as a tuple. You can always initialize
        # individual modules manually, but pygame.init is a convenient way to
        # get everything started. The init() functions for individual modules
        # will raise exceptions when they fail.
        # 
        # You may want to initalise the different modules seperately to speed
        # up your program or to not use things your game does not.
        # 
        # It is safe to call this init() more than once: repeated calls will
        # have no effect. This is true even if you have pygame.quit() all the
        # modules.
        # 



        # Make sure everything is not init
        self.not_init_assertions()
    
        # Initiate it
        pygame.init()
        
        # Check
        self.init_assertions()

        # Quit
        pygame.quit()
        
        # All modules have quit
        self.not_init_assertions()


    def todo_test_segfault(self):

        # __doc__ (as of 2008-08-02) for pygame.base.segfault:

          # crash

        self.fail() 

if __name__ == '__main__':
    unittest.main()
