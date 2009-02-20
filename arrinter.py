from ctypes import *

class PyArrayInterface(Structure):
    _fields_ = [('two', c_int), ('nd', c_int), ('typekind', c_char),
                ('itemsize', c_int), ('flags', c_int),
                ('shape', POINTER(c_int)),
                ('strides', POINTER(c_int)),
                ('data', c_void_p), ('descr', py_object)]

PAI_Ptr = POINTER(PyArrayInterface)
PyCObject_AsVoidPtr = pythonapi.PyCObject_AsVoidPtr
PyCObject_AsVoidPtr.restype = c_void_p
PyCObject_AsVoidPtr.argtypes = [py_object]

PAI_CONTIGUOUS = 0x01
PAI_FORTRAN = 0x02
PAI_ALIGNED = 0x100
PAI_NOTSWAPPED = 0x200
PAI_WRITEABLE = 0x400
PAI_ARR_HAS_DESCR = 0x800

class Inter(object):
    def __init__(self, arr):
        self._cobj = arr.__array_struct__
        vp = PyCObject_AsVoidPtr(self._cobj)
        self._inter = cast(vp, PAI_Ptr)[0]

    def __getattr__(self, name):
        return getattr(self._inter, name)

    def __str__(self):
        return ("nd: %i\n"
                "typekind: %s\n"
                "itemsize: %i\n"
                "flags: %s\n"
                "shape: %s\n"
                "strides: %s\n" %
                (self.nd, self.typekind, self.itemsize,
                 format_flags(self.flags),
                 format_shape(self.nd, self.shape),
                 format_strides(self.nd, self.strides)))

def format_flags(flags):
    names = []
    for flag, name in [(PAI_CONTIGUOUS, 'CONTIGUOUS'),
                       (PAI_FORTRAN, 'FORTRAN'),
                       (PAI_ALIGNED, 'ALIGNED'),
                       (PAI_NOTSWAPPED, 'NOTSWAPPED'),
                       (PAI_WRITEABLE, 'WRITEABLE'),
                       (PAI_ARR_HAS_DESCR, 'ARR_HAS_DESCR')]:
        if flag & flags:
            names.append(name)
    return ', '.join(names)

def format_shape(nd, shape):
    return ', '.join([str(shape[i]) for i in range(nd)])

def format_strides(nd, strides):
    return ', '.join([str(strides[i]) for i in range(nd)])
