import sys
from ctypes import *

__all__ = ['PAI_CONTIGUOUS', 'PAI_FORTRAN', 'PAI_ALIGNED',
           'PAI_NOTSWAPPED', 'PAI_WRITEABLE', 'PAI_ARR_HAS_DESCR',
           'ArrayInterface',]

PY3 = 0
if sys.version_info >= (3,):
    PY3 = 1

class PyArrayInterface(Structure):
    _fields_ = [('two', c_int), ('nd', c_int), ('typekind', c_char),
                ('itemsize', c_int), ('flags', c_int),
                ('shape', POINTER(c_int)),
                ('strides', POINTER(c_int)),
                ('data', c_void_p), ('descr', py_object)]

PAI_Ptr = POINTER(PyArrayInterface)
try:
    PyCObject_AsVoidPtr = pythonapi.PyCObject_AsVoidPtr
except AttributeError:
    def PyCObject_AsVoidPtr(o):
        raise TypeError("Not available")
else:
    PyCObject_AsVoidPtr.restype = c_void_p
    PyCObject_AsVoidPtr.argtypes = [py_object]
    PyCObject_GetDesc = pythonapi.PyCObject_GetDesc
    PyCObject_GetDesc.restype = c_void_p
    PyCObject_GetDesc.argtypes = [py_object]
try:
    PyCapsule_IsValid = pythonapi.PyCapsule_IsValid
except AttributeError:
    def PyCapsule_IsValid(capsule, name):
        return 0
else:
    PyCapsule_IsValid.restype = c_int
    PyCapsule_IsValid.argtypes = [py_object, c_char_p]
    PyCapsule_GetPointer = pythonapi.PyCapsule_GetPointer
    PyCapsule_GetPointer.restype = c_void_p
    PyCapsule_GetPointer.argtypes = [py_object, c_char_p]
    PyCapsule_GetContext = pythonapi.PyCapsule_GetContext
    PyCapsule_GetContext.restype = c_void_p
    PyCapsule_GetContext.argtypes = [py_object]

PAI_CONTIGUOUS = 0x01
PAI_FORTRAN = 0x02
PAI_ALIGNED = 0x100
PAI_NOTSWAPPED = 0x200
PAI_WRITEABLE = 0x400
PAI_ARR_HAS_DESCR = 0x800

class ArrayInterface(object):
    def __init__(self, arr):
        try:
            self._cobj = arr.__array_struct__
        except AttributeError:
            raise TypeError("The array object lacks an array structure")
        if not self._cobj:
            raise TypeError("The array object has a NULL array structure value")
        try:
            vp = PyCObject_AsVoidPtr(self._cobj)
        except TypeError:
            if PyCapsule_IsValid(self._cobj, None):
                vp = PyCapsule_GetPointer(self._cobj, None)
            else:
                raise TypeError("The array object has an invalid array structure")
            self.desc = PyCapsule_GetContext(self._cobj)
        else:
            self.desc = PyCObject_GetDesc(self._cobj)
        self._inter = cast(vp, PAI_Ptr)[0]

    def __getattr__(self, name):
        return getattr(self._inter, name)

    def __str__(self):
        if isinstance(self.desc, tuple):
            ver = self.desc[0]
        else:
            ver = "N/A"
        return ("nd: %i\n"
                "typekind: %s\n"
                "itemsize: %i\n"
                "flags: %s\n"
                "shape: %s\n"
                "strides: %s\n"
                "ver: %s\n" %
                (self.nd, self.typekind, self.itemsize,
                 format_flags(self.flags),
                 format_shape(self.nd, self.shape),
                 format_strides(self.nd, self.strides), ver))

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
