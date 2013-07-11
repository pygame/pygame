import sys
from ctypes import *

__all__ = ['PAI_CONTIGUOUS', 'PAI_FORTRAN', 'PAI_ALIGNED',
           'PAI_NOTSWAPPED', 'PAI_WRITEABLE', 'PAI_ARR_HAS_DESCR',
           'ArrayInterface',]

PY3 = 0
if sys.version_info >= (3,):
    PY3 = 1

SIZEOF_VOID_P = sizeof(c_void_p)
if SIZEOF_VOID_P <= sizeof(c_int):
    Py_intptr_t = c_int
elif SIZEOF_VOID_P <= sizeof(c_long):
    Py_intptr_t = c_long
elif 'c_longlong' in globals() and SIZEOF_VOID_P <= sizeof(c_longlong):
    Py_intptr_t = c_longlong
else:
    raise RuntimeError("Unrecognized pointer size %i" % (pointer_size,))

class PyArrayInterface(Structure):
    _fields_ = [('two', c_int), ('nd', c_int), ('typekind', c_char),
                ('itemsize', c_int), ('flags', c_int),
                ('shape', POINTER(Py_intptr_t)),
                ('strides', POINTER(Py_intptr_t)),
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

if PY3:
    PyCapsule_Destructor = CFUNCTYPE(None, py_object)
    PyCapsule_New = pythonapi.PyCapsule_New
    PyCapsule_New.restype = py_object
    PyCapsule_New.argtypes = [c_void_p, c_char_p, POINTER(PyCapsule_Destructor)]
    def capsule_new(p):
        return PyCapsule_New(addressof(p), None, None)
else:
    PyCObject_Destructor = CFUNCTYPE(None, c_void_p)
    PyCObject_FromVoidPtr = pythonapi.PyCObject_FromVoidPtr
    PyCObject_FromVoidPtr.restype = py_object
    PyCObject_FromVoidPtr.argtypes = [c_void_p, POINTER(PyCObject_Destructor)]
    def capsule_new(p):
        return PyCObject_FromVoidPtr(addressof(p), None)

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
        if (name == 'typekind'):
            return self._inter.typekind.decode('latin-1')
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

class Exporter(object):
    def __init__(self, shape,
                 typekind=None, itemsize=None, strides=None,
                 descr=None, flags=None):
        if typekind is None:
            typekind = 'u'
        if itemsize is None:
            itemsize = 1
        if flags is None:
            flags = PAI_WRITEABLE | PAI_ALIGNED | PAI_NOTSWAPPED
        if descr is not None:
            flags |= PAI_ARR_HAS_DESCR
        if len(typekind) != 1:
            raise ValueError("Argument 'typekind' must be length 1 string")
        nd = len(shape)
        self.typekind = typekind
        self.itemsize = itemsize
        self.nd = nd
        self.shape = tuple(shape)
        self._shape = (c_ssize_t * self.nd)(*self.shape)
        if strides is None:
            self._strides = (c_ssize_t * self.nd)()
            self._strides[self.nd - 1] = self.itemsize
            for i in range(self.nd - 1, 0, -1):
                self._strides[i - 1] = self.shape[i] * self._strides[i]
            strides = tuple(self._strides)
            self.strides = strides
        elif len(strides) == nd:
            self.strides = tuple(strides)
            self._strides = (c_ssize_t * self.nd)(*self.strides)
        else:
            raise ValueError("Mismatch in length of strides and shape")
        self.descr = descr
        if self.is_contiguous('C'):
            flags |= PAI_CONTIGUOUS
        if self.is_contiguous('F'):
            flags |= PAI_FORTRAN
        self.flags = flags
        sz = max(shape[i] * strides[i] for i in range(nd))
        self._data = (c_ubyte * sz)()
        self.data = addressof(self._data)
        self._inter = PyArrayInterface(2, nd, typekind.encode('latin_1'),
                                       itemsize, flags, self._shape,
                                       self._strides, self.data, descr)
        self.len = itemsize
        for i in range(nd):
            self.len *= self.shape[i]

    __array_struct__ = property(lambda self: capsule_new(self._inter))

    def is_contiguous(self, fortran):
        if fortran in "CA":
            if self.strides[-1] == self.itemsize:
                for i in range(self.nd - 1, 0, -1):
                    if self.strides[i - 1] != self.shape[i] * self.strides[i]:
                        break
                else:
                    return True
        if fortran in "FA":
            if self.strides[0] == self.itemsize:
                for i in range(0, self.nd - 1):
                    if self.strides[i + 1] != self.shape[i] * self.strides[i]:
                        break
                else:
                    return True
        return False
