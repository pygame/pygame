"""Mock C data types for range checking

This module contains integer and fake C pointer types as well as
a struct. The integer types can be used to check for overflow
and underflow. The pointer types do bounds checking on addresses.

"""

# Requires Python 2.5

class Undef(object):
    pass
undef = Undef()

class UndefinedError(ValueError):
    pass

class Cell(object):
    def __init__(self, value=undef):
        if value is undef:
            self._value = undef
        else:
            self.set_value(value)
    def get_value(self):
        if self._value is undef:
            raise UndefinedError("Undefined value")
        return self._value
    def set_value(self, value):
        if isinstance(value, Cell):
            value = value.get_value()
        self._value = value
    def del_value(self):
        self._value = undef
    def copy(self):
        cls = self.__class__
        obj = cls.__new__(cls)
        obj.__dict__.update(self.__dict__)
        return obj

class RangeError(ValueError):
    pass

class Range(Cell):
    def __init__(self, minimum, maximum, value=undef):
        if isinstance(minimum, Range):
            minimum = minimum.get_value()
        if isinstance(maximum, Range):
            maximum = maximum.get_value()
        if maximum < minimum:
            return ValueError("upper limit is less than minimum: %i", maximum)
        self._minimum = minimum
        self._maximum = maximum
        if value is undef:
            Cell.__init__(self)
        else:
            self.set_value(value)
    def set_value(self, value):
        if isinstance(value, Cell):
            value = value.get_value()
        if not (self._minimum <= value <= self._maximum):
            raise RangeError("%s value %i out of range (%i, %i)" %
                             (self.__class__.__name__,
                              value, self._minimum, self._maximum))
        Cell.set_value(self, value)
    def __str__(self):
        try:
            return str(self.get_value())
        except UndefinedError:
            return "**undefined**"
    def __repr__(self):
        try:
            return ("%s(%i, %i, %i)" %
                    (self.__class__.__name__,
                     self._minimum, self._maximum, self.get_value()))
        except UndefinedError:
            return ("%s(%i, %i)" %
                    (self.__class__.__name__, self._minimum, self._maximum))

class int_(Range):
    def __iadd__(self, other):
        if isinstance(other, int_):
            other = other.get_value()
        if not isinstance(other, (int, long)):
            return NotImplemented
        self.set_value(self.get_value() + other)
        return self
    def __add__(self, other):
        return self.copy().__iadd__(other)
    def __radd__(self, other):
        return self.__add__(other)
    def __imul__(self, other):
        if isinstance(other, int_):
            other = other.get_value()
        if not isinstance(other, (int, long)):
            return NotImplemented
        self.set_value(self.get_value() * other)
        return self
    def __mul__(self, other):
        return self.copy().__imul__(other)
    def __rmul__(self, other):
        return self.__mul__(other)
    def __nonzero__(self):
        return bool(self.get_value())
    def __floordiv__(self, other):
        if isinstance(other, int_):
            other = other.get_value()
        if not isinstance(other, (int, long)):
            return NotImplemented
        obj = self.copy()
        obj.set_value(self.get_value() // other)
        return obj
    def __rfloordiv__(self, other):
        if isinstance(other, int_):
            other = other.get_value()
        if not isinstance(other, (int, long)):
            return NotImplemented
        obj = self.copy()
        obj.set_value(other // self.get_value())
        return obj
    def __iand__(self, other):
        if isinstance(other, int_):
            other = other.get_value()
        if not isinstance(other, (int, long)):
            return NotImplemented
        self.set_value(self.get_value() & other)
        return self
    def __and__(self, other):
        return self.copy().__iand__(other)
    def __ior__(self, other):
        if isinstance(other, int_):
            other = other.get_value()
        if not isinstance(other, (int, long)):
            return NotImplemented
        self.set_value(self.get_value() | other)
        return self
    def __or__(self, other):
        return self.copy().__ior__(other)
    def __rand__(self, other):
        return self.__and__(other)
    def __eq__(self, other):
        if isinstance(other, int_):
            other = other.get_value()
        return self.get_value() == other
    def __gt__(self, other):
        if isinstance(other, int_):
            other = other.get_value()
        return self.get_value() > other
    def __ge__(self, other):
        if isinstance(other, int_):
            other = other.get_value()
        return self.get_value() >= other
    def __isub__(self, other):
        if isinstance(other, int_):
            other = other.get_value()
        if not isinstance(other, (int, long)):
            return NotImplemented
        self.set_value(self.get_value() - other)
        return self
    def __sub__(self, other):
        return self.copy().__isub__(other)
    def __rsub__(self, other):
        if isinstance(other, int_):
            other = other.get_value()
        if not isinstance(other, (int, long)):
            return NotImplemented
        obj = self.copy()
        obj.set_value(other - self.get_value())
        return obj
    def __neg__(self):
        obj = self.copy()
        obj.set_value(-self.get_value())
        return obj
    def __imod__(self, other):
        if isinstance(other, int_):
            other = other.get_value()
        if not isinstance(other, (int, long)):
            return NotImplemented
        self.set_value(self.get_value % other)
        return self
    def __mod__(self, other):
        return self.copy().__imod__(other)
    def __rmod__(self, other):
        if isinstance(other, int_):
            other = other.get_value()
        if not isinstance(other, (int, long)):
            return NotImplemented
        obj = self.copy()
        obj.set_value(other % self.get_value())
        return obj
    def __index__(self):
        return self.get_value()

class intX(int_):
    def __repr__(self):
        try:
            return "%s(%i)" % (self.__class__.__name__, self.get_value())
        except UndefinedError:
            return "%s()" % self.__class__.__name__

class Uint8(intX):
    def __init__(self, value=undef):
        intX.__init__(self, 0, 255, value)

class Sint16(intX):
    def __init__(self, value=undef):
        intX.__init__(self, -0x8000, 0x7FFF, value)

class Uint16(intX):
    def __init__(self, value=undef):
        intX.__init__(self, 0, 0xFFFF, value)

class Uint32(intX):
    def __init__(self, value=undef):
        intX.__init__(self, 0, 0xFFFFFFFFL, value)

class posint_(intX):
    def __init__(self, value=undef):
        intX.__init__(self, 0, 0x7FFFFFFF, value)

class VoidP(Range):
    def __init__(self, lower, upper, address=undef):
        if not (0 <= lower <= 0xFFFFFFFFL):
            raise ValueError("lower bounds out of range: %i" % lower)
        if not (lower <= upper <= 0xFFFFFFFFL):
            raise ValueError("Upper bounds out of range: %i" % upper)
        Range.__init__(self, 0, 0xFFFFFFFFL, address)
        self._lower = lower
        self._upper = upper
    def clamped_lower(self):
        obj = self.copy()
        obj._lower = self.get_value()
        return obj
    def clamped_upper(self):
        obj = self.copy()
        obj._upper = self.get_value()
        return obj
    def set_value(self, value):
        Range.set_value(self, value)
        if isinstance(value, VoidP):
            lower = value._lower
            upper = value._upper
            if lower > self._lower:
                self._lower = lower
            if upper < self._upper:
                self._upper = upper
    @classmethod
    def cast(cls, other):
        return cls(other._lower, other._upper, other.get_value())
    def __getitem__(self, index):
        if index != 0:
            raise IndexError("Void pointer doesn't support non-zero index")
        if not (self._lower <= self.get_value() <= self._upper):
            raise RangeError("address %i outside of range (%i, %i)" %
                             (self._value, self._lower, self._value))
    def __repr__(self):
        try:
            return ("%s(%i, %i, %i)" %
                    (self.__class__.__name__,
                     self._lower, self._upper, self.get_value()))
        except UndefinedError:
            return ("%s(%i, %i)" %
                    (self.__class__.__name__, self._lower, self._upper))
    def __eq__(self, other):
        if isinstance(other, VoidP):
            other = other.get_value()
        return self.get_value() == other
    def __gt__(self, other):
        if isinstance(other, VoidP):
            other = other.get_value()
        return self.get_value() > other
    def __ge__(self, other):
        if isinstance(other, VoidP):
            other = other.get_value()
        return self.get_value() >= other

class Uint8P(VoidP):
    item_size = 1
    def __iadd__(self, other):
        try:
            i = other.__index__()
        except AttributeError:
            return NotImplemented
        self.set_value(self.get_value() + (self.item_size * i))
        return self
    def __add__(self, other):
        return self.copy().__iadd__(other)
    def __radd__(self, other):
        return self.__add__(other)
    def __isub__(self, other):
        try:
            i = other.__index__()
        except AttributeError:
            return NotImplemented
        self.set_value(self.get_value() - (self.item_size * i))
        return self
    def __sub__(self, other):
        if isinstance(other, Uint8P):
            if other.item_size != self.item_size:
                raise TypeError("Other pointer has different item size")
            return (self.get_value() - other.get_value()) // self.item_size
        return self.copy().__isub__(other)
    def __getitem__(self, index):
        address = self.get_value() + self.item_size * index.__index__()
        if not (self._lower <= address <= self._upper):
            raise RangeError("address %i outside of range (%i, %i)" %
                             (address, self._lower, self._upper))
        return 1L << (8 * self.item_size - 2)

class Ref(Cell):
    def copy(self):
        return self.get_value()
    def __str__(self):
        return "&%s" % str(self.get_value())
    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, repr(self.get_value()))

class Struct(object):
    class Substruct(object):
        def __init__(self, struct):
            self._struct = struct
        def copy(self):
            return self._struct
        def set_value(self, value):
            self._struct.set_value(value)
        def __str__(self):
            return str(self._struct)
        def __repr__(self):
            return repr(self._struct)
    def __init__(self, **kwds):
        for name, value in kwds.items():
            if isinstance(value, Struct):
                value = self.Substruct(value)
            object.__setattr__(self, '_%s' % name, value)
    def set_value(self, other):
        for key, val in other.__dict__.iteritems():
            setattr(self, key, val)
    def copy(self):
        kwds = {}
        for name, attr in self.__dict__.iteritems():
            if isinstance(attr, self.Substruct):
                attr = attr.copy().copy()
            elif isinstance(attr, Ref):
                attr = Ref(attr.copy())
            kwds[name[1:]] = attr.copy()
        return self.__class__(**kwds)
    def __getattr__(self, name):
        return object.__getattribute__(self, '_%s' % name).copy()
    def __setattr__(self, name, value):
        object.__getattribute__(self, '_%s' % name).set_value(value)
    def __str__(self):
        return ("[%s]" %
                ", ".join("%s: %s" % (name[1:], val)
                          for name, val in self.__dict__.iteritems()))
    def __repr__(self):
        return ("Struct(%s)" %
                ", ".join("%s=%s" % (name[1:], repr(val))
                            for name, val in self.__dict__.iteritems()))

def malloc(size):
    address = 0x01000000
    return VoidP(address, address + size - 1, address)

