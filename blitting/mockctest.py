"""Do some confirmation of the mock C data types module"""

# Requires Python 2.5

from mockc import Uint8, RangeError, Uint8P, malloc

a = Uint8()
try:
    a.get_value()
except ValueError:
    pass
else:
    assert False

a.set_value(42)
assert a == 42

b = a
a += 2
assert a is b
assert a == 44

a -= 10
assert a is b
assert a == 34

assert isinstance(a + 1, Uint8)
assert (a + 8) == 42
assert (8 + a) == 42

assert isinstance(a - 1, Uint8)
assert (a - 18) == 16
assert isinstance(100 - a, Uint8)
assert (34 - a) == 0

p = Uint8P.cast(malloc(2048))

p[0]
p[2047]
try:
    p[2048]
except RangeError:
    pass
else:
    assert False

q = p - 10
assert isinstance(q, Uint8P)
try:
    q[0]
except RangeError:
    pass
else:
    assert False

q = p + 1024
q[0]
try:
    q[1024]
except RangeError:
    pass
else:
    assert False
