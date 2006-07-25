#!/usr/bin/env python

'''Pygame object for storing rectangular coordinates.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id$'

import copy

import SDL.video

class _RectProxy(object):
    '''Proxy for SDL_Rect that can handle negative size.'''

    __slots__ = ['x', 'y', 'w', 'h']

    def __init__(self, r):
        if isinstance(r, SDL.SDL_Rect) or isinstance(r, Rect):
            self.x = r.x
            self.y = r.y
            self.w = r.w
            self.h = r.h
        else:
            self.x = r[0]
            self.y = r[1]
            self.w = r[2]
            self.h = r[3]

    def _get_as_parameter_(self):
        return SDL.SDL_Rect(self.x, self.y, self.w, self.h)

    _as_parameter_ = property(_get_as_parameter_)

class Rect(object):
    __slots__ = ['_r']

    def __init__(self, *args):
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, Rect):
                object.__setattr__(self, '_r', copy.copy(arg._r))
                return
            elif isinstance(arg, SDL.SDL_Rect):
                object.__setattr__(self, '_r', copy.copy(arg))
                return
            elif hasattr(arg, 'rect'):
                arg = arg.rect
                if callable(arg):
                    arg = arg()
                self.__init__(arg)
                return
            elif hasattr(arg, '__len__'):
                args = arg
            else:
                raise TypeError, 'Argument must be rect style object'
        if len(args) == 4:
            if args[2] < 0 or args[3] < 0:
                object.__setattr__(self, '_r', _RectProxy((int(args[0]),
                                                          int(args[1]),
                                                          int(args[2]),
                                                          int(args[3]))))
            else:
                object.__setattr__(self, '_r', SDL.SDL_Rect(int(args[0]),
                                                            int(args[1]),
                                                            int(args[2]),
                                                            int(args[3])))
        elif len(args) == 2:
            if args[1][0] < 0 or args[1][1] < 0:
                object.__setattr__(self, '_r', 
                                   _RectProxy((int(args[0][0]), 
                                              int(args[0][1]), 
                                              int(args[1][0]), 
                                              int(args[1][1]))))
            else:
                object.__setattr__(self, '_r', 
                                   SDL.SDL_Rect(int(args[0][0]), 
                                                int(args[0][1]), 
                                                int(args[1][0]), 
                                                int(args[1][1])))
        else:
            raise TypeError, 'Argument must be rect style object'

    def __copy__(self):
        return Rect(self)

    def __repr__(self):
        return '<rect(%d, %d, %d, %d)>' % \
            (self._r.x, self._r.y, self._r.w, self._r.h)

    def __cmp__(self, *other):
        other = _rect_from_object(other)

        if self._r.x != other._r.x:
            return cmp(self._r.x, other._r.x)
        if self._r.y != other._r.y:
            return cmp(self._r.y, other._r.y)
        if self._r.w != other._r.w:
            return cmp(self._r.w, other._r.w)
        if self._r.h != other._r.h:
            return cmp(self._r.h, other._r.h)
        return 0

    def __nonzero__(self):
        return self._r.w != 0 and self._r.h != 0

    def __getattr__(self, name):
        if name == 'top':
            return self._r.y
        elif name == 'left':
            return self._r.x
        elif name == 'bottom':
            return self._r.y + self._r.h
        elif name == 'right':
            return self._r.x + self._r.w
        elif name == 'topleft':
            return self._r.x, self._r.y
        elif name == 'bottomleft':
            return self._r.x, self._r.y + self._r.h
        elif name == 'topright':
            return self._r.x + self._r.w, self._r.y
        elif name == 'bottomright':
            return self._r.x + self._r.w, self._r.y + self._r.h
        elif name == 'midtop':
            return self._r.x + self._r.w / 2, self._r.y
        elif name == 'midleft':
            return self._r.x, self._r.y + self._r.h / 2
        elif name == 'midbottom':
            return self._r.x + self._r.w / 2, self._r.y + self._r.h
        elif name == 'midright':
            return self._r.x + self._r.w, self._r.y + self._r.h / 2
        elif name == 'center':
            return self._r.x + self._r.w / 2, self._r.y + self._r.h / 2
        elif name == 'centerx':
            return self._r.x + self._r.w / 2
        elif name == 'centery':
            return self._r.y + self._r.h / 2
        elif name == 'size':
            return self._r.w, self._r.h
        elif name == 'width':
            return self._r.w
        elif name == 'height':
            return self._r.h
        else:
            raise AttributeError, name

    def __setattr__(self, name, value):
        if name == 'top' or name == 'y':
            self._r.y = value
        elif name == 'left' or name == 'x':
            self._r.x = value
        elif name == 'bottom':
            self._r.y = value - self._r.h
        elif name == 'right':
            self._r.x = value - self._r.w
        elif name == 'topleft':
            self._r.x, self._r.y = value
        elif name == 'bottomleft':
            self._r.x = value[0]
            self._r.y = value[1] - self._r.h
        elif name == 'topright':
            self._r.x = value[0] - self._r.w
            self._r.y = value[1]
        elif name == 'bottomright':
            self._r.x = value[0] - self._r.w
            self._r.y = value[1] - self._r.h
        elif name == 'midtop':
            self._r.x = value[0] - self._r.w / 2
            self._r.y = value[1]
        elif name == 'midleft':
            self._r.x = value[0]
            self._r.y = value[1] - self._r.h / 2
        elif name == 'midbottom':
            self._r.x = value[0] - self._r.w / 2
            self._r.y = value[1] - self._r.h
        elif name == 'midright':
            self._r.x = value[0] - self._r.w
            self._r.y = value[1] - self._r.h / 2
        elif name == 'center':
            self._r.x = value[0] - self._r.w / 2
            self._r.y = value[1] - self._r.h / 2
        elif name == 'centerx':
            self._r.x = value - self._r.w / 2
        elif name == 'centery':
            self._r.y = value - self._r.h / 2
        elif name == 'size':
            if value[0] < 0 or value[1] < 0:
                self._ensure_proxy()
            self._r.w, self._r.h = value
        elif name == 'width':
            if value < 0:
                self._ensure_proxy()
            self._r.w = value
        elif name == 'height':
            if value < 0:
                self._ensure_proxy()
            self._r.h = value
        else:
            raise AttributeError, name

    def _ensure_proxy(self):
        if not isinstance(self._r, _RectProxy):
            object.__setattr__(self, '_r', _RectProxy(self._r))

    def __len__(self):
        return 4

    def __getitem__(self, key):
        return (self._r.x, self._r.y, self._r.w, self._r.h)[key]

    def __setitem__(self, key, value):
        r = [self._r.x, self._r.y, self._r.w, self._r.h]
        r[key] = value
        self._r.x, self._r.y, self._r.w, self._r.h = r

    def __coerce__(self, *other):
        try:
            return self, Rect(*other)
        except TypeError:
            return None

    def move(self, *pos):
        x, y = _two_ints_from_args(pos)
        return Rect(self._r.x + x, self._r.y + y, self._r.w, self._r.h)

    def move_ip(self, *pos):
        x, y = _two_ints_from_args(pos)
        self._r.x += x
        self._r.y += y

    def inflate(self, x, y):
        return Rect(self._r.x - x / 2, self._r.y - y / 2, 
                    self._r.w + x, self._r.h + y)

    def inflate_ip(self, x, y):
        self._r.x -= x / 2
        self._r.y -= y / 2
        self._r.w += x
        self._r.h += y

    def clamp(self, *other):
        r = Rect(self)
        r.clamp_ip(*other)
        return r

    def clamp_ip(self, *other):
        other = _rect_from_object(other)._r
        if self._r.w >= other.w:
            x = other.x + other.w / 2 - self._r.w / 2
        elif self._r.x < other.x:
            x = other.x
        elif self._r.x + self._r.w > other.x + other.w:
            x = other.x + other.w - self._r.w
        else:
            x = self._r.x

        if self._r.h >= other.h:
            y = other.y + other.h / 2 - self._r.h / 2
        elif self._r.y < other.y:
            y = other.y
        elif self._r.y + self._r.h > other.y + other.h:
            y = other.y + other.h - self._r.h
        else:
            y = self._r.y

        self._r.x, self._r.y = x, y

    def clip(self, *other):
        r = Rect(self)
        r.clip_ip(*other)
        return r

    def clip_ip(self, *other):
        other = _rect_from_object(other)._r
        x = max(self._r.x, other.x)
        w = min(self._r.x + self._r.w, other.x + other.w) - x
        y = max(self._r.y, other.y)
        h = min(self._r.y + self._r.h, other.y + other.h) - y

        if w <= 0 or h <= 0:
            self._r.w, self._r.h = 0, 0
        else:
            self._r.x, self._r.y, self._r.w, self._r.h = x, y, w, h

    def union(self, *other):
        r = Rect(self)
        r.union_ip(*other)
        return r

    def union_ip(self, *other):
        other = _rect_from_object(other)._r
        x = min(self._r.x, other.x)
        y = min(self._r.y, other.y)
        w = max(self._r.x + self._r.w, other.x + other.w) - x
        h = max(self._r.y + self._r.h, other.y + other.h) - y
        self._r.x, self._r.y, self._r.w, self._r.h = x, y, w, h

    def unionall(self, others):
        r = Rect(self)
        r.unionall_ip(others)
        return r

    def unionall_ip(self, others):
        l = self._r.x
        r = self._r.x + self._r.w
        t = self._r.y
        b = self._r.y + self._r.h
        for other in others:
            other = _rect_from_object(other)._r
            l = min(l, other.x)
            r = max(r, other.x + other.w)
            t = min(t, other.y)
            b = max(b, other.y + other.h)
        self._r.x, self._r.y, self._r.w, self._r.h = l, t, r - l, b - t

    def fit(self, *other):
        r = Rect(self)
        r.fit_ip(*other)
        return r
    
    def fit_ip(self, *other):
        other = _rect_from_object(other)._r

        xratio = self._r.w / float(other.w)
        yratio = self._r.h / float(other.h)
        maxratio = max(xratio, yratio)
        self._r.w = int(self._r.w / maxratio)
        self._r.h = int(self._r.h / maxratio)
        self._r.x = other.x + (other.w - self._r.w) / 2
        self._r.y = other.y + (other.h - self._r.h) / 2

    def normalize(self):
        if self._r.w < 0:
            self._r.x += self._r.w
            self._r.w = -self._r.w
        if self._r.h < 0:
            self._r.y += self._r.h
            self._r.h = -self._r.h
        if isinstance(self._r, _RectProxy):
            object.__setattr__(self, '_r', SDL.SDL_Rect(self._r.x,
                                                        self._r.y,
                                                        self._r.w,
                                                        self._r.h))

    def contains(self, *other):
        other = _rect_from_object(other)._r
        return self._r.x <= other.x and \
               self._r.y <= other.y and \
               self._r.x + self._r.w >= other.x + other.w and \
               self._r.y + self._r.h >= other.y + other.h and \
               self._r.x + self._r.w > other.x and \
               self._r.y + self._r.h > other.y

    def collidepoint(self, x, y):
        return x >= self._r.x and \
               y >= self._r.y and \
               x < self._r.x + self._r.w and \
               y < self._r.y + self._r.h
    
    def colliderect(self, *other):
        return _rect_collide(self._r, _rect_from_object(other)._r)
        
    def collidelist(self, others):
        for i in range(len(others)):
            if _rect_collide(self._r, _rect_from_object(others[i])._r):
                return i
        return -1

    def collidelistall(self, others):
        matches = []
        for i in range(len(others)):
            if _rect_collide(self._r, _rect_from_object(others[i])._r):
                matches.append(i)
        return matches

    def collidedict(self, d):
        for key, other in d.items():
            if _rect_collide(self._r, _rect_from_object(other)._r):
                return key, other
        return None

    def collidedictall(self, d):
        matches = []
        for key, other in d.items():
            if _rect_collide(self._r, _rect_from_object(other)._r):
                matches.append((key, other))
        return matches

def _rect_from_object(obj):
    if isinstance(obj, Rect):
        return obj
    if type(obj) in (tuple, list):
        return Rect(*obj)
    else:
        return Rect(obj)

def _rect_collide(a, b):
    return a.x + a.w > b.x and b.x + b.w > a.x and \
           a.y + a.h > b.y and b.y + b.h > a.y

def _two_ints_from_args(arg):
    if len(arg) == 1:
        return _two_ints_from_args(arg[0])
    else:
        return arg[:2]
