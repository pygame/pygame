__tags__ = ['array']

exclude = False

try:
    import numpy
except ImportError:
    try:
        import Numeric
        import pygame._numericsndarray
    except ImportError:
        exclude = True

if exclude:
    __tags__.extend(('ignore', 'subprocess_ignore'))
