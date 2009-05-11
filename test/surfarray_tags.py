__tags__ = ['array']

exclude = False

try:
    import numpy
except ImportError:
    try:
        import Numeric
        import pygame._numericsurfarray
        import pygame._arraysurfarray
    except ImportError:
        exclude = True
else:
    try:
        import pygame._arraysurfarray
    except ImportError:
        exclude = True

if exclude:
    __tags__.extend(('ignore', 'subprocess_ignore'))
