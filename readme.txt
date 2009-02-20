These are some tools used to validate parts of Pygame.

arrinter.py is a ctypes based module that wraps an objects array struct
interface in a Python class. It was useful in developing the
_arraysurfarray module.

Directory blitting contains several modules and programs used to check
Pygame's blitter algorithms. It contains mockc.py, a module of types
that emulate C integers and pointer, alphablit.py, a partial implementation
of alphablit.c using mockc, and blittest.py, a program that runs some
tests on alphablit.py.
