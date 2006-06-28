#!/usr/bin/env python

'''
Mangle the SDL module into something a documentation generator can
handle.  Basically this means rewriting all anonymous functions as
named functions.  Imports are also removed so everything appears in
both SDL/__init__.py and the module where it's defined (e.g.,
SDL/video.py).  

The end result is hopefully something that looks a lot closer to
what the programmer sees, not what Python sees.  This is quite hacky
and SDL specific at the moment.

Usage:
  # Generate source code for SDL module in build_doc:
  python support/prep_doc.py build_doc/

This script is called automatically from setup.py.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

import ctypes
import inspect
import os.path
import sys
import StringIO

base_dir = os.path.abspath(sys.argv[1])
modules = {}
done_modules = []

def write_function(func, file):
    if hasattr(func, '__doc__'):
        docstring = func.__doc__
    else:
        docstring = ''
    if hasattr(func, '_args'):
        args = func._args
    else:
        args, foo, foo, defaults = inspect.getargspec(func)
        defaults = defaults or []
        nondefault = len(args) - len(defaults)
        args = args[:nondefault] + \
            ['%s=%s' % (args[i+nondefault], defaults[i]) \
                for i in range(len(defaults))]
    print >> file, 'def %s(%s):' % (func.__name__, ','.join(args))
    print >> file, "    '''%s'''" % docstring
    print >> file

def write_class(cls, file):
    if ctypes.Structure in cls.__bases__ or \
       ctypes.Union in cls.__bases__:
        print >> file, 'class %s:' % cls.__name__
        print >> file, '    %s' % repr(cls.__doc__ or '')
        for field in cls._fields_:
            if field[0] != '_':
                print >> file, '    %s = None' % field[0]
    else:
        print >> file, '%s' % inspect.getsource(cls)

def write_variable(child_name, child, file):
    print >> file, '%s = %s' % (child_name, repr(child))

def write_module(module):
    done_modules.append(module)
    f = module_file(module.__name__)
    if not f:
        return

    print >> f, "'''%s'''\n" % module.__doc__
    print >> f, '__docformat__ = "restructuredtext"'
    for child_name in dir(module):
        # Ignore privates
        if child_name[:2] == '__':
            continue
        
        child = getattr(module, child_name)
        child_module = inspect.getmodule(child) or module
        if child_module is not module and child_module is not SDL.dll: 
            if child_module not in done_modules:
                write_module(child_module)
            if module is not SDL:
                continue
        if child_module.__name__[:3] != 'SDL':
            continue

        if inspect.isfunction(child) and child_name[0] != '_':
            write_function(child, f)
        elif inspect.isclass(child):
            write_class(child, f)
        elif inspect.ismodule(child):
            pass
        elif module in (SDL, SDL.constants):
            write_variable(child_name, child, f)

def module_file(module_name):
    if module_name[:3] != 'SDL' or module_name in ('SDL.dll', ):
        return None

    if module_name in modules:
        return modules[module_name]
    else:
        f = open(os.path.join(base_dir, 
                              '%s.py' % module_name.replace('.', '/')), 'w')
        modules[module_name] = f
        return f

if __name__ == '__main__':
    try:
        os.makedirs(os.path.join(base_dir, 'SDL'))
    except:
        pass
    modules['SDL'] = open(os.path.join(base_dir, 'SDL/__init__.py'), 'w')
    import SDL
    import SDL.ttf
    import SDL.mixer
    write_module(SDL)
    write_module(SDL.ttf)
    write_module(SDL.mixer)
