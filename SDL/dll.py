#!/usr/bin/env python

'''
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

from ctypes import *
from ctypes.util import find_library
import sys

# Private version checking declared before SDL.version can be
# imported.
class _SDL_version(Structure):
    _fields_ = [('major', c_ubyte),
                ('minor', c_ubyte),
                ('patch', c_ubyte)]

    def __repr__(self):
        return '%d.%d.%d' % \
            (self.major, self.minor, self.patch)

def _version_parts(v):
    '''Return a tuple (major, minor, patch) for `v`, which can be
    an _SDL_version, string or tuple.'''
    if hasattr(v, 'major') and hasattr(v, 'minor') and hasattr(v, 'patch'):
        return v.major, v.minor, v.patch
    elif type(v) == tuple:
        return v
    elif type(v) == str:
        return tuple([int(i) for i in v.split('.')])
    else:
        raise TypeError

def _version_string(v):
    return '%d.%d.%d' % _version_parts(v)

def _platform_library_name(library):
    if sys.platform[:5] == 'linux':
        return 'lib%s.so' % library
    elif sys.platform == 'darwin':
        return '%s.framework' % library
    elif sys.platform == 'windows':
        return '%s.dll' % library
    return library

class SDL_DLL:
    def __init__(self, library_name, version_function_name):
        self.library_name = library_name
        library = find_library(library_name)
        if not library:
            raise ImportError, 'Dynamic library "%s" was not found' % \
                _platform_library_name(library_name)
        self._dll = getattr(cdll, library)
        
        # Get the version of the DLL we're using
        if version_function_name:
            try:
                version_function = getattr(self._dll, version_function_name)
                version_function.restype = POINTER(_SDL_version)
                self._version = _version_parts(version_function().contents)
            except AttributeError:
                self._version = (0, 0, 0)
        else:
            self._version = (0, 0, 0)

    def version_compatible(self, v):
        '''Returns True iff `v` is equal to or later than the loaded library
        version.'''
        v = _version_parts(v) 
        for i in range(3):
            if self._version[i] < v[i]:
                return False
        return True

    def assert_version_compatible(self, name, since):
        '''Raises an exception if `since` is later than the loaded library.'''
        if not version_compatible(since):
            import SDL.error
            raise SDL.error.SDL_NotImplementedError, \
                '%s requires SDL version %s; currently using version %s' % \
                (name, _version_string(since), _version_string(self._version))



    def private_function(self, name, **kwargs):
        '''Construct a wrapper function for ctypes with internal documentation
        and no argument names.'''
        kwargs['doc'] = 'Private wrapper for %s' % name
        kwargs['args'] = []
        return self.function(name, **kwargs)

    def function(self, name, doc, args=[], arg_types=[], 
                 return_type=None, 
                 dereference_return=False, 
                 require_return=False,
                 success_return=None,
                 error_return=None,
                 since=None):
        '''Construct a wrapper function for ctypes.

        :Parameters:
            `name`
                The name of the function as it appears in the shared library.
            `doc`
                Docstring to associate with the wrapper function.
            `args`
                List of strings giving the argument names.
            `arg_types`
                List of ctypes classes giving the argument types.
            `return_type`
                The ctypes class giving the wrapped function's native
                return type.
            `dereference_return`
                If True, the return value is assumed to be a pointer and
                will be dereferenced via ``.contents`` before being
                returned to the user application.
            `require_return`
                Used in conjunction with `dereference_return`; if True, an
                exception will be raised if the result is NULL; if False
                None will be returned when the result is NULL.
            `success_return`
                If not None, the expected result of the wrapped function.
                If the return value does not equal success_return, an
                exception will be raised.
            `error_return`
                If not None, the error result of the wrapped function.  If
                the return value equals error_return, an exception will be
                raised.  Cannot be used in conjunction with
                `success_return`.
            `since`
                Tuple (major, minor, patch) or string 'x.y.z' of the first
                version of SDL in which this function appears.  If the
                loaded version predates it, a placeholder function that
                raises `SDL_NotImplementedError` will be returned instead.
                Set to None if the function is in all versions of SDL.

        '''
        # Check for version compatibility first
        if since and not self.version_compatible(since):
            def _f(*args, **kwargs):
                import SDL.error
                raise SDL.error.SDL_NotImplementedError, \
                      '%s requires %s %s; currently using version %s' % \
                      (name, self.library_name, _version_string(since), 
                       _version_string(self._version))
            if args:
                _f._args = args
            _f.__doc__ = doc
            try:
                _f.func_name = name
            except TypeError: # read-only in Python 2.3
                pass
            return _f

        # Ok, get function from ctypes
        func = getattr(self._dll, name)
        func.argtypes = arg_types
        func.restype = return_type
        if dereference_return:
            if require_return:
                # Construct a function which dereferences the pointer result,
                # or raises an exception if NULL is returned.
                def _f(*args, **kwargs):
                    result = func(*args, **kwargs)
                    if result:
                        return result.contents
                    import SDL.error
                    raise SDL.error.SDL_Exception, SDL.error.SDL_GetError()
            else:
                # Construct a function which dereferences the pointer result,
                # or returns None if NULL is returned.
                def _f(*args, **kwargs):
                    result = func(*args, **kwargs)
                    if result:
                        return result.contents
                    return None
        elif success_return is not None:
            # Construct a function which returns None, but raises an exception
            # if the C function returns a failure code.
            def _f(*args, **kwargs):
                result = func(*args, **kwargs)
                if result != success_return:
                    import SDL.error
                    raise SDL.error.SDL_Exception, SDL.error.SDL_GetError()
                return result
        elif error_return is not None:
            # Construct a function which returns None, but raises an exception
            # if the C function returns a failure code.
            def _f(*args, **kwargs):
                result = func(*args, **kwargs)
                if result == error_return:
                    import SDL.error
                    raise SDL.error.SDL_Exception, SDL.error.SDL_GetError()
                return result
        elif require_return:
            # Construct a function which returns the usual result, or returns
            # None if NULL is returned.
            def _f(*args, **kwargs):
                result = func(*args, **kwargs)
                if not result: 
                    import SDL.error
                    raise SDL.error.SDL_Exception, SDL.error.SDL_GetError()
                return result
        else:
            # Construct a function which returns the C function's return
            # value.
            def _f(*args, **kwargs):
                return func(*args, **kwargs)
        if args:
            _f._args = args
        _f.__doc__ = doc
        try:
            _f.func_name = name
        except TypeError: # read-only in Python 2.3
            pass
        return _f

# Shortcuts to the SDL core library
_dll = SDL_DLL('SDL', 'SDL_Linked_Version')
version_compatible = _dll.version_compatible
assert_version_compatible = _dll.assert_version_compatible
private_function = _dll.private_function
function = _dll.function
