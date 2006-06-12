#!/usr/bin/env python

'''
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

from ctypes import *

_dll = cdll.SDL

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
    if type(v) == _SDL_version:
        return v.major, v.minor, v.patch
    elif type(v) == tuple:
        return v
    elif type(v) == string:
        return tuple([int(i) for i in v.split('.')])
    else:
        raise TypeError

def version_compatible(v):
    '''Returns True iff `v` is equal to or later than the loaded library
    version.'''
    v = _version_parts(v) 
    for i in range(3):
        if _version[i] < v[i]:
            return False
    return True

def assert_version_compatible(name, since):
    '''Raises an exception if `since` is later than the loaded library.'''
    if not version_compatible(version):
        import SDL.error
        raise SDL.error.SDL_NotImplementedError, \
            '%s requires SDL version %s; currently using version %s' % \
            (name, _version_string(since), _version_string(_version))

def _version_string(v):
    return '%d.%d.%d' % _version_parts(v)

# Get the version of the DLL we're using
_dll.SDL_Linked_Version.restype = POINTER(_SDL_version)
_version = _version_parts(_dll.SDL_Linked_Version().contents)

def private_function(name, **kwargs):
    '''Construct a wrapper function for ctypes with internal documentation
    and no argument names.'''
    kwargs['doc'] = 'Private wrapper for %s' % name
    kwargs['args'] = []
    return function(name, **kwargs)

def function(name, doc, args=[], arg_types=[], 
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
            The ctypes class giving the wrapped function's native return
            type.
        `dereference_return`
            If True, the return value is assumed to be a pointer and
            will be dereferenced via ``.contents`` before being returned
            to the user application.
        `require_return`
            Used in conjunction with `dereference_return`; if True, an
            exception will be raised if the result is NULL; if False
            None will be returned when the result is NULL.
        `success_return`
            If not None, the expected result of the wrapped function.  If
            the return value does not equal success_return, an exception
            will be raised.
        `error_return`
            If not None, the error result of the wrapped function.  If the
            return value equals error_return, an exception will be raised.
            Cannot be used in conjunction with `success_return`.
        `since`
            Tuple (major, minor, patch) or string 'x.y.z' of the first version
            of SDL in which this function appears.  If the loaded version
            predates it, a placeholder function that raises
            `SDL_NotImplementedError` will be returned instead.  Set to None
            if the function is in all versions of SDL.
    '''
    # Check for version compatibility first
    if since and not version_compatible(since):
        def _f(*args, **kwargs):
            import SDL.error
            raise SDL.error.SDL_NotImplementedError, \
                  '%s requires SDL version %s; currently using version %s' % \
                  (name, _version_string(since), _version_string(_version))
        if args:
            _f._args = args
        _f.__doc__ = doc
        _f.func_name = name
        return _f

    # Ok, get function from ctypes
    func = getattr(_dll, name)
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
    _f.func_name = name
    return _f

