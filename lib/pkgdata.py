"""
pkgdata is a simple, extensible way for a package to acquire data file 
resources.

The getResource function is equivalent to the standard idioms, such as
the following minimal implementation:
    
    import sys, os

    def getResource(identifier, pkgname=__name__):
        pkgpath = os.path.dirname(sys.modules[pkgname].__file__)
        path = os.path.join(pkgpath, identifier)
        return file(os.path.normpath(path), mode='rb')

When a __loader__ is present on the module given by __name__, it will defer
getResource to its get_data implementation and return it as a file-like
object (such as StringIO).
"""

__all__ = ['getResource']
import sys
import os
from pygame.compat import get_BytesIO
BytesIO = get_BytesIO()

try:
    from pkg_resources import resource_stream, resource_exists
except ImportError:
    def resource_exists(package_or_requirement, resource_name):
        return False
    def resource_stream(package_of_requirement, resource_name):
        raise NotImplementedError

def getResource(identifier, pkgname=__name__):
    """
    Acquire a readable object for a given package name and identifier.
    An IOError will be raised if the resource can not be found.

    For example:
        mydata = getResource('mypkgdata.jpg').read()

    Note that the package name must be fully qualified, if given, such
    that it would be found in sys.modules.

    In some cases, getResource will return a real file object.  In that
    case, it may be useful to use its name attribute to get the path
    rather than use it as a file-like object.  For example, you may
    be handing data off to a C API.
    """
    if resource_exists(pkgname, identifier):
        return resource_stream(pkgname, identifier)

    mod = sys.modules[pkgname]
    fn = getattr(mod, '__file__', None)
    if fn is None:
        raise IOError("%s has no __file__!" % repr(mod))
    path = os.path.join(os.path.dirname(fn), identifier)
    loader = getattr(mod, '__loader__', None)
    if loader is not None:
        try:
            data = loader.get_data(path)
        except IOError:
            pass
        else:
            return BytesIO(data)
    return open(os.path.normpath(path), 'rb')
