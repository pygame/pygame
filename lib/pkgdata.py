"""
pkgdata is a simple, extensible way for a package to acquire data file 
resources.

The functions are equivalent to the standard idioms, such as the following
minimal implementation:
    
    import sys, os

    def getResourcePath(identifier, pkgname=__name__):
        pkgpath = os.path.dirname(sys.modules[pkgname].__file__)
        path = os.path.join(pkgpath, identifier)
        if not os.path.exists(path):
            raise IOError, "%r not found near %s" % (identifier, pkgname)

    def getResource(identifier, pkgname=__name__):
        return file(getResourcePath(pkgname, identifier), mode='rb')

When a __loader__ is present on the module given by __name__, it will defer
getResource to its get_data implementation.
"""

__all__ = ['getResourcePath', 'getResource']
import sys
import os
from cStringIO import StringIO

def getResource(identifier, pkgname=__name__):
    """
    Acquire a readable object for a given package name and identifier.
    An IOError will be raised if the resource can not be found.

    For example:
        mydata = getResource('mypkgdata.jpg').read()

    Note that the package name must be fully qualified, if given, such
    that it would be found in sys.modules.
    """

    mod = sys.modules[pkgname]
    fn = getattr(mod, '__file__', None)
    if fn is None:
        raise IOError, "%r has no __file__!"
    path = os.path.join(os.path.dirname(fn), identifier)
    loader = getattr(mod, '__loader__', None)
    if loader is not None:
        try:
            data = loader.get_data(path)
        except IOError:
            pass
        else:
            return StringIO(data)
    return file(path, 'rb')

def getResourcePath(identifier, pkgname=__name__):
    """
    Acquire a full path name for a given package name and identifier.
    An IOError will be raised if the resource can not be found.

    For example:
        mydatafile = getResourcePath('mypkgdata.jpg')

    Note that the package name must be fully qualified, if given, such
    that it would be found in sys.modules.    

    Also note that getResource(...) is preferable, as it will work
    more often.
    """
    
    mod = sys.modules[pkgname]
    fn = getattr(mod, '__file__', None)
    if fn is None:
        raise IOError, "%r has no __file__!"
    path = os.path.join(os.path.dirname(fn), identifier)
    if not os.path.exists(path):
        raise IOError, "%r not found near %r" % (identifier, mod)
    return path
