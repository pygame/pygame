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
    def resource_exists(_package_or_requirement, _resource_name):
        """
        A stub for when we fail to import this function.

        :return: Always returns False
        """
        return False

    def resource_stream(_package_of_requirement, _resource_name):
        """
        A stub for when we fail to import this function.

        Always raises a NotImplementedError when called.
        """
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
    
    # When pyinstaller (or similar tools) are used, resource_exists may raise NotImplemented error
    try:
        if resource_exists(pkgname, identifier):
            return resource_stream(pkgname, identifier)
    except NotImplementedError:
        pass

    mod = sys.modules[pkgname]
    path_to_file = getattr(mod, '__file__', None)
    if path_to_file is None:
        raise IOError("%s has no __file__!" % repr(mod))
    path = os.path.join(os.path.dirname(path_to_file), identifier)
    if sys.version_info < (3, 3):
        loader = getattr(mod, '__loader__', None)
        if loader is not None:
            try:
                data = loader.get_data(path)
            except IOError:
                pass
            else:
                return BytesIO(data)
    try:
        file = open(os.path.normpath(path), 'rb')
        return file
    except:
        pass

    def splitall(path):
        allparts = []
        while 1:
            parts = os.path.split(path)
            if parts[0] == path:  # sentinel for absolute paths
                allparts.insert(0, parts[0])
                break
            elif parts[1] == path: # sentinel for relative paths
                allparts.insert(0, parts[1])
                break
            else:
                path = parts[0]
                allparts.insert(0, parts[1])
        return allparts

    abs_path = path
     
    split_path = splitall(abs_path)

    if not split_path:
        raise IOError("Path is empty")

    reconstructed_path = split_path[0]  


    if len(split_path) == 1 or "zip" in split_path[-1]:
        raise IOError("Must be a valid zip path (example: file.zip/file.txt)")

    if "zip" in reconstructed_path:
        try:
            file_path = "/".join(split_path[i:])
            assets_path = "Assets/" + file_path
            file = open(os.path.normpath(assets_path), 'rb')
        except:
            raise IOError("Invalid zip or file contained by it")        

    file = None

    for i in range(1, len(split_path) + 1):
        if "zip" in reconstructed_path:
            try:
                file_path = "/".join(split_path[i:])
                assets_path = "Assets/" + file_path
                file = open(os.path.normpath(assets_path), 'rb')
                break
            except:
                raise IOError("Invalid path")       
        reconstructed_path = reconstructed_path + "/" + split_path[i]

    return_file = file
    return return_file