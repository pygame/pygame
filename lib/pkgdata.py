"""
pkgdata is a simple, extensible way for a package to acquire data file 
resources.

The implementation is lightweight, and is intended to be included *inside*
your Python packages, rather than as an external dependency.  Therefore,
it should always be imported absolutely, like this:
    
    from mypkg.pkgdata import getResource, getResourcePath

If used as an external dependency, the pkgname keyword argument MUST be
given!

It's extensible using PyProtocols.  Without a PyProtocols based extension,
the functions are equivalent to the standard idioms, such as the following
minimal implementation:
    
    import sys, os

    def getResourcePath(identifier, pkgname=__name__):
        pkgpath = os.path.dirname(sys.modules[pkgname].__file__)
        return os.path.join(pkgpath, identifier)

    def getResource(identifier, pkgname=__name__):
        return file(getResourcePath(pkgname, identifier), mode='rb')

Extension using PyProtocols is intended for the exclusive use of packaging 
software, such as py2exe, freeze, and bundlebuilder.  However, it would be
done like so:
    
    #
    # This is a Mac OS X / PyObjC / Bundlebuilder extension for pkgdata.
    #
    # data files are located in the resources folder as such:
    #   %(appName)s/Contents/Resources/pkgdata/%(pkgname)s/%(identifier)s
    #   

    import os
    from protocols import declareAdapter, protocolForURI
    from Foundation import NSBundle

    __apiversion__ = '0'
    BASEURI = 'org.undefined.pkgdata.' + __apiversion__

    def localProtocol(s):
        return protocolForURI(BASEURI + '.' + s)

    IPackageFileTuple = localProtocol('packagefiletuple')
    IReadableFileLike = localProtocol('readablefilelike')
    IReadableFilePath = localProtocol('readablefilepath')

    def pathForPackageFileTuple(pkgtuple, protocol):
        bndl = NSBundle.mainBundle()
        if bndl is None:
            raise IOError(2, 'No main bundle')
        pkgname, identifier = pkgtuple
        pkgname = '.'.join(pkgname.split('.')[:-1])
        dataPath = os.path.join(
            bndl.resourcePath(), 'pkgdata', pkgname, identifier
        )
        if not os.path.exists(dataPath):
            raise IOError(2, 'No such file or directory', dataPath)
        return dataPath
    declareAdapter(
        pathForPackageFileTuple,
        provides=[IReadableFilePath],
        forProtocols=[IPackageFileTuple],
    )


    def openFilePath(path, protocol):
        return file(path, mode='rb')
    declareAdapter(
        openFilePath,
        provides=[IReadableFileLike],
        forProtocols=[IReadableFilePath],
    )
"""

# ----------
#  Metadata
# ----------

__version__ = '0.1'
__apiversion__ = '0'
__all__ = ['getResourcePath', 'getResource']

# --------------
#  Dependencies 
# --------------

import sys, os

try:
    from protocols import adapt, protocolForURI, declareImplementation, declareAdapterForType
except ImportError:
    class NOT_GIVEN:
        pass

    def adapt(obj, protocol, factory=NOT_GIVEN, default=NOT_GIVEN):
        if default is not NOT_GIVEN:
            return default
        if factory is not NOT_GIVEN:
            return factory(obj, protocol)
        raise NotImplementedError("Can't adapt", obj, protocol)

    def protocolForURI(s):
        return s

    def declareImplementation(typ, instancesProvide=(), instancesDoNotProvide=()):
        pass

    def declareAdapterForType(protocol, adapter, typ, depth=1):
        pass

# -----------
#  Protocols 
# -----------

# we include the version so that we can provide compatibility
BASEURI = 'org.undefined.pkgdata.' + __apiversion__
def localProtocol(s):
    return protocolForURI(BASEURI + '.' + s)

# the actual protocols
IPackageFileTuple = localProtocol('packagefiletuple')
IReadableFileLike = localProtocol('readablefilelike')
IReadableFilePath = localProtocol('readablefilepath')

# ----------------
#  Implementation
# ----------------

class packageFileTuple(tuple):
    """
    Tuple subclass, should be limited to two elements:
        package_name:
            a fully qualified sys.modules package name
        identifier:
            the local path of a data file
    """
declareImplementation(packageFileTuple, instancesProvide=(IPackageFileTuple,))

def resFactory((pkgname, identifier), protocol):
    try:
        pkgpath = os.path.dirname(sys.modules[pkgname].__file__)
    except:
        raise IOError(2, 'Not a valid package name', pkgname)

    fullpath = os.path.join(pkgpath, identifier)
    if not os.path.exists(fullpath):
        raise IOError(2, 'No such file or directory', fullpath)

    if protocol is IReadableFilePath:
        return fullpath
    elif protocol is IReadableFileLike:
        return file(fullpath, mode='rb')
    else:
        raise NotImplementedError("This should never, ever, happen.")

# -----
#  API
# -----

def getResource(identifier, pkgname=__name__):
    """
    Acquire a readable object for a given package name and identifier.
    An IOError will be raised if the resource can not be found.

    For example:
        mydata = getResource('mypkgdata.jpg').read()

    Note that the package name must be fully qualified, if given, such
    that it would be found in sys.modules.
    """
    
    obj = packageFileTuple((pkgname, identifier))
    return adapt(obj, IReadableFileLike, factory=resFactory)

def getResourcePath(identifier, pkgname=__name__):
    """
    Acquire a full path name for a given package name and identifier.
    An IOError will be raised if the resource can not be found.

    For example:
        mydatafile = getResourcePath('mypkgdata.jpg')

    Note that the package name must be fully qualified, if given, such
    that it would be found in sys.modules.    

    Also note that getResource(...) is preferable, as it is likely that
    some resource acquisition schemes may need to create a temporary
    file to perform this operation.
    """

    obj = packageFileTuple((pkgname, identifier))
    return adapt(obj, IReadableFilePath, factory=resFactory)
