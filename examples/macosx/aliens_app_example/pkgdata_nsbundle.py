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
