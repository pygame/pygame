import sys

def getversion ():
    return sys.version_info[0:3]

def geterror ():
    return sys.exc_info()[1]
