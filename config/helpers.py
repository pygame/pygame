import sys, traceback

def getversion ():
    return sys.version_info[0:3]

def geterror ():
    return sys.exc_info()[1]

def gettraceback ():
    return traceback.format_exc ()
