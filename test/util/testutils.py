##
## This file is placed under the public domain.
##

import os

try:
    getinput = raw_input
except NameError:
    getinput = input

def getanswer (question):
    answer = getinput ("%s [y/n]: " % question)
    return answer.lower ().strip () == 'y'

def doprint (text):
    getinput ("%s (press enter to continue) " % text)

class interactive (object):
    """Simple interactive question decorator for unit test methods.
    """
    def __init__ (self, question):
        self.question = question

    def __call__ (self, func):
        def wrapper (*fargs, **kw):
            if fargs and getattr (fargs[0], "__class__", None):
                instance = fargs[0]
                funcargs = fargs[1:]
                print (os.linesep)
                func (instance, *funcargs, **kw)
                if not getanswer (self.question):
                    instance.fail ()

        wrapper.__name__ = func.__name__
        wrapper.__dict__.update (func.__dict__)
        wrapper.__tags__ = [ 'interactive' ]
        return wrapper

class ignore (object):
    """Simple ignore flag decorator."""
    def __init__ (self, func):
        self.func = func
    
    def __call__ (self):
        pass
