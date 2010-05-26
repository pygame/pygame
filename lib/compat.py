##    pygame - Python Game Library
##    Copyright (C) 2009 Marcus von Appen
##
##    This library is free software; you can redistribute it and/or
##    modify it under the terms of the GNU Library General Public
##    License as published by the Free Software Foundation; either
##    version 2 of the License, or (at your option) any later version.
##
##    This library is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
##    Library General Public License for more details.
##
##    You should have received a copy of the GNU Library General Public
##    License along with this library; if not, write to the Free
##    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
##

##
## This file is placed under the Public Domain.
##

"""Simple deprecation helper and compatibility functions."""

import warnings

class deprecated (object):
    """deprecated () -> decorator

    A simple decorator to mark functions and methods as deprecated.
    """
    def __init__ (self):
        pass

    def __call__ (self, func):
        def wrapper (*fargs, **kw):
            warnings.warn ("%s is deprecated." % func.__name__,
                           category=DeprecationWarning, stacklevel=2)
            return func (*fargs, **kw)
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__dict__.update (func.__dict__)

def deprecation (message):
    """deprecation (message) -> None
    
    Prints a deprecation message.
    """
    warnings.warn (message, category=DeprecationWarning, stacklevel=2)
