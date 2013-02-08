

# this lets me know that the module has not been imported.
#  we store it so we don't reimport a module each time the isgp2x function is called.
_is_gp2x = -1

def isgp2x():
    """ Returns True if we are running on a gp2x, else False
    """

    if _is_gp2x == -1:
        #TODO: FIXME: HACK: need to find a good way to do this.  
        #   Use configure to put 'gp2x' in the version string?
        import sys

        if "arm" in sys.version:
            _is_gp2x = True
        else:
            _is_gp2x = False
    else:
        return _is_gp2x



