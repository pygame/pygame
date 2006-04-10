#NOTE: the docs for this will be inserted from the scrap.doc file.  
#        see pygame/__init__.py
try:
    from AppKit import *
    from Foundation import *
except:
    pass

import pygame.locals



def init():
    # FIXME: NOTE: I don't know what type of init we need here.
    return 1

def get(scrap_type):
    if scrap_type == pygame.locals.SCRAP_TEXT:
	board = NSPasteboard.generalPasteboard()
	content = board.stringForType_(NSStringPboardType)
	return content
    else:
        raise "Unsupported scrap_type"


def put(scrap_type, thing):
    if scrap_type == pygame.locals.SCRAP_TEXT:
        board = NSPasteboard.generalPasteboard()
        board.declareTypes_owner_([NSStringPboardType], None)
        board.setString_forType_(unicode(thing), NSStringPboardType)
    else:
        raise "Unsupported scrap_type"

def lost():
    #FIXME: TODO: how do we do this on mac?
    return 0
    pass


