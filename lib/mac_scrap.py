#
# These methods are called internally by pygame.scrap
#
from AppKit import *
from Foundation import *

import pygame.locals

ScrapPboardType = u'org.pygame.scrap'

def init():
    return 1

def get(scrap_type):
    if scrap_type == pygame.locals.SCRAP_TEXT:
        board = NSPasteboard.generalPasteboard()
        content = board.stringForType_(NSStringPboardType)
        return content
    else:
        raise ValueError("Unsupported scrap_type: %r" % (scrap_type,))


def put(scrap_type, thing):
    if scrap_type == pygame.locals.SCRAP_TEXT:
        board = NSPasteboard.generalPasteboard()
        board.declareTypes_owner_([NSStringPboardType, ScrapPboardType], None)
        if isinstance(thing, unicode):
            text_thing = thing
        else:
            text_thing = unicode(text_thing, 'utf-8')
        board.setString_forType_(text_thing, NSStringPboardType)
        board.setString_forType_(u'', ScrapPboardType)
    else:
        raise ValueError("Unsupported scrap_type: %r" % (scrap_type,))

def lost():
    board = NSPasteboard.generalPasteboard()
    return not board.availableTypeFromArray_([ScrapPboardType])
