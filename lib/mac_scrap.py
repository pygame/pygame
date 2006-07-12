#
# These methods are called internally by pygame.scrap
#
from AppKit import *
from Foundation import *

import sys
import tempfile
import pygame.image
from pygame.locals import SCRAP_TEXT, SCRAP_BMP
from cStringIO import StringIO

ScrapPboardType = u'org.pygame.scrap'

def init():
    return 1

def get(scrap_type):
    board = NSPasteboard.generalPasteboard()
    if scrap_type == SCRAP_TEXT:
        content = board.stringForType_(NSStringPboardType)
        return content
    elif scrap_type == SCRAP_BMP:
        # We could try loading directly but I don't trust pygame's TIFF loading.
        # This is slow and stupid but it does happen to work.
        if not NSImage.canInitWithPasteboard_(board):
            return None
        img = NSImage.alloc().initWithPasteboard_(board)
        data = img.TIFFRepresentation()
        rep = NSBitmapImageRep.alloc().initWithData_(data)
        if rep is None:
            return None
        data = rep.representationUsingType_properties_(NSBMPFileType, None)
        bmp = StringIO(data)
        return pygame.image.load(bmp, "scrap.bmp")
    else:
        raise ValueError("Unsupported scrap_type: %r" % (scrap_type,))


def put(scrap_type, thing):
    board = NSPasteboard.generalPasteboard()
    if scrap_type == SCRAP_TEXT:
        board.declareTypes_owner_([NSStringPboardType, ScrapPboardType], None)
        if isinstance(thing, unicode):
            text_thing = thing
        else:
            text_thing = unicode(text_thing, 'utf-8')
        board.setString_forType_(text_thing, NSStringPboardType)
        board.setString_forType_(u'', ScrapPboardType)
    elif scrap_type == SCRAP_BMP:
        # This is pretty silly, we shouldn't have to do this...
        fh = tempfile.NamedTemporaryFile(suffix='.png')
        pygame.image.save(thing, fh.name)
        path = fh.name
        if not isinstance(path, unicode):
            path = unicode(path, sys.getfilesystemencoding())
        img = NSImage.alloc().initByReferencingFile_(path)
        tiff = img.TIFFRepresentation()
        fh.close()
        board.declareTypes_owner_([NSTIFFPboardType, ScrapPboardType], None)
        board.setData_forType_(tiff, NSTIFFPboardType)
        board.setString_forType_(u'', ScrapPboardType)
    else:
        raise ValueError("Unsupported scrap_type: %r" % (scrap_type,))

def lost():
    board = NSPasteboard.generalPasteboard()
    return not board.availableTypeFromArray_([ScrapPboardType])
