#
# These methods are called internally by pygame.scrap
#
from AppKit import *
from Foundation import *

import sys
import tempfile
import pygame.image
from pygame.locals import SCRAP_TEXT, SCRAP_BMP, SCRAP_SELECTION, SCRAP_CLIPBOARD
from cStringIO import StringIO
from pygame.compat import unicode_

ScrapPboardType = unicode_('org.pygame.scrap')


err = "Only text has been implemented for scrap on mac. See lib/mac_scrap.py to debug."



def init():
    return 1

def get(scrap_type):
    board = NSPasteboard.generalPasteboard()
    
    if 0:
        print (board.types)
        print (dir(board.types))
        print (dir(board))
        print (board.__doc__)

    if scrap_type == SCRAP_TEXT:
        return board.stringForType_(NSStringPboardType)
    elif 1:
        raise NotImplementedError(err)


    elif 0 and scrap_type == SCRAP_BMP:
        # We could try loading directly but I don't trust pygame's TIFF
        # loading.  This is slow and stupid but it does happen to work.
        if not NSImage.canInitWithPasteboard_(board):
            return None
        img = NSImage.alloc().initWithPasteboard_(board)
        data = img.TIFFRepresentation()
        rep = NSBitmapImageRep.alloc().initWithData_(data)
        if rep is None:
            return None

        # bug with bmp, out of memory error... so we use png.
        #data = rep.representationUsingType_properties_(NSBMPFileType, None)
        data = rep.representationUsingType_properties_(NSPNGFileType, None)
        bmp = StringIO(data)
        return pygame.image.load(bmp, "scrap.png")
    #elif scrap_type in board.types:
    elif scrap_type == SCRAP_BMP:
        return board.dataForType_(scrap_type)
    else:
        return board.stringForType_(scrap_type)

def put(scrap_type, thing):
    board = NSPasteboard.generalPasteboard()
    if scrap_type == SCRAP_TEXT:
        board.declareTypes_owner_([NSStringPboardType, ScrapPboardType], None)
        if isinstance(thing, unicode):
            text_thing = thing
        else:
            text_thing = unicode(thing, 'utf-8')
        board.setString_forType_(text_thing, NSStringPboardType)
        board.setString_forType_(unicode_(''), ScrapPboardType)
    elif 1:
        raise NotImplementedError(err)





    elif 0 and scrap_type == SCRAP_BMP:
        # Don't use this code... we put the data in as a string.

        #if type(thing) != type(pygame.Surface((1,1))):
        #    thing = pygame.image.fromstring(thing, len(thing) * 4, "RGBA")


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
        board.setString_forType_(unicode_(''), ScrapPboardType)
    elif scrap_type == SCRAP_BMP:
        
        other_type = scrap_type
        board.declareTypes_owner_([other_type], None)
        board.setData_forType_(thing, other_type)

    else:
        other_type = scrap_type
        if 0:
            board.declareTypes_owner_([NSStringPboardType, other_type], None)
            board.setString_forType_(text_thing, NSStringPboardType)
        elif 0:
            board.declareTypes_owner_([other_type], None)
            #board.setString_forType_(thing, other_type)
            board.setData_forType_(thing, other_type)
        else:
            board.declareTypes_owner_([NSStringPboardType, other_type], None)
            board.setString_forType_(thing, NSStringPboardType)

        #board.setData_forType_(thing, other_type)





def set_mode (mode):
    # No diversion between clipboard and selection modes on MacOS X.
    if mode not in [SCRAP_SELECTION, SCRAP_CLIPBOARD]:
        raise ValueError("invalid clipboard mode")

def contains (scrap_type):
    return scrap_type in NSPasteboard.generalPasteboard ().types ()

def get_types ():
    typelist = []
    types = NSPasteboard.generalPasteboard ().types ()
    for t in types:
        typelist.append (t)
    return typelist

def lost ():
    board = NSPasteboard.generalPasteboard ()
    return not board.availableTypeFromArray_ ([ScrapPboardType])
