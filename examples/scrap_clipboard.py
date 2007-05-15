#!/usr/bin/env python
"""
Demonstrates the clipboard capabilities of pygame.
"""
import pygame
from pygame.locals import *
import pygame.scrap as scrap
import StringIO


pygame.init ()
screen = pygame.display.set_mode ((200, 200))
c = pygame.time.Clock ()
going = True

# Initialize the scrap module and use the clipboard mode.
scrap.init ()
scrap.set_mode (SCRAP_CLIPBOARD)

while going:
    for e in pygame.event.get ():
        if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
            going = False

        if e.type == KEYDOWN and e.key == K_g:
            # This means to look for data.
            print "Getting the different clipboard data.."
            for t in scrap.get_types ():
                r = scrap.get (t)
                if r and len (r) > 500:
                    print "Type %s : (large buffer)" % t
                else:
                    print "Type %s : %s" % (t, r)
                if "image" in t:
                    namehint = t.split("/")[1]
                    if namehint in ['bmp', 'png', 'jpg']:
                        f = StringIO.StringIO(r)
                        loaded_surf = pygame.image.load(f, "." + namehint)
                        screen.blit(loaded_surf, (0,0))


        if e.type == KEYDOWN and e.key == K_p:
            # Place some text into the selection.
            print "Placing clipboard text."
            scrap.put (SCRAP_TEXT, "Hello. This is a message from scrap.")

        if e.type == KEYDOWN and e.key == K_a:
            # Get all available types.
            print "Getting the available types from the clipboard."
            types = scrap.get_types ()
            print types
            if len (types) > 0:
                print "Contains %s: %s" % (types[0], scrap.contains (types[0]))
                print "Contains _INVALID_: ", scrap.contains ("_INVALID_")

        if e.type == KEYDOWN and e.key == K_i:
            print "Putting image into the clipboard."
            scrap.set_mode (SCRAP_CLIPBOARD)
            fp = open ("data/liquid.bmp", "rb")
            buf = fp.read ()
            scrap.put ("image/bmp", buf)
            fp.close ()

    pygame.display.flip()
    c.tick(40)




