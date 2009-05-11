#!/usr/bin/env python
"""
Demonstrates the clipboard capabilities of pygame.
"""
import os

import pygame
from pygame.locals import *
import pygame.scrap as scrap
import StringIO

def usage ():
    print ("Press the 'g' key to get all of the current clipboard data")
    print ("Press the 'p' key to put a string into the clipboard")
    print ("Press the 'a' key to get a list of the currently available types")
    print ("Press the 'i' key to put an image into the clipboard")

main_dir = os.path.split(os.path.abspath(__file__))[0]

pygame.init ()
screen = pygame.display.set_mode ((200, 200))
c = pygame.time.Clock ()
going = True

# Initialize the scrap module and use the clipboard mode.
scrap.init ()
scrap.set_mode (SCRAP_CLIPBOARD)

usage ()

while going:
    for e in pygame.event.get ():
        if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
            going = False

        elif e.type == KEYDOWN and e.key == K_g:
            # This means to look for data.
            print ("Getting the different clipboard data..")
            for t in scrap.get_types ():
                r = scrap.get (t)
                if r and len (r) > 500:
                    print ("Type %s : (large buffer)" % t)
                else:
                    print ("Type %s : %s" % (t, r))
                if "image" in t:
                    namehint = t.split("/")[1]
                    if namehint in ['bmp', 'png', 'jpg']:
                        f = StringIO.StringIO(r)
                        loaded_surf = pygame.image.load(f, "." + namehint)
                        screen.blit(loaded_surf, (0,0))


        elif e.type == KEYDOWN and e.key == K_p:
            # Place some text into the selection.
            print ("Placing clipboard text.")
            scrap.put (SCRAP_TEXT, "Hello. This is a message from scrap.")

        elif e.type == KEYDOWN and e.key == K_a:
            # Get all available types.
            print ("Getting the available types from the clipboard.")
            types = scrap.get_types ()
            print (types)
            if len (types) > 0:
                print ("Contains %s: %s" % (types[0], scrap.contains (types[0])))
                print ("Contains _INVALID_: ", scrap.contains ("_INVALID_"))

        elif e.type == KEYDOWN and e.key == K_i:
            print ("Putting image into the clipboard.")
            scrap.set_mode (SCRAP_CLIPBOARD)
            fp = open (os.path.join(main_dir, 'data', 'liquid.bmp'), 'rb')
            buf = fp.read ()
            scrap.put ("image/bmp", buf)
            fp.close ()

        elif e.type in (KEYDOWN, MOUSEBUTTONDOWN):
            usage ()
    pygame.display.flip()
    c.tick(40)




