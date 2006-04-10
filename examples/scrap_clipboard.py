import pygame
from pygame.locals import *
pygame.init()

pygame.display.set_mode((200, 200))

import pygame.scrap
pygame.scrap.init()


c = pygame.time.Clock()

going = True


while going:
    for e in pygame.event.get():
        if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
            going = False
        if e.type == KEYDOWN and e.key == K_g:

            # this number means to look for text data.
            r = pygame.scrap.get(SCRAP_TEXT)
            print type(r)
            print ":%s:" % r

            if type(r) == type(""):
                print repr(r)
                print len(r)
                print "replaced :%s:" % r.replace("\r", "\n")

        if e.type == KEYDOWN and e.key == K_p:

            # this number means to look for text data.
            pygame.scrap.put(SCRAP_TEXT, "Hello.  This is a message from scrap.")

            #r = pygame.scrap.get(SCRAP_TEXT)
            #print type(r)
            #print ":%s:" % r



        if e.type == KEYDOWN and e.key == K_b:
            r = pygame.scrap.get(1)
            print type(r)
            print r

    pygame.display.flip()
    c.tick(40)




