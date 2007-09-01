# 8 bit and 24 bit surfaces are currently broken
import pygame
from pygame.locals import *

pygame.init ()

screen = pygame.display.set_mode ((400, 400))
screen.fill ((100, 100, 100))

sf = pygame.Surface ((50, 50), 0, 32)

sf.fill ((0,0,0))
rect = sf.get_rect ()
screen.blit (sf, (50, 50))
pygame.display.flip ()

while 1:
    event = pygame.event.wait ()
    if event.type == QUIT:
        break
    if event.type == KEYDOWN:
        if event.key == K_m:
            print "Creating array"
            ar = pygame.PixelArray (sf)
            ar[3] = 0xffffff
            ar[-2] = 0x808080
            for i in xrange (rect.w):
                ar[i][2] = (100, 100, 200)
            del ar

            screen.blit (sf, (50, 50))
            pygame.display.flip ()
