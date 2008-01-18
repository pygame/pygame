import pygame
from pygame.locals import *

pygame.init ()

screen = pygame.display.set_mode ((200, 200))
pygame.display.flip ()
rect = screen.get_rect ()

while 1:
    event = pygame.event.wait ()
    if event.type == QUIT:
        break
    if event.type == MOUSEBUTTONDOWN:

        # Create  the PixelArray
        ar = pygame.PixelArray (screen)

        # Fill the x columns with a white color. This will create two
        # vertical, small rects.
        ar[3:5] = (255, 255, 255)
        ar[-4:-2] = (255, 255, 255)

        # 
        for px in xrange (rect.width):
            # A diagonal line from the topleft to the bottomright
            ar[px][px] = (255, 255, 255)

            # A diagonal line from the bottomright to the topleft.
            ar[px][-px] = (255, 255, 255)

            # Horizontal, small rects.
            ar[px][3:5] = (255, 255, 255)
            ar[px][-4:-2] = (255, 255, 255)

            # Note, that something like
            #
            #   array[2:4][3:5] = ...
            #
            # will _not_ cause a rectangular manipulation. Instead it
            # will be first sliced to a two-column array, which then
            # shall be sliced by columns once more, which will fail due
            # an IndexError.
            #
            # This is caused by the slicing mechanisms in python and an
            # absolutely correct behaviour. Instead create a single
            # columned slice first, which you can manipulate then (as
            # done above in e.g. ar[px][3:5] = (255, 255, 255).
            #
            
        del ar

        pygame.display.flip ()
