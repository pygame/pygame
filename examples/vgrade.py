#!/usr/bin/env python

"""This example demonstrates creating an image with Numeric
python, and displaying that through SDL. You can look at the
method of importing numeric and pygame.surfarray. This method
will fail 'gracefully' if it is not available.
I've tried mixing in a lot of comments where the code might
not be self explanatory, nonetheless it may still seem a bit
strange. Learning to use numeric for images like this takes a
bit of learning, but the payoff is extremely fast image
manipulation in python.
The code also demonstrates use of the timer events."""


import pygame
from pygame.locals import *

try:
    from Numeric import *
    from RandomArray import *
    import pygame.surfarray
except ImportError:
    raise SystemExit, 'This example requires Numeric and the pygame surfarray module'


timer = 0
def stopwatch(message = None):
    "simple routine to time python code"
    global timer
    if not message:
        timer = pygame.time.get_ticks()
        return
    now = pygame.time.get_ticks()
    runtime = (now - timer)/1000.0 + .001
    print message, runtime, ('seconds\t(%.2ffps)'%(1.0/runtime))
    timer = now



def VertGrad3D(surf, topcolor, bottomcolor):
    "creates a new 3d vertical gradient array"
    topcolor = array(topcolor, copy=0)
    bottomcolor = array(bottomcolor, copy=0)
    diff = bottomcolor - topcolor
    width, height = surf.get_size()
    # create array from 0.0 to 1.0 triplets
    column = arange(height, typecode=Float)/height
    column = repeat(column[:, NewAxis], [3], 1)
    # create a single column of gradient
    column = topcolor + (diff * column).astype(Int)
    # make the column a 3d image column by adding X
    column = column.astype(UnsignedInt8)[NewAxis,:,:]
    # stretch the column into a full image
    return resize(column, (width, height, 3))



def DisplayGradient(surf):
    "choose random colors and show them"
    stopwatch()
    colors = randint(0, 255, (2, 3))
    grade = VertGrad3D(surf, colors[0], colors[1])
    pygame.surfarray.blit_array(surf, grade)
    pygame.display.flip()
    stopwatch('Gradient:')



def main():
    pygame.init()
    size = (640, 480)
    screen = pygame.display.set_mode(size, 0, 32)
    finished = 0
    pygame.event.set_blocked(MOUSEMOTION) #keep our queue cleaner
    pygame.time.set_timer(500, USEREVENT+5)
    while 1:
        event = pygame.event.poll()
        if event.type in (QUIT, KEYDOWN, MOUSEBUTTONDOWN):
            break
        elif event.type == USEREVENT+5:
            DisplayGradient(screen)

    

if __name__ == '__main__': main()
