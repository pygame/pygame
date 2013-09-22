#!/usr/bin/env python

"""
This is the full and final example from the Pygame Tutorial,
"How Do I Make It Move". It creates 10 objects and animates
them on the screen.

Note it's a bit scant on error checking, but it's easy to read. :]
Fortunately, this is python, and we needn't wrestle with a pile of
error codes.
"""


#import everything
import os, pygame
from pygame.locals import *

main_dir = os.path.split(os.path.abspath(__file__))[0]

#our game object class
class GameObject:
    def __init__(self, image, height, speed):
        self.speed = speed
        self.image = image
        self.pos = image.get_rect().move(0, height)
    def move(self):
        self.pos = self.pos.move(self.speed, 0)
        if self.pos.right > 600:
            self.pos.left = 0


#quick function to load an image
def load_image(name):
    path = os.path.join(main_dir, 'data', name)
    return pygame.image.load(path).convert()


#here's the full code
def main():
    pygame.init()
    screen = pygame.display.set_mode((640, 480))

    player = load_image('player1.gif')
    background = load_image('liquid.bmp')

    # scale the background image so that it fills the window and
    #   successfully overwrites the old sprite position.
    background = pygame.transform.scale2x(background)
    background = pygame.transform.scale2x(background)

    screen.blit(background, (0, 0))

    objects = []
    for x in range(10):
        o = GameObject(player, x*40, x)
        objects.append(o)

    while 1:
        for event in pygame.event.get():
            if event.type in (QUIT, KEYDOWN):
                return

        for o in objects:
            screen.blit(background, o.pos, o.pos)
        for o in objects:
            o.move()
            screen.blit(o.image, o.pos)

        pygame.display.update()



if __name__ == '__main__': main()
