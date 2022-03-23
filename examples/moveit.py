#!/usr/bin/env python
""" pygame.examples.moveit

This is the full and final example from the Pygame Tutorial,
"How Do I Make It Move". It creates 5 objects and animates
them on the screen.

It also has a separate player character that can be controlled with arrow keys.

Note it's a bit scant on error checking, but it's easy to read. :]
Fortunately, this is python, and we needn't wrestle with a pile of
error codes.
"""
import os
import pygame as pg

main_dir = os.path.split(os.path.abspath(__file__))[0]

# our game object class
class GameObject:
    def __init__(self, image, height, speed):
        self.speed = speed
        self.image = image
        self.pos = image.get_rect().move(0, height)

    # move the object. Defaults to moving right.
    def move(self, up=0, down=0, left=0, right=1):
        if right:
            self.pos.right += self.speed
        if left:
            self.pos.right -= self.speed
        if down:
            self.pos.top += self.speed
        if up:
            self.pos.top -= self.speed
        
        # controls the object such that it cannot leave the screen's viewpoint
        if self.pos.right > 640:
            self.pos.left = 0
        if self.pos.top > 420:
            self.pos.top = 0
        if self.pos.right < 79:
            self.pos.right = 640
        if self.pos.top < 0:
            self.pos.top = 420


# quick function to load an image
def load_image(name):
    path = os.path.join(main_dir, "data", name)
    return pg.image.load(path).convert()


# here's the full code
def main():
    pg.init()
    clock = pg.time.Clock()
    screen = pg.display.set_mode((640, 480))

    player = load_image("player1.gif")
    entity = load_image("alien1.gif")
    background = load_image("liquid.bmp")

    # scale the background image so that it fills the window and
    #   successfully overwrites the old sprite position.
    background = pg.transform.scale2x(background)
    background = pg.transform.scale2x(background)

    screen.blit(background, (0, 0))

    objects = []
    p = GameObject(player, 10, 3)
    for x in range(10):
        o = GameObject(entity, x * 40, x)
        objects.append(o)

    # Player controls
    up = 0
    down = 0
    right = 0
    left = 0

    # This is a simple event handler that enables player input.
    while True:
        screen.blit(background, (0, 0))
        for e in pg.event.get():

            # if a key is pressed, make sure that a value is set so that it can move in multiple directions
            if e.type == pg.KEYDOWN:
                if e.key == pg.K_DOWN:
                    down = 1
                elif e.key == pg.K_UP:
                    up = 1
                elif e.key == pg.K_LEFT:
                    left = 1
                elif e.key == pg.K_RIGHT:
                    right = 1
                elif e.key == pg.K_ESCAPE:
                    exit()

            # if a key is lifted, make sure the value is removed so it can stop moving
            if e.type == pg.KEYUP:
                if e.key == pg.K_DOWN:
                    down = 0
                elif e.key == pg.K_UP:
                    up = 0
                elif e.key == pg.K_LEFT:
                    left = 0
                elif e.key == pg.K_RIGHT:
                    right = 0
            
            # quit upon screen exit
            if e.type == pg.QUIT:
                exit()

        # move the player in accordance to the values set in the event handling
        p.move(up,down,left,right)

        for o in objects:
            screen.blit(background, o.pos, o.pos)
        for o in objects:
            o.move()
            screen.blit(o.image, o.pos)
        screen.blit(p.image, p.pos)
        clock.tick(60)
        pg.display.update()
    

if __name__ == "__main__":
    main()
    pg.quit()
