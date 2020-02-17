#!/usr/bin/env python
""" pygame.examples.mask

A pygame.mask collision detection production.




Brought

       to
             you
                     by

    the

pixels
               0000000000000
      and
         111111


This is 32 bits:
    11111111111111111111111111111111

There are 32 or 64 bits in a computer 'word'.
Rather than using one word for a pixel,
the mask module represents 32 or 64 pixels in one word.
As you can imagine, this makes things fast, and saves memory.

Compute intensive things like collision detection,
and computer vision benefit greatly from this.


This module can also be run as a stand-alone program, excepting
one or more image file names as command line arguments.
"""

import sys
import os
import random

import pygame as pg


def maskFromSurface(surface, threshold=127):
    return pg.mask.from_surface(surface, threshold)


def vadd(x, y):
    return [x[0] + y[0], x[1] + y[1]]


def vsub(x, y):
    return [x[0] - y[0], x[1] - y[1]]


def vdot(x, y):
    return x[0] * y[0] + x[1] * y[1]


class Sprite:
    def __init__(self, surface, mask=None):
        self.surface = surface
        if mask:
            self.mask = mask
        else:
            self.mask = maskFromSurface(self.surface)
        self.setPos([0, 0])
        self.setVelocity([0, 0])

    def setPos(self, pos):
        self.pos = [pos[0], pos[1]]

    def setVelocity(self, vel):
        self.vel = [vel[0], vel[1]]

    def move(self, dr):
        self.pos = vadd(self.pos, dr)

    def kick(self, impulse):
        self.vel[0] += impulse[0]
        self.vel[1] += impulse[1]

    def collide(self, s):
        """Test if the sprites are colliding and
        resolve the collision in this case."""
        offset = [int(x) for x in vsub(s.pos, self.pos)]
        overlap = self.mask.overlap_area(s.mask, offset)
        if overlap == 0:
            return
        """Calculate collision normal"""
        nx = self.mask.overlap_area(
            s.mask, (offset[0] + 1, offset[1])
        ) - self.mask.overlap_area(s.mask, (offset[0] - 1, offset[1]))
        ny = self.mask.overlap_area(
            s.mask, (offset[0], offset[1] + 1)
        ) - self.mask.overlap_area(s.mask, (offset[0], offset[1] - 1))
        if nx == 0 and ny == 0:
            """One sprite is inside another"""
            return
        n = [nx, ny]
        dv = vsub(s.vel, self.vel)
        J = vdot(dv, n) / (2 * vdot(n, n))
        if J > 0:
            """Can scale up to 2*J here to get bouncy collisions"""
            J *= 1.9
            self.kick([nx * J, ny * J])
            s.kick([-J * nx, -J * ny])
        return

        # """Separate the sprites"""
        # c1 = -overlap/vdot(n,n)
        # c2 = -c1/2
        # self.move([c2*nx,c2*ny])
        # s.move([(c1+c2)*nx,(c1+c2)*ny])

    def update(self, dt):
        self.pos[0] += dt * self.vel[0]
        self.pos[1] += dt * self.vel[1]


def main(*args):
    """Display multiple images bounce off each other using collision detection

    Positional arguments:
      one or more image file names.

    This pg.masks demo will display multiple moving sprites bouncing
    off each other. More than one sprite image can be provided.
    """

    if len(args) == 0:
        raise ValueError("Require at least one image file name: non given")
    print("Press any key to quit")
    screen = pg.display.set_mode((640, 480))
    if any("fist.bmp" in x for x in args):
        pg.display.set_caption("Punch Nazis")
    images = []
    masks = []
    for impath in args:
        images.append(pg.image.load(impath).convert_alpha())
        masks.append(maskFromSurface(images[-1]))

    numtimes = 10
    import time

    t1 = time.time()
    for x in range(numtimes):
        unused_mask = maskFromSurface(images[-1])
    t2 = time.time()

    print("python maskFromSurface :%s" % (t2 - t1))

    t1 = time.time()
    for x in range(numtimes):
        unused_mask = pg.mask.from_surface(images[-1])
    t2 = time.time()

    print("C pg.mask.from_surface :%s" % (t2 - t1))

    sprites = []
    for i in range(20):
        j = i % len(images)
        s = Sprite(images[j], masks[j])
        s.setPos(
            (
                random.uniform(0, screen.get_width()),
                random.uniform(0, screen.get_height()),
            )
        )
        s.setVelocity((random.uniform(-5, 5), random.uniform(-5, 5)))
        sprites.append(s)
    pg.time.set_timer(pg.USEREVENT, 33)
    while 1:
        event = pg.event.wait()
        if event.type == pg.QUIT:
            return
        elif event.type == pg.USEREVENT:

            # Do both mechanics and screen update
            screen.fill((240, 220, 100))
            for i in range(len(sprites)):
                for j in range(i + 1, len(sprites)):
                    sprites[i].collide(sprites[j])
            for s in sprites:
                s.update(1)
                if s.pos[0] < -s.surface.get_width() - 3:
                    s.pos[0] = screen.get_width()
                elif s.pos[0] > screen.get_width() + 3:
                    s.pos[0] = -s.surface.get_width()
                if s.pos[1] < -s.surface.get_height() - 3:
                    s.pos[1] = screen.get_height()
                elif s.pos[1] > screen.get_height() + 3:
                    s.pos[1] = -s.surface.get_height()
                screen.blit(s.surface, s.pos)
            pg.display.update()
        elif event.type == pg.KEYDOWN:
            return


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: mask.py <IMAGE> [<IMAGE> ...]")
        print("Let many copies of IMAGE(s) bounce against each other")
        print("Press any key to quit")
        main_dir = os.path.split(os.path.abspath(__file__))[0]
        imagename = os.path.join(main_dir, "data", "fist.bmp")
        main(imagename)
    else:
        main(*sys.argv[1:])
