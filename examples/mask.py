#!/usr/bin/env python
"""
pygame.examples.mask

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

import os
import random
import sys

import pygame as pg


class Sprite:
    def __init__(self, pos, vel, surface: pg.Surface, mask: pg.mask.Mask = None):
        self.surface = surface
        self.width, self.height = self.surface.get_size()
        if mask is not None:
            self.mask = mask
        else:
            self.mask = pg.mask.from_surface(self.surface)

        self.pos = pg.Vector2(pos)
        self.vel = pg.Vector2(vel)

    def collide(self, sprite):
        """
        Test if the sprites are colliding and
        resolve the collision in this case.

        Positional arguments:
            sprite: other sprite to test for collisions
        """
        offset = [int(x) for x in sprite.pos - self.pos]
        overlap = self.mask.overlap_area(sprite.mask, offset)
        if overlap == 0:
            return
        # Calculate collision normal
        nx = self.mask.overlap_area(
            sprite.mask, (offset[0] + 1, offset[1])
        ) - self.mask.overlap_area(sprite.mask, (offset[0] - 1, offset[1]))
        ny = self.mask.overlap_area(
            sprite.mask, (offset[0], offset[1] + 1)
        ) - self.mask.overlap_area(sprite.mask, (offset[0], offset[1] - 1))

        if nx == 0 and ny == 0:
            # One sprite is inside another
            return

        n = pg.Vector2(nx, ny)
        dv = sprite.vel - self.vel
        J = dv * n / (2 * n * n)
        if J > 0:
            # Can scale up to 2*J here to get bouncy collisions
            J *= 1.9
            self.vel += [nx * J, ny * J]
            sprite.vel += [-J * nx, -J * ny]

        # # Separate the sprites
        # c1 = -overlap / vdot(n, n)
        # c2 = -c1 / 2
        # self.pos += [c2 * nx, c2 * ny]
        # sprite.pos += [(c1 + c2) * nx, (c1 + c2) * ny]

    def update(self):
        self.pos += self.vel


def main(*args):
    """
    Display multiple images bounce off each other using collision detection

    Positional arguments:
      one or more image file names.

    This pg.masks demo will display multiple moving sprites bouncing
    off each other. More than one sprite image can be provided.
    """

    if len(args) == 0:
        raise ValueError("Require at least one image file name: non given")
    print("Press any key to quit")
    pg.init()

    screen_width, screen_height = 640, 480
    screen = pg.display.set_mode((screen_width, screen_height))
    clock = pg.time.Clock()

    images = []
    masks = []
    for image_path in args:
        images.append(pg.image.load(image_path).convert_alpha())
        masks.append(pg.mask.from_surface(images[-1]))

    sprites = []
    for i in range(20):
        j = i % len(images)
        s = Sprite(
            pos=(
                random.uniform(0, screen_width),
                random.uniform(0, screen_width),
            ),
            vel=(
                random.uniform(-5, 5),
                random.uniform(-5, 5),
            ),
            surface=images[j],
            mask=masks[j],
        )
        sprites.append(s)

    while True:
        for event in pg.event.get():
            if event.type in (pg.QUIT, pg.KEYDOWN):
                return

        screen.fill((240, 220, 100))

        for sprite_index, sprite in enumerate(sprites):
            for other_sprite in sprites[sprite_index + 1 :]:
                sprite.collide(other_sprite)

            sprite.update()

            # If the sprite is outside of the screen on the left
            if sprite.pos.x < -sprite.width:
                sprite.pos.x = screen_width
            # right
            elif sprite.pos.x > screen_width:
                sprite.pos.x = -sprite.width
            # top
            if sprite.pos.y < -sprite.height:
                sprite.pos.y = screen_height
            # down
            elif sprite.pos.y > screen_height:
                sprite.pos.y = -sprite.height

            screen.blit(sprite.surface, sprite.pos)

        clock.tick(30)
        pg.display.flip()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: mask.py <IMAGE> [<IMAGE> ...]")
        print("Let many copies of IMAGE(s) bounce against each other")
        print("Press any key to quit")
        main_dir = os.path.split(os.path.abspath(__file__))[0]
        image_name = os.path.join(main_dir, "data", "alien1.png")
        main(image_name)

    else:
        main(*sys.argv[1:])
    pg.quit()
