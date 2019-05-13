""" _sdl2.video.Image in a backwards compatible way with pygame.sprite groups.

Experimental! Uses APIs which may disapear in the next release (_sdl2 is private).
"""

import pygame

if pygame.get_sdl_version()[0] < 2:
    raise SystemExit('This example requires pygame 2 and SDL2.')

import os
data_dir = os.path.join(os.path.split(os.path.abspath(__file__))[0],
                        'data')

from pygame._sdl2 import (
    Window,
    Texture,
    Image,
    Renderer
)

def load_img(file):
    return pygame.image.load(os.path.join(data_dir, file))

pygame.display.init()
pygame.key.set_repeat(10, 10)

win = Window('asdf', resizable=True)
renderer = Renderer(win)
tex = Texture.from_surface(renderer, load_img('alien1.gif'))

class Something(pygame.sprite.Sprite):
    def __init__(self, img):
        pygame.sprite.Sprite.__init__(self)

        self.rect = img.get_rect()
        self.image = img

        self.rect.w *= 5
        self.rect.h *= 5

        img.origin = self.rect.w / 2, self.rect.h / 2

sprite = Something(Image(tex, (0, 0, tex.width/2, tex.height/2)))
sprite.rect.x = 250
sprite.rect.y = 50

#sprite2 = Something(Image(sprite.image))
sprite2 = Something(Image(tex))
sprite2.rect.x = 250
sprite2.rect.y = 250
sprite2.rect.w /= 2
sprite2.rect.h /= 2

group = pygame.sprite.Group()
group.add(sprite2)
group.add(sprite)

import math
t = 0
running = True
clock = pygame.time.Clock()
renderer.draw_color = (255, 0, 0, 255)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_LEFT:
                sprite.rect.x -= 5
            elif event.key == pygame.K_RIGHT:
                sprite.rect.x += 5
            elif event.key == pygame.K_DOWN:
                sprite.rect.y += 5
            elif event.key == pygame.K_UP:
                sprite.rect.y -= 5

    renderer.clear()
    t += 1

    img = sprite.image
    img.angle += 1
    img.flipX = t % 50 < 25
    img.flipY = t % 100 < 50
    img.color[0] = int(255.0 * (.5 + math.sin(.5 * t + 10.0) / 2.0))
    img.alpha = int(255.0 * (.5 + math.sin(.1 * t) / 2.0))
    #img.draw(dstrect=(x, y, 5 * img.srcrect['w'], 5 * img.srcrect['h']))

    group.draw(renderer)

    renderer.present()

    clock.tick(60)
    win.title = str('FPS: {}'.format(clock.get_fps()))
