import pygame


if pygame.get_sdl_version()[0] < 2:
    raise SystemExit('This example requires pygame 2 and SDL2.')

import os
data_dir = os.path.join(os.path.split(os.path.abspath(__file__))[0],
                        'data')

from pygame._sdl2 import (
    Window,
    Texture,
    Renderer,
    get_drivers,
    messagebox,
    MESSAGEBOX_WARNING
)

def load_img(file):
    return pygame.image.load(os.path.join(data_dir, file))

pygame.display.init()
pygame.key.set_repeat(1000, 10)

for driver in get_drivers():
    print(driver)

import random
answer = messagebox("I will open two windows! Continue?", "Hello!", type=MESSAGEBOX_WARNING,
                    buttons=('Yes', 'No', 'Chance'),
                    return_button=0, escape_button=1)
if answer == 1 or (answer == 2 and random.random() < .5):
    import sys
    sys.exit(0)

win = Window('asdf', resizable=True)
renderer = Renderer(win)
tex = Texture(renderer, load_img('alien1.gif'))

running = True

x, y = 250, 50
clock = pygame.time.Clock()

backgrounds = [(255,0,0,255), (0,255,0,255), (0,0,255,255)]
bg_index = 0

renderer.draw_color = backgrounds[bg_index]

win2 = Window('2nd window', size=(256, 256), always_on_top=True)
win2.opacity = 0.5
win2.set_icon(load_img('bomb.gif'))
renderer2 = Renderer(win2)
tex2 = Texture(renderer2, load_img('asprite.bmp'))
renderer2.clear()
renderer2.copy(tex2)
renderer2.present()
del tex2

full = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif getattr(event, 'window', None) == win2:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE or\
               event.type == pygame.WINDOWEVENT and event.event == pygame.WINDOWEVENT_CLOSE:
                win2.destroy()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_LEFT:
                x -= 5
            elif event.key == pygame.K_RIGHT:
                x += 5
            elif event.key == pygame.K_DOWN:
                y += 5
            elif event.key == pygame.K_UP:
                y -= 5
            elif event.key == pygame.K_f:
                if full == 0:
                    win.set_fullscreen(True)
                    full = 1
                else:
                    win.set_windowed()
                    full = 0
            elif event.key == pygame.K_SPACE:
                bg_index = (bg_index + 1) % len(backgrounds)
                renderer.draw_color = backgrounds[bg_index]

    renderer.clear()
    renderer.copy_pos(tex, x, y)
    renderer.present()

    clock.tick(60)
    win.title = str('FPS: {}'.format(clock.get_fps()))
