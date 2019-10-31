import pygame as pg
from pygame._sdl2 import Window, Texture, Image, Renderer

import os
pg.init()

if pg.get_sdl_version()[0] < 2:
    raise SystemExit(
        "This example requires pg 2 and SDL2. _sdl2 is experimental and will change."
    )


RES = (160, 120)
FPS = 30
clock = pg.time.Clock()

screen = pg.display.set_mode(RES, pg.SCALED | pg.RESIZABLE)

done = False

i = 0
j = 0

screen = pg.display.set_mode(RES, pg.SCALED | pg.RESIZABLE)
win = Window.get_window_from_ID(pg.display.get_window_ID())
renderer = Renderer.get_window_renderer(win)

win.size=800,600
renderer.set_viewport((0,0,80,60))

while not done:
    for event in pg.event.get():
        if event.type == pg.KEYDOWN and event.key == pg.K_q:
            done = True
        if event.type == pg.QUIT:
            done = True
        if event.type == pg.KEYDOWN and event.key == pg.K_f:
            pg.display.toggle_fullscreen()
        if event.type == pg.VIDEORESIZE:
            pg.display.resize_event(event)

    i += 1
    i = i % screen.get_width()
    j += i % 2
    j = j % screen.get_height()

    screen.fill((255, 0, 255))
    pg.draw.circle(screen, (0, 0, 0), (100, 100), 20)
    pg.draw.circle(screen, (0, 0, 200), (0, 0), 10)
    pg.draw.circle(screen, (200, 0, 0), (160, 120), 30)
    pg.draw.line(screen, (250, 250, 0), (0, 120), (160, 0))
    pg.draw.circle(screen, (255, 255, 255), (i, j), 5)

    pg.display.flip()
    clock.tick(FPS)
