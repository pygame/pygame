import pygame as pg

pg.init()
# define variables
screen = pg.display.set_mode((1280, 720))

clock = pg.time.Clock()


# main loop
while True:
    # fill the screen with the color purple
    screen.fill("purple")
    # get pygame events
    for event in pg.event.get():
        # if pygame is quitted, end the program
        if event.type == pg.QUIT:
            pg.quit()
            raise SystemExit
    # update the screen
    pg.display.flip()

    # make the game go at 60 fps
    clock.tick(60)
