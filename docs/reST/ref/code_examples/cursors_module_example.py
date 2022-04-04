# pygame setup
import pygame as pg

pg.init()
screen = pg.display.set_mode([600, 400])
pg.display.set_caption("Example code for the cursors module")

# create a system cursor
system = pg.cursors.Cursor(pg.SYSTEM_CURSOR_NO)

# create bitmap cursors
bitmap_1 = pg.cursors.Cursor(*pg.cursors.arrow)
bitmap_2 = pg.cursors.Cursor(
    (24, 24), (0, 0), *pg.cursors.compile(pg.cursors.thickarrow_strings)
)

# create a color cursor
surf = pg.Surface((40, 40)) # you could also load an image 
surf.fill((120, 50, 50))        # and use that as your surface
color = pg.cursors.Cursor((20, 20), surf)

cursors = [system, bitmap_1, bitmap_2, color]
cursor_index = 0

pg.mouse.set_cursor(cursors[cursor_index])

clock = pg.time.Clock()
going = True
while going:
    clock.tick(60)
    screen.fill((0, 75, 30))
    pg.display.flip()

    for event in pg.event.get():
        if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
            going = False

        # if the mouse is clicked it will switch to a new cursor
        if event.type == pg.MOUSEBUTTONDOWN:
            cursor_index += 1
            cursor_index %= len(cursors)
            pg.mouse.set_cursor(cursors[cursor_index])

pg.quit()
