import pygame as pg

pg.init()
display = pg.display.set_mode((640, 480))

pg.event.set_grab(True)
pg.mouse.set_visible(False)
visible, grabbed = False, True
font = pg.font.SysFont('arial', 24)

def draw_text_line(text, y=0):
    """
    Draws a line of text onto the display surface
    The text will be centered horizontally at the given y postition
    The text's height is added to y and returned to the caller
    """
    display = pg.display.get_surface()
    surf = font.render(text, 1, (255, 255, 255))
    y += surf.get_height()
    x = (display.get_width() - surf.get_width()) / 2
    display.blit(surf, (x, y))
    return y

clock = pg.time.Clock()
Running = True
virtualx = 0
virtualy = 0

while Running:
    for ev in pg.event.get():
        if ev.type == pg.QUIT:
            Running = False
        elif ev.type == pg.KEYDOWN:
            if ev.key == pg.K_ESCAPE:
                Running = False
        elif ev.type == pg.MOUSEBUTTONDOWN:
            if ev.button == 1:
                if pg.mouse.get_visible():
                    visible = False
                    pg.mouse.set_visible(False)
                else:
                    visible = True
                    pg.mouse.set_visible(True)
            else:
                if pg.event.get_grab():
                    grabbed = False
                    pg.event.set_grab(False)
                else:
                    grabbed = True
                    pg.event.set_grab(True)

    newx, newy = pg.mouse.get_rel()
    virtualx += newx
    virtualy += newy

    display.fill((0,0,0))
    y = draw_text_line(
        'visible = {}: left click to toggle'.format(visible), 50)
    y = draw_text_line(
        'grabbed = {}: right click to toggle'.format(grabbed), y)
    y = draw_text_line('x={}, y={}'.format(virtualx, virtualy), y)

    if grabbed and not visible:
    	draw_text_line('Virtual Mouse Mode Enabled!', y + 50)

    clock.tick(30)
    pg.display.flip()
