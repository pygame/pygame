#!/usr/bin/env python
""" pygame.examples.cursors

Click a mouse button and the cursor will change.

This example will show you:
*The different types of cursors that exist
*How to create a cursor
*How to set a cursor

"""
import pygame as pg
import os

#Create bitmap cursor from simple strings

# sized 24x24
thickarrow_strings = (
    "XX                      ",
    "XXX                     ",
    "XXXX                    ",
    "XX.XX                   ",
    "XX..XX                  ",
    "XX...XX                 ",
    "XX....XX                ",
    "XX.....XX               ",
    "XX......XX              ",
    "XX.......XX             ",
    "XX........XX            ",
    "XX........XXX           ",
    "XX......XXXXX           ",
    "XX.XXX..XX              ",
    "XXXX XX..XX             ",
    "XX   XX..XX             ",
    "     XX..XX             ",
    "      XX..XX            ",
    "      XX..XX            ",
    "       XXXX             ",
    "       XX               ",
    "                        ",
    "                        ",
    "                        ",
)

bitmapCursor1 = pg.cursors.Cursor(
    (24, 24), (0, 0), *pg.cursors.compile(thickarrow_strings, black='X', white='.', xor='o') 
)

# sized 24x16
sizer_x_strings = (
    "     X      X           ",
    "    XX      XX          ",
    "   X.X      X.X         ",
    "  X..X      X..X        ",
    " X...XXXXXXXX...X       ",
    "X................X      ",
    " X...XXXXXXXX...X       ",
    "  X..X      X..X        ",
    "   X.X      X.X         ",
    "    XX      XX          ",
    "     X      X           ",
    "                        ",
    "                        ",
    "                        ",
    "                        ",
    "                        ",
)

bitmapCursor2 = pg.cursors.Cursor(
    (24, 16), (0, 0), *pg.cursors.compile(sizer_x_strings, black='X', white='.', xor='o') 
)


#Create bitmap cursor from premade simple strings

bitmapCursor3 = pg.cursors.diamond


#Create a system cursor

systemCursor = pg.SYSTEM_CURSOR_CROSSHAIR


#Create color cursor

surf = pg.Surface((40, 40)) 
surf.fill((120, 50, 50))     
colorCursor = pg.cursors.Cursor((20, 20), surf)


#Load an image and use it as cursor surface

main_dir = os.path.split(os.path.abspath(__file__))[0]
imagename = os.path.join(main_dir, "data", "cursor.png")
image = pg.image.load(imagename)

imageCursor = pg.cursors.Cursor((0, 0), image)

def main():
    pg.init()
    pg.font.init()
    font = pg.font.Font(None, 30)
    bg = pg.display.set_mode((400, 300))
    bg.fill((220,220,220))
    text = font.render("Click to change cursor", True, (0, 0, 0))
    text_rect = text.get_rect(center=(200, 130))
    bg.blit(text, text_rect)
    pg.display.set_caption("Cursors Example")
    pg.display.update()

    cursors = [bitmapCursor1, bitmapCursor2, bitmapCursor3, systemCursor, colorCursor, imageCursor]
    index = 0
    
    pg.mouse.set_cursor(cursors[index])

    while True:
        for event in pg.event.get():
            if event.type == pg.MOUSEBUTTONDOWN:
                index += 1
                index %= len(cursors)
                pg.mouse.set_cursor(cursors[index])

            if event.type == pg.QUIT:
                pg.quit()
                raise SystemExit



if __name__ == "__main__":
    main()
