#!/usr/bin/env python
""" pygame.examples.cursors

Click a mouse button (if you have one!) and the cursor changes.

"""
import pygame as pg


arrow = (
    "xX                      ",
    "X.X                     ",
    "X..X                    ",
    "X...X                   ",
    "X....X                  ",
    "X.....X                 ",
    "X......X                ",
    "X.......X               ",
    "X........X              ",
    "X.........X             ",
    "X......XXXXX            ",
    "X...X..X                ",
    "X..XX..X                ",
    "X.X XX..X               ",
    "XX   X..X               ",
    "X     X..X              ",
    "      X..X              ",
    "       X..X             ",
    "       X..X             ",
    "        XX              ",
    "                        ",
    "                        ",
    "                        ",
    "                        ",
)


no = (
    "                        ",
    "                        ",
    "         XXXXXX         ",
    "       XX......XX       ",
    "      X..........X      ",
    "     X....XXXX....X     ",
    "    X...XX    XX...X    ",
    "   X.....X      X...X   ",
    "   X..X...X      X..X   ",
    "  X...XX...X     X...X  ",
    "  X..X  X...X     X..X  ",
    "  X..X   X...X    X..X  ",
    "  X..X    X.,.X   X..X  ",
    "  X..X     X...X  X..X  ",
    "  X...X     X...XX...X  ",
    "   X..X      X...X..X   ",
    "   X...X      X.....X   ",
    "    X...XX     X...X    ",
    "     X....XXXXX...X     ",
    "      X..........X      ",
    "       XX......XX       ",
    "         XXXXXX         ",
    "                        ",
    "                        ",
)


def TestCursor(arrow):
    hotspot = None
    for y, line in enumerate(arrow):
        for x, char in enumerate(line):
            if char in ["x", ",", "O"]:
                hotspot = x, y
                break
        if hotspot is not None:
            break
    if hotspot is None:
        raise Exception("No hotspot specified for cursor '%s'!" % arrow)
    s2 = []
    for line in arrow:
        s2.append(line.replace("x", "X").replace(",", ".").replace("O", "o"))
    cursor, mask = pg.cursors.compile(s2, "X", ".", "o")
    size = len(arrow[0]), len(arrow)
    pg.mouse.set_cursor(size, hotspot, cursor, mask)


def main():
    pg.init()
    pg.font.init()
    font = pg.font.Font(None, 24)
    bg = pg.display.set_mode((800, 600), 0, 24)
    bg.fill((255, 255, 255))
    bg.blit(font.render("Click to advance", 1, (0, 0, 0)), (0, 0))
    pg.display.update()
    for cursor in [no, arrow]:
        TestCursor(cursor)
        going = True
        while going:
            pg.event.pump()
            for e in pg.event.get():
                if e.type == pg.MOUSEBUTTONDOWN:
                    going = False
    pg.quit()


if __name__ == "__main__":
    main()
