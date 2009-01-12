#!/usr/bin/env python

import pygame


arrow = ( "xX                      ",
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
          "                        ")


no = ("                        ",
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
    for y in range(len(arrow)):
        for x in range(len(arrow[y])):
            if arrow[y][x] in ['x', ',', 'O']:
                hotspot = x,y
                break
        if hotspot != None:
            break
    if hotspot == None:
        raise Exception("No hotspot specified for cursor '%s'!" %
cursorname)
    s2 = []
    for line in arrow:
        s2.append(line.replace('x', 'X').replace(',', '.').replace('O',
'o'))
    cursor, mask = pygame.cursors.compile(s2, 'X', '.', 'o')
    size = len(arrow[0]), len(arrow)
    pygame.mouse.set_cursor(size, hotspot, cursor, mask)

def main():
    pygame.init()
    pygame.font.init()
    font = pygame.font.Font(None, 24)
    bg = pygame.display.set_mode((800, 600), 0, 24)
    bg.fill((255,255,255))
    bg.blit(font.render("Click to advance", 1, (0, 0, 0)), (0, 0))
    pygame.display.update()
    for cursor in [no, arrow]:
        TestCursor(cursor)
        going = True
        while going:
            pygame.event.pump()
            for e in pygame.event.get():
                if e.type == pygame.MOUSEBUTTONDOWN:
                    going = False
    pygame.quit()


if __name__ == '__main__':
    main()

