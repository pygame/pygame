
"""Here we load a .TTF font file, and display it in
a basic pygame window. It demonstrates several of the
Font object attributes. Nothing exciting in here, but
it makes a great example for basic window, event, and
font management."""


import pygame
from pygame.locals import *


def main():
    #initialize
    pygame.init()
    resolution = 400, 200
    screen = pygame.display.set_mode(resolution)

    #the python 1.5.2 way to set the cursor
    apply(pygame.mouse.set_cursor, pygame.cursors.diamond)
    #the python 2.0 way to set the cursor
    #pygame.mouse.set_cursor(*pygame.cursors.diamond)

    fg = 250, 240, 230
    bg = 5, 5, 5
    wincolor = 40, 40, 90

    #fill background
    screen.fill(wincolor)

    #load font, prepare values
    font = pygame.font.Font(None, 80)
    text = 'Fonty'
    size = font.size(text)

    #no AA, no transparancy, normal
    ren = font.render(text, 0, fg, bg)
    screen.blit(ren, (10, 10))

    #no AA, transparancy, underline
    font.set_underline(1)
    ren = font.render(text, 0, fg)
    screen.blit(ren, (10, 40 + size[1]))
    font.set_underline(0)

    #AA, no transparancy, bold
    font.set_bold(1)
    ren = font.render(text, 1, fg, bg)
    screen.blit(ren, (30 + size[0], 10))
    font.set_bold(0)

    #AA, transparancy, italic
    font.set_italic(1)
    ren = font.render(text, 1, fg)
    screen.blit(ren, (30 + size[0], 40 + size[1]))
    font.set_italic(0)

    #show the surface and await user quit
    pygame.display.flip()
    while 1:
        #use event.wait to keep from polling 100% cpu
        if pygame.event.wait().type in (QUIT, KEYDOWN, MOUSEBUTTONDOWN):
            break



if __name__ == '__main__': main()
    