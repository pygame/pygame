#!/usr/bin/env python

'''An example of using the SDL.ttf module with 2D graphics.
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id: $'

import sys

from SDL import *
from SDL.ttf import *

DEFAULT_PTSIZE = 18
DEFAULT_TEXT = 'The quick brown fox jumped over the lazy dog'
NUM_COLORS = 256

Usage = 'Usage: %s [-solid] [-b] [-i] [-u] [-fgcol r,g,b] [-bgcol r,g,b] \
<font>.ttf [ptsize] [text]' % sys.argv[0]

if __name__ == '__main__':
    white = SDL_Color(0xff, 0xff, 0xff)
    black = SDL_Color(0, 0, 0)

    dump = 0
    rendersolid = 0
    renderstyle = TTF_STYLE_NORMAL

    forecol = black
    backcol = white
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg[0] != '-':
            break
        if arg == '-solid':
            rendersolid = 1
        elif arg == '-b':
            renderstyle |= TTF_STYLE_BOLD
        elif arg == '-i':
            renderstyle |= TTF_STYLE_ITALIC
        elif arg == '-u':
            renderstyle |= TTF_STYLE_UNDERLINE
        elif arg == '-dump':
            dump = 1
        elif arg == '-fgcol':
            i += 1
            r, g, b = sys.argv[i].split(',')
            forecol.r = int(r)
            forecol.g = int(g)
            forecol.b = int(b)
        elif arg == '-bgcol':
            i += 1
            r, g, b = sys.argv[i].split(',')
            backcol.r = int(r)
            backcol.g = int(g)
            backcol.b = int(b)
        else:
            print >> sys.stderr, Usage
            sys.exit(1)
        i += 1

    if i >= len(sys.argv):
        print >> sys.stderr, Usage
        sys.exit(1)

    SDL_Init(SDL_INIT_VIDEO)
    TTF_Init()

    fontfile = sys.argv[i]
    i += 1

    try:
        ptsize = int(sys.argv[i])
        i += 1
    except:
        ptsize = DEFAULT_PTSIZE

    try:
        message = sys.argv[i]
    except:
        message = DEFAULT_TEXT

    font = TTF_OpenFont(fontfile, ptsize)
    TTF_SetFontStyle(font, renderstyle)

    if dump:
        for i in range(48, 123):
            glyph = TTF_RenderGlyph_Shaded(font, unichr(i), forecol, backcol)
            SDL_SaveBMP(glyph, 'glyph-%d.bmp' % i)
        TTF_Quit()
        SDL_Quit()
        sys.exit(0)

    screen = SDL_SetVideoMode(640, 480, 8, SDL_SWSURFACE)

    # Set a palette that is good for the foreground colored text
    rdiff = backcol.r - forecol.r
    gdiff = backcol.g - forecol.g
    bdiff = backcol.b - forecol.b
    colors = []
    for i in range(NUM_COLORS):
        colors.append(SDL_Color(forecol.r + (i * rdiff) / 4,
                                forecol.g + (i * gdiff) / 4,
                                forecol.b + (i * bdiff) / 4) )
    SDL_SetColors(screen, colors, 0)

    SDL_FillRect(screen, None, 
                 SDL_MapRGB(screen.format, backcol.r, backcol.g, backcol.b))
    SDL_UpdateRect(screen, 0, 0, 0, 0)

    s = 'Font file: %s' % fontfile
    if rendersolid:
        text = TTF_RenderText_Solid(font, s, forecol)
    else:
        text = TTF_RenderText_Shaded(font, s, forecol, backcol)
    
    dstrect = SDL_Rect(4, 4, text.w, text.h)
    SDL_BlitSurface(text, None, screen, dstrect)
    SDL_FreeSurface(text)


    if rendersolid:
        text = TTF_RenderText_Solid(font, message, forecol)
    else:
        text = TTF_RenderText_Shaded(font, message, forecol, backcol)

    dstrect.x = (screen.w - text.w) / 2
    dstrect.y = (screen.h - text.h) / 2
    dstrect.w = text.w
    dstrect.h = text.h
    print 'Font is generally %d big, and string is %d big' % \
        (TTF_FontHeight(font), text.h)

    SDL_BlitSurface(text, None, screen, dstrect)
    SDL_UpdateRect(screen, 0, 0, 0, 0)

    # Set the text colorkey and convert to display format
    SDL_SetColorKey(text, SDL_SRCCOLORKEY | SDL_RLEACCEL, 0)
    temp = SDL_DisplayFormat(text)
    SDL_FreeSurface(text)
    text = temp

    # Wait for a keystroke, and blit text on mouse press
    done = 0
    while not done:
        event = SDL_WaitEventAndReturn()
        if event.type == SDL_MOUSEBUTTONDOWN:
            dstrect.x = event.x - text.w / 2
            dstrect.y = event.y - text.h / 2
            dstrect.w = text.w
            dstrect.h = text.h
            SDL_BlitSurface(text, None, screen, dstrect)
            SDL_UpdateRects(screen, [dstrect])
        elif event.type in (SDL_KEYDOWN, SDL_QUIT):
            done = 1
    
    SDL_FreeSurface(text)
    TTF_CloseFont(font)
    TTF_Quit()
    SDL_Quit()
