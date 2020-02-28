#!/usr/bin/env python
""" pygame.examples.freetype_misc


Miscellaneous (or misc) means:
  "consisting of a mixture of various things that are not
   usually connected with each other"
   Adjective


All those words you read on computers, magazines, books, and such over the years?
Probably a lot of them were constructed with...

The FreeType Project:  a free, high-quality and portable Font engine.
https://freetype.org

Next time you're reading something. Think of them.


Herein lies a *BOLD* demo consisting of a mixture of various things.

        Not only is it a *BOLD* demo, it's an
        italics demo,
        a rotated demo,
        it's a blend,
        and is sized to go nicely with a cup of tea*.

        * also goes well with coffee.

Enjoy!
"""
import os
import pygame as pg
import pygame.freetype as freetype

colors = {
    "grey_light": pg.Color(200, 200, 200),
    "grey_dark": pg.Color(100, 100, 100),
    "green": pg.Color(50, 255, 63),
    "red": pg.Color(220, 30, 30),
    "blue": pg.Color(50, 75, 245),
}


def run():
    pg.init()

    fontdir = os.path.dirname(os.path.abspath(__file__))
    font = freetype.Font(os.path.join(fontdir, "data", "sans.ttf"))

    screen = pg.display.set_mode((800, 600))
    screen.fill(colors["grey_light"])

    font.underline_adjustment = 0.5
    font.pad = True
    font.render_to(
        screen,
        (32, 32),
        "Hello World",
        colors["red"],
        colors["grey_dark"],
        size=64,
        style=freetype.STYLE_UNDERLINE | freetype.STYLE_OBLIQUE,
    )
    font.pad = False

    font.render_to(
        screen,
        (32, 128),
        "abcdefghijklm",
        colors["grey_dark"],
        colors["green"],
        size=64,
    )

    font.vertical = True
    font.render_to(screen, (32, 200), "Vertical?", colors["blue"], None, size=32)
    font.vertical = False

    font.render_to(
        screen, (64, 190), "Let's spin!", colors["red"], None, size=48, rotation=55
    )

    font.render_to(
        screen, (160, 290), "All around!", colors["green"], None, size=48, rotation=-55
    )

    font.render_to(
        screen, (250, 220), "and BLEND", pg.Color(255, 0, 0, 128), None, size=64
    )

    font.render_to(
        screen, (265, 237), "or BLAND!", pg.Color(0, 0xCC, 28, 128), None, size=64
    )

    # Some pinwheels
    font.origin = True
    for angle in range(0, 360, 45):
        font.render_to(
            screen, (150, 420), ")", pg.Color("black"), size=48, rotation=angle
        )
    font.vertical = True
    for angle in range(15, 375, 30):
        font.render_to(
            screen, (600, 400), "|^*", pg.Color("orange"), size=48, rotation=angle
        )
    font.vertical = False
    font.origin = False

    utext = pg.compat.as_unicode(r"I \u2665 Unicode")
    font.render_to(screen, (298, 320), utext, pg.Color(0, 0xCC, 0xDD), None, size=64)

    utext = pg.compat.as_unicode(r"\u2665")
    font.render_to(
        screen, (480, 32), utext, colors["grey_light"], colors["red"], size=148
    )

    font.render_to(
        screen,
        (380, 380),
        "...yes, this is an SDL surface",
        pg.Color(0, 0, 0),
        None,
        size=24,
        style=freetype.STYLE_STRONG,
    )

    font.origin = True
    r = font.render_to(
        screen,
        (100, 530),
        "stretch",
        pg.Color("red"),
        None,
        size=(24, 24),
        style=freetype.STYLE_NORMAL,
    )
    font.render_to(
        screen,
        (100 + r.width, 530),
        " VERTICAL",
        pg.Color("red"),
        None,
        size=(24, 48),
        style=freetype.STYLE_NORMAL,
    )

    r = font.render_to(
        screen,
        (100, 580),
        "stretch",
        pg.Color("blue"),
        None,
        size=(24, 24),
        style=freetype.STYLE_NORMAL,
    )
    font.render_to(
        screen,
        (100 + r.width, 580),
        " HORIZONTAL",
        pg.Color("blue"),
        None,
        size=(48, 24),
        style=freetype.STYLE_NORMAL,
    )

    pg.display.flip()

    while 1:
        if pg.event.wait().type in (pg.QUIT, pg.KEYDOWN, pg.MOUSEBUTTONDOWN):
            break

    pg.quit()


if __name__ == "__main__":
    run()
