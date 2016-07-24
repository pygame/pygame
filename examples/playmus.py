#!/usr/bin/env python

"""A simple music player.

   Use pygame.mixer.music to play an audio file. A window is
   created to handle keyboard events for playback commands.

"""

from __future__ import print_function
import pygame
import pygame.freetype
from pygame.locals import *
import sys
import os

class Window(object):
    """The application's Pygame window

    A Window instance manages the creation of and drawing to a
    window. It is a singleton class. Only one instance can exist.

    """

    instance = None

    def __new__(cls, *args, **kwds):
        """Return an open Pygame window"""

        if Window.instance is not None:
            return Window.instance
        self = object.__new__(cls)
        pygame.display.init()
        self.screen = pygame.display.set_mode((600, 400))
        Window.instance = self
        return self

    def __init__(self, title):
        pygame.display.set_caption(title)
        self.screen.fill(Color('white'))
        pygame.display.flip()

        pygame.freetype.init()
        self.font = pygame.freetype.Font(None, 20)
        self.font.origin = True
        self.ascender = int(self.font.get_sized_ascender() * 1.5)
        self.descender = int(self.font.get_sized_descender() * 1.5)
        self.line_height = self.ascender - self.descender

        self.write_lines("'q', ESCAPE or close this window to quit\n"
                         "SPACE to play/pause\n"
                         "'r' to rewind\n"
                         "'f' to faid out over 5 seconds\n", 0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        pygame.display.quit()
        Window.instance = None

    def write_lines(self, text, line=0):
        w, h = self.screen.get_size()
        line_height = self.line_height
        nlines = h // line_height
        if line < 0:
            line = nlines + line
        for i, text_line in enumerate(text.split('\n'), line):
            y = i * line_height + self.ascender
            # Clear the line first.
            self.screen.fill(Color('white'),
                             (0, i * line_height, w, line_height))

            # Write new text.
            self.font.render_to(self.screen, (15, y), text_line, Color('blue'))
        pygame.display.flip()


def show_usage_message():
    print("Usage: python playmus.py <file>")
    print("       python -m pygame.examples.playmus <file>")

def main(file_path):
    """Play an audio file with pygame.mixer.music"""

    with Window(file_path) as win:
        win.write_lines('Loading ...', -1)
        pygame.mixer.init(frequency=44100)
        try:
            paused = False
            pygame.mixer.music.load(file_path)

            # Make sure the event loop ticks over at least every 0.5 seconds.
            pygame.time.set_timer(USEREVENT, 500)

            pygame.mixer.music.play()
            win.write_lines("Playing ...\n", -1)

            while pygame.mixer.music.get_busy():
                e = pygame.event.wait()
                if e.type == pygame.KEYDOWN:
                    key = e.key
                    if key == K_SPACE:
                        if paused:
                            pygame.mixer.music.unpause()
                            paused = False
                            win.write_lines("Playing ...\n", -1)
                        else:
                            pygame.mixer.music.pause()
                            paused = True
                            win.write_lines("Paused ...\n", -1)
                    elif key == K_r:
                        pygame.mixer.music.rewind()
                        if paused:
                            win.write_lines("Rewound.", -1)
                    elif key == K_f:
                        win.write_lines("Faiding out ...\n", -1)
                        pygame.mixer.music.fadeout(5000)
                        # when finished get_busy() will return 0.
                    elif key in [K_q, K_ESCAPE]:
                        pygame.mixer.music.stop()
                        # get_busy() will now return 0.
                elif e.type == QUIT:
                    pygame.mixer.music.stop()
                    # get_busy() will now return 0.
            pygame.time.set_timer(USEREVENT, 0)
        finally:
            pygame.mixer.quit()

if __name__ == '__main__':
# Check the only command line argument, a file path
    if len(sys.argv) != 2:
        show_usage_message()
    else:
        main(sys.argv[1])

