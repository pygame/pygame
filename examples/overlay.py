#!/usr/bin/env python
""" pygame.examples.overlay

The overlay module is deprecated now.
It is an olden days way to draw video quickly.
"""
import sys
import pygame as pg
from pygame.compat import xrange_

SR = (800, 600)
ovl = None

########################################################################
# Simple video player
def vPlayer(fName):
    global ovl
    f = open(fName, "rb")
    fmt = f.readline().strip()
    res = f.readline().strip()
    unused_col = f.readline().strip()
    if fmt != "P5":
        print("Unknown format( len %d ). Exiting..." % len(fmt))
        return

    w, h = [int(x) for x in res.split(" ")]
    h = (h * 2) / 3
    # Read into strings
    y = f.read(w * h)
    u = []
    v = []
    for _ in xrange_(0, h / 2):
        u.append(f.read(w / 2))
        v.append(f.read(w / 2))

    u = "".join(u)
    v = "".join(v)

    # Open overlay with the resolution specified
    ovl = pg.Overlay(pg.YV12_OVERLAY, (w, h))
    ovl.set_location(0, 0, w, h)

    ovl.display((y, u, v))
    while 1:
        pg.time.wait(10)
        for ev in pg.event.get():
            if ev.type in (pg.KEYDOWN, pg.QUIT):
                return


def main(fname):
    """play video file fname"""
    pg.init()
    try:
        pg.display.set_mode(SR)
        vPlayer(fname)
    finally:
        pg.quit()


# Test all modules
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: play_file <file_pattern>")
    else:
        main(sys.argv[1])
