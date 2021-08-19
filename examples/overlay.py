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
    fmt = f.readline().strip().decode()
    res = f.readline().strip().decode()
    unused_col = f.readline().strip()
    if fmt != "P5":
        print("Unknown format( len %d ). Exiting..." % len(fmt))
        return

    w, h = [int(x) for x in res.split(" ")]
    h = int((h * 2) / 3)
    # Read into strings
    y = f.read(w * h)
    u = bytes()
    v = bytes()
    for _ in xrange_(0, int(h / 2)):
        u += f.read(int(w / 2))
        v += f.read(int(w / 2))

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
        print("Example usage: python overlay.py data/yuv_1.pgm")
    else:
        main(sys.argv[1])

# Uncomment the code below for a quick test
# ------------------------------------------
# if __name__ == "__main__":
#     main('data/yuv_1.pgm')
