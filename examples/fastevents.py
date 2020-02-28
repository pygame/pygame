#!/usr/bin/env python
""" pygame.examples.fastevents

This is a stress test for the fastevents module.

If you are using threads, then fastevents is useful.
"""
import time as pytime
from threading import Thread

import pygame as pg

# the config to try different settings out with the event queues.

# use the fastevent module or not.
event_module = pg.fastevent
# event_module = event

# use pg.display.flip().
#    otherwise we test raw event processing throughput.
with_display = 1

# limit the game loop to 40 fps.
slow_tick = 0

NUM_EVENTS_TO_POST = 200000


class PostThem(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.done = []
        self.stop = []

    def run(self):
        self.done = []
        self.stop = []
        for x in range(NUM_EVENTS_TO_POST):
            ee = pg.event.Event(pg.USEREVENT)
            try_post = 1

            # the pg.event.post raises an exception if the event
            #   queue is full.  so wait a little bit, and try again.
            while try_post:
                try:
                    event_module.post(ee)
                    try_post = 0
                except pg.error:
                    pytime.sleep(0.001)
                    try_post = 1

            if self.stop:
                return
        self.done.append(1)


def main():
    pg.init()

    if hasattr(event_module, "init"):
        event_module.init()

    c = pg.time.Clock()

    pg.display.set_mode((640, 480), pg.RESIZABLE)
    pg.display.set_caption("fastevent Workout")

    poster = PostThem()

    t1 = pytime.time()
    poster.start()

    going = True
    while going:
        for e in event_module.get():
            if e.type == pg.QUIT:
                print(c.get_fps())
                poster.stop.append(1)
                going = False
            if e.type == pg.KEYDOWN:
                if e.key == pg.K_ESCAPE:
                    print(c.get_fps())
                    poster.stop.append(1)
                    going = False
        if poster.done:
            print(c.get_fps())
            print(c)
            t2 = pytime.time()
            print("total time:%s" % (t2 - t1))
            print("events/second:%s" % (NUM_EVENTS_TO_POST / (t2 - t1)))
            going = False
        if with_display:
            pg.display.flip()
        if slow_tick:
            c.tick(40)

    pg.quit()


if __name__ == "__main__":
    main()
