#!/usr/bin/env python
"""  This is a stress test for the fastevents module.

*Fast events does not appear faster!*

So far it looks like normal pygame.event is faster by up to two times.
So maybe fastevent isn't fast at all.

Tested on windowsXP sp2 athlon, and freebsd.

However... on my debian duron 850 machine fastevents is faster.
"""

import pygame
from pygame import *

# the config to try different settings out with the event queues.

# use the fastevent module or not.
use_fast_events = 1

# use pygame.display.flip().
#    otherwise we test raw event processing throughput.
with_display = 1

# limit the game loop to 40 fps.
slow_tick = 0

NUM_EVENTS_TO_POST = 200000



if use_fast_events:
    event_module = fastevent
else:
    event_module = event




from threading import Thread

class post_them(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.done = []
        self.stop = []

    def run(self):
        self.done = []
        self.stop = []
        for x in range(NUM_EVENTS_TO_POST):
            ee = event.Event(USEREVENT)
            try_post = 1

            # the pygame.event.post raises an exception if the event
            #   queue is full.  so wait a little bit, and try again.
            while try_post:
                try:
                    event_module.post(ee)
                    try_post = 0
                except:
                    pytime.sleep(0.001)
                    try_post = 1

            if self.stop:
                return
        self.done.append(1)



import time as pytime

def main():
    init()

    if use_fast_events:
        fastevent.init()

    c = time.Clock()

    win = display.set_mode((640, 480), RESIZABLE)
    display.set_caption("fastevent Workout")

    poster = post_them()

    t1 = pytime.time()
    poster.start()

    going = True
    while going:
#        for e in event.get():
        #for x in range(200):
        #    ee = event.Event(USEREVENT)
        #    r = event_module.post(ee)
        #    print (r)

        #for e in event_module.get():
        event_list = []
        event_list = event_module.get()

        for e in event_list:
            if e.type == QUIT:
                print (c.get_fps())
                poster.stop.append(1)
                going = False
            if e.type == KEYDOWN:
                if e.key == K_ESCAPE:
                    print (c.get_fps())
                    poster.stop.append(1)
                    going = False
        if poster.done:
            print (c.get_fps())
            print (c)
            t2 = pytime.time()
            print ("total time:%s" % (t2 - t1))
            print ("events/second:%s" % (NUM_EVENTS_TO_POST / (t2 - t1)))
            going = False
        if with_display:
            display.flip()
        if slow_tick:
            c.tick(40)


    pygame.quit()



if __name__ == '__main__':
    main()
