"""  This is a stress test for the fastevents module.

So far it looks like normal pygame.event is faster by up to two times.
So maybe fastevent isn't fast at all.

tested on windowsXP sp2 athlon.
"""

from pygame import *



# the config to try different settings out with the event queues.


with_display = 1
slow_tick = 0
use_fast_events = 1
NUM_EVENTS_TO_POST = 200000


if use_fast_events:
    event_module = fastevent
else:
    event_module = event




from threading import Thread

class post_them(Thread):
    
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
    
    fastevent.init()

    c = time.Clock()

    win = display.set_mode((640, 480), RESIZABLE)
    display.set_caption("fastevent Workout")

    poster = post_them()

    t1 = pytime.time()
    poster.start()


    while 1:
#        for e in event.get():
        #for x in range(200):
        #    ee = event.Event(USEREVENT)
        #    r = event_module.post(ee)
        #    print r
        
        #for e in event_module.get():
        event_list = []
        event_list = event_module.get()

        for e in event_list:
            if e.type == QUIT:
                print c.get_fps()
                poster.stop.append(1)
                return
            if e.type == KEYDOWN:
                if e.key == K_ESCAPE:
                    print c.get_fps()
                    poster.stop.append(1)
                    return
        if poster.done:
            print c.get_fps()
            print c
            t2 = pytime.time()
            print "total time:%s" % (t2 - t1)
            print "events/second:%s" % (NUM_EVENTS_TO_POST / (t2 - t1))
            return
        if with_display:
            display.flip()
        if slow_tick:
            c.tick(40)
        
        




if __name__ == '__main__': main()
