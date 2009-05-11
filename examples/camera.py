#!/usr/bin/env python

# 1. Basic image capturing and displaying using the camera module

import pygame
import pygame.camera
from pygame.locals import *


class VideoCapturePlayer(object):

   size = ( 640, 480 )
   def __init__(self, **argd):
       self.__dict__.update(**argd)
       super(VideoCapturePlayer, self).__init__(**argd)

       # create a display surface. standard pygame stuff
       self.display = pygame.display.set_mode( self.size, 0 )

       # gets a list of available cameras.
       self.clist = pygame.camera.list_cameras()
       if not self.clist:
           raise ValueError("Sorry, no cameras detected.")

       # creates the camera of the specified size and in RGB colorspace
       self.camera = pygame.camera.Camera(self.clist[0], self.size, "RGB")

       # starts the camera
       self.camera.start()

       self.clock = pygame.time.Clock()

       # create a surface to capture to.  for performance purposes, you want the
       # bit depth to be the same as that of the display surface.
       self.snapshot = pygame.surface.Surface(self.size, 0, self.display)

   def get_and_flip(self):
       # if you don't want to tie the framerate to the camera, you can check and
       # see if the camera has an image ready.  note that while this works
       # on most cameras, some will never return true.
       if 0 and self.camera.query_image():
           # capture an image

           self.snapshot = self.camera.get_image(self.snapshot)
       self.snapshot = self.camera.get_image(self.snapshot)
       #self.snapshot = self.camera.get_image()

       # blit it to the display surface.  simple!
       self.display.blit(self.snapshot, (0,0))
       pygame.display.flip()

   def main(self):
       going = True
       while going:
           events = pygame.event.get()
           for e in events:
               if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
                   going = False

           self.get_and_flip()
           self.clock.tick()
           print (self.clock.get_fps())

def main():
    pygame.init()
    pygame.camera.init()
    VideoCapturePlayer().main()
    pygame.quit()

if __name__ == '__main__':
    main()
