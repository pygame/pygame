
import pygame
import numpy

import opencv
#this is important for capturing/displaying images
from opencv import highgui 



def list_cameras():
    """
    """
    # -1 for opencv means get any of them.
    return [-1]



class Camera:

    def __init__(self, device =0):
        """
        """
        self.camera = highgui.cvCreateCameraCapture(device)


    def set_resolution(self, width, height):
        """Sets the capture resolution. (without dialog)
        """
        # nothing to do here.
        pass

    def get_buffer(self):
        """Returns a string containing the raw pixel data.
        """
        return self.get_surface().get_buffer()

    def get_image(self):
        return self.get_surface()

    def get_surface(self):
        camera = self.camera

        im = highgui.cvQueryFrame(camera)
        #convert Ipl image to PIL image
        #print type(im)
        if im:
            xx = opencv.adaptors.Ipl2NumPy(im) 
            #print type(xx)
            #print xx.iscontiguous()
            #print dir(xx)
            #print xx.shape
            xxx = numpy.reshape(xx, (numpy.product(xx.shape),))
      
            if xx.shape[2] != 3:
                raise ValueError("not sure what to do about this size")

            pg_img = pygame.image.frombuffer(xxx, (xx.shape[1],xx.shape[0]), "RGB")
            return pg_img
            #return xxx
            #return opencv.adaptors.Ipl2PIL(im) 


    def get_surfacexx(self):
        """Returns a pygame Surface.
        """
        abuffer, width, height = self.get_buffer()
        if abuffer:
            if 1:
                surf = pygame.image.frombuffer(abuffer, (width, height), "RGB")

                # swap it from a BGR surface to an RGB surface.
                r,g,b,a = surf.get_masks()
                surf.set_masks((b,g,r,a))

                r,g,b,a = surf.get_shifts()
                surf.set_shifts((b,g,r,a))

                surf = pygame.transform.flip(surf, 0,1)

            else:

                # Need to flip the image.
                surf = pygame.image.fromstring(abuffer, (width, height), "RGB", 1)
                # swap it from a BGR surface to an RGB surface.
                r,g,b,a = surf.get_masks()
                surf.set_masks((b,g,r,a))

                r,g,b,a = surf.get_shifts()
                surf.set_shifts((b,g,r,a))
            return surf


if __name__ == "__main__":

    from pygame.locals import *
    pygame.init()

    #print list_cameras()
    #raise ""


    c = Camera(0)
    c.set_resolution(640,480)
    #c.display_capture_pin_properties()
    #c.display_capture_filter_properties()



    #import time
    #time.sleep(1.0)
    screen = pygame.display.set_mode((640,480))
    clk = pygame.time.Clock()

    going = True
    while going:
        events = pygame.event.get()
        for e in events:
            if e.type in [QUIT, KEYDOWN]:
                going = False

        asurf = c.get_surface()
        screen.blit(asurf, (0,0))
        pygame.display.flip()

        clk.tick()
        print clk.get_fps()




