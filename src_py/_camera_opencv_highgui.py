
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

def init():
    pass

def quit():
    pass


class Camera:

    def __init__(self, device=0, size=(640, 480), mode="RGB"):
        """
        """
        self.camera = highgui.cvCreateCameraCapture(device)
        if not self.camera:
            raise ValueError("Could not open camera.  Sorry.")

    def set_controls(self, **kwargs):
        """
        """

    def set_resolution(self, width, height):
        """Sets the capture resolution. (without dialog)
        """
        # nothing to do here.
        pass

    def query_image(self):
        return True

    def stop(self):
        pass

    def start(self):
        # do nothing here... since the camera is already open.
        pass

    def get_buffer(self):
        """Returns a string containing the raw pixel data.
        """
        return self.get_surface().get_buffer()

    def get_image(self, dest_surf=None):
        return self.get_surface(dest_surf)

    def get_surface(self, dest_surf=None):
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

            pg_img = pygame.image.frombuffer(xxx, (xx.shape[1], xx.shape[0]), "RGB")

            # if there is a destination surface given, we blit onto that.
            if dest_surf:
                dest_surf.blit(pg_img, (0, 0))
            return dest_surf
            #return pg_img


if __name__ == "__main__":

    # try and use this camera stuff with the pygame camera example.
    import pygame.examples.camera

    pygame.camera.Camera = Camera
    pygame.camera.list_cameras = list_cameras
    pygame.examples.camera.main()
