
import pygame

def list_cameras():
    return [0]

    # this just cycles through all the cameras trying to open them
    cameras = []
    for x in range(256):
        try:
            c = Camera(x)
        except:
            break
        cameras.append(x)

    return cameras


def init():
    global vidcap
    import vidcap as vc
    vidcap = vc

def quit():
    global vidcap
    pass
    del vidcap



class Camera:

    def __init__(self, device =0,
                       size = (640,480),
                       mode = "RGB",
                       show_video_window=0):
        """device:  VideoCapture enumerates the available video capture devices
                    on your system.  If you have more than one device, specify
                    the desired one here.  The device number starts from 0.

           show_video_window: 0 ... do not display a video window (the default)
                              1 ... display a video window

                            Mainly used for debugging, since the video window
                            can not be closed or moved around.
        """
        self.dev = vidcap.new_Dev(device, show_video_window)
        width, height = size
        self.dev.setresolution(width, height)

    def display_capture_filter_properties(self):
        """Displays a dialog containing the property page of the capture filter.

        For VfW drivers you may find the option to select the resolution most
        likely here.
        """
        self.dev.displaycapturefilterproperties()

    def display_capture_pin_properties(self):
        """Displays a dialog containing the property page of the capture pin.

        For WDM drivers you may find the option to select the resolution most
        likely here.
        """
        self.dev.displaycapturepinproperties()

    def set_resolution(self, width, height):
        """Sets the capture resolution. (without dialog)
        """
        self.dev.setresolution(width, height)

    def get_buffer(self):
        """Returns a string containing the raw pixel data.
        """
        return self.dev.getbuffer()

    def start(self):
        """
        """
    def set_controls(self, **kwargs):
        """
        """

    def stop(self):
        """
        """

    def get_image(self, dest_surf = None):
        return self.get_surface(dest_surf)

    def get_surface(self, dest_surf = None):
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

                # if there is a destination surface given, we blit onto that.
                if dest_surf:
                    dest_surf.blit(surf, (0,0))
                return dest_surf

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
    import pygame.examples.camera

    pygame.camera.Camera = Camera
    pygame.camera.list_cameras = list_cameras
    pygame.examples.camera.main()


