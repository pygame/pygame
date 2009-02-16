


def init():
    global list_cameras, Camera, colorspace


    # TODO: select the camera module to import here.
    import _camera
    list_cameras = _camera.list_cameras
    Camera = _camera.Camera
    colorspace = _camera.colorspace


    pass


def quit():
    pass
 


def list_cameras():
    """
    """
    raise NotImplementedError()


class Camera:

    def __init__(self, device =0, size = (320, 200), mode = "RGB"):
        """
        """
        raise NotImplementedError()

    def set_resolution(self, width, height):
        """Sets the capture resolution. (without dialog)
        """
        pass

    def start(self):
        """
        """

    def stop(self):
        """
        """

    def get_buffer(self):
        """
        """


    def get_image(self, dest_surf = None):
        """
        """

    def get_surface(self, dest_surf = None):
        """
        """



if __name__ == "__main__":

    # try and use this camera stuff with the pygame camera example.
    import pygame.examples.camera

    #pygame.camera.Camera = Camera
    #pygame.camera.list_cameras = list_cameras
    pygame.examples.camera.main()

    

