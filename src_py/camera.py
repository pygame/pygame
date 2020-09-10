_is_init = 0

def init():
    global list_cameras, Camera, colorspace, _is_init

    import os, sys

    use_opencv = False
    use_vidcapture = False
    use__camera = True

    # Check the operating system
    if sys.platform == 'win32':  # If windows 32bit
        use_vidcapture = True
        use__camera = False
    elif "linux" in sys.platform: # If linux
        use__camera = True
    elif "darwin" in sys.platform: # If darwin
        use__camera = True
    else: # If any other operating system
        use_opencv = True

    # see if we have any user specified defaults in environments.
    camera_env = os.environ.get("PYGAME_CAMERA", "")
    if camera_env == "opencv": # If the pygame camera is set to use opencv
        use_opencv = True
    if camera_env == "vidcapture": # If the pygame camera is set to use vidcapture
        use_vidcapture = True

    # select the camera module to import here.

    # the _camera module has some code which can be reused by other modules.
    #  it will also be the default one.
    if use__camera:
        from pygame import _camera
        colorspace = _camera.colorspace

        list_cameras = _camera.list_cameras
        Camera = _camera.Camera

    if use_opencv: # If using Opencv for pygame camera
        try:
            from pygame import _camera_opencv_highgui
        except ImportError: # If cant import pygame camera
            _camera_opencv_highgui = None

        if _camera_opencv_highgui:
            _camera_opencv_highgui.init()

            list_cameras = _camera_opencv_highgui.list_cameras
            Camera = _camera_opencv_highgui.Camera

    if use_vidcapture: # If using VidCapture for pygame camera
        try:
            from pygame import _camera_vidcapture
        except ImportError: # If cant import pygame camera
            _camera_vidcapture = None

        if _camera_vidcapture:
            _camera_vidcapture.init()
            list_cameras = _camera_vidcapture.list_cameras
            Camera = _camera_vidcapture.Camera

    _is_init = 1


def quit():
    global _is_init
    _is_init = 0


def _check_init():
    # This function will check whether you have initialized the camera
    global _is_init
    if not _is_init:
        raise ValueError("Need to call camera.init() before using.")


def list_cameras():
    """
    """
    _check_init()
    raise NotImplementedError()


class Camera:
    # This will store all the infomation about the camera
    def __init__(self, device=0, size=(320, 200), mode="RGB"):
        """
        """
        _check_init()
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

    def set_controls(self, **kwargs):
        """
        """

    def get_image(self, dest_surf=None):
        """
        """

    def get_surface(self, dest_surf=None):
        """
        """


if __name__ == "__main__":

    # try and use this camera stuff with the pygame camera example.
    import pygame.examples.camera

    #pygame.camera.Camera = Camera
    #pygame.camera.list_cameras = list_cameras
    pygame.examples.camera.main()
