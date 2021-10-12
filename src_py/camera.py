import os
import sys
import platform

_is_init = 0

def setup_opencv_mac():
    global list_cameras, Camera, colorspace

    from pygame import _camera_opencv

    list_cameras = _camera_opencv.list_cameras_darwin
    Camera = _camera_opencv.CameraMac

def setup_opencv():
    global list_cameras, Camera, colorspace

    from pygame import _camera_opencv

    list_cameras = _camera_opencv.list_cameras
    Camera = _camera_opencv.Camera

def setup_opencv_legacy():
    global list_cameras, Camera, colorspace

    from pygame import _camera_opencv_highgui

    list_cameras = _camera_opencv_highgui.list_cameras
    Camera = _camera_opencv_highgui.Camera

def setup__camera():
    global list_cameras, Camera, colorspace

    from pygame import _camera
    
    list_cameras = _camera.list_cameras
    Camera = _camera.Camera
    colorspace = _camera.colorspace

def setup_vidcapture():
    global list_cameras, Camera, colorspace

    from pygame import _camera_vidcapture

    _camera_vidcapture.init()
    list_cameras = _camera_vidcapture.list_cameras
    Camera = _camera_vidcapture.Camera

def get_backends():
    possible_backends = []

    if sys.platform == 'win32' and sys.version_info > (3,) and int(platform.win32_ver()[0]) > 8:
        possible_backends.append("_camera (MSMF)")
    
    if "linux" in sys.platform:
        possible_backends.append("_camera (V4L2)")

    if "darwin" in sys.platform:
        possible_backends.append("OpenCV-Mac")

    possible_backends.append("OpenCV")

    if sys.platform == 'win32':
        possible_backends.append("VidCapture")

    possible_backends.append("OpenCV-Legacy")

    # see if we have any user specified defaults in environments.
    camera_env = os.environ.get("PYGAME_CAMERA", "")
    if camera_env == "opencv": # prioritize opencv legacy
        if "OpenCV-Legacy" in possible_backends:
            possible_backends.remove("OpenCV-Legacy")
        possible_backends = ["OpenCV-Legacy"] + possible_backends
    if camera_env == "vidcapture":
        if "VidCapture" in possible_backends:
            possible_backends.remove("VidCapture")
        possible_backends = ["VidCapture"] + possible_backends

    return possible_backends

backend_table = {"OpenCV-Mac": setup_opencv_mac,
                 "OpenCV": setup_opencv,
                 "OpenCV-Legacy": setup_opencv_legacy,
                 "_camera (MSMF)": setup__camera,
                 "_camera (V4l2)": setup__camera,
                 "VidCapture": setup_vidcapture}

def set_backend(backend):
    if backend not in backend_table:
        raise ValueError("unrecognized backend name")

    backend_table[backend]()

def init():
    global _is_init
    # select the camera module to import here.

    backends = get_backends()
    if backends:
        set_backend(backends[0])

    _is_init = 1

def quit():
    global _is_init
    _is_init = 0


def _check_init():
    global _is_init
    if not _is_init:
        raise ValueError("Need to call camera.init() before using.")


def list_cameras():
    """
    """
    _check_init()
    raise NotImplementedError()


class Camera:

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
