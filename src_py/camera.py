import os
import sys
import platform
import warnings

_is_init = 0

def _setup_opencv_mac():
    global list_cameras, Camera, colorspace

    from pygame import _camera_opencv
    try:
        from pygame import _camera
    except ImportError:
        _camera = None

    list_cameras = _camera_opencv.list_cameras_darwin
    Camera = _camera_opencv.CameraMac
    if _camera:
        colorspace = _camera.colorspace

def _setup_opencv():
    global list_cameras, Camera, colorspace

    from pygame import _camera_opencv
    try:
        from pygame import _camera
    except ImportError:
        _camera = None

    list_cameras = _camera_opencv.list_cameras
    Camera = _camera_opencv.Camera
    if _camera:
        colorspace = _camera.colorspace

def _setup__camera():
    global list_cameras, Camera, colorspace

    from pygame import _camera
    
    list_cameras = _camera.list_cameras
    Camera = _camera.Camera
    colorspace = _camera.colorspace

def _setup_vidcapture():
    global list_cameras, Camera, colorspace

    from pygame import _camera_vidcapture
    try:
        from pygame import _camera
    except ImportError:
        _camera = None

    warnings.warn("The VideoCapture backend is not recommended and may be removed."
                  "For Python3 and Windows 8+, there is now a native Windows backend built into pygame.",
                  DeprecationWarning, stacklevel=2)

    _camera_vidcapture.init()
    list_cameras = _camera_vidcapture.list_cameras
    Camera = _camera_vidcapture.Camera
    if _camera:
        colorspace = _camera.colorspace

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

    # see if we have any user specified defaults in environments.
    camera_env = os.environ.get("PYGAME_CAMERA", "")
    if camera_env == "opencv": # prioritize opencv
        if "OpenCV" in possible_backends:
            possible_backends.remove("OpenCV")
        possible_backends = ["OpenCV"] + possible_backends
    if camera_env == "vidcapture": # prioritize vidcapture
        if "VideoCapture" in possible_backends:
            possible_backends.remove("VideoCapture")
        possible_backends = ["VideoCapture"] + possible_backends

    return possible_backends

backend_table = {"opencv-mac": _setup_opencv_mac,
                 "opencv": _setup_opencv,
                 "_camera (msmf)": _setup__camera,
                 "_camera (v4l2)": _setup__camera,
                 "videocapture": _setup_vidcapture}

def init(backend=None):
    global _is_init
    # select the camera module to import here.

    backends = get_backends()

    if not backends:
        _is_init = 1
        return
    else:
        backends = [b.lower() for b in backends]

    if not backend:
        backend = backends[0]
    else:
        backend = backend.lower()

    if backend not in backend_table:
        raise ValueError("unrecognized backend name")

    if backend not in backends:
        warnings.warn("We don't think this is a supported backend on this system, but we'll try it...", 
                       Warning, stacklevel=2)
        
    backend_table[backend]()

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
