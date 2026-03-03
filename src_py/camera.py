import os
import platform
import sys
import warnings
from abc import ABC, abstractmethod

from pygame import error

_is_init = False


class AbstractCamera(ABC):
    # set_controls and get_controls are not a part of the AbstractCamera ABC,
    # because implementations of the same can vary across different Camera
    # types
    @abstractmethod
    def __init__(self, *args, **kwargs):
        """ """

    @abstractmethod
    def start(self):
        """ """

    @abstractmethod
    def stop(self):
        """ """

    @abstractmethod
    def get_size(self):
        """ """

    @abstractmethod
    def query_image(self):
        """ """

    @abstractmethod
    def get_image(self, dest_surf=None):
        """ """

    @abstractmethod
    def get_raw(self):
        """ """


def _pre_init_placeholder():
    if not _is_init:
        raise error("pygame.camera is not initialized")

    # camera was init, and yet functions are not monkey patched. This should
    # not happen
    raise NotImplementedError()


def _pre_init_placeholder_varargs(*_, **__):
    _pre_init_placeholder()


class _PreInitPlaceholderCamera(AbstractCamera):
    def __init__(self, *args, **kwargs):
        _pre_init_placeholder()

    start = _pre_init_placeholder_varargs
    stop = _pre_init_placeholder_varargs
    get_controls = _pre_init_placeholder_varargs
    set_controls = _pre_init_placeholder_varargs
    get_size = _pre_init_placeholder_varargs
    query_image = _pre_init_placeholder_varargs
    get_image = _pre_init_placeholder_varargs
    get_raw = _pre_init_placeholder_varargs


list_cameras = _pre_init_placeholder
Camera = _PreInitPlaceholderCamera


def _colorspace_not_available(*args):
    raise RuntimeError("pygame is not built with colorspace support")


try:
    from pygame import _camera

    colorspace = _camera.colorspace
except ImportError:
    # Should not happen in most cases
    colorspace = _colorspace_not_available


def _setup_backend(backend):
    global list_cameras, Camera
    if backend == "opencv-mac":
        from pygame import _camera_opencv

        list_cameras = _camera_opencv.list_cameras_darwin
        Camera = _camera_opencv.CameraMac

    elif backend == "opencv":
        from pygame import _camera_opencv

        list_cameras = _camera_opencv.list_cameras
        Camera = _camera_opencv.Camera

    elif backend in ("_camera (msmf)", "_camera (v4l2)"):
        from pygame import _camera

        list_cameras = _camera.list_cameras
        Camera = _camera.Camera

    else:
        raise ValueError("unrecognized backend name")


def get_backends():
    possible_backends = []

    if sys.platform == "win32" and int(platform.win32_ver()[0].split(".")[0]) >= 8:
        possible_backends.append("_camera (MSMF)")

    if sys.platform == "linux":
        possible_backends.append("_camera (V4L2)")

    if "darwin" in sys.platform:
        possible_backends.append("OpenCV-Mac")

    possible_backends.append("OpenCV")

    # see if we have any user specified defaults in environments.
    camera_env = os.environ.get("PYGAME_CAMERA", "").lower()
    if camera_env == "opencv":  # prioritize opencv
        if "OpenCV" in possible_backends:
            possible_backends.remove("OpenCV")
        possible_backends = ["OpenCV"] + possible_backends

    return possible_backends


def init(backend=None):
    global _is_init
    # select the camera module to import here.

    backends = [b.lower() for b in get_backends()]
    if not backends:
        raise error("No camera backends are supported on your platform!")

    backend = backends[0] if backend is None else backend.lower()
    if backend not in backends:
        warnings.warn(
            "We don't think this is a supported backend on this system, "
            "but we'll try it...",
            Warning,
            stacklevel=2,
        )

    try:
        _setup_backend(backend)
    except ImportError:
        emsg = f"Backend '{backend}' is not supported on your platform!"
        if backend in ("opencv", "opencv-mac"):
            dep = "OpenCV"
            emsg += (
                f" Make sure you have '{dep}' installed to be able to use this backend"
            )

        raise error(emsg)

    _is_init = True


def quit():
    global _is_init, Camera, list_cameras
    # reset to their respective pre-init placeholders
    list_cameras = _pre_init_placeholder
    Camera = _PreInitPlaceholderCamera

    _is_init = False


if __name__ == "__main__":
    # try to use this camera stuff with the pygame camera example.
    import pygame.examples.camera

    # pygame.camera.Camera = Camera
    # pygame.camera.list_cameras = list_cameras
    pygame.examples.camera.main()
