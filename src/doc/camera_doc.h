/* Auto generated file: with makeref.py .  Docs go in src/ *.doc . */
#define DOC_PYGAMECAMERA "pygame module for camera use"

#define DOC_PYGAMECAMERACOLORSPACE "colorspace(Surface, format, DestSurface = None) -> Surface\nSurface colorspace conversion"

#define DOC_PYGAMECAMERALISTCAMERAS "list_cameras() -> [cameras]\nreturns a list of available cameras"

#define DOC_PYGAMECAMERACAMERA "Camera(device, (width, height), format) -> Camera\nload a camera"

#define DOC_CAMERASTART "start() -> None\nopens, initializes, and starts capturing"

#define DOC_CAMERASTOP "stop() -> None\nstops, uninitializes, and closes the camera"

#define DOC_CAMERAGETCONTROLS "get_controls() -> (hflip = bool, vflip = bool, brightness)\ngets current values of user controls"

#define DOC_CAMERASETCONTROLS "set_controls(hflip = bool, vflip = bool, brightness) -> (hflip = bool, vflip = bool, brightness)\nchanges camera settings if supported by the camera"

#define DOC_CAMERAGETSIZE "get_size() -> (width, height)\nreturns the dimensions of the images being recorded"

#define DOC_CAMERAQUERYIMAGE "query_image() -> bool\nchecks if a frame is ready"

#define DOC_CAMERAGETIMAGE "get_image(Surface = None) -> Surface\ncaptures an image as a Surface"

#define DOC_CAMERAGETRAW "get_raw() -> string\nreturns an unmodified image as a string"



/* Docs in a comment... slightly easier to read. */

/*

pygame.camera
pygame module for camera use

pygame.camera.colorspace
 colorspace(Surface, format, DestSurface = None) -> Surface
Surface colorspace conversion

pygame.camera.list_cameras
 list_cameras() -> [cameras]
returns a list of available cameras

pygame.camera.Camera
 Camera(device, (width, height), format) -> Camera
load a camera

pygame.camera.Camera.start
 start() -> None
opens, initializes, and starts capturing

pygame.camera.Camera.stop
 stop() -> None
stops, uninitializes, and closes the camera

pygame.camera.Camera.get_controls
 get_controls() -> (hflip = bool, vflip = bool, brightness)
gets current values of user controls

pygame.camera.Camera.set_controls
 set_controls(hflip = bool, vflip = bool, brightness) -> (hflip = bool, vflip = bool, brightness)
changes camera settings if supported by the camera

pygame.camera.Camera.get_size
 get_size() -> (width, height)
returns the dimensions of the images being recorded

pygame.camera.Camera.query_image
 query_image() -> bool
checks if a frame is ready

pygame.camera.Camera.get_image
 get_image(Surface = None) -> Surface
captures an image as a Surface

pygame.camera.Camera.get_raw
 get_raw() -> string
returns an unmodified image as a string

*/