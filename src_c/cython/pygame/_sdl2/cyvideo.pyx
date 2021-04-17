from cpython cimport PyObject
from cpython.pycapsule cimport PyCapsule_GetPointer
from . import error
from . import error as errorfnc
from libc.stdlib cimport free, malloc

MESSAGEBOX_ERROR = _SDL_MESSAGEBOX_ERROR
MESSAGEBOX_WARNING = _SDL_MESSAGEBOX_WARNING
MESSAGEBOX_INFORMATION = _SDL_MESSAGEBOX_INFORMATION

cdef extern from "SDL.h" nogil:
    Uint32 SDL_GetWindowPixelFormat(SDL_Window* window)
    SDL_bool SDL_IntersectRect(const SDL_Rect* A,
                               const SDL_Rect* B,
                               SDL_Rect*       result)
    SDL_bool SDL_GetRelativeMouseMode()
    SDL_Renderer* SDL_GetRenderer(SDL_Window* window)


cdef extern from "pygame.h" nogil:
    ctypedef struct pgSurfaceObject:
        pass

    int pgSurface_Check(object surf)
    SDL_Surface* pgSurface_AsSurface(object surf)
    void import_pygame_surface()

    void import_pygame_base()

    int pgRect_Check(object rect)
    SDL_Rect *pgRect_FromObject(object obj, SDL_Rect *temp)
    object pgRect_New(SDL_Rect *r)
    object pgRect_New4(int x, int y, int w, int h)
    SDL_Rect pgRect_AsRect(object rect)
    void import_pygame_rect()

    object pgColor_New(Uint8 rgba[])
    object pgColor_NewLength(Uint8 rgba[], Uint8 length)
    void import_pygame_color()
    pgSurfaceObject *pgSurface_New2(SDL_Surface *info, int owner)

cdef extern from "pgcompat.h" nogil:
    pass

import_pygame_base()
import_pygame_color()
import_pygame_surface()
import_pygame_rect()

class RendererDriverInfo:
    def __repr__(self):
        return "<%s(name: %s, flags: 0x%02x, num_texture_formats: %d, max_texture_width: %d, max_texture_height: %d)>" % (
            self.__class__.__name__,
            self.name,
            self.flags,
            self.num_texture_formats,
            self.max_texture_width,
            self.max_texture_height,
        )

def get_drivers():
    cdef int num = SDL_GetNumRenderDrivers()
    cdef SDL_RendererInfo info
    cdef int ind
    for ind from 0 <= ind < num:
        SDL_GetRenderDriverInfo(ind, &info)
        ret = RendererDriverInfo()
        ret.name = info.name
        ret.flags = info.flags
        ret.num_texture_formats = info.num_texture_formats
        ret.max_texture_width = info.max_texture_width
        ret.max_texture_height = info.max_texture_height
        yield ret

def messagebox(title, message,
               window=None,
               bint info=False,
               bint warn=False,
               bint error=False,
               buttons=('OK', ),
               return_button=0,
               escape_button=0):
    """ Display a message box.

    :param str title: A title string or None.
    :param str message: A message string.
    :param bool info: If ``True``, display an info message.
    :param bool warn: If ``True``, display a warning message.
    :param bool error: If ``True``, display an error message.
    :param tuple buttons: An optional sequence of buttons to show to the user (strings).
    :param int return_button: Button index to use if the return key is hit (-1 for none).
    :param int escape_button: Button index to use if the escape key is hit (-1 for none).
    :return: The index of the button that was pushed.
    """
    # TODO: type check
    # TODO: color scheme
    cdef SDL_MessageBoxButtonData* c_buttons = NULL

    cdef SDL_MessageBoxData data
    data.flags = 0
    if warn:
        data.flags |= _SDL_MESSAGEBOX_WARNING
    if error:
        data.flags |= _SDL_MESSAGEBOX_ERROR
    if info:
        data.flags |= _SDL_MESSAGEBOX_INFORMATION

    if not window:
        data.window = NULL
    else:
        x = window._get_cpointer()
        # what does the window field of MessageBoxData do?
        data.window = <SDL_Window *>PyCapsule_GetPointer(x, NULL)

    if title is not None:
        title = title.encode('utf8')
        data.title = title
    else:
        data.title = NULL
    message = message.encode('utf8')
    data.message = message
    data.colorScheme = NULL

    cdef SDL_MessageBoxButtonData button
    if not buttons:
        button.flags = _SDL_MESSAGEBOX_BUTTON_RETURNKEY_DEFAULT |\
                       _SDL_MESSAGEBOX_BUTTON_ESCAPEKEY_DEFAULT
        button.buttonid = 0
        button.text = "OK"
        data.buttons = &button
        data.numbuttons = 1
    else:
        buttons_utf8 = [s.encode('utf8') for s in buttons]
        data.numbuttons = len(buttons)
        c_buttons =\
            <SDL_MessageBoxButtonData*>malloc(data.numbuttons * sizeof(SDL_MessageBoxButtonData))
        if not c_buttons:
            raise MemoryError()
        for i, but in enumerate(reversed(buttons_utf8)):
            c_buttons[i].flags = 0
            c_buttons[i].buttonid = data.numbuttons - i - 1
            if c_buttons[i].buttonid == return_button:
                c_buttons[i].flags |= _SDL_MESSAGEBOX_BUTTON_RETURNKEY_DEFAULT
            if c_buttons[i].buttonid == escape_button:
                c_buttons[i].flags |= _SDL_MESSAGEBOX_BUTTON_ESCAPEKEY_DEFAULT
            c_buttons[i].text = but
        data.buttons = c_buttons

    cdef int buttonid
    if SDL_ShowMessageBox(&data, &buttonid):
        free(c_buttons)
        raise errorfnc()

    free(c_buttons)
    return buttonid