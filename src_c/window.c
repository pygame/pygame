#define PYGAMEAPI_WINDOW_INTERNAL

#include "pygame.h"

#include "pgcompat.h"
#include "pgopengl.h"

#include "doc/sdl2_video_doc.h"
#include "doc/window_doc.h"

static int is_window_mod_init = 0;

#if !defined(__APPLE__)
static char *icon_defaultname = "pygame_icon.bmp";
static int icon_colorkey = 0;
#else
static char *icon_defaultname = "pygame_icon_mac.bmp";
static int icon_colorkey = -1;
#endif

static char *pkgdatamodule_name = "pygame.pkgdata";
static char *imagemodule_name = "pygame.image";
static char *resourcefunc_name = "getResource";
static char *load_basicfunc_name = "load_basic";

// Copied from display.c
static void
pg_close_file(PyObject *fileobj)
{
    PyObject *result = PyObject_CallMethod(fileobj, "close", NULL);
    if (result) {
        Py_DECREF(result);
    }
    else {
        PyErr_Clear();
    }
}

// Copied from display.c
static PyObject *
pg_display_resource(char *filename)
{
    PyObject *imagemodule = NULL;
    PyObject *pkgdatamodule = NULL;
    PyObject *fresult = NULL;
    PyObject *result = NULL;
    PyObject *name = NULL;

    pkgdatamodule = PyImport_ImportModule(pkgdatamodule_name);
    if (!pkgdatamodule) {
        goto display_resource_end;
    }

    imagemodule = PyImport_ImportModule(imagemodule_name);
    if (!imagemodule) {
        goto display_resource_end;
    }

    fresult =
        PyObject_CallMethod(pkgdatamodule, resourcefunc_name, "s", filename);
    if (!fresult) {
        goto display_resource_end;
    }

    name = PyObject_GetAttrString(fresult, "name");
    if (name != NULL) {
        if (PyUnicode_Check(name)) {
            pg_close_file(fresult);
            Py_DECREF(fresult);
            fresult = name;
            name = NULL;
        }
    }
    else {
        PyErr_Clear();
    }

    result =
        PyObject_CallMethod(imagemodule, load_basicfunc_name, "O", fresult);
    if (!result) {
        goto display_resource_end;
    }

display_resource_end:
    Py_XDECREF(pkgdatamodule);
    Py_XDECREF(imagemodule);
    Py_XDECREF(fresult);
    Py_XDECREF(name);
    return result;
}

static PyTypeObject pgWindow_Type;
static GL_glViewport_Func p_glViewport = NULL;

#define pgWindow_Check(x) \
    (PyObject_IsInstance((x), (PyObject *)&pgWindow_Type))

static inline PyObject *
pg_get_pg_window(SDL_Window *win)
{
#if SDL_VERSION_ATLEAST(3, 0, 0)
    return SDL_GetPointerProperty(SDL_GetWindowProperties(win), "pg_window",
                                  NULL);
#else
    return SDL_GetWindowData(win, "pg_window");
#endif
}

static inline void
pg_set_pg_window(SDL_Window *win, PyObject *pg_win)
{
#if SDL_VERSION_ATLEAST(3, 0, 0)
    SDL_SetPointerProperty(SDL_GetWindowProperties(win), "pg_window", pg_win);
#else
    SDL_SetWindowData(win, "pg_window", pg_win);
#endif
}

static PyObject *
get_grabbed_window(PyObject *self, PyObject *_null)
{
    SDL_Window *grabbed = SDL_GetGrabbedWindow();
    PyObject *win_obj = NULL;
    if (grabbed) {
        win_obj = pg_get_pg_window(grabbed);
        if (!win_obj) {
            Py_RETURN_NONE;
        }
        Py_INCREF(win_obj);
        return win_obj;
    }
    Py_RETURN_NONE;
}

static PyObject *
window_destroy(pgWindowObject *self, PyObject *_null)
{
    if (self->_win) {
        if (self->_is_borrowed && pg_GetDefaultWindow() == self->_win) {
            pgSurface_AsSurface(pg_GetDefaultWindowSurface()) = NULL;
            pg_SetDefaultWindowSurface(NULL);
            pg_SetDefaultWindow(NULL);
        }

        if (self->context != NULL) {
            SDL_GL_DeleteContext(self->context);
        }

        SDL_DestroyWindow(self->_win);
        self->_win = NULL;
    }
    if (self->surf) {
        // Set the internal surface to NULL to make pygame surface invalid
        // since this surface will be deallocated by SDL when the window is
        // destroyed.
        self->surf->surf = NULL;

        Py_DECREF(self->surf);
        self->surf = NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
window_get_surface(pgWindowObject *self, PyObject *_null)
{
    PyObject *surf = NULL;
    SDL_Surface *_surf;

    if (self->_is_borrowed) {
        surf = (PyObject *)pg_GetDefaultWindowSurface();
        if (!surf) {
            return RAISE(pgExc_SDLError,
                         "display.set_mode has not been called yet.");
        }
        Py_INCREF(surf);
        return surf;
    }

    _surf = SDL_GetWindowSurface(self->_win);
    if (!_surf) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    if (pg_GetDefaultConvertFormat() == 0) {
        pg_SetDefaultConvertFormat(PG_SURF_FORMATENUM(_surf));
    }

    if (self->surf == NULL) {
        self->surf = pgSurface_New2(_surf, SDL_FALSE);
        if (!self->surf) {
            return NULL;
        }
    }
    self->surf->surf = _surf;

    Py_INCREF(self->surf);
    return (PyObject *)self->surf;
}

static PyObject *
window_flip(pgWindowObject *self, PyObject *_null)
{
    bool success;

    if (self->context == NULL) {
        if (!self->surf) {
            return RAISE(pgExc_SDLError,
                         "the Window has no surface associated with it, did "
                         "you forget to call Window.get_surface()");
        }

        Py_BEGIN_ALLOW_THREADS;
        success = PG_UpdateWindowSurface(self->_win);
        Py_END_ALLOW_THREADS;
        if (!success) {
            return RAISE(pgExc_SDLError, SDL_GetError());
        }
    }
    else {
        Py_BEGIN_ALLOW_THREADS;
        SDL_GL_SwapWindow(self->_win);
        Py_END_ALLOW_THREADS;
    }
    Py_RETURN_NONE;
}

/* Exception already set */
static int
_window_opengl_set_viewport(SDL_Window *window, SDL_GLContext context,
                            int wnew, int hnew)
{
#if SDL_VERSION_ATLEAST(3, 0, 0)
    if (!SDL_GL_MakeCurrent(window, context))
#else
    if (SDL_GL_MakeCurrent(window, context) < 0)
#endif
    {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return -1;
    }
    if (p_glViewport == NULL) {
        PyErr_SetString(pgExc_SDLError, "glViewport function is unavailable");
        return -1;
    }
    p_glViewport(0, 0, wnew, hnew);
    return 0;
}

// Callback function for surface auto resize or OpenGL viewport update
#if SDL_VERSION_ATLEAST(3, 0, 0)
static bool SDLCALL
#else
static int SDLCALL
#endif
_resize_event_watch(void *userdata, SDL_Event *event)
{
    pgWindowObject *event_window_pg;
    SDL_Window *event_window;
#if SDL_VERSION_ATLEAST(3, 0, 0)
    if ((event->type != SDL_WINDOWEVENT_SIZE_CHANGED)) {
        return 0;
    }
#else
    if ((event->type != SDL_WINDOWEVENT)) {
        return 0;
    }
    if (event->window.event != SDL_WINDOWEVENT_SIZE_CHANGED) {
        return 0;
    }
#endif
    event_window = SDL_GetWindowFromID(event->window.windowID);
    event_window_pg = (pgWindowObject *)pg_get_pg_window(event_window);

    if (!event_window_pg) {
        return 0;
    }

    if (event_window_pg->_is_borrowed) {
        // have been handled by event watch in display.c
        return 0;
    }

    if (event_window_pg->context != NULL) {
        if (_window_opengl_set_viewport(event_window, event_window_pg->context,
                                        event->window.data1,
                                        event->window.data2) < 0) {
            return PyErr_WarnEx(PyExc_RuntimeWarning,
                                "Failed to set OpenGL viewport", 0);
        }
    }

    if (!event_window_pg->surf) {
        return 0;
    }

    event_window_pg->surf->surf = SDL_GetWindowSurface(event_window);
    return 0;
}

static PyObject *
window_set_windowed(pgWindowObject *self, PyObject *_null)
{
#if SDL_VERSION_ATLEAST(3, 0, 0)
    if (!SDL_SetWindowFullscreen(self->_win, 0)) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }
#else
    if (SDL_SetWindowFullscreen(self->_win, 0)) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }
#endif
    Py_RETURN_NONE;
}

#if SDL_VERSION_ATLEAST(3, 0, 0)
/* Returns 1 on success, 0 on failure */
static int
pg_window_set_fullscreen(SDL_Window *window, int desktop)
{
    int ret = 0;
    SDL_DisplayMode **modes = NULL;
    SDL_DisplayMode *chosen_mode = NULL;
    if (!desktop) {
        /* if not desktop fullscreen, get the first display mode available */
        SDL_DisplayID disp = SDL_GetDisplayForWindow(window);
        if (!disp) {
            goto end;
        }
        modes = SDL_GetFullscreenDisplayModes(disp, NULL);
        if (!modes) {
            goto end;
        }
        chosen_mode = modes[0];
        if (!chosen_mode) {
            SDL_SetError("Could not get fullscreen display mode");
            goto end;
        }
    }

    if (!SDL_SetWindowFullscreen(window, 1)) {
        goto end;
    }
    if (!SDL_SetWindowFullscreenMode(window, chosen_mode)) {
        goto end;
    }

    ret = 1;
end:
    SDL_free(modes);
    return ret;
}
#endif

static PyObject *
window_set_fullscreen(pgWindowObject *self, PyObject *args, PyObject *kwargs)
{
    int desktop = 0;
    char *kwids[] = {"desktop", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|p", kwids, &desktop)) {
        return NULL;
    }
#if SDL_VERSION_ATLEAST(3, 0, 0)
    if (!pg_window_set_fullscreen(self->_win, desktop)) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }
#else
    if (SDL_SetWindowFullscreen(self->_win, desktop
                                                ? SDL_WINDOW_FULLSCREEN_DESKTOP
                                                : SDL_WINDOW_FULLSCREEN)) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }
#endif
    Py_RETURN_NONE;
}

static PyObject *
window_focus(pgWindowObject *self, PyObject *args, PyObject *kwargs)
{
    int input_only = SDL_FALSE;
    char *kwids[] = {"input_only", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|p", kwids, &input_only)) {
        return NULL;
    }
#if SDL_VERSION_ATLEAST(3, 0, 0)
    /* input_only ignored on SDL3 */
    SDL_RaiseWindow(self->_win);
#else
    if (input_only) {
        if (PyErr_WarnEx(PyExc_DeprecationWarning,
                         "The input_only kwarg has been deprecated and may be "
                         "removed in a future version",
                         1) == -1) {
            return NULL;
        }
        if (SDL_SetWindowInputFocus(self->_win)) {
            return RAISE(pgExc_SDLError, SDL_GetError());
        }
    }
    else {
        SDL_RaiseWindow(self->_win);
    }
#endif
    Py_RETURN_NONE;
}

static PyObject *
window_get_focused(pgWindowObject *self, void *v)
{
    return PyBool_FromLong(
        (SDL_GetWindowFlags(self->_win) & SDL_WINDOW_INPUT_FOCUS) != 0);
}

static PyObject *
window_hide(pgWindowObject *self, PyObject *_null)
{
    SDL_HideWindow(self->_win);
    Py_RETURN_NONE;
}

static PyObject *
window_show(pgWindowObject *self, PyObject *_null)
{
    SDL_ShowWindow(self->_win);
    Py_RETURN_NONE;
}

static PyObject *
window_restore(pgWindowObject *self, PyObject *_null)
{
    SDL_RestoreWindow(self->_win);
    Py_RETURN_NONE;
}

static PyObject *
window_maximize(pgWindowObject *self, PyObject *_null)
{
    SDL_MaximizeWindow(self->_win);
    Py_RETURN_NONE;
}

static PyObject *
window_minimize(pgWindowObject *self, PyObject *_null)
{
    SDL_MinimizeWindow(self->_win);
    Py_RETURN_NONE;
}

/* Based on code from sdl2-compat */
static int SDLCALL
PG_SetWindowModalFor(SDL_Window *modal_window, SDL_Window *parent_window)
{
#if SDL_VERSION_ATLEAST(3, 0, 0)
    if (!modal_window) {
        SDL_SetError("Invalid window");
        return -1;
    }
    if (SDL_GetWindowFlags(modal_window) & SDL_WINDOW_MODAL) {
        SDL_SetWindowModal(modal_window, false);
    }
    if (SDL_SetWindowParent(modal_window, parent_window)) {
        int ret = 0;
        if (parent_window) {
            ret = SDL_SetWindowModal(modal_window, true) ? 0 : -1;
        }
        return ret;
    }

    return -1;
#else
    return SDL_SetWindowModalFor(modal_window, parent_window);
#endif
}

static PyObject *
window_set_modal_for(pgWindowObject *self, PyObject *arg)
{
    if (!pgWindow_Check(arg)) {
        return RAISE(PyExc_TypeError,
                     "Argument to set_modal_for must be a Window.");
    }
    if (PG_SetWindowModalFor(self->_win, ((pgWindowObject *)arg)->_win) < 0) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }
    Py_RETURN_NONE;
}

static PyObject *
window_set_icon(pgWindowObject *self, PyObject *arg)
{
    if (!pgSurface_Check(arg)) {
        return RAISE(PyExc_TypeError,
                     "Argument to set_icon must be a Surface.");
    }
    SDL_SetWindowIcon(self->_win, pgSurface_AsSurface(arg));
    Py_RETURN_NONE;
}

static int
window_set_grab_mouse(pgWindowObject *self, PyObject *arg, void *v)
{
    int enable = PyObject_IsTrue(arg);
    if (enable == -1) {
        return -1;
    }

#if SDL_VERSION_ATLEAST(2, 0, 16)
    SDL_SetWindowMouseGrab(self->_win, enable);
#else
    SDL_SetWindowGrab(self->_win, enable);
#endif

    return 0;
}

static PyObject *
window_get_grab_mouse(pgWindowObject *self, void *v)
{
#if SDL_VERSION_ATLEAST(2, 0, 16)
    return PyBool_FromLong(SDL_GetWindowFlags(self->_win) &
                           SDL_WINDOW_MOUSE_GRABBED);
#else
    return PyBool_FromLong(SDL_GetWindowFlags(self->_win) &
                           SDL_WINDOW_INPUT_GRABBED);
#endif
}

static PyObject *
window_get_mouse_grabbed(pgWindowObject *self, void *v)
{
#if SDL_VERSION_ATLEAST(2, 0, 16)
    return PyBool_FromLong(SDL_GetWindowMouseGrab(self->_win));
#else
    return PyBool_FromLong(SDL_GetWindowGrab(self->_win));
#endif
}

static int
window_set_grab_keyboard(pgWindowObject *self, PyObject *arg, void *v)
{
#if SDL_VERSION_ATLEAST(2, 0, 16)
    int enable = PyObject_IsTrue(arg);
    if (enable == -1) {
        return -1;
    }

    SDL_SetWindowKeyboardGrab(self->_win, enable);
#else
    if (PyErr_WarnEx(PyExc_Warning, "'grab_keyboard' requires SDL 2.0.16+",
                     1) == -1) {
        return -1;
    }
#endif
    return 0;
}

static PyObject *
window_get_grab_keyboard(pgWindowObject *self, void *v)
{
#if SDL_VERSION_ATLEAST(2, 0, 16)
    return PyBool_FromLong(SDL_GetWindowFlags(self->_win) &
                           SDL_WINDOW_KEYBOARD_GRABBED);
#else
    if (PyErr_WarnEx(PyExc_Warning, "'grab_keyboard' requires SDL 2.0.16+",
                     1) == -1) {
        return NULL;
    }
    return PyBool_FromLong(SDL_FALSE);
#endif
}

static PyObject *
window_get_keyboard_grabbed(pgWindowObject *self, void *v)
{
#if SDL_VERSION_ATLEAST(2, 0, 16)
    return PyBool_FromLong(SDL_GetWindowKeyboardGrab(self->_win));
#else
    if (PyErr_WarnEx(PyExc_Warning, "'keyboard_captured' requires SDL 2.0.16+",
                     1) == -1) {
        return NULL;
    }
    return PyBool_FromLong(SDL_FALSE);
#endif
}

static int
window_set_title(pgWindowObject *self, PyObject *arg, void *v)
{
    const char *title;
    if (!PyUnicode_Check(arg)) {
        PyErr_SetString(PyExc_TypeError,
                        "Argument to set_title must be a str.");
        return -1;
    }
    title = PyUnicode_AsUTF8(arg);
    SDL_SetWindowTitle(self->_win, title);
    return 0;
}

static PyObject *
window_get_title(pgWindowObject *self, void *v)
{
    const char *title = SDL_GetWindowTitle(self->_win);
    return PyUnicode_FromString(title);
}

static int
window_set_resizable(pgWindowObject *self, PyObject *arg, void *v)
{
    int enable = PyObject_IsTrue(arg);
    if (enable == -1) {
        return -1;
    }

    SDL_SetWindowResizable(self->_win, enable);

    return 0;
}

static PyObject *
window_get_resizable(pgWindowObject *self, void *v)
{
    return PyBool_FromLong(SDL_GetWindowFlags(self->_win) &
                           SDL_WINDOW_RESIZABLE);
}

static int
window_set_borderless(pgWindowObject *self, PyObject *arg, void *v)
{
    int enable = PyObject_IsTrue(arg);
    if (enable == -1) {
        return -1;
    }

    SDL_SetWindowBordered(self->_win, !enable);

    return 0;
}

static PyObject *
window_get_borderless(pgWindowObject *self, void *v)
{
    return PyBool_FromLong(SDL_GetWindowFlags(self->_win) &
                           SDL_WINDOW_BORDERLESS);
}

static int
window_set_always_on_top(pgWindowObject *self, PyObject *arg, void *v)
{
#if SDL_VERSION_ATLEAST(2, 0, 16)
    int enable = PyObject_IsTrue(arg);
    if (enable == -1) {
        return -1;
    }

    SDL_SetWindowAlwaysOnTop(self->_win, enable);
#else
    if (PyErr_WarnEx(PyExc_Warning,
                     "Setting 'always_on_top' requires SDL 2.0.16+",
                     1) == -1) {
        return -1;
    }
#endif  // SDL_VERSION_ATLEAST(2, 0, 16)
    return 0;
}

static PyObject *
window_get_always_on_top(pgWindowObject *self, void *v)
{
    return PyBool_FromLong(SDL_GetWindowFlags(self->_win) &
                           SDL_WINDOW_ALWAYS_ON_TOP);
}

static PyObject *
window_get_window_id(pgWindowObject *self, PyObject *_null)
{
    Uint32 window_id = SDL_GetWindowID(self->_win);
    if (!window_id) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }
    return PyLong_FromLong(window_id);
}

static int
window_set_mouse_rect(pgWindowObject *self, PyObject *arg, void *v)
{
#if SDL_VERSION_ATLEAST(2, 0, 18)
    SDL_Rect tmp_rect;
    SDL_Rect *mouse_rect_p = pgRect_FromObject(arg, &tmp_rect);
    if (mouse_rect_p == NULL && arg != Py_None) {
        PyErr_SetString(PyExc_TypeError,
                        "mouse_rect should be a Rect-like object or None");
        return -1;
    }
#if SDL_VERSION_ATLEAST(3, 0, 0)
    if (!SDL_SetWindowMouseRect(self->_win, mouse_rect_p)) {
#else
    if (SDL_SetWindowMouseRect(self->_win, mouse_rect_p) < 0) {
#endif
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return -1;
    }
#else
    if (PyErr_WarnEx(PyExc_Warning,
                     "Setting 'mouse_rect' requires SDL 2.0.18+", 1) == -1) {
        return -1;
    }
#endif  // SDL_VERSION_ATLEAST(2, 0, 18)
    return 0;
}

static PyObject *
window_get_mouse_rect(pgWindowObject *self, void *v)
{
#if SDL_VERSION_ATLEAST(2, 0, 18)
    const SDL_Rect *mouse_rect_p = SDL_GetWindowMouseRect(self->_win);
    if (mouse_rect_p == NULL) {
        Py_RETURN_NONE;
    }
    return pgRect_New((SDL_Rect *)mouse_rect_p);
#else
    if (PyErr_WarnEx(PyExc_Warning,
                     "Getting 'mouse_rect' requires SDL 2.0.18+", 1) == -1) {
        return NULL;
    }
    Py_RETURN_NONE;
#endif  // SDL_VERSION_ATLEAST(2, 0, 18)
}

static int
window_set_size(pgWindowObject *self, PyObject *arg, void *v)
{
    int w, h;

    if (!pg_TwoIntsFromObj(arg, &w, &h)) {
        PyErr_SetString(PyExc_TypeError, "invalid size argument");
        return -1;
    }

    if (w <= 0 || h <= 0) {
        PyErr_SetString(
            PyExc_ValueError,
            "width or height should not be less than or equal to zero");
        return -1;
    }

    SDL_SetWindowSize(self->_win, w, h);
    if (self->surf) {
        /* Ensure that the underlying surf is immediately updated, instead of
         * relying on the event callback */
        self->surf->surf = SDL_GetWindowSurface(self->_win);
    }
    if (self->context != NULL) {
        /* Update the OpenGL viewport immediately instead of relying on the
         * event callback */
        if (_window_opengl_set_viewport(self->_win, self->context, w, h) < 0) {
            return -1;
        }
    }

    return 0;
}

static PyObject *
window_get_size(pgWindowObject *self, void *v)
{
    int w, h;
    SDL_GetWindowSize(self->_win, &w, &h);

    return pg_tuple_couple_from_values_int(w, h);
}

static int
window_set_minimum_size(pgWindowObject *self, PyObject *arg, void *v)
{
    int w, h;
    int max_w, max_h;

    if (!pg_TwoIntsFromObj(arg, &w, &h)) {
        PyErr_SetString(PyExc_TypeError, "invalid size argument");
        return -1;
    }

    if (w < 0 || h < 0) {
        PyErr_SetString(
            PyExc_ValueError,
            "minimum width or height should not be less than zero");
        return -1;
    }

    SDL_GetWindowMaximumSize(self->_win, &max_w, &max_h);
    if ((max_w > 0 && max_h > 0) && (w > max_w || h > max_h)) {
        PyErr_SetString(PyExc_ValueError,
                        "minimum width or height should not be greater than "
                        "maximum width or height respectively");
        return -1;
    }

    SDL_SetWindowMinimumSize(self->_win, w, h);

    return 0;
}

static PyObject *
window_get_minimum_size(pgWindowObject *self, void *v)
{
    int w, h;
    SDL_GetWindowMinimumSize(self->_win, &w, &h);

    return pg_tuple_couple_from_values_int(w, h);
}

static int
window_set_maximum_size(pgWindowObject *self, PyObject *arg, void *v)
{
    int w, h;
    int min_w, min_h;

    if (!pg_TwoIntsFromObj(arg, &w, &h)) {
        PyErr_SetString(PyExc_TypeError, "invalid size argument");
        return -1;
    }

    if (w < 0 || h < 0) {
        PyErr_SetString(
            PyExc_ValueError,
            "maximum width or height should not be less than zero");
        return -1;
    }

    SDL_GetWindowMinimumSize(self->_win, &min_w, &min_h);
    if (w < min_w || h < min_h) {
        PyErr_SetString(PyExc_ValueError,
                        "maximum width or height should not be less than "
                        "minimum width or height respectively");
        return -1;
    }

    SDL_SetWindowMaximumSize(self->_win, w, h);

    return 0;
}

static PyObject *
window_get_maximum_size(pgWindowObject *self, void *v)
{
    int w, h;
    SDL_GetWindowMaximumSize(self->_win, &w, &h);

    return pg_tuple_couple_from_values_int(w, h);
}

static int
window_set_position(pgWindowObject *self, PyObject *arg, void *v)
{
    int x, y;

    if (PyLong_Check(arg)) {
        x = y = PyLong_AsLong(arg);
        if (x != SDL_WINDOWPOS_CENTERED && x != SDL_WINDOWPOS_UNDEFINED) {
            PyErr_SetString(PyExc_TypeError, "invalid position argument");
            return -1;
        }
    }
    else if (!pg_TwoIntsFromObj(arg, &x, &y)) {
        PyErr_SetString(PyExc_TypeError, "invalid position argument");
        return -1;
    }

    SDL_SetWindowPosition(self->_win, x, y);

    return 0;
}

static PyObject *
window_get_position(pgWindowObject *self, void *v)
{
    int x, y;
    SDL_GetWindowPosition(self->_win, &x, &y);

    return pg_tuple_couple_from_values_int(x, y);
}

static int
window_set_opacity(pgWindowObject *self, PyObject *arg, void *v)
{
    float opacity;
    opacity = (float)PyFloat_AsDouble(arg);
    if (PyErr_Occurred()) {
        return -1;
    }
#if SDL_VERSION_ATLEAST(3, 0, 0)
    if (!SDL_SetWindowOpacity(self->_win, opacity))
#else
    if (SDL_SetWindowOpacity(self->_win, opacity))
#endif
    {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return -1;
    }
    return 0;
}

static PyObject *
window_get_opacity(pgWindowObject *self, void *v)
{
#if SDL_VERSION_ATLEAST(3, 0, 0)
    float opacity = SDL_GetWindowOpacity(self->_win);
    if (opacity < 0) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }
#else
    float opacity;
    if (SDL_GetWindowOpacity(self->_win, &opacity)) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }
#endif
    return PyFloat_FromDouble((double)opacity);
}

static PyObject *
window_get_opengl(pgWindowObject *self, void *v)
{
    long hasGL;
    if (!self->_is_borrowed) {
        hasGL = self->context != NULL;
    }
    else {
        /* This is not a reliable way to test that OPENGL was requested by the
         * user. SDL can implicitly create and use an opengl context in some
         * platforms and in that case hasGL=1 even when the user didn't
         * request for it. As borrowed windows are deprecated functionality we
         * can ignore this issue. */
        hasGL = (SDL_GetWindowFlags(self->_win) & SDL_WINDOW_OPENGL) > 0;
    }
    return PyBool_FromLong(hasGL);
}

static PyObject *
window_get_utility(pgWindowObject *self, void *v)
{
    return PyBool_FromLong(SDL_GetWindowFlags(self->_win) &
                           SDL_WINDOW_UTILITY);
}

static void
window_dealloc(pgWindowObject *self, PyObject *_null)
{
    if (self->_win) {
        if (!self->_is_borrowed) {
            if (self->context != NULL) {
                SDL_GL_DeleteContext(self->context);
            }
            SDL_DestroyWindow(self->_win);
        }
        else if (pg_get_pg_window(self->_win) != NULL) {
            pg_set_pg_window(self->_win, NULL);
        }
    }
    if (self->surf) {
        // Set the internal surface to NULL to make pygame surface invalid
        // since this surface will be deallocated by SDL when the window is
        // destroyed.
        self->surf->surf = NULL;

        Py_DECREF(self->surf);
    }

    Py_TYPE(self)->tp_free(self);
}

/* Based on code from sdl2-compat */
static SDL_Window *
PG_CreateWindow(const char *title, int x, int y, int w, int h, Uint32 flags)
{
#if SDL_VERSION_ATLEAST(3, 0, 0)
    SDL_Window *window = NULL;

    SDL_PropertiesID props = SDL_CreateProperties();
    if (!props) {
        return NULL;
    }

    if (title && *title) {
        SDL_SetStringProperty(props, SDL_PROP_WINDOW_CREATE_TITLE_STRING,
                              title);
    }
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_X_NUMBER, x);
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_Y_NUMBER, y);
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_WIDTH_NUMBER, w);
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_HEIGHT_NUMBER, h);
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_FLAGS_NUMBER, flags);
    SDL_SetBooleanProperty(
        props, SDL_PROP_WINDOW_CREATE_EXTERNAL_GRAPHICS_CONTEXT_BOOLEAN,
        SDL_GetHintBoolean("SDL_VIDEO_EXTERNAL_CONTEXT", false));

    window = SDL_CreateWindowWithProperties(props);
    SDL_DestroyProperties(props);
    return window;
#else
    return SDL_CreateWindow(title, x, y, w, h, flags);
#endif
}

static int
window_init(pgWindowObject *self, PyObject *args, PyObject *kwargs)
{
    char *title = "pygame window";
    PyObject *size = NULL;
    int size_w = 640, size_h = 480;
    PyObject *position = NULL;
    int pos_x = SDL_WINDOWPOS_UNDEFINED;
    int pos_y = SDL_WINDOWPOS_UNDEFINED;
    Uint32 flags = 0;
    SDL_Window *_win = NULL;

    Py_ssize_t dict_pos = 0;
    PyObject *_key, *_value, *_kw;
    const char *_key_str;
    int _value_bool;

    // ensure display is init at this point, display init automatically calls
    // the window init in this module
    if (!pg_mod_autoinit(IMPPREFIX "display")) {
        return -1;
    }

    _kw = PyDict_New();
    if (!_kw) {
        return -1;
    }

#if SDL_VERSION_ATLEAST(3, 0, 0)
    int fullscreen_non_desktop = 0;
#endif
    if (kwargs) {
        while (PyDict_Next(kwargs, &dict_pos, &_key, &_value)) {
            if (!PyUnicode_Check(_key)) {
                PyErr_SetString(PyExc_TypeError, "keywords must be strings");
                return -1;
            }

            _key_str = PyUnicode_AsUTF8(_key);
            if (!_key_str) {
                return -1;
            }

            if (!strcmp(_key_str, "title") || !strcmp(_key_str, "size") ||
                !strcmp(_key_str, "position")) {
                PyDict_SetItem(_kw, _key, _value);
            }

            // handle **flags
            else {
                _value_bool = PyObject_IsTrue(_value);
                if (_value_bool == -1) {
                    return -1;
                }

                if (!strcmp(_key_str, "opengl")) {
                    if (_value_bool) {
                        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
                        flags |= SDL_WINDOW_OPENGL;
                    }
                    else {
                        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 0);
                    }
                }
                else if (!strcmp(_key_str, "fullscreen")) {
#if SDL_VERSION_ATLEAST(3, 0, 0)
                    fullscreen_non_desktop = 1;
#endif
                    if (_value_bool) {
                        flags |= SDL_WINDOW_FULLSCREEN;
                    }
                }
                else if (!strcmp(_key_str, "fullscreen_desktop")) {
#if SDL_VERSION_ATLEAST(3, 0, 0)
                    if (_value_bool) {
                        flags |= SDL_WINDOW_FULLSCREEN;
                    }
#else
                    if (_value_bool) {
                        flags |= SDL_WINDOW_FULLSCREEN_DESKTOP;
                    }
#endif
                }
                else if (!strcmp(_key_str, "hidden")) {
                    if (_value_bool) {
                        flags |= SDL_WINDOW_HIDDEN;
                    }
                }
                else if (!strcmp(_key_str, "borderless")) {
                    if (_value_bool) {
                        flags |= SDL_WINDOW_BORDERLESS;
                    }
                }
                else if (!strcmp(_key_str, "resizable")) {
                    if (_value_bool) {
                        flags |= SDL_WINDOW_RESIZABLE;
                    }
                }
                else if (!strcmp(_key_str, "minimized")) {
                    if (_value_bool) {
                        flags |= SDL_WINDOW_MINIMIZED;
                    }
                }
                else if (!strcmp(_key_str, "maximized")) {
                    if (_value_bool) {
                        flags |= SDL_WINDOW_MAXIMIZED;
                    }
                }
                else if (!strcmp(_key_str, "mouse_grabbed")) {
                    if (_value_bool)
#if SDL_VERSION_ATLEAST(2, 0, 16)
                        flags |= SDL_WINDOW_MOUSE_GRABBED;
#else
                        flags |= SDL_WINDOW_INPUT_GRABBED;
#endif
                }
                else if (!strcmp(_key_str, "keyboard_grabbed")) {
                    if (_value_bool) {
#if SDL_VERSION_ATLEAST(2, 0, 16)
                        flags |= SDL_WINDOW_KEYBOARD_GRABBED;
#else
                        if (PyErr_WarnEx(PyExc_Warning,
                                         "Keyword 'keyboard_grabbed' requires "
                                         "SDL 2.0.16+",
                                         1) == -1) {
                            return -1;
                        }
#endif
                    }
                }
                else if (!strcmp(_key_str, "input_focus")) {
                    if (_value_bool) {
                        flags |= SDL_WINDOW_INPUT_FOCUS;
                    }
                }
                else if (!strcmp(_key_str, "mouse_focus")) {
                    if (_value_bool) {
                        flags |= SDL_WINDOW_MOUSE_FOCUS;
                    }
                }
                else if (!strcmp(_key_str, "foreign")) {
                    if (PyErr_WarnEx(PyExc_DeprecationWarning,
                                     "The foreign kwarg has been deprecated "
                                     "and may be removed in a future version",
                                     1) == -1) {
                        return -1;
                    }
                    if (_value_bool) {
#if SDL_VERSION_ATLEAST(3, 0, 0)
                        flags |= SDL_WINDOW_EXTERNAL;
#else
                        flags |= SDL_WINDOW_FOREIGN;
#endif
                    }
                }
                else if (!strcmp(_key_str, "allow_high_dpi")) {
                    if (_value_bool) {
                        flags |= SDL_WINDOW_ALLOW_HIGHDPI;
                    }
                }
                else if (!strcmp(_key_str, "mouse_capture")) {
                    if (_value_bool) {
                        flags |= SDL_WINDOW_MOUSE_CAPTURE;
                    }
                }
                else if (!strcmp(_key_str, "always_on_top")) {
                    if (_value_bool) {
                        flags |= SDL_WINDOW_ALWAYS_ON_TOP;
                    }
                }
                else if (!strcmp(_key_str, "vulkan")) {
                    if (_value_bool) {
                        flags |= SDL_WINDOW_VULKAN;
                    }
                }
                else if (!strcmp(_key_str, "utility")) {
                    if (_value_bool) {
                        flags |= SDL_WINDOW_UTILITY;
#if !SDL_VERSION_ATLEAST(3, 0, 0)
                        flags |= SDL_WINDOW_SKIP_TASKBAR;
#endif
                    }
                }
                else {
                    PyErr_Format(PyExc_TypeError,
                                 "__init__ got an unexpected flag \'%s\'",
                                 _key_str);
                    return -1;
                }
            }
        }
    }

    char *kwids[] = {"title", "size", "position", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, _kw, "|sOO", kwids, &title, &size,
                                     &position)) {
        return -1;
    }

    if (size) {
        if (!pg_TwoIntsFromObj(size, &size_w, &size_h)) {
            PyErr_SetString(PyExc_TypeError, "invalid size argument");
            return -1;
        }
    }

    if (size_w <= 0 || size_h <= 0) {
        PyErr_SetString(
            PyExc_ValueError,
            "width or height should not be less than or equal to zero.");
        return -1;
    }

    if (position) {
        if (Py_TYPE(position) == &PyLong_Type) {
            pos_x = pos_y = PyLong_AsLong(position);
            if (pos_x != SDL_WINDOWPOS_CENTERED &&
                pos_x != SDL_WINDOWPOS_UNDEFINED) {
                PyErr_SetString(PyExc_TypeError, "invalid position argument");
                return -1;
            }
        }
        else if (!pg_TwoIntsFromObj(position, &pos_x, &pos_y)) {
            PyErr_SetString(PyExc_TypeError, "invalid position argument");
            return -1;
        }
    }

    _win = PG_CreateWindow(title, pos_x, pos_y, size_w, size_h, flags);
    if (!_win) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return -1;
    }
#if SDL_VERSION_ATLEAST(3, 0, 0)
    if (fullscreen_non_desktop) {
        if (!pg_window_set_fullscreen(_win, 0)) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            return -1;
        }
    }
#endif
    self->_win = _win;
    self->_is_borrowed = SDL_FALSE;
    self->surf = NULL;

    if (flags & SDL_WINDOW_OPENGL) {
        SDL_GLContext context = SDL_GL_CreateContext(self->_win);
        if (context == NULL) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            return -1;
        }
        /* As stated in the 'Remarks' of the docs
         * (https://wiki.libsdl.org/SDL2/SDL_GL_GetProcAddress) on Windows
         * SDL_GL_GetProcAddress is only valid after an OpenGL context has been
         * created */
        p_glViewport = (GL_glViewport_Func)SDL_GL_GetProcAddress("glViewport");
        self->context = context;
    }
    else {
        self->context = NULL;
    }

    pg_set_pg_window(_win, (PyObject *)self);

    PyObject *icon = pg_display_resource(icon_defaultname);
    if (!icon) {
        return -1;
    }
    if (icon_colorkey != -1) {
        if (!PG_SetSurfaceColorKey(pgSurface_AsSurface(icon), SDL_TRUE,
                                   icon_colorkey)) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            return -1;
        }
    }
    SDL_SetWindowIcon(self->_win, pgSurface_AsSurface(icon));

    return 0;
}

static PyObject *
window_from_display_module(PyTypeObject *cls, PyObject *_null)
{
    if (PyErr_WarnEx(PyExc_DeprecationWarning,
                     "Please use Window.get_surface and Window.flip to use "
                     "surface-rendering with Window. This method will be "
                     "removed in a future version.",
                     1) == -1) {
        return NULL;
    }

    // ensure display is init at this point, display init automatically calls
    // the window init in this module
    if (!pg_mod_autoinit(IMPPREFIX "display")) {
        return NULL;
    }

    SDL_Window *window = pg_GetDefaultWindow();
    if (!window) {
        return RAISE(pgExc_SDLError,
                     "display.set_mode has not been called yet.");
    }

    pgWindowObject *self = (pgWindowObject *)pg_get_pg_window(window);
    if (self != NULL) {
        Py_INCREF(self);
        return (PyObject *)self;
    }

    self = (pgWindowObject *)(cls->tp_new(cls, NULL, NULL));
    self->_win = window;
    self->_is_borrowed = SDL_TRUE;
    pg_set_pg_window(window, (PyObject *)self);
    return (PyObject *)self;
}

static PyObject *
window_flash(pgWindowObject *self, PyObject *arg)
{
#if SDL_VERSION_ATLEAST(2, 0, 16)
    long operation = PyLong_AsLong(arg);
    if (operation == -1 && PyErr_Occurred()) {
        return RAISE(PyExc_TypeError,
                     "'operation' must be an integer. "
                     "Must correspond with FLASH_CANCEL, FLASH_BRIEFLY, or "
                     "FLASH_UNTIL_FOCUSED.");
    }

    if (operation != SDL_FLASH_CANCEL && operation != SDL_FLASH_BRIEFLY &&
        operation != SDL_FLASH_UNTIL_FOCUSED) {
        return RAISE(PyExc_ValueError, "Unsupported window flash operation.");
    }

#if SDL_VERSION_ATLEAST(3, 0, 0)
    if (!SDL_FlashWindow(self->_win, operation)) {
#else
    if (SDL_FlashWindow(self->_win, operation) < 0) {
#endif
        return RAISE(pgExc_SDLError, SDL_GetError());
    }
    Py_RETURN_NONE;
#else
    return RAISE(pgExc_SDLError, "'Window.flash' requires SDL 2.0.16+");
#endif /* SDL_VERSION_ATLEAST(2, 0, 16) */
}

PyObject *
window_repr(pgWindowObject *self)
{
    const char *title;
    int win_id;
    if (!self->_win) {
        return PyUnicode_FromString("<Window(Destroyed)>");
    }

    if (self->_is_borrowed) {
        return PyUnicode_FromString("<Window(From Display)>");
    }

    title = SDL_GetWindowTitle(self->_win);
    win_id = SDL_GetWindowID(self->_win);
    if (win_id == 0) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    return PyUnicode_FromFormat("<Window(title='%s', id=%d)>", title, win_id);
}

static PyObject *
_window_internal_mod_init(PyObject *self, PyObject *_null)
{
    if (!is_window_mod_init) {
        SDL_AddEventWatch(_resize_event_watch, NULL);
        is_window_mod_init = 1;
    }
    Py_RETURN_NONE;
}

static PyObject *
_window_internal_mod_quit(PyObject *self, PyObject *_null)
{
    if (is_window_mod_init) {
        SDL_DelEventWatch(_resize_event_watch, NULL);
        is_window_mod_init = 0;
    }
    Py_RETURN_NONE;
}

static PyMethodDef window_methods[] = {
    {"destroy", (PyCFunction)window_destroy, METH_NOARGS, DOC_WINDOW_DESTROY},
    {"set_windowed", (PyCFunction)window_set_windowed, METH_NOARGS,
     DOC_WINDOW_SETWINDOWED},
    {"set_fullscreen", (PyCFunction)window_set_fullscreen,
     METH_VARARGS | METH_KEYWORDS, DOC_WINDOW_SETFULLSCREEN},
    {"focus", (PyCFunction)window_focus, METH_VARARGS | METH_KEYWORDS,
     DOC_WINDOW_FOCUS},
    {"hide", (PyCFunction)window_hide, METH_NOARGS, DOC_WINDOW_HIDE},
    {"show", (PyCFunction)window_show, METH_NOARGS, DOC_WINDOW_SHOW},
    {"restore", (PyCFunction)window_restore, METH_NOARGS, DOC_WINDOW_RESTORE},
    {"maximize", (PyCFunction)window_maximize, METH_NOARGS,
     DOC_WINDOW_MAXIMIZE},
    {"minimize", (PyCFunction)window_minimize, METH_NOARGS,
     DOC_WINDOW_MINIMIZE},
    {"set_modal_for", (PyCFunction)window_set_modal_for, METH_O,
     DOC_WINDOW_SETMODALFOR},
    {"set_icon", (PyCFunction)window_set_icon, METH_O, DOC_WINDOW_SETICON},
    {"flip", (PyCFunction)window_flip, METH_NOARGS, DOC_WINDOW_FLIP},
    {"get_surface", (PyCFunction)window_get_surface, METH_NOARGS,
     DOC_WINDOW_GETSURFACE},
    {"from_display_module", (PyCFunction)window_from_display_module,
     METH_CLASS | METH_NOARGS, DOC_WINDOW_FROMDISPLAYMODULE},
    {"flash", (PyCFunction)window_flash, METH_O, DOC_WINDOW_FLASH},
    {NULL, NULL, 0, NULL}};

static PyGetSetDef _window_getset[] = {
    {"grab_mouse", (getter)window_get_grab_mouse,
     (setter)window_set_grab_mouse, DOC_WINDOW_GRABMOUSE, NULL},
    {"grab_keyboard", (getter)window_get_grab_keyboard,
     (setter)window_set_grab_keyboard, DOC_WINDOW_GRABKEYBOARD, NULL},
    {"mouse_grabbed", (getter)window_get_mouse_grabbed, NULL,
     DOC_WINDOW_MOUSEGRABBED, NULL},
    {"keyboard_grabbed", (getter)window_get_keyboard_grabbed, NULL,
     DOC_WINDOW_KEYBOARDGRABBED, NULL},
    {"focused", (getter)window_get_focused, NULL, DOC_WINDOW_FOCUSED, NULL},
    {"title", (getter)window_get_title, (setter)window_set_title,
     DOC_WINDOW_TITLE, NULL},
    {"resizable", (getter)window_get_resizable, (setter)window_set_resizable,
     DOC_WINDOW_RESIZABLE, NULL},
    {"borderless", (getter)window_get_borderless,
     (setter)window_set_borderless, DOC_WINDOW_BORDERLESS, NULL},
    {"always_on_top", (getter)window_get_always_on_top,
     (setter)window_set_always_on_top, DOC_WINDOW_ALWAYSONTOP, NULL},
    {"mouse_rect", (getter)window_get_mouse_rect,
     (setter)window_set_mouse_rect, DOC_WINDOW_MOUSERECT, NULL},
    {"size", (getter)window_get_size, (setter)window_set_size, DOC_WINDOW_SIZE,
     NULL},
    {"minimum_size", (getter)window_get_minimum_size,
     (setter)window_set_minimum_size, DOC_WINDOW_MINIMUMSIZE, NULL},
    {"maximum_size", (getter)window_get_maximum_size,
     (setter)window_set_maximum_size, DOC_WINDOW_MAXIMUMSIZE, NULL},
    {"position", (getter)window_get_position, (setter)window_set_position,
     DOC_WINDOW_POSITION, NULL},
    {"opacity", (getter)window_get_opacity, (setter)window_set_opacity,
     DOC_WINDOW_OPACITY, NULL},
    {"id", (getter)window_get_window_id, NULL, DOC_WINDOW_ID, NULL},
    {"opengl", (getter)window_get_opengl, NULL, DOC_WINDOW_OPENGL, NULL},
    {"utility", (getter)window_get_utility, NULL, DOC_WINDOW_UTILITY, NULL},
    {NULL, 0, NULL, NULL, NULL} /* Sentinel */
};

static PyTypeObject pgWindow_Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pygame.window.Window",
    .tp_basicsize = sizeof(pgWindowObject),
    .tp_dealloc = (destructor)window_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = DOC_WINDOW,
    .tp_methods = window_methods,
    .tp_init = (initproc)window_init,
    .tp_new = PyType_GenericNew,
    .tp_getset = _window_getset,
    .tp_repr = (reprfunc)window_repr};

static PyMethodDef _window_methods[] = {
    {"get_grabbed_window", (PyCFunction)get_grabbed_window, METH_NOARGS,
     DOC_SDL2_VIDEO_GETGRABBEDWINDOW},
    {"_internal_mod_init", (PyCFunction)_window_internal_mod_init, METH_NOARGS,
     "auto initialize for window module"},
    {"_internal_mod_quit", (PyCFunction)_window_internal_mod_quit, METH_NOARGS,
     "auto quit for window module"},
    {NULL, NULL, 0, NULL}};

MODINIT_DEFINE(window)
{
    PyObject *module, *apiobj;
    static void *c_api[PYGAMEAPI_WINDOW_NUMSLOTS];

    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "window",
                                         "docs_needed",
                                         -1,
                                         _window_methods,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL};

    /* imported needed apis; Do this first so if there is an error
       the module is not loaded.
    */
    import_pygame_base();
    if (PyErr_Occurred()) {
        return NULL;
    }

    import_pygame_surface();
    if (PyErr_Occurred()) {
        return NULL;
    }

    import_pygame_rect();
    if (PyErr_Occurred()) {
        return NULL;
    }

    /* create the module */
    module = PyModule_Create(&_module);
    if (module == 0) {
        return NULL;
    }

    if (PyModule_AddType(module, &pgWindow_Type)) {
        Py_DECREF(module);
        return NULL;
    }

    c_api[0] = &pgWindow_Type;
    apiobj = encapsulate_api(c_api, "window");
    if (PyModule_AddObject(module, PYGAMEAPI_LOCAL_ENTRY, apiobj)) {
        Py_XDECREF(apiobj);
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
