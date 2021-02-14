/*
  pygame - Python Game Library
  Copyright (C) 2000-2001  Pete Shinners

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Library General Public
  License as published by the Free Software Foundation; either
  version 2 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Library General Public License for more details.

  You should have received a copy of the GNU Library General Public
  License along with this library; if not, write to the Free
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

  Pete Shinners
  pete@shinners.org
*/

/*
 *  pygame display module
 */
#define PYGAMEAPI_DISPLAY_INTERNAL
#include "pygame.h"

#include "pgcompat.h"
#include "pgopengl.h"

#include "doc/display_doc.h"

#include <SDL_syswm.h>

static PyTypeObject pgVidInfo_Type;

#if IS_SDLv1

static PyObject *
pgVidInfo_New(const SDL_VideoInfo *info);

static void
pg_do_set_icon(PyObject *surface);
static pgSurfaceObject *pgDisplaySurfaceObject = NULL;
static int icon_was_set = 0;
static int _allow_screensaver = 0;

#else /* IS_SDLv2 */

static PyObject *
pgVidInfo_New(const pg_VideoInfo *info);

static SDL_Renderer *pg_renderer = NULL;
static SDL_Texture *pg_texture = NULL;

typedef struct _display_state_s {
    char *title;
    PyObject *icon;
    Uint16 *gamma_ramp;
    SDL_GLContext gl_context;
    int toggle_windowed_w;
    int toggle_windowed_h;
    Uint8 using_gl; /* using an OPENGL display without renderer */
    Uint8 scaled_gl;
    int scaled_gl_w;
    int scaled_gl_h;
    int fullscreen_backup_x;
    int fullscreen_backup_y;
    SDL_bool auto_resize;
} _DisplayState;

static int
pg_flip_internal(_DisplayState *state);

#if PY3
#ifndef PYPY_VERSION
static struct PyModuleDef _module;
#define DISPLAY_MOD_STATE(mod) ((_DisplayState *)PyModule_GetState(mod))
#define DISPLAY_STATE DISPLAY_MOD_STATE(PyState_FindModule(&_module))
#else /* PYPY_VERSION */
static struct PyModuleDef _module;
static _DisplayState _modstate = {0};
#define DISPLAY_MOD_STATE(mod) (&_modstate)
#define DISPLAY_STATE DISPLAY_MOD_STATE(0)
#endif /* PYPY_VERSION */
#else  /* PY2 */
static _DisplayState _modstate = {0};
#define DISPLAY_MOD_STATE(mod) (&_modstate)
#define DISPLAY_STATE DISPLAY_MOD_STATE(0)
#endif /* PY2 */

static void
_display_state_cleanup(_DisplayState *state)
{
    if (state->title) {
        free(state->title);
        state->title = NULL;
    }
    if (state->icon) {
        Py_XDECREF(state->icon);
        state->icon = NULL;
    }
    if (state->gl_context) {
        SDL_GL_DeleteContext(state->gl_context);
        state->gl_context = NULL;
    }
    if (state->gamma_ramp) {
        free(state->gamma_ramp);
        state->gamma_ramp = NULL;
    }
}

#endif /* IS_SDLv2 */

static char *icon_defaultname = "pygame_icon.bmp";
static char *pkgdatamodule_name = "pygame.pkgdata";
static char *imagemodule_name = "pygame.image";
static char *resourcefunc_name = "getResource";
static char *load_basicfunc_name = "load_basic";

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

static PyObject *
pg_display_resource(char *filename)
{
    PyObject *imagemodule = NULL;
    PyObject *load_basicfunc = NULL;
    PyObject *pkgdatamodule = NULL;
    PyObject *resourcefunc = NULL;
    PyObject *fresult = NULL;
    PyObject *result = NULL;
#if PY3
    PyObject *name = NULL;
#endif

    pkgdatamodule = PyImport_ImportModule(pkgdatamodule_name);
    if (!pkgdatamodule)
        goto display_resource_end;

    resourcefunc = PyObject_GetAttrString(pkgdatamodule, resourcefunc_name);
    if (!resourcefunc)
        goto display_resource_end;

    imagemodule = PyImport_ImportModule(imagemodule_name);
    if (!imagemodule)
        goto display_resource_end;

    load_basicfunc = PyObject_GetAttrString(imagemodule, load_basicfunc_name);
    if (!load_basicfunc)
        goto display_resource_end;

    fresult = PyObject_CallFunction(resourcefunc, "s", filename);
    if (!fresult)
        goto display_resource_end;

#if PY3
    name = PyObject_GetAttrString(fresult, "name");
    if (name != NULL) {
        if (Text_Check(name)) {
            pg_close_file(fresult);
            Py_DECREF(fresult);
            fresult = name;
            name = NULL;
        }
    }
    else {
        PyErr_Clear();
    }
#else
    if (PyFile_Check(fresult)) {
        PyObject *tmp = PyFile_Name(fresult);
        Py_INCREF(tmp);
        pg_close_file(fresult);
        Py_DECREF(fresult);
        fresult = tmp;
    }
#endif

    result = PyObject_CallFunction(load_basicfunc, "O", fresult);
    if (!result)
        goto display_resource_end;

display_resource_end:
    Py_XDECREF(pkgdatamodule);
    Py_XDECREF(resourcefunc);
    Py_XDECREF(imagemodule);
    Py_XDECREF(load_basicfunc);
    Py_XDECREF(fresult);
#if PY3
    Py_XDECREF(name);
#endif
    return result;
}

/* init routines */
#if IS_SDLv2
static void
pg_display_autoquit(void)
{
    _DisplayState *state = DISPLAY_STATE;
    _display_state_cleanup(state);
    if (pg_GetDefaultWindowSurface()) {
        pgSurface_AsSurface(pg_GetDefaultWindowSurface()) = NULL;
        pg_SetDefaultWindowSurface(NULL);
        pg_SetDefaultWindow(NULL);
    }
}
#else  /* IS_SDLv1 */
static void
pg_display_autoquit(void)
{
    if (pgDisplaySurfaceObject) {
        pgSurface_AsSurface(pgDisplaySurfaceObject) = NULL;
        Py_DECREF((PyObject *)pgDisplaySurfaceObject);
        pgDisplaySurfaceObject = NULL;
        icon_was_set = 0;
    }
}
#endif /* IS_SDLv1 */

static PyObject *
pg_display_autoinit(PyObject *self, PyObject *arg)
{
    pg_RegisterQuit(pg_display_autoquit);
    return PyInt_FromLong(1);
}

static PyObject *
pg_quit(PyObject *self, PyObject *arg)
{
    pgVideo_AutoQuit();
    pg_display_autoquit();
    Py_RETURN_NONE;
}

static PyObject *
pg_init(PyObject *self, PyObject *args)
{

#if IS_SDLv2
    /* Compatibility:
        windib video driver was renamed in SDL2, and we don't want it to fail.
    */
    const char *drivername = SDL_getenv("SDL_VIDEODRIVER");
    if (drivername && SDL_strncasecmp("windib", drivername, SDL_strlen(drivername)) == 0) {
        SDL_setenv("SDL_VIDEODRIVER", "windows", 1);
    }
#endif

    if (!pgVideo_AutoInit())
        return RAISE(pgExc_SDLError, SDL_GetError());
    if (!pg_display_autoinit(NULL, NULL))
        return NULL;
    Py_RETURN_NONE;
}

static PyObject *
pg_get_init(PyObject *self, PyObject *args)
{
    return PyBool_FromLong(SDL_WasInit(SDL_INIT_VIDEO) != 0);
}

#if IS_SDLv2
static PyObject *
pg_get_active(PyObject *self, PyObject *args)
{
    Uint32 flags = SDL_GetWindowFlags(pg_GetDefaultWindow());
    return PyBool_FromLong((flags & SDL_WINDOW_SHOWN) &&
                           !(flags & SDL_WINDOW_MINIMIZED));
}
#else  /* IS_SDLv1 */
static PyObject *
pg_get_active(PyObject *self, PyObject *args)
{
  if (!pgDisplaySurfaceObject)
       return PyBool_FromLong(0);
  return PyBool_FromLong((SDL_GetAppState() & SDL_APPACTIVE) != 0);
}
#endif /* IS_SDLv1 */

/* vidinfo object */
static void
pg_vidinfo_dealloc(PyObject *self)
{
    PyObject_DEL(self);
}

static PyObject *
pg_vidinfo_getattr(PyObject *self, char *name)
{
#if IS_SDLv1
    SDL_VideoInfo *info = &((pgVidInfoObject *)self)->info;
#else
    pg_VideoInfo *info = &((pgVidInfoObject *)self)->info;
#endif

    int current_w = -1;
    int current_h = -1;

    SDL_version versioninfo;
    SDL_VERSION(&versioninfo);

    if (versioninfo.major > 1 ||
        (versioninfo.minor >= 2 && versioninfo.patch >= 10)) {
        current_w = info->current_w;
        current_h = info->current_h;
    }

    if (!strcmp(name, "hw"))
        return PyInt_FromLong(info->hw_available);
    else if (!strcmp(name, "wm"))
        return PyInt_FromLong(info->wm_available);
    else if (!strcmp(name, "blit_hw"))
        return PyInt_FromLong(info->blit_hw);
    else if (!strcmp(name, "blit_hw_CC"))
        return PyInt_FromLong(info->blit_hw_CC);
    else if (!strcmp(name, "blit_hw_A"))
        return PyInt_FromLong(info->blit_hw_A);
    else if (!strcmp(name, "blit_sw"))
        return PyInt_FromLong(info->blit_hw);
    else if (!strcmp(name, "blit_sw_CC"))
        return PyInt_FromLong(info->blit_hw_CC);
    else if (!strcmp(name, "blit_sw_A"))
        return PyInt_FromLong(info->blit_hw_A);
    else if (!strcmp(name, "blit_fill"))
        return PyInt_FromLong(info->blit_fill);
    else if (!strcmp(name, "video_mem"))
        return PyInt_FromLong(info->video_mem);
    else if (!strcmp(name, "bitsize"))
        return PyInt_FromLong(info->vfmt->BitsPerPixel);
    else if (!strcmp(name, "bytesize"))
        return PyInt_FromLong(info->vfmt->BytesPerPixel);
    else if (!strcmp(name, "masks"))
        return Py_BuildValue("(iiii)", info->vfmt->Rmask, info->vfmt->Gmask,
                             info->vfmt->Bmask, info->vfmt->Amask);
    else if (!strcmp(name, "shifts"))
        return Py_BuildValue("(iiii)", info->vfmt->Rshift, info->vfmt->Gshift,
                             info->vfmt->Bshift, info->vfmt->Ashift);
    else if (!strcmp(name, "losses"))
        return Py_BuildValue("(iiii)", info->vfmt->Rloss, info->vfmt->Gloss,
                             info->vfmt->Bloss, info->vfmt->Aloss);
    else if (!strcmp(name, "current_h"))
        return PyInt_FromLong(current_h);
    else if (!strcmp(name, "current_w"))
        return PyInt_FromLong(current_w);

    return RAISE(PyExc_AttributeError, "does not exist in vidinfo");
}

PyObject *
pg_vidinfo_str(PyObject *self)
{
    char str[1024];
    int current_w = -1;
    int current_h = -1;
#if IS_SDLv1
    SDL_VideoInfo *info = &((pgVidInfoObject *)self)->info;
#else
    pg_VideoInfo *info = &((pgVidInfoObject *)self)->info;
#endif

    SDL_version versioninfo;
    SDL_VERSION(&versioninfo);

    if (versioninfo.major > 1 ||
        (versioninfo.minor >= 2 && versioninfo.patch >= 10)) {
        current_w = info->current_w;
        current_h = info->current_h;
    }

    sprintf(str,
            "<VideoInfo(hw = %d, wm = %d,video_mem = %d\n"
            "         blit_hw = %d, blit_hw_CC = %d, blit_hw_A = %d,\n"
            "         blit_sw = %d, blit_sw_CC = %d, blit_sw_A = %d,\n"
            "         bitsize  = %d, bytesize = %d,\n"
            "         masks =  (%d, %d, %d, %d),\n"
            "         shifts = (%d, %d, %d, %d),\n"
            "         losses =  (%d, %d, %d, %d),\n"
            "         current_w = %d, current_h = %d\n"
            ">\n",
            info->hw_available, info->wm_available, info->video_mem,
            info->blit_hw, info->blit_hw_CC, info->blit_hw_A, info->blit_sw,
            info->blit_sw_CC, info->blit_sw_A, info->vfmt->BitsPerPixel,
            info->vfmt->BytesPerPixel, info->vfmt->Rmask, info->vfmt->Gmask,
            info->vfmt->Bmask, info->vfmt->Amask, info->vfmt->Rshift,
            info->vfmt->Gshift, info->vfmt->Bshift, info->vfmt->Ashift,
            info->vfmt->Rloss, info->vfmt->Gloss, info->vfmt->Bloss,
            info->vfmt->Aloss, current_w, current_h);
    return Text_FromUTF8(str);
}

static PyTypeObject pgVidInfo_Type = {
    PyVarObject_HEAD_INIT(NULL,0)
    "VidInfo",                    /*name*/
    sizeof(pgVidInfoObject),      /*basic size*/
    0,                            /*itemsize*/
    pg_vidinfo_dealloc,           /*dealloc*/
    0,                            /*print*/
    pg_vidinfo_getattr,           /*getattr*/
    NULL,                         /*setattr*/
    NULL,                         /*compare*/
    pg_vidinfo_str,               /*repr*/
    NULL,                         /*as_number*/
    NULL,                         /*as_sequence*/
    NULL,                         /*as_mapping*/
    (hashfunc)NULL,               /*hash*/
    (ternaryfunc)NULL,            /*call*/
    (reprfunc)NULL,               /*str*/
};

#if IS_SDLv1
static PyObject *
pgVidInfo_New(const SDL_VideoInfo *i)
#else
static PyObject *
pgVidInfo_New(const pg_VideoInfo *i)
#endif
{
    pgVidInfoObject *info;
    if (!i)
        return RAISE(pgExc_SDLError, SDL_GetError());
    info = PyObject_NEW(pgVidInfoObject, &pgVidInfo_Type);
    if (!info)
        return NULL;
#if IS_SDLv1
    memcpy(&info->info, i, sizeof(SDL_VideoInfo));
#else
    info->info = *i;
    info->info.vfmt = &info->info.vfmt_data;
#endif
    return (PyObject *)info;
}

#if IS_SDLv2
static pg_VideoInfo *
pg_GetVideoInfo(pg_VideoInfo *info)
{
    SDL_DisplayMode mode;
    SDL_PixelFormat *tempformat;
    Uint32 formatenum;
    pgSurfaceObject *winsurfobj;
    SDL_Surface *winsurf;

#pragma PG_WARN(hardcoding wm_available to 1)
#pragma PG_WARN(setting available video RAM to 0 KB)

    memset(info, 0, sizeof(pg_VideoInfo));
    info->wm_available = 1;

    winsurfobj = pg_GetDefaultWindowSurface();
    if (winsurfobj) {
        winsurf = pgSurface_AsSurface(winsurfobj);
        info->current_w = winsurf->w;
        info->current_h = winsurf->h;
        info->vfmt_data = *(winsurf->format);
        info->vfmt = &info->vfmt_data;
    }
    else {
        if (SDL_GetCurrentDisplayMode(0, &mode) == 0) {
            info->current_w = mode.w;
            info->current_h = mode.h;
            formatenum = mode.format;
        }
        else {
            info->current_w = -1;
            info->current_h = -1;
            formatenum = SDL_PIXELFORMAT_UNKNOWN;
        }

        if ((tempformat = SDL_AllocFormat(formatenum))) {
            info->vfmt_data = *tempformat;
            info->vfmt = &info->vfmt_data;
            SDL_FreeFormat(tempformat);
        }
        else {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            return (pg_VideoInfo *)NULL;
        }
    }

    return info;
}
#endif /* IS_SDLv2 */

static PyObject *
pgInfo(PyObject *self, PyObject *args)
{
#if IS_SDLv1
    const SDL_VideoInfo *info;
    VIDEO_INIT_CHECK();
    info = SDL_GetVideoInfo();
    return pgVidInfo_New(info);
#else  /* IS_SDLv2 */
    pg_VideoInfo info;
    VIDEO_INIT_CHECK();
    return pgVidInfo_New(pg_GetVideoInfo(&info));
#endif /* IS_SDLv2 */
}

static PyObject *
pg_get_wm_info(PyObject *self, PyObject *args)
{
    PyObject *dict;
    PyObject *tmp;
    SDL_SysWMinfo info;
#if IS_SDLv2
    SDL_Window *win;
#endif

    VIDEO_INIT_CHECK();

    SDL_VERSION(&(info.version))
    dict = PyDict_New();
    if (!dict)
        return NULL;

#if IS_SDLv1
    if (!SDL_GetWMInfo(&info))
        return dict;

/*scary #ifdef's match SDL_syswm.h*/
#if (defined(unix) || defined(__unix__) || defined(_AIX) ||     \
     defined(__OpenBSD__)) &&                                   \
    (defined(SDL_VIDEO_DRIVER_X11) && !defined(__CYGWIN32__) && \
     !defined(ENABLE_NANOX) && !defined(__QNXNTO__))

    tmp = PyInt_FromLong(info.info.x11.window);
    PyDict_SetItemString(dict, "window", tmp);
    Py_DECREF(tmp);

    tmp = PyCapsule_New(info.info.x11.display, "display", NULL);
    PyDict_SetItemString(dict, "display", tmp);
    Py_DECREF(tmp);

    tmp = PyCapsule_New(info.info.x11.lock_func, "lock_func", NULL);
    PyDict_SetItemString(dict, "lock_func", tmp);
    Py_DECREF(tmp);

    tmp = PyCapsule_New(info.info.x11.unlock_func, "unlock_func", NULL);
    PyDict_SetItemString(dict, "unlock_func", tmp);
    Py_DECREF(tmp);

    tmp = PyInt_FromLong(info.info.x11.fswindow);
    PyDict_SetItemString(dict, "fswindow", tmp);
    Py_DECREF(tmp);

    tmp = PyInt_FromLong(info.info.x11.wmwindow);
    PyDict_SetItemString(dict, "wmwindow", tmp);
    Py_DECREF(tmp);

#elif defined(ENABLE_NANOX)
    tmp = PyInt_FromLong(info.window);
    PyDict_SetItemString(dict, "window", tmp);
    Py_DECREF(tmp);
#elif defined(WIN32)
    tmp = PyInt_FromLong((long)info.window);
    PyDict_SetItemString(dict, "window", tmp);
    Py_DECREF(tmp);

    tmp = PyInt_FromLong((long)info.hglrc);
    PyDict_SetItemString(dict, "hglrc", tmp);
    Py_DECREF(tmp);
#elif defined(__riscos__)
    tmp = PyInt_FromLong(info.window);
    PyDict_SetItemString(dict, "window", tmp);
    Py_DECREF(tmp);

    tmp = PyInt_FromLong(info.wimpVersion);
    PyDict_SetItemString(dict, "wimpVersion", tmp);
    Py_DECREF(tmp);

    tmp = PyInt_FromLong(info.taskHandle);
    PyDict_SetItemString(dict, "taskHandle", tmp);
    Py_DECREF(tmp);
#elif (defined(__APPLE__) && defined(__MACH__))
        /* do nothing. */
#else
    tmp = PyInt_FromLong(info.data);
    PyDict_SetItemString(dict, "data", tmp);
    Py_DECREF(tmp);
#endif

#else /* IS_SDLv2 */

    win = pg_GetDefaultWindow();
    if (!win)
        return dict;
    if (!SDL_GetWindowWMInfo(win, &info))
        return dict;

#if defined(SDL_VIDEO_DRIVER_WINDOWS)
    tmp = PyLong_FromLong((long)info.info.win.window);
    PyDict_SetItemString(dict, "window", tmp);
    Py_DECREF(tmp);

    tmp = PyLong_FromLong((long)info.info.win.hdc);
    PyDict_SetItemString(dict, "hdc", tmp);
    Py_DECREF(tmp);

    tmp = PyLong_FromLong((long)info.info.win.hinstance);
    PyDict_SetItemString(dict, "hinstance", tmp);
    Py_DECREF(tmp);
#endif
#if defined(SDL_VIDEO_DRIVER_WINRT)
    tmp = PyCapsule_New(info.info.winrt.window, "window", NULL);
    PyDict_SetItemString(dict, "window", tmp);
    Py_DECREF(tmp);
#endif
#if defined(SDL_VIDEO_DRIVER_X11)
    tmp = PyInt_FromLong(info.info.x11.window);
    PyDict_SetItemString(dict, "window", tmp);
    Py_DECREF(tmp);

    tmp = PyCapsule_New(info.info.x11.display, "display", NULL);
    PyDict_SetItemString(dict, "display", tmp);
    Py_DECREF(tmp);
#endif
#if defined(SDL_VIDEO_DRIVER_DIRECTFB)
    tmp = PyCapsule_New(info.info.dfb.dfb, "dfb", NULL);
    PyDict_SetItemString(dict, "dfb", tmp);
    Py_DECREF(tmp);

    tmp = PyCapsule_New(info.info.dfb.window, "window", NULL);
    PyDict_SetItemString(dict, "window", tmp);
    Py_DECREF(tmp);

    tmp = PyCapsule_New(info.info.dfb.surface, "surface", NULL);
    PyDict_SetItemString(dict, "surface", tmp);
    Py_DECREF(tmp);
#endif
#if defined(SDL_VIDEO_DRIVER_COCOA)
    tmp = PyCapsule_New(info.info.cocoa.window, "window", NULL);
    PyDict_SetItemString(dict, "window", tmp);
    Py_DECREF(tmp);
#endif
#if defined(SDL_VIDEO_DRIVER_UIKIT)
    tmp = PyCapsule_New(info.info.uikit.window, "window", NULL);
    PyDict_SetItemString(dict, "window", tmp);
    Py_DECREF(tmp);

    tmp = PyLong_FromLong(info.info.uikit.framebuffer);
    PyDict_SetItemString(dict, "framebuffer", tmp);
    Py_DECREF(tmp);

    tmp = PyLong_FromLong(info.info.uikit.colorbuffer);
    PyDict_SetItemString(dict, "colorbuffer", tmp);
    Py_DECREF(tmp);

    tmp = PyLong_FromLong(info.info.uikit.resolveFramebuffer);
    PyDict_SetItemString(dict, "resolveFramebuffer", tmp);
    Py_DECREF(tmp);
#endif
#if defined(SDL_VIDEO_DRIVER_WAYLAND)
    tmp = PyCapsule_New(info.info.wl.display, "display", NULL);
    PyDict_SetItemString(dict, "display", tmp);
    Py_DECREF(tmp);

    tmp = PyCapsule_New(info.info.wl.surface, "surface", NULL);
    PyDict_SetItemString(dict, "surface", tmp);
    Py_DECREF(tmp);

    tmp = PyCapsule_New(info.info.wl.shell_surface, "shell_surface", NULL);
    PyDict_SetItemString(dict, "shell_surface", tmp);
    Py_DECREF(tmp);
#endif
#if defined(SDL_VIDEO_DRIVER_MIR) /* no longer available, left for API/ABI \
                                     compatibility. Remove in 2.1! */
    tmp = PyCapsule_New(info.info.mir.connection, "connection", NULL);
    PyDict_SetItemString(dict, "connection", tmp);
    Py_DECREF(tmp);

    tmp = PyCapsule_New(info.info.mir.surface, "surface", NULL);
    PyDict_SetItemString(dict, "surface", tmp);
    Py_DECREF(tmp);
#endif
#if defined(SDL_VIDEO_DRIVER_ANDROID)
    tmp = PyCapsule_New(info.info.android.window, "window", NULL);
    PyDict_SetItemString(dict, "window", tmp);
    Py_DECREF(tmp);

    tmp = PyLong_FromLong((long)info.info.android.surface);
    PyDict_SetItemString(dict, "surface", tmp);
    Py_DECREF(tmp);
#endif
#if defined(SDL_VIDEO_DRIVER_VIVANTE)
    tmp = PyLong_FromLong((long)info.info.vivante.display);
    PyDict_SetItemString(dict, "display", tmp);
    Py_DECREF(tmp);

    tmp = PyLong_FromLong((long)info.info.vivante.window);
    PyDict_SetItemString(dict, "window", tmp);
    Py_DECREF(tmp);
#endif

#endif /* IS_SDLv2 */

    return dict;
}

/* display functions */
#if IS_SDLv2
static PyObject *
pg_get_driver(PyObject *self, PyObject *args)
{
    const char *name = NULL;
    VIDEO_INIT_CHECK();
    name = SDL_GetCurrentVideoDriver();
    if (!name)
        Py_RETURN_NONE;
    return Text_FromUTF8(name);
}

static PyObject *
pg_get_surface(PyObject *self, PyObject *args)
{
    _DisplayState *state = DISPLAY_MOD_STATE(self);
    SDL_Window *win = pg_GetDefaultWindow();

    if (pg_renderer!=NULL || state->using_gl) {
        pgSurfaceObject *surface = pg_GetDefaultWindowSurface();
        if (!surface)
            Py_RETURN_NONE;
        Py_INCREF(surface);
        return (PyObject *)surface;
    }
    else if (win==NULL) {
        Py_RETURN_NONE;
    }
    else {
        SDL_Surface *sdl_surface = SDL_GetWindowSurface(win);
        pgSurfaceObject *old_surface = pg_GetDefaultWindowSurface();
        if (sdl_surface != old_surface->surf) {
            pgSurfaceObject *new_surface = pgSurface_New2(sdl_surface, SDL_FALSE);
            if (!new_surface)
                return NULL;
            pg_SetDefaultWindowSurface(new_surface);
            Py_INCREF((PyObject *)new_surface);
            return (PyObject *)new_surface;
        }
        Py_INCREF(old_surface);
        return (PyObject *)old_surface;
    }
    return NULL;
}
#else  /* IS_SDLv1 */
static PyObject *
pg_get_driver(PyObject *self, PyObject *args)
{
    char buf[256];
    VIDEO_INIT_CHECK();
    if (!SDL_VideoDriverName(buf, sizeof(buf)))
        Py_RETURN_NONE;
    return Text_FromUTF8(buf);
}

static PyObject *
pg_get_surface(PyObject *self, PyObject *args)
{
    if (!pgDisplaySurfaceObject)
        Py_RETURN_NONE;
    Py_INCREF((PyObject *)pgDisplaySurfaceObject);
    return (PyObject *)pgDisplaySurfaceObject;
}
#endif /* IS_SDLv1 */

static PyObject *
pg_gl_set_attribute(PyObject *self, PyObject *arg)
{
    int flag, value, result;
    VIDEO_INIT_CHECK();
    if (!PyArg_ParseTuple(arg, "ii", &flag, &value))
        return NULL;
    if (flag == -1) /*an undefined/unsupported val, ignore*/
        Py_RETURN_NONE;
    result = SDL_GL_SetAttribute(flag, value);
    if (result == -1)
        return RAISE(pgExc_SDLError, SDL_GetError());
    Py_RETURN_NONE;
}

static PyObject *
pg_gl_get_attribute(PyObject *self, PyObject *arg)
{
    int flag, value, result;
    VIDEO_INIT_CHECK();
    if (!PyArg_ParseTuple(arg, "i", &flag))
        return NULL;
    result = SDL_GL_GetAttribute(flag, &value);
    if (result == -1)
        return RAISE(pgExc_SDLError, SDL_GetError());
    return PyInt_FromLong(value);
}

#if IS_SDLv2

/*
** Looks at the SDL1 environment variables:
**    - SDL_VIDEO_WINDOW_POS
*         "x,y"
*         "center"
**    - SDL_VIDEO_CENTERED
*         if set the window should be centered.
*
*  Returns:
*      0 if we do not want to position the window.
*      1 if we set the x and y.
*          x, and y are set to the x and y.
*          center_window is set to 0.
*      2 if we want the window centered.
*          center_window is set to 1.
*/
int
_get_video_window_pos(int *x, int *y, int *center_window)
{
    const char *sdl_video_window_pos = SDL_getenv("SDL_VIDEO_WINDOW_POS");
    const char *sdl_video_centered = SDL_getenv("SDL_VIDEO_CENTERED");
    int xx, yy;
    if (sdl_video_window_pos) {
        if (SDL_sscanf(sdl_video_window_pos, "%d,%d", &xx, &yy) == 2) {
            *x = xx;
            *y = yy;
            *center_window = 0;
            return 1;
        }
        if (SDL_strcmp(sdl_video_window_pos, "center") == 0) {
            sdl_video_centered = sdl_video_window_pos;
        }
    }
    if (sdl_video_centered) {
        *center_window = 1;
        return 2;
    }
    return 0;
}

static int SDLCALL
pg_ResizeEventWatch(void *userdata, SDL_Event *event) {
    SDL_Window *pygame_window;
    PyObject *self;
    _DisplayState *state;
    SDL_Window *window;

    if (event->type != SDL_WINDOWEVENT)
        return 0;

    self= (PyObject *) userdata;
    pygame_window = pg_GetDefaultWindow();
    state = DISPLAY_MOD_STATE(self);

    window = SDL_GetWindowFromID(event->window.windowID);
    if (window != pygame_window)
        return 0;

    if (pg_renderer!=NULL) {
#if (SDL_VERSION_ATLEAST(2, 0, 5))

        if (event->window.event == SDL_WINDOWEVENT_MAXIMIZED) {
            SDL_RenderSetIntegerScale(pg_renderer,
                                      SDL_FALSE);
        }
        if (event->window.event == SDL_WINDOWEVENT_RESTORED) {
            SDL_RenderSetIntegerScale(pg_renderer,
                                      !(SDL_GetHintBoolean("SDL_HINT_RENDER_SCALE_QUALITY",SDL_FALSE)));
        }
#endif
        return 0;
    }

    if (state->using_gl) {
        if (event->window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {

            GL_glViewport_Func p_glViewport = (GL_glViewport_Func)SDL_GL_GetProcAddress("glViewport");
            int wnew=event->window.data1;
            int hnew=event->window.data2;
            SDL_GL_MakeCurrent(pygame_window, state->gl_context);
            if (state->scaled_gl) {
                float saved_aspect_ratio =
                    ((float)state->scaled_gl_w) / (float)state->scaled_gl_h;
                float window_aspect_ratio = ((float)wnew) / (float)hnew;

                if (window_aspect_ratio > saved_aspect_ratio) {
                    int width = (int)(hnew * saved_aspect_ratio);
                    p_glViewport((wnew - width) / 2, 0, width, hnew);
                }
                else {
                    p_glViewport(0, 0, wnew, (int)(wnew / saved_aspect_ratio));
                }
            }
            else {
                p_glViewport(0, 0, wnew, hnew);
            }
        }
        return 0;
    }

    if (event->window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
        if (window == pygame_window) {
            SDL_Surface *sdl_surface = SDL_GetWindowSurface(window);
            pgSurfaceObject *old_surface = pg_GetDefaultWindowSurface();
            if (sdl_surface != old_surface->surf) {
                old_surface->surf=sdl_surface;
            }
        }
    }
    return 0;
}

static PyObject *
pg_display_set_autoresize(PyObject *self, PyObject *args)
{
    SDL_bool do_resize;
    _DisplayState *state = DISPLAY_MOD_STATE(self);

#if PY3
    if (!PyArg_ParseTuple(args, "p",  &do_resize))
        return NULL;
#else
    if (!PyArg_ParseTuple(args, "i", &do_resize))
        return NULL;
#endif

    state->auto_resize=do_resize;
    SDL_DelEventWatch(pg_ResizeEventWatch, self);

    if(do_resize) {
        SDL_AddEventWatch(pg_ResizeEventWatch, self);
        Py_RETURN_TRUE;
    }
    else {
        Py_RETURN_FALSE;
    }
}


int
_get_display(SDL_Window *win)
{
    char *display_env = SDL_getenv("PYGAME_DISPLAY");
    int display = 0; /* default display 0 */

    if (win != NULL) {
        display = SDL_GetWindowDisplayIndex(win);
        return display;
    }
    else if (display_env != NULL) {
        display = SDL_atoi(display_env);
        return display;
    }
    /* On e.g. Linux X11, checking the mouse pointer requires that the
     * video subsystem is initialized to avoid crashes.
     *
     * Note that we do not bother raising an error here; the condition will
     * be rechecked after parsing the arguments and the function will throw
     * the relevant error there.
     */
    else if (SDL_WasInit(SDL_INIT_VIDEO)) {
        /* get currently "active" desktop, containing mouse ptr */
        int num_displays, i;
        SDL_Rect display_bounds;
        SDL_Point mouse_position;
        SDL_GetGlobalMouseState(&mouse_position.x, &mouse_position.y);
        num_displays = SDL_GetNumVideoDisplays();

        for (i = 0; i < num_displays; i++) {
            if (SDL_GetDisplayBounds(i, &display_bounds) == 0) {
                if (SDL_PointInRect(&mouse_position, &display_bounds)) {
                    display = i;
                    break;
                }
            }
        }
    }
    return display;
}

static PyObject *
pg_set_mode(PyObject *self, PyObject *arg, PyObject *kwds)
{
    static const char *const DefaultTitle = "pygame window";

    _DisplayState *state = DISPLAY_MOD_STATE(self);
    SDL_Window *win = pg_GetDefaultWindow();
    pgSurfaceObject *surface = pg_GetDefaultWindowSurface();
    SDL_Surface *surf = NULL;
    SDL_Surface *newownedsurf = NULL;
    int depth = 0;
    int flags = 0;
    int w = 0;
    int h = 0;
    int vsync = SDL_FALSE;
    /* display will get overwritten by ParseTupleAndKeywords only if display
       parameter is given. By default, put the new window on the same
       screen as the old one */
    int display = _get_display(win);
    char *title = state->title;
    int init_flip = 0;
    char *scale_env;

    char *keywords[] = {"size", "flags", "depth", "display", "vsync", NULL};

    scale_env = SDL_getenv("PYGAME_FORCE_SCALE");

    if (!PyArg_ParseTupleAndKeywords(arg, kwds, "|(ii)iiii", keywords, &w, &h,
                                     &flags, &depth, &display, &vsync))
        return NULL;

    if (scale_env != NULL) {
        flags |= PGS_SCALED;
        if (strcmp(scale_env, "photo") == 0) {
            SDL_SetHintWithPriority(SDL_HINT_RENDER_SCALE_QUALITY, "best",
                                    SDL_HINT_NORMAL);
        }
    }

    if (w < 0 || h < 0)
        return RAISE(pgExc_SDLError, "Cannot set negative sized display mode");

    if (!SDL_WasInit(SDL_INIT_VIDEO)) {
        /*note SDL works special like this too*/
        if (!pg_init(NULL, NULL))
            return NULL;
    }

    state->using_gl = (flags & PGS_OPENGL) != 0;
    state->scaled_gl = state->using_gl && (flags & PGS_SCALED) != 0;

    if (state->scaled_gl) {
        if (PyErr_WarnEx(PyExc_FutureWarning,
                         "SCALED|OPENGL is experimental and subject to change",
                         1) != 0)
            return NULL;
    }

    if (!state->title) {
        state->title = malloc((strlen(DefaultTitle) + 1) * sizeof(char));
        if (!state->title)
            return PyErr_NoMemory();
        strcpy(state->title, DefaultTitle);
        title = state->title;
    }

    // if (vsync && !(flags & (PGS_SCALED | PGS_OPENGL))) {
    //     return RAISE(pgExc_SDLError,
    //                  "vsync needs either SCALED or OPENGL flag");
    // }

    /* set these only in toggle_fullscreen, clear on set_mode */
    state->toggle_windowed_w = 0;
    state->toggle_windowed_h = 0;

    if (pg_texture) {
        SDL_DestroyTexture(pg_texture);
        pg_texture = NULL;
    }

    if (pg_renderer) {
        SDL_DestroyRenderer(pg_renderer);
        pg_renderer = NULL;
    }

    SDL_DelEventWatch(pg_ResizeEventWatch, self);

    {
        Uint32 sdl_flags = 0;
        SDL_DisplayMode display_mode;

        if (SDL_GetDesktopDisplayMode(display, &display_mode) != 0) {
            return RAISE(pgExc_SDLError, SDL_GetError());
        }

        if (w == 0 && h == 0 && !(flags & PGS_SCALED)) {
            /* We are free to choose a resolution in this case, so we can
           avoid changing the physical resolution. This used to default
           to the max supported by the monitor, but we can use current
           desktop resolution without breaking compatibility. */
            w = display_mode.w;
            h = display_mode.h;
        }

        if (flags & PGS_FULLSCREEN) {
            if (flags & PGS_SCALED) {
                sdl_flags |= SDL_WINDOW_FULLSCREEN_DESKTOP;
            }
            else if (w == display_mode.w && h == display_mode.h) {
                /* No need to change physical resolution.
               Borderless fullscreen is preferred when possible */
                sdl_flags |= SDL_WINDOW_FULLSCREEN_DESKTOP;
            }
            else {
                sdl_flags |= SDL_WINDOW_FULLSCREEN;
            }
        }

        if (flags & PGS_SCALED) {
            if (w == 0 || h == 0)
                return RAISE(pgExc_SDLError,
                             "Cannot set 0 sized SCALED display mode");
        }

        if (flags & PGS_OPENGL)
            sdl_flags |= SDL_WINDOW_OPENGL;
        if (flags & PGS_NOFRAME)
            sdl_flags |= SDL_WINDOW_BORDERLESS;
        if (flags & PGS_RESIZABLE) {
            sdl_flags |= SDL_WINDOW_RESIZABLE;
            if(state->auto_resize)
                SDL_AddEventWatch(pg_ResizeEventWatch, self);
        }
        if (flags & PGS_SHOWN)
            sdl_flags |= SDL_WINDOW_SHOWN;
        if (flags & PGS_HIDDEN)
            sdl_flags |= SDL_WINDOW_HIDDEN;
        if (!(sdl_flags & SDL_WINDOW_HIDDEN))
            sdl_flags |= SDL_WINDOW_SHOWN;
        if (flags & PGS_OPENGL) {
            /* Must be called before creating context */
            if (flags & PGS_DOUBLEBUF) {
                flags &= ~PGS_DOUBLEBUF;
                SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
            }
            else
                SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 0);
        }

#pragma PG_WARN(Not setting bpp ?)
#pragma PG_WARN(Add mode stuff.)
        {
            int w_1 = w, h_1 = h;
            int scale = 1;
            int center_window = 0;
            int x = SDL_WINDOWPOS_UNDEFINED_DISPLAY(display);
            int y = SDL_WINDOWPOS_UNDEFINED_DISPLAY(display);

            _get_video_window_pos(&x, &y, &center_window);
            if (center_window) {
                x = SDL_WINDOWPOS_CENTERED_DISPLAY(display);
                y = SDL_WINDOWPOS_CENTERED_DISPLAY(display);
            }

            if (win) {
                if (SDL_GetWindowDisplayIndex(win) == display) {
                    // fullscreen windows don't hold window x and y as needed
                    if (SDL_GetWindowFlags(win) & (SDL_WINDOW_FULLSCREEN | SDL_WINDOW_FULLSCREEN_DESKTOP)) { 
                        x = state->fullscreen_backup_x;
                        y = state->fullscreen_backup_y;

                        // if the program goes into fullscreen first the "saved x and y" are "undefined position"
                        // that should be interpreted as a cue to center the window
                        if (x == SDL_WINDOWPOS_UNDEFINED_DISPLAY(display))
                            x = SDL_WINDOWPOS_CENTERED_DISPLAY(display);
                        if (y == SDL_WINDOWPOS_UNDEFINED_DISPLAY(display))
                            y = SDL_WINDOWPOS_CENTERED_DISPLAY(display);
                    }
                    else {
                        SDL_GetWindowPosition(win, &x, &y);
                    }
                    
                }
                if (!(flags & PGS_OPENGL) !=
                    !(SDL_GetWindowFlags(win) & SDL_WINDOW_OPENGL)) {
                    pg_SetDefaultWindow(NULL);
                    win = NULL;
                }
            }

            if (flags & PGS_SCALED && !(flags & PGS_FULLSCREEN)) {
                SDL_Rect display_bounds;
                int fractional_scaling = SDL_FALSE;

#if (SDL_VERSION_ATLEAST(2, 0, 5))
                if (0 !=
                    SDL_GetDisplayUsableBounds(display, &display_bounds)) {
                    return RAISE(pgExc_SDLError, SDL_GetError());
                }
#else
                display_bounds.w = display_mode.w - 80;
                display_bounds.h = display_mode.h - 30;
#endif

#if (SDL_VERSION_ATLEAST(2, 0, 5))
                if (SDL_GetHintBoolean("SDL_HINT_RENDER_SCALE_QUALITY",
                                       SDL_FALSE))
                    fractional_scaling = SDL_TRUE;
#endif
                if (state->scaled_gl)
                    fractional_scaling = SDL_TRUE;

                if (fractional_scaling) {
                    float aspect_ratio = ((float)w) / (float)h;

                    w_1 = display_bounds.w;
                    h_1 = display_bounds.h;

                    if (((float)w_1) / (float)h_1 > aspect_ratio) {
                        w_1 = h_1 * aspect_ratio;
                    }
                    else {
                        h_1 = w_1 / aspect_ratio;
                    }
                }
                else {
                    int xscale, yscale;

                    xscale = display_bounds.w / w;
                    yscale = display_bounds.h / h;

                    scale = xscale < yscale ? xscale : yscale;

                    if (scale < 1)
                        scale = 1;

                    w_1 = w * scale;
                    h_1 = h * scale;
                }
            }

            // SDL doesn't preserve window position in fullscreen mode
            // However, windows coming out of fullscreen need these to go back into the correct position
            if (sdl_flags & (SDL_WINDOW_FULLSCREEN | SDL_WINDOW_FULLSCREEN_DESKTOP)) {
                state->fullscreen_backup_x = x;
                state->fullscreen_backup_y = y;
            }

            if (!win) {
                /*open window*/
                win = SDL_CreateWindow(title, x, y, w_1, h_1, sdl_flags);
                if (!win)
                    return RAISE(pgExc_SDLError, SDL_GetError());
                init_flip = 1;
            }
            else {
                /* change existing window.
                 this invalidates the display surface*/
                SDL_SetWindowTitle(win, title);
                SDL_SetWindowSize(win, w_1, h_1);

#if (SDL_VERSION_ATLEAST(2, 0, 5))
                SDL_SetWindowResizable(win, flags & PGS_RESIZABLE);
#endif
                SDL_SetWindowBordered(win, (flags & PGS_NOFRAME) == 0);

                if ((flags & PGS_SHOWN) || !(flags & PGS_HIDDEN))
                    SDL_ShowWindow(win);
                else if (flags & PGS_HIDDEN)
                    SDL_HideWindow(win);

                if (0 !=
                    SDL_SetWindowFullscreen(
                        win, sdl_flags & (SDL_WINDOW_FULLSCREEN |
                                          SDL_WINDOW_FULLSCREEN_DESKTOP))) {
                    return RAISE(pgExc_SDLError, SDL_GetError());
                }

                SDL_SetWindowPosition(win, x, y);

                assert(surface);
            }
        }

        if (state->using_gl) {
            if (!state->gl_context) {
                state->gl_context = SDL_GL_CreateContext(win);
                if (!state->gl_context) {
                    _display_state_cleanup(state);
                    PyErr_SetString(pgExc_SDLError, SDL_GetError());
                    goto DESTROY_WINDOW;
                }
                /* SDL_GetWindowSurface can not be used when using GL.
                According to https://wiki.libsdl.org/SDL_GetWindowSurface

                So we make a fake surface.
                */
                surf = SDL_CreateRGBSurface(SDL_SWSURFACE, w, h, 32,
                                            0xff << 16, 0xff << 8, 0xff, 0);
                newownedsurf = surf;
            }
            else {
                surf = pgSurface_AsSurface(surface);
            }
            if (flags & PGS_SCALED) {
                state->scaled_gl_w = w;
                state->scaled_gl_h = h;
            }

            /* Even if this succeeds, we can never *really* know if vsync
               actually works. There may be screen tearing, blocking double
               buffering, triple buffering, render-offloading where the driver
               for the on-board graphics *doesn't* have vsync enabled, or cases
               where the driver lies to us because the user has configured
               vsync to be aways on or always off, or vsync is on by default
               for the whole desktop because of wayland GL compositing. */
            if (vsync) {
                if (SDL_GL_SetSwapInterval(-1) != 0) {
                    if (PyErr_WarnEx(PyExc_Warning,
                                     "adaptive vsync for OpenGL not "
                                     "available, trying regular",
                                     1) != 0) {
                        _display_state_cleanup(state);
                        goto DESTROY_WINDOW;
                    }
                    if (SDL_GL_SetSwapInterval(1) != 0) {
                        if (PyErr_WarnEx(PyExc_Warning,
                                         "regular vsync for OpenGL *also* not "
                                         "available",
                                         1) != 0) {
                            _display_state_cleanup(state);
                            goto DESTROY_WINDOW;
                        }
                    }
                }
            }
            else {
                SDL_GL_SetSwapInterval(0);
            }
        }
        else {
            if (state->gl_context) {
                SDL_GL_DeleteContext(state->gl_context);
                state->gl_context = NULL;
            }

            if (flags & PGS_SCALED) {
                if (pg_renderer == NULL) {
                    SDL_RendererInfo info;

                    SDL_SetHintWithPriority(SDL_HINT_RENDER_SCALE_QUALITY,
                                            "nearest", SDL_HINT_DEFAULT);

                    if (vsync) {
                        pg_renderer = SDL_CreateRenderer(
                            win, -1, SDL_RENDERER_PRESENTVSYNC);
                    }
                    else {
                        pg_renderer = SDL_CreateRenderer(win, -1, 0);
                    }

                    if (pg_renderer == NULL) {
                        return RAISE(pgExc_SDLError,
                                     "failed to create renderer");
                    }

#if (SDL_VERSION_ATLEAST(2, 0, 5))

                    /* use whole screen with uneven pixels on fullscreen,
                       exact scale otherwise.
                       we chose the window size for this to work */
                    SDL_RenderSetIntegerScale(
                        pg_renderer,
                        !(flags & PGS_FULLSCREEN ||
                          SDL_GetHintBoolean("SDL_HINT_RENDER_SCALE_QUALITY",
                                             SDL_FALSE)));
#endif
                    SDL_RenderSetLogicalSize(pg_renderer, w, h);
                    /* this must be called after creating the renderer!*/
                    SDL_SetWindowMinimumSize(win, w, h);

                    SDL_GetRendererInfo(pg_renderer, &info);
                    if (vsync && !(info.flags & SDL_RENDERER_PRESENTVSYNC)) {
                        if (PyErr_WarnEx(PyExc_Warning,
                                         "could not enable vsync", 1) != 0) {
                            _display_state_cleanup(state);
                            goto DESTROY_WINDOW;
                        }
                    }
                    if (!(info.flags & SDL_RENDERER_ACCELERATED)) {
                        if (PyErr_WarnEx(PyExc_Warning,
                                         "no fast renderer available",
                                         1) != 0) {
                            _display_state_cleanup(state);
                            goto DESTROY_WINDOW;
                        }
                    }

                    pg_texture = SDL_CreateTexture(
                        pg_renderer, SDL_PIXELFORMAT_ARGB8888,
                        SDL_TEXTUREACCESS_STREAMING, w, h);
                }
                surf = SDL_CreateRGBSurface(SDL_SWSURFACE, w, h, 32,
                                            0xff << 16, 0xff << 8, 0xff, 0);
                newownedsurf = surf;
            }
            else {
                surf = SDL_GetWindowSurface(win);
            }
        }
        if (state->gamma_ramp) {
            int result = SDL_SetWindowGammaRamp(win, state->gamma_ramp,
                                                state->gamma_ramp + 256,
                                                state->gamma_ramp + 512);
            if (result) /* SDL Error? */
            {
                /* Discard a possibly faulty gamma ramp. */
                _display_state_cleanup(state);

                /* Recover error, then destroy the window */
                PyErr_SetString(pgExc_SDLError, SDL_GetError());
                goto DESTROY_WINDOW;
            }
        }

        if (state->using_gl && pg_renderer != NULL) {
            _display_state_cleanup(state);
            PyErr_SetString(
                pgExc_SDLError,
                "GL context and SDL_Renderer created at the same time");
            goto DESTROY_WINDOW;
        }

        if (!surf) {
            _display_state_cleanup(state);
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            goto DESTROY_WINDOW;
        }
        if (!surface) {
            surface = pgSurface_New2(surf, newownedsurf != NULL);
        }
        else {
            pgSurface_SetSurface(surface, surf, newownedsurf != NULL);
            Py_INCREF(surface);
        }
        if (!surface) {
            if (newownedsurf)
                SDL_FreeSurface(newownedsurf);
            _display_state_cleanup(state);
            goto DESTROY_WINDOW;
        }

        /*no errors; make the window available*/
        pg_SetDefaultWindow(win);
        pg_SetDefaultWindowSurface(surface);
        Py_DECREF(surface);

        /* ensure window is initially black */
        if (init_flip)
            pg_flip_internal(state);
    }

    /*set the window icon*/
    if (!state->icon) {
        state->icon = pg_display_resource(icon_defaultname);
        if (!state->icon)
            PyErr_Clear();
        else {
            SDL_SetColorKey(pgSurface_AsSurface(state->icon), SDL_TRUE, 0);
        }
    }
    if (state->icon)
        SDL_SetWindowIcon(win, pgSurface_AsSurface(state->icon));

    /*probably won't do much, but can't hurt, and might help*/
    SDL_PumpEvents();

    /*return the window's surface (screen)*/
    Py_INCREF(surface);
    return (PyObject *)surface;

DESTROY_WINDOW:

    if (win == pg_GetDefaultWindow())
        pg_SetDefaultWindow(NULL);
    else if (win)
        SDL_DestroyWindow(win);
    return NULL;
}

static int
_pg_get_default_display_masks(int bpp, Uint32 *Rmask, Uint32 *Gmask,
                              Uint32 *Bmask)
{
    switch (bpp) {
        case 8:
            *Rmask = 0;
            *Gmask = 0;
            *Bmask = 0;
            break;
        case 12:
            *Rmask = 0xFF >> 4 << 8;
            *Gmask = 0xFF >> 4 << 4;
            *Bmask = 0xFF >> 4;
            break;
        case 15:
            *Rmask = 0xFF >> 3 << 10;
            *Gmask = 0xFF >> 3 << 5;
            *Bmask = 0xFF >> 3;
            break;
        case 16:
            *Rmask = 0xFF >> 3 << 11;
            *Gmask = 0xFF >> 2 << 5;
            *Bmask = 0xFF >> 3;
            break;
        case 24:
        case 32:
            *Rmask = 0xFF << 16;
            *Gmask = 0xFF << 8;
            *Bmask = 0xFF;
            break;
        default:
            PyErr_SetString(PyExc_ValueError, "nonstandard bit depth given");
            return -1;
    }
    return 0;
}

static PyObject *
pg_window_size(PyObject *self, PyObject *args)
{
    SDL_Window *win = pg_GetDefaultWindow();
    int w, h;
    if (!win)
        return RAISE(pgExc_SDLError, "No open window");
    SDL_GetWindowSize(win, &w, &h);
    return Py_BuildValue("(ii)", w, h);
}

static PyObject *
pg_mode_ok(PyObject *self, PyObject *args, PyObject *kwds)
{
    SDL_DisplayMode desired, closest;
    int bpp = 0;
    int flags = SDL_SWSURFACE;
    int display_index = 0;

    char *keywords[] = {"size", "flags", "depth", "display", NULL};

    VIDEO_INIT_CHECK();

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "(ii)|iii", keywords,
                                     &desired.w, &desired.h, &flags, &bpp,
                                     &display_index)) {
        return NULL;
    }
    if (display_index < 0 || display_index >= SDL_GetNumVideoDisplays()) {
        return RAISE(PyExc_ValueError,
                     "The display index must be between 0"
                     " and the number of displays.");
    }
#pragma PG_WARN(Ignoring most flags)

    desired.driverdata = 0;
    desired.refresh_rate = 0;

    if (bpp == 0) {
        desired.format = 0;
    }
    else {
        Uint32 Rmask, Gmask, Bmask;
        if (_pg_get_default_display_masks(bpp, &Rmask, &Gmask, &Bmask)) {
            PyErr_Clear();
            return PyInt_FromLong((long)0);
        }
        desired.format =
            SDL_MasksToPixelFormatEnum(bpp, Rmask, Gmask, Bmask, 0);
    }
    if (!SDL_GetClosestDisplayMode(display_index, &desired, &closest)) {
        if (flags & PGS_FULLSCREEN)
            return PyInt_FromLong((long)0);
        closest.format = desired.format;
    }
    if ((flags & PGS_FULLSCREEN) &&
        (closest.w != desired.w || closest.h != desired.h))
        return PyInt_FromLong((long)0);
    return PyInt_FromLong(SDL_BITSPERPIXEL(closest.format));
}

static PyObject *
pg_list_modes(PyObject *self, PyObject *args, PyObject *kwds)
{
    SDL_DisplayMode mode;
    int nummodes;
    int bpp = 0;
    int flags = PGS_FULLSCREEN;
    int display_index = 0;
    PyObject *list, *size;
    int i;

    char *keywords[] = {"depth", "flags", "display", NULL};

    VIDEO_INIT_CHECK();

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|bii", keywords, &bpp,
                                     &flags, &display_index)) {
        return NULL;
    }

    if (display_index < 0 || display_index >= SDL_GetNumVideoDisplays()) {
        return RAISE(PyExc_ValueError,
                     "The display index must be between 0"
                     " and the number of displays.");
    }
#pragma PG_WARN(Ignoring flags)

    if (bpp == 0) {
        SDL_DisplayMode curmode;
        if (SDL_GetCurrentDisplayMode(display_index, &curmode))
            return RAISE(pgExc_SDLError, SDL_GetError());
        bpp = SDL_BITSPERPIXEL(curmode.format);
    }

    nummodes = SDL_GetNumDisplayModes(display_index);
    if (nummodes < 0)
        return RAISE(pgExc_SDLError, SDL_GetError());

    if (!(list = PyList_New(0)))
        return NULL;

    for (i = 0; i < nummodes; i++) {
        if (SDL_GetDisplayMode(display_index, i, &mode) < 0) {
            Py_DECREF(list);
            return RAISE(pgExc_SDLError, SDL_GetError());
        }
        /* use reasonable defaults (cf. SDL_video.c) */
        if (!mode.format)
            mode.format = SDL_PIXELFORMAT_RGB888;
        if (!mode.w)
            mode.w = 640;
        if (!mode.h)
            mode.h = 480;
        if (SDL_BITSPERPIXEL(mode.format) != bpp)
            continue;
        if (!(size = Py_BuildValue("(ii)", mode.w, mode.h))) {
            Py_DECREF(list);
            return NULL;
        }
        if (0 != PyList_Append(list, size)) {
            Py_DECREF(list);
            Py_DECREF(size);
            return NULL; /* Exception already set. */
        }
        Py_DECREF(size);
    }
    return list;
}

static int
pg_flip_internal(_DisplayState *state)
{
    SDL_Window *win = pg_GetDefaultWindow();
    int status = 0;

    /* Same check as VIDEO_INIT_CHECK() but returns -1 instead of NULL on
     * fail. */
    if (!SDL_WasInit(SDL_INIT_VIDEO)) {
        PyErr_SetString(pgExc_SDLError, "video system not initialized");
        return -1;
    }

    if (!win) {
        PyErr_SetString(pgExc_SDLError, "Display mode not set");
        return -1;
    }

    Py_BEGIN_ALLOW_THREADS;
    if (state->using_gl) {
        SDL_GL_SwapWindow(win);
    }
    else {
#if IS_SDLv2
        if (pg_renderer != NULL) {
            SDL_Surface *screen =
                pgSurface_AsSurface(pg_GetDefaultWindowSurface());
            SDL_UpdateTexture(pg_texture, NULL, screen->pixels, screen->pitch);
            SDL_RenderClear(pg_renderer);
            SDL_RenderCopy(pg_renderer, pg_texture, NULL, NULL);
            SDL_RenderPresent(pg_renderer);
        }
        else {
            /* Force a re-initialization of the surface in case it
             * has been resized to avoid "please call SDL_GetWindowSurface"
             * errors that the programmer cannot fix
             */
            pgSurfaceObject *screen = pg_GetDefaultWindowSurface();
            SDL_Surface *new_surface = SDL_GetWindowSurface(win);

            if (new_surface != ((pgSurfaceObject *)screen)->surf) {
                screen->surf = new_surface;
            }
            status = SDL_UpdateWindowSurface(win);
        }
#else
        status = SDL_UpdateWindowSurface(win);
#endif
    }
    Py_END_ALLOW_THREADS;

    if (status < 0) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return -1;
    }

    return 0;
}

static PyObject *
pg_flip(PyObject *self, PyObject *args)
{
    if (pg_flip_internal(DISPLAY_MOD_STATE(self)) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
pg_num_displays(PyObject *self, PyObject *args)
{
    int ret = SDL_GetNumVideoDisplays();
    if (ret < 0)
        return RAISE(pgExc_SDLError, SDL_GetError());
    return PyInt_FromLong(ret);
}

#else /* IS_SDLv1 */
static PyObject *
pg_set_mode(PyObject *self, PyObject *arg, PyObject *kwds)
{
    SDL_Surface *surf;
    int depth = 0;
    int flags = SDL_SWSURFACE;
    int w = 0;
    int h = 0;
    int display = 0;
    int hasbuf;
    char *title, *icontitle;

    char *keywords[] = {"size", "flags", "depth", "display", NULL};

    if (!PyArg_ParseTupleAndKeywords(arg, kwds, "|(ii)iii", keywords, &w, &h,
                                     &flags, &depth, &display))
        return NULL;

    if (w < 0 || h < 0)
        return RAISE(pgExc_SDLError, "Cannot set negative sized display mode");

    if (w == 0 || h == 0) {
        SDL_version versioninfo;
        SDL_VERSION(&versioninfo);
        if (!(versioninfo.major > 1 ||
              (versioninfo.major == 1 && versioninfo.minor > 2) ||
              (versioninfo.major == 1 && versioninfo.minor == 2 &&
               versioninfo.patch >= 10))) {
            return RAISE(pgExc_SDLError, "Cannot set 0 sized display mode");
        }
    }

    if (!SDL_WasInit(SDL_INIT_VIDEO)) {
        /*note SDL works special like this too*/
        if (!pg_init(NULL, NULL))
            return NULL;
    }

    if (flags & SDL_OPENGL) {
        if (flags & SDL_DOUBLEBUF) {
            flags &= ~SDL_DOUBLEBUF;
            SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
        }
        else
            SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 0);
        if (depth)
            SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, depth);
        surf = SDL_SetVideoMode(w, h, depth, flags);
        if (!surf)
            return RAISE(pgExc_SDLError, SDL_GetError());

        SDL_GL_GetAttribute(SDL_GL_DOUBLEBUFFER, &hasbuf);
        if (hasbuf)
            surf->flags |= SDL_DOUBLEBUF;
    }
    else {
        if (!depth)
            flags |= SDL_ANYFORMAT;
        Py_BEGIN_ALLOW_THREADS;
        surf = SDL_SetVideoMode(w, h, depth, flags);
        Py_END_ALLOW_THREADS;
        if (!surf)
            return RAISE(pgExc_SDLError, SDL_GetError());
    }
    SDL_WM_GetCaption(&title, &icontitle);
    if (!title || !*title)
        SDL_WM_SetCaption("pygame window", "pygame");

    /*probably won't do much, but can't hurt, and might help*/
    SDL_PumpEvents();

    if (pgDisplaySurfaceObject)
        pgDisplaySurfaceObject->surf = surf;
    else
        pgDisplaySurfaceObject = pgSurface_New(surf);

#if !defined(darwin)
    if (!icon_was_set) {
        PyObject *iconsurf = pg_display_resource(icon_defaultname);
        if (!iconsurf)
            PyErr_Clear();
        else {
            SDL_SetColorKey(pgSurface_AsSurface(iconsurf), SDL_SRCCOLORKEY, 0);
            pg_do_set_icon(iconsurf);
            Py_DECREF(iconsurf);
        }
    }
#endif
    Py_INCREF((PyObject *)pgDisplaySurfaceObject);
    return (PyObject *)pgDisplaySurfaceObject;
}

static PyObject *
pg_window_size(PyObject *self, PyObject *args)
{
    if (!pgDisplaySurfaceObject) {
        return RAISE(pgExc_SDLError, "No open window");
    }
    return Py_BuildValue("(ii)",
                         pgSurface_AsSurface(pgDisplaySurfaceObject)->w,
                         pgSurface_AsSurface(pgDisplaySurfaceObject)->h);
}

/* SDL1 mode_ok. Note, there is a separate SDL2 version of this. */
static PyObject *
pg_mode_ok(PyObject *self, PyObject *args, PyObject *kwds)
{
    int depth = 0;
    int w, h;
    int flags = SDL_SWSURFACE;
    int display = 0;
    char *keywords[] = {"size", "flags", "depth", "display", NULL};

    VIDEO_INIT_CHECK();

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "(ii)|iii", keywords, &w, &h,
                                     &flags, &depth, &display))
        return NULL;
    if (!depth)
        depth = SDL_GetVideoInfo()->vfmt->BitsPerPixel;

    return PyInt_FromLong(SDL_VideoModeOK(w, h, depth, flags));
}

static PyObject *
pg_list_modes(PyObject *self, PyObject *args, PyObject *kwds)
{
    SDL_PixelFormat format;
    SDL_Rect **rects;
    int flags = SDL_FULLSCREEN;
    int display_index = 0; /* SDL1 does not use a display_index. */
    PyObject *list, *size;
    char *keywords[] = {"depth", "flags", "display", NULL};

    format.BitsPerPixel = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|bii", keywords,
                                     &format.BitsPerPixel, &flags,
                                     &display_index)) {
        return NULL;
    }

    VIDEO_INIT_CHECK();

    if (!format.BitsPerPixel)
        format.BitsPerPixel = SDL_GetVideoInfo()->vfmt->BitsPerPixel;

    rects = SDL_ListModes(&format, flags);

    if (rects == (SDL_Rect **)-1)
        return PyInt_FromLong(-1);

    if (!(list = PyList_New(0)))
        return NULL;
    if (!rects)
        return list;

    for (; *rects; ++rects) {
        if (!(size = Py_BuildValue("(ii)", (*rects)->w, (*rects)->h))) {
            Py_DECREF(list);
            return NULL;
        }
        if (0 != PyList_Append(list, size)) {
            Py_DECREF(list);
            Py_DECREF(size);
            return NULL; /* Exception already set. */
        }
        Py_DECREF(size);
    }
    return list;
}

static PyObject *
pg_flip(PyObject *self, PyObject *args)
{
    SDL_Surface *screen;
    int status = 0;

    VIDEO_INIT_CHECK();

    screen = SDL_GetVideoSurface();
    if (!screen)
        return RAISE(pgExc_SDLError, "Display mode not set");

    Py_BEGIN_ALLOW_THREADS;
    if (screen->flags & SDL_OPENGL)
        SDL_GL_SwapBuffers();
    else
        status = SDL_Flip(screen);
    Py_END_ALLOW_THREADS;

    if (status < 0)
        return RAISE(pgExc_SDLError, SDL_GetError());
    Py_RETURN_NONE;
}

static PyObject *
pg_num_displays(PyObject *self, PyObject *args)
{
    return PyInt_FromLong(1);
}

#endif /* IS_SDLv1 */

/*BAD things happen when out-of-bound rects go to updaterect*/
static SDL_Rect *
pg_screencroprect(GAME_Rect *r, int w, int h, SDL_Rect *cur)
{
    if (r->x > w || r->y > h || (r->x + r->w) <= 0 || (r->y + r->h) <= 0)
        return 0;
    else {
        int right = MIN(r->x + r->w, w);
        int bottom = MIN(r->y + r->h, h);
        cur->x = (short)MAX(r->x, 0);
        cur->y = (short)MAX(r->y, 0);
        cur->w = (unsigned short)right - cur->x;
        cur->h = (unsigned short)bottom - cur->y;
    }
    return cur;
}

static PyObject *
pg_update(PyObject *self, PyObject *arg)
{
#if IS_SDLv2
    SDL_Window *win = pg_GetDefaultWindow();
    _DisplayState *state = DISPLAY_MOD_STATE(self);
#else  /* IS_SDLv1 */
    SDL_Surface *screen;
#endif /* IS_SDLv1 */
    GAME_Rect *gr, temp = {0};
    int wide, high;
    PyObject *obj;

    VIDEO_INIT_CHECK();

#if IS_SDLv2
    if (!win)
        return RAISE(pgExc_SDLError, "Display mode not set");
    if (pg_renderer != NULL) {
        return pg_flip(self, NULL);
    }
    SDL_GetWindowSize(win, &wide, &high);

    if (state->using_gl)
        return RAISE(pgExc_SDLError, "Cannot update an OPENGL display");
#else  /* IS_SDLv1 */
    screen = SDL_GetVideoSurface();
    if (!screen)
        return RAISE(pgExc_SDLError, SDL_GetError());
    wide = screen->w;
    high = screen->h;
    if (screen->flags & SDL_OPENGL)
        return RAISE(pgExc_SDLError, "Cannot update an OPENGL display");
#endif /* IS_SDLv1 */

    /*determine type of argument we got*/
    if (PyTuple_Size(arg) == 0) {
#if IS_SDLv2
        return pg_flip(self, NULL);
#else  /* IS_SDLv1 */
        SDL_UpdateRect(screen, 0, 0, 0, 0);

        Py_RETURN_NONE;
#endif /* IS_SDLv1 */
    }
    else {
        obj = PyTuple_GET_ITEM(arg, 0);
        if (obj == Py_None)
            gr = &temp;
        else {
            gr = pgRect_FromObject(arg, &temp);
            if (!gr)
                PyErr_Clear();
            else if (gr != &temp) {
                memcpy(&temp, gr, sizeof(temp));
                gr = &temp;
            }
        }
    }

    if (gr) {
#if IS_SDLv2
        SDL_Rect sdlr;

        if (pg_screencroprect(gr, wide, high, &sdlr))
            SDL_UpdateWindowSurfaceRects(win, &sdlr, 1);
#else  /* IS_SDLv1 */
        SDL_Rect sdlr;

        if (pg_screencroprect(gr, wide, high, &sdlr))
            SDL_UpdateRect(screen, sdlr.x, sdlr.y, sdlr.w, sdlr.h);
#endif /* IS_SDLv1 */
    }
    else {
        PyObject *seq;
        PyObject *r;
        Py_ssize_t loop, num;
        int count;
        SDL_Rect *rects;
        if (PyTuple_Size(arg) != 1)
            return RAISE(
                PyExc_ValueError,
                "update requires a rectstyle or sequence of recstyles");
        seq = PyTuple_GET_ITEM(arg, 0);
        if (!seq || !PySequence_Check(seq))
            return RAISE(
                PyExc_ValueError,
                "update requires a rectstyle or sequence of recstyles");

        num = PySequence_Length(seq);
        rects = PyMem_New(SDL_Rect, num);
        if (!rects)
            return NULL;
        count = 0;
        for (loop = 0; loop < num; ++loop) {
            SDL_Rect *cur_rect = (rects + count);

            /*get rect from the sequence*/
            r = PySequence_GetItem(seq, loop);
            if (r == Py_None) {
                Py_DECREF(r);
                continue;
            }
            gr = pgRect_FromObject(r, &temp);
            Py_XDECREF(r);
            if (!gr) {
                PyMem_Free((char *)rects);
                return RAISE(PyExc_ValueError,
                             "update_rects requires a single list of rects");
            }

            if (gr->w < 1 && gr->h < 1)
                continue;

            /*bail out if rect not onscreen*/
            if (!pg_screencroprect(gr, wide, high, cur_rect))
                continue;

            ++count;
        }

        if (count) {
#if IS_SDLv1
            Py_BEGIN_ALLOW_THREADS;
            SDL_UpdateRects(screen, count, rects);
            Py_END_ALLOW_THREADS;
#else  /* IS_SDLv2 */
            Py_BEGIN_ALLOW_THREADS;
            SDL_UpdateWindowSurfaceRects(win, rects, count);
            Py_END_ALLOW_THREADS;
#endif /* IS_SDLv2 */
        }

        PyMem_Free((char *)rects);
    }
    Py_RETURN_NONE;
}

#if IS_SDLv2
static PyObject *
pg_set_palette(PyObject *self, PyObject *args)
{
    pgSurfaceObject *surface = pg_GetDefaultWindowSurface();
    SDL_Surface *surf;
    SDL_Palette *pal;
    SDL_Color *colors;
    PyObject *list, *item = NULL;
    int i, len;
    int r, g, b;

    VIDEO_INIT_CHECK();
    if (!PyArg_ParseTuple(args, "|O", &list))
        return NULL;
    if (!surface)
        return RAISE(pgExc_SDLError, "No display mode is set");
    Py_INCREF(surface);
    surf = pgSurface_AsSurface(surface);
    pal = surf->format->palette;
    if (surf->format->BytesPerPixel != 1 || !pal) {
        Py_DECREF(surface);
        return RAISE(pgExc_SDLError, "Display mode is not colormapped");
    }

    if (!list) {
        Py_DECREF(surface);
        Py_RETURN_NONE;
    }

    if (!PySequence_Check(list)) {
        Py_DECREF(surface);
        return RAISE(PyExc_ValueError, "Argument must be a sequence type");
    }

    len = MIN(pal->ncolors, PySequence_Length(list));

    colors = (SDL_Color *)malloc(len * sizeof(SDL_Color));
    if (!colors) {
        Py_DECREF(surface);
        return PyErr_NoMemory();
    }

    for (i = 0; i < len; i++) {
        item = PySequence_GetItem(list, i);
        if (!PySequence_Check(item) || PySequence_Length(item) != 3) {
            Py_DECREF(item);
            free((char *)colors);
            Py_DECREF(surface);
            return RAISE(PyExc_TypeError,
                         "takes a sequence of sequence of RGB");
        }
        if (!pg_IntFromObjIndex(item, 0, &r) ||
            !pg_IntFromObjIndex(item, 1, &g) ||
            !pg_IntFromObjIndex(item, 2, &b)) {
            Py_DECREF(item);
            free((char *)colors);
            Py_DECREF(surface);
            return RAISE(PyExc_TypeError,
                         "RGB sequence must contain numeric values");
        }

        colors[i].r = (unsigned char)r;
        colors[i].g = (unsigned char)g;
        colors[i].b = (unsigned char)b;
        colors[i].a = SDL_ALPHA_OPAQUE;

        Py_DECREF(item);
    }

    pal = SDL_AllocPalette(len);
    if (!pal) {
        free((char *)colors);
        Py_DECREF(surface);
        return RAISE(pgExc_SDLError, SDL_GetError());
    }
    if (!SDL_SetPaletteColors(pal, colors, 0, len)) {
        SDL_FreePalette(pal);
        free((char *)colors);
        Py_DECREF(surface);
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    SDL_SetSurfacePalette(surf, pal);
    SDL_FreePalette(pal);
    free((char *)colors);
    Py_DECREF(surface);
    Py_RETURN_NONE;
}

static PyObject *
pg_set_gamma(PyObject *self, PyObject *arg)
{
    float r, g, b;
    int result = 0;
    _DisplayState *state = DISPLAY_MOD_STATE(self);
    SDL_Window *win = pg_GetDefaultWindow();
    Uint16 *gamma_ramp;

    if (!PyArg_ParseTuple(arg, "f|ff", &r, &g, &b))
        return NULL;
    if (PyTuple_Size(arg) == 1)
        g = b = r;
    VIDEO_INIT_CHECK();

    gamma_ramp = (Uint16 *)malloc((3 * 256) * sizeof(Uint16));
    if (!gamma_ramp)
        return PyErr_NoMemory();
    SDL_CalculateGammaRamp(r, gamma_ramp);
    SDL_CalculateGammaRamp(g, gamma_ramp + 256);
    SDL_CalculateGammaRamp(b, gamma_ramp + 256);
    if (win) {
        result = SDL_SetWindowGammaRamp(win, gamma_ramp, gamma_ramp + 256,
                                        gamma_ramp + 512);
        if (result) {
            /* Discard a possibly faulty gamma ramp */
            free(gamma_ramp);
            gamma_ramp = NULL;
        }
    }
    if (gamma_ramp) {
        if (state->gamma_ramp)
            free(state->gamma_ramp);
        state->gamma_ramp = gamma_ramp;
    }
    return PyBool_FromLong(result == 0);
}

#else  /* IS_SDLv1 */
static PyObject *
pg_set_palette(PyObject *self, PyObject *args)
{
    SDL_Surface *surf;
    SDL_Palette *pal;
    SDL_Color *colors;
    PyObject *list, *item = NULL;
    int i, len;
    int r, g, b;

    VIDEO_INIT_CHECK();
    if (!PyArg_ParseTuple(args, "|O", &list))
        return NULL;
    surf = SDL_GetVideoSurface();
    if (!surf)
        return RAISE(pgExc_SDLError, "No display mode is set");
    pal = surf->format->palette;
    if (surf->format->BytesPerPixel != 1 || !pal)
        return RAISE(pgExc_SDLError, "Display mode is not colormapped");

    if (!list) {
        colors = pal->colors;
        len = pal->ncolors;
        SDL_SetPalette(surf, SDL_PHYSPAL, colors, 0, len);
        Py_RETURN_NONE;
    }

    if (!PySequence_Check(list))
        return RAISE(PyExc_ValueError, "Argument must be a sequence type");

    len = MIN(pal->ncolors, PySequence_Length(list));

    colors = (SDL_Color *)malloc(len * sizeof(SDL_Color));
    if (!colors)
        return NULL;

    for (i = 0; i < len; i++) {
        item = PySequence_GetItem(list, i);
        if (!PySequence_Check(item) || PySequence_Length(item) != 3) {
            Py_DECREF(item);
            free((char *)colors);
            return RAISE(PyExc_TypeError,
                         "takes a sequence of sequence of RGB");
        }
        if (!pg_IntFromObjIndex(item, 0, &r) ||
            !pg_IntFromObjIndex(item, 1, &g) ||
            !pg_IntFromObjIndex(item, 2, &b)) {
            Py_DECREF(item);
            free((char *)colors);
            return RAISE(PyExc_TypeError,
                         "RGB sequence must contain numeric values");
        }

        colors[i].r = (unsigned char)r;
        colors[i].g = (unsigned char)g;
        colors[i].b = (unsigned char)b;

        Py_DECREF(item);
    }

    SDL_SetPalette(surf, SDL_PHYSPAL, colors, 0, len);

    free((char *)colors);
    Py_RETURN_NONE;
}

static PyObject *
pg_set_gamma(PyObject *self, PyObject *arg)
{
    float r, g, b;
    int result = 0;
    if (!PyArg_ParseTuple(arg, "f|ff", &r, &g, &b))
        return NULL;
    if (PyTuple_Size(arg) == 1)
        g = b = r;
    VIDEO_INIT_CHECK();
    result = SDL_SetGamma(r, g, b);
    return PyBool_FromLong(result == 0);
}
#endif /* IS_SDLv1 */

static int
pg_convert_to_uint16(PyObject *python_array, Uint16 *c_uint16_array)
{
    int i;
    PyObject *item;

    if (!c_uint16_array) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Memory not allocated for c_uint16_array.");
        return 0;
    }
    if (!PySequence_Check(python_array)) {
        PyErr_SetString(PyExc_TypeError, "Array must be sequence type");
        return 0;
    }
    if (PySequence_Size(python_array) != 256) {
        PyErr_SetString(PyExc_ValueError,
                        "gamma ramp must be 256 elements long");
        return 0;
    }
    for (i = 0; i < 256; i++) {
        item = PySequence_GetItem(python_array, i);
        if (!PyInt_Check(item)) {
            PyErr_SetString(PyExc_ValueError,
                            "gamma ramp must contain integer elements");
            return 0;
        }
        c_uint16_array[i] = (Uint16)PyInt_AsLong(item);
        Py_XDECREF(item);
    }
    return 1;
}

#if IS_SDLv2
static PyObject *
pg_set_gamma_ramp(PyObject *self, PyObject *arg)
{
    _DisplayState *state = DISPLAY_MOD_STATE(self);
    SDL_Window *win = pg_GetDefaultWindow();
    Uint16 *gamma_ramp = (Uint16 *)malloc((3 * 256) * sizeof(Uint16));
    Uint16 *r, *g, *b;
    int result = 0;
    if (!gamma_ramp)
        return PyErr_NoMemory();
    r = gamma_ramp;
    g = gamma_ramp + 256;
    b = gamma_ramp + 512;
    if (!PyArg_ParseTuple(arg, "O&O&O&", pg_convert_to_uint16, r,
                          pg_convert_to_uint16, g, pg_convert_to_uint16, b)) {
        free(gamma_ramp);
        return NULL;
    }
    VIDEO_INIT_CHECK();
    if (win) {
        result = SDL_SetWindowGammaRamp(win, gamma_ramp, gamma_ramp + 256,
                                        gamma_ramp + 512);
        if (result) {
            /* Discard a possibly faulty gamma ramp */
            free(gamma_ramp);
            gamma_ramp = NULL;
        }
    }
    if (gamma_ramp) {
        if (state->gamma_ramp)
            free(state->gamma_ramp);
        state->gamma_ramp = gamma_ramp;
    }
    return PyBool_FromLong(result == 0);
}

static PyObject *
pg_set_caption(PyObject *self, PyObject *arg)
{
    _DisplayState *state = DISPLAY_MOD_STATE(self);
    SDL_Window *win = pg_GetDefaultWindow();
    char *title, *icontitle = NULL;
    if (!PyArg_ParseTuple(arg, "es|es", "UTF-8", &title, "UTF-8", &icontitle))
        return NULL;

    if (state->title)
        free(state->title);
    state->title = (char *)malloc((strlen(title) + 1) * sizeof(char));
    if (!state->title) {
        PyErr_NoMemory();
        goto error;
    }
    strcpy(state->title, title);
    if (win)
        SDL_SetWindowTitle(win, title);
    /* TODO: icon title? */

    PyMem_Free(title);
    PyMem_Free(icontitle);
    Py_RETURN_NONE;

error:
    PyMem_Free(title);
    PyMem_Free(icontitle);
    return NULL;
}

static PyObject *
pg_get_caption(PyObject *self, PyObject *args)
{
    _DisplayState *state = DISPLAY_MOD_STATE(self);
    SDL_Window *win = pg_GetDefaultWindow();
    const char *title = win ? SDL_GetWindowTitle(win) : state->title;

    if (title && *title) {
        PyObject *titleObj = Text_FromUTF8(title);
        PyObject *ret = PyTuple_Pack(2, titleObj, titleObj);
        Py_DECREF(titleObj);
        /* TODO: icon title? */
        return ret;
    }
    return PyTuple_New(0);
}

static PyObject *
pg_set_icon(PyObject *self, PyObject *arg)
{
    _DisplayState *state = DISPLAY_MOD_STATE(self);
    SDL_Window *win = pg_GetDefaultWindow();
    PyObject *surface;
    if (!PyArg_ParseTuple(arg, "O!", &pgSurface_Type, &surface))
        return NULL;
    if (!pgVideo_AutoInit())
        return RAISE(pgExc_SDLError, SDL_GetError());
    Py_INCREF(surface);
    Py_XDECREF(state->icon);
    state->icon = surface;
    if (win)
        SDL_SetWindowIcon(win, pgSurface_AsSurface(surface));
    Py_RETURN_NONE;
}

static PyObject *
pg_iconify(PyObject *self, PyObject *args)
{
    SDL_Window *win = pg_GetDefaultWindow();
    VIDEO_INIT_CHECK();
    if (!win)
        return RAISE(pgExc_SDLError, "No open window");
    SDL_MinimizeWindow(win);
#pragma PG_WARN(Does this send the app an SDL_ActiveEvent loss event ?)
    return PyBool_FromLong(1);
}

/* This is only here for debugging purposes. Games should not rely on the
 * implementation details of specific renderers, only on the documented
 * behaviour of SDL_Renderer. It's fine to debug-print which renderer a game is
 * running on, or to inform the user when the game is not running with HW
 * acceleration, but openGL can still be available without HW acceleration. */
static PyObject *
pg_get_scaled_renderer_info(PyObject *self, PyObject *args)
{
    SDL_Window *win = pg_GetDefaultWindow();
    SDL_RendererInfo r_info;

    VIDEO_INIT_CHECK();
    if (!win)
        return RAISE(pgExc_SDLError, "No open window");

    if (pg_renderer != NULL) {
        if (SDL_GetRendererInfo(pg_renderer, &r_info) == 0) {
            return PyTuple_Pack(2, PyUnicode_FromString(r_info.name),
                                PyLong_FromLong(r_info.flags));
        }
        else {
            Py_RETURN_NONE;
        }
    }
    else {
        Py_RETURN_NONE;
    }
}

static PyObject *
pg_get_desktop_screen_sizes(PyObject *self, PyObject *args)
{
    int display_count, i;
    SDL_DisplayMode dm;
    PyObject *result;

    VIDEO_INIT_CHECK();

    display_count = SDL_GetNumVideoDisplays();

    result = PyList_New(display_count);
    if (result == NULL) {
        Py_RETURN_NONE;
    }
    for (i = 0; i < display_count; i++) {
        if (SDL_GetDesktopDisplayMode(i, &dm) != 0) {
            Py_RETURN_NONE;
        }
        if (PyList_SetItem(result, i,
                           PyTuple_Pack(2, PyLong_FromLong(dm.w),
                                        PyLong_FromLong(dm.h))) != 0) {
            Py_RETURN_NONE;
        }
    }
    return result;
}

static PyObject *
pg_is_fullscreen(PyObject *self, PyObject *args)
{
    SDL_Window *win = pg_GetDefaultWindow();
    int flags;

    VIDEO_INIT_CHECK();
    if (!win)
        return RAISE(pgExc_SDLError, "No open window");

    flags = SDL_GetWindowFlags(win) & SDL_WINDOW_FULLSCREEN_DESKTOP;

    if (flags & SDL_WINDOW_FULLSCREEN)
        Py_RETURN_TRUE;
    else
        Py_RETURN_FALSE;
}

static PyObject *
pg_toggle_fullscreen(PyObject *self, PyObject *args)
{
    SDL_Window *win = pg_GetDefaultWindow();
    int result, flags;
    int window_w, window_h, w, h, window_display;
    SDL_DisplayMode display_mode;
    pgSurfaceObject *display_surface;
    _DisplayState *state = DISPLAY_MOD_STATE(self);
    GL_glViewport_Func p_glViewport = NULL;
    SDL_SysWMinfo wm_info;
    SDL_RendererInfo r_info;

    VIDEO_INIT_CHECK();
    if (!win)
        return RAISE(pgExc_SDLError, "No open window");

    flags = SDL_GetWindowFlags(win) & SDL_WINDOW_FULLSCREEN_DESKTOP;
    /* SDL_WINDOW_FULLSCREEN_DESKTOP includes SDL_WINDOW_FULLSCREEN */

    SDL_VERSION(&wm_info.version);
    if (!SDL_GetWindowWMInfo(win, &wm_info)) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    if (state->using_gl && pg_renderer != NULL) {
        return RAISE(pgExc_SDLError,
                     "OPENGL and SDL_Renderer active at the same time");
    }

    if (pg_renderer != NULL) {
        if (SDL_GetRendererInfo(pg_renderer, &r_info) != 0) {
            return RAISE(pgExc_SDLError, SDL_GetError());
        }
    }

    switch (wm_info.subsystem) {
        // if we get this to work correctly with more systems, move them here
        case SDL_SYSWM_WINDOWS:
        case SDL_SYSWM_X11:
        case SDL_SYSWM_COCOA:
#if SDL_VERSION_ATLEAST(2, 0, 2)
        case SDL_SYSWM_WAYLAND:
#endif
            break;

            // These probably have fullscreen/windowed, but not tested yet.
            // before merge, this section should be handled by moving items
            // into the "supported" category, or returning early.

#if SDL_VERSION_ATLEAST(2, 0, 3)
        case SDL_SYSWM_WINRT:  // currently not supported by pygame?
#endif
            return PyInt_FromLong(-1);

        // On these platforms, everything is fullscreen at all times anyway
        // So we silently fail
        // In the future, add consoles like xbone/switch here
        case SDL_SYSWM_DIRECTFB:
        case SDL_SYSWM_UIKIT:  // iOS currently not supported by pygame
#if SDL_VERSION_ATLEAST(2, 0, 4)
        case SDL_SYSWM_ANDROID:  // currently not supported by pygame
#endif
            if (PyErr_WarnEx(PyExc_Warning,
                             "cannot leave FULLSCREEN on this platform",
                             1) != 0) {
                return NULL;
            }
            return PyInt_FromLong(-1);

            // Untested and unsupported platforms
#if SDL_VERSION_ATLEAST(2, 0, 2)
        case SDL_SYSWM_MIR:  // nobody uses mir any more, wayland has won
#endif
#if SDL_VERSION_ATLEAST(2, 0, 5)
        case SDL_SYSWM_VIVANTE:
#endif
        case SDL_SYSWM_UNKNOWN:
        default:
            return RAISE(pgExc_SDLError, "Unsupported platform");
    }

    display_surface = pg_GetDefaultWindowSurface();

    // could also take the size of the old display surface
    SDL_GetWindowSize(win, &window_w, &window_h);
    window_display = SDL_GetWindowDisplayIndex(win);
    if (SDL_GetDesktopDisplayMode(window_display, &display_mode) != 0) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    /*
      if (pg_renderer != NULL) {
        SDL_RenderGetLogicalSize(pg_renderer, &w, &h);
    } else
    */
    if (state->using_gl) {
        p_glViewport = (GL_glViewport_Func)SDL_GL_GetProcAddress("glViewport");
        SDL_GL_GetDrawableSize(win, &w, &h);
    }
    else {
        w = display_surface->surf->w;
        h = display_surface->surf->h;
    }

    if (flags & SDL_WINDOW_FULLSCREEN) {
        /* TOGGLE FULLSCREEN OFF */

        if (pg_renderer != NULL) {
            int scale = 1;
            int xscale, yscale;

            xscale = window_w / w;
            yscale = window_h / h;
            scale = xscale < yscale ? xscale : yscale;
            if (scale < 1) {
                scale = 1;
            }
            result = SDL_SetWindowFullscreen(win, 0);
            if (result != 0) {
                return RAISE(pgExc_SDLError, SDL_GetError());
            }
            SDL_SetWindowSize(win, w * scale, h * scale);

            if (r_info.flags & SDL_RENDERER_SOFTWARE &&
                wm_info.subsystem == SDL_SYSWM_X11) {
                /* display surface lost? */
                SDL_DestroyTexture(pg_texture);
                SDL_DestroyRenderer(pg_renderer);
                pg_renderer =
                    SDL_CreateRenderer(win, -1, SDL_RENDERER_SOFTWARE);
                pg_texture =
                    SDL_CreateTexture(pg_renderer, SDL_PIXELFORMAT_ARGB8888,
                                      SDL_TEXTUREACCESS_STREAMING, w, h);
            }
            SDL_RenderSetLogicalSize(pg_renderer, w, h);

#if (SDL_VERSION_ATLEAST(2, 0, 5))
            /* use exact integer scale in windowed mode */
            SDL_RenderSetIntegerScale(
                pg_renderer, !SDL_GetHintBoolean(
                                 "SDL_HINT_RENDER_SCALE_QUALITY", SDL_FALSE));
#endif
            SDL_SetWindowMinimumSize(win, w, h);
        }
        else if (state->using_gl) {
            /* this is literally the only place where state->toggle_windowed_w
             * should ever be read. We only use it because with GL, there is no
             * display surface we can query for dimensions. */
            result = SDL_SetWindowFullscreen(win, 0);
            if (result != 0) {
                return RAISE(pgExc_SDLError, SDL_GetError());
            }
            SDL_GL_MakeCurrent(win, state->gl_context);
            if (state->toggle_windowed_w > 0 && state->toggle_windowed_h > 0) {
                if (state->scaled_gl) {
                    float saved_aspect_ratio =
                        ((float)state->toggle_windowed_w) /
                        (float)state->toggle_windowed_h;
                    float window_aspect_ratio =
                        ((float)display_mode.w) / (float)display_mode.h;

                    if (window_aspect_ratio > saved_aspect_ratio) {
                        int width = (int)(state->toggle_windowed_h *
                                          saved_aspect_ratio);
                        p_glViewport((state->toggle_windowed_w - width) / 2, 0,
                                     width, state->toggle_windowed_h);
                    }
                    else {
                        p_glViewport(0, 0, state->toggle_windowed_w,
                                     (int)(state->toggle_windowed_w /
                                           saved_aspect_ratio));
                    }
                }
                else {
                    p_glViewport(0, 0, state->toggle_windowed_w,
                                 state->toggle_windowed_h);
                }
            }
        }
        else if ((flags & SDL_WINDOW_FULLSCREEN_DESKTOP) ==
                 SDL_WINDOW_FULLSCREEN_DESKTOP) {
            result = SDL_SetWindowFullscreen(win, 0);
            if (result != 0) {
                return RAISE(pgExc_SDLError, SDL_GetError());
            }
            display_surface->surf = SDL_GetWindowSurface(win);
        }
        else if (wm_info.subsystem == SDL_SYSWM_X11) {
            /* This is a HACK, specifically to work around faulty behaviour of
             * SDL_SetWindowFullscreen on X11 when switching out of fullscreen
             * would change the physical resolution of the display back to the
             * desktop resolution in SDL 2.0.8 (unsure about other versions).
             * The display surface gets messed up, so we re-create the window.
             * This is only relevant in the non-GL case. */
            int wx = SDL_WINDOWPOS_UNDEFINED_DISPLAY(window_display);
            int wy = SDL_WINDOWPOS_UNDEFINED_DISPLAY(window_display);
            if (PyErr_WarnEx(PyExc_Warning,
                             "re-creating window in toggle_fullscreen",
                             1) != 0) {
                return NULL;
            }
            win = SDL_CreateWindow(state->title, wx, wy, w, h, 0);
            if (win == NULL) {
                return RAISE(pgExc_SDLError, SDL_GetError());
            }
            else {
                result = 0;
            }
            display_surface->surf = SDL_GetWindowSurface(win);
            pg_SetDefaultWindow(win);
        }
        else {
            result = SDL_SetWindowFullscreen(win, 0);
            if (result != 0) {
                return RAISE(pgExc_SDLError, SDL_GetError());
            }
            display_surface->surf = SDL_GetWindowSurface(win);
        }
        state->toggle_windowed_w = 0;
        state->toggle_windowed_h = 0;
    }
    else {
        /* TOGGLE FULLSCREEN ON */

        state->toggle_windowed_w = w;
        state->toggle_windowed_h = h;
        if (pg_renderer != NULL) {
            result =
                SDL_SetWindowFullscreen(win, SDL_WINDOW_FULLSCREEN_DESKTOP);
            if (result != 0) {
                return RAISE(pgExc_SDLError, SDL_GetError());
            }
            if (r_info.flags & SDL_RENDERER_SOFTWARE &&
                wm_info.subsystem == SDL_SYSWM_X11) {
                if (PyErr_WarnEx(
                        PyExc_Warning,
                        "recreating software renderer in toggle_fullscreen",
                        1) != 0) {
                    return NULL;
                }
                /* display surface lost? only on x11? */
                SDL_DestroyTexture(pg_texture);
                SDL_DestroyRenderer(pg_renderer);
                pg_renderer =
                    SDL_CreateRenderer(win, -1, SDL_RENDERER_SOFTWARE);
                pg_texture =
                    SDL_CreateTexture(pg_renderer, SDL_PIXELFORMAT_ARGB8888,
                                      SDL_TEXTUREACCESS_STREAMING, w, h);
            }

            SDL_RenderSetLogicalSize(pg_renderer, w, h);
#if (SDL_VERSION_ATLEAST(2, 0, 5))
            SDL_RenderSetIntegerScale(pg_renderer, SDL_FALSE);
#endif
        }
        else if (state->using_gl) {
            result =
                SDL_SetWindowFullscreen(win, SDL_WINDOW_FULLSCREEN_DESKTOP);
            if (result != 0) {
                return RAISE(pgExc_SDLError, SDL_GetError());
            }
            SDL_GL_MakeCurrent(win, state->gl_context);
            if (state->scaled_gl) {
                float saved_aspect_ratio =
                    ((float)state->scaled_gl_w) / (float)state->scaled_gl_h;
                float window_aspect_ratio =
                    ((float)display_mode.w) / (float)display_mode.h;

                if (window_aspect_ratio > saved_aspect_ratio) {
                    int width = (int)(display_mode.h * saved_aspect_ratio);
                    p_glViewport((display_mode.w - width) / 2, 0, width,
                                 display_mode.h);
                }
                else {
                    p_glViewport(0, 0, display_mode.w,
                                 (int)(display_mode.w / saved_aspect_ratio));
                }
            }
            else {
                p_glViewport(0, 0, display_mode.w, display_mode.h);
            }
        }
        else if (w == display_mode.w && h == display_mode.h) {
            result =
                SDL_SetWindowFullscreen(win, SDL_WINDOW_FULLSCREEN_DESKTOP);
            if (result != 0) {
                return RAISE(pgExc_SDLError, SDL_GetError());
            }
            display_surface->surf = SDL_GetWindowSurface(win);
        }
        else if (wm_info.subsystem == SDL_SYSWM_WAYLAND) {
            if (PyErr_WarnEx(PyExc_Warning,
                             "skipping toggle_fullscreen on wayland",
                             1) != 0) {
                return NULL;
            }
            return PyInt_FromLong(-1);
        }
        else {
            result = SDL_SetWindowFullscreen(win, SDL_WINDOW_FULLSCREEN);
            if (result != 0) {
                return RAISE(pgExc_SDLError, SDL_GetError());
            }
            display_surface->surf = SDL_GetWindowSurface(win);
            if (w != display_surface->surf->w ||
                h != display_surface->surf->h) {
                int wx = SDL_WINDOWPOS_UNDEFINED_DISPLAY(window_display);
                int wy = SDL_WINDOWPOS_UNDEFINED_DISPLAY(window_display);
                win = SDL_CreateWindow(state->title, wx, wy, w, h, 0);
                if (win == NULL) {
                    return RAISE(pgExc_SDLError, SDL_GetError());
                }
                display_surface->surf = SDL_GetWindowSurface(win);
                pg_SetDefaultWindow(win);
                if (PyErr_WarnEx(PyExc_Warning,
                                 "re-creating window in toggle_fullscreen",
                                 1) != 0) {
                    return NULL;
                }
                return PyInt_FromLong(-1);
            }
        }
    }
    return PyInt_FromLong(result != 0);
}

/* This API is provisional, and, not finalised, and should not be documented
 * in any user-facing docs until we are sure when this is safe to call and when
 * it should raise an exception */
static PyObject *
pg_display_resize_event(PyObject *self, PyObject *event)
{
    /* Call this from your game if you want to use RESIZABLE with SCALED
     * TODO: Document, handle bad args, bail on FULLSCREEN
     */
    int wnew = PyLong_AsLong(PyObject_GetAttrString(event, "w"));
    int hnew = PyLong_AsLong(PyObject_GetAttrString(event, "h"));
    SDL_Window *win = pg_GetDefaultWindow();
    int flags;
    int window_w, window_h, w, h, window_display, result;
    SDL_DisplayMode display_mode;
    _DisplayState *state = DISPLAY_MOD_STATE(self);
    GL_glViewport_Func p_glViewport = NULL;

    VIDEO_INIT_CHECK();
    if (!win)
        return RAISE(pgExc_SDLError, "No open window");

    flags = SDL_GetWindowFlags(win) &
            (SDL_WINDOW_FULLSCREEN | SDL_WINDOW_FULLSCREEN_DESKTOP);

    if (flags) {
        return PyInt_FromLong(-1);
    }

    // could also take the size of the old display surface
    SDL_GetWindowSize(win, &window_w, &window_h);
    window_display = SDL_GetWindowDisplayIndex(win);
    if (SDL_GetDesktopDisplayMode(window_display, &display_mode) != 0) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    if (state->using_gl) {
        p_glViewport = (GL_glViewport_Func)SDL_GL_GetProcAddress("glViewport");
        SDL_SetWindowSize(win, wnew, hnew);
        SDL_GL_MakeCurrent(win, state->gl_context);
        if (state->scaled_gl) {
            float saved_aspect_ratio =
                ((float)state->scaled_gl_w) / (float)state->scaled_gl_h;
            float window_aspect_ratio = ((float)wnew) / (float)hnew;

            if (window_aspect_ratio > saved_aspect_ratio) {
                int width = (int)(hnew * saved_aspect_ratio);
                p_glViewport((wnew - width) / 2, 0, width, hnew);
            }
            else {
                p_glViewport(0, 0, wnew, (int)(wnew / saved_aspect_ratio));
            }
        }
        else {
            p_glViewport(0, 0, wnew, hnew);
        }
    }
    else if (pg_renderer != NULL) {
        SDL_RenderGetLogicalSize(pg_renderer, &w, &h);
        SDL_SetWindowSize(win, (w > wnew) ? w : wnew, (h > hnew) ? h : hnew);
        result = SDL_RenderSetLogicalSize(pg_renderer, w, h);
        if (result != 0) {
            return RAISE(pgExc_SDLError, SDL_GetError());
        }
    }
    else {
        /* do not do anything that would invalidate a display surface! */
        return PyInt_FromLong(-1);
    }
    return PyInt_FromLong(0);
}

#else  /* IS_SDLv1 */
static PyObject *
pg_set_gamma_ramp(PyObject *self, PyObject *arg)
{
    Uint16 *r, *g, *b;
    int result;
    r = (Uint16 *)malloc(256 * sizeof(Uint16));
    if (!r)
        return NULL;
    g = (Uint16 *)malloc(256 * sizeof(Uint16));
    if (!g) {
        free(r);
        return NULL;
    }
    b = (Uint16 *)malloc(256 * sizeof(Uint16));
    if (!b) {
        free(r);
        free(g);
        return NULL;
    }
    if (!PyArg_ParseTuple(arg, "O&O&O&", pg_convert_to_uint16, r,
                          pg_convert_to_uint16, g, pg_convert_to_uint16, b)) {
        free(r);
        free(g);
        free(b);
        return NULL;
    }
    VIDEO_INIT_CHECK();
    result = SDL_SetGammaRamp(r, g, b);
    free((char *)r);
    free((char *)g);
    free((char *)b);
    return PyBool_FromLong(result == 0);
}

static PyObject *
pg_set_caption(PyObject *self, PyObject *arg)
{
    char *title, *icontitle = NULL;
    if (!PyArg_ParseTuple(arg, "es|es", "UTF-8", &title, "UTF-8", &icontitle))
        return NULL;
    SDL_WM_SetCaption(title, icontitle ? icontitle : title);
    PyMem_Free(title);
    PyMem_Free(icontitle);
    Py_RETURN_NONE;
}

static PyObject *
pg_get_caption(PyObject *self, PyObject *args)
{
    char *title, *icontitle;
    SDL_WM_GetCaption(&title, &icontitle);
    if (title && *title) {
        PyObject *titleObj = Text_FromUTF8(title);
        PyObject *iconObj = Text_FromUTF8(icontitle);
        PyObject *ret = PyTuple_Pack(2, titleObj, iconObj);
        Py_DECREF(titleObj);
        Py_DECREF(iconObj);
        return ret;
    }
    return PyTuple_New(0);
}

static void
pg_do_set_icon(PyObject *surface)
{
    SDL_Surface *surf = pgSurface_AsSurface(surface);
    SDL_WM_SetIcon(surf, NULL);
    icon_was_set = 1;
}

static PyObject *
pg_set_icon(PyObject *self, PyObject *arg)
{
    PyObject *surface;
    if (!PyArg_ParseTuple(arg, "O!", &pgSurface_Type, &surface))
        return NULL;
    if (!pgVideo_AutoInit())
        return RAISE(pgExc_SDLError, SDL_GetError());
    pg_do_set_icon(surface);
    Py_RETURN_NONE;
}

static PyObject *
pg_iconify(PyObject *self, PyObject *args)
{
    int result;
    VIDEO_INIT_CHECK();
    result = SDL_WM_IconifyWindow();

    /* If the application is running in a window managed environment SDL
       attempts to iconify/minimise it. If SDL_WM_IconifyWindow is successful,
       the application will receive a SDL_APPACTIVE loss event (see
       SDL_ActiveEvent).
    */
    return PyInt_FromLong(result != 0);
}

static PyObject *
pg_toggle_fullscreen(PyObject *self, PyObject *args)
{
    SDL_Surface *screen;
    int result;
    VIDEO_INIT_CHECK();
    screen = SDL_GetVideoSurface();
    if (!screen)
        return RAISE(pgExc_SDLError, SDL_GetError());

    result = SDL_WM_ToggleFullScreen(screen);
    return PyInt_FromLong(result != 0);
}
#endif /* IS_SDLv1 */


static PyObject *
pg_get_allow_screensaver(PyObject *self) {
    /* SDL_IsScreenSaverEnabled() unconditionally returns SDL_True if
     * the video system is not initialized.  Therefore we insist on
     * the video being initialized before calling it.
     */
   VIDEO_INIT_CHECK();
#if IS_SDLv2
    return PyBool_FromLong(SDL_IsScreenSaverEnabled() == SDL_TRUE);
#else /* IS_SDLv1*/
    return PyBool_FromLong(_allow_screensaver);
#endif /* IS_SDLv1*/
}

static PyObject *
pg_set_allow_screensaver(PyObject *self, PyObject *arg, PyObject *kwargs) {
    int val = 1;
    static char *keywords[] = {"value", NULL};

    if (!PyArg_ParseTupleAndKeywords(arg, kwargs, "|i", keywords, &val)) {
        return NULL;
    }

    VIDEO_INIT_CHECK();
    if (val) {
#if IS_SDLv2
        SDL_EnableScreenSaver();
#else
    _allow_screensaver = 1;
#endif
    } else {
#if IS_SDLv2
        SDL_DisableScreenSaver();
#else
        _allow_screensaver = 0;
#endif
    }

    Py_RETURN_NONE;
}

static PyMethodDef _pg_display_methods[] = {
    {"__PYGAMEinit__", pg_display_autoinit, 1,
     "auto initialize function for display."},
    {"init", pg_init, METH_NOARGS, DOC_PYGAMEDISPLAYINIT},
    {"quit", pg_quit, METH_NOARGS, DOC_PYGAMEDISPLAYQUIT},
    {"get_init", pg_get_init, METH_NOARGS, DOC_PYGAMEDISPLAYGETINIT},
    {"get_active", pg_get_active, METH_NOARGS, DOC_PYGAMEDISPLAYGETACTIVE},

    /*    { "set_driver", set_driver, 1, doc_set_driver },*/
    {"get_driver", pg_get_driver, METH_NOARGS, DOC_PYGAMEDISPLAYGETDRIVER},
    {"get_wm_info", pg_get_wm_info, METH_NOARGS, DOC_PYGAMEDISPLAYGETWMINFO},
    {"Info", pgInfo, METH_NOARGS, DOC_PYGAMEDISPLAYINFO},
    {"get_surface", pg_get_surface, METH_NOARGS, DOC_PYGAMEDISPLAYGETSURFACE},
    {"get_window_size", pg_window_size, METH_NOARGS,
     DOC_PYGAMEDISPLAYGETWINDOWSIZE},

    {"set_mode", (PyCFunction)pg_set_mode, METH_VARARGS | METH_KEYWORDS,
     DOC_PYGAMEDISPLAYSETMODE},
    {"mode_ok", (PyCFunction)pg_mode_ok, METH_VARARGS | METH_KEYWORDS,
     DOC_PYGAMEDISPLAYMODEOK},
    {"list_modes", (PyCFunction)pg_list_modes, METH_VARARGS | METH_KEYWORDS,
     DOC_PYGAMEDISPLAYLISTMODES},
    {"get_num_displays", pg_num_displays, METH_NOARGS,
     DOC_PYGAMEDISPLAYGETNUMDISPLAYS},

    {"flip", pg_flip, METH_NOARGS, DOC_PYGAMEDISPLAYFLIP},
    {"update", pg_update, METH_VARARGS, DOC_PYGAMEDISPLAYUPDATE},

    {"set_palette", pg_set_palette, METH_VARARGS, DOC_PYGAMEDISPLAYSETPALETTE},
    {"set_gamma", pg_set_gamma, METH_VARARGS, DOC_PYGAMEDISPLAYSETGAMMA},
    {"set_gamma_ramp", pg_set_gamma_ramp, METH_VARARGS,
     DOC_PYGAMEDISPLAYSETGAMMARAMP},

    {"set_caption", pg_set_caption, METH_VARARGS, DOC_PYGAMEDISPLAYSETCAPTION},
    {"get_caption", pg_get_caption, METH_NOARGS, DOC_PYGAMEDISPLAYGETCAPTION},
    {"set_icon", pg_set_icon, METH_VARARGS, DOC_PYGAMEDISPLAYSETICON},

    {"iconify", pg_iconify, METH_NOARGS, DOC_PYGAMEDISPLAYICONIFY},
    {"toggle_fullscreen", pg_toggle_fullscreen, METH_NOARGS,
     DOC_PYGAMEDISPLAYTOGGLEFULLSCREEN},

#if IS_SDLv2
    {"_set_autoresize", (PyCFunction)pg_display_set_autoresize
     , METH_VARARGS, "provisional API, subject to change"},
    {"_resize_event", (PyCFunction)pg_display_resize_event, METH_O,
     "provisional API, subject to change"},
    {"_get_renderer_info", (PyCFunction)pg_get_scaled_renderer_info,
     METH_NOARGS, "provisional API, subject to change"},
    {"get_desktop_sizes", (PyCFunction)pg_get_desktop_screen_sizes,
     METH_NOARGS, "provisional API, subject to change"},
    {"is_fullscreen", (PyCFunction)pg_is_fullscreen, METH_NOARGS,
     "provisional API, subject to change"},
#endif

    {"gl_set_attribute", pg_gl_set_attribute, METH_VARARGS,
     DOC_PYGAMEDISPLAYGLSETATTRIBUTE},
    {"gl_get_attribute", pg_gl_get_attribute, METH_VARARGS,
     DOC_PYGAMEDISPLAYGLGETATTRIBUTE},

    {"get_allow_screensaver", (PyCFunction)pg_get_allow_screensaver, METH_NOARGS,
     DOC_PYGAMEDISPLAYGETALLOWSCREENSAVER},
    {"set_allow_screensaver", (PyCFunction)pg_set_allow_screensaver, METH_VARARGS | METH_KEYWORDS,
     DOC_PYGAMEDISPLAYSETALLOWSCREENSAVER},

    {NULL, NULL, 0, NULL}};

#if IS_SDLv2
#if PY3
#ifndef PYPY_VERSION
static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                     "display",
                                     DOC_PYGAMEDISPLAY,
                                     sizeof(_DisplayState),
                                     _pg_display_methods,
#pragma PG_WARN(At some point should add GC slot functions.)
                                     NULL,
                                     NULL,
                                     NULL,
                                     NULL};
#else  /* PYPY_VERSION */
static struct PyModuleDef _module = {
    PyModuleDef_HEAD_INIT,
    "display",
    DOC_PYGAMEDISPLAY,
    -1, /* PyModule_GetState() not implemented */
    _pg_display_methods,
    NULL,
    NULL,
    NULL,
    NULL};
#endif /* PYPY_VERSION */
#endif /* PY3 */

MODINIT_DEFINE(display)
{
    PyObject *module;
    _DisplayState *state;

    /* imported needed apis; Do this first so if there is an error
       the module is not loaded.
    */
    import_pygame_base();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }
    import_pygame_rect();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }
    import_pygame_surface();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }

    /* type preparation */
    if (PyType_Ready(&pgVidInfo_Type) < 0) {
        MODINIT_ERROR;
    }

    /* create the module */
#if PY3
    module = PyModule_Create(&_module);
#else
    module = Py_InitModule3(MODPREFIX "display", _pg_display_methods,
                            DOC_PYGAMEDISPLAY);
#endif
    if (module == NULL) {
        MODINIT_ERROR;
    }
    state = DISPLAY_MOD_STATE(module);
    state->title = NULL;
    state->icon = NULL;
    state->gamma_ramp = NULL;
    state->using_gl = 0;
    state->auto_resize = SDL_TRUE;

    MODINIT_RETURN(module);
}
#else /* IF_SDLv1 */

MODINIT_DEFINE(display)
{
    PyObject *module, *dict, *apiobj;
    int ecode;
    static void *c_api[PYGAMEAPI_DISPLAY_NUMSLOTS];
#if PY3
    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "display",
                                         DOC_PYGAMEDISPLAY,
                                         -1,
                                         _pg_display_methods,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL};
#endif /* PY3 */

    /* imported needed apis; Do this first so if there is an error
       the module is not loaded.
    */
    import_pygame_base();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }
    import_pygame_rect();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }
    import_pygame_surface();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }

    /* type preparation */
    if (PyType_Ready(&pgVidInfo_Type) < 0) {
        MODINIT_ERROR;
    }

    /* create the module */
#if PY3
    module = PyModule_Create(&_module);
#else
    module = Py_InitModule3(MODPREFIX "display", _pg_display_methods,
                            DOC_PYGAMEDISPLAY);
#endif
    if (module == NULL) {
        MODINIT_ERROR;
    }
    dict = PyModule_GetDict(module);

    /* export the c api */
    c_api[0] = &pgVidInfo_Type;
    c_api[1] = pgVidInfo_New;
    apiobj = encapsulate_api(c_api, "display");
    if (apiobj == NULL) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    ecode = PyDict_SetItemString(dict, PYGAMEAPI_LOCAL_ENTRY, apiobj);
    Py_DECREF(apiobj);
    if (ecode) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    MODINIT_RETURN(module);
}
#endif /* IF_SDLv1 */
