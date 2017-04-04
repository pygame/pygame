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
#include "doc/display_doc.h"

typedef struct _display_state_s {
    char* title;
    PyObject* icon;
    Uint16 *gamma_ramp;
} _DisplayState;

#if PY3
static struct PyModuleDef _module = {
    PyModuleDef_HEAD_INIT,
    "display",
    DOC_PYGAMEDISPLAY,
    sizeof(_DisplayState),
    _display_methods,
#warning At some point should add GC slot functions.
    NULL, NULL, NULL, NULL
};

#define DISPLAY_MOD_STATE(mod) ((_DisplayState*)PyModule_GetState(mod))
#define DISPLAY_STATE \
    DISPLAY_MOD_STATE(PyState_FindModule(&_module))

#else
static _DisplayState _modstate;

#define DISPLAY_MOD_STATE(mod) (&_modstate)
#define DISPLAY_STATE DISPLAY_MOD_STATE(0)

#endif

#if 0
static PyTypeObject PyVidInfo_Type;
static PyObject* PyVidInfo_New (const SDL_VideoInfo* info);
#endif

#if (!defined(darwin))
static char* icon_defaultname = "pygame_icon.bmp";
static char* pkgdatamodule_name = "pygame.pkgdata";
static char* imagemodule_name = "pygame.image";
static char* resourcefunc_name = "getResource";
static char* load_basicfunc_name = "load_basic";

static void
close_file (PyObject *fileobj)
{
    PyObject* result = PyObject_CallMethod (fileobj, "close", NULL);
    if (result) {
        Py_DECREF (result);
    }
    else {
        PyErr_Clear ();
    }
}

static PyObject*
display_resource (char *filename)
{
    PyObject* imagemodule = NULL;
    PyObject* load_basicfunc = NULL;
    PyObject* pkgdatamodule = NULL;
    PyObject* resourcefunc = NULL;
    PyObject* fresult = NULL;
    PyObject* result = NULL;
#if PY3
    PyObject* name = NULL;
#endif

    pkgdatamodule = PyImport_ImportModule (pkgdatamodule_name);
    if (!pkgdatamodule)
        goto display_resource_end;

    resourcefunc = PyObject_GetAttrString (pkgdatamodule, resourcefunc_name);
    if (!resourcefunc)
        goto display_resource_end;

    imagemodule = PyImport_ImportModule (imagemodule_name);
    if (!imagemodule)
        goto display_resource_end;

    load_basicfunc = PyObject_GetAttrString
        (imagemodule, load_basicfunc_name);
    if (!load_basicfunc)
        goto display_resource_end;

    fresult = PyObject_CallFunction (resourcefunc, "s", filename);
    if (!fresult)
        goto display_resource_end;

#if PY3
    name = PyObject_GetAttrString (fresult, "name");
    if (name != NULL) {
        if (Text_Check (name)) {
            close_file (fresult);
            Py_DECREF (fresult);
            fresult = name;
            name = NULL;
        }
    }
    else {
        PyErr_Clear ();
    }
#else
    if (PyFile_Check (fresult)) {
        PyObject *tmp = PyFile_Name (fresult);
        Py_INCREF (tmp);
        close_file (fresult);
        Py_DECREF (fresult);
        fresult = tmp;
    }
#endif

    result = PyObject_CallFunction (load_basicfunc, "O", fresult);
    if (!result) goto display_resource_end;

display_resource_end:
    Py_XDECREF (pkgdatamodule);
    Py_XDECREF (resourcefunc);
    Py_XDECREF (imagemodule);
    Py_XDECREF (load_basicfunc);
    Py_XDECREF (fresult);
#if PY3
    Py_XDECREF (name);
#endif
    return result;
}
#endif

/* init routines */
static void
display_autoquit (void)
{
    if (Py_GetDefaultWindowSurface ())
    {
#warning DISPLAY_STATE: Does this really work in autoquit?
        _DisplayState* state = DISPLAY_STATE;

        Py_SetDefaultWindowSurface (NULL);

        if (state->title)
        {
            free (state->title);
            state->title = NULL;
        }

        Py_XDECREF (state->icon);
        state->icon = NULL;

        if (state->gamma_ramp)
            free (state->gamma_ramp);
        state->gamma_ramp = NULL;

        Py_SetDefaultWindow (NULL);
    }
}

static PyObject*
display_autoinit (PyObject* self, PyObject* arg)
{
    PyGame_RegisterQuit (display_autoquit);
    return PyInt_FromLong (1);
}

static PyObject*
quit (PyObject* self, PyObject* arg)
{
    PyGame_Video_AutoQuit ();
    display_autoquit ();

    Py_RETURN_NONE;
}

static PyObject*
init (PyObject* self)
{
    if (!PyGame_Video_AutoInit())
        return RAISE (PyExc_SDLError, SDL_GetError ());
    if (!display_autoinit (NULL, NULL))
        return NULL;

    Py_RETURN_NONE;
}

static PyObject*
get_init (PyObject* self)
{
    return PyInt_FromLong (SDL_WasInit (SDL_INIT_VIDEO) != 0);
}

static PyObject*
get_active (PyObject* self)
{
    Uint32 flags = SDL_GetWindowFlags (Py_GetDefaultWindow ());

    return PyInt_FromLong ((flags & SDL_WINDOW_SHOWN) != 0);
}

#if 0
/* vidinfo object */
static void
vidinfo_dealloc (PyObject* self)
{
    PyObject_DEL (self);
}

static PyObject*
vidinfo_getattr (PyObject *self, char *name)
{
    SDL_VideoInfo* info = &((PyVidInfoObject*) self)->info;

    int current_w = -1;
    int current_h = -1;

    SDL_version versioninfo;
    SDL_VERSION (&versioninfo);

    if (versioninfo.major >= 1 && versioninfo.minor >= 2
        && versioninfo.patch >= 10 )
    {
        current_w = info->current_w;
        current_h = info->current_h;
    }

    if (!strcmp (name, "hw"))
        return PyInt_FromLong (info->hw_available);
    else if (!strcmp (name, "wm"))
        return PyInt_FromLong (info->wm_available);
    else if (!strcmp (name, "blit_hw"))
        return PyInt_FromLong (info->blit_hw);
    else if (!strcmp (name, "blit_hw_CC"))
        return PyInt_FromLong (info->blit_hw_CC);
    else if (!strcmp (name, "blit_hw_A"))
        return PyInt_FromLong (info->blit_hw_A);
    else if (!strcmp (name, "blit_sw"))
        return PyInt_FromLong (info->blit_hw);
    else if (!strcmp (name, "blit_sw_CC"))
        return PyInt_FromLong (info->blit_hw_CC);
    else if (!strcmp (name, "blit_sw_A"))
        return PyInt_FromLong (info->blit_hw_A);
    else if (!strcmp (name, "blit_fill"))
        return PyInt_FromLong (info->blit_fill);
    else if (!strcmp (name, "video_mem"))
        return PyInt_FromLong (info->video_mem);
    else if (!strcmp (name, "bitsize"))
        return PyInt_FromLong (info->vfmt->BitsPerPixel);
    else if (!strcmp (name, "bytesize"))
        return PyInt_FromLong (info->vfmt->BytesPerPixel);
    else if (!strcmp (name, "masks"))
        return Py_BuildValue ("(iiii)", info->vfmt->Rmask, info->vfmt->Gmask,
                              info->vfmt->Bmask, info->vfmt->Amask);
    else if (!strcmp (name, "shifts"))
        return Py_BuildValue ("(iiii)", info->vfmt->Rshift, info->vfmt->Gshift,
                              info->vfmt->Bshift, info->vfmt->Ashift);
    else if (!strcmp (name, "losses"))
        return Py_BuildValue ("(iiii)", info->vfmt->Rloss, info->vfmt->Gloss,
                              info->vfmt->Bloss, info->vfmt->Aloss);
    else if (!strcmp (name, "current_h"))
        return PyInt_FromLong (current_h);
    else if (!strcmp (name, "current_w"))
        return PyInt_FromLong (current_w);

    return RAISE (PyExc_AttributeError, "does not exist in vidinfo");
}

PyObject*
vidinfo_str (PyObject* self)
{
    char str[1024];
    int current_w = -1;
    int current_h = -1;
    SDL_VideoInfo* info = &((PyVidInfoObject*) self)->info;

    SDL_version versioninfo;
    SDL_VERSION (&versioninfo);

    if(versioninfo.major >= 1 && versioninfo.minor >= 2
       && versioninfo.patch >= 10 )
    {
        current_w = info->current_w;
        current_h = info->current_h;
    }

    sprintf (str, "<VideoInfo(hw = %d, wm = %d,video_mem = %d\n"
             "         blit_hw = %d, blit_hw_CC = %d, blit_hw_A = %d,\n"
             "         blit_sw = %d, blit_sw_CC = %d, blit_sw_A = %d,\n"
             "         bitsize  = %d, bytesize = %d,\n"
             "         masks =  (%d, %d, %d, %d),\n"
             "         shifts = (%d, %d, %d, %d),\n"
             "         losses =  (%d, %d, %d, %d),\n"
             "         current_w = %d, current_h = %d\n"
             ">\n",
             info->hw_available, info->wm_available, info->video_mem,
             info->blit_hw, info->blit_hw_CC, info->blit_hw_A,
             info->blit_sw, info->blit_sw_CC, info->blit_sw_A,
             info->vfmt->BitsPerPixel, info->vfmt->BytesPerPixel,
             info->vfmt->Rmask, info->vfmt->Gmask, info->vfmt->Bmask,
             info->vfmt->Amask,
             info->vfmt->Rshift, info->vfmt->Gshift, info->vfmt->Bshift,
             info->vfmt->Ashift,
             info->vfmt->Rloss, info->vfmt->Gloss, info->vfmt->Bloss,
             info->vfmt->Aloss,
             current_w, current_h);
    return Text_FromUTF8 (str);
}

static PyTypeObject PyVidInfo_Type =
{
    TYPE_HEAD (NULL, 0)
    "VidInfo",               /*name*/
    sizeof(PyVidInfoObject), /*basic size*/
    0,                       /*itemsize*/
    vidinfo_dealloc,         /*dealloc*/
    0,                       /*print*/
    vidinfo_getattr,         /*getattr*/
    NULL,                    /*setattr*/
    NULL,                    /*compare*/
    vidinfo_str,             /*repr*/
    NULL,                    /*as_number*/
    NULL,                    /*as_sequence*/
    NULL,                    /*as_mapping*/
    (hashfunc)NULL,          /*hash*/
    (ternaryfunc)NULL,       /*call*/
    (reprfunc)NULL,          /*str*/
};

static PyObject*
PyVidInfo_New (const SDL_VideoInfo* i)
{
    PyVidInfoObject* info;

    if (!i)
        return RAISE (PyExc_SDLError, SDL_GetError ());

    info = PyObject_NEW (PyVidInfoObject, &PyVidInfo_Type);
    if (!info)
        return NULL;

    memcpy (&info->info, i, sizeof (SDL_VideoInfo));
    return (PyObject*)info;
}
#endif /* May or may not get ported stuff. */

/* display functions */
static PyObject*
get_driver (PyObject* self)
{
    const char* name = NULL;

    VIDEO_INIT_CHECK ();

    name = SDL_GetCurrentVideoDriver ();
    if (!name)
        Py_RETURN_NONE;
    return Text_FromUTF8 (name);
}

#if 0
static PyObject*
get_wm_info (PyObject* self)
{
    PyObject *dict;
    PyObject *tmp;
    SDL_SysWMinfo info;

    VIDEO_INIT_CHECK ();

    SDL_VERSION (&(info.version))
    dict = PyDict_New ();
    if (!dict || !SDL_GetWMInfo (&info))
        return dict;

/*scary #ifdef's match SDL_syswm.h*/
#if (defined(unix) || defined(__unix__) ||              \
     defined(_AIX) || defined(__OpenBSD__)) &&          \
    (defined(SDL_VIDEO_DRIVER_X11) && !defined(__CYGWIN32__) && \
     !defined(ENABLE_NANOX) && !defined(__QNXNTO__))

    tmp = PyInt_FromLong (info.info.x11.window);
    PyDict_SetItemString (dict, "window", tmp);
    Py_DECREF (tmp);

    tmp = PyCapsule_New (info.info.x11.display, "display", NULL);
    PyDict_SetItemString (dict, "display", tmp);
    Py_DECREF (tmp);

    tmp = PyCapsule_New (info.info.x11.lock_func, "lock_func", NULL);
    PyDict_SetItemString (dict, "lock_func", tmp);
    Py_DECREF (tmp);

    tmp = PyCapsule_New (info.info.x11.unlock_func, "unlock_func", NULL);
    PyDict_SetItemString (dict, "unlock_func", tmp);
    Py_DECREF (tmp);

    tmp = PyInt_FromLong (info.info.x11.fswindow);
    PyDict_SetItemString (dict, "fswindow", tmp);
    Py_DECREF (tmp);

    tmp = PyInt_FromLong (info.info.x11.wmwindow);
    PyDict_SetItemString (dict, "wmwindow", tmp);
    Py_DECREF (tmp);

#elif defined(ENABLE_NANOX)
    tmp = PyInt_FromLong (info.window);
    PyDict_SetItemString (dict, "window", tmp);
    Py_DECREF (tmp);
#elif defined(WIN32)
    tmp = PyInt_FromLong ((long)info.window);
    PyDict_SetItemString (dict, "window", tmp);
    Py_DECREF (tmp);

    tmp = PyInt_FromLong ((long)info.hglrc);
    PyDict_SetItemString (dict, "hglrc", tmp);
    Py_DECREF (tmp);
#elif defined(__riscos__)
    tmp = PyInt_FromLong (info.window);
    PyDict_SetItemString (dict, "window", tmp);
    Py_DECREF (tmp);

    tmp = PyInt_FromLong (info.wimpVersion);
    PyDict_SetItemString (dict, "wimpVersion", tmp);
    Py_DECREF (tmp);

    tmp = PyInt_FromLong (info.taskHandle);
    PyDict_SetItemString (dict, "taskHandle", tmp);
    Py_DECREF (tmp);
#elif (defined(__APPLE__) && defined(__MACH__))
    /* do nothing. */
#else
    tmp = PyInt_FromLong (info.data);
    PyDict_SetItemString (dict, "data", tmp);
    Py_DECREF (tmp);
#endif

    return dict;
}

static PyObject*
Info (PyObject* self)
{
    const SDL_VideoInfo* info;

    VIDEO_INIT_CHECK ();

    info = SDL_GetVideoInfo ();
    return PyVidInfo_New (info);
}
#endif /* May or may not get ported stuff. */

static PyObject*
get_surface (PyObject* self)
{
    PyObject* surface = Py_GetDefaultWindowSurface ();

    if (!surface)
        Py_RETURN_NONE;

    Py_INCREF (surface);
    return surface;
}

static PyObject*
gl_set_attribute (PyObject* self, PyObject* arg)
{
    int flag, value, result;

    VIDEO_INIT_CHECK ();

    if (!PyArg_ParseTuple (arg, "ii", &flag, &value))
        return NULL;
    if (flag == -1) /*an undefined/unsupported val, ignore*/
        Py_RETURN_NONE;

    result = SDL_GL_SetAttribute (flag, value);
    if (result == -1)
        return RAISE (PyExc_SDLError, SDL_GetError());
    Py_RETURN_NONE;
}

static PyObject*
gl_get_attribute (PyObject* self, PyObject* arg)
{
    int flag, value, result;

    VIDEO_INIT_CHECK ();

    if (!PyArg_ParseTuple (arg, "i", &flag))
        return NULL;

    result = SDL_GL_GetAttribute (flag, &value);
    if (result == -1)
        return RAISE (PyExc_SDLError, SDL_GetError ());

    return PyInt_FromLong (value);
}

static PyObject*
set_mode (PyObject* self, PyObject* arg)
{
    static const char* const DefaultTitle = "pygame window";

    _DisplayState* state = DISPLAY_MOD_STATE (self);
    SDL_Window* win = Py_GetDefaultWindow ();
    PyObject* surface = Py_GetDefaultWindowSurface ();
    SDL_Surface* surf = NULL;
    int depth = 0;
    int flags = 0;
    int w = 0;
    int h = 0;
    char *title = state->title;

    if (!PyArg_ParseTuple (arg, "|(ii)ii", &w, &h, &flags, &depth))
        return NULL;

    if (w < 0 || h < 0)
        return RAISE (PyExc_SDLError, "Cannot set negative sized display mode");

    if (w == 0 || h == 0)
    {
        SDL_version versioninfo;
        SDL_VERSION (&versioninfo);
        if (!(versioninfo.major != 1 ||
             (versioninfo.major == 1 && versioninfo.minor > 2) ||
             (versioninfo.major == 1 && versioninfo.minor == 2 && versioninfo.patch >= 10 )))
        {
            return RAISE (PyExc_SDLError, "Cannot set 0 sized display mode");
        }
    }

    if (!SDL_WasInit (SDL_INIT_VIDEO))
    {
        /*note SDL works special like thir too*/
        if (!init (NULL))
            return NULL;
    }

    if (!state->title)
    {
        state->title = malloc ((strlen(DefaultTitle) + 1) * sizeof(char *));
        if (!state->title)
            return PyErr_NoMemory ();
        strcpy (state->title, DefaultTitle);
        title = state->title;
    }

    if (win)
    {
        /*change existing window*/
        SDL_SetWindowTitle (win, title);
        SDL_SetWindowSize (win, w, h);
#warning Add mode stuff.
        surf = SDL_GetWindowSurface (win);
        if (!surf) {
            PyErr_SetString (PyExc_SDLError, SDL_GetError ());
            Py_SetDefaultWindow (NULL);
            return NULL;
        }
        assert (surface);
        PySurface_AsSurface (surface) = surf;
    }
    else
    {
        /*open window*/
        Uint32 sdl_flags = 0;
        if (flags & PGS_FULLSCREEN)
            sdl_flags |= SDL_WINDOW_FULLSCREEN;
        if (flags & PGS_OPENGL)
            sdl_flags |= SDL_WINDOW_OPENGL;
        if (flags & PGS_NOFRAME)
            sdl_flags |= SDL_WINDOW_BORDERLESS;
        if (flags & PGS_RESIZABLE)
            sdl_flags |= SDL_WINDOW_RESIZABLE;
#warning Only use SDL_WINDOW_SHOWN when window not hidden.
        if (!sdl_flags)
            sdl_flags = SDL_WINDOW_SHOWN;
        win = SDL_CreateWindow (title,
                                SDL_WINDOWPOS_UNDEFINED,
                                SDL_WINDOWPOS_UNDEFINED,
                                w,
                                h,
                                sdl_flags);
        if (!win)
            return RAISE (PyExc_SDLError, SDL_GetError ());

        if (state->gamma_ramp)
        {
            int result = SDL_SetWindowGammaRamp (win,
                                                 state->gamma_ramp,
                                                 state->gamma_ramp + 256,
                                                 state->gamma_ramp + 512);
            if (result) /* SDL Error? */
            {
                /* Discard a possibly faulty gamma ramp. */
                free (state->gamma_ramp);
                state->gamma_ramp = NULL;

                /* Recover error, then destroy the window */
                RAISE (PyExc_SDLError, SDL_GetError ());
                SDL_DestroyWindow (win);
                return NULL;
            }
        }
        surf = SDL_GetWindowSurface (win);
        if (!surf)
        {
            PyErr_SetString (PyExc_SDLError, SDL_GetError ());
            SDL_DestroyWindow (win);
            return NULL;
        }
        surface = PySurface_NewNoOwn (surf);
        if (!surface)
        {
            SDL_DestroyWindow (win);
            return 0;
        }

        /*no errors; make the window available*/
        Py_SetDefaultWindow (win);
        Py_SetDefaultWindowSurface (surface);
        Py_DECREF (surface);
    }

#if !defined(darwin)
    /*set the window icon*/
    if (!state->icon)
    {
        state->icon = display_resource (icon_defaultname);
        if (!state->icon)
            PyErr_Clear ();
        else
        {
            SDL_SetColorKey (PySurface_AsSurface (state->icon),
                             SDL_TRUE,
                             0);
        }
    }
    if (state->icon)
        SDL_SetWindowIcon (win, PySurface_AsSurface (state->icon));
#endif

    /*probably won't do much, but can't hurt, and might help*/
    SDL_PumpEvents ();

    /*return the window's surface (screen)*/
    Py_INCREF (surface);
    return surface;
}

static PyObject*
mode_ok (PyObject* self, PyObject* args)
{
#if 0
    int depth = 0;
    int w, h;
    int flags = SDL_SWSURFACE;

    VIDEO_INIT_CHECK ();

    if (!PyArg_ParseTuple (args, "(ii)|ii", &w, &h, &flags, &depth))
        return NULL;
#error Reimplement
    if (!depth)
        depth = SDL_GetVideoInfo ()->vfmt->BitsPerPixel;

    return PyInt_FromLong (SDL_VideoModeOK (w, h, depth, flags));
#endif
    return PyInt_FromLong ((long)0);
}

static PyObject*
list_modes (PyObject* self, PyObject* args)
{
#if 0
#error Reimplement
    SDL_PixelFormat format;
    SDL_Rect** rects;
    int flags=SDL_FULLSCREEN;
    PyObject *list, *size;

    format.BitsPerPixel = 0;
    if (PyTuple_Size (args)!=0
        && !PyArg_ParseTuple (args, "|bi", &format.BitsPerPixel, &flags))
        return NULL;

    VIDEO_INIT_CHECK ();

    if (!format.BitsPerPixel)
        format.BitsPerPixel = SDL_GetVideoInfo ()->vfmt->BitsPerPixel;

    rects = SDL_ListModes (&format, flags);

    if(rects == (SDL_Rect**)-1)
        return PyInt_FromLong (-1);

    if(!(list = PyList_New (0)))
        return NULL;
    if (!rects)
        return list;

    for (; *rects; ++rects)
    {
        if (!(size = Py_BuildValue ("(ii)", (*rects)->w, (*rects)->h)))
        {
            Py_DECREF (list);
            return NULL;
        }
        PyList_Append (list, size);
        Py_DECREF (size);
    }
    return list;
#endif
    Py_RETURN_NONE;
}

static PyObject*
flip (PyObject* self)
{
    SDL_Window* win = Py_GetDefaultWindow ();
    int status = 0;

    VIDEO_INIT_CHECK ();

    if (!win)
        return RAISE (PyExc_SDLError, "Display mode not set");

    Py_BEGIN_ALLOW_THREADS;
    if (SDL_GetWindowFlags (win) & SDL_WINDOW_OPENGL)
        SDL_GL_SwapWindow (win);
    else
        status = SDL_UpdateWindowSurface (win) == -1;
    Py_END_ALLOW_THREADS;

    if (status == -1)
        return RAISE (PyExc_SDLError, SDL_GetError ());
    Py_RETURN_NONE;
}

/*BAD things happen when out-of-bound rects go to updaterect*/
static SDL_Rect*
screencroprect (GAME_Rect* r, int w, int h, SDL_Rect* cur)
{
    if (r->x > w || r->y > h || (r->x + r->w) <= 0 || (r->y + r->h) <= 0)
        return 0;
    else
    {
        int right = MIN (r->x + r->w, w);
        int bottom = MIN (r->y + r->h, h);
        cur->x = (short) MAX (r->x, 0);
        cur->y = (short) MAX (r->y, 0);
        cur->w = (unsigned short) right - cur->x;
        cur->h = (unsigned short) bottom - cur->y;
    }
    return cur;
}

static PyObject*
update (PyObject* self, PyObject* arg)
{
    SDL_Window* win = Py_GetDefaultWindow ();
    GAME_Rect *gr, temp = { 0 };
    int wide, high;
    PyObject* obj;

    VIDEO_INIT_CHECK ();

    if (!win)
        return RAISE (PyExc_SDLError, "Display mode not set");
    SDL_GetWindowSize (win, &wide, &high);
    if (SDL_GetWindowFlags (win) & SDL_WINDOW_OPENGL)
        return RAISE (PyExc_SDLError, "Cannot update an OPENGL display");

    /*determine type of argument we got*/
    if (PyTuple_Size (arg) == 0)
    {
        SDL_UpdateWindowSurface (win);
        Py_RETURN_NONE;
    }
    else
    {
        obj = PyTuple_GET_ITEM (arg, 0);
        if (obj == Py_None)
            gr = &temp;
        else
        {
            gr = GameRect_FromObject (arg, &temp);
            if (!gr)
                PyErr_Clear ();
            else if (gr != &temp)
            {
                memcpy (&temp, gr, sizeof (temp));
                gr = &temp;
            }
        }
    }

    if (gr)
    {
        SDL_Rect sdlr;
        if (screencroprect (gr, wide, high, &sdlr))
            SDL_UpdateWindowSurfaceRects (win, &sdlr, 1);
    }
    else
    {
        PyObject* seq;
        PyObject* r;
        int loop, num, count;
        SDL_Rect* rects;
        if (PyTuple_Size (arg) != 1)
            return RAISE
                (PyExc_ValueError,
                 "update requires a rectstyle or sequence of recstyles");
        seq = PyTuple_GET_ITEM (arg, 0);
        if (!seq || !PySequence_Check (seq))
            return RAISE
                (PyExc_ValueError,
                 "update requires a rectstyle or sequence of recstyles");

        num = PySequence_Length (seq);
        rects = PyMem_New (SDL_Rect, num);
        if (!rects)
            return NULL;
        count = 0;
        for (loop = 0; loop < num; ++loop)
        {
            SDL_Rect* cur_rect = (rects + count);

            /*get rect from the sequence*/
            r = PySequence_GetItem (seq, loop);
            if (r == Py_None)
            {
                Py_DECREF(r);
                continue;
            }
            gr = GameRect_FromObject (r, &temp);
            Py_XDECREF (r);
            if (!gr)
            {
                PyMem_Free ((char*)rects);
                return RAISE (PyExc_ValueError,
                              "update_rects requires a single list of rects");
            }

            if (gr->w < 1 && gr->h < 1)
                continue;

            /*bail out if rect not onscreen*/
            if (!screencroprect (gr, wide, high, cur_rect))
                continue;

            ++count;
        }

        if (count) {
            Py_BEGIN_ALLOW_THREADS;
            SDL_UpdateWindowSurfaceRects (win, rects, count);
            Py_END_ALLOW_THREADS;
        }

        PyMem_Free ((char*)rects);
    }
    Py_RETURN_NONE;
}

static PyObject*
set_palette (PyObject* self, PyObject* args)
{
    PyObject* surface = Py_GetDefaultWindowSurface ();
    SDL_Surface* surf;
    SDL_Palette* pal;
    SDL_Color* colors;
    PyObject* list, *item = NULL;
    int i, len;
    int r, g, b;

    VIDEO_INIT_CHECK ();
    if (!PyArg_ParseTuple (args, "|O", &list))
        return NULL;
    if (!surface)
        return RAISE (PyExc_SDLError, "No display mode is set");
    Py_INCREF (surface);
    surf = PySurface_AsSurface (surface);
    pal = surf->format->palette;
    if (surf->format->BytesPerPixel != 1 || !pal)
    {
        Py_DECREF (surface);
        return RAISE (PyExc_SDLError, "Display mode is not colormapped");
    }

    if (!list)
    {
        Py_DECREF (surface);
        Py_RETURN_NONE;
    }

    if (!PySequence_Check (list))
    {
        Py_DECREF (surface);
        return RAISE (PyExc_ValueError, "Argument must be a sequence type");
    }

    len = MIN (pal->ncolors, PySequence_Length (list));

    colors = (SDL_Color*)malloc (len * sizeof (SDL_Color));
    if (!colors)
    {
        Py_DECREF (surface);
        return PyErr_NoMemory ();
    }

    for (i = 0; i < len; i++)
    {
        item = PySequence_GetItem (list, i);
        if (!PySequence_Check (item) || PySequence_Length (item) != 3)
        {
            Py_DECREF (item);
            free ((char*)colors);
            Py_DECREF (surface);
            return RAISE (PyExc_TypeError,
                          "takes a sequence of sequence of RGB");
        }
        if(!IntFromObjIndex (item, 0, &r) || !IntFromObjIndex (item, 1, &g)
           || !IntFromObjIndex (item, 2, &b))
        {
            Py_DECREF (item);
            free ((char*)colors);
            Py_DECREF (surface);
            return RAISE (PyExc_TypeError,
                          "RGB sequence must contain numeric values");
        }

        colors[i].r = (unsigned char)r;
        colors[i].g = (unsigned char)g;
        colors[i].b = (unsigned char)b;

        Py_DECREF (item);
    }

    pal = SDL_AllocPalette (len);
    if (!pal)
    {
        free ((char*)colors);
        Py_DECREF (surface);
        return RAISE (PyExc_SDLError, SDL_GetError ());
    }
    if (!SDL_SetPaletteColors (pal, colors, 0, len))
    {
        SDL_FreePalette (pal);
        free ((char*)colors);
        Py_DECREF (surface);
        return RAISE (PyExc_SDLError, SDL_GetError ());
    }

    SDL_SetSurfacePalette (surf, pal);
    SDL_FreePalette (pal);
    free ((char*)colors);
    Py_DECREF (surface);
    Py_RETURN_NONE;
}

static PyObject*
set_gamma (PyObject* self, PyObject* arg)
{
    _DisplayState* state = DISPLAY_MOD_STATE (self);
    SDL_Window* win = Py_GetDefaultWindow ();
    float r, g, b;
    Uint16* gamma_ramp;
    int result = 0;

    if (!PyArg_ParseTuple (arg, "f|ff", &r, &g, &b))
        return NULL;
    if (PyTuple_Size (arg) == 1)
        g = b = r;

    VIDEO_INIT_CHECK ();

    gamma_ramp = (Uint16 *) malloc ((3 * 256) * sizeof (Uint16));
    if (!gamma_ramp)
        return PyErr_NoMemory ();

    SDL_CalculateGammaRamp (r, gamma_ramp);
    SDL_CalculateGammaRamp (g, gamma_ramp + 256);
    SDL_CalculateGammaRamp (b, gamma_ramp + 256);
    if (win)
    {
        result = SDL_SetWindowGammaRamp (win,
                                         gamma_ramp,
                                         gamma_ramp + 256,
                                         gamma_ramp + 512);
        if (result)
        {
            /* Discard a possibly faulty gamma ramp */
            free (gamma_ramp);
            gamma_ramp = NULL;
        }
    }
    if (gamma_ramp)
    {
        if (state->gamma_ramp)
            free (state->gamma_ramp);
        state->gamma_ramp = gamma_ramp;
    }
    return PyInt_FromLong (result == 0);
}

static int
convert_to_uint16 (PyObject* python_array, Uint16* c_uint16_array)
{
    int i;
    PyObject* item;

    if (!c_uint16_array)
    {
        RAISE (PyExc_RuntimeError, "Memory not allocated for c_uint16_array.");
        return 0;
    }

    if (!PySequence_Check (python_array))
    {
        RAISE (PyExc_TypeError, "Array must be sequence type");
        return 0;
    }

    if (PySequence_Size (python_array) != 256)
    {
        RAISE (PyExc_ValueError, "gamma ramp must be 256 elements long");
        return 0;
    }
    for (i = 0; i < 256; i++)
    {
        item = PySequence_GetItem (python_array, i);
        if(!PyInt_Check (item))
        {
            RAISE (PyExc_ValueError,
                   "gamma ramp must contain integer elements");
            return 0;
        }
        c_uint16_array[i] = (Uint16)PyInt_AsLong (item);
        Py_XDECREF (item);
    }
    return 1;
}


static PyObject*
set_gamma_ramp (PyObject* self, PyObject* arg)
{
    _DisplayState* state = DISPLAY_MOD_STATE (self);
    SDL_Window* win = Py_GetDefaultWindow ();
    Uint16* gamma_ramp = (Uint16 *) malloc ((3 * 256) * sizeof (Uint16));
    Uint16 *r, *g, *b;
    int result;

    if (!gamma_ramp)
        return PyErr_NoMemory ();

    r = gamma_ramp;
    g = gamma_ramp + 256;
    b = gamma_ramp + 512;

    if(!PyArg_ParseTuple (arg, "O&O&O&",
                          convert_to_uint16, r,
                          convert_to_uint16, g,
                          convert_to_uint16, b))
    {
        free (gamma_ramp);
        return NULL;
    }

    VIDEO_INIT_CHECK ();

    if (win)
    {
        result = SDL_SetWindowGammaRamp (win,
                                         gamma_ramp,
                                         gamma_ramp + 256,
                                         gamma_ramp + 512);
        if (result)
        {
            /* Discard a possibly faulty gamma ramp */
            free (gamma_ramp);
            gamma_ramp = NULL;
        }
    }
    if (gamma_ramp)
    {
        if (state->gamma_ramp)
            free (state->gamma_ramp);
        state->gamma_ramp = gamma_ramp;
    }
    return PyInt_FromLong (result == 0);
}

static PyObject*
set_caption (PyObject* self, PyObject* arg)
{
    _DisplayState* state = DISPLAY_MOD_STATE (self);
    SDL_Window* win = Py_GetDefaultWindow ();
    char *title, *icontitle=NULL;

    if (!PyArg_ParseTuple (arg, "s|s", &title, &icontitle))
        return NULL;

    if (state->title)
        free (state->title);
    state->title = (char *)malloc ((strlen(title) + 1) * sizeof(char *));
    if (!state->title)
        return PyErr_NoMemory ();
    strcpy (state->title, title);
    if (win) 
        SDL_SetWindowTitle (win, title);
    Py_RETURN_NONE;
}

static PyObject*
get_caption (PyObject* self)
{
    _DisplayState* state = DISPLAY_MOD_STATE (self);
    SDL_Window* win = Py_GetDefaultWindow ();
    const char* title = win ? SDL_GetWindowTitle (win) : state->title;

    if (title && title[0]) /* Conditional && */
        return Py_BuildValue ("(ss)", title, title);

    return Py_BuildValue ("()");
}

static PyObject*
set_icon (PyObject* self, PyObject* arg)
{
    _DisplayState* state = DISPLAY_MOD_STATE (self);
    SDL_Window* win = Py_GetDefaultWindow ();
    PyObject* surface;
    if (!PyArg_ParseTuple (arg, "O!", &PySurface_Type, &surface))
        return NULL;
    if (!PyGame_Video_AutoInit ())
        return RAISE (PyExc_SDLError, SDL_GetError ());
#if (!defined(darwin))
    Py_INCREF (surface);
    Py_XDECREF (state->icon);
    state->icon = surface;
    if (win)
        SDL_SetWindowIcon (win, PySurface_AsSurface (surface));
#endif
    Py_RETURN_NONE;
}

static PyObject*
iconify (PyObject* self)
{
    SDL_Window* win = Py_GetDefaultWindow ();

    VIDEO_INIT_CHECK ();
    if (!win)
        return RAISE (PyExc_SDLError, "No open window");
    SDL_MinimizeWindow (win);
    Py_RETURN_NONE;
}

static PyObject*
toggle_fullscreen (PyObject* self)
{
    SDL_Window* win = Py_GetDefaultWindow ();
    int result;

    VIDEO_INIT_CHECK ();
    if (!win)
        return RAISE (PyExc_SDLError, "No open window");

    if (SDL_GetWindowFlags (win) & SDL_WINDOW_FULLSCREEN)
        result = SDL_SetWindowFullscreen (win, SDL_WINDOW_FULLSCREEN);
    else
        result = SDL_SetWindowFullscreen (win, 0);
    return PyInt_FromLong (result != 0);
}

static PyMethodDef _display_methods[] =
{
    { "__PYGAMEinit__", display_autoinit, 1,
      "auto initialize function for display." },
    { "init", (PyCFunction) init, METH_NOARGS, DOC_PYGAMEDISPLAYINIT },
    { "quit", (PyCFunction) quit, METH_NOARGS, DOC_PYGAMEDISPLAYQUIT },
    { "get_init", (PyCFunction) get_init, METH_NOARGS,
      DOC_PYGAMEDISPLAYGETINIT },
    { "get_active", (PyCFunction) get_active, METH_NOARGS,
      DOC_PYGAMEDISPLAYGETACTIVE },

/*    { "set_driver", set_driver, 1, doc_set_driver },*/
    { "get_driver", (PyCFunction) get_driver, METH_NOARGS,
      DOC_PYGAMEDISPLAYGETDRIVER },
#if 0
    { "get_wm_info", (PyCFunction) get_wm_info, METH_NOARGS,
      DOC_PYGAMEDISPLAYGETWMINFO },
    { "Info", (PyCFunction) Info, METH_NOARGS, DOC_PYGAMEDISPLAYINFO },
#endif
    { "get_surface", (PyCFunction) get_surface, METH_NOARGS,
      DOC_PYGAMEDISPLAYGETSURFACE },

    { "set_mode", set_mode, METH_VARARGS, DOC_PYGAMEDISPLAYSETMODE },
    { "mode_ok", mode_ok, METH_VARARGS, DOC_PYGAMEDISPLAYMODEOK },
    { "list_modes", list_modes, METH_VARARGS, DOC_PYGAMEDISPLAYLISTMODES },

    { "flip", (PyCFunction) flip, METH_NOARGS, DOC_PYGAMEDISPLAYFLIP },
    { "update", update, METH_VARARGS, DOC_PYGAMEDISPLAYUPDATE },

    { "set_palette", set_palette, METH_VARARGS, DOC_PYGAMEDISPLAYSETPALETTE },
    { "set_gamma", set_gamma, METH_VARARGS, DOC_PYGAMEDISPLAYSETGAMMA },
    { "set_gamma_ramp", set_gamma_ramp, METH_VARARGS,
      DOC_PYGAMEDISPLAYSETGAMMARAMP },

    { "set_caption", set_caption, METH_VARARGS, DOC_PYGAMEDISPLAYSETCAPTION },
    { "get_caption", (PyCFunction) get_caption, METH_NOARGS,
      DOC_PYGAMEDISPLAYGETCAPTION },
    { "set_icon", set_icon, METH_VARARGS, DOC_PYGAMEDISPLAYSETICON },

    { "iconify", (PyCFunction) iconify, METH_NOARGS, DOC_PYGAMEDISPLAYICONIFY },
    { "toggle_fullscreen", (PyCFunction) toggle_fullscreen, METH_NOARGS,
      DOC_PYGAMEDISPLAYTOGGLEFULLSCREEN },

    { "gl_set_attribute", gl_set_attribute, METH_VARARGS,
      DOC_PYGAMEDISPLAYGLSETATTRIBUTE },
    { "gl_get_attribute", gl_get_attribute, METH_VARARGS,
      DOC_PYGAMEDISPLAYGLGETATTRIBUTE },

    { NULL, NULL, 0, NULL }
};



MODINIT_DEFINE (display)
{
    PyObject* module;
    _DisplayState* state;
#if 0
    PyObject* dict;
    PyObject* apiobj;
    int ecode;
    static void* c_api[PYGAMEAPI_DISPLAY_NUMSLOTS];
#endif

    /* imported needed apis; Do this first so if there is an error
       the module is not loaded.
    */
    import_pygame_base ();
    if (PyErr_Occurred ()) {
        MODINIT_ERROR;
    }
    import_pygame_rect ();
    if (PyErr_Occurred ()) {
        MODINIT_ERROR;
    }
    import_pygame_surface ();
    if (PyErr_Occurred ()) {
        MODINIT_ERROR;
    }

#if 0
    /* type preparation */
    if (PyType_Ready (&PyVidInfo_Type) < 0) {
        MODINIT_ERROR;
    }
#endif

    /* create the module */
#if PY3
    module = PyModule_Create (&_module);
#else
    module = Py_InitModule3 (MODPREFIX "display",
                             _display_methods,
                             DOC_PYGAMEDISPLAY);
#endif
    if (module == NULL) {
        MODINIT_ERROR;
    }
    state = DISPLAY_MOD_STATE (module);
    state->title = NULL;
    state->icon = NULL;
    state->gamma_ramp = NULL;
#if 0
    dict = PyModule_GetDict (module);

    /* export the c api */
    c_api[0] = &PyVidInfo_Type;
    c_api[1] = PyVidInfo_New;
    apiobj = encapsulate_api (c_api, "display");
    if (apiobj == NULL) {
        DECREF_MOD (module);
        MODINIT_ERROR;
    }
    ecode = PyDict_SetItemString (dict, PYGAMEAPI_LOCAL_ENTRY, apiobj);
    Py_DECREF (apiobj);
    if (ecode) {
        DECREF_MOD (module);
        MODINIT_ERROR;
    }
#endif
    MODINIT_RETURN (module);
}
