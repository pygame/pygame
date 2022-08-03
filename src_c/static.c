#define NO_PYGAME_C_API

#define PYGAMEAPI_RECT_INTERNAL
#define PYGAMEAPI_EVENT_INTERNAL
#define PYGAMEAPI_JOYSTICK_INTERNAL
#define PYGAMEAPI_BASE_INTERNAL
#define PYGAMEAPI_SURFACE_INTERNAL

#define pgSurface_New(surface) (pgSurfaceObject *)pgSurface_New2((surface), 1)
#define pgSurface_NewNoOwn(surface) \
    (pgSurfaceObject *)pgSurface_New2((surface), 0)

#include "pygame.h"
#include "Python.h"

#if defined(__EMSCRIPTEN__)
#undef WITH_THREAD
#endif

#if defined(BUILD_STATIC)
#undef import_pygame_base
#undef import_pygame_rect
#undef import_pygame_surface
#undef import_pygame_color
#undef import_pygame_bufferproxy
#undef import_pygame_rwobject
#undef import_pygame_event

void
import_pygame_base(void)
{
}

void
import_pygame_rect(void)
{
}

void
import_pygame_surface(void)
{
}

void
import_pygame_color(void)
{
}

void
import_pygame_bufferproxy(void)
{
}

void
import_pygame_rwobject(void)
{
}

void
import_pygame_event(void)
{
}

void
import_pygame_joystick(void)
{
}

PyMODINIT_FUNC
PyInit_base(void);
PyMODINIT_FUNC
PyInit_color(void);
PyMODINIT_FUNC
PyInit_constants(void);
PyMODINIT_FUNC
PyInit_version(void);
PyMODINIT_FUNC
PyInit_rect(void);
PyMODINIT_FUNC
PyInit_surflock(void);
PyMODINIT_FUNC
PyInit_rwobject(void);
PyMODINIT_FUNC
PyInit_bufferproxy(void);

PyMODINIT_FUNC
PyInit_surface(void);
PyMODINIT_FUNC
PyInit_display(void);
PyMODINIT_FUNC
PyInit__freetype(void);
PyMODINIT_FUNC
PyInit_font(void);

PyMODINIT_FUNC
PyInit_draw(void);
PyMODINIT_FUNC
PyInit_mouse(void);
PyMODINIT_FUNC
PyInit_key(void);
PyMODINIT_FUNC
PyInit_event(void);

PyMODINIT_FUNC
PyInit_joystick(void);

PyMODINIT_FUNC
PyInit_imageext(void);

PyMODINIT_FUNC
PyInit_image(void);

PyMODINIT_FUNC
PyInit_mask(void);

PyMODINIT_FUNC
PyInit_mixer_music(void);

PyMODINIT_FUNC
PyInit_pg_mixer(void);

PyMODINIT_FUNC
PyInit_pg_math(void);

PyMODINIT_FUNC
PyInit_pg_time(void);

PyMODINIT_FUNC
PyInit_sdl2(void);

PyMODINIT_FUNC
PyInit_mixer(void);

PyMODINIT_FUNC
PyInit_context(void);

PyMODINIT_FUNC
PyInit_controller(void);

PyMODINIT_FUNC
PyInit_transform(void);

PyMODINIT_FUNC
PyInit_video(void);

PyMODINIT_FUNC
PyInit__sprite(void);

PyMODINIT_FUNC
PyInit_pixelcopy(void);

PyMODINIT_FUNC
PyInit_gfxdraw(void);

PyMODINIT_FUNC
PyInit_audio(void);

// pygame_static module

void
load_submodule(const char *parent, PyObject *mod, const char *alias)
{
    char fqn[1024];
    snprintf(fqn, sizeof(fqn), "%s.%s", parent, alias);

    PyObject *modules = PyImport_GetModuleDict();

    PyObject *pmod = PyDict_GetItemString(modules, parent);

    if (!mod) {
        snprintf(fqn, sizeof(fqn), "ERROR: %s.%s", parent, alias);
        puts(fqn);
        PyErr_Print();
        PyErr_Clear();
    }
    else {
        PyDict_SetItemString(modules, fqn, mod);
        PyDict_SetItemString(PyModule_GetDict(mod), "__name__",
                             PyUnicode_FromString(fqn));
        PyModule_AddObjectRef(pmod, alias, mod);
        Py_XDECREF(mod);
    }
}

void
load_submodule_mphase(const char *parent, PyObject *mdef, PyObject *spec,
                      const char *alias)
{
    char fqn[1024];
    snprintf(fqn, sizeof(fqn), "%s.%s", parent, alias);

    PyObject *modules = PyImport_GetModuleDict();

    Py_DECREF(PyObject_GetAttrString(spec, "name"));

    PyObject_SetAttrString(spec, "name", PyUnicode_FromString(alias));

    PyObject *pmod = PyDict_GetItemString(modules, parent);

    PyObject *mod = PyModule_FromDefAndSpec((PyModuleDef *)mdef, spec);

    PyDict_SetItemString(PyModule_GetDict(mod), "__package__",
                         PyUnicode_FromString(parent));

    // TODO SET PACKAGE

    PyModule_ExecDef(mod, (PyModuleDef *)mdef);

    if (!mod) {
        snprintf(fqn, sizeof(fqn), "ERROR: %s.%s", parent, alias);
        puts(fqn);
        PyErr_Print();
        PyErr_Clear();
    }
    else {
        PyDict_SetItemString(modules, fqn, mod);
        PyDict_SetItemString(PyModule_GetDict(mod), "__name__",
                             PyUnicode_FromString(fqn));
        PyModule_AddObjectRef(pmod, alias, mod);
        Py_XDECREF(mod);
    }
}

static PyObject *
mod_pygame_import_cython(PyObject *self, PyObject *spec)
{
    load_submodule_mphase("pygame._sdl2", PyInit_sdl2(), spec, "sdl2");
    load_submodule_mphase("pygame._sdl2", PyInit_mixer(), spec, "mixer");
    load_submodule_mphase("pygame._sdl2", PyInit_controller(), spec,
                          "controller");
    load_submodule_mphase("pygame._sdl2", PyInit_audio(), spec, "audio");
    load_submodule_mphase("pygame._sdl2", PyInit_video(), spec, "video");
    // depends on pygame._sdl2.video
    load_submodule_mphase("pygame", PyInit__sprite(), spec, "_sprite");
    Py_RETURN_NONE;
}

static PyMethodDef mod_pygame_static_methods[] = {
    {"import_cython", (PyCFunction)mod_pygame_import_cython, METH_O,
     "pygame._sdl2.sdl2"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef mod_pygame_static = {PyModuleDef_HEAD_INIT,
                                               "pygame_static", NULL, -1,
                                               mod_pygame_static_methods};

PyMODINIT_FUNC
PyInit_pygame_static()
{
    load_submodule("pygame", PyInit_base(), "base");
    load_submodule("pygame", PyInit_constants(), "constants");
    load_submodule("pygame", PyInit_surflock(), "surflock");
    load_submodule("pygame", PyInit_rwobject(), "rwobject");
    load_submodule("pygame", PyInit_pg_math(), "math");
    load_submodule("pygame", PyInit_display(), "display");
    load_submodule("pygame", PyInit_surface(), "surface");
    load_submodule("pygame", PyInit_context(), "context");
    load_submodule("pygame", PyInit_key(), "key");

    load_submodule("pygame", PyInit_rect(), "rect");
    load_submodule("pygame", PyInit_gfxdraw(), "gfxdraw");
    load_submodule("pygame", PyInit_pg_time(), "time");
    load_submodule("pygame", PyInit__freetype(), "_freetype");

    load_submodule("pygame", PyInit_imageext(), "imageext");

    load_submodule("pygame", PyInit_image(), "image");
    load_submodule("pygame", PyInit_font(), "font");
    load_submodule("pygame", PyInit_pixelcopy(), "pixelcopy");

    load_submodule("pygame", PyInit_color(), "color");
    load_submodule("pygame", PyInit_bufferproxy(), "bufferproxy");

    load_submodule("pygame", PyInit_transform(), "transform");
    load_submodule("pygame", PyInit_draw(), "draw");

    load_submodule("pygame", PyInit_mask(), "mask");
    load_submodule("pygame", PyInit_mouse(), "mouse");
    load_submodule("pygame", PyInit_event(), "event");
    load_submodule("pygame", PyInit_joystick(), "joystick");

    load_submodule("pygame", PyInit_pg_mixer(), "mixer");

    load_submodule("pygame.mixer", PyInit_mixer_music(), "music");

    return PyModule_Create(&mod_pygame_static);
}

void
pygame_Inittab()
{
    PyImport_AppendInittab("pygame_static", PyInit_pygame_static);
}

#endif  // defined(BUILD_STATIC)

#include "base.c"

#include "rect.c"

#undef pgSurface_Lock
#undef pgSurface_Unlock
#undef pgSurface_LockBy
#undef pgSurface_UnlockBy
#undef pgSurface_Prep
#undef pgSurface_Unprep
#undef pgLifetimeLock_Type
#undef pgSurface_LockLifetime

#include "surflock.c"

#undef pgColor_New
#undef pgColor_NewLength
#undef pg_RGBAFromColorObj
#undef pg_RGBAFromFuzzyColorObj
#undef pgColor_Type

#include "color.c"

#undef pgBufproxy_New

#include "bufferproxy.c"

#undef pgSurface_Blit
#undef pgSurface_New
#undef pgSurface_Type
#undef pgSurface_SetSurface

#include "surface.c"

#undef pgVidInfo_Type
#undef pgVidInfo_New

#include "display.c"

#include "draw.c"

#undef pg_EncodeString
#undef pg_EncodeFilePath
#undef pgRWops_IsFileObject
#undef pgRWops_GetFileExtension
#undef pgRWops_FromFileObject
#undef pgRWops_ReleaseObject
#undef pgRWops_FromObject

#include "rwobject.c"

#define pgSurface_New(surface) (pgSurfaceObject *)pgSurface_New2((surface), 1)
#include "image.c"

#include "imageext.c"

#include "mask.c"

#undef pg_EnableKeyRepeat
#undef pg_GetKeyRepeat
#undef pgEvent_FillUserEvent
#undef pgEvent_Type
#undef pgEvent_New

#include "event.c"

#include "mouse.c"

#include "key.c"

#include "joystick.c"

#include "time.c"

#include "context.c"

#include "_freetype.c"
#include "freetype/ft_wrap.c"
#include "freetype/ft_render.c"
#include "freetype/ft_render_cb.c"
#include "freetype/ft_cache.c"
#include "freetype/ft_layout.c"
#include "freetype/ft_unicode.c"

#undef DOC_FONTUNDERLINE
#undef DOC_FONTRENDER
#undef DOC_FONTSIZE

#include "font.c"

#include "mixer.c"

#include "music.c"

#include "gfxdraw.c"

#include "alphablit.c"

#include "surface_fill.c"
#include "pixelarray.c"
#include "pixelcopy.c"
#include "newbuffer.c"

#include "_sdl2/controller.c"
#include "_sdl2/touch.c"
#include "transform.c"
// that remove some warnings
#undef MAX
#undef MIN
#include "scale2x.c"
