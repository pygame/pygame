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

void
PyGame_static_init()
{
    PyImport_AppendInittab("pygame_base", PyInit_base);
    PyImport_AppendInittab("pygame_color", PyInit_color);
    PyImport_AppendInittab("pygame_constants", PyInit_constants);
    PyImport_AppendInittab("pygame_rect", PyInit_rect);
    PyImport_AppendInittab("pygame_surflock", PyInit_surflock);
    PyImport_AppendInittab("pygame_rwobject", PyInit_rwobject);
    PyImport_AppendInittab("pygame_bufferproxy", PyInit_bufferproxy);
    PyImport_AppendInittab("pygame_math", PyInit_pg_math);
    PyImport_AppendInittab("pygame_surface", PyInit_surface);
    PyImport_AppendInittab("pygame_pixelcopy", PyInit_pixelcopy);
    PyImport_AppendInittab("pygame_transform", PyInit_transform);
    PyImport_AppendInittab("pygame_display", PyInit_display);
    PyImport_AppendInittab("pygame__freetype", PyInit__freetype);
    PyImport_AppendInittab("pygame_font", PyInit_font);
    PyImport_AppendInittab("pygame_draw", PyInit_draw);
    PyImport_AppendInittab("pygame_gfxdraw", PyInit_gfxdraw);
    PyImport_AppendInittab("pygame_imageext", PyInit_imageext);
    PyImport_AppendInittab("pygame_image", PyInit_image);
    PyImport_AppendInittab("pygame_mask", PyInit_mask);
    PyImport_AppendInittab("pygame_mixer_music", PyInit_mixer_music);
    PyImport_AppendInittab("pygame_mixer", PyInit_pg_mixer);
    PyImport_AppendInittab("pygame_mouse", PyInit_mouse);
    PyImport_AppendInittab("pygame_key", PyInit_key);
    PyImport_AppendInittab("pygame_event", PyInit_event);
    PyImport_AppendInittab("pygame_joystick", PyInit_joystick);
    PyImport_AppendInittab("pygame_time", PyInit_pg_time);
    PyImport_AppendInittab("pygame_sdl2_video", PyInit_video);
    PyImport_AppendInittab("pygame_context", PyInit_context);
    PyImport_AppendInittab("pygame_sprite", PyInit__sprite);
    PyImport_AppendInittab("pygame__sdl2_sdl2", PyInit_sdl2);
    PyImport_AppendInittab("pygame__sdl2_sdl2_mixer", PyInit_mixer);
    PyImport_AppendInittab("pygame__sdl2_controller", PyInit_controller);
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
