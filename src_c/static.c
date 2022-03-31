#define NO_PYGAME_C_API
#define PYGAMEAPI_RECT_INTERNAL
#define PYGAMEAPI_EVENT_INTERNAL
#define PYGAMEAPI_JOYSTICK_INTERNAL
#define PYGAMEAPI_BASE_INTERNAL
#define PYGAMEAPI_SURFACE_INTERNAL

#define pgSurface_New(surface) pgSurface_New2((surface), 1)
#define pgSurface_NewNoOwn(surface) pgSurface_New2((surface), 0)


#include "pygame.h"
#include "Python.h"

#undef import_pygame_base
#undef import_pygame_rect
#undef import_pygame_surface
#undef import_pygame_color
#undef import_pygame_bufferproxy
#undef import_pygame_rwobject
#undef import_pygame_event

void import_pygame_base(void) {
}
void import_pygame_rect(void) {
}
void import_pygame_surface(void) {
}
void import_pygame_color(void) {
}
void import_pygame_bufferproxy(void) {
}
void import_pygame_rwobject(void) {
}
void import_pygame_event(void) {
}

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


#define pgSurface_New(surface) pgSurface_New2((surface), 1)
#include "image.c"

#include "imageext.c"


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

#include "font.c"

#include "mixer.c"

#include "music.c"

void import_pygame_joystick(void) {}
#include "_sdl2/controller.c"

#include "alphablit.c"

#include "surface_fill.c"
