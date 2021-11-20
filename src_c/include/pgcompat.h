/* Python 2.x/3.x and SDL compatibility tools
 */

#if !defined(PGCOMPAT_H)
#define PGCOMPAT_H

#include <Python.h>

/* define common types where SDL is not included */
#ifndef SDL_VERSION_ATLEAST
#ifdef _MSC_VER
typedef unsigned __int8 uint8_t;
typedef unsigned __int32 uint32_t;
#else
#include <stdint.h>
#endif
typedef uint32_t Uint32;
typedef uint8_t Uint8;
#endif /* no SDL */

#if defined(SDL_VERSION_ATLEAST)

#ifndef SDL_WINDOW_VULKAN
#define SDL_WINDOW_VULKAN 0
#endif

#ifndef SDL_WINDOW_ALWAYS_ON_TOP
#define SDL_WINDOW_ALWAYS_ON_TOP 0
#endif

#ifndef SDL_WINDOW_SKIP_TASKBAR
#define SDL_WINDOW_SKIP_TASKBAR 0
#endif

#ifndef SDL_WINDOW_UTILITY
#define SDL_WINDOW_UTILITY 0
#endif

#ifndef SDL_WINDOW_TOOLTIP
#define SDL_WINDOW_TOOLTIP 0
#endif

#ifndef SDL_WINDOW_POPUP_MENU
#define SDL_WINDOW_POPUP_MENU 0
#endif

#ifndef SDL_WINDOW_INPUT_GRABBED
#define SDL_WINDOW_INPUT_GRABBED 0
#endif

#ifndef SDL_WINDOW_INPUT_FOCUS
#define SDL_WINDOW_INPUT_FOCUS 0
#endif

#ifndef SDL_WINDOW_MOUSE_FOCUS
#define SDL_WINDOW_MOUSE_FOCUS 0
#endif

#ifndef SDL_WINDOW_FOREIGN
#define SDL_WINDOW_FOREIGN 0
#endif

#ifndef SDL_WINDOW_ALLOW_HIGHDPI
#define SDL_WINDOW_ALLOW_HIGHDPI 0
#endif

#ifndef SDL_WINDOW_MOUSE_CAPTURE
#define SDL_WINDOW_MOUSE_CAPTURE 0
#endif

#ifndef SDL_WINDOW_ALWAYS_ON_TOP
#define SDL_WINDOW_ALWAYS_ON_TOP 0
#endif

#ifndef SDL_WINDOW_SKIP_TASKBAR
#define SDL_WINDOW_SKIP_TASKBAR 0
#endif

#ifndef SDL_WINDOW_UTILITY
#define SDL_WINDOW_UTILITY 0
#endif

#ifndef SDL_WINDOW_TOOLTIP
#define SDL_WINDOW_TOOLTIP 0
#endif

#ifndef SDL_WINDOW_POPUP_MENU
#define SDL_WINDOW_POPUP_MENU 0
#endif

#if SDL_VERSION_ATLEAST(2, 0, 4)
/* To control the use of:
 * SDL_AUDIODEVICEADDED
 * SDL_AUDIODEVICEREMOVED
 *
 * Ref: https://wiki.libsdl.org/SDL_EventType
 * Ref: https://wiki.libsdl.org/SDL_AudioDeviceEvent
 */
#define SDL2_AUDIODEVICE_SUPPORTED
#endif

#ifndef SDL_MOUSEWHEEL_FLIPPED
#define NO_SDL_MOUSEWHEEL_FLIPPED
#endif

#endif /* defined(SDL_VERSION_ATLEAST) */

#endif /* ~defined(PGCOMPAT_H) */
