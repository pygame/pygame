/* Python 2.x/3.x and SDL compatibility tools
 */

#if !defined(PGCOMPAT_H)
#define PGCOMPAT_H

#include <Python.h>

/* Cobjects vanish in Python 3.2; so we will code as though we use capsules */
#if defined(Py_CAPSULE_H)
#define PG_HAVE_CAPSULE 1
#else
#define PG_HAVE_CAPSULE 0
#endif
#if defined(Py_COBJECT_H)
#define PG_HAVE_COBJECT 1
#else
#define PG_HAVE_COBJECT 0
#endif
#if !PG_HAVE_CAPSULE
#define PyCapsule_New(ptr, n, dfn) PyCObject_FromVoidPtr(ptr, dfn)
#define PyCapsule_GetPointer(obj, n) PyCObject_AsVoidPtr(obj)
#define PyCapsule_CheckExact(obj) PyCObject_Check(obj)
#endif

/* Pygame uses Py_buffer (PEP 3118) to exchange array information internally;
 * define here as needed.
 */
#if !defined(PyBUF_SIMPLE)
typedef struct bufferinfo {
	void *buf;
	PyObject *obj;
	Py_ssize_t len;
	Py_ssize_t itemsize;
	int readonly;
	int ndim;
	char *format;
	Py_ssize_t *shape;
	Py_ssize_t *strides;
	Py_ssize_t *suboffsets;
	void *internal;
} Py_buffer;

/* Flags for getting buffers */
#define PyBUF_SIMPLE 0
#define PyBUF_WRITABLE 0x0001
/*  we used to include an E, backwards compatible alias  */
#define PyBUF_WRITEABLE PyBUF_WRITABLE
#define PyBUF_FORMAT 0x0004
#define PyBUF_ND 0x0008
#define PyBUF_STRIDES (0x0010 | PyBUF_ND)
#define PyBUF_C_CONTIGUOUS (0x0020 | PyBUF_STRIDES)
#define PyBUF_F_CONTIGUOUS (0x0040 | PyBUF_STRIDES)
#define PyBUF_ANY_CONTIGUOUS (0x0080 | PyBUF_STRIDES)
#define PyBUF_INDIRECT (0x0100 | PyBUF_STRIDES)

#define PyBUF_CONTIG (PyBUF_ND | PyBUF_WRITABLE)
#define PyBUF_CONTIG_RO (PyBUF_ND)

#define PyBUF_STRIDED (PyBUF_STRIDES | PyBUF_WRITABLE)
#define PyBUF_STRIDED_RO (PyBUF_STRIDES)

#define PyBUF_RECORDS (PyBUF_STRIDES | PyBUF_WRITABLE | PyBUF_FORMAT)
#define PyBUF_RECORDS_RO (PyBUF_STRIDES | PyBUF_FORMAT)

#define PyBUF_FULL (PyBUF_INDIRECT | PyBUF_WRITABLE | PyBUF_FORMAT)
#define PyBUF_FULL_RO (PyBUF_INDIRECT | PyBUF_FORMAT)

#define PyBUF_READ 0x100
#define PyBUF_WRITE 0x200
#define PyBUF_SHADOW 0x400

typedef int(*getbufferproc)(PyObject *, Py_buffer *, int);
typedef void(*releasebufferproc)(Py_buffer *);
#endif /* ~defined(PyBUF_SIMPLE) */

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
