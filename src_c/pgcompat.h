/* Python 2.x/3.x compatibility tools (internal)
 */
#ifndef PGCOMPAT_INTERNAL_H
#define PGCOMPAT_INTERNAL_H

#include "include/pgcompat.h"

/* Module init function returns new module instance. */
#define MODINIT_DEFINE(mod_name) PyMODINIT_FUNC PyInit_##mod_name(void)

/* Defaults for unicode file path encoding */
#if defined(MS_WIN32)
#define UNICODE_DEF_FS_ERROR "replace"
#else
#define UNICODE_DEF_FS_ERROR "surrogateescape"
#endif

#define RELATIVE_MODULE(m) ("." m)

#ifndef Py_TPFLAGS_HAVE_NEWBUFFER
#define Py_TPFLAGS_HAVE_NEWBUFFER 0
#endif

#define Slice_GET_INDICES_EX(slice, length, start, stop, step, slicelength) \
    PySlice_GetIndicesEx(slice, length, start, stop, step, slicelength)

#if defined(SDL_VERSION_ATLEAST)
#if !(SDL_VERSION_ATLEAST(2, 0, 5))
/* These functions require SDL 2.0.5 or greater.

  https://wiki.libsdl.org/SDL_SetWindowResizable
*/
void
SDL_SetWindowResizable(SDL_Window *window, SDL_bool resizable);
int
SDL_GetWindowOpacity(SDL_Window *window, float *opacity);
int
SDL_SetWindowOpacity(SDL_Window *window, float opacity);
int
SDL_SetWindowModalFor(SDL_Window *modal_window, SDL_Window *parent_window);
int
SDL_SetWindowInputFocus(SDL_Window *window);
SDL_Surface *
SDL_CreateRGBSurfaceWithFormat(Uint32 flags, int width, int height, int depth,
                               Uint32 format);
#endif /* !(SDL_VERSION_ATLEAST(2, 0, 5)) */
#endif /* defined(SDL_VERSION_ATLEAST) */

#endif /* ~PGCOMPAT_INTERNAL_H */
