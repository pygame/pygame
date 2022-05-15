/*
  pygame - Python Game Library
  Copyright (C) 2000-2001  Pete Shinners
  Copyright (C) 2006 Rene Dudfield

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
 *  extended image module for pygame, note this only has
 *  the extended load and save functions, which are autmatically used
 *  by the normal pygame.image module if it is available.
 */
#include "pygame.h"

/* Keep a stray macro from conflicting with python.h */
#if defined(HAVE_PROTOTYPES)
#undef HAVE_PROTOTYPES
#endif
/* Remove GCC macro redefine warnings. */
#if defined(HAVE_STDDEF_H) /* also defined in pygame.h (python.h) */
#undef HAVE_STDDEF_H
#endif
#if defined(HAVE_STDLIB_H) /* also defined in pygame.h (SDL.h) */
#undef HAVE_STDLIB_H
#endif

#ifdef __SYMBIAN32__ /* until PNG support is done for Symbian */
#include <stdio.h>
#else
// PNG_SKIP_SETJMP_CHECK : non-regression on #662 (build error on old libpng)
#define PNG_SKIP_SETJMP_CHECK
#include <png.h>
#endif

#include "pgcompat.h"

#include "doc/image_doc.h"

#include "pgopengl.h"

#include <SDL_image.h>
#ifdef WIN32
#define strcasecmp _stricmp
#else
#include <strings.h>
#endif
#include <string.h>

#define JPEG_QUALITY 85

#ifdef WITH_THREAD
static SDL_mutex *_pg_img_mutex = 0;
#endif /* WITH_THREAD */

#ifdef WIN32
#include <windows.h>
#define pg_RWflush(rwops) \
    (FlushFileBuffers((HANDLE)(rwops)->hidden.windowsio.h) ? 0 : -1)

#else /* ~WIN32 */
#define pg_RWflush(rwops) (fflush((rwops)->hidden.stdio.fp) ? -1 : 0)
#endif /* ~WIN32 */

static char *
find_extension(char *fullname)
{
    char *dot;

    if (fullname == NULL) {
        return NULL;
    }

    dot = strrchr(fullname, '.');
    if (dot == NULL) {
        return fullname;
    }
    return dot + 1;
}

static PyObject *
image_load_ext(PyObject *self, PyObject *arg)
{
    PyObject *obj;
    PyObject *final;
    char *name = NULL, *ext = NULL;
    SDL_Surface *surf;
    SDL_RWops *rw = NULL;

    if (!PyArg_ParseTuple(arg, "O|s", &obj, &name)) {
        return NULL;
    }

    rw = pgRWops_FromObject(obj);
    if (rw == NULL) /* stop on NULL, error already set */
        return NULL;
    ext = pgRWops_GetFileExtension(rw);
    if (name) /* override extension with namehint if given */
        ext = find_extension(name);

#ifdef WITH_THREAD
    /*
    if (ext)
        lock_mutex = !strcasecmp(ext, "gif");
    */
    Py_BEGIN_ALLOW_THREADS;

    /* using multiple threads does not work for (at least) SDL_image
     * <= 2.0.4
    SDL_LockMutex(_pg_img_mutex);
    surf = IMG_LoadTyped_RW(rw, 1, ext);
    SDL_UnlockMutex(_pg_img_mutex);
    */

    surf = IMG_LoadTyped_RW(rw, 1, ext);
    Py_END_ALLOW_THREADS;
#else  /* ~WITH_THREAD */
    surf = IMG_LoadTyped_RW(rw, 1, ext);
#endif /* ~WITH_THREAD */

    if (surf == NULL)
        return RAISE(pgExc_SDLError, IMG_GetError());

    final = (PyObject *)pgSurface_New(surf);
    if (final == NULL) {
        SDL_FreeSurface(surf);
    }
    return final;
}

#ifdef PNG_H

static void
png_write_fn(png_structp png_ptr, png_bytep data, png_size_t length)
{
    SDL_RWops *rwops = (SDL_RWops *)png_get_io_ptr(png_ptr);
    if (SDL_RWwrite(rwops, data, 1, length) != length) {
        SDL_RWclose(rwops);
        png_error(png_ptr,
                  "Error while writing to the PNG file (SDL_RWwrite)");
    }
}

static void
png_flush_fn(png_structp png_ptr)
{
    SDL_RWops *rwops = (SDL_RWops *)png_get_io_ptr(png_ptr);
    if (pg_RWflush(rwops)) {
        SDL_RWclose(rwops);
        png_error(png_ptr, "Error while writing to PNG file (flush)");
    }
}

static int
write_png(SDL_RWops *rwops, png_bytep *rows,
          SDL_Palette *palette, int w, int h, int colortype, int bitdepth)
{
    png_structp png_ptr = NULL;
    png_infop info_ptr = NULL;
    png_colorp color_ptr = NULL;
    char *doing;

    doing = "create png write struct";
    if (!(png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL,
                                            NULL)))
        goto fail;

    doing = "create png info struct";
    if (!(info_ptr = png_create_info_struct(png_ptr)))
        goto fail;

    if (setjmp(png_jmpbuf(png_ptr)))
        goto fail;

    /* doing = "init IO"; */
    png_set_write_fn(png_ptr, rwops, png_write_fn, png_flush_fn);

    /* doing = "write header"; */
    png_set_IHDR(png_ptr, info_ptr, w, h, bitdepth, colortype,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE,
                 PNG_FILTER_TYPE_BASE);

    if (palette) {
        doing = "set pallete";
        const int ncolors = palette->ncolors;
        int i;
        if (!(color_ptr =
                  (png_colorp)SDL_malloc(sizeof(png_colorp) * ncolors)))
            goto fail;
        for (i = 0; i < ncolors; i++) {
            color_ptr[i].red = palette->colors[i].r;
            color_ptr[i].green = palette->colors[i].g;
            color_ptr[i].blue = palette->colors[i].b;
        }
        png_set_PLTE(png_ptr, info_ptr, color_ptr, ncolors);
        SDL_free(color_ptr);
    }

    /* doing = "write info"; */
    png_write_info(png_ptr, info_ptr);

    /* doing = "write image"; */
    png_write_image(png_ptr, rows);

    /* doing = "write end"; */
    png_write_end(png_ptr, NULL);

    png_destroy_write_struct(&png_ptr, &info_ptr);
    return 0;

fail:
    /*
     * I don't see how to handle the case where png_ptr
     * was allocated but info_ptr was not. However, those
     * calls should only fail if memory is out and you are
     * probably screwed regardless then. The resulting memory
     * leak is the least of your concerns.
     */
    if (png_ptr && info_ptr) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
    }
    SDL_SetError("SavePNG: could not %s", doing);
    return -1;
}

static int
SavePNG_RW(SDL_Surface *surface, SDL_RWops *rw)
{
    static unsigned char **ss_rows;
    static int ss_size;
    static int ss_w, ss_h;
    SDL_Surface *ss_surface;
    SDL_Rect ss_rect;
    int r, i;
    int alpha = 0;
    SDL_Palette *palette;
    Uint8 surf_alpha = 255;
    Uint32 surf_colorkey;
    int has_colorkey = 0;
    SDL_BlendMode surf_mode;

    ss_rows = 0;
    ss_size = 0;
    ss_surface = NULL;

    palette = surface->format->palette;
    ss_w = surface->w;
    ss_h = surface->h;

    if (surface->format->Amask) {
        alpha = 1;
        ss_surface = SDL_CreateRGBSurface(0, ss_w, ss_h, 32,
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
                                          0xff000000, 0xff0000, 0xff00, 0xff
#else
                                          0xff, 0xff00, 0xff0000, 0xff000000
#endif
        );
    }
    else {
        ss_surface = SDL_CreateRGBSurface(0, ss_w, ss_h, 24,
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
                                          0xff0000, 0xff00, 0xff, 0
#else
                                          0xff, 0xff00, 0xff0000, 0
#endif
        );
    }

    if (ss_surface == NULL)
        return -1;

    SDL_GetSurfaceAlphaMod(surface, &surf_alpha);
    SDL_SetSurfaceAlphaMod(surface, 255);
    SDL_GetSurfaceBlendMode(surface, &surf_mode);
    SDL_SetSurfaceBlendMode(surface, SDL_BLENDMODE_NONE);

    if (SDL_GetColorKey(surface, &surf_colorkey) == 0) {
        has_colorkey = 1;
        SDL_SetColorKey(surface, SDL_FALSE, surf_colorkey);
    }

    ss_rect.x = 0;
    ss_rect.y = 0;
    ss_rect.w = ss_w;
    ss_rect.h = ss_h;
    SDL_BlitSurface(surface, &ss_rect, ss_surface, NULL);

#ifdef _MSC_VER
    /* Make MSVC static analyzer happy by assuring ss_size >= 2 to supress
     * a false analyzer report */
    __analysis_assume(ss_size >= 2);
#endif

    if (ss_size == 0) {
        ss_size = ss_h;
        ss_rows = (unsigned char **)malloc(sizeof(unsigned char *) * ss_size);
        if (ss_rows == NULL)
            return -1;
    }
    if (has_colorkey)
        SDL_SetColorKey(surface, SDL_TRUE, surf_colorkey);
    SDL_SetSurfaceAlphaMod(surface, surf_alpha);
    SDL_SetSurfaceBlendMode(surface, surf_mode);

    for (i = 0; i < ss_h; i++) {
        ss_rows[i] =
            ((unsigned char *)ss_surface->pixels) + i * ss_surface->pitch;
    }

    if (palette) {
        r = write_png(rw, ss_rows, palette, surface->w, surface->h,
                      PNG_COLOR_TYPE_PALETTE, 8);
    }
    else if (alpha) {
        r = write_png(rw, ss_rows, NULL, surface->w, surface->h,
                      PNG_COLOR_TYPE_RGB_ALPHA, 8);
    }
    else {
        r = write_png(rw, ss_rows, NULL, surface->w, surface->h,
                      PNG_COLOR_TYPE_RGB, 8);
    }

    free(ss_rows);
    SDL_FreeSurface(ss_surface);
    ss_surface = NULL;
    return r;
}

#endif /* end if PNG_H */

static PyObject *
image_save_ext(PyObject *self, PyObject *arg)
{
    pgSurfaceObject *surfobj;
    PyObject *obj;
    char *namehint = NULL;
    PyObject *oencoded = NULL;
    SDL_Surface *surf;
    int result = 1;
    char *name = NULL;
    SDL_RWops *rw = NULL;

    if (!PyArg_ParseTuple(arg, "O!O|s", &pgSurface_Type, &surfobj, &obj,
                          &namehint)) {
        return NULL;
    }

    surf = pgSurface_AsSurface(surfobj);
    pgSurface_Prep(surfobj);

    rw = pgRWops_FromObjectAndMode(obj, "wb");
    if (rw == NULL) { /* propagate error immediately */
        return NULL;
    }

    char* ext = pgRWops_GetFileExtension(rw);
    if (namehint) {
        ext = find_extension(namehint);
    }

    /* TODO: Py_BEGIN_ALLOW_THREADS anywhere here? */

    if (!strcasecmp(ext, "jpeg") || !strcasecmp(ext, "jpg")) {
        if (result = IMG_SaveJPG_RW(surf, rw, 0, JPEG_QUALITY) < 0) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
        }
    }

    if (!strcasecmp(ext, "png")) {
        if (result = SavePNG_RW(surf, rw) < 0) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
        }
    }

    pgRWops_ReleaseObject(rw);
    pgSurface_Unprep(surfobj);

    /* result 1 means no image type was ever found to match */
    if (result == 1) {
        PyErr_Format(pgExc_SDLError, "Unrecognized image type: %s", ext);
        return NULL;
    }

    /* result < 0 means error, can be propagated as python error */
    if (result < 0) {
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject *
image_get_sdl_image_version(PyObject *self, PyObject *_null)
{
    return Py_BuildValue("iii", SDL_IMAGE_MAJOR_VERSION,
                         SDL_IMAGE_MINOR_VERSION, SDL_IMAGE_PATCHLEVEL);
}

#ifdef WITH_THREAD
static void
_imageext_free(void *ptr)
{
    if (_pg_img_mutex) {
        SDL_DestroyMutex(_pg_img_mutex);
        _pg_img_mutex = 0;
    }
}
#endif /* WITH_THREAD */

static PyMethodDef _imageext_methods[] = {
    {"load_extended", image_load_ext, METH_VARARGS, DOC_PYGAMEIMAGE},
    {"save_extended", image_save_ext, METH_VARARGS, DOC_PYGAMEIMAGE},
    {"_get_sdl_image_version", image_get_sdl_image_version, METH_NOARGS,
     "_get_sdl_image_version() -> (major, minor, patch)\n"
     "Note: Should not be used directly."},
    {NULL, NULL, 0, NULL}};

/*DOC*/ static char _imageext_doc[] =
    /*DOC*/ "additional image loaders";

MODINIT_DEFINE(imageext)
{
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT, "imageext", _imageext_doc, -1,
        _imageext_methods,     NULL,       NULL,          NULL,
#ifdef WITH_THREAD
        _imageext_free};
#else  /* ~WITH_THREAD */
                                         0};
#endif /* ~WITH_THREAD */

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
    import_pygame_rwobject();

    if (PyErr_Occurred()) {
        return NULL;
    }

#ifdef WITH_THREAD
    _pg_img_mutex = SDL_CreateMutex();
    if (!_pg_img_mutex) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return NULL;
    }
#endif /* WITH_THREAD */

    /* create the module */
    return PyModule_Create(&_module);
}
