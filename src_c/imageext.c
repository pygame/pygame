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
 *  the extended load and save functions, which are automatically used
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

// PNG_SKIP_SETJMP_CHECK : non-regression on #662 (build error on old libpng)
#define PNG_SKIP_SETJMP_CHECK
#include <png.h>

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

/*
#ifdef WITH_THREAD
static SDL_mutex *_pg_img_mutex = 0;
#endif
*/

#ifdef WIN32
#include <windows.h>
#define pg_RWflush(rwops) \
    (FlushFileBuffers((HANDLE)(rwops)->hidden.windowsio.h) ? 0 : -1)

#else /* ~WIN32 */
#define pg_RWflush(rwops) (fflush((rwops)->hidden.stdio.fp) ? -1 : 0)
#endif /* ~WIN32 */

static char *
iext_find_extension(char *fullname)
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
        ext = iext_find_extension(name);

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

/* This entire png saving code is directly copied from the SDL_image source
 * (with minor changes)
 * Eventually this should be removed, and we should start using the SDL_image
 * functions directly */
#ifdef PNG_H

#if SDL_BYTEORDER == SDL_LIL_ENDIAN
static const Uint32 png_format = SDL_PIXELFORMAT_ABGR8888;
#else
static const Uint32 png_format = SDL_PIXELFORMAT_RGBA8888;
#endif

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
pg_SavePNG_RW(SDL_Surface *surface, SDL_RWops *dst, int freedst)
{
    if (dst) {
        png_structp png_ptr;
        png_infop info_ptr;
        png_colorp color_ptr = NULL;
        Uint8 transparent_table[256];
        SDL_Surface *source = surface;
        SDL_Palette *palette;
        int png_color_type = PNG_COLOR_TYPE_RGB_ALPHA;

        png_ptr =
            png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
        if (png_ptr == NULL) {
            IMG_SetError(
                "Couldn't allocate memory for PNG file or incompatible PNG "
                "dll");
            return -1;
        }

        info_ptr = png_create_info_struct(png_ptr);
        if (info_ptr == NULL) {
            png_destroy_write_struct(&png_ptr, NULL);
            IMG_SetError("Couldn't create image information for PNG file");
            return -1;
        }
#ifdef PNG_SETJMP_SUPPORTED
#ifndef LIBPNG_VERSION_12
        if (setjmp(*png_set_longjmp_fn(png_ptr, longjmp, sizeof(jmp_buf))))
#else
        if (setjmp(png_ptr->jmpbuf))
#endif
#endif
        {
            png_destroy_write_struct(&png_ptr, &info_ptr);
            IMG_SetError("Error writing the PNG file.");
            return -1;
        }

        palette = surface->format->palette;
        if (palette) {
            const int ncolors = palette->ncolors;
            int i;
            int last_transparent = -1;

            color_ptr = (png_colorp)SDL_malloc(sizeof(png_colorp) * ncolors);
            if (color_ptr == NULL) {
                png_destroy_write_struct(&png_ptr, &info_ptr);
                IMG_SetError("Couldn't create palette for PNG file");
                return -1;
            }
            for (i = 0; i < ncolors; i++) {
                color_ptr[i].red = palette->colors[i].r;
                color_ptr[i].green = palette->colors[i].g;
                color_ptr[i].blue = palette->colors[i].b;
                if (palette->colors[i].a != 255) {
                    last_transparent = i;
                }
            }
            png_set_PLTE(png_ptr, info_ptr, color_ptr, ncolors);
            png_color_type = PNG_COLOR_TYPE_PALETTE;

            if (last_transparent >= 0) {
                for (i = 0; i <= last_transparent; ++i) {
                    transparent_table[i] = palette->colors[i].a;
                }
                png_set_tRNS(png_ptr, info_ptr, transparent_table,
                             last_transparent + 1, NULL);
            }
        }
        else if (surface->format->format == SDL_PIXELFORMAT_RGB24) {
            /* If the surface is exactly the right RGB format it is just passed
             * through */
            png_color_type = PNG_COLOR_TYPE_RGB;
        }
        else if (!SDL_ISPIXELFORMAT_ALPHA(surface->format->format)) {
            /* If the surface is not exactly the right RGB format but does not
               have alpha information, it should be converted to RGB24 before
               being passed through */
            png_color_type = PNG_COLOR_TYPE_RGB;
            source =
                SDL_ConvertSurfaceFormat(surface, SDL_PIXELFORMAT_RGB24, 0);
        }
        else if (surface->format->format != png_format) {
            /* Otherwise, (surface has alpha data), and it is not in the exact
               right format , so it should be converted to that */
            source = SDL_ConvertSurfaceFormat(surface, png_format, 0);
        }

        png_set_write_fn(png_ptr, dst, png_write_fn, png_flush_fn);

        png_set_IHDR(png_ptr, info_ptr, surface->w, surface->h, 8,
                     png_color_type, PNG_INTERLACE_NONE,
                     PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

        if (source) {
            png_bytep *row_pointers;
            int row;

            row_pointers =
                (png_bytep *)SDL_malloc(sizeof(png_bytep) * source->h);
            if (!row_pointers) {
                png_destroy_write_struct(&png_ptr, &info_ptr);
                IMG_SetError("Out of memory");
                return -1;
            }
            for (row = 0; row < (int)source->h; row++) {
                row_pointers[row] =
                    (png_bytep)(Uint8 *)source->pixels + row * source->pitch;
            }

            png_set_rows(png_ptr, info_ptr, row_pointers);
            png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);

            SDL_free(row_pointers);
            if (source != surface) {
                SDL_FreeSurface(source);
            }
        }
        png_destroy_write_struct(&png_ptr, &info_ptr);
        if (color_ptr) {
            SDL_free(color_ptr);
        }
        if (freedst) {
            SDL_RWclose(dst);
        }
    }
    else {
        IMG_SetError("Passed NULL dst");
        return -1;
    }
    return 0;
}

int
pg_SavePNG(SDL_Surface *surface, const char *file)
{
    SDL_RWops *dst = SDL_RWFromFile(file, "wb");
    if (dst) {
        return pg_SavePNG_RW(surface, dst, 1);
    }
    else {
        return -1;
    }
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

    oencoded = pg_EncodeString(obj, "UTF-8", NULL, pgExc_SDLError);
    if (oencoded == NULL) {
        result = -2;
    }
    else if (oencoded == Py_None) {
        rw = pgRWops_FromFileObject(obj);
        if (rw == NULL) {
            PyErr_Format(PyExc_TypeError,
                         "Expected a string or file object for the file "
                         "argument: got %.1024s",
                         Py_TYPE(obj)->tp_name);
            result = -2;
        }
        else {
            name = namehint;
        }
    }
    else {
        name = PyBytes_AS_STRING(oencoded);
    }

    if (result > 0) {
        char *ext = iext_find_extension(name);
        if (!strcasecmp(ext, "jpeg") || !strcasecmp(ext, "jpg")) {
            if (rw != NULL) {
                result = IMG_SaveJPG_RW(surf, rw, 0, JPEG_QUALITY);
            }
            else {
                result = IMG_SaveJPG(surf, name, JPEG_QUALITY);
            }
        }
        else if (!strcasecmp(ext, "png")) {
#ifdef PNG_H
            /*Py_BEGIN_ALLOW_THREADS; */
            if (rw != NULL) {
                result = pg_SavePNG_RW(surf, rw, 0);
            }
            else {
                result = pg_SavePNG(surf, name);
            }
            /*Py_END_ALLOW_THREADS; */
#else
            PyErr_SetString(pgExc_SDLError, "No support for png compiled in.");
            result = -2;
#endif /* ~PNG_H */
        }
    }

    pgSurface_Unprep(surfobj);

    Py_XDECREF(oencoded);
    if (result == -2) {
        /* Python error raised elsewhere */
        return NULL;
    }
    if (result == -1) {
        /* SDL error: translate to Python error */
        return RAISE(pgExc_SDLError, SDL_GetError());
    }
    if (result == 1) {
        return RAISE(pgExc_SDLError, "Unrecognized image type");
    }

    Py_RETURN_NONE;
}

static PyObject *
imageext_get_sdl_image_version(PyObject *self, PyObject *_null)
{
    return Py_BuildValue("iii", SDL_IMAGE_MAJOR_VERSION,
                         SDL_IMAGE_MINOR_VERSION, SDL_IMAGE_PATCHLEVEL);
}

/*
static void
_imageext_free(void *ptr)
{
#ifdef WITH_THREAD
    if (_pg_img_mutex) {
        SDL_DestroyMutex(_pg_img_mutex);
        _pg_img_mutex = 0;
    }
#endif
}
*/

static PyMethodDef _imageext_methods[] = {
    {"load_extended", image_load_ext, METH_VARARGS, DOC_PYGAMEIMAGE},
    {"save_extended", image_save_ext, METH_VARARGS, DOC_PYGAMEIMAGE},
    {"_get_sdl_image_version", imageext_get_sdl_image_version, METH_NOARGS,
     "_get_sdl_image_version() -> (major, minor, patch)\n"
     "Note: Should not be used directly."},
    {NULL, NULL, 0, NULL}};

/*DOC*/ static char _imageext_doc[] =
    /*DOC*/ "additional image loaders";

MODINIT_DEFINE(imageext)
{
    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "imageext",
                                         _imageext_doc,
                                         -1,
                                         _imageext_methods,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL}; /* _imageext_free commented */

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

    /*
    #ifdef WITH_THREAD
        _pg_img_mutex = SDL_CreateMutex();
        if (!_pg_img_mutex) {
            PyErr_SetString(pgExc_SDLError, SDL_GetError());
            return NULL;
        }
    #endif
    */

    /* create the module */
    return PyModule_Create(&_module);
}
