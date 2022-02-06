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

#include <jerror.h>
#include <jpeglib.h>

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
write_png(const char *file_name, SDL_RWops *rw, png_bytep *rows, int w, int h,
          int colortype, int bitdepth)
{
    png_structp png_ptr = NULL;
    png_infop info_ptr = NULL;
    SDL_RWops *rwops;
    char *doing;

    if (rw == NULL) {
        if (!(rwops = SDL_RWFromFile(file_name, "wb"))) {
            return -1;
        }
    }
    else {
        rwops = rw;
    }

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

    /* doing = "write info"; */
    png_write_info(png_ptr, info_ptr);

    /* doing = "write image"; */
    png_write_image(png_ptr, rows);

    /* doing = "write end"; */
    png_write_end(png_ptr, NULL);

    if (rw == NULL) {
        doing = "close file";
        if (0 != SDL_RWclose(rwops))
            goto fail;
    }
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
SavePNG(SDL_Surface *surface, const char *file, SDL_RWops *rw)
{
    static unsigned char **ss_rows;
    static int ss_size;
    static int ss_w, ss_h;
    SDL_Surface *ss_surface;
    SDL_Rect ss_rect;
    int r, i;
    int alpha = 0;

    Uint8 surf_alpha = 255;
    Uint32 surf_colorkey;
    int has_colorkey = 0;
    SDL_BlendMode surf_mode;

    ss_rows = 0;
    ss_size = 0;
    ss_surface = NULL;

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

    if (alpha) {
        r = write_png(file, rw, ss_rows, surface->w, surface->h,
                      PNG_COLOR_TYPE_RGB_ALPHA, 8);
    }
    else {
        r = write_png(file, rw, ss_rows, surface->w, surface->h,
                      PNG_COLOR_TYPE_RGB, 8);
    }

    free(ss_rows);
    SDL_FreeSurface(ss_surface);
    ss_surface = NULL;
    return r;
}

#endif /* end if PNG_H */

#ifdef JPEGLIB_H

#define NUM_LINES_TO_WRITE 500

/* Avoid conflicts with the libjpeg libraries C runtime bindings.
 * Adapted from code in the libjpeg file jdatadst.c .
 */

#define OUTPUT_BUF_SIZE 4096 /* choose an efficiently fwrite'able size */

/* Expanded data destination object for stdio output */
typedef struct {
    struct jpeg_destination_mgr pub; /* public fields */

    SDL_RWops *outfile; /* target stream */
    JOCTET *buffer;     /* start of buffer */
} j_outfile_mgr;

static void
j_init_destination(j_compress_ptr cinfo)
{
    j_outfile_mgr *dest = (j_outfile_mgr *)cinfo->dest;

    /* Allocate the output buffer --- it will be released when done with
     * image
     */
    dest->buffer = (JOCTET *)(*cinfo->mem->alloc_small)(
        (j_common_ptr)cinfo, JPOOL_IMAGE, OUTPUT_BUF_SIZE * sizeof(JOCTET));

    dest->pub.next_output_byte = dest->buffer;
    dest->pub.free_in_buffer = OUTPUT_BUF_SIZE;
}

static boolean
j_empty_output_buffer(j_compress_ptr cinfo)
{
    j_outfile_mgr *dest = (j_outfile_mgr *)cinfo->dest;

    if (SDL_RWwrite(dest->outfile, dest->buffer, 1, OUTPUT_BUF_SIZE) !=
        (size_t)OUTPUT_BUF_SIZE) {
        ERREXIT(cinfo, JERR_FILE_WRITE);
    }
    dest->pub.next_output_byte = dest->buffer;
    dest->pub.free_in_buffer = OUTPUT_BUF_SIZE;

    return 1;
}

static void
j_term_destination(j_compress_ptr cinfo)
{
    j_outfile_mgr *dest = (j_outfile_mgr *)cinfo->dest;
    size_t datacount = OUTPUT_BUF_SIZE - dest->pub.free_in_buffer;

    /* Write any data remaining in the buffer */
    if (datacount > 0) {
        if (SDL_RWwrite(dest->outfile, dest->buffer, 1, datacount) !=
            datacount) {
            ERREXIT(cinfo, JERR_FILE_WRITE);
        }
    }
    if (pg_RWflush(dest->outfile)) {
        ERREXIT(cinfo, JERR_FILE_WRITE);
    }
}

static void
j_stdio_dest(j_compress_ptr cinfo, SDL_RWops *outfile)
{
    j_outfile_mgr *dest;

    /* The destination object is made permanent so that multiple JPEG
     * images can be written to the same file without re-executing
     * jpeg_stdio_dest. This makes it dangerous to use this manager and a
     * different destination manager serially with the same JPEG object,
     * because their private object sizes may be different.  Caveat
     * programmer.
     */
    if (cinfo->dest == NULL) { /* first time for this JPEG object? */
        cinfo->dest =
            (struct jpeg_destination_mgr *)(*cinfo->mem->alloc_small)(
                (j_common_ptr)cinfo, JPOOL_PERMANENT, sizeof(j_outfile_mgr));
    }

    dest = (j_outfile_mgr *)cinfo->dest;
    dest->pub.init_destination = j_init_destination;
    dest->pub.empty_output_buffer = j_empty_output_buffer;
    dest->pub.term_destination = j_term_destination;
    dest->outfile = outfile;
}

/* End borrowed code
 */

int
write_jpeg(const char *file_name, unsigned char **image_buffer,
           int image_width, int image_height, int quality)
{
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    SDL_RWops *outfile;
    JSAMPROW row_pointer[NUM_LINES_TO_WRITE];
    JDIMENSION i, num_lines_to_write;

    num_lines_to_write = NUM_LINES_TO_WRITE;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    if (!(outfile = SDL_RWFromFile(file_name, "wb"))) {
        return -1;
    }
    j_stdio_dest(&cinfo, outfile);

    cinfo.image_width = image_width;
    cinfo.image_height = image_height;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;
    /* cinfo.optimize_coding = FALSE;
     */
    /* cinfo.optimize_coding = FALSE;
     */

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, 1);

    jpeg_start_compress(&cinfo, 1);

    /* try and write many scanlines at once.  */
    while (cinfo.next_scanline < cinfo.image_height) {
        if (num_lines_to_write >
            (cinfo.image_height - cinfo.next_scanline) - 1) {
            num_lines_to_write = (cinfo.image_height - cinfo.next_scanline);
        }
        /* copy the memory from the buffers */
        for (i = 0; i < num_lines_to_write; i++) {
            row_pointer[i] = image_buffer[cinfo.next_scanline + i];
        }

        jpeg_write_scanlines(&cinfo, row_pointer, num_lines_to_write);
    }

    jpeg_finish_compress(&cinfo);
    SDL_RWclose(outfile);
    jpeg_destroy_compress(&cinfo);
    return 0;
}

int
SaveJPEG(SDL_Surface *surface, const char *file)
{
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
#define RED_MASK 0xff0000
#define GREEN_MASK 0xff00
#define BLUE_MASK 0xff
#else
#define RED_MASK 0xff
#define GREEN_MASK 0xff00
#define BLUE_MASK 0xff0000
#endif

    static unsigned char **ss_rows;
    static int ss_size;
    static int ss_w, ss_h;
    SDL_Surface *ss_surface;
    SDL_Rect ss_rect;
    int r, i;
    int pixel_bits = 32;
    int free_ss_surface = 1;

    if (!surface) {
        return -1;
    }

    ss_rows = 0;
    ss_size = 0;
    ss_surface = NULL;

    ss_w = surface->w;
    ss_h = surface->h;

    pixel_bits = 24;

    /* See if the Surface is suitable for using directly.
       So no conversion is needed.  24bit, RGB
    */

    if (surface->format->format == SDL_PIXELFORMAT_RGB24) {
        /*
           printf("not creating...\n");
        */
        ss_surface = surface;

        free_ss_surface = 0;
    }
    else {
        /*
        printf("creating...\n");
        */

        /* If it is not, then we need to make a new surface.
         */

        ss_surface = SDL_CreateRGBSurface(0, ss_w, ss_h, pixel_bits, RED_MASK,
                                          GREEN_MASK, BLUE_MASK, 0);
        if (ss_surface == NULL) {
            return -1;
        }

        ss_rect.x = 0;
        ss_rect.y = 0;
        ss_rect.w = ss_w;
        ss_rect.h = ss_h;
        SDL_BlitSurface(surface, &ss_rect, ss_surface, NULL);

        free_ss_surface = 1;
    }

#ifdef _MSC_VER
    /* Make MSVC static analyzer happy by assuring ss_size >= 2 to supress
     * a false analyzer report */
    __analysis_assume(ss_size >= 2);
#endif

    ss_size = ss_h;
    ss_rows = (unsigned char **)malloc(sizeof(unsigned char *) * ss_size);
    if (ss_rows == NULL) {
        /* clean up the allocated surface too */
        if (free_ss_surface) {
            SDL_FreeSurface(ss_surface);
        }
        return -1;
    }

    /* copy pointers to the scanlines... since they might not be packed.
     */
    for (i = 0; i < ss_h; i++) {
        ss_rows[i] =
            ((unsigned char *)ss_surface->pixels) + i * ss_surface->pitch;
    }
    r = write_jpeg(file, ss_rows, surface->w, surface->h, JPEG_QUALITY);

    free(ss_rows);

    if (free_ss_surface) {
        SDL_FreeSurface(ss_surface);
        ss_surface = NULL;
    }
    return r;
}

#endif /* end if JPEGLIB_H */

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
        char *ext = find_extension(name);
        if (!strcasecmp(ext, "jpeg") || !strcasecmp(ext, "jpg")) {
#if (SDL_IMAGE_MAJOR_VERSION * 1000 + SDL_IMAGE_MINOR_VERSION * 100 + \
     SDL_IMAGE_PATCHLEVEL) < 2002
            /* SDL_Image is a version less than 2.0.2 and therefore does
             * not have the functions IMG_SaveJPG() and IMG_SaveJPG_RW().
             */
            if (rw != NULL) {
                PyErr_SetString(pgExc_SDLError,
                                "SDL_Image 2.0.2 or newer needed to save "
                                "jpeg to a fileobject.");
                result = -2;
            }
            else {
#ifdef JPEGLIB_H
                /* jpg save functions seem *NOT* thread safe at least on
                 * windows.
                 */
                /*
                Py_BEGIN_ALLOW_THREADS;
                */
                result = SaveJPEG(surf, name);
                /*
                Py_END_ALLOW_THREADS;
                */
#else
                PyErr_SetString(pgExc_SDLError,
                                "No support for jpg compiled in.");
                result = -2;
#endif /* ~JPEGLIB_H */
            }
#else
            /* SDL_Image is version 2.0.2 or newer and therefore does
             * have the functions IMG_SaveJPG() and IMG_SaveJPG_RW().
             */
            if (rw != NULL) {
                result = IMG_SaveJPG_RW(surf, rw, 0, JPEG_QUALITY);
            }
            else {
                result = IMG_SaveJPG(surf, name, JPEG_QUALITY);
#ifdef JPEGLIB_H
                /* In the unlikely event that pygame is compiled with support
                 * for jpg but SDL_Image was not, then we can catch that and
                 * try calling the pygame SaveJPEG function.
                 */
                if (result == -1) {
                    if (strstr(SDL_GetError(), "not built with jpeglib") !=
                        NULL) {
                        SDL_ClearError();
                        result = SaveJPEG(surf, name);
                    }
                }
#endif /* JPEGLIB_H */
            }
#endif /* SDL_Image >= 2.0.2 */
        }
        else if (!strcasecmp(ext, "png")) {
#ifdef PNG_H
            /*Py_BEGIN_ALLOW_THREADS; */
            result = SavePNG(surf, name, rw);
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
image_get_sdl_image_version(PyObject *self, PyObject *arg)
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
