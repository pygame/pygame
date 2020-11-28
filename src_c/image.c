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
 *  image module for pygame
 */
#include "pygame.h"

#include "pgcompat.h"

#include "doc/image_doc.h"

#if PG_COMPILE_SSE4_2 && SDL_VERSION_ATLEAST(2, 0, 0)
#include <emmintrin.h>
/* SSSE 3 */
#include <tmmintrin.h>
#endif

#if IS_SDLv1
#include "pgopengl.h"
#endif /* IS_SDLv1 */

static int
SaveTGA(SDL_Surface *surface, const char *file, int rle);
static int
SaveTGA_RW(SDL_Surface *surface, SDL_RWops *out, int rle);
#if IS_SDLv1
static SDL_Surface *
opengltosdl(void);
#endif /* IS_SDLv1 */

#define DATAROW(data, row, width, height, flipped)             \
    ((flipped) ? (((char *)data) + (height - row - 1) * width) \
               : (((char *)data) + row * width))

static PyObject *extloadobj = NULL;
static PyObject *extsaveobj = NULL;
static PyObject *extverobj = NULL;

static const char *
find_extension(const char *fullname)
{
    const char *dot;

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
image_load_basic(PyObject *self, PyObject *obj)
{
    PyObject *final;
    PyObject *oencoded;
    SDL_Surface *surf;
    SDL_RWops *rw;
    
    oencoded = pg_EncodeString(obj, "UTF-8", NULL, pgExc_SDLError);
    if (oencoded == NULL) {
        return NULL;
    }
    
    if (oencoded != Py_None) {
        Py_BEGIN_ALLOW_THREADS;
        surf = SDL_LoadBMP(Bytes_AS_STRING(oencoded));
        Py_END_ALLOW_THREADS;
        Py_DECREF(oencoded);
    }
    else {
        Py_DECREF(oencoded);
        rw = pgRWops_FromFileObject(obj);
        if (rw == NULL) {
            return NULL;
        }
        Py_BEGIN_ALLOW_THREADS;
        surf = SDL_LoadBMP_RW(rw, 1);
        Py_END_ALLOW_THREADS;
    }

    if (surf == NULL) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    final = (PyObject *)pgSurface_New(surf);
    if (final == NULL) {
        SDL_FreeSurface(surf);
    }
    return final;
}

static PyObject *
image_load_extended(PyObject *self, PyObject *arg)
{
    if (extloadobj == NULL)
        return RAISE(PyExc_NotImplementedError, 
                     "loading images of extended format is not available");
    else
        return PyObject_CallObject(extloadobj, arg);
}

static PyObject *
image_load(PyObject *self, PyObject *arg)
{
    PyObject *obj;
    const char *name = NULL;
    
    if (extloadobj == NULL) {
        if (!PyArg_ParseTuple(arg, "O|s", &obj, &name)) {
            return NULL;
        }
        return image_load_basic(self, obj);
    }
    else
        return image_load_extended(self, arg);
}

#if IS_SDLv1
static SDL_Surface *
opengltosdl()
{
    /*we need to get ahold of the pyopengl glReadPixels function*/
    /*we use pyopengl's so we don't need to link with opengl at compiletime*/
    SDL_Surface *surf = NULL;
    Uint32 rmask, gmask, bmask;
    int i;
    unsigned char *pixels = NULL;

    GL_glReadPixels_Func p_glReadPixels = NULL;

    p_glReadPixels =
        (GL_glReadPixels_Func)SDL_GL_GetProcAddress("glReadPixels");

    surf = SDL_GetVideoSurface();

    if (!surf) {
        return (SDL_Surface *)RAISE(PyExc_RuntimeError,
                                    "Cannot get video surface.");

    }

    if (!p_glReadPixels) {
        return (SDL_Surface *)RAISE(PyExc_RuntimeError,
                                    "Cannot find glReadPixels function.");

    }

    /*
    GL_UNSIGNED_BYTE = 5121
    GL_RGB = 6407
    */

    pixels = (unsigned char *)malloc(surf->w * surf->h * 3);

    if (!pixels) {
        return (SDL_Surface *)RAISE(
                                  PyExc_MemoryError,
                                  "Cannot allocate enough memory for pixels.");

    }
    // p_glReadPixels(0, 0, surf->w, surf->h, 6407, 5121, pixels);
    // glReadPixels(0, 0, surf->w, surf->h, 0x1907, 0x1401, pixels);
    p_glReadPixels(0, 0, surf->w, surf->h, 0x1907, 0x1401, pixels);

    if (SDL_BYTEORDER == SDL_LIL_ENDIAN) {
        rmask = 0x000000FF;
        gmask = 0x0000FF00;
        bmask = 0x00FF0000;
    }
    else {
        rmask = 0x00FF0000;
        gmask = 0x0000FF00;
        bmask = 0x000000FF;
    }
    surf = SDL_CreateRGBSurface(SDL_SWSURFACE, surf->w, surf->h, 24, rmask,
                                gmask, bmask, 0);
    if (!surf) {
        free(pixels);
        return (SDL_Surface *)RAISE(pgExc_SDLError, SDL_GetError());
    }

    for (i = 0; i < surf->h; ++i) {
        memcpy(((char *)surf->pixels) + surf->pitch * i,
               pixels + 3 * surf->w * (surf->h - i - 1), surf->w * 3);
    }

    free(pixels);
    return surf;
}
#endif /* IS_SDLv1 */

#ifdef WIN32
#define strcasecmp _stricmp
#else
#include <strings.h>
#endif

static PyObject *
image_save_extended(PyObject *self, PyObject *arg)
{
    if (extsaveobj == NULL)
        return RAISE(PyExc_NotImplementedError, 
                     "saving images of extended format is not available");
    else
        return PyObject_CallObject(extsaveobj, arg);
}

static PyObject *
image_save(PyObject *self, PyObject *arg)
{
    pgSurfaceObject *surfobj;
    PyObject *obj;
    const char *namehint = NULL;
    PyObject *oencoded;
    PyObject *ret;
    SDL_Surface *surf;
    int result = 1;
#if IS_SDLv1
    SDL_Surface *temp = NULL;
#endif /* IS_SDLv1 */

    if (!PyArg_ParseTuple(arg, "O!O|s", &pgSurface_Type, &surfobj,
                &obj, &namehint)) {
        return NULL;
    }

    surf = pgSurface_AsSurface(surfobj);
#if IS_SDLv1
    if (surf->flags & SDL_OPENGL) {
        temp = surf = opengltosdl();
        if (surf == NULL) {
            return NULL;
        }
    }
    else {
        pgSurface_Prep(surfobj);
    }
#else  /* IS_SDLv2 */
    pgSurface_Prep(surfobj);
#endif /* IS_SDLv2 */

    oencoded = pg_EncodeString(obj, "UTF-8", NULL, pgExc_SDLError);
    if (oencoded == NULL) {
        result = -2;
    }
    else {
        const char *name = NULL;
        const char * ext = NULL;
        if (oencoded == Py_None) {
            name = (namehint ? namehint: "tga");
        }
        else {
            name = Bytes_AS_STRING(oencoded);
        }
        
        ext = find_extension(name);
        if (!strcasecmp(ext, "png") ||
                !strcasecmp(ext, "jpg") ||
                !strcasecmp(ext, "jpeg")) {
            /* If it is .png .jpg .jpeg use the extended module. */
            /* try to get extended formats */
            ret = image_save_extended(self, arg);
            result = (ret == NULL ? -2 : 0);
        }
        else if (oencoded == Py_None) {
            SDL_RWops *rw = pgRWops_FromFileObject(obj);
            if (rw != NULL) {
                if (!strcasecmp(ext, "bmp")) {
                    /* The SDL documentation didn't specify which negative number
                     * is returned upon error. We want to be sure that result is
                     * either 0 or -1: */
                    result = (SDL_SaveBMP_RW(surf, rw, 0) == 0 ? 0 : -1);
                }
                else {
                    result = SaveTGA_RW(surf, rw, 1);
                }
            }
            else {
                result = -2;
            }
        }
        else {
            if (!strcasecmp(ext, "bmp")) {
                Py_BEGIN_ALLOW_THREADS;
                /* The SDL documentation didn't specify which negative number
                 * is returned upon error. We want to be sure that result is
                 * either 0 or -1: */
                result = (SDL_SaveBMP(surf, name) == 0 ? 0 : -1);
                Py_END_ALLOW_THREADS;
            }
            else {
                Py_BEGIN_ALLOW_THREADS;
                result = SaveTGA(surf, name, 1);
                Py_END_ALLOW_THREADS;
            }
        }
    }
    Py_XDECREF(oencoded);

#if IS_SDLv1
    if (temp) {
        SDL_FreeSurface(temp);
    }
    else {
        pgSurface_Unprep(surfobj);
    }
#else  /* IS_SDLv2 */
    pgSurface_Unprep(surfobj);
#endif /* IS_SDLv2 */

    if (result == -2) {
        /* Python error raised elsewhere */
        return NULL;
    }
    if (result == -1) {
        /* SDL error: translate to Python error */
        return RAISE(pgExc_SDLError, SDL_GetError());
    }
    if (result == 1) {
        /* Should never get here */
        return RAISE(pgExc_SDLError, "Unrecognized image type");
    }

    Py_RETURN_NONE;
}

static PyObject *
image_get_extended(PyObject *self)
{
    if (extverobj == NULL)
        Py_RETURN_FALSE;
    else
        Py_RETURN_TRUE;
}

static PyObject *
image_get_sdl_image_version(PyObject *self)
{
    if (extverobj == NULL)
        Py_RETURN_NONE;
    else
        return PyObject_CallObject(extverobj, NULL);
}

#if PG_COMPILE_SSE4_2 && SDL_VERSION_ATLEAST(2, 0, 0)
#define SSE42_ALIGN_NEEDED 16
#define SSE42_ALIGN __attribute__((aligned(SSE42_ALIGN_NEEDED)))

#define _SHIFT_N_STEP2ALIGN(shift, step) (shift/8 + step * 4)

#if PYGAME_DEBUG_SSE
/* Useful for debugging/comparing the SSE vectors */
static void _debug_print128_num(__m128i var, const char *msg)
{
    uint32_t val[4];
    memcpy(val, &var, sizeof(val));
    fprintf(stderr, "%s: %04x%04x%04x%04x\n",
           msg, val[0], val[1], val[2], val[3]);
}
#define DEBUG_PRINT128_NUM(var, msg) _debug_print128_num(var, msg)
#else
#define DEBUG_PRINT128_NUM(var, msg) do { /* do nothing */ } while (0)
#endif

/*
 * Generates an SSE vector useful for reordering a SSE vector
 * based on the "in-memory layout" to match the "tostring layout"
 * It is only useful as second parameter to _mm_shuffle_epi8.
 *
 * A short _mm_shuffle_epi8 primer:
 *
 * - If the highest bit of a byte in the reorder vector is set,
 *   the matching byte in the original vector is cleared:
 * - Otherwise, the 3 lowest bit of the byte in the reorder vector
 *   represent the byte index of where to find the relevant value
 *   from the original vector.
 *
 * As an example, given the following in memory layout (bytes):
 *
 *    R1 G1 B1 A1 R2 G2 B2 A2 ...
 *
 * And we want:
 *
 *    A1 R1 G1 B1 A2 R2 G2 B2
 *
 * Then the reorder vector should look like (in hex):
 *
 *    03 00 01 02 07 04 05 06
 *
 * This is exactly the type of vector that compute_align_vector
 * produces (based on the pixel format and where the alpha should
 * be placed in the output).
 */
static PG_INLINE __m128i
compute_align_vector(SDL_PixelFormat *format, int color_offset,
                     int alpha_offset) {
    int output_align[4];
    size_t i;
    size_t limit = sizeof(output_align) / sizeof(int);
    int a_shift = alpha_offset * 8;
    int r_shift = (color_offset + 0) * 8;
    int g_shift = (color_offset + 1) * 8;
    int b_shift = (color_offset + 2) * 8;
    for (i = 0; i < limit; i++) {
        int p = 3 - i;
        output_align[i] = _SHIFT_N_STEP2ALIGN(format->Rshift, p) << r_shift
                        | _SHIFT_N_STEP2ALIGN(format->Gshift, p) << g_shift
                        | _SHIFT_N_STEP2ALIGN(format->Bshift, p) << b_shift
                        | _SHIFT_N_STEP2ALIGN(format->Ashift, p) << a_shift;
    }
    return _mm_set_epi32(output_align[0], output_align[1],
                         output_align[2], output_align[3]);
}


static PG_INLINE PG_FUNCTION_TARGET_SSE4_2 void
tostring_pixels_32bit_sse4(const __m128i *row, __m128i *data, int loop_max,
                           __m128i mask_vector, __m128i align_vector) {
    int w;
    for (w = 0; w < loop_max; ++w) {
        __m128i pvector = _mm_loadu_si128(row + w);
        DEBUG_PRINT128_NUM(pvector, "Load");
        pvector = _mm_and_si128(pvector, mask_vector);
        DEBUG_PRINT128_NUM(pvector, "after _mm_and_si128 (and)");
        pvector = _mm_shuffle_epi8(pvector, align_vector);
        DEBUG_PRINT128_NUM(pvector, "after _mm_shuffle_epi8 (reorder)");
        _mm_storeu_si128(data + w, pvector);
    }
}

/*
 * SSE4.2 variant of tostring_surf_32bpp.
 *
 * It is a lot faster but only works on a subset of the surfaces
 * (plus requires SSE4.2 support from the CPU).
 */
static PG_FUNCTION_TARGET_SSE4_2 void
tostring_surf_32bpp_sse42(SDL_Surface *surf, int flipped, char *data,
                          int color_offset, int alpha_offset) {
    const int step_size = 4;
    int h;
    SDL_PixelFormat *format = surf->format;
    int loop_max = surf->w / step_size;
    int mask = (format->Rloss ? 0 : format->Rmask)
             | (format->Gloss ? 0 : format->Gmask)
             | (format->Bloss ? 0 : format->Bmask)
             | (format->Aloss ? 0 : format->Amask);

    __m128i mask_vector = _mm_set_epi32(mask, mask, mask, mask);
    __m128i align_vector = compute_align_vector(surf->format,
                                                color_offset, alpha_offset);
    /* How much we would overshoot if we overstep loop_max */
    int rollback_count = surf->w % step_size;
    if (rollback_count) {
        rollback_count = step_size - rollback_count;
    }

    DEBUG_PRINT128_NUM(mask_vector, "mask-vector");
    DEBUG_PRINT128_NUM(align_vector, "align-vector");

    /* This code will be horribly wrong if these assumptions do not hold.
     * They are intended as a debug/testing guard to ensure that nothing
     * calls this function without ensuring the assumptions during
     * development
     */
    assert(sizeof(int) == sizeof(Uint32));
    assert(4 * sizeof(Uint32) == sizeof(__m128i));
    /* If this assertion does not hold, the fallback code will overrun
     * the buffers.
     */
    assert(surf->w >= step_size);
    assert(format->Rloss % 8 == 0);
    assert(format->Gloss % 8 == 0);
    assert(format->Bloss % 8 == 0);
    assert(format->Aloss % 8 == 0);

    for (h = 0; h < surf->h; ++h) {
        const char *row = (char *)DATAROW(
            surf->pixels, h, surf->pitch, surf->h, flipped);
        tostring_pixels_32bit_sse4((const __m128i*)row, (__m128i *)data,
                                   loop_max, mask_vector, align_vector);
        row += sizeof(__m128i) * loop_max;
        data += sizeof(__m128i) * loop_max;
        if (rollback_count) {
            /* Back up a bit to ensure we stay within the memory boundaries
             * Technically, we end up redoing part of the computations, but
             * it does not really matter as the runtime of these operations
             * are fixed and the results are deterministic.
             */
            row -= rollback_count * sizeof(Uint32);
            data -= rollback_count * sizeof(Uint32);

            tostring_pixels_32bit_sse4((const __m128i*)row, (__m128i *)data,
                                       1, mask_vector, align_vector);

            row += sizeof(__m128i);
            data += sizeof(__m128i);
        }
    }
}
#endif /* PG_COMPILE_SSE4_2  && SDL_VERSION_ATLEAST(2, 0, 0) */


#if IS_SDLv2
static void
tostring_surf_32bpp(SDL_Surface *surf, int flipped,
                    int hascolorkey, Uint32 colorkey,
                    char *serialized_image,
                    int color_offset, int alpha_offset
)
#else
static void
tostring_surf_32bpp(SDL_Surface *surf, int flipped,
                    int hascolorkey, int colorkey,
                    char *serialized_image,
                    int color_offset, int alpha_offset
)
#endif /* !IS_SDLv2*/
{
    int w, h;

    Uint32 Rmask = surf->format->Rmask;
    Uint32 Gmask = surf->format->Gmask;
    Uint32 Bmask = surf->format->Bmask;
    Uint32 Amask = surf->format->Amask;
    Uint32 Rshift = surf->format->Rshift;
    Uint32 Gshift = surf->format->Gshift;
    Uint32 Bshift = surf->format->Bshift;
    Uint32 Ashift = surf->format->Ashift;
    Uint32 Rloss = surf->format->Rloss;
    Uint32 Gloss = surf->format->Gloss;
    Uint32 Bloss = surf->format->Bloss;
    Uint32 Aloss = surf->format->Aloss;

#if PG_COMPILE_SSE4_2 && SDL_VERSION_ATLEAST(2, 0, 0)
    if (/* SDL uses Uint32, SSE uses int for building vectors.
         * Related, we assume that Uint32 is packed so 4 of
         * them perfectly matches an __m128i.
         * If these assumptions do not match up, we will
         * produce incorrect results.
         */
        sizeof(int) == sizeof(Uint32)
        && 4 * sizeof(Uint32) == sizeof(__m128i)
        && !hascolorkey /* No color key */
        && SDL_HasSSE42() == SDL_TRUE
        /* The SSE code assumes it will always read at least 4 pixels */
        && surf->w >= 4
        /* Our SSE code assumes masks are at most 0xff */
        && (surf->format->Rmask >> surf->format->Rshift) <= 0x0ff
        && (surf->format->Gmask >> surf->format->Gshift) <= 0x0ff
        && (surf->format->Bmask >> surf->format->Bshift) <= 0x0ff
        && (Amask >> Ashift) <= 0x0ff
        /* Our SSE code cannot handle losses other than 0 or 8
         * Note the mask check above ensures that losses can be
         * at most be 8 (assuming the pixel format makes sense
         * at all).
         */
        && (surf->format->Rloss % 8) == 0
        && (surf->format->Bloss % 8) == 0
        && (surf->format->Gloss % 8) == 0
        && (Aloss % 8) == 0
        ) {
        tostring_surf_32bpp_sse42(surf, flipped, serialized_image,
                                  color_offset, alpha_offset);
        return;
    }
#endif /* PG_COMPILE_SSE4_2 && SDL_VERSION_ATLEAST(2, 0, 0) */

    for (h = 0; h < surf->h; ++h) {
        Uint32 *pixel_row = (Uint32 *)DATAROW(
            surf->pixels, h, surf->pitch, surf->h, flipped);
        for (w = 0; w < surf->w; ++w) {
            Uint32 color = *pixel_row++;
            serialized_image[color_offset + 0] =
                 (char)(((color & Rmask) >> Rshift) << Rloss);
            serialized_image[color_offset + 1] =
                 (char)(((color & Gmask) >> Gshift) << Gloss);
            serialized_image[color_offset + 2] =
                 (char)(((color & Bmask) >> Bshift) << Bloss);
            serialized_image[alpha_offset] =
                hascolorkey
                    ? (char)(color != colorkey) * 255
                    : (char)(Amask ? (((color & Amask) >> Ashift)
                                      << Aloss)
                                   : 255);
            serialized_image += 4;
        }
    }
}

PyObject *
image_tostring(PyObject *self, PyObject *arg)
{
    pgSurfaceObject *surfobj = NULL;
    PyObject *string = NULL;
    char *format, *data, *pixels;
    SDL_Surface *surf;
    int w, h, flipped = 0;
    Py_ssize_t len;
    Uint32 Rmask, Gmask, Bmask, Amask, Rshift, Gshift, Bshift, Ashift, Rloss,
        Gloss, Bloss, Aloss;
    int hascolorkey;
#if IS_SDLv1
    SDL_Surface *temp = NULL;
    int color, colorkey;
#else  /* IS_SDLv2 */
    Uint32 color, colorkey;
#endif /* IS_SDLv2 */
    Uint32 alpha;

    if (!PyArg_ParseTuple(arg, "O!s|i", &pgSurface_Type, &surfobj, &format,
                          &flipped))
        return NULL;
    surf = pgSurface_AsSurface(surfobj);
#if IS_SDLv1
    if (surf->flags & SDL_OPENGL) {
        temp = surf = opengltosdl();
        if (!surf)
            return NULL;
    }
#endif /* IS_SDLv1 */

    Rmask = surf->format->Rmask;
    Gmask = surf->format->Gmask;
    Bmask = surf->format->Bmask;
    Amask = surf->format->Amask;
    Rshift = surf->format->Rshift;
    Gshift = surf->format->Gshift;
    Bshift = surf->format->Bshift;
    Ashift = surf->format->Ashift;
    Rloss = surf->format->Rloss;
    Gloss = surf->format->Gloss;
    Bloss = surf->format->Bloss;
    Aloss = surf->format->Aloss;
#if IS_SDLv1
    hascolorkey = (surf->flags & SDL_SRCCOLORKEY) && !Amask;
    colorkey = surf->format->colorkey;
#else  /* IS_SDLv2 */
    hascolorkey = (SDL_GetColorKey(surf, &colorkey) == 0);
#endif /* IS_SDLv2 */

    if (!strcmp(format, "P")) {
        if (surf->format->BytesPerPixel != 1)
            return RAISE(
                PyExc_ValueError,
                "Can only create \"P\" format data with 8bit Surfaces");
        string = Bytes_FromStringAndSize(NULL, (Py_ssize_t)surf->w * surf->h);
        if (!string)
            return NULL;
        Bytes_AsStringAndSize(string, &data, &len);

        pgSurface_Lock(surfobj);
        pixels = (char *)surf->pixels;
        for (h = 0; h < surf->h; ++h)
            memcpy(DATAROW(data, h, surf->w, surf->h, flipped),
                   pixels + (h * surf->pitch), surf->w);
        pgSurface_Unlock(surfobj);
    }
    else if (!strcmp(format, "RGB")) {
        string =
            Bytes_FromStringAndSize(NULL, (Py_ssize_t)surf->w * surf->h * 3);
        if (!string)
            return NULL;
        Bytes_AsStringAndSize(string, &data, &len);

#if IS_SDLv1
        if (!temp)
            pgSurface_Lock(surfobj);
#else  /* IS_SDLv2 */
        pgSurface_Lock(surfobj);
#endif /* IS_SDLv2 */

        pixels = (char *)surf->pixels;
        switch (surf->format->BytesPerPixel) {
            case 1:
                for (h = 0; h < surf->h; ++h) {
                    Uint8 *ptr = (Uint8 *)DATAROW(surf->pixels, h, surf->pitch,
                                                  surf->h, flipped);
                    for (w = 0; w < surf->w; ++w) {
                        color = *ptr++;
                        data[0] = (char)surf->format->palette->colors[color].r;
                        data[1] = (char)surf->format->palette->colors[color].g;
                        data[2] = (char)surf->format->palette->colors[color].b;
                        data += 3;
                    }
                }
                break;
            case 2:
                for (h = 0; h < surf->h; ++h) {
                    Uint16 *ptr = (Uint16 *)DATAROW(
                        surf->pixels, h, surf->pitch, surf->h, flipped);
                    for (w = 0; w < surf->w; ++w) {
                        color = *ptr++;
                        data[0] = (char)(((color & Rmask) >> Rshift) << Rloss);
                        data[1] = (char)(((color & Gmask) >> Gshift) << Gloss);
                        data[2] = (char)(((color & Bmask) >> Bshift) << Bloss);
                        data += 3;
                    }
                }
                break;
            case 3:
                for (h = 0; h < surf->h; ++h) {
                    Uint8 *ptr = (Uint8 *)DATAROW(surf->pixels, h, surf->pitch,
                                                  surf->h, flipped);
                    for (w = 0; w < surf->w; ++w) {
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                        color = ptr[0] + (ptr[1] << 8) + (ptr[2] << 16);
#else
                        color = ptr[2] + (ptr[1] << 8) + (ptr[0] << 16);
#endif
                        ptr += 3;
                        data[0] = (char)(((color & Rmask) >> Rshift) << Rloss);
                        data[1] = (char)(((color & Gmask) >> Gshift) << Gloss);
                        data[2] = (char)(((color & Bmask) >> Bshift) << Bloss);
                        data += 3;
                    }
                }
                break;
            case 4:
                for (h = 0; h < surf->h; ++h) {
                    Uint32 *ptr = (Uint32 *)DATAROW(
                        surf->pixels, h, surf->pitch, surf->h, flipped);
                    for (w = 0; w < surf->w; ++w) {
                        color = *ptr++;
                        data[0] = (char)(((color & Rmask) >> Rshift) << Rloss);
                        data[1] = (char)(((color & Gmask) >> Gshift) << Rloss);
                        data[2] = (char)(((color & Bmask) >> Bshift) << Rloss);
                        data += 3;
                    }
                }
                break;
        }

#if IS_SDLv1
        if (!temp)
            pgSurface_Unlock(surfobj);
#else  /* IS_SDLv2 */
        pgSurface_Unlock(surfobj);
#endif /* IS_SDLv2 */
    }
    else if (!strcmp(format, "RGBX") || !strcmp(format, "RGBA")) {
        if (strcmp(format, "RGBA"))
            hascolorkey = 0;

        string =
            Bytes_FromStringAndSize(NULL, (Py_ssize_t)surf->w * surf->h * 4);
        if (!string)
            return NULL;
        Bytes_AsStringAndSize(string, &data, &len);

        pgSurface_Lock(surfobj);
        pixels = (char *)surf->pixels;
        switch (surf->format->BytesPerPixel) {
            case 1:
                for (h = 0; h < surf->h; ++h) {
                    Uint8 *ptr = (Uint8 *)DATAROW(surf->pixels, h, surf->pitch,
                                                  surf->h, flipped);
                    for (w = 0; w < surf->w; ++w) {
                        color = *ptr++;
                        data[0] = (char)surf->format->palette->colors[color].r;
                        data[1] = (char)surf->format->palette->colors[color].g;
                        data[2] = (char)surf->format->palette->colors[color].b;
                        data[3] = hascolorkey ? (char)(color != colorkey) * 255
                                              : (char)255;
                        data += 4;
                    }
                }
                break;
            case 2:
                for (h = 0; h < surf->h; ++h) {
                    Uint16 *ptr = (Uint16 *)DATAROW(
                        surf->pixels, h, surf->pitch, surf->h, flipped);
                    for (w = 0; w < surf->w; ++w) {
                        color = *ptr++;
                        data[0] = (char)(((color & Rmask) >> Rshift) << Rloss);
                        data[1] = (char)(((color & Gmask) >> Gshift) << Gloss);
                        data[2] = (char)(((color & Bmask) >> Bshift) << Bloss);
                        data[3] =
                            hascolorkey
                                ? (char)(color != colorkey) * 255
                                : (char)(Amask ? (((color & Amask) >> Ashift)
                                                  << Aloss)
                                               : 255);
                        data += 4;
                    }
                }
                break;
            case 3:
                for (h = 0; h < surf->h; ++h) {
                    Uint8 *ptr = (Uint8 *)DATAROW(surf->pixels, h, surf->pitch,
                                                  surf->h, flipped);
                    for (w = 0; w < surf->w; ++w) {
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                        color = ptr[0] + (ptr[1] << 8) + (ptr[2] << 16);
#else
                        color = ptr[2] + (ptr[1] << 8) + (ptr[0] << 16);
#endif
                        ptr += 3;
                        data[0] = (char)(((color & Rmask) >> Rshift) << Rloss);
                        data[1] = (char)(((color & Gmask) >> Gshift) << Gloss);
                        data[2] = (char)(((color & Bmask) >> Bshift) << Bloss);
                        data[3] =
                            hascolorkey
                                ? (char)(color != colorkey) * 255
                                : (char)(Amask ? (((color & Amask) >> Ashift)
                                                  << Aloss)
                                               : 255);
                        data += 4;
                    }
                }
                break;
            case 4:
                tostring_surf_32bpp(surf, flipped, hascolorkey, colorkey,
                                    data, 0, 3);
                break;
        }
        pgSurface_Unlock(surfobj);
    }
    else if (!strcmp(format, "ARGB")) {
        hascolorkey = 0;

        string =
            Bytes_FromStringAndSize(NULL, (Py_ssize_t)surf->w * surf->h * 4);
        if (!string)
            return NULL;
        Bytes_AsStringAndSize(string, &data, &len);

        pgSurface_Lock(surfobj);
        pixels = (char *)surf->pixels;
        switch (surf->format->BytesPerPixel) {
            case 1:
                for (h = 0; h < surf->h; ++h) {
                    Uint8 *ptr = (Uint8 *)DATAROW(surf->pixels, h, surf->pitch,
                                                  surf->h, flipped);
                    for (w = 0; w < surf->w; ++w) {
                        color = *ptr++;
                        data[1] = (char)surf->format->palette->colors[color].r;
                        data[2] = (char)surf->format->palette->colors[color].g;
                        data[3] = (char)surf->format->palette->colors[color].b;
                        data[0] = (char)255;
                        data += 4;
                    }
                }
                break;
            case 2:
                for (h = 0; h < surf->h; ++h) {
                    Uint16 *ptr = (Uint16 *)DATAROW(
                        surf->pixels, h, surf->pitch, surf->h, flipped);
                    for (w = 0; w < surf->w; ++w) {
                        color = *ptr++;
                        data[1] = (char)(((color & Rmask) >> Rshift) << Rloss);
                        data[2] = (char)(((color & Gmask) >> Gshift) << Gloss);
                        data[3] = (char)(((color & Bmask) >> Bshift) << Bloss);
                        data[0] = (char)(Amask ? (((color & Amask) >> Ashift)
                                                  << Aloss)
                                               : 255);
                        data += 4;
                    }
                }
                break;
            case 3:
                for (h = 0; h < surf->h; ++h) {
                    Uint8 *ptr = (Uint8 *)DATAROW(surf->pixels, h, surf->pitch,
                                                  surf->h, flipped);
                    for (w = 0; w < surf->w; ++w) {
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                        color = ptr[0] + (ptr[1] << 8) + (ptr[2] << 16);
#else
                        color = ptr[2] + (ptr[1] << 8) + (ptr[0] << 16);
#endif
                        ptr += 3;
                        data[1] = (char)(((color & Rmask) >> Rshift) << Rloss);
                        data[2] = (char)(((color & Gmask) >> Gshift) << Gloss);
                        data[3] = (char)(((color & Bmask) >> Bshift) << Bloss);
                        data[0] = (char)(Amask ? (((color & Amask) >> Ashift)
                                                  << Aloss)
                                               : 255);
                        data += 4;
                    }
                }
                break;
            case 4:
                tostring_surf_32bpp(surf, flipped, hascolorkey, colorkey,
                                    data, 1, 0);
                break;
        }
        pgSurface_Unlock(surfobj);
    }
    else if (!strcmp(format, "RGBA_PREMULT")) {
        if (surf->format->BytesPerPixel == 1 || surf->format->Amask == 0)
            return RAISE(PyExc_ValueError,
                         "Can only create pre-multiplied alpha strings if the "
                         "surface has per-pixel alpha");

        hascolorkey = 0;

        string =
            Bytes_FromStringAndSize(NULL, (Py_ssize_t)surf->w * surf->h * 4);
        if (!string)
            return NULL;
        Bytes_AsStringAndSize(string, &data, &len);

        pgSurface_Lock(surfobj);
        pixels = (char *)surf->pixels;
        switch (surf->format->BytesPerPixel) {
            case 2:
                for (h = 0; h < surf->h; ++h) {
                    Uint16 *ptr = (Uint16 *)DATAROW(
                        surf->pixels, h, surf->pitch, surf->h, flipped);
                    for (w = 0; w < surf->w; ++w) {
                        color = *ptr++;
                        alpha = ((color & Amask) >> Ashift) << Aloss;
                        data[0] =
                            (char)((((color & Rmask) >> Rshift) << Rloss) *
                                   alpha / 255);
                        data[1] =
                            (char)((((color & Gmask) >> Gshift) << Gloss) *
                                   alpha / 255);
                        data[2] =
                            (char)((((color & Bmask) >> Bshift) << Bloss) *
                                   alpha / 255);
                        data[3] = (char)alpha;
                        data += 4;
                    }
                }
                break;
            case 3:
                for (h = 0; h < surf->h; ++h) {
                    Uint8 *ptr = (Uint8 *)DATAROW(surf->pixels, h, surf->pitch,
                                                  surf->h, flipped);
                    for (w = 0; w < surf->w; ++w) {
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                        color = ptr[0] + (ptr[1] << 8) + (ptr[2] << 16);
#else
                        color = ptr[2] + (ptr[1] << 8) + (ptr[0] << 16);
#endif
                        ptr += 3;
                        alpha = ((color & Amask) >> Ashift) << Aloss;
                        data[0] =
                            (char)((((color & Rmask) >> Rshift) << Rloss) *
                                   alpha / 255);
                        data[1] =
                            (char)((((color & Gmask) >> Gshift) << Gloss) *
                                   alpha / 255);
                        data[2] =
                            (char)((((color & Bmask) >> Bshift) << Bloss) *
                                   alpha / 255);
                        data[3] = (char)alpha;
                        data += 4;
                    }
                }
                break;
            case 4:
                for (h = 0; h < surf->h; ++h) {
                    Uint32 *ptr = (Uint32 *)DATAROW(
                        surf->pixels, h, surf->pitch, surf->h, flipped);
                    for (w = 0; w < surf->w; ++w) {
                        color = *ptr++;
                        alpha = ((color & Amask) >> Ashift) << Aloss;
                        if (alpha == 0) {
                            data[0] = data[1] = data[2] = 0;
                        }
                        else {
                            data[0] =
                                (char)((((color & Rmask) >> Rshift) << Rloss) *
                                       alpha / 255);
                            data[1] =
                                (char)((((color & Gmask) >> Gshift) << Gloss) *
                                       alpha / 255);
                            data[2] =
                                (char)((((color & Bmask) >> Bshift) << Bloss) *
                                       alpha / 255);
                        }
                        data[3] = (char)alpha;
                        data += 4;
                    }
                }
                break;
        }
        pgSurface_Unlock(surfobj);
    }
    else if (!strcmp(format, "ARGB_PREMULT")) {
        if (surf->format->BytesPerPixel == 1 || surf->format->Amask == 0)
            return RAISE(PyExc_ValueError,
                         "Can only create pre-multiplied alpha strings if the "
                         "surface has per-pixel alpha");

        hascolorkey = 0;

        string =
            Bytes_FromStringAndSize(NULL, (Py_ssize_t)surf->w * surf->h * 4);
        if (!string)
            return NULL;
        Bytes_AsStringAndSize(string, &data, &len);

        pgSurface_Lock(surfobj);
        pixels = (char *)surf->pixels;
        switch (surf->format->BytesPerPixel) {
            case 2:
                for (h = 0; h < surf->h; ++h) {
                    Uint16 *ptr = (Uint16 *)DATAROW(
                        surf->pixels, h, surf->pitch, surf->h, flipped);
                    for (w = 0; w < surf->w; ++w) {
                        color = *ptr++;
                        alpha = ((color & Amask) >> Ashift) << Aloss;
                        data[1] =
                            (char)((((color & Rmask) >> Rshift) << Rloss) *
                                   alpha / 255);
                        data[2] =
                            (char)((((color & Gmask) >> Gshift) << Gloss) *
                                   alpha / 255);
                        data[3] =
                            (char)((((color & Bmask) >> Bshift) << Bloss) *
                                   alpha / 255);
                        data[0] = (char)alpha;
                        data += 4;
                    }
                }
                break;
            case 3:
                for (h = 0; h < surf->h; ++h) {
                    Uint8 *ptr = (Uint8 *)DATAROW(surf->pixels, h, surf->pitch,
                                                  surf->h, flipped);
                    for (w = 0; w < surf->w; ++w) {
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                        color = ptr[0] + (ptr[1] << 8) + (ptr[2] << 16);
#else
                        color = ptr[2] + (ptr[1] << 8) + (ptr[0] << 16);
#endif
                        ptr += 3;
                        alpha = ((color & Amask) >> Ashift) << Aloss;
                        data[1] =
                            (char)((((color & Rmask) >> Rshift) << Rloss) *
                                   alpha / 255);
                        data[2] =
                            (char)((((color & Gmask) >> Gshift) << Gloss) *
                                   alpha / 255);
                        data[3] =
                            (char)((((color & Bmask) >> Bshift) << Bloss) *
                                   alpha / 255);
                        data[0] = (char)alpha;
                        data += 4;
                    }
                }
                break;
            case 4:
                for (h = 0; h < surf->h; ++h) {
                    Uint32 *ptr = (Uint32 *)DATAROW(
                        surf->pixels, h, surf->pitch, surf->h, flipped);
                    for (w = 0; w < surf->w; ++w) {
                        color = *ptr++;
                        alpha = ((color & Amask) >> Ashift) << Aloss;
                        if (alpha == 0) {
                            data[1] = data[2] = data[3] = 0;
                        }
                        else {
                            data[1] =
                                (char)((((color & Rmask) >> Rshift) << Rloss) *
                                       alpha / 255);
                            data[2] =
                                (char)((((color & Gmask) >> Gshift) << Gloss) *
                                       alpha / 255);
                            data[3] =
                                (char)((((color & Bmask) >> Bshift) << Bloss) *
                                       alpha / 255);
                        }
                        data[0] = (char)alpha;
                        data += 4;
                    }
                }
                break;
        }
        pgSurface_Unlock(surfobj);
    }
    else {
#if IS_SDLv1
        if (temp)
            SDL_FreeSurface(temp);
#endif /* IS_SDLv1 */

        return RAISE(PyExc_ValueError, "Unrecognized type of format");
    }

#if IS_SDLv1
    if (temp)
        SDL_FreeSurface(temp);
#endif /* IS_SDLv1 */

    return string;
}

PyObject *
image_fromstring(PyObject *self, PyObject *arg)
{
    PyObject *string;
    char *format, *data;
    SDL_Surface *surf = NULL;
    int w, h, flipped = 0;
    Py_ssize_t len;
    int loopw, looph;

    if (!PyArg_ParseTuple(arg, "O!(ii)s|i", &Bytes_Type, &string, &w, &h,
                          &format, &flipped))
        return NULL;

    if (w < 1 || h < 1)
        return RAISE(PyExc_ValueError, "Resolution must be positive values");

    Bytes_AsStringAndSize(string, &data, &len);

    if (!strcmp(format, "P")) {
        if (len != (Py_ssize_t)w * h)
            return RAISE(
                PyExc_ValueError,
                "String length does not equal format and resolution size");

        surf = SDL_CreateRGBSurface(0, w, h, 8, 0, 0, 0, 0);
        if (!surf)
            return RAISE(pgExc_SDLError, SDL_GetError());
        SDL_LockSurface(surf);
        for (looph = 0; looph < h; ++looph)
            memcpy(((char *)surf->pixels) + looph * surf->pitch,
                   DATAROW(data, looph, w, h, flipped), w);
        SDL_UnlockSurface(surf);
    }
    else if (!strcmp(format, "RGB")) {
        if (len != (Py_ssize_t)w * h * 3)
            return RAISE(
                PyExc_ValueError,
                "String length does not equal format and resolution size");
        surf =
            SDL_CreateRGBSurface(0, w, h, 24, 0xFF << 16, 0xFF << 8, 0xFF, 0);
        if (!surf)
            return RAISE(pgExc_SDLError, SDL_GetError());
        SDL_LockSurface(surf);
        for (looph = 0; looph < h; ++looph) {
            Uint8 *pix =
                (Uint8 *)DATAROW(surf->pixels, looph, surf->pitch, h, flipped);
            for (loopw = 0; loopw < w; ++loopw) {
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                pix[2] = data[0];
                pix[1] = data[1];
                pix[0] = data[2];
#else
                pix[0] = data[0];
                pix[1] = data[1];
                pix[2] = data[2];
#endif
                pix += 3;
                data += 3;
            }
        }
        SDL_UnlockSurface(surf);
    }
    else if (!strcmp(format, "RGBA") || !strcmp(format, "RGBX")) {
        int alphamult = !strcmp(format, "RGBA");
        if (len != (Py_ssize_t)w * h * 4)
            return RAISE(
                PyExc_ValueError,
                "String length does not equal format and resolution size");
        surf = SDL_CreateRGBSurface((alphamult ? SDL_SRCALPHA : 0), w, h, 32,
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                                    0xFF, 0xFF << 8, 0xFF << 16,
                                    (alphamult ? 0xFF << 24 : 0));
#else
                                    0xFF << 24, 0xFF << 16, 0xFF << 8,
                                    (alphamult ? 0xFF : 0));
#endif
        if (!surf)
            return RAISE(pgExc_SDLError, SDL_GetError());
        SDL_LockSurface(surf);
        for (looph = 0; looph < h; ++looph) {
            Uint32 *pix = (Uint32 *)DATAROW(surf->pixels, looph, surf->pitch,
                                            h, flipped);
            memcpy(pix, data, w * sizeof(Uint32));
            data += w * sizeof(Uint32);
        }
        SDL_UnlockSurface(surf);
    }
    else if (!strcmp(format, "ARGB")) {
        if (len != (Py_ssize_t)w * h * 4)
            return RAISE(
                PyExc_ValueError,
                "String length does not equal format and resolution size");
        surf = SDL_CreateRGBSurface(SDL_SRCALPHA, w, h, 32,
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                                    0xFF << 8, 0xFF << 16, 0xFF << 24, 0xFF);
#else
                                    0xFF << 16, 0xFF << 8, 0xFF, 0xFF << 24);
#endif
        if (!surf)
            return RAISE(pgExc_SDLError, SDL_GetError());
        SDL_LockSurface(surf);
        for (looph = 0; looph < h; ++looph) {
            Uint32 *pix = (Uint32 *)DATAROW(surf->pixels, looph, surf->pitch,
                                            h, flipped);
            memcpy(pix, data, w * sizeof(Uint32));
            data += w * sizeof(Uint32);
        }
        SDL_UnlockSurface(surf);
    }
    else
        return RAISE(PyExc_ValueError, "Unrecognized type of format");

    return (PyObject *)pgSurface_New(surf);
}

static int
_as_read_buffer(PyObject *obj, const void **buffer, Py_ssize_t *buffer_len)
{
    Py_buffer view;

    if (obj == NULL || buffer == NULL || buffer_len == NULL) {
        return -1;
    }
    if (PyObject_GetBuffer(obj, &view, PyBUF_SIMPLE) != 0)
        return -1;

    *buffer = view.buf;
    *buffer_len = view.len;
    PyBuffer_Release(&view);
    return 0;
}
/*
pgObject_AsCharBuffer is backwards compatible for PyObject_AsCharBuffer.
Because PyObject_AsCharBuffer is deprecated.
*/
int
pgObject_AsCharBuffer(PyObject *obj, const char **buffer,
                      Py_ssize_t *buffer_len)
{
    return _as_read_buffer(obj, (const void **)buffer, buffer_len);
}

PyObject *
image_frombuffer(PyObject *self, PyObject *arg)
{
    PyObject *buffer;
    char *format, *data;
    SDL_Surface *surf = NULL;
    int w, h;
    Py_ssize_t len;
    pgSurfaceObject *surfobj;

    if (!PyArg_ParseTuple(arg, "O(ii)s|i", &buffer, &w, &h, &format))
        return NULL;

    if (w < 1 || h < 1)
        return RAISE(PyExc_ValueError, "Resolution must be positive values");

    /* breaking constness here, we should really not change this string */
    if (pgObject_AsCharBuffer(buffer, (const char **)&data, &len) == -1)
        return NULL;

    if (!strcmp(format, "P")) {
        if (len != (Py_ssize_t)w * h)
            return RAISE(
                PyExc_ValueError,
                "Buffer length does not equal format and resolution size");

        surf = SDL_CreateRGBSurfaceFrom(data, w, h, 8, w, 0, 0, 0, 0);
    }
    else if (!strcmp(format, "RGB")) {
        if (len != (Py_ssize_t)w * h * 3)
            return RAISE(
                PyExc_ValueError,
                "Buffer length does not equal format and resolution size");
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
        surf = SDL_CreateRGBSurfaceFrom(data, w, h, 24, w * 3, 0xFF, 0xFF << 8,
                                        0xFF << 16, 0);
#else
        surf = SDL_CreateRGBSurfaceFrom(data, w, h, 24, w * 3, 0xFF << 16,
                                        0xFF << 8, 0xFF, 0);
#endif
    }
    else if (!strcmp(format, "BGR")) {
        if (len != (Py_ssize_t)w * h * 3)
            return RAISE(
                PyExc_ValueError,
                "Buffer length does not equal format and resolution size");
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
        surf = SDL_CreateRGBSurfaceFrom(data, w, h, 24, w * 3,
                                        0xFF << 16, 0xFF << 8,
                                        0xFF, 0);
#else
        surf = SDL_CreateRGBSurfaceFrom(data, w, h, 24, w * 3,
                                        0xFF, 0xFF << 8,
                                        0xFF << 16, 0);
#endif
    }
    else if (!strcmp(format, "RGBA") || !strcmp(format, "RGBX")) {
        int alphamult = !strcmp(format, "RGBA");
        if (len != (Py_ssize_t)w * h * 4)
            return RAISE(
                PyExc_ValueError,
                "Buffer length does not equal format and resolution size");
        surf = SDL_CreateRGBSurfaceFrom(data, w, h, 32, w * 4,
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                                        0xFF, 0xFF << 8, 0xFF << 16,
                                        (alphamult ? 0xFF << 24 : 0));
#else
                                        0xFF << 24, 0xFF << 16, 0xFF << 8,
                                        (alphamult ? 0xFF : 0));
#endif
        if (alphamult)
            surf->flags |= SDL_SRCALPHA;
    }
    else if (!strcmp(format, "ARGB")) {
        if (len != (Py_ssize_t)w * h * 4)
            return RAISE(
                PyExc_ValueError,
                "Buffer length does not equal format and resolution size");
        surf =
            SDL_CreateRGBSurfaceFrom(data, w, h, 32, w * 4,
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
                                    0xFF << 8, 0xFF << 16, 0xFF << 24, 0xFF);
#else
                                    0xFF << 16, 0xFF << 8, 0xFF, 0xFF << 24);
#endif
        surf->flags |= SDL_SRCALPHA;
    }
    else
        return RAISE(PyExc_ValueError, "Unrecognized type of format");

    if (!surf)
        return RAISE(pgExc_SDLError, SDL_GetError());
    surfobj = pgSurface_New(surf);
    Py_INCREF(buffer);
    surfobj->dependency = buffer;
    return (PyObject *)surfobj;
}

/*******************************************************/
/* tga code by Mattias Engdegard, in the public domain */
/*******************************************************/
struct TGAheader {
    Uint8 infolen;  /* length of info field */
    Uint8 has_cmap; /* 1 if image has colormap, 0 otherwise */
    Uint8 type;

    Uint8 cmap_start[2]; /* index of first colormap entry */
    Uint8 cmap_len[2];   /* number of entries in colormap */
    Uint8 cmap_bits;     /* bits per colormap entry */

    Uint8 yorigin[2]; /* image origin (ignored here) */
    Uint8 xorigin[2];
    Uint8 width[2]; /* image size */
    Uint8 height[2];
    Uint8 pixel_bits; /* bits/pixel */
    Uint8 flags;
};

enum tga_type {
    TGA_TYPE_INDEXED = 1,
    TGA_TYPE_RGB = 2,
    TGA_TYPE_BW = 3,
    TGA_TYPE_RLE = 8 /* additive */
};

#define TGA_INTERLEAVE_MASK 0xc0
#define TGA_INTERLEAVE_NONE 0x00
#define TGA_INTERLEAVE_2WAY 0x40
#define TGA_INTERLEAVE_4WAY 0x80

#define TGA_ORIGIN_MASK 0x30
#define TGA_ORIGIN_LEFT 0x00
#define TGA_ORIGIN_RIGHT 0x10
#define TGA_ORIGIN_LOWER 0x00
#define TGA_ORIGIN_UPPER 0x20

/* read/write unaligned little-endian 16-bit ints */
#define LE16(p) ((p)[0] + ((p)[1] << 8))
#define SETLE16(p, v) ((p)[0] = (v), (p)[1] = (v) >> 8)

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#define TGA_RLE_MAX 128 /* max length of a TGA RLE chunk */
/* return the number of bytes in the resulting buffer after RLE-encoding
   a line of TGA data */
static int
rle_line(Uint8 *src, Uint8 *dst, int w, int bpp)
{
    int x = 0;
    int out = 0;
    int raw = 0;
    while (x < w) {
        Uint32 pix;
        int x0 = x;
        memcpy(&pix, src + x * bpp, bpp);
        x++;
        while (x < w && memcmp(&pix, src + x * bpp, bpp) == 0 &&
               x - x0 < TGA_RLE_MAX)
            x++;
        /* use a repetition chunk iff the repeated pixels would consume
           two bytes or more */
        if ((x - x0 - 1) * bpp >= 2 || x == w) {
            /* output previous raw chunks */
            while (raw < x0) {
                int n = MIN(TGA_RLE_MAX, x0 - raw);
                dst[out++] = n - 1;
                memcpy(dst + out, src + raw * bpp, (size_t)n * bpp);
                out += n * bpp;
                raw += n;
            }

            if (x - x0 > 0) {
                /* output new repetition chunk */
                dst[out++] = 0x7f + x - x0;
                memcpy(dst + out, &pix, bpp);
                out += bpp;
            }
            raw = x;
        }
    }
    return out;
}

/*
 * Save a surface to an output stream in TGA format.
 * 8bpp surfaces are saved as indexed images with 24bpp palette, or with
 *     32bpp palette if colourkeying is used.
 * 15, 16, 24 and 32bpp surfaces are saved as 24bpp RGB images,
 * or as 32bpp RGBA images if alpha channel is used.
 *
 * RLE compression is not used in the output file.
 *
 * Returns -1 upon error, 0 if success
 */
static int
SaveTGA_RW(SDL_Surface *surface, SDL_RWops *out, int rle)
{
    SDL_Surface *linebuf = NULL;
    int alpha = 0;
#if IS_SDLv1
    int ckey = -1;
#endif /* IS_SDLv1 */
    struct TGAheader h;
    int srcbpp;
#if IS_SDLv1
    unsigned surf_flags;
    unsigned surf_alpha;
#else  /* IS_SDLv2 */
    Uint8 surf_alpha;
    int have_surf_colorkey = 0;
    Uint32 surf_colorkey;
#endif /* IS_SDLv2 */
    Uint32 rmask, gmask, bmask, amask;
    SDL_Rect r;
    int bpp;
    Uint8 *rlebuf = NULL;

    h.infolen = 0;
    SETLE16(h.cmap_start, 0);

    srcbpp = surface->format->BitsPerPixel;
    if (srcbpp < 8) {
        SDL_SetError("cannot save <8bpp images as TGA");
        return -1;
    }

#if IS_SDLv2
    SDL_GetSurfaceAlphaMod(surface, &surf_alpha);
    have_surf_colorkey = (SDL_GetColorKey(surface, &surf_colorkey) == 0);
#endif /* IS_SDLv2 */

    if (srcbpp == 8) {
        h.has_cmap = 1;
        h.type = TGA_TYPE_INDEXED;
#if IS_SDLv1
        if (surface->flags & SDL_SRCCOLORKEY) {
            ckey = surface->format->colorkey;
            h.cmap_bits = 32;
        }
#else  /* IS_SDLv2 */
        if (have_surf_colorkey)
            h.cmap_bits = 32;
#endif /* IS_SDLv2 */
        else
            h.cmap_bits = 24;
        SETLE16(h.cmap_len, surface->format->palette->ncolors);
        h.pixel_bits = 8;
        rmask = gmask = bmask = amask = 0;
    }
    else {
        h.has_cmap = 0;
        h.type = TGA_TYPE_RGB;
        h.cmap_bits = 0;
        SETLE16(h.cmap_len, 0);
        if (surface->format->Amask) {
            alpha = 1;
            h.pixel_bits = 32;
        }
        else
            h.pixel_bits = 24;
        if (SDL_BYTEORDER == SDL_BIG_ENDIAN) {
            int s = alpha ? 0 : 8;
            amask = 0x000000ff >> s;
            rmask = 0x0000ff00 >> s;
            gmask = 0x00ff0000 >> s;
            bmask = 0xff000000 >> s;
        }
        else {
            amask = alpha ? 0xff000000 : 0;
            rmask = 0x00ff0000;
            gmask = 0x0000ff00;
            bmask = 0x000000ff;
        }
    }
    bpp = h.pixel_bits >> 3;
    if (rle)
        h.type += TGA_TYPE_RLE;

    SETLE16(h.yorigin, 0);
    SETLE16(h.xorigin, 0);
    SETLE16(h.width, surface->w);
    SETLE16(h.height, surface->h);
    h.flags = TGA_ORIGIN_UPPER | (alpha ? 8 : 0);

    if (!SDL_RWwrite(out, &h, sizeof(h), 1))
        return -1;

    if (h.has_cmap) {
        int i;
        SDL_Palette *pal = surface->format->palette;
        Uint8 entry[4];
        for (i = 0; i < pal->ncolors; i++) {
            entry[0] = pal->colors[i].b;
            entry[1] = pal->colors[i].g;
            entry[2] = pal->colors[i].r;
#if IS_SDLv1
            entry[3] = (i == ckey) ? 0 : 0xff;
#else  /* IS_SDLv2 */
            entry[3] = ((unsigned)i == surf_colorkey) ? 0 : 0xff;
#endif /* IS_SDLv2 */
            if (!SDL_RWwrite(out, entry, h.cmap_bits >> 3, 1))
                return -1;
        }
    }

    linebuf = SDL_CreateRGBSurface(SDL_SWSURFACE, surface->w, 1, h.pixel_bits,
                                   rmask, gmask, bmask, amask);
    if (!linebuf)
        return -1;

    if (h.has_cmap) {
#if IS_SDLv1
        SDL_SetColors(linebuf, surface->format->palette->colors, 0,
                      surface->format->palette->ncolors);
#else  /* IS_SDLv2 */
        if (0 != SDL_SetPaletteColors(linebuf->format->palette,
                                      surface->format->palette->colors, 0,
                                      surface->format->palette->ncolors)) {
            /* SDL error already set. */
            goto error;
        }
#endif /* IS_SDLv2 */
    }

    if (rle) {
        rlebuf = malloc(bpp * surface->w + 1 + surface->w / TGA_RLE_MAX);
        if (!rlebuf) {
            SDL_SetError("out of memory");
            goto error;
        }
    }

    /* Temporarily remove colourkey and alpha from surface so copies are
       opaque */
#if IS_SDLv1
    surf_flags = surface->flags & (SDL_SRCALPHA | SDL_SRCCOLORKEY);
    surf_alpha = surface->format->alpha;
    if (surf_flags & SDL_SRCALPHA)
        SDL_SetAlpha(surface, 0, 255);
    if (surf_flags & SDL_SRCCOLORKEY)
        SDL_SetColorKey(surface, 0, surface->format->colorkey);
#else  /* IS_SDLv2 */
    SDL_SetSurfaceAlphaMod(surface, SDL_ALPHA_OPAQUE);
    if (have_surf_colorkey)
        SDL_SetColorKey(surface, SDL_FALSE, surf_colorkey);
#endif /* IS_SDLv2 */

    r.x = 0;
    r.w = surface->w;
    r.h = 1;
    for (r.y = 0; r.y < surface->h; r.y++) {
        int n;
        void *buf;
        if (SDL_BlitSurface(surface, &r, linebuf, NULL) < 0)
            break;
        if (rle) {
            buf = rlebuf;
            n = rle_line(linebuf->pixels, rlebuf, surface->w, bpp);
        }
        else {
            buf = linebuf->pixels;
            n = surface->w * bpp;
        }
        if (!SDL_RWwrite(out, buf, n, 1))
            break;
    }

    /* restore flags */
#if IS_SDLv1
    if (surf_flags & SDL_SRCALPHA)
        SDL_SetAlpha(surface, SDL_SRCALPHA, (Uint8)surf_alpha);
    if (surf_flags & SDL_SRCCOLORKEY)
        SDL_SetColorKey(surface, SDL_SRCCOLORKEY, surface->format->colorkey);
#else  /* IS_SDLv2 */
    SDL_SetSurfaceAlphaMod(surface, surf_alpha);
    if (have_surf_colorkey)
        SDL_SetColorKey(surface, SDL_TRUE, surf_colorkey);
#endif /* IS_SDLv2 */

    free(rlebuf);
    SDL_FreeSurface(linebuf);
    return 0;

error:
    free(rlebuf);
    SDL_FreeSurface(linebuf);
    return -1;
}

static int
SaveTGA(SDL_Surface *surface, const char *file, int rle)
{
    SDL_RWops *out = SDL_RWFromFile(file, "wb");
    int ret;
    if (!out)
        return -1;
    ret = SaveTGA_RW(surface, out, rle);
    SDL_RWclose(out);
    return ret;
}

static PyMethodDef _image_methods[] = {
    {"load_basic", (PyCFunction)image_load_basic, METH_O, DOC_PYGAMEIMAGELOADBASIC},
    {"load_extended", image_load_extended, METH_VARARGS, DOC_PYGAMEIMAGELOADEXTENDED},
    {"load", image_load, METH_VARARGS, DOC_PYGAMEIMAGELOAD},
    
    {"save_extended", image_save_extended, METH_VARARGS, DOC_PYGAMEIMAGESAVEEXTENDED},
    {"save", image_save, METH_VARARGS, DOC_PYGAMEIMAGESAVE},
    {"get_extended", (PyCFunction)image_get_extended, METH_NOARGS,
     DOC_PYGAMEIMAGEGETEXTENDED},
    {"get_sdl_image_version", (PyCFunction)image_get_sdl_image_version, METH_NOARGS,
     DOC_PYGAMEIMAGEGETSDLIMAGEVERSION},

    {"tostring", image_tostring, METH_VARARGS, DOC_PYGAMEIMAGETOSTRING},
    {"fromstring", image_fromstring, METH_VARARGS, DOC_PYGAMEIMAGEFROMSTRING},
    {"frombuffer", image_frombuffer, METH_VARARGS, DOC_PYGAMEIMAGEFROMBUFFER},
    {NULL, NULL, 0, NULL}};

MODINIT_DEFINE(image)
{
    PyObject *module;
    PyObject *extmodule;

#if PY3
    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "image",
                                         DOC_PYGAMEIMAGE,
                                         -1,
                                         _image_methods,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL};
#endif

    /* imported needed apis; Do this first so if there is an error
       the module is not loaded.
    */
    import_pygame_base();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }
    import_pygame_surface();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }
    import_pygame_rwobject();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }

    /* create the module */
#if PY3
    module = PyModule_Create(&_module);
#else
    module =
        Py_InitModule3(MODPREFIX "image", _image_methods, DOC_PYGAMEIMAGE);
#endif
    if (module == NULL) {
        MODINIT_ERROR;
    }

    /* try to get extended formats */
    extmodule = PyImport_ImportModule(IMPPREFIX "imageext");
    if (extmodule) {
        extloadobj = PyObject_GetAttrString(extmodule, "load_extended");
        if (!extloadobj) {
            goto error;
        }
        extsaveobj = PyObject_GetAttrString(extmodule, "save_extended");
        if (!extsaveobj) {
            goto error;
        }
        extverobj = PyObject_GetAttrString(extmodule, "_get_sdl_image_version");
        if (!extverobj) {
            goto error;
        }
        Py_DECREF(extmodule);
    }
    else {
        // if the module could not be loaded, dont treat it like an error
        PyErr_Clear();
    }
    MODINIT_RETURN(module);
    
    error:
        Py_XDECREF(extloadobj);
        Py_XDECREF(extsaveobj);
        Py_XDECREF(extverobj);
        Py_DECREF(extmodule);
        DECREF_MOD(module);
        MODINIT_ERROR;
}
