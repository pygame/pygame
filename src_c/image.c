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

#if PG_COMPILE_SSE4_2
#include <emmintrin.h>
/* SSSE 3 */
#include <tmmintrin.h>
#endif

static int
SaveTGA(SDL_Surface *surface, const char *file, int rle);
static int
SaveTGA_RW(SDL_Surface *surface, SDL_RWops *out, int rle);

#define DATAROW(data, row, width, height, flipped)             \
    ((flipped) ? (((char *)data) + (height - row - 1) * width) \
               : (((char *)data) + row * width))

static PyObject *extloadobj = NULL;
static PyObject *extsaveobj = NULL;
static PyObject *extverobj = NULL;
static PyObject *ext_load_sized_svg = NULL;
static PyObject *ext_load_animation = NULL;

static inline void
pad(char **data, int padding)
{
    if (padding) {
        memset(*data, 0, padding);
        *data += padding;
    }
}

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
    SDL_Surface *surf;

    SDL_RWops *rw = pgRWops_FromObject(obj, NULL);
    if (rw == NULL) {
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS;
    surf = SDL_LoadBMP_RW(rw, 1);
    Py_END_ALLOW_THREADS;

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
image_load_extended(PyObject *self, PyObject *arg, PyObject *kwarg)
{
    if (extloadobj == NULL) {
        return RAISE(PyExc_NotImplementedError,
                     "loading images of extended format is not available");
    }
    else {
        return PyObject_Call(extloadobj, arg, kwarg);
    }
}

static PyObject *
image_load(PyObject *self, PyObject *arg, PyObject *kwarg)
{
    PyObject *obj;
    const char *name = NULL;
    static char *kwds[] = {"file", "namehint", NULL};

    if (extloadobj == NULL) {
        if (!PyArg_ParseTupleAndKeywords(arg, kwarg, "O|s", kwds, &obj,
                                         &name)) {
            return NULL;
        }
        return image_load_basic(self, obj);
    }
    else {
        return image_load_extended(self, arg, kwarg);
    }
}

#ifdef WIN32
#define strcasecmp _stricmp
#else
#include <strings.h>
#endif

static PyObject *
image_save_extended(PyObject *self, PyObject *arg, PyObject *kwarg)
{
    if (extsaveobj == NULL) {
        return RAISE(PyExc_NotImplementedError,
                     "saving images of extended format is not available");
    }
    else {
        return PyObject_Call(extsaveobj, arg, kwarg);
    }
}

static PyObject *
image_save(PyObject *self, PyObject *arg, PyObject *kwarg)
{
    pgSurfaceObject *surfobj;
    PyObject *obj;
    const char *namehint = NULL;
    PyObject *oencoded;
    PyObject *ret;
    SDL_Surface *surf;
    int result = 1;
    static char *kwds[] = {"surface", "file", "namehint", NULL};

    if (!PyArg_ParseTupleAndKeywords(arg, kwarg, "O!O|s", kwds,
                                     &pgSurface_Type, &surfobj, &obj,
                                     &namehint)) {
        return NULL;
    }

    surf = pgSurface_AsSurface(surfobj);
    pgSurface_Prep(surfobj);

    oencoded = pg_EncodeString(obj, "UTF-8", NULL, pgExc_SDLError);
    if (oencoded == NULL) {
        result = -2;
    }
    else {
        const char *name = NULL;
        const char *ext = NULL;
        if (oencoded == Py_None) {
            name = (namehint ? namehint : "tga");
        }
        else {
            name = PyBytes_AS_STRING(oencoded);
        }

        ext = find_extension(name);
        if (!strcasecmp(ext, "png") || !strcasecmp(ext, "jpg") ||
            !strcasecmp(ext, "jpeg")) {
            /* If it is .png .jpg .jpeg use the extended module. */
            /* try to get extended formats */
            ret = image_save_extended(self, arg, kwarg);
            result = (ret == NULL ? -2 : 0);
        }
        else if (oencoded == Py_None) {
            SDL_RWops *rw = pgRWops_FromFileObject(obj);
            if (rw != NULL) {
                if (!strcasecmp(ext, "bmp")) {
#if SDL_VERSION_ATLEAST(3, 0, 0)
                    result = (SDL_SaveBMP_IO(surf, rw, 0) ? 0 : -1);
#else
                    /* The SDL documentation didn't specify which negative
                     * number is returned upon error. We want to be sure that
                     * result is either 0 or -1: */
                    result = (SDL_SaveBMP_RW(surf, rw, 0) == 0 ? 0 : -1);
#endif
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
#if SDL_VERSION_ATLEAST(3, 0, 0)
                result = (SDL_SaveBMP(surf, name) ? 0 : -1);
#else
                /* The SDL documentation didn't specify which negative number
                 * is returned upon error. We want to be sure that result is
                 * either 0 or -1: */
                result = (SDL_SaveBMP(surf, name) == 0 ? 0 : -1);
#endif
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

    pgSurface_Unprep(surfobj);

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
image_get_extended(PyObject *self, PyObject *_null)
{
    if (extverobj == NULL) {
        Py_RETURN_FALSE;
    }
    else {
        Py_RETURN_TRUE;
    }
}

static PyObject *
image_get_sdl_image_version(PyObject *self, PyObject *args, PyObject *kwargs)
{
    if (extverobj == NULL) {
        Py_RETURN_NONE;
    }
    else {
        return PyObject_Call(extverobj, args, kwargs);
    }
}

#if PG_COMPILE_SSE4_2
#define SSE42_ALIGN_NEEDED 16
#define SSE42_ALIGN __attribute__((aligned(SSE42_ALIGN_NEEDED)))

#define _SHIFT_N_STEP2ALIGN(shift, step) (shift / 8 + step * 4)

#if PYGAME_DEBUG_SSE
/* Useful for debugging/comparing the SSE vectors */
static void
_debug_print128_num(__m128i var, const char *msg)
{
    uint32_t val[4];
    memcpy(val, &var, sizeof(val));
    fprintf(stderr, "%s: %04x%04x%04x%04x\n", msg, val[0], val[1], val[2],
            val[3]);
}
#define DEBUG_PRINT128_NUM(var, msg) _debug_print128_num(var, msg)
#else
#define DEBUG_PRINT128_NUM(var, msg) \
    do { /* do nothing */            \
    } while (0)
#endif

/*
 * Generates an SSE vector useful for reordering a SSE vector
 * based on the "in-memory layout" to match the "tobytes layout"
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
                     int alpha_offset)
{
    int output_align[4];
    size_t i;
    size_t limit = sizeof(output_align) / sizeof(int);
    int a_shift = alpha_offset * 8;
    int r_shift = (color_offset + 0) * 8;
    int g_shift = (color_offset + 1) * 8;
    int b_shift = (color_offset + 2) * 8;
    for (i = 0; i < limit; i++) {
        int p = 3 - i;
        output_align[i] = _SHIFT_N_STEP2ALIGN(format->Rshift, p) << r_shift |
                          _SHIFT_N_STEP2ALIGN(format->Gshift, p) << g_shift |
                          _SHIFT_N_STEP2ALIGN(format->Bshift, p) << b_shift |
                          _SHIFT_N_STEP2ALIGN(format->Ashift, p) << a_shift;
    }
    return _mm_set_epi32(output_align[0], output_align[1], output_align[2],
                         output_align[3]);
}

static PG_INLINE PG_FUNCTION_TARGET_SSE4_2 void
tobytes_pixels_32bit_sse4(const __m128i *row, __m128i *data, int loop_max,
                          __m128i mask_vector, __m128i align_vector)
{
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
 * SSE4.2 variant of tobytes_surf_32bpp.
 *
 * It is a lot faster but only works on a subset of the surfaces
 * (plus requires SSE4.2 support from the CPU).
 */
static PG_FUNCTION_TARGET_SSE4_2 void
tobytes_surf_32bpp_sse42(SDL_Surface *surf, int flipped, char *data,
                         int color_offset, int alpha_offset)
{
    const int step_size = 4;
    int h;
    SDL_PixelFormat *format = surf->format;
    int loop_max = surf->w / step_size;
    int mask = (format->Rloss ? 0 : format->Rmask) |
               (format->Gloss ? 0 : format->Gmask) |
               (format->Bloss ? 0 : format->Bmask) |
               (format->Aloss ? 0 : format->Amask);

    __m128i mask_vector = _mm_set_epi32(mask, mask, mask, mask);
    __m128i align_vector =
        compute_align_vector(surf->format, color_offset, alpha_offset);
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
        const char *row =
            (char *)DATAROW(surf->pixels, h, surf->pitch, surf->h, flipped);
        tobytes_pixels_32bit_sse4((const __m128i *)row, (__m128i *)data,
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

            tobytes_pixels_32bit_sse4((const __m128i *)row, (__m128i *)data, 1,
                                      mask_vector, align_vector);

            row += sizeof(__m128i);
            data += sizeof(__m128i);
        }
    }
}
#endif /* PG_COMPILE_SSE4_2 */

static void
#if SDL_VERSION_ATLEAST(3, 0, 0)
tobytes_surf_32bpp(SDL_Surface *surf,
                   const SDL_PixelFormatDetails *format_details, int flipped,
                   int hascolorkey, Uint32 colorkey, char *serialized_image,
                   int color_offset, int alpha_offset, int padding)
#else
tobytes_surf_32bpp(SDL_Surface *surf, SDL_PixelFormat *format_details,
                   int flipped, int hascolorkey, Uint32 colorkey,
                   char *serialized_image, int color_offset, int alpha_offset,
                   int padding)
#endif
{
    int w, h;

    Uint32 Rloss = PG_FORMAT_R_LOSS(format_details);
    Uint32 Gloss = PG_FORMAT_G_LOSS(format_details);
    Uint32 Bloss = PG_FORMAT_B_LOSS(format_details);
    Uint32 Aloss = PG_FORMAT_A_LOSS(format_details);
    Uint32 Rmask = format_details->Rmask;
    Uint32 Gmask = format_details->Gmask;
    Uint32 Bmask = format_details->Bmask;
    Uint32 Amask = format_details->Amask;
    Uint32 Rshift = format_details->Rshift;
    Uint32 Gshift = format_details->Gshift;
    Uint32 Bshift = format_details->Bshift;
    Uint32 Ashift = format_details->Ashift;

#if PG_COMPILE_SSE4_2
    if (/* SDL uses Uint32, SSE uses int for building vectors.
         * Related, we assume that Uint32 is packed so 4 of
         * them perfectly matches an __m128i.
         * If these assumptions do not match up, we will
         * produce incorrect results.
         */
        sizeof(int) == sizeof(Uint32) &&
        4 * sizeof(Uint32) == sizeof(__m128i) &&
        !hascolorkey /* No color key */
        && !padding &&
        SDL_HasSSE42() == SDL_TRUE
        /* The SSE code assumes it will always read at least 4 pixels */
        && surf->w >= 4
        /* Our SSE code assumes masks are at most 0xff */
        && (Rmask >> Rshift) <= 0x0ff && (Gmask >> Gshift) <= 0x0ff &&
        (Bmask >> Bshift) <= 0x0ff &&
        (Amask >> Ashift) <= 0x0ff
        /* Our SSE code cannot handle losses other than 0 or 8
         * Note the mask check above ensures that losses can be
         * at most be 8 (assuming the pixel format makes sense
         * at all).
         */
        && (Rloss % 8) == 0 && (Bloss % 8) == 0 && (Gloss % 8) == 0 &&
        (Aloss % 8) == 0) {
        tobytes_surf_32bpp_sse42(surf, flipped, serialized_image, color_offset,
                                 alpha_offset);
        return;
    }
#endif /* PG_COMPILE_SSE4_2 */

    for (h = 0; h < surf->h; ++h) {
        Uint32 *pixel_row =
            (Uint32 *)DATAROW(surf->pixels, h, surf->pitch, surf->h, flipped);
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
                    : (char)(Amask ? (((color & Amask) >> Ashift) << Aloss)
                                   : 255);
            serialized_image += 4;
        }
        pad(&serialized_image, padding);
    }
}

#define PREMUL_PIXEL_ALPHA(pixel, alpha) (char)((((pixel) + 1) * (alpha)) >> 8)

PyObject *
image_tobytes(PyObject *self, PyObject *arg, PyObject *kwarg)
{
    pgSurfaceObject *surfobj;
    PyObject *bytes = NULL;
    char *format, *data;
    SDL_Surface *surf;
    int w, h, flipped = 0, pitch = -1;
    int byte_width, padding;
    Py_ssize_t len;
    Uint32 Rmask, Gmask, Bmask, Amask, Rshift, Gshift, Bshift, Ashift, Rloss,
        Gloss, Bloss, Aloss;
    int hascolorkey = 0;
    Uint32 color, colorkey;
    Uint32 alpha;
    static char *kwds[] = {"surface", "format", "flipped", "pitch", NULL};

#ifdef _MSC_VER
    /* MSVC static analyzer false alarm: assure format is NULL-terminated by
     * making analyzer assume it was initialised */
    __analysis_assume(format = "inited");
#endif

    if (!PyArg_ParseTupleAndKeywords(arg, kwarg, "O!s|ii", kwds,
                                     &pgSurface_Type, &surfobj, &format,
                                     &flipped, &pitch)) {
        return NULL;
    }
    surf = pgSurface_AsSurface(surfobj);

#if SDL_VERSION_ATLEAST(3, 0, 0)
    const SDL_PixelFormatDetails *format_details =
        SDL_GetPixelFormatDetails(surf->format);
    if (!format_details) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }
    SDL_Palette *surf_palette = PG_GetSurfacePalette(surf);
#else
    SDL_PixelFormat *format_details = surf->format;
    SDL_Palette *surf_palette = surf->format->palette;
#endif
    Rloss = PG_FORMAT_R_LOSS(format_details);
    Gloss = PG_FORMAT_G_LOSS(format_details);
    Bloss = PG_FORMAT_B_LOSS(format_details);
    Aloss = PG_FORMAT_A_LOSS(format_details);
    Rmask = format_details->Rmask;
    Gmask = format_details->Gmask;
    Bmask = format_details->Bmask;
    Amask = format_details->Amask;
    Rshift = format_details->Rshift;
    Gshift = format_details->Gshift;
    Bshift = format_details->Bshift;
    Ashift = format_details->Ashift;

    if (!strcmp(format, "P")) {
        if (PG_SURF_BytesPerPixel(surf) != 1) {
            return RAISE(
                PyExc_ValueError,
                "Can only create \"P\" format data with 8bit Surfaces");
        }
        byte_width = surf->w;
    }
    else if (!strcmp(format, "RGB")) {
        byte_width = surf->w * 3;
    }
    else if (!strcmp(format, "RGBA")) {
        if ((hascolorkey = SDL_HasColorKey(surf))) {
            SDL_GetColorKey(surf, &colorkey);
        }
        byte_width = surf->w * 4;
    }
    else if (!strcmp(format, "RGBX") || !strcmp(format, "ARGB") ||
             !strcmp(format, "BGRA") || !strcmp(format, "ABGR")) {
        byte_width = surf->w * 4;
    }
    else if (!strcmp(format, "RGBA_PREMULT") ||
             !strcmp(format, "ARGB_PREMULT")) {
        if (PG_SURF_BytesPerPixel(surf) == 1 || Amask == 0) {
            return RAISE(PyExc_ValueError,
                         "Can only create pre-multiplied alpha bytes if "
                         "the surface has per-pixel alpha");
        }
        byte_width = surf->w * 4;
    }
    else {
        return RAISE(PyExc_ValueError, "Unrecognized type of format");
    }

    if (pitch == -1) {
        pitch = byte_width;
        padding = 0;
    }
    else if (pitch < byte_width) {
        return RAISE(PyExc_ValueError,
                     "Pitch must be greater than or equal to the width "
                     "as per the format");
    }
    else {
        padding = pitch - byte_width;
    }

    bytes = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)pitch * surf->h);
    if (!bytes) {
        return NULL;
    }
    PyBytes_AsStringAndSize(bytes, &data, &len);

    if (!strcmp(format, "P")) {
        pgSurface_Lock(surfobj);
        for (h = 0; h < surf->h; ++h) {
            Uint8 *ptr = (Uint8 *)DATAROW(data, h, pitch, surf->h, flipped);
            memcpy(ptr, (char *)surf->pixels + (h * surf->pitch), surf->w);
            if (padding) {
                memset(ptr + byte_width, 0, padding);
            }
        }
        pgSurface_Unlock(surfobj);
    }
    else if (!strcmp(format, "RGB")) {
        pgSurface_Lock(surfobj);

        switch (PG_SURF_BytesPerPixel(surf)) {
            case 1:
                for (h = 0; h < surf->h; ++h) {
                    Uint8 *ptr = (Uint8 *)DATAROW(surf->pixels, h, surf->pitch,
                                                  surf->h, flipped);
                    for (w = 0; w < surf->w; ++w) {
                        color = *ptr++;
                        data[0] = (char)surf_palette->colors[color].r;
                        data[1] = (char)surf_palette->colors[color].g;
                        data[2] = (char)surf_palette->colors[color].b;
                        data += 3;
                    }
                    pad(&data, padding);
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
                    pad(&data, padding);
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
                    pad(&data, padding);
                }
                break;
            case 4:
                for (h = 0; h < surf->h; ++h) {
                    Uint32 *ptr = (Uint32 *)DATAROW(
                        surf->pixels, h, surf->pitch, surf->h, flipped);
                    for (w = 0; w < surf->w; ++w) {
                        color = *ptr++;
                        data[0] = (char)(((color & Rmask) >> Rshift) << Rloss);
                        data[1] = (char)(((color & Gmask) >> Gshift) << Gloss);
                        data[2] = (char)(((color & Bmask) >> Bshift) << Bloss);
                        data += 3;
                    }
                    pad(&data, padding);
                }
                break;
        }

        pgSurface_Unlock(surfobj);
    }
    else if (!strcmp(format, "RGBX") || !strcmp(format, "RGBA")) {
        pgSurface_Lock(surfobj);
        switch (PG_SURF_BytesPerPixel(surf)) {
            case 1:
                for (h = 0; h < surf->h; ++h) {
                    Uint8 *ptr = (Uint8 *)DATAROW(surf->pixels, h, surf->pitch,
                                                  surf->h, flipped);
                    for (w = 0; w < surf->w; ++w) {
                        color = *ptr++;
                        data[0] = (char)surf_palette->colors[color].r;
                        data[1] = (char)surf_palette->colors[color].g;
                        data[2] = (char)surf_palette->colors[color].b;
                        data[3] = hascolorkey ? (char)(color != colorkey) * 255
                                              : (char)255;
                        data += 4;
                    }
                    pad(&data, padding);
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
                    pad(&data, padding);
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
                    pad(&data, padding);
                }
                break;
            case 4:
                tobytes_surf_32bpp(surf, format_details, flipped, hascolorkey,
                                   colorkey, data, 0, 3, padding);
                break;
        }
        pgSurface_Unlock(surfobj);
    }
    else if (!strcmp(format, "ARGB")) {
        pgSurface_Lock(surfobj);
        switch (PG_SURF_BytesPerPixel(surf)) {
            case 1:
                for (h = 0; h < surf->h; ++h) {
                    Uint8 *ptr = (Uint8 *)DATAROW(surf->pixels, h, surf->pitch,
                                                  surf->h, flipped);
                    for (w = 0; w < surf->w; ++w) {
                        color = *ptr++;
                        data[1] = (char)surf_palette->colors[color].r;
                        data[2] = (char)surf_palette->colors[color].g;
                        data[3] = (char)surf_palette->colors[color].b;
                        data[0] = (char)255;
                        data += 4;
                    }
                    pad(&data, padding);
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
                    pad(&data, padding);
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
                    pad(&data, padding);
                }
                break;
            case 4:
                tobytes_surf_32bpp(surf, format_details, flipped, hascolorkey,
                                   colorkey, data, 1, 0, padding);
                break;
        }
        pgSurface_Unlock(surfobj);
    }
    else if (!strcmp(format, "BGRA")) {
        pgSurface_Lock(surfobj);
        switch (PG_SURF_BytesPerPixel(surf)) {
            case 1:
                for (h = 0; h < surf->h; ++h) {
                    Uint8 *ptr = (Uint8 *)DATAROW(surf->pixels, h, surf->pitch,
                                                  surf->h, flipped);
                    for (w = 0; w < surf->w; ++w) {
                        color = *ptr++;
                        data[2] = (char)surf_palette->colors[color].r;
                        data[1] = (char)surf_palette->colors[color].g;
                        data[0] = (char)surf_palette->colors[color].b;
                        data[3] = (char)255;
                        data += 4;
                    }
                    pad(&data, padding);
                }
                break;
            case 2:
                for (h = 0; h < surf->h; ++h) {
                    Uint16 *ptr = (Uint16 *)DATAROW(
                        surf->pixels, h, surf->pitch, surf->h, flipped);
                    for (w = 0; w < surf->w; ++w) {
                        color = *ptr++;
                        data[2] = (char)(((color & Rmask) >> Rshift) << Rloss);
                        data[1] = (char)(((color & Gmask) >> Gshift) << Gloss);
                        data[0] = (char)(((color & Bmask) >> Bshift) << Bloss);
                        data[3] = (char)(Amask ? (((color & Amask) >> Ashift)
                                                  << Aloss)
                                               : 255);
                        data += 4;
                    }
                    pad(&data, padding);
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
                        data[2] = (char)(((color & Rmask) >> Rshift) << Rloss);
                        data[1] = (char)(((color & Gmask) >> Gshift) << Gloss);
                        data[0] = (char)(((color & Bmask) >> Bshift) << Bloss);
                        data[3] = (char)(Amask ? (((color & Amask) >> Ashift)
                                                  << Aloss)
                                               : 255);
                        data += 4;
                    }
                    pad(&data, padding);
                }
                break;
            case 4:
                for (h = 0; h < surf->h; ++h) {
                    Uint32 *ptr = (Uint32 *)DATAROW(
                        surf->pixels, h, surf->pitch, surf->h, flipped);
                    for (w = 0; w < surf->w; ++w) {
                        color = *ptr++;
                        data[2] = (char)(((color & Rmask) >> Rshift) << Rloss);
                        data[1] = (char)(((color & Gmask) >> Gshift) << Gloss);
                        data[0] = (char)(((color & Bmask) >> Bshift) << Bloss);
                        data[3] = (char)(Amask ? (((color & Amask) >> Ashift)
                                                  << Aloss)
                                               : 255);
                        data += 4;
                    }
                    pad(&data, padding);
                }
                break;
        }
        pgSurface_Unlock(surfobj);
    }
    else if (!strcmp(format, "ABGR")) {
        pgSurface_Lock(surfobj);
        switch (PG_SURF_BytesPerPixel(surf)) {
            case 1:
                for (h = 0; h < surf->h; ++h) {
                    Uint8 *ptr = (Uint8 *)DATAROW(surf->pixels, h, surf->pitch,
                                                  surf->h, flipped);
                    for (w = 0; w < surf->w; ++w) {
                        color = *ptr++;
                        data[3] = (char)surf_palette->colors[color].r;
                        data[2] = (char)surf_palette->colors[color].g;
                        data[1] = (char)surf_palette->colors[color].b;
                        data[0] = (char)255;
                        data += 4;
                    }
                    pad(&data, padding);
                }
                break;
            case 2:
                for (h = 0; h < surf->h; ++h) {
                    Uint16 *ptr = (Uint16 *)DATAROW(
                        surf->pixels, h, surf->pitch, surf->h, flipped);
                    for (w = 0; w < surf->w; ++w) {
                        color = *ptr++;
                        data[3] = (char)(((color & Rmask) >> Rshift) << Rloss);
                        data[2] = (char)(((color & Gmask) >> Gshift) << Gloss);
                        data[1] = (char)(((color & Bmask) >> Bshift) << Bloss);
                        data[0] = (char)(Amask ? (((color & Amask) >> Ashift)
                                                  << Aloss)
                                               : 255);
                        data += 4;
                    }
                    pad(&data, padding);
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
                        data[3] = (char)(((color & Rmask) >> Rshift) << Rloss);
                        data[2] = (char)(((color & Gmask) >> Gshift) << Gloss);
                        data[1] = (char)(((color & Bmask) >> Bshift) << Bloss);
                        data[0] = (char)(Amask ? (((color & Amask) >> Ashift)
                                                  << Aloss)
                                               : 255);
                        data += 4;
                    }
                    pad(&data, padding);
                }
                break;
            case 4:
                for (h = 0; h < surf->h; ++h) {
                    Uint32 *ptr = (Uint32 *)DATAROW(
                        surf->pixels, h, surf->pitch, surf->h, flipped);
                    for (w = 0; w < surf->w; ++w) {
                        color = *ptr++;
                        data[3] = (char)(((color & Rmask) >> Rshift) << Rloss);
                        data[2] = (char)(((color & Gmask) >> Gshift) << Gloss);
                        data[1] = (char)(((color & Bmask) >> Bshift) << Bloss);
                        data[0] = (char)(Amask ? (((color & Amask) >> Ashift)
                                                  << Aloss)
                                               : 255);
                        data += 4;
                    }
                    pad(&data, padding);
                }
                break;
        }
        pgSurface_Unlock(surfobj);
    }
    else if (!strcmp(format, "RGBA_PREMULT")) {
        pgSurface_Lock(surfobj);
        switch (PG_SURF_BytesPerPixel(surf)) {
            case 2:
                for (h = 0; h < surf->h; ++h) {
                    Uint16 *ptr = (Uint16 *)DATAROW(
                        surf->pixels, h, surf->pitch, surf->h, flipped);
                    for (w = 0; w < surf->w; ++w) {
                        color = *ptr++;
                        alpha = ((color & Amask) >> Ashift) << Aloss;
                        data[0] = PREMUL_PIXEL_ALPHA(
                            ((color & Rmask) >> Rshift) << Rloss, alpha);
                        data[1] = PREMUL_PIXEL_ALPHA(
                            ((color & Gmask) >> Gshift) << Gloss, alpha);
                        data[2] = PREMUL_PIXEL_ALPHA(
                            ((color & Bmask) >> Bshift) << Bloss, alpha);
                        data[3] = (char)alpha;
                        data += 4;
                    }
                    pad(&data, padding);
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
                        data[0] = PREMUL_PIXEL_ALPHA(
                            ((color & Rmask) >> Rshift) << Rloss, alpha);
                        data[1] = PREMUL_PIXEL_ALPHA(
                            ((color & Gmask) >> Gshift) << Gloss, alpha);
                        data[2] = PREMUL_PIXEL_ALPHA(
                            ((color & Bmask) >> Bshift) << Bloss, alpha);
                        data[3] = (char)alpha;
                        data += 4;
                    }
                    pad(&data, padding);
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
                            data[0] = PREMUL_PIXEL_ALPHA(
                                ((color & Rmask) >> Rshift) << Rloss, alpha);
                            data[1] = PREMUL_PIXEL_ALPHA(
                                ((color & Gmask) >> Gshift) << Gloss, alpha);
                            data[2] = PREMUL_PIXEL_ALPHA(
                                ((color & Bmask) >> Bshift) << Bloss, alpha);
                        }
                        data[3] = (char)alpha;
                        data += 4;
                    }
                    pad(&data, padding);
                }
                break;
        }
        pgSurface_Unlock(surfobj);
    }
    else if (!strcmp(format, "ARGB_PREMULT")) {
        pgSurface_Lock(surfobj);
        switch (PG_SURF_BytesPerPixel(surf)) {
            case 2:
                for (h = 0; h < surf->h; ++h) {
                    Uint16 *ptr = (Uint16 *)DATAROW(
                        surf->pixels, h, surf->pitch, surf->h, flipped);
                    for (w = 0; w < surf->w; ++w) {
                        color = *ptr++;
                        alpha = ((color & Amask) >> Ashift) << Aloss;
                        data[1] = PREMUL_PIXEL_ALPHA(
                            ((color & Rmask) >> Rshift) << Rloss, alpha);
                        data[2] = PREMUL_PIXEL_ALPHA(
                            ((color & Gmask) >> Gshift) << Gloss, alpha);
                        data[3] = PREMUL_PIXEL_ALPHA(
                            ((color & Bmask) >> Bshift) << Bloss, alpha);
                        data[0] = (char)alpha;
                        data += 4;
                    }
                    pad(&data, padding);
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
                        data[1] = PREMUL_PIXEL_ALPHA(
                            ((color & Rmask) >> Rshift) << Rloss, alpha);
                        data[2] = PREMUL_PIXEL_ALPHA(
                            ((color & Gmask) >> Gshift) << Gloss, alpha);
                        data[3] = PREMUL_PIXEL_ALPHA(
                            ((color & Bmask) >> Bshift) << Bloss, alpha);
                        data[0] = (char)alpha;
                        data += 4;
                    }
                    pad(&data, padding);
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
                            data[1] = PREMUL_PIXEL_ALPHA(
                                ((color & Rmask) >> Rshift) << Rloss, alpha);
                            data[2] = PREMUL_PIXEL_ALPHA(
                                ((color & Gmask) >> Gshift) << Gloss, alpha);
                            data[3] = PREMUL_PIXEL_ALPHA(
                                ((color & Bmask) >> Bshift) << Bloss, alpha);
                        }
                        data[0] = (char)alpha;
                        data += 4;
                    }
                    pad(&data, padding);
                }
                break;
        }
        pgSurface_Unlock(surfobj);
    }

    return bytes;
}

PyObject *
image_frombytes(PyObject *self, PyObject *arg, PyObject *kwds)
{
    PyObject *bytes;
    char *format, *data;
    SDL_Surface *surf = NULL;
    int w, h, flipped = 0, pitch = -1;
    Py_ssize_t len;
    int loopw, looph;

#ifdef _MSC_VER
    /* MSVC static analyzer false alarm: assure format is NULL-terminated by
     * making analyzer assume it was initialised */
    __analysis_assume(format = "inited");
#endif

    static char *kwids[] = {"bytes",   "size",  "format",
                            "flipped", "pitch", NULL};
    if (!PyArg_ParseTupleAndKeywords(arg, kwds, "O!(ii)s|ii", kwids,
                                     &PyBytes_Type, &bytes, &w, &h, &format,
                                     &flipped, &pitch)) {
        return NULL;
    }

    if (w < 1 || h < 1) {
        return RAISE(PyExc_ValueError,
                     "Resolution must be nonzero positive values");
    }

    PyBytes_AsStringAndSize(bytes, &data, &len);

    if (!strcmp(format, "P")) {
        if (pitch == -1) {
            pitch = w;
        }
        else if (pitch < w) {
            return RAISE(PyExc_ValueError,
                         "Pitch must be greater than or equal to the width "
                         "as per the format");
        }

        if (len != (Py_ssize_t)pitch * h) {
            return RAISE(
                PyExc_ValueError,
                "Bytes length does not equal format and resolution size");
        }

        surf = PG_CreateSurface(w, h, SDL_PIXELFORMAT_INDEX8);
        if (!surf) {
            return RAISE(pgExc_SDLError, SDL_GetError());
        }
        SDL_LockSurface(surf);
        for (looph = 0; looph < h; ++looph) {
            memcpy(((char *)surf->pixels) + looph * surf->pitch,
                   DATAROW(data, looph, pitch, h, flipped), w);
        }
        SDL_UnlockSurface(surf);
    }
    else if (!strcmp(format, "RGB")) {
        if (pitch == -1) {
            pitch = w * 3;
        }
        else if (pitch < w * 3) {
            return RAISE(PyExc_ValueError,
                         "Pitch must be greater than or equal to the width * "
                         "3 as per the format");
        }

        if (len != (Py_ssize_t)pitch * h) {
            return RAISE(
                PyExc_ValueError,
                "Bytes length does not equal format and resolution size");
        }

#if SDL_BYTEORDER == SDL_LIL_ENDIAN
        surf = PG_CreateSurface(w, h, SDL_PIXELFORMAT_BGR24);
#else
        surf = PG_CreateSurface(w, h, SDL_PIXELFORMAT_RGB24);
#endif
        if (!surf) {
            return RAISE(pgExc_SDLError, SDL_GetError());
        }
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

            data += pitch - w * 3;
        }
        SDL_UnlockSurface(surf);
    }
    else if (!strcmp(format, "RGBA") || !strcmp(format, "RGBX")) {
        if (pitch == -1) {
            pitch = w * 4;
        }
        else if (pitch < w * 4) {
            return RAISE(PyExc_ValueError,
                         "Pitch must be greater than or equal to the width * "
                         "4 as per the format");
        }

        int alphamult = !strcmp(format, "RGBA");
        if (len != (Py_ssize_t)pitch * h) {
            return RAISE(
                PyExc_ValueError,
                "Bytes length does not equal format and resolution size");
        }
        surf = PG_CreateSurface(
            w, h,
            (alphamult ? SDL_PIXELFORMAT_RGBA32 : PG_PIXELFORMAT_RGBX32));
        if (!surf) {
            return RAISE(pgExc_SDLError, SDL_GetError());
        }
        SDL_LockSurface(surf);
        for (looph = 0; looph < h; ++looph) {
            Uint32 *pix = (Uint32 *)DATAROW(surf->pixels, looph, surf->pitch,
                                            h, flipped);
            memcpy(pix, data, w * sizeof(Uint32));
            data += pitch;
        }
        SDL_UnlockSurface(surf);
    }
    else if (!strcmp(format, "BGRA")) {
        if (pitch == -1) {
            pitch = w * 4;
        }
        else if (pitch < w * 4) {
            return RAISE(PyExc_ValueError,
                         "Pitch must be greater than or equal to the width * "
                         "4 as per the format");
        }

        if (len != (Py_ssize_t)pitch * h) {
            return RAISE(
                PyExc_ValueError,
                "Bytes length does not equal format and resolution size");
        }
        surf = PG_CreateSurface(w, h, SDL_PIXELFORMAT_BGRA32);
        if (!surf) {
            return RAISE(pgExc_SDLError, SDL_GetError());
        }
        SDL_LockSurface(surf);
        for (looph = 0; looph < h; ++looph) {
            Uint32 *pix = (Uint32 *)DATAROW(surf->pixels, looph, surf->pitch,
                                            h, flipped);
            memcpy(pix, data, w * sizeof(Uint32));
            data += pitch;
        }
        SDL_UnlockSurface(surf);
    }
    else if (!strcmp(format, "ARGB")) {
        if (pitch == -1) {
            pitch = w * 4;
        }
        else if (pitch < w * 4) {
            return RAISE(PyExc_ValueError,
                         "Pitch must be greater than or equal to the width * "
                         "4 as per the format");
        }

        if (len != (Py_ssize_t)pitch * h) {
            return RAISE(
                PyExc_ValueError,
                "Bytes length does not equal format and resolution size");
        }
        surf = PG_CreateSurface(w, h, SDL_PIXELFORMAT_ARGB32);
        if (!surf) {
            return RAISE(pgExc_SDLError, SDL_GetError());
        }
        SDL_LockSurface(surf);
        for (looph = 0; looph < h; ++looph) {
            Uint32 *pix = (Uint32 *)DATAROW(surf->pixels, looph, surf->pitch,
                                            h, flipped);
            memcpy(pix, data, w * sizeof(Uint32));
            data += pitch;
        }
        SDL_UnlockSurface(surf);
    }
    else if (!strcmp(format, "ABGR")) {
        if (pitch == -1) {
            pitch = w * 4;
        }
        else if (pitch < w * 4) {
            return RAISE(PyExc_ValueError,
                         "Pitch must be greater than or equal to the width * "
                         "4 as per the format");
        }

        if (len != (Py_ssize_t)pitch * h) {
            return RAISE(
                PyExc_ValueError,
                "Bytes length does not equal format and resolution size");
        }
        surf = PG_CreateSurface(w, h, SDL_PIXELFORMAT_ABGR32);
        if (!surf) {
            return RAISE(pgExc_SDLError, SDL_GetError());
        }
        SDL_LockSurface(surf);
        for (looph = 0; looph < h; ++looph) {
            Uint32 *pix = (Uint32 *)DATAROW(surf->pixels, looph, surf->pitch,
                                            h, flipped);
            memcpy(pix, data, w * sizeof(Uint32));
            data += pitch;
        }
        SDL_UnlockSurface(surf);
    }
    else {
        return RAISE(PyExc_ValueError, "Unrecognized type of format");
    }

    return (PyObject *)pgSurface_New(surf);
}

PyObject *
image_tostring(PyObject *self, PyObject *arg, PyObject *kwarg)
{
    if (PyErr_WarnEx(PyExc_DeprecationWarning,
                     "pygame.image.tostring deprecated since 2.3.0",
                     1) == -1) {
        return NULL;
    }

    return image_tobytes(self, arg, kwarg);
}

PyObject *
image_fromstring(PyObject *self, PyObject *arg, PyObject *kwarg)
{
    if (PyErr_WarnEx(PyExc_DeprecationWarning,
                     "pygame.image.fromstring deprecated since 2.3.0",
                     1) == -1) {
        return NULL;
    }

    return image_frombytes(self, arg, kwarg);
}

static int
_as_read_buffer(PyObject *obj, const void **buffer, Py_ssize_t *buffer_len)
{
    Py_buffer view;

    if (obj == NULL || buffer == NULL || buffer_len == NULL) {
        return -1;
    }
    if (PyObject_GetBuffer(obj, &view, PyBUF_SIMPLE) != 0) {
        return -1;
    }

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
image_frombuffer(PyObject *self, PyObject *arg, PyObject *kwds)
{
    PyObject *buffer;
    char *format, *data;
    SDL_Surface *surf = NULL;
    int w, h, pitch = -1;
    Py_ssize_t len;
    pgSurfaceObject *surfobj;

#ifdef _MSC_VER
    /* MSVC static analyzer false alarm: assure format is NULL-terminated by
     * making analyzer assume it was initialised */
    __analysis_assume(format = "inited");
#endif

    static char *kwids[] = {"buffer", "size", "format", "pitch", NULL};
    if (!PyArg_ParseTupleAndKeywords(arg, kwds, "O(ii)s|i", kwids, &buffer, &w,
                                     &h, &format, &pitch)) {
        return NULL;
    }

    if (w < 1 || h < 1) {
        return RAISE(PyExc_ValueError,
                     "Resolution must be nonzero positive values");
    }

    /* breaking constness here, we should really not change this string */
    if (pgObject_AsCharBuffer(buffer, (const char **)&data, &len) == -1) {
        return NULL;
    }

    if (!strcmp(format, "P")) {
        if (pitch == -1) {
            pitch = w;
        }
        else if (pitch < w) {
            return RAISE(PyExc_ValueError,
                         "Pitch must be greater than or equal to the width "
                         "as per the format");
        }

        if (len != (Py_ssize_t)pitch * h) {
            return RAISE(
                PyExc_ValueError,
                "Buffer length does not equal format and resolution size");
        }

        surf = PG_CreateSurfaceFrom(w, h, SDL_PIXELFORMAT_INDEX8, data, pitch);
    }
    else if (!strcmp(format, "RGB")) {
        if (pitch == -1) {
            pitch = w * 3;
        }
        else if (pitch < w * 3) {
            return RAISE(PyExc_ValueError,
                         "Pitch must be greater than or equal to the width * "
                         "3 as per the format");
        }

        if (len != (Py_ssize_t)pitch * h) {
            return RAISE(
                PyExc_ValueError,
                "Buffer length does not equal format and resolution size");
        }
        surf = PG_CreateSurfaceFrom(w, h, SDL_PIXELFORMAT_RGB24, data, pitch);
    }
    else if (!strcmp(format, "BGR")) {
        if (pitch == -1) {
            pitch = w * 3;
        }
        else if (pitch < w * 3) {
            return RAISE(PyExc_ValueError,
                         "Pitch must be greater than or equal to the width * "
                         "3 as per the format");
        }

        if (len != (Py_ssize_t)pitch * h) {
            return RAISE(
                PyExc_ValueError,
                "Buffer length does not equal format and resolution size");
        }
        surf = PG_CreateSurfaceFrom(w, h, SDL_PIXELFORMAT_BGR24, data, pitch);
    }
    else if (!strcmp(format, "BGRA")) {
        if (pitch == -1) {
            pitch = w * 4;
        }
        else if (pitch < w * 4) {
            return RAISE(PyExc_ValueError,
                         "Pitch must be greater than or equal to the width * "
                         "4 as per the format");
        }

        if (len != (Py_ssize_t)pitch * h) {
            return RAISE(
                PyExc_ValueError,
                "Buffer length does not equal format and resolution size");
        }
        surf = PG_CreateSurfaceFrom(w, h, SDL_PIXELFORMAT_BGRA32, data, pitch);
    }
    else if (!strcmp(format, "RGBA") || !strcmp(format, "RGBX")) {
        if (pitch == -1) {
            pitch = w * 4;
        }
        else if (pitch < w * 4) {
            return RAISE(PyExc_ValueError,
                         "Pitch must be greater than or equal to the width * "
                         "4 as per the format");
        }

        int alphamult = !strcmp(format, "RGBA");
        if (len != (Py_ssize_t)pitch * h) {
            return RAISE(
                PyExc_ValueError,
                "Buffer length does not equal format and resolution size");
        }
        surf = PG_CreateSurfaceFrom(
            w, h, (alphamult ? SDL_PIXELFORMAT_RGBA32 : PG_PIXELFORMAT_RGBX32),
            data, pitch);
    }
    else if (!strcmp(format, "ARGB")) {
        if (pitch == -1) {
            pitch = w * 4;
        }
        else if (pitch < w * 4) {
            return RAISE(PyExc_ValueError,
                         "Pitch must be greater than or equal to the width * "
                         "4 as per the format");
        }

        if (len != (Py_ssize_t)pitch * h) {
            return RAISE(
                PyExc_ValueError,
                "Buffer length does not equal format and resolution size");
        }
        surf = PG_CreateSurfaceFrom(w, h, SDL_PIXELFORMAT_ARGB32, data, pitch);
    }
    else {
        return RAISE(PyExc_ValueError, "Unrecognized type of format");
    }

    if (!surf) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }
    surfobj = (pgSurfaceObject *)pgSurface_New(surf);
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
               x - x0 < TGA_RLE_MAX) {
            x++;
        }
        /* use a repetition chunk iff the repeated pixels would consume
           two bytes or more */
        if ((x - x0 - 1) * bpp >= 2 || x == w) {
            /* output previous raw chunks */
            if (x - x0 == 1) {
                /* No need for repeat chunk, do a raw chunk */
                x0++;
            }
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
 *     32bpp palette if colorkeying is used.
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
    struct TGAheader h;
    int srcbpp;
    SDL_BlendMode surf_blendmode;
    int have_surf_colorkey = 0;
    Uint32 surf_colorkey;
    SDL_Rect r;
    int bpp;
    Uint8 *rlebuf = NULL;

    h.infolen = 0;
    SETLE16(h.cmap_start, 0);

    srcbpp = PG_SURF_BitsPerPixel(surface);
    if (srcbpp < 8) {
        SDL_SetError("cannot save <8bpp images as TGA");
        return -1;
    }

#if SDL_VERSION_ATLEAST(3, 0, 0)
    const SDL_PixelFormatDetails *surf_format =
        SDL_GetPixelFormatDetails(surface->format);
    if (!surf_format) {
        return -1;
    }
    SDL_PixelFormat output_format;

    SDL_Palette *surf_palette = PG_GetSurfacePalette(surface);
#else
    SDL_PixelFormat *surf_format = surface->format;
    SDL_Palette *surf_palette = surface->format->palette;
    SDL_PixelFormatEnum output_format;
#endif

    if (!PG_GetSurfaceBlendMode(surface, &surf_blendmode)) {
        return -1;
    }

    if ((have_surf_colorkey = SDL_HasColorKey(surface))) {
        SDL_GetColorKey(surface, &surf_colorkey);
    }

    if (srcbpp == 8) {
        h.has_cmap = 1;
        h.type = TGA_TYPE_INDEXED;
        if (have_surf_colorkey) {
            h.cmap_bits = 32;
        }
        else {
            h.cmap_bits = 24;
        }
        SETLE16(h.cmap_len, surf_palette->ncolors);
        h.pixel_bits = 8;
        output_format = SDL_PIXELFORMAT_INDEX8;
    }
    else {
        h.has_cmap = 0;
        h.type = TGA_TYPE_RGB;
        h.cmap_bits = 0;
        SETLE16(h.cmap_len, 0);
        if (surf_format->Amask) {
            alpha = 1;
            h.pixel_bits = 32;
            output_format = SDL_PIXELFORMAT_BGRA32;
        }
        else {
            h.pixel_bits = 24;
            output_format = SDL_PIXELFORMAT_BGR24;
        }
    }
    bpp = h.pixel_bits >> 3;
    if (rle) {
        h.type += TGA_TYPE_RLE;
    }

    SETLE16(h.yorigin, 0);
    SETLE16(h.xorigin, 0);
    SETLE16(h.width, surface->w);
    SETLE16(h.height, surface->h);
    h.flags = TGA_ORIGIN_UPPER | (alpha ? 8 : 0);

#if SDL_VERSION_ATLEAST(3, 0, 0)
    if (!SDL_WriteIO(out, &h, sizeof(h)))
#else
    if (!SDL_RWwrite(out, &h, sizeof(h), 1))
#endif
        return -1;

    if (h.has_cmap) {
        int i;
        Uint8 entry[4];
        for (i = 0; i < surf_palette->ncolors; i++) {
            entry[0] = surf_palette->colors[i].b;
            entry[1] = surf_palette->colors[i].g;
            entry[2] = surf_palette->colors[i].r;
            entry[3] = ((unsigned)i == surf_colorkey) ? 0 : 0xff;
#if SDL_VERSION_ATLEAST(3, 0, 0)
            if (!SDL_WriteIO(out, entry, h.cmap_bits >> 3))
#else
            if (!SDL_RWwrite(out, entry, h.cmap_bits >> 3, 1))
#endif
                return -1;
        }
    }

    linebuf = PG_CreateSurface(surface->w, 1, output_format);
    if (!linebuf) {
        return -1;
    }

    if (h.has_cmap) {
        if (!PG_SetSurfacePalette(linebuf, surf_palette)) {
            goto error; /* SDL error already set. */
        }
    }

    if (rle) {
        rlebuf = malloc(bpp * surface->w + 1 + surface->w / TGA_RLE_MAX);
        if (!rlebuf) {
            SDL_SetError("out of memory");
            goto error;
        }
    }

    /* Temporarily set SDL_BLENDMODE_NONE so that copies are opaque */
    if (!PG_SetSurfaceBlendMode(surface, SDL_BLENDMODE_NONE)) {
        goto error;
    }

    r.x = 0;
    r.w = surface->w;
    r.h = 1;
    for (r.y = 0; r.y < surface->h; r.y++) {
        int n;
        void *buf;
#if SDL_VERSION_ATLEAST(3, 0, 0)
        if (!SDL_BlitSurface(surface, &r, linebuf, NULL))
#else
        if (SDL_BlitSurface(surface, &r, linebuf, NULL) < 0)
#endif
            break;
        if (rle) {
            buf = rlebuf;
            n = rle_line(linebuf->pixels, rlebuf, surface->w, bpp);
        }
        else {
            buf = linebuf->pixels;
            n = surface->w * bpp;
        }
#if SDL_VERSION_ATLEAST(3, 0, 0)
        if (!SDL_WriteIO(out, buf, n))
#else
        if (!SDL_RWwrite(out, buf, n, 1))
#endif
            break;
    }

    /* restore flags */
    if (!PG_SetSurfaceBlendMode(surface, surf_blendmode)) {
        goto error;
    }

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
    if (!out) {
        return -1;
    }
    ret = SaveTGA_RW(surface, out, rle);
    SDL_RWclose(out);
    return ret;
}

static PyObject *
image_load_sized_svg(PyObject *self, PyObject *args, PyObject *kwargs)
{
    if (ext_load_sized_svg) {
        return PyObject_Call(ext_load_sized_svg, args, kwargs);
    }

    return RAISE(PyExc_NotImplementedError,
                 "Support for sized svg image loading was not compiled in.");
}

static PyObject *
image_load_animation(PyObject *self, PyObject *args, PyObject *kwargs)
{
    if (ext_load_animation) {
        return PyObject_Call(ext_load_animation, args, kwargs);
    }

    return RAISE(PyExc_NotImplementedError,
                 "Support for animation loading was not compiled in.");
}

static PyMethodDef _image_methods[] = {
    {"load_basic", (PyCFunction)image_load_basic, METH_O, DOC_IMAGE_LOADBASIC},
    {"load_extended", (PyCFunction)image_load_extended,
     METH_VARARGS | METH_KEYWORDS, DOC_IMAGE_LOADEXTENDED},
    {"load", (PyCFunction)image_load, METH_VARARGS | METH_KEYWORDS,
     DOC_IMAGE_LOAD},
    {"load_sized_svg", (PyCFunction)image_load_sized_svg,
     METH_VARARGS | METH_KEYWORDS, DOC_IMAGE_LOADSIZEDSVG},
    {"load_animation", (PyCFunction)image_load_animation,
     METH_VARARGS | METH_KEYWORDS, DOC_IMAGE_LOADANIMATION},

    {"save_extended", (PyCFunction)image_save_extended,
     METH_VARARGS | METH_KEYWORDS, DOC_IMAGE_SAVEEXTENDED},
    {"save", (PyCFunction)image_save, METH_VARARGS | METH_KEYWORDS,
     DOC_IMAGE_SAVE},
    {"get_extended", (PyCFunction)image_get_extended, METH_NOARGS,
     DOC_IMAGE_GETEXTENDED},
    {"get_sdl_image_version", (PyCFunction)image_get_sdl_image_version,
     METH_VARARGS | METH_KEYWORDS, DOC_IMAGE_GETSDLIMAGEVERSION},

    {"tostring", (PyCFunction)image_tostring, METH_VARARGS | METH_KEYWORDS,
     DOC_IMAGE_TOSTRING},
    {"tobytes", (PyCFunction)image_tobytes, METH_VARARGS | METH_KEYWORDS,
     DOC_IMAGE_TOBYTES},
    {"fromstring", (PyCFunction)image_fromstring, METH_VARARGS | METH_KEYWORDS,
     DOC_IMAGE_FROMSTRING},
    {"frombytes", (PyCFunction)image_frombytes, METH_VARARGS | METH_KEYWORDS,
     DOC_IMAGE_FROMBYTES},
    {"frombuffer", (PyCFunction)image_frombuffer, METH_VARARGS | METH_KEYWORDS,
     DOC_IMAGE_FROMBUFFER},
    {NULL, NULL, 0, NULL}};

MODINIT_DEFINE(image)
{
    PyObject *module;
    PyObject *extmodule;

    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "image",
                                         DOC_IMAGE,
                                         -1,
                                         _image_methods,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL};

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

    /* create the module */
    module = PyModule_Create(&_module);
    if (module == NULL) {
        return NULL;
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
        extverobj =
            PyObject_GetAttrString(extmodule, "_get_sdl_image_version");
        if (!extverobj) {
            goto error;
        }
        ext_load_sized_svg =
            PyObject_GetAttrString(extmodule, "_load_sized_svg");
        if (!ext_load_sized_svg) {
            goto error;
        }
        ext_load_animation =
            PyObject_GetAttrString(extmodule, "_load_animation");
        if (!ext_load_animation) {
            goto error;
        }
        Py_DECREF(extmodule);
    }
    else {
        // if the module could not be loaded, don't treat it like an error
        PyErr_Clear();
    }
    return module;

error:
    Py_XDECREF(extloadobj);
    Py_XDECREF(extsaveobj);
    Py_XDECREF(extverobj);
    Py_XDECREF(ext_load_sized_svg);
    Py_XDECREF(ext_load_animation);
    Py_DECREF(extmodule);
    Py_DECREF(module);
    return NULL;
}
