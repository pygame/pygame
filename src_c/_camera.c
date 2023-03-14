/*
  pygame - Python Game Library

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

*/

/*
 * Camera - webcam support for pygame
 * Author: Nirav Patel
 *
 * This module allows for use of v4l2 webcams in pygame. The code is written
 * such that adding support for vfw cameras should be possible without
 * much modification of existing functions. v4l2 functions are kept separate
 * from functions available to pygame users and generic functions like
 * colorspace conversion.
 *
 * There is currently support for cameras that support MMAP and use
 * pixelformats of RGB24, RGB444, YUYV, SBGGR8, and YUV420. To add support for
 * additional pixelformats, add them to v4l2_init_device and
 * v4l2_process_image, and add functions to convert the format to packed RGB,
 * YUV, and HSV.
 */

#include "camera.h"
#include "pgcompat.h"

/*
#if defined(__unix__) || !defined(__APPLE__)
#else
    #define V4L2_PIX_FMT_RGB24 1
    #define V4L2_PIX_FMT_RGB444 1
#endif
*/

/* functions available to pygame users */
PyObject *
surf_colorspace(PyObject *self, PyObject *arg);
PyObject *
list_cameras(PyObject *self, PyObject *arg);
PyObject *
camera_start(pgCameraObject *self, PyObject *args);
PyObject *
camera_stop(pgCameraObject *self, PyObject *args);
PyObject *
camera_get_controls(pgCameraObject *self, PyObject *args);
PyObject *
camera_set_controls(pgCameraObject *self, PyObject *arg, PyObject *kwds);
PyObject *
camera_get_size(pgCameraObject *self, PyObject *args);
PyObject *
camera_query_image(pgCameraObject *self, PyObject *args);
PyObject *
camera_get_image(pgCameraObject *self, PyObject *arg);
PyObject *
camera_get_raw(pgCameraObject *self, PyObject *args);

/*
 * Functions available to pygame users.  The idea is to make these as simple as
 * possible, and merely have them call functions specific to the type of
 * camera being used to do all the real work.  It currently just calls v4l2_*
 * functions, but it could check something like self->cameratype and depending
 * on the result, call v4l, v4l2, vfw, or other functions.
 */

/* colorspace() - Surface colorspace conversion */
PyObject *
surf_colorspace(PyObject *self, PyObject *arg)
{
    pgSurfaceObject *surfobj, *surfobj2;
    SDL_Surface *surf, *newsurf;
    char *color;
    int cspace;
    surfobj2 = NULL;

#ifdef _MSC_VER
    /* MSVC static analyzer false alarm: assure color is NULL-terminated by
     * making analyzer assume it was initialised */
    __analysis_assume(color = "inited");
#endif

    /*get all the arguments*/
    if (!PyArg_ParseTuple(arg, "O!s|O!", &pgSurface_Type, &surfobj, &color,
                          &pgSurface_Type, &surfobj2))
        return NULL;

    if (!strcmp(color, "YUV")) {
        cspace = YUV_OUT;
    }
    else if (!strcmp(color, "HSV")) {
        cspace = HSV_OUT;
    }
    else {
        return RAISE(PyExc_ValueError, "Incorrect colorspace value");
    }

    surf = pgSurface_AsSurface(surfobj);

    if (!surf)
        return RAISE(pgExc_SDLError, "display Surface quit");

    if (!surfobj2) {
        newsurf = SDL_CreateRGBSurface(
            0, surf->w, surf->h, surf->format->BitsPerPixel,
            surf->format->Rmask, surf->format->Gmask, surf->format->Bmask,
            surf->format->Amask);
        if (!newsurf) {
            return NULL;
        }
    }
    else {
        newsurf = pgSurface_AsSurface(surfobj2);
        if (!newsurf)
            return RAISE(pgExc_SDLError, "display Surface quit");
    }

    /* check to see if the size is the same. */
    if (newsurf->w != surf->w || newsurf->h != surf->h)
        return RAISE(PyExc_ValueError,
                     "Surfaces not the same width and height.");

    /* check to see if the format of the surface is the same. */
    if (surf->format->BitsPerPixel != newsurf->format->BitsPerPixel)
        return RAISE(PyExc_ValueError, "Surfaces not the same depth");

    SDL_LockSurface(newsurf);
    pgSurface_Lock(surfobj);

    Py_BEGIN_ALLOW_THREADS;
    colorspace(surf, newsurf, cspace);
    Py_END_ALLOW_THREADS;

    pgSurface_Unlock(surfobj);
    SDL_UnlockSurface(newsurf);

    if (surfobj2) {
        Py_INCREF(surfobj2);
        return (PyObject *)surfobj2;
    }
    else
        return (PyObject *)pgSurface_New(newsurf);
}

/* list_cameras() - lists cameras available on the computer */
PyObject *
list_cameras(PyObject *self, PyObject *_null)
{
#if defined(__unix__) || defined(PYGAME_WINDOWS_CAMERA)
    PyObject *ret_list;
    PyObject *string;
#if !defined(PYGAME_WINDOWS_CAMERA)
    char **devices;
#else
    WCHAR **devices;
#endif
    int j, i = 0, num_devices = 0;

    /* TODO for future PRs: errors in these functions are being ignored as
     * of now, and an empty list is being returned by this function */
#if defined(__unix__)
    devices = v4l2_list_cameras(&num_devices);
#elif defined(PYGAME_WINDOWS_CAMERA)
    devices = windows_list_cameras(&num_devices);
#endif

    ret_list = PyList_New(num_devices);
    if (!ret_list) {
        goto error;
    }

    for (i = 0; i < num_devices; i++) {
#if defined(PYGAME_WINDOWS_CAMERA)
        string = PyUnicode_FromWideChar(devices[i], -1);
#else
        string = PyUnicode_FromString(devices[i]);
#endif
        if (!string) {
            goto error;
        }
        /* steals reference to 'string' */
        PyList_SET_ITEM(ret_list, i, string);
        free(devices[i]);
    }
    free(devices);

    return ret_list;
error:
    /* free the remaining devices */
    for (j = i; j < num_devices; j++) {
        free(devices[j]);
    }
    free(devices);
    Py_XDECREF(ret_list);
    return NULL;

#else
    Py_RETURN_NONE;
#endif
}

/* start() - opens, inits, and starts capturing on the camera */
PyObject *
camera_start(pgCameraObject *self, PyObject *_null)
{
#if defined(__unix__)
    if (v4l2_open_device(self) == 0) {
        v4l2_close_device(self);
        return NULL;
    }
    else {
        self->camera_type = CAM_V4L2;
        if (v4l2_init_device(self) == 0) {
            v4l2_close_device(self);
            return NULL;
        }
        if (v4l2_start_capturing(self) == 0) {
            v4l2_close_device(self);
            return NULL;
        }
    }
#elif defined(PYGAME_WINDOWS_CAMERA)
    if (self->open) { /* camera already started */
        Py_RETURN_NONE;
    }

    if (!windows_open_device(self)) {
        windows_close_device(self);
        return NULL;
    }
#endif
    Py_RETURN_NONE;
}

/* stop() - stops capturing, uninits, and closes the camera */
PyObject *
camera_stop(pgCameraObject *self, PyObject *_null)
{
#if defined(__unix__)
    if (v4l2_stop_capturing(self) == 0)
        return NULL;
    if (v4l2_uninit_device(self) == 0)
        return NULL;
    if (v4l2_close_device(self) == 0)
        return NULL;
#elif defined(PYGAME_WINDOWS_CAMERA)
    if (self->open) { /* camera started */
        if (!windows_close_device(self))
            return NULL;
    }
#endif
    Py_RETURN_NONE;
}

/* get_controls() - gets current values of user controls */
/* TODO: Support brightness, contrast, and other common controls */
PyObject *
camera_get_controls(pgCameraObject *self, PyObject *_null)
{
#if defined(__unix__)
    int value;
    if (v4l2_get_control(self->fd, V4L2_CID_HFLIP, &value))
        self->hflip = value;

    if (v4l2_get_control(self->fd, V4L2_CID_VFLIP, &value))
        self->vflip = value;

    if (v4l2_get_control(self->fd, V4L2_CID_BRIGHTNESS, &value))
        self->brightness = value;

    return Py_BuildValue("(NNN)", PyBool_FromLong(self->hflip),
                         PyBool_FromLong(self->vflip),
                         PyLong_FromLong(self->brightness));
#elif defined(PYGAME_WINDOWS_CAMERA)
    return Py_BuildValue("(NNN)", PyBool_FromLong(self->hflip),
                         PyBool_FromLong(self->vflip), PyLong_FromLong(-1));
#endif
    Py_RETURN_NONE;
}

/* set_controls() - changes camera settings if supported by the camera */
PyObject *
camera_set_controls(pgCameraObject *self, PyObject *arg, PyObject *kwds)
{
#if defined(__unix__)
    int hflip = 0, vflip = 0, brightness = 0;
    char *kwids[] = {"hflip", "vflip", "brightness", NULL};

    camera_get_controls(self, NULL);
    hflip = self->hflip;
    vflip = self->vflip;
    brightness = self->brightness;

    if (!PyArg_ParseTupleAndKeywords(arg, kwds, "|iii", kwids, &hflip, &vflip,
                                     &brightness))
        return NULL;

    /* #if defined(__unix__)         */
    if (v4l2_set_control(self->fd, V4L2_CID_HFLIP, hflip))
        self->hflip = hflip;

    if (v4l2_set_control(self->fd, V4L2_CID_VFLIP, vflip))
        self->vflip = vflip;

    if (v4l2_set_control(self->fd, V4L2_CID_BRIGHTNESS, brightness))
        self->brightness = brightness;

    return Py_BuildValue("(NNN)", PyBool_FromLong(self->hflip),
                         PyBool_FromLong(self->vflip),
                         PyLong_FromLong(self->brightness));

#elif defined(PYGAME_WINDOWS_CAMERA)
    int hflip = 0, vflip = 0, brightness = 0;
    char *kwids[] = {"hflip", "vflip", "brightness", NULL};

    hflip = self->hflip;
    vflip = self->vflip;
    brightness = -1;

    if (!PyArg_ParseTupleAndKeywords(arg, kwds, "|iii", kwids, &hflip, &vflip,
                                     &brightness))
        return NULL;

    self->hflip = hflip;
    self->vflip = vflip;

    return Py_BuildValue("(NNN)", PyBool_FromLong(self->hflip),
                         PyBool_FromLong(self->vflip), PyLong_FromLong(-1));
#endif
    Py_RETURN_NONE;
}

/* get_size() - returns the dimensions of the images being recorded */
PyObject *
camera_get_size(pgCameraObject *self, PyObject *_null)
{
#if defined(__unix__) || defined(PYGAME_WINDOWS_CAMERA)
    return Py_BuildValue("(ii)", self->width, self->height);
#endif
    Py_RETURN_NONE;
}

/* query_image() - checks if a frame is ready */
PyObject *
camera_query_image(pgCameraObject *self, PyObject *_null)
{
#if defined(__unix__)
    return PyBool_FromLong(v4l2_query_buffer(self));
#elif defined(PYGAME_WINDOWS_CAMERA)
    int ready;
    if (!windows_frame_ready(self, &ready))
        return NULL;

    return PyBool_FromLong(ready);
#endif
    Py_RETURN_TRUE;
}

/* get_image() - returns an RGB Surface */
/* code to reuse Surface from René Dudfield */
PyObject *
camera_get_image(pgCameraObject *self, PyObject *arg)
{
#if defined(__unix__)
    SDL_Surface *surf = NULL;
    pgSurfaceObject *surfobj = NULL;
    int ret, errno_code = 0;

    if (!PyArg_ParseTuple(arg, "|O!", &pgSurface_Type, &surfobj))
        return NULL;

    if (!surfobj) {
        surf = SDL_CreateRGBSurface(0, self->width, self->height, 24,
                                    0xFF << 16, 0xFF << 8, 0xFF, 0);
    }
    else {
        surf = pgSurface_AsSurface(surfobj);
    }

    if (!surf)
        return NULL;

    if (surf->w != self->width || surf->h != self->height) {
        return RAISE(PyExc_ValueError,
                     "Destination surface not the correct width or height.");
    }

    Py_BEGIN_ALLOW_THREADS;
    ret = v4l2_read_frame(self, surf, &errno_code);
    Py_END_ALLOW_THREADS;
    if (!ret) {
        /* error occurred */
        if (errno_code) {
            PyErr_Format(PyExc_SystemError,
                         "ioctl(VIDIOC_DQBUF) failure : %d, %s", errno_code,
                         strerror(errno_code));
            return NULL;
        }
        return RAISE(PyExc_SystemError, "image processing error");
    }

    if (surfobj) {
        Py_INCREF(surfobj);
        return (PyObject *)surfobj;
    }
    else {
        return (PyObject *)pgSurface_New(surf);
    }
#elif defined(PYGAME_WINDOWS_CAMERA)
    SDL_Surface *surf = NULL;
    pgSurfaceObject *surfobj = NULL;

    int width = self->width;
    int height = self->height;

    if (!PyArg_ParseTuple(arg, "|O!", &pgSurface_Type, &surfobj))
        return NULL;

    if (!surfobj) {
        surf = SDL_CreateRGBSurface(0, width, height, 32,  // 24?
                                    0xFF << 16, 0xFF << 8, 0xFF, 0);
    }
    else {
        surf = pgSurface_AsSurface(surfobj);
    }

    if (!surf)
        return NULL;

    if (surf->w != self->width || surf->h != self->height) {
        return RAISE(PyExc_ValueError,
                     "Destination surface not the correct width or height.");
    }

    if (!windows_read_frame(self, surf))
        return NULL;

    if (surfobj) {
        Py_INCREF(surfobj);
        return (PyObject *)surfobj;
    }
    else {
        return (PyObject *)pgSurface_New(surf);
    }

#endif
    Py_RETURN_NONE;
}

/* get_raw() - returns an unmodified image as a string from the buffer */
PyObject *
camera_get_raw(pgCameraObject *self, PyObject *_null)
{
#if defined(__unix__)
    return v4l2_read_raw(self);
#elif defined(PYGAME_WINDOWS_CAMERA)
    return windows_read_raw(self);
#endif
    Py_RETURN_NONE;
}

/*
 * Pixelformat conversion functions
 */

/* converts from rgb Surface to yuv or hsv */
/* TODO: Allow for conversion from yuv and hsv to all */
void
colorspace(SDL_Surface *src, SDL_Surface *dst, int cspace)
{
    switch (cspace) {
        case YUV_OUT:
            rgb_to_yuv(src->pixels, dst->pixels, src->h * src->w, 0,
                       src->format);
            break;
        case HSV_OUT:
            rgb_to_hsv(src->pixels, dst->pixels, src->h * src->w, 0,
                       src->format);
            break;
    }
}

/* converts pretty directly if its already RGB24 */
void
rgb24_to_rgb(const void *src, void *dst, int length, SDL_PixelFormat *format)
{
    Uint8 *s = (Uint8 *)src;
    Uint8 *d8;
    Uint16 *d16;
    Uint32 *d32;
    Uint8 r, g, b;
    int rshift, gshift, bshift, rloss, gloss, bloss;

    rshift = format->Rshift;
    gshift = format->Gshift;
    bshift = format->Bshift;
    rloss = format->Rloss;
    gloss = format->Gloss;
    bloss = format->Bloss;

    switch (format->BytesPerPixel) {
        case 1:
            d8 = (Uint8 *)dst;
            while (length--) {
                r = *s++;
                g = *s++;
                b = *s++;
                *d8++ = ((r >> rloss) << rshift) | ((g >> gloss) << gshift) |
                        ((b >> bloss) << bshift);
            }
            break;
        case 2:
            d16 = (Uint16 *)dst;
            while (length--) {
                r = *s++;
                g = *s++;
                b = *s++;
                *d16++ = ((r >> rloss) << rshift) | ((g >> gloss) << gshift) |
                         ((b >> bloss) << bshift);
            }
            break;
        case 3:
            d8 = (Uint8 *)dst;
            while (length--) {
                *d8++ = *(s + 2); /* blue */
                *d8++ = *(s + 1); /* green */
                *d8++ = *s;       /* red */
                s += 3;
            }
            break;
        default:
            d32 = (Uint32 *)dst;
            while (length--) {
                r = *s++;
                g = *s++;
                b = *s++;
                *d32++ = ((r >> rloss) << rshift) | ((g >> gloss) << gshift) |
                         ((b >> bloss) << bshift);
            }
            break;
    }
}

/* slight variation on rgb24_to_rgb, just drops the 4th byte of each pixel,
 * changes the R, G, B ordering. */
void
bgr32_to_rgb(const void *src, void *dst, int length, SDL_PixelFormat *format)
{
    Uint8 *s = (Uint8 *)src;
    Uint8 *d8;
    Uint16 *d16;
    Uint32 *d32;
    Uint8 r, g, b;
    int rshift, gshift, bshift, rloss, gloss, bloss;

    rshift = format->Rshift;
    gshift = format->Gshift;
    bshift = format->Bshift;
    rloss = format->Rloss;
    gloss = format->Gloss;
    bloss = format->Bloss;

    switch (format->BytesPerPixel) {
        case 1:
            d8 = (Uint8 *)dst;
            while (length--) {
                b = *s++;
                g = *s++;
                r = *s++;
                s++;
                *d8++ = ((r >> rloss) << rshift) | ((g >> gloss) << gshift) |
                        ((b >> bloss) << bshift);
            }
            break;
        case 2:
            d16 = (Uint16 *)dst;
            while (length--) {
                b = *s++;
                g = *s++;
                r = *s++;
                s++;
                *d16++ = ((r >> rloss) << rshift) | ((g >> gloss) << gshift) |
                         ((b >> bloss) << bshift);
            }
            break;
        case 3:
            d8 = (Uint8 *)dst;
            while (length--) {
                *d8++ = *s;       /* red */
                *d8++ = *(s + 1); /* green */
                *d8++ = *(s + 2); /* blue */
                s += 4;
            }
            break;
        default:
            d32 = (Uint32 *)dst;
            while (length--) {
                b = *s++;
                g = *s++;
                r = *s++;
                s++;
                *d32++ = ((r >> rloss) << rshift) | ((g >> gloss) << gshift) |
                         ((b >> bloss) << bshift);
            }
            break;
    }
}

/* converts packed rgb to packed hsv. formulas modified from wikipedia */
void
rgb_to_hsv(const void *src, void *dst, int length, unsigned long source,
           SDL_PixelFormat *format)
{
    Uint8 *s8, *d8;
    Uint16 *s16, *d16;
    Uint32 *s32, *d32;
    Uint8 r, g, b, p1, p2, h, s, v, max, min, delta;
    int rshift, gshift, bshift, rloss, gloss, bloss;

    s8 = (Uint8 *)src;
    s16 = (Uint16 *)src;
    s32 = (Uint32 *)src;
    d8 = (Uint8 *)dst;
    d16 = (Uint16 *)dst;
    d32 = (Uint32 *)dst;
    rshift = format->Rshift;
    gshift = format->Gshift;
    bshift = format->Bshift;
    rloss = format->Rloss;
    gloss = format->Gloss;
    bloss = format->Bloss;

    /* you could stick the if statement inside the loop, but I'm sacrificing a
       a few hundred bytes for a little performance */
    /* 10 yrs later... about that... */
    if (source == V4L2_PIX_FMT_RGB444 || source == V4L2_PIX_FMT_RGB24 ||
        source == V4L2_PIX_FMT_XBGR32) {
        while (length--) {
            if (source == V4L2_PIX_FMT_RGB444) {
                p1 = *s8++;
                p2 = *s8++;
                b = p2 << 4;
                g = p1 & 0xF0;
                r = p1 << 4;
            }
            else if (source == V4L2_PIX_FMT_RGB24) {
                r = *s8++;
                g = *s8++;
                b = *s8++;
            }
            else {
                b = *s8++;
                g = *s8++;
                r = *s8++;
                s8++;
            }
            max = MAX(MAX(r, g), b);
            min = MIN(MIN(r, g), b);
            delta = max - min;
            v = max;      /* value (similar to luminosity) */
            if (!delta) { /* grey, zero hue and saturation */
                s = 0;
                h = 0;
            }
            else {
                s = 255 * delta / max; /* saturation */
                if (r == max) {        /* set hue based on max color */
                    h = 43 * (g - b) / delta;
                }
                else if (g == max) {
                    h = 85 + 43 * (b - r) / delta;
                }
                else {
                    h = 170 + 43 * (r - g) / delta;
                }
            }
            switch (format->BytesPerPixel) {
                case 1:
                    *d8++ = ((h >> rloss) << rshift) |
                            ((s >> gloss) << gshift) |
                            ((v >> bloss) << bshift);
                    break;
                case 2:
                    *d16++ = ((h >> rloss) << rshift) |
                             ((s >> gloss) << gshift) |
                             ((v >> bloss) << bshift);
                    break;
                case 3:
                    *d8++ = v;
                    *d8++ = s;
                    *d8++ = h;
                    break;
                default:
                    *d32++ = ((h >> rloss) << rshift) |
                             ((s >> gloss) << gshift) |
                             ((v >> bloss) << bshift);
                    break;
            }
        }
    }
    else { /* for use as stage 2 in yuv or bayer to hsv, r and b switched */
        while (length--) {
            switch (format->BytesPerPixel) {
                case 1:
                    r = *s8 >> rshift << rloss;
                    g = *s8 >> gshift << gloss;
                    b = *s8++ >> bshift << bloss;
                    break;
                case 2:
                    r = *s16 >> rshift << rloss;
                    g = *s16 >> gshift << gloss;
                    b = *s16++ >> bshift << bloss;
                    break;
                case 3:
                    b = *s8++;
                    g = *s8++;
                    r = *s8++;
                    break;
                default:
                    r = *s32 >> rshift << rloss;
                    g = *s32 >> gshift << gloss;
                    b = *s32++ >> bshift << bloss;
                    break;
            }
            max = MAX(MAX(r, g), b);
            min = MIN(MIN(r, g), b);
            delta = max - min;
            v = max;      /* value (similar to luminosity) */
            if (!delta) { /* grey, zero hue and saturation */
                s = 0;
                h = 0;
            }
            else {
                s = 255 * delta / max; /* saturation */
                if (r == max) {        /* set hue based on max color */
                    h = 43 * (g - b) / delta;
                }
                else if (g == max) {
                    h = 85 + 43 * (b - r) / delta;
                }
                else {
                    h = 170 + 43 * (r - g) / delta;
                }
            }
            switch (format->BytesPerPixel) {
                case 1:
                    *d8++ = ((h >> rloss) << rshift) |
                            ((s >> gloss) << gshift) |
                            ((v >> bloss) << bshift);
                    break;
                case 2:
                    *d16++ = ((h >> rloss) << rshift) +
                             ((s >> gloss) << gshift) +
                             ((v >> bloss) << bshift);
                    break;
                case 3:
                    *d8++ = v;
                    *d8++ = s;
                    *d8++ = h;
                    break;
                default:
                    *d32++ = ((h >> rloss) << rshift) |
                             ((s >> gloss) << gshift) |
                             ((v >> bloss) << bshift);
                    break;
            }
        }
    }
}

/* convert packed rgb to yuv. Note that unlike many implementations of YUV,
   this has a full range of 0-255 for Y, not 16-235. Formulas from wikipedia */
void
rgb_to_yuv(const void *src, void *dst, int length, unsigned long source,
           SDL_PixelFormat *format)
{
    Uint8 *s8, *d8;
    Uint16 *s16, *d16;
    Uint32 *s32, *d32;
    Uint8 r, g, b, y, u, v;
    Uint8 p1, p2;
    int rshift, gshift, bshift, rloss, gloss, bloss;

    s8 = (Uint8 *)src;
    s16 = (Uint16 *)src;
    s32 = (Uint32 *)src;
    d8 = (Uint8 *)dst;
    d16 = (Uint16 *)dst;
    d32 = (Uint32 *)dst;
    rshift = format->Rshift;
    gshift = format->Gshift;
    bshift = format->Bshift;
    rloss = format->Rloss;
    gloss = format->Gloss;
    bloss = format->Bloss;

    if (source == V4L2_PIX_FMT_RGB444 || source == V4L2_PIX_FMT_RGB24 ||
        source == V4L2_PIX_FMT_XBGR32) {
        while (length--) {
            if (source == V4L2_PIX_FMT_RGB444) {
                p1 = *s8++;
                p2 = *s8++;
                b = p2 << 4;
                g = p1 & 0xF0;
                r = p1 << 4;
            }
            else if (source == V4L2_PIX_FMT_RGB24) {
                r = *s8++;
                g = *s8++;
                b = *s8++;
            }
            else {
                b = *s8++;
                g = *s8++;
                r = *s8++;
                s8++;
            }

            v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;  /* V */
            u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128; /* U */
            y = (77 * r + 150 * g + 29 * b + 128) >> 8;          /* Y */
            switch (format->BytesPerPixel) {
                case 1:
                    *d8++ = ((y >> rloss) << rshift) |
                            ((u >> gloss) << gshift) |
                            ((v >> bloss) << bshift);
                    break;
                case 2:
                    *d16++ = ((y >> rloss) << rshift) |
                             ((u >> gloss) << gshift) |
                             ((v >> bloss) << bshift);
                    break;
                case 3:
                    *d8++ = v;
                    *d8++ = u;
                    *d8++ = y;
                    break;
                default:
                    *d32++ = ((y >> rloss) << rshift) |
                             ((u >> gloss) << gshift) |
                             ((v >> bloss) << bshift);
                    break;
            }
        }
    }
    else { /* for use as stage 2 in bayer to yuv, r and b switched */
        switch (format->BytesPerPixel) {
            case 1:
                while (length--) {
                    r = *s8 >> rshift << rloss;
                    g = *s8 >> gshift << gloss;
                    b = *s8++ >> bshift << bloss;
                    *d8++ =
                        ((((77 * r + 150 * g + 29 * b + 128) >> 8) >> rloss)
                         << rshift) |
                        (((((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128) >>
                          gloss)
                         << gshift) |
                        (((((112 * r - 94 * g - 18 * b + 128) >> 8) + 128) >>
                          bloss)
                         << bshift);
                }
                break;
            case 2:
                while (length--) {
                    r = *s16 >> rshift << rloss;
                    g = *s16 >> gshift << gloss;
                    b = *s16++ >> bshift << bloss;
                    *d16++ =
                        ((((77 * r + 150 * g + 29 * b + 128) >> 8) >> rloss)
                         << rshift) |
                        (((((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128) >>
                          gloss)
                         << gshift) |
                        (((((112 * r - 94 * g - 18 * b + 128) >> 8) + 128) >>
                          bloss)
                         << bshift);
                }
                break;
            case 3:
                while (length--) {
                    b = *s8++;
                    g = *s8++;
                    r = *s8++;
                    *d8++ = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
                    *d8++ = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
                    *d8++ = (77 * r + 150 * g + 29 * b + 128) >> 8;
                }
                break;
            default:
                while (length--) {
                    r = *s32 >> rshift << rloss;
                    g = *s32 >> gshift << gloss;
                    b = *s32++ >> bshift << bloss;
                    *d32++ =
                        ((((77 * r + 150 * g + 29 * b + 128) >> 8) >> rloss)
                         << rshift) |
                        (((((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128) >>
                          gloss)
                         << gshift) |
                        (((((112 * r - 94 * g - 18 * b + 128) >> 8) + 128) >>
                          bloss)
                         << bshift);
                }
                break;
        }
    }
}

/* Converts from rgb444 (R444) to rgb24 (RGB3) */
void
rgb444_to_rgb(const void *src, void *dst, int length, SDL_PixelFormat *format)
{
    Uint8 *s, *d8;
    Uint16 *d16;
    Uint32 *d32;
    Uint8 p1, p2, r, g, b;
    int rshift, gshift, bshift, rloss, gloss, bloss;

    s = (Uint8 *)src;
    rshift = format->Rshift;
    gshift = format->Gshift;
    bshift = format->Bshift;
    rloss = format->Rloss;
    gloss = format->Gloss;
    bloss = format->Bloss;

    switch (format->BytesPerPixel) {
        case 1:
            d8 = (Uint8 *)dst;
            while (length--) {
                r = *s << 4;
                g = *s++ & 0xF0;
                b = *s++ << 4;
                *d8++ = ((r >> rloss) << rshift) | ((g >> gloss) << gshift) |
                        ((b >> bloss) << bshift);
            }
            break;
        case 2:
            d16 = (Uint16 *)dst;
            while (length--) {
                r = *s << 4;
                g = *s++ & 0xF0;
                b = *s++ << 4;
                *d16++ = ((r >> rloss) << rshift) | ((g >> gloss) << gshift) |
                         ((b >> bloss) << bshift);
            }
            break;
        case 3:
            d8 = (Uint8 *)dst;
            while (length--) {
                p1 = *s++;
                p2 = *s++;
                *d8++ = p2 << 4;   /* blue */
                *d8++ = p1 & 0xF0; /* green */
                *d8++ = p1 << 4;   /* red */
            }
            break;
        default:
            d32 = (Uint32 *)dst;
            while (length--) {
                r = *s << 4;
                g = *s++ & 0xF0;
                b = *s++ << 4;
                *d32++ = ((r >> rloss) << rshift) | ((g >> gloss) << gshift) |
                         ((b >> bloss) << bshift);
            }
            break;
    }
}

/* convert from 4:2:2 YUYV interlaced to RGB */
/* colorspace conversion routine from libv4l. Licensed LGPL 2.1
   (C) 2008 Hans de Goede <j.w.r.degoede@hhs.nl> */
void
yuyv_to_rgb(const void *src, void *dst, int length, SDL_PixelFormat *format)
{
    Uint8 *s, *d8;
    Uint16 *d16;
    Uint32 *d32;
    int i;
    int r1, g1, b1, r2, b2, g2;
    int rshift, gshift, bshift, rloss, gloss, bloss, y1, y2, u, v, u1, rg, v1;

    rshift = format->Rshift;
    gshift = format->Gshift;
    bshift = format->Bshift;
    rloss = format->Rloss;
    gloss = format->Gloss;
    bloss = format->Bloss;

    d8 = (Uint8 *)dst;
    d16 = (Uint16 *)dst;
    d32 = (Uint32 *)dst;
    i = length >> 1;
    s = (Uint8 *)src;

    /* yuyv packs 2 pixels into every 4 bytes, sharing the u and v color
       terms between the 2, with each pixel having a unique y luminance term.
       Thus, we will operate on 2 pixels at a time. */
    while (i--) {
        y1 = *s++;
        u = *s++;
        y2 = *s++;
        v = *s++;
        /* The lines from here to the switch statement are from libv4l */
        u1 = (((u - 128) << 7) + (u - 128)) >> 6;
        rg = (((u - 128) << 1) + (u - 128) + ((v - 128) << 2) +
              ((v - 128) << 1)) >>
             3;
        v1 = (((v - 128) << 1) + (v - 128)) >> 1;

        r1 = SAT2(y1 + v1);
        g1 = SAT2(y1 - rg);
        b1 = SAT2(y1 + u1);

        r2 = SAT2(y2 + v1);
        g2 = SAT2(y2 - rg);
        b2 = SAT2(y2 + u1);

        /* choose the right pixel packing for the destination surface depth */
        switch (format->BytesPerPixel) {
            case 1:
                *d8++ = ((r1 >> rloss) << rshift) | ((g1 >> gloss) << gshift) |
                        ((b1 >> bloss) << bshift);
                *d8++ = ((r2 >> rloss) << rshift) | ((g2 >> gloss) << gshift) |
                        ((b2 >> bloss) << bshift);
                break;
            case 2:
                *d16++ = ((r1 >> rloss) << rshift) |
                         ((g1 >> gloss) << gshift) | ((b1 >> bloss) << bshift);
                *d16++ = ((r2 >> rloss) << rshift) |
                         ((g2 >> gloss) << gshift) | ((b2 >> bloss) << bshift);
                break;
            case 3:
                *d8++ = b1;
                *d8++ = g1;
                *d8++ = r1;
                *d8++ = b2;
                *d8++ = g2;
                *d8++ = r2;
                break;
            default:
                *d32++ = ((r1 >> rloss) << rshift) |
                         ((g1 >> gloss) << gshift) | ((b1 >> bloss) << bshift);
                *d32++ = ((r2 >> rloss) << rshift) |
                         ((g2 >> gloss) << gshift) | ((b2 >> bloss) << bshift);
                break;
        }
    }
}

/* turn yuyv into packed yuv. */
void
yuyv_to_yuv(const void *src, void *dst, int length, SDL_PixelFormat *format)
{
    Uint8 *s, *d8;
    Uint8 y1, u, y2, v;
    Uint16 *d16;
    Uint32 *d32;
    int i = length >> 1;
    int rshift, gshift, bshift, rloss, gloss, bloss;

    rshift = format->Rshift;
    gshift = format->Gshift;
    bshift = format->Bshift;
    rloss = format->Rloss;
    gloss = format->Gloss;
    bloss = format->Bloss;
    s = (Uint8 *)src;

    switch (format->BytesPerPixel) {
        case 1:
            d8 = (Uint8 *)dst;
            while (i--) {
                y1 = *s++;
                u = *s++;
                y2 = *s++;
                v = *s++;
                *d8++ = ((y1 >> rloss) << rshift) | ((u >> gloss) << gshift) |
                        ((v >> bloss) << bshift);
                *d8++ = ((y2 >> rloss) << rshift) | ((u >> gloss) << gshift) |
                        ((v >> bloss) << bshift);
            }
            break;
        case 2:
            d16 = (Uint16 *)dst;
            while (i--) {
                y1 = *s++;
                u = *s++;
                y2 = *s++;
                v = *s++;
                *d16++ = ((y1 >> rloss) << rshift) | ((u >> gloss) << gshift) |
                         ((v >> bloss) << bshift);
                *d16++ = ((y2 >> rloss) << rshift) | ((u >> gloss) << gshift) |
                         ((v >> bloss) << bshift);
            }
            break;
        case 3:
            d8 = (Uint8 *)dst;
            while (i--) {
                *d8++ = *(s + 3); /* v */
                *d8++ = *(s + 1); /* u */
                *d8++ = *s;       /* y1 */
                *d8++ = *(s + 3); /* v */
                *d8++ = *(s + 1); /* u */
                *d8++ = *(s + 2); /* y2 */
                s += 4;
            }
            break;
        default:
            d32 = (Uint32 *)dst;
            while (i--) {
                y1 = *s++;
                u = *s++;
                y2 = *s++;
                v = *s++;
                *d32++ = ((y1 >> rloss) << rshift) | ((u >> gloss) << gshift) |
                         ((v >> bloss) << bshift);
                *d32++ = ((y2 >> rloss) << rshift) | ((u >> gloss) << gshift) |
                         ((v >> bloss) << bshift);
            }
            break;
    }
}

/* cribbed from above, but modified for uyvy ordering */
void
uyvy_to_rgb(const void *src, void *dst, int length, SDL_PixelFormat *format)
{
    Uint8 *s, *d8;
    Uint16 *d16;
    Uint32 *d32;
    int i;
    int r1, g1, b1, r2, b2, g2;
    int rshift, gshift, bshift, rloss, gloss, bloss, y1, y2, u, v, u1, rg, v1;

    rshift = format->Rshift;
    gshift = format->Gshift;
    bshift = format->Bshift;
    rloss = format->Rloss;
    gloss = format->Gloss;
    bloss = format->Bloss;

    d8 = (Uint8 *)dst;
    d16 = (Uint16 *)dst;
    d32 = (Uint32 *)dst;
    i = length >> 1;
    s = (Uint8 *)src;

    /* yuyv packs 2 pixels into every 4 bytes, sharing the u and v color
       terms between the 2, with each pixel having a unique y luminance term.
       Thus, we will operate on 2 pixels at a time. */
    while (i--) {
        u = *s++;
        y1 = *s++;
        v = *s++;
        y2 = *s++;

        /* The lines from here to the switch statement are from libv4l */
        u1 = (((u - 128) << 7) + (u - 128)) >> 6;
        rg = (((u - 128) << 1) + (u - 128) + ((v - 128) << 2) +
              ((v - 128) << 1)) >>
             3;
        v1 = (((v - 128) << 1) + (v - 128)) >> 1;

        r1 = SAT2(y1 + v1);
        g1 = SAT2(y1 - rg);
        b1 = SAT2(y1 + u1);

        r2 = SAT2(y2 + v1);
        g2 = SAT2(y2 - rg);
        b2 = SAT2(y2 + u1);

        /* choose the right pixel packing for the destination surface depth */
        switch (format->BytesPerPixel) {
            case 1:
                *d8++ = ((r1 >> rloss) << rshift) | ((g1 >> gloss) << gshift) |
                        ((b1 >> bloss) << bshift);
                *d8++ = ((r2 >> rloss) << rshift) | ((g2 >> gloss) << gshift) |
                        ((b2 >> bloss) << bshift);
                break;
            case 2:
                *d16++ = ((r1 >> rloss) << rshift) |
                         ((g1 >> gloss) << gshift) | ((b1 >> bloss) << bshift);
                *d16++ = ((r2 >> rloss) << rshift) |
                         ((g2 >> gloss) << gshift) | ((b2 >> bloss) << bshift);
                break;
            case 3:
                *d8++ = b1;
                *d8++ = g1;
                *d8++ = r1;
                *d8++ = b2;
                *d8++ = g2;
                *d8++ = r2;
                break;
            default:
                *d32++ = ((r1 >> rloss) << rshift) |
                         ((g1 >> gloss) << gshift) | ((b1 >> bloss) << bshift);
                *d32++ = ((r2 >> rloss) << rshift) |
                         ((g2 >> gloss) << gshift) | ((b2 >> bloss) << bshift);
                break;
        }
    }
}
/* turn uyvy into packed yuv. */
void
uyvy_to_yuv(const void *src, void *dst, int length, SDL_PixelFormat *format)
{
    Uint8 *s, *d8;
    Uint8 y1, u, y2, v;
    Uint16 *d16;
    Uint32 *d32;
    int i = length >> 1;
    int rshift, gshift, bshift, rloss, gloss, bloss;

    rshift = format->Rshift;
    gshift = format->Gshift;
    bshift = format->Bshift;
    rloss = format->Rloss;
    gloss = format->Gloss;
    bloss = format->Bloss;
    s = (Uint8 *)src;

    switch (format->BytesPerPixel) {
        case 1:
            d8 = (Uint8 *)dst;
            while (i--) {
                u = *s++;
                y1 = *s++;
                v = *s++;
                y2 = *s++;
                *d8++ = ((y1 >> rloss) << rshift) | ((u >> gloss) << gshift) |
                        ((v >> bloss) << bshift);
                *d8++ = ((y2 >> rloss) << rshift) | ((u >> gloss) << gshift) |
                        ((v >> bloss) << bshift);
            }
            break;
        case 2:
            d16 = (Uint16 *)dst;
            while (i--) {
                u = *s++;
                y1 = *s++;
                v = *s++;
                y2 = *s++;
                *d16++ = ((y1 >> rloss) << rshift) | ((u >> gloss) << gshift) |
                         ((v >> bloss) << bshift);
                *d16++ = ((y2 >> rloss) << rshift) | ((u >> gloss) << gshift) |
                         ((v >> bloss) << bshift);
            }
            break;
        case 3:
            d8 = (Uint8 *)dst;
            while (i--) {
                *d8++ = *(s + 2); /* v */
                *d8++ = *s;       /* u */
                *d8++ = *(s + 1); /* y1 */
                *d8++ = *(s + 2); /* v */
                *d8++ = *s;       /* u */
                *d8++ = *(s + 3); /* y2 */
                s += 4;
            }
            break;
        default:
            d32 = (Uint32 *)dst;
            while (i--) {
                y1 = *s++;
                u = *s++;
                y2 = *s++;
                v = *s++;
                *d32++ = ((y1 >> rloss) << rshift) | ((u >> gloss) << gshift) |
                         ((v >> bloss) << bshift);
                *d32++ = ((y2 >> rloss) << rshift) | ((u >> gloss) << gshift) |
                         ((v >> bloss) << bshift);
            }
            break;
    }
}

/* Converts from 8 bit Bayer (BA81) to rgb24 (RGB3), based on:
 * Sonix SN9C101 based webcam basic I/F routines
 * Copyright (C) 2004 Takafumi Mizuno <taka-qce@ls-a.jp>
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */
/* TODO: Certainly not the most efficient way of doing this conversion. */
void
sbggr8_to_rgb(const void *src, void *dst, int width, int height,
              SDL_PixelFormat *format)
{
    Uint8 *rawpt, *d8;
    Uint16 *d16;
    Uint32 *d32;
    Uint8 r, g, b;
    int rshift, gshift, bshift, rloss, gloss, bloss;
    int i = width * height;
    rawpt = (Uint8 *)src;
    rshift = format->Rshift;
    gshift = format->Gshift;
    bshift = format->Bshift;
    rloss = format->Rloss;
    gloss = format->Gloss;
    bloss = format->Bloss;

    d8 = (Uint8 *)dst;
    d16 = (Uint16 *)dst;
    d32 = (Uint32 *)dst;

    while (i--) {
        if ((i / width) % 2 == 0) {
            /* even row (BGBGBGBG)*/
            if ((i % 2) == 0) {
                /* B */
                if ((i > width) && ((i % width) > 0)) {
                    b = *rawpt; /* B */
                    g = (*(rawpt - 1) + *(rawpt + 1) + *(rawpt + width) +
                         *(rawpt - width)) /
                        4; /* G */
                    r = (*(rawpt - width - 1) + *(rawpt - width + 1) +
                         *(rawpt + width - 1) + *(rawpt + width + 1)) /
                        4; /* R */
                }
                else {
                    /* first line or left column */
                    b = *rawpt;                                /* B */
                    g = (*(rawpt + 1) + *(rawpt + width)) / 2; /* G */
                    r = *(rawpt + width + 1);                  /* R */
                }
            }
            else {
                /* (B)G */
                if ((i > width) && ((i % width) < (width - 1))) {
                    b = (*(rawpt - 1) + *(rawpt + 1)) / 2;         /* B */
                    g = *rawpt;                                    /* G */
                    r = (*(rawpt + width) + *(rawpt - width)) / 2; /* R */
                }
                else {
                    /* first line or right column */
                    b = *(rawpt - 1);     /* B */
                    g = *rawpt;           /* G */
                    r = *(rawpt + width); /* R */
                }
            }
        }
        else {
            /* odd row (GRGRGRGR) */
            if ((i % 2) == 0) {
                /* G(R) */
                if ((i < (width * (height - 1))) && ((i % width) > 0)) {
                    b = (*(rawpt + width) + *(rawpt - width)) / 2; /* B */
                    g = *rawpt;                                    /* G */
                    r = (*(rawpt - 1) + *(rawpt + 1)) / 2;         /* R */
                }
                else {
                    /* bottom line or left column */
                    b = *(rawpt - width); /* B */
                    g = *rawpt;           /* G */
                    r = *(rawpt + 1);     /* R */
                }
            }
            else {
                /* R */
                if (i < (width * (height - 1)) &&
                    ((i % width) < (width - 1))) {
                    b = (*(rawpt - width - 1) + *(rawpt - width + 1) +
                         *(rawpt + width - 1) + *(rawpt + width + 1)) /
                        4; /* B */
                    g = (*(rawpt - 1) + *(rawpt + 1) + *(rawpt - width) +
                         *(rawpt + width)) /
                        4;      /* G */
                    r = *rawpt; /* R */
                }
                else {
                    /* bottom line or right column */
                    b = *(rawpt - width - 1);                  /* B */
                    g = (*(rawpt - 1) + *(rawpt - width)) / 2; /* G */
                    r = *rawpt;                                /* R */
                }
            }
        }
        rawpt++;
        switch (format->BytesPerPixel) {
            case 1:
                *d8++ = ((r >> rloss) << rshift) | ((g >> gloss) << gshift) |
                        ((b >> bloss) << bshift);
                break;
            case 2:
                *d16++ = ((r >> rloss) << rshift) | ((g >> gloss) << gshift) |
                         ((b >> bloss) << bshift);
                break;
            case 3:
                *d8++ = b;
                *d8++ = g;
                *d8++ = r;
                break;
            default:
                *d32++ = ((r >> rloss) << rshift) | ((g >> gloss) << gshift) |
                         ((b >> bloss) << bshift);
                break;
        }
    }
}

/* convert from YUV 4:2:0 (YU12) to RGB24 */
/* based on v4lconvert_yuv420_to_rgb24 in libv4l (C) 2008 Hans de Goede. LGPL
 */
void
yuv420_to_rgb(const void *src, void *dst, int width, int height,
              SDL_PixelFormat *format)
{
    int rshift, gshift, bshift, rloss, gloss, bloss, i, j, u1, v1, rg, y;
    const Uint8 *y1, *y2, *u, *v;
    Uint8 *d8_1, *d8_2;
    Uint16 *d16_1, *d16_2;
    Uint32 *d32_1, *d32_2;

    rshift = format->Rshift;
    gshift = format->Gshift;
    bshift = format->Bshift;
    rloss = format->Rloss;
    gloss = format->Gloss;
    bloss = format->Bloss;

    /* see http://en.wikipedia.org/wiki/YUV for an explanation of YUV420 */
    y1 = (Uint8 *)src;
    y2 = y1 + width;
    u = y1 + width * height;
    v = u + (width * height) / 4;
    j = height / 2;
    /* prepare the destination pointers for different surface depths. */
    d8_1 = (Uint8 *)dst;
    /* the following is because d8 used for both 8 and 24 bit surfaces */
    d8_2 = d8_1 + (format->BytesPerPixel == 3 ? width * 3 : 3);
    d16_1 = (Uint16 *)dst;
    d16_2 = d16_1 + width;
    d32_1 = (Uint32 *)dst;
    d32_2 = d32_1 + width;

    /* for the sake of speed, the nested while loops are inside of the switch
      statement for the different surface bit depths */
    switch (format->BytesPerPixel) {
        case 1:
            while (j--) {
                i = width / 2;
                while (i--) {
                    /* These formulas are from libv4l */
                    u1 = (((*u - 128) << 7) + (*u - 128)) >> 6;
                    rg = (((*u - 128) << 1) + (*u - 128) + ((*v - 128) << 2) +
                          ((*v - 128) << 1)) >>
                         3;
                    v1 = (((*v - 128) << 1) + (*v - 128)) >> 1;
                    u++;
                    v++;
                    /* do the pixels on row 1 */
                    y = *y1++;
                    *d8_1++ = ((SAT2(y + v1) >> rloss) << rshift) |
                              ((SAT2(y - rg) >> gloss) << gshift) |
                              ((SAT2(y + u1) >> bloss) << bshift);
                    y = *y1++;
                    *d8_1++ = ((SAT2(y + v1) >> rloss) << rshift) |
                              ((SAT2(y - rg) >> gloss) << gshift) |
                              ((SAT2(y + u1) >> bloss) << bshift);
                    /* do the pixels on row 2 */
                    y = *y2++;
                    *d8_2++ = ((SAT2(y + v1) >> rloss) << rshift) |
                              ((SAT2(y - rg) >> gloss) << gshift) |
                              ((SAT2(y + u1) >> bloss) << bshift);
                    y = *y2++;
                    *d8_2++ = ((SAT2(y + v1) >> rloss) << rshift) |
                              ((SAT2(y - rg) >> gloss) << gshift) |
                              ((SAT2(y + u1) >> bloss) << bshift);
                }
                /* y2 is at the beginning of a new row, make it the new row 1
                 */
                y1 = y2;
                /* and make row 2 one row further */
                y2 += width;
                d8_1 = d8_2;
                d8_2 += width;
            }
            break;
        case 2:
            while (j--) {
                i = width / 2;
                while (i--) {
                    /* These formulas are from libv4l */
                    u1 = (((*u - 128) << 7) + (*u - 128)) >> 6;
                    rg = (((*u - 128) << 1) + (*u - 128) + ((*v - 128) << 2) +
                          ((*v - 128) << 1)) >>
                         3;
                    v1 = (((*v - 128) << 1) + (*v - 128)) >> 1;
                    u++;
                    v++;
                    /* do the pixels on row 1 */
                    y = *y1++;
                    *d16_1++ = ((SAT2(y + v1) >> rloss) << rshift) |
                               ((SAT2(y - rg) >> gloss) << gshift) |
                               ((SAT2(y + u1) >> bloss) << bshift);
                    y = *y1++;
                    *d16_1++ = ((SAT2(y + v1) >> rloss) << rshift) |
                               ((SAT2(y - rg) >> gloss) << gshift) |
                               ((SAT2(y + u1) >> bloss) << bshift);
                    /* do the pixels on row 2 */
                    y = *y2++;
                    *d16_2++ = ((SAT2(y + v1) >> rloss) << rshift) |
                               ((SAT2(y - rg) >> gloss) << gshift) |
                               ((SAT2(y + u1) >> bloss) << bshift);
                    y = *y2++;
                    *d16_2++ = ((SAT2(y + v1) >> rloss) << rshift) |
                               ((SAT2(y - rg) >> gloss) << gshift) |
                               ((SAT2(y + u1) >> bloss) << bshift);
                }
                /* y2 is at the beginning of a new row, make it the new row 1
                 */
                y1 = y2;
                /* and make row 2 one row further */
                y2 += width;
                d16_1 = d16_2;
                d16_2 += width;
            }
            break;
        case 3:
            while (j--) {
                i = width / 2;
                while (i--) {
                    /* These formulas are from libv4l */
                    u1 = (((*u - 128) << 7) + (*u - 128)) >> 6;
                    rg = (((*u - 128) << 1) + (*u - 128) + ((*v - 128) << 2) +
                          ((*v - 128) << 1)) >>
                         3;
                    v1 = (((*v - 128) << 1) + (*v - 128)) >> 1;
                    u++;
                    v++;
                    /* do the pixels on row 1 */
                    y = *y1++;
                    *d8_1++ = SAT2(y + u1);
                    *d8_1++ = SAT2(y - rg);
                    *d8_1++ = SAT2(y + v1);
                    y = *y1++;
                    *d8_1++ = SAT2(y + u1);
                    *d8_1++ = SAT2(y - rg);
                    *d8_1++ = SAT2(y + v1);
                    /* do the pixels on row 2 */
                    y = *y2++;
                    *d8_2++ = SAT2(y + u1);
                    *d8_2++ = SAT2(y - rg);
                    *d8_2++ = SAT2(y + v1);
                    y = *y2++;
                    *d8_2++ = SAT2(y + u1);
                    *d8_2++ = SAT2(y - rg);
                    *d8_2++ = SAT2(y + v1);
                }
                /* y2 is at the beginning of a new row, make it the new row 1
                 */
                y1 = y2;
                /* and make row 2 one row further */
                y2 += width;
                d8_1 = d8_2;
                /* since it is 3 bytes per pixel */
                d8_2 += width * 3;
            }
            break;
        default:
            while (j--) {
                i = width / 2;
                while (i--) {
                    /* These formulas are from libv4l */
                    u1 = (((*u - 128) << 7) + (*u - 128)) >> 6;
                    rg = (((*u - 128) << 1) + (*u - 128) + ((*v - 128) << 2) +
                          ((*v - 128) << 1)) >>
                         3;
                    v1 = (((*v - 128) << 1) + (*v - 128)) >> 1;
                    u++;
                    v++;
                    /* do the pixels on row 1 */
                    y = *y1++;
                    *d32_1++ = ((SAT2(y + v1) >> rloss) << rshift) |
                               ((SAT2(y - rg) >> gloss) << gshift) |
                               ((SAT2(y + u1) >> bloss) << bshift);
                    y = *y1++;
                    *d32_1++ = ((SAT2(y + v1) >> rloss) << rshift) |
                               ((SAT2(y - rg) >> gloss) << gshift) |
                               ((SAT2(y + u1) >> bloss) << bshift);
                    /* do the pixels on row 2 */
                    y = *y2++;
                    *d32_2++ = ((SAT2(y + v1) >> rloss) << rshift) |
                               ((SAT2(y - rg) >> gloss) << gshift) |
                               ((SAT2(y + u1) >> bloss) << bshift);
                    y = *y2++;
                    *d32_2++ = ((SAT2(y + v1) >> rloss) << rshift) |
                               ((SAT2(y - rg) >> gloss) << gshift) |
                               ((SAT2(y + u1) >> bloss) << bshift);
                }
                /* y2 is at the beginning of a new row, make it the new row 1
                 */
                y1 = y2;
                /* and make row 2 one row further */
                y2 += width;
                d32_1 = d32_2;
                d32_2 += width;
            }
            break;
    }
}

/* turn yuv420 into packed yuv. */
void
yuv420_to_yuv(const void *src, void *dst, int width, int height,
              SDL_PixelFormat *format)
{
    const Uint8 *y1, *y2, *u, *v;
    Uint8 *d8_1, *d8_2;
    Uint16 *d16_1, *d16_2;
    Uint32 *d32_1, *d32_2;
    int rshift, gshift, bshift, rloss, gloss, bloss, j, i;

    rshift = format->Rshift;
    gshift = format->Gshift;
    bshift = format->Bshift;
    rloss = format->Rloss;
    gloss = format->Gloss;
    bloss = format->Bloss;

    d8_1 = (Uint8 *)dst;
    d8_2 = d8_1 + (format->BytesPerPixel == 3 ? width * 3 : 3);
    d16_1 = (Uint16 *)dst;
    d16_2 = d16_1 + width;
    d32_1 = (Uint32 *)dst;
    d32_2 = d32_1 + width;
    y1 = (Uint8 *)src;
    y2 = y1 + width;
    u = y1 + width * height;
    v = u + (width * height) / 4;
    j = height / 2;

    switch (format->BytesPerPixel) {
        case 1:
            while (j--) {
                i = width / 2;
                while (i--) {
                    *d8_1++ = ((*y1++ >> rloss) << rshift) |
                              ((*u >> gloss) << gshift) |
                              ((*v >> bloss) << bshift);
                    *d8_1++ = ((*y1++ >> rloss) << rshift) |
                              ((*u >> gloss) << gshift) |
                              ((*v >> bloss) << bshift);
                    *d8_2++ = ((*y2++ >> rloss) << rshift) |
                              ((*u >> gloss) << gshift) |
                              ((*v >> bloss) << bshift);
                    *d8_2++ = ((*y2++ >> rloss) << rshift) |
                              ((*u++ >> gloss) << gshift) |
                              ((*v++ >> bloss) << bshift);
                }
                y1 = y2;
                y2 += width;
                d8_1 = d8_2;
                d8_2 += width;
            }
            break;
        case 2:
            while (j--) {
                i = width / 2;
                while (i--) {
                    *d16_1++ = ((*y1++ >> rloss) << rshift) |
                               ((*u >> gloss) << gshift) |
                               ((*v >> bloss) << bshift);
                    *d16_1++ = ((*y1++ >> rloss) << rshift) |
                               ((*u >> gloss) << gshift) |
                               ((*v >> bloss) << bshift);
                    *d16_2++ = ((*y2++ >> rloss) << rshift) |
                               ((*u >> gloss) << gshift) |
                               ((*v >> bloss) << bshift);
                    *d16_2++ = ((*y2++ >> rloss) << rshift) |
                               ((*u++ >> gloss) << gshift) |
                               ((*v++ >> bloss) << bshift);
                }
                y1 = y2;
                y2 += width;
                d16_1 = d16_2;
                d16_2 += width;
            }
            break;
        case 3:
            while (j--) {
                i = width / 2;
                while (i--) {
                    *d8_1++ = *v;
                    *d8_1++ = *u;
                    *d8_1++ = *y1++;
                    *d8_1++ = *v;
                    *d8_1++ = *u;
                    *d8_1++ = *y1++;
                    *d8_2++ = *v;
                    *d8_2++ = *u;
                    *d8_2++ = *y2++;
                    *d8_2++ = *v++;
                    *d8_2++ = *u++;
                    *d8_2++ = *y2++;
                }
                y1 = y2;
                y2 += width;
                d8_1 = d8_2;
                d8_2 += width * 3;
            }
            break;
        default:
            while (j--) {
                i = width / 2;
                while (i--) {
                    *d32_1++ = ((*y1++ >> rloss) << rshift) |
                               ((*u >> gloss) << gshift) |
                               ((*v >> bloss) << bshift);
                    *d32_1++ = ((*y1++ >> rloss) << rshift) |
                               ((*u >> gloss) << gshift) |
                               ((*v >> bloss) << bshift);
                    *d32_2++ = ((*y2++ >> rloss) << rshift) |
                               ((*u >> gloss) << gshift) |
                               ((*v >> bloss) << bshift);
                    *d32_2++ = ((*y2++ >> rloss) << rshift) |
                               ((*u++ >> gloss) << gshift) |
                               ((*v++ >> bloss) << bshift);
                }
                y1 = y2;
                y2 += width;
                d32_1 = d32_2;
                d32_2 += width;
            }
            break;
    }
}

/*
 * Python API stuff
 */

/* Camera class definition */
PyMethodDef cameraobj_builtins[] = {
    {"start", (PyCFunction)camera_start, METH_NOARGS, DOC_CAMERASTART},
    {"stop", (PyCFunction)camera_stop, METH_NOARGS, DOC_CAMERASTOP},
    {"get_controls", (PyCFunction)camera_get_controls, METH_NOARGS,
     DOC_CAMERAGETCONTROLS},
    {"set_controls", (PyCFunction)camera_set_controls,
     METH_VARARGS | METH_KEYWORDS, DOC_CAMERASETCONTROLS},
    {"get_size", (PyCFunction)camera_get_size, METH_NOARGS, DOC_CAMERAGETSIZE},
    {"query_image", (PyCFunction)camera_query_image, METH_NOARGS,
     DOC_CAMERAQUERYIMAGE},
    {"get_image", (PyCFunction)camera_get_image, METH_VARARGS,
     DOC_CAMERAGETIMAGE},
    {"get_raw", (PyCFunction)camera_get_raw, METH_NOARGS, DOC_CAMERAGETRAW},
    {NULL, NULL, 0, NULL}};

void
camera_dealloc(PyObject *self)
{
#if defined(PYGAME_WINDOWS_CAMERA)
    if (((pgCameraObject *)self)->open) {
        windows_close_device((pgCameraObject *)self);
    }
    windows_dealloc_device((pgCameraObject *)self);
#else
    free(((pgCameraObject *)self)->device_name);
#endif
    PyObject_Free(self);
}
/*
PyObject* camera_getattr(PyObject* self, char* attrname) {
    return Py_FindMethod(cameraobj_builtins, self, attrname);
}
*/

static int
camera_init(pgCameraObject *self, PyObject *arg, PyObject *kwargs)
{
#if defined(__unix__)
    int w, h;
    char *dev_name = NULL;
    char *color = NULL;

    w = DEFAULT_WIDTH;
    h = DEFAULT_HEIGHT;

    if (!PyArg_ParseTuple(arg, "s|(ii)s", &dev_name, &w, &h, &color)) {
        return -1;
    }

    self->device_name = (char *)malloc((strlen(dev_name) + 1) * sizeof(char));
    if (!self->device_name) {
        PyErr_NoMemory();
        return -1;
    }
    strcpy(self->device_name, dev_name);
    self->camera_type = 0;
    self->pixelformat = 0;
    if (color) {
        if (!strcmp(color, "YUV")) {
            self->color_out = YUV_OUT;
        }
        else if (!strcmp(color, "HSV")) {
            self->color_out = HSV_OUT;
        }
        else {
            self->color_out = RGB_OUT;
        }
    }
    else {
        self->color_out = RGB_OUT;
    }
    self->buffers = NULL;
    self->n_buffers = 0;
    self->width = w;
    self->height = h;
    self->size = 0;
    self->hflip = 0;
    self->vflip = 0;
    self->brightness = 0;
    self->fd = -1;

    return 0;
#elif defined(PYGAME_WINDOWS_CAMERA)
    PyObject *name_obj = NULL;
    WCHAR *dev_name = NULL;
    int w, h;
    char *color = NULL;
    IMFActivate *p = NULL;

    w = DEFAULT_WIDTH;
    h = DEFAULT_HEIGHT;

    if (!PyArg_ParseTuple(arg, "O|(ii)s", &name_obj, &w, &h, &color)) {
        return -1;
    }

    /* needs to be freed with PyMem_Free later */
    dev_name = PyUnicode_AsWideCharString(name_obj, NULL);

    p = windows_device_from_name(dev_name);

    if (!p) {
        PyErr_SetString(PyExc_ValueError,
                        "Couldn't find a camera with that name");
        return -1;
    }

    if (color) {
        if (!strcmp(color, "YUV")) {
            self->color_out = YUV_OUT;
        }
        else if (!strcmp(color, "HSV")) {
            self->color_out = HSV_OUT;
        }
        else {
            self->color_out = RGB_OUT;
        }
    }
    else {
        self->color_out = RGB_OUT;
    }

    self->device_name = dev_name;
    self->width = w;
    self->height = h;
    self->open = 0;
    self->hflip = 0;
    self->vflip = 0;
    self->last_vflip = 0;

    self->raw_buf = NULL;
    self->buf = NULL;
    self->t_handle = NULL;

    if (!windows_init_device(NULL)) {
        return -1;
    }

    return 0;
#else
    PyErr_SetString(PyExc_RuntimeError,
                    "_camera backend not available on your platform");
    return -1;
#endif
}

PyTypeObject pgCamera_Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pygame.camera.Camera",
    .tp_basicsize = sizeof(pgCameraObject),
    .tp_dealloc = camera_dealloc,
    .tp_doc = DOC_PYGAMECAMERACAMERA,
    .tp_methods = cameraobj_builtins,
    .tp_init = (initproc)camera_init,
    .tp_new = PyType_GenericNew,
};

/* Camera module definition */
PyMethodDef camera_builtins[] = {
    {"colorspace", surf_colorspace, METH_VARARGS, DOC_PYGAMECAMERACOLORSPACE},
    {"list_cameras", list_cameras, METH_NOARGS, DOC_PYGAMECAMERALISTCAMERAS},
    {NULL, NULL, 0, NULL}};

MODINIT_DEFINE(_camera)
{
    PyObject *module;
    /* imported needed apis; Do this first so if there is an error
     * the module is not loaded.
     */

    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "_camera",
                                         DOC_PYGAMECAMERA,
                                         -1,
                                         camera_builtins,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL};

    import_pygame_base();
    if (PyErr_Occurred()) {
        return NULL;
    }
    import_pygame_surface();
    if (PyErr_Occurred()) {
        return NULL;
    }

    /* type preparation */
    // PyType_Init(pgCamera_Type);
    pgCamera_Type.tp_new = PyType_GenericNew;
    if (PyType_Ready(&pgCamera_Type) < 0) {
        return NULL;
    }

    /* create the module */
    module = PyModule_Create(&_module);
    if (!module) {
        return NULL;
    }

    Py_INCREF(&pgCamera_Type);
    if (PyModule_AddObject(module, "CameraType", (PyObject *)&pgCamera_Type)) {
        Py_DECREF(&pgCamera_Type);
        Py_DECREF(module);
        return NULL;
    }
    Py_INCREF(&pgCamera_Type);
    if (PyModule_AddObject(module, "Camera", (PyObject *)&pgCamera_Type)) {
        Py_DECREF(&pgCamera_Type);
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
