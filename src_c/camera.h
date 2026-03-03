#ifndef CAMERA_H
#define CAMERA_H
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

#include "pygame.h"
#include "pgcompat.h"
#include "doc/camera_doc.h"

#if defined(__unix__)
#include <structmember.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <fcntl.h> /* low-level i/o */
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

/* on freebsd there is no asm/types */
#ifdef linux
#include <asm/types.h> /* for videodev2.h */
#endif

#include <linux/videodev2.h>
#endif

/* At the time of writing of this comment, _camera does not compile on windows
 * while using the MinGW compiler (due to missing API). So we do a _MSC_VER
 * check here to compile this only under the MSVC compiler */
#if defined(__WIN32__) && defined(_MSC_VER)
#define PYGAME_WINDOWS_CAMERA 1

#include <mfapi.h>
#include <mfobjects.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <combaseapi.h>
#include <mftransform.h>
#endif

#ifndef v4l2_fourcc
/* Four-character-code (FOURCC), taken from v4l source */
#define v4l2_fourcc(a, b, c, d)                               \
    ((Uint32)(a) | ((Uint32)(b) << 8) | ((Uint32)(c) << 16) | \
     ((Uint32)(d) << 24))
#endif

/* some constants used which are not defined on non-v4l machines. */
#ifndef V4L2_PIX_FMT_RGB24
#define V4L2_PIX_FMT_RGB24 v4l2_fourcc('R', 'G', 'B', '3')
#endif
#ifndef V4L2_PIX_FMT_RGB444
#define V4L2_PIX_FMT_RGB444 v4l2_fourcc('R', '4', '4', '4')
#endif
#ifndef V4L2_PIX_FMT_YUYV
#define V4L2_PIX_FMT_YUYV v4l2_fourcc('Y', 'U', 'Y', 'V')
#endif
#ifndef V4L2_PIX_FMT_XBGR32
#define V4L2_PIX_FMT_XBGR32 v4l2_fourcc('X', 'R', '2', '4')
#endif

#define CLEAR(x) memset(&(x), 0, sizeof(x))
#define SAT(c)        \
    if (c & (~255)) { \
        if (c < 0)    \
            c = 0;    \
        else          \
            c = 255;  \
    }
#define SAT2(c) ((c) & (~255) ? ((c) < 0 ? 0 : 255) : (c))
#define DEFAULT_WIDTH 640
#define DEFAULT_HEIGHT 480
#define RGB_OUT 1
#define YUV_OUT 2
#define HSV_OUT 4
#define CAM_V4L \
    1 /* deprecated. the incomplete support in pygame was removed */
#define CAM_V4L2 2

struct buffer {
    void *start;
    size_t length;
};

#if defined(__unix__)
typedef struct pgCameraObject {
    PyObject_HEAD char *device_name;
    int camera_type;
    unsigned long pixelformat;
    unsigned int color_out;
    struct buffer *buffers;
    unsigned int n_buffers;
    int width;
    int height;
    int size;
    int hflip;
    int vflip;
    int brightness;
    int fd;
} pgCameraObject;
#elif defined(PYGAME_WINDOWS_CAMERA)
typedef struct pgCameraObject {
    PyObject_HEAD WCHAR *device_name;
    IMFSourceReader *reader;
    IMFTransform *transform;
    IMFVideoProcessorControl *control;
    IMFMediaBuffer *buf;
    IMFMediaBuffer *raw_buf;
    int buffer_ready;
    short open; /* used to signal the update_function to exit */
    HANDLE t_handle;
    HRESULT t_error;
    int t_error_line;
    int width;
    int height;
    int hflip;
    int vflip;
    int last_vflip;
    int color_out;
    unsigned long pixelformat;
} pgCameraObject;

#else
/* generic definition.
 */

typedef struct pgCameraObject {
    PyObject_HEAD char *device_name;
    int camera_type;
    unsigned long pixelformat;
    unsigned int color_out;
    struct buffer *buffers;
    unsigned int n_buffers;
    int width;
    int height;
    int size;
    int hflip;
    int vflip;
    int brightness;
    int fd;
} pgCameraObject;
#endif

/* internal functions for colorspace conversion */
void
colorspace(SDL_Surface *src, PG_PixelFormat *src_fmt, SDL_Surface *dst,
           int cspace);
void
rgb24_to_rgb(const void *src, void *dst, int length, PG_PixelFormat *format);
void
bgr32_to_rgb(const void *src, void *dst, int length, PG_PixelFormat *format);
void
rgb444_to_rgb(const void *src, void *dst, int length, PG_PixelFormat *format);
void
rgb_to_yuv(const void *src, void *dst, int length, unsigned long source,
           PG_PixelFormat *format);
void
rgb_to_hsv(const void *src, void *dst, int length, unsigned long source,
           PG_PixelFormat *format);
void
yuyv_to_rgb(const void *src, void *dst, int length, PG_PixelFormat *format);
void
yuyv_to_yuv(const void *src, void *dst, int length, PG_PixelFormat *format);
void
uyvy_to_rgb(const void *src, void *dst, int length, PG_PixelFormat *format);
void
uyvy_to_yuv(const void *src, void *dst, int length, PG_PixelFormat *format);
void
sbggr8_to_rgb(const void *src, void *dst, int width, int height,
              PG_PixelFormat *format);
void
yuv420_to_rgb(const void *src, void *dst, int width, int height,
              PG_PixelFormat *format);
void
yuv420_to_yuv(const void *src, void *dst, int width, int height,
              PG_PixelFormat *format);

#if defined(__unix__)
/* internal functions specific to v4l2 */
char **
v4l2_list_cameras(int *num_devices);
int
v4l2_get_control(int fd, int id, int *value);
int
v4l2_set_control(int fd, int id, int value);
PyObject *
v4l2_read_raw(pgCameraObject *self);
int
v4l2_xioctl(int fd, int request, void *arg);
int
v4l2_process_image(pgCameraObject *self, const void *image, int buffer_size,
                   SDL_Surface *surf);
int
v4l2_query_buffer(pgCameraObject *self);
int
v4l2_read_frame(pgCameraObject *self, SDL_Surface *surf, int *errno_code);
int
v4l2_stop_capturing(pgCameraObject *self);
int
v4l2_start_capturing(pgCameraObject *self);
int
v4l2_uninit_device(pgCameraObject *self);
int
v4l2_init_mmap(pgCameraObject *self);
int
v4l2_init_device(pgCameraObject *self);
int
v4l2_close_device(pgCameraObject *self);
int
v4l2_open_device(pgCameraObject *self);

#elif defined(PYGAME_WINDOWS_CAMERA)
/* internal functions specific to WINDOWS */
WCHAR **
windows_list_cameras(int *num_devices);
int
windows_init_device(pgCameraObject *self);
int
windows_open_device(pgCameraObject *self);
IMFActivate *
windows_device_from_name(WCHAR *device_name);
int
windows_close_device(pgCameraObject *self);
int
windows_read_frame(pgCameraObject *self, SDL_Surface *surf);
int
windows_frame_ready(pgCameraObject *self, int *result);
PyObject *
windows_read_raw(pgCameraObject *self);
int
windows_process_image(pgCameraObject *self, BYTE *data, DWORD buffer_size,
                      SDL_Surface *surf);
void
windows_dealloc_device(pgCameraObject *self);
int
windows_init_device(pgCameraObject *self);

#endif

#endif /* !CAMERA_H */
