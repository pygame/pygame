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

#if defined(__unix__)

#include "_camera.h"
#include "pgcompat.h"

int v4l2_pixelformat (int fd, struct v4l2_format* fmt,
                             unsigned long pixelformat);

char** v4l2_list_cameras (int* num_devices)
{
    char** devices;
    char* device;
    int num, i, fd;

    num = *num_devices;

    devices = (char**) malloc(sizeof(char *)*65);

    device = (char*) malloc(sizeof(char)*13);
    strcpy(device,"/dev/video");
    fd = open(device, O_RDONLY);
    if (fd != -1) {
        devices[num] = device;
        num++;
        device = (char*) malloc(sizeof(char)*13);
    }
    close(fd);
    /* v4l2 cameras can be /dev/video and /dev/video0 to /dev/video63 */
    for (i = 0; i < 64; i++) {
        sprintf(device,"/dev/video%d",i);
        fd = open(device, O_RDONLY);
        if (fd != -1) {
            devices[num] = device;
            num++;
            device = (char*) malloc(sizeof(char)*13);
        }
        close(fd);
    }

    if (num == *num_devices) {
        free(device);
    } else {
        *num_devices = num;
    }

    return devices;
}

/* A wrapper around a VIDIOC_S_FMT ioctl to check for format compatibility */
int v4l2_pixelformat (int fd, struct v4l2_format* fmt,
                             unsigned long pixelformat)
{
    fmt->fmt.pix.pixelformat = pixelformat;

    if (-1 == v4l2_xioctl (fd, VIDIOC_S_FMT, fmt)) {
        return 0;
    }

    if (fmt->fmt.pix.pixelformat == pixelformat) {
        return 1;
    } else {
        return 0;
    }
}

/* gets the value of a specific camera control if available */
int v4l2_get_control (int fd, int id, int *value)
{
    struct v4l2_control control;
    CLEAR(control);

    control.id = id;

    if (-1 == v4l2_xioctl (fd, VIDIOC_G_CTRL, &control)) {
        return 0;
    }

    *value = control.value;
    return 1;
}

/* sets a control if supported. the camera may round the value */
int v4l2_set_control (int fd, int id, int value)
{
    struct v4l2_control control;
    CLEAR(control);

    control.id = id;
    control.value = value;

    if (-1 == v4l2_xioctl (fd, VIDIOC_S_CTRL, &control)) {
        return 0;
    }

    return 1;
}

/* returns a string of the buffer from the camera */
/* TODO: fold this into the regular read_frame. lots of duplicate code */
PyObject* v4l2_read_raw (pgCameraObject* self)
{
    struct v4l2_buffer buf;
    PyObject* raw;

    CLEAR (buf);

    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (-1 == v4l2_xioctl (self->fd, VIDIOC_DQBUF, &buf)) {
        PyErr_Format(PyExc_SystemError, "ioctl(VIDIOC_DQBUF) failure : %d, %s",
                     errno, strerror (errno));
        return NULL;
    }

    assert (buf.index < self->n_buffers);

    raw = Bytes_FromStringAndSize(self->buffers[buf.index].start,
                                  self->buffers[buf.index].length);

    if (-1 == v4l2_xioctl (self->fd, VIDIOC_QBUF, &buf)) {
        PyErr_Format(PyExc_SystemError, "ioctl(VIDIOC_QBUF) failure : %d, %s",
                     errno, strerror (errno));
        return NULL;
    }
    return raw;
}

/*
 * Functions for v4l2 cameras.
 * This code is based partly on pyvideograb by Laurent Pointal at
 * http://laurent.pointal.org/python/projets/pyvideograb
 * the v4l2 capture example at
 * http://www.linuxtv.org/download/video4linux/API/V4L2_API/spec/
 * and the HighGUI library in OpenCV.
 */

int v4l2_xioctl (int fd, int request, void *arg)
{
    int r;

    do r = ioctl (fd, request, arg);
    while (-1 == r && EINTR == errno);

    return r;
}

/* sends the image to the conversion function based on input format and
   desired output format.  Note that some of the less common conversions are
   currently two step processes. */
/* TODO: Write single step conversions where they may actually be useful */
int v4l2_process_image (pgCameraObject* self, const void *image,
                               unsigned int buffer_size, SDL_Surface* surf)
{

    if (!surf)
        return 0;

    SDL_LockSurface (surf);

    switch (self->pixelformat) {
        case V4L2_PIX_FMT_RGB24:
            if (buffer_size >= self->size * 3) {
                switch (self->color_out) {
                    case RGB_OUT:
                        rgb24_to_rgb(image, surf->pixels, self->size, surf->format);
                        break;
                    case HSV_OUT:
                        rgb_to_hsv(image, surf->pixels, self->size, V4L2_PIX_FMT_RGB24, surf->format);
                        break;
                    case YUV_OUT:
                        rgb_to_yuv(image, surf->pixels, self->size, V4L2_PIX_FMT_RGB24, surf->format);
                        break;
                }
            } else {
                SDL_UnlockSurface (surf);
                return 0;
            }
            break;
        case V4L2_PIX_FMT_RGB444:
            if (buffer_size >= self->size * 2) {
                switch (self->color_out) {
                    case RGB_OUT:
                        rgb444_to_rgb(image, surf->pixels, self->size, surf->format);
                        break;
                    case HSV_OUT:
                        rgb_to_hsv(image, surf->pixels, self->size, V4L2_PIX_FMT_RGB444, surf->format);
                        break;
                    case YUV_OUT:
                        rgb_to_yuv(image, surf->pixels, self->size, V4L2_PIX_FMT_RGB444, surf->format);
                        break;
                }
            } else {
                SDL_UnlockSurface (surf);
                return 0;
            }
            break;
        case V4L2_PIX_FMT_YUYV:
            if (buffer_size >= self->size * 2) {
                switch (self->color_out) {
                    case YUV_OUT:
                        yuyv_to_yuv(image, surf->pixels, self->size, surf->format);
                        break;
                    case RGB_OUT:
                        yuyv_to_rgb(image, surf->pixels, self->size, surf->format);
                        break;
                    case HSV_OUT:
                        yuyv_to_rgb(image, surf->pixels, self->size, surf->format);
                        rgb_to_hsv(surf->pixels, surf->pixels, self->size, V4L2_PIX_FMT_YUYV, surf->format);
                        break;
                }
            } else {
                SDL_UnlockSurface (surf);
                return 0;
            }
            break;
    case V4L2_PIX_FMT_UYVY:
            if (buffer_size >= self->size * 2) {
                switch (self->color_out) {
                    case YUV_OUT:
                        uyvy_to_yuv(image, surf->pixels, self->size, surf->format);
                        break;
                    case RGB_OUT:
                        uyvy_to_rgb(image, surf->pixels, self->size, surf->format);
                        break;
                    case HSV_OUT:
                        uyvy_to_rgb(image, surf->pixels, self->size, surf->format);
                        rgb_to_hsv(surf->pixels, surf->pixels, self->size, V4L2_PIX_FMT_YUYV, surf->format);
                        break;
                }
            } else {
                SDL_UnlockSurface (surf);
                return 0;
            }
            break;
        case V4L2_PIX_FMT_SBGGR8:
            if (buffer_size >= self->size) {
                switch (self->color_out) {
                    case RGB_OUT:
                        sbggr8_to_rgb(image, surf->pixels, self->width, self->height, surf->format);
                        break;
                    case HSV_OUT:
                        sbggr8_to_rgb(image, surf->pixels, self->width, self->height, surf->format);
                        rgb_to_hsv(surf->pixels, surf->pixels, self->size, V4L2_PIX_FMT_SBGGR8, surf->format);
                        break;
                    case YUV_OUT:
                        sbggr8_to_rgb(image, surf->pixels, self->width, self->height, surf->format);
                        rgb_to_yuv(surf->pixels, surf->pixels, self->size, V4L2_PIX_FMT_SBGGR8, surf->format);
                        break;
                }
            } else {
                SDL_UnlockSurface (surf);
                return 0;
            }
            break;
        case V4L2_PIX_FMT_YUV420:
            if (buffer_size >= (self->size * 3) / 2) {
                switch (self->color_out) {
                    case YUV_OUT:
                        yuv420_to_yuv(image, surf->pixels, self->width, self->height, surf->format);
                        break;
                    case RGB_OUT:
                        yuv420_to_rgb(image, surf->pixels, self->width, self->height, surf->format);
                        break;
                    case HSV_OUT:
                        yuv420_to_rgb(image, surf->pixels, self->width, self->height, surf->format);
                        rgb_to_hsv(surf->pixels, surf->pixels, self->size, V4L2_PIX_FMT_YUV420, surf->format);
                        break;
                }
            } else {
                SDL_UnlockSurface (surf);
                return 0;
            }
            break;
    }
    SDL_UnlockSurface (surf);
    return 1;
}

/* query each buffer to see if it contains a frame ready to take */
/* FIXME: There needs to be a better way to implement non-blocking frame
   grabbing than only doing a get_image if query_image returns true. Many
   cameras will always return false, and will only respond to blocking calls. */
int v4l2_query_buffer (pgCameraObject* self)
{
    int i;

    for (i = 0; i < self->n_buffers; ++i) {
        struct v4l2_buffer buf;

        CLEAR (buf);

        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        if (-1 == v4l2_xioctl (self->fd, VIDIOC_QUERYBUF, &buf)) {
            PyErr_Format(PyExc_MemoryError, "ioctl(VIDIOC_QUERYBUF) failure : %d, %s",
                errno, strerror (errno));
            return 0;
        }

        /*  is there a buffer on outgoing queue ready for us to take? */
        if (buf.flags & V4L2_BUF_FLAG_DONE)
            return 1;
    }

    /* no buffer ready to take */
    return 0;
}

int v4l2_read_frame (pgCameraObject* self, SDL_Surface* surf)
{
    struct v4l2_buffer buf;

    CLEAR (buf);

    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (-1 == v4l2_xioctl (self->fd, VIDIOC_DQBUF, &buf)) {
        PyErr_Format(PyExc_SystemError, "ioctl(VIDIOC_DQBUF) failure : %d, %s",
                     errno, strerror (errno));
        return 0;
    }

    assert (buf.index < self->n_buffers);

    if (!v4l2_process_image (self, self->buffers[buf.index].start,
                             self->buffers[buf.index].length, surf)) {
        PyErr_Format(PyExc_SystemError, "image processing error");
        return 0;
    }

    if (-1 == v4l2_xioctl (self->fd, VIDIOC_QBUF, &buf)) {
        PyErr_Format(PyExc_SystemError, "ioctl(VIDIOC_QBUF) failure : %d, %s",
                     errno, strerror (errno));
        return 0;
    }
    return 1;
}

int v4l2_stop_capturing (pgCameraObject* self)
{
    enum v4l2_buf_type type;

    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    if (-1 == v4l2_xioctl (self->fd, VIDIOC_STREAMOFF, &type)) {
        PyErr_Format(PyExc_SystemError, "ioctl(VIDIOC_STREAMOFF) failure : %d, %s",
                     errno, strerror (errno));
        return 0;
    }

    return 1;
}

int v4l2_start_capturing (pgCameraObject* self)
{
    unsigned int i;
    enum v4l2_buf_type type;

    for (i = 0; i < self->n_buffers; ++i) {
        struct v4l2_buffer buf;

        CLEAR (buf);

        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        if (-1 == v4l2_xioctl (self->fd, VIDIOC_QBUF, &buf)) {
            PyErr_Format(PyExc_EnvironmentError, "ioctl(VIDIOC_QBUF) failure : %d, %s",
                         errno, strerror (errno));
            return 0;
        }
    }

    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    if (-1 == v4l2_xioctl (self->fd, VIDIOC_STREAMON, &type)) {
        PyErr_Format(PyExc_EnvironmentError, "ioctl(VIDIOC_STREAMON) failure : %d, %s",
                      errno, strerror (errno));
        return 0;
    }

    return 1;
}

int v4l2_uninit_device (pgCameraObject* self)
{
    unsigned int i;

    for (i = 0; i < self->n_buffers; ++i) {
        if (-1 == munmap (self->buffers[i].start, self->buffers[i].length)) {
            PyErr_Format(PyExc_MemoryError, "munmap failure: %d, %s",
                         errno, strerror (errno));
            return 0;
        }
    }

    free (self->buffers);

    return 1;
}

int v4l2_init_mmap (pgCameraObject* self)
{
    struct v4l2_requestbuffers req;

    CLEAR (req);

    /* 2 is the minimum possible, and some drivers will force a higher count.
       It will likely result in buffer overruns, but for purposes of gaming,
       it is probably better to drop frames than get old frames. */
    req.count = 2;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (-1 == v4l2_xioctl (self->fd, VIDIOC_REQBUFS, &req)) {
        if (EINVAL == errno) {
            PyErr_Format(PyExc_MemoryError, "%s does not support memory mapping",
                self->device_name);
            return 0;
        }
        else {
            PyErr_Format(PyExc_MemoryError, "ioctl(VIDIOC_REQBUFS) failure : %d, %s",
                errno, strerror (errno));
            return 0;
        }
    }

    if (req.count < 2) {
        PyErr_Format(PyExc_MemoryError, "Insufficient buffer memory on %s\n",
            self->device_name);
        return 0;
    }

    self->buffers = calloc (req.count, sizeof (*self->buffers));

    if (!self->buffers) {
        PyErr_Format(PyExc_MemoryError, "Out of memory");
        return 0;
    }

    for (self->n_buffers = 0; self->n_buffers < req.count; ++self->n_buffers) {
        struct v4l2_buffer buf;

        CLEAR (buf);

        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = self->n_buffers;

        if (-1 == v4l2_xioctl (self->fd, VIDIOC_QUERYBUF, &buf)) {
            PyErr_Format(PyExc_MemoryError, "ioctl(VIDIOC_QUERYBUF) failure : %d, %s",
                errno, strerror (errno));
            return 0;
        }

        self->buffers[self->n_buffers].length = buf.length;
        self->buffers[self->n_buffers].start =
            mmap (NULL /* start anywhere */,
                buf.length,
                PROT_READ | PROT_WRITE /* required */,
                MAP_SHARED /* recommended */,
                self->fd, buf.m.offset);

        if (MAP_FAILED == self->buffers[self->n_buffers].start) {
            PyErr_Format(PyExc_MemoryError, "mmap failure : %d, %s",
                errno, strerror (errno));
            return 0;
        }
    }

    return 1;
}

int v4l2_init_device (pgCameraObject* self)
{
    struct v4l2_capability cap;
    struct v4l2_format fmt;
    unsigned int min;

    if (-1 == v4l2_xioctl (self->fd, VIDIOC_QUERYCAP, &cap)) {
        if (EINVAL == errno) {
            PyErr_Format(PyExc_SystemError, "%s is not a V4L2 device",
                self->device_name);
            return 0;
        }
        else {
            PyErr_Format(PyExc_SystemError, "ioctl(VIDIOC_QUERYCAP) failure : %d, %s",
                errno, strerror (errno));
            return 0;
        }
    }

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        PyErr_Format(PyExc_SystemError, "%s is not a video capture device",
            self->device_name);
        return 0;
    }

    if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
        PyErr_Format(PyExc_SystemError, "%s does not support streaming i/o",
            self->device_name);
        return 0;
    }

    CLEAR (fmt);

    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = self->width;
    fmt.fmt.pix.height = self->height;
    fmt.fmt.pix.field = V4L2_FIELD_ANY;

    /* Find the pixelformat supported by the camera that will take the least
       processing power to convert to the desired output.  Thus, for YUV out,
       YUYVand YUV420 are first, while for RGB and HSV, the packed RGB formats
       are first. */
    switch (self->color_out) {
        case YUV_OUT:
            if (v4l2_pixelformat(self->fd, &fmt, V4L2_PIX_FMT_YUYV)) {
                self->pixelformat = V4L2_PIX_FMT_YUYV;
            } else if (v4l2_pixelformat(self->fd, &fmt, V4L2_PIX_FMT_YUV420)) {
                self->pixelformat = V4L2_PIX_FMT_YUV420;
            } else if (v4l2_pixelformat(self->fd, &fmt, V4L2_PIX_FMT_UYVY)) {
                self->pixelformat = V4L2_PIX_FMT_UYVY;
            } else if (v4l2_pixelformat(self->fd, &fmt, V4L2_PIX_FMT_RGB24)) {
                self->pixelformat = V4L2_PIX_FMT_RGB24;
            } else if (v4l2_pixelformat(self->fd, &fmt, V4L2_PIX_FMT_RGB444)) {
                self->pixelformat = V4L2_PIX_FMT_RGB444;
            } else if (v4l2_pixelformat(self->fd, &fmt, V4L2_PIX_FMT_SBGGR8)) {
                self->pixelformat = V4L2_PIX_FMT_SBGGR8;
            } else {
                PyErr_Format(PyExc_SystemError,
                           "ioctl(VIDIOC_S_FMT) failure: no supported formats");
                return 0;
            }
            break;
        default:
            if (v4l2_pixelformat(self->fd, &fmt, V4L2_PIX_FMT_RGB24)) {
                self->pixelformat = V4L2_PIX_FMT_RGB24;
            } else if (v4l2_pixelformat(self->fd, &fmt, V4L2_PIX_FMT_RGB444)) {
                self->pixelformat = V4L2_PIX_FMT_RGB444;
            } else if (v4l2_pixelformat(self->fd, &fmt, V4L2_PIX_FMT_YUYV)) {
                self->pixelformat = V4L2_PIX_FMT_YUYV;
            } else if (v4l2_pixelformat(self->fd, &fmt, V4L2_PIX_FMT_SBGGR8)) {
                self->pixelformat = V4L2_PIX_FMT_SBGGR8;
            } else if (v4l2_pixelformat(self->fd, &fmt, V4L2_PIX_FMT_YUV420)) {
                self->pixelformat = V4L2_PIX_FMT_YUV420;
            } else if (v4l2_pixelformat(self->fd, &fmt, V4L2_PIX_FMT_UYVY)) {
                self->pixelformat = V4L2_PIX_FMT_UYVY;
            } else {
                PyErr_Format(PyExc_SystemError,
                           "ioctl(VIDIOC_S_FMT) failure: no supported formats");
                return 0;
            }
            break;
    }

    /* Note VIDIOC_S_FMT may change width and height. */
    self->width = fmt.fmt.pix.width;
    self->height = fmt.fmt.pix.height;
    self->size = self->width * self->height;
    self->pixelformat = fmt.fmt.pix.pixelformat;

    /* Buggy driver paranoia. */
    min = fmt.fmt.pix.width * 2;
    if (fmt.fmt.pix.bytesperline < min)
        fmt.fmt.pix.bytesperline = min;
    min = fmt.fmt.pix.bytesperline * fmt.fmt.pix.height;
    if (fmt.fmt.pix.sizeimage < min)
        fmt.fmt.pix.sizeimage = min;

    v4l2_init_mmap (self);

    return 1;
}

int v4l2_close_device (pgCameraObject* self)
{
    if (self->fd==-1)
        return 1;

    if (-1 == close (self->fd)) {
        PyErr_Format(PyExc_SystemError, "Cannot close '%s': %d, %s",
            self->device_name, errno, strerror (errno));
        return 0;
    }
    self->fd = -1;

    return 1;
}

int v4l2_open_device (pgCameraObject* self)
{
    struct stat st;

    if (-1 == stat (self->device_name, &st)) {
        PyErr_Format(PyExc_SystemError, "Cannot identify '%s': %d, %s",
            self->device_name, errno, strerror (errno));
        return 0;
    }

    if (!S_ISCHR (st.st_mode)) {
        PyErr_Format(PyExc_SystemError, "%s is no device",self->device_name);
        return 0;
    }

    self->fd = open (self->device_name, O_RDWR /* required | O_NONBLOCK */, 0);

    if (-1 == self->fd) {
        PyErr_Format(PyExc_SystemError, "Cannot open '%s': %d, %s",
            self->device_name, errno, strerror (errno));
        return 0;
    }

    return 1;
}
#endif
