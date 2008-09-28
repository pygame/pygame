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
#include "camera.h"

/*
 * V4L functions
 */
 
int v4l_open_device (PyCameraObject* self)
{
    struct stat st;
    struct video_capability cap;
    struct video_mbuf buf;
    
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
    
    if(ioctl(self->fd, VIDIOCGCAP, cap) == -1) {
        PyErr_Format(PyExc_SystemError, "%s is not a V4L device",
            self->device_name);        
	return 0;
    }
    
    if(!(cap.type & VID_TYPE_CAPTURE)) {
        PyErr_Format(PyExc_SystemError, "%s is not a video capture device",
            self->device_name);
        return 0;
    }
    
    if( ioctl(self->fd , VIDIOCGMBUF , buf ) == -1 ) {
        PyErr_Format(PyExc_SystemError, "%s does not support streaming i/o",
            self->device_name);
	return 0;
    }
    
    return 1;
}

int v4l_init_device(PyCameraObject* self)
{
    return 0;
}

int v4l_start_capturing(PyCameraObject* self)
{
    return 0;
}
#endif
