//
//  camera_mac.m
//
//  Created by Werner Laurensse on 2009-05-28.
//  Copyright (c) 2009 . All rights reserved.
//

#import "camera.h"
#import "pgcompat.h"

#if defined(PYGAME_MAC_CAMERA_OLD)



/* Flips the image array horizontally and/or vertically by reverse copying
 * a 'depth' number of bytes to flipped_image.*/
/* todo speed up... */
void flip_image(const void* image, 
                void* flipped_image, 
                int width, int height,
                short depth, 
                int hflip, int vflip) 
{

    if (!hflip && vflip) {
        int i, j;
        int width_size = width*depth;
        const void* tmp_image = image;
        
        for(i=0; i<=height-1; i++) {
            for(j=0; j<=width; j++) {
                memcpy(flipped_image+width_size-j*depth-3,
                       tmp_image+j*depth,
                       depth);
            }
            tmp_image += width_size;
            flipped_image += width_size;
        }
    } else if (hflip && !vflip) {
        int i;
        int width_size = width*depth;
        void* tmp_image = flipped_image+width_size*height;
        
        for(i=0; i<height; i++) {
            tmp_image -= width_size;
            memcpy(tmp_image, image+i*width_size, width_size);
        }
    } else if (hflip && vflip) {
        int i, j;
        int width_size = width*depth;
        void* tmp_image = flipped_image + width_size*height;
    
        for(i=0; i<=height-1; i++) {
            for(j=0; j<=width; j++) {
                memcpy(tmp_image-j*depth-3,
                       image+j*depth,
                       depth);
            }
            tmp_image -= width_size;
            image += width_size;
        }
    } else {
        memcpy(flipped_image, image, height*width*depth);
    }
}





/* 
 * return: an array of the available cameras ids.
 * num_cameras: number of cameras in array.
 */
char** mac_list_cameras(int* num_cameras) {
    char** cameras;
    char* camera;
    
    // The ComponentDescription allows you to search for Components that can be used to capture
    // images, and thus can be used as camera.
    ComponentDescription cameraDescription;
    memset(&cameraDescription, 0, sizeof(ComponentDescription));
    cameraDescription.componentType = SeqGrabComponentType;
    
    // Count the number of cameras on the system, and allocate an array for the cameras
    *num_cameras = (int) CountComponents(&cameraDescription);
    cameras = (char **) malloc(sizeof(char*) * *num_cameras);
    
    // Try to find a camera, and add the camera's 'number' as a string to the array of cameras 
    Component cameraComponent = FindNextComponent(0, &cameraDescription);
    short num = 0;
    while(cameraComponent != NULL) {
        camera = malloc(sizeof(char) * 50); //TODO: find a better way to do this...
        sprintf(camera, "%d", cameraComponent);
        cameras[num] = camera;
        cameraComponent = FindNextComponent(cameraComponent, &cameraDescription);
        num++;
    }
    return cameras;
}

/* Open a Camera component. */
int mac_open_device (PyCameraObject* self) {
    OSErr theErr;

    // Initialize movie toolbox
    theErr = EnterMovies();
    if (theErr != noErr) {
        PyErr_Format(PyExc_SystemError,
        "Cannot initializes the Movie Toolbox");
        return 0;
    }
    
    // Open camera component
    SeqGrabComponent component = OpenComponent((Component) atoi(self->device_name));
    if (component == NULL) {
        PyErr_Format(PyExc_SystemError,
        "Cannot open '%s'", self->device_name);
        return 0;
    }
    self->component = component;
    
    return 1;
}

/* Make the Camera object ready for capturing images. */
int mac_init_device(PyCameraObject* self) {
    OSErr theErr;

    if (self->color_out == YUV_OUT) {
        self->pixelformat = kYUVSPixelFormat;
        self->depth = 2;
    } else {
        self->pixelformat = k24RGBPixelFormat;
        self->depth = 3;
    }
    
    int rowlength = self->boundsRect.right * self->depth;
	
	theErr = SGInitialize(self->component);
    if (theErr != noErr) {
        PyErr_Format(PyExc_SystemError,
        "Cannot initialize sequence grabber component");
        return 0;
    }
    
	
    theErr = SGSetDataRef(self->component, 0, 0, seqGrabDontMakeMovie);
    if (theErr != noErr) {
        PyErr_Format(PyExc_SystemError,
        "Cannot set the sequence grabber destination data reference for a record operation");
        return 0;
    }
        
    theErr = SGNewChannel(self->component, VideoMediaType, &self->channel);
    if (theErr != noErr) {
        PyErr_Format(PyExc_SystemError,
        "Cannot creates a sequence grabber channel and assigns a channel component to the channel");
        return 0;
    }
    
    //theErr = SGSettingsDialog (self->component, self->channel, 0, NULL, 0, NULL, 0);
    
    theErr = SGSetChannelBounds(self->channel, &self->boundsRect);
    if (theErr != noErr) {
        PyErr_Format(PyExc_SystemError,
        "Cannot specifie a channel's display boundary rectangle");
        return 0;
    }
	
	/*
	theErr = SGSetFrameRate (vc, fps);
    if(theErr != noErr){
        PyErr_Format(PyExc_SystemError,
        "Cannot set the frame rate of the sequence grabber");
        return 0;
    }
    */
      
    theErr = SGSetChannelUsage(self->channel, seqGrabPreview);
    if (theErr != noErr) {
        PyErr_Format(PyExc_SystemError,
        "Cannot specifie how a channel is to be used by the sequence grabber componen");
        return 0;
    }
    
	theErr = SGSetChannelPlayFlags(self->channel, channelPlayAllData);
	if (theErr != noErr) {
        PyErr_Format(PyExc_SystemError,
        "Cannot adjust the speed and quality with which the sequence grabber displays data from a channel");
        return 0;
	}
	
    self->pixels.length = self->boundsRect.right * self->boundsRect.bottom * self->depth;
	self->pixels.start = (unsigned char*) malloc(self->pixels.length);
	
	theErr = QTNewGWorldFromPtr(&self->gworld,
								self->pixelformat,
                                &self->boundsRect, 
                                NULL, 
                                NULL, 
                                0, 
                                self->pixels.start, 
                                rowlength);
        
	if (theErr != noErr) {
	    PyErr_Format(PyExc_SystemError,
	    "Cannot wrap a graphics world and pixel map structure around an existing block of memory containing an image, "
	    "failed to run QTNewGWorldFromPtr");
		free(self->pixels.start);
		self->pixels.start = NULL;
        self->pixels.length = 0;
		return 0;
	}  
	
    if (self->gworld == NULL) {
		PyErr_Format(PyExc_SystemError,
		"Cannot wrap a graphics world and pixel map structure around an existing block of memory containing an image, "
		"gworld is NULL");
		free(self->pixels.start);
		self->pixels.start = NULL;
        self->pixels.length = 0;
		return 0;
	}
	
    theErr = SGSetGWorld(self->component, (CGrafPtr)self->gworld, NULL);
	if (theErr != noErr) {
		PyErr_Format(PyExc_SystemError,
		"Cannot establishe the graphics port and device for a sequence grabber component");
        free(self->pixels.start);
		self->pixels.start = NULL;
        self->pixels.length = 0;
		return 0;
	}
	    
    return 1;
}

/* Start Capturing */
int mac_start_capturing(PyCameraObject* self) {
    OSErr theErr;
    
    theErr = SGPrepare(self->component, true, false);
    if (theErr != noErr) {
        PyErr_Format(PyExc_SystemError,
        "Cannot istruct a sequence grabber to get ready to begin a preview or record operation");
        free(self->pixels.start);
		self->pixels.start = NULL;
        self->pixels.length = 0;
        return 0;
	}
	
	theErr = SGStartPreview(self->component);
    if (theErr != noErr) {
        PyErr_Format(PyExc_SystemError,
        "Cannot instruct the sequence grabber to begin processing data from its channels");
        free(self->pixels.start);
		self->pixels.start = NULL;
        self->pixels.length = 0;
        return 0;
	}
	
    return 1;
}

/* Close the camera component, and stop the image capturing if necessary. */
int mac_close_device (PyCameraObject* self) {
    ComponentResult theErr;
    
    // Stop recording
   	if (self->component)
   		SGStop(self->component);

    // Close sequence grabber component
   	if (self->component) {
   		theErr = CloseComponent(self->component);
   		if (theErr != noErr) {
   			PyErr_Format(PyExc_SystemError,
   			"Cannot close sequence grabber component");
            return 0;
   		}
   		self->component = NULL;
   	}
    
    // Dispose of GWorld
   	if (self->gworld) {
   		DisposeGWorld(self->gworld);
   		self->gworld = NULL;
   	}
   	// Dispose of pixels buffer
    free(self->pixels.start);
    self->pixels.start = NULL;
    self->pixels.length = 0;
    return 1;
}

/* Stop capturing. */
int mac_stop_capturing (PyCameraObject* self) {
    OSErr theErr = SGStop(self->component);
    if (theErr != noErr) {
        PyErr_Format(PyExc_SystemError,
        "Could not stop the sequence grabber with previewing");
        return 0;
    }
    return 1;
}

/* Read a frame, and put the raw data into a python string. */
PyObject *mac_read_raw(PyCameraObject *self) {
    if (self->gworld == NULL) {
        PyErr_Format(PyExc_SystemError,
        "Cannot set convert gworld to surface because gworls is 0");
        return 0;
    }
    
    if (mac_camera_idle(self) == 0) {
        return 0;
    }
    
    PyObject *raw;
    PixMapHandle pixmap_handle = GetGWorldPixMap(self->gworld);
    LockPixels(pixmap_handle);
    raw = Bytes_FromStringAndSize(self->pixels.start, self->pixels.length);
    UnlockPixels(pixmap_handle);
    return raw;
}

/* Read a frame from the camera and copy it to a surface. */
int mac_read_frame(PyCameraObject* self, SDL_Surface* surf) {
    if (mac_camera_idle(self) != 0) {
        return mac_process_image(self, self->pixels.start, self->pixels.length, surf);
    } else {
        return 0;
    }
}

// TODO sometimes it is posible to directly grab the image in the desired pixel format,
// but this format needs to be known at the beginning of the initiation of the camera.
int mac_process_image(PyCameraObject* self, const void *image, unsigned int buffer_size, SDL_Surface* surf) {
    if (!surf)
        return 0;
    
    void* new_pixels;
    if (self->hflip || self->vflip) {
        new_pixels = malloc(self->pixels.length);
        flip_image(self->pixels.start,
                   new_pixels,
                   self->boundsRect.right,
                   self->boundsRect.bottom,
                   self->depth,
                   self->hflip,
                   self->vflip);
    } else {
        new_pixels = image;
    }
    
    SDL_LockSurface(surf);
    
    switch (self->pixelformat) {
        case k24RGBPixelFormat:
            if (buffer_size >= self->size * 3) {
                switch (self->color_out) {
                    case RGB_OUT:
                        rgb24_to_rgb(new_pixels, surf->pixels, self->size, surf->format);
                        break;
                    case HSV_OUT:
                        rgb_to_hsv(new_pixels, surf->pixels, self->size, V4L2_PIX_FMT_RGB24, surf->format);
                        break;
                    case YUV_OUT:
                        rgb_to_yuv(new_pixels, surf->pixels, self->size, V4L2_PIX_FMT_RGB24, surf->format);
                        break;
                }
            } else {
                SDL_UnlockSurface(surf);
                free(new_pixels);
                return 0;
            }
            break;
        
        case kYUVSPixelFormat:
            if (buffer_size >= self->size * 2) {
                switch (self->color_out) {
                    case YUV_OUT:
                        yuyv_to_yuv(new_pixels, surf->pixels, self->size, surf->format);
                        break;
                    case RGB_OUT:
                        yuyv_to_rgb(new_pixels, surf->pixels, self->size, surf->format);
                        break;
                    case HSV_OUT:
                        yuyv_to_rgb(new_pixels, surf->pixels, self->size, surf->format);
                        rgb_to_hsv(surf->pixels, surf->pixels, self->size, V4L2_PIX_FMT_YUYV, surf->format);
                        break;
                }
            } else {
                SDL_UnlockSurface(surf);
                free(new_pixels);
                return 0;
            }
            break;
    }
    SDL_UnlockSurface(surf);
    if (self->hflip || self->vflip)
        free(new_pixels);
    
    return 1;
}

/* Put the camera in idle mode. */
int mac_camera_idle(PyCameraObject* self) {
    OSErr theErr = SGIdle(self->component);
    if (theErr != noErr) {
        PyErr_Format(PyExc_SystemError, "SGIdle failed");
        return 0;
    }
    
    return 1;
}


#endif
