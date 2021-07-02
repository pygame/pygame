/*
 * Windows Camera - webcam support for pygame
 * Original Author: Charlie Hayden, 2021
 *
 * This sub-module adds native support for windows webcams to pygame,
 * made possible by Microsoft Media Foundation.
 * 
 * Since Media Foundation is still maturing, support is best in Windows 10 or
 * above, but I believe this will work in Windows 8 as well.
 * 
 * The Media Foundation documentation is written for use in C++, but the API
 * supports C as well. What would be a call to a class method in C++ is a call
 * through a pointer table lpVtbl.
 * For example obj->func(arguments) would be obj->lpVtbl->func(obj, arguments)
 * Also, GUIDs need to be addressed using '&' to work.
 */

#include "_camera.h"
#include "pgcompat.h"

/*these are already included in camera.h, but having them here
 * makes all the types be recognized by VS code */
#include <mfapi.h>
#include <mfobjects.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <combaseapi.h>
#include <mftransform.h>

#include <math.h>

#define RELEASE(obj) if (obj) {obj->lpVtbl->Release(obj);}

/* HRESULT failure numbers can be looked up on 
 * hresult.info to get the actual name */
#define CHECKHR(hr) if FAILED(hr) {PyErr_Format(pgExc_SDLError, "Media Foundation HRESULT failure %i on line %i", hr, __LINE__); return 0;}

#define FIRST_VIDEO MF_SOURCE_READER_FIRST_VIDEO_STREAM
#define DEVSOURCE_VIDCAP_GUID MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID
#define DEVSOURCE_NAME MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME

#define DISTANCE(x, y) (int)pow(x, 2) + (int)pow(y, 2)

//TODO: make camera able to be restarted in place, started and restarted, etc.
//TODO: check if started for get_image(), other functions?
//TOOD: improve memory management everywhere
//TODO: fix up get_raw() docs and stubs. It returns bytes

/* These are the only supported input types
 * (TODO?) broaden in the future by enumerating MFTs to find decoders?
 * drawn from:
 * https://docs.microsoft.com/en-us/windows/win32/medfound/video-processor-mft */
#define NUM_FORM 20
const GUID* inp_types[NUM_FORM] = {&MFVideoFormat_ARGB32, &MFVideoFormat_AYUV,
                                   &MFVideoFormat_I420, &MFVideoFormat_IYUV,
                                   &MFVideoFormat_NV11, &MFVideoFormat_NV12,
                                   &MFVideoFormat_RGB24, &MFVideoFormat_RGB32,
                                   &MFVideoFormat_RGB555, &MFVideoFormat_RGB8,
                                   &MFVideoFormat_RGB565, &MFVideoFormat_UYVY,
                                   &MFVideoFormat_v410, &MFVideoFormat_Y216,
                                   &MFVideoFormat_Y41P, &MFVideoFormat_Y41T,
                                   &MFVideoFormat_Y42T, &MFVideoFormat_YUY2,
                                   &MFVideoFormat_YV12, &MFVideoFormat_YVYU};

int _is_supported_input_format(GUID format) {
    for (int i=0; i<NUM_FORM; i++) {
        if (format.Data1 == inp_types[i]->Data1)
            return 1;
    }
    return 0;   
}

WCHAR *
get_attr_string(IMFActivate *pActive) 
{
    HRESULT hr = S_OK;
    UINT32 cchLength = 0;
    WCHAR *res = NULL;

    hr = pActive->lpVtbl->GetStringLength(pActive, &DEVSOURCE_NAME,
                                          &cchLength);
    
    if (SUCCEEDED(hr)) {
        res = malloc(sizeof(WCHAR)*(cchLength+1));
        if (res == NULL)
            hr = E_OUTOFMEMORY;
    }

    if (SUCCEEDED(hr)) {        
        hr = pActive->lpVtbl->GetString(pActive, &DEVSOURCE_NAME, res, 
                                        cchLength + 1, &cchLength);
    }

    return (WCHAR *)res;
}

WCHAR **
windows_list_cameras(int *num_devices) 
{
    WCHAR** devices = NULL;
    IMFAttributes *pAttributes = NULL;
    IMFActivate **ppDevices = NULL;

    HRESULT hr = MFCreateAttributes(&pAttributes, 1);
    if (FAILED(hr)) {
        printf("oof\n");
    }

    hr = pAttributes->lpVtbl->SetGUID(pAttributes, 
                                      &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
                                      &DEVSOURCE_VIDCAP_GUID);

    if (FAILED(hr)) {
        printf("oof2\n");
    }

    UINT32 count = -1;
    hr = MFEnumDeviceSources(pAttributes, &ppDevices, &count);

    if (FAILED(hr)) {
        printf("oof_3\n");
    }

    devices = (WCHAR **)malloc(sizeof(WCHAR *) * count);

    for (int i=0; i<(int)count; i++) {
        devices[i] = get_attr_string(ppDevices[i]);
    }

    for (int i=0; i<(int)count; i++) {
        RELEASE(ppDevices[i]);  
    }
    RELEASE(pAttributes)
    *num_devices = count;
    return devices;
}

IMFActivate *
windows_device_from_name(WCHAR* device_name) 
{
    IMFAttributes *pAttributes = NULL;
    IMFActivate **ppDevices = NULL;
    WCHAR* _device_name = NULL;

    HRESULT hr = MFCreateAttributes(&pAttributes, 1);
    if (FAILED(hr)) {
        printf("oof\n");
    }

    hr = pAttributes->lpVtbl->SetGUID(pAttributes, 
                                      &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, 
                                      &DEVSOURCE_VIDCAP_GUID);

    if (FAILED(hr)) {
        printf("oof2\n");
    }

    UINT32 count = -1;
    hr = MFEnumDeviceSources(pAttributes, &ppDevices, &count);

    if (FAILED(hr)) {
        printf("oof_3\n");
    }

    for (int i=0; i<(int)count; i++) {
        _device_name = get_attr_string(ppDevices[i]);
        if (!wcscmp(_device_name, device_name)) {
            free(_device_name);
            return ppDevices[i];
        }
        free(_device_name);
    }

    for (int i=0; i<(int)count; i++) {
        RELEASE(ppDevices[i]);  
    }
    RELEASE(pAttributes);
    return NULL;
}

int
_create_media_type(IMFMediaType** mp, IMFSourceReader* reader, int width,
                   int height)
{
    HRESULT hr;
    IMFMediaType* media_type = NULL;
    UINT64 size;
    int type_count = 0;
    UINT32 t_width, t_height;

    /* Find out how many native media types there are
     * (different resolution modes, mostly) */
    while(1) {
        hr = reader->lpVtbl->GetNativeMediaType(reader, FIRST_VIDEO,
                                                type_count, &media_type);
        if (hr) {
            break;
        }
        type_count++;
        RELEASE(media_type);
    }

    if (!type_count) {
        PyErr_SetString(pgExc_SDLError, 
                        "Could not find any video types on this camera");
        return 0;
    }

    IMFMediaType** native_types = malloc(sizeof(IMFMediaType*) * type_count);
    int* diagonal_distances = malloc(sizeof(int) * type_count);

    GUID subtype;

    /* This is interesting
     * https://social.msdn.microsoft.com/Forums/windowsdesktop/en-US/9d6a8704-764f-46df-a41c-8e9d84f7f0f3/mjpg-encoded-media-type-is-not-available-for-usbuvc-webcameras-after-windows-10-version-1607-os
     * Compressed video modes listed below are "backwards compatibility modes"
     * And will always have an uncompressed counterpart (NV12 / YUY2)
     * At least in Windows 10 since 2016 */

    for (int i=0; i < type_count; i++) {
        hr = reader->lpVtbl->GetNativeMediaType(reader, FIRST_VIDEO, i, 
                                                &media_type);
        CHECKHR(hr);

        hr = media_type->lpVtbl->GetUINT64(media_type, &MF_MT_FRAME_SIZE,
                                           &size);
        CHECKHR(hr);
    
        t_width = size >> 32;
        t_height = size << 32 >> 32;

        hr = media_type->lpVtbl->GetGUID(media_type, &MF_MT_SUBTYPE,
                                         &subtype);
        CHECKHR(hr);

        /* Debug printf for seeing what media types are available */
        /*
        printf("media_type: %li x %li, subtype id = %li\n", t_width, t_height,
               subtype.Data1);
        */

        if (_is_supported_input_format(subtype)) {
            native_types[i] = media_type;
            diagonal_distances[i] = DISTANCE(t_width, t_height);
        }
        else {
            /* types that are unsupported are set to NULL, which is checked in
             * the selection loop */
            native_types[i] = NULL;
            diagonal_distances[i] = 0;
        }
    }

    int difference = 100000000;
    int current_difference;
    int index = -1;
    int requested_diagonal = DISTANCE(width, height);

    for (int i=0; i < type_count; i++) {
        current_difference = diagonal_distances[i] - requested_diagonal;
        current_difference = abs(current_difference);
        if (current_difference < difference && native_types[i]) {
            index = i;
            difference = current_difference;
        }
    }

    if (index == -1) {
        PyErr_SetString(pgExc_SDLError,
                        "Could not find a valid video type in any supported"
                        " format");
        return 0;
    }

    printf("chosen index # =%i\n", index);

    hr = native_types[index]->lpVtbl->GetUINT64(native_types[index],
                                                &MF_MT_FRAME_SIZE, &size);
    CHECKHR(hr);

    printf("chosen width=%lli, chosen height=%lli\n", size >> 32, size << 32 >> 32);

    /* Can't hurt to tell the webcam to use its highest possible framerate
     * Although I haven't seen any upside from this either */
    UINT64 fps_max_ratio;
    hr = native_types[index]->lpVtbl->GetUINT64(native_types[index],
                                                &MF_MT_FRAME_RATE_RANGE_MAX,
                                                &fps_max_ratio);
    CHECKHR(hr);
    hr = native_types[index]->lpVtbl->SetUINT64(native_types[index],
                                                &MF_MT_FRAME_RATE,
                                                fps_max_ratio);
    CHECKHR(hr);

    *mp = native_types[index];
    return 1;
}

DWORD WINAPI 
update_function(LPVOID lpParam)
{
    pgCameraObject* self = (pgCameraObject*) lpParam;

    IMFSample *sample;
    HRESULT hr;
    DWORD pdwStreamFlags;

    IMFSourceReader* reader = self->reader;

    IMFMediaType* output_type;
    INT32 stride;

    while(1) {
        sample = NULL;

        /* flip control subsystem -
         * I managed to do it all within media foundation, without
         * camera_mac.m's flip_image, or v4l2's set_controls API.
         * The self->transform (Video Processor MFT) also exposes a control
         * interface, which can be used to set vertical or horizontal
         * mirroring, but not both. So for vertical mirroring, it changes the
         * stride of the image so it comes out of the transform upside down */
        if (self->hflip) {
            hr = self->control->lpVtbl->SetMirror(self->control,
                                                  MIRROR_HORIZONTAL);
            CHECKHR(hr);
        }
        else {
            hr = self->control->lpVtbl->SetMirror(self->control,
                                                  MIRROR_NONE);
            CHECKHR(hr);            
        }

        if (self->vflip != self->last_vflip) {
            hr = self->transform->lpVtbl->GetOutputCurrentType(self->transform,
                                                               0,
                                                               &output_type);
            CHECKHR(hr);
            hr = output_type->lpVtbl->GetUINT32(output_type,
                                                &MF_MT_DEFAULT_STRIDE,
                                                &stride);
            CHECKHR(hr);
            hr = output_type->lpVtbl->SetUINT32(output_type, 
                                                &MF_MT_DEFAULT_STRIDE,
                                                -stride);
            CHECKHR(hr);
            hr = self->transform->lpVtbl->SetOutputType(self->transform, 
                                                        0, output_type, 0);
            CHECKHR(hr);
            self->last_vflip = self->vflip;
        }

        hr = reader->lpVtbl->ReadSample(reader, FIRST_VIDEO, 0, 0,
                                        &pdwStreamFlags, NULL, &sample);
        if (hr == -1072875772) { //MF_E_HW_MFT_FAILED_START_STREAMING
            PyErr_SetString(PyExc_SystemError, "Camera already in use");
            return 0;
            //TODO: how are errors from this thread going to work?
        }
        CHECKHR(hr);

        if (!self->open) {
            RELEASE(sample);
            break;
        }

        if (sample) {
            /* Put unprocessed sample into another buffer for get_raw() */
            RELEASE(self->raw_buf);
            sample->lpVtbl->ConvertToContiguousBuffer(sample, &self->raw_buf);

            hr = self->transform->lpVtbl->ProcessInput(self->transform, 0,
                                                       sample, 0);
            CHECKHR(hr);

            MFT_OUTPUT_DATA_BUFFER mft_buffer[1];
            MFT_OUTPUT_DATA_BUFFER x;

            IMFSample* ns;
            hr = MFCreateSample(&ns);
            CHECKHR(hr);

            CHECKHR(ns->lpVtbl->AddBuffer(ns, self->buf));

            x.pSample = ns;
            x.dwStreamID = 0;
            x.dwStatus = 0;
            x.pEvents = NULL;
            mft_buffer[0] = x;

            DWORD out;
            hr = self->transform->lpVtbl->ProcessOutput(self->transform, 0, 1,
                                                        mft_buffer, &out);
            CHECKHR(hr);

            self->buffer_ready = 1;
        }

        RELEASE(sample);
    }

    printf("exiting 2nd thread...\n");
    ExitThread(0);
}

int
windows_open_device(pgCameraObject *self)
{
    IMFMediaSource *source;
    IMFSourceReader *reader = NULL;
    IMFMediaType *media_type = NULL;
    HRESULT hr;

    /* setup the stuff before MFCreateSourceReaderFromMediaSource is called */
    
    /* I wanted to use COINIT_MULTITHREADED, but something inside SDL started
     * by display.set_mode() uses COINIT_APARTMENTTHREADED, and you can't have
     * a thread in two modes at once */
    hr = CoInitializeEx(0, COINIT_APARTMENTTHREADED);
    CHECKHR(hr);

    hr = MFStartup(MF_VERSION, MFSTARTUP_LITE);
    CHECKHR(hr);

    hr = self->activate->lpVtbl->ActivateObject(self->activate,
                                                &IID_IMFMediaSource, &source);
    CHECKHR(hr);
    self->source = source;

    /* The commented out code below sets the source reader to use video
     * processing, which guarantees it can output RGB32 from any format.
     * Sadly, using the source reader in this way would only let me open
     * the webcam in a single size, so I replaced it with a video processor
     * MFT. (Which also helps with the flip controls). */
    /*
    IMFAttributes *rsa;
    MFCreateAttributes(&rsa, 1);
    rsa->lpVtbl->SetUINT32(rsa, &MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, 1);

    hr = MFCreateSourceReaderFromMediaSource(source, rsa, &reader);
    RELEASE(rsa);
    CHECKHR(hr);
    */

    hr = MFCreateSourceReaderFromMediaSource(source, NULL, &reader);
    CHECKHR(hr);

    self->reader = reader;
    
    if(!_create_media_type(&media_type, reader, self->width, self->height)) {
        return 0;
    }

    UINT64 size;
    hr = media_type->lpVtbl->GetUINT64(media_type, &MF_MT_FRAME_SIZE, &size);
    CHECKHR(hr);

    self->width = size >> 32;
    self->height = size << 32 >> 32;

    hr = reader->lpVtbl->SetCurrentMediaType(reader, FIRST_VIDEO, NULL, 
                                             media_type);
    CHECKHR(hr);

    IMFVideoProcessorControl* control;
    IMFTransform* transform;

    hr = CoCreateInstance(&CLSID_VideoProcessorMFT, NULL,
                          CLSCTX_INPROC_SERVER, &IID_IMFTransform,
                          &transform);
    CHECKHR(hr);

    hr = transform->lpVtbl->QueryInterface(transform,
                                           &IID_IMFVideoProcessorControl,
                                           &control);
    CHECKHR(hr);

    self->control = control;
    self->transform = transform;

    hr = transform->lpVtbl->SetInputType(transform, 0, media_type, 0);
    CHECKHR(hr);

    IMFMediaType* conv_type;
    CHECKHR(MFCreateMediaType(&conv_type));
    hr = conv_type->lpVtbl->SetGUID(conv_type, &MF_MT_MAJOR_TYPE,
                                    &MFMediaType_Video);
    CHECKHR(hr);

    int depth;

    hr = conv_type->lpVtbl->SetUINT64(conv_type, &MF_MT_FRAME_SIZE, size);
    CHECKHR(hr);

    if (self->color_out != YUV_OUT) {
        hr = conv_type->lpVtbl->SetGUID(conv_type, &MF_MT_SUBTYPE,
                                        &MFVideoFormat_RGB32);
        CHECKHR(hr);

        hr = transform->lpVtbl->SetOutputType(transform, 0, conv_type, MFT_SET_TYPE_TEST_ONLY);
        CHECKHR(hr);
        self->pixelformat = MFVideoFormat_RGB32.Data1;
        depth = 4;
    }

    else {
        hr = conv_type->lpVtbl->SetGUID(conv_type, &MF_MT_SUBTYPE,
                                        &MFVideoFormat_YUY2);
        CHECKHR(hr);

        hr = transform->lpVtbl->SetOutputType(transform, 0, conv_type, MFT_SET_TYPE_TEST_ONLY);
        if (hr == -1072875852) { //MF_E_INVALIDMEDIATYPE
            hr = conv_type->lpVtbl->SetGUID(conv_type, &MF_MT_SUBTYPE,
                                            &MFVideoFormat_RGB32);
            CHECKHR(hr);

            hr = transform->lpVtbl->SetOutputType(transform, 0, conv_type, MFT_SET_TYPE_TEST_ONLY);
            CHECKHR(hr);
            self->pixelformat = MFVideoFormat_RGB32.Data1;
            depth = 4;
        }
        else {
            CHECKHR(hr);
            self->pixelformat = MFVideoFormat_YUY2.Data1;
            depth = 2;
        }   
    }

    /* make sure the output is right side up by default
     * multiplied by 4 because that is the number of bytes per pixel */
    hr = conv_type->lpVtbl->SetUINT32(conv_type, &MF_MT_DEFAULT_STRIDE, 
                                      self->width * depth);
    CHECKHR(hr);

    hr = transform->lpVtbl->SetOutputType(transform, 0, conv_type, 0);
    CHECKHR(hr);

    MFT_OUTPUT_STREAM_INFO info;
    hr = self->transform->lpVtbl->GetOutputStreamInfo(self->transform, 0, 
                                                      &info);
    CHECKHR(hr);

    CHECKHR(MFCreateMemoryBuffer(info.cbSize, &self->buf));

    HANDLE update_thread = CreateThread(NULL, 0, update_function, self, 0, 
                                        NULL);
    self->t_handle = update_thread;

    return 1;
}

int
windows_close_device(pgCameraObject *self)
{
    self->open = 0;
    WaitForSingleObject(self->t_handle, 3000);

    RELEASE(self->reader);
    CHECKHR(MFShutdown());

    CoUninitialize();
    return 1;
}

int
windows_process_image(pgCameraObject *self, BYTE* data, DWORD length,
                      SDL_Surface *surf)
{
    SDL_LockSurface(surf);

    int size = self->width * self->height;

    /* apparently RGB32 is actually BGR to Microsoft */
    if (self->pixelformat == MFVideoFormat_RGB32.Data1) {
        switch(self->color_out) {
            case RGB_OUT:
                /* optimized for 32 bit output surfaces
                 * this won't be possible always, TODO implement switching logic */
                //memcpy(surf->pixels, data, length);

                bgr32_to_rgb(data, surf->pixels, size, surf->format);
                break;
            case YUV_OUT:
                rgb_to_yuv(data, surf->pixels, size, V4L2_PIX_FMT_XBGR32, surf->format);
                break;
            case HSV_OUT:
                rgb_to_hsv(data, surf->pixels, size, V4L2_PIX_FMT_XBGR32, surf->format);
                break;
        }
    }

    if (self->pixelformat == MFVideoFormat_YUY2.Data1) {
        switch (self->color_out) {
            case YUV_OUT:
                yuyv_to_yuv(data, surf->pixels, size, surf->format);
                break;
            case RGB_OUT:
                yuyv_to_rgb(data, surf->pixels, size, surf->format);
                break;
            case HSV_OUT:
                yuyv_to_rgb(data, surf->pixels, size, surf->format);
                rgb_to_hsv(surf->pixels, surf->pixels, size, V4L2_PIX_FMT_YUYV, surf->format);
                break;
        }
    }

    SDL_UnlockSurface(surf);

    return 1;
}   

int
windows_read_frame(pgCameraObject *self, SDL_Surface *surf)
{
    HRESULT hr;

    if (self->buf) {
        BYTE *buf_data;
        DWORD buf_max_length;
        DWORD buf_length;
        hr = self->buf->lpVtbl->Lock(self->buf, &buf_data, &buf_max_length,
                                     &buf_length);
        CHECKHR(hr);

        if (!windows_process_image(self, buf_data, buf_length, surf)) {
            return 0;
        }

        hr = self->buf->lpVtbl->Unlock(self->buf);
        CHECKHR(hr);

        self->buffer_ready = 0;
    }

    return 1;
}

int
windows_frame_ready(pgCameraObject *self)
{
    return self->buffer_ready;
}              

PyObject*
windows_read_raw(pgCameraObject *self)
{
    PyObject* data = NULL;
    HRESULT hr;

    if (self->raw_buf) {
        BYTE *buf_data;
        DWORD buf_max_length;
        DWORD buf_length;
        hr = self->raw_buf->lpVtbl->Lock(self->raw_buf, &buf_data,
                                         &buf_max_length, &buf_length);
        CHECKHR(hr);

        data = Bytes_FromStringAndSize(buf_data, buf_length);

        hr = self->raw_buf->lpVtbl->Unlock(self->raw_buf);
        CHECKHR(hr);

        self->buffer_ready = 0;

        return data;
    }
    
    /* should this be an error instead? */
    Py_RETURN_NONE;
}