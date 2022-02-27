/*
  pygame - Python Game Library
  Copyright (C) 2021 Charlie Hayden

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

#define RELEASE(obj)               \
    if (obj) {                     \
        obj->lpVtbl->Release(obj); \
        obj = NULL;                \
    }

/* HRESULT failure numbers can be looked up on
 * hresult.info to get the actual name */
#define FORMATHR(hr, line)                                 \
    PyErr_Format(pgExc_SDLError,                           \
                 "Media Foundation "                       \
                 "HRESULT failure %i on camera_windows.c " \
                 "line %i",                                \
                 hr, line);

#define CHECKHR(hr)            \
    if (FAILED(hr)) {          \
        FORMATHR(hr, __LINE__) \
        return 0;              \
    }

#define HANDLEHR(hr)           \
    if (FAILED(hr)) {          \
        FORMATHR(hr, __LINE__) \
        goto cleanup;          \
    }

#define T_HANDLEHR(hr)                 \
    if (FAILED(hr)) {                  \
        self->t_error = hr;            \
        self->t_error_line = __LINE__; \
        break;                         \
    }

#define T_FORMATHR(hr, line)                               \
    PyErr_Format(pgExc_SDLError,                           \
                 "Media Foundation "                       \
                 "HRESULT failure %i on camera_windows.c " \
                 "line %i (Capture Thread Quit)",          \
                 hr, line);

int
_check_integrity(pgCameraObject *self)
{
    if (FAILED(self->t_error)) {
        /* MF_E_HW_MFT_FAILED_START_STREAMING */
        if (self->t_error == (HRESULT)-1072875772) {
            PyErr_SetString(PyExc_SystemError,
                            "Camera already in use (Capture Thread Quit)");
        }
        /* MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED */
        else if (self->t_error == (HRESULT)-1072873822) {
            PyErr_SetString(PyExc_SystemError,
                            "Camera disconnected (Capture Thread Quit)");
        }
        else {
            T_FORMATHR(self->t_error, self->t_error_line)
        }
        return 0;
    }
    return 1;
}

#define FIRST_VIDEO MF_SOURCE_READER_FIRST_VIDEO_STREAM
#define DEVSOURCE_VIDCAP_GUID MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID
#define DEVSOURCE_NAME MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME

#define HR_UPSTREAM_FAILURE -12345

#define DISTANCE(x, y) (int)pow(x, 2) + (int)pow(y, 2)

/* These are the only supported input types
 * (TODO?) broaden in the future by enumerating MFTs to find decoders?
 * drawn from:
 * https://docs.microsoft.com/en-us/windows/win32/medfound/video-processor-mft
 */
#define NUM_FORM 20
const GUID *inp_types[NUM_FORM] = {
    &MFVideoFormat_ARGB32, &MFVideoFormat_AYUV,   &MFVideoFormat_I420,
    &MFVideoFormat_IYUV,   &MFVideoFormat_NV11,   &MFVideoFormat_NV12,
    &MFVideoFormat_RGB24,  &MFVideoFormat_RGB32,  &MFVideoFormat_RGB555,
    &MFVideoFormat_RGB8,   &MFVideoFormat_RGB565, &MFVideoFormat_UYVY,
    &MFVideoFormat_v410,   &MFVideoFormat_Y216,   &MFVideoFormat_Y41P,
    &MFVideoFormat_Y41T,   &MFVideoFormat_Y42T,   &MFVideoFormat_YUY2,
    &MFVideoFormat_YV12,   &MFVideoFormat_YVYU};

int
_is_supported_input_format(GUID format)
{
    for (int i = 0; i < NUM_FORM; i++) {
        if (format.Data1 == inp_types[i]->Data1)
            return 1;
    }
    return 0;
}

int
_get_name_from_activate(IMFActivate *pActive, WCHAR **ppAttr)
{
    HRESULT hr;
    UINT32 cchLength = 0;
    WCHAR *res = NULL;

    hr =
        pActive->lpVtbl->GetStringLength(pActive, &DEVSOURCE_NAME, &cchLength);
    CHECKHR(hr);

    res = malloc(sizeof(WCHAR) * (cchLength + 1));
    if (!res) {
        hr = E_OUTOFMEMORY;
        CHECKHR(hr);
    }

    hr = pActive->lpVtbl->GetString(pActive, &DEVSOURCE_NAME, res,
                                    cchLength + 1, &cchLength);
    HANDLEHR(hr);

    *ppAttr = res;
    return 1;

cleanup:
    free(res);
    return 0;
}

WCHAR **
windows_list_cameras(int *num_devices)
{
    WCHAR **devices = NULL;
    IMFAttributes *pAttributes = NULL;
    IMFActivate **ppDevices = NULL;
    UINT32 count = 0;
    HRESULT hr;

    hr = MFCreateAttributes(&pAttributes, 1);
    HANDLEHR(hr);

    hr = pAttributes->lpVtbl->SetGUID(pAttributes,
                                      &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
                                      &DEVSOURCE_VIDCAP_GUID);

    HANDLEHR(hr);

    hr = MFEnumDeviceSources(pAttributes, &ppDevices, &count);
    HANDLEHR(hr);

    /* freed by caller of this function */
    devices = (WCHAR **)malloc(sizeof(WCHAR *) * count);
    if (!devices) {
        hr = E_OUTOFMEMORY;
        HANDLEHR(hr);
    }

    for (int i = 0; i < (int)count; i++) {
        if (!_get_name_from_activate(ppDevices[i], &devices[i])) {
            hr = HR_UPSTREAM_FAILURE;
            goto cleanup;
        }
    }

    *num_devices = count;
    goto cleanup;

cleanup:
    RELEASE(pAttributes);
    for (int i = 0; i < (int)count; i++) {
        RELEASE(ppDevices[i]);
    }
    CoTaskMemFree(ppDevices);
    if (FAILED(hr)) {
        if (devices) {
            for (int i = 0; i < (int)count; i++) {
                free(devices[i]);
            }
            free(devices);
            return NULL;
        }
        else {
            return NULL;
        }
    }
    else {
        return devices;
    }
}

IMFActivate *
windows_device_from_name(WCHAR *device_name)
{
    IMFAttributes *pAttributes = NULL;
    IMFActivate **ppDevices = NULL;
    IMFActivate *ret_device;
    WCHAR *_device_name = NULL;
    UINT32 count = 0;
    HRESULT hr;

    hr = MFCreateAttributes(&pAttributes, 1);
    HANDLEHR(hr);

    hr = pAttributes->lpVtbl->SetGUID(pAttributes,
                                      &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
                                      &DEVSOURCE_VIDCAP_GUID);
    HANDLEHR(hr);

    hr = MFEnumDeviceSources(pAttributes, &ppDevices, &count);
    HANDLEHR(hr);

    for (int i = 0; i < (int)count; i++) {
        if (!_get_name_from_activate(ppDevices[i], &_device_name)) {
            hr = HR_UPSTREAM_FAILURE;
            goto cleanup;
        }
        if (!wcscmp(_device_name, device_name)) {
            free(_device_name);
            for (int j = 0; j < (int)count; j++) {
                if (i != j) {
                    RELEASE(ppDevices[j]);
                }
            }
            ret_device = ppDevices[i];
            CoTaskMemFree(ppDevices);
            return ret_device;
        }
        free(_device_name);
    }

    goto cleanup;

cleanup:
    RELEASE(pAttributes);
    for (int i = 0; i < (int)count; i++) {
        RELEASE(ppDevices[i]);
    }
    CoTaskMemFree(ppDevices);
    return NULL;
}

/* Enumerates the formats supported "natively" by the webcam, and picks one
 * that is A) supported by the video processor MFT used for processing, and
 * B) close to the requested size */
int
_select_source_type(pgCameraObject *self, IMFMediaType **mp)
{
    HRESULT hr;
    IMFMediaType *media_type = NULL;
    int type_count = 0;
    UINT32 t_width, t_height;
    IMFMediaType **native_types = NULL;
    int *diagonal_distances = NULL;
    UINT64 size;
    GUID subtype;
    int difference = 100000000;
    int current_difference;
    int index = -1;
    int requested_diagonal = DISTANCE(self->width, self->height);
    UINT64 fps_max_ratio;

    /* Find out how many native media types there are
     * (different resolution modes, mostly) */
    while (1) {
        hr = self->reader->lpVtbl->GetNativeMediaType(
            self->reader, FIRST_VIDEO, type_count, &media_type);
        if (FAILED(hr)) {
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

    native_types = malloc(sizeof(IMFMediaType *) * type_count);
    diagonal_distances = malloc(sizeof(int) * type_count);
    if (!native_types || !diagonal_distances) {
        hr = E_OUTOFMEMORY;
        HANDLEHR(hr);
    }

    /* This is interesting
     * https://social.msdn.microsoft.com/Forums/windowsdesktop/en-US/9d6a8704-764f-46df-a41c-8e9d84f7f0f3/mjpg-encoded-media-type-is-not-available-for-usbuvc-webcameras-after-windows-10-version-1607-os
     * Compressed video modes listed below are "backwards compatibility modes"
     * And will always have an uncompressed counterpart (NV12 / YUY2)
     * At least in Windows 10 since 2016 */

    for (int i = 0; i < type_count; i++) {
        hr = self->reader->lpVtbl->GetNativeMediaType(
            self->reader, FIRST_VIDEO, i, &media_type);
        HANDLEHR(hr);

        hr = media_type->lpVtbl->GetUINT64(media_type, &MF_MT_FRAME_SIZE,
                                           &size);
        HANDLEHR(hr);

        t_width = size >> 32;
        t_height = size << 32 >> 32;

        hr = media_type->lpVtbl->GetGUID(media_type, &MF_MT_SUBTYPE, &subtype);
        HANDLEHR(hr);

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

    for (int i = 0; i < type_count; i++) {
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
        hr = HR_UPSTREAM_FAILURE;
        goto cleanup;
    }

    hr = native_types[index]->lpVtbl->GetUINT64(native_types[index],
                                                &MF_MT_FRAME_SIZE, &size);
    HANDLEHR(hr);

    self->width = size >> 32;
    self->height = size << 32 >> 32;

    /* More debug printfs that could be handy in the future */
    /*
    printf("chosen index # =%i\n", index);
    printf("chosen width=%lli, chosen height=%lli\n", size >> 32,
           size << 32 >> 32);
    */

    /* Can't hurt to tell the webcam to use its highest possible framerate
     * Although I haven't seen any upside from this either */
    hr = native_types[index]->lpVtbl->GetUINT64(
        native_types[index], &MF_MT_FRAME_RATE_RANGE_MAX, &fps_max_ratio);
    HANDLEHR(hr);
    hr = native_types[index]->lpVtbl->SetUINT64(
        native_types[index], &MF_MT_FRAME_RATE, fps_max_ratio);
    HANDLEHR(hr);

    *mp = native_types[index];
    goto cleanup;

cleanup:
    free(diagonal_distances);

    for (int i = 0; i < type_count; i++) {
        if (i != index) {
            RELEASE(native_types[i]);
        }
    }

    if (FAILED(hr)) {
        if (index > 0) {
            RELEASE(native_types[index]);
        }
        free(native_types);
        return 0;
    }
    else {
        free(native_types);
        return 1;
    }
}

/* Based on the requested video format out (RGB, HSV, YUV), attempts to find
 * a conversion type that works with the source type. MSMF makes very few
 * guarantees, but RGB32 has the best support. If even RGB32 fails for some
 * reason, it raises an exception. The conversion type is the output of the
 * video processor MFT, the source type is the input. */
int
_select_conversion_type(pgCameraObject *self, IMFMediaType **mp)
{
    IMFMediaType *conv_type = NULL;
    HRESULT hr;
    int depth;
    IMFTransform *transform = self->transform;
    UINT64 size;

    hr = MFCreateMediaType(&conv_type);
    HANDLEHR(hr);

    hr = conv_type->lpVtbl->SetGUID(conv_type, &MF_MT_MAJOR_TYPE,
                                    &MFMediaType_Video);
    HANDLEHR(hr);

    /* reconstruct the size num out of width/height set by the source */
    size = self->width;
    size = size << 32;
    size += self->height;
    hr = conv_type->lpVtbl->SetUINT64(conv_type, &MF_MT_FRAME_SIZE, size);
    HANDLEHR(hr);

    if (self->color_out != YUV_OUT) {
        hr = conv_type->lpVtbl->SetGUID(conv_type, &MF_MT_SUBTYPE,
                                        &MFVideoFormat_RGB32);
        HANDLEHR(hr);

        hr = transform->lpVtbl->SetOutputType(transform, 0, conv_type,
                                              MFT_SET_TYPE_TEST_ONLY);
        HANDLEHR(hr);
        self->pixelformat = MFVideoFormat_RGB32.Data1;
        depth = 4;
    }

    else {
        hr = conv_type->lpVtbl->SetGUID(conv_type, &MF_MT_SUBTYPE,
                                        &MFVideoFormat_YUY2);
        HANDLEHR(hr);

        hr = transform->lpVtbl->SetOutputType(transform, 0, conv_type,
                                              MFT_SET_TYPE_TEST_ONLY);
        if (hr == (HRESULT)-1072875852) {  // MF_E_INVALIDMEDIATYPE
            hr = conv_type->lpVtbl->SetGUID(conv_type, &MF_MT_SUBTYPE,
                                            &MFVideoFormat_RGB32);
            HANDLEHR(hr);

            hr = transform->lpVtbl->SetOutputType(transform, 0, conv_type,
                                                  MFT_SET_TYPE_TEST_ONLY);
            HANDLEHR(hr);
            self->pixelformat = MFVideoFormat_RGB32.Data1;
            depth = 4;
        }
        else {
            HANDLEHR(hr);
            self->pixelformat = MFVideoFormat_YUY2.Data1;
            depth = 2;
        }
    }

    /* make sure the output is right side up by default
     * multiplied by 4 because that is the number of bytes per pixel */
    hr = conv_type->lpVtbl->SetUINT32(conv_type, &MF_MT_DEFAULT_STRIDE,
                                      self->width * depth);
    HANDLEHR(hr);

    *mp = conv_type;
    return 1;

cleanup:
    RELEASE(conv_type);
    return 0;
}

DWORD WINAPI
update_function(LPVOID lpParam)
{
    pgCameraObject *self = (pgCameraObject *)lpParam;

    IMFSample *sample;
    HRESULT hr;
    DWORD pdwStreamFlags;

    IMFSourceReader *reader = self->reader;

    IMFMediaType *output_type;
    INT32 stride;

    while (1) {
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
            T_HANDLEHR(hr);
        }
        else {
            hr = self->control->lpVtbl->SetMirror(self->control, MIRROR_NONE);
            T_HANDLEHR(hr);
        }

        if (self->vflip != self->last_vflip) {
            hr = self->transform->lpVtbl->GetOutputCurrentType(
                self->transform, 0, &output_type);
            T_HANDLEHR(hr);
            hr = output_type->lpVtbl->GetUINT32(
                output_type, &MF_MT_DEFAULT_STRIDE, &stride);
            T_HANDLEHR(hr);
            hr = output_type->lpVtbl->SetUINT32(
                output_type, &MF_MT_DEFAULT_STRIDE, -stride);
            T_HANDLEHR(hr);
            hr = self->transform->lpVtbl->SetOutputType(self->transform, 0,
                                                        output_type, 0);
            T_HANDLEHR(hr);
            self->last_vflip = self->vflip;
        }

        hr = reader->lpVtbl->ReadSample(reader, FIRST_VIDEO, 0, 0,
                                        &pdwStreamFlags, NULL, &sample);
        T_HANDLEHR(hr);

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
            T_HANDLEHR(hr);

            MFT_OUTPUT_DATA_BUFFER mft_buffer[1];
            MFT_OUTPUT_DATA_BUFFER x;

            IMFSample *ns;
            hr = MFCreateSample(&ns);
            T_HANDLEHR(hr);

            hr = ns->lpVtbl->AddBuffer(ns, self->buf);
            T_HANDLEHR(hr);

            x.pSample = ns;
            x.dwStreamID = 0;
            x.dwStatus = 0;
            x.pEvents = NULL;
            mft_buffer[0] = x;

            DWORD out;
            hr = self->transform->lpVtbl->ProcessOutput(self->transform, 0, 1,
                                                        mft_buffer, &out);
            T_HANDLEHR(hr);

            self->buffer_ready = 1;
        }

        RELEASE(sample);
    }

    /* printf("exiting 2nd thread...\n"); */
    ExitThread(0);
}

int
windows_open_device(pgCameraObject *self)
{
    IMFMediaSource *source = NULL;
    IMFSourceReader *reader = NULL;
    IMFVideoProcessorControl *control = NULL;
    IMFTransform *transform = NULL;
    IMFMediaType *source_type = NULL;
    IMFMediaType *conv_type = NULL;
    MFT_OUTPUT_STREAM_INFO info;
    HRESULT hr;

    IMFActivate *act = windows_device_from_name(self->device_name);

    hr = act->lpVtbl->ActivateObject(act, &IID_IMFMediaSource, &source);
    RELEASE(act);
    HANDLEHR(hr);

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
    self->reader = reader;
    RELEASE(source);
    HANDLEHR(hr);

    if (!_select_source_type(self, &source_type)) {
        return 0;
    }

    hr = reader->lpVtbl->SetCurrentMediaType(reader, FIRST_VIDEO, NULL,
                                             source_type);
    HANDLEHR(hr);

    hr = CoCreateInstance(&CLSID_VideoProcessorMFT, NULL, CLSCTX_INPROC_SERVER,
                          &IID_IMFTransform, &transform);
    self->transform = transform;
    HANDLEHR(hr);

    hr = transform->lpVtbl->QueryInterface(
        transform, &IID_IMFVideoProcessorControl, &control);
    self->control = control;
    HANDLEHR(hr);

    hr = transform->lpVtbl->SetInputType(transform, 0, source_type, 0);
    HANDLEHR(hr);

    if (!_select_conversion_type(self, &conv_type)) {
        return 0;
    }

    hr = transform->lpVtbl->SetOutputType(transform, 0, conv_type, 0);
    HANDLEHR(hr);

    hr = self->transform->lpVtbl->GetOutputStreamInfo(self->transform, 0,
                                                      &info);
    HANDLEHR(hr);

    hr = MFCreateMemoryBuffer(info.cbSize, &self->buf);
    HANDLEHR(hr);

    self->t_handle = CreateThread(NULL, 0, update_function, self, 0, NULL);

    self->open = 1; /* set here, since this shouldn't happen on error */
    self->t_error = S_OK;
    self->t_error_line = 0;
    return 1;

cleanup:
    windows_close_device(self);
    return 0;
}

int
windows_close_device(pgCameraObject *self)
{
    self->open = 0;
    if (self->t_handle) {
        WaitForSingleObject(self->t_handle, 3000);
    }

    RELEASE(self->reader);
    RELEASE(self->transform);
    RELEASE(self->control);
    RELEASE(self->buf);
    RELEASE(self->raw_buf);

    return 1;
}

int
windows_init_device(pgCameraObject *self)
{
    HRESULT hr;

    /* I wanted to use COINIT_MULTITHREADED, but something inside SDL started
     * by display.set_mode() uses COINIT_APARTMENTTHREADED, and you can't have
     * a thread in two modes at once */
    hr = CoInitializeEx(0, COINIT_APARTMENTTHREADED);
    CHECKHR(hr);

    hr = MFStartup(MF_VERSION, MFSTARTUP_LITE);
    CHECKHR(hr);

    return 1;
}

void
windows_dealloc_device(pgCameraObject *self)
{
    PyMem_Free(self->device_name);

    MFShutdown();
    CoUninitialize();
}

int
windows_process_image(pgCameraObject *self, BYTE *data, DWORD length,
                      SDL_Surface *surf)
{
    SDL_LockSurface(surf);

    int size = self->width * self->height;

    /* apparently RGB32 is actually BGR to Microsoft */
    if (self->pixelformat == MFVideoFormat_RGB32.Data1) {
        switch (self->color_out) {
            case RGB_OUT:
                /* optimized version for 32 bit output surfaces*/
                /* memcpy(surf->pixels, data, length); */

                bgr32_to_rgb(data, surf->pixels, size, surf->format);
                break;
            case YUV_OUT:
                rgb_to_yuv(data, surf->pixels, size, V4L2_PIX_FMT_XBGR32,
                           surf->format);
                break;
            case HSV_OUT:
                rgb_to_hsv(data, surf->pixels, size, V4L2_PIX_FMT_XBGR32,
                           surf->format);
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
                rgb_to_hsv(surf->pixels, surf->pixels, size, V4L2_PIX_FMT_YUYV,
                           surf->format);
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

    if (!self->open) {
        PyErr_SetString(pgExc_SDLError,
                        "Camera needs to be started to read data");
        return 0;
    }

    if (!_check_integrity(self)) {
        return 0;
    };

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
windows_frame_ready(pgCameraObject *self, int *result)
{
    *result = self->buffer_ready;

    if (!self->open) {
        PyErr_SetString(pgExc_SDLError,
                        "Camera needs to be started to read data");
        return 0;
    }

    if (!_check_integrity(self)) {
        return 0;
    };

    return 1;
}

PyObject *
windows_read_raw(pgCameraObject *self)
{
    PyObject *data = NULL;
    HRESULT hr;

    if (!self->open) {
        PyErr_SetString(pgExc_SDLError,
                        "Camera needs to be started to read data");
        return 0;
    }

    if (!_check_integrity(self)) {
        return 0;
    };

    if (self->raw_buf) {
        BYTE *buf_data;
        DWORD buf_max_length;
        DWORD buf_length;
        hr = self->raw_buf->lpVtbl->Lock(self->raw_buf, &buf_data,
                                         &buf_max_length, &buf_length);
        CHECKHR(hr);

        data = PyBytes_FromStringAndSize(buf_data, buf_length);
        if (!data) {
            PyErr_SetString(pgExc_SDLError,
                            "Error constructing bytes from data");
            return 0;
        }

        hr = self->raw_buf->lpVtbl->Unlock(self->raw_buf);
        CHECKHR(hr);

        self->buffer_ready = 0;

        return data;
    }

    /* should this be an error instead? */
    Py_RETURN_NONE;
}
