/*
    pygame - Python Game Library
    Copyright (C) 2006, 2007 Rene Dudfield, Marcus von Appen

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
#include <SDL.h>
#include <SDL_syswm.h>

#if defined(SDL_VIDEO_DRIVER_WINDIB) || defined(SDL_VIDEO_DRIVER_DDRAW) || defined(SDL_VIDEO_DRIVER_GAPI)

#include <windows.h>
#include "scrap_win.h"

#if !defined(CF_DIBV5)
/* Missing from the MinGW win32api-3.11 winuser.h header */
#define CF_DIBV5 17
#endif

static HWND _sdlwindow;
#define MAX_CHUNK_SIZE INT_MAX
static UINT _format_MIME_PLAIN;

static UINT _convert_format (char *type);
static UINT _convert_internal_type (char *type);
static int _lookup_clipboard_format (UINT format, char *buf, int size);
static char* _create_dib_buffer (char* data, unsigned int *count);


/**
 * \brief Converts the passed type into a system specific clipboard type
 *        to use for the clipboard.
 *
 * \param type The type to convert.
 * \return A system specific type.
 */
static UINT
_convert_format (char *type)
{
    return RegisterClipboardFormat (type);
}

/**
 * \brief Gets a system specific clipboard format type for a certain type.
 *
 * \param type The name of the format to get the mapped format type for.
 * \return The format type or -1 if no such type was found.
 */
static UINT
_convert_internal_type (char *type)
{
    if (strcmp (type, SCRAP_FORMAT_TEXT) == 0)
        return CF_TEXT;
    if (strcmp (type, "text/plain;charset=utf-8") == 0)
        return CF_UNICODETEXT;
    if (strcmp (type, "image/tiff") == 0)
        return CF_TIFF;
    if (strcmp (type, SCRAP_FORMAT_BMP) == 0)
        return CF_DIB;
    if (strcmp (type, "audio/wav") == 0)
        return CF_WAVE;
    return -1;
}

/**
 * \brief Looks up the name for the specific clipboard format type.
 *
 * \param format The format to get the name for.
 * \param buf The buffer to copy the name into.
 * \param size The size of the buffer.
 * \return The length of the format name.
 */
static int
_lookup_clipboard_format (UINT format, char *buf, int size)
{
    int len;
    char *cpy;

    memset (buf, 0, size);
    switch (format)
    {
    case CF_TEXT:
        len = strlen (SCRAP_FORMAT_TEXT);
        cpy = SCRAP_FORMAT_TEXT;
        break;
    case CF_UNICODETEXT:
        len = 24;
        cpy = "text/plain;charset=utf-8";
        break;
    case CF_TIFF:
        len = 10;
        cpy = "image/tiff";
        break;
    case CF_DIB:
        len = strlen (SCRAP_FORMAT_BMP);
        cpy = SCRAP_FORMAT_BMP;
        break;
    case CF_WAVE:
        len = 9;
        cpy = "audio/wav";
        break;
    default:
        len = GetClipboardFormatName (format, buf, size);
        return len;
    }
    if (len != 0)
        memcpy (buf, cpy, len);
    return len;
}

/**
 * \brief Creates a BMP character buffer with all headers from a DIB
 *        HANDLE. The caller has to free the returned buffer.
 * \param data The DIB handle data.
 * \param count The size of the DIB handle.
 * \return The character buffer containing the BMP information.
 */
static char*
_create_dib_buffer (char* data, unsigned int *count)
{
    BITMAPFILEHEADER hdr;
    LPBITMAPINFOHEADER bihdr;
    char *buf;

    if (!data)
    {
        SDL_SetError ("data argument NULL");
        return NULL;
    }
    bihdr = (LPBITMAPINFOHEADER) data;

    /* Create the BMP header. */
    hdr.bfType = 'M' << 8 | 'B'; /* Specs say, it is always BM */
	hdr.bfReserved1 = 0;
	hdr.bfReserved2 = 0;
	hdr.bfSize = (DWORD) (sizeof (BITMAPFILEHEADER) + bihdr->biSize
                             + bihdr->biClrUsed * sizeof (RGBQUAD)
                             + bihdr->biSizeImage);
    hdr.bfOffBits = (DWORD) (sizeof (BITMAPFILEHEADER) + bihdr->biSize
                             + bihdr->biClrUsed * sizeof (RGBQUAD));

    /* Copy both to the buffer. */
    buf = malloc (sizeof (hdr) + (*count));
    if (!buf)
    {
        SDL_SetError ("could not allocate memory for image buffer");
        return NULL;
    }
    memcpy (buf, &hdr, sizeof (hdr));
    memcpy (buf + sizeof (BITMAPFILEHEADER), data, *count);

    /* Increase count for the correct size. */
    *count += sizeof (hdr);
    return buf;
}

int
scrap_init_win (void)
{
    SDL_SysWMinfo info;

    /* Grab the window manager specific information */
    SDL_VERSION (&info.version);
    if (SDL_GetWMInfo (&info))
    {
        /* Save the information for later use */
        _sdlwindow = info.window;
        _format_MIME_PLAIN = RegisterClipboardFormat (SCRAP_FORMAT_TEXT);
        return 1;
    }
    SDL_SetError ("SDL is not running on Windows");
    return 0;
}

void
scrap_quit_win (void)
{
    _sdlwindow = NULL;
    _format_MIME_PLAIN = 0;
}

int
scrap_contains_win (char *type)
{
    return IsClipboardFormatAvailable (_convert_format (type));
}

int
scrap_lost_win (void)
{
    return (GetClipboardOwner () != _sdlwindow);
}

ScrapType
scrap_get_mode_win (void)
{
    /* Only clipboard mode is supported. */
    return SCRAP_CLIPBOARD;
}

ScrapType
scrap_set_mode_win (ScrapType mode)
{
    /* Only clipboard mode is supported. */
    return SCRAP_CLIPBOARD;
}

int
scrap_get_win (char *type, char **data, unsigned int *size)
{
   UINT format = _convert_format (type);

    if (!OpenClipboard (_sdlwindow))
    {
        SDL_SetError ("could not access clipboard");
        return -1;
    }
    
    if (!IsClipboardFormatAvailable (format))
    {
        /* The format was not found - was it a mapped type? */
        format = _convert_internal_type (type);
        if (format == (UINT)-1)
        {
            CloseClipboard ();
            SDL_SetError ("no matching format on clipboard found");
            return -1;
        }
    }

    if (IsClipboardFormatAvailable (format))
    {
        HANDLE hMem;
        char *src = NULL;
        int retval = 0;
        
        hMem = GetClipboardData (format);
        if (hMem)
        {
            *size = 0;

            /* CF_BITMAP is not a global, so do not lock it. */
            if (format != CF_BITMAP)
            {
                src = GlobalLock (hMem);
                if (!src)
                {
                    CloseClipboard ();
                    SDL_SetError ("could not acquire the memory pointer");
                    return -1;
                }
                *size = GlobalSize (hMem);
            }
            
            if (format == CF_DIB || format == CF_DIBV5)
            {
                /* size will be increased accordingly in
                 * _create_dib_buffer.
                 */
                *data = _create_dib_buffer (src, size);
                retval = 1;
            }
            else if (*size != 0)
            {
                *data = malloc (*size);
                if (*data)
                {
                    memset (*data, 0, *size);
                    memcpy (*data, src, *size);
                    retval = 1;
                }
                else
                {
                    SDL_SetError ("could not allocate memory");
                    retval = -1;
                }
            }
            GlobalUnlock (hMem);
            CloseClipboard ();
            return retval;
        }
    }

    CloseClipboard ();
    return 0;
}

int
scrap_put_win (char *type, char *data, unsigned int size)
{
    UINT format;
    int nulledlen = size + 1;
    HANDLE hMem;

    format = _convert_internal_type (type);
    if (format == (UINT)-1)
        format = _convert_format (type);

    if (!OpenClipboard (_sdlwindow))
    {
        SDL_SetError ("could not access clipboard");
        return -1; /* Could not open the clipboard. */
    }
    if (format == CF_DIB || format == CF_DIBV5)
        nulledlen -= sizeof (BITMAPFILEHEADER); /* We won't copy the header */
    
    hMem = GlobalAlloc ((GMEM_MOVEABLE | GMEM_DDESHARE), nulledlen);
    if (hMem)
    {
        char *dst = GlobalLock (hMem);

        memset (dst, 0, nulledlen);
        if (format == CF_DIB || format == CF_DIBV5)
            memcpy (dst, data + sizeof (BITMAPFILEHEADER), nulledlen - 1);
        else
            memcpy (dst, data, size);

        GlobalUnlock (hMem);
        EmptyClipboard ();
        SetClipboardData (format, hMem);
        
        if (format == _format_MIME_PLAIN) 
        {
            /* Setting SCRAP_TEXT, also set CF_TEXT. */
            SetClipboardData (CF_TEXT, hMem);
        }
    }
    else
    {
        /* Could not access the clipboard, raise an error. */
        SDL_SetError ("could not acquire the memory pointer");
        CloseClipboard ();
        return -1;
    }
    
    CloseClipboard ();
    return 1;
}

int
scrap_get_types_win (char** types)
{
    UINT format = 0;
    char **tmptypes;
    int count = -1;
    int i, len, size;
    char tmp[100] = { '\0' };

    if (!OpenClipboard (_sdlwindow))
    {
        SDL_SetError ("could not access clipboard");
        return -1;
    }

    size = CountClipboardFormats ();
    if (size == 0)
    {
        CloseClipboard ();
        return 0; /* No clipboard data. */
    }

    for (i = 0; i < size; i++)
    {
        format = EnumClipboardFormats (format);
        if (format == 0)
        {
            /* Something wicked happened. */
            while (i > 0)
                free (types[i]);
            free (types);
            CloseClipboard ();
            SDL_SetError ("error on retrieving the formats");
            return -1;
        }

        /* No predefined name, get the (truncated) name. */
        len = _lookup_clipboard_format (format, tmp, 100);
        if (len == 0)
            continue;
        count++;

        tmptypes = realloc (types, sizeof (char *) * (count + 1));
        if (!tmptypes)
        {
            while (count > 0)
            {
                free (types[count]);
                count--;
            }
            free (types);
            CloseClipboard ();
            SDL_SetError ("could allocate memory");
            return -1;
        }
        types = tmptypes;
        types[count] = malloc (sizeof (char) * (len + 1));
        if (!types[count])
        {
            while (count > 0)
            {
                free (types[count]);
                count--;
            }
            free (types);
            CloseClipboard ();
            SDL_SetError ("could allocate memory");
            return -1;
        }

        memset (types[count], 0, len + 1);
        memcpy (types[count], tmp, len);
    }

    tmptypes = realloc (types, sizeof (char *) * (count + 1));
    if (!tmptypes)
    {
        while (count > 0)
        {
            free (types[count]);
            count--;
        }
        free (types);
        CloseClipboard ();
        SDL_SetError ("could allocate memory");
        return -1;
    }
    types = tmptypes;
    types[count] = NULL;
    CloseClipboard ();
    return 1;
}

#endif /* SDL_VIDEO_DRIVER_WINDIB || SDL_VIDEO_DRIVER_DDRAW || SDL_VIDEO_DRIVER_GAPI */
