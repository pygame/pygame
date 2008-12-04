/*
    pygame - Python Game Library
    Copyright (C) 2006, 2007 Rene Dudfield, Marcus von Appen

    Originally written and put in the public domain by Sam Lantinga.

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
#include <Windows.h>

#if !defined(CF_DIBV5)
/* Missing from the MinGW win32api-3.11 winuser.h header */
#define CF_DIBV5 17
#endif

static HWND SDL_Window;
#define MAX_CHUNK_SIZE INT_MAX

static UINT _format_MIME_PLAIN;

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
    if (strcmp (type, PYGAME_SCRAP_TEXT) == 0)
        return CF_TEXT;
    if (strcmp (type, "text/plain;charset=utf-8") == 0)
        return CF_UNICODETEXT;
    if (strcmp (type, "image/tiff") == 0)
        return CF_TIFF;
    if (strcmp (type, PYGAME_SCRAP_BMP) == 0)
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
        len = strlen (PYGAME_SCRAP_TEXT);
        cpy = PYGAME_SCRAP_TEXT;
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
        len = strlen (PYGAME_SCRAP_BMP);
        cpy = PYGAME_SCRAP_BMP;
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
_create_dib_buffer (char* data, unsigned long *count)
{
    BITMAPFILEHEADER hdr;
    LPBITMAPINFOHEADER bihdr;
    char *buf;

    if (!data)
        return NULL;
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
        return NULL;
    memcpy (buf, &hdr, sizeof (hdr));
    memcpy (buf + sizeof (BITMAPFILEHEADER), data, *count);

    /* Increase count for the correct size. */
    *count += sizeof (hdr);
    return buf;
}

int
pygame_scrap_init (void)
{
    SDL_SysWMinfo info;
    int retval = 0;

    /* Grab the window manager specific information */
    SDL_SetError ("SDL is not running on known window manager");

    SDL_VERSION (&info.version);
    if (SDL_GetWMInfo (&info))
    {
        /* Save the information for later use */
        SDL_Window = info.window;
        retval = 1;
    }
    if (retval)
        _scrapinitialized = 1;
    
    _format_MIME_PLAIN = RegisterClipboardFormat (PYGAME_SCRAP_TEXT);
    return retval;
}

int
pygame_scrap_lost (void)
{
    if (!pygame_scrap_initialized ())
    {
        PyErr_SetString (PyExc_SDLError, "scrap system not initialized.");
        return 0;
    }
    return (GetClipboardOwner () != SDL_Window);
}

int
pygame_scrap_put (char *type, int srclen, char *src)
{
    UINT format;
    int nulledlen = srclen + 1;
    HANDLE hMem;

    if (!pygame_scrap_initialized ())
    {
        PyErr_SetString (PyExc_SDLError, "scrap system not initialized.");
        return 0;
    }

    format = _convert_internal_type (type);
    if (format == -1)
        format = _convert_format (type);

    if (!OpenClipboard (SDL_Window))
        return 0; /* Could not open the clipboard. */
    
    if (format == CF_DIB || format == CF_DIBV5)
        nulledlen -= sizeof (BITMAPFILEHEADER); /* We won't copy the header */
    
    hMem = GlobalAlloc ((GMEM_MOVEABLE | GMEM_DDESHARE), nulledlen);
    if (hMem)
    {
        char *dst = GlobalLock (hMem);

        memset (dst, 0, nulledlen);
        if (format == CF_DIB || format == CF_DIBV5)
            memcpy (dst, src + sizeof (BITMAPFILEHEADER), nulledlen - 1);
        else
            memcpy (dst, src, srclen);

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
        CloseClipboard ();
        return 0;
    }
    
    CloseClipboard ();
    return 1;
}

char*
pygame_scrap_get (char *type, unsigned long *count)
{
    UINT format = _convert_format (type);
    char *retval = NULL;

    if (!pygame_scrap_initialized ())
    {
        PyErr_SetString (PyExc_SDLError, "scrap system not initialized.");
        return NULL;
    }

    if (!pygame_scrap_lost ())
        return PyString_AsString (PyDict_GetItemString (_clipdata, type));

    if (!OpenClipboard (SDL_Window))
        return NULL;

    if (!IsClipboardFormatAvailable (format))
    {
        /* The format was not found - was it a mapped type? */
        format = _convert_internal_type (type);
        if (format == -1)
        {
            CloseClipboard ();
            return NULL;
        }
    }

    if (IsClipboardFormatAvailable (format))
    {
        HANDLE hMem;
        char *src;
        src = NULL;

        hMem = GetClipboardData (format);
        if (hMem)
        {
            *count = 0;

            /* CF_BITMAP is not a global, so do not lock it. */
            if (format != CF_BITMAP)
            {
                src = GlobalLock (hMem);
                if (!src)
                {
                    CloseClipboard ();
                    return NULL;
                }
                *count = GlobalSize (hMem);
            }

            if (format == CF_DIB || format == CF_DIBV5)
            {
                /* Count will be increased accordingly in
                 * _create_dib_buffer.
                 */
                src = _create_dib_buffer (src, count);
                GlobalUnlock (hMem);
                CloseClipboard ();
                return src;
            }
            else if (*count != 0)
            {
                /* weird error, shouldn't get here. */
                if(!src) {
                    return NULL;
                }

                retval = malloc (*count);
                if (retval)
                {
                    memset (retval, 0, *count);
                    memcpy (retval, src, *count);
                }
            }
            GlobalUnlock (hMem);
        }
    }

    CloseClipboard ();
    return retval;
}

char**
pygame_scrap_get_types (void)
{
    UINT format = 0;
    char **types = NULL;
    char **tmptypes;
    int i = 0;
    int count = -1;
    int len;
    char tmp[100] = { '\0' };
    int size = 0;

    if (!OpenClipboard (SDL_Window))
        return NULL;

    size = CountClipboardFormats ();
    if (size == 0)
    {
        CloseClipboard ();
        return NULL; /* No clipboard data. */
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
            return NULL;
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
            return NULL;
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
            return NULL;
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
        return NULL;
    }
    types = tmptypes;
    types[count] = NULL;
    CloseClipboard ();
    return types;
}

int
pygame_scrap_contains (char *type)
{
    return IsClipboardFormatAvailable (_convert_format(type));
}
