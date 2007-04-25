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

static HWND SDL_Window;
#define MAX_CHUNK_SIZE INT_MAX

static UINT _format_TEXT;

/**
 * \brief Converts the passed type into a system specific type to use
 *        for the clipboard.
 *
 * \param type The type to convert.
 * \return A system specific type.
 */
static UINT
_convert_format (char *type)
{
    return RegisterClipboardFormat (type);
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
    
    _format_MIME_PLAIN = RegisterClipboardFormat ("text/plain");
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

    format = _convert_format (type);

    if (!OpenClipboard (SDL_Window))
        return 0; /* Could not open the clipboard. */
    
    hMem = GlobalAlloc ((GMEM_MOVEABLE | GMEM_DDESHARE), nulledlen);
    if (hMem)
    {
        char *dst = GlobalLock (hMem);

        memset (dst, 0, nulledlen);
        memcpy (dst, src, srclen);

        GlobalUnlock (hMem);
        EmptyClipboard ();
        SetClipboardData (format, hMem);
        
        if (format == _format_MIME_PLAIN) 
        {
            /* Setting SCRAP_TEXT, also set CF_TEXT. */
            SetClipboardData (CF_TEXT, hMem);
            PyDict_SetItemString (_clipdata, "TEXT", PyString_FromString (src));
        }
        CloseClipboard ();
    }
    else
    {
        /* Could not access the clipboard, raise an error. */
        CloseClipboard ();
        return 0;
    }

    return 1;
}

char*
pygame_scrap_get (char *type)
{
    UINT format = _convert_format (type);
    char *retval = NULL;

    if (!pygame_scrap_initialized ())
    {
        PyErr_SetString (PyExc_SDLError, "scrap system not initialized.");
        return 0;
    }

    if (!pygame_scrap_lost ())
        return PyString_AsString (PyDict_GetItemString (_clipdata, type));

    if (IsClipboardFormatAvailable (format) && OpenClipboard (SDL_Window))
    {
        HANDLE hMem;
        char *src;
        
        hMem = GetClipboardData (format);
        if (hMem)
        {
            int len = 0;
            
            /* TODO: Is there any mechanism to detect the amount of bytes
             * in the HANDLE? strlen() won't work as supposed, if the
             * sequence contains NUL bytes. Can this even happen in the 
             * Win32 clipboard or is NUL the usual delimiter?
             */
            src = GlobalLock (hMem);
            len = strlen (src) + 1;
            
            retval = malloc (len);
            if (retval)
            {
                memset (retval, 0, len);
                memcpy (retval, src, len);
            }
            GlobalUnlock (hMem);
        }
        CloseClipboard ();
    }
    
    return retval;
}

char**
pygame_scrap_get_types (void)
{
    UINT format = 0;
    char **types;
    int i = 0;
    int len;
    char tmp[100] = { NULL };
    int size =  CountClipboardFormats ();

    if (size == 0)
        return NULL; /* No clipboard data. */

    types = malloc (sizeof (char *) * (size + 1));
    for (i = 0; i < size; i++)
    {
        format = EnumClipboardFormats (format);
        if (format == 0)
        {
            /* Something wicked happened. */
            while (i > 0)
                free (types[i]);
            free (types);
            return NULL;
            break;
        }

        /* No predefined name, get the (truncated) name. */
        memset (tmp, 0, sizeof (tmp));
        len = GetClipboardFormatName (format, tmp, sizeof (tmp));
        types[i] = malloc (sizeof (char) * (len + 1));
        if (!types[i])
        {
            while (i > 0)
                free (types[i]);
            free (types);
            return NULL;
        }
        memset (types[i], 0, len + 1);
        memcpy (types[i], tmp, len);
    }
    types[size] = NULL;
    return NULL;
}

int
pygame_scrap_contains (char *type)
{
    return IsClipboardFormatAvailable (format);
}
