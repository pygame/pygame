
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

/* Needed for deprecation warnings */
#include <Python.h>

static unsigned short InputGroup;
#define MAX_CHUNK_SIZE INT_MAX

/**
 * \brief Converts the passed type into a system specific type to use
 *        for the clipboard.
 *
 * \param type The type to convert.
 * \return A system specific type.
 */
static uint32_t
_convert_format(char *type)
{
    switch (type) {
        case PYGAME_SCRAP_TEXT:
            return Ph_CL_TEXT;
        default: /* PYGAME_SCRAP_BMP et al. */
        {
            /* TODO */
            return 0;
        }
    }
}

int
pygame_scrap_init(void)
{
    SDL_SysWMinfo info;
    int retval = 0;

    if (PyErr_WarnEx(PyExc_DeprecationWarning,
                    "pygame.scrap.init deprecated since 2.1.4",
                    1) == -1) {
        return NULL;
    }

    /* Grab the window manager specific information */
    SDL_SetError("SDL is not running on known window manager");

    SDL_VERSION(&info.version);
    if (SDL_GetWMInfo(&info)) {
        /* Save the information for later use */
        InputGroup = PhInputGroup(NULL);
        retval = 1;
    }
    if (retval)
        _scrapinitialized = 1;

    return retval;
}

int
pygame_scrap_lost(void)
{
    if (!pygame_scrap_initialized()) {
        PyErr_SetString(pgExc_SDLError, "scrap system not initialized.");
        return 0;
    }

    if (PyErr_WarnEx(PyExc_DeprecationWarning,
                     "pygame.scrap.lost deprecated since 2.1.4",
                     1) == -1) {
        return NULL;
    }

    return (PhInputGroup(NULL) != InputGroup);
}

int
pygame_scrap_put(char *type, Py_ssize_t srclen, char *src)
{
    uint32_t format;
    int nulledlen = srclen + 1;

    if (!pygame_scrap_initialized()) {
        PyErr_SetString(pgExc_SDLError, "scrap system not initialized.");
        return 0;
    }

    if (PyErr_WarnEx(PyExc_DeprecationWarning,
                     "pygame.scrap.put deprecated since 2.1.4. Consider using" 
                     " pygame.scrap.put_text instead.",
                     1) == -1) {
        return NULL;
    }

    format = _convert_format(type);

    /* Clear old buffer and copy the new content. */
    if (_clipbuffer)
        free(_clipbuffer);

    _clipbuffer = malloc(nulledlen);
    if (!_clipbuffer)
        return 0; /* Allocation failed. */
    memset(_clipbuffer, 0, nulledlen);
    memcpy(_clipbuffer, src, srclen);
    _clipsize = srclen;
    _cliptype = format;

#if (_NTO_VERSION < 620) /* Before 6.2.0 releases. */
    {
        PhClipHeader clheader = {Ph_CLIPBOARD_TYPE_TEXT, 0, NULL};
        int *cldata;
        int status;

        cldata = (int *)_clipbuffer;
        *cldata = type;
        clheader.data = _clipbuffer;
        if (dstlen > 65535)
            clheader.length = 65535; /* Maximum photon clipboard size. :( */
        else
            clheader.length = nulledlen;

        status = PhClipboardCopy(InputGroup, 1, &clheader);
        if (status == -1) {
            /* Could not access the clipboard, raise an error. */
            CLEAN_CLIP_BUFFER();
            return 0;
        }
    }

#else /* 6.2.0 and 6.2.1 and future releases. */
    {
        PhClipboardHdr clheader = {Ph_CLIPBOARD_TYPE_TEXT, 0, NULL};
        int *cldata;
        int status;

        cldata = (int *)_clipbuffer;
        *cldata = type;
        clheader.data = _clipbuffer;
        clheader.length = nulledlen;

        status = PhClipboardWrite(InputGroup, 1, &clheader);
        if (status == -1) {
            /* Could not access the clipboard, raise an error. */
            CLEAN_CLIP_BUFFER();
            return 0;
        }
    }
#endif
    return 1;
}

char *
pygame_get_scrap(char *type)
{
    uint32_t format = _convert_format(type);
    char *retval = NULL;

    if (!pygame_scrap_initialized()) {
        PyErr_SetString(pgExc_SDLError, "scrap system not initialized.");
        return 0;
    }

    if (PyErr_WarnEx(PyExc_DeprecationWarning,
                     "pygame.scrap.get deprecated since 2.1.4. Consider using"
                     " pygame.scrap.get_text instead.",
                     1) == -1) {
        return NULL;
    }

    /* If we are the owner, simply return the clip buffer, if it matches
     * the request type. */
    if (!pygame_scrap_lost()) {
        if (format != _cliptype)
            return NULL;

        if (_clipbuffer) {
            retval = malloc(_clipsize + 1);
            if (!retval)
                return NULL;
            memset(retval, 0, _clipsize + 1);
            memcpy(retval, _clipbuffer, _clipsize + 1);
            return retval;
        }
        return NULL;
    }

#if (_NTO_VERSION < 620) /* before 6.2.0 releases */
    {
        void *clhandle;
        PhClipHeader *clheader;
        int *cldata;

        clhandle = PhClipboardPasteStart(InputGroup);

        if (clhandle) {
            clheader = PhClipboardPasteType(clhandle, Ph_CLIPBOARD_TYPE_TEXT);
            if (clheader) {
                cldata = clheader->data;
                if (*cldata == type)
                    retval = malloc(clheader->length + 1);

                if (retval) {
                    memset(retval, 0, clheader->length + 1);
                    memcpy(retval, cldata, clheader->length + 1);
                }
            }
            PhClipboardPasteFinish(clhandle);
        }
    }
#else /* 6.2.0 and 6.2.1 and future releases */
    {
        void *clhandle;
        PhClipboardHdr *clheader;
        int *cldata;

        clheader = PhClipboardRead(InputGroup, Ph_CLIPBOARD_TYPE_TEXT);
        if (clheader) {
            cldata = clheader->data;
            if (*cldata == type)
                retval = malloc(clheader->length + 1);

            if (retval) {
                memset(retval, 0, clheader->length + 1);
                memcpy(retval, cldata, clheader->length + 1);
            }
            /* According to the QNX 6.x docs, the clheader pointer is a
             * newly created one that must be freed manually. */
            free(clheader->data);
            free(clheader);
        }
    }
#endif

    return retval;
}

char **
pygame_scrap_get_types(void)
{
    if (PyErr_WarnEx(PyExc_DeprecationWarning,
                     "pygame.scrap.get_types deprecated since 2.1.4",
                     1) == -1) {
        return NULL;
    }

    return NULL;
}

int
pygame_scrap_contains(char *type)
{
    if (PyErr_WarnEx(PyExc_DeprecationWarning,
                     "pygame.scrap.contains deprecated since 2.1.4",
                     1) == -1) {
        return NULL;
    }
    
    return 0;
}
