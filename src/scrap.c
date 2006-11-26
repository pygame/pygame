/*
    pygame - Python Game Library
    Copyright (C) 2006 Rene Dudfield, Marcus von Appen

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

/* Handle clipboard text and data in arbitrary formats */

#include <stdio.h>
#include <limits.h>

#include "SDL.h"
#include "SDL_syswm.h"

#include "scrap.h"
#include "pygame.h"
#include "pygamedocs.h"


/* Python < 2.3/2.4 backwards compatibility - should be placed in a
 * private header by time. */
#ifndef Py_RETURN_TRUE
#define Py_RETURN_TRUE return Py_INCREF (Py_True), Py_True
#endif

#ifndef Py_RETURN_FALSE
#define Py_RETURN_FALSE return Py_INCREF (Py_False), Py_False
#endif

#ifndef Py_RETURN_NONE
#define Py_RETURN_NONE return Py_INCREF (Py_None), Py_None
#endif

/**
 * Format prefix to use.
 */
#define FORMAT_PREFIX "SDL_scrap_0x"

/* Determine what type of clipboard we are using */
#if defined(__unix__) && !defined(__QNXNTO__) && !defined(DISABLE_X11)
    #define X11_SCRAP
    #include <time.h> /* Needed for clipboard timeouts. */
#elif defined(__WIN32__)
    #define WIN_SCRAP
#elif defined(__QNXNTO__)
    #define QNX_SCRAP
#elif defined(__APPLE__)
    #define MAC_SCRAP
#else
    #error Unknown window manager for clipboard handling
#endif /* scrap type */

/* MAC_SCRAP delegates all functionality, we just need a small stub. */
#if defined(MAC_SCRAP)

static PyObject*
mac_scrap_call (char *name, PyObject *args)
{
    static PyObject *mac_scrap_module = NULL;
    PyObject *method;
    PyObject *result;

    if (!mac_scrap_module)
        mac_scrap_module = PyImport_ImportModule ("pygame.mac_scrap");
    if (!mac_scrap_module)
        return NULL;
    
    method = PyObject_GetAttrString (mac_scrap_module, name);
    if (!method)
        return NULL;
    result = PyObject_CallObject (method, args);
    Py_DECREF (method);
    return result;
}

static PyObject*
scrap_init (PyObject *self, PyObject *args)
{
    return mac_scrap_call ("init", args);
}

static PyObject*
scrap_get_scrap (PyObject *self, PyObject *args)
{
    return mac_scrap_call ("get", args);
}

static PyObject*
scrap_put_scrap (PyObject *self, PyObject *args)
{
    return mac_scrap_call ("put", args);
}

static PyObject*
scrap_lost_scrap (PyObject *self, PyObject *args)
{
    return mac_scrap_call ("lost", args);
}

#else /* defined(MAC_SCRAP) */

/**
 * Indicates, whether pygame.scrap was initialized or not.
 */
static int _scrapinitialized = 0;

/**
 * The internal clipboard buffer.
 */
static char *_clipbuffer = NULL;
static int _clipsize = 0;
static int _cliptype = 0;

/**
 * Clip buffer cleaning macro.
 */
#define CLEAN_CLIP_BUFFER() \
    { if (_clipbuffer) \
            free (_clipbuffer); \
        _clipbuffer = NULL; \
        _clipsize = 0; \
        _cliptype = 0; \
    }

/* System dependent data types and variables. */
#if defined(X11_SCRAP)
typedef Atom scrap_type;
static Display *SDL_Display;
static Window SDL_Window;
static void (*Lock_Display)(void);
static void (*Unlock_Display)(void);

#elif defined(WIN_SCRAP)
typedef UINT scrap_type;
static HWND SDL_Window;

#elif defined(QNX_SCRAP)
typedef uint32_t scrap_type;
static unsigned short InputGroup;

#endif /* types */

/**
 * \brief Converts the passed type into a system specific scrap_type to
 *        use for the clipboard.
 *
 * \param type The type to convert.
 * \return A system specific scrap_type.
 */
static scrap_type
_convert_format (int type)
{
  switch (type)
    {
    case PYGAME_SCRAP_TEXT:
#if defined(X11_SCRAP)
        return XA_STRING;
#elif defined(WIN_SCRAP)
        return CF_TEXT;
#elif defined(QNX_SCRAP)
        return Ph_CL_TEXT;
#endif

    default: /* PYGAME_SCRAP_BMP et al. */
    {
        char format[sizeof (FORMAT_PREFIX) + 8 + 1];
        sprintf (format, "%s%08lx", FORMAT_PREFIX, (unsigned long) type);

#if defined(X11_SCRAP)
        return XInternAtom (SDL_Display, format, False);
#elif defined(WIN_SCRAP)
        return RegisterClipboardFormat (format);
#endif
    }
    }
}

/**
 * X11 specific methods we need here.
 */
#if defined(X11_SCRAP)

/**
 * \brief System message filter function -- handles X11 clipboard messages.
 *
 * \param event The SDL_Event to check.
 * \return Always 1.
 */
static int
_clipboard_filter (const SDL_Event *event)
{
    /* Post all non-window manager specific events */
    if (event->type != SDL_SYSWMEVENT)
        return 1;

    /* Handle window-manager specific clipboard events */
    switch (event->syswm.msg->event.xevent.type)
    {
    case SelectionClear:
        /* Looks like another window takes control over the clipboard.
         * Release the internally saved buffer. */
        CLEAN_CLIP_BUFFER ();
        break;

    case SelectionNotify:
        /* This one will be handled directly in the pygame_get_scrap ()
         * function. */
        break;

    case SelectionRequest:
    {
        Atom request;
        int found = 0;
        /*  unused?
        int seln_format;
        unsigned long nbytes;
        unsigned long overflow;
        unsigned char *seln_data;
        */
        XSelectionRequestEvent *req =
            &event->syswm.msg->event.xevent.xselectionrequest;
        XEvent ev;

        if (!_clipbuffer)
            return 1;
        
        request = XInternAtom (SDL_Display, "UTF8_STRING", False);
        if (req->target != request)
        {
            request = XInternAtom (SDL_Display, "TEXT", False);
            if (req->target != request)
            {
                request = XInternAtom (SDL_Display, "COMPOUND_TEXT", False);
                if (req->target != request)
                {
                    if (req->target == XA_STRING)
                    {
                        request = XA_STRING;
                        found = 1;
                    }
                }
                else
                    found = 1; /* Want COMPOUND_TEXT. */
            }
            else
                found = 1; /* Want TEXT. */
        }
        else
            found = 1; /* Want UTF8_STRING. */


        /* No text requested. Try SCRAP_BMP for another pygame window,
         * that might request the data. */
        if (!found)
        {
            request = _convert_format (PYGAME_SCRAP_BMP);
            found = req->target == request;
        }
        
        if (found)
            XChangeProperty (req->display, req->requestor, req->property,
                             request, 8, PropModeReplace, (unsigned char *)_clipbuffer,
                             strlen (_clipbuffer));

        /* Prepare answer. */
        ev.xselection.type = SelectionNotify;
        ev.xselection.property = req->property;
        ev.xselection.display = req->display;
        ev.xselection.requestor = req->requestor;
        ev.xselection.selection = req->selection;
        ev.xselection.target = req->target;
        ev.xselection.time = req->time;

        XSendEvent (req->display, req->requestor, False, 0, &ev);
        break;
    }
    }

    /* Post the event for X11 clipboard reading above */
    return 1;
}

/**
 * \brief Tries to determine the X window with a valid selection.
 *        Default is to check
 *         - passed parameter
 *         - XA_PRIMARY
 *         - XA_SECONDARY
 *         - XA_CUT_BUFFER0
 *        
 *         in this order.
 * 
 * \param selection The Atom type, that should be tried before any of the
 *                  fixed XA_* buffers.
 * \return The Window handle, that owns the selection or None if none was
 *         found. 
 */
static Window
_get_scrap_owner (Atom *selection)
{
    int i = 0;
    static Atom buffers[] = { XA_PRIMARY, XA_SECONDARY, XA_CUT_BUFFER0,
                              XA_CUT_BUFFER1, XA_CUT_BUFFER2, XA_CUT_BUFFER3,
                              XA_CUT_BUFFER4, XA_CUT_BUFFER5, XA_CUT_BUFFER6,
                              XA_CUT_BUFFER7 };

    Window owner = XGetSelectionOwner (SDL_Display, *selection);
    if (owner != None)
        return owner;

    while (i < 10)
    {
        owner = XGetSelectionOwner (SDL_Display, buffers[i]);
        if (owner != None)
        {
            *selection = buffers[i];
            return owner;
        }
        i++;
    }
    return None;
}

#endif /* X11_SCRAP */

int
pygame_scrap_initialized (void)
{
    return _scrapinitialized;
}

int
pygame_init_scrap (void)
{
    SDL_SysWMinfo info;
    int retval = 0;

    /* Grab the window manager specific information */
    SDL_SetError ("SDL is not running on known window manager");

    SDL_VERSION (&info.version);
    if (SDL_GetWMInfo (&info))
    {
        /* Save the information for later use */
#if defined(X11_SCRAP)
        if (info.subsystem == SDL_SYSWM_X11)
        {
            SDL_Display = info.info.x11.display;
            SDL_Window = info.info.x11.window;
            Lock_Display = info.info.x11.lock_func;
            Unlock_Display = info.info.x11.unlock_func;
          
            /* Enable the special window hook events */
            SDL_EventState (SDL_SYSWMEVENT, SDL_ENABLE);
            SDL_SetEventFilter (_clipboard_filter);

            retval = 1;
        }
        else
            SDL_SetError ("SDL is not running on X11");

#elif defined(WIN_SCRAP)
        SDL_Window = info.window;
        retval = 1;

#elif defined(QNX_SCRAP)
        InputGroup = PhInputGroup (NULL);
        retval = 1;

#endif /* scrap type */
    }
    if (retval)
        _scrapinitialized = 1;

    return retval;
}

int
pygame_lost_scrap (void)
{
    int retval;

    if (!pygame_scrap_initialized ())
    {
        PyErr_SetString (PyExc_SDLError, "scrap system not initialized.");
        return 0;
    }

#if defined(X11_SCRAP)
    Lock_Display ();
    retval = (XGetSelectionOwner (SDL_Display, XA_PRIMARY) != SDL_Window);
    Unlock_Display ();

#elif defined(WIN_SCRAP)
    retval = (GetClipboardOwner () != SDL_Window);

#elif defined(QNX_SCRAP)
    retval = (PhInputGroup (NULL) != InputGroup);

#endif /* scrap type */

  return retval;
}

int
pygame_put_scrap (int type, int srclen, char *src)
{
    scrap_type format;
    int nulledlen = srclen + 1;
#if defined(WIN_SCRAP)
    HANDLE hMem;
#endif



    if (!pygame_scrap_initialized ())
    {
        PyErr_SetString (PyExc_SDLError, "scrap system not initialized.");
        return 0;
    }

    format = _convert_format (type);

    /* Clear old buffer and copy the new content. */
    if (_clipbuffer)
        free (_clipbuffer);

    _clipbuffer = malloc (nulledlen);
    if (!_clipbuffer)
        return 0; /* Allocation failed. */
    memset (_clipbuffer, 0, nulledlen);
    memcpy (_clipbuffer, src, srclen);
    _clipsize = srclen;
    _cliptype = format;

#if defined(X11_SCRAP)
    Lock_Display ();

    /* Update the clipboard property with the buffer. */
    XChangeProperty (SDL_Display, SDL_Window, XA_PRIMARY, format, 8,
                     PropModeReplace, (unsigned char *)_clipbuffer, srclen);

  /* Set the selection owner to the own window. */
    XSetSelectionOwner (SDL_Display, XA_PRIMARY, SDL_Window, CurrentTime);
    if (XGetSelectionOwner (SDL_Display, XA_PRIMARY) != SDL_Window)
    {
        /* Ouch, we could not toggle the selection owner. Raise an error,
         * as it's not guaranteed, that the clipboard contains valid
         * data. */
        CLEAN_CLIP_BUFFER ();
        Unlock_Display ();
        return 0;
    }

    Unlock_Display ();

#elif defined(WIN_SCRAP)
    
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
        CloseClipboard ();
    }
    else
    {
        /* Could not access the clipboard, raise an error. */
        CLEAN_CLIP_BUFFER ();
        CloseClipboard ();
        return 0;
    }

#elif defined(QNX_SCRAP)
  #if (_NTO_VERSION < 620) /* Before 6.2.0 releases. */
    {
        PhClipHeader clheader = { Ph_CLIPBOARD_TYPE_TEXT, 0, NULL };
        int* cldata;
        int status;
        
        cldata = (int *) _clipbuffer;
        *cldata = type;
        clheader.data = _clipbuffer;
        if (dstlen > 65535)
            clheader.length = 65535; /* Maximum photon clipboard size. :( */
        else
            clheader.length = nulledlen;
        
        status = PhClipboardCopy (InputGroup, 1, &clheader);
        if (status == -1)
        {
            /* Could not access the clipboard, raise an error. */
            CLEAN_CLIP_BUFFER ();
            return 0;
        }
    }

  #else /* 6.2.0 and 6.2.1 and future releases. */
    {
        PhClipboardHdr clheader = { Ph_CLIPBOARD_TYPE_TEXT, 0, NULL };
        int* cldata;
        int status;
        
        cldata = (int *) _clipbuffer;
        *cldata = type;
        clheader.data = _clipbuffer;
        clheader.length = nulledlen;

        status = PhClipboardWrite (InputGroup, 1, &clheader);
        if (status == -1)
        {
            /* Could not access the clipboard, raise an error. */
            CLEAN_CLIP_BUFFER ();
            return 0;
        }
    }
  #endif
#endif /* scrap type */

    return 1;
}

char*
pygame_get_scrap (int type)
{
    scrap_type format = _convert_format (type);
    char *retval = NULL;

    if (!pygame_scrap_initialized ())
    {
        PyErr_SetString (PyExc_SDLError, "scrap system not initialized.");
        return 0;
    }

    /* If we are the owner, simply return the clip buffer, if it matches
     * the request type. */
    if (!pygame_lost_scrap ())
    {
        if (format != _cliptype)
            return NULL;

        if (_clipbuffer)
        {
            retval = malloc (_clipsize + 1);
            if (!retval)
                return NULL;
            memset (retval, 0, _clipsize + 1);
            memcpy (retval, _clipbuffer, _clipsize + 1);
            return retval;
        }
        return NULL;
    }

#if defined(X11_SCRAP)
    {
        Window owner;
        Atom source = XA_PRIMARY;
        Atom selection;
        time_t start;
        Atom sel_type;
        int sel_format;
        unsigned long nbytes;
        unsigned long overflow;
        unsigned char *src;
        XEvent ev;

        Lock_Display ();
    
        /* Find a selection owner. */
        owner = _get_scrap_owner (&source);
        if (owner == None)
            return NULL;

        selection = XInternAtom (SDL_Display, "SDL_SELECTION", False);
        /* Copy and convert the selection into our SDL_SELECTION atom of the
         * window. 
         * Flush afterwards, so we have an immediate effect and do not receive
         * the old buffer anymore. */
        XConvertSelection (SDL_Display, source, format, selection, SDL_Window,
                           CurrentTime);
        XSync (SDL_Display, False);

        /* Let's wait for the SelectionNotify event from the callee and
         * react upon it as soon as it is received. */
        for (start = time (0);;)
        {
            if (XCheckTypedWindowEvent (SDL_Display, SDL_Window,
                                        SelectionNotify, &ev))
                break;
            if (time (0) - start >= 5)
            {
                /* Timeout, damn. */
                Unlock_Display ();
                return NULL;
            }
        }
        
        /* Get any property type and check the sel_type afterwards to decide
         * what to do. */
        if (XGetWindowProperty (SDL_Display, ev.xselection.requestor,
                                selection, 0, INT_MAX / 4, True,
                                AnyPropertyType, &sel_type, &sel_format,
                                &nbytes, &overflow, &src) != Success)
        {
            XFree (src);
            return NULL;
        }

        /* In case we requested an XA_STRING (SCRAP_TEXT), any property
         * type of XA_STRING, XA_COMPOUND_TEXT, XA_UTF8_STRING and
         * XA_TEXT is valid. */ 
        if (format == PYGAME_SCRAP_TEXT &&
            (sel_type != XInternAtom (SDL_Display, "UTF8_STRING", False)
             && sel_type != XInternAtom (SDL_Display, "UTF8_STRING", False)
             && sel_type != XInternAtom (SDL_Display, "TEXT", False)
             && sel_type != XInternAtom (SDL_Display, "COMPOUND_TEXT", False)
             && sel_type != XA_STRING))
        {
            /* No matching text type found. Return nothing then. */
            XFree (src);
            return NULL;
        }
        else if (format == PYGAME_SCRAP_BMP && sel_type != PYGAME_SCRAP_BMP)
        {
            /* No matching bitmap type found. Return nothing then. */
            XFree (src);
            return NULL;
        }
        
        /* Anything is fine, so copy the buffer and return it. */
        retval = malloc (nbytes + 1);
        memset (retval, 0, nbytes + 1);
        memcpy (retval, src, nbytes);
        XFree (src);

        Unlock_Display ();
    }

#elif defined(WIN_SCRAP)
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
             * Win32 clipboard or is NUL the usual delimiter? */
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
    
#elif defined(QNX_SCRAP)
  #if (_NTO_VERSION < 620) /* before 6.2.0 releases */
    {
        void *clhandle;
        PhClipHeader *clheader;
        int *cldata;
        
        clhandle = PhClipboardPasteStart (InputGroup);
        
        if (clhandle)
        {
            clheader = PhClipboardPasteType (clhandle, Ph_CLIPBOARD_TYPE_TEXT);
            if (clheader)
            {
                cldata = clheader->data;
                if (*cldata == type)
                    retval = malloc (clheader->length + 1);
                
                if (retval)
                {
                    memset (retval, 0, clheader->length + 1);
                    memcpy (retval, cldata, clheader->length + 1);
                }
            }
            PhClipboardPasteFinish (clhandle);
        }
    }
  #else /* 6.2.0 and 6.2.1 and future releases */
    {
        void* clhandle;
        PhClipboardHdr* clheader;
        int* cldata;
        
        clheader = PhClipboardRead (InputGroup, Ph_CLIPBOARD_TYPE_TEXT);
        if (clheader)
        {
            cldata = clheader->data;
            if (*cldata == type)
                retval = malloc (clheader->length + 1);
            
            if (retval)
            {
                memset (retval, 0, clheader->length + 1);
                memcpy (retval, cldata, clheader->length + 1);
            }
            /* According to the QNX 6.x docs, the clheader pointer is a
             * newly created one that must be freed manually. */
            free (clheader->data);
            free (clheader);
        }
    }
  #endif
#endif /* scrap type */

    return retval;
}

/*
 * The python specific stuff.
 */

static PyObject*
scrap_init (PyObject *self, PyObject *args)
{
    VIDEO_INIT_CHECK ();
    if (!pygame_init_scrap ())
        return RAISE (PyExc_SDLError, SDL_GetError ());
    Py_RETURN_NONE;
}

/*
 * this will return a python string of the clipboard.
 */
static PyObject*
scrap_get_scrap (PyObject* self, PyObject* args)
{
    char *scrap = NULL;
    PyObject *return_string;
    int scrap_type;

    PYGAME_SCRAP_INIT_CHECK ();

    if(!PyArg_ParseTuple (args, "i", &scrap_type))
        return NULL;

    /* pygame_get_scrap() only returns NULL or !NULL, but won't set any
     * errors. */
    scrap = pygame_get_scrap (scrap_type);
    if (!scrap)
        Py_RETURN_NONE;

    return_string = PyString_FromString (scrap);
    free (scrap);
    return return_string;
}

/*
 * this will put a python string into the clipboard.
 */
static PyObject*
scrap_put_scrap (PyObject* self, PyObject* args)
{
    int scraplen;
    char *scrap = NULL;
    int scrap_type;

    PYGAME_SCRAP_INIT_CHECK ();

    if(!PyArg_ParseTuple (args, "it#", &scrap_type, &scrap, &scraplen))
        return NULL;

    if (!pygame_put_scrap (scrap_type, scraplen, scrap))
        return RAISE (PyExc_SDLError,
                      "content could not be placed in clipboard.");
    Py_RETURN_NONE;
}

static PyObject*
scrap_lost_scrap (PyObject* self, PyObject* args)
{
    PYGAME_SCRAP_INIT_CHECK ();

    if (pygame_lost_scrap ())
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

#endif /* !defined(MAC_SCRAP) */

static PyMethodDef scrap_builtins[] =
{
    /*
     * Only initialise these functions for ones we know about.
     *
     * Note, the macosx stuff is done in pygame/__init__.py 
     *   by importing pygame.mac_scrap
     */
#if defined(X11_SCRAP) || defined(WIN_SCRAP) || defined(QNX_SCRAP) || defined(MAC_SCRAP)
	{ "init", scrap_init, 1, DOC_PYGAMESCRAPINIT },
	{ "get", scrap_get_scrap, 1, DOC_PYGAMESCRAPGET },
	{ "put", scrap_put_scrap, 1, DOC_PYGAMESCRAPPUT },
	{ "lost", scrap_lost_scrap, 1, DOC_PYGAMESCRAPLOST},
#endif
	{ NULL, NULL }
};

PYGAME_EXPORT
void initscrap (void)
{
    /* create the module */
    Py_InitModule3 ("scrap", scrap_builtins, NULL);

    /*imported needed apis*/
    import_pygame_base ();
}
