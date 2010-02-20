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

#ifdef SDL_VIDEO_DRIVER_X11

#ifndef MIN
#define MIN(x,y) (((x) < (y)) ? (x) : (y))
#endif
#ifndef MAX
#define MAX(x,y) (((x) > (y)) ? (x) : (y))
#endif

#include <time.h>
#include <X11/Xutil.h>
#include "scrap_x11.h"

typedef struct
{
  Atom type;
  char *data;
  unsigned int size;
} _ClipData;

static _ClipData *_selectiondata = NULL;
static _ClipData *_clipboarddata = NULL;
static ScrapType _currentmode;
static Display *_sdldisplay = NULL;
static Window _sdlwindow = 0;
static void (*Lock_Display)(void);
static void (*Unlock_Display)(void);

/* Atoms used. */
static Atom _atom_UTF8;
static Atom _atom_TEXT;
static Atom _atom_COMPOUND;
static Atom _atom_MIME_PLAIN;
static Atom _atom_MIME_UTF8;
static Atom _atom_TARGETS;
static Atom _atom_TIMESTAMP;
static Atom _atom_SDL;
static Atom _atom_BMP;
static Atom _atom_CLIPBOARD;

/* Timestamps for the requests. */
static Time _cliptime = CurrentTime;
static Time _selectiontime = CurrentTime;

/* Maximum size to send or receive per request. */
#define MAX_CHUNK_SIZE(display)                                    \
    MIN(262144, /* 65536 * 4 */                                    \
        (XExtendedMaxRequestSize (display)) == 0                   \
        ? XMaxRequestSize (display) - 100                          \
        : XExtendedMaxRequestSize (display) - 100)
#define GET_CLIPATOM(x) ((x == SCRAP_SELECTION) ? XA_PRIMARY : _atom_CLIPBOARD)
#define GET_CLIPLIST(x) \
    ((x == SCRAP_SELECTION) ? _selectiondata : _clipboarddata)

static int _add_clip_data (Atom type, char *data, unsigned int size);
static Atom _convert_format (char *type);
static char* _atom_to_string (Atom a);
static void _init_atom_types (void);
static void _set_targets (_ClipData *data, Display *display, Window window,
    Atom property);
static int _set_data (_ClipData *datalist, Display *display, Window window,
    Atom property, Atom target);
static int _clipboard_filter (const SDL_Event *event);
static Window _get_scrap_owner (Atom *selection);
static int _get_data_as (Atom source, Atom format, char **data,
    unsigned int *size);

/**
 * Adds additional data to the currently selected clipboard and Window
 * if it does not already exist.
 *
 * \param cliptype The Atom to set the data for.
 * \param data The data to set.
 * \param size The length of the data.
 */
static int
_add_clip_data (Atom type, char *data, unsigned int size)
{
    _ClipData *tmp;
    int entries;
    _ClipData *datalist = GET_CLIPLIST (_currentmode);
    Atom clip = GET_CLIPATOM (_currentmode);
    
    entries = (sizeof (datalist) / sizeof (datalist[0]));
    tmp = realloc (datalist, (entries + 1) * sizeof (_ClipData));
    if (!tmp)
        return 0;
    datalist = tmp;
    datalist[entries].type = type;
    datalist[entries].data = data;
    datalist[entries].size = size;
    
    XChangeProperty (_sdldisplay, _sdlwindow, clip, type,
        8, PropModeReplace, (unsigned char *) data, (int)size);
    return 0;
}
/**
 * \brief Converts the passed type into a system specific type to use
 *        for the clipboard.
 *
 * \param type The type to convert.
 * \return A system specific type.
 */
static Atom
_convert_format (char *type)
{
    if (strcmp (type, SCRAP_FORMAT_PPM) == 0)
        return XA_PIXMAP;
    if (strcmp (type, SCRAP_FORMAT_PBM) == 0)
        return XA_BITMAP;
    return XInternAtom (_sdldisplay, type, False);
}

/**
 * \brief Returns the name of the passed Atom. The result has to be
 * freed by the caller using free().
 *
 * \param a The Atom to get the name for.
 * \return The name of the Atom or NULL in case of an error.
 */
static char*
_atom_to_string (Atom a)
{
    char *name, *retval;

    if (!a)
    {
        SDL_SetError ("atom argument NULL");
        return NULL;
    }
    name = XGetAtomName (_sdldisplay, a);
    retval = strdup (name);
    XFree (name);
    if (!retval)
        SDL_SetError ("name could not be copied");
    return retval;
}

/**
 * \brief Initializes the used atom types.
 */
static void
_init_atom_types (void)
{
    _atom_UTF8 = XInternAtom (_sdldisplay, "UTF8_STRING", False);
    _atom_TEXT = XInternAtom (_sdldisplay, "TEXT", False);
    _atom_COMPOUND = XInternAtom (_sdldisplay, "COMPOUND_TEXT", False);
    _atom_MIME_PLAIN = XInternAtom (_sdldisplay, "text/plain", False);
    _atom_MIME_UTF8  = XInternAtom (_sdldisplay, "text/plain;charset=utf-8",
        False);
    _atom_TARGETS = XInternAtom (_sdldisplay, "TARGETS", False);
    _atom_TIMESTAMP = XInternAtom (_sdldisplay, "TIMESTAMP", False);
    _atom_SDL = XInternAtom (_sdldisplay, "SDL_SELECTION", False);
    _atom_BMP = XInternAtom (_sdldisplay, SCRAP_FORMAT_BMP, False);
    _atom_CLIPBOARD = XInternAtom (_sdldisplay, "CLIPBOARD", False);
}

/**
 * Sets the list of target atoms available in the clipboard.
 *
 * \param data The clipboard dictionary.
 * \param display The requesting Display.
 * \param window The requesting Window.
 * \param property The request property to place the list into.
 */
static void
_set_targets (_ClipData *data, Display *display, Window window, Atom property)
{
    int i, amount = (sizeof (data) / sizeof (data[0]));
    
    /* All types plus the TARGETS and a TIMESTAMP atom. */
    Atom *targets = malloc ((amount + 2) * sizeof (Atom));
    if (targets == NULL)
        return;
    memset (targets, 0, (amount + 2) * sizeof (Atom));
    targets[0] = _atom_TARGETS;
    targets[1] = _atom_TIMESTAMP;
    for (i = 0; i < amount; i++)
        targets[i + 2] = data[i].type;

    XChangeProperty (display, window, property, XA_ATOM, 32, PropModeReplace,
        (unsigned char*) targets, amount + 2);
    free (targets);
}

/**
 * Places the requested Atom data into a Window.
 *
 * \param datalist The clipboard dictionary.
 * \param display The requesting Display.
 * \param window The requesting Window.
 * \param property The request property to place the list into.
 * \param target The target porperty to place the list into.
 * \return 0 if no data for the target is available, 1 on success.
 */
static int
_set_data (_ClipData *datalist, Display *display, Window window, Atom property,
    Atom target)
{
    char *value = NULL;
    _ClipData *data = NULL;
    int i, size = (sizeof (datalist) / sizeof (datalist[0]));

    for (i = 0; i < size; i++)
        if (datalist[i].type == target)
        {
            data = &(datalist[i]);
            break;
        }

            
    if (!data)
        return 0;

    size = data->size;
    value = data->data;

    /* Send data. */
    XChangeProperty (display, window, property, target, 8, PropModeReplace,
        (unsigned char *) value, size);
    return 1;
}

/**
 * \brief System message filter function -- handles X11 clipboard messages.
 *
 * \param event The SDL_Event to check.
 * \return Always 1.
 */
static int
_clipboard_filter (const SDL_Event *event)
{
    _ClipData *datalist;
    Time timestamp = CurrentTime;

    /* Post all non-window manager specific events */
    if (event->type != SDL_SYSWMEVENT)
        return 1;

    XEvent xevent = event->syswm.msg->event.xevent;

    /* Handle window manager specific clipboard events */
    switch (xevent.type)
    {
    case PropertyNotify:
    {
        /* Handled in scrap_put_x11(). */
        break;
    }
    case SelectionClear:
    {
        XSelectionClearEvent *clear = &xevent.xselectionclear;

        /* Looks like another window takes control over the clipboard.
         * Release the internally saved buffer, if any.
         */
        if (clear->selection == XA_PRIMARY)
            timestamp = _selectiontime;
        else if(clear->selection == _atom_CLIPBOARD)
            timestamp = _cliptime;
        else
            break;

        /* Do not do anything, if the times do not match. */
        if (timestamp != CurrentTime && xevent.xselectionclear.time < timestamp)
            break;

        /* Clean the dictionaries. */
        if (clear->selection == XA_PRIMARY)
            free (_selectiondata);
        else if (clear->selection != _atom_CLIPBOARD)
            free (_clipboarddata);
        break;
    }
    case SelectionNotify:
        /* This one will be handled directly in the scrap_get_x11() function.
         */
        break;

    case SelectionRequest:
    {
        XSelectionRequestEvent *req = &xevent.xselectionrequest;
        XEvent ev;
        
        /* Prepare answer. */
        ev.xselection.type      = SelectionNotify;
        ev.xselection.display   = req->display;
        ev.xselection.requestor = req->requestor;
        ev.xselection.selection = req->selection;
        ev.xselection.target    = req->target;
        ev.xselection.property  = None;
        ev.xselection.time      = req->time;

        /* Which clipboard type was requested? */
        if (req->selection == XA_PRIMARY)
        {
            datalist = _selectiondata;
            timestamp = _selectiontime;
        }
        else if (req->selection == _atom_CLIPBOARD)
        {
            datalist = _clipboarddata;
            timestamp = _cliptime;

        }
        else /* Anything else's not supported. */
        {
            XSendEvent (req->display, req->requestor, False, NoEventMask, &ev);
            return 1;
        }

        /* No data? */
        if ((sizeof (datalist) / sizeof (datalist[0])) == 0)
        {
            XSendEvent (req->display, req->requestor, False, NoEventMask, &ev);
            return 1;
        }

        /* We do not own the selection anymore. */
        if (timestamp == CurrentTime
            || (req->time != CurrentTime && timestamp > req->time))
        {
            XSendEvent (req->display, req->requestor, False, NoEventMask, &ev);
            return 1;
        }

        /* 
         * TODO: We have to make it ICCCM compatible at some point by
         * implementing the MULTIPLE atom request.
         */

        /* Old client? */
        if (req->property == None)
            ev.xselection.property = req->target;
        
        /* TODO */
        if (req->target == _atom_TARGETS)
        {
            /* The requestor wants to know, what we've got. */
            _set_targets (datalist, req->display, req->requestor,
                req->property);
        }
        else
        {
            _set_data (datalist, req->display, req->requestor, req->property,
                req->target);
        }

        ev.xselection.property = req->property;
        /* Post the event for X11 clipboard reading above */
        XSendEvent (req->display, req->requestor, False, 0, &ev);
        break;
    }
    }
    return 1;
}


/**
 * \brief Tries to determine the X window with a valid selection.
 *        Default is to check
 *         - passed parameter
 *         - CLIPBOARD
 *         - XA_PRIMARY
 *         - XA_SECONDARY
 *         - XA_CUT_BUFFER0 - 7
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
        XA_CUT_BUFFER1, XA_CUT_BUFFER2, XA_CUT_BUFFER3, XA_CUT_BUFFER4,
        XA_CUT_BUFFER5, XA_CUT_BUFFER6, XA_CUT_BUFFER7 };

    Window owner = XGetSelectionOwner (_sdldisplay, *selection);
    if (owner != None)
        return owner;

    owner = XGetSelectionOwner (_sdldisplay, _atom_CLIPBOARD);
    if (owner != None)
        return owner;

    while (i < 10)
    {
        owner = XGetSelectionOwner (_sdldisplay, buffers[i]);
        if (owner != None)
        {
            *selection = buffers[i];
            return owner;
        }
        i++;
    }
    
    return None;
}

/**
 * Retrieves the data from a certain source Atom using a specific
 * format.
 *
 * \param source The currently active clipboard Atom to get the data from.
 * \param format The format of the data to get.
 * \param length Out parameter that contains the length of the returned
 * buffer.
 * \return The requested content or NULL in case no content exists or an
 * error occured.
 */
static int
_get_data_as (Atom source, Atom format, char **data, unsigned int *size)
{
    Window owner;
    time_t start;
    Atom sel_type;
    int sel_format;
    unsigned long nbytes, overflow;
    *size = 0;
    unsigned char *src;
    unsigned long offset = 0;
    unsigned long chunk = 0;
    int step = 1;
    XEvent ev;
    Time timestamp;

    /* If we are the owner, simply return the clip buffer, if it matches
     * the request type.
     */
    if (!scrap_lost_x11 ())
    {
        char *fmt;

        fmt = _atom_to_string (format);
        /* TODO */
        free (fmt);

        return 1;
    }

    Lock_Display ();
    
    /* Find a selection owner. */
    owner = _get_scrap_owner (&source);
    if (owner == None)
    {
        Unlock_Display ();
        SDL_SetError ("no clipboard owner found");
        return 0;
    }

    timestamp = (source == XA_PRIMARY) ?  _selectiontime : _cliptime;

    /* Copy and convert the selection into our SDL_SELECTION atom of the
     * window. 
     * Flush afterwards, so we have an immediate effect and do not receive
     * the old buffer anymore.
     */
    XConvertSelection (_sdldisplay, source, format, _atom_SDL, _sdlwindow,
        timestamp);
    XSync (_sdldisplay, False);

    /* Let's wait for the SelectionNotify event from the callee and
     * react upon it as soon as it is received.
     */
    for (start = time (0);;)
    {
        if (XCheckTypedWindowEvent (_sdldisplay, _sdlwindow,
                SelectionNotify, &ev))
            break;
        if (time (0) - start >= 5)
        {
            /* Timeout, damn. */
            Unlock_Display ();
            SDL_SetError ("timeout on retrieving the clipboard data");
            return 0;
        }
    }
    /* Get any property type and check the sel_type afterwards to decide
     * what to do.
     */
    if (XGetWindowProperty (_sdldisplay, ev.xselection.requestor, _atom_SDL,
            0, 0, True, AnyPropertyType, &sel_type, &sel_format, &nbytes,
            &overflow, &src) != Success)
    {
        XFree (src);
        Unlock_Display ();
        SDL_SetError ("could not receive clipboard buffer");
        return 0;
    }

    /* In case we requested a SCRAP_TEXT, any property type of
     * XA_STRING, XA_COMPOUND_TEXT, UTF8_STRING and TEXT is valid.
     */
    if (format == _atom_MIME_PLAIN &&
        (sel_type != _atom_UTF8 && sel_type != _atom_TEXT
            && sel_type != _atom_COMPOUND && sel_type != XA_STRING))
    {
        /* No matching text type found. Return nothing then. */
        XFree (src);
        Unlock_Display ();
        SDL_SetError ("text format not found");
        return 0;
    }

    /* Anything is fine, so copy the buffer and return it. */
    switch (sel_format)
    {
    case 16:
        step = sizeof (short) / 2;
        break;
    case 32:
        step = sizeof (long) / 4;
        break;
    case 8:
    default:
        step = sizeof (char);
        *size = overflow; /* 8 bit size is already correctly set in nbytes.*/
        break;
    }

    /* X11 guarantees NULL termination, add an extra byte. */
    *size = step * overflow;
    *data = malloc (*size + 1);
    if (*data)
    {
        unsigned long boffset = 0;
        chunk = MAX_CHUNK_SIZE(_sdldisplay);
        memset (*data, 0, (size_t) (*size + 1));

        /* Read as long as there is data. */
        while (overflow)
        {
            if (XGetWindowProperty (_sdldisplay, ev.xselection.requestor,
                    _atom_SDL, offset, chunk, True, AnyPropertyType, &sel_type,
                    &sel_format, &nbytes, &overflow, &src) != Success)
            {
                break;
            }
            
            offset += nbytes / (32 / sel_format);
            nbytes *= step * sel_format / 8;
            memcpy ((*data) + boffset, src, nbytes);
            boffset += nbytes;
            XFree (src);
        }
    }
    else
    {
        /* ENOMEM */
        SDL_SetError ("could not allocate memory");
        return 0;
    }
    /* In case we've got a COMPOUND_TEXT, convert it to the current
     * multibyte locale.
     */
    if (sel_type == _atom_COMPOUND && sel_format == 8)
    {
        char **list = NULL;
        int count, status = 0;
        XTextProperty p;

        p.encoding = sel_type;
        p.format = sel_format;
        p.nitems = nbytes;
        p.value = (*data);

        status = XmbTextPropertyToTextList (_sdldisplay, &p, &list, &count);
        if (status == XLocaleNotSupported || status == XConverterNotFound)
        {
            free (*data);
            SDL_SetError ("current locale is not supported for conversion.");
            return 0;
        }
        else if (status == XNoMemory)
        {
            free (*data);
            SDL_SetError ("could not allocate memory");
            return 0;
        }
        else if (status == Success)
        {
            if (count && list)
            {
                int i = 0;
                int ioffset = 0;
                char *tmp;

                free (*data);
                *data = NULL;
                for (i = 0; i < count; i++)
                {
                    *size = strlen (list[i]);
                    tmp = *data;
                    *data = realloc (*data, (*size) + 1);
                    if (!(*data))
                    {
                        free (tmp);
                        SDL_SetError ("could not allocate memory");
                        return 0;
                    }
                    ioffset += *size;

                    memcpy (*data, list[i], *size);
                    memset ((*data) + ioffset, '\n', 1);
                }
                memset ((*data) + ioffset, 0, 1);
            }
        }

        if (list)
            XFreeStringList (list);
    }

    Unlock_Display ();
    return 1;
}

int
scrap_init_x11 (void)
{
    SDL_SysWMinfo info;
    int retval = 0;

    SDL_VERSION (&info.version);
    if (SDL_GetWMInfo (&info))
    {
        /* Save the information for later use */
        if (info.subsystem == SDL_SYSWM_X11)
        {
            XWindowAttributes setattrs;
            XSetWindowAttributes newattrs;

            newattrs.event_mask = PropertyChangeMask;

            _sdldisplay = info.info.x11.display;
            _sdlwindow  = info.info.x11.window;
            Lock_Display = info.info.x11.lock_func;
            Unlock_Display = info.info.x11.unlock_func;
            
            Lock_Display ();

            /* We need the PropertyNotify event for the timestap, so
             * modify the event attributes.
             */
            XGetWindowAttributes (_sdldisplay, _sdlwindow, &setattrs);
            newattrs.event_mask |= setattrs.all_event_masks;
            XChangeWindowAttributes (_sdldisplay, _sdlwindow, CWEventMask,
                &newattrs);

            Unlock_Display ();

            /* Enable the special window hook events */
            SDL_EventState (SDL_SYSWMEVENT, SDL_ENABLE);
            SDL_SetEventFilter (_clipboard_filter);

            /* Create the atom types we need. */
            _init_atom_types ();

            retval = 1;
        }
        else
            SDL_SetError ("SDL is not running on X11");
    }
    return retval;
}

void
scrap_quit_x11 (void)
{
    _sdlwindow = 0;
    _sdldisplay = NULL;
    if (_selectiondata)
        free (_selectiondata);
    if (_clipboarddata)
        free (_clipboarddata);
    _clipboarddata = NULL;
    _selectiondata = NULL;
}

int
scrap_contains_x11 (char *type)
{
    int i = 0;
    char **types = NULL;

    if (scrap_get_types_x11 (types) == -1)
        return -1;
    while (types[i])
    {
        if (strcmp (type, types[i]) == 0)
        {
            int x = 0;
            while (types[x])
            {
                free (types[x]);
                x++;
            }
            free (types);
            return 1;
        }
        i++;
    }

    i = 0;
    while (types[i])
    {
        free (types[i]);
        i++;
    }
    free (types);
    return 0;
}

int
scrap_lost_x11 (void)
{
    int retval;

    Lock_Display ();
    retval = (XGetSelectionOwner (_sdldisplay, GET_CLIPATOM (_currentmode)) !=
        _sdlwindow);
    Unlock_Display ();
    return retval;
}

ScrapType
scrap_get_mode_x11 (void)
{
    return _currentmode;
}

ScrapType
scrap_set_mode_x11 (ScrapType mode)
{
    ScrapType oldmode = _currentmode;
    _currentmode = mode;
    return oldmode;
}

int
scrap_get_x11 (char *type, char **data, unsigned int *size)
{
    return _get_data_as (GET_CLIPATOM (_currentmode), _convert_format (type),
        data, size);
}

int
scrap_put_x11 (char *type, char *data, unsigned int size)
{
    Atom clip, cliptype;
    Time timestamp = CurrentTime;
    time_t start;
    XEvent ev;

    Lock_Display ();

    clip = GET_CLIPATOM (_currentmode);
    cliptype = _convert_format (type);

    /* We've some types which should not be set by the user. */
    if (cliptype == _atom_TARGETS || cliptype == _atom_SDL ||
        cliptype == _atom_TIMESTAMP)
    {
        SDL_SetError ("the requested format type is reserved");
        Unlock_Display ();
        return -1;
    }

    /* Update the clipboard property with the buffer. */
    XChangeProperty (_sdldisplay, _sdlwindow, clip, cliptype, 8,
        PropModeReplace, (unsigned char *) data, (int) size);
    
    _add_clip_data (cliptype, data, size);
    if (cliptype == _atom_MIME_PLAIN)
    {
        /* Set SCRAP_FORMAT_TEXT. Also set XA_STRING, TEXT and
         * UTF8_STRING if they are not set in the dictionary.
         */
        _add_clip_data (XA_STRING, data, size);
        _add_clip_data (_atom_UTF8, data, size);
        _add_clip_data (_atom_TEXT, data, size);
    }
    XSync (_sdldisplay, False);

    /* Update the timestamp */
    for (start = time (0);;)
    {
        if (XCheckTypedWindowEvent (_sdldisplay, _sdlwindow, PropertyNotify,
            &ev))
            break;
        if (time (0) - start >= 5)
        {
            /* Timeout, damn. */
            Unlock_Display ();
            goto SETSELECTIONOWNER;
        }
    }
    if (ev.xproperty.atom == clip)
    {
        timestamp = ev.xproperty.time;

        if (cliptype == XA_PRIMARY)
            _selectiontime = timestamp;
        else
            _cliptime = timestamp;
    }
    else
        timestamp = (cliptype == XA_PRIMARY) ? _selectiontime : _cliptime;

SETSELECTIONOWNER:
    /* Set the selection owner to the own window. */
    XSetSelectionOwner (_sdldisplay, clip, _sdlwindow, timestamp);
    if (XGetSelectionOwner (_sdldisplay, clip) != _sdlwindow)
    {
        /* Ouch, we could not toggle the selection owner. Raise an
         * error, as it's not guaranteed, that the clipboard
         * contains valid data.
         */
        Unlock_Display ();
        SDL_SetError ("could not set the proper owner for the clipboard");
        return -1;
    }

    Unlock_Display ();
    return 1;
}

int
scrap_get_types_x11 (char** types)
{
    Atom *targetdata;
    unsigned int length;
    _ClipData *datalist = GET_CLIPLIST(_currentmode);
    
    if (!scrap_lost_x11 ())
    {
        size_t i = 0;
        size_t size = (sizeof (datalist) / sizeof (datalist[0]));
        
        types = malloc (sizeof (char*) * size);
        if (!types)
        {
            SDL_SetError ("could not allocate memory");
            return -1;
        }
        
        memset (types, 0, size);
        for (i = 0; i < size; i++)
        {
            types[i] = _atom_to_string (datalist[i].type);
            if (!types[i])
            {
                int j = 0;
                while (types[j])
                {
                    free (types[j]);
                    j++;
                }
                free (types);
                return -1;
            }
            i++;
        }
        types[i] = NULL;
        return 1;
    }
    
    if (_get_data_as (GET_CLIPATOM (_currentmode), _atom_TARGETS,
        ((char**) &targetdata), &length) == -1)
        return -1;

    if (length > 0 && targetdata)
    {
        int i, count = length / sizeof (Atom);
        
        types = malloc (sizeof (char *) * (count + 1));
        if (!types)
        {
            free (targetdata);
            return -1;
        }
        memset (types, 0, sizeof (char *) * (count + 1));

        for (i = 0; i < count; i++)
        {
            types[i] = _atom_to_string (datalist[i].type);
            if (!types[i])
            {
                int j = 0;
                while (types[j])
                {
                    free (types[j]);
                    j++;
                }
                free (types);
                return -1;
            }
            i++;
        }
        free (targetdata);
        return 1;
    }
    return 0;
}

#endif /* SDL_VIDEO_DRIVER_X11 */
