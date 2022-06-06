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

#include <X11/Xutil.h>

static Display *SDL_Display;
static Window SDL_Window;
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
#define MAX_CHUNK_SIZE(display)                 \
    MIN(262144, /* 65536 * 4 */                 \
        (XExtendedMaxRequestSize(display)) == 0 \
            ? XMaxRequestSize(display) - 100    \
            : XExtendedMaxRequestSize(display) - 100)

#define GET_CLIPATOM(x) ((x == SCRAP_SELECTION) ? XA_PRIMARY : _atom_CLIPBOARD)

static Atom
_convert_format(char *type);
static void
_init_atom_types(void);
static char *
_atom_to_string(Atom a);
static void
_add_clip_data(Atom type, char *data, int srclen);
static int
_clipboard_filter(const SDL_Event *event);
static void
_set_targets(PyObject *data, Display *display, Window window, Atom property);
static int
_set_data(PyObject *dict, Display *display, Window window, Atom property,
          Atom target);
static Window
_get_scrap_owner(Atom *selection);
static char *
_get_data_as(Atom source, Atom format, unsigned long *length);

/**
 * \brief Converts the passed type into a system specific type to use
 *        for the clipboard.
 *
 * \param type The type to convert.
 * \return A system specific type.
 */
static Atom
_convert_format(char *type)
{
    if (strcmp(type, PYGAME_SCRAP_PPM) == 0)
        return XA_PIXMAP;
    if (strcmp(type, PYGAME_SCRAP_PBM) == 0)
        return XA_BITMAP;
    return XInternAtom(SDL_Display, type, False);
}

/**
 * \brief Initializes the used atom types.
 */
static void
_init_atom_types(void)
{
    _atom_UTF8 = XInternAtom(SDL_Display, "UTF8_STRING", False);
    _atom_TEXT = XInternAtom(SDL_Display, "TEXT", False);
    _atom_COMPOUND = XInternAtom(SDL_Display, "COMPOUND_TEXT", False);
    _atom_MIME_PLAIN = XInternAtom(SDL_Display, "text/plain", False);
    _atom_MIME_UTF8 =
        XInternAtom(SDL_Display, "text/plain;charset=utf-8", False);
    _atom_TARGETS = XInternAtom(SDL_Display, "TARGETS", False);
    _atom_TIMESTAMP = XInternAtom(SDL_Display, "TIMESTAMP", False);
    _atom_SDL = XInternAtom(SDL_Display, "SDL_SELECTION", False);
    _atom_BMP = XInternAtom(SDL_Display, PYGAME_SCRAP_BMP, False);
    _atom_CLIPBOARD = XInternAtom(SDL_Display, "CLIPBOARD", False);
}

/**
 * \brief Returns the name of the passed Atom. The result has to be
 * freed by the caller using free().
 *
 * \param a The Atom to get the name for.
 * \return The name of the Atom.
 */
static char *
_atom_to_string(Atom a)
{
    char *name;
    char *retval;

    if (!a)
        return NULL;
    name = XGetAtomName(SDL_Display, a);
    retval = strdup(name);
    XFree(name);
    return retval;
}

/**
 * Adds additional data to the currently selected clipboard and Window
 * if it does not already exist.
 *
 * \param cliptype The Atom to set the data for.
 * \param data The data to set.
 * \param srclen The length of the data.
 */
static void
_add_clip_data(Atom cliptype, char *data, int srclen)
{
    Atom clip = GET_CLIPATOM(_currentmode);
    PyObject *dict =
        (_currentmode == SCRAP_CLIPBOARD) ? _clipdata : _selectiondata;
    PyObject *tmp;
    char *key = _atom_to_string(cliptype);

    tmp = PyBytes_FromStringAndSize(data, srclen);
    PyDict_SetItemString(dict, key, tmp);
    Py_DECREF(tmp);
    XChangeProperty(SDL_Display, SDL_Window, clip, cliptype, 8,
                    PropModeReplace, (unsigned char *)data, srclen);
    free(key);
}

/**
 * \brief System message filter function -- handles X11 clipboard messages.
 *
 * \param event The SDL_Event to check.
 * \return Always 1.
 */
static int
_clipboard_filter(const SDL_Event *event)
{
    PyObject *dict = NULL;
    Time timestamp = CurrentTime;

    /* Post all non-window manager specific events */
    if (event->type != SDL_SYSWMEVENT)
        return 1;

    XEvent xevent = event->syswm.msg->event.xevent;

    /* Handle window-manager specific clipboard events */
    switch (xevent.type) {
        case PropertyNotify: {
            /* Handled in pygame_scrap_put(). */
            break;
        }
        case SelectionClear: {
            XSelectionClearEvent *clear = &xevent.xselectionclear;

            /* Looks like another window takes control over the clipboard.
             * Release the internally saved buffer, if any.
             */
            if (clear->selection == XA_PRIMARY)
                timestamp = _selectiontime;
            else if (clear->selection == _atom_CLIPBOARD)
                timestamp = _cliptime;
            else
                break;

            /* Do not do anything, if the times do not match. */
            if (timestamp != CurrentTime &&
                xevent.xselectionclear.time < timestamp)
                break;

            /* Clean the dictionaries. */
            if (clear->selection == XA_PRIMARY)
                PyDict_Clear(_selectiondata);
            else if (clear->selection != _atom_CLIPBOARD)
                PyDict_Clear(_clipdata);
            break;
        }
        case SelectionNotify:
            /* This one will be handled directly in the pygame_scrap_get ()
             * function.
             */
            break;

        case SelectionRequest: {
            XSelectionRequestEvent *req = &xevent.xselectionrequest;
            XEvent ev;

            /* Prepare answer. */
            ev.xselection.type = SelectionNotify;
            ev.xselection.display = req->display;
            ev.xselection.requestor = req->requestor;
            ev.xselection.selection = req->selection;
            ev.xselection.target = req->target;
            ev.xselection.property = None;
            ev.xselection.time = req->time;

            /* Which clipboard type was requested? */
            if (req->selection == XA_PRIMARY) {
                dict = _selectiondata;
                timestamp = _selectiontime;
            }
            else if (req->selection == _atom_CLIPBOARD) {
                dict = _clipdata;
                timestamp = _cliptime;
            }
            else /* Anything else's not supported. */
            {
                XSendEvent(req->display, req->requestor, False, NoEventMask,
                           &ev);
                return 1;
            }

            /* No data? */
            if (PyDict_Size(dict) == 0) {
                XSendEvent(req->display, req->requestor, False, NoEventMask,
                           &ev);
                return 1;
            }

            /* We do not own the selection anymore. */
            if (timestamp == CurrentTime ||
                (req->time != CurrentTime && timestamp > req->time)) {
                XSendEvent(req->display, req->requestor, False, NoEventMask,
                           &ev);
                return 1;
            }

            /*
             * TODO: We have to make it ICCCM compatible at some point by
             * implementing the MULTIPLE atom request.
             */

            /* Old client? */
            if (req->property == None)
                ev.xselection.property = req->target;

            if (req->target == _atom_TARGETS) {
                /* The requestor wants to know, what we've got. */
                _set_targets(dict, req->display, req->requestor,
                             req->property);
            }
            else {
                _set_data(dict, req->display, req->requestor, req->property,
                          req->target);
            }

            ev.xselection.property = req->property;
            /* Post the event for X11 clipboard reading above */
            XSendEvent(req->display, req->requestor, False, 0, &ev);
            break;
        }
    }
    return 1;
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
_set_targets(PyObject *data, Display *display, Window window, Atom property)
{
    int i;
    char *format;
    PyObject *list = PyDict_Keys(data);
    PyObject *chars;
    int amount = PyList_Size(list);
    /* All types plus the TARGETS and a TIMESTAMP atom. */
    Atom *targets = malloc((amount + 2) * sizeof(Atom));
    if (targets == NULL)
        return;
    memset(targets, 0, (amount + 2) * sizeof(Atom));
    targets[0] = _atom_TARGETS;
    targets[1] = _atom_TIMESTAMP;
    for (i = 0; i < amount; i++) {
        chars = PyUnicode_AsASCIIString(PyList_GetItem(list, i));
        if (!chars) {
            return;
        }
        format = PyBytes_AsString(chars);
        targets[i + 2] = _convert_format(format);
        Py_DECREF(chars);
    }
    XChangeProperty(display, window, property, XA_ATOM, 32, PropModeReplace,
                    (unsigned char *)targets, amount + 2);
}

/**
 * Places the requested Atom data into a Window.
 *
 * \param data The clipboard dictionary.
 * \param display The requesting Display.
 * \param window The requesting Window.
 * \param property The request property to place the list into.
 * \param target The target property to place the list into.
 * \return 0 if no data for the target is available, 1 on success.
 */
static int
_set_data(PyObject *data, Display *display, Window window, Atom property,
          Atom target)
{
    char *name = _atom_to_string(target);
    PyObject *val = PyDict_GetItemString(data, name);
    char *value = NULL;
    int size;

    if (!val) {
        XFree(name);
        return 0;
    }
    size = PyBytes_Size(val);
    value = PyBytes_AsString(val);

    /* Send data. */
    XChangeProperty(display, window, property, target, 8, PropModeReplace,
                    (unsigned char *)value, size);
    XFree(name);
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
_get_scrap_owner(Atom *selection)
{
    int i = 0;
    static Atom buffers[] = {XA_PRIMARY,     XA_SECONDARY,   XA_CUT_BUFFER0,
                             XA_CUT_BUFFER1, XA_CUT_BUFFER2, XA_CUT_BUFFER3,
                             XA_CUT_BUFFER4, XA_CUT_BUFFER5, XA_CUT_BUFFER6,
                             XA_CUT_BUFFER7};

    Window owner = XGetSelectionOwner(SDL_Display, *selection);
    if (owner != None)
        return owner;

    owner = XGetSelectionOwner(SDL_Display, _atom_CLIPBOARD);
    if (owner != None)
        return owner;

    while (i < 10) {
        owner = XGetSelectionOwner(SDL_Display, buffers[i]);
        if (owner != None) {
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
 * error occurred.
 */
static char *
_get_data_as(Atom source, Atom format, unsigned long *length)
{
    unsigned char *retval = NULL;
    Window owner;
    time_t start;
    Atom sel_type;
    int sel_format;
    unsigned long nbytes;
    unsigned long overflow;
    *length = 0;
    unsigned char *src;
    unsigned long offset = 0;
    unsigned long chunk = 0;
    int step = 1;
    XEvent ev;
    Time timestamp;

    /* If we are the owner, simply return the clip buffer, if it matches
     * the request type.
     */
    if (!pygame_scrap_lost()) {
        char *fmt;
        char *data;

        fmt = _atom_to_string(format);

        if (_currentmode == SCRAP_SELECTION)
            data = PyBytes_AsString(PyDict_GetItemString(_selectiondata, fmt));
        else
            data = PyBytes_AsString(PyDict_GetItemString(_clipdata, fmt));
        free(fmt);

        return data;
    }

    Lock_Display();

    /* Find a selection owner. */
    owner = _get_scrap_owner(&source);
    if (owner == None) {
        Unlock_Display();
        return NULL;
    }

    timestamp = (source == XA_PRIMARY) ? _selectiontime : _cliptime;

    /* Copy and convert the selection into our SDL_SELECTION atom of the
     * window.
     * Flush afterwards, so we have an immediate effect and do not receive
     * the old buffer anymore.
     */
    XConvertSelection(SDL_Display, source, format, _atom_SDL, SDL_Window,
                      timestamp);
    XSync(SDL_Display, False);

    /* Let's wait for the SelectionNotify event from the callee and
     * react upon it as soon as it is received.
     */
    for (start = time(0);;) {
        if (XCheckTypedWindowEvent(SDL_Display, SDL_Window, SelectionNotify,
                                   &ev))
            break;
        if (time(0) - start >= 5) {
            /* Timeout, damn. */
            Unlock_Display();
            return NULL;
        }
    }

    /* Get any property type and check the sel_type afterwards to decide
     * what to do.
     */
    if (XGetWindowProperty(SDL_Display, ev.xselection.requestor, _atom_SDL, 0,
                           0, True, AnyPropertyType, &sel_type, &sel_format,
                           &nbytes, &overflow, &src) != Success) {
        XFree(src);
        Unlock_Display();
        return NULL;
    }

    /* In case we requested a SCRAP_TEXT, any property type of
     * XA_STRING, XA_COMPOUND_TEXT, UTF8_STRING and TEXT is valid.
     */
    if (format == _atom_MIME_PLAIN &&
        (sel_type != _atom_UTF8 && sel_type != _atom_TEXT &&
         sel_type != _atom_COMPOUND && sel_type != XA_STRING)) {
        /* No matching text type found. Return nothing then. */
        XFree(src);
        Unlock_Display();
        return NULL;
    }

    /* Anything is fine, so copy the buffer and return it. */
    switch (sel_format) {
        case 16:
            step = sizeof(short) / 2;
            break;
        case 32:
            step = sizeof(long) / 4;
            break;
        case 8:
        default:
            step = sizeof(char);
            *length =
                overflow; /* 8 bit size is already correctly set in nbytes.*/
            break;
    }

    /* X11 guarantees NULL termination, add an extra byte. */
    *length = step * overflow;
    retval = malloc(*length + 1);
    if (retval) {
        unsigned long boffset = 0;
        chunk = MAX_CHUNK_SIZE(SDL_Display);
        memset(retval, 0, (size_t)(*length + 1));

        /* Read as long as there is data. */
        while (overflow) {
            if (XGetWindowProperty(SDL_Display, ev.xselection.requestor,
                                   _atom_SDL, offset, chunk, True,
                                   AnyPropertyType, &sel_type, &sel_format,
                                   &nbytes, &overflow, &src) != Success) {
                break;
            }

            offset += nbytes / (32 / sel_format);
            nbytes *= step * sel_format / 8;
            memcpy(retval + boffset, src, nbytes);
            boffset += nbytes;
            XFree(src);
        }
    }
    else {
        /* ENOMEM */
        return NULL;
    }

    /* In case we've got a COMPOUND_TEXT, convert it to the current
     * multibyte locale.
     */
    if (sel_type == _atom_COMPOUND && sel_format == 8) {
        char **list = NULL;
        int count;
        int status = 0;
        XTextProperty p;

        p.encoding = sel_type;
        p.format = sel_format;
        p.nitems = nbytes;
        p.value = retval;

        status = XmbTextPropertyToTextList(SDL_Display, &p, &list, &count);
        if (status == XLocaleNotSupported || status == XConverterNotFound) {
            free(retval);
            PyErr_SetString(pgExc_SDLError,
                            "current locale is not supported for conversion.");
            return NULL;
        }
        else if (status == XNoMemory) {
            free(retval);
            return NULL;
        }
        else if (status == Success) {
            if (count && list) {
                int i = 0;
                int ioffset = 0;
                unsigned char *tmp;

                free(retval);
                retval = NULL;
                for (i = 0; i < count; i++) {
                    *length = strlen(list[i]);
                    tmp = retval;
                    retval = realloc(retval, (*length) + 1);
                    if (!retval) {
                        free(tmp);
                        return NULL;
                    }
                    ioffset += *length;

                    memcpy(retval, list[i], *length);
                    memset(retval + ioffset, '\n', 1);
                }
                memset(retval + ioffset, 0, 1);
            }
        }

        if (list)
            XFreeStringList(list);
    }

    Unlock_Display();
    return (char *)retval;
}

int
pygame_scrap_init(void)
{
    SDL_SysWMinfo info;
    int retval = 0;

    /* Grab the window manager specific information */
    SDL_SetError("SDL is not running on known window manager");

    SDL_VERSION(&info.version);
    if (SDL_GetWMInfo(&info)) {
        /* Save the information for later use */
        if (info.subsystem == SDL_SYSWM_X11) {
            XWindowAttributes setattrs;
            XSetWindowAttributes newattrs;

            newattrs.event_mask = PropertyChangeMask;

            SDL_Display = info.info.x11.display;
            SDL_Window = info.info.x11.window;
            Lock_Display = info.info.x11.lock_func;
            Unlock_Display = info.info.x11.unlock_func;

            Lock_Display();

            /* We need the PropertyNotify event for the timestap, so
             * modify the event attributes.
             */
            XGetWindowAttributes(SDL_Display, SDL_Window, &setattrs);
            newattrs.event_mask |= setattrs.all_event_masks;
            XChangeWindowAttributes(SDL_Display, SDL_Window, CWEventMask,
                                    &newattrs);

            Unlock_Display();

            /* Enable the special window hook events */
            SDL_EventState(SDL_SYSWMEVENT, SDL_ENABLE);
            SDL_SetEventFilter(_clipboard_filter);

            /* Create the atom types we need. */
            _init_atom_types();

            retval = 1;
        }
        else
            SDL_SetError("SDL is not running on X11");
    }
    if (retval)
        _scrapinitialized = 1;

    return retval;
}

int
pygame_scrap_lost(void)
{
    int retval;

    if (!pygame_scrap_initialized()) {
        PyErr_SetString(pgExc_SDLError, "scrap system not initialized.");
        return 0;
    }

    Lock_Display();
    retval = (XGetSelectionOwner(SDL_Display, GET_CLIPATOM(_currentmode)) !=
              SDL_Window);
    Unlock_Display();

    return retval;
}

int
pygame_scrap_put(char *type, Py_ssize_t srclen, char *src)
{
    Atom clip;
    Atom cliptype;
    Time timestamp = CurrentTime;
    time_t start;
    XEvent ev;

    if (!pygame_scrap_initialized()) {
        PyErr_SetString(pgExc_SDLError, "scrap system not initialized.");
        return 0;
    }

    Lock_Display();

    clip = GET_CLIPATOM(_currentmode);
    cliptype = _convert_format(type);

    /* We've some types which should not be set by the user. */
    if (cliptype == _atom_TARGETS || cliptype == _atom_SDL ||
        cliptype == _atom_TIMESTAMP) {
        PyErr_SetString(PyExc_ValueError, "the requested type is reserved.");
        Unlock_Display();
        return 0;
    }

    /* Update the clipboard property with the buffer. */
    XChangeProperty(SDL_Display, SDL_Window, clip, cliptype, 8,
                    PropModeReplace, (unsigned char *)src, srclen);

    if (cliptype == _atom_MIME_PLAIN) {
        /* Set PYGAME_SCRAP_TEXT. Also set XA_STRING, TEXT and
         * UTF8_STRING if they are not set in the dictionary.
         */
        _add_clip_data(XA_STRING, src, srclen);
        _add_clip_data(_atom_UTF8, src, srclen);
        _add_clip_data(_atom_TEXT, src, srclen);
    }
    XSync(SDL_Display, False);

    /* Update the timestamp */
    for (start = time(0);;) {
        if (XCheckTypedWindowEvent(SDL_Display, SDL_Window, PropertyNotify,
                                   &ev))
            break;
        if (time(0) - start >= 5) {
            /* Timeout, damn. */
            Unlock_Display();
            goto SETSELECTIONOWNER;
        }
    }
    if (ev.xproperty.atom == clip) {
        timestamp = ev.xproperty.time;

        if (clip == XA_PRIMARY)
            _selectiontime = timestamp;
        else
            _cliptime = timestamp;
    }
    else
        timestamp = (clip == XA_PRIMARY) ? _selectiontime : _cliptime;

SETSELECTIONOWNER:
    /* Set the selection owner to the own window. */
    XSetSelectionOwner(SDL_Display, clip, SDL_Window, timestamp);
    if (XGetSelectionOwner(SDL_Display, clip) != SDL_Window) {
        /* Ouch, we could not toggle the selection owner. Raise an
         * error, as it's not guaranteed, that the clipboard
         * contains valid data.
         */
        Unlock_Display();
        return 0;
    }

    Unlock_Display();
    return 1;
}

char *
pygame_scrap_get(char *type, size_t *count)
{
    if (!pygame_scrap_initialized()) {
        PyErr_SetString(pgExc_SDLError, "scrap system not initialized.");
        return NULL;
    }
    return _get_data_as(GET_CLIPATOM(_currentmode), _convert_format(type),
                        count);
}

int
pygame_scrap_contains(char *type)
{
    int i = 0;
    char **types = pygame_scrap_get_types();
    while (types[i]) {
        if (strcmp(type, types[i]) == 0)
            return 1;
        i++;
    }
    return 0;
}

char **
pygame_scrap_get_types(void)
{
    char **types;
    Atom *targetdata;
    unsigned long length;

    if (!pygame_scrap_lost()) {
        PyObject *key;
        PyObject *chars;
        Py_ssize_t pos = 0;
        int i = 0;
        PyObject *dict =
            (_currentmode == SCRAP_SELECTION) ? _selectiondata : _clipdata;

        types = malloc(sizeof(char *) * (PyDict_Size(dict) + 1));
        if (!types)
            return NULL;

        memset(types, 0, (size_t)(PyDict_Size(dict) + 1));
        while (PyDict_Next(dict, &pos, &key, NULL)) {
            chars = PyUnicode_AsASCIIString(key);
            if (chars) {
                types[i] = strdup(PyBytes_AsString(chars));
                Py_DECREF(chars);
            }
            else {
                types[i] = NULL;
            }
            if (!types[i]) {
                /* Could not allocate memory, free anything. */
                int j = 0;
                while (types[j]) {
                    free(types[j]);
                    j++;
                }
                free(types);
                return NULL;
            }
            i++;
        }
        types[i] = NULL;
        return types;
    }

    targetdata = (Atom *)_get_data_as(GET_CLIPATOM(_currentmode),
                                      _atom_TARGETS, &length);
    if (length > 0 && targetdata != NULL) {
        Atom *data = targetdata;
        int count = length / sizeof(Atom);
        int i;
        char **targets = malloc(sizeof(char *) * (count + 1));

        if (targets == NULL) {
            free(targetdata);
            return NULL;
        }
        memset(targets, 0, sizeof(char *) * (count + 1));

        for (i = 0; i < count; i++)
            targets[i] = _atom_to_string(data[i]);

        free(targetdata);
        return targets;
    }
    return NULL;
}
