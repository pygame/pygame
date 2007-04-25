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

/**
 * Indicates, whether pygame.scrap was initialized or not.
 */
static int _scrapinitialized = 0;

/**
 * Currently active Clipboard object.
 */
static ScrapClipType _currentmode;
static PyObject* _selectiondata;
static PyObject* _clipdata;

/* Forward declarations. */
static PyObject* _scrap_get_types (PyObject *self, PyObject *args);
static PyObject* _scrap_contains (PyObject *self, PyObject *args);
static PyObject* _scrap_get_scrap (PyObject* self, PyObject* args);
static PyObject* _scrap_put_scrap (PyObject* self, PyObject* args);
static PyObject* _scrap_lost_scrap (PyObject* self, PyObject* args);
static PyObject* _scrap_set_mode (PyObject* self, PyObject* args);

/* Determine what type of clipboard we are using */
#if defined(__unix__) && !defined(__QNXNTO__) && !defined(DISABLE_X11)
    #define X11_SCRAP
    #include <time.h> /* Needed for clipboard timeouts. */
    #include "scrap_x11.c"
#elif defined(__WIN32__)
    #define WIN_SCRAP
static UINT _cliptype = 0;
    #include "scrap_win.c"
#elif defined(__QNXNTO__)
    #define QNX_SCRAP
static uint32_t _cliptype = 0;
    #include "scrap_qnx.c"
#elif defined(__APPLE__)
    #define MAC_SCRAP
    #include "scrap_mac.c"
#else
    #error Unknown window manager for clipboard handling
#endif /* scrap type */

/**
 * \brief Indicates whether the scrap module is already initialized.
 *
 * \return 0 if the module is not initialized, 1, if it is.
 */
int
pygame_scrap_initialized (void)
{
    return _scrapinitialized;
}

#if !defined(MAC_SCRAP)

/*DOC*/ static char doc_scrap_init[] = 
/*DOC*/    "scrap.init () -> None\n"
/*DOC*/    "Initializes the scrap module.\n"
/*DOC*/    "\n"
/*DOC*/    "Tries to initialize the scrap module and raises an exception, if\n"
/*DOC*/    "it fails\n";

/*
 * Initializes the pygame scrap module.
 */
static PyObject*
_scrap_init (PyObject *self, PyObject *args)
{
    VIDEO_INIT_CHECK ();
    _clipdata = PyDict_New ();
    _selectiondata = PyDict_New ();

    if (!pygame_scrap_init ())
        return RAISE (PyExc_SDLError, SDL_GetError ());

    Py_RETURN_NONE;
}

/*DOC*/ static char doc_scrap_get_types[] = 
/*DOC*/    "scrap.get_types () -> list\n"
/*DOC*/    "Gets a list of the available clipboard types.\n"
/*DOC*/    "\n"
/*DOC*/    "Gets a list of strings with the identifiers for the available\n"
/*DOC*/    "clipboard types. Each identifier can be used in the scrap.get()\n"
/*DOC*/    "method to get the clipboard content of the specific type.\n"
/*DOC*/    "If there is no data in the clipboard, an empty list is returned.";

/*
 * Gets the currently available types from the active clipboard.
 */
static PyObject*
_scrap_get_types (PyObject *self, PyObject *args)
{
    int i = 0;
    char **types;
    PyObject *list;
    
    PYGAME_SCRAP_INIT_CHECK ();
    if (!pygame_scrap_lost ())
    {
        switch (_currentmode)
        {
        case SCRAP_SELECTION:
            return PyDict_Keys (_selectiondata);
        case SCRAP_CLIPBOARD:
        default:
            return PyDict_Keys (_clipdata);
        }
    }

    list = PyList_New (0);
    types = pygame_scrap_get_types ();
    if (!types)
        return list;
    while (types[i] != NULL)
    {
        PyList_Append (list, PyString_FromString (types[i]));
        i++;
    }
    return list;
}

/*DOC*/ static char doc_scrap_contains[] =
/*DOC*/    "scrap.contains (type) -> bool\n"
/*DOC*/    "Checks, whether a certain type is available in the clipboard.\n"
/*DOC*/    "\n"
/*DOC*/    "Returns True, if data fpr the passed type is available in the\n"
/*DOC*/    "clipboard, False otherwise.";

/*
 * Checks whether the active clipboard contains a certain type.
 */
static PyObject*
_scrap_contains (PyObject *self, PyObject *args)
{
    char *type = NULL;

    if (!PyArg_ParseTuple (args, "s", &type))
        return NULL;
    if (pygame_scrap_contains (type))
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

/*DOC*/ static char doc_scrap_get_scrap[] =
/*DOC*/    "scrap.get (type) -> string\n"
/*DOC*/    "Gets the data for the specified type from the clipboard.\n"
/*DOC*/    "\n"
/*DOC*/    "Returns the data for the specified type from the clipboard.\n"
/*DOC*/    "The data is returned as string and might need further processing.\n"
/*DOC*/    "If no data for the passed type is available, None is returned.";

/*
 * Gets the content for a certain type from the active clipboard.
 */
static PyObject*
_scrap_get_scrap (PyObject* self, PyObject* args)
{
    char *scrap = NULL;
    PyObject *retval;
    char *scrap_type;
    PyObject *val;
    unsigned long count;

    PYGAME_SCRAP_INIT_CHECK ();

    if(!PyArg_ParseTuple (args, "s", &scrap_type))
        return NULL;

    if (!pygame_scrap_lost ())
    {
        /* We are still the active one. */
        switch (_currentmode)
        {
        case SCRAP_SELECTION:
            val = PyDict_GetItemString (_selectiondata, scrap_type);
            break;
        case SCRAP_CLIPBOARD:
        default:
            val = PyDict_GetItemString (_clipdata, scrap_type);
            break;
        }
        Py_XINCREF (val);
        return val;
    }

    /* pygame_get_scrap() only returns NULL or !NULL, but won't set any
     * errors. */
    scrap = pygame_scrap_get (scrap_type, &count);
    if (!scrap)
        Py_RETURN_NONE;

    retval = PyString_FromStringAndSize (scrap, count);
    return retval;
}

/*DOC*/ static char doc_scrap_put_scrap[] =
/*DOC*/    "scrap.put(type, data) -> None\n"
/*DOC*/    "Places data into the clipboard.\n"
/*DOC*/    "\n"
/*DOC*/    "Places data for a specific clipboard type into the clipboard.\n"
/*DOC*/    "The data must be a string buffer.\n"
/*DOC*/    "The method raises an exception, if the content could not be placed\n"
/*DOC*/    "into the clipboard.\n";

/*
 * This will put a python string into the clipboard.
 */
static PyObject*
_scrap_put_scrap (PyObject* self, PyObject* args)
{
    int scraplen;
    char *scrap = NULL;
    char *scrap_type;

    PYGAME_SCRAP_INIT_CHECK ();

    if (!PyArg_ParseTuple (args, "st#", &scrap_type, &scrap, &scraplen))
        return NULL;

    /* Set it in the clipboard. */
    if (!pygame_scrap_put (scrap_type, scraplen, scrap))
        return RAISE (PyExc_SDLError,
                      "content could not be placed in clipboard.");

    /* Add or replace the set value. */
    switch (_currentmode)
    {
    case SCRAP_SELECTION:
        PyDict_SetItemString (_selectiondata, scrap_type,
                              PyString_FromStringAndSize (scrap, scraplen));
        break;
    case SCRAP_CLIPBOARD:
    default:
        PyDict_SetItemString (_clipdata, scrap_type,
                              PyString_FromStringAndSize (scrap, scraplen));
        break;
    }

    Py_RETURN_NONE;
}

/*DOC*/ static char doc_scrap_lost_scrap[] =
/*DOC*/    "scrap.lost() -> bool\n"
/*DOC*/    "Checks whether the clipboard is currently owned by the application\n"
/*DOC*/    "\n"
/*DOC*/    "Returns true, if the clipboard is currently owned by the pygame\n"
/*DOC*/    "application, false otherwise.\n";

/*
 * Checks whether the pygame window has lost the clipboard.
 */
static PyObject*
_scrap_lost_scrap (PyObject* self, PyObject* args)
{
    PYGAME_SCRAP_INIT_CHECK ();

    if (pygame_scrap_lost ())
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

/*DOC*/ static char doc_scrap_set_mode[] =
/*DOC*/    "scrap.set_mode(mode) -> None\n"
/*DOC*/    "Sets the clipboard access mode.\n"
/*DOC*/    "\n"
/*DOC*/    "Sets the access mode for the clipboard. This is only of interest\n"
/*DOC*/    "for X11 environments, where clipboard modes for mouse selections\n"
/*DOC*/    "(SRAP_SELECTION) and the clipboard (SCRAP_CLIPBOARD) are\n"
/*DOC*/    " available. The method does not have any influence on other\n"
/*DOC*/    " environments.";

/*
 * Sets the clipboard mode. This only works for the X11 environment, which
 * diverses between mouse selections and the clipboard.
 */
static PyObject*
_scrap_set_mode (PyObject* self, PyObject* args)
{
    PYGAME_SCRAP_INIT_CHECK ();
    if (!PyArg_ParseTuple (args, "i", &_currentmode))
        return NULL;

#ifndef X11_SCRAP
    /* Force the clipboard, if not in a X11 environment. */
    _currentmode = SCRAP_CLIPBOARD;
#endif
    Py_RETURN_NONE;
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
#if (defined(X11_SCRAP) || defined(WIN_SCRAP) || defined(QNX_SCRAP) \
     || defined(MAC_SCRAP))

    { "init", _scrap_init, 1, doc_scrap_init },
    { "contains", _scrap_contains, METH_VARARGS, doc_scrap_contains, },
    { "get", _scrap_get_scrap, METH_VARARGS, doc_scrap_get_scrap },
    { "get_types", _scrap_get_types, METH_NOARGS, doc_scrap_get_types },
    { "put", _scrap_put_scrap, METH_VARARGS, doc_scrap_put_scrap },
    { "lost", _scrap_lost_scrap, METH_NOARGS, doc_scrap_lost_scrap },
    { "set_mode", _scrap_set_mode, METH_VARARGS, doc_scrap_set_mode },

#endif
    { NULL, NULL, 0, NULL}
};

PYGAME_EXPORT
void initscrap (void)
{
    PyObject *mod;

    /* create the module */
    mod = Py_InitModule3 ("scrap", scrap_builtins, NULL);

    /*imported needed apis*/
    import_pygame_base ();
}
