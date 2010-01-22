/*
  pygame - Python Game Library
  Copyright (C) 2000-2001 Pete Shinners, 2008 Marcus von Appen

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
#define PYGAME_SDLCDROM_INTERNAL

#include "pymacros.h"
#include "cdrommod.h"
#include "pgsdl.h"
#include "sdlcdrom_doc.h"

static int _cdrom_clear (PyObject *mod);

static PyObject* _sdl_cdinit (PyObject *self);
static PyObject* _sdl_cdwasinit (PyObject *self);
static PyObject* _sdl_cdquit (PyObject *self);
static PyObject* _sdl_cdnumdrives (PyObject *self);
static PyObject* _sdl_cdgetname (PyObject *self, PyObject *args);

static PyMethodDef _cdrom_methods[] = {
    { "init", (PyCFunction) _sdl_cdinit, METH_NOARGS, DOC_CDROM_INIT },
    { "was_init", (PyCFunction) _sdl_cdwasinit, METH_NOARGS,
      DOC_CDROM_WAS_INIT },
    { "quit", (PyCFunction) _sdl_cdquit, METH_NOARGS, DOC_CDROM_QUIT },
    { "num_drives", (PyCFunction) _sdl_cdnumdrives, METH_NOARGS,
      DOC_CDROM_NUM_DRIVES },
    { "get_name", _sdl_cdgetname, METH_O, DOC_CDROM_GET_NAME },
    { NULL, NULL, 0, NULL }
};

static PyObject*
_sdl_cdinit (PyObject *self)
{
    if (SDL_WasInit (SDL_INIT_CDROM))
        Py_RETURN_NONE;
        
    if (SDL_InitSubSystem (SDL_INIT_CDROM) == -1)
    {
        PyErr_SetString (PyExc_PyGameError, SDL_GetError ());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
_sdl_cdwasinit (PyObject *self)
{
    if (SDL_WasInit (SDL_INIT_CDROM))
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject*
_sdl_cdquit (PyObject *self)
{
    int i;
    _SDLCDromState *state = SDLCDROM_MOD_STATE(self);
    for (i = 0; i < MAX_CDROMS; i++)
    {
        /* Close all open cdroms. */
        if (state->cdrom_drives[i])
        {
            SDL_CDClose (state->cdrom_drives[i]);
            state->cdrom_drives[i] = NULL;
        }
    }

    if (SDL_WasInit (SDL_INIT_CDROM))
        SDL_QuitSubSystem (SDL_INIT_CDROM);
    Py_RETURN_NONE;
}

static PyObject*
_sdl_cdnumdrives (PyObject *self)
{
    ASSERT_CDROM_INIT (NULL);
    return PyInt_FromLong (SDL_CDNumDrives ());
}

static PyObject*
_sdl_cdgetname (PyObject *self, PyObject *args)
{
    int drive;
    ASSERT_CDROM_INIT (NULL);

    if (!IntFromObj (args, &drive))
        return NULL;
    
    if (drive < 0 || drive >= SDL_CDNumDrives ())
    {
        PyErr_SetString (PyExc_ValueError, "invalid cdrom index");
        return NULL;
    }
    
    return Text_FromUTF8 (SDL_CDName (drive));
}

static int
_cdrom_clear (PyObject *mod)
{
    int i;
    _SDLCDromState *state = SDLCDROM_MOD_STATE(mod);
    for (i = 0; i < MAX_CDROMS; i++)
    {
        /* Close all open cdroms. */
        if (state->cdrom_drives[i])
        {
            SDL_CDClose (state->cdrom_drives[i]);
            state->cdrom_drives[i] = NULL;
        }
    }
    return 0;
}

void
cdrommod_add_drive (int _index, SDL_CD *cdrom)
{
    if (_index < 0 || _index >= MAX_CDROMS)
        return;
    SDLCDROM_STATE->cdrom_drives[_index] = cdrom;
}

void
cdrommod_remove_drive (int _index)
{
    if (_index < 0 || _index >= MAX_CDROMS)
        return;
    SDLCDROM_STATE->cdrom_drives[_index] = NULL;
}

SDL_CD*
cdrommod_get_drive (int _index)
{
    if (_index < 0 || _index >= MAX_CDROMS)
        return NULL;
    return SDLCDROM_STATE->cdrom_drives[_index];
}

#ifdef IS_PYTHON_3
struct PyModuleDef _cdrommodule = {
    PyModuleDef_HEAD_INIT,
    "cdrom",
    DOC_CDROM,
    sizeof (_SDLCDromState),
    _cdrom_methods,
    NULL,
    NULL,
    _cdrom_clear,
    NULL
};
#else
_SDLCDromState _modstate;
#endif


#ifdef IS_PYTHON_3
PyMODINIT_FUNC PyInit_cdrom (void)
#else
PyMODINIT_FUNC initcdrom (void)
#endif
{
    int i;
    PyObject *mod = NULL;
    PyObject *c_api_obj;
    _SDLCDromState *state;

    static void *c_api[PYGAME_SDLCDROM_SLOTS];

#ifdef IS_PYTHON_3
    mod = PyModule_Create (&_cdrommodule);
#else
    mod = Py_InitModule3 ("cdrom", _cdrom_methods, DOC_CDROM);
#endif
    if (!mod)
        goto fail;

    state = SDLCDROM_MOD_STATE(mod);
    for (i = 0; i < MAX_CDROMS; i++)
        state->cdrom_drives[i] = NULL;

    
    /* Complete types */
    if (PyType_Ready (&PyCD_Type) < 0)
        goto fail;
    if (PyType_Ready (&PyCDTrack_Type) < 0)
        goto fail;

    ADD_OBJ_OR_FAIL (mod, "CD", PyCD_Type, fail);
    ADD_OBJ_OR_FAIL (mod, "CDTrack", PyCDTrack_Type, fail);

    cdrom_export_capi (c_api);
    cdtrack_export_capi (c_api);

    c_api_obj = PyCObject_FromVoidPtr ((void *) c_api, NULL);
    if (c_api_obj)
    {
        if (PyModule_AddObject (mod, PYGAME_SDLCDROM_ENTRY, c_api_obj) == -1)
        {
            Py_DECREF (c_api_obj);
            goto fail;
        }
    }
    if (import_pygame2_base () < 0)
        goto fail;
    if (import_pygame2_sdl_base () < 0)
        goto fail;
    
    MODINIT_RETURN(mod);
fail:
    Py_XDECREF (mod);
    MODINIT_RETURN (NULL);
}
