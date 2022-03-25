#include "pygame.h"

#include "pgcompat.h"

#include "doc/context_doc.h"

static PyObject *
pg_context_get_pref_path(PyObject *self, PyObject *args, PyObject *kwargs)
{
    char *org, *project;
    static char *kwids[] = {"org", "app", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ss", kwids, &org,
                                     &project)) {
        return NULL;
    }

    char *path = SDL_GetPrefPath(org, project);
    if (path == NULL) {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

    PyObject *ret = Py_BuildValue("s", path);
    SDL_free(path);

    return ret;
}

static PyMethodDef _context_methods[] = {
    {"get_pref_path", (PyCFunction)pg_context_get_pref_path,
     METH_VARARGS | METH_KEYWORDS, DOC_PYGAMECONTEXTGETPREFPATH},
    {NULL, NULL, 0, NULL}};

MODINIT_DEFINE(context)
{
    PyObject *module;
    static struct PyModuleDef _module = {
        .m_base = PyModuleDef_HEAD_INIT,
        .m_name = "context",
        .m_doc = DOC_PYGAMECONTEXT,
        .m_size = -1,
        .m_methods = _context_methods,
    };

    /* need to import base module, just so SDL is happy. Do this first so if
       the module is there is an error the module is not loaded.
    */
    import_pygame_base();
    if (PyErr_Occurred()) {
        return NULL;
    }

    /* create the module */
    module = PyModule_Create(&_module);
    if (!module) {
        return NULL;
    }

    return module;
}
