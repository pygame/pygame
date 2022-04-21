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

    PyObject *ret = PyUnicode_FromString(path);
    SDL_free(path);

    return ret;
}

static PyObject *
pg_context_get_pref_locales(PyObject *self, PyObject *_null)
{
    PyObject *ret_list = PyList_New(0);
    if (!ret_list) {
        return NULL;
    }

#if SDL_VERSION_ATLEAST(2, 0, 14)
    PyObject *dict, *val = NULL;
    SDL_Locale *locales = SDL_GetPreferredLocales();
    if (!locales) {
        /* Return an empty list if SDL function does not return any useful
         * information */
        return ret_list;
    }

    SDL_Locale *current_locale = locales;

    /* The array is terminated when the language attribute of the last struct
     * in the array is NULL */
    while (current_locale->language) {
        dict = PyDict_New();
        if (!dict) {
            goto error;
        }

        val = PyUnicode_FromString(current_locale->language);
        if (!val) {
            goto error;
        }
        if (PyDict_SetItemString(dict, "language", val)) {
            goto error;
        }
        Py_DECREF(val);

        if (current_locale->country) {
            val = PyUnicode_FromString(current_locale->country);
            if (!val) {
                goto error;
            }
        }
        else {
            Py_INCREF(Py_None);
            val = Py_None;
        }
        if (PyDict_SetItemString(dict, "country", val)) {
            goto error;
        }
        Py_DECREF(val);

        /* reset val to NULL because goto XDECREF's this */
        val = NULL;
        if (PyList_Append(ret_list, dict)) {
            goto error;
        }
        Py_DECREF(dict);
        current_locale++;
    }

    SDL_free(locales);
    return ret_list;
error:
    Py_XDECREF(val);
    Py_XDECREF(dict);
    SDL_free(locales);
    Py_DECREF(ret_list);
    return NULL;
#else
    return ret_list;
#endif
}

static PyMethodDef _context_methods[] = {
    {"get_pref_path", (PyCFunction)pg_context_get_pref_path,
     METH_VARARGS | METH_KEYWORDS, DOC_PYGAMECONTEXTGETPREFPATH},
    {"get_pref_locales", pg_context_get_pref_locales, METH_NOARGS,
     DOC_PYGAMECONTEXTGETPREFLOCALES},
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
