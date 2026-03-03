#include "pygame.h"

#include "pgcompat.h"

#include "doc/system_doc.h"

static PyObject *
pg_system_get_cpu_instruction_sets(PyObject *self, PyObject *_null)
{
    PyObject *tmp_bool = NULL;
    PyObject *instruction_sets = NULL;

    instruction_sets = PyDict_New();
    if (!instruction_sets) {
        goto error;
    }

#define INSERT_INSTRUCTIONSET_INFO(name, SDL_fn_name)               \
    tmp_bool = PyBool_FromLong((SDL_fn_name)());                    \
    if (PyDict_SetItemString(instruction_sets, (name), tmp_bool)) { \
        Py_DECREF(tmp_bool);                                        \
        goto error;                                                 \
    }                                                               \
    Py_DECREF(tmp_bool);

    // Don't insert SDL_HasRDTSC because it's been removed in SDL3
    INSERT_INSTRUCTIONSET_INFO("ALTIVEC", SDL_HasAltiVec);
    INSERT_INSTRUCTIONSET_INFO("MMX", SDL_HasMMX);
    INSERT_INSTRUCTIONSET_INFO("SSE", SDL_HasSSE);
    INSERT_INSTRUCTIONSET_INFO("SSE2", SDL_HasSSE2);
    INSERT_INSTRUCTIONSET_INFO("SSE3", SDL_HasSSE3);
    INSERT_INSTRUCTIONSET_INFO("SSE41", SDL_HasSSE41);
    INSERT_INSTRUCTIONSET_INFO("SSE42", SDL_HasSSE42);
    INSERT_INSTRUCTIONSET_INFO("AVX", SDL_HasAVX);
    INSERT_INSTRUCTIONSET_INFO("AVX2", SDL_HasAVX2);
    INSERT_INSTRUCTIONSET_INFO("AVX512F", SDL_HasAVX512F);
    INSERT_INSTRUCTIONSET_INFO("NEON", SDL_HasNEON);
    INSERT_INSTRUCTIONSET_INFO("ARMSIMD", SDL_HasARMSIMD);
#if SDL_VERSION_ATLEAST(2, 24, 0)
    INSERT_INSTRUCTIONSET_INFO("LSX", SDL_HasLSX);
    INSERT_INSTRUCTIONSET_INFO("LASX", SDL_HasLASX);
#else
    if (PyDict_SetItemString(instruction_sets, "LSX", Py_False)) {
        goto error;
    }
    if (PyDict_SetItemString(instruction_sets, "LASX", Py_False)) {
        goto error;
    }
#endif

#undef INSERT_INSTRUCTIONSET_INFO

    return instruction_sets;

error:
    Py_XDECREF(instruction_sets);
    return NULL;
}

static PyObject *
pg_system_get_total_ram(PyObject *self, PyObject *_null)
{
    return PyLong_FromLong(SDL_GetSystemRAM());
}

static PyObject *
pg_system_get_pref_path(PyObject *self, PyObject *args, PyObject *kwargs)
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
pg_system_get_pref_locales(PyObject *self, PyObject *_null)
{
    PyObject *ret_list = PyList_New(0);
    if (!ret_list) {
        return NULL;
    }

    // Sorry about the SDL3 gnarliness here, this was the best way I could
    // think of to support SDL2/SDL3 at once. The approach is that each
    // version is responsible for coming up with a list and a count,
    // then the iteration over the list is shared (except for the indexing
    // strategy, where SDL2/3 are different)

    PyObject *dict, *val = NULL;
    int num_locales;
    SDL_Locale *current_locale;

#if SDL_VERSION_ATLEAST(3, 0, 0)
    SDL_Locale **locales = SDL_GetPreferredLocales(&num_locales);
    if (!locales) {
        /* Return an empty list if SDL function does not return any useful
         * information */
        return ret_list;
    }
#else
    SDL_Locale *locales = SDL_GetPreferredLocales();
    if (!locales) {
        /* Return an empty list if SDL function does not return any useful
         * information */
        return ret_list;
    }

    num_locales = 0;
    current_locale = locales;
    /* The array is terminated when the language attribute of the last struct
     * in the array is NULL */
    while (current_locale->language) {
        num_locales++;
        current_locale++;
    }
#endif

    for (int i = 0; i < num_locales; i++) {
        dict = PyDict_New();
        if (!dict) {
            goto error;
        }

#if SDL_VERSION_ATLEAST(3, 0, 0)
        current_locale = locales[i];
#else
        current_locale = locales + i;
#endif
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
    }

    SDL_free(locales);
    return ret_list;
error:
    Py_XDECREF(val);
    Py_XDECREF(dict);
    SDL_free(locales);
    Py_DECREF(ret_list);
    return NULL;
}

static PyObject *PowerState_class = NULL;

static PyObject *
pg_system_get_power_state(PyObject *self, PyObject *_null)
{
    int sec, pct;
    SDL_PowerState power_state;
    PyObject *return_args;
    PyObject *return_kwargs;
    PyObject *sec_py, *pct_py;

    power_state = SDL_GetPowerInfo(&sec, &pct);

    if (power_state == SDL_POWERSTATE_UNKNOWN) {
        Py_RETURN_NONE;
    }

    if (sec == -1) {
        sec_py = Py_None;
        Py_INCREF(Py_None);
    }
    else {
        sec_py = PyLong_FromLong(sec);
    }

    if (pct == -1) {
        pct_py = Py_None;
        Py_INCREF(Py_None);
    }
    else {
        pct_py = PyLong_FromLong(pct);
    }
    // Error check will be done in Py_BuildValue

    int on_battery = (power_state == SDL_POWERSTATE_ON_BATTERY);
    int no_battery = (power_state == SDL_POWERSTATE_NO_BATTERY);
    int charging = (power_state == SDL_POWERSTATE_CHARGING);
    int charged = (power_state == SDL_POWERSTATE_CHARGED);

    // clang-format off
    return_kwargs = Py_BuildValue(
        "{s:N,s:N,s:N,s:N,s:N,s:N,s:N,s:N}",
        "battery_percent", pct_py,
        "battery_seconds", sec_py,
        "on_battery", PyBool_FromLong(on_battery),
        "no_battery", PyBool_FromLong(no_battery),
        "charging", PyBool_FromLong(charging),
        "charged", PyBool_FromLong(charged),
        "plugged_in", PyBool_FromLong(!on_battery),
        "has_battery", PyBool_FromLong(on_battery || !no_battery)
    );
    // clang-format on

    if (!return_kwargs) {
        return NULL;
    }

    return_args = Py_BuildValue("()");

    if (!return_args) {
        return NULL;
    }

    if (!PowerState_class) {
        return RAISE(PyExc_SystemError, "PowerState class is not imported.");
    }

    return PyObject_Call(PowerState_class, return_args, return_kwargs);
}

static PyMethodDef _system_methods[] = {
    {"get_cpu_instruction_sets", pg_system_get_cpu_instruction_sets,
     METH_NOARGS, DOC_SYSTEM_GETCPUINSTRUCTIONSETS},
    {"get_total_ram", pg_system_get_total_ram, METH_NOARGS,
     DOC_SYSTEM_GETTOTALRAM},
    {"get_pref_path", (PyCFunction)pg_system_get_pref_path,
     METH_VARARGS | METH_KEYWORDS, DOC_SYSTEM_GETPREFPATH},
    {"get_pref_locales", pg_system_get_pref_locales, METH_NOARGS,
     DOC_SYSTEM_GETPREFLOCALES},
    {"get_power_state", pg_system_get_power_state, METH_NOARGS,
     DOC_SYSTEM_GETPOWERSTATE},
    {NULL, NULL, 0, NULL}};

MODINIT_DEFINE(system)
{
    PyObject *module;
    static struct PyModuleDef _module = {
        .m_base = PyModuleDef_HEAD_INIT,
        .m_name = "system",
        .m_doc = DOC_SYSTEM,
        .m_size = -1,
        .m_methods = _system_methods,
    };

    /* need to import base module, just so SDL is happy. Do this first so if
       the module is there is an error the module is not loaded.
    */
    import_pygame_base();
    if (PyErr_Occurred()) {
        return NULL;
    }

    PyObject *data_classes_module =
        PyImport_ImportModule("pygame._data_classes");
    if (!data_classes_module) {
        return NULL;
    }

    PowerState_class =
        PyObject_GetAttrString(data_classes_module, "PowerState");
    if (!PowerState_class) {
        return NULL;
    }
    Py_DECREF(data_classes_module);

    /* create the module */
    module = PyModule_Create(&_module);
    if (!module) {
        return NULL;
    }

    if (PyModule_AddObject(module, "PowerState", PowerState_class)) {
        Py_DECREF(PowerState_class);
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
