/*
  pygame - Python Game Library
  Copyright (C) 2000-2001  Pete Shinners

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

  Pete Shinners
  pete@shinners.org
*/

/*
 *  pygame keyboard module
 */
#include "pygame.h"

#include "pgcompat.h"

#include "doc/key_doc.h"

/* keyboard module functions */
static PyObject *
key_set_repeat(PyObject *self, PyObject *args)
{
    int delay = 0, interval = 0;

    if (!PyArg_ParseTuple(args, "|ii", &delay, &interval))
        return NULL;

    VIDEO_INIT_CHECK();

    if (delay && !interval)
        interval = delay;

    if (pg_EnableKeyRepeat(delay, interval) == -1)
        return NULL;

    Py_RETURN_NONE;
}

static PyObject *
key_get_repeat(PyObject *self, PyObject *_null)
{
    int delay = 0, interval = 0;

    VIDEO_INIT_CHECK();
    pg_GetKeyRepeat(&delay, &interval);
    return Py_BuildValue("(ii)", delay, interval);
}

/*
 * pgScancodeWrapper is for key_get_pressed in SDL2.
 * It converts key symbol indices to scan codes, as suggested in
 *     https://github.com/pygame/pygame/issues/659
 * so that they work with SDL_GetKeyboardState().
 */
#define _PG_SCANCODEWRAPPER_TYPE_NAME "ScancodeWrapper"
#define _PG_SCANCODEWRAPPER_TYPE_FULLNAME \
    "pygame.key." _PG_SCANCODEWRAPPER_TYPE_NAME

typedef struct {
    PyTupleObject tuple;
} pgScancodeWrapper;

static PyObject *
pg_scancodewrapper_subscript(pgScancodeWrapper *self, PyObject *item)
{
    long index;
    PyObject *adjustedvalue, *ret;
    if ((index = PyLong_AsLong(item)) == -1 && PyErr_Occurred())
        return NULL;
    index = SDL_GetScancodeFromKey(index);
    adjustedvalue = PyLong_FromLong(index);
    ret = PyTuple_Type.tp_as_mapping->mp_subscript((PyObject *)self,
                                                   adjustedvalue);
    Py_DECREF(adjustedvalue);
    return ret;
}

/*
static PyObject *
pg_iter_raise(PyObject *self)
{
    PyErr_SetString(PyExc_TypeError,
                    "Iterating over key states is not supported");
    return NULL;
}
*/

/**
 * There is an issue in PyPy that causes __iter__ to be called
 * on creation of a ScandcodeWrapper. This stops this from
 * happening.
 */
#ifdef PYPY_VERSION
/*
static PyObject *
pg_scancodewrapper_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds)
{
    PyObject *tuple = NULL;
    Py_ssize_t size = PyTuple_Size(args);
    if (size == 1) {
        tuple = PyTuple_GET_ITEM(args, 0);
        if (PyTuple_Check(tuple)) {
            size = PyTuple_Size(tuple);
        }
        else {
            tuple = NULL;
        }
    }

    pgScancodeWrapper *obj =
        (pgScancodeWrapper *)(subtype->tp_alloc(subtype, size));

    if (obj && tuple) {
        for (Py_ssize_t i = 0; i < size; ++i) {
            PyObject *item = PyTuple_GET_ITEM((PyObject *)tuple, i);
            PyTuple_SET_ITEM((PyObject *)obj, i, item);
        }
        Py_DECREF(tuple);
    }

    return (PyObject *)obj;
}
*/
#endif /* PYPY_VERSION */

static PyMappingMethods pg_scancodewrapper_mapping = {
    .mp_subscript = (binaryfunc)pg_scancodewrapper_subscript,
};

static PyObject *
pg_scancodewrapper_repr(pgScancodeWrapper *self)
{
    PyObject *baserepr = PyTuple_Type.tp_repr((PyObject *)self);
    PyObject *ret =
        PyUnicode_FromFormat(_PG_SCANCODEWRAPPER_TYPE_FULLNAME "%S", baserepr);
    Py_DECREF(baserepr);
    return ret;
}

static PyTypeObject pgScancodeWrapper_Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pygame.key.ScancodeWrapper",
    .tp_repr = (reprfunc)pg_scancodewrapper_repr,
    .tp_as_mapping = &pg_scancodewrapper_mapping,
    /*
        .tp_iter = (getiterfunc)pg_iter_raise,
        .tp_iternext = (iternextfunc)pg_iter_raise,
    #ifdef PYPY_VERSION
        .tp_new = pg_scancodewrapper_new,
    #endif
    */
    .tp_flags =
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_TUPLE_SUBCLASS | Py_TPFLAGS_BASETYPE,
};

static PyObject *
key_get_pressed(PyObject *self, PyObject *_null)
{
    int num_keys;
    const Uint8 *key_state;
    PyObject *ret_obj = NULL;
    PyObject *key_tuple;
    int i;

    VIDEO_INIT_CHECK();

    key_state = SDL_GetKeyboardState(&num_keys);

    if (!key_state || !num_keys)
        Py_RETURN_NONE;

    if (!(key_tuple = PyTuple_New(num_keys)))
        return NULL;

    for (i = 0; i < num_keys; i++) {
        PyObject *key_elem;
        key_elem = PyBool_FromLong(key_state[i]);
        if (!key_elem) {
            Py_DECREF(key_tuple);
            return NULL;
        }
        PyTuple_SET_ITEM(key_tuple, i, key_elem);
    }

    ret_obj = PyObject_CallFunctionObjArgs((PyObject *)&pgScancodeWrapper_Type,
                                           key_tuple, NULL);
    Py_DECREF(key_tuple);
    return ret_obj;
}

/* Keep our own key-name table for backwards compatibility.
 * This has to be kept updated (only new things can be added, existing records
 * in this must not be changed).
 * Here the constant values are hardcoded so that this table remains compatible
 * with older SDL2 versions without the need for many SDL version check macro
 * fences
 */
static const struct {
    const SDL_Keycode key;
    const char *name;
} pg_key_and_name[] = {
    {0, ""},                         /* K_UNKNOWN */
    {8, "backspace"},                /* K_BACKSPACE */
    {9, "tab"},                      /* K_TAB */
    {13, "return"},                  /* K_RETURN */
    {27, "escape"},                  /* K_ESCAPE */
    {32, "space"},                   /* K_SPACE */
    {33, "!"},                       /* K_EXCLAIM */
    {34, "\""},                      /* K_QUOTEDBL */
    {35, "#"},                       /* K_HASH */
    {36, "$"},                       /* K_DOLLAR */
    {37, "%"},                       /* K_PERCENT */
    {38, "&"},                       /* K_AMPERSAND */
    {39, "'"},                       /* K_QUOTE */
    {40, "("},                       /* K_LEFTPAREN */
    {41, ")"},                       /* K_RIGHTPAREN */
    {42, "*"},                       /* K_ASTERISK */
    {43, "+"},                       /* K_PLUS */
    {44, ","},                       /* K_COMMA */
    {45, "-"},                       /* K_MINUS */
    {46, "."},                       /* K_PERIOD */
    {47, "/"},                       /* K_SLASH */
    {48, "0"},                       /* K_0 */
    {49, "1"},                       /* K_1 */
    {50, "2"},                       /* K_2 */
    {51, "3"},                       /* K_3 */
    {52, "4"},                       /* K_4 */
    {53, "5"},                       /* K_5 */
    {54, "6"},                       /* K_6 */
    {55, "7"},                       /* K_7 */
    {56, "8"},                       /* K_8 */
    {57, "9"},                       /* K_9 */
    {58, ":"},                       /* K_COLON */
    {59, ";"},                       /* K_SEMICOLON */
    {60, "<"},                       /* K_LESS */
    {61, "="},                       /* K_EQUALS */
    {62, ">"},                       /* K_GREATER */
    {63, "?"},                       /* K_QUESTION */
    {64, "@"},                       /* K_AT */
    {91, "["},                       /* K_LEFTBRACKET */
    {92, "\\"},                      /* K_BACKSLASH */
    {93, "]"},                       /* K_RIGHTBRACKET */
    {94, "^"},                       /* K_CARET */
    {95, "_"},                       /* K_UNDERSCORE */
    {96, "`"},                       /* K_BACKQUOTE */
    {97, "a"},                       /* K_a */
    {98, "b"},                       /* K_b */
    {99, "c"},                       /* K_c */
    {100, "d"},                      /* K_d */
    {101, "e"},                      /* K_e */
    {102, "f"},                      /* K_f */
    {103, "g"},                      /* K_g */
    {104, "h"},                      /* K_h */
    {105, "i"},                      /* K_i */
    {106, "j"},                      /* K_j */
    {107, "k"},                      /* K_k */
    {108, "l"},                      /* K_l */
    {109, "m"},                      /* K_m */
    {110, "n"},                      /* K_n */
    {111, "o"},                      /* K_o */
    {112, "p"},                      /* K_p */
    {113, "q"},                      /* K_q */
    {114, "r"},                      /* K_r */
    {115, "s"},                      /* K_s */
    {116, "t"},                      /* K_t */
    {117, "u"},                      /* K_u */
    {118, "v"},                      /* K_v */
    {119, "w"},                      /* K_w */
    {120, "x"},                      /* K_x */
    {121, "y"},                      /* K_y */
    {122, "z"},                      /* K_z */
    {127, "delete"},                 /* K_DELETE */
    {1073741881, "caps lock"},       /* K_CAPSLOCK */
    {1073741882, "f1"},              /* K_F1 */
    {1073741883, "f2"},              /* K_F2 */
    {1073741884, "f3"},              /* K_F3 */
    {1073741885, "f4"},              /* K_F4 */
    {1073741886, "f5"},              /* K_F5 */
    {1073741887, "f6"},              /* K_F6 */
    {1073741888, "f7"},              /* K_F7 */
    {1073741889, "f8"},              /* K_F8 */
    {1073741890, "f9"},              /* K_F9 */
    {1073741891, "f10"},             /* K_F10 */
    {1073741892, "f11"},             /* K_F11 */
    {1073741893, "f12"},             /* K_F12 */
    {1073741894, "print screen"},    /* K_PRINT, K_PRINTSCREEN */
    {1073741895, "scroll lock"},     /* K_SCROLLLOCK, K_SCROLLOCK */
    {1073741896, "break"},           /* K_BREAK, K_PAUSE */
    {1073741897, "insert"},          /* K_INSERT */
    {1073741898, "home"},            /* K_HOME */
    {1073741899, "page up"},         /* K_PAGEUP */
    {1073741901, "end"},             /* K_END */
    {1073741902, "page down"},       /* K_PAGEDOWN */
    {1073741903, "right"},           /* K_RIGHT */
    {1073741904, "left"},            /* K_LEFT */
    {1073741905, "down"},            /* K_DOWN */
    {1073741906, "up"},              /* K_UP */
    {1073741907, "numlock"},         /* K_NUMLOCK, K_NUMLOCKCLEAR */
    {1073741908, "[/]"},             /* K_KP_DIVIDE */
    {1073741909, "[*]"},             /* K_KP_MULTIPLY */
    {1073741910, "[-]"},             /* K_KP_MINUS */
    {1073741911, "[+]"},             /* K_KP_PLUS */
    {1073741912, "enter"},           /* K_KP_ENTER */
    {1073741913, "[1]"},             /* K_KP1, K_KP_1 */
    {1073741914, "[2]"},             /* K_KP2, K_KP_2 */
    {1073741915, "[3]"},             /* K_KP3, K_KP_3 */
    {1073741916, "[4]"},             /* K_KP4, K_KP_4 */
    {1073741917, "[5]"},             /* K_KP5, K_KP_5 */
    {1073741918, "[6]"},             /* K_KP6, K_KP_6 */
    {1073741919, "[7]"},             /* K_KP7, K_KP_7 */
    {1073741920, "[8]"},             /* K_KP8, K_KP_8 */
    {1073741921, "[9]"},             /* K_KP9, K_KP_9 */
    {1073741922, "[0]"},             /* K_KP0, K_KP_0 */
    {1073741923, "[.]"},             /* K_KP_PERIOD */
    {1073741926, "power"},           /* K_POWER */
    {1073741927, "equals"},          /* K_KP_EQUALS */
    {1073741928, "f13"},             /* K_F13 */
    {1073741929, "f14"},             /* K_F14 */
    {1073741930, "f15"},             /* K_F15 */
    {1073741941, "help"},            /* K_HELP */
    {1073741942, "menu"},            /* K_MENU */
    {1073741978, "sys req"},         /* K_SYSREQ */
    {1073741980, "clear"},           /* K_CLEAR */
    {1073742004, "euro"},            /* K_CURRENCYUNIT, K_EURO */
    {1073742005, "CurrencySubUnit"}, /* K_CURRENCYSUBUNIT */
    {1073742048, "left ctrl"},       /* K_LCTRL */
    {1073742049, "left shift"},      /* K_LSHIFT */
    {1073742050, "left alt"},        /* K_LALT */
    {1073742051, "left meta"},       /* K_LGUI, K_LMETA, K_LSUPER */
    {1073742052, "right ctrl"},      /* K_RCTRL */
    {1073742053, "right shift"},     /* K_RSHIFT */
    {1073742054, "right alt"},       /* K_RALT */
    {1073742055, "right meta"},      /* K_RGUI, K_RMETA, K_RSUPER */
    {1073742081, "alt gr"},          /* K_MODE */
    {1073742094, "AC Back"},         /* K_AC_BACK */
};

/* Get name from keycode using pygame compat table */
static const char *
_pg_get_keycode_name(SDL_Keycode key)
{
    int i;
    for (i = 0; i < (int)SDL_arraysize(pg_key_and_name); i++) {
        if (pg_key_and_name[i].key == key) {
            return pg_key_and_name[i].name;
        }
    }
    return pg_key_and_name[SDLK_UNKNOWN].name;
}

/* Lighter version of SDL_GetKeyFromName, uses custom compat table first */
static SDL_Keycode
_pg_get_key_from_name(const char *name)
{
    int i;
    for (i = 0; i < (int)SDL_arraysize(pg_key_and_name); i++) {
        if (!SDL_strcasecmp(name, pg_key_and_name[i].name)) {
            return pg_key_and_name[i].key;
        }
    }

    /* fallback to SDL function here */
    return SDL_GetKeyFromName(name);
}

static PyObject *
key_name(PyObject *self, PyObject *args, PyObject *kwargs)
{
    int key, use_compat = 1;
    static char *kwids[] = {"key", "use_compat", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i|p", kwids, &key,
                                     &use_compat))
        return NULL;

    if (use_compat) {
        /* Use our backcompat function, that has names hardcoded in pygame
         * source */
        return PyUnicode_FromString(_pg_get_keycode_name(key));
    }

    /* This check only needs to run when use_compat=False because SDL API calls
     * only happen in this case.
     * This is not checked at the top of this function to not break compat with
     * older API usage that does not expect this function to need init */
    VIDEO_INIT_CHECK();
    return PyUnicode_FromString(SDL_GetKeyName(key));
}

static PyObject *
key_code(PyObject *self, PyObject *args, PyObject *kwargs)
{
    const char *name;
    SDL_Keycode code;

    static char *kwids[] = {"name", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s", kwids, &name))
        return NULL;

    /* in the future, this should be an error. For now it's a warning to not
     * break existing code */
    if (!SDL_WasInit(SDL_INIT_VIDEO)) {
        if (PyErr_WarnEx(PyExc_Warning,
                         "pygame.init() has not been called. This function "
                         "may return incorrect results",
                         1) != 0) {
            return NULL;
        }
    }

    code = _pg_get_key_from_name(name);
    if (code == SDLK_UNKNOWN) {
        return RAISE(PyExc_ValueError, "unknown key name");
    }
    return PyLong_FromLong(code);
}

static PyObject *
key_get_mods(PyObject *self, PyObject *_null)
{
    VIDEO_INIT_CHECK();

    return PyLong_FromLong(SDL_GetModState());
}

static PyObject *
key_set_mods(PyObject *self, PyObject *args)
{
    int mods;

    if (!PyArg_ParseTuple(args, "i", &mods))
        return NULL;

    VIDEO_INIT_CHECK();

    SDL_SetModState(mods);
    Py_RETURN_NONE;
}

static PyObject *
key_get_focused(PyObject *self, PyObject *_null)
{
    VIDEO_INIT_CHECK();

    return PyBool_FromLong(SDL_GetKeyboardFocus() != NULL);
}

static PyObject *
key_start_text_input(PyObject *self, PyObject *_null)
{
    /* https://wiki.libsdl.org/SDL_StartTextInput */
    SDL_StartTextInput();
    Py_RETURN_NONE;
}

static PyObject *
key_stop_text_input(PyObject *self, PyObject *_null)
{
    /* https://wiki.libsdl.org/SDL_StopTextInput */
    SDL_StopTextInput();
    Py_RETURN_NONE;
}

static PyObject *
key_set_text_input_rect(PyObject *self, PyObject *obj)
{
    /* https://wiki.libsdl.org/SDL_SetTextInputRect */
    SDL_Rect *rect, temp;
    SDL_Window *sdlWindow = pg_GetDefaultWindow();
    SDL_Renderer *sdlRenderer = SDL_GetRenderer(sdlWindow);

    if (obj == Py_None) {
        Py_RETURN_NONE;
    }
    rect = pgRect_FromObject(obj, &temp);
    if (!rect)
        return RAISE(PyExc_TypeError, "Invalid rect argument");

    if (sdlRenderer != NULL) {
        SDL_Rect vprect, rect2;
        /* new rect so we do not overwrite the input rect */
        float scalex, scaley;

        SDL_RenderGetScale(sdlRenderer, &scalex, &scaley);
        SDL_RenderGetViewport(sdlRenderer, &vprect);

        rect2.x = (int)(rect->x * scalex + vprect.x);
        rect2.y = (int)(rect->y * scaley + vprect.y);
        rect2.w = (int)(rect->w * scalex);
        rect2.h = (int)(rect->h * scaley);

        SDL_SetTextInputRect(&rect2);
        Py_RETURN_NONE;
    }

    SDL_SetTextInputRect(rect);

    Py_RETURN_NONE;
}

static PyMethodDef _key_methods[] = {
    {"set_repeat", key_set_repeat, METH_VARARGS, DOC_PYGAMEKEYSETREPEAT},
    {"get_repeat", key_get_repeat, METH_NOARGS, DOC_PYGAMEKEYGETREPEAT},
    {"get_pressed", key_get_pressed, METH_NOARGS, DOC_PYGAMEKEYGETPRESSED},
    {"name", (PyCFunction)key_name, METH_VARARGS | METH_KEYWORDS,
     DOC_PYGAMEKEYNAME},
    {"key_code", (PyCFunction)key_code, METH_VARARGS | METH_KEYWORDS,
     DOC_PYGAMEKEYKEYCODE},
    {"get_mods", key_get_mods, METH_NOARGS, DOC_PYGAMEKEYGETMODS},
    {"set_mods", key_set_mods, METH_VARARGS, DOC_PYGAMEKEYSETMODS},
    {"get_focused", key_get_focused, METH_NOARGS, DOC_PYGAMEKEYGETFOCUSED},
    {"start_text_input", key_start_text_input, METH_NOARGS,
     DOC_PYGAMEKEYSTARTTEXTINPUT},
    {"stop_text_input", key_stop_text_input, METH_NOARGS,
     DOC_PYGAMEKEYSTOPTEXTINPUT},
    {"set_text_input_rect", key_set_text_input_rect, METH_O,
     DOC_PYGAMEKEYSETTEXTINPUTRECT},

    {NULL, NULL, 0, NULL}};

MODINIT_DEFINE(key)
{
    PyObject *module;

    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "key",
                                         DOC_PYGAMEKEY,
                                         -1,
                                         _key_methods,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL};

    /* imported needed apis; Do this first so if there is an error
       the module is not loaded.
    */
    import_pygame_base();
    if (PyErr_Occurred()) {
        return NULL;
    }
    import_pygame_rect();
    if (PyErr_Occurred()) {
        return NULL;
    }
    import_pygame_event();
    if (PyErr_Occurred()) {
        return NULL;
    }
    /* type preparation */
    pgScancodeWrapper_Type.tp_base = &PyTuple_Type;
    if (PyType_Ready(&pgScancodeWrapper_Type) < 0) {
        return NULL;
    }

    /* create the module */
    module = PyModule_Create(&_module);
    if (module == NULL) {
        return NULL;
    }

    Py_INCREF(&pgScancodeWrapper_Type);
    if (PyModule_AddObject(module, _PG_SCANCODEWRAPPER_TYPE_NAME,
                           (PyObject *)&pgScancodeWrapper_Type)) {
        Py_DECREF(&pgScancodeWrapper_Type);
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
