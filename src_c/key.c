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
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = _PG_SCANCODEWRAPPER_TYPE_FULLNAME,
    .tp_repr = (reprfunc)pg_scancodewrapper_repr,
    .tp_as_mapping = &pg_scancodewrapper_mapping,
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

/* Keep our own table for backwards compatibility. This table is directly taken
 * from SDL2 source (but some SDL1 names are patched in it at runtime).
 * This has to be kept updated (only new things can be added, existing names in
 * this must not be changed) */
static const char *SDL1_scancode_names[SDL_NUM_SCANCODES] = {
    NULL,
    NULL,
    NULL,
    NULL,
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "0",
    "Return",
    "Escape",
    "Backspace",
    "Tab",
    "Space",
    "-",
    "=",
    "[",
    "]",
    "\\",
    "#",
    ";",
    "'",
    "`",
    ",",
    ".",
    "/",
    "CapsLock",
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "F6",
    "F7",
    "F8",
    "F9",
    "F10",
    "F11",
    "F12",
    "PrintScreen",
    "ScrollLock",
    "Pause",
    "Insert",
    "Home",
    "PageUp",
    "Delete",
    "End",
    "PageDown",
    "Right",
    "Left",
    "Down",
    "Up",
    "Numlock",
    "Keypad /",
    "Keypad *",
    "Keypad -",
    "Keypad +",
    "Keypad Enter",
    "Keypad 1",
    "Keypad 2",
    "Keypad 3",
    "Keypad 4",
    "Keypad 5",
    "Keypad 6",
    "Keypad 7",
    "Keypad 8",
    "Keypad 9",
    "Keypad 0",
    "Keypad .",
    NULL,
    "Application",
    "Power",
    "Keypad =",
    "F13",
    "F14",
    "F15",
    "F16",
    "F17",
    "F18",
    "F19",
    "F20",
    "F21",
    "F22",
    "F23",
    "F24",
    "Execute",
    "Help",
    "Menu",
    "Select",
    "Stop",
    "Again",
    "Undo",
    "Cut",
    "Copy",
    "Paste",
    "Find",
    "Mute",
    "VolumeUp",
    "VolumeDown",
    NULL,
    NULL,
    NULL,
    "Keypad ,",
    "Keypad = (AS400)",
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    "AltErase",
    "SysReq",
    "Cancel",
    "Clear",
    "Prior",
    "Return",
    "Separator",
    "Out",
    "Oper",
    "Clear / Again",
    "CrSel",
    "ExSel",
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    "Keypad 00",
    "Keypad 000",
    "ThousandsSeparator",
    "DecimalSeparator",
    "CurrencyUnit",
    "CurrencySubUnit",
    "Keypad (",
    "Keypad )",
    "Keypad {",
    "Keypad }",
    "Keypad Tab",
    "Keypad Backspace",
    "Keypad A",
    "Keypad B",
    "Keypad C",
    "Keypad D",
    "Keypad E",
    "Keypad F",
    "Keypad XOR",
    "Keypad ^",
    "Keypad %",
    "Keypad <",
    "Keypad >",
    "Keypad &",
    "Keypad &&",
    "Keypad |",
    "Keypad ||",
    "Keypad :",
    "Keypad #",
    "Keypad Space",
    "Keypad @",
    "Keypad !",
    "Keypad MemStore",
    "Keypad MemRecall",
    "Keypad MemClear",
    "Keypad MemAdd",
    "Keypad MemSubtract",
    "Keypad MemMultiply",
    "Keypad MemDivide",
    "Keypad +/-",
    "Keypad Clear",
    "Keypad ClearEntry",
    "Keypad Binary",
    "Keypad Octal",
    "Keypad Decimal",
    "Keypad Hexadecimal",
    NULL,
    NULL,
    "Left Ctrl",
    "Left Shift",
    "Left Alt",
    "Left GUI",
    "Right Ctrl",
    "Right Shift",
    "Right Alt",
    "Right GUI",
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    "ModeSwitch",
    "AudioNext",
    "AudioPrev",
    "AudioStop",
    "AudioPlay",
    "AudioMute",
    "MediaSelect",
    "WWW",
    "Mail",
    "Calculator",
    "Computer",
    "AC Search",
    "AC Home",
    "AC Back",
    "AC Forward",
    "AC Stop",
    "AC Refresh",
    "AC Bookmarks",
    "BrightnessDown",
    "BrightnessUp",
    "DisplaySwitch",
    "KBDIllumToggle",
    "KBDIllumDown",
    "KBDIllumUp",
    "Eject",
    "Sleep",
    "App1",
    "App2",
    "AudioRewind",
    "AudioFastForward",
};

/* Taken from SDL_iconv() */
char *
pg_UCS4ToUTF8(Uint32 ch, char *dst)
{
    Uint8 *p = (Uint8 *)dst;
    if (ch <= 0x7F) {
        *p = (Uint8)ch;
        ++dst;
    }
    else if (ch <= 0x7FF) {
        p[0] = 0xC0 | (Uint8)((ch >> 6) & 0x1F);
        p[1] = 0x80 | (Uint8)(ch & 0x3F);
        dst += 2;
    }
    else if (ch <= 0xFFFF) {
        p[0] = 0xE0 | (Uint8)((ch >> 12) & 0x0F);
        p[1] = 0x80 | (Uint8)((ch >> 6) & 0x3F);
        p[2] = 0x80 | (Uint8)(ch & 0x3F);
        dst += 3;
    }
    else {
        p[0] = 0xF0 | (Uint8)((ch >> 18) & 0x07);
        p[1] = 0x80 | (Uint8)((ch >> 12) & 0x3F);
        p[2] = 0x80 | (Uint8)((ch >> 6) & 0x3F);
        p[3] = 0x80 | (Uint8)(ch & 0x3F);
        dst += 4;
    }
    return dst;
}

/* Patch in pygame 1 compat names in our key name compat table */
static void
_use_sdl1_key_names(void)
{
    /* mostly copied from SDL_keyboard.c
     * ASCII keys are already handled correctly, it does not break anything
     * if some of these keys have missing keycodes */

    /* These are specialcases in the _get_keycode_name implementation */
    SDL1_scancode_names[SDL_SCANCODE_BACKSPACE] = "backspace";
    SDL1_scancode_names[SDL_SCANCODE_TAB] = "tab";
    SDL1_scancode_names[SDL_SCANCODE_RETURN] = "return";
    SDL1_scancode_names[SDL_SCANCODE_PAUSE] = "pause";
    SDL1_scancode_names[SDL_SCANCODE_ESCAPE] = "escape";
    SDL1_scancode_names[SDL_SCANCODE_SPACE] = "space";
    SDL1_scancode_names[SDL_SCANCODE_CLEAR] = "clear";
    SDL1_scancode_names[SDL_SCANCODE_DELETE] = "delete";
    SDL1_scancode_names[SDL_SCANCODE_KP_0] = "[0]";
    SDL1_scancode_names[SDL_SCANCODE_KP_1] = "[1]";
    SDL1_scancode_names[SDL_SCANCODE_KP_2] = "[2]";
    SDL1_scancode_names[SDL_SCANCODE_KP_3] = "[3]";
    SDL1_scancode_names[SDL_SCANCODE_KP_4] = "[4]";
    SDL1_scancode_names[SDL_SCANCODE_KP_5] = "[5]";
    SDL1_scancode_names[SDL_SCANCODE_KP_6] = "[6]";
    SDL1_scancode_names[SDL_SCANCODE_KP_7] = "[7]";
    SDL1_scancode_names[SDL_SCANCODE_KP_8] = "[8]";
    SDL1_scancode_names[SDL_SCANCODE_KP_9] = "[9]";
    SDL1_scancode_names[SDL_SCANCODE_KP_PERIOD] = "[.]";
    SDL1_scancode_names[SDL_SCANCODE_KP_DIVIDE] = "[/]";
    SDL1_scancode_names[SDL_SCANCODE_KP_MULTIPLY] = "[*]";
    SDL1_scancode_names[SDL_SCANCODE_KP_MINUS] = "[-]";
    SDL1_scancode_names[SDL_SCANCODE_KP_PLUS] = "[+]";
    SDL1_scancode_names[SDL_SCANCODE_KP_ENTER] = "enter";
    SDL1_scancode_names[SDL_SCANCODE_KP_EQUALS] = "equals";
    SDL1_scancode_names[SDL_SCANCODE_UP] = "up";
    SDL1_scancode_names[SDL_SCANCODE_DOWN] = "down";
    SDL1_scancode_names[SDL_SCANCODE_RIGHT] = "right";
    SDL1_scancode_names[SDL_SCANCODE_LEFT] = "left";
    SDL1_scancode_names[SDL_SCANCODE_DOWN] = "down";
    SDL1_scancode_names[SDL_SCANCODE_INSERT] = "insert";
    SDL1_scancode_names[SDL_SCANCODE_HOME] = "home";
    SDL1_scancode_names[SDL_SCANCODE_END] = "end";
    SDL1_scancode_names[SDL_SCANCODE_PAGEUP] = "page up";
    SDL1_scancode_names[SDL_SCANCODE_PAGEDOWN] = "page down";
    SDL1_scancode_names[SDL_SCANCODE_F1] = "f1";
    SDL1_scancode_names[SDL_SCANCODE_F2] = "f2";
    SDL1_scancode_names[SDL_SCANCODE_F3] = "f3";
    SDL1_scancode_names[SDL_SCANCODE_F4] = "f4";
    SDL1_scancode_names[SDL_SCANCODE_F5] = "f5";
    SDL1_scancode_names[SDL_SCANCODE_F6] = "f6";
    SDL1_scancode_names[SDL_SCANCODE_F7] = "f7";
    SDL1_scancode_names[SDL_SCANCODE_F8] = "f8";
    SDL1_scancode_names[SDL_SCANCODE_F9] = "f9";
    SDL1_scancode_names[SDL_SCANCODE_F10] = "f10";
    SDL1_scancode_names[SDL_SCANCODE_F11] = "f11";
    SDL1_scancode_names[SDL_SCANCODE_F12] = "f12";
    SDL1_scancode_names[SDL_SCANCODE_F13] = "f13";
    SDL1_scancode_names[SDL_SCANCODE_F14] = "f14";
    SDL1_scancode_names[SDL_SCANCODE_F15] = "f15";
    SDL1_scancode_names[SDL_SCANCODE_NUMLOCKCLEAR] = "numlock";
    SDL1_scancode_names[SDL_SCANCODE_CAPSLOCK] = "caps lock";
    SDL1_scancode_names[SDL_SCANCODE_SCROLLLOCK] = "scroll lock";
    SDL1_scancode_names[SDL_SCANCODE_RSHIFT] = "right shift";
    SDL1_scancode_names[SDL_SCANCODE_LSHIFT] = "left shift";
    SDL1_scancode_names[SDL_SCANCODE_RCTRL] = "right ctrl";
    SDL1_scancode_names[SDL_SCANCODE_LCTRL] = "left ctrl";
    SDL1_scancode_names[SDL_SCANCODE_RALT] = "right alt";
    SDL1_scancode_names[SDL_SCANCODE_LALT] = "left alt";
    SDL1_scancode_names[SDL_SCANCODE_RGUI] = "right meta";
    SDL1_scancode_names[SDL_SCANCODE_LGUI] = "left meta";
    SDL1_scancode_names[SDL_SCANCODE_MODE] = "alt gr";
    SDL1_scancode_names[SDL_SCANCODE_APPLICATION] =
        "compose"; /*  Application / Compose / Context Menu (Windows) key */
    SDL1_scancode_names[SDL_SCANCODE_HELP] = "help";
    SDL1_scancode_names[SDL_SCANCODE_PRINTSCREEN] = "print screen";
    SDL1_scancode_names[SDL_SCANCODE_SYSREQ] = "sys req";
    SDL1_scancode_names[SDL_SCANCODE_PAUSE] = "break";
    SDL1_scancode_names[SDL_SCANCODE_MENU] = "menu";
    SDL1_scancode_names[SDL_SCANCODE_POWER] = "power";
    SDL1_scancode_names[SDL_SCANCODE_CURRENCYUNIT] =
        "euro"; /* EURO (SDL1) is an alias for CURRENCYUNIT in SDL2. But we
                need to retain the SDL1 name in compat mode */
    SDL1_scancode_names[SDL_SCANCODE_UNDO] = "undo";
}

static const char *
_get_scancode_name(SDL_Scancode scancode)
{
    /* this only differs SDL_GetScancodeName() in that we use the (mostly)
     * backward-compatible table above */
    const char *name;
    if (((int)scancode) < ((int)SDL_SCANCODE_UNKNOWN) ||
        scancode >= SDL_NUM_SCANCODES) {
        SDL_InvalidParamError("scancode");
        return "";
    }

    name = SDL1_scancode_names[scancode];
    if (name)
        return name;
    else
        return "";
}

static const char *
_get_keycode_name(SDL_Keycode key)
{
    static char name[8];
    char *end;

    if (key & SDLK_SCANCODE_MASK) {
        return _get_scancode_name((SDL_Scancode)(key & ~SDLK_SCANCODE_MASK));
    }

    switch (key) {
        case SDLK_RETURN:
            return _get_scancode_name(SDL_SCANCODE_RETURN);
        case SDLK_ESCAPE:
            return _get_scancode_name(SDL_SCANCODE_ESCAPE);
        case SDLK_BACKSPACE:
            return _get_scancode_name(SDL_SCANCODE_BACKSPACE);
        case SDLK_TAB:
            return _get_scancode_name(SDL_SCANCODE_TAB);
        case SDLK_SPACE:
            return _get_scancode_name(SDL_SCANCODE_SPACE);
        case SDLK_DELETE:
            return _get_scancode_name(SDL_SCANCODE_DELETE);
        default:
            /* SDL2 converts lowercase letters to uppercase here, but we don't
             * for pygame 1 compatibility */
            end = pg_UCS4ToUTF8((Uint32)key, name);
            *end = '\0';
            return name;
    }
}

/* Lighter version of SDL_GetKeyFromName, uses custom compat table first */
static SDL_Keycode
_get_key_from_name(const char *name)
{
    int i;
    for (i = 0; i < SDL_NUM_SCANCODES; ++i) {
        if (!SDL1_scancode_names[i]) {
            continue;
        }
        if (!SDL_strcasecmp(name, SDL1_scancode_names[i])) {
            return SDL_GetKeyFromScancode((SDL_Scancode)i);
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
        return PyUnicode_FromString(_get_keycode_name(key));
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

    code = _get_key_from_name(name);
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

    _use_sdl1_key_names();

    return module;
}
