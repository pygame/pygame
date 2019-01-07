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

#if IS_SDLv1
    if (SDL_EnableKeyRepeat(delay, interval) == -1)
        return RAISE(pgExc_SDLError, SDL_GetError());
#else  /* IS_SDLv2 */
    if (pg_EnableKeyRepeat(delay, interval) == -1)
        return NULL;
#endif /* IS_SDLv2 */

    Py_RETURN_NONE;
}

#if SDL_VERSION_ATLEAST(1, 2, 10)
static PyObject *
key_get_repeat(PyObject *self, PyObject *args)
{
    int delay = 0, interval = 0;

    VIDEO_INIT_CHECK();
#if IS_SDLv1
    SDL_GetKeyRepeat(&delay, &interval);
#else  /* IS_SDLv2 */
    pg_GetKeyRepeat(&delay, &interval);
#endif /* IS_SDLv2 */
    return Py_BuildValue("(ii)", delay, interval);
}
#else  /* not SDL_VERSION_ATLEAST(1, 2, 10) */
static PyObject *
key_get_repeat(PyObject *self, PyObject *args)
{
    Py_RETURN_NONE;
}
#endif /* not SDL_VERSION_ATLEAST(1, 2, 10) */


#if IS_SDLv2
/*
* pgScancodeWrapper is for key_get_pressed in SDL2.
* It converts key symbol indices to scan codes, as suggested in
*     https://github.com/pygame/pygame/issues/659
* so that they work with SDL_GetKeyboardState().
*/
typedef struct {
    PyObject_HEAD
} pgScancodeWrapper;

static PyObject*
pg_scancodewrapper_subscript(pgScancodeWrapper *self, PyObject *item)
{
    long index;
    PyObject *adjustedvalue, *ret;
    if ((index = PyLong_AsLong(item)) == -1 && PyErr_Occurred())
        return NULL;
    index = SDL_GetScancodeFromKey(index);
    adjustedvalue = PyLong_FromLong(index);
    ret = ((PyObject*)self)->ob_type->tp_base->
          tp_as_mapping->mp_subscript(self, adjustedvalue);
    Py_DECREF(adjustedvalue);
    return ret;
}

static PyMappingMethods pg_scancodewrapper_mapping = {
    NULL,
    pg_scancodewrapper_subscript,
    NULL
};

static void
pg_scancodewrapper_dealloc(pgScancodeWrapper *self)
{
    ((PyObject*)self)->ob_type->tp_free(self);
}

static PyObject*
pg_scancodewrapper_repr(pgScancodeWrapper *self)
{
    PyObject *baserepr = ((PyObject*)self)->ob_type->tp_base->tp_repr(self);
#if PY3
    PyObject *ret = Text_FromFormat("pygame._ScancodeWrapper%S", baserepr);
#else /* PY2 */
    PyObject *ret = Text_FromFormat("pygame._ScancodeWrapper%s",
                                    PyString_AsString(baserepr));
#endif /* PY2 */
    Py_DECREF(baserepr);
    return ret;
}

static PyTypeObject pgScancodeWrapper_Type = {
    TYPE_HEAD(NULL, 0) "pygame._ScancodeWrapper", /* name */
    sizeof(pgScancodeWrapper),                    /* basic size */
    0,                                            /* itemsize */
    pg_scancodewrapper_dealloc,                   /* dealloc */
    0,                                            /* print */
    NULL,                                         /* getattr */
    NULL,                                         /* setattr */
    NULL,                                         /* compare */
    pg_scancodewrapper_repr,                      /* repr */
    NULL,                                         /* as_number */
    NULL,                                         /* as_sequence */
    &pg_scancodewrapper_mapping,                  /* as_mapping */
    (hashfunc)NULL,                               /* hash */
    (ternaryfunc)NULL,                            /* call */
    (reprfunc)NULL,                               /* str */
    0,                                            /* tp_getattro */
    0L,
    0L,
    Py_TPFLAGS_DEFAULT |
    Py_TPFLAGS_TUPLE_SUBCLASS,                /* tp_flags */
    NULL,                                     /* Documentation string */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                                        /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    0,                                        /* tp_methods */
    0,                                        /* tp_members */
    0,                                        /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    0,                                        /* tp_init */
    0,                                        /* tp_alloc */
    0                                         /* tp_new */
};
#endif /* IS_SDLv2 */

static PyObject *
key_get_pressed(PyObject *self)
{
    int num_keys;
    Uint8 *key_state;
    PyObject *key_tuple;
    int i;

    VIDEO_INIT_CHECK();

#if IS_SDLv1
    key_state = SDL_GetKeyState(&num_keys);
#else  /* IS_SDLv2 */
    key_state = SDL_GetKeyboardState(&num_keys);
#endif /* IS_SDLv2 */

    if (!key_state || !num_keys)
        Py_RETURN_NONE;

    if (!(key_tuple = PyTuple_New(num_keys)))
        return NULL;

    for (i = 0; i < num_keys; i++) {
        PyObject *key_elem;
        key_elem = PyInt_FromLong(key_state[i]);
        if (!key_elem) {
            Py_DECREF(key_tuple);
            return NULL;
        }
        PyTuple_SET_ITEM(key_tuple, i, key_elem);
    }

#if IS_SDLv1
    return key_tuple;
#else
    return PyObject_CallFunctionObjArgs(&pgScancodeWrapper_Type,
                                        key_tuple, NULL);
#endif
}

#if IS_SDLv2
/* keep our own table for backward compatibility */
static const char *SDL1_scancode_names[SDL_NUM_SCANCODES] = {
    NULL, NULL, NULL, NULL,
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
    NULL, NULL, NULL,
    "Keypad ,",
    "Keypad = (AS400)",
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL, NULL,
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
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
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
    NULL, NULL,
    "Left Ctrl",
    "Left Shift",
    "Left Alt",
    "Left GUI",
    "Right Ctrl",
    "Right Shift",
    "Right Alt",
    "Right GUI",
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
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

static void
_use_sdl1_key_names()
{
    /* mostly copied from SDL_keyboard.c */
    SDL1_scancode_names[SDL_SCANCODE_BACKSPACE] = "backspace";
    SDL1_scancode_names[SDL_SCANCODE_TAB] = "tab";
    SDL1_scancode_names[SDL_SCANCODE_CLEAR] = "clear";
    SDL1_scancode_names[SDL_SCANCODE_RETURN] = "return";
    SDL1_scancode_names[SDL_SCANCODE_PAUSE] = "pause";
    SDL1_scancode_names[SDL_SCANCODE_ESCAPE] = "escape";
    SDL1_scancode_names[SDL_SCANCODE_SPACE] = "space";
    /*SDL1_scancode_names[SDL_SCANCODE_EXCLAIM] = "!";
    SDL1_scancode_names[SDL_SCANCODE_QUOTEDBL] = "\"";
    SDL1_scancode_names[SDL_SCANCODE_HASH] = "#";
    SDL1_scancode_names[SDL_SCANCODE_DOLLAR] = "$";
    SDL1_scancode_names[SDL_SCANCODE_AMPERSAND] = "&";
    SDL1_scancode_names[SDL_SCANCODE_QUOTE] = "'";
    SDL1_scancode_names[SDL_SCANCODE_LEFTPAREN] = "(";
    SDL1_scancode_names[SDL_SCANCODE_RIGHTPAREN] = ")";
    SDL1_scancode_names[SDL_SCANCODE_ASTERISK] = "*";
    SDL1_scancode_names[SDL_SCANCODE_PLUS] = "+";*/ /* these have no scancode */
    SDL1_scancode_names[SDL_SCANCODE_COMMA] = ",";
    SDL1_scancode_names[SDL_SCANCODE_MINUS] = "-";
    SDL1_scancode_names[SDL_SCANCODE_PERIOD] = ".";
    SDL1_scancode_names[SDL_SCANCODE_SLASH] = "/";
    SDL1_scancode_names[SDL_SCANCODE_0] = "0";
    SDL1_scancode_names[SDL_SCANCODE_1] = "1";
    SDL1_scancode_names[SDL_SCANCODE_2] = "2";
    SDL1_scancode_names[SDL_SCANCODE_3] = "3";
    SDL1_scancode_names[SDL_SCANCODE_4] = "4";
    SDL1_scancode_names[SDL_SCANCODE_5] = "5";
    SDL1_scancode_names[SDL_SCANCODE_6] = "6";
    SDL1_scancode_names[SDL_SCANCODE_7] = "7";
    SDL1_scancode_names[SDL_SCANCODE_8] = "8";
    SDL1_scancode_names[SDL_SCANCODE_9] = "9";
    /*SDL1_scancode_names[SDL_SCANCODE_COLON] = ":";*/ /* no scancode */
    SDL1_scancode_names[SDL_SCANCODE_SEMICOLON] = ";";
    /*SDL1_scancode_names[SDL_SCANCODE_LESS] = "<";*/ /* no scancode */
    SDL1_scancode_names[SDL_SCANCODE_EQUALS] = "=";
    /*SDL1_scancode_names[SDL_SCANCODE_GREATER] = ">";
    SDL1_scancode_names[SDL_SCANCODE_QUESTION] = "?";
    SDL1_scancode_names[SDL_SCANCODE_AT] = "@";*/ /* no scancode */
    SDL1_scancode_names[SDL_SCANCODE_LEFTBRACKET] = "[";
    SDL1_scancode_names[SDL_SCANCODE_BACKSLASH] = "\\";
    SDL1_scancode_names[SDL_SCANCODE_RIGHTBRACKET] = "]";
    /*SDL1_scancode_names[SDL_SCANCODE_CARET] = "^";
    SDL1_scancode_names[SDL_SCANCODE_UNDERSCORE] = "_";
    SDL1_scancode_names[SDL_SCANCODE_BACKQUOTE] = "`";*/ /* no scancode */
    SDL1_scancode_names[SDL_SCANCODE_A] = "a";
    SDL1_scancode_names[SDL_SCANCODE_B] = "b";
    SDL1_scancode_names[SDL_SCANCODE_C] = "c";
    SDL1_scancode_names[SDL_SCANCODE_D] = "d";
    SDL1_scancode_names[SDL_SCANCODE_E] = "e";
    SDL1_scancode_names[SDL_SCANCODE_F] = "f";
    SDL1_scancode_names[SDL_SCANCODE_G] = "g";
    SDL1_scancode_names[SDL_SCANCODE_H] = "h";
    SDL1_scancode_names[SDL_SCANCODE_I] = "i";
    SDL1_scancode_names[SDL_SCANCODE_J] = "j";
    SDL1_scancode_names[SDL_SCANCODE_K] = "k";
    SDL1_scancode_names[SDL_SCANCODE_L] = "l";
    SDL1_scancode_names[SDL_SCANCODE_M] = "m";
    SDL1_scancode_names[SDL_SCANCODE_N] = "n";
    SDL1_scancode_names[SDL_SCANCODE_O] = "o";
    SDL1_scancode_names[SDL_SCANCODE_P] = "p";
    SDL1_scancode_names[SDL_SCANCODE_Q] = "q";
    SDL1_scancode_names[SDL_SCANCODE_R] = "r";
    SDL1_scancode_names[SDL_SCANCODE_S] = "s";
    SDL1_scancode_names[SDL_SCANCODE_T] = "t";
    SDL1_scancode_names[SDL_SCANCODE_U] = "u";
    SDL1_scancode_names[SDL_SCANCODE_V] = "v";
    SDL1_scancode_names[SDL_SCANCODE_W] = "w";
    SDL1_scancode_names[SDL_SCANCODE_X] = "x";
    SDL1_scancode_names[SDL_SCANCODE_Y] = "y";
    SDL1_scancode_names[SDL_SCANCODE_Z] = "z";
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
    /*SDL1_scancode_names[SDL_SCANCODE_LSUPER] = "left super";
    SDL1_scancode_names[SDL_SCANCODE_RSUPER] = "right super"; */ /* same as "meta" now? */
    SDL1_scancode_names[SDL_SCANCODE_MODE] = "alt gr";
    SDL1_scancode_names[SDL_SCANCODE_APPLICATION] = "compose"; /*  Application / Compose / Context Menu (Windows) key */

    SDL1_scancode_names[SDL_SCANCODE_HELP] = "help";
    SDL1_scancode_names[SDL_SCANCODE_PRINTSCREEN] = "print screen";
    SDL1_scancode_names[SDL_SCANCODE_SYSREQ] = "sys req";
    SDL1_scancode_names[SDL_SCANCODE_PAUSE] = "break";
    SDL1_scancode_names[SDL_SCANCODE_MENU] = "menu";
    SDL1_scancode_names[SDL_SCANCODE_POWER] = "power";
    /*SDL1_scancode_names[SDL_SCANCODE_EURO] = "euro"; */ /* changed to CurrencyUnit */
    SDL1_scancode_names[SDL_SCANCODE_UNDO] = "undo";
}

static const char *
_get_scancode_name(SDL_Scancode scancode)
{
    /* this only differs SDL_GetScancodeName() in that we use the (mostly) backward-compatible table above */
    const char *name;
    if (((int)scancode) < ((int)SDL_SCANCODE_UNKNOWN) || scancode >= SDL_NUM_SCANCODES) {
          SDL_InvalidParamError("scancode");
          return "";
    }

    name = SDL1_scancode_names[scancode];
    if (name)
        return name;
    else
        return "";
}

/* copied from SDL */
static char *
SDL_UCS4ToUTF8(Uint32 ch, char *dst)
{
    Uint8 *p = (Uint8 *) dst;
    if (ch <= 0x7F) {
        *p = (Uint8) ch;
        ++dst;
    } else if (ch <= 0x7FF) {
        p[0] = 0xC0 | (Uint8) ((ch >> 6) & 0x1F);
        p[1] = 0x80 | (Uint8) (ch & 0x3F);
        dst += 2;
    } else if (ch <= 0xFFFF) {
        p[0] = 0xE0 | (Uint8) ((ch >> 12) & 0x0F);
        p[1] = 0x80 | (Uint8) ((ch >> 6) & 0x3F);
        p[2] = 0x80 | (Uint8) (ch & 0x3F);
        dst += 3;
    } else if (ch <= 0x1FFFFF) {
        p[0] = 0xF0 | (Uint8) ((ch >> 18) & 0x07);
        p[1] = 0x80 | (Uint8) ((ch >> 12) & 0x3F);
        p[2] = 0x80 | (Uint8) ((ch >> 6) & 0x3F);
        p[3] = 0x80 | (Uint8) (ch & 0x3F);
        dst += 4;
    } else if (ch <= 0x3FFFFFF) {
        p[0] = 0xF8 | (Uint8) ((ch >> 24) & 0x03);
        p[1] = 0x80 | (Uint8) ((ch >> 18) & 0x3F);
        p[2] = 0x80 | (Uint8) ((ch >> 12) & 0x3F);
        p[3] = 0x80 | (Uint8) ((ch >> 6) & 0x3F);
        p[4] = 0x80 | (Uint8) (ch & 0x3F);
        dst += 5;
    } else {
        p[0] = 0xFC | (Uint8) ((ch >> 30) & 0x01);
        p[1] = 0x80 | (Uint8) ((ch >> 24) & 0x3F);
        p[2] = 0x80 | (Uint8) ((ch >> 18) & 0x3F);
        p[3] = 0x80 | (Uint8) ((ch >> 12) & 0x3F);
        p[4] = 0x80 | (Uint8) ((ch >> 6) & 0x3F);
        p[5] = 0x80 | (Uint8) (ch & 0x3F);
        dst += 6;
    }
    return dst;
}

static const char *
_get_keycode_name(SDL_Keycode key)
{
#pragma WARN(Add missing keycode names here? )

    static char name[8];
    char *end;

    if (key & SDLK_SCANCODE_MASK) {
        return
            _get_scancode_name((SDL_Scancode) (key & ~SDLK_SCANCODE_MASK));
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
        end = SDL_UCS4ToUTF8((Uint32) key, name);
        *end = '\0';
        return name;
    }
}

#endif /* IS_SDLv2 */

static PyObject *
key_name(PyObject *self, PyObject *args)
{
    int key;

    if (!PyArg_ParseTuple(args, "i", &key))
        return NULL;

#if IS_SDLv2
    return Text_FromUTF8(_get_keycode_name(key));
#else
    return Text_FromUTF8(SDL_GetKeyName(key));
#endif
}

static PyObject *
key_get_mods(PyObject *self)
{
    VIDEO_INIT_CHECK();

    return PyInt_FromLong(SDL_GetModState());
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
key_get_focused(PyObject *self)
{
    VIDEO_INIT_CHECK();

#if IS_SDLv1
    return PyInt_FromLong((SDL_GetAppState() & SDL_APPINPUTFOCUS) != 0);
#else  /* IS_SDLv2 */
    return PyInt_FromLong(SDL_GetKeyboardFocus() != NULL);
#endif /* IS_SDLv2 */
}

static PyObject *
key_start_text_input(PyObject *self)
{
#if IS_SDLv2
    /* https://wiki.libsdl.org/SDL_StartTextInput */
    SDL_StartTextInput();
#endif /* IS_SDLv2 */
    Py_RETURN_NONE;
}

static PyObject *
key_stop_text_input(PyObject *self)
{
#if IS_SDLv2
    /* https://wiki.libsdl.org/SDL_StopTextInput */
    SDL_StopTextInput();
#endif /* IS_SDLv2 */
    Py_RETURN_NONE;
}

static PyObject *
key_set_text_input_rect(PyObject *self, PyObject *obj)
{
    /* https://wiki.libsdl.org/SDL_SetTextInputRect */
#if IS_SDLv2
    SDL_Rect *rect, temp;
    if (obj == Py_None) {
        Py_RETURN_NONE;
    }
    rect = pgRect_FromObject(obj, &temp);
    if (!rect)
        return RAISE(PyExc_TypeError, "Invalid rect argument");
    SDL_SetTextInputRect(rect);
#endif /* IS_SDLv2 */
    Py_RETURN_NONE;
}

static PyMethodDef _key_methods[] = {
    {"set_repeat", key_set_repeat, METH_VARARGS, DOC_PYGAMEKEYSETREPEAT},
    {"get_repeat", key_get_repeat, METH_NOARGS, DOC_PYGAMEKEYGETREPEAT},
    {"get_pressed", (PyCFunction)key_get_pressed, METH_NOARGS,
     DOC_PYGAMEKEYGETPRESSED},
    {"name", key_name, METH_VARARGS, DOC_PYGAMEKEYNAME},
    {"get_mods", (PyCFunction)key_get_mods, METH_NOARGS, DOC_PYGAMEKEYGETMODS},
    {"set_mods", key_set_mods, METH_VARARGS, DOC_PYGAMEKEYSETMODS},
    {"get_focused", (PyCFunction)key_get_focused, METH_NOARGS,
     DOC_PYGAMEKEYGETFOCUSED},
    {"start_text_input", (PyCFunction)key_start_text_input, METH_NOARGS,
     DOC_PYGAMEKEYSTARTTEXTINPUT},
    {"stop_text_input", (PyCFunction)key_stop_text_input, METH_NOARGS,
     DOC_PYGAMEKEYSTOPTEXTINPUT},
    {"set_text_input_rect", (PyCFunction)key_set_text_input_rect, METH_O,
     DOC_PYGAMEKEYSETTEXTINPUTRECT},

    {NULL, NULL, 0, NULL}};

MODINIT_DEFINE(key)
{
    PyObject *module;

#if PY3
    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "key",
                                         DOC_PYGAMEKEY,
                                         -1,
                                         _key_methods,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL};
#endif

    /* imported needed apis; Do this first so if there is an error
       the module is not loaded.
    */
    import_pygame_base();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }
#if IS_SDLv2
    import_pygame_rect();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }
    import_pygame_event();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }
    /* type preparation */
    pgScancodeWrapper_Type.tp_base = &PyTuple_Type;
    if (PyType_Ready(&pgScancodeWrapper_Type) < 0) {
        MODINIT_ERROR;
    }
#endif /* IS_SDLv2 */

    /* create the module */
#if PY3
    module = PyModule_Create(&_module);
#else
    module = Py_InitModule3(MODPREFIX "key", _key_methods, DOC_PYGAMEKEY);
#endif
    if (module == NULL) {
        MODINIT_ERROR;
    }

#if IS_SDLv2
    _use_sdl1_key_names();
#endif /* IS_SDLv2 */

    MODINIT_RETURN(module);
}
