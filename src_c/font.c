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
 *  font module for pygame
 */
#define PYGAMEAPI_FONT_INTERNAL
#include "font.h"

#include <stdio.h>
#include <string.h>

#include "pygame.h"

#include "pgcompat.h"

#include "doc/font_doc.h"

#include "structmember.h"

#ifndef SDL_TTF_VERSION_ATLEAST
#define SDL_TTF_COMPILEDVERSION                                  \
    SDL_VERSIONNUM(SDL_TTF_MAJOR_VERSION, SDL_TTF_MINOR_VERSION, \
                   SDL_TTF_PATCHLEVEL)
#define SDL_TTF_VERSION_ATLEAST(X, Y, Z) \
    (SDL_TTF_COMPILEDVERSION >= SDL_VERSIONNUM(X, Y, Z))
#endif

#define RAISE_TEXT_TYPE_ERROR() \
    RAISE(PyExc_TypeError, "text must be a unicode or bytes");

/* For filtering out UCS-4 and larger characters when Python is
 * built with Py_UNICODE_WIDE.
 */
#if defined(PYPY_VERSION)
#define Py_UNICODE_IS_SURROGATE(ch) (0xD800 <= (ch) && (ch) <= 0xDFFF)
#endif

static PyTypeObject PyFont_Type;
static PyObject *
PyFont_New(TTF_Font *);
#define PyFont_Check(x) ((x)->ob_type == &PyFont_Type)

static int font_initialized = 0;
static unsigned int current_ttf_generation = 0;
static const char font_defaultname[] = "freesansbold.ttf";
static const char pkgdatamodule_name[] = "pygame.pkgdata";
static const char resourcefunc_name[] = "getResource";

/*
 */
#if !SDL_TTF_VERSION_ATLEAST(2, 0, 15)
static int
utf_8_needs_UCS_4(const char *str)
{
    static const Uint8 first = '\xF0';

    while (*str) {
        if ((Uint8)*str >= first) {
            return 1;
        }
        ++str;
    }
    return 0;
}
#endif

/* Return an encoded file path, a file-like object or a NULL pointer.
 * May raise a Python error. Use PyErr_Occurred to check.
 */
static PyObject *
font_resource(const char *filename)
{
    PyObject *pkgdatamodule = NULL;
    PyObject *resourcefunc = NULL;
    PyObject *result = NULL;
    PyObject *tmp;

    pkgdatamodule = PyImport_ImportModule(pkgdatamodule_name);
    if (pkgdatamodule == NULL) {
        return NULL;
    }

    resourcefunc = PyObject_GetAttrString(pkgdatamodule, resourcefunc_name);
    Py_DECREF(pkgdatamodule);
    if (resourcefunc == NULL) {
        return NULL;
    }

    result = PyObject_CallFunction(resourcefunc, "s", filename);
    Py_DECREF(resourcefunc);
    if (result == NULL) {
        return NULL;
    }

    tmp = PyObject_GetAttrString(result, "name");
    if (tmp != NULL) {
        PyObject *closeret;
        if (!(closeret = PyObject_CallMethod(result, "close", NULL))) {
            Py_DECREF(result);
            Py_DECREF(tmp);
            return NULL;
        }
        Py_DECREF(closeret);
        Py_DECREF(result);
        result = tmp;
    }
    else if (!PyErr_ExceptionMatches(PyExc_MemoryError)) {
        PyErr_Clear();
    }

    tmp = pg_EncodeString(result, "UTF-8", NULL, NULL);
    if (tmp == NULL) {
        Py_DECREF(result);
        return NULL;
    }
    else if (tmp != Py_None) {
        Py_DECREF(result);
        result = tmp;
    }
    else {
        Py_DECREF(tmp);
    }

    return result;
}

static PyObject *
fontmodule_init(PyObject *self)
{
    if (!font_initialized) {
        if (TTF_Init())
            return RAISE(pgExc_SDLError, SDL_GetError());

        font_initialized = 1;
    }
    Py_RETURN_NONE;
}

static PyObject *
fontmodule_quit(PyObject *self)
{
    if (font_initialized) {
        TTF_Quit();
        font_initialized = 0;
        current_ttf_generation++;
    }
    Py_RETURN_NONE;
}

static PyObject *
get_init(PyObject *self)
{
    return PyBool_FromLong(font_initialized);
}

/* font object methods */
static PyObject *
font_get_height(PyObject *self, PyObject *args)
{
    TTF_Font *font = PyFont_AsFont(self);
    return PyLong_FromLong(TTF_FontHeight(font));
}

static PyObject *
font_get_descent(PyObject *self, PyObject *args)
{
    TTF_Font *font = PyFont_AsFont(self);
    return PyLong_FromLong(TTF_FontDescent(font));
}

static PyObject *
font_get_ascent(PyObject *self, PyObject *args)
{
    TTF_Font *font = PyFont_AsFont(self);
    return PyLong_FromLong(TTF_FontAscent(font));
}

static PyObject *
font_get_linesize(PyObject *self, PyObject *args)
{
    TTF_Font *font = PyFont_AsFont(self);
    return PyLong_FromLong(TTF_FontLineSkip(font));
}

static PyObject *
_font_get_style_flag_as_py_bool(PyObject *self, int flag)
{
    TTF_Font *font = PyFont_AsFont(self);
    return PyBool_FromLong((TTF_GetFontStyle(font) & flag) != 0);
}

static void
_font_set_or_clear_style_flag(TTF_Font *font, int flag, int set_flag)
{
    int style = TTF_GetFontStyle(font);
    if (set_flag)
        style |= flag;
    else
        style &= ~flag;
    TTF_SetFontStyle(font, style);
}

/* Implements getter for the bold attribute */
static PyObject *
font_getter_bold(PyObject *self, void *closure)
{
    return _font_get_style_flag_as_py_bool(self, TTF_STYLE_BOLD);
}

/* Implements setter for the bold attribute */
static int
font_setter_bold(PyObject *self, PyObject *value, void *closure)
{
    TTF_Font *font = PyFont_AsFont(self);
    int val;

    DEL_ATTR_NOT_SUPPORTED_CHECK("bold", value);

    val = PyObject_IsTrue(value);
    if (val == -1) {
        return -1;
    }

    _font_set_or_clear_style_flag(font, TTF_STYLE_BOLD, val);
    return 0;
}

/* Implements get_bold() */
static PyObject *
font_get_bold(PyObject *self, PyObject *args)
{
    return _font_get_style_flag_as_py_bool(self, TTF_STYLE_BOLD);
}

/* Implements set_bold(bool) */
static PyObject *
font_set_bold(PyObject *self, PyObject *args)
{
    TTF_Font *font = PyFont_AsFont(self);
    int val;
    if (!PyArg_ParseTuple(args, "p", &val))
        return NULL;

    _font_set_or_clear_style_flag(font, TTF_STYLE_BOLD, val);

    Py_RETURN_NONE;
}

/* Implements getter for the italic attribute */
static PyObject *
font_getter_italic(PyObject *self, void *closure)
{
    return _font_get_style_flag_as_py_bool(self, TTF_STYLE_ITALIC);
}

/* Implements setter for the italic attribute */
static int
font_setter_italic(PyObject *self, PyObject *value, void *closure)
{
    TTF_Font *font = PyFont_AsFont(self);
    int val;

    DEL_ATTR_NOT_SUPPORTED_CHECK("italic", value);

    val = PyObject_IsTrue(value);
    if (val == -1) {
        return -1;
    }

    _font_set_or_clear_style_flag(font, TTF_STYLE_ITALIC, val);
    return 0;
}

/* Implements get_italic() */
static PyObject *
font_get_italic(PyObject *self, PyObject *args)
{
    return _font_get_style_flag_as_py_bool(self, TTF_STYLE_ITALIC);
}

/* Implements set_italic(bool) */
static PyObject *
font_set_italic(PyObject *self, PyObject *args)
{
    TTF_Font *font = PyFont_AsFont(self);
    int val;

    if (!PyArg_ParseTuple(args, "p", &val))
        return NULL;

    _font_set_or_clear_style_flag(font, TTF_STYLE_ITALIC, val);

    Py_RETURN_NONE;
}

/* Implements getter for the underline attribute */
static PyObject *
font_getter_underline(PyObject *self, void *closure)
{
    return _font_get_style_flag_as_py_bool(self, TTF_STYLE_UNDERLINE);
}

/* Implements setter for the underline attribute */
static int
font_setter_underline(PyObject *self, PyObject *value, void *closure)
{
    TTF_Font *font = PyFont_AsFont(self);
    int val;

    DEL_ATTR_NOT_SUPPORTED_CHECK("underline", value);

    val = PyObject_IsTrue(value);
    if (val == -1) {
        return -1;
    }

    _font_set_or_clear_style_flag(font, TTF_STYLE_UNDERLINE, val);
    return 0;
}

/* Implements get_underline() */
static PyObject *
font_get_underline(PyObject *self, PyObject *args)
{
    return _font_get_style_flag_as_py_bool(self, TTF_STYLE_UNDERLINE);
}

/* Implements set_underline(bool) */
static PyObject *
font_set_underline(PyObject *self, PyObject *args)
{
    TTF_Font *font = PyFont_AsFont(self);
    int val;

    if (!PyArg_ParseTuple(args, "p", &val))
        return NULL;

    _font_set_or_clear_style_flag(font, TTF_STYLE_UNDERLINE, val);

    Py_RETURN_NONE;
}

static PyObject *
font_render(PyObject *self, PyObject *args)
{
    TTF_Font *font = PyFont_AsFont(self);
    int aa;
    PyObject *text, *final;
    PyObject *fg_rgba_obj, *bg_rgba_obj = NULL;
    Uint8 rgba[] = {0, 0, 0, 0};
    SDL_Surface *surf;
    SDL_Color foreg, backg;
    int just_return;

    if (!PyArg_ParseTuple(args, "OpO|O", &text, &aa, &fg_rgba_obj,
                          &bg_rgba_obj)) {
        return NULL;
    }

    if (!pg_RGBAFromFuzzyColorObj(fg_rgba_obj, rgba)) {
        /* Exception already set for us */
        return NULL;
    }
    foreg.r = rgba[0];
    foreg.g = rgba[1];
    foreg.b = rgba[2];
    foreg.a = SDL_ALPHA_OPAQUE;
    if (bg_rgba_obj == Py_None) {
        /* Explicit None is the same as not passing a color for us */
        bg_rgba_obj = NULL;
    }
    if (bg_rgba_obj != NULL) {
        if (!pg_RGBAFromFuzzyColorObj(bg_rgba_obj, rgba)) {
            /* Exception already set for us */
            return NULL;
        }
        else {
            backg.r = rgba[0];
            backg.g = rgba[1];
            backg.b = rgba[2];
            backg.a = SDL_ALPHA_OPAQUE;
        }
    }
    else {
        backg.r = 0;
        backg.g = 0;
        backg.b = 0;
        backg.a = SDL_ALPHA_OPAQUE;
    }

    just_return = PyObject_Not(text);
    if (just_return) {
        int height = TTF_FontHeight(font);

        if (just_return == -1 || !(PyUnicode_Check(text) ||
                                   PyBytes_Check(text) || text == Py_None)) {
            PyErr_Clear();
            return RAISE_TEXT_TYPE_ERROR();
        }
        surf = SDL_CreateRGBSurface(SDL_SWSURFACE, 0, height, 32, 0xff << 16,
                                    0xff << 8, 0xff, 0);
        if (surf == NULL) {
            return RAISE(pgExc_SDLError, SDL_GetError());
        }
        if (bg_rgba_obj != NULL) {
            Uint32 c = SDL_MapRGB(surf->format, backg.r, backg.g, backg.b);
            SDL_FillRect(surf, NULL, c);
        }
        else {
            SDL_SetColorKey(surf, SDL_SRCCOLORKEY, 0);
        }
    }
    else if (PyUnicode_Check(text)) {
        PyObject *bytes = PyUnicode_AsEncodedString(text, "utf-8", "replace");
        const char *astring = NULL;

        if (!bytes) {
            return NULL;
        }
        astring = PyBytes_AsString(bytes);
        if (strlen(astring) != (size_t)PyBytes_GET_SIZE(bytes)) {
            Py_DECREF(bytes);
            return RAISE(PyExc_ValueError,
                         "A null character was found in the text");
        }
#if !SDL_TTF_VERSION_ATLEAST(2, 0, 15)
        if (utf_8_needs_UCS_4(astring)) {
            Py_DECREF(bytes);
            return RAISE(PyExc_UnicodeError,
                         "A Unicode character above '\\uFFFF' was found;"
                         " not supported with SDL_ttf version below 2.0.15");
        }
#endif
        if (aa) {
            if (bg_rgba_obj == NULL) {
                surf = TTF_RenderUTF8_Blended(font, astring, foreg);
            }
            else {
                surf = TTF_RenderUTF8_Shaded(font, astring, foreg, backg);
            }
        }
        else {
            surf = TTF_RenderUTF8_Solid(font, astring, foreg);
        }
        Py_DECREF(bytes);
    }
    else if (PyBytes_Check(text)) {
        const char *astring = PyBytes_AsString(text);

        if (strlen(astring) != (size_t)PyBytes_GET_SIZE(text)) {
            return RAISE(PyExc_ValueError,
                         "A null character was found in the text");
        }
        if (aa) {
            if (bg_rgba_obj == NULL) {
                surf = TTF_RenderText_Blended(font, astring, foreg);
            }
            else {
                surf = TTF_RenderText_Shaded(font, astring, foreg, backg);
            }
        }
        else {
            surf = TTF_RenderText_Solid(font, astring, foreg);
        }
    }
    else {
        return RAISE_TEXT_TYPE_ERROR();
    }
    if (surf == NULL) {
        return RAISE(pgExc_SDLError, TTF_GetError());
    }
    if (!aa && (bg_rgba_obj != NULL) && !just_return) {
        /* turn off transparency */
        SDL_SetColorKey(surf, 0, 0);
        surf->format->palette->colors[0].r = backg.r;
        surf->format->palette->colors[0].g = backg.g;
        surf->format->palette->colors[0].b = backg.b;
    }
    final = (PyObject *)pgSurface_New(surf);
    if (final == NULL) {
        SDL_FreeSurface(surf);
    }
    return final;
}

static PyObject *
font_size(PyObject *self, PyObject *args)
{
    TTF_Font *font = PyFont_AsFont(self);
    int w, h;
    PyObject *text;
    const char *string;

    if (!PyArg_ParseTuple(args, "O", &text)) {
        return NULL;
    }

    if (PyUnicode_Check(text)) {
        PyObject *bytes = PyUnicode_AsEncodedString(text, "utf-8", "strict");
        int ecode;

        if (!bytes) {
            return NULL;
        }
        string = PyBytes_AS_STRING(bytes);
        ecode = TTF_SizeUTF8(font, string, &w, &h);
        Py_DECREF(bytes);
        if (ecode) {
            return RAISE(pgExc_SDLError, TTF_GetError());
        }
    }
    else if (PyBytes_Check(text)) {
        string = PyBytes_AS_STRING(text);
        if (TTF_SizeText(font, string, &w, &h)) {
            return RAISE(pgExc_SDLError, TTF_GetError());
        }
    }
    else {
        return RAISE_TEXT_TYPE_ERROR();
    }
    return Py_BuildValue("(ii)", w, h);
}

static PyObject *
font_metrics(PyObject *self, PyObject *args)
{
    TTF_Font *font = PyFont_AsFont(self);
    PyObject *list;
    PyObject *textobj;
    Py_ssize_t length;
    Py_ssize_t i;
    int minx;
    int maxx;
    int miny;
    int maxy;
    int advance;
    PyObject *obj;
    PyObject *listitem;
    Uint16 *buffer;
    Uint16 ch;
    PyObject *temp;
    int surrogate;

    if (!PyArg_ParseTuple(args, "O", &textobj)) {
        return NULL;
    }

    if (PyUnicode_Check(textobj)) {
        obj = textobj;
        Py_INCREF(obj);
    }
    else if (PyBytes_Check(textobj)) {
        obj = PyUnicode_FromEncodedObject(textobj, "UTF-8", NULL);
        if (!obj) {
            return NULL;
        }
    }
    else {
        return RAISE_TEXT_TYPE_ERROR();
    }
    temp = PyUnicode_AsUTF16String(obj);
    Py_DECREF(obj);
    if (!temp)
        return NULL;
    obj = temp;

    list = PyList_New(0);
    if (!list) {
        Py_DECREF(obj);
        return NULL;
    }
    buffer = (Uint16 *)PyBytes_AS_STRING(obj);
    length = PyBytes_GET_SIZE(obj) / sizeof(Uint16);
    for (i = 1 /* skip BOM */; i < length; i++) {
        ch = buffer[i];
        surrogate = Py_UNICODE_IS_SURROGATE(ch);
        /* TODO:
         * TTF_GlyphMetrics() seems to return a value for any character,
         * using the default invalid character, if the char is not found.
         */
        if (!surrogate && /* conditional and */
            !TTF_GlyphMetrics(font, (Uint16)ch, &minx, &maxx, &miny, &maxy,
                              &advance)) {
            listitem =
                Py_BuildValue("(iiiii)", minx, maxx, miny, maxy, advance);
            if (!listitem) {
                Py_DECREF(list);
                Py_DECREF(obj);
                return NULL;
            }
        }
        else {
            /* Not UCS-2 or no matching metrics. */
            Py_INCREF(Py_None);
            listitem = Py_None;
            if (surrogate)
                i++;
        }
        if (0 != PyList_Append(list, listitem)) {
            Py_DECREF(list);
            Py_DECREF(listitem);
            Py_DECREF(obj);
            return NULL; /* Exception already set. */
        }
        Py_DECREF(listitem);
    }
    Py_DECREF(obj);
    return list;
}

/**
 * Getters and setters for the pgFontObject.
 */
static PyGetSetDef font_getsets[] = {
    {"bold", (getter)font_getter_bold, (setter)font_setter_bold, DOC_FONTBOLD,
     NULL},
    {"italic", (getter)font_getter_italic, (setter)font_setter_italic,
     DOC_FONTITALIC, NULL},
    {"underline", (getter)font_getter_underline, (setter)font_setter_underline,
     DOC_FONTUNDERLINE, NULL},
    {NULL, NULL, NULL, NULL, NULL}};

static PyMethodDef font_methods[] = {
    {"get_height", font_get_height, METH_NOARGS, DOC_FONTGETHEIGHT},
    {"get_descent", font_get_descent, METH_NOARGS, DOC_FONTGETDESCENT},
    {"get_ascent", font_get_ascent, METH_NOARGS, DOC_FONTGETASCENT},
    {"get_linesize", font_get_linesize, METH_NOARGS, DOC_FONTGETLINESIZE},

    {"get_bold", font_get_bold, METH_NOARGS, DOC_FONTGETBOLD},
    {"set_bold", font_set_bold, METH_VARARGS, DOC_FONTSETBOLD},
    {"get_italic", font_get_italic, METH_NOARGS, DOC_FONTGETITALIC},
    {"set_italic", font_set_italic, METH_VARARGS, DOC_FONTSETITALIC},
    {"get_underline", font_get_underline, METH_NOARGS, DOC_FONTGETUNDERLINE},
    {"set_underline", font_set_underline, METH_VARARGS, DOC_FONTSETUNDERLINE},

    {"metrics", font_metrics, METH_VARARGS, DOC_FONTMETRICS},
    {"render", font_render, METH_VARARGS, DOC_FONTRENDER},
    {"size", font_size, METH_VARARGS, DOC_FONTSIZE},

    {NULL, NULL, 0, NULL}};

/*font object internals*/
static void
font_dealloc(PyFontObject *self)
{
    TTF_Font *font = PyFont_AsFont(self);
    if (font && font_initialized) {
        if (self->ttf_init_generation != current_ttf_generation) {
            // Since TTF_Font is a private structure
            // it's impossible to access face field in a common way.
            long **face_pp = (long **)font;
            *face_pp = NULL;
        }
        TTF_CloseFont(font);
        self->font = NULL;
    }

    if (self->weakreflist)
        PyObject_ClearWeakRefs((PyObject *)self);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int
font_init(PyFontObject *self, PyObject *args, PyObject *kwds)
{
    int fontsize;
    TTF_Font *font = NULL;
    PyObject *obj;
    SDL_RWops *rw;

    self->font = NULL;
    if (!PyArg_ParseTuple(args, "Oi", &obj, &fontsize)) {
        return -1;
    }

    if (!font_initialized) {
        PyErr_SetString(pgExc_SDLError, "font not initialized");
        return -1;
    }

    /* Incref obj, needs to be decref'd later */
    Py_INCREF(obj);

    if (fontsize <= 1) {
        fontsize = 1;
    }

    if (obj == Py_None) {
        /* default font */
        Py_DECREF(obj);
        obj = font_resource(font_defaultname);
        if (obj == NULL) {
            if (PyErr_Occurred() == NULL) {
                PyErr_Format(PyExc_RuntimeError,
                             "default font '%.1024s' not found",
                             font_defaultname);
            }
            goto error;
        }
        fontsize = (int)(fontsize * .6875);
    }

    rw = pgRWops_FromObject(obj);

    if (rw == NULL && PyUnicode_Check(obj)) {
        if (!PyUnicode_CompareWithASCIIString(obj, font_defaultname)) {
            /* clear out existing file loading error before attempt to get
             * default font */
            PyErr_Clear();
            Py_DECREF(obj);
            obj = font_resource(font_defaultname);
            if (obj == NULL) {
                if (PyErr_Occurred() == NULL) {
                    PyErr_Format(PyExc_RuntimeError,
                                 "default font '%.1024s' not found",
                                 font_defaultname);
                }
                goto error;
            }
            /* Unlike when the default font is loaded with None, the fontsize
             * is not scaled down here. This was probably unintended
             * implementation detail,
             * but this rewritten code aims to keep the exact behavior as the
             * old one */

            rw = pgRWops_FromObject(obj);
        }
    }

    if (rw == NULL) {
        goto error;
    }

    if (fontsize <= 1)
        fontsize = 1;

    Py_BEGIN_ALLOW_THREADS;
    font = TTF_OpenFontRW(rw, 1, fontsize);
    Py_END_ALLOW_THREADS;

    Py_DECREF(obj);
    self->font = font;
    self->ttf_init_generation = current_ttf_generation;

    return 0;

error:
    Py_XDECREF(obj);
    return -1;
}

static PyTypeObject PyFont_Type = {
    PyVarObject_HEAD_INIT(NULL, 0) "pygame.font.Font",
    sizeof(PyFontObject),
    0,
    (destructor)font_dealloc,
    0,
    0, /*getattr*/
    0,
    0,
    0,
    0,
    NULL,
    0,
    (hashfunc)NULL,
    (ternaryfunc)NULL,
    (reprfunc)NULL,
    0L,
    0L,
    0L,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    DOC_PYGAMEFONTFONT,                       /* Documentation string */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                                        /* tp_richcompare */
    offsetof(PyFontObject, weakreflist),      /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    font_methods,                             /* tp_methods */
    0,                                        /* tp_members */
    font_getsets,                             /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    (initproc)font_init,                      /* tp_init */
    0,                                        /* tp_alloc */
    0,                                        /* tp_new */
};

//    PyType_GenericNew,                        /* tp_new */

/*font module methods*/
static PyObject *
get_default_font(PyObject *self)
{
    return PyUnicode_FromString(font_defaultname);
}

static PyMethodDef _font_methods[] = {
    {"init", (PyCFunction)fontmodule_init, METH_NOARGS, DOC_PYGAMEFONTINIT},
    {"quit", (PyCFunction)fontmodule_quit, METH_NOARGS, DOC_PYGAMEFONTQUIT},
    {"get_init", (PyCFunction)get_init, METH_NOARGS, DOC_PYGAMEFONTGETINIT},
    {"get_default_font", (PyCFunction)get_default_font, METH_NOARGS,
     DOC_PYGAMEFONTGETDEFAULTFONT},
    {NULL, NULL, 0, NULL}};

static PyObject *
PyFont_New(TTF_Font *font)
{
    PyFontObject *fontobj;

    if (!font)
        return RAISE(PyExc_RuntimeError, "unable to load font.");
    fontobj = (PyFontObject *)PyFont_Type.tp_new(&PyFont_Type, NULL, NULL);

    if (fontobj)
        fontobj->font = font;

    return (PyObject *)fontobj;
}

MODINIT_DEFINE(font)
{
    PyObject *module, *apiobj;
    static void *c_api[PYGAMEAPI_FONT_NUMSLOTS];

    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "font",
                                         DOC_PYGAMEFONT,
                                         -1,
                                         _font_methods,
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
    import_pygame_color();
    if (PyErr_Occurred()) {
        return NULL;
    }
    import_pygame_surface();
    if (PyErr_Occurred()) {
        return NULL;
    }
    import_pygame_rwobject();
    if (PyErr_Occurred()) {
        return NULL;
    }

    /* type preparation */
    if (PyType_Ready(&PyFont_Type) < 0) {
        return NULL;
    }
    PyFont_Type.tp_new = PyType_GenericNew;

    module = PyModule_Create(&_module);
    if (module == NULL) {
        return NULL;
    }

    Py_INCREF(&PyFont_Type);
    if (PyModule_AddObject(module, "FontType", (PyObject *)&PyFont_Type)) {
        Py_DECREF(&PyFont_Type);
        Py_DECREF(module);
        return NULL;
    }

    Py_INCREF(&PyFont_Type);
    if (PyModule_AddObject(module, "Font", (PyObject *)&PyFont_Type)) {
        Py_DECREF(&PyFont_Type);
        Py_DECREF(module);
        return NULL;
    }

#if SDL_TTF_VERSION_ATLEAST(2, 0, 15)
    /* So people can check for UCS4 support. */
    if (PyModule_AddIntConstant(module, "UCS4", 1)) {
        Py_DECREF(module);
        return NULL;
    }
#endif

    /* export the c api */
    c_api[0] = &PyFont_Type;
    c_api[1] = PyFont_New;
    c_api[2] = &font_initialized;
    apiobj = encapsulate_api(c_api, "font");
    if (PyModule_AddObject(module, PYGAMEAPI_LOCAL_ENTRY, apiobj)) {
        Py_XDECREF(apiobj);
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
