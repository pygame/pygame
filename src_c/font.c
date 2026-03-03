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

#define RAISE_TEXT_TYPE_ERROR() \
    RAISE(PyExc_TypeError, "text must be a unicode or bytes");

#define RAISE_FONT_QUIT_ERROR_RETURN(r) \
    RAISERETURN(pgExc_SDLError,         \
                "Invalid font (font module quit since font created)", r)

#define RAISE_FONT_QUIT_ERROR() \
    RAISE(pgExc_SDLError,       \
          "Invalid font (font module quit since font created)");

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

static unsigned int current_ttf_generation = 0;

#define PgFont_GenerationCheck(x) \
    (((PyFontObject *)(x))->ttf_init_generation == current_ttf_generation)

#if defined(BUILD_STATIC)
// SDL_Init + TTF_Init()  are made in main before CPython process the module
// inittab so the emscripten handler knows it will use SDL2 next cycle.
static int font_initialized = 1;
#else
static int font_initialized = 0;
static const char pkgdatamodule_name[] = "pygame.pkgdata";
static const char resourcefunc_name[] = "getResource";
#endif
static const char font_defaultname[] = "freesansbold.ttf";
static const int font_defaultsize = 20;

/* Return an encoded file path, a file-like object or a NULL pointer.
 * May raise a Python error. Use PyErr_Occurred to check.
 */
static PyObject *
font_resource(const char *filename)
{
    PyObject *pkgdatamodule = NULL;
    PyObject *result = NULL;
    PyObject *tmp;

    pkgdatamodule = PyImport_ImportModule(pkgdatamodule_name);
    if (pkgdatamodule == NULL) {
        return NULL;
    }

    result =
        PyObject_CallMethod(pkgdatamodule, resourcefunc_name, "s", filename);
    Py_DECREF(pkgdatamodule);
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
fontmodule_init(PyObject *self, PyObject *_null)
{
    if (!font_initialized) {
#if SDL_TTF_VERSION_ATLEAST(3, 0, 0)
        if (!TTF_Init())
#else
        if (TTF_Init())
#endif
        {
            return RAISE(pgExc_SDLError, SDL_GetError());
        }
        font_initialized = 1;
    }
    Py_RETURN_NONE;
}

static PyObject *
fontmodule_quit(PyObject *self, PyObject *_null)
{
    if (font_initialized) {
        TTF_Quit();
        font_initialized = 0;
        current_ttf_generation++;
    }
    Py_RETURN_NONE;
}

static PyObject *
pg_font_get_init(PyObject *self, PyObject *_null)
{
    return PyBool_FromLong(font_initialized);
}

/* font object methods */
static PyObject *
font_get_height(PyObject *self, PyObject *_null)
{
    if (!PgFont_GenerationCheck(self)) {
        return RAISE_FONT_QUIT_ERROR();
    }

    TTF_Font *font = PyFont_AsFont(self);
#if SDL_TTF_VERSION_ATLEAST(3, 0, 0)
    return PyLong_FromLong(TTF_GetFontHeight(font));
#else
    return PyLong_FromLong(TTF_FontHeight(font));
#endif
}

static PyObject *
font_get_descent(PyObject *self, PyObject *_null)
{
    if (!PgFont_GenerationCheck(self)) {
        return RAISE_FONT_QUIT_ERROR();
    }

    TTF_Font *font = PyFont_AsFont(self);
#if SDL_TTF_VERSION_ATLEAST(3, 0, 0)
    return PyLong_FromLong(TTF_GetFontDescent(font));
#else
    return PyLong_FromLong(TTF_FontDescent(font));
#endif
}

static PyObject *
font_get_ascent(PyObject *self, PyObject *_null)
{
    if (!PgFont_GenerationCheck(self)) {
        return RAISE_FONT_QUIT_ERROR();
    }

    TTF_Font *font = PyFont_AsFont(self);
#if SDL_TTF_VERSION_ATLEAST(3, 0, 0)
    return PyLong_FromLong(TTF_GetFontAscent(font));
#else
    return PyLong_FromLong(TTF_FontAscent(font));
#endif
}

static PyObject *
font_get_linesize(PyObject *self, PyObject *_null)
{
    if (!PgFont_GenerationCheck(self)) {
        return RAISE_FONT_QUIT_ERROR();
    }

    TTF_Font *font = PyFont_AsFont(self);
#if SDL_TTF_VERSION_ATLEAST(3, 0, 0)
    return PyLong_FromLong(TTF_GetFontLineSkip(font));
#else
    return PyLong_FromLong(TTF_FontLineSkip(font));
#endif
}

static PyObject *
font_set_linesize(PyObject *self, PyObject *arg)
{
    if (!PgFont_GenerationCheck(self)) {
        return RAISE_FONT_QUIT_ERROR();
    }

#if SDL_TTF_VERSION_ATLEAST(2, 24, 0)
    TTF_Font *font = PyFont_AsFont(self);

    if (!PyLong_Check(arg)) {
        return RAISE(PyExc_TypeError, "linesize must be an integer");
    }
    int linesize = PyLong_AsLong(arg);
    if (linesize == -1 && PyErr_Occurred()) {
        return NULL;
    }

    if (linesize < 0) {
        return RAISE(PyExc_ValueError, "linesize must be >= 0");
    }

    TTF_SetFontLineSkip(font, linesize);

    Py_RETURN_NONE;
#else
    return RAISE(
        PyExc_NotImplementedError,
        "TTF_SetFontLineSkip is not available in this version of SDL_ttf");
#endif
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
    if (set_flag) {
        style |= flag;
    }
    else {
        style &= ~flag;
    }
    TTF_SetFontStyle(font, style);
}

/* Implements getter for the bold attribute */
static PyObject *
font_getter_bold(PyObject *self, void *closure)
{
    if (!PgFont_GenerationCheck(self)) {
        return RAISE_FONT_QUIT_ERROR();
    }

    return _font_get_style_flag_as_py_bool(self, TTF_STYLE_BOLD);
}

/* Implements setter for the bold attribute */
static int
font_setter_bold(PyObject *self, PyObject *value, void *closure)
{
    if (!PgFont_GenerationCheck(self)) {
        RAISE_FONT_QUIT_ERROR_RETURN(-1);
    }

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
font_get_bold(PyObject *self, PyObject *_null)
{
    if (!PgFont_GenerationCheck(self)) {
        return RAISE_FONT_QUIT_ERROR();
    }

    return _font_get_style_flag_as_py_bool(self, TTF_STYLE_BOLD);
}

/* Implements set_bold(bool) */
static PyObject *
font_set_bold(PyObject *self, PyObject *arg)
{
    if (!PgFont_GenerationCheck(self)) {
        return RAISE_FONT_QUIT_ERROR();
    }

    TTF_Font *font = PyFont_AsFont(self);
    int val = PyObject_IsTrue(arg);
    if (val == -1) {
        return NULL;
    }

    _font_set_or_clear_style_flag(font, TTF_STYLE_BOLD, val);

    Py_RETURN_NONE;
}

/* Implements getter for the italic attribute */
static PyObject *
font_getter_italic(PyObject *self, void *closure)
{
    if (!PgFont_GenerationCheck(self)) {
        return RAISE_FONT_QUIT_ERROR();
    }

    return _font_get_style_flag_as_py_bool(self, TTF_STYLE_ITALIC);
}

/* Implements setter for the italic attribute */
static int
font_setter_italic(PyObject *self, PyObject *value, void *closure)
{
    if (!PgFont_GenerationCheck(self)) {
        RAISE_FONT_QUIT_ERROR_RETURN(-1);
    }

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
font_get_italic(PyObject *self, PyObject *_null)
{
    if (!PgFont_GenerationCheck(self)) {
        return RAISE_FONT_QUIT_ERROR();
    }

    return _font_get_style_flag_as_py_bool(self, TTF_STYLE_ITALIC);
}

/* Implements set_italic(bool) */
static PyObject *
font_set_italic(PyObject *self, PyObject *arg)
{
    if (!PgFont_GenerationCheck(self)) {
        return RAISE_FONT_QUIT_ERROR();
    }

    TTF_Font *font = PyFont_AsFont(self);
    int val = PyObject_IsTrue(arg);
    if (val == -1) {
        return NULL;
    }

    _font_set_or_clear_style_flag(font, TTF_STYLE_ITALIC, val);

    Py_RETURN_NONE;
}

/* Implements getter for the underline attribute */
static PyObject *
font_getter_underline(PyObject *self, void *closure)
{
    if (!PgFont_GenerationCheck(self)) {
        return RAISE_FONT_QUIT_ERROR();
    }

    return _font_get_style_flag_as_py_bool(self, TTF_STYLE_UNDERLINE);
}

/* Implements setter for the underline attribute */
static int
font_setter_underline(PyObject *self, PyObject *value, void *closure)
{
    if (!PgFont_GenerationCheck(self)) {
        RAISE_FONT_QUIT_ERROR_RETURN(-1);
    }

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
font_get_underline(PyObject *self, PyObject *_null)
{
    if (!PgFont_GenerationCheck(self)) {
        return RAISE_FONT_QUIT_ERROR();
    }

    return _font_get_style_flag_as_py_bool(self, TTF_STYLE_UNDERLINE);
}

/* Implements set_underline(bool) */
static PyObject *
font_set_underline(PyObject *self, PyObject *arg)
{
    if (!PgFont_GenerationCheck(self)) {
        return RAISE_FONT_QUIT_ERROR();
    }

    TTF_Font *font = PyFont_AsFont(self);
    int val = PyObject_IsTrue(arg);
    if (val == -1) {
        return NULL;
    }

    _font_set_or_clear_style_flag(font, TTF_STYLE_UNDERLINE, val);

    Py_RETURN_NONE;
}

/* Implements getter for the strikethrough attribute */
static PyObject *
font_getter_strikethrough(PyObject *self, void *closure)
{
    if (!PgFont_GenerationCheck(self)) {
        return RAISE_FONT_QUIT_ERROR();
    }

    return _font_get_style_flag_as_py_bool(self, TTF_STYLE_STRIKETHROUGH);
}

/* Implements setter for the strikethrough attribute */
static int
font_setter_strikethrough(PyObject *self, PyObject *value, void *closure)
{
    if (!PgFont_GenerationCheck(self)) {
        RAISE_FONT_QUIT_ERROR_RETURN(-1);
    }

    TTF_Font *font = PyFont_AsFont(self);
    int val;

    DEL_ATTR_NOT_SUPPORTED_CHECK("strikethrough", value);

    val = PyObject_IsTrue(value);
    if (val == -1) {
        return -1;
    }

    _font_set_or_clear_style_flag(font, TTF_STYLE_STRIKETHROUGH, val);
    return 0;
}

/* Implements getter for the align attribute */
static PyObject *
font_getter_align(PyObject *self, void *closure)
{
    if (!PgFont_GenerationCheck(self)) {
        return RAISE_FONT_QUIT_ERROR();
    }

#if SDL_TTF_VERSION_ATLEAST(3, 0, 0)
    TTF_Font *font = PyFont_AsFont(self);
    return PyLong_FromLong(TTF_GetFontWrapAlignment(font));
#elif SDL_TTF_VERSION_ATLEAST(2, 20, 0)
    TTF_Font *font = PyFont_AsFont(self);
    return PyLong_FromLong(TTF_GetFontWrappedAlign(font));
#else
    return RAISE(pgExc_SDLError,
                 "pygame.font not compiled with a new enough SDL_ttf version. "
                 "Needs SDL_ttf 2.20.0 or above.");
#endif
}

/* Implements setter for the align attribute */
static int
font_setter_align(PyObject *self, PyObject *value, void *closure)
{
    if (!PgFont_GenerationCheck(self)) {
        RAISE_FONT_QUIT_ERROR_RETURN(-1);
    }

#if SDL_TTF_VERSION_ATLEAST(2, 20, 0)
    TTF_Font *font = PyFont_AsFont(self);

    DEL_ATTR_NOT_SUPPORTED_CHECK("align", value);

    long val = PyLong_AsLong(value);
    if (val == -1 && PyErr_Occurred()) {
        PyErr_SetString(
            PyExc_TypeError,
            "font.align must be an integer. "
            "Must correspond with FONT_LEFT, FONT_CENTER, or FONT_RIGHT.");
        return -1;
    }

    if (val < 0 || val > 2) {
        PyErr_SetString(pgExc_SDLError,
                        "font.align must be FONT_LEFT, FONT_CENTER, or "
                        "FONT_RIGHT.");
        return -1;
    }

#if SDL_TTF_VERSION_ATLEAST(3, 0, 0)
    TTF_SetFontWrapAlignment(font, val);
#else
    TTF_SetFontWrappedAlign(font, val);
#endif
    return 0;
#else
    PyErr_SetString(pgExc_SDLError,
                    "pygame.font not compiled with a new enough SDL_ttf "
                    "version. Needs SDL_ttf 2.20.0 or above.");
    return -1;
#endif
}

/* Implements get_strikethrough() */
static PyObject *
font_get_strikethrough(PyObject *self, PyObject *args)
{
    if (!PgFont_GenerationCheck(self)) {
        return RAISE_FONT_QUIT_ERROR();
    }

    return _font_get_style_flag_as_py_bool(self, TTF_STYLE_STRIKETHROUGH);
}

/* Implements set_strikethrough(bool) */
static PyObject *
font_set_strikethrough(PyObject *self, PyObject *arg)
{
    if (!PgFont_GenerationCheck(self)) {
        return RAISE_FONT_QUIT_ERROR();
    }

    TTF_Font *font = PyFont_AsFont(self);
    int val = PyObject_IsTrue(arg);
    if (val == -1) {
        return NULL;
    }

    _font_set_or_clear_style_flag(font, TTF_STYLE_STRIKETHROUGH, val);

    Py_RETURN_NONE;
}

static PyObject *
font_render(PyObject *self, PyObject *args, PyObject *kwds)
{
    if (!PgFont_GenerationCheck(self)) {
        return RAISE_FONT_QUIT_ERROR();
    }

    TTF_Font *font = PyFont_AsFont(self);
    int antialias;
    PyObject *text, *final;
    PyObject *fg_rgba_obj, *bg_rgba_obj = Py_None;
    Uint8 rgba[] = {0, 0, 0, 0};
    SDL_Surface *surf;
    const char *astring = "";
    int wraplength = 0;

    if (!PgFont_GenerationCheck(self)) {
        return RAISE_FONT_QUIT_ERROR()
    }

    static char *kwlist[] = {"text",    "antialias",  "color",
                             "bgcolor", "wraplength", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OpO|Oi", kwlist, &text,
                                     &antialias, &fg_rgba_obj, &bg_rgba_obj,
                                     &wraplength)) {
        return NULL;
    }

    if (!pg_RGBAFromObjEx(fg_rgba_obj, rgba, PG_COLOR_HANDLE_ALL)) {
        /* Exception already set for us */
        return NULL;
    }

    SDL_Color foreg = {rgba[0], rgba[1], rgba[2], SDL_ALPHA_OPAQUE};
    /* might be overridden right below, with an explicit background color */
    SDL_Color backg = {0, 0, 0, SDL_ALPHA_OPAQUE};

    if (bg_rgba_obj != Py_None) {
        if (!pg_RGBAFromObjEx(bg_rgba_obj, rgba, PG_COLOR_HANDLE_ALL)) {
            /* Exception already set for us */
            return NULL;
        }
        backg = (SDL_Color){rgba[0], rgba[1], rgba[2], SDL_ALPHA_OPAQUE};
    }

    if (!PyUnicode_Check(text) && !PyBytes_Check(text) && text != Py_None) {
        return RAISE_TEXT_TYPE_ERROR();
    }

    if (wraplength < 0) {
        return RAISE(PyExc_ValueError,
                     "wraplength parameter must be positive");
    }

    if (PyUnicode_Check(text)) {
        Py_ssize_t _size = -1;
        astring = PyUnicode_AsUTF8AndSize(text, &_size);
        if (astring == NULL) { /* exception already set */
            return NULL;
        }
        if (strlen(astring) != (size_t)_size) {
            return RAISE(PyExc_ValueError,
                         "A null character was found in the text");
        }
    }

    else if (PyBytes_Check(text)) {
        /* Bytes_AsStringAndSize with NULL arg for length emits
           ValueError if internal NULL bytes are present */
        if (PyBytes_AsStringAndSize(text, (char **)&astring, NULL) == -1) {
            return NULL; /* exception already set */
        }
    }

    /* if text is Py_None, leave astring as a null byte to represent 0
       length string */

    if (strlen(astring) == 0) { /* special 0 string case */
#if SDL_TTF_VERSION_ATLEAST(3, 0, 0)
        int height = TTF_GetFontHeight(font);
#else
        int height = TTF_FontHeight(font);
#endif
        surf = PG_CreateSurface(0, height, SDL_PIXELFORMAT_XRGB8888);
    }
    else { /* normal case */
        if (antialias && bg_rgba_obj == Py_None) {
#if SDL_TTF_VERSION_ATLEAST(3, 0, 0)
            surf = TTF_RenderText_Blended_Wrapped(font, astring, 0, foreg,
                                                  wraplength);
#elif SDL_TTF_VERSION_ATLEAST(2, 0, 18)
            surf = TTF_RenderUTF8_Blended_Wrapped(font, astring, foreg,
                                                  wraplength);
#else
            surf = TTF_RenderUTF8_Blended(font, astring, foreg);
#endif
        }
        else if (antialias) {
#if SDL_TTF_VERSION_ATLEAST(3, 0, 0)
            surf = TTF_RenderText_Shaded_Wrapped(font, astring, 0, foreg,
                                                 backg, wraplength);
#elif SDL_TTF_VERSION_ATLEAST(2, 0, 18)
            surf = TTF_RenderUTF8_Shaded_Wrapped(font, astring, foreg, backg,
                                                 wraplength);
#else
            surf = TTF_RenderUTF8_Shaded(font, astring, foreg, backg);
#endif
        }
        else {
#if SDL_TTF_VERSION_ATLEAST(3, 0, 0)
            surf = TTF_RenderText_Solid_Wrapped(font, astring, 0, foreg,
                                                wraplength);
#elif SDL_TTF_VERSION_ATLEAST(2, 0, 18)
            surf =
                TTF_RenderUTF8_Solid_Wrapped(font, astring, foreg, wraplength);
#else
            surf = TTF_RenderUTF8_Solid(font, astring, foreg);
#endif
            /* If an explicit background was provided and the rendering options
            resolve to Render_Solid, that needs to be explicitly handled. */
            if (surf != NULL && bg_rgba_obj != Py_None) {
                SDL_SetColorKey(surf, 0, 0);
                SDL_Palette *palette = PG_GetSurfacePalette(surf);
                if (palette) {
                    palette->colors[0].r = backg.r;
                    palette->colors[0].g = backg.g;
                    palette->colors[0].b = backg.b;
                }
            }
        }
    }

    if (surf == NULL) {
        return RAISE(pgExc_SDLError, TTF_GetError());
    }

    final = (PyObject *)pgSurface_New(surf);
    if (final == NULL) {
        SDL_FreeSurface(surf);
    }
    return final;
}

static PyObject *
font_size(PyObject *self, PyObject *text)
{
    if (!PgFont_GenerationCheck(self)) {
        return RAISE_FONT_QUIT_ERROR();
    }

    TTF_Font *font = PyFont_AsFont(self);
    int w, h;
    const char *string;

    if (!PgFont_GenerationCheck(self)) {
        return RAISE_FONT_QUIT_ERROR();
    }

    if (PyUnicode_Check(text)) {
        PyObject *bytes = PyUnicode_AsEncodedString(text, "utf-8", "strict");
        int ecode;

        if (!bytes) {
            return NULL;
        }
        string = PyBytes_AS_STRING(bytes);
#if SDL_TTF_VERSION_ATLEAST(3, 0, 0)
        ecode = TTF_GetStringSize(font, string, 0, &w, &h) ? 0 : -1;
#else
        ecode = TTF_SizeUTF8(font, string, &w, &h);
#endif
        Py_DECREF(bytes);
        if (ecode) {
            return RAISE(pgExc_SDLError, TTF_GetError());
        }
    }
    else if (PyBytes_Check(text)) {
        string = PyBytes_AS_STRING(text);
#if SDL_TTF_VERSION_ATLEAST(3, 0, 0)
        if (!TTF_GetStringSize(font, string, 0, &w, &h))
#else
        if (TTF_SizeText(font, string, &w, &h))
#endif
        {
            return RAISE(pgExc_SDLError, TTF_GetError());
        }
    }
    else {
        return RAISE_TEXT_TYPE_ERROR();
    }
    return pg_tuple_couple_from_values_int(w, h);
}

static PyObject *
font_getter_point_size(PyFontObject *self, void *closure)
{
    if (!PgFont_GenerationCheck(self)) {
        return RAISE_FONT_QUIT_ERROR();
    }

#if SDL_TTF_VERSION_ATLEAST(2, 0, 18)
    return PyLong_FromLong(self->ptsize);
#else
    return RAISE(pgExc_SDLError,
                 "Incorrect SDL_TTF version (requires 2.0.18)");
#endif
}

static int
font_setter_point_size(PyFontObject *self, PyObject *value, void *closure)
{
    if (!PgFont_GenerationCheck(self)) {
        RAISE_FONT_QUIT_ERROR_RETURN(-1);
    }

#if SDL_TTF_VERSION_ATLEAST(2, 0, 18)
    TTF_Font *font = PyFont_AsFont(self);
    int val = PyLong_AsLong(value);

    if (PyErr_Occurred() && val == -1) {
        return -1;
    }

    if (val <= 0) {
        PyErr_SetString(PyExc_ValueError,
                        "point_size cannot be equal to, or less than 0");
        return -1;
    }

#if SDL_TTF_VERSION_ATLEAST(3, 0, 0)
    /* TODO: can consider supporting float in python API */
    if (!TTF_SetFontSize(font, (float)val))
#else
    if (TTF_SetFontSize(font, val) == -1)
#endif
    {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return -1;
    }
    self->ptsize = val;

    return 0;
#else
    PyErr_SetString(pgExc_SDLError,
                    "Incorrect SDL_TTF version (requires 2.0.18)");
    return -1;
#endif
}

static PyObject *
font_get_ptsize(PyObject *self, PyObject *args)
{
    if (!PgFont_GenerationCheck(self)) {
        return RAISE_FONT_QUIT_ERROR();
    }

#if SDL_TTF_VERSION_ATLEAST(2, 0, 18)
    return PyLong_FromLong(((PyFontObject *)self)->ptsize);
#else
    return RAISE(pgExc_SDLError,
                 "Incorrect SDL_TTF version (requires 2.0.18)");
#endif
}

static PyObject *
font_set_ptsize(PyObject *self, PyObject *arg)
{
    if (!PgFont_GenerationCheck(self)) {
        return RAISE_FONT_QUIT_ERROR();
    }

#if SDL_TTF_VERSION_ATLEAST(2, 0, 18)
    TTF_Font *font = PyFont_AsFont(self);
    int val = PyLong_AsLong(arg);

    if (PyErr_Occurred() && val == -1) {
        return NULL;
    }

    if (val <= 0) {
        return RAISE(PyExc_ValueError,
                     "point_size cannot be equal to, or less than 0");
    }

#if SDL_TTF_VERSION_ATLEAST(3, 0, 0)
    /* TODO: can consider supporting float in python API */
    if (!TTF_SetFontSize(font, (float)val))
#else
    if (TTF_SetFontSize(font, val) == -1)
#endif
    {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }
    ((PyFontObject *)self)->ptsize = val;

    Py_RETURN_NONE;
#else
    return RAISE(pgExc_SDLError,
                 "Incorrect SDL_TTF version (requires 2.0.18)");
#endif
}

static PyObject *
font_getter_outline(PyObject *self, void *closure)
{
    if (!PgFont_GenerationCheck(self)) {
        return RAISE_FONT_QUIT_ERROR();
    }

    TTF_Font *font = PyFont_AsFont(self);
    return PyLong_FromLong(TTF_GetFontOutline(font));
}

static int
font_setter_outline(PyObject *self, PyObject *value, void *closure)
{
    if (!PgFont_GenerationCheck(self)) {
        RAISE_FONT_QUIT_ERROR_RETURN(-1);
    }
    TTF_Font *font = PyFont_AsFont(self);

    DEL_ATTR_NOT_SUPPORTED_CHECK("outline", value);

    long val = PyLong_AsLong(value);
    if (val == -1 && PyErr_Occurred()) {
        return -1;
    }
    if (val < 0) {
        PyErr_SetString(PyExc_ValueError, "outline must be >= 0");
        return -1;
    }

#if SDL_TTF_VERSION_ATLEAST(3, 0, 0)
    if (!TTF_SetFontOutline(font, (int)val)) {
        PyErr_SetString(pgExc_SDLError, SDL_GetError());
        return -1;
    }
#else
    TTF_SetFontOutline(font, (int)val);
#endif
    return 0;
}

static PyObject *
font_getter_name(PyObject *self, void *closure)
{
    if (!PgFont_GenerationCheck(self)) {
        return RAISE_FONT_QUIT_ERROR();
    }

    TTF_Font *font = PyFont_AsFont(self);
#if SDL_TTF_VERSION_ATLEAST(3, 0, 0)
    const char *font_name = TTF_GetFontFamilyName(font);
#else
    const char *font_name = TTF_FontFaceFamilyName(font);
#endif

    return PyUnicode_FromString(font_name ? font_name : "");
}

static PyObject *
font_getter_style_name(PyObject *self, void *closure)
{
    if (!PgFont_GenerationCheck(self)) {
        return RAISE_FONT_QUIT_ERROR();
    }

    TTF_Font *font = PyFont_AsFont(self);
#if SDL_TTF_VERSION_ATLEAST(3, 0, 0)
    const char *font_style_name = TTF_GetFontStyleName(font);
#else
    const char *font_style_name = TTF_FontFaceStyleName(font);
#endif
    return PyUnicode_FromString(font_style_name ? font_style_name : "");
}

static PyObject *
font_metrics(PyObject *self, PyObject *textobj)
{
    if (!PgFont_GenerationCheck(self)) {
        return RAISE_FONT_QUIT_ERROR();
    }

    TTF_Font *font = PyFont_AsFont(self);
    PyObject *list;
    Py_ssize_t i;
    int minx;
    int maxx;
    int miny;
    int maxy;
    int advance;
    PyObject *listitem;
    Py_UCS4 *buffer;
    Py_UCS4 ch;
    if (!PgFont_GenerationCheck(self)) {
        return RAISE_FONT_QUIT_ERROR();
    }

    if (PyUnicode_Check(textobj)) {
        Py_INCREF(textobj);
    }
    else if (PyBytes_Check(textobj)) {
        textobj = PyUnicode_FromEncodedObject(textobj, "UTF-8", NULL);
        if (!textobj) {
            return NULL;
        }
    }
    else {
        return RAISE_TEXT_TYPE_ERROR();
    }
    buffer = PyUnicode_AsUCS4Copy(textobj);
    Py_DECREF(textobj);
    if (!buffer) {
        return NULL;
    }

    list = PyList_New(0);
    if (!list) {
        PyMem_Free(buffer);
        return NULL;
    }
    for (i = 0; (ch = buffer[i]); i++) {
        /* TODO:
         * TTF_GlyphMetrics() seems to return a value for any character,
         * using the default invalid character, if the char is not found.
         */
#if SDL_TTF_VERSION_ATLEAST(3, 0, 0)
        if (TTF_GetGlyphMetrics(font, ch, &minx, &maxx, &miny, &maxy,
                                &advance))
#elif SDL_TTF_VERSION_ATLEAST(2, 0, 18)
        if (!TTF_GlyphMetrics32(font, ch, &minx, &maxx, &miny, &maxy,
                                &advance))
#else
        if (ch <= 0xFFFF && /* conditional and */
            !TTF_GlyphMetrics(font, (Uint16)ch, &minx, &maxx, &miny, &maxy,
                              &advance))
#endif
        {
            listitem =
                Py_BuildValue("(iiiii)", minx, maxx, miny, maxy, advance);
            if (!listitem) {
                Py_DECREF(list);
                PyMem_Free(buffer);
                return NULL;
            }
        }
        else {
            /* Not UCS-2 (and old SDL) or no matching metrics. */
            Py_INCREF(Py_None);
            listitem = Py_None;
        }
        if (0 != PyList_Append(list, listitem)) {
            Py_DECREF(list);
            Py_DECREF(listitem);
            PyMem_Free(buffer);
            return NULL; /* Exception already set. */
        }
        Py_DECREF(listitem);
    }
    PyMem_Free(buffer);
    return list;
}

/* This is taken from the harfbuzz header file. It converts script name in the
 * format expected by sdl2 (a 4 char string) to the format expected by sdl3
 * (a single uint32 tag) */
#define HB_TAG(c1, c2, c3, c4)                  \
    ((Uint32)((((uint32_t)(c1) & 0xFF) << 24) | \
              (((uint32_t)(c2) & 0xFF) << 16) | \
              (((uint32_t)(c3) & 0xFF) << 8) | ((uint32_t)(c4) & 0xFF)))

static PyObject *
font_set_script(PyObject *self, PyObject *arg)
{
    if (!PgFont_GenerationCheck(self)) {
        return RAISE_FONT_QUIT_ERROR();
    }

#if SDL_TTF_VERSION_ATLEAST(2, 20, 0)
    TTF_Font *font = PyFont_AsFont(self);
    Py_ssize_t size;
    const char *script_code;

    if (!PyUnicode_Check(arg)) {
        return RAISE(PyExc_TypeError, "script code must be a string");
    }

    script_code = PyUnicode_AsUTF8AndSize(arg, &size);

    if (size != 4) {
        return RAISE(PyExc_ValueError,
                     "script code must be exactly 4 characters");
    }

#if SDL_TTF_VERSION_ATLEAST(3, 0, 0)
    if (!TTF_SetFontScript(font, HB_TAG(script_code[0], script_code[1],
                                        script_code[2], script_code[3])))
#else
    if (TTF_SetFontScriptName(font, script_code) < 0)
#endif
    {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }
#else
    return RAISE(pgExc_SDLError,
                 "pygame.font not compiled with a new enough SDL_ttf version. "
                 "Needs SDL_ttf 2.20.0 or above.");
#endif
    Py_RETURN_NONE;
}

static PyObject *
font_set_direction(PyObject *self, PyObject *arg, PyObject *kwarg)
{
    if (!PgFont_GenerationCheck(self)) {
        return RAISE_FONT_QUIT_ERROR();
    }

#if SDL_TTF_VERSION_ATLEAST(2, 20, 0)
    TTF_Font *font = PyFont_AsFont(self);
    int direction;
    char *kwds[] = {"direction", NULL};

    if (!PyArg_ParseTupleAndKeywords(arg, kwarg, "i", kwds, &direction)) {
        return NULL;
    }

    if (direction < 0 || direction > 3) {
        return RAISE(PyExc_ValueError,
                     "invalid input parameter for Font.set_direction");
    }

    TTF_Direction dir;
    switch (direction) {
        case 0: {
            dir = TTF_DIRECTION_LTR;
            break;
        }

        case 1: {
            dir = TTF_DIRECTION_RTL;
            break;
        }

/*  There is a bug in SDL_ttf up to 2.22.0 (the next release version as of
   writing this) This bug flips the top-to-bottom and bottom-to-top rendering.
   So, this is a compat patch for that behavior
 */
#if !SDL_TTF_VERSION_ATLEAST(2, 22, 0)
        case 2: {
            dir = TTF_DIRECTION_BTT;
            break;
        }

        case 3: {
            dir = TTF_DIRECTION_TTB;
            break;
        }
#else
        case 2: {
            dir = TTF_DIRECTION_TTB;
            break;
        }

        case 3: {
            dir = TTF_DIRECTION_BTT;
            break;
        }
#endif

        default: {
            // should NEVER reach this point
            return RAISE(PyExc_RuntimeError,
                         "Something went wrong in Font.set_direction. Please "
                         "report this"
                         " to https://github.com/pygame/pygame");
        }
    }
#if SDL_TTF_VERSION_ATLEAST(3, 0, 0)
    if (!TTF_SetFontDirection(font, dir))
#else
    if (TTF_SetFontDirection(font, dir))
#endif
    {
        return RAISE(pgExc_SDLError, SDL_GetError());
    }

#else
    return RAISE(pgExc_SDLError,
                 "pygame.font not compiled with a new enough SDL_ttf version. "
                 "Needs SDL_ttf 2.20.0 or above.");
#endif
    Py_RETURN_NONE;
}

/**
 * Getters and setters for the pgFontObject.
 */
static PyGetSetDef font_getsets[] = {
    {"name", (getter)font_getter_name, NULL, DOC_FONT_FONT_NAME, NULL},
    {"style_name", (getter)font_getter_style_name, NULL,
     DOC_FONT_FONT_STYLENAME, NULL},
    {"bold", (getter)font_getter_bold, (setter)font_setter_bold,
     DOC_FONT_FONT_BOLD, NULL},
    {"italic", (getter)font_getter_italic, (setter)font_setter_italic,
     DOC_FONT_FONT_ITALIC, NULL},
    {"underline", (getter)font_getter_underline, (setter)font_setter_underline,
     DOC_FONT_FONT_UNDERLINE, NULL},
    {"strikethrough", (getter)font_getter_strikethrough,
     (setter)font_setter_strikethrough, DOC_FONT_FONT_STRIKETHROUGH, NULL},
    {"outline", (getter)font_getter_outline, (setter)font_setter_outline,
     DOC_FONT_FONT_OUTLINE, NULL},
    {"align", (getter)font_getter_align, (setter)font_setter_align,
     DOC_FONT_FONT_ALIGN, NULL},
    {"point_size", (getter)font_getter_point_size,
     (setter)font_setter_point_size, DOC_FONT_FONT_POINTSIZE, NULL},
    {NULL, NULL, NULL, NULL, NULL}};

static PyMethodDef font_methods[] = {
    {"get_height", font_get_height, METH_NOARGS, DOC_FONT_FONT_GETHEIGHT},
    {"get_descent", font_get_descent, METH_NOARGS, DOC_FONT_FONT_GETDESCENT},
    {"get_ascent", font_get_ascent, METH_NOARGS, DOC_FONT_FONT_GETASCENT},
    {"get_linesize", font_get_linesize, METH_NOARGS,
     DOC_FONT_FONT_GETLINESIZE},
    {"set_linesize", font_set_linesize, METH_O, DOC_FONT_FONT_SETLINESIZE},
    {"get_bold", font_get_bold, METH_NOARGS, DOC_FONT_FONT_GETBOLD},
    {"set_bold", font_set_bold, METH_O, DOC_FONT_FONT_SETBOLD},
    {"get_italic", font_get_italic, METH_NOARGS, DOC_FONT_FONT_GETITALIC},
    {"set_italic", font_set_italic, METH_O, DOC_FONT_FONT_SETITALIC},
    {"get_underline", font_get_underline, METH_NOARGS,
     DOC_FONT_FONT_GETUNDERLINE},
    {"set_underline", font_set_underline, METH_O, DOC_FONT_FONT_SETUNDERLINE},
    {"get_strikethrough", font_get_strikethrough, METH_NOARGS,
     DOC_FONT_FONT_GETSTRIKETHROUGH},
    {"set_strikethrough", font_set_strikethrough, METH_O,
     DOC_FONT_FONT_SETSTRIKETHROUGH},
    {"get_point_size", font_get_ptsize, METH_NOARGS,
     DOC_FONT_FONT_GETPOINTSIZE},
    {"set_point_size", font_set_ptsize, METH_O, DOC_FONT_FONT_SETPOINTSIZE},
    {"metrics", font_metrics, METH_O, DOC_FONT_FONT_METRICS},
    {"render", (PyCFunction)font_render, METH_VARARGS | METH_KEYWORDS,
     DOC_FONT_FONT_RENDER},
    {"size", font_size, METH_O, DOC_FONT_FONT_SIZE},
    {"set_script", font_set_script, METH_O, DOC_FONT_FONT_SETSCRIPT},
    {"set_direction", (PyCFunction)font_set_direction,
     METH_VARARGS | METH_KEYWORDS, DOC_FONT_FONT_SETDIRECTION},
    {NULL, NULL, 0, NULL}};

/*font object internals*/
static void
font_dealloc(PyFontObject *self)
{
    TTF_Font *font = PyFont_AsFont(self);
    if (font && font_initialized) {
        // In SDL3_ttf, it seems that closing a font after its library was
        // destroyed segfaults. So only close if same generation.
        // TODO SDL3:
        // TTF docs say "A well-written program should call TTF_CloseFont()
        // on any open fonts before calling this function!"
#if SDL_VERSION_ATLEAST(3, 0, 0)
        if (self->ttf_init_generation == current_ttf_generation) {
            TTF_CloseFont(font);
        }
        self->font = NULL;
#else
        if (self->ttf_init_generation != current_ttf_generation) {
            // Since TTF_Font is a private structure
            // it's impossible to access face field in a common way.
            long **face_pp = (long **)font;
            *face_pp = NULL;
        }
        TTF_CloseFont(font);
        self->font = NULL;
#endif
    }

    if (self->weakreflist) {
        PyObject_ClearWeakRefs((PyObject *)self);
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int
font_init(PyFontObject *self, PyObject *args, PyObject *kwds)
{
    int fontsize = font_defaultsize;
    TTF_Font *font = NULL;
    PyObject *obj = Py_None;
    SDL_RWops *rw;

    static char *kwlist[] = {"filename", "size", NULL};

    self->font = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|Oi", kwlist, &obj,
                                     &fontsize)) {
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

    rw = pgRWops_FromObject(obj, NULL);

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

            rw = pgRWops_FromObject(obj, NULL);
        }
    }

    if (rw == NULL) {
        goto error;
    }

    if (fontsize <= 1) {
        fontsize = 1;
    }

    if (SDL_RWsize(rw) <= 0) {
        PyErr_Format(PyExc_ValueError,
                     "Font file object has an invalid file size: %lld",
                     SDL_RWsize(rw));
        goto error;
    }

    Py_BEGIN_ALLOW_THREADS;
#if SDL_VERSION_ATLEAST(3, 0, 0)
    /* TODO: can consider supporting float in python API */
    font = TTF_OpenFontIO(rw, 1, (float)fontsize);
#else
    font = TTF_OpenFontRW(rw, 1, fontsize);
#endif
    Py_END_ALLOW_THREADS;

    Py_DECREF(obj);
    self->font = font;
    self->ptsize = fontsize;
    self->ttf_init_generation = current_ttf_generation;

    return 0;

error:
    Py_XDECREF(obj);
    return -1;
}

static PyTypeObject PyFont_Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pygame.font.Font",
    .tp_basicsize = sizeof(PyFontObject),
    .tp_dealloc = (destructor)font_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = DOC_FONT_FONT,
    .tp_weaklistoffset = offsetof(PyFontObject, weakreflist),
    .tp_methods = font_methods,
    .tp_getset = font_getsets,
    .tp_init = (initproc)font_init,
    .tp_new = PyType_GenericNew,
};

/*font module methods*/
static PyObject *
get_default_font(PyObject *self, PyObject *_null)
{
    return PyUnicode_FromString(font_defaultname);
}

static PyObject *
get_ttf_version(PyObject *self, PyObject *args, PyObject *kwargs)
{
    int linked = 1; /* Default is linked version. */
#if SDL_VERSION_ATLEAST(3, 0, 0)
    int version = SDL_TTF_VERSION;
#else
    SDL_version version;
    TTF_VERSION(&version);
#endif

    static char *keywords[] = {"linked", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|p", keywords, &linked)) {
        return NULL; /* Exception already set. */
    }

    if (linked) {
#if SDL_VERSION_ATLEAST(3, 0, 0)
        version = TTF_Version();
#else
        version = *TTF_Linked_Version();
#endif
    }

    return Py_BuildValue("iii", PG_FIND_VNUM_MAJOR(version),
                         PG_FIND_VNUM_MINOR(version),
                         PG_FIND_VNUM_MICRO(version));
}

static PyMethodDef _font_methods[] = {
    {"init", (PyCFunction)fontmodule_init, METH_NOARGS, DOC_FONT_INIT},
    {"quit", (PyCFunction)fontmodule_quit, METH_NOARGS, DOC_FONT_QUIT},
    {"get_init", (PyCFunction)pg_font_get_init, METH_NOARGS, DOC_FONT_GETINIT},
    {"get_default_font", (PyCFunction)get_default_font, METH_NOARGS,
     DOC_FONT_GETDEFAULTFONT},
    {"get_sdl_ttf_version", (PyCFunction)get_ttf_version,
     METH_VARARGS | METH_KEYWORDS, DOC_FONT_GETINIT},

    {NULL, NULL, 0, NULL}};

static PyObject *
PyFont_New(TTF_Font *font)
{
    PyFontObject *fontobj;

    if (!font) {
        return RAISE(PyExc_RuntimeError, "unable to load font.");
    }
    fontobj = (PyFontObject *)PyFont_Type.tp_new(&PyFont_Type, NULL, NULL);

    if (fontobj) {
        fontobj->font = font;
    }

    return (PyObject *)fontobj;
}

MODINIT_DEFINE(font)
{
    PyObject *module, *apiobj;
    static void *c_api[PYGAMEAPI_FONT_NUMSLOTS];

    static struct PyModuleDef _module = {PyModuleDef_HEAD_INIT,
                                         "font",
                                         DOC_FONT,
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

    module = PyModule_Create(&_module);
    if (module == NULL) {
        return NULL;
    }

    if (PyModule_AddObjectRef(module, "FontType", (PyObject *)&PyFont_Type)) {
        Py_DECREF(module);
        return NULL;
    }

    if (PyModule_AddObjectRef(module, "Font", (PyObject *)&PyFont_Type)) {
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
