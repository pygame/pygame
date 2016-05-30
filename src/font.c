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

/* Require SDL_ttf 2.0.6 or later for rwops support */
#ifdef TTF_MAJOR_VERSION
#define FONT_HAVE_RWOPS 1
#else
#define FONT_HAVE_RWOPS 0
#endif

#if PY3
#define RAISE_TEXT_TYPE_ERROR() \
    RAISE(PyExc_TypeError, "text must be a unicode or bytes");
#else
#define RAISE_TEXT_TYPE_ERROR() \
    RAISE(PyExc_TypeError, "text must be a string or unicode");
#endif

/* For filtering out UCS-4 and larger characters when Python is
 * built with Py_UNICODE_WIDE.
 */
#if defined(Py_UNICODE_WIDE)
#define IS_UCS_2(c) ((c) < 0x10000L)
#else
#define IS_UCS_2(c) 1
#endif

static PyTypeObject PyFont_Type;
static PyObject* PyFont_New (TTF_Font*);
#define PyFont_Check(x) ((x)->ob_type == &PyFont_Type)

static int font_initialized = 0;
static const char font_defaultname[] = "freesansbold.ttf";
static const char pkgdatamodule_name[] = "pygame.pkgdata";
static const char resourcefunc_name[] = "getResource";


/*
 */
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

/* Return an encoded file path, a file-like object or a NULL pointer.
 * May raise a Python error. Use PyErr_Occurred to check.
 */
static PyObject*
font_resource (const char *filename)
{
    PyObject* pkgdatamodule = NULL;
    PyObject* resourcefunc = NULL;
    PyObject* result = NULL;
    PyObject* tmp;

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

#if PY3
    tmp = PyObject_GetAttrString(result, "name");
    if (tmp != NULL) {
        Py_DECREF(result);
        result = tmp;
    }
    else if (!PyErr_ExceptionMatches(PyExc_MemoryError)) {
        PyErr_Clear();
    }
#else
    if (PyFile_Check(result))
    {
        tmp = PyFile_Name(result);
        Py_INCREF(tmp);
        Py_DECREF(result);
        result = tmp;
    }
#endif

    tmp = RWopsEncodeFilePath(result, NULL);
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

static void
font_autoquit (void)
{
    if (font_initialized)
    {
        font_initialized = 0;
        TTF_Quit ();
    }
}


static PyObject*
font_autoinit (PyObject* self)
{
    if (!font_initialized)
    {
        PyGame_RegisterQuit (font_autoquit);

        if (TTF_Init ())
            return PyInt_FromLong (0);
        font_initialized = 1;

    }
    return PyInt_FromLong (font_initialized);
}

static PyObject*
fontmodule_quit (PyObject* self)
{
    font_autoquit ();
    Py_RETURN_NONE;
}


static PyObject*
fontmodule_init (PyObject* self)
{
    PyObject* result;
    int istrue;

    result = font_autoinit (self);
    if (result == NULL)
        return NULL;
    istrue = PyObject_IsTrue (result);
    Py_DECREF (result);
    if (!istrue)
        return RAISE (PyExc_SDLError, SDL_GetError ());
    Py_RETURN_NONE;
}

static PyObject*
get_init (PyObject* self)
{
    return PyInt_FromLong (font_initialized);
}

/* font object methods */
static PyObject*
font_get_height (PyObject* self)
{
    TTF_Font* font = PyFont_AsFont (self);
    return PyInt_FromLong (TTF_FontHeight (font));
}

static PyObject*
font_get_descent (PyObject* self)
{
    TTF_Font* font = PyFont_AsFont (self);
    return PyInt_FromLong (TTF_FontDescent (font));
}

static PyObject*
font_get_ascent (PyObject* self)
{
    TTF_Font* font = PyFont_AsFont (self);
    return PyInt_FromLong (TTF_FontAscent (font));
}

static PyObject*
font_get_linesize (PyObject* self)
{
    TTF_Font* font = PyFont_AsFont (self);
    return PyInt_FromLong (TTF_FontLineSkip (font));
}

static PyObject*
font_get_bold (PyObject* self)
{
    TTF_Font* font = PyFont_AsFont (self);
    return PyInt_FromLong ((TTF_GetFontStyle (font) & TTF_STYLE_BOLD) != 0);
}

static PyObject*
font_set_bold (PyObject* self, PyObject* args)
{
    TTF_Font* font = PyFont_AsFont (self);
    int style, val;

    if (!PyArg_ParseTuple (args, "i", &val))
        return NULL;

    style = TTF_GetFontStyle (font);
    if (val)
        style |= TTF_STYLE_BOLD;
    else
        style &= ~TTF_STYLE_BOLD;
    TTF_SetFontStyle (font, style);

    Py_RETURN_NONE;
}

static PyObject*
font_get_italic (PyObject* self)
{
    TTF_Font* font = PyFont_AsFont (self);
    return PyInt_FromLong ((TTF_GetFontStyle (font) & TTF_STYLE_ITALIC) != 0);
}

static PyObject*
font_set_italic (PyObject* self, PyObject* args)
{
    TTF_Font* font = PyFont_AsFont (self);
    int style, val;

    if (!PyArg_ParseTuple (args, "i", &val))
        return NULL;

    style = TTF_GetFontStyle (font);
    if(val)
        style |= TTF_STYLE_ITALIC;
    else
        style &= ~TTF_STYLE_ITALIC;
    TTF_SetFontStyle (font, style);

    Py_RETURN_NONE;
}

static PyObject*
font_get_underline (PyObject* self)
{
    TTF_Font* font = PyFont_AsFont (self);
    return PyInt_FromLong
        ((TTF_GetFontStyle (font) & TTF_STYLE_UNDERLINE) != 0);
}

static PyObject*
font_set_underline (PyObject* self, PyObject* args)
{
    TTF_Font* font = PyFont_AsFont (self);
    int style, val;

    if (!PyArg_ParseTuple (args, "i", &val))
        return NULL;

    style = TTF_GetFontStyle (font);
    if(val)
        style |= TTF_STYLE_UNDERLINE;
    else
        style &= ~TTF_STYLE_UNDERLINE;
    TTF_SetFontStyle (font, style);

    Py_RETURN_NONE;
}

static PyObject*
font_render(PyObject* self, PyObject* args)
{
    TTF_Font* font = PyFont_AsFont (self);
    int aa;
    PyObject* text, *final;
    PyObject* fg_rgba_obj, *bg_rgba_obj = NULL;
    Uint8 rgba[] = {0, 0, 0, 0};
    SDL_Surface* surf;
    SDL_Color foreg, backg;
    int just_return;

    if (!PyArg_ParseTuple(args, "OiO|O", &text, &aa, &fg_rgba_obj,
                          &bg_rgba_obj)) {
        return NULL;
    }

    if (!RGBAFromColorObj(fg_rgba_obj, rgba)) {
        return RAISE(PyExc_TypeError, "Invalid foreground RGBA argument");
    }
    foreg.r = rgba[0];
    foreg.g = rgba[1];
    foreg.b = rgba[2];
    foreg.unused = 0;
    if (bg_rgba_obj != NULL) {
        if (!RGBAFromColorObj(bg_rgba_obj, rgba)) {
            bg_rgba_obj = NULL;
            backg.r = 0;
            backg.g = 0;
            backg.b = 0;
            backg.unused = 0;
        }
        else
        {
            backg.r = rgba[0];
            backg.g = rgba[1];
            backg.b = rgba[2];
            backg.unused = 0;
        }
    }
    else {
        backg.r = 0;
        backg.g = 0;
        backg.b = 0;
        backg.unused = 0;
    }

    just_return = PyObject_Not(text);
    if (just_return) {
        int height = TTF_FontHeight(font);

        if (just_return == -1 ||
            !(PyUnicode_Check(text) || Bytes_Check(text) || text == Py_None)) {
            PyErr_Clear();
            return RAISE_TEXT_TYPE_ERROR();
        }
        surf = SDL_CreateRGBSurface(SDL_SWSURFACE, 1, height, 32,
                                    0xff<<16, 0xff<<8, 0xff, 0);
        if (surf == NULL) {
            return RAISE(PyExc_SDLError, SDL_GetError());
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
        astring = Bytes_AsString(bytes);
        if (strlen(astring) != Bytes_GET_SIZE(bytes)) {
            Py_DECREF(bytes);
            return RAISE(PyExc_ValueError,
                         "A null character was found in the text");
        }
        if (utf_8_needs_UCS_4(astring)) {
            Py_DECREF(bytes);
            return RAISE(PyExc_UnicodeError,
                         "A Unicode character above '\\uFFFF' was found;"
                         " not supported");
        }
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
    else if (Bytes_Check(text)) {
        const char *astring = Bytes_AsString(text);

        if (strlen(astring) != Bytes_GET_SIZE(text)) {
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
        return RAISE(PyExc_SDLError, TTF_GetError());
    }
    if (!aa && (bg_rgba_obj != NULL) && !just_return) {
        /* turn off transparancy */
        SDL_SetColorKey(surf, 0, 0);
        surf->format->palette->colors[0].r = backg.r;
        surf->format->palette->colors[0].g = backg.g;
        surf->format->palette->colors[0].b = backg.b;
    }
    final = PySurface_New(surf);
    if (final == NULL) {
        SDL_FreeSurface(surf);
    }
    return final;
}

static PyObject*
font_size(PyObject* self, PyObject* args)
{
    TTF_Font* font = PyFont_AsFont(self);
    int w, h;
    PyObject *text;
    const char *string;

    if (!PyArg_ParseTuple(args, "O", &text)) {
        return NULL;
    }

    if (PyUnicode_Check(text))
    {
        PyObject* bytes = PyUnicode_AsEncodedString(text, "utf-8", "strict");
        int ecode;

        if (!bytes) {
            return NULL;
        }
        string = Bytes_AS_STRING(bytes);
        ecode = TTF_SizeUTF8(font, string, &w, &h);
        Py_DECREF(bytes);
        if (ecode) {
            return RAISE (PyExc_SDLError, TTF_GetError());
        }
    }
    else if (Bytes_Check(text)) {
        string = Bytes_AS_STRING(text);
        if (TTF_SizeText(font, string, &w, &h)) {
            return RAISE (PyExc_SDLError, TTF_GetError());
        }
    }
    else {
        return RAISE_TEXT_TYPE_ERROR();
    }
    return Py_BuildValue("(ii)", w, h);
}

static PyObject*
font_metrics(PyObject* self, PyObject* args)
{
    TTF_Font *font = PyFont_AsFont (self);
    PyObject *list;
    PyObject *textobj;
    Py_ssize_t length;
    Py_ssize_t i;
    int minx;
    int maxx;
    int miny;
    int maxy;
    int advance;
    PyObject *unicodeobj;
    PyObject *listitem;
    Py_UNICODE *buffer;
    Py_UNICODE ch;

    if (!PyArg_ParseTuple(args, "O", &textobj)) {
        return NULL;
    }

    if (PyUnicode_Check (textobj)) {
        unicodeobj = textobj;
        Py_INCREF (unicodeobj);
    }
    else if (Bytes_Check (textobj)) {
        unicodeobj = PyUnicode_FromEncodedObject(textobj, "latin-1", NULL);
        if (!unicodeobj) {
            return NULL;
        }
    }
    else {
        return RAISE_TEXT_TYPE_ERROR ();
    }

    length = PyUnicode_GET_SIZE(unicodeobj);
    list = PyList_New(length);
    if (!list) {
        Py_DECREF (unicodeobj);
        return NULL;
    }
    buffer = PyUnicode_AS_UNICODE(unicodeobj);
    for (i = 0; i != length; ++i) {
        ch = buffer[i];
        /* TODO:
         * TTF_GlyphMetrics() seems to return a value for any character,
         * using the default invalid character, if the char is not found.
         */
        if (IS_UCS_2(ch) &&  /* conditional and */
            !TTF_GlyphMetrics(font, (Uint16) ch, &minx,
                              &maxx, &miny, &maxy, &advance)) {
            listitem = Py_BuildValue("(iiiii)",
                                     minx, maxx, miny, maxy, advance);
            if (!listitem) {
                Py_DECREF(list);
                Py_DECREF(unicodeobj);
                return NULL;
            }
        }
        else {
            /* Not UCS-2 or no matching metrics. */
            Py_INCREF(Py_None);
            listitem = Py_None;
        }
        PyList_SET_ITEM(list, i, listitem);
    }
    Py_DECREF(unicodeobj);
    return list;
}

static PyMethodDef font_methods[] =
{
    { "get_height", (PyCFunction) font_get_height, METH_NOARGS,
      DOC_FONTGETHEIGHT },
    { "get_descent", (PyCFunction) font_get_descent, METH_NOARGS,
      DOC_FONTGETDESCENT },
    { "get_ascent", (PyCFunction) font_get_ascent, METH_NOARGS,
      DOC_FONTGETASCENT },
    { "get_linesize", (PyCFunction) font_get_linesize, METH_NOARGS,
      DOC_FONTGETLINESIZE },

    { "get_bold", (PyCFunction) font_get_bold, METH_NOARGS,
      DOC_FONTGETBOLD },
    { "set_bold", font_set_bold, METH_VARARGS, DOC_FONTSETBOLD },
    { "get_italic", (PyCFunction) font_get_italic, METH_NOARGS,
      DOC_FONTGETITALIC },
    { "set_italic", font_set_italic, METH_VARARGS, DOC_FONTSETITALIC },
    { "get_underline", (PyCFunction) font_get_underline, METH_NOARGS,
      DOC_FONTGETUNDERLINE },
    { "set_underline", font_set_underline, METH_VARARGS, DOC_FONTSETUNDERLINE },

    { "metrics", font_metrics, METH_VARARGS, DOC_FONTMETRICS },
    { "render", font_render, METH_VARARGS, DOC_FONTRENDER },
    { "size", font_size, METH_VARARGS, DOC_FONTSIZE },

    { NULL, NULL, 0, NULL }
};

/*font object internals*/
static void
font_dealloc (PyFontObject* self)
{
    TTF_Font* font = PyFont_AsFont (self);

    if (font && font_initialized)
        TTF_CloseFont (font);

    if (self->weakreflist)
        PyObject_ClearWeakRefs ((PyObject*) self);
    Py_TYPE(self)->tp_free ((PyObject*) self);
}

static int
font_init(PyFontObject *self, PyObject *args, PyObject *kwds)
{
    int fontsize;
    TTF_Font *font = NULL;
    PyObject *obj;
    PyObject *oencoded;

    self->font = NULL;
    if (!PyArg_ParseTuple(args, "Oi", &obj, &fontsize)) {
        return -1;
    }

    if (!font_initialized) {
        RAISE(PyExc_SDLError, "font not initialized");
        return -1;
    }

    Py_INCREF(obj);

    if (fontsize <= 1) {
        fontsize = 1;
    }

    if (obj == Py_None) {
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
        if (fontsize <= 1) {
            fontsize = 1;
        }
    }
    else {
        oencoded = RWopsEncodeFilePath(obj, NULL);
        if (oencoded == NULL) {
            goto error;
        }
        if (oencoded == Py_None) {
            Py_DECREF(oencoded);
        }
        else {
            Py_DECREF(obj);
            obj = oencoded;
        }
    }
    if (Bytes_Check(obj)) {
        const char *filename = Bytes_AS_STRING(obj);
        FILE *test;

        /*check if it is a valid file, else SDL_ttf segfaults*/
        test = fopen(filename, "rb");
        if (test == NULL) {
            PyObject *tmp = NULL;

            if (strcmp(filename, font_defaultname) == 0) {
                /* filename is the default font; get it's resource
                 */
                tmp = font_resource(font_defaultname);
            }
            if (tmp == NULL) {
                if (PyErr_Occurred() == NULL) {
                    PyErr_Format(PyExc_IOError,
                                 "unable to read font file '%.1024s'",
                                 filename);
                }
                goto error;
            }
            Py_DECREF(obj);
            obj = tmp;
            if (Bytes_Check(obj)) {
                filename = Bytes_AS_STRING(obj);
                test = fopen(filename, "rb");
                if (test == NULL) {
                    PyErr_Format(PyExc_IOError,
                                 "unable to read font file '%.1024s'",
                                 filename);
                    goto error;
                }
            }
        }
        if (Bytes_Check(obj)) {
            fclose(test);
            Py_BEGIN_ALLOW_THREADS;
            font = TTF_OpenFont(filename, fontsize);
            Py_END_ALLOW_THREADS;
        }
    }
    if (font == NULL)  {
#if FONT_HAVE_RWOPS
        SDL_RWops *rw = RWopsFromFileObject(obj);

        if (rw == NULL) {
            goto error;
        }

        if (RWopsCheckObject(rw)) {
            font = TTF_OpenFontIndexRW(rw, 1, fontsize, 0);
        }
        else {
             Py_BEGIN_ALLOW_THREADS;
             font = TTF_OpenFontIndexRW(rw, 1, fontsize, 0);
             Py_END_ALLOW_THREADS;
        }
#else
        RAISE (PyExc_NotImplementedError,
               "nonstring fonts require SDL_ttf-2.0.6");
        goto error;
#endif
    }

    if (font == NULL) {
        RAISE(PyExc_RuntimeError, SDL_GetError());
        goto error;
    }

    Py_DECREF(obj);
    self->font = font;
    return 0;

error:
    Py_DECREF(obj);
    return -1;
}

static PyTypeObject PyFont_Type =
{
    TYPE_HEAD (NULL, 0)
    "pygame.font.Font",
    sizeof(PyFontObject),
    0,
    (destructor)font_dealloc,
    0,
    0,                                        /*getattr*/
    0,
    0,
    0,
    0,
    NULL,
    0,
    (hashfunc)NULL,
    (ternaryfunc)NULL,
    (reprfunc)NULL,
    0L,0L,0L,
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
    0,                                        /* tp_getset */
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
static PyObject*
get_default_font (PyObject* self)
{
    return Text_FromUTF8 (font_defaultname);
}

static PyMethodDef _font_methods[] =
{
    { "__PYGAMEinit__", (PyCFunction) font_autoinit, METH_NOARGS,
      "auto initialize function for font" },
    { "init", (PyCFunction) fontmodule_init, METH_NOARGS, DOC_PYGAMEFONTINIT },
    { "quit", (PyCFunction) fontmodule_quit, METH_NOARGS, DOC_PYGAMEFONTQUIT },
    { "get_init", (PyCFunction) get_init, METH_NOARGS, DOC_PYGAMEFONTGETINIT },
    { "get_default_font", (PyCFunction) get_default_font, METH_NOARGS,
      DOC_PYGAMEFONTGETDEFAULTFONT },
    { NULL, NULL, 0, NULL }
};



static PyObject*
PyFont_New (TTF_Font* font)
{
    PyFontObject* fontobj;

    if (!font)
        return RAISE (PyExc_RuntimeError, "unable to load font.");
    fontobj = (PyFontObject *) PyFont_Type.tp_new (&PyFont_Type, NULL, NULL);

    if (fontobj)
        fontobj->font = font;

    return (PyObject*) fontobj;
}

MODINIT_DEFINE (font)
{
    PyObject *module, *apiobj;
    static void* c_api[PYGAMEAPI_FONT_NUMSLOTS];

#if PY3
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "font",
        DOC_PYGAMEFONT,
        -1,
        _font_methods,
        NULL, NULL, NULL, NULL
    };
#endif

    /* imported needed apis; Do this first so if there is an error
       the module is not loaded.
    */
    import_pygame_base ();
    if (PyErr_Occurred ()) {
        MODINIT_ERROR;
    }
    import_pygame_color ();
    if (PyErr_Occurred ()) {
        MODINIT_ERROR;
    }
    import_pygame_surface ();
    if (PyErr_Occurred ()) {
        MODINIT_ERROR;
    }
    import_pygame_rwobject ();
    if (PyErr_Occurred ()) {
        MODINIT_ERROR;
    }

    /* type preparation */
    if (PyType_Ready (&PyFont_Type) < 0) {
        MODINIT_ERROR;
    }
    PyFont_Type.tp_new = PyType_GenericNew;

#if PY3
    module = PyModule_Create (&_module);
#else
    module = Py_InitModule3 (MODPREFIX "font",
                             _font_methods,
                             DOC_PYGAMEFONT);
#endif
    if (module == NULL) {
        MODINIT_ERROR;
    }

    Py_INCREF ((PyObject*) &PyFont_Type);
    if (PyModule_AddObject (module,
                            "FontType",
                            (PyObject *) &PyFont_Type) == -1) {
        Py_DECREF ((PyObject *) &PyFont_Type);
        DECREF_MOD (module);
        MODINIT_ERROR;
    }

    Py_INCREF ((PyObject*) &PyFont_Type);
    if (PyModule_AddObject (module,
                            "Font",
                            (PyObject *) &PyFont_Type) == -1) {
        Py_DECREF ((PyObject *) &PyFont_Type);
        DECREF_MOD (module);
        MODINIT_ERROR;
    }

    /* export the c api */
    c_api[0] = &PyFont_Type;
    c_api[1] = PyFont_New;
    c_api[2] = &font_initialized;
    apiobj = encapsulate_api (c_api, "font");
    if (apiobj == NULL) {
        DECREF_MOD (module);
        MODINIT_ERROR;
    }
    if (PyModule_AddObject (module, PYGAMEAPI_LOCAL_ENTRY, apiobj) == -1) {
        Py_DECREF (apiobj);
        DECREF_MOD (module);
        MODINIT_ERROR;
    }
    MODINIT_RETURN (module);
}
