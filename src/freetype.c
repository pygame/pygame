/*
  pygame - Python Game Library
  Copyright (C) 2009 Vicent Marti

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

#define PYGAME_FREETYPE_INTERNAL
#define PYGAME_FREETYPE_FONT_INTERNAL

#include "freetype.h"
#include "freetype/ft_wrap.h"
#include "doc/freetype_doc.h"

#define MODULE_NAME "freetype"
#define FONT_TYPE_NAME "Font"

/*
 * FreeType module declarations
 */
#if PY3
static int _ft_traverse(PyObject *, visitproc, void *);
static int _ft_clear(PyObject *);
#endif

static PyObject *_ft_quit(PyObject *);
static PyObject *_ft_init(PyObject *, PyObject *, PyObject *);
static PyObject *_ft_get_version(PyObject *);
static PyObject *_ft_get_error(PyObject *);
static PyObject *_ft_was_init(PyObject *);
static PyObject *_ft_autoinit(PyObject *);
static PyObject *_ft_get_default_resolution(PyObject *);
static PyObject *_ft_set_default_resolution(PyObject *, PyObject *);
static PyObject *_ft_get_default_font(PyObject* self);

/*
 * Constructor/init/destructor
 */
static PyObject *_ftfont_new(PyTypeObject *, PyObject *, PyObject *);
static void _ftfont_dealloc(PyFreeTypeFont *);
static PyObject *_ftfont_repr(PyObject *);
static int _ftfont_init(PyObject *, PyObject *, PyObject *);

/*
 * Main methods
 */
static PyObject *_ftfont_getrect(PyObject *, PyObject *, PyObject *);
static PyObject *_ftfont_getmetrics(PyObject *, PyObject *, PyObject *);
static PyObject *_ftfont_render(PyObject *, PyObject *, PyObject *);
static PyObject *_ftfont_render_raw(PyObject *, PyObject *, PyObject *);
static PyObject *_ftfont_getsizedascender(PyFreeTypeFont *, PyObject *);
static PyObject *_ftfont_getsizeddescender(PyFreeTypeFont *, PyObject *);
static PyObject *_ftfont_getsizedheight(PyFreeTypeFont *, PyObject *);
static PyObject *_ftfont_getsizedglyphheight(PyFreeTypeFont *, PyObject *);
static PyObject *_ftfont_gettransform(PyFreeTypeFont *);
static PyObject *_ftfont_settransform(PyFreeTypeFont *, PyObject *);
static PyObject *_ftfont_deletetransform(PyFreeTypeFont *);

/* static PyObject *_ftfont_copy(PyObject *); */

/*
 * Getters/setters
 */
static PyObject *_ftfont_getstyle(PyObject *, void *);
static int _ftfont_setstyle(PyObject *, PyObject *, void *);
static PyObject *_ftfont_getheight(PyObject *, void *);
static PyObject *_ftfont_getascender(PyObject *, void *);
static PyObject *_ftfont_getdescender(PyObject *, void *);
static PyObject *_ftfont_getname(PyObject *, void *);
static PyObject *_ftfont_getpath(PyObject *, void *);
static PyObject *_ftfont_getfixedwidth(PyObject *, void *);

static PyObject *_ftfont_getvertical(PyObject *, void *);
static int _ftfont_setvertical(PyObject *, PyObject *, void *);
static PyObject *_ftfont_getantialias(PyObject *, void *);
static int _ftfont_setantialias(PyObject *, PyObject *, void *);
static PyObject *_ftfont_getkerning(PyObject *, void *);
static int _ftfont_setkerning(PyObject *, PyObject *, void *);
static PyObject *_ftfont_getucs4(PyObject *, void *);
static int _ftfont_setucs4(PyObject *, PyObject *, void *);
static PyObject *_ftfont_getorigin(PyObject *, void *);
static int _ftfont_setorigin(PyObject *, PyObject *, void *);
static PyObject *_ftfont_getpad(PyObject *, void *);
static int _ftfont_setpad(PyObject *, PyObject *, void *);

static PyObject *_ftfont_getresolution(PyObject *, void *);

static PyObject *_ftfont_getstyle_flag(PyObject *, void *);
static int _ftfont_setstyle_flag(PyObject *, PyObject *, void *);

#if defined(PGFT_DEBUG_CACHE)
static PyObject *_ftfont_getdebugcachestats(PyObject *, void *);
#endif

/*
 * Internal helpers
 */
static PyObject *get_metrics(FreeTypeInstance *, FontRenderMode *,
                             PyFreeTypeFont *, PGFT_String *);
static PyObject *load_font_res(const char *);
static int parse_dest(PyObject *, PyObject **, int *, int *);


/*
 * Auxiliar defines
 */
#define ASSERT_SELF_IS_ALIVE(s)                     \
if (!PyFreeTypeFont_IS_ALIVE(s)) {                  \
    return RAISE(PyExc_RuntimeError,                \
        MODULE_NAME "." FONT_TYPE_NAME              \
        " instance is not initialized");            \
}

#define PGFT_CHECK_BOOL(_pyobj, _var)               \
    if (_pyobj) {                                   \
        if (!PyBool_Check(_pyobj)) {                \
            PyErr_SetString(PyExc_TypeError,        \
                #_var " must be a boolean value");  \
            return NULL;                            \
        }                                           \
                                                    \
        _var = PyObject_IsTrue(_pyobj);             \
    }

#define DEFAULT_FONT_NAME   "freesansbold.ttf"
#define PKGDATA_MODULE_NAME "pygame.pkgdata"
#define RESOURCE_FUNC_NAME  "getResource"

static PyObject *
load_font_res(const char *filename)
{
    PyObject *load_basicfunc = NULL;
    PyObject *pkgdatamodule = NULL;
    PyObject *resourcefunc = NULL;
    PyObject *result = NULL;
    PyObject *tmp;

    pkgdatamodule = PyImport_ImportModule(PKGDATA_MODULE_NAME);
    if (!pkgdatamodule)
        goto font_resource_end;

    resourcefunc = PyObject_GetAttrString(pkgdatamodule, RESOURCE_FUNC_NAME);
    if (!resourcefunc)
        goto font_resource_end;

    result = PyObject_CallFunction(resourcefunc, "s", filename);
    if (!result)
        goto font_resource_end;

#if PY3
    tmp = PyObject_GetAttrString(result, "name");
    if (tmp != NULL) {
        Py_DECREF(result);
        result = tmp;
    }
    else  {
        PyErr_Clear();
    }
#else
    if (PyFile_Check(result)) {
        tmp = PyFile_Name(result);
        Py_INCREF(tmp);
        Py_DECREF(result);
        result = tmp;
    }
#endif

font_resource_end:
    Py_XDECREF(pkgdatamodule);
    Py_XDECREF(resourcefunc);
    Py_XDECREF(load_basicfunc);
    return result;
}

static int
parse_dest(PyObject *dest, PyObject **surf, int *x, int *y)
{
    PyObject *s = PySequence_GetItem(dest, 0);
    int len = PySequence_Length(dest);
    PyObject *oi;
    PyObject *oj;
    int i, j;

    if (!PySurface_Check(s)) {
        PyErr_Format(PyExc_TypeError,
                     "expected a Surface as element 0 of dest:"
                     " got type %.1024s",
                     Py_TYPE(s)->tp_name);
        Py_DECREF(s);
        return -1;
    }
    if (len == 2) {
        PyObject *size = PySequence_GetItem(dest, 1);

        if (size == NULL) {
            Py_DECREF(s);
            return -1;
        }
        if (!PySequence_Check(size)) {
            PyErr_Format(PyExc_TypeError,
                         "expected an (x,y) position for element 1"
                         " of dest: got type %.1024s",
                         Py_TYPE(size)->tp_name);
            Py_DECREF(s);
            Py_DECREF(size);
            return -1;
        }
        len = PySequence_Length(size);
        if (len < 2) {
            PyErr_Format(PyExc_TypeError,
                         "expected at least a length 2 sequence for element 1"
                         " of dest: not length %d", len);
            Py_DECREF(s);
            Py_DECREF(size);
            return -1;
        }
        oi = PySequence_GetItem(size, 0);
        if (oi == NULL) {
            Py_DECREF(s);
            Py_DECREF(size);
            return -1;
        }
        oj = PySequence_GetItem(size, 1);
        Py_DECREF(size);
        if (oj == NULL) {
            Py_DECREF(s);
            Py_DECREF(oi);
            return -1;
        }
        if (!PyNumber_Check(oi) || !PyNumber_Check(oj)) {
            Py_DECREF(s);
            Py_DECREF(oi);
            Py_DECREF(oj);
            PyErr_Format(PyExc_TypeError,
                         "expected a pair of numbers for element 1 of dest:"
                         " got types %.1024s and %.1024s",
                         Py_TYPE(oi)->tp_name, Py_TYPE(oj)->tp_name);
            return -1;
        }
    }
    else if (len == 3) {
        oi = PySequence_GetItem(dest, 1);
        if (oi == NULL) {
            Py_DECREF(s);
            return -1;
        }
        oj = PySequence_GetItem(dest, 2);
        if (oj == NULL) {
            Py_DECREF(oi);
            Py_DECREF(s);
            return -1;
        }
        if (!PyNumber_Check(oi) || !PyNumber_Check(oj)) {
            Py_DECREF(s);
            PyErr_Format(PyExc_TypeError,
                         "for dest expected a pair of numbers"
                         "for elements 1 and 2: got types %.1024s and %1024s",
                         Py_TYPE(oi)->tp_name, Py_TYPE(oj)->tp_name);
            Py_DECREF(oi);
            Py_DECREF(oj);
            return -1;
        }
    }
    else {
        Py_DECREF(s);
        PyErr_Format(PyExc_TypeError,
                     "for dest expected a sequence of either 2 or 3:"
                     " not length %d", len);
        return -1;
    }
    i = PyInt_AsLong(oi);
    Py_DECREF(oi);
    if (i == -1 && PyErr_Occurred()) {
        Py_DECREF(s);
        Py_DECREF(oj);
        return -1;
    }
    j = PyInt_AsLong(oj);
    Py_DECREF(oj);
    if (j == -1 && PyErr_Occurred()) {
        Py_DECREF(s);
        return -1;
    }
    *surf = s;
    *x = i;
    *y = j;
    return 0;
}

/*
 * FREETYPE MODULE METHODS TABLE
 */
static PyMethodDef _ft_methods[] = {
    {
        "__PYGAMEinit__",
        (PyCFunction) _ft_autoinit,
        METH_NOARGS,
        "auto initialize function for font"
    },
    {
        "init",
        (PyCFunction) _ft_init,
        METH_VARARGS | METH_KEYWORDS,
        DOC_PYGAMEFREETYPEINIT
    },
    {
        "quit",
        (PyCFunction) _ft_quit,
        METH_NOARGS,
        DOC_PYGAMEFREETYPEQUIT
    },
    {
        "was_init",
        (PyCFunction) _ft_was_init,
        METH_NOARGS,
        DOC_PYGAMEFREETYPEWASINIT
    },
    {
        "get_error",
        (PyCFunction) _ft_get_error,
        METH_NOARGS,
        DOC_PYGAMEFREETYPEGETERROR
    },
    {
        "get_version",
        (PyCFunction) _ft_get_version,
        METH_NOARGS,
        DOC_PYGAMEFREETYPEGETVERSION
    },
    {
        "get_default_resolution",
        (PyCFunction) _ft_get_default_resolution,
        METH_NOARGS,
        DOC_PYGAMEFREETYPEGETDEFAULTRESOLUTION
    },
    {
        "set_default_resolution",
        (PyCFunction) _ft_set_default_resolution,
        METH_VARARGS,
        DOC_PYGAMEFREETYPESETDEFAULTRESOLUTION
    },
    {
        "get_default_font",
        (PyCFunction) _ft_get_default_font,
        METH_NOARGS,
        DOC_PYGAMEFREETYPEGETDEFAULTFONT
    },

    { NULL, NULL, 0, NULL }
};


/*
 * FREETYPE FONT METHODS TABLE
 */
static PyMethodDef _ftfont_methods[] = {
    {
        "get_sized_height",
        (PyCFunction) _ftfont_getsizedheight,
        METH_VARARGS,
        DOC_FONTGETSIZEDHEIGHT
    },
    {
        "get_sized_ascender",
        (PyCFunction) _ftfont_getsizedascender,
        METH_VARARGS,
        DOC_FONTGETSIZEDASCENDER
    },
    {
        "get_sized_descender",
        (PyCFunction) _ftfont_getsizeddescender,
        METH_VARARGS,
        DOC_FONTGETSIZEDDESCENDER
    },
    {
        "get_sized_glyph_height",
        (PyCFunction) _ftfont_getsizedglyphheight,
        METH_VARARGS,
        DOC_FONTGETSIZEDGLYPHHEIGHT
    },
    {
        "get_rect",
        (PyCFunction) _ftfont_getrect,
        METH_VARARGS | METH_KEYWORDS,
        DOC_FONTGETRECT
    },
    {
        "get_metrics",
        (PyCFunction) _ftfont_getmetrics,
        METH_VARARGS | METH_KEYWORDS,
        DOC_FONTGETMETRICS
    },
    {
        "render",
        (PyCFunction)_ftfont_render,
        METH_VARARGS | METH_KEYWORDS,
        DOC_FONTRENDER
    },
    {
        "render_raw",
        (PyCFunction)_ftfont_render_raw,
        METH_VARARGS | METH_KEYWORDS,
        DOC_FONTRENDERRAW
    },
    {
        "get_transform",
        (PyCFunction)_ftfont_gettransform,
        METH_NOARGS,
        DOC_FONTGETTRANSFORM
    },
    {
        "set_transform",
        (PyCFunction)_ftfont_settransform,
        METH_VARARGS,
        DOC_FONTSETTRANSFORM
    },
    {
        "delete_transform",
        (PyCFunction)_ftfont_deletetransform,
        METH_NOARGS,
        DOC_FONTDELETETRANSFORM
    },

    { NULL, NULL, 0, NULL }
};

/*
 * FREETYPE FONT GETTERS/SETTERS TABLE
 */
static PyGetSetDef _ftfont_getsets[] = {
    {
        "style",
        _ftfont_getstyle,
        _ftfont_setstyle,
        DOC_FONTSTYLE,
        NULL
    },
    {
        "height",
        _ftfont_getheight,
        NULL,
        DOC_FONTHEIGHT,
        NULL
    },
    {
        "ascender",
        _ftfont_getascender,
        NULL,
        DOC_FONTASCENDER,
        NULL
    },
    {
        "descender",
        _ftfont_getdescender,
        NULL,
        DOC_FONTASCENDER,
        NULL
    },
    {
        "name",
        _ftfont_getname,
        NULL,
        DOC_FONTNAME,
        NULL
    },
    {
        "path",
        _ftfont_getpath,
        NULL,
        DOC_FONTPATH,
        NULL
    },
    {
        "fixed_width",
        _ftfont_getfixedwidth,
        NULL,
        DOC_FONTFIXEDWIDTH,
        NULL
    },
    {
        "antialiased",
        _ftfont_getantialias,
        _ftfont_setantialias,
        DOC_FONTANTIALIASED,
        NULL
    },
    {
        "kerning",
        _ftfont_getkerning,
        _ftfont_setkerning,
        DOC_FONTKERNING,
        NULL
    },
    {
        "vertical",
        _ftfont_getvertical,
        _ftfont_setvertical,
        DOC_FONTVERTICAL,
        NULL
    },
    {
        "pad",
        _ftfont_getpad,
        _ftfont_setpad,
        DOC_FONTPAD,
        NULL
    },
    {
        "oblique",
        _ftfont_getstyle_flag,
        _ftfont_setstyle_flag,
        DOC_FONTOBLIQUE,
        (void *)FT_STYLE_OBLIQUE
    },
    {
        "bold",
        _ftfont_getstyle_flag,
        _ftfont_setstyle_flag,
        DOC_FONTBOLD,
        (void *)FT_STYLE_BOLD
    },
    {
        "underline",
        _ftfont_getstyle_flag,
        _ftfont_setstyle_flag,
        DOC_FONTUNDERLINE,
        (void *)FT_STYLE_UNDERLINE
    },
    {
        "underscore",
        _ftfont_getstyle_flag,
        _ftfont_setstyle_flag,
        DOC_FONTUNDERSCORE,
        (void *)FT_STYLE_UNDERSCORE
    },
    {
        "wide",
        _ftfont_getstyle_flag,
        _ftfont_setstyle_flag,
        DOC_FONTWIDE,
        (void *)FT_STYLE_WIDE
    },
    {
        "ucs4",
        _ftfont_getucs4,
        _ftfont_setucs4,
        DOC_FONTUCS4,
        NULL
    },
    {
        "resolution",
        _ftfont_getresolution,
        NULL,
        DOC_FONTRESOLUTION,
        NULL
    },
    {
        "origin",
        _ftfont_getorigin,
        _ftfont_setorigin,
        DOC_FONTORIGIN,
        NULL
    },
#if defined(PGFT_DEBUG_CACHE)
    {
        "_debug_cache_stats",
        _ftfont_getdebugcachestats,
        NULL,
        "_debug cache fields as a tuple",
        NULL
    },
#endif

    { NULL, NULL, NULL, NULL, NULL }
};

/*
 * FREETYPE FONT BASE TYPE TABLE
 */
#define FULL_TYPE_NAME MODULE_NAME FONT_TYPE_NAME

PyTypeObject PyFreeTypeFont_Type = {
    TYPE_HEAD(NULL,0)
    FULL_TYPE_NAME,             /* tp_name */
    sizeof (PyFreeTypeFont),    /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor)_ftfont_dealloc,/* tp_dealloc */
    0,                          /* tp_print */
    0,                          /* tp_getattr */
    0,                          /* tp_setattr */
    0,                          /* tp_compare */
    (reprfunc)_ftfont_repr,     /* tp_repr */
    0,                          /* tp_as_number */
    0,                          /* tp_as_sequence */
    0,                          /* tp_as_mapping */
    0,                          /* tp_hash */
    0,                          /* tp_call */
    0,                          /* tp_str */
    0,                          /* tp_getattro */
    0,                          /* tp_setattro */
    0,                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    DOC_PYGAMEFREETYPEFONT,     /* docstring */
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    _ftfont_methods,            /* tp_methods */
    0,                          /* tp_members */
    _ftfont_getsets,            /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc) _ftfont_init,    /* tp_init */
    0,                          /* tp_alloc */
    (newfunc) _ftfont_new,      /* tp_new */
    0,                          /* tp_free */
    0,                          /* tp_is_gc */
    0,                          /* tp_bases */
    0,                          /* tp_mro */
    0,                          /* tp_cache */
    0,                          /* tp_subclasses */
    0,                          /* tp_weaklist */
    0,                          /* tp_del */
#if PY_VERSION_HEX >= 0x02060000
    0                           /* tp_version_tag */
#endif
};

#undef FULL_TYPE_NAME


/****************************************************
 * CONSTRUCTOR/INIT/DESTRUCTOR
 ****************************************************/
static PyObject *
_ftfont_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds)
{
    PyFreeTypeFont *obj = (PyFreeTypeFont *)(subtype->tp_alloc(subtype, 0));

    if (obj != NULL) {
        obj->id.open_args.flags = 0;
        obj->id.open_args.pathname = NULL;
        obj->path = NULL;
        obj->resolution = 0;
        obj->_internals = NULL;
        obj->ptsize = -1;
        obj->style = FT_STYLE_NORMAL;
        obj->vertical = 0;
        obj->antialias = 1;
        obj->kerning = 0;
        obj->ucs4 = 0;
        obj->origin = 0;
	obj->pad = 0;
        obj->do_transform = 0;
        obj->transform.xx = 0x00010000;
        obj->transform.xy = 0x00000000;
        obj->transform.yx = 0x00000000;
        obj->transform.yy = 0x00010000;
   }
    return (PyObject *)obj;
}

static void
_ftfont_dealloc(PyFreeTypeFont *self)
{
    /* Always try to unload the font even if we cannot grab
     * a freetype instance. */
    _PGFT_UnloadFont(FREETYPE_STATE->freetype, self);

    Py_XDECREF(self->path);
    ((PyObject *)self)->ob_type->tp_free((PyObject *)self);
}

static int
_ftfont_init(PyObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] =  {
        "font", "ptsize", "style", "face_index", "vertical",
        "ucs4", "resolution", "origin", NULL
    };

    PyFreeTypeFont *font = (PyFreeTypeFont *)self;

    PyObject *file, *original_file;
    int face_index = 0;
    int ptsize;
    int font_style;
    int ucs4;
    int vertical;
    unsigned resolution = 0;
    int origin;

    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, -1);

    ptsize = font->ptsize;
    font_style = font->style;
    ucs4 = font->ucs4;
    vertical = font->vertical;
    origin = font->origin;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|iiIiiIi", kwlist,
                                     &file, &ptsize, &font_style, &face_index,
                                     &vertical, &ucs4, &resolution, &origin))
        return -1;

    original_file = file;

    _PGFT_UnloadFont(ft, font);
    Py_XDECREF(font->path);
    font->path = NULL;

    if (_PGFT_CheckStyle(font_style)) {
        PyErr_Format(PyExc_ValueError,
                     "Invalid style value %x", (int)font_style);
        return -1;
    }
    font->ptsize = (FT_Int16)((ptsize <= 0) ? -1 : ptsize);
    font->style = (FT_Byte)font_style;
    font->ucs4 = ucs4 ? (FT_Byte)1 : (FT_Byte)0;
    font->vertical = vertical;
    font->origin = (FT_Byte)origin;
    if (resolution) {
        font->resolution = (FT_UInt)resolution;
    }
    else {
        font->resolution = FREETYPE_STATE->resolution;
    }
    if (file == Py_None) {
        file = load_font_res(DEFAULT_FONT_NAME);

        if (file == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to find default font");
            goto end;
        }
    }

    file = RWopsEncodeFilePath(file, NULL);
    if (file == NULL) {
        return -1;
    }
    if (Bytes_Check(file)) {
        if (_PGFT_TryLoadFont_Filename(ft, font, Bytes_AS_STRING(file),
                                       face_index)) {
            goto end;
        }

        if (PyUnicode_Check(original_file)) {
            /* Make sure to save a pure Unicode object to prevent possible
             * cycles from a derived class. This means no tp_traverse or
             * tp_clear for the PyFreetypeFont type.
             */
            font->path = Object_Unicode(original_file);
        }
        else {
            font->path = PyUnicode_FromEncodedObject(file, "raw_unicode_escape",
                                                     "replace");
        }
    }
    else {
        SDL_RWops *source = RWopsFromFileObject(original_file);
        PyObject *str = NULL;
        PyObject *path = NULL;

        if (source == NULL) {
            goto end;
        }

        if (_PGFT_TryLoadFont_RWops(ft, font, source, face_index)) {
            goto end;
        }

        path = PyObject_GetAttrString(original_file, "name");
        if (!path) {
            PyErr_Clear();
            str = Bytes_FromFormat("<%s instance at %p>",
                                   Py_TYPE(file)->tp_name, (void *)file);
            if (str) {
                font->path = PyUnicode_FromEncodedObject(str,
                                                         "ascii", "strict");
                Py_DECREF(str);
            }
        }
        else if (PyUnicode_Check(path)) {
            /* Make sure to save a pure Unicode object to prevent possible
             * cycles from a derived class. This means no tp_traverse or
             * tp_clear for the PyFreetypeFont type.
             */
            font->path = Object_Unicode(path);
        }
        else if (Bytes_Check(path)) {
            font->path = PyUnicode_FromEncodedObject(path,
                                               "unicode_escape", "replace");
        }
        else {
            font->path = Object_Unicode(path);
        }
        Py_XDECREF(path);
    }

end:

    if (file != original_file) {
        Py_XDECREF(file);
    }

    return PyErr_Occurred() ? -1 : 0;
}

static PyObject *
_ftfont_repr(PyObject *self)
{
    PyFreeTypeFont *font = (PyFreeTypeFont *)self;

    if (PyFreeTypeFont_IS_ALIVE(font)) {
#if PY3
        return PyUnicode_FromFormat("Font('%.1024u')", font->path);
#else
        PyObject *str = PyUnicode_AsEncodedString(font->path,
                                                  "raw_unicode_escape",
                                                  "replace");
        PyObject *rval = NULL;

        if (str) {
            rval = PyString_FromFormat("Font('%.1024s')",
                                       PyString_AS_STRING(str));
            Py_DECREF(str);
        }
        return rval;
#endif
    }
    return Text_FromFormat("<uninitialized Font object at %p>", (void *)self);
}


/****************************************************
 * GETTERS/SETTERS
 ****************************************************/

/** Vertical attribute */
static PyObject *
_ftfont_getvertical(PyObject *self, void *closure)
{
    PyFreeTypeFont *font = (PyFreeTypeFont *)self;

    return PyBool_FromLong(font->vertical);
}

static int
_ftfont_setvertical(PyObject *self, PyObject *value, void *closure)
{
    PyFreeTypeFont *font = (PyFreeTypeFont *)self;

    if (!PyBool_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Expecting 'bool' type");
        return -1;
    }
    font->vertical = (FT_Byte)PyObject_IsTrue(value);
    return 0;
}


/** Antialias attribute */
static PyObject *
_ftfont_getantialias(PyObject *self, void *closure)
{
    PyFreeTypeFont *font = (PyFreeTypeFont *)self;

    return PyBool_FromLong(font->antialias);
}

static int
_ftfont_setantialias(PyObject *self, PyObject *value, void *closure)
{
    PyFreeTypeFont *font = (PyFreeTypeFont *)self;

    if (!PyBool_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Expecting 'bool' type");
        return -1;
    }
    font->antialias = (FT_Byte)PyObject_IsTrue(value);
    return 0;
}


/** kerning attribute */
static PyObject *
_ftfont_getkerning(PyObject *self, void *closure)
{
    PyFreeTypeFont *font = (PyFreeTypeFont *)self;

    return PyBool_FromLong(font->kerning);
}

static int
_ftfont_setkerning(PyObject *self, PyObject *value, void *closure)
{
    PyFreeTypeFont *font = (PyFreeTypeFont *)self;

    if (!PyBool_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Expecting 'bool' type");
        return -1;
    }
    font->kerning = (FT_Byte)PyObject_IsTrue(value);
    return 0;
}


/** Generic style attributes */
static PyObject *
_ftfont_getstyle_flag(PyObject *self, void *closure)
{
    const int style_flag = (int)closure;
    PyFreeTypeFont *font = (PyFreeTypeFont *)self;

    return PyBool_FromLong(font->style & style_flag);
}

static int
_ftfont_setstyle_flag(PyObject *self, PyObject *value, void *closure)
{
    const int style_flag = (int)closure;
    PyFreeTypeFont *font = (PyFreeTypeFont *)self;

    if (!PyBool_Check(value)) {
        PyErr_SetString(PyExc_TypeError,
                "The style value must be a boolean");
        return -1;
    }

    if (PyObject_IsTrue(value)) {
        font->style |= (FT_Byte)style_flag;
    }
    else {
        font->style &= (FT_Byte)(~style_flag);
    }

    return 0;
}


/** Style attribute */
static PyObject *
_ftfont_getstyle (PyObject *self, void *closure)
{
    PyFreeTypeFont *font = (PyFreeTypeFont *)self;

    return PyInt_FromLong(font->style);
}

static int
_ftfont_setstyle(PyObject *self, PyObject *value, void *closure)
{
    PyFreeTypeFont *font = (PyFreeTypeFont *)self;
    FT_UInt32 style;

    if (!PyInt_Check(value)) {
        PyErr_SetString(PyExc_TypeError,
                "The style value must be an integer"
                " from the FT constants module");
        return -1;
    }

    style = (FT_UInt32)PyInt_AsLong(value);

    if (_PGFT_CheckStyle(style) != 0) {
        PyErr_Format(PyExc_ValueError,
                     "Invalid style value %x", (int)style);
        return -1;
    }

    font->style = (FT_Byte)style;
    return 0;
}

static PyObject *
_ftfont_getascender(PyObject *self, void *closure)
{
    FreeTypeInstance *ft;
    long height;
    ASSERT_GRAB_FREETYPE(ft, NULL);

    ASSERT_SELF_IS_ALIVE(self);
    height = _PGFT_Face_GetAscender(ft, (PyFreeTypeFont *)self);
    if (!height && PyErr_Occurred()) {
        return NULL;
    }
    return PyInt_FromLong(height);
}

static PyObject *
_ftfont_getdescender(PyObject *self, void *closure)
{
    FreeTypeInstance *ft;
    long height;
    ASSERT_GRAB_FREETYPE(ft, NULL);

    ASSERT_SELF_IS_ALIVE(self);
    height = _PGFT_Face_GetDescender(ft, (PyFreeTypeFont *)self);
    if (!height && PyErr_Occurred()) {
        return NULL;
    }
    return PyInt_FromLong(height);
}

static PyObject *
_ftfont_getheight(PyObject *self, void *closure)
{
    FreeTypeInstance *ft;
    long height;
    ASSERT_GRAB_FREETYPE(ft, NULL);

    ASSERT_SELF_IS_ALIVE(self);
    height = _PGFT_Face_GetHeight(ft, (PyFreeTypeFont *)self);
    if (!height && PyErr_Occurred()) {
        return NULL;
    }
    return PyInt_FromLong(height);
}

static PyObject *
_ftfont_getname(PyObject *self, void *closure)
{
    FreeTypeInstance *ft;
    const char *name;
    ASSERT_GRAB_FREETYPE(ft, NULL);

    if (PyFreeTypeFont_IS_ALIVE(self)) {
        name = _PGFT_Face_GetName(ft, (PyFreeTypeFont *)self);
        return name != NULL ? Text_FromUTF8(name) : NULL;
    }
    return PyObject_Repr(self);
}

static PyObject *
_ftfont_getpath(PyObject *self, void *closure)
{
    PyObject *path = ((PyFreeTypeFont *)self)->path;

    if (!path) {
        PyErr_SetString(PyExc_AttributeError, "path unavailable");
        return NULL;
    }
    Py_INCREF(path);
    return path;
}

static PyObject *
_ftfont_getfixedwidth(PyObject *self, void *closure)
{
    FreeTypeInstance *ft;
    long fixed_width;
    ASSERT_GRAB_FREETYPE(ft, NULL);

    ASSERT_SELF_IS_ALIVE(self);
    fixed_width = _PGFT_Face_IsFixedWidth(ft, (PyFreeTypeFont *)self);
    return fixed_width >= 0 ? PyBool_FromLong(fixed_width) : NULL;
}


/** ucs4 unicode text handling attribute */
static PyObject *
_ftfont_getucs4(PyObject *self, void *closure)
{
    PyFreeTypeFont *font = (PyFreeTypeFont *)self;

    return PyBool_FromLong(font->ucs4);
}

static int
_ftfont_setucs4(PyObject *self, PyObject *value, void *closure)
{
    PyFreeTypeFont *font = (PyFreeTypeFont *)self;

    if (!PyBool_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Expecting 'bool' type");
        return -1;
    }
    font->ucs4 = (FT_Byte)PyObject_IsTrue(value);
    return 0;
}


/** render at text origin attribute */
static PyObject *
_ftfont_getorigin(PyObject *self, void *closure)
{
    PyFreeTypeFont *font = (PyFreeTypeFont *)self;

    return PyBool_FromLong(font->origin);
}

static int
_ftfont_setorigin(PyObject *self, PyObject *value, void *closure)
{
    PyFreeTypeFont *font = (PyFreeTypeFont *)self;

    if (!PyBool_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Expecting 'bool' type");
        return -1;
    }
    font->origin = (FT_Byte)PyObject_IsTrue(value);
    return 0;
}


/** pad bounding box as font.Font does */
static PyObject *
_ftfont_getpad(PyObject *self, void *closure)
{
    PyFreeTypeFont *font = (PyFreeTypeFont *)self;

    return PyBool_FromLong(font->pad);
}

static int
_ftfont_setpad(PyObject *self, PyObject *value, void *closure)
{
    PyFreeTypeFont *font = (PyFreeTypeFont *)self;

    if (!PyBool_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Expecting 'bool' type");
        return -1;
    }
    font->pad = (FT_Byte)PyObject_IsTrue(value);
    return 0;
}


/** resolution pixel size attribute */
static PyObject *
_ftfont_getresolution(PyObject *self, void *closure)
{
    PyFreeTypeFont *font = (PyFreeTypeFont *)self;

    return PyLong_FromUnsignedLong((unsigned long)font->resolution);
}


/** testing and debugging */
#if defined(_PGFT_DEBUG_CACHE)
static PyObject *
_ftfont_getdebugcachestats(PyObject *self, void *closure)
{
    /* Yes, this kind of breaches the boundary between the top level
     * freetype.c and the lower level ft_text.c. But it is built
     * conditionally, and it keeps some of the Python api out
     * of ft_text.c and ft_cache.c (hoping to remove the Python
     * api completely from ft_text.c and support C modules at some point.)
     */
    const FontCache *cache = &PGFT_FONT_CACHE((PyFreeTypeFont *)self);

    return Py_BuildValue("kkkkk",
                         (unsigned long)cache->_debug_count,
                         (unsigned long)cache->_debug_delete_count,
                         (unsigned long)cache->_debug_access,
                         (unsigned long)cache->_debug_hit,
                         (unsigned long)cache->_debug_miss);
}
#endif

/****************************************************
 * MAIN METHODS
 ****************************************************/
static PyObject *
_ftfont_getrect(PyObject *self, PyObject *args, PyObject *kwds)
{
/* MODIFIED
 */
    /* keyword list */
    static char *kwlist[] =  {
        "text", "style", "rotation", "ptsize", NULL
    };

    PyFreeTypeFont *font = (PyFreeTypeFont *)self;
    PyObject *textobj;
    PGFT_String *text;
    PyObject *rectobj = NULL;
    FT_Error error;
    int ptsize = -1;
    SDL_Rect r;

    FontRenderMode render;
    int rotation = 0;
    int style = 0;

    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, NULL);

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|iii", kwlist,
                                     &textobj, &style, &rotation, &ptsize))
        return NULL;

    /* Encode text */
    text = _PGFT_EncodePyString(textobj, font->ucs4);
    if (text == NULL) {
        return NULL;
    }

    ASSERT_SELF_IS_ALIVE(self);

    /* Build rendering mode, always anti-aliased by default */
    if (_PGFT_BuildRenderMode(ft, font, &render,
                             ptsize, style, rotation) != 0) {
        return NULL;
    }

    error = _PGFT_GetTextRect(ft, font, &render, text, &r);
    _PGFT_FreeString(text);

    if (!error)
        rectobj = PyRect_New(&r);

    return rectobj;
}

static PyObject *
get_metrics(FreeTypeInstance *ft, FontRenderMode *render,
          PyFreeTypeFont *font, PGFT_String *text)
{
    Py_ssize_t length = PGFT_String_GET_LENGTH(text);
    PGFT_char *data = PGFT_String_GET_DATA(text);
    PyObject *list, *item;
    FT_UInt gindex;
    long minx, miny;
    long maxx, maxy;
    double advance_x;
    double advance_y;
    Py_ssize_t i;

    list = PyList_New(length);
    if (list == NULL) {
        return NULL;
    }
    for (i = 0; i < length; ++i) {
        if (_PGFT_GetMetrics(ft, font, data[i], render,
                             &gindex, &minx, &maxx, &miny, &maxy,
                             &advance_x, &advance_y) == 0) {
            if (gindex == 0) {
                Py_INCREF(Py_None);
                item = Py_None;
            }
            else {
                item = Py_BuildValue("lllldd", minx, maxx, miny, maxy,
                                     advance_x, advance_y);
            }
            if (!item) {
                Py_DECREF(list);
                return NULL;
            }
        }
        else {
            Py_INCREF(Py_None);
            item = Py_None;
        }
        PyList_SET_ITEM(list, i, item);
    }

    return list;
}

static PyObject *
_ftfont_getmetrics(PyObject *self, PyObject *args, PyObject *kwds)
{
    /* keyword list */
    static char *kwlist[] =  {
        "text", "ptsize", NULL
    };

    PyFreeTypeFont *font = (PyFreeTypeFont *)self;
    FontRenderMode render;
    PyObject *list = NULL;

    /* arguments */
    PyObject *textobj;
    PGFT_String *text;
    int ptsize = -1;

    /* grab freetype */
    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, NULL);

    /* parse args */
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|i", kwlist,
                                     &textobj, &ptsize))
        return NULL;

    /* Encode text */
    text = _PGFT_EncodePyString(textobj, font->ucs4);
    if (text == NULL) {
        return NULL;
    }

    ASSERT_SELF_IS_ALIVE(self);

    /*
     * Build the render mode with the given size and no
     * rotation/styles/vertical text
     */
    if (_PGFT_BuildRenderMode(ft, font, &render, ptsize, FT_STYLE_NORMAL, 0)) {
        _PGFT_FreeString(text);
        return NULL;
    }

    /* get metrics */
    list = get_metrics(ft, &render, font, text);

    _PGFT_FreeString(text);
    return list;
}

static PyObject *
_ftfont_getsizedascender(PyFreeTypeFont *self, PyObject *args)
{
    int pt_size = -1;
    long value;

    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, NULL);

    if (!PyArg_ParseTuple(args, "|i", &pt_size)) {
        return NULL;
    }

    if (pt_size == -1) {
        if (self->ptsize == -1) {
            RAISE(PyExc_ValueError,
                  "No font point size specified"
                  " and no default font size in typeface");
            return NULL;
        }

        pt_size = self->ptsize;
    }

    if (pt_size <= 0) {
        RAISE(PyExc_ValueError, "Invalid point size for font.");
        return NULL;
    }
    value = (long)_PGFT_Face_GetAscenderSized(ft, self, (FT_UInt16)pt_size);
    if (!value && PyErr_Occurred()) {
        return NULL;
    }
    return PyInt_FromLong(value);
}

static PyObject *
_ftfont_getsizeddescender(PyFreeTypeFont *self, PyObject *args)
{
    int pt_size = -1;
    long value;

    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, NULL);

    if (!PyArg_ParseTuple(args, "|i", &pt_size)) {
        return NULL;
    }

    if (pt_size == -1) {
        if (self->ptsize == -1) {
            RAISE(PyExc_ValueError,
                  "No font point size specified"
                  " and no default font size in typeface");
            return NULL;
        }

        pt_size = self->ptsize;
    }

    if (pt_size <= 0) {
        RAISE(PyExc_ValueError, "Invalid point size for font.");
        return NULL;
    }
    value = (long)_PGFT_Face_GetDescenderSized(ft, self, (FT_UInt16)pt_size);
    if (!value && PyErr_Occurred()) {
        return NULL;
    }
    return PyInt_FromLong(value);
}

static PyObject *
_ftfont_getsizedheight(PyFreeTypeFont *self, PyObject *args)
{
    int pt_size = -1;
    long value;

    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, NULL);

    if (!PyArg_ParseTuple(args, "|i", &pt_size)) {
        return NULL;
    }

    if (pt_size == -1) {
        if (self->ptsize == -1) {
            RAISE(PyExc_ValueError,
                  "No font point size specified"
                  " and no default font size in typeface");
            return NULL;
        }

        pt_size = self->ptsize;
    }

    if (pt_size <= 0) {
        RAISE(PyExc_ValueError, "Invalid point size for font.");
        return NULL;
    }
    value = _PGFT_Face_GetHeightSized(ft, self, (FT_UInt16)pt_size);
    if (!value && PyErr_Occurred()) {
        return NULL;
    }
    return PyInt_FromLong(value);
}

static PyObject *
_ftfont_getsizedglyphheight(PyFreeTypeFont *self, PyObject *args)
{
    int pt_size = -1;
    long value;

    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, NULL);

    if (!PyArg_ParseTuple(args, "|i", &pt_size)) {
        return NULL;
    }

    if (pt_size == -1) {
        if (self->ptsize == -1) {
            RAISE(PyExc_ValueError,
                  "No font point size specified"
                  " and no default font size in typeface");
            return NULL;
        }

        pt_size = self->ptsize;
    }

    if (pt_size <= 0) {
        RAISE(PyExc_ValueError, "Invalid point size for font.");
        return NULL;
    }
    value = (long)_PGFT_Face_GetGlyphHeightSized(ft, self, (FT_UInt16)pt_size);
    if (!value && PyErr_Occurred()) {
        return NULL;
    }
    return PyInt_FromLong(value);
}

static PyObject *
_ftfont_render_raw(PyObject *self, PyObject *args, PyObject *kwds)
{
    /* keyword list */
    static char *kwlist[] =  {
        "text", "style", "rotation", "ptsize", NULL
    };

    PyFreeTypeFont *font = (PyFreeTypeFont *)self;
    FontRenderMode render;

    /* input arguments */
    PyObject *textobj;
    PGFT_String *text;
    int rotation = 0;
    int ptsize = -1;
    int style = FT_STYLE_DEFAULT;

    /* output arguments */
    PyObject *rbuffer = NULL;
    PyObject *rtuple;
    int width, height;

    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, NULL);

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|iii", kwlist,
                                     &textobj, &style, &rotation, &ptsize))
        return NULL;

    /* Encode text */
    text = _PGFT_EncodePyString(textobj, font->ucs4);
    if (text == NULL) {
        return NULL;
    }

    ASSERT_SELF_IS_ALIVE(self);

    /*
     * Build the render mode with the given size and no
     * rotation/styles/vertical text
     */
    if (_PGFT_BuildRenderMode(ft, font,
                              &render, ptsize, style, rotation) != 0) {
        _PGFT_FreeString(text);
        return NULL;
    }

    rbuffer = _PGFT_Render_PixelArray(ft, font, &render, text, &width, &height);
    _PGFT_FreeString(text);

    if (!rbuffer) {
        return NULL;
    }
    rtuple = Py_BuildValue("O(ii)", rbuffer, width, height);
    Py_DECREF(rbuffer);
    return rtuple;
}

static PyObject *
_ftfont_render(PyObject *self, PyObject *args, PyObject *kwds)
{
#ifndef HAVE_PYGAME_SDL_VIDEO

    PyErr_SetString(PyExc_RuntimeError,
		    "SDL support is missing. Cannot render on surfaces");
    return NULL;

#else
    /* keyword list */
    static char *kwlist[] =  {
        "dest", "text", "fgcolor", "bgcolor",
        "style", "rotation", "ptsize", NULL
    };

    PyFreeTypeFont *font = (PyFreeTypeFont *)self;

    /* input arguments */
    PyObject *textobj = NULL;
    PGFT_String *text;
    int ptsize = -1;
    PyObject *dest = NULL;
    PyObject *surface_obj = NULL;
    int xpos = 0;
    int ypos = 0;
    PyObject *fg_color_obj = NULL;
    PyObject *bg_color_obj = NULL;
    int rotation = 0;
    int style = FT_STYLE_DEFAULT;

    /* output arguments */
    PyObject *rtuple = NULL;
    SDL_Rect r;
    PyObject *rect_obj;

    FontColor fg_color;
    FontColor bg_color;
    FontRenderMode render;

    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, NULL);

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOO|Oiii", kwlist,
                &dest, &textobj, &fg_color_obj, /* required */
                &bg_color_obj, &style, &rotation, &ptsize)) /* optional */
        return NULL;

    if (!RGBAFromColorObj(fg_color_obj, (Uint8 *)&fg_color)) {
        PyErr_SetString(PyExc_TypeError, "fgcolor must be a Color");
        return NULL;
    }

    if (bg_color_obj) {
        if (bg_color_obj == Py_None)
            bg_color_obj = NULL;

        else if (!RGBAFromColorObj(bg_color_obj, (Uint8 *)&bg_color)) {
            PyErr_SetString(PyExc_TypeError, "bgcolor must be a Color");
            return NULL;
        }
    }

    /* Encode text */
    text = _PGFT_EncodePyString(textobj, font->ucs4);
    if (text == NULL) {
        return NULL;
    }

    ASSERT_SELF_IS_ALIVE(self);

    if (_PGFT_BuildRenderMode(ft, font,
                              &render, ptsize, style, rotation) != 0) {
        _PGFT_FreeString(text);
        return NULL;
    }

    if (dest == Py_None) {
        SDL_Surface *r_surface = NULL;

        r_surface = _PGFT_Render_NewSurface(ft, font, &render, text, &fg_color,
                                            bg_color_obj ? &bg_color : NULL,
                                            &r);
        _PGFT_FreeString(text);

        if (!r_surface) {
            return NULL;
        }

        surface_obj = PySurface_New(r_surface);
        if (!surface_obj) {
            return NULL;
        }
    }
    else if (PySequence_Check(dest) &&  /* conditional and */
             PySequence_Size(dest) > 1) {
        SDL_Surface *surface = NULL;
        int rcode;

        if (parse_dest(dest, &surface_obj, &xpos, &ypos)) {
            _PGFT_FreeString(text);
            return NULL;
        }

        surface = PySurface_AsSurface(surface_obj);

        rcode = _PGFT_Render_ExistingSurface(ft, font, &render, text, surface,
                                             xpos, ypos, &fg_color,
                                             bg_color_obj ? &bg_color : NULL,
                                             &r);
        _PGFT_FreeString(text);
        if (rcode) {
            Py_DECREF(surface_obj);
            return NULL;
        }
    }
    else {
        _PGFT_FreeString(text);
        return PyErr_Format(PyExc_TypeError,
                            "Expected a (surface, posn) or None for"
			    " dest argument:"
                            " got type %.1024s",
                            Py_TYPE(dest)->tp_name);
    }
    rect_obj = PyRect_New(&r);
    if (rect_obj != NULL) {
        rtuple = PyTuple_Pack(2, surface_obj, rect_obj);
        Py_DECREF(rect_obj);
    }
    Py_DECREF(surface_obj);

    return rtuple;

#endif // HAVE_PYGAME_SDL_VIDEO
}

static PyObject *
_ftfont_gettransform(PyFreeTypeFont *self)
{
    const double scale_factor = 1.5259e-5; /* 1 / 65536.0 */

    if (!self->do_transform) {
        Py_RETURN_NONE;
    }
    return Py_BuildValue("dddd",
                         self->transform.xx * scale_factor,
                         self->transform.xy * scale_factor,
                         self->transform.yx * scale_factor,
                         self->transform.yy * scale_factor);
}

static PyObject *
_ftfont_settransform(PyFreeTypeFont *self, PyObject *args)
{
    const double scale_factor = 65536.0;
    double xx;
    double xy;
    double yx;
    double yy;

    if (!PyArg_ParseTuple(args, "dddd", &xx, &xy, &yx, &yy)) {
        return NULL;
    }
    /* Sanity check: only accept numbers between -2.0 and 2.0 inclusive */
    if (xx < -2.0 || xx > 2.0 || xy < -2.0 || xy > 2.0 ||
        yx < -2.0 || yx > 2.0 || yy < -2.0 || yy > 2.0) {
        PyErr_SetString(PyExc_ValueError,
                        "received a value outside range [-2.0,2.0]");
        return NULL;
    }
    self->transform.xx = (FT_Fixed)(xx * scale_factor);
    self->transform.xy = (FT_Fixed)(xy * scale_factor);
    self->transform.yx = (FT_Fixed)(yx * scale_factor);
    self->transform.yy = (FT_Fixed)(yy * scale_factor);
    self->do_transform = 1;
    Py_RETURN_NONE;
}

static PyObject *
_ftfont_deletetransform(PyFreeTypeFont *self)
{
    self->do_transform = 0;
    self->transform.xx = 0x00010000;
    self->transform.xy = 0x00000000;
    self->transform.yx = 0x00000000;
    self->transform.yy = 0x00010000;
    Py_RETURN_NONE;
}

/****************************************************
 * C API CALLS
 ****************************************************/
static PyObject *
PyFreeTypeFont_New(const char *filename, int face_index)
{
    PyFreeTypeFont *font;

    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, NULL);

    if (!filename)
        return NULL;

    font = (PyFreeTypeFont *)PyFreeTypeFont_Type.tp_new(
            &PyFreeTypeFont_Type, NULL, NULL);

    if (!font)
        return NULL;

    if (_PGFT_TryLoadFont_Filename(ft, font, filename, face_index) != 0) {
        return NULL;
    }

    return (PyObject *) font;
}







/****************************************************
 * FREETYPE MODULE METHODS
 ****************************************************/

/***************************************************************
 *
 * Bindings for initialization/cleanup functions
 *
 * Explicit init/quit functions are required to work around
 * some issues regarding module caching and multi-threaded apps.
 * It's always good to let the user choose when to initialize
 * the module.
 *
 * TODO: These bindings can be removed once proper threading
 * support is in place.
 *
 ***************************************************************/

static PyObject *
_ft_autoinit(PyObject *self)
{
    FT_Error result = 1;

    if (FREETYPE_MOD_STATE(self)->freetype == NULL) {
        result = (_PGFT_Init(&(FREETYPE_MOD_STATE(self)->freetype),
                            FREETYPE_MOD_STATE(self)->cache_size) == 0);
        if (!result)
            return NULL;
    }

    return PyInt_FromLong(result);
}

static PyObject *
_ft_quit(PyObject *self)
{
    if (FREETYPE_MOD_STATE(self)->freetype != NULL) {
        _PGFT_Quit(FREETYPE_MOD_STATE(self)->freetype);
        FREETYPE_MOD_STATE(self)->freetype = NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
_ft_init(PyObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] =  {
        "cache_size", "resolution", NULL
    };

    PyObject *result;
    int cache_size = PGFT_DEFAULT_CACHE_SIZE;
    unsigned resolution = 0;
    _FreeTypeState *state = FREETYPE_MOD_STATE(self);

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iI", kwlist,
                                     &cache_size, &resolution))
        return NULL;

    if (!state->freetype) {
        state->resolution = (resolution ?
                             (FT_UInt)resolution : PGFT_DEFAULT_RESOLUTION);
        result = _ft_autoinit(self);

        if (!PyObject_IsTrue(result)) {
            PyErr_SetString(PyExc_RuntimeError,
                            "Failed to initialize the FreeType2 library");
            return NULL;
        }
    }

    Py_RETURN_NONE;
}


static PyObject *
_ft_get_error(PyObject *self)
{
    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, NULL);

    if (ft->_error_msg[0]) {
        return Text_FromUTF8(ft->_error_msg);
    }

    Py_RETURN_NONE;
}

static PyObject *
_ft_get_version(PyObject *self)
{
    /* Return the linked FreeType2 version */
    return Py_BuildValue("iii", FREETYPE_MAJOR, FREETYPE_MINOR, FREETYPE_PATCH);
}

static PyObject *
_ft_get_default_resolution(PyObject *self)
{
    return PyLong_FromUnsignedLong((unsigned long)(FREETYPE_STATE->resolution));
}

static PyObject *
_ft_set_default_resolution(PyObject *self, PyObject *args)
{
    unsigned resolution = 0;
    _FreeTypeState *state = FREETYPE_MOD_STATE(self);

    if (!PyArg_ParseTuple(args, "|I", &resolution))
        return NULL;

    state->resolution = (resolution ?
                         (FT_UInt)resolution : PGFT_DEFAULT_RESOLUTION);
    Py_RETURN_NONE;
}

static PyObject *
_ft_was_init(PyObject *self)
{
    return PyBool_FromLong((FREETYPE_MOD_STATE (self)->freetype != NULL));
}

static PyObject*
_ft_get_default_font(PyObject* self)
{
    return Text_FromUTF8(DEFAULT_FONT_NAME);
}

#if PY3
static int
_ft_traverse (PyObject *mod, visitproc visit, void *arg)
{
    return 0;
}

static int
_ft_clear (PyObject *mod)
{
    if (FREETYPE_MOD_STATE(mod)->freetype) {
        _PGFT_Quit(FREETYPE_MOD_STATE(mod)->freetype);
        FREETYPE_MOD_STATE(mod)->freetype = NULL;
    }
    return 0;
}
#endif



/****************************************************
 * FREETYPE MODULE DECLARATION
 ****************************************************/
#if PY3
struct PyModuleDef _freetypemodule =
{
    PyModuleDef_HEAD_INIT,
    MODULE_NAME,
    DOC_PYGAMEFREETYPE,
    sizeof(_FreeTypeState),
    _ft_methods,
    NULL,
    _ft_traverse,
    _ft_clear,
    NULL
};
#else
_FreeTypeState _modstate;
#endif

MODINIT_DEFINE (freetype)
{
    PyObject *module, *apiobj, *pygame, *pygame_register_quit, *quit, *rval;
    static void* c_api[PYGAMEAPI_FREETYPE_NUMSLOTS];

    import_pygame_base();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }

    import_pygame_surface();
    if (PyErr_Occurred())  {
        MODINIT_ERROR;
    }

    import_pygame_color();
    if (PyErr_Occurred())  {
        MODINIT_ERROR;
    }

    import_pygame_rwobject();
    if (PyErr_Occurred())  {
        MODINIT_ERROR;
    }

    import_pygame_rect();
    if (PyErr_Occurred())  {
        MODINIT_ERROR;
    }

    /* import needed modules. Do this first so if there is an error
       the module is not loaded.
    */
    pygame = PyImport_ImportModule ("pygame");
    if (!pygame) {
        MODINIT_ERROR;
    }
    pygame_register_quit = PyObject_GetAttrString (pygame, "register_quit");
    Py_DECREF (pygame);
    if (!pygame_register_quit) {
        MODINIT_ERROR;
    }

    /* type preparation */
    if (PyType_Ready(&PyFreeTypeFont_Type) < 0)  {
        Py_DECREF(pygame_register_quit);
        MODINIT_ERROR;
    }

#if PY3
    module = PyModule_Create(&_freetypemodule);
#else
    /* TODO: DOC */
    module = Py_InitModule3(MODULE_NAME, _ft_methods, DOC_PYGAMEFREETYPE);
#endif

    if (module == NULL)  {
        Py_DECREF(pygame_register_quit);
        MODINIT_ERROR;
    }

    FREETYPE_MOD_STATE(module)->freetype = NULL;
    FREETYPE_MOD_STATE(module)->cache_size = PGFT_DEFAULT_CACHE_SIZE;
    FREETYPE_MOD_STATE(module)->resolution = PGFT_DEFAULT_RESOLUTION;

    Py_INCREF((PyObject *)&PyFreeTypeFont_Type);
    if (PyModule_AddObject(module, FONT_TYPE_NAME,
                           (PyObject *)&PyFreeTypeFont_Type) == -1)  {
        Py_DECREF(pygame_register_quit);
        Py_DECREF((PyObject *) &PyFreeTypeFont_Type);
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

#   define DEC_CONST(x)  PyModule_AddIntConstant(module, #x, (int)FT_##x)

    DEC_CONST(STYLE_NORMAL);
    DEC_CONST(STYLE_BOLD);
    DEC_CONST(STYLE_OBLIQUE);
    DEC_CONST(STYLE_UNDERLINE);
    DEC_CONST(STYLE_UNDERSCORE);
    DEC_CONST(STYLE_WIDE);

    DEC_CONST(BBOX_EXACT);
    DEC_CONST(BBOX_EXACT_GRIDFIT);
    DEC_CONST(BBOX_PIXEL);
    DEC_CONST(BBOX_PIXEL_GRIDFIT);

    /* export the c api */
#   if PYGAMEAPI_FREETYPE_NUMSLOTS != 2
#       error Mismatch between number of api slots and actual exports.
#   endif
    c_api[0] = &PyFreeTypeFont_Type;
    c_api[1] = &PyFreeTypeFont_New;

    apiobj = encapsulate_api(c_api, "freetype");
    if (apiobj == NULL)  {
        Py_DECREF (pygame_register_quit);
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

    if (PyModule_AddObject(module, PYGAMEAPI_LOCAL_ENTRY, apiobj) == -1)  {
        Py_DECREF(pygame_register_quit);
        Py_DECREF(apiobj);
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

    quit = PyObject_GetAttrString (module, "quit");
    if (quit == NULL) {  /* assertion */
        Py_DECREF (pygame_register_quit);
        DECREF_MOD (module);
        MODINIT_ERROR;
    }
    rval = PyObject_CallFunctionObjArgs (pygame_register_quit, quit, NULL);
    Py_DECREF (pygame_register_quit);
    Py_DECREF (quit);
    if (rval == NULL) {
        DECREF_MOD (module);
        MODINIT_ERROR;
    }
    Py_DECREF (rval);

    MODINIT_RETURN(module);
}
