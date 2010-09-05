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
 * Auxiliar defines
 */
#define ASSERT_SELF_IS_ALIVE(s)                     \
if (!PyFreeTypeFont_IS_ALIVE(s)) {                  \
    return RAISE(PyExc_RuntimeError,                \
        MODULE_NAME "." FONT_TYPE_NAME              \
        " instance is not initialized");            \
}

#define PGFT_CHECK_BOOL(_pyobj, _var)               \
    if (_pyobj)                                     \
    {                                               \
        if (!PyBool_Check(_pyobj))                  \
        {                                           \
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
    if (tmp != NULL)
    {
        Py_DECREF(result);
        result = tmp;
    }
    else 
    {
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

    if (!PySurface_Check(s))
    {
        PyErr_Format(PyExc_TypeError,
                     "expected a Surface as element 0 of dest:"
                     " got type %.1024s",
                     Py_TYPE(s)->tp_name);
        Py_DECREF(s);
        return -1;
    }
    if (len == 2)
    {
        PyObject *size = PySequence_GetItem(dest, 1);

        if (size == NULL)
        {
            Py_DECREF(s);
            return -1;
        }
        if (!PySequence_Check(size))
        {
            PyErr_Format(PyExc_TypeError,
                         "expected an (x,y) position for element 1"
                         " of dest: got type %.1024s",
                         Py_TYPE(size)->tp_name);
            Py_DECREF(s);
            Py_DECREF(size);
            return -1;
        }
        len = PySequence_Length(size);
        if (len < 2)
        {
            PyErr_Format(PyExc_TypeError,
                         "expected at least a length 2 sequence for element 1"
                         " of dest: not length %d", len);
            Py_DECREF(s);
            Py_DECREF(size);
            return -1;
        }
        oi = PySequence_GetItem(size, 0);
        if (oi == NULL)
        {
            Py_DECREF(s);
            Py_DECREF(size);
            return -1;
        }
        oj = PySequence_GetItem(size, 1);
        Py_DECREF(size);
        if (oj == NULL)
        {
            Py_DECREF(s);
            Py_DECREF(oi);
            return -1;
        }
        if (!PyNumber_Check(oi) || !PyNumber_Check(oj))
        {
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
    else if (len == 3)
    {
        oi = PySequence_GetItem(dest, 1);
        if (oi == NULL)
        {
            Py_DECREF(s);
            return -1;
        }
        oj = PySequence_GetItem(dest, 2);
        if (oj == NULL)
        {
            Py_DECREF(oi);
            Py_DECREF(s);
            return -1;
        }
        if (!PyNumber_Check(oi) || !PyNumber_Check(oj))
        {
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
    else
    {
        Py_DECREF(s);
        PyErr_Format(PyExc_TypeError,
                     "for dest expected a sequence of either 2 or 3:"
                     " not length %d", len);
        return -1;
    }              
    i = PyInt_AsLong(oi);
    Py_DECREF(oi);
    if (i == -1 && PyErr_Occurred())
    {
        Py_DECREF(s);
        Py_DECREF(oj);
        return -1;
    }
    j = PyInt_AsLong(oj);
    Py_DECREF(oj);
    if (j == -1 && PyErr_Occurred())
    {
        Py_DECREF(s);
        return -1;
    }
    *surf = s;
    *x = i;
    *y = j;
    return 0;
}

/*
 * FreeType module declarations
 */
#if PY3
static int _ft_traverse(PyObject *mod, visitproc visit, void *arg);
static int _ft_clear(PyObject *mod);
#endif

static PyObject *_ft_quit(PyObject *self);
static PyObject *_ft_init(PyObject *self, PyObject *args);
static PyObject *_ft_get_version(PyObject *self);
static PyObject *_ft_get_error(PyObject *self);
static PyObject *_ft_was_init(PyObject *self);
static PyObject *_ft_autoinit(PyObject *self);

/*
 * Constructor/init/destructor
 */
static PyObject *_ftfont_new(PyTypeObject *subtype,
                             PyObject *args, PyObject *kwds);
static void _ftfont_dealloc(PyFreeTypeFont *self);
static PyObject *_ftfont_repr(PyObject *self);
static int _ftfont_init(PyObject *self, PyObject *args, PyObject *kwds);

/*
 * Main methods
 */
static PyObject *_ftfont_getsize(PyObject *self, PyObject *args, PyObject *kwds);
static PyObject *_ftfont_getmetrics(PyObject *self, PyObject *args, PyObject *kwds);
static PyObject *_ftfont_render(PyObject *self, PyObject *args, PyObject *kwds);
static PyObject *_ftfont_render_raw(PyObject *self, PyObject *args, PyObject *kwds);

/* static PyObject *_ftfont_copy(PyObject *self); */

/*
 * Getters/setters
 */
static PyObject *_ftfont_getstyle(PyObject *self, void *closure);
static int _ftfont_setstyle(PyObject *self, PyObject *value, void *closure);
static PyObject *_ftfont_getheight(PyObject *self, void *closure);
static PyObject *_ftfont_getname(PyObject *self, void *closure);
static PyObject *_ftfont_getfixedwidth(PyObject *self, void *closure);

static PyObject *_ftfont_getvertical(PyObject *self, void *closure);
static int _ftfont_setvertical(PyObject *self, PyObject *value, void *closure);
static PyObject *_ftfont_getantialias(PyObject *self, void *closure);
static int _ftfont_setantialias(PyObject *self, PyObject *value, void *closure);

static PyObject *_ftfont_getstyle_flag(PyObject *self, void *closure);
static int _ftfont_setstyle_flag(PyObject *self, PyObject *value, void *closure);

/*
 * FREETYPE MODULE METHODS TABLE
 */
static PyMethodDef _ft_methods[] = 
{

    {   
        "__PYGAMEinit__", 
        (PyCFunction) _ft_autoinit, 
        METH_NOARGS,
        "auto initialize function for font" 
    },
    { 
        "init", 
        (PyCFunction) _ft_init, 
        METH_VARARGS, 
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
    { NULL, NULL, 0, NULL },
};


/*
 * FREETYPE FONT METHODS TABLE
 */
static PyMethodDef _ftfont_methods[] = 
{
    {
        "get_size", 
        (PyCFunction) _ftfont_getsize,
        METH_VARARGS | METH_KEYWORDS,
        DOC_FONTGETSIZE
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

    { NULL, NULL, 0, NULL }
};

/*
 * FREETYPE FONT GETTERS/SETTERS TABLE
 */
static PyGetSetDef _ftfont_getsets[] = 
{
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
        "name", 
        _ftfont_getname, 
        NULL,
        DOC_FONTNAME,
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
        "vertical",
        _ftfont_getvertical,
        _ftfont_setvertical,
        DOC_FONTVERTICAL,
        NULL
    },
    {
        "italic",
        _ftfont_getstyle_flag,
        _ftfont_setstyle_flag,
        DOC_FONTITALIC,
        (void *)FT_STYLE_ITALIC
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
    { NULL, NULL, NULL, NULL, NULL }
};

/*
 * FREETYPE FONT BASE TYPE TABLE
 */
#define FULL_TYPE_NAME MODULE_NAME FONT_TYPE_NAME

PyTypeObject PyFreeTypeFont_Type =
{
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
    DOC_PYGAMEFREETYPEFONT, 	/* docstring */
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
PyObject *
_ftfont_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds)
{
    PyFreeTypeFont *obj = (PyFreeTypeFont *)(subtype->tp_alloc(subtype, 0));
    
    if (obj != NULL) {
        obj->id.open_args.flags = 0;
        obj->_internals = NULL;
        /* Set defaults here so not reset by __init__ */
        obj->ptsize = -1;
        obj->style = FT_STYLE_NORMAL;
        obj->vertical = 0;
        obj->antialias = 1;
    }
    return (PyObject *)obj;
}
 
void
_ftfont_dealloc(PyFreeTypeFont *self)
{
    /* Always try to unload the font even if we cannot grab
     * a freetype instance. */
    PGFT_UnloadFont(FREETYPE_STATE->freetype, self);

    ((PyObject *)self)->ob_type->tp_free((PyObject *)self);
}

int
_ftfont_init(PyObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = 
    { 
        "font", "ptsize", "style", "face_index", NULL
    };

    PyFreeTypeFont *font = (PyFreeTypeFont *)self;
    
    PyObject *file, *original_file;
    int face_index = 0;
    int ptsize;
    int font_style;

    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, -1);

    ptsize = font->ptsize;
    font_style = font->style;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|iii", kwlist, 
                &file, &ptsize, &font_style, &face_index))
        return -1;

    original_file = file;

    if (face_index < 0)
    {
        PyErr_SetString(PyExc_ValueError, "Face index cannot be negative");
        goto end;
    }

    PGFT_UnloadFont(ft, font);
    
    /* TODO: Ask for vertical? */

    if (PGFT_CheckStyle(font_style))
    {
        PyErr_Format(PyExc_ValueError,
                     "Invalid style value %x", (int)font_style);
        return -1;
    }
    font->ptsize = (FT_Int16)((ptsize <= 0) ? -1 : ptsize);
    font->style = (FT_Byte)font_style;

    if (file == Py_None)
    {
        file = load_font_res(DEFAULT_FONT_NAME);

        if (file == NULL)
        {
            PyErr_SetString(PyExc_RuntimeError, "Failed to find default font");
            goto end;
        }
    }

    if (PyUnicode_Check(file)) 
    {
        file = PyUnicode_AsASCIIString(file);

        if (file == NULL)
        {
            PyErr_SetString(PyExc_ValueError, "Failed to decode filename");
            goto end;
        }
    }

    if (Bytes_Check(file))
    {
        PGFT_TryLoadFont_Filename(ft, font, Bytes_AS_STRING(file), face_index);
    }
    else
    {
        SDL_RWops *source = RWopsFromPython(file);

        if (source == NULL)
        {
            goto end;
        }

        if (PGFT_TryLoadFont_RWops(ft, font, source, face_index) != 0);
        {
            goto end;
        }

        Py_INCREF(file);
        Py_INCREF(file); /* Is this necessary? */
        Py_INCREF(file);
        Py_INCREF(file);
        return 0;
    }

end:

    if (file != original_file)
    {
        Py_XDECREF(file);
    }

    return PyErr_Occurred() ? -1 : 0;
}

PyObject *
_ftfont_repr(PyObject *self)
{
    PyFreeTypeFont *font = (PyFreeTypeFont *)self;

    if (PyFreeTypeFont_IS_ALIVE(font))
        return Text_FromFormat("Font('%.1024s')", font->id.open_args.pathname);
    return Text_FromFormat("<uninitialized Font object at %p>", (void *)self);
}


/****************************************************
 * GETTERS/SETTERS
 ****************************************************/

/** Vertical attribute */
PyObject *
_ftfont_getvertical(PyObject *self, void *closure)
{
    PyFreeTypeFont *font = (PyFreeTypeFont *)self;
    
    return PyBool_FromLong(font->vertical);
}

int
_ftfont_setvertical(PyObject *self, PyObject *value, void *closure)
{
    PyFreeTypeFont *font = (PyFreeTypeFont *)self;
    
    if (!PyBool_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "Expecting 'bool' type");
        return -1;
    }
    font->vertical = (FT_Byte)PyObject_IsTrue(value);
    return 0;
}


/** Antialias attribute */
PyObject *
_ftfont_getantialias(PyObject *self, void *closure)
{
    PyFreeTypeFont *font = (PyFreeTypeFont *)self;
    
    return PyBool_FromLong(font->antialias);
}

int
_ftfont_setantialias(PyObject *self, PyObject *value, void *closure)
{
    PyFreeTypeFont *font = (PyFreeTypeFont *)self;
    
    if (!PyBool_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "Expecting 'bool' type");
        return -1;
    }
    font->antialias = (FT_Byte)PyObject_IsTrue(value);
    return 0;
}


/** Generic style attributes */
PyObject *
_ftfont_getstyle_flag(PyObject *self, void *closure)
{
    const int style_flag = (int)closure;
    PyFreeTypeFont *font = (PyFreeTypeFont *)self;

    return PyBool_FromLong(font->style & style_flag);
}

int
_ftfont_setstyle_flag(PyObject *self, PyObject *value, void *closure)
{
    const int style_flag = (int)closure;
    PyFreeTypeFont *font = (PyFreeTypeFont *)self;

    if (!PyBool_Check(value))
    {
        PyErr_SetString(PyExc_TypeError,
                "The style value must be a boolean");
        return -1;
    }

    if (PyObject_IsTrue(value))
    {
        font->style |= (FT_Byte)style_flag;
    }
    else
    {
        font->style &= (FT_Byte)(~style_flag);
    }

    return 0;
}


/** Style attribute */
PyObject *
_ftfont_getstyle (PyObject *self, void *closure)
{
    PyFreeTypeFont *font = (PyFreeTypeFont *)self;

    return PyInt_FromLong(font->style);
}

int
_ftfont_setstyle(PyObject *self, PyObject *value, void *closure)
{
    PyFreeTypeFont *font = (PyFreeTypeFont *)self;
    FT_UInt32 style;

    if (!PyInt_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, 
                "The style value must be an integer"
                " from the FT constants module");
        return -1;
    }

    style = (FT_UInt32)PyInt_AsLong(value);

    if (PGFT_CheckStyle(style) != 0)
    {
        PyErr_Format(PyExc_ValueError,
                     "Invalid style value %x", (int)style);
        return -1;
    }

    font->style = (FT_Byte)style;
    return 0;
}

PyObject *
_ftfont_getheight(PyObject *self, void *closure)
{
    FreeTypeInstance *ft;
    long height;
    ASSERT_GRAB_FREETYPE(ft, NULL);

    ASSERT_SELF_IS_ALIVE(self);
    height = PGFT_Face_GetHeight(ft, (PyFreeTypeFont *)self);
    return height >= 0 ? PyInt_FromLong(height) : NULL;
}

PyObject *
_ftfont_getname(PyObject *self, void *closure)
{
    FreeTypeInstance *ft;
    const char *name;
    ASSERT_GRAB_FREETYPE(ft, NULL);

    if (PyFreeTypeFont_IS_ALIVE(self))
    {
        name = PGFT_Face_GetName(ft, (PyFreeTypeFont *)self);
        return name != NULL ? Text_FromUTF8(name) : NULL;
    }
    return PyObject_Repr(self);
}

PyObject *
_ftfont_getfixedwidth(PyObject *self, void *closure)
{
    FreeTypeInstance *ft;
    long fixed_width;
    ASSERT_GRAB_FREETYPE(ft, NULL);

    ASSERT_SELF_IS_ALIVE(self);
    fixed_width = PGFT_Face_IsFixedWidth(ft, (PyFreeTypeFont *)self);
    return fixed_width >= 0 ? PyBool_FromLong(fixed_width) : NULL;
}



/****************************************************
 * MAIN METHODS
 ****************************************************/
PyObject *
_ftfont_getsize(PyObject *self, PyObject *args, PyObject *kwds)
{
    /* keyword list */
    static char *kwlist[] = 
    { 
        "text", "style", "rotation", "ptsize", "surrogates", NULL
    };

    PyFreeTypeFont *font = (PyFreeTypeFont *)self;
    PyObject *textobj;
    PGFT_String *text;
    PyObject *rtuple = NULL;
    FT_Error error;
    int ptsize = -1;
    int surrogates = 0;
    int width, height;

    FontRenderMode render;
    int rotation = 0;
    int style = 0;

    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, NULL);

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|iiii", kwlist, 
                &textobj, &style, &rotation, &ptsize, &surrogates))
        return NULL;

    /* Encode text */
    text = PGFT_EncodePyString(textobj, surrogates);
    if (text == NULL)
    {
        return NULL;
    }

    ASSERT_SELF_IS_ALIVE(self);

    /* Build rendering mode, always anti-aliased by default */
    if (PGFT_BuildRenderMode(ft, font, &render, 
                ptsize, style, rotation) != 0)
    {
        return NULL;
    }

    error = PGFT_GetTextSize(ft, font, &render, text, &width, &height);
    PGFT_FreeString(text);

    if (!error)
        rtuple = Py_BuildValue ("(ii)", width, height);

    return rtuple;
}

#define _TUPLE_FORMAT(_ot) ("(" _ot _ot _ot _ot _ot ")")
#define _DEFINE_GET_METRICS(_mt, _ot)                   \
static PyObject *                                       \
_PGFT_get_metrics_##_mt(FreeTypeInstance *ft,           \
                        FontRenderMode render,          \
                        int bbmode,                     \
                        PyFreeTypeFont *font,           \
                        PGFT_String *text);             \
PyObject *                                              \
_PGFT_get_metrics_##_mt(FreeTypeInstance *ft,           \
                        FontRenderMode render,          \
                        int bbmode,                     \
                        PyFreeTypeFont *font,           \
                        PGFT_String *text)              \
{                                                       \
    Py_ssize_t length = PGFT_String_GET_LENGTH(text);   \
    PGFT_char *data = PGFT_String_GET_DATA(text);       \
    PyObject *list, *item;                              \
    _mt minx_##_mt, miny_##_mt;                         \
    _mt maxx_##_mt, maxy_##_mt;                         \
    _mt advance_##_mt;                                  \
    Py_ssize_t i;                                       \
                                                        \
    list = PyList_New(length);                          \
    if (list == NULL) {                                 \
        return NULL;                                    \
    }                                                   \
    for (i = 0; i < length; ++i) {                      \
        if (PGFT_GetMetrics(ft, font, data[i],          \
                &render, bbmode,                        \
                &minx_##_mt, &maxx_##_mt,               \
                &miny_##_mt, &maxy_##_mt,               \
                &advance_##_mt) == 0) {                 \
            item = Py_BuildValue(_TUPLE_FORMAT(_ot),    \
                       minx_##_mt, maxx_##_mt,          \
                       miny_##_mt, maxy_##_mt,          \
                       advance_##_mt);                  \
            if (!item) {                                \
                Py_DECREF(list);                        \
                return NULL;                            \
            }                                           \
        }                                               \
        else {                                          \
            Py_INCREF(Py_None);                         \
            item = Py_None;                             \
        }                                               \
        PyList_SET_ITEM(list, i, item);                 \
    }                                                   \
                                                        \
    return list;                                        \
}
#define _GET_METRICS(_mt, ft, r, b, fo, o) \
    _PGFT_get_metrics_##_mt((ft), (r), (b), (fo), (o))

_DEFINE_GET_METRICS(int, "i")
_DEFINE_GET_METRICS(float, "f")

PyObject *
_ftfont_getmetrics(PyObject *self, PyObject *args, PyObject *kwds)
{
    /* keyword list */
    static char *kwlist[] = 
    { 
        "text", "ptsize", "bbmode", "surrogates", NULL
    };

    PyFreeTypeFont *font = (PyFreeTypeFont *)self;
    FontRenderMode render;
    PyObject *list = NULL;

    /* arguments */
    PyObject *textobj;
    PGFT_String *text;
    int ptsize = -1;
    int bbmode = FT_BBOX_PIXEL_GRIDFIT;
    int surrogates = 0;

    /* grab freetype */
    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, NULL);

    /* parse args */
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|iii", kwlist,
                &textobj, &ptsize, &bbmode, &surrogates))
        return NULL;

    /* Encode text */
    text = PGFT_EncodePyString(textobj, surrogates);
    if (text == NULL)
    {
        return NULL;
    }

    ASSERT_SELF_IS_ALIVE(self);

    /*
     * Build the render mode with the given size and no
     * rotation/styles/vertical text
     */
    if (PGFT_BuildRenderMode(ft, font, 
                &render, ptsize, FT_STYLE_NORMAL, 0) != 0)
    {
        PGFT_FreeString(text);
        return NULL;
    }

    /* get metrics */
    if (bbmode == FT_BBOX_EXACT || bbmode == FT_BBOX_EXACT_GRIDFIT)
    {
        list = _GET_METRICS(float, ft, render, bbmode, font, text);
    }
    else if (bbmode == FT_BBOX_PIXEL || bbmode == FT_BBOX_PIXEL_GRIDFIT)
    {
        list = _GET_METRICS(int, ft, render, bbmode, font, text);
    }
    else
    {
        PyErr_SetString(PyExc_ValueError, "Invalid bbox mode specified");
    }

    PGFT_FreeString(text);
    return list;
}

#undef _TUPLE_FORMAT
#undef _DEFINE_GET_METRICS
#undef _GET_METRICS

PyObject *
_ftfont_render_raw(PyObject *self, PyObject *args, PyObject *kwds)
{
    /* keyword list */
    static char *kwlist[] = 
    { 
        "text", "ptsize", "surrogates", NULL
    };

    PyFreeTypeFont *font = (PyFreeTypeFont *)self;
    FontRenderMode render;

    /* input arguments */
    PyObject *textobj;
    PGFT_String *text;
    int ptsize = -1;
    int surrogates = 0;

    /* output arguments */
    PyObject *rbuffer = NULL;
    PyObject *rtuple;
    int width, height;

    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, NULL);

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|ii", kwlist,
                                     &textobj, &ptsize, &surrogates))
        return NULL;

    /* Encode text */
    text = PGFT_EncodePyString(textobj, surrogates);
    if (text == NULL)
    {
        return NULL;
    }

    ASSERT_SELF_IS_ALIVE(self);
    
    /*
     * Build the render mode with the given size and no
     * rotation/styles/vertical text
     */
    if (PGFT_BuildRenderMode(ft, font, 
                &render, ptsize, FT_STYLE_NORMAL, 0) != 0)
    {
        PGFT_FreeString(text);
        return NULL;
    }

    rbuffer = PGFT_Render_PixelArray(ft, font, &render, text, &width, &height);
    PGFT_FreeString(text);

    if (!rbuffer)
    {
        return NULL;
    }

    rtuple = Py_BuildValue("(O(ii))", rbuffer, width, height);
    Py_DECREF(rbuffer);
    return rtuple;
}

PyObject *
_ftfont_render(PyObject *self, PyObject *args, PyObject *kwds)
{
#ifndef HAVE_PYGAME_SDL_VIDEO

    PyErr_SetString(PyExc_RuntimeError, "SDL support is missing. Cannot render on surfaces");
    return NULL;

#else
    /* keyword list */
    static char *kwlist[] = 
    { 
        "dest", "text", "fgcolor", "bgcolor", 
        "style", "rotation", "ptsize", "surrogates", NULL
    };

    PyFreeTypeFont *font = (PyFreeTypeFont *)self;

    /* input arguments */
    PyObject *textobj = NULL;
    PGFT_String *text;
    int ptsize = -1;
    int surrogates = 0;
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
    int width, height;
    PyObject *rect_obj;

    FontColor fg_color, bg_color;
    FontRenderMode render;

    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, NULL);

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOO|Oiiii", kwlist,
                &dest, &textobj, &fg_color_obj, /* required */
                &bg_color_obj, &style, &rotation, /* optional */
                &ptsize, &surrogates)) /* optional */
        return NULL;

    if (!RGBAFromColorObj(fg_color_obj, (Uint8 *)&fg_color))
    {
        PyErr_SetString(PyExc_TypeError, "fgcolor must be a Color");
        return NULL;
    }

    if (bg_color_obj)
    {
        if (bg_color_obj == Py_None)
            bg_color_obj = NULL;
        
        else if (!RGBAFromColorObj(bg_color_obj, (Uint8 *)&bg_color))
        {
            PyErr_SetString(PyExc_TypeError, "bgcolor must be a Color");
            return NULL;
        }
    }

    /* Encode text */
    text = PGFT_EncodePyString(textobj, surrogates);
    if (text == NULL)
    {
        return NULL;
    }

    ASSERT_SELF_IS_ALIVE(self);

    if (PGFT_BuildRenderMode(ft, font, 
                &render, ptsize, style, rotation) != 0)
    {
        PGFT_FreeString(text);
        return NULL;
    }

    if (dest == Py_None)
    {
        SDL_Surface *r_surface = NULL;

        r_surface = PGFT_Render_NewSurface(ft, font, &render, text,
                &fg_color, bg_color_obj ? &bg_color : NULL, 
                &width, &height);
        PGFT_FreeString(text);

        if (!r_surface)
        {
            return NULL;
        }

        surface_obj = PySurface_New(r_surface);
        if (!surface_obj)
        {
            return NULL;
        }
    }
    else if (PySequence_Check(dest) &&  /* conditional and */
             PySequence_Size(dest) > 1)
    {
        SDL_Surface *surface = NULL;
        int rcode;

        if (parse_dest(dest, &surface_obj, &xpos, &ypos))
        {
            PGFT_FreeString(text);
            return NULL;
        }

        surface = PySurface_AsSurface(surface_obj);

        rcode = PGFT_Render_ExistingSurface(ft, font, &render, 
                    text, surface, xpos, ypos, 
                    &fg_color, bg_color_obj ? &bg_color : NULL,
                    &width, &height);
        PGFT_FreeString(text);
        if (rcode)
        {
            Py_DECREF(surface_obj);
            return NULL;
        }
    }
    else
    {
        PGFT_FreeString(text);
        return PyErr_Format(PyExc_TypeError,
                            "Expected a (surface, posn) or None for dest argument:"
                            " got type %.1024s",
                            Py_TYPE(dest)->tp_name);
    }
    rect_obj = PyRect_New4(xpos, ypos, width, height);
    if (rect_obj != NULL)
    {
        rtuple = PyTuple_Pack(2, surface_obj, rect_obj);
        Py_DECREF(rect_obj);
    }
    Py_DECREF(surface_obj);

    return rtuple;

#endif // HAVE_PYGAME_SDL_VIDEO
}

/****************************************************
 * C API CALLS
 ****************************************************/
PyObject *
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

    if (PGFT_TryLoadFont_Filename(ft, font, filename, face_index) != 0)
    {
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

PyObject *
_ft_autoinit(PyObject *self)
{
    FT_Error result = 1;

    if (FREETYPE_MOD_STATE(self)->freetype == NULL)
    {
        result = (PGFT_Init(&(FREETYPE_MOD_STATE(self)->freetype), 
                    PGFT_DEFAULT_CACHE_SIZE) == 0);
        if (!result)
            return NULL;
    }

    return PyInt_FromLong(result);
}

PyObject *
_ft_quit(PyObject *self)
{
    if (FREETYPE_MOD_STATE(self)->freetype != NULL)
    {
        PGFT_Quit(FREETYPE_MOD_STATE(self)->freetype);
        FREETYPE_MOD_STATE(self)->freetype = NULL;
    }
    Py_RETURN_NONE;
}

PyObject *
_ft_init(PyObject *self, PyObject *args)
{
    PyObject *result;
    FT_Int cache_size = PGFT_DEFAULT_CACHE_SIZE;

    if (!PyArg_ParseTuple(args, "|i", &cache_size))
        return NULL;

    result = _ft_autoinit(self);

    if (!PyObject_IsTrue(result))
    {
        PyErr_SetString(PyExc_RuntimeError, 
                "Failed to initialize the FreeType2 library");
        return NULL;
    }

    FREETYPE_MOD_STATE(self)->freetype->cache_size = cache_size;
    Py_RETURN_NONE;
}


PyObject *
_ft_get_error(PyObject *self)
{
    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, NULL);

    if (ft->_error_msg[0])
    {
        return Text_FromUTF8(ft->_error_msg);
    }

    Py_RETURN_NONE;
}

PyObject *
_ft_get_version(PyObject *self)
{
    /* Return the linked FreeType2 version */
    return Py_BuildValue("(iii)", FREETYPE_MAJOR, FREETYPE_MINOR, FREETYPE_PATCH);
}

PyObject *
_ft_was_init(PyObject *self)
{
    return PyBool_FromLong((FREETYPE_MOD_STATE (self)->freetype != NULL));
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
    if (FREETYPE_MOD_STATE(mod)->freetype)
    {
        PGFT_Quit(FREETYPE_MOD_STATE(mod)->freetype);
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

    PyFREETYPE_C_API[0] = PyFREETYPE_C_API[0]; 

    import_pygame_base();
    if (PyErr_Occurred())
    {   
        MODINIT_ERROR;
    }

    import_pygame_surface();
    if (PyErr_Occurred()) 
    {
	    MODINIT_ERROR;
    }

    import_pygame_color();
    if (PyErr_Occurred()) 
    {
	    MODINIT_ERROR;
    }

    import_pygame_rwobject();
    if (PyErr_Occurred()) 
    {
	    MODINIT_ERROR;
    }

    import_pygame_rect();
    if (PyErr_Occurred()) 
    {
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
    if (PyType_Ready(&PyFreeTypeFont_Type) < 0) 
    {
        Py_DECREF(pygame_register_quit);
        MODINIT_ERROR;
    }

#if PY3
    module = PyModule_Create(&_freetypemodule);
#else
    /* TODO: DOC */
    module = Py_InitModule3(MODULE_NAME, _ft_methods, DOC_PYGAMEFREETYPE); 
#endif

    if (module == NULL) 
    {
        Py_DECREF(pygame_register_quit);
        MODINIT_ERROR;
    }

    Py_INCREF((PyObject *)&PyFreeTypeFont_Type);
    if (PyModule_AddObject(module, FONT_TYPE_NAME,
                           (PyObject *)&PyFreeTypeFont_Type) == -1) 
    {
        Py_DECREF(pygame_register_quit);
        Py_DECREF((PyObject *) &PyFreeTypeFont_Type);
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

#   define DEC_CONST(x)  PyModule_AddIntConstant(module, #x, (int)FT_##x)

    DEC_CONST(STYLE_NORMAL);
    DEC_CONST(STYLE_BOLD);
    DEC_CONST(STYLE_ITALIC);
    DEC_CONST(STYLE_UNDERLINE);

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

    apiobj = PyCObject_FromVoidPtr(c_api, NULL);
    if (apiobj == NULL) 
    {
        Py_DECREF (pygame_register_quit);
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

    if (PyModule_AddObject(module, PYGAMEAPI_LOCAL_ENTRY, apiobj) == -1) 
    {
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
