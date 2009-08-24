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

/*
 * Auxiliar defines
 */
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

static PyObject*
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

/*
 * FreeType module declarations
 */
static int _ft_traverse (PyObject *mod, visitproc visit, void *arg);
static int _ft_clear (PyObject *mod);

static PyObject *_ft_quit(PyObject *self);
static PyObject *_ft_init(PyObject *self, PyObject *args);
static PyObject *_ft_get_version(PyObject *self);
static PyObject *_ft_get_error(PyObject *self);
static PyObject *_ft_was_init(PyObject *self);
static PyObject* _ft_autoinit(PyObject* self);
static void _ft_autoquit(void);

/*
 * Constructor/init/destructor
 */
static void _ftfont_dealloc(PyFreeTypeFont *self);
static PyObject *_ftfont_repr(PyObject *self);
static int _ftfont_init(PyObject *self, PyObject *args, PyObject *kwds);

/*
 * Main methods
 */
static PyObject* _ftfont_getsize(PyObject *self, PyObject* args, PyObject *kwds);
static PyObject* _ftfont_getmetrics(PyObject *self, PyObject* args, PyObject *kwds);
static PyObject* _ftfont_render(PyObject *self, PyObject* args, PyObject *kwds);
static PyObject* _ftfont_render_raw(PyObject *self, PyObject* args, PyObject *kwds);

/* static PyObject* _ftfont_copy(PyObject *self); */

/*
 * Getters/setters
 */
static PyObject* _ftfont_getstyle(PyObject *self, void *closure);
static int _ftfont_setstyle(PyObject *self, PyObject *value, void *closure);
static PyObject* _ftfont_getheight(PyObject *self, void *closure);
static PyObject* _ftfont_getname(PyObject *self, void *closure);
static PyObject* _ftfont_getfixedwidth(PyObject *self, void *closure);

static PyObject* _ftfont_getvertical(PyObject *self, void *closure);
static int _ftfont_setvertical(PyObject *self, PyObject *value, void *closure);
static PyObject* _ftfont_getantialias(PyObject *self, void *closure);
static int _ftfont_setantialias(PyObject *self, PyObject *value, void *closure);

static PyObject* _ftfont_getstyle_flag(PyObject *self, void *closure);
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
/*    { 
        "render_raw", 
        (PyCFunction)_ftfont_render_raw, 
        METH_VARARGS | METH_KEYWORDS,
        DOC_PYGAMEFREETYPEFONTRENDERRAW
    }, */
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
PyTypeObject PyFreeTypeFont_Type =
{
    TYPE_HEAD(NULL,0)
    "freetype.Font",            /* tp_name */
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
    0,                          /* tp_new */
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



/****************************************************
 * CONSTRUCTOR/INIT/DESTRUCTOR
 ****************************************************/
static void
_ftfont_dealloc(PyFreeTypeFont *self)
{
    /* Always try to unload the font even if we cannot grab
     * a freetype instance. */
    PGFT_UnloadFont(FREETYPE_STATE->freetype, self);

    ((PyObject*)self)->ob_type->tp_free((PyObject *)self);
}

static int
_ftfont_init(PyObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = 
    { 
        "font", "ptsize", "style", "face_index", NULL
    };

    PyFreeTypeFont *font = (PyFreeTypeFont *)self;
    
    PyObject *file, *original_file;
    int face_index = 0;
    int ptsize = -1;
    int font_style = FT_STYLE_NORMAL;

    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, -1);

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|iii", kwlist, 
                &file, &ptsize, &font_style, &face_index))
        return -1;

    original_file = file;

    if (face_index < 0)
    {
        PyErr_SetString(PyExc_ValueError, "Face index cannot be negative");
        goto end;
    }
    
    /* TODO: Check style? Ask for vertical? */

    font->ptsize = (FT_Int16)((ptsize <= 0) ? -1 : ptsize);
    font->style = (FT_Byte)font_style;
    font->antialias = 1;
    font->vertical = 0;

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
        char *filename;
        filename = Bytes_AsString(file);

        if (filename == NULL)
        {
            PyErr_SetString(PyExc_ValueError, "Failed to decode filename");
            goto end;
        }

        if (PGFT_TryLoadFont_Filename(ft, font, filename, face_index) != 0)
        {
            PyErr_SetString(PyExc_RuntimeError, PGFT_GetError(ft));
            goto end;
        }
    }
    else
    {
        SDL_RWops *source;
        source = RWopsFromPython(file);

        if (source == NULL)
        {
            PyErr_SetString(PyExc_ValueError, 
                    "Invalid 'file' parameter (must be a File object or a file name)");
            goto end;
        }

        if (PGFT_TryLoadFont_RWops(ft, font, source, face_index) != 0);
        {
            PyErr_SetString(PyExc_RuntimeError, PGFT_GetError(ft));
            goto end;
        }

        Py_INCREF(file);
        Py_INCREF(file);
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

static PyObject*
_ftfont_repr(PyObject *self)
{
    PyFreeTypeFont *font = (PyFreeTypeFont *)self;
    return Text_FromFormat("Font('%s')", font->id.open_args.pathname);
}


/****************************************************
 * GETTERS/SETTERS
 ****************************************************/

/** Vertical attribute */
static PyObject*
_ftfont_getvertical(PyObject *self, void *closure)
{
    PyFreeTypeFont *font = (PyFreeTypeFont *)self;
    return PyBool_FromLong(font->vertical);
}

static int
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
static PyObject*
_ftfont_getantialias(PyObject *self, void *closure)
{
    PyFreeTypeFont *font = (PyFreeTypeFont *)self;
    return PyBool_FromLong(font->antialias);
}

static int
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
static PyObject*
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
static PyObject*
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
        PyErr_SetString(PyExc_ValueError, "Invalid style value");
        return -1;
    }

    font->style = (FT_Byte)style;
    return 0;
}

static PyObject*
_ftfont_getheight(PyObject *self, void *closure)
{
    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, NULL);

    return PyInt_FromLong(PGFT_Face_GetHeight(ft, (PyFreeTypeFont *)self));
}

static PyObject*
_ftfont_getname(PyObject *self, void *closure)
{
    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, NULL);

    return Text_FromUTF8(PGFT_Face_GetName(ft, (PyFreeTypeFont *)self));
}

static PyObject*
_ftfont_getfixedwidth(PyObject *self, void *closure)
{
    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, NULL);

    return PyBool_FromLong(PGFT_Face_IsFixedWidth(ft, (PyFreeTypeFont *)self));
}



/****************************************************
 * MAIN METHODS
 ****************************************************/
static PyObject*
_ftfont_getsize(PyObject *self, PyObject* args, PyObject *kwds)
{
    /* keyword list */
    static char *kwlist[] = 
    { 
        "text", "style", "rotation", "ptsize", NULL
    };

    PyFreeTypeFont *font = (PyFreeTypeFont *)self;
    PyObject *text, *rtuple = NULL;
    FT_Error error;
    int ptsize = -1;
    int width, height;

    FontRenderMode render;
    int rotation = 0;
    int style = 0;

    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, NULL);

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|iii", kwlist, 
                &text, &style, &rotation, &ptsize))
        return NULL;

    /* Build rendering mode, always anti-aliased by default */
    if (PGFT_BuildRenderMode(ft, font, &render, 
                ptsize, style, rotation) != 0)
    {
        PyErr_SetString(PyExc_ValueError, PGFT_GetError(ft));
        return NULL;
    }

    error = PGFT_GetTextSize(ft, font, &render,text, &width, &height);

    if (error)
        PyErr_SetString(PyExc_RuntimeError, PGFT_GetError(ft));
    else
        rtuple = Py_BuildValue ("(ii)", width, height);

    return rtuple;
}

static PyObject *
_ftfont_getmetrics(PyObject *self, PyObject* args, PyObject *kwds)
{
    /* keyword list */
    static char *kwlist[] = 
    { 
        "text", "ptsize", "bbmode", NULL
    };

    PyFreeTypeFont *font = (PyFreeTypeFont *)self;
    FontRenderMode render;

    /* aux vars */
    void *buf = NULL;
    int char_id, length, i, isunicode = 0;

    /* arguments */
    PyObject *text, *list;
    int ptsize = -1;
    int bbmode = FT_BBOX_PIXEL_GRIDFIT;

    /* grab freetype */
    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, NULL);

    /* parse args */
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|ii", kwlist,
                &text, &ptsize, &bbmode))
        return NULL;

    /*
     * Build the render mode with the given size and no
     * rotation/styles/vertical text
     */
    if (PGFT_BuildRenderMode(ft, font, 
                &render, ptsize, FT_STYLE_NORMAL, 0) != 0)
    {
        PyErr_SetString(PyExc_ValueError, PGFT_GetError(ft));
        return NULL;
    }

    /* check text */
    if (PyUnicode_Check(text))
    {
        buf = PyUnicode_AsUnicode(text);
        isunicode = 1;
    }
    else if (Bytes_Check(text))
    {
        buf = Bytes_AS_STRING(text);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError,
            "argument must be a string or unicode");
    }

    if (!buf)
        return NULL;

    if (isunicode)
        length = PyUnicode_GetSize(text);
    else
        length = Bytes_Size(text);

#define _GET_METRICS(_mt, _tuple_format) {              \
    for (i = 0; i < length; i++)                        \
    {                                                   \
        _mt minx_##_mt, miny_##_mt;                     \
        _mt maxx_##_mt, maxy_##_mt;                     \
        _mt advance_##_mt;                              \
                                                        \
        if (isunicode) char_id = ((Py_UNICODE *)buf)[i];\
        else char_id = ((char *)buf)[i];                \
                                                        \
        if (PGFT_GetMetrics(ft,                         \
                (PyFreeTypeFont *)self, char_id,        \
                &render, bbmode,                        \
                &minx_##_mt, &maxx_##_mt,               \
                &miny_##_mt, &maxy_##_mt,               \
                &advance_##_mt) == 0)                   \
        {                                               \
            PyList_SetItem (list, i,                    \
                    Py_BuildValue(_tuple_format,        \
                        minx_##_mt, maxx_##_mt,         \
                        miny_##_mt, maxy_##_mt,         \
                        advance_##_mt));                \
        }                                               \
        else                                            \
        {                                               \
            Py_INCREF (Py_None);                        \
            PyList_SetItem (list, i, Py_None);          \
        }                                               \
    }}

    /* get metrics */
    if (bbmode == FT_BBOX_EXACT || bbmode == FT_BBOX_EXACT_GRIDFIT)
    {
        list = PyList_New(length);
        _GET_METRICS(float, "(fffff)");
    }
    else if (bbmode == FT_BBOX_PIXEL || bbmode == FT_BBOX_PIXEL_GRIDFIT)
    {
        list = PyList_New(length);
        _GET_METRICS(int, "(iiiii)");
    }
    else
    {
        PyErr_SetString(PyExc_ValueError, "Invalid bbox mode specified");
        return NULL;
    }

#undef _GET_METRICS

    return list;
}

static PyObject*
_ftfont_render_raw(PyObject *self, PyObject* args, PyObject *kwds)
{
    /* keyword list */
    static char *kwlist[] = 
    { 
        "text", "ptsize", NULL
    };

    PyFreeTypeFont *font = (PyFreeTypeFont *)self;
    FontRenderMode render;

    /* input arguments */
    PyObject *text = NULL;
    int ptsize = -1;

    /* output arguments */
    PyObject *rbuffer = NULL;
    int width, height;

    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, NULL);

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|i", kwlist, &text, &ptsize))
        return NULL;

    /*
     * Build the render mode with the given size and no
     * rotation/styles/vertical text
     */
    if (PGFT_BuildRenderMode(ft, font, 
                &render, ptsize, FT_STYLE_NORMAL, 0) != 0)
    {
        PyErr_SetString(PyExc_ValueError, PGFT_GetError(ft));
        return NULL;
    }

    rbuffer = PGFT_Render_PixelArray(ft, font, &render, text, &width, &height);

    if (!rbuffer)
    {
        PyErr_SetString(PyExc_RuntimeError, PGFT_GetError(ft));
        return NULL;
    }

    return Py_BuildValue("(iiO)", width, height, rbuffer);
}

static PyObject*
_ftfont_render(PyObject *self, PyObject* args, PyObject *kwds)
{
#ifndef HAVE_PYGAME_SDL_VIDEO

    PyErr_SetString(PyExc_RuntimeError, "SDL support is missing. Cannot render on surfaces");
    return NULL;

#else
    /* keyword list */
    static char *kwlist[] = 
    { 
        "dest", "text", "fgcolor", "bgcolor", 
        "style", "rotation", "ptsize", NULL
    };

    PyFreeTypeFont *font = (PyFreeTypeFont *)self;

    /* input arguments */
    PyObject *text = NULL;
    int ptsize = -1;
    PyObject *target_tuple = NULL;
    PyObject *fg_color_obj = NULL;
    PyObject *bg_color_obj = NULL;
    int rotation = 0;
    int style = FT_STYLE_DEFAULT;

    /* output arguments */
    PyObject *rtuple = NULL;
    int width, height;

    FontColor fg_color, bg_color;
    FontRenderMode render;

    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, NULL);

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOO|Oiii", kwlist,
                &target_tuple, &text, &fg_color_obj, /* required */
                &bg_color_obj, &style, &rotation, &ptsize)) /* optional */
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

    if (PGFT_BuildRenderMode(ft, font, 
                &render, ptsize, style, rotation) != 0)
    {
        PyErr_SetString(PyExc_ValueError, PGFT_GetError(ft));
        return NULL;
    }

    if (target_tuple == Py_None)
    {
        SDL_Surface *r_surface = NULL;

        r_surface = PGFT_Render_NewSurface(ft, font, &render, text,
                &fg_color, bg_color_obj ? &bg_color : NULL, 
                &width, &height);

        if (!r_surface)
        {
            PyErr_SetString(PyExc_RuntimeError, PGFT_GetError(ft));
            return NULL;
        }

        rtuple = Py_BuildValue("(Oii)", 
                PySurface_New(r_surface), width, height);
               
    }
    else
    {
        SDL_Surface *surface = NULL;
        PyObject *surface_obj = NULL;
        int xpos = 0, ypos = 0;

        if (!PyArg_ParseTuple(target_tuple, "Oii", &surface_obj, &xpos, &ypos))
            return NULL;

        if (!PySurface_Check(surface_obj))
        {
            PyErr_SetString(PyExc_TypeError, "Target surface must be a SDL surface");
            return NULL;
        }

        surface = PySurface_AsSurface(surface_obj);

        if (PGFT_Render_ExistingSurface(ft, font, &render, 
                text, surface, xpos, ypos, 
                &fg_color, bg_color_obj ? &bg_color : NULL,
                &width, &height) != 0)
        {
            PyErr_SetString(PyExc_RuntimeError, PGFT_GetError(ft));
            return NULL;
        }

        rtuple = Py_BuildValue("(iiO)", width, height);
    }

    return rtuple;

#endif // HAVE_PYGAME_SDL_VIDEO
}

/****************************************************
 * C API CALLS
 ****************************************************/
PyObject*
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

    font->ptsize = -1;
    font->style = FT_STYLE_NORMAL;
    font->antialias = 1;
    font->vertical = 0;

    if (PGFT_TryLoadFont_Filename(ft, font, filename, face_index) != 0)
    {
        PyErr_SetString(PyExc_RuntimeError, PGFT_GetError(ft));
        return NULL;
    }

    return (PyObject*) font;
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

static PyObject*
_ft_autoinit(PyObject* self)
{
    FT_Error result = 1;

    if (FREETYPE_MOD_STATE(self)->freetype == NULL)
    {
        result = (PGFT_Init(&(FREETYPE_MOD_STATE(self)->freetype), 
                    PGFT_DEFAULT_CACHE_SIZE) == 0);
    }

    return PyInt_FromLong(result);
}

static PyObject *
_ft_quit(PyObject *self)
{
    if (FREETYPE_MOD_STATE(self)->freetype != NULL)
    {
        PGFT_Quit(FREETYPE_MOD_STATE(self)->freetype);
        FREETYPE_MOD_STATE(self)->freetype = NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
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


static PyObject *
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

static PyObject *
_ft_get_version(PyObject *self)
{
    /* Return the linked FreeType2 version */
    return Py_BuildValue("(iii)", FREETYPE_MAJOR, FREETYPE_MINOR, FREETYPE_PATCH);
}

static PyObject *
_ft_was_init(PyObject *self)
{
    return PyBool_FromLong((FREETYPE_MOD_STATE (self)->freetype != NULL));
}

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



/****************************************************
 * FREETYPE MODULE DECLARATION
 ****************************************************/
#if PY3
struct PyModuleDef _freetypemodule = 
{
    PyModuleDef_HEAD_INIT,
    "freetype",
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
    PyObject *module, *apiobj, *base, *base_register_quit, *quit, *rval;
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

    /* import needed modules. Do this first so if there is an error
       the module is not loaded.
    */
    base = PyImport_ImportModule ("base");
    if (!base) {
        MODINIT_ERROR;
    }
    base_register_quit = PyObject_GetAttrString (base, "register_quit");
    Py_DECREF (base);
    if (!base_register_quit) {
        MODINIT_ERROR;
    }

    /* type preparation */
    if (PyType_Ready(&PyFreeTypeFont_Type) < 0) 
    {
        Py_DECREF(base_register_quit);
        MODINIT_ERROR;
    }

    PyFreeTypeFont_Type.tp_new = PyType_GenericNew;

#if PY3
    module = PyModule_Create(&_freetypemodule);
#else
    /* TODO: DOC */
    module = Py_InitModule3("freetype", _ft_methods, DOC_PYGAMEFREETYPE); 
#endif

    if (module == NULL) 
    {
        Py_DECREF(base_register_quit);
        MODINIT_ERROR;
    }

    Py_INCREF((PyObject *)&PyFreeTypeFont_Type);
    if (PyModule_AddObject(module, "Font", (PyObject *)&PyFreeTypeFont_Type) == -1) 
    {
        Py_DECREF(base_register_quit);
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
    c_api[0] = &PyFreeTypeFont_Type;
    c_api[1] = &PyFreeTypeFont_New;

    apiobj = PyCObject_FromVoidPtr(c_api, NULL);
    if (apiobj == NULL) 
    {
        Py_DECREF (base_register_quit);
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

    if (PyModule_AddObject(module, PYGAMEAPI_LOCAL_ENTRY, apiobj) == -1) 
    {
        Py_DECREF(base_register_quit);
        Py_DECREF(apiobj);
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

    quit = PyObject_GetAttrString (module, "quit");
    if (quit == NULL) {  /* assertion */
        Py_DECREF (base_register_quit);
        DECREF_MOD (module);
        MODINIT_ERROR;
    }
    rval = PyObject_CallFunctionObjArgs (base_register_quit, quit, NULL);
    Py_DECREF (base_register_quit);
    Py_DECREF (quit);
    if (rval == NULL) {
        DECREF_MOD (module);
        MODINIT_ERROR;
    }
    Py_DECREF (rval);

    MODINIT_RETURN(module);
}
