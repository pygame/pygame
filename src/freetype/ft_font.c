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

#include "ft_wrap.h"
#include "pgfreetype.h"
#include "ft_mod.h"
#include "freetypebase_doc.h"


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

/*
 * Constructor/init/destructor
 */
static PyObject *_ftfont_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
static int _ftfont_init(PyObject *chunk, PyObject *args, PyObject *kwds);
static void _ftfont_dealloc(PyFreeTypeFont *self);
static PyObject *_ftfont_repr(PyObject *self);

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
 * FREETYPE FONT METHODS TABLE
 */
static PyMethodDef _ftfont_methods[] = 
{
    {
        "get_size", 
        (PyCFunction) _ftfont_getsize,
        METH_VARARGS | METH_KEYWORDS,
        DOC_BASE_FONT_GET_SIZE 
    },
    {
        "get_metrics", 
        (PyCFunction) _ftfont_getmetrics,
        METH_VARARGS | METH_KEYWORDS,
        DOC_BASE_FONT_GET_METRICS 
    },
    { 
        "render", 
        (PyCFunction)_ftfont_render, 
        METH_VARARGS | METH_KEYWORDS,
        DOC_BASE_FONT_RENDER 
    },
    { 
        "render_raw", 
        (PyCFunction)_ftfont_render_raw, 
        METH_VARARGS | METH_KEYWORDS,
        DOC_BASE_FONT_RENDER_RAW
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
        DOC_BASE_FONT_STYLE,    
        NULL 
    },
    { 
        "height",
        _ftfont_getheight,  
        NULL,
        DOC_BASE_FONT_HEIGHT,   
        NULL
    },
    { 
        "name", 
        _ftfont_getname, 
        NULL,
        DOC_BASE_FONT_NAME, 
        NULL 
    },
    {
        "fixed_width",
        _ftfont_getfixedwidth,
        NULL,
        DOC_BASE_FONT_FIXED_WIDTH,
        NULL
    },
    {
        "antialiased",
        _ftfont_getantialias,
        _ftfont_setantialias,
        DOC_BASE_FONT_ANTIALIASED,
        NULL
    },
    {
        "vertical",
        _ftfont_getvertical,
        _ftfont_setvertical,
        DOC_BASE_FONT_VERTICAL,
        NULL
    },
    {
        "italic",
        _ftfont_getstyle_flag,
        _ftfont_setstyle_flag,
        DOC_BASE_FONT_ITALIC, 
        (void *)FT_STYLE_ITALIC
    },
    {
        "bold",
        _ftfont_getstyle_flag,
        _ftfont_setstyle_flag,
        DOC_BASE_FONT_BOLD, 
        (void *)FT_STYLE_BOLD
    },
    {
        "underline",
        _ftfont_getstyle_flag,
        _ftfont_setstyle_flag,
        DOC_BASE_FONT_UNDERLINE, 
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
    DOC_BASE_FONT,              /* docstring */
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
    _ftfont_new,                /* tp_new */
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

static PyObject*
_ftfont_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyFreeTypeFont *font = (PyFreeTypeFont*)type->tp_alloc(type, 0);

    if (!font)
        return NULL;

    memset(&font->id, 0, sizeof(FontId));

    font->pyfont.get_height = _ftfont_getheight;
    font->pyfont.get_name = _ftfont_getname;
    font->pyfont.get_style = _ftfont_getstyle;
    font->pyfont.set_style = _ftfont_setstyle;
    font->pyfont.get_size = _ftfont_getsize;
    font->pyfont.render = _ftfont_render;
    font->pyfont.copy = NULL; /* TODO */

    return (PyObject*)font;
}

static int
_ftfont_init(PyObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = 
    { 
        "font", "ptsize", "style", "face_index", NULL
    };

    PyFreeTypeFont *font = (PyFreeTypeFont *)self;
    
    PyObject *file;
    int face_index = 0;
    int ptsize = -1;
    int font_style = FT_STYLE_NORMAL;

    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, -1);

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|iii", kwlist, 
                &file, &ptsize, &font_style, &face_index))
        return -1;

    if (face_index < 0)
    {
        PyErr_SetString(PyExc_ValueError, "Face index cannot be negative");
        return -1;
    }

    font->ptsize = (FT_Int16)((ptsize <= 0) ? -1 : ptsize);
    font->style = (FT_Byte)font_style;
    font->antialias = 1;
    font->vertical = 0;

    /*
     * TODO: Handle file-like objects
     */

    if (IsTextObj(file))
    {
        PyObject *tmp;
        char *filename;

        if (!UTF8FromObject(file, &filename, &tmp))
        {
            PyErr_SetString(PyExc_ValueError, "Failed to decode file name");
            return -1;
        }

        Py_XDECREF(tmp);

        if (PGFT_TryLoadFont_Filename(ft, font, filename, face_index) != 0)
        {
            PyErr_SetString(PyExc_RuntimeError, PGFT_GetError(ft));
            return -1;
        }
    }
    else
    {
        PyErr_SetString(PyExc_ValueError, 
            "Invalid 'file' parameter (must be a File object or a file name)");
        return -1;
    }

    return 0;
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
_ftfont_getstyle(PyObject *self, void *closure)
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


/** Height attribute */
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

    if (!IsTextObj (text))
    {
        PyErr_SetString (PyExc_TypeError,
            "text must be a string or unicode object");
        return NULL;
    }

    /* Build rendering mode, always anti-aliased by default */
    if (PGFT_BuildRenderMode(ft, font, &render, 
                ptsize, style, rotation) != 0)
    {
        PyErr_SetString(PyExc_ValueError, PGFT_GetError(ft));
        return NULL;
    }
    
    error = PGFT_GetTextSize(ft, font, &render, text, &width, &height);
    
    if (error)
    {
        if (!PyErr_Occurred ())
            PyErr_SetString(PyExc_RuntimeError, PGFT_GetError(ft));
    }
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
        PyErr_SetString(PyExc_PyGameError, PGFT_GetError(ft));
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

    /* parsed vars */
    FontRenderMode render;
    FontColor fg_color;
    FontColor bg_color;

    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, NULL);

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOO|Oiii", kwlist,
                &target_tuple, &text, &fg_color_obj, /* required */
                &bg_color_obj, &style, &rotation, &ptsize)) /* optional */
        return NULL;


    if (PyColor_Check(fg_color_obj))
    {
        PyColor *c = (PyColor *)fg_color_obj;
        fg_color.r = c->r;
        fg_color.g = c->g;
        fg_color.b = c->b;
        fg_color.a = c->a;
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "fgcolor must be a Color");
        return NULL;
    }

    if (bg_color_obj)
    {
        if (bg_color_obj == Py_None)
        {
            bg_color_obj = NULL;
        }
        else if (PyColor_Check(bg_color_obj))
        {
            PyColor *c = (PyColor *)bg_color_obj;
            bg_color.r = c->r;
            bg_color.g = c->g;
            bg_color.b = c->b;
            bg_color.a = c->a;
        }
        else
        {
            PyErr_SetString(PyExc_TypeError, "bgcolor must be a Color");
            return NULL;
        }
    }

    if (PGFT_BuildRenderMode(ft, font, &render, ptsize, style, rotation) != 0)
    {
        PyErr_SetString(PyExc_ValueError, PGFT_GetError(ft));
        return NULL;
    }

    if (target_tuple == Py_None)
    {
        SDL_Surface *r_surface = NULL;

        r_surface = PGFT_Render_NewSurface(ft, font, &render, text,
                &fg_color, bg_color_obj ? &bg_color : NULL, &width, &height);

        if (!r_surface)
        {
            PyErr_SetString(PyExc_RuntimeError, PGFT_GetError(ft));
            return NULL;
        }

        rtuple = Py_BuildValue("(iiO)", width, height,
                PySDLSurface_NewFromSDLSurface(r_surface));
    }
    else
    {
        SDL_Surface *surface = NULL;
        PyObject *surface_obj = NULL;
        int xpos = 0, ypos = 0;

        if (!PyArg_ParseTuple(target_tuple, "Oii", &surface_obj, &xpos, &ypos))
            return NULL;

        if (!PySDLSurface_Check(surface_obj))
        {
            PyErr_SetString(PyExc_TypeError, "Target surface must be a SDL surface");
            return NULL;
        }

        surface = PySDLSurface_AsSDLSurface(surface_obj);
            
        if (PGFT_Render_ExistingSurface(ft, font, &render, 
                text, surface, xpos, ypos, 
                &fg_color, bg_color_obj ? &bg_color : NULL,
                &width, &height) != 0)
        {
            PyErr_SetString(PyExc_PyGameError, PGFT_GetError(ft));
            return NULL;
        }

        rtuple = Py_BuildValue("(ii)", width, height);
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

void
ftfont_export_capi(void **capi)
{
    capi[PYGAME_FREETYPE_FONT_FIRSTSLOT + 0] = &PyFreeTypeFont_Type;
    capi[PYGAME_FREETYPE_FONT_FIRSTSLOT + 1] = &PyFreeTypeFont_New;
}
