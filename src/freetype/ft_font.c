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

#include "ft_mod.h"
#include "ft_wrap.h"
#include "pgfreetype.h"
#include "freetypebase_doc.h"

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
static PyObject* _ftfont_render(PyObject *self, PyObject* args, PyObject *kwds);
static PyObject* _ftfont_getmetrics(PyObject *self, PyObject* args, PyObject *kwds);
/* static PyObject* _ftfont_copy(PyObject *self); */

/*
 * Getters/setters
 */
static PyObject* _ftfont_getstyle(PyObject *self, void *closure);
static int _ftfont_setstyle(PyObject *self, PyObject *value, void *closure);
static PyObject* _ftfont_getheight(PyObject *self, void *closure);
static PyObject* _ftfont_getname(PyObject *self, void *closure);
static PyObject* _ftfont_getfixedwidth(PyObject *self, void *closure);

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
    FreeTypeInstance *ft;

    if ((ft = _get_freetype()) != NULL)
        PGFT_UnloadFont(ft, self);

    free(self->id.open_args.pathname);

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
    PyFreeTypeFont *font = (PyFreeTypeFont *)self;
    
    PyObject *file;
    int face_index = 0;

    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, -1);

    if (!PyArg_ParseTuple(args, "O|i", &file, &face_index))
        return -1;

    if (face_index < 0)
    {
        PyErr_SetString(PyExc_ValueError, "Face index cannot be negative");
        return -1;
    }

    font->id.face_index = face_index;

    /*
     * TODO: Handle file-like objects
     */

    if (IsTextObj(file))
    {
        PyObject *tmp;
        char *filename, *filename_alloc;
        size_t file_len;

        if (!UTF8FromObject(file, &filename, &tmp))
        {
            PyErr_SetString(PyExc_ValueError, "Failed to decode file name");
            return -1;
        }

        file_len = strlen(filename);
        filename_alloc = malloc(file_len + 1);

        strcpy(filename_alloc, filename);
        filename_alloc[file_len] = 0;
        
        font->id.open_args.flags = FT_OPEN_PATHNAME;
        font->id.open_args.pathname = filename_alloc;
        Py_XDECREF(tmp);
    }
    else
    {
        PyErr_SetString(PyExc_ValueError, 
            "Invalid 'file' parameter (must be a File object or a file name)");
        return -1;
    }


    if (PGFT_TryLoadFont(ft, font) != 0)
    {
        /* TODO: Get a proper error string */
        PyErr_SetString(PyExc_RuntimeError, PGFT_GetError(ft));
        return -1;
    }

    return 0;
}

static PyObject*
_ftfont_repr(PyObject *self)
{
    return Text_FromUTF8("font");
/*    return Text_FromFormat("%s (FreeType %s font)", 
        PGFT_Face_GetName((PyFreeTypeFont *)self),
        PGFT_Face_GetFormat((PyFreeTypeFont *)self)); */
}


/****************************************************
 * GETTERS/SETTERS
 ****************************************************/
static PyObject*
_ftfont_getstyle (PyObject *self, void *closure)
{
    /* TODO */
}

static int
_ftfont_setstyle(PyObject *self, PyObject *value, void *closure)
{
    /* TODO */
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
    PyObject *text, *rtuple = NULL;
    FT_Error error;
    int text_size, free_buffer;
    int width, height;
    FT_UInt16 *text_buffer = NULL;

    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, NULL);

    if (!PyArg_ParseTuple(args, "iO", &text_size, &text))
        return NULL;

    text_buffer = PGFT_BuildUnicodeString(text, &free_buffer);

    if (!text_buffer)
    {
        PyErr_SetString(PyExc_ValueError, "Expecting unicode/bytes string");
        return NULL;
    }

    error = PGFT_GetTextSize(ft, (PyFreeTypeFont *)self, 
            text_buffer, text_size, &width, &height);

    if (error)
        PyErr_SetString(PyExc_RuntimeError, PGFT_GetError(ft));
    else
        rtuple = Py_BuildValue ("(ii)", width, height);

    if (free_buffer)
        free(text_buffer);

    return rtuple;
}

static PyObject *
_ftfont_getmetrics(PyObject *self, PyObject* args, PyObject *kwds)
{
    PyObject *text, *list;
    void *buf = NULL;
    int isunicode = 0;
    int text_size, char_id, length, i;
    int minx, miny, maxx, maxy, advance;

    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, NULL);

    if (!PyArg_ParseTuple(args, "iO", &text_size, &text))
        return NULL;

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

    if (length == 0)
        Py_RETURN_NONE;

    list = PyList_New(length);
    if (!list)
        return NULL;

    for (i = 0; i < length; i++)
    {
        if (isunicode)
            char_id = ((Py_UNICODE *)buf)[i];
        else
            char_id = ((char *)buf)[i];

        if (PGFT_GetMetrics(ft, (PyFreeTypeFont *)self, char_id, 
                    text_size, &minx, &maxx, &miny, &maxy, &advance) == 0)
        {
            PyList_SetItem (list, i, Py_BuildValue(
                        "(iiiii)", minx, maxx, miny, maxy, advance));
        }
        else
        {
            Py_INCREF (Py_None);
            PyList_SetItem (list, i, Py_None); 
        }
    }

    return list;
}

static PyObject*
_ftfont_render(PyObject *self, PyObject* args, PyObject *kwds)
{
    /* TODO */
}



/****************************************************
 * C API CALLS
 ****************************************************/
PyObject*
PyFreeTypeFont_New(char *file, int ptsize)
{
    /* TODO */
    return NULL;
}

void
ftfont_export_capi(void **capi)
{
    capi[PYGAME_FREETYPE_FONT_FIRSTSLOT + 0] = &PyFreeTypeFont_Type;
    capi[PYGAME_FREETYPE_FONT_FIRSTSLOT + 1] = &PyFreeTypeFont_New;
}
