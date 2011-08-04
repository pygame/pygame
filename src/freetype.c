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
#define FACE_TYPE_NAME "Face"

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
static PyObject *_ftface_new(PyTypeObject *, PyObject *, PyObject *);
static void _ftface_dealloc(PgFaceObject *);
static PyObject *_ftface_repr(PgFaceObject *);
static int _ftface_init(PgFaceObject *, PyObject *, PyObject *);

/*
 * Main methods
 */
static PyObject *_ftface_getrect(PgFaceObject *, PyObject *, PyObject *);
static PyObject *_ftface_getmetrics(PgFaceObject *, PyObject *, PyObject *);
static PyObject *_ftface_render(PgFaceObject *, PyObject *, PyObject *);
static PyObject *_ftface_render_to(PgFaceObject *, PyObject *, PyObject *);
static PyObject *_ftface_render_raw(PgFaceObject *, PyObject *, PyObject *);
static PyObject *_ftface_getsizedascender(PgFaceObject *, PyObject *);
static PyObject *_ftface_getsizeddescender(PgFaceObject *, PyObject *);
static PyObject *_ftface_getsizedheight(PgFaceObject *, PyObject *);
static PyObject *_ftface_getsizedglyphheight(PgFaceObject *, PyObject *);
static PyObject *_ftface_gettransform(PgFaceObject *);
static PyObject *_ftface_settransform(PgFaceObject *, PyObject *);
static PyObject *_ftface_deletetransform(PgFaceObject *);

/* static PyObject *_ftface_copy(PgFaceObject *); */

/*
 * Getters/setters
 */
static PyObject *_ftface_getstyle(PgFaceObject *, void *);
static int _ftface_setstyle(PgFaceObject *, PyObject *, void *);
static PyObject *_ftface_getname(PgFaceObject *, void *);
static PyObject *_ftface_getpath(PgFaceObject *, void *);
static PyObject *_ftface_getfixedwidth(PgFaceObject *, void *);
static PyObject *_ftface_getstrength(PgFaceObject *, void *);
static int _ftface_setstrength(PgFaceObject *, PyObject *, void *);
static PyObject *_ftface_getunderlineadjustment(PgFaceObject *, void *);
static int _ftface_setunderlineadjustment(PgFaceObject *, PyObject *, void *);

static PyObject *_ftface_getresolution(PgFaceObject *, void *);

static PyObject *_ftface_getfacemetric(PgFaceObject *, void *);

static PyObject *_ftface_getstyle_flag(PgFaceObject *, void *);
static int _ftface_setstyle_flag(PgFaceObject *, PyObject *, void *);

static PyObject *_ftface_getrender_flag(PgFaceObject *, void *);
static int _ftface_setrender_flag(PgFaceObject *, PyObject *, void *);

#if defined(PGFT_DEBUG_CACHE)
static PyObject *_ftface_getdebugcachestats(PgFaceObject *, void *);
#endif

/*
 * Internal helpers
 */
static PyObject *get_metrics(FreeTypeInstance *, FaceRenderMode *,
                             PgFaceObject *, PGFT_String *);
static PyObject *load_font_res(const char *);
static int parse_dest(PyObject *, int *, int *);


/*
 * Auxiliar defines
 */
#define ASSERT_SELF_IS_ALIVE(s)                     \
if (!PgFace_IS_ALIVE(s)) {                  \
    return RAISE(PyExc_RuntimeError,                \
        MODULE_NAME "." FACE_TYPE_NAME              \
        " instance is not initialized");            \
}

#define PGFT_CHECK_BOOL(_pyobj, _var)               \
    if (_pyobj) {                                   \
        if (!PyBool_Check(_pyobj)) {                \
            PyErr_SetString(PyExc_TypeError,        \
                #_var " must be a boolean value");  \
            return 0;                            \
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
    PyObject *load_basicfunc = 0;
    PyObject *pkgdatamodule = 0;
    PyObject *resourcefunc = 0;
    PyObject *result = 0;
    PyObject *tmp;

    pkgdatamodule = PyImport_ImportModule(PKGDATA_MODULE_NAME);
    if (!pkgdatamodule) {
        goto font_resource_end;
    }

    resourcefunc = PyObject_GetAttrString(pkgdatamodule, RESOURCE_FUNC_NAME);
    if (!resourcefunc) {
        goto font_resource_end;
    }

    result = PyObject_CallFunction(resourcefunc, "s", filename);
    if (!result) {
        goto font_resource_end;
    }

#if PY3
    tmp = PyObject_GetAttrString(result, "name");
    if (tmp) {
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
parse_dest(PyObject *dest, int *x, int *y)
{
    PyObject *oi;
    PyObject *oj;
    int i, j;

    if (!PySequence_Check(dest) ||  /* conditional and */
        !PySequence_Size(dest) > 1) {
        PyErr_Format(PyExc_TypeError,
                     "Expected length 2 sequence for dest argument:"
                     " got type %.1024s",
                     Py_TYPE(dest)->tp_name);
        return -1;
    }
    oi = PySequence_GetItem(dest, 0);
    if (!oi) {
        return -1;
    }
    oj = PySequence_GetItem(dest, 1);
    if (!oj) {
        Py_DECREF(oi);
        return -1;
    }
    if (!PyNumber_Check(oi) || !PyNumber_Check(oj)) {
        PyErr_Format(PyExc_TypeError,
                     "for dest expected a pair of numbers"
                     "for elements 1 and 2: got types %.1024s and %1024s",
                     Py_TYPE(oi)->tp_name, Py_TYPE(oj)->tp_name);
        Py_DECREF(oi);
        Py_DECREF(oj);
        return -1;
    }
    i = PyInt_AsLong(oi);
    Py_DECREF(oi);
    if (i == -1 && PyErr_Occurred()) {
        Py_DECREF(oj);
        return -1;
    }
    j = PyInt_AsLong(oj);
    Py_DECREF(oj);
    if (j == -1 && PyErr_Occurred()) {
        return -1;
    }
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

    { 0, 0, 0, 0 }
};


/*
 * FREETYPE FACE METHODS TABLE
 */
static PyMethodDef _ftface_methods[] = {
    {
        "get_sized_height",
        (PyCFunction) _ftface_getsizedheight,
        METH_VARARGS,
        DOC_FACEGETSIZEDHEIGHT
    },
    {
        "get_sized_ascender",
        (PyCFunction) _ftface_getsizedascender,
        METH_VARARGS,
        DOC_FACEGETSIZEDASCENDER
    },
    {
        "get_sized_descender",
        (PyCFunction) _ftface_getsizeddescender,
        METH_VARARGS,
        DOC_FACEGETSIZEDDESCENDER
    },
    {
        "get_sized_glyph_height",
        (PyCFunction) _ftface_getsizedglyphheight,
        METH_VARARGS,
        DOC_FACEGETSIZEDGLYPHHEIGHT
    },
    {
        "get_rect",
        (PyCFunction) _ftface_getrect,
        METH_VARARGS | METH_KEYWORDS,
        DOC_FACEGETRECT
    },
    {
        "get_metrics",
        (PyCFunction) _ftface_getmetrics,
        METH_VARARGS | METH_KEYWORDS,
        DOC_FACEGETMETRICS
    },
    {
        "render",
        (PyCFunction)_ftface_render,
        METH_VARARGS | METH_KEYWORDS,
        DOC_FACERENDER
    },
    {
        "render_to",
        (PyCFunction)_ftface_render_to,
        METH_VARARGS | METH_KEYWORDS,
        DOC_FACERENDERTO
    },
    {
        "render_raw",
        (PyCFunction)_ftface_render_raw,
        METH_VARARGS | METH_KEYWORDS,
        DOC_FACERENDERRAW
    },
    {
        "get_transform",
        (PyCFunction)_ftface_gettransform,
        METH_NOARGS,
        DOC_FACEGETTRANSFORM
    },
    {
        "set_transform",
        (PyCFunction)_ftface_settransform,
        METH_VARARGS,
        DOC_FACESETTRANSFORM
    },
    {
        "delete_transform",
        (PyCFunction)_ftface_deletetransform,
        METH_NOARGS,
        DOC_FACEDELETETRANSFORM
    },

    { 0, 0, 0, 0 }
};

/*
 * FREETYPE FACE GETTERS/SETTERS TABLE
 */
static PyGetSetDef _ftface_getsets[] = {
    {
        "style",
        (getter)_ftface_getstyle,
        (setter)_ftface_setstyle,
        DOC_FACESTYLE,
        0
    },
    {
        "height",
        (getter)_ftface_getfacemetric,
        0,
        DOC_FACEHEIGHT,
        (void *)_PGFT_Face_GetHeight
    },
    {
        "ascender",
        (getter)_ftface_getfacemetric,
        0,
        DOC_FACEASCENDER,
        (void *)_PGFT_Face_GetAscender
    },
    {
        "descender",
        (getter)_ftface_getfacemetric,
        0,
        DOC_FACEASCENDER,
        (void *)_PGFT_Face_GetDescender
    },
    {
        "name",
        (getter)_ftface_getname,
        0,
        DOC_FACENAME,
        0
    },
    {
        "path",
        (getter)_ftface_getpath,
        0,
        DOC_FACEPATH,
        0
    },
    {
        "fixed_width",
        (getter)_ftface_getfixedwidth,
        0,
        DOC_FACEFIXEDWIDTH,
        0
    },
    {
        "antialiased",
        (getter)_ftface_getrender_flag,
        (setter)_ftface_setrender_flag,
        DOC_FACEANTIALIASED,
        (void *)FT_RFLAG_ANTIALIAS
    },
    {
        "kerning",
        (getter)_ftface_getrender_flag,
        (setter)_ftface_setrender_flag,
        DOC_FACEKERNING,
        (void *)FT_RFLAG_KERNING
    },
    {
        "vertical",
        (getter)_ftface_getrender_flag,
        (setter)_ftface_setrender_flag,
        DOC_FACEVERTICAL,
        (void *)FT_RFLAG_VERTICAL
    },
    {
        "pad",
        (getter)_ftface_getrender_flag,
        (setter)_ftface_setrender_flag,
        DOC_FACEPAD,
        (void *)FT_RFLAG_PAD
    },
    {
        "oblique",
        (getter)_ftface_getstyle_flag,
        (setter)_ftface_setstyle_flag,
        DOC_FACEOBLIQUE,
        (void *)FT_STYLE_OBLIQUE
    },
    {
        "strong",
        (getter)_ftface_getstyle_flag,
        (setter)_ftface_setstyle_flag,
        DOC_FACESTRONG,
        (void *)FT_STYLE_STRONG
    },
    {
        "underline",
        (getter)_ftface_getstyle_flag,
        (setter)_ftface_setstyle_flag,
        DOC_FACEUNDERLINE,
        (void *)FT_STYLE_UNDERLINE
    },
    {
        "wide",
        (getter)_ftface_getstyle_flag,
        (setter)_ftface_setstyle_flag,
        DOC_FACEWIDE,
        (void *)FT_STYLE_WIDE
    },
    {
        "strength",
        (getter)_ftface_getstrength,
        (setter)_ftface_setstrength,
        DOC_FACESTRENGTH,
        0
    }, 
    {
        "underline_adjustment",
        (getter)_ftface_getunderlineadjustment,
        (setter)_ftface_setunderlineadjustment,
        DOC_FACEUNDERLINEADJUSTMENT,
        0
    },
    {
        "ucs4",
        (getter)_ftface_getrender_flag,
        (setter)_ftface_setrender_flag,
        DOC_FACEUCS4,
        (void *)FT_RFLAG_UCS4
    },
    {
        "resolution",
        (getter)_ftface_getresolution,
        0,
        DOC_FACERESOLUTION,
        0
    },
    {
        "origin",
        (getter)_ftface_getrender_flag,
        (setter)_ftface_setrender_flag,
        DOC_FACEORIGIN,
        (void *)FT_RFLAG_ORIGIN
    },
#if defined(PGFT_DEBUG_CACHE)
    {
        "_debug_cache_stats",
        (getter)_ftface_getdebugcachestats,
        0,
        "_debug cache fields as a tuple",
        0
    },
#endif

    { 0, 0, 0, 0, 0 }
};

/*
 * FREETYPE FACE BASE TYPE TABLE
 */
#define FULL_TYPE_NAME MODULE_NAME FACE_TYPE_NAME

PyTypeObject PgFace_Type = {
    TYPE_HEAD(0,0)
    FULL_TYPE_NAME,             /* tp_name */
    sizeof (PgFaceObject),      /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor)_ftface_dealloc,/* tp_dealloc */
    0,                          /* tp_print */
    0,                          /* tp_getattr */
    0,                          /* tp_setattr */
    0,                          /* tp_compare */
    (reprfunc)_ftface_repr,     /* tp_repr */
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
    DOC_PYGAMEFREETYPEFACE,     /* docstring */
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    _ftface_methods,            /* tp_methods */
    0,                          /* tp_members */
    _ftface_getsets,            /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc) _ftface_init,    /* tp_init */
    0,                          /* tp_alloc */
    (newfunc) _ftface_new,      /* tp_new */
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
_ftface_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds)
{
    PgFaceObject *obj = (PgFaceObject *)(subtype->tp_alloc(subtype, 0));

    if (obj) {
        obj->id.open_args.flags = 0;
        obj->id.open_args.pathname = 0;
        obj->path = 0;
        obj->resolution = 0;
        obj->_internals = 0;
        obj->ptsize = -1;
        obj->style = FT_STYLE_NORMAL;
        obj->render_flags = FT_RFLAG_DEFAULTS;
        obj->strength = PGFT_DBL_DEFAULT_STRENGTH;
        obj->underline_adjustment = 1.0;
        obj->transform.xx = FX16_ONE;
        obj->transform.xy = 0;
        obj->transform.yx = 0;
        obj->transform.yy = FX16_ONE;
    }
    return (PyObject *)obj;
}

static void
_ftface_dealloc(PgFaceObject *self)
{
    /* Always try to unload the font even if we cannot grab
     * a freetype instance. */
    _PGFT_UnloadFace(FREETYPE_STATE->freetype, self);

    Py_XDECREF(self->path);
    ((PyObject *)self)->ob_type->tp_free((PyObject *)self);
}

static int
_ftface_init(PgFaceObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] =  {
        "font", "ptsize", "style", "face_index", "vertical",
        "ucs4", "resolution", "origin", 0
    };

    PyObject *file, *original_file;
    int face_index = 0;
    int ptsize;
    int style;
    int ucs4;
    int vertical;
    unsigned resolution = 0;
    int origin;

    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, -1);

    ptsize = self->ptsize;
    style = self->style;
    ucs4 = self->render_flags & FT_RFLAG_UCS4 ? 1 : 0;
    vertical = self->render_flags & FT_RFLAG_VERTICAL ? 1 : 0;
    origin = self->render_flags & FT_RFLAG_ORIGIN ? 1 : 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|iiIiiIi", kwlist,
                                     &file, &ptsize, &style, &face_index,
                                     &vertical, &ucs4, &resolution, &origin)) {
        return -1;
    }

    original_file = file;

    _PGFT_UnloadFace(ft, self);
    Py_XDECREF(self->path);
    self->path = 0;

    if (_PGFT_CheckStyle(style)) {
        PyErr_Format(PyExc_ValueError,
                     "Invalid style value %x", (int)style);
        return -1;
    }
    self->ptsize = (FT_Int16)((ptsize <= 0) ? -1 : ptsize);
    self->style = (FT_Int16)style;
    if (ucs4) {
        self->render_flags |= FT_RFLAG_UCS4;
    }
    else {
        self->render_flags &= ~FT_RFLAG_UCS4;
    }
    if (vertical) {
        self->render_flags |= FT_RFLAG_VERTICAL;
    }
    else {
        self->render_flags &= ~FT_RFLAG_VERTICAL;
    }
    if (origin) {
        self->render_flags |= FT_RFLAG_ORIGIN;
    }
    else {
        self->render_flags &= ~FT_RFLAG_ORIGIN;
    }
    if (resolution) {
        self->resolution = (FT_UInt)resolution;
    }
    else {
        self->resolution = FREETYPE_STATE->resolution;
    }
    if (file == Py_None) {
        file = load_font_res(DEFAULT_FONT_NAME);

        if (!file) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to find default font");
            goto end;
        }
    }

    file = RWopsEncodeFilePath(file, 0);
    if (!file) {
        return -1;
    }
    if (Bytes_Check(file)) {
        if (_PGFT_TryLoadFont_Filename(ft, self, Bytes_AS_STRING(file),
                                       face_index)) {
            goto end;
        }

        if (PyUnicode_Check(original_file)) {
            /* Make sure to save a pure Unicode object to prevent possible
             * cycles from a derived class. This means no tp_traverse or
             * tp_clear for the PyFreetypeFont type.
             */
            self->path = Object_Unicode(original_file);
        }
        else {
            self->path = PyUnicode_FromEncodedObject(file, "raw_unicode_escape",
                                                     "replace");
        }
    }
    else {
        SDL_RWops *source = RWopsFromFileObjectThreaded(original_file);
        PyObject *str = 0;
        PyObject *path = 0;

        if (!source) {
            goto end;
        }

        if (_PGFT_TryLoadFont_RWops(ft, self, source, face_index)) {
            goto end;
        }

        path = PyObject_GetAttrString(original_file, "name");
        if (!path) {
            PyErr_Clear();
            str = Bytes_FromFormat("<%s instance at %p>",
                                   Py_TYPE(file)->tp_name, (void *)file);
            if (str) {
                self->path = PyUnicode_FromEncodedObject(str,
                                                         "ascii", "strict");
                Py_DECREF(str);
            }
        }
        else if (PyUnicode_Check(path)) {
            /* Make sure to save a pure Unicode object to prevent possible
             * cycles from a derived class. This means no tp_traverse or
             * tp_clear for the PyFreetypeFont type.
             */
            self->path = Object_Unicode(path);
        }
        else if (Bytes_Check(path)) {
            self->path = PyUnicode_FromEncodedObject(path,
                                               "unicode_escape", "replace");
        }
        else {
            self->path = Object_Unicode(path);
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
_ftface_repr(PgFaceObject *self)
{
    if (PgFace_IS_ALIVE(self)) {
#if PY3
        return PyUnicode_FromFormat("Face('%.1024u')", self->path);
#else
        PyObject *str = PyUnicode_AsEncodedString(self->path,
                                                  "raw_unicode_escape",
                                                  "replace");
        PyObject *rval = 0;

        if (str) {
            rval = PyString_FromFormat("Face('%.1024s')",
                                       PyString_AS_STRING(str));
            Py_DECREF(str);
        }
        return rval;
#endif
    }
    return Text_FromFormat("<uninitialized Face object at %p>", (void *)self);
}


/****************************************************
 * GETTERS/SETTERS
 ****************************************************/

/** Generic style attributes */
static PyObject *
_ftface_getstyle_flag(PgFaceObject *self, void *closure)
{
    const int style_flag = (int)closure;

    return PyBool_FromLong(self->style & style_flag);
}

static int
_ftface_setstyle_flag(PgFaceObject *self, PyObject *value, void *closure)
{
    const int style_flag = (int)closure;

    if (!PyBool_Check(value)) {
        PyErr_SetString(PyExc_TypeError,
                "The style value must be a boolean");
        return -1;
    }

    if (PyObject_IsTrue(value)) {
        self->style |= (FT_UInt16)style_flag;
    }
    else {
        self->style &= (FT_UInt16)(~style_flag);
    }

    return 0;
}


/** Style attribute */
static PyObject *
_ftface_getstyle (PgFaceObject *self, void *closure)
{
    return PyInt_FromLong(self->style);
}

static int
_ftface_setstyle(PgFaceObject *self, PyObject *value, void *closure)
{
    FT_UInt32 style;

    if (!PyInt_Check(value)) {
        PyErr_SetString(PyExc_TypeError,
                "The style value must be an integer"
                " from the FT constants module");
        return -1;
    }

    style = (FT_UInt32)PyInt_AsLong(value);

    if (_PGFT_CheckStyle(style)) {
        PyErr_Format(PyExc_ValueError,
                     "Invalid style value %x", (int)style);
        return -1;
    }

    self->style = (FT_UInt16)style;
    return 0;
}

static PyObject *
_ftface_getstrength(PgFaceObject *self, void *closure)
{
    return PyFloat_FromDouble(self->strength);
}

static int
_ftface_setstrength(PgFaceObject *self, PyObject *value, void *closure)
{
    PyObject *strengthobj = PyNumber_Float(value);
    double strength;

    if (!strengthobj) {
        return -1;
    }
    strength = PyFloat_AS_DOUBLE(strengthobj);
    Py_DECREF(strengthobj);
    if (strength < 0.0 || strength > 1.0) {
        char msg[80];

        sprintf(msg, "strength value %.4e is outside range [0, 1]", strength);
        PyErr_SetString(PyExc_ValueError, msg);
        return -1;
    }
    self->strength = strength;
    return 0;
}

static PyObject *
_ftface_getunderlineadjustment(PgFaceObject *self, void *closure)
{
    return PyFloat_FromDouble(self->underline_adjustment);
}

static int
_ftface_setunderlineadjustment(PgFaceObject *self, PyObject *value,
                               void *closure)
{
    PyObject *adjustmentobj = PyNumber_Float(value);
    double adjustment;

    if (!adjustmentobj) {
        return -1;
    }
    adjustment = PyFloat_AS_DOUBLE(adjustmentobj);
    Py_DECREF(adjustmentobj);
    if (adjustment < -2.0 || adjustment > 2.0) {
        char msg[100];

        sprintf(msg,
                "underline adjustment value %.4e is outside range [-2.0, 2.0]",
                adjustment);
        PyErr_SetString(PyExc_ValueError, msg);
        return -1;
    }
    self->underline_adjustment = adjustment;
    return 0;
}


/** general face attributes */

static PyObject *
_ftface_getfacemetric(PgFaceObject *self, void *closure)
{
    typedef long (*getter)(FreeTypeInstance *, PgFaceObject *);
    FreeTypeInstance *ft;
    long height;
    ASSERT_GRAB_FREETYPE(ft, 0);

    ASSERT_SELF_IS_ALIVE(self);
    height = ((getter)closure)(ft, self);
    if (!height && PyErr_Occurred()) {
        return 0;
    }
    return PyInt_FromLong(height);
}

static PyObject *
_ftface_getname(PgFaceObject *self, void *closure)
{
    FreeTypeInstance *ft;
    const char *name;
    ASSERT_GRAB_FREETYPE(ft, 0);

    if (PgFace_IS_ALIVE(self)) {
        name = _PGFT_Face_GetName(ft, self);
        return name ? Text_FromUTF8(name) : 0;
    }
    return PyObject_Repr((PyObject *)self);
}

static PyObject *
_ftface_getpath(PgFaceObject *self, void *closure)
{
    PyObject *path = ((PgFaceObject *)self)->path;

    if (!path) {
        PyErr_SetString(PyExc_AttributeError, "path unavailable");
        return 0;
    }
    Py_INCREF(path);
    return path;
}

static PyObject *
_ftface_getfixedwidth(PgFaceObject *self, void *closure)
{
    FreeTypeInstance *ft;
    long fixed_width;
    ASSERT_GRAB_FREETYPE(ft, 0);

    ASSERT_SELF_IS_ALIVE(self);
    fixed_width = _PGFT_Face_IsFixedWidth(ft, (PgFaceObject *)self);
    return fixed_width >= 0 ? PyBool_FromLong(fixed_width) : 0;
}


/** Generic render flag attributes */
static PyObject *
_ftface_getrender_flag(PgFaceObject *self, void *closure)
{
    const int render_flag = (int)closure;

    return PyBool_FromLong(self->render_flags & render_flag);
}

static int
_ftface_setrender_flag(PgFaceObject *self, PyObject *value, void *closure)
{
    const int render_flag = (int)closure;

    if (!PyBool_Check(value)) {
        PyErr_SetString(PyExc_TypeError,
                "The style value must be a boolean");
        return -1;
    }

    if (PyObject_IsTrue(value)) {
        self->render_flags |= (FT_UInt16)render_flag;
    }
    else {
        self->render_flags &= (FT_UInt16)(~render_flag);
    }

    return 0;
}


/** resolution pixel size attribute */
static PyObject *
_ftface_getresolution(PgFaceObject *self, void *closure)
{
    return PyLong_FromUnsignedLong((unsigned long)self->resolution);
}


/** testing and debugging */
#if defined(PGFT_DEBUG_CACHE)
static PyObject *
_ftface_getdebugcachestats(PgFaceObject *self, void *closure)
{
    /* Yes, this kind of breaches the boundary between the top level
     * freetype.c and the lower level ft_text.c. But it is built
     * conditionally, and it keeps some of the Python api out
     * of ft_text.c and ft_cache.c (hoping to remove the Python
     * api completely from ft_text.c and support C modules at some point.)
     */
    const FaceCache *cache = &PGFT_FACE_CACHE(self);

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
_ftface_getrect(PgFaceObject *self, PyObject *args, PyObject *kwds)
{
/* MODIFIED
 */
    /* keyword list */
    static char *kwlist[] =  {
        "text", "style", "rotation", "ptsize", 0
    };

    PyObject *textobj;
    PGFT_String *text;
    PyObject *rectobj = 0;
    FT_Error error;
    int ptsize = -1;
    SDL_Rect r;

    FaceRenderMode render;
    int rotation = 0;
    int style = 0;

    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, 0);

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|iii", kwlist,
                                     &textobj, &style, &rotation, &ptsize)) {
        return 0;
    }

    /* Encode text */
    text = _PGFT_EncodePyString(textobj, self->render_flags & FT_RFLAG_UCS4);
    if (!text) {
        return 0;
    }

    ASSERT_SELF_IS_ALIVE(self);

    /* Build rendering mode, always anti-aliased by default */
    if (_PGFT_BuildRenderMode(ft, self, &render,
                             ptsize, style, rotation)) {
        return 0;
    }

    error = _PGFT_GetTextRect(ft, self, &render, text, &r);
    _PGFT_FreeString(text);

    if (!error) {
        rectobj = PyRect_New(&r);
    }

    return rectobj;
}

static PyObject *
get_metrics(FreeTypeInstance *ft, FaceRenderMode *render,
          PgFaceObject *face, PGFT_String *text)
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
    if (!list) {
        return 0;
    }
    for (i = 0; i < length; ++i) {
        if (_PGFT_GetMetrics(ft, face, data[i], render,
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
                return 0;
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
_ftface_getmetrics(PgFaceObject *self, PyObject *args, PyObject *kwds)
{
    /* keyword list */
    static char *kwlist[] =  {
        "text", "ptsize", 0
    };

    FaceRenderMode render;
    PyObject *list = 0;

    /* arguments */
    PyObject *textobj;
    PGFT_String *text;
    int ptsize = -1;

    /* grab freetype */
    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, 0);

    /* parse args */
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|i", kwlist,
                                     &textobj, &ptsize)) {
        return 0;
    }

    /* Encode text */
    text = _PGFT_EncodePyString(textobj, self->render_flags & FT_RFLAG_UCS4);
    if (!text) {
        return 0;
    }

    ASSERT_SELF_IS_ALIVE(self);

    /*
     * Build the render mode with the given size and no
     * rotation/styles/vertical text
     */
    if (_PGFT_BuildRenderMode(ft, self, &render, ptsize, FT_STYLE_NORMAL, 0)) {
        _PGFT_FreeString(text);
        return 0;
    }

    /* get metrics */
    list = get_metrics(ft, &render, self, text);

    _PGFT_FreeString(text);
    return list;
}

static PyObject *
_ftface_getsizedascender(PgFaceObject *self, PyObject *args)
{
    int pt_size = -1;
    long value;

    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, 0);

    if (!PyArg_ParseTuple(args, "|i", &pt_size)) {
        return 0;
    }

    if (pt_size == -1) {
        if (self->ptsize == -1) {
            RAISE(PyExc_ValueError,
                  "No font point size specified"
                  " and no default font size in typeface");
            return 0;
        }

        pt_size = self->ptsize;
    }

    if (pt_size <= 0) {
        RAISE(PyExc_ValueError, "Invalid point size for font.");
        return 0;
    }
    value = (long)_PGFT_Face_GetAscenderSized(ft, self, (FT_UInt16)pt_size);
    if (!value && PyErr_Occurred()) {
        return 0;
    }
    return PyInt_FromLong(value);
}

static PyObject *
_ftface_getsizeddescender(PgFaceObject *self, PyObject *args)
{
    int pt_size = -1;
    long value;

    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, 0);

    if (!PyArg_ParseTuple(args, "|i", &pt_size)) {
        return 0;
    }

    if (pt_size == -1) {
        if (self->ptsize == -1) {
            RAISE(PyExc_ValueError,
                  "No font point size specified"
                  " and no default font size in typeface");
            return 0;
        }

        pt_size = self->ptsize;
    }

    if (pt_size <= 0) {
        RAISE(PyExc_ValueError, "Invalid point size for font.");
        return 0;
    }
    value = (long)_PGFT_Face_GetDescenderSized(ft, self, (FT_UInt16)pt_size);
    if (!value && PyErr_Occurred()) {
        return 0;
    }
    return PyInt_FromLong(value);
}

static PyObject *
_ftface_getsizedheight(PgFaceObject *self, PyObject *args)
{
    int pt_size = -1;
    long value;

    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, 0);

    if (!PyArg_ParseTuple(args, "|i", &pt_size)) {
        return 0;
    }

    if (pt_size == -1) {
        if (self->ptsize == -1) {
            RAISE(PyExc_ValueError,
                  "No font point size specified"
                  " and no default font size in typeface");
            return 0;
        }

        pt_size = self->ptsize;
    }

    if (pt_size <= 0) {
        RAISE(PyExc_ValueError, "Invalid point size for font.");
        return 0;
    }
    value = _PGFT_Face_GetHeightSized(ft, self, (FT_UInt16)pt_size);
    if (!value && PyErr_Occurred()) {
        return 0;
    }
    return PyInt_FromLong(value);
}

static PyObject *
_ftface_getsizedglyphheight(PgFaceObject *self, PyObject *args)
{
    int pt_size = -1;
    long value;

    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, 0);

    if (!PyArg_ParseTuple(args, "|i", &pt_size)) {
        return 0;
    }

    if (pt_size == -1) {
        if (self->ptsize == -1) {
            RAISE(PyExc_ValueError,
                  "No font point size specified"
                  " and no default font size in typeface");
            return 0;
        }

        pt_size = self->ptsize;
    }

    if (pt_size <= 0) {
        RAISE(PyExc_ValueError, "Invalid point size for font.");
        return 0;
    }
    value = (long)_PGFT_Face_GetGlyphHeightSized(ft, self, (FT_UInt16)pt_size);
    if (!value && PyErr_Occurred()) {
        return 0;
    }
    return PyInt_FromLong(value);
}

static PyObject *
_ftface_render_raw(PgFaceObject *self, PyObject *args, PyObject *kwds)
{
    /* keyword list */
    static char *kwlist[] =  {
        "text", "style", "rotation", "ptsize", 0
    };

    FaceRenderMode mode;

    /* input arguments */
    PyObject *textobj;
    PGFT_String *text;
    int rotation = 0;
    int ptsize = -1;
    int style = FT_STYLE_DEFAULT;

    /* output arguments */
    PyObject *rbuffer = 0;
    PyObject *rtuple;
    int width, height;

    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, 0);

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|iii", kwlist,
                                     &textobj, &style, &rotation, &ptsize)) {
        return 0;
    }

    /* Encode text */
    text = _PGFT_EncodePyString(textobj, self->render_flags & FT_RFLAG_UCS4);
    if (!text) {
        return 0;
    }

    ASSERT_SELF_IS_ALIVE(self);

    /*
     * Build the render mode with the given size and no
     * rotation/styles/vertical text
     */
    if (_PGFT_BuildRenderMode(ft, self, &mode, ptsize, style, rotation)) {
        _PGFT_FreeString(text);
        return 0;
    }

    rbuffer = _PGFT_Render_PixelArray(ft, self, &mode, text, &width, &height);
    _PGFT_FreeString(text);

    if (!rbuffer) {
        return 0;
    }
    rtuple = Py_BuildValue("O(ii)", rbuffer, width, height);
    Py_DECREF(rbuffer);
    return rtuple;
}

static PyObject *
_ftface_render(PgFaceObject *self, PyObject *args, PyObject *kwds)
{
#ifndef HAVE_PYGAME_SDL_VIDEO

    PyErr_SetString(PyExc_RuntimeError,
		    "SDL support is missing. Cannot render on surfaces");
    return 0;

#else
    /* keyword list */
    static char *kwlist[] =  {
        "text", "fgcolor", "bgcolor", "style", "rotation", "ptsize", 0
    };

    /* input arguments */
    PyObject *textobj = 0;
    PGFT_String *text;
    int ptsize = -1;
    PyObject *fg_color_obj = 0;
    PyObject *bg_color_obj = 0;
    int rotation = 0;
    int style = FT_STYLE_DEFAULT;

    /* output arguments */
    SDL_Surface *surface;
    PyObject *surface_obj = 0;
    PyObject *rtuple = 0;
    SDL_Rect r;
    PyObject *rect_obj;

    FaceColor fg_color;
    FaceColor bg_color;
    FaceRenderMode render;

    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, 0);

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|Oiii", kwlist,
                                     /* required */
                                     &textobj, &fg_color_obj,
                                     /* optional */
                                     &bg_color_obj, &style,
                                     &rotation, &ptsize)) {
        return 0;
    }

    if (!RGBAFromColorObj(fg_color_obj, (Uint8 *)&fg_color)) {
        PyErr_SetString(PyExc_TypeError, "fgcolor must be a Color");
        return 0;
    }
    if (bg_color_obj) {
        if (bg_color_obj == Py_None) {
            bg_color_obj = 0;
        }
        else if (!RGBAFromColorObj(bg_color_obj, (Uint8 *)&bg_color)) {
            PyErr_SetString(PyExc_TypeError, "bgcolor must be a Color");
            return 0;
        }
    }

    /* Encode text */
    text = _PGFT_EncodePyString(textobj, self->render_flags & FT_RFLAG_UCS4);
    if (!text) {
        return 0;
    }

    ASSERT_SELF_IS_ALIVE(self);

    if (_PGFT_BuildRenderMode(ft, self, &render, ptsize, style, rotation)) {
        _PGFT_FreeString(text);
        return 0;
    }

    surface = _PGFT_Render_NewSurface(ft, self, &render, text, &fg_color,
                                      bg_color_obj ? &bg_color : 0, &r);
    _PGFT_FreeString(text);
    if (!surface) {
        return 0;
    }
    surface_obj = PySurface_New(surface);
    if (!surface_obj) {
        SDL_FreeSurface(surface);
        return 0;
    }

    rect_obj = PyRect_New(&r);
    if (rect_obj) {
        rtuple = PyTuple_Pack(2, surface_obj, rect_obj);
        Py_DECREF(rect_obj);
    }
    Py_DECREF(surface_obj);

    return rtuple;

#endif // HAVE_PYGAME_SDL_VIDEO
}

static PyObject *
_ftface_render_to(PgFaceObject *self, PyObject *args, PyObject *kwds)
{
#ifndef HAVE_PYGAME_SDL_VIDEO

    PyErr_SetString(PyExc_RuntimeError,
		    "SDL support is missing. Cannot render on surfaces");
    return 0;

#else
    /* keyword list */
    static char *kwlist[] =  {
        "surf", "dest", "text", "fgcolor", "bgcolor",
        "style", "rotation", "ptsize", 0
    };

    /* input arguments */
    PyObject *surface_obj = 0;
    PyObject *textobj = 0;
    PGFT_String *text;
    int ptsize = -1;
    PyObject *dest = 0;
    int xpos = 0;
    int ypos = 0;
    PyObject *fg_color_obj = 0;
    PyObject *bg_color_obj = 0;
    int rotation = 0;
    int style = FT_STYLE_DEFAULT;
    SDL_Surface *surface = 0;

    /* output arguments */
    SDL_Rect r;
    int rcode;

    FaceColor fg_color;
    FaceColor bg_color;
    FaceRenderMode render;

    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, 0);

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!OOO|Oiii", kwlist,
                                     /* required */
                                     &PySurface_Type, &surface_obj, &dest,
                                     &textobj, &fg_color_obj,
                                     /* optional */
                                     &bg_color_obj, &style,
                                     &rotation, &ptsize)) {
        return 0;
    }

    if (parse_dest(dest, &xpos, &ypos)) {
        return 0;
    }
    if (!RGBAFromColorObj(fg_color_obj, (Uint8 *)&fg_color)) {
        PyErr_SetString(PyExc_TypeError, "fgcolor must be a Color");
        return 0;
    }
    if (bg_color_obj) {
        if (bg_color_obj == Py_None) {
            bg_color_obj = 0;
        }
        else if (!RGBAFromColorObj(bg_color_obj, (Uint8 *)&bg_color)) {
            PyErr_SetString(PyExc_TypeError, "bgcolor must be a Color");
            return 0;
        }
    }

    ASSERT_SELF_IS_ALIVE(self);

    /* Encode text */
    text = _PGFT_EncodePyString(textobj, self->render_flags & FT_RFLAG_UCS4);
    if (!text) {
        return 0;
    }

    if (_PGFT_BuildRenderMode(ft, self, &render, ptsize, style, rotation)) {
        _PGFT_FreeString(text);
        return 0;
    }

    surface = PySurface_AsSurface(surface_obj);
    rcode = _PGFT_Render_ExistingSurface(ft, self, &render, text, surface,
                                         xpos, ypos, &fg_color,
                                         bg_color_obj ? &bg_color : 0, &r);
    _PGFT_FreeString(text);
    if (rcode) {
        return 0;
    }

    return PyRect_New(&r);

#endif // HAVE_PYGAME_SDL_VIDEO
}

static PyObject *
_ftface_gettransform(PgFaceObject *self)
{
    if (!(self->render_flags & FT_RFLAG_TRANSFORM)) {
        Py_RETURN_NONE;
    }
    return Py_BuildValue("dddd",
                         FX16_TO_DBL(self->transform.xx),
                         FX16_TO_DBL(self->transform.xy),
                         FX16_TO_DBL(self->transform.yx),
                         FX16_TO_DBL(self->transform.yy));
}

static PyObject *
_ftface_settransform(PgFaceObject *self, PyObject *args)
{
    double xx;
    double xy;
    double yx;
    double yy;

    if (!PyArg_ParseTuple(args, "dddd", &xx, &xy, &yx, &yy)) {
        return 0;
    }
    /* Sanity check: only accept numbers between -2.0 and 2.0 inclusive */
    if (xx < -2.0 || xx > 2.0 || xy < -2.0 || xy > 2.0 ||
        yx < -2.0 || yx > 2.0 || yy < -2.0 || yy > 2.0) {
        PyErr_SetString(PyExc_ValueError,
                        "received a value outside range [-2.0,2.0]");
        return 0;
    }
    self->transform.xx = FX16_TO_DBL(xx);
    self->transform.xy = FX16_TO_DBL(xy);
    self->transform.yx = FX16_TO_DBL(yx);
    self->transform.yy = FX16_TO_DBL(yy);
    self->render_flags |= FT_RFLAG_TRANSFORM;
    Py_RETURN_NONE;
}

static PyObject *
_ftface_deletetransform(PgFaceObject *self)
{
    self->render_flags &= ~FT_RFLAG_TRANSFORM;
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
PgFace_New(const char *filename, int face_index)
{
    PgFaceObject *font;

    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, 0);

    if (!filename) {
        return 0;
    }

    font = (PgFaceObject *)PgFace_Type.tp_new(
            &PgFace_Type, 0, 0);

    if (!font) {
        return 0;
    }

    if (_PGFT_TryLoadFont_Filename(ft, font, filename, face_index)) {
        return 0;
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

    if (!FREETYPE_MOD_STATE(self)->freetype) {
        if (_PGFT_Init(&(FREETYPE_MOD_STATE(self)->freetype),
                       FREETYPE_MOD_STATE(self)->cache_size)) {
            return 0;
        }
    }

    return PyInt_FromLong(result);
}

static PyObject *
_ft_quit(PyObject *self)
{
    if (FREETYPE_MOD_STATE(self)->freetype) {
        _PGFT_Quit(FREETYPE_MOD_STATE(self)->freetype);
        FREETYPE_MOD_STATE(self)->freetype = 0;
    }
    Py_RETURN_NONE;
}

static PyObject *
_ft_init(PyObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] =  {
        "cache_size", "resolution", 0
    };

    PyObject *result;
    int cache_size = PGFT_DEFAULT_CACHE_SIZE;
    unsigned resolution = 0;
    _FreeTypeState *state = FREETYPE_MOD_STATE(self);

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iI", kwlist,
                                     &cache_size, &resolution)) {
        return 0;
    }

    if (!state->freetype) {
        state->resolution = (resolution ?
                             (FT_UInt)resolution : PGFT_DEFAULT_RESOLUTION);
        result = _ft_autoinit(self);

        if (!PyObject_IsTrue(result)) {
            PyErr_SetString(PyExc_RuntimeError,
                            "Failed to initialize the FreeType2 library");
            return 0;
        }
    }

    Py_RETURN_NONE;
}


static PyObject *
_ft_get_error(PyObject *self)
{
    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, 0);

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

    if (!PyArg_ParseTuple(args, "|I", &resolution)) {
        return 0;
    }

    state->resolution = (resolution ?
                         (FT_UInt)resolution : PGFT_DEFAULT_RESOLUTION);
    Py_RETURN_NONE;
}

static PyObject *
_ft_was_init(PyObject *self)
{
    return PyBool_FromLong(FREETYPE_MOD_STATE(self)->freetype ? 1 : 0);
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
        FREETYPE_MOD_STATE(mod)->freetype = 0;
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
    0,
    _ft_traverse,
    _ft_clear,
    0
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
    if (PyType_Ready(&PgFace_Type) < 0)  {
        Py_DECREF(pygame_register_quit);
        MODINIT_ERROR;
    }

#if PY3
    module = PyModule_Create(&_freetypemodule);
#else
    /* TODO: DOC */
    module = Py_InitModule3(MODULE_NAME, _ft_methods, DOC_PYGAMEFREETYPE);
#endif

    if (!module)  {
        Py_DECREF(pygame_register_quit);
        MODINIT_ERROR;
    }

    FREETYPE_MOD_STATE(module)->freetype = 0;
    FREETYPE_MOD_STATE(module)->cache_size = PGFT_DEFAULT_CACHE_SIZE;
    FREETYPE_MOD_STATE(module)->resolution = PGFT_DEFAULT_RESOLUTION;

    Py_INCREF((PyObject *)&PgFace_Type);
    if (PyModule_AddObject(module, FACE_TYPE_NAME,
                           (PyObject *)&PgFace_Type) == -1)  {
        Py_DECREF(pygame_register_quit);
        Py_DECREF((PyObject *) &PgFace_Type);
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

#   define DEC_CONST(x)  PyModule_AddIntConstant(module, #x, (int)FT_##x)

    DEC_CONST(STYLE_NORMAL);
    DEC_CONST(STYLE_STRONG);
    DEC_CONST(STYLE_OBLIQUE);
    DEC_CONST(STYLE_UNDERLINE);
    DEC_CONST(STYLE_WIDE);

    DEC_CONST(BBOX_EXACT);
    DEC_CONST(BBOX_EXACT_GRIDFIT);
    DEC_CONST(BBOX_PIXEL);
    DEC_CONST(BBOX_PIXEL_GRIDFIT);

    /* export the c api */
#   if PYGAMEAPI_FREETYPE_NUMSLOTS != 2
#       error Mismatch between number of api slots and actual exports.
#   endif
    c_api[0] = &PgFace_Type;
    c_api[1] = &PgFace_New;

    apiobj = encapsulate_api(c_api, "freetype");
    if (!apiobj)  {
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
    if (!quit) {  /* assertion */
        Py_DECREF (pygame_register_quit);
        DECREF_MOD (module);
        MODINIT_ERROR;
    }
    rval = PyObject_CallFunctionObjArgs (pygame_register_quit, quit, 0);
    Py_DECREF (pygame_register_quit);
    Py_DECREF (quit);
    if (!rval) {
        DECREF_MOD (module);
        MODINIT_ERROR;
    }
    Py_DECREF (rval);

    MODINIT_RETURN(module);
}
