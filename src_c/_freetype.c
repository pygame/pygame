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

#define MODULE_NAME "_freetype"
#define FONT_TYPE_NAME "Font"

/*
 * FreeType module declarations
 */
static const Scale_t FACE_SIZE_NONE = {0, 0};

#if PY3
static int
_ft_traverse(PyObject *, visitproc, void *);
static int
_ft_clear(PyObject *);
#endif

static PyObject *
_ft_quit(PyObject *);
static PyObject *
_ft_init(PyObject *, PyObject *, PyObject *);
static PyObject *
_ft_get_version(PyObject *);
static PyObject *
_ft_get_error(PyObject *);
static PyObject *
_ft_was_init(PyObject *);
static PyObject *
_ft_autoinit(PyObject *);
static void
_ft_autoquit(void);
static PyObject *
_ft_get_cache_size(PyObject *);
static PyObject *
_ft_get_default_resolution(PyObject *);
static PyObject *
_ft_set_default_resolution(PyObject *, PyObject *);
static PyObject *
_ft_get_default_font(PyObject *self);

/*
 * Constructor/init/destructor
 */
static PyObject *
_ftfont_new(PyTypeObject *, PyObject *, PyObject *);
static void
_ftfont_dealloc(pgFontObject *);
static PyObject *
_ftfont_repr(pgFontObject *);
static int
_ftfont_init(pgFontObject *, PyObject *, PyObject *);

/*
 * Main methods
 */
static PyObject *
_ftfont_getrect(pgFontObject *, PyObject *, PyObject *);
static PyObject *
_ftfont_getmetrics(pgFontObject *, PyObject *, PyObject *);
static PyObject *
_ftfont_render(pgFontObject *, PyObject *, PyObject *);
static PyObject *
_ftfont_render_to(pgFontObject *, PyObject *, PyObject *);
static PyObject *
_ftfont_render_raw(pgFontObject *, PyObject *, PyObject *);
static PyObject *
_ftfont_render_raw_to(pgFontObject *, PyObject *, PyObject *);
static PyObject *
_ftfont_getsizedascender(pgFontObject *, PyObject *);
static PyObject *
_ftfont_getsizeddescender(pgFontObject *, PyObject *);
static PyObject *
_ftfont_getsizedheight(pgFontObject *, PyObject *);
static PyObject *
_ftfont_getsizedglyphheight(pgFontObject *, PyObject *);
static PyObject *
_ftfont_getsizes(pgFontObject *);

/* static PyObject *_ftfont_copy(pgFontObject *); */

/*
 * Getters/setters
 */
static PyObject *
_ftfont_getsize(pgFontObject *, void *);
static int
_ftfont_setsize(pgFontObject *, PyObject *, void *);
static PyObject *
_ftfont_getstyle(pgFontObject *, void *);
static int
_ftfont_setstyle(pgFontObject *, PyObject *, void *);
static PyObject *
_ftfont_getname(pgFontObject *, void *);
static PyObject *
_ftfont_getpath(pgFontObject *, void *);
static PyObject *
_ftfont_getscalable(pgFontObject *, void *);
static PyObject *
_ftfont_getfixedwidth(pgFontObject *, void *);
static PyObject *
_ftfont_getfixedsizes(pgFontObject *, void *);
static PyObject *
_ftfont_getstrength(pgFontObject *, void *);
static int
_ftfont_setstrength(pgFontObject *, PyObject *, void *);
static PyObject *
_ftfont_getunderlineadjustment(pgFontObject *, void *);
static int
_ftfont_setunderlineadjustment(pgFontObject *, PyObject *, void *);
static PyObject *
_ftfont_getrotation(pgFontObject *, void *);
static int
_ftfont_setrotation(pgFontObject *, PyObject *, void *);
static PyObject *
_ftfont_getfgcolor(pgFontObject *, void *);
static int
_ftfont_setfgcolor(pgFontObject *, PyObject *, void *);

static PyObject *
_ftfont_getresolution(pgFontObject *, void *);

static PyObject *
_ftfont_getfontmetric(pgFontObject *, void *);

static PyObject *
_ftfont_getstyle_flag(pgFontObject *, void *);
static int
_ftfont_setstyle_flag(pgFontObject *, PyObject *, void *);

static PyObject *
_ftfont_getrender_flag(pgFontObject *, void *);
static int
_ftfont_setrender_flag(pgFontObject *, PyObject *, void *);

#if defined(PGFT_DEBUG_CACHE)
static PyObject *
_ftfont_getdebugcachestats(pgFontObject *, void *);
#endif

/*
 * Internal helpers
 */
static PyObject *
get_metrics(FontRenderMode *, pgFontObject *, PGFT_String *);
static PyObject *
load_font_res(const char *);
static int
parse_dest(PyObject *, int *, int *);
static int
obj_to_scale(PyObject *, void *);
static int
objs_to_scale(PyObject *, PyObject *, Scale_t *);
static int
numbers_to_scale(PyObject *, PyObject *, Scale_t *);
static int
build_scale(PyObject *, PyObject *, Scale_t *);
static FT_UInt
number_to_FX6_unsigned(PyObject *);
static int
obj_to_rotation(PyObject *, void *);
static void
free_string(PGFT_String *);

/*
 * Auxiliar defines
 */
#define ASSERT_SELF_IS_ALIVE(s)                                          \
    if (!pgFont_IS_ALIVE(s)) {                                           \
        return RAISE(PyExc_RuntimeError, MODULE_NAME                     \
                     "." FONT_TYPE_NAME " instance is not initialized"); \
    }

#define PGFT_CHECK_BOOL(_pyobj, _var)                          \
    if (_pyobj) {                                              \
        if (!PyBool_Check(_pyobj)) {                           \
            PyErr_SetString(PyExc_TypeError,                   \
                            #_var " must be a boolean value"); \
            return 0;                                          \
        }                                                      \
                                                               \
        _var = PyObject_IsTrue(_pyobj);                        \
    }

#define DEFAULT_FONT_NAME "freesansbold.ttf"
#define PKGDATA_MODULE_NAME "pygame.pkgdata"
#define RESOURCE_FUNC_NAME "getResource"

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
    else {
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

    if (!PySequence_Check(dest) || /* conditional and */
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

/** Point size PyArg_ParseTuple converter: int -> Scale_t */
static int
obj_to_scale(PyObject *o, void *p)
{
    if (PyTuple_Check(o)) {
        if (PyTuple_GET_SIZE(o) != 2) {
            PyErr_Format(PyExc_TypeError,
                         "expected a 2-tuple for size, got %zd-tuple",
                         PyTuple_GET_SIZE(o));
            return 0;
        }
        return objs_to_scale(PyTuple_GET_ITEM(o, 0), PyTuple_GET_ITEM(o, 1),
                             (Scale_t *)p);
    }
    return objs_to_scale(o, 0, (Scale_t *)p);
}

static int
objs_to_scale(PyObject *x, PyObject *y, Scale_t *size)
{
    PyObject *o;
    int do_y;

    for (o = x, do_y = 1; o; o = (do_y--) ? y : 0) {
        if (!PyLong_Check(o) &&
#if PY2
            !PyInt_Check(o) &&
#endif
            !PyFloat_Check(o)) {
            if (y) {
                PyErr_Format(PyExc_TypeError,
                             "expected a (float, float) tuple for size"
                             ", got (%128s, %128s)",
                             Py_TYPE(x)->tp_name, Py_TYPE(y)->tp_name);
            }
            else {
                PyErr_Format(PyExc_TypeError,
                             "expected a float for size, got %128s",
                             Py_TYPE(o)->tp_name);
            }
            return 0;
        }
    }

    return numbers_to_scale(x, y, size);
}

static int
numbers_to_scale(PyObject *x, PyObject *y, Scale_t *size)
{
    PyObject *o;
    PyObject *min_obj = 0;
    PyObject *max_obj = 0;
    int do_y;
    int cmp_result;
    int rval = 0;

    min_obj = PyFloat_FromDouble(0.0);
    if (!min_obj)
        goto finish;
    max_obj = PyFloat_FromDouble(FX6_TO_DBL(FX6_MAX));
    if (!max_obj)
        goto finish;

    for (o = x, do_y = 1; o; o = (do_y--) ? y : 0) {
        cmp_result = PyObject_RichCompareBool(o, min_obj, Py_LT);
        if (cmp_result == -1)
            goto finish;
        if (cmp_result == 1) {
            PyErr_Format(PyExc_OverflowError,
                         "%128s value is negative"
                         " while size value is zero or positive",
                         Py_TYPE(o)->tp_name);
            goto finish;
        }
        cmp_result = PyObject_RichCompareBool(o, max_obj, Py_GT);
        if (cmp_result == -1)
            goto finish;
        if (cmp_result == 1) {
            PyErr_Format(PyExc_OverflowError,
                         "%128s value too large to convert to a size value",
                         Py_TYPE(o)->tp_name);
            goto finish;
        }
    }

    rval = build_scale(x, y, size);

finish:
    Py_XDECREF(min_obj);
    Py_XDECREF(max_obj);
    return rval;
}

static int
build_scale(PyObject *x, PyObject *y, Scale_t *size)
{
    FT_UInt sz_x = 0, sz_y = 0;

    sz_x = number_to_FX6_unsigned(x);
    if (PyErr_Occurred()) {
        return 0;
    }
    if (y) {
        sz_y = number_to_FX6_unsigned(y);
        if (PyErr_Occurred()) {
            return 0;
        }
    }
    if (sz_x == 0 && sz_y != 0) {
        PyErr_SetString(PyExc_ValueError,
                        "expected zero size height when width is zero");
        return 0;
    }
    size->x = sz_x;
    size->y = sz_y;
    return 1;
}

static FT_UInt
number_to_FX6_unsigned(PyObject *n)
{
    PyObject *f_obj = PyNumber_Float(n);
    double f;

    if (!f_obj)
        return 0;
    f = PyFloat_AsDouble(f_obj);
    Py_XDECREF(f_obj);
    if (PyErr_Occurred())
        return 0;
    return DBL_TO_FX6(f);
}

/** rotation: int -> Angle_t */
int
obj_to_rotation(PyObject *o, void *p)
{
    PyObject *full_circle_obj = 0;
    PyObject *angle_obj = 0;
    long angle;
    int rval = 0;

    if (PyLong_Check(o)) {
        ;
    }
#if PY2
    else if (PyInt_Check(o)) {
        ;
    }
#endif
    else {
        PyErr_Format(PyExc_TypeError, "integer rotation expected, got %s",
                     Py_TYPE(o)->tp_name);
        goto finish;
    }
    full_circle_obj = PyLong_FromLong(360L);
    if (!full_circle_obj)
        goto finish;
    angle_obj = PyNumber_Remainder(o, full_circle_obj);
    if (!angle_obj)
        goto finish;
    angle = PyLong_AsLong(angle_obj);
    if (angle == -1)
        goto finish;
    *(Angle_t *)p = (Angle_t)INT_TO_FX16(angle);
    rval = 1;

finish:
    Py_XDECREF(full_circle_obj);
    Py_XDECREF(angle_obj);
    return rval;
}

/** This accepts a NULL PGFT_String pointer */
static void
free_string(PGFT_String *p)
{
    if (p)
        _PGFT_FreeString(p);
}

/*
 * FREETYPE MODULE METHODS TABLE
 */
static PyMethodDef _ft_methods[] = {
    {"__PYGAMEinit__", (PyCFunction)_ft_autoinit, METH_NOARGS,
     "auto initialize function for _freetype"},
    {"init", (PyCFunction)_ft_init, METH_VARARGS | METH_KEYWORDS,
     DOC_PYGAMEFREETYPEINIT},
    {"quit", (PyCFunction)_ft_quit, METH_NOARGS, DOC_PYGAMEFREETYPEQUIT},
    {"was_init", (PyCFunction)_ft_was_init, METH_NOARGS,
     DOC_PYGAMEFREETYPEWASINIT},
    {"get_error", (PyCFunction)_ft_get_error, METH_NOARGS,
     DOC_PYGAMEFREETYPEGETERROR},
    {"get_version", (PyCFunction)_ft_get_version, METH_NOARGS,
     DOC_PYGAMEFREETYPEGETVERSION},
    {"get_cache_size", (PyCFunction)_ft_get_cache_size, METH_NOARGS,
     DOC_PYGAMEFREETYPEGETCACHESIZE},
    {"get_default_resolution", (PyCFunction)_ft_get_default_resolution,
     METH_NOARGS, DOC_PYGAMEFREETYPEGETDEFAULTRESOLUTION},
    {"set_default_resolution", (PyCFunction)_ft_set_default_resolution,
     METH_VARARGS, DOC_PYGAMEFREETYPESETDEFAULTRESOLUTION},
    {"get_default_font", (PyCFunction)_ft_get_default_font, METH_NOARGS,
     DOC_PYGAMEFREETYPEGETDEFAULTFONT},

    {0, 0, 0, 0}};

/*
 * FREETYPE FONT METHODS TABLE
 */
static PyMethodDef _ftfont_methods[] = {
    {"get_sized_height", (PyCFunction)_ftfont_getsizedheight, METH_VARARGS,
     DOC_FONTGETSIZEDHEIGHT},
    {"get_sized_ascender", (PyCFunction)_ftfont_getsizedascender, METH_VARARGS,
     DOC_FONTGETSIZEDASCENDER},
    {"get_sized_descender", (PyCFunction)_ftfont_getsizeddescender,
     METH_VARARGS, DOC_FONTGETSIZEDDESCENDER},
    {"get_sized_glyph_height", (PyCFunction)_ftfont_getsizedglyphheight,
     METH_VARARGS, DOC_FONTGETSIZEDGLYPHHEIGHT},
    {"get_rect", (PyCFunction)_ftfont_getrect, METH_VARARGS | METH_KEYWORDS,
     DOC_FONTGETRECT},
    {"get_metrics", (PyCFunction)_ftfont_getmetrics,
     METH_VARARGS | METH_KEYWORDS, DOC_FONTGETMETRICS},
    {"get_sizes", (PyCFunction)_ftfont_getsizes, METH_NOARGS,
     DOC_FONTGETSIZES},
    {"render", (PyCFunction)_ftfont_render, METH_VARARGS | METH_KEYWORDS,
     DOC_FONTRENDER},
    {"render_to", (PyCFunction)_ftfont_render_to, METH_VARARGS | METH_KEYWORDS,
     DOC_FONTRENDERTO},
    {"render_raw", (PyCFunction)_ftfont_render_raw,
     METH_VARARGS | METH_KEYWORDS, DOC_FONTRENDERRAW},
    {"render_raw_to", (PyCFunction)_ftfont_render_raw_to,
     METH_VARARGS | METH_KEYWORDS, DOC_FONTRENDERRAWTO},

    {0, 0, 0, 0}};

/*
 * FREETYPE FONT GETTERS/SETTERS TABLE
 */
static PyGetSetDef _ftfont_getsets[] = {
    {"size", (getter)_ftfont_getsize, (setter)_ftfont_setsize, DOC_FONTSIZE,
     0},
    {"style", (getter)_ftfont_getstyle, (setter)_ftfont_setstyle,
     DOC_FONTSTYLE, 0},
    {"height", (getter)_ftfont_getfontmetric, 0, DOC_FONTHEIGHT,
     (void *)_PGFT_Font_GetHeight},
    {"ascender", (getter)_ftfont_getfontmetric, 0, DOC_FONTASCENDER,
     (void *)_PGFT_Font_GetAscender},
    {"descender", (getter)_ftfont_getfontmetric, 0, DOC_FONTASCENDER,
     (void *)_PGFT_Font_GetDescender},
    {"name", (getter)_ftfont_getname, 0, DOC_FONTNAME, 0},
    {"path", (getter)_ftfont_getpath, 0, DOC_FONTPATH, 0},
    {"scalable", (getter)_ftfont_getscalable, 0, DOC_FONTSCALABLE, 0},
    {"fixed_width", (getter)_ftfont_getfixedwidth, 0, DOC_FONTFIXEDWIDTH, 0},
    {"fixed_sizes", (getter)_ftfont_getfixedsizes, 0, DOC_FONTFIXEDSIZES, 0},
    {"antialiased", (getter)_ftfont_getrender_flag,
     (setter)_ftfont_setrender_flag, DOC_FONTANTIALIASED,
     (void *)FT_RFLAG_ANTIALIAS},
    {"kerning", (getter)_ftfont_getrender_flag, (setter)_ftfont_setrender_flag,
     DOC_FONTKERNING, (void *)FT_RFLAG_KERNING},
    {"vertical", (getter)_ftfont_getrender_flag,
     (setter)_ftfont_setrender_flag, DOC_FONTVERTICAL,
     (void *)FT_RFLAG_VERTICAL},
    {"pad", (getter)_ftfont_getrender_flag, (setter)_ftfont_setrender_flag,
     DOC_FONTPAD, (void *)FT_RFLAG_PAD},
    {"oblique", (getter)_ftfont_getstyle_flag, (setter)_ftfont_setstyle_flag,
     DOC_FONTOBLIQUE, (void *)FT_STYLE_OBLIQUE},
    {"strong", (getter)_ftfont_getstyle_flag, (setter)_ftfont_setstyle_flag,
     DOC_FONTSTRONG, (void *)FT_STYLE_STRONG},
    {"underline", (getter)_ftfont_getstyle_flag, (setter)_ftfont_setstyle_flag,
     DOC_FONTUNDERLINE, (void *)FT_STYLE_UNDERLINE},
    {"wide", (getter)_ftfont_getstyle_flag, (setter)_ftfont_setstyle_flag,
     DOC_FONTWIDE, (void *)FT_STYLE_WIDE},
    {"strength", (getter)_ftfont_getstrength, (setter)_ftfont_setstrength,
     DOC_FONTSTRENGTH, 0},
    {"underline_adjustment", (getter)_ftfont_getunderlineadjustment,
     (setter)_ftfont_setunderlineadjustment, DOC_FONTUNDERLINEADJUSTMENT, 0},
    {"ucs4", (getter)_ftfont_getrender_flag, (setter)_ftfont_setrender_flag,
     DOC_FONTUCS4, (void *)FT_RFLAG_UCS4},
    {"use_bitmap_strikes", (getter)_ftfont_getrender_flag,
     (setter)_ftfont_setrender_flag, DOC_FONTUSEBITMAPSTRIKES,
     (void *)FT_RFLAG_USE_BITMAP_STRIKES},
    {"resolution", (getter)_ftfont_getresolution, 0, DOC_FONTRESOLUTION, 0},
    {"rotation", (getter)_ftfont_getrotation, (setter)_ftfont_setrotation,
     DOC_FONTROTATION, 0},
    {"fgcolor", (getter)_ftfont_getfgcolor, (setter)_ftfont_setfgcolor,
     DOC_FONTFGCOLOR, 0},
    {"origin", (getter)_ftfont_getrender_flag, (setter)_ftfont_setrender_flag,
     DOC_FONTORIGIN, (void *)FT_RFLAG_ORIGIN},
#if defined(PGFT_DEBUG_CACHE)
    {"_debug_cache_stats", (getter)_ftfont_getdebugcachestats, 0,
     "_debug cache fields as a tuple", 0},
#endif

    {0, 0, 0, 0, 0}};

/*
 * FREETYPE FONT BASE TYPE TABLE
 */
#define FULL_TYPE_NAME MODULE_NAME "." FONT_TYPE_NAME

PyTypeObject pgFont_Type = {
    TYPE_HEAD(0, 0) FULL_TYPE_NAME,           /* tp_name */
    sizeof(pgFontObject),                     /* tp_basicsize */
    0,                                        /* tp_itemsize */
    (destructor)_ftfont_dealloc,              /* tp_dealloc */
    0,                                        /* tp_print */
    0,                                        /* tp_getattr */
    0,                                        /* tp_setattr */
    0,                                        /* tp_compare */
    (reprfunc)_ftfont_repr,                   /* tp_repr */
    0,                                        /* tp_as_number */
    0,                                        /* tp_as_sequence */
    0,                                        /* tp_as_mapping */
    0,                                        /* tp_hash */
    0,                                        /* tp_call */
    0,                                        /* tp_str */
    0,                                        /* tp_getattro */
    0,                                        /* tp_setattro */
    0,                                        /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    DOC_PYGAMEFREETYPEFONT,                   /* docstring */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                                        /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    _ftfont_methods,                          /* tp_methods */
    0,                                        /* tp_members */
    _ftfont_getsets,                          /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    (initproc)_ftfont_init,                   /* tp_init */
    0,                                        /* tp_alloc */
    (newfunc)_ftfont_new,                     /* tp_new */
    0,                                        /* tp_free */
    0,                                        /* tp_is_gc */
    0,                                        /* tp_bases */
    0,                                        /* tp_mro */
    0,                                        /* tp_cache */
    0,                                        /* tp_subclasses */
    0,                                        /* tp_weaklist */
    0,                                        /* tp_del */
    0 /* tp_version_tag */
};

#undef FULL_TYPE_NAME

/****************************************************
 * CONSTRUCTOR/INIT/DESTRUCTOR
 ****************************************************/
static PyObject *
_ftfont_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds)
{
    pgFontObject *obj = (pgFontObject *)(subtype->tp_alloc(subtype, 0));

    if (obj) {
        obj->id.open_args.flags = 0;
        obj->id.open_args.pathname = 0;
        obj->path = 0;
        obj->resolution = 0;
        obj->is_scalable = 0;
        obj->freetype = 0;
        obj->_internals = 0;
        obj->face_size = FACE_SIZE_NONE;
        obj->style = FT_STYLE_NORMAL;
        obj->render_flags = FT_RFLAG_DEFAULTS;
        obj->strength = PGFT_DBL_DEFAULT_STRENGTH;
        obj->underline_adjustment = 1.0;
        obj->rotation = 0;
        obj->transform.xx = FX16_ONE;
        obj->transform.xy = 0;
        obj->transform.yx = 0;
        obj->transform.yy = FX16_ONE;
        obj->fgcolor[0] = 0; /* rgba opaque black */
        obj->fgcolor[1] = 0;
        obj->fgcolor[2] = 0;
        obj->fgcolor[3] = 255;
    }
    return (PyObject *)obj;
}

static void
_ftfont_dealloc(pgFontObject *self)
{
#ifdef HAVE_PYGAME_SDL_RWOPS
    SDL_RWops *src = _PGFT_GetRWops(self);
#endif
    _PGFT_UnloadFont(self->freetype, self);
#ifdef HAVE_PYGAME_SDL_RWOPS
    if (src) {
        pgRWopsFreeFromObject(src);
    }
#endif
    _PGFT_Quit(self->freetype);

    Py_XDECREF(self->path);
    ((PyObject *)self)->ob_type->tp_free((PyObject *)self);
}

static int
_ftfont_init(pgFontObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"file",       "size", "font_index",
                             "resolution", "ucs4", 0};

    PyObject *file, *original_file;
    long font_index = 0;
    Scale_t face_size = self->face_size;
    int ucs4 = self->render_flags & FT_RFLAG_UCS4 ? 1 : 0;
    unsigned resolution = 0;
    long size = 0;
    long height = 0;
    long width = 0;
    double x_ppem = 0;
    double y_ppem = 0;
    int rval = -1;
    SDL_RWops *source;

    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, -1);

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O&lIi", kwlist, &file,
                                     obj_to_scale, (void *)&face_size,
                                     &font_index, &resolution, &ucs4)) {
        return -1;
    }

    original_file = file;

    if (self->freetype) {
        /* Font.__init__ was previously called on this object. Reset */
        _PGFT_UnloadFont(self->freetype, self);
        _PGFT_Quit(self->freetype);
        self->freetype = 0;
    }
    Py_XDECREF(self->path);
    self->path = 0;
    self->is_scalable = 0;

    self->face_size = face_size;
    if (ucs4) {
        self->render_flags |= FT_RFLAG_UCS4;
    }
    else {
        self->render_flags &= ~FT_RFLAG_UCS4;
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

#if !defined(WIN32) || !defined(HAVE_PYGAME_SDL_RWOPS)
    file = pgRWopsEncodeString(file, "UTF-8", NULL, NULL);
    if (!file) {
        goto end;
    }
    if (Bytes_Check(file)) {
        if (PyUnicode_Check(original_file)) {
            /* Make sure to save a pure Unicode object to prevent possible
             * cycles from a derived class. This means no tp_traverse or
             * tp_clear for the PyFreetypeFont type.
             */
            self->path = Object_Unicode(original_file);
        }
        else {
            self->path = PyUnicode_FromEncodedObject(
                file, "UTF-8", NULL);
        }
        if (!self->path) {
            goto end;
        }

        if (_PGFT_TryLoadFont_Filename(ft, self, Bytes_AS_STRING(file),
                                       font_index)) {
            goto end;
        }
    } else {
        PyObject *str = 0;
        PyObject *path = 0;
        source = pgRWopsFromFileObjectThreaded(original_file);
        if (!source) {
            goto end;
        }

        path = PyObject_GetAttrString(original_file, "name");
        if (!path) {
            PyErr_Clear();
            str = Bytes_FromFormat("<%s instance at %p>",
                                   Py_TYPE(file)->tp_name, (void *)file);
            if (str) {
                self->path =
                    PyUnicode_FromEncodedObject(str, "ascii", "strict");
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
            self->path = PyUnicode_FromEncodedObject(
                path, "UTF-8", NULL);
        }
        else {
            self->path = Object_Unicode(path);
        }
        Py_XDECREF(path);
        if (!self->path) {
            goto end;
        }

        if (_PGFT_TryLoadFont_RWops(ft, self, source, font_index)) {
            goto end;
        }
    }
#else /* WIN32 && HAVE_PYGAME_SDL_RWOPS */
    /* FT uses fopen(); as a workaround, always use RWops */
    if (file == original_file)
        Py_INCREF(file);
    source = pgRWopsFromObjectThreaded(file);
    if (!source) {
        goto end;
    } else {
        PyObject *path = 0;

        if (pgRWopsCheckObjectThreaded(source)) {
            path = PyObject_GetAttrString(file, "name");
        } else {
            Py_INCREF(file);
            path = file;
        }
        if (!path) {
            PyObject *str;
            PyErr_Clear();
            str = Bytes_FromFormat("<%s instance at %p>",
                                   Py_TYPE(file)->tp_name, (void *)file);
            if (str) {
                self->path =
                    PyUnicode_FromEncodedObject(str, "ascii", "strict");
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
            self->path = PyUnicode_FromEncodedObject(
                path, "UTF-8", NULL);
        }
        else {
            self->path = Object_Unicode(path);
        }
        Py_XDECREF(path);
        if (!self->path) {
            goto end;
        }

        if (_PGFT_TryLoadFont_RWops(ft, self, source, font_index)) {
            goto end;
        }
    }
#endif /* WIN32 && HAVE_PYGAME_SDL_RWOPS */

    if (!self->is_scalable && self->face_size.x == 0) {
        if (_PGFT_Font_GetAvailableSize(ft, self, 0, &size, &height, &width,
                                        &x_ppem, &y_ppem)) {
            self->face_size.x = DBL_TO_FX6(x_ppem);
            self->face_size.y = DBL_TO_FX6(y_ppem);
        }
        else {
            PyErr_Clear();
        }
    }

    /* Keep the current freetype 2 connection open while this object exists.
       Otherwise, the freetype library may be closed before the object frees
       its local resources. See Pygame issue #187
    */
    self->freetype = ft;
    ++ft->ref_count;

    rval = 0;

end:
    Py_XDECREF(file);
    return rval;
}

static PyObject *
_ftfont_repr(pgFontObject *self)
{
    if (pgFont_IS_ALIVE(self)) {
#if PY3
        return PyUnicode_FromFormat("Font('%.1024U')", self->path);
#else
        PyObject *str = PyUnicode_AsEncodedString(
            self->path, "raw_unicode_escape", "replace");
        PyObject *rval = 0;

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

/** Generic style attributes */
static PyObject *
_ftfont_getstyle_flag(pgFontObject *self, void *closure)
{
    const int style_flag = (int)closure;

    return PyBool_FromLong(self->style & style_flag);
}

static int
_ftfont_setstyle_flag(pgFontObject *self, PyObject *value, void *closure)
{
    const int style_flag = (int)closure;

    if (!PyBool_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "The style value must be a boolean");
        return -1;
    }

    if ((style_flag & FT_STYLES_SCALABLE_ONLY) && !self->is_scalable) {
        if (pgFont_IS_ALIVE(self)) {
            PyErr_SetString(PyExc_AttributeError,
                            "this style is unsupported for a bitmap font");
        }
        else {
            PyErr_SetString(PyExc_RuntimeError, MODULE_NAME
                            "." FONT_TYPE_NAME " instance is not initialized");
        }
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
_ftfont_getstyle(pgFontObject *self, void *closure)
{
    return PyInt_FromLong(self->style);
}

static int
_ftfont_setstyle(pgFontObject *self, PyObject *value, void *closure)
{
    FT_UInt32 style;

    if (!PyInt_Check(value)) {
        PyErr_SetString(PyExc_TypeError,
                        "The style value must be an integer"
                        " from the FT constants module");
        return -1;
    }

    style = (FT_UInt32)PyInt_AsLong(value);

    if (style == FT_STYLE_DEFAULT) {
        /* The Font object's style property is the Font's default style,
         * so leave unchanged.
         */
        return 0;
    }
    if (_PGFT_CheckStyle(style)) {
        PyErr_Format(PyExc_ValueError, "Invalid style value %x", (int)style);
        return -1;
    }
    if ((style & FT_STYLES_SCALABLE_ONLY) && !self->is_scalable) {
        if (pgFont_IS_ALIVE(self)) {
            PyErr_SetString(PyExc_AttributeError,
                            "this style is unsupported for a bitmap font");
        }
        else {
            PyErr_SetString(PyExc_RuntimeError, MODULE_NAME
                            "." FONT_TYPE_NAME " instance is not initialized");
        }
        return -1;
    }

    self->style = (FT_UInt16)style;
    return 0;
}

static PyObject *
_ftfont_getstrength(pgFontObject *self, void *closure)
{
    return PyFloat_FromDouble(self->strength);
}

static int
_ftfont_setstrength(pgFontObject *self, PyObject *value, void *closure)
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
_ftfont_getsize(pgFontObject *self, void *closure)
{
    if (self->face_size.y == 0) {
        return PyFloat_FromDouble(FX6_TO_DBL(self->face_size.x));
    }
    return Py_BuildValue("dd", FX6_TO_DBL(self->face_size.x),
                         FX6_TO_DBL(self->face_size.y));
}

static int
_ftfont_setsize(pgFontObject *self, PyObject *value, void *closure)
{
    Scale_t face_size;

    if (!obj_to_scale(value, &face_size))
        goto error;
    self->face_size = face_size;
    return 0;

error:
    return -1;
}

static PyObject *
_ftfont_getunderlineadjustment(pgFontObject *self, void *closure)
{
    return PyFloat_FromDouble(self->underline_adjustment);
}

static int
_ftfont_setunderlineadjustment(pgFontObject *self, PyObject *value,
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

/** general font attributes */

static PyObject *
_ftfont_getfontmetric(pgFontObject *self, void *closure)
{
    typedef long (*getter)(FreeTypeInstance *, pgFontObject *);
    long height;

    ASSERT_SELF_IS_ALIVE(self);
    height = ((getter)closure)(self->freetype, self);
    if (!height && PyErr_Occurred()) {
        return 0;
    }
    return PyInt_FromLong(height);
}

static PyObject *
_ftfont_getname(pgFontObject *self, void *closure)
{
    const char *name;

    if (pgFont_IS_ALIVE(self)) {
        name = _PGFT_Font_GetName(self->freetype, self);
        return name ? Text_FromUTF8(name) : 0;
    }
    return PyObject_Repr((PyObject *)self);
}

static PyObject *
_ftfont_getpath(pgFontObject *self, void *closure)
{
    PyObject *path = ((pgFontObject *)self)->path;

    if (!path) {
        PyErr_SetString(PyExc_AttributeError, "path unavailable");
        return 0;
    }
    Py_INCREF(path);
    return path;
}

static PyObject *
_ftfont_getscalable(pgFontObject *self, void *closure)
{
    ASSERT_SELF_IS_ALIVE(self)
    return PyBool_FromLong(self->is_scalable);
}

static PyObject *
_ftfont_getfixedwidth(pgFontObject *self, void *closure)
{
    long fixed_width;

    ASSERT_SELF_IS_ALIVE(self);
    fixed_width =
        _PGFT_Font_IsFixedWidth(self->freetype, (pgFontObject *)self);
    return fixed_width >= 0 ? PyBool_FromLong(fixed_width) : 0;
}

static PyObject *
_ftfont_getfixedsizes(pgFontObject *self, void *closure)
{
    long num_fixed_sizes;

    ASSERT_SELF_IS_ALIVE(self);
    num_fixed_sizes = _PGFT_Font_NumFixedSizes(self->freetype, self);
    return num_fixed_sizes >= 0 ? PyInt_FromLong(num_fixed_sizes) : 0;
}

/** Generic render flag attributes */
static PyObject *
_ftfont_getrender_flag(pgFontObject *self, void *closure)
{
    const int render_flag = (int)closure;

    return PyBool_FromLong(self->render_flags & render_flag);
}

static int
_ftfont_setrender_flag(pgFontObject *self, PyObject *value, void *closure)
{
    const int render_flag = (int)closure;

    if (!PyBool_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "The style value must be a boolean");
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
_ftfont_getresolution(pgFontObject *self, void *closure)
{
    return PyLong_FromUnsignedLong((unsigned long)self->resolution);
}

/** text rotation attribute */
static PyObject *
_ftfont_getrotation(pgFontObject *self, void *closure)
{
    return PyLong_FromLong((long)FX16_ROUND_TO_INT(self->rotation));
}

static int
_ftfont_setrotation(pgFontObject *self, PyObject *value, void *closure)
{
    if (!self->is_scalable) {
        if (pgFont_IS_ALIVE(self)) {
            PyErr_SetString(PyExc_AttributeError,
                            "rotation is unsupported for a bitmap font");
        }
        else {
            PyErr_SetString(PyExc_RuntimeError, MODULE_NAME
                            "." FONT_TYPE_NAME " instance is not initialized");
        }
        return -1;
    }
    return obj_to_rotation(value, &self->rotation) ? 0 : -1;
}

/** default glyph color */
static PyObject *
_ftfont_getfgcolor(pgFontObject *self, void *closure)
{
    return pgColor_New(self->fgcolor);
}

static int
_ftfont_setfgcolor(pgFontObject *self, PyObject *value, void *closure)
{
    if (!pg_RGBAFromObj(value, self->fgcolor)) {
        PyErr_Format(PyExc_AttributeError,
                     "unable to convert %128s object to a color",
                     Py_TYPE(value)->tp_name);
        return -1;
    }
    return 0;
}

/** testing and debugging */
#if defined(PGFT_DEBUG_CACHE)
static PyObject *
_ftfont_getdebugcachestats(pgFontObject *self, void *closure)
{
    /* Yes, this kind of breaches the boundary between the top level
     * freetype.c and the lower level ft_text.c. But it is built
     * conditionally, and it keeps some of the Python api out
     * of ft_text.c and ft_cache.c (hoping to remove the Python
     * api completely from ft_text.c and support C modules at some point.)
     */
    const FontCache *cache = &PGFT_FONT_CACHE(self);

    return Py_BuildValue("kkkkk", (unsigned long)cache->_debug_count,
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
_ftfont_getrect(pgFontObject *self, PyObject *args, PyObject *kwds)
{
    /* MODIFIED
     */
    /* keyword list */
    static char *kwlist[] = {"text", "style", "rotation", "size", 0};

    PyObject *textobj;
    PGFT_String *text = 0;
    Scale_t face_size = FACE_SIZE_NONE;
    SDL_Rect r;

    FontRenderMode render;
    Angle_t rotation = self->rotation;
    int style = FT_STYLE_DEFAULT;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "O|iO&O&", kwlist, &textobj, &style, obj_to_rotation,
            (void *)&rotation, obj_to_scale, (void *)&face_size))
        goto error;

    /* Encode text */
    if (textobj != Py_None) {
        text =
            _PGFT_EncodePyString(textobj, self->render_flags & FT_RFLAG_UCS4);
        if (!text)
            goto error;
    }

    ASSERT_SELF_IS_ALIVE(self);

    /* Build rendering mode, always anti-aliased by default */
    if (_PGFT_BuildRenderMode(self->freetype, self, &render, face_size, style,
                              rotation))
        goto error;

    if (_PGFT_GetTextRect(self->freetype, self, &render, text, &r))
        goto error;
    free_string(text);

    return pgRect_New(&r);

error:
    free_string(text);
    return 0;
}

static PyObject *
get_metrics(FontRenderMode *render, pgFontObject *font, PGFT_String *text)
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

    if (!_PGFT_GetFontSized(font->freetype, font, render->face_size)) {
        PyErr_SetString(pgExc_SDLError, _PGFT_GetError(font->freetype));
        return 0;
    }
    list = PyList_New(length);
    if (!list) {
        return 0;
    }
    for (i = 0; i < length; ++i) {
        if (_PGFT_GetMetrics(font->freetype, font, data[i], render, &gindex,
                             &minx, &maxx, &miny, &maxy, &advance_x,
                             &advance_y) == 0) {
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
_ftfont_getmetrics(pgFontObject *self, PyObject *args, PyObject *kwds)
{
    /* keyword list */
    static char *kwlist[] = {"text", "size", 0};

    FontRenderMode render;
    PyObject *list = 0;

    /* arguments */
    PyObject *textobj;
    PGFT_String *text = 0;
    Scale_t face_size = FACE_SIZE_NONE;

    /* parse args */
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O&", kwlist, &textobj,
                                     obj_to_scale, (void *)&face_size))
        goto error;

    /* Encode text */
    text = _PGFT_EncodePyString(textobj, self->render_flags & FT_RFLAG_UCS4);
    if (!text)
        goto error;

    ASSERT_SELF_IS_ALIVE(self);

    /*
     * Build the render mode with the given size and support
     * for rotation/styles/size changes in text
     */
    if (_PGFT_BuildRenderMode(self->freetype, self, &render, face_size,
                              FT_STYLE_DEFAULT, self->rotation))
        goto error;

    /* get metrics */
    list = get_metrics(&render, self, text);
    if (!list)
        goto error;
    free_string(text);

    return list;

error:
    free_string(text);
    Py_XDECREF(list);
    return 0;
}

static PyObject *
_ftfont_getsizedascender(pgFontObject *self, PyObject *args)
{
    Scale_t face_size = FACE_SIZE_NONE;
    long value;

    if (!PyArg_ParseTuple(args, "|O&", obj_to_scale, (void *)&face_size)) {
        return 0;
    }

    if (face_size.x == 0) {
        if (self->face_size.x == 0) {
            RAISE(PyExc_ValueError,
                  "No font point size specified"
                  " and no default font size in typefont");
            return 0;
        }

        face_size = self->face_size;
    }
    value = (long)_PGFT_Font_GetAscenderSized(self->freetype, self, face_size);
    if (!value && PyErr_Occurred()) {
        return 0;
    }
    return PyInt_FromLong(value);
}

static PyObject *
_ftfont_getsizeddescender(pgFontObject *self, PyObject *args)
{
    Scale_t face_size = FACE_SIZE_NONE;
    long value;

    if (!PyArg_ParseTuple(args, "|O&", obj_to_scale, (void *)&face_size)) {
        return 0;
    }

    if (face_size.x == 0) {
        if (self->face_size.x == 0) {
            RAISE(PyExc_ValueError,
                  "No font point size specified"
                  " and no default font size in typefont");
            return 0;
        }

        face_size = self->face_size;
    }
    value =
        (long)_PGFT_Font_GetDescenderSized(self->freetype, self, face_size);
    if (!value && PyErr_Occurred()) {
        return 0;
    }
    return PyInt_FromLong(value);
}

static PyObject *
_ftfont_getsizedheight(pgFontObject *self, PyObject *args)
{
    Scale_t face_size = FACE_SIZE_NONE;
    long value;

    if (!PyArg_ParseTuple(args, "|O&", obj_to_scale, (void *)&face_size)) {
        return 0;
    }

    if (face_size.x == 0) {
        if (self->face_size.x == 0) {
            RAISE(PyExc_ValueError,
                  "No font point size specified"
                  " and no default font size in typeface");
            return 0;
        }

        face_size = self->face_size;
    }
    value = _PGFT_Font_GetHeightSized(self->freetype, self, face_size);
    if (!value && PyErr_Occurred()) {
        return 0;
    }
    return PyInt_FromLong(value);
}

static PyObject *
_ftfont_getsizedglyphheight(pgFontObject *self, PyObject *args)
{
    Scale_t face_size = FACE_SIZE_NONE;
    long value;

    if (!PyArg_ParseTuple(args, "|O&", obj_to_scale, (void *)&face_size)) {
        return 0;
    }

    if (face_size.x == 0) {
        if (self->face_size.x == 0) {
            RAISE(PyExc_ValueError,
                  "No font point size specified"
                  " and no default font size in typeface");
            return 0;
        }

        face_size = self->face_size;
    }
    value =
        (long)_PGFT_Font_GetGlyphHeightSized(self->freetype, self, face_size);
    if (!value && PyErr_Occurred()) {
        return 0;
    }
    return PyInt_FromLong(value);
}

static PyObject *
_ftfont_getsizes(pgFontObject *self)
{
    int nsizes;
    unsigned i;
    int rc;
    long size = 0;
    long height = 0, width = 0;
    double x_ppem = 0.0, y_ppem = 0.0;
    PyObject *size_list = 0;
    PyObject *size_item;

    nsizes = _PGFT_Font_NumFixedSizes(self->freetype, self);
    if (nsizes < 0)
        goto error;
    size_list = PyList_New(nsizes);
    if (!size_list)
        goto error;
    for (i = 0; i < nsizes; ++i) {
        rc = _PGFT_Font_GetAvailableSize(self->freetype, self, i, &size,
                                         &height, &width, &x_ppem, &y_ppem);
        if (rc < 0)
            goto error;
        assert(rc > 0);
        size_item =
            Py_BuildValue("llldd", size, height, width, x_ppem, y_ppem);
        if (!size_item)
            goto error;
        PyList_SET_ITEM(size_list, i, size_item);
    }
    return size_list;

error:
    Py_XDECREF(size_list);
    return 0;
}

static PyObject *
_ftfont_render_raw(pgFontObject *self, PyObject *args, PyObject *kwds)
{
    /* keyword list */
    static char *kwlist[] = {"text", "style", "rotation", "size", "invert", 0};

    FontRenderMode mode;

    /* input arguments */
    PyObject *textobj;
    PGFT_String *text = 0;
    int style = FT_STYLE_DEFAULT;
    Angle_t rotation = self->rotation;
    Scale_t face_size = FACE_SIZE_NONE;
    int invert = 0;

    /* output arguments */
    PyObject *rbuffer = 0;
    PyObject *rtuple = 0;
    int width, height;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "O|iO&O&i", kwlist, &textobj, &style, obj_to_rotation,
            (void *)&rotation, obj_to_scale, (void *)&face_size, &invert))
        goto error;

    /* Encode text */
    if (textobj != Py_None) {
        text =
            _PGFT_EncodePyString(textobj, self->render_flags & FT_RFLAG_UCS4);
        if (!text)
            goto error;
    }

    ASSERT_SELF_IS_ALIVE(self);

    /*
     * Build the render mode with the given size and no
     * rotation/styles/vertical text
     */
    if (_PGFT_BuildRenderMode(self->freetype, self, &mode, face_size, style,
                              rotation))
        goto error;

    rbuffer = _PGFT_Render_PixelArray(self->freetype, self, &mode, text,
                                      invert, &width, &height);
    if (!rbuffer)
        goto error;
    free_string(text);
    rtuple = Py_BuildValue("O(ii)", rbuffer, width, height);
    if (!rtuple)
        goto error;
    Py_DECREF(rbuffer);

    return rtuple;

error:
    free_string(text);
    Py_XDECREF(rbuffer);
    Py_XDECREF(rtuple);
    return 0;
}

static PyObject *
_ftfont_render_raw_to(pgFontObject *self, PyObject *args, PyObject *kwds)
{
    /* keyword list */
    static char *kwlist[] = {"array",    "text", "dest",   "style",
                             "rotation", "size", "invert", 0};

    FontRenderMode mode;

    /* input arguments */
    PyObject *arrayobj;
    PyObject *textobj;
    PGFT_String *text = 0;
    PyObject *dest = 0;
    int xpos = 0;
    int ypos = 0;
    int style = FT_STYLE_DEFAULT;
    Angle_t rotation = self->rotation;
    Scale_t face_size = FACE_SIZE_NONE;
    int invert = 0;

    /* output arguments */
    SDL_Rect r;

    ASSERT_SELF_IS_ALIVE(self);

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "OO|OiO&O&i", kwlist, &arrayobj, &textobj, &dest,
            &style, obj_to_rotation, (void *)&rotation, obj_to_scale,
            (void *)&face_size, &invert))
        goto error;

    if (dest && dest != Py_None) {
        if (parse_dest(dest, &xpos, &ypos))
            goto error;
    }

    /* Encode text */
    if (textobj != Py_None) {
        text =
            _PGFT_EncodePyString(textobj, self->render_flags & FT_RFLAG_UCS4);
        if (!text)
            goto error;
    }

    /*
     * Build the render mode with the given size and no
     * rotation/styles/vertical text
     */
    if (_PGFT_BuildRenderMode(self->freetype, self, &mode, face_size, style,
                              rotation))
        goto error;

    if (_PGFT_Render_Array(self->freetype, self, &mode, arrayobj, text, invert,
                           xpos, ypos, &r))
        goto error;
    free_string(text);

    return pgRect_New(&r);

error:
    free_string(text);
    return 0;
}

static PyObject *
_ftfont_render(pgFontObject *self, PyObject *args, PyObject *kwds)
{
#ifndef HAVE_PYGAME_SDL_VIDEO

    PyErr_SetString(PyExc_RuntimeError,
                    "SDL support is missing. Cannot render on surfonts");
    return 0;

#else
    /* keyword list */
    static char *kwlist[] = {"text",     "fgcolor", "bgcolor", "style",
                             "rotation", "size",    0};

    /* input arguments */
    PyObject *textobj = 0;
    PGFT_String *text = 0;
    Scale_t face_size = FACE_SIZE_NONE;
    PyObject *fg_color_obj = 0;
    PyObject *bg_color_obj = 0;
    Angle_t rotation = self->rotation;
    int style = FT_STYLE_DEFAULT;

    /* output arguments */
    SDL_Surface *surface = 0;
    PyObject *surface_obj = 0;
    PyObject *rtuple = 0;
    SDL_Rect r;
    PyObject *rect_obj = 0;

    FontColor fg_color;
    FontColor bg_color;
    FontRenderMode render;

    ASSERT_SELF_IS_ALIVE(self);

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOiO&O&", kwlist,
                                     /* required */
                                     &textobj,
                                     /* optional */
                                     &fg_color_obj, &bg_color_obj, &style,
                                     obj_to_rotation, (void *)&rotation,
                                     obj_to_scale, (void *)&face_size))
        goto error;

    if (fg_color_obj == Py_None) {
        fg_color_obj = 0;
    }
    if (bg_color_obj == Py_None) {
        bg_color_obj = 0;
    }

    if (fg_color_obj) {
        if (!pg_RGBAFromColorObj(fg_color_obj, (Uint8 *)&fg_color)) {
            PyErr_SetString(PyExc_TypeError, "fgcolor must be a Color");
            goto error;
        }
    }
    else {
        fg_color.r = self->fgcolor[0];
        fg_color.g = self->fgcolor[1];
        fg_color.b = self->fgcolor[2];
        fg_color.a = self->fgcolor[3];
    }
    if (bg_color_obj) {
        if (!pg_RGBAFromColorObj(bg_color_obj, (Uint8 *)&bg_color)) {
            PyErr_SetString(PyExc_TypeError, "bgcolor must be a Color");
            goto error;
        }
    }

    /* Encode text */
    if (textobj != Py_None) {
        text =
            _PGFT_EncodePyString(textobj, self->render_flags & FT_RFLAG_UCS4);
        if (!text)
            goto error;
    }

    if (_PGFT_BuildRenderMode(self->freetype, self, &render, face_size, style,
                              rotation))
        goto error;

    surface =
        _PGFT_Render_NewSurface(self->freetype, self, &render, text, &fg_color,
                                bg_color_obj ? &bg_color : 0, &r);
    if (!surface)
        goto error;
    free_string(text);
    surface_obj = pgSurface_New(surface);
    if (!surface_obj)
        goto error;

    rect_obj = pgRect_New(&r);
    if (!rect_obj)
        goto error;
    rtuple = PyTuple_Pack(2, surface_obj, rect_obj);
    if (!rtuple)
        goto error;
    Py_DECREF(surface_obj);
    Py_DECREF(rect_obj);

    return rtuple;

error:
    free_string(text);
    if (surface_obj) {
        Py_DECREF(surface_obj);
    }
    else if (surface) {
        SDL_FreeSurface(surface);
    }
    Py_XDECREF(rect_obj);
    Py_XDECREF(rtuple);
    return 0;

#endif  // HAVE_PYGAME_SDL_VIDEO
}

static PyObject *
_ftfont_render_to(pgFontObject *self, PyObject *args, PyObject *kwds)
{
#ifndef HAVE_PYGAME_SDL_VIDEO

    PyErr_SetString(PyExc_RuntimeError,
                    "SDL support is missing. Cannot render on surfaces");
    return 0;

#else
    /* keyword list */
    static char *kwlist[] = {"surf",  "dest",     "text", "fgcolor", "bgcolor",
                             "style", "rotation", "size", 0};

    /* input arguments */
    PyObject *surface_obj = 0;
    PyObject *textobj = 0;
    PGFT_String *text = 0;
    Scale_t face_size = FACE_SIZE_NONE;
    PyObject *dest = 0;
    int xpos = 0;
    int ypos = 0;
    PyObject *fg_color_obj = 0;
    PyObject *bg_color_obj = 0;
    Angle_t rotation = self->rotation;
    int style = FT_STYLE_DEFAULT;
    SDL_Surface *surface = 0;

    /* output arguments */
    SDL_Rect r;

    FontColor fg_color;
    FontColor bg_color;
    FontRenderMode render;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "O!OO|OOiO&O&", kwlist,
            /* required */
            &pgSurface_Type, &surface_obj, &dest, &textobj, &fg_color_obj,
            /* optional */
            &bg_color_obj, &style, obj_to_rotation, (void *)&rotation,
            obj_to_scale, (void *)&face_size))
        goto error;

    if (fg_color_obj == Py_None) {
        fg_color_obj = 0;
    }
    if (bg_color_obj == Py_None) {
        bg_color_obj = 0;
    }

    if (parse_dest(dest, &xpos, &ypos))
        goto error;
    if (fg_color_obj) {
        if (!pg_RGBAFromColorObj(fg_color_obj, (Uint8 *)&fg_color)) {
            PyErr_SetString(PyExc_TypeError, "fgcolor must be a Color");
            goto error;
        }
    }
    else {
        fg_color.r = self->fgcolor[0];
        fg_color.g = self->fgcolor[1];
        fg_color.b = self->fgcolor[2];
        fg_color.a = self->fgcolor[3];
    }
    if (bg_color_obj) {
        if (!pg_RGBAFromColorObj(bg_color_obj, (Uint8 *)&bg_color)) {
            PyErr_SetString(PyExc_TypeError, "bgcolor must be a Color");
            goto error;
        }
    }

    ASSERT_SELF_IS_ALIVE(self);

    /* Encode text */
    if (textobj != Py_None) {
        text =
            _PGFT_EncodePyString(textobj, self->render_flags & FT_RFLAG_UCS4);
        if (!text)
            goto error;
    }

    if (_PGFT_BuildRenderMode(self->freetype, self, &render, face_size, style,
                              rotation))
        goto error;

    surface = pgSurface_AsSurface(surface_obj);
    if (!surface) {
        PyErr_SetString(pgExc_SDLError, "display Surface quit");
        goto error;
    }
    if (_PGFT_Render_ExistingSurface(self->freetype, self, &render, text,
                                     surface, xpos, ypos, &fg_color,
                                     bg_color_obj ? &bg_color : 0, &r))
        goto error;
    free_string(text);

    return pgRect_New(&r);

error:
    free_string(text);
    return 0;
#endif  // HAVE_PYGAME_SDL_VIDEO
}

/****************************************************
 * C API CALLS
 ****************************************************/
static PyObject *
pgFont_New(const char *filename, long font_index)
{
    pgFontObject *font;

    FreeTypeInstance *ft;
    ASSERT_GRAB_FREETYPE(ft, 0);

    if (!filename) {
        return 0;
    }

    font = (pgFontObject *)pgFont_Type.tp_new(&pgFont_Type, 0, 0);

    if (!font) {
        return 0;
    }

    if (_PGFT_TryLoadFont_Filename(ft, font, filename, font_index)) {
        return 0;
    }

    return (PyObject *)font;
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
    int cache_size = FREETYPE_MOD_STATE(self)->cache_size;
    FT_Error result = 1;

    if (!FREETYPE_MOD_STATE(self)->freetype) {
        pg_RegisterQuit(_ft_autoquit);

        if (cache_size == 0) {
            cache_size = PGFT_DEFAULT_CACHE_SIZE;
        }
        if (_PGFT_Init(&(FREETYPE_MOD_STATE(self)->freetype), cache_size)) {
            return 0;
        }
        FREETYPE_MOD_STATE(self)->cache_size = cache_size;
    }

    return PyInt_FromLong(result);
}

static void
_ft_autoquit(void)
{
    _FreeTypeState *state = FREETYPE_STATE;

    if (state->freetype) {
        _PGFT_Quit(state->freetype);
        state->cache_size = 0;
        state->freetype = 0;
    }
}

static PyObject *
_ft_quit(PyObject *self)
{
    _ft_autoquit();
    Py_RETURN_NONE;
}

static PyObject *
_ft_init(PyObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"cache_size", "resolution", 0};

    PyObject *result;
    unsigned cache_size = 0;
    unsigned resolution = 0;
    _FreeTypeState *state = FREETYPE_MOD_STATE(self);

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|II", kwlist, &cache_size,
                                     &resolution)) {
        return 0;
    }

    if (!state->freetype) {
        state->cache_size = cache_size;
        state->resolution =
            (resolution ? (FT_UInt)resolution : PGFT_DEFAULT_RESOLUTION);
        result = _ft_autoinit(self);

        if (!result) {
            PyErr_Clear();
            PyErr_SetString(PyExc_RuntimeError,
                            "Failed to initialize the FreeType2 library");
            return 0;
        }
        Py_DECREF(result);
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
    return Py_BuildValue("iii", FREETYPE_MAJOR, FREETYPE_MINOR,
                         FREETYPE_PATCH);
}

static PyObject *
_ft_get_cache_size(PyObject *self)
{
    return PyLong_FromUnsignedLong(
        (unsigned long)(FREETYPE_STATE->cache_size));
}

static PyObject *
_ft_get_default_resolution(PyObject *self)
{
    return PyLong_FromUnsignedLong(
        (unsigned long)(FREETYPE_STATE->resolution));
}

static PyObject *
_ft_set_default_resolution(PyObject *self, PyObject *args)
{
    unsigned resolution = 0;
    _FreeTypeState *state = FREETYPE_MOD_STATE(self);

    if (!PyArg_ParseTuple(args, "|I", &resolution)) {
        return 0;
    }

    state->resolution =
        (resolution ? (FT_UInt)resolution : PGFT_DEFAULT_RESOLUTION);
    Py_RETURN_NONE;
}

static PyObject *
_ft_was_init(PyObject *self)
{
    return PyBool_FromLong(FREETYPE_MOD_STATE(self)->freetype ? 1 : 0);
}

static PyObject *
_ft_get_default_font(PyObject *self)
{
    return Text_FromUTF8(DEFAULT_FONT_NAME);
}

#if PY3
static int
_ft_traverse(PyObject *mod, visitproc visit, void *arg)
{
    return 0;
}

static int
_ft_clear(PyObject *mod)
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
#ifndef PYPY_VERSION
struct PyModuleDef _freetypemodule = {
    PyModuleDef_HEAD_INIT,  MODULE_NAME, DOC_PYGAMEFREETYPE,
    sizeof(_FreeTypeState), _ft_methods, 0,
    _ft_traverse,           _ft_clear,   0};
#else /* PYPY_VERSION */
_FreeTypeState _modstate;
struct PyModuleDef _freetypemodule = {
    PyModuleDef_HEAD_INIT,  MODULE_NAME, DOC_PYGAMEFREETYPE,
    -1 /* PyModule_GetState() not implemented */, _ft_methods, 0,
    _ft_traverse, _ft_clear, 0};
#endif /* PYPY_VERSION */
#else /* PY2 */
_FreeTypeState _modstate;
#endif /* PY2 */

MODINIT_DEFINE(_freetype)
{
    PyObject *module, *apiobj;
    static void *c_api[PYGAMEAPI_FREETYPE_NUMSLOTS];

    import_pygame_base();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }

    import_pygame_surface();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }

    import_pygame_color();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }

    import_pygame_rwobject();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }

    import_pygame_rect();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }

    /* type preparation */
    if (PyType_Ready(&pgFont_Type) < 0) {
        MODINIT_ERROR;
    }

#if PY3
    module = PyModule_Create(&_freetypemodule);
#else
    /* TODO: DOC */
    module = Py_InitModule3(MODULE_NAME, _ft_methods, DOC_PYGAMEFREETYPE);
#endif

    if (!module) {
        MODINIT_ERROR;
    }

    FREETYPE_MOD_STATE(module)->freetype = 0;
    FREETYPE_MOD_STATE(module)->cache_size = 0;
    FREETYPE_MOD_STATE(module)->resolution = PGFT_DEFAULT_RESOLUTION;

    Py_INCREF((PyObject *)&pgFont_Type);
    if (PyModule_AddObject(module, FONT_TYPE_NAME, (PyObject *)&pgFont_Type) ==
        -1) {
        Py_DECREF((PyObject *)&pgFont_Type);
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

#define DEC_CONST(x) PyModule_AddIntConstant(module, #x, (int)FT_##x)

    DEC_CONST(STYLE_NORMAL);
    DEC_CONST(STYLE_STRONG);
    DEC_CONST(STYLE_OBLIQUE);
    DEC_CONST(STYLE_UNDERLINE);
    DEC_CONST(STYLE_WIDE);
    DEC_CONST(STYLE_DEFAULT);

    DEC_CONST(BBOX_EXACT);
    DEC_CONST(BBOX_EXACT_GRIDFIT);
    DEC_CONST(BBOX_PIXEL);
    DEC_CONST(BBOX_PIXEL_GRIDFIT);

    /* export the c api */
#if PYGAMEAPI_FREETYPE_NUMSLOTS != 2
#error Mismatch between number of api slots and actual exports.
#endif
    c_api[0] = &pgFont_Type;
    c_api[1] = &pgFont_New;

    apiobj = encapsulate_api(c_api, "freetype");
    if (!apiobj) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

    if (PyModule_AddObject(module, PYGAMEAPI_LOCAL_ENTRY, apiobj) == -1) {
        Py_DECREF(apiobj);
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

    MODINIT_RETURN(module);
}
