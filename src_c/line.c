#include "doc/geometry_doc.h"
#include "geometry_common.h"

static double
pgLine_Length(pgLineBase *line)
{
    double dx = line->bx - line->ax;
    double dy = line->by - line->ay;
    return sqrt(dx * dx + dy * dy);
}

static PyObject *
_pg_line_subtype_new4(PyTypeObject *type, double ax, double ay, double bx,
                      double by)
{
    pgLineObject *line = (pgLineObject *)pgLine_Type.tp_new(type, NULL, NULL);

    if (line) {
        line->line.ax = ax;
        line->line.ay = ay;
        line->line.bx = bx;
        line->line.by = by;
    }
    return (PyObject *)line;
}

static PyObject *
pg_line_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    pgLineObject *self = (pgLineObject *)type->tp_alloc(type, 0);

    if (self != NULL) {
        memset(&self->line, 0, sizeof(pgLineBase));
    }

    return (PyObject *)self;
}

static void
pg_line_dealloc(pgLineObject *self)
{
    if (self->weakreflist != NULL) {
        PyObject_ClearWeakRefs((PyObject *)self);
    }

    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int
pg_line_init(pgLineObject *self, PyObject *args, PyObject *kwds)
{
    if (!pgLine_FromObject(args, &self->line)) {
        PyErr_SetString(PyExc_TypeError,
                        "Invalid line end points, expected 4 "
                        "numbers or 2 sequences of 2 numbers");
        return -1;
    }
    return 0;
}

static PyObject *
pgLine_New(pgLineBase *l)
{
    return _pg_line_subtype_new4(&pgLine_Type, l->ax, l->ay, l->bx, l->by);
}

static PyObject *
pg_line_copy(pgLineObject *self, PyObject *_null)
{
    return _pg_line_subtype_new4(Py_TYPE(self), self->line.ax, self->line.ay,
                                 self->line.bx, self->line.by);
}

static PyObject *
pg_line_update(pgLineObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (!pgLine_FromObjectFastcall(args, nargs, &self->line)) {
        return RAISE(PyExc_TypeError,
                     "Line.update requires a line or LineLike object");
    }
    Py_RETURN_NONE;
}

static PyObject *
pg_line_move(pgLineObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    double Dx, Dy;

    if (!pg_TwoDoublesFromFastcallArgs(args, nargs, &Dx, &Dy)) {
        return RAISE(PyExc_TypeError, "move requires a pair of numbers");
    }

    return _pg_line_subtype_new4(Py_TYPE(self), self->line.ax + Dx,
                                 self->line.ay + Dy, self->line.bx + Dx,
                                 self->line.by + Dy);
}

static PyObject *
pg_line_move_ip(pgLineObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    double Dx, Dy;

    if (!pg_TwoDoublesFromFastcallArgs(args, nargs, &Dx, &Dy)) {
        return RAISE(PyExc_TypeError, "move_ip requires a pair of numbers");
    }

    self->line.ax += Dx;
    self->line.ay += Dy;
    self->line.bx += Dx;
    self->line.by += Dy;

    Py_RETURN_NONE;
}

static PyObject *
pg_line_flip(pgLineObject *self, PyObject *_null)
{
    return _pg_line_subtype_new4(Py_TYPE(self), self->line.bx, self->line.by,
                                 self->line.ax, self->line.ay);
}

static PyObject *
pg_line_flip_ab_ip(pgLineObject *self, PyObject *_null)
{
    double tx = self->line.bx;
    double ty = self->line.by;

    self->line.bx = self->line.ax;
    self->line.by = self->line.ay;

    self->line.ax = tx;
    self->line.ay = ty;

    Py_RETURN_NONE;
}

static PG_FORCEINLINE double
_lerp_helper(double start, double end, double amount)
{
    return start + (end - start) * amount;
}

static int
_line_scale_helper(pgLineBase *line, double factor, double origin)
{
    if (factor == 1.0) {
        return 1;
    }
    else if (factor <= 0.0) {
        PyErr_SetString(PyExc_ValueError,
                        "Can only scale by a positive non zero number");
        return 0;
    }

    if (origin < 0.0 || origin > 1.0) {
        PyErr_SetString(PyExc_ValueError, "Origin must be between 0 and 1");
        return 0;
    }

    double ax = line->ax;
    double ay = line->ay;
    double bx = line->bx;
    double by = line->by;

    double x1_factor = ax * factor;
    double y1_factor = ay * factor;
    double x2_factor = bx * factor;
    double y2_factor = by * factor;

    double fac_m_one = factor - 1;
    double dx = _lerp_helper(fac_m_one * ax, fac_m_one * bx, origin);
    double dy = _lerp_helper(fac_m_one * ay, fac_m_one * by, origin);

    line->ax = x1_factor - dx;
    line->ay = y1_factor - dy;
    line->bx = x2_factor - dx;
    line->by = y2_factor - dy;

    return 1;
}

static PyObject *
pg_line_scale(pgLineObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    double factor, origin;

    if (!pg_TwoDoublesFromFastcallArgs(args, nargs, &factor, &origin)) {
        return RAISE(PyExc_TypeError,
                     "scale requires a sequence of two numbers");
    }

    PyObject *line;
    if (!(line = pgLine_New(&self->line))) {
        return NULL;
    }

    if (!_line_scale_helper(&pgLine_AsLine(line), factor, origin)) {
        Py_DECREF(line);
        return NULL;
    }

    return line;
}

static PyObject *
pg_line_scale_ip(pgLineObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    double factor, origin;

    if (!pg_TwoDoublesFromFastcallArgs(args, nargs, &factor, &origin)) {
        return RAISE(PyExc_TypeError,
                     "scale_ip requires a sequence of two numbers");
    }

    if (!_line_scale_helper(&pgLine_AsLine(self), factor, origin)) {
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject *
_line_project_helper(pgLineBase *line, double *point, int clamp)
{
    // this is a vector that goes from one point of the line to another
    double line_vector[2] = {line->bx - line->ax, line->by - line->ay};
    double squared_line_length =
        line_vector[0] * line_vector[0] + line_vector[1] * line_vector[1];

    if (squared_line_length == 0.0) {
        double projected_point[2];
        projected_point[0] = line->ax;
        projected_point[1] = line->ay;
        return pg_tuple_couple_from_values_double(projected_point[0],
                                                  projected_point[1]);
    }

    // this is a vector that goes from the start of the line to the point we
    // are projecting onto the line
    double vector_from_line_start_to_point[2] = {point[0] - line->ax,
                                                 point[1] - line->ay};

    double dot_product =
        (vector_from_line_start_to_point[0] * line_vector[0] +
         vector_from_line_start_to_point[1] * line_vector[1]) /
        squared_line_length;

    double projection[2] = {dot_product * line_vector[0],
                            dot_product * line_vector[1]};

    if (clamp) {
        if (dot_product < 0) {
            projection[0] = 0;
            projection[1] = 0;
        }
        else if (projection[0] * projection[0] +
                     projection[1] * projection[1] >
                 squared_line_length) {
            projection[0] = line_vector[0];
            projection[1] = line_vector[1];
        }
    }

    double projected_point[2] = {line->ax + projection[0],
                                 line->ay + projection[1]};

    return pg_tuple_couple_from_values_double(projected_point[0],
                                              projected_point[1]);
}

static PyObject *
pg_line_project(pgLineObject *self, PyObject *args, PyObject *kwnames)
{
    double point[2] = {0.f, 0.f};
    int clamp = 0;

    PyObject *point_obj = NULL;

    static char *kwlist[] = {"point", "clamp", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwnames, "O|p:project", kwlist,
                                     &point_obj, &clamp)) {
        return RAISE(
            PyExc_TypeError,
            "project requires a sequence(point) and an optional clamp flag");
    }

    if (!pg_TwoDoublesFromObj(point_obj, &point[0], &point[1])) {
        return RAISE(PyExc_TypeError,
                     "project requires a sequence of two numbers");
    }

    return _line_project_helper(&pgLine_AsLine(self), point, clamp);
}

static struct PyMethodDef pg_line_methods[] = {
    {"__copy__", (PyCFunction)pg_line_copy, METH_NOARGS, DOC_LINE_COPY},
    {"copy", (PyCFunction)pg_line_copy, METH_NOARGS, DOC_LINE_COPY},
    {"update", (PyCFunction)pg_line_update, METH_FASTCALL, DOC_LINE_UPDATE},
    {"move", (PyCFunction)pg_line_move, METH_FASTCALL, DOC_LINE_MOVE},
    {"move_ip", (PyCFunction)pg_line_move_ip, METH_FASTCALL, DOC_LINE_MOVEIP},
    {"flip_ab", (PyCFunction)pg_line_flip, METH_NOARGS, DOC_LINE_FLIPAB},
    {"flip_ab_ip", (PyCFunction)pg_line_flip_ab_ip, METH_NOARGS,
     DOC_LINE_FLIPABIP},
    {"scale", (PyCFunction)pg_line_scale, METH_FASTCALL, DOC_LINE_SCALE},
    {"scale_ip", (PyCFunction)pg_line_scale_ip, METH_FASTCALL,
     DOC_LINE_SCALEIP},
    {"project", (PyCFunction)pg_line_project, METH_VARARGS | METH_KEYWORDS,
     DOC_LINE_PROJECT},
    {NULL, NULL, 0, NULL}};

static PyObject *
pg_line_repr(pgLineObject *self)
{
    PyObject *result, *ax, *ay, *bx, *by;

    ax = PyFloat_FromDouble(self->line.ax);
    if (!ax) {
        return NULL;
    }
    ay = PyFloat_FromDouble(self->line.ay);
    if (!ay) {
        Py_DECREF(ax);
        return NULL;
    }
    bx = PyFloat_FromDouble(self->line.bx);
    if (!bx) {
        Py_DECREF(ax);
        Py_DECREF(ay);
        return NULL;
    }
    by = PyFloat_FromDouble(self->line.by);
    if (!by) {
        Py_DECREF(ax);
        Py_DECREF(ay);
        Py_DECREF(bx);
        return NULL;
    }

    result = PyUnicode_FromFormat("Line((%R, %R), (%R, %R))", ax, ay, bx, by);

    Py_DECREF(ax);
    Py_DECREF(ay);
    Py_DECREF(bx);
    Py_DECREF(by);

    return result;
}

static PyObject *
pg_line_str(pgLineObject *self)
{
    return pg_line_repr(self);
}

#define __LINE_GETSET_NAME(name)                                          \
    static PyObject *pg_line_get##name(pgLineObject *self, void *closure) \
    {                                                                     \
        return PyFloat_FromDouble(self->line.name);                       \
    }                                                                     \
    static int pg_line_set##name(pgLineObject *self, PyObject *value,     \
                                 void *closure)                           \
    {                                                                     \
        double val;                                                       \
        DEL_ATTR_NOT_SUPPORTED_CHECK_NO_NAME(value);                      \
        if (pg_DoubleFromObj(value, &val)) {                              \
            self->line.name = val;                                        \
            return 0;                                                     \
        }                                                                 \
        PyErr_SetString(PyExc_TypeError, "Expected a number");            \
        return -1;                                                        \
    }

__LINE_GETSET_NAME(ax)
__LINE_GETSET_NAME(ay)
__LINE_GETSET_NAME(bx)
__LINE_GETSET_NAME(by)
#undef __LINE_GETSET_NAME

static PyObject *
pg_line_geta(pgLineObject *self, void *closure)
{
    return pg_tuple_couple_from_values_double(self->line.ax, self->line.ay);
}

static int
pg_line_seta(pgLineObject *self, PyObject *value, void *closure)
{
    double x, y;
    DEL_ATTR_NOT_SUPPORTED_CHECK_NO_NAME(value);
    if (pg_TwoDoublesFromObj(value, &x, &y)) {
        self->line.ax = x;
        self->line.ay = y;
        return 0;
    }
    PyErr_SetString(PyExc_TypeError, "Expected a sequence of 2 numbers");
    return -1;
}

static PyObject *
pg_line_getb(pgLineObject *self, void *closure)
{
    return pg_tuple_couple_from_values_double(self->line.bx, self->line.by);
}

static int
pg_line_setb(pgLineObject *self, PyObject *value, void *closure)
{
    double x, y;
    DEL_ATTR_NOT_SUPPORTED_CHECK_NO_NAME(value);
    if (pg_TwoDoublesFromObj(value, &x, &y)) {
        self->line.bx = x;
        self->line.by = y;
        return 0;
    }
    PyErr_SetString(PyExc_TypeError, "Expected a sequence of 2 numbers");
    return -1;
}

static PyObject *
pg_line_getlength(pgLineObject *self, void *closure)
{
    return PyFloat_FromDouble(pgLine_Length(&self->line));
}

static PyGetSetDef pg_line_getsets[] = {
    {"ax", (getter)pg_line_getax, (setter)pg_line_setax, DOC_LINE_AX, NULL},
    {"ay", (getter)pg_line_getay, (setter)pg_line_setay, DOC_LINE_AY, NULL},
    {"bx", (getter)pg_line_getbx, (setter)pg_line_setbx, DOC_LINE_BX, NULL},
    {"by", (getter)pg_line_getby, (setter)pg_line_setby, DOC_LINE_BY, NULL},
    {"a", (getter)pg_line_geta, (setter)pg_line_seta, DOC_LINE_A, NULL},
    {"b", (getter)pg_line_getb, (setter)pg_line_setb, DOC_LINE_B, NULL},
    {"length", (getter)pg_line_getlength, NULL, DOC_LINE_LENGTH, NULL},
    {NULL, 0, NULL, NULL, NULL}};

static PyTypeObject pgLine_Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pygame.geometry.Line",
    .tp_basicsize = sizeof(pgLineObject),
    .tp_dealloc = (destructor)pg_line_dealloc,
    .tp_repr = (reprfunc)pg_line_repr,
    .tp_str = (reprfunc)pg_line_str,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = DOC_LINE,
    .tp_weaklistoffset = offsetof(pgLineObject, weakreflist),
    .tp_methods = pg_line_methods,
    .tp_getset = pg_line_getsets,
    .tp_init = (initproc)pg_line_init,
    .tp_new = pg_line_new,
};
