#include "doc/geometry_doc.h"
#include "geometry_common.h"

static PyObject *
_pg_circle_subtype_new(PyTypeObject *type, pgCircleBase *circle)
{
    pgCircleObject *circle_obj =
        (pgCircleObject *)pgCircle_Type.tp_new(type, NULL, NULL);

    if (circle_obj) {
        circle_obj->circle = *circle;
    }
    return (PyObject *)circle_obj;
}

static PyObject *
_pg_circle_subtype_new3(PyTypeObject *type, double x, double y, double r)
{
    pgCircleObject *circle_obj =
        (pgCircleObject *)pgCircle_Type.tp_new(type, NULL, NULL);

    if (circle_obj) {
        circle_obj->circle.x = x;
        circle_obj->circle.y = y;
        circle_obj->circle.r = r;
    }
    return (PyObject *)circle_obj;
}

static PyObject *
pg_circle_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    pgCircleObject *self = (pgCircleObject *)type->tp_alloc(type, 0);

    if (self) {
        self->circle.x = self->circle.y = 0;
        self->circle.r = 1;
        self->weakreflist = NULL;
    }
    return (PyObject *)self;
}

static int
pg_circle_init(pgCircleObject *self, PyObject *args, PyObject *kwds)
{
    if (!pgCircle_FromObject(args, &self->circle)) {
        PyErr_SetString(
            PyExc_TypeError,
            "Arguments must be a Circle, a sequence of length 3 or 2, or an "
            "object with an attribute called 'circle', all with corresponding "
            "nonnegative radius argument");
        return -1;
    }
    return 0;
}

static void
pg_circle_dealloc(pgCircleObject *self)
{
    if (self->weakreflist) {
        PyObject_ClearWeakRefs((PyObject *)self);
    }

    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
pg_circle_copy(pgCircleObject *self, PyObject *_null)
{
    return _pg_circle_subtype_new(Py_TYPE(self), &self->circle);
}

static PyObject *
pg_circle_repr(pgCircleObject *self)
{
    PyObject *x, *y, *r;

    x = PyFloat_FromDouble(self->circle.x);
    if (!x) {
        return NULL;
    }
    y = PyFloat_FromDouble(self->circle.y);
    if (!y) {
        Py_DECREF(x);
        return NULL;
    }
    r = PyFloat_FromDouble(self->circle.r);
    if (!r) {
        Py_DECREF(x);
        Py_DECREF(y);
        return NULL;
    }

    PyObject *result = PyUnicode_FromFormat("Circle((%R, %R), %R)", x, y, r);

    Py_DECREF(x);
    Py_DECREF(y);
    Py_DECREF(r);

    return result;
}

static PyObject *
pg_circle_str(pgCircleObject *self)
{
    return pg_circle_repr(self);
}

static PyObject *
pg_circle_move(pgCircleObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    double Dx, Dy;

    if (!pg_TwoDoublesFromFastcallArgs(args, nargs, &Dx, &Dy)) {
        return RAISE(PyExc_TypeError, "move requires a pair of numbers");
    }

    return _pg_circle_subtype_new3(Py_TYPE(self), self->circle.x + Dx,
                                   self->circle.y + Dy, self->circle.r);
}

static PyObject *
pg_circle_move_ip(pgCircleObject *self, PyObject *const *args,
                  Py_ssize_t nargs)
{
    double Dx, Dy;

    if (!pg_TwoDoublesFromFastcallArgs(args, nargs, &Dx, &Dy)) {
        return RAISE(PyExc_TypeError, "move_ip requires a pair of numbers");
    }

    self->circle.x += Dx;
    self->circle.y += Dy;

    Py_RETURN_NONE;
}

static PyObject *
pg_circle_update(pgCircleObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (!pgCircle_FromObjectFastcall(args, nargs, &self->circle)) {
        PyErr_SetString(
            PyExc_TypeError,
            "Circle.update requires a circle or CircleLike object");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
pg_circle_collidepoint(pgCircleObject *self, PyObject *const *args,
                       Py_ssize_t nargs)
{
    double px, py;

    if (!pg_TwoDoublesFromFastcallArgs(args, nargs, &px, &py)) {
        return RAISE(
            PyExc_TypeError,
            "Circle.collidepoint requires a point or PointLike object");
    }

    return PyBool_FromLong(pgCollision_CirclePoint(&self->circle, px, py));
}

static PyObject *
pg_circle_collidecircle(pgCircleObject *self, PyObject *const *args,
                        Py_ssize_t nargs)
{
    pgCircleBase other_circle;

    if (!pgCircle_FromObjectFastcall(args, nargs, &other_circle)) {
        return RAISE(PyExc_TypeError, "A CircleType object was expected");
    }

    return PyBool_FromLong(
        pgCollision_CircleCircle(&self->circle, &other_circle));
}

static PyObject *
pg_circle_colliderect(pgCircleObject *self, PyObject *const *args,
                      Py_ssize_t nargs)
{
    double x, y, w, h;

    if (nargs == 1) {
        SDL_FRect temp, *tmp;
        if (!(tmp = pgFRect_FromObject(args[0], &temp))) {
            return RAISE(PyExc_TypeError,
                         "Invalid rect, must be RectType or sequence of 4 "
                         "numbers");
        }
        x = (double)tmp->x;
        y = (double)tmp->y;
        w = (double)tmp->w;
        h = (double)tmp->h;
    }
    else if (nargs == 2) {
        if (!pg_TwoDoublesFromObj(args[0], &x, &y) ||
            !pg_TwoDoublesFromObj(args[1], &w, &h)) {
            return RAISE(PyExc_TypeError,
                         "Invalid rect, all 4 fields must be numeric");
        }
    }
    else if (nargs == 4) {
        if (!pg_DoubleFromObj(args[0], &x) || !pg_DoubleFromObj(args[1], &y) ||
            !pg_DoubleFromObj(args[2], &w) || !pg_DoubleFromObj(args[3], &h)) {
            return RAISE(PyExc_TypeError,
                         "Invalid rect, all 4 fields must be numeric");
        }
    }
    else {
        PyErr_Format(
            PyExc_TypeError,
            "Invalid number of arguments, expected 1, 2 or 4 (got %zd)",
            nargs);
        return NULL;
    }

    return PyBool_FromLong(pgCollision_RectCircle(x, y, w, h, &self->circle));
}

static void
_pg_rotate_circle_helper(pgCircleBase *circle, double angle, double rx,
                         double ry)
{
    if (angle == 0.0 || fmod(angle, 360.0) == 0.0) {
        return;
    }

    double x = circle->x - rx;
    double y = circle->y - ry;

    const double angle_rad = DEG_TO_RAD(angle);

    double cos_theta = cos(angle_rad);
    double sin_theta = sin(angle_rad);

    circle->x = rx + x * cos_theta - y * sin_theta;
    circle->y = ry + x * sin_theta + y * cos_theta;
}

static PyObject *
pg_circle_rotate(pgCircleObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (!nargs || nargs > 2) {
        return RAISE(PyExc_TypeError, "rotate requires 1 or 2 arguments");
    }

    pgCircleBase *circle = &self->circle;
    double angle, rx, ry;

    rx = circle->x;
    ry = circle->y;

    if (!pg_DoubleFromObj(args[0], &angle)) {
        return RAISE(PyExc_TypeError,
                     "Invalid angle argument, must be numeric");
    }

    if (nargs != 2) {
        return _pg_circle_subtype_new(Py_TYPE(self), circle);
    }

    if (!pg_TwoDoublesFromObj(args[1], &rx, &ry)) {
        return RAISE(PyExc_TypeError,
                     "Invalid rotation point argument, must be a sequence of "
                     "2 numbers");
    }

    PyObject *circle_obj = _pg_circle_subtype_new(Py_TYPE(self), circle);
    if (!circle_obj) {
        return NULL;
    }

    _pg_rotate_circle_helper(&pgCircle_AsCircle(circle_obj), angle, rx, ry);

    return circle_obj;
}

static PyObject *
pg_circle_rotate_ip(pgCircleObject *self, PyObject *const *args,
                    Py_ssize_t nargs)
{
    if (!nargs || nargs > 2) {
        return RAISE(PyExc_TypeError, "rotate requires 1 or 2 arguments");
    }

    pgCircleBase *circle = &self->circle;
    double angle, rx, ry;

    rx = circle->x;
    ry = circle->y;

    if (!pg_DoubleFromObj(args[0], &angle)) {
        return RAISE(PyExc_TypeError,
                     "Invalid angle argument, must be numeric");
    }

    if (nargs != 2) {
        /* just return None */
        Py_RETURN_NONE;
    }

    if (!pg_TwoDoublesFromObj(args[1], &rx, &ry)) {
        return RAISE(PyExc_TypeError,
                     "Invalid rotation point argument, must be a sequence "
                     "of 2 numbers");
    }

    _pg_rotate_circle_helper(circle, angle, rx, ry);

    Py_RETURN_NONE;
}

static PG_FORCEINLINE int
_pg_circle_collideswith(pgCircleBase *scirc, PyObject *arg)
{
    int result = 0;
    if (pgCircle_Check(arg)) {
        result = pgCollision_CircleCircle(&pgCircle_AsCircle(arg), scirc);
    }
    else if (pgRect_Check(arg)) {
        SDL_Rect *argrect = &pgRect_AsRect(arg);
        result = pgCollision_RectCircle((double)argrect->x, (double)argrect->y,
                                        (double)argrect->w, (double)argrect->h,
                                        scirc);
    }
    else if (pgFRect_Check(arg)) {
        SDL_FRect *argrect = &pgFRect_AsRect(arg);
        result = pgCollision_RectCircle((double)argrect->x, (double)argrect->y,
                                        (double)argrect->w, (double)argrect->h,
                                        scirc);
    }
    else if (PySequence_Check(arg)) {
        double x, y;
        if (!pg_TwoDoublesFromObj(arg, &x, &y)) {
            PyErr_SetString(
                PyExc_TypeError,
                "Invalid point argument, must be a sequence of two numbers");
            return -1;
        }
        result = pgCollision_CirclePoint(scirc, x, y);
    }
    else {
        PyErr_SetString(
            PyExc_TypeError,
            "Invalid point argument, must be a sequence of 2 numbers");
        return -1;
    }

    return result;
}

static PyObject *
pg_circle_collideswith(pgCircleObject *self, PyObject *arg)
{
    int result = _pg_circle_collideswith(&self->circle, arg);
    if (result == -1) {
        return NULL;
    }

    return PyBool_FromLong(result);
}

static PyObject *
pg_circle_collidelist(pgCircleObject *self, PyObject *arg)
{
    Py_ssize_t i;
    pgCircleBase *scirc = &self->circle;
    int colliding;

    if (!PySequence_Check(arg)) {
        return RAISE(PyExc_TypeError, "colliders argument must be a sequence");
    }

    /* fast path */
    if (pgSequenceFast_Check(arg)) {
        PyObject **items = PySequence_Fast_ITEMS(arg);
        for (i = 0; i < PySequence_Fast_GET_SIZE(arg); i++) {
            if ((colliding = _pg_circle_collideswith(scirc, items[i])) == -1) {
                /*invalid shape*/
                return NULL;
            }
            if (colliding) {
                return PyLong_FromSsize_t(i);
            }
        }
        return PyLong_FromLong(-1);
    }

    /* general sequence path */
    for (i = 0; i < PySequence_Length(arg); i++) {
        PyObject *obj = PySequence_ITEM(arg, i);
        if (!obj) {
            return NULL;
        }

        if ((colliding = _pg_circle_collideswith(scirc, obj)) == -1) {
            /*invalid shape*/
            Py_DECREF(obj);
            return NULL;
        }
        Py_DECREF(obj);

        if (colliding) {
            return PyLong_FromSsize_t(i);
        }
    }

    return PyLong_FromLong(-1);
}

static PyObject *
pg_circle_collidelistall(pgCircleObject *self, PyObject *arg)
{
    PyObject *ret;
    Py_ssize_t i;
    pgCircleBase *scirc = &self->circle;
    int colliding;

    if (!PySequence_Check(arg)) {
        return RAISE(PyExc_TypeError, "Argument must be a sequence");
    }

    ret = PyList_New(0);
    if (!ret) {
        return NULL;
    }

    /* fast path */
    if (pgSequenceFast_Check(arg)) {
        PyObject **items = PySequence_Fast_ITEMS(arg);

        for (i = 0; i < PySequence_Fast_GET_SIZE(arg); i++) {
            if ((colliding = _pg_circle_collideswith(scirc, items[i])) == -1) {
                /*invalid shape*/
                Py_DECREF(ret);
                return NULL;
            }

            if (!colliding) {
                continue;
            }

            PyObject *num = PyLong_FromSsize_t(i);
            if (!num) {
                Py_DECREF(ret);
                return NULL;
            }

            if (PyList_Append(ret, num)) {
                Py_DECREF(num);
                Py_DECREF(ret);
                return NULL;
            }
            Py_DECREF(num);
        }

        return ret;
    }

    /* general sequence path */
    for (i = 0; i < PySequence_Length(arg); i++) {
        PyObject *obj = PySequence_ITEM(arg, i);
        if (!obj) {
            Py_DECREF(ret);
            return NULL;
        }

        if ((colliding = _pg_circle_collideswith(scirc, obj)) == -1) {
            /*invalid shape*/
            Py_DECREF(ret);
            Py_DECREF(obj);
            return NULL;
        }
        Py_DECREF(obj);

        if (!colliding) {
            continue;
        }

        PyObject *num = PyLong_FromSsize_t(i);
        if (!num) {
            Py_DECREF(ret);
            return NULL;
        }

        if (PyList_Append(ret, num)) {
            Py_DECREF(num);
            Py_DECREF(ret);
            return NULL;
        }
        Py_DECREF(num);
    }

    return ret;
}

static PyObject *
pg_circle_as_rect(pgCircleObject *self, PyObject *_null)
{
    pgCircleBase *scirc = &self->circle;
    int diameter = (int)(scirc->r * 2.0);
    int x = (int)(scirc->x - scirc->r);
    int y = (int)(scirc->y - scirc->r);

    return pgRect_New4(x, y, diameter, diameter);
}

static PyObject *
pg_circle_as_frect(pgCircleObject *self, PyObject *_null)
{
    pgCircleBase *scirc = &self->circle;
    double diameter = scirc->r * 2.0;
    double x = scirc->x - scirc->r;
    double y = scirc->y - scirc->r;

    return pgFRect_New4((float)x, (float)y, (float)diameter, (float)diameter);
}

#define RECT_CIRCLE_CONTAINS(rect, circle, result)                          \
    if (pgCollision_CirclePoint(scirc, (double)rect->x, (double)rect->y) && \
        pgCollision_CirclePoint(circle, (double)(rect->x + rect->w),        \
                                (double)rect->y) &&                         \
        pgCollision_CirclePoint(circle, (double)rect->x,                    \
                                (double)(rect->y + rect->h)) &&             \
        pgCollision_CirclePoint(circle, (double)(rect->x + rect->w),        \
                                (double)(rect->y + rect->h))) {             \
        result = 1;                                                         \
    }

static PyObject *
pg_circle_contains(pgCircleObject *self, PyObject *arg)
{
    int result = 0;
    pgCircleBase *scirc = &self->circle;
    double x, y;

    if (pgCircle_Check(arg)) {
        pgCircleBase *circle = &pgCircle_AsCircle(arg);
        /*a circle is always contained within itself*/
        if (circle == scirc) {
            Py_RETURN_TRUE;
        }
        /* a bigger circle can't be contained within a smaller circle */
        if (circle->r > scirc->r) {
            Py_RETURN_FALSE;
        }

        const double dx = circle->x - scirc->x;
        const double dy = circle->y - scirc->y;
        const double dr = circle->r - scirc->r;

        result = (dx * dx + dy * dy) <= (dr * dr);
    }
    else if (pgRect_Check(arg)) {
        SDL_Rect *rect = &pgRect_AsRect(arg);
        RECT_CIRCLE_CONTAINS(rect, scirc, result);
    }
    else if (pgFRect_Check(arg)) {
        SDL_FRect *frect = &pgFRect_AsRect(arg);
        RECT_CIRCLE_CONTAINS(frect, scirc, result);
    }
    else if (pg_TwoDoublesFromObj(arg, &x, &y)) {
        result = pgCollision_CirclePoint(scirc, x, y);
    }
    else {
        return RAISE(PyExc_TypeError,
                     "Invalid shape argument, must be a Circle, Rect / Frect "
                     "or a coordinate");
    }

    return PyBool_FromLong(result);
}

static PyObject *
pg_circle_intersect(pgCircleObject *self, PyObject *arg)
{
    pgCircleBase *scirc = &self->circle;

    /* max number of intersections when supporting: Circle (2), */
    double intersections[4];
    int num = 0;

    if (pgCircle_Check(arg)) {
        pgCircleBase *other = &pgCircle_AsCircle(arg);
        num = pgIntersection_CircleCircle(scirc, other, intersections);
    }
    else {
        PyErr_Format(PyExc_TypeError, "Argument must be a CircleType, got %s",
                     Py_TYPE(arg)->tp_name);
        return NULL;
    }

    return pg_PointList_FromArrayDouble(intersections, num * 2);
}

static struct PyMethodDef pg_circle_methods[] = {
    {"collidepoint", (PyCFunction)pg_circle_collidepoint, METH_FASTCALL,
     DOC_CIRCLE_COLLIDEPOINT},
    {"collidecircle", (PyCFunction)pg_circle_collidecircle, METH_FASTCALL,
     DOC_CIRCLE_COLLIDECIRCLE},
    {"move", (PyCFunction)pg_circle_move, METH_FASTCALL, DOC_CIRCLE_MOVE},
    {"move_ip", (PyCFunction)pg_circle_move_ip, METH_FASTCALL,
     DOC_CIRCLE_MOVEIP},
    {"colliderect", (PyCFunction)pg_circle_colliderect, METH_FASTCALL,
     DOC_CIRCLE_COLLIDERECT},
    {"update", (PyCFunction)pg_circle_update, METH_FASTCALL,
     DOC_CIRCLE_UPDATE},
    {"collideswith", (PyCFunction)pg_circle_collideswith, METH_O,
     DOC_CIRCLE_COLLIDESWITH},
    {"collidelist", (PyCFunction)pg_circle_collidelist, METH_O,
     DOC_CIRCLE_COLLIDELIST},
    {"collidelistall", (PyCFunction)pg_circle_collidelistall, METH_O,
     DOC_CIRCLE_COLLIDELISTALL},
    {"as_rect", (PyCFunction)pg_circle_as_rect, METH_NOARGS,
     DOC_CIRCLE_ASRECT},
    {"as_frect", (PyCFunction)pg_circle_as_frect, METH_NOARGS,
     DOC_CIRCLE_ASFRECT},
    {"__copy__", (PyCFunction)pg_circle_copy, METH_NOARGS, DOC_CIRCLE_COPY},
    {"copy", (PyCFunction)pg_circle_copy, METH_NOARGS, DOC_CIRCLE_COPY},
    {"rotate", (PyCFunction)pg_circle_rotate, METH_FASTCALL,
     DOC_CIRCLE_ROTATE},
    {"rotate_ip", (PyCFunction)pg_circle_rotate_ip, METH_FASTCALL,
     DOC_CIRCLE_ROTATEIP},
    {"contains", (PyCFunction)pg_circle_contains, METH_O, DOC_CIRCLE_CONTAINS},
    {"intersect", (PyCFunction)pg_circle_intersect, METH_O,
     DOC_CIRCLE_INTERSECT},
    {NULL, NULL, 0, NULL}};

#define GETTER_SETTER(name)                                                   \
    static PyObject *pg_circle_get##name(pgCircleObject *self, void *closure) \
    {                                                                         \
        return PyFloat_FromDouble(self->circle.name);                         \
    }                                                                         \
    static int pg_circle_set##name(pgCircleObject *self, PyObject *value,     \
                                   void *closure)                             \
    {                                                                         \
        double val;                                                           \
        DEL_ATTR_NOT_SUPPORTED_CHECK_NO_NAME(value);                          \
        if (!pg_DoubleFromObj(value, &val)) {                                 \
            PyErr_Format(PyExc_TypeError, "Expected a number, got '%s'",      \
                         Py_TYPE(value)->tp_name);                            \
            return -1;                                                        \
        }                                                                     \
        self->circle.name = val;                                              \
        return 0;                                                             \
    }

GETTER_SETTER(x)
GETTER_SETTER(y)

#undef GETTER_SETTER

static PyObject *
pg_circle_getr(pgCircleObject *self, void *closure)
{
    return PyFloat_FromDouble(self->circle.r);
}

static int
pg_circle_setr(pgCircleObject *self, PyObject *value, void *closure)
{
    double radius;

    DEL_ATTR_NOT_SUPPORTED_CHECK_NO_NAME(value);

    if (!pg_DoubleFromObj(value, &radius)) {
        PyErr_Format(PyExc_TypeError, "Expected a number, got '%s'",
                     Py_TYPE(value)->tp_name);
        return -1;
    }

    if (radius < 0) {
        PyErr_SetString(PyExc_ValueError, "Radius must be nonnegative");
        return -1;
    }

    self->circle.r = radius;

    return 0;
}

static PyObject *
pg_circle_getr_sqr(pgCircleObject *self, void *closure)
{
    return PyFloat_FromDouble(self->circle.r * self->circle.r);
}

static int
pg_circle_setr_sqr(pgCircleObject *self, PyObject *value, void *closure)
{
    double radius_squared;

    DEL_ATTR_NOT_SUPPORTED_CHECK_NO_NAME(value);

    if (!pg_DoubleFromObj(value, &radius_squared)) {
        PyErr_SetString(PyExc_TypeError,
                        "Invalid type for radius squared, must be numeric");
        return -1;
    }

    if (radius_squared < 0) {
        PyErr_SetString(PyExc_ValueError,
                        "Invalid radius squared value, must be nonnegative");
        return -1;
    }

    self->circle.r = sqrt(radius_squared);

    return 0;
}

static PyObject *
pg_circle_getcenter(pgCircleObject *self, void *closure)
{
    return pg_tuple_couple_from_values_double(self->circle.x, self->circle.y);
}

static int
pg_circle_setcenter(pgCircleObject *self, PyObject *value, void *closure)
{
    DEL_ATTR_NOT_SUPPORTED_CHECK_NO_NAME(value);
    if (!pg_TwoDoublesFromObj(value, &self->circle.x, &self->circle.y)) {
        PyErr_SetString(PyExc_TypeError, "Expected a sequence of 2 numbers");
        return -1;
    }
    return 0;
}

static PyObject *
pg_circle_getarea(pgCircleObject *self, void *closure)
{
    return PyFloat_FromDouble(M_PI * self->circle.r * self->circle.r);
}

static int
pg_circle_setarea(pgCircleObject *self, PyObject *value, void *closure)
{
    double area;

    DEL_ATTR_NOT_SUPPORTED_CHECK_NO_NAME(value);

    if (!pg_DoubleFromObj(value, &area)) {
        PyErr_SetString(PyExc_TypeError,
                        "Invalid type for area, must be numeric");
        return -1;
    }

    if (area < 0) {
        PyErr_SetString(PyExc_ValueError,
                        "Invalid area value, must be nonnegative");
        return -1;
    }

    self->circle.r = sqrt(area / M_PI);

    return 0;
}

static PyObject *
pg_circle_getcircumference(pgCircleObject *self, void *closure)
{
    return PyFloat_FromDouble(M_TWOPI * self->circle.r);
}

static int
pg_circle_setcircumference(pgCircleObject *self, PyObject *value,
                           void *closure)
{
    double circumference;

    DEL_ATTR_NOT_SUPPORTED_CHECK_NO_NAME(value);

    if (!pg_DoubleFromObj(value, &circumference)) {
        PyErr_SetString(PyExc_TypeError,
                        "Invalid type for circumference, must be numeric");
        return -1;
    }

    if (circumference < 0) {
        PyErr_SetString(PyExc_ValueError,
                        "Invalid circumference value, must be nonnegative");
        return -1;
    }

    self->circle.r = circumference / M_TWOPI;

    return 0;
}

static PyObject *
pg_circle_getdiameter(pgCircleObject *self, void *closure)
{
    return PyFloat_FromDouble(2 * self->circle.r);
}

static int
pg_circle_setdiameter(pgCircleObject *self, PyObject *value, void *closure)
{
    double diameter;

    DEL_ATTR_NOT_SUPPORTED_CHECK_NO_NAME(value);

    if (!pg_DoubleFromObj(value, &diameter)) {
        PyErr_SetString(PyExc_TypeError,
                        "Invalid type for diameter, must be numeric");
        return -1;
    }

    if (diameter < 0) {
        PyErr_SetString(PyExc_ValueError,
                        "Invalid diameter value, must be nonnegative");
        return -1;
    }

    self->circle.r = diameter / 2;

    return 0;
}

static PyObject *
pg_circle_gettop(pgCircleObject *self, void *closure)
{
    return pg_tuple_couple_from_values_double(self->circle.x,
                                              self->circle.y - self->circle.r);
}

static int
pg_circle_settop(pgCircleObject *self, PyObject *value, void *closure)
{
    double x, y;

    DEL_ATTR_NOT_SUPPORTED_CHECK_NO_NAME(value);

    if (!pg_TwoDoublesFromObj(value, &x, &y)) {
        PyErr_SetString(PyExc_TypeError, "Expected a sequence of 2 numbers");
        return -1;
    }

    self->circle.y = y + self->circle.r;
    self->circle.x = x;

    return 0;
}

static PyObject *
pg_circle_getleft(pgCircleObject *self, void *closure)
{
    return pg_tuple_couple_from_values_double(self->circle.x - self->circle.r,
                                              self->circle.y);
}

static int
pg_circle_setleft(pgCircleObject *self, PyObject *value, void *closure)
{
    double x, y;

    DEL_ATTR_NOT_SUPPORTED_CHECK_NO_NAME(value);

    if (!pg_TwoDoublesFromObj(value, &x, &y)) {
        PyErr_SetString(PyExc_TypeError, "Expected a sequence of 2 numbers");
        return -1;
    }

    self->circle.x = x + self->circle.r;
    self->circle.y = y;

    return 0;
}

static PyObject *
pg_circle_getbottom(pgCircleObject *self, void *closure)
{
    return pg_tuple_couple_from_values_double(self->circle.x,
                                              self->circle.y + self->circle.r);
}

static int
pg_circle_setbottom(pgCircleObject *self, PyObject *value, void *closure)
{
    double x, y;

    DEL_ATTR_NOT_SUPPORTED_CHECK_NO_NAME(value);

    if (!pg_TwoDoublesFromObj(value, &x, &y)) {
        PyErr_SetString(PyExc_TypeError, "Expected a sequence of 2 numbers");
        return -1;
    }

    self->circle.y = y - self->circle.r;
    self->circle.x = x;

    return 0;
}

static PyObject *
pg_circle_getright(pgCircleObject *self, void *closure)
{
    return pg_tuple_couple_from_values_double(self->circle.x + self->circle.r,
                                              self->circle.y);
}

static int
pg_circle_setright(pgCircleObject *self, PyObject *value, void *closure)
{
    double x, y;

    DEL_ATTR_NOT_SUPPORTED_CHECK_NO_NAME(value);

    if (!pg_TwoDoublesFromObj(value, &x, &y)) {
        PyErr_SetString(PyExc_TypeError, "Expected a sequence of 2 numbers");
        return -1;
    }

    self->circle.x = x - self->circle.r;
    self->circle.y = y;

    return 0;
}

static PyObject *
pg_circle_richcompare(PyObject *self, PyObject *other, int op)
{
    pgCircleBase c1, c2;
    int equal;

    if (!pgCircle_FromObject(self, &c1) || !pgCircle_FromObject(other, &c2)) {
        equal = 0;
    }
    else {
        equal = double_compare(c1.x, c2.x) && double_compare(c1.y, c2.y) &&
                double_compare(c1.r, c2.r);
    }

    switch (op) {
        case Py_EQ:
            return PyBool_FromLong(equal);
        case Py_NE:
            return PyBool_FromLong(!equal);
        default:
            Py_RETURN_NOTIMPLEMENTED;
    }
}

static PyGetSetDef pg_circle_getsets[] = {
    {"x", (getter)pg_circle_getx, (setter)pg_circle_setx, DOC_CIRCLE_X, NULL},
    {"y", (getter)pg_circle_gety, (setter)pg_circle_sety, DOC_CIRCLE_Y, NULL},
    {"r", (getter)pg_circle_getr, (setter)pg_circle_setr, DOC_CIRCLE_R, NULL},
    {"radius", (getter)pg_circle_getr, (setter)pg_circle_setr, DOC_CIRCLE_R,
     NULL},
    {"r_sqr", (getter)pg_circle_getr_sqr, (setter)pg_circle_setr_sqr,
     DOC_CIRCLE_RSQR, NULL},
    {"d", (getter)pg_circle_getdiameter, (setter)pg_circle_setdiameter,
     DOC_CIRCLE_DIAMETER, NULL},
    {"diameter", (getter)pg_circle_getdiameter, (setter)pg_circle_setdiameter,
     DOC_CIRCLE_DIAMETER, NULL},
    {"center", (getter)pg_circle_getcenter, (setter)pg_circle_setcenter,
     DOC_CIRCLE_CENTER, NULL},
    {"area", (getter)pg_circle_getarea, (setter)pg_circle_setarea,
     DOC_CIRCLE_AREA, NULL},
    {"circumference", (getter)pg_circle_getcircumference,
     (setter)pg_circle_setcircumference, DOC_CIRCLE_CIRCUMFERENCE, NULL},
    {"top", (getter)pg_circle_gettop, (setter)pg_circle_settop, DOC_CIRCLE_TOP,
     NULL},
    {"left", (getter)pg_circle_getleft, (setter)pg_circle_setleft,
     DOC_CIRCLE_LEFT, NULL},
    {"bottom", (getter)pg_circle_getbottom, (setter)pg_circle_setbottom,
     DOC_CIRCLE_BOTTOM, NULL},
    {"right", (getter)pg_circle_getright, (setter)pg_circle_setright,
     DOC_CIRCLE_RIGHT, NULL},
    {NULL, 0, NULL, NULL, NULL}};

static PyTypeObject pgCircle_Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pygame.geometry.Circle",
    .tp_basicsize = sizeof(pgCircleObject),
    .tp_dealloc = (destructor)pg_circle_dealloc,
    .tp_repr = (reprfunc)pg_circle_repr,
    .tp_str = (reprfunc)pg_circle_str,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = DOC_CIRCLE,
    .tp_weaklistoffset = offsetof(pgCircleObject, weakreflist),
    .tp_methods = pg_circle_methods,
    .tp_getset = pg_circle_getsets,
    .tp_init = (initproc)pg_circle_init,
    .tp_new = pg_circle_new,
    .tp_richcompare = pg_circle_richcompare,
};
