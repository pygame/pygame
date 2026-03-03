#include "geometry_common.h"

int
_pg_circle_set_radius(PyObject *value, pgCircleBase *circle)
{
    double radius = 0.0;
    if (!pg_DoubleFromObj(value, &radius) || radius < 0.0) {
        return 0;
    }
    circle->r = radius;
    return 1;
}

int
pgCircle_FromObject(PyObject *obj, pgCircleBase *out)
{
    Py_ssize_t length;

    if (pgCircle_Check(obj)) {
        *out = pgCircle_AsCircle(obj);
        return 1;
    }

    /* Paths for sequences */
    if (pgSequenceFast_Check(obj)) {
        PyObject **f_arr = PySequence_Fast_ITEMS(obj);
        length = PySequence_Fast_GET_SIZE(obj);

        switch (length) {
            case 1:
                return pgCircle_FromObject(f_arr[0], out);
            case 2:
                return pg_TwoDoublesFromObj(f_arr[0], &out->x, &out->y) &&
                       _pg_circle_set_radius(f_arr[1], out);
            case 3:
                return pg_DoubleFromObj(f_arr[0], &out->x) &&
                       pg_DoubleFromObj(f_arr[1], &out->y) &&
                       _pg_circle_set_radius(f_arr[2], out);
            default:
                return 0;
        }
    }
    else if (PySequence_Check(obj)) {
        PyObject *tmp = NULL;
        length = PySequence_Length(obj);

        if (length == 3) {
            tmp = PySequence_ITEM(obj, 0);
            if (!pg_DoubleFromObj(tmp, &out->x)) {
                Py_DECREF(tmp);
                return 0;
            }
            Py_DECREF(tmp);

            tmp = PySequence_ITEM(obj, 1);
            if (!pg_DoubleFromObj(tmp, &out->y)) {
                Py_DECREF(tmp);
                return 0;
            }
            Py_DECREF(tmp);

            tmp = PySequence_ITEM(obj, 2);
            if (!_pg_circle_set_radius(tmp, out)) {
                Py_DECREF(tmp);
                return 0;
            }
            Py_DECREF(tmp);

            return 1;
        }
        else if (length == 2) {
            tmp = PySequence_ITEM(obj, 0);
            if (!pg_TwoDoublesFromObj(tmp, &out->x, &out->y)) {
                Py_DECREF(tmp);
                return 0;
            }
            Py_DECREF(tmp);

            tmp = PySequence_ITEM(obj, 1);
            if (!_pg_circle_set_radius(tmp, out)) {
                Py_DECREF(tmp);
                return 0;
            }
            Py_DECREF(tmp);

            return 1;
        }
        else if (length == 1) {
            tmp = PySequence_ITEM(obj, 0);
            if (PyUnicode_Check(obj) || !pgCircle_FromObject(tmp, out)) {
                Py_DECREF(tmp);
                return 0;
            }
            Py_DECREF(tmp);
            return 1;
        }
        else {
            return 0;
        }
    }

    /* Path for objects that have a circle attribute */
    PyObject *circleattr;
    if (!(circleattr = PyObject_GetAttrString(obj, "circle"))) {
        PyErr_Clear();
        return 0;
    }

    if (PyCallable_Check(circleattr)) /*call if it's a method*/
    {
        PyObject *circleresult = PyObject_CallNoArgs(circleattr);
        Py_DECREF(circleattr);
        if (!circleresult) {
            PyErr_Clear();
            return 0;
        }
        circleattr = circleresult;
    }

    if (!pgCircle_FromObject(circleattr, out)) {
        PyErr_Clear();
        Py_DECREF(circleattr);
        return 0;
    }

    Py_DECREF(circleattr);

    return 1;
}

int
pgCircle_FromObjectFastcall(PyObject *const *args, Py_ssize_t nargs,
                            pgCircleBase *out)
{
    switch (nargs) {
        case 1:
            return pgCircle_FromObject(args[0], out);
        case 2:
            return pg_TwoDoublesFromObj(args[0], &out->x, &out->y) &&
                   _pg_circle_set_radius(args[1], out);
        case 3:
            return pg_DoubleFromObj(args[0], &out->x) &&
                   pg_DoubleFromObj(args[1], &out->y) &&
                   _pg_circle_set_radius(args[2], out);
        default:
            return 0;
    }
}

int
pgLine_FromObject(PyObject *obj, pgLineBase *out)
{
    Py_ssize_t length;

    if (pgLine_Check(obj)) {
        *out = pgLine_AsLine(obj);
        return 1;
    }

    /* Paths for sequences */
    if (pgSequenceFast_Check(obj)) {
        length = PySequence_Fast_GET_SIZE(obj);
        PyObject **farray = PySequence_Fast_ITEMS(obj);

        switch (length) {
            case 4:
                return pg_DoubleFromObj(farray[0], &out->ax) &&
                       pg_DoubleFromObj(farray[1], &out->ay) &&
                       pg_DoubleFromObj(farray[2], &out->bx) &&
                       pg_DoubleFromObj(farray[3], &out->by);
            case 2:
                return pg_TwoDoublesFromObj(farray[0], &out->ax, &out->ay) &&
                       pg_TwoDoublesFromObj(farray[1], &out->bx, &out->by);
            case 1: /*looks like an arg?*/
                if (PyUnicode_Check(farray[0]) ||
                    !pgLine_FromObject(farray[0], out)) {
                    return 0;
                }
                return 1;
            default:
                return 0;
        }
    }
    else if (PySequence_Check(obj)) {
        length = PySequence_Length(obj);
        if (length == 4) {
            PyObject *tmp;
            tmp = PySequence_ITEM(obj, 0);
            if (!pg_DoubleFromObj(tmp, &out->ax)) {
                Py_DECREF(tmp);
                return 0;
            }
            Py_DECREF(tmp);
            tmp = PySequence_ITEM(obj, 1);
            if (!pg_DoubleFromObj(tmp, &out->ay)) {
                Py_DECREF(tmp);
                return 0;
            }
            Py_DECREF(tmp);
            tmp = PySequence_ITEM(obj, 2);
            if (!pg_DoubleFromObj(tmp, &out->bx)) {
                Py_DECREF(tmp);
                return 0;
            }
            Py_DECREF(tmp);
            tmp = PySequence_ITEM(obj, 3);
            if (!pg_DoubleFromObj(tmp, &out->by)) {
                Py_DECREF(tmp);
                return 0;
            }
            Py_DECREF(tmp);
            return 1;
        }
        else if (length == 2) {
            PyObject *tmp;
            tmp = PySequence_ITEM(obj, 0);
            if (!pg_TwoDoublesFromObj(tmp, &out->ax, &out->ay)) {
                Py_DECREF(tmp);
                return 0;
            }
            Py_DECREF(tmp);
            tmp = PySequence_ITEM(obj, 1);
            if (!pg_TwoDoublesFromObj(tmp, &out->bx, &out->by)) {
                Py_DECREF(tmp);
                return 0;
            }
            Py_DECREF(tmp);
            return 1;
        }
        else if (PyTuple_Check(obj) && length == 1) /*looks like an arg?*/ {
            PyObject *sub = PySequence_ITEM(obj, 0);
            if (PyUnicode_Check(sub) || !pgLine_FromObject(sub, out)) {
                Py_DECREF(sub);
                return 0;
            }
            Py_DECREF(sub);
            return 1;
        }
        else {
            return 0;
        }
    }

    /* Path for objects that have a line attribute */
    PyObject *lineattr;
    if (!(lineattr = PyObject_GetAttrString(obj, "line"))) {
        PyErr_Clear();
        return 0;
    }

    if (PyCallable_Check(lineattr)) /*call if it's a method*/
    {
        PyObject *lineresult = PyObject_CallNoArgs(lineattr);
        Py_DECREF(lineattr);
        if (!lineresult) {
            PyErr_Clear();
            return 0;
        }
        lineattr = lineresult;
    }

    if (!pgLine_FromObject(lineattr, out)) {
        PyErr_Clear();
        Py_DECREF(lineattr);
        return 0;
    }

    Py_DECREF(lineattr);

    return 1;
}

int
pgLine_FromObjectFastcall(PyObject *const *args, Py_ssize_t nargs,
                          pgLineBase *out)
{
    switch (nargs) {
        case 1:
            return pgLine_FromObject(args[0], out);
        case 2:
            return pg_TwoDoublesFromObj(args[0], &out->ax, &out->ay) &&
                   pg_TwoDoublesFromObj(args[1], &out->bx, &out->by);
        case 4:
            return pg_DoubleFromObj(args[0], &out->ax) &&
                   pg_DoubleFromObj(args[1], &out->ay) &&
                   pg_DoubleFromObj(args[2], &out->bx) &&
                   pg_DoubleFromObj(args[3], &out->by);
        default:
            return 0;
    }
}

static inline int
double_compare(double a, double b)
{
    /* Uses both a fixed epsilon and an adaptive epsilon */
    const double e = 1e-6;
    return fabs(a - b) < e || fabs(a - b) <= e * MAX(fabs(a), fabs(b));
}
