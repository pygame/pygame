#ifndef PYGAME_CE_GEOMETRY_COMMON_H
#define PYGAME_CE_GEOMETRY_COMMON_H

#include "geometry.h"

int
_pg_circle_set_radius(PyObject *value, pgCircleBase *circle);

int
pgCircle_FromObject(PyObject *obj, pgCircleBase *out);

int
pgCircle_FromObjectFastcall(PyObject *const *args, Py_ssize_t nargs,
                            pgCircleBase *out);

int
pgLine_FromObject(PyObject *obj, pgLineBase *out);

int
pgLine_FromObjectFastcall(PyObject *const *args, Py_ssize_t nargs,
                          pgLineBase *out);

static inline int
double_compare(double a, double b);

/* === Collision Functions === */

static inline int
pgCollision_CirclePoint(pgCircleBase *circle, double Cx, double Cy)
{
    double dx = circle->x - Cx;
    double dy = circle->y - Cy;
    return dx * dx + dy * dy <= circle->r * circle->r;
}

static inline int
pgCollision_CircleCircle(pgCircleBase *A, pgCircleBase *B)
{
    double dx, dy;
    double sum_radii;

    dx = A->x - B->x;
    dy = A->y - B->y;
    sum_radii = A->r + B->r;

    return dx * dx + dy * dy <= sum_radii * sum_radii;
}

static inline int
pgCollision_RectCircle(double rx, double ry, double rw, double rh,
                       pgCircleBase *circle)
{
    const double cx = circle->x, cy = circle->y;
    const double r_bottom = ry + rh, r_right = rx + rw;

    const double test_x = (cx < rx) ? rx : ((cx > r_right) ? r_right : cx);
    const double test_y = (cy < ry) ? ry : ((cy > r_bottom) ? r_bottom : cy);

    return pgCollision_CirclePoint(circle, test_x, test_y);
}

static inline int
pgIntersection_CircleCircle(pgCircleBase *A, pgCircleBase *B,
                            double *intersections)
{
    double dx = B->x - A->x;
    double dy = B->y - A->y;
    double d2 = dx * dx + dy * dy;
    double r_sum = A->r + B->r;
    double r_diff = A->r - B->r;
    double r_sum2 = r_sum * r_sum;
    double r_diff2 = r_diff * r_diff;

    if (d2 > r_sum2 || d2 < r_diff2) {
        return 0;
    }

    if (double_compare(d2, 0) && double_compare(A->r, B->r)) {
        return 0;
    }

    double d = sqrt(d2);
    double a = (d2 + A->r * A->r - B->r * B->r) / (2 * d);
    double h = sqrt(A->r * A->r - a * a);

    double xm = A->x + a * (dx / d);
    double ym = A->y + a * (dy / d);

    double xs1 = xm + h * (dy / d);
    double ys1 = ym - h * (dx / d);
    double xs2 = xm - h * (dy / d);
    double ys2 = ym + h * (dx / d);

    if (double_compare(d2, r_sum2) || double_compare(d2, r_diff2)) {
        intersections[0] = xs1;
        intersections[1] = ys1;
        return 1;
    }

    intersections[0] = xs1;
    intersections[1] = ys1;
    intersections[2] = xs2;
    intersections[3] = ys2;
    return 2;
}

#endif  // PYGAME_CE_GEOMETRY_COMMON_H
