#ifndef PYGAME_CE_GEOMETRY_H
#define PYGAME_CE_GEOMETRY_H

#include "pygame.h"
#include "pgcompat.h"
#include <stddef.h>

typedef struct {
    double x, y, r;
} pgCircleBase;

typedef struct {
    PyObject_HEAD pgCircleBase circle;
    PyObject *weakreflist;
} pgCircleObject;

#define pgCircle_CAST(o) ((pgCircleObject *)(o))
#define pgCircle_AsCircle(o) (pgCircle_CAST(o)->circle)
#define pgCircle_Check(o) ((o)->ob_type == &pgCircle_Type)

typedef struct {
    double ax, ay;
    double bx, by;
} pgLineBase;

typedef struct {
    PyObject_HEAD pgLineBase line;
    PyObject *weakreflist;
} pgLineObject;

#define pgLine_CAST(o) ((pgLineObject *)(o))
#define pgLine_AsLine(o) (pgLine_CAST(o)->line)
#define pgLine_Check(o) ((o)->ob_type == &pgLine_Type)

static PyTypeObject pgCircle_Type;
static PyTypeObject pgLine_Type;

/* Constants */

/* PI */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* 2PI */
#ifndef M_TWOPI
#define M_TWOPI 6.28318530717958647692
#endif

/* PI/180 */
#ifndef M_PI_QUO_180
#define M_PI_QUO_180 0.01745329251994329577
#endif

/* Converts degrees to radians */
static inline double
DEG_TO_RAD(double deg)
{
    return deg * M_PI_QUO_180;
}

#endif  // PYGAME_CE_GEOMETRY_H
