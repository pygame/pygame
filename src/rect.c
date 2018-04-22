/*
  pygame - Python Game Library
  Copyright (C) 2000-2001  Pete Shinners

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

  Pete Shinners
  pete@shinners.org
*/

/*
 *  Python Rect Object -- useful 2d rectangle class
 */
#define PYGAMEAPI_RECT_INTERNAL
#include "pygame.h"
#include "doc/rect_doc.h"
#include "structmember.h"
#include "pgcompat.h"
#include <limits.h>

static PyTypeObject pgRect_Type;
#define pgRect_Check(x) ((x)->ob_type == &pgRect_Type)

static int pg_rect_init(pgRectObject *, PyObject *, PyObject *);


/* We store some rect objects which have been allocated already.
   Mostly to work around an old pypy cpyext performance issue.
*/
#ifdef PYPY_VERSION
#define PG_RECT_NUM 49152
const int PG_RECT_FREELIST_MAX = PG_RECT_NUM;
static PyRectObject *pg_rect_freelist[PG_RECT_NUM];
int pg_rect_freelist_num = -1;
#endif


static PyObject *
_pg_rect_subtype_new4(PyTypeObject *type, int x, int y, int w, int h)
{
    pgRectObject *rect = (pgRectObject *)pgRect_Type.tp_new(type, NULL, NULL);

    if (rect) {
        rect->r.x = x;
        rect->r.y = y;
        rect->r.w = w;
        rect->r.h = h;
    }
    return (PyObject*)rect;
}

static PyObject *
pg_rect_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    pgRectObject *self;

#ifdef PYPY_VERSION
    if (pg_rect_freelist_num > -1) {
        self = pg_rect_freelist[pg_rect_freelist_num];
        Py_INCREF(self);
        /* This is so that pypy garbage collector thinks it is a new obj
           TODO: May be a hack. Is a hack.
           See https://github.com/pygame/pygame/issues/430
        */
        ((PyObject *)(self))->ob_pypy_link = 0;
        pg_rect_freelist_num--;
    }
    else {
        self = (PyRectObject *)type->tp_alloc(type, 0);
    }
#else
    self = (PyRectObject *)type->tp_alloc(type, 0);
#endif

    if (self != NULL) {
        self->r.x = self->r.y = 0;
        self->r.w = self->r.h = 0;
        self->weakreflist = NULL;
    }
    return (PyObject *)self;
}

/* object type functions */
static void
pg_rect_dealloc(PyRectObject *self)
{
    if (self->weakreflist != NULL) {
        PyObject_ClearWeakRefs((PyObject *)self);
    }

#ifdef PYPY_VERSION
    if (pg_rect_freelist_num < PG_RECT_FREELIST_MAX) {
        pg_rect_freelist_num++;
        pg_rect_freelist[pg_rect_freelist_num] = self;
    }
    else {
        Py_TYPE(self)->tp_free((PyObject *)self);
    }
#else
    Py_TYPE(self)->tp_free((PyObject *)self);
#endif
}

static GAME_Rect *
pgRect_FromObject(PyObject *obj, GAME_Rect *temp)
{
    int val;
    int length;

    if (pgRect_Check(obj)) {
        return &((pgRectObject*) obj)->r;
    }
    if (PySequence_Check(obj) && (length = PySequence_Length(obj)) > 0) {
        if (length == 4) {
            if (!pg_IntFromObjIndex(obj, 0, &val)) {
                return NULL;
            }
            temp->x = val;
            if (!pg_IntFromObjIndex(obj, 1, &val)) {
                return NULL;
            }
            temp->y = val;
            if (!pg_IntFromObjIndex(obj, 2, &val)) {
                return NULL;
            }
            temp->w = val;
            if (!pg_IntFromObjIndex(obj, 3, &val)) {
                return NULL;
            }
            temp->h = val;
            return temp;
        }
        if (length == 2) {
            PyObject *sub = PySequence_GetItem(obj, 0);
            if (!sub || !PySequence_Check(sub) || PySequence_Length(sub) != 2) {
                Py_XDECREF(sub);
                return NULL;
            }
            if (!pg_IntFromObjIndex(sub, 0, &val)) {
                Py_DECREF(sub);
                return NULL;
            }
            temp->x = val;
            if (!pg_IntFromObjIndex(sub, 1, &val)) {
                Py_DECREF(sub);
                return NULL;
            }
            temp->y = val;
            Py_DECREF(sub);

            sub = PySequence_GetItem(obj, 1);
            if (sub == NULL ||
                !PySequence_Check(sub) ||
                PySequence_Length(sub) != 2) {
                Py_XDECREF(sub);
                return NULL;
            }
            if (!pg_IntFromObjIndex(sub, 0, &val)) {
                Py_DECREF(sub);
                return NULL;
            }
            temp->w = val;
            if (!pg_IntFromObjIndex(sub, 1, &val)) {
                Py_DECREF(sub);
                return NULL;
            }
            temp->h = val;
            Py_DECREF(sub);
            return temp;
        }
        if (PyTuple_Check(obj) && length == 1) /*looks like an arg?*/ {
            PyObject* sub = PyTuple_GET_ITEM(obj, 0);
            if (sub) {
                return pgRect_FromObject(sub, temp);
            }
        }
    }
    if (PyObject_HasAttrString(obj, "rect")) {
        PyObject *rectattr;
        GAME_Rect *returnrect;
        rectattr = PyObject_GetAttrString(obj, "rect");
        if (PyCallable_Check(rectattr)) /*call if it's a method*/
        {
            PyObject *rectresult = PyObject_CallObject(rectattr, NULL);
            Py_DECREF(rectattr);
            if (rectresult == NULL) {
                return NULL;
            }
            rectattr = rectresult;
        }
        returnrect = pgRect_FromObject(rectattr, temp);
        Py_DECREF(rectattr);
        return returnrect;
    }
    return NULL;
}

static PyObject *
pgRect_New(SDL_Rect* r)
{
    return _pg_rect_subtype_new4(&pgRect_Type, r->x, r->y, r->w, r->h);
}

static PyObject *
pgRect_New4(int x, int y, int w, int h)
{
    return _pg_rect_subtype_new4(&pgRect_Type, x, y, w, h);
}

static int
_pg_do_rects_intersect(GAME_Rect *A, GAME_Rect *B)
{
    //A.topleft < B.bottomright &&
    //A.bottomright > B.topleft
    return (A->x < B->x + B->w && A->y < B->y + B->h &&
            A->x + A->w > B->x && A->y + A->h > B->y);
}

static PyObject *
pg_rect_normalize(pgRectObject *self)
{
    if (self->r.w < 0) {
        self->r.x += self->r.w;
        self->r.w = -self->r.w;
    }
    if (self->r.h < 0) {
        self->r.y += self->r.h;
        self->r.h = -self->r.h;
    }

    Py_RETURN_NONE;
}

static PyObject *
pg_rect_move(pgRectObject *self, PyObject* args)
{
    int x, y;

    if (!pg_TwoIntsFromObj(args, &x, &y)) {
        return RAISE(PyExc_TypeError, "argument must contain two numbers");
    }

    return _pg_rect_subtype_new4(Py_TYPE(self),
                                 self->r.x + x, self->r.y + y,
                                 self->r.w, self->r.h);
}

static PyObject *
pg_rect_move_ip(pgRectObject *self, PyObject* args)
{
    int x, y;

    if (!pg_TwoIntsFromObj(args, &x, &y)) {
        return RAISE(PyExc_TypeError, "argument must contain two numbers");
    }

    self->r.x += x;
    self->r.y += y;
    Py_RETURN_NONE;
}

static PyObject *
pg_rect_inflate(pgRectObject *self, PyObject* args)
{
    int x, y;

    if (!pg_TwoIntsFromObj(args, &x, &y)) {
        return RAISE(PyExc_TypeError, "argument must contain two numbers");
    }

    return _pg_rect_subtype_new4(Py_TYPE(self),
                                 self->r.x - x / 2, self->r.y - y / 2,
                                 self->r.w + x, self->r.h + y);
}

static PyObject *
pg_rect_inflate_ip(pgRectObject *self, PyObject* args)
{
    int x, y;

    if (!pg_TwoIntsFromObj(args, &x, &y)) {
        return RAISE(PyExc_TypeError, "argument must contain two numbers");
    }
    self->r.x -= x / 2;
    self->r.y -= y / 2;
    self->r.w += x;
    self->r.h += y;
    Py_RETURN_NONE;
}

static PyObject *
pg_rect_union(pgRectObject *self, PyObject* args)
{
    GAME_Rect *argrect, temp;
    int x, y, w, h;

    if (!(argrect = pgRect_FromObject(args, &temp))) {
        return RAISE(PyExc_TypeError, "Argument must be rect style object");
    }
    x = MIN(self->r.x, argrect->x);
    y = MIN(self->r.y, argrect->y);
    w = MAX(self->r.x + self->r.w, argrect->x + argrect->w) - x;
    h = MAX(self->r.y + self->r.h, argrect->y + argrect->h) - y;
    return _pg_rect_subtype_new4(Py_TYPE(self), x, y, w, h);
}

static PyObject *
pg_rect_union_ip(pgRectObject *self, PyObject* args)
{
    GAME_Rect *argrect, temp;
    int x, y, w, h;

    if(!(argrect = pgRect_FromObject(args, &temp)))
        return RAISE(PyExc_TypeError, "Argument must be rect style object");

    x = MIN(self->r.x, argrect->x);
    y = MIN(self->r.y, argrect->y);
    w = MAX(self->r.x + self->r.w, argrect->x + argrect->w) - x;
    h = MAX(self->r.y + self->r.h, argrect->y + argrect->h) - y;
    self->r.x = x;
    self->r.y = y;
    self->r.w = w;
    self->r.h = h;
    Py_RETURN_NONE;
}

static PyObject *
pg_rect_unionall(pgRectObject *self, PyObject* args)
{
    GAME_Rect *argrect, temp;
    int loop, size;
    PyObject* list, *obj;
    int t, l, b, r;

    if (!PyArg_ParseTuple(args, "O", &list)) {
        return NULL;
    }
    if (!PySequence_Check(list)) {
        return RAISE(PyExc_TypeError,
                     "Argument must be a sequence of rectstyle objects.");
    }

    l = self->r.x;
    t = self->r.y;
    r = self->r.x + self->r.w;
    b = self->r.y + self->r.h;
    size = PySequence_Length(list); /*warning, size could be -1 on error?*/
    if (size < 1) {
        if (size < 0) {
            /*Error.*/
            return NULL;
        }
        /*Empty list: nothing to be done.*/
        return _pg_rect_subtype_new4(Py_TYPE(self), l, t, r-l, b-t);
    }

    for (loop = 0; loop < size; ++loop)
    {
        obj = PySequence_GetItem(list, loop);
        if(!obj || !(argrect = pgRect_FromObject(obj, &temp)))
        {
            RAISE(PyExc_TypeError,
                  "Argument must be a sequence of rectstyle objects.");
            Py_XDECREF(obj);
            break;
        }
        l = MIN(l, argrect->x);
        t = MIN(t, argrect->y);
        r = MAX(r, argrect->x + argrect->w);
        b = MAX(b, argrect->y + argrect->h);
        Py_DECREF(obj);
    }
    return _pg_rect_subtype_new4(Py_TYPE(self), l, t, r-l, b-t);
}

static PyObject *
pg_rect_unionall_ip(pgRectObject *self, PyObject* args)
{
    GAME_Rect *argrect, temp;
    int loop, size;
    PyObject* list, *obj;
    int t, l, b, r;

    if (!PyArg_ParseTuple(args, "O", &list)) {
        return NULL;
    }
    if (!PySequence_Check(list)) {
        return RAISE(PyExc_TypeError,
                     "Argument must be a sequence of rectstyle objects.");
    }

    l = self->r.x;
    t = self->r.y;
    r = self->r.x + self->r.w;
    b = self->r.y + self->r.h;

    size = PySequence_Length(list); /*warning, size could be -1 on error?*/
    if (size < 1) {
        if (size < 0) {
            /*Error.*/
            return NULL;
        }
        /*Empty list: nothing to be done.*/
        Py_RETURN_NONE;
    }

    for (loop = 0; loop < size; ++loop)
    {
        obj = PySequence_GetItem(list, loop);
        if (!obj || !(argrect = pgRect_FromObject(obj, &temp))) {
            RAISE(PyExc_TypeError,
                  "Argument must be a sequence of rectstyle objects.");
            Py_XDECREF(obj);
            break;
        }
        l = MIN(l, argrect->x);
        t = MIN(t, argrect->y);
        r = MAX(r, argrect->x + argrect->w);
        b = MAX(b, argrect->y + argrect->h);
        Py_DECREF(obj);
    }

    self->r.x = l;
    self->r.y = t;
    self->r.w = r - l;
    self->r.h = b - t;
    Py_RETURN_NONE;
}

static PyObject *
pg_rect_collidepoint(pgRectObject *self, PyObject* args)
{
    int x, y;
    int inside;

    if (!pg_TwoIntsFromObj(args, &x, &y)) {
        return RAISE(PyExc_TypeError, "argument must contain two numbers");
    }

    inside = x >= self->r.x && x < self->r.x + self->r.w &&
        y >= self->r.y && y < self->r.y + self->r.h;

    return PyInt_FromLong(inside);
}

static PyObject *
pg_rect_colliderect(pgRectObject *self, PyObject* args)
{
    GAME_Rect *argrect, temp;

    if (!(argrect = pgRect_FromObject(args, &temp))) {
        return RAISE(PyExc_TypeError, "Argument must be rect style object");
    }
    return PyInt_FromLong(_pg_do_rects_intersect(&self->r, argrect));
}

static PyObject *
pg_rect_collidelist(pgRectObject *self, PyObject* args)
{
    GAME_Rect *argrect, temp;
    int loop, size;
    PyObject* list, *obj;
    PyObject* ret = NULL;

    if (!PyArg_ParseTuple(args, "O", &list)) {
        return NULL;
    }

    if (!PySequence_Check(list)) {
        return RAISE(PyExc_TypeError,
                     "Argument must be a sequence of rectstyle objects.");
    }

    size = PySequence_Length(list); /*warning, size could be -1 on error?*/
    for (loop = 0; loop < size; ++loop)
    {
        obj = PySequence_GetItem(list, loop);
        if (!obj || !(argrect = pgRect_FromObject(obj, &temp))) {
            RAISE(PyExc_TypeError,
                  "Argument must be a sequence of rectstyle objects.");
            Py_XDECREF(obj);
            break;
        }
        if (_pg_do_rects_intersect(&self->r, argrect)) {
            ret = PyInt_FromLong(loop);
            Py_DECREF(obj);
            break;
        }
        Py_DECREF(obj);
    }
    if (loop == size) {
        ret = PyInt_FromLong(-1);
    }

    return ret;
}

static PyObject *
pg_rect_collidelistall(pgRectObject *self, PyObject* args)
{
    GAME_Rect *argrect, temp;
    int loop, size;
    PyObject* list, *obj;
    PyObject* ret = NULL;

    if (!PyArg_ParseTuple(args, "O", &list)) {
        return NULL;
    }

    if (!PySequence_Check(list)) {
        return RAISE(PyExc_TypeError,
                     "Argument must be a sequence of rectstyle objects.");
    }

    ret = PyList_New(0);
    if (!ret) {
        return NULL;
    }

    size = PySequence_Length(list); /*warning, size could be -1?*/
    for (loop = 0; loop < size; ++loop)
    {
        obj = PySequence_GetItem(list, loop);

        if(!obj || !(argrect = pgRect_FromObject(obj, &temp)))
        {
            Py_XDECREF(obj);
            Py_DECREF(ret);
            return RAISE(PyExc_TypeError,
                         "Argument must be a sequence of rectstyle objects.");
        }

        if (_pg_do_rects_intersect(&self->r, argrect)) {
            PyObject* num = PyInt_FromLong(loop);
            if (!num) {
                Py_DECREF(obj);
                return NULL;
            }
            PyList_Append(ret, num);
            Py_DECREF(num);
        }
        Py_DECREF(obj);
    }

    return ret;
}

static PyObject *
pg_rect_collidedict(pgRectObject* self, PyObject* args)
{
    GAME_Rect *argrect, temp;
    Py_ssize_t loop=0;
    Py_ssize_t values=0;
    PyObject* dict, *key, *val;
    PyObject* ret = NULL;

    if (!PyArg_ParseTuple(args, "O|i", &dict, &values)) {
        return NULL;
    }
    if (!PyDict_Check(dict)) {
        return RAISE(PyExc_TypeError,
                     "Argument must be a dict with rectstyle keys.");
    }

    while (PyDict_Next(dict, &loop, &key, &val))
    {
        if(values) {
            if (!(argrect = pgRect_FromObject(val, &temp))) {
                RAISE(PyExc_TypeError,
                      "Argument must be a dict with rectstyle values.");
                break;
            }
        } else {
            if (!(argrect = pgRect_FromObject(key, &temp))) {
                RAISE(PyExc_TypeError,
                      "Argument must be a dict with rectstyle keys.");
                break;
            }
        }


        if (_pg_do_rects_intersect(&self->r, argrect)) {
            ret = Py_BuildValue("(OO)", key, val);
            break;
        }
    }

    if (!ret) {
        Py_RETURN_NONE;
    }
    return ret;
}

static PyObject *
pg_rect_collidedictall(pgRectObject *self, PyObject* args)
{
    GAME_Rect *argrect, temp;
    Py_ssize_t loop=0;
    /* should we use values or keys? */
    Py_ssize_t values=0;

    PyObject* dict, *key, *val;
    PyObject* ret = NULL;

    if (!PyArg_ParseTuple(args, "O|i", &dict, &values)) {
        return NULL;
    }
    if (!PyDict_Check(dict)) {
        return RAISE(PyExc_TypeError,
                     "Argument must be a dict with rectstyle keys.");
    }

    ret = PyList_New(0);
    if(!ret)
        return NULL;

    while (PyDict_Next(dict, &loop, &key, &val))
    {
        if (values) {
            if (!(argrect = pgRect_FromObject(val, &temp))) {
                Py_DECREF(ret);
                return RAISE(PyExc_TypeError,
                             "Argument must be a dict with rectstyle values.");
            }
        } else {
            if (!(argrect = pgRect_FromObject(key, &temp))) {
                Py_DECREF(ret);
                return RAISE(PyExc_TypeError,
                             "Argument must be a dict with rectstyle keys.");
            }
        }

        if (_pg_do_rects_intersect(&self->r, argrect)) {
            PyObject* num = Py_BuildValue("(OO)", key, val);
            if(!num)
                return NULL;
            PyList_Append(ret, num);
            Py_DECREF(num);
        }
    }

    return ret;
}

static PyObject *
pg_rect_clip(pgRectObject *self, PyObject* args)
{
    GAME_Rect *A, *B, temp;
    int x, y, w, h;

    A = &self->r;
    if (!(B = pgRect_FromObject(args, &temp))) {
        return RAISE(PyExc_TypeError, "Argument must be rect style object");
    }

    /* Left */
    if ((A->x >= B->x) && (A->x < (B->x + B->w))) {
        x = A->x;
    }
    else if ((B->x >= A->x) && (B->x < (A->x + A->w)))
        x = B->x;
    else
        goto nointersect;

    /* Right */
    if (((A->x + A->w) > B->x) && ((A->x + A->w) <= (B->x + B->w))) {
        w = (A->x + A->w) - x;
    }
    else if (((B->x + B->w) > A->x) && ((B->x + B->w) <= (A->x + A->w)))
        w = (B->x + B->w) - x;
    else
        goto nointersect;

    /* Top */
    if ((A->y >= B->y) && (A->y < (B->y + B->h))) {
        y = A->y;
    }
    else if ((B->y >= A->y) && (B->y < (A->y + A->h)))
        y = B->y;
    else
        goto nointersect;

    /* Bottom */
    if (((A->y + A->h) > B->y) && ((A->y + A->h) <= (B->y + B->h))) {
        h = (A->y + A->h) - y;
    }
    else if (((B->y + B->h) > A->y) && ((B->y + B->h) <= (A->y + A->h)))
        h = (B->y + B->h) - y;
    else
        goto nointersect;

    return _pg_rect_subtype_new4(Py_TYPE(self), x, y, w, h);

nointersect:
    return _pg_rect_subtype_new4(Py_TYPE(self), A->x, A->y, 0, 0);
}

static PyObject *
pg_rect_contains(pgRectObject *self, PyObject* args)
{
    int contained;
    GAME_Rect *argrect, temp;

    if (!(argrect = pgRect_FromObject(args, &temp))) {
        return RAISE(PyExc_TypeError, "Argument must be rect style object");
    }

    contained = (self->r.x <= argrect->x) && (self->r.y <= argrect->y) &&
        (self->r.x + self->r.w >= argrect->x + argrect->w) &&
        (self->r.y + self->r.h >= argrect->y + argrect->h) &&
        (self->r.x + self->r.w > argrect->x) &&
        (self->r.y + self->r.h > argrect->y);

    return PyInt_FromLong(contained);
}

static PyObject *
pg_rect_clamp(pgRectObject *self, PyObject* args)
{
    GAME_Rect *argrect, temp;
    int x, y;

    if (!(argrect = pgRect_FromObject(args, &temp))) {
        return RAISE(PyExc_TypeError, "Argument must be rect style object");
    }

    if (self->r.w >= argrect->w) {
        x = argrect->x + argrect->w / 2 - self->r.w / 2;
    }
    else if (self->r.x < argrect->x)
        x = argrect->x;
    else if (self->r.x + self->r.w > argrect->x + argrect->w)
        x = argrect->x + argrect->w - self->r.w;
    else
        x = self->r.x;

    if (self->r.h >= argrect->h) {
        y = argrect->y + argrect->h / 2 - self->r.h / 2;
    }
    else if (self->r.y < argrect->y)
        y = argrect->y;
    else if (self->r.y + self->r.h > argrect->y + argrect->h)
        y = argrect->y + argrect->h - self->r.h;
    else
        y = self->r.y;

    return _pg_rect_subtype_new4(Py_TYPE(self), x, y, self->r.w, self->r.h);
}

static PyObject *
pg_rect_fit(pgRectObject *self, PyObject* args)
{
    GAME_Rect *argrect, temp;
    int w, h, x, y;
    float xratio, yratio, maxratio;

    if (!(argrect = pgRect_FromObject(args, &temp))) {
        return RAISE(PyExc_TypeError, "Argument must be rect style object");
    }

    xratio = (float) self->r.w / (float) argrect->w;
    yratio = (float) self->r.h / (float) argrect->h;
    maxratio = (xratio > yratio) ? xratio : yratio;

    w = (int) (self->r.w / maxratio);
    h = (int) (self->r.h / maxratio);

    x = argrect->x + (argrect->w - w)/2;
    y = argrect->y + (argrect->h - h)/2;

    return _pg_rect_subtype_new4(Py_TYPE(self), x, y, w, h);
}

static PyObject *
pg_rect_clamp_ip(pgRectObject *self, PyObject* args)
{
    GAME_Rect *argrect, temp;
    int x, y;

    if (!(argrect = pgRect_FromObject(args, &temp))) {
        return RAISE(PyExc_TypeError, "Argument must be rect style object");
    }

    if (self->r.w >= argrect->w) {
        x = argrect->x + argrect->w / 2 - self->r.w / 2;
    }
    else if (self->r.x < argrect->x)
        x = argrect->x;
    else if (self->r.x + self->r.w > argrect->x + argrect->w)
        x = argrect->x + argrect->w - self->r.w;
    else
        x = self->r.x;

    if (self->r.h >= argrect->h) {
        y = argrect->y + argrect->h / 2 - self->r.h / 2;
    }
    else if (self->r.y < argrect->y)
        y = argrect->y;
    else if (self->r.y + self->r.h > argrect->y + argrect->h)
        y = argrect->y + argrect->h - self->r.h;
    else
        y = self->r.y;

    self->r.x = x;
    self->r.y = y;
    Py_RETURN_NONE;
}

/* for pickling */
static PyObject *
pg_rect_reduce(pgRectObject *self)
{
    return Py_BuildValue("(O(iiii))", Py_TYPE(self),
                          (int)self->r.x, (int)self->r.y,
                          (int)self->r.w, (int)self->r.h);
}

/* for copy module */
static PyObject *
pg_rect_copy(pgRectObject *self)
{
    return _pg_rect_subtype_new4(Py_TYPE(self),
                                 self->r.x, self->r.y, self->r.w, self->r.h);
}

static struct PyMethodDef pg_rect_methods[] =
{
    { "normalize", (PyCFunction)pg_rect_normalize, METH_NOARGS,
      DOC_RECTNORMALIZE },
    { "clip", (PyCFunction)pg_rect_clip, METH_VARARGS, DOC_RECTCLIP},
    { "clamp", (PyCFunction)pg_rect_clamp, METH_VARARGS, DOC_RECTCLAMP},
    { "clamp_ip", (PyCFunction)pg_rect_clamp_ip, METH_VARARGS, DOC_RECTCLAMPIP},
    { "copy", (PyCFunction) pg_rect_copy, METH_NOARGS, DOC_RECTCOPY},
    { "fit", (PyCFunction)pg_rect_fit, METH_VARARGS, DOC_RECTFIT},
    { "move", (PyCFunction)pg_rect_move, METH_VARARGS, DOC_RECTMOVE},
    { "inflate",  (PyCFunction)pg_rect_inflate, METH_VARARGS, DOC_RECTINFLATE},
    { "union",  (PyCFunction)pg_rect_union, METH_VARARGS, DOC_RECTUNION},
    { "unionall",  (PyCFunction)pg_rect_unionall, METH_VARARGS, DOC_RECTUNIONALL},
    { "move_ip",  (PyCFunction)pg_rect_move_ip, METH_VARARGS, DOC_RECTMOVEIP},
    { "inflate_ip", (PyCFunction)pg_rect_inflate_ip, METH_VARARGS, DOC_RECTINFLATEIP},
    { "union_ip", (PyCFunction)pg_rect_union_ip, METH_VARARGS, DOC_RECTUNIONIP},
    { "unionall_ip", (PyCFunction)pg_rect_unionall_ip, METH_VARARGS, DOC_RECTUNIONALLIP},
    { "collidepoint", (PyCFunction)pg_rect_collidepoint, METH_VARARGS, DOC_RECTCOLLIDEPOINT},
    { "colliderect", (PyCFunction)pg_rect_colliderect, METH_VARARGS, DOC_RECTCOLLIDERECT},
    { "collidelist", (PyCFunction)pg_rect_collidelist, METH_VARARGS, DOC_RECTCOLLIDELIST},
    { "collidelistall", (PyCFunction)pg_rect_collidelistall, METH_VARARGS,
      DOC_RECTCOLLIDELISTALL},
    { "collidedict", (PyCFunction)pg_rect_collidedict, METH_VARARGS, DOC_RECTCOLLIDEDICT},
    { "collidedictall", (PyCFunction)pg_rect_collidedictall, METH_VARARGS,
      DOC_RECTCOLLIDEDICTALL},
    { "contains", (PyCFunction)pg_rect_contains, METH_VARARGS, DOC_RECTCONTAINS},
    { "__reduce__", (PyCFunction)pg_rect_reduce, METH_NOARGS, NULL},
    { "__copy__", (PyCFunction)pg_rect_copy, METH_NOARGS, NULL},
    { NULL, NULL, 0, NULL }
};

/* sequence functions */

static Py_ssize_t
pg_rect_length(PyObject *_self)
{
    return 4;
}

static PyObject*
pg_rect_item(PyRectObject *self, Py_ssize_t i)
{
    int* data = (int*)&self->r;

    if (i < 0 || i > 3) {
        if (i > -5 && i < 0) {
            i += 4;
        }
        else {
            return RAISE (PyExc_IndexError, "Invalid rect Index");
        }
    }
    return PyInt_FromLong(data[i]);
}

static int
pg_rect_ass_item(PyRectObject *self, Py_ssize_t i, PyObject *v)
{
    int val;
    int* data = (int*)&self->r;

    if (i < 0 || i > 3) {
        if (i > -5 && i < 0) {
            i += 4;
        }
        else {
            RAISE (PyExc_IndexError, "Invalid rect Index");
            return -1;
        }
    }
    if (!pg_IntFromObj(v, &val)) {
        RAISE(PyExc_TypeError, "Must assign numeric values");
        return -1;
    }
    data[i] = val;
    return 0;
}

static PySequenceMethods pg_rect_as_sequence =
{
    pg_rect_length,                     /*length*/
    NULL,                               /*concat*/
    NULL,                               /*repeat*/
    (ssizeargfunc)pg_rect_item,         /*item*/
    NULL,                               /*slice*/
    (ssizeobjargproc)pg_rect_ass_item,  /*ass_item*/
    NULL,                               /*ass_slice*/
};

static PyObject *
pg_rect_subscript(PyRectObject *self, PyObject *op)
{
    int *data = (int *)&self->r;

    if (PyIndex_Check(op)) {
        PyObject *index = PyNumber_Index(op);
        Py_ssize_t i;

        if (index == NULL) {
            return NULL;
        }
        i = PyNumber_AsSsize_t(index, NULL);
        Py_DECREF(index);
        return pg_rect_item(self, i);
    }
    else if (op == Py_Ellipsis) {
        return Py_BuildValue("[iiii]", data[0], data[1], data[2], data[3]);
    }
    else if (PySlice_Check(op)) {
        PyObject *slice;
        Py_ssize_t start;
        Py_ssize_t stop;
        Py_ssize_t step;
        Py_ssize_t slicelen;
        Py_ssize_t i;
        PyObject *n;

#if PY_VERSION_HEX >= 0x03020000
        if (PySlice_GetIndicesEx(op, 4, &start, &stop, &step, &slicelen)) {
            return NULL;
        }
#else
        if (PySlice_GetIndicesEx((PySliceObject *)op, 4,
                                 &start, &stop, &step, &slicelen)) {
            return NULL;
        }
#endif
        slice = PyList_New(slicelen);
        if (slice == NULL) {
            return NULL;
        }
        for (i = 0; i < slicelen; ++i) {
            n = PyInt_FromSsize_t(data[start + (step * i)]);
            if (n == NULL) {
                Py_DECREF(slice);
                return NULL;
            }
            PyList_SET_ITEM(slice, i, n);
        }
        return slice;
    }

    RAISE(PyExc_TypeError, "Invalid Rect slice");
    return NULL;
}

static int
pg_rect_ass_subscript(PyRectObject *self, PyObject *op, PyObject *value)
{
    if (PyIndex_Check(op)) {
        PyObject *index;
        Py_ssize_t i;

        index = PyNumber_Index(op);
        if (index == NULL) {
            return -1;
        }
        i = PyNumber_AsSsize_t(index, NULL);
        Py_DECREF(index);
        return pg_rect_ass_item(self, i, value);
    }
    else if (op == Py_Ellipsis) {
        int val;

        if (IntFromObj(value, &val)) {
            self->r.x = val;
            self->r.y = val;
            self->r.w = val;
            self->r.h = val;
        }
        else if (PyObject_IsInstance(value, (PyObject *)&PyRect_Type)) {
            PyRectObject *rect = (PyRectObject *)value;

            self->r.x = rect->r.x;
            self->r.y = rect->r.y;
            self->r.w = rect->r.w;
            self->r.h = rect->r.h;
        }
        else if (PySequence_Check(value)) {
            PyObject *item;
            int values[4];
            Py_ssize_t i;

            if (PySequence_Size(value) != 4) {
                RAISE(PyExc_TypeError, "Expect a length 4 sequence");
                return -1;
            }
            for (i = 0; i < 4; ++i) {
                item = PySequence_ITEM(value, i);
                if (!IntFromObj(item, values + i)) {
                    PyErr_Format(PyExc_TypeError,
                                 "Expected an integer between %d and %d",
                                 INT_MIN, INT_MAX);
                }
            }
            self->r.x = values[0];
            self->r.y = values[1];
            self->r.w = values[2];
            self->r.h = values[3];
        }
        else {
            RAISE(PyExc_TypeError, "Expected an integer or sequence");
            return -1;
        }
    }
    else if (PySlice_Check(op)) {
        int *data = (int *)&self->r;
        Py_ssize_t start;
        Py_ssize_t stop;
        Py_ssize_t step;
        Py_ssize_t slicelen;
        int val;
        Py_ssize_t i;

#if PY_VERSION_HEX >= 0x03020000
        if (PySlice_GetIndicesEx(op, 4, &start, &stop, &step, &slicelen)) {
            return -1;
        }
#else
        if (PySlice_GetIndicesEx((PySliceObject *)op, 4,
                                  &start, &stop, &step, &slicelen)) {
            return -1;
        }
#endif
        if (IntFromObj(value, &val)) {
            for (i = 0; i < slicelen; ++i) {
                data[start + step * i] = val;
            }
        }
        else if (PySequence_Check(value)) {
            PyObject *item;
            int values[4];
            Py_ssize_t i;
            Py_ssize_t size = PySequence_Size(value);

            if (size != slicelen) {
                PyErr_Format(PyExc_TypeError,
                             "Expected a length %zd sequence",
                             slicelen);
                return -1;
            }
            for (i = 0; i < slicelen; ++i) {
                item = PySequence_ITEM(value, i);
                if (!IntFromObj(item, values + i)) {
                    PyErr_Format(PyExc_TypeError,
                                 "Expected an integer between %d and %d",
                                 INT_MIN, INT_MAX);
                }
            }
            for (i = 0; i < slicelen; ++i) {
                data[start + step * i] = values[i];
            }
        }
        else {
            RAISE(PyExc_TypeError, "Expected an integer or sequence");
            return -1;
        }
    }
    else {
        RAISE(PyExc_TypeError, "Invalid Rect slice");
        return -1;
    }
    return 0;
}

static PyMappingMethods pg_rect_as_mapping =
{
    (lenfunc)pg_rect_length,             /*mp_length*/
    (binaryfunc)pg_rect_subscript,       /*mp_subscript*/
    (objobjargproc)pg_rect_ass_subscript /*mp_ass_subscript*/
};

/* numeric functions */
static int
pg_rect_bool(pgRectObject *self)
{
    return self->r.w != 0 && self->r.h != 0;
}

#if !PY3
static int
pg_rect_coerce(PyObject** o1, PyObject** o2)
{
    PyObject* new1;
    PyObject* new2;
    GAME_Rect* r, temp;

    if (pgRect_Check(*o1)) {
        new1 = *o1;
        Py_INCREF(new1);
    }
    else if ((r = pgRect_FromObject(*o1, &temp)))
        new1 = pgRect_New4(r->x, r->y, r->w, r->h);
    else
        return 1;

    if (pgRect_Check(*o2)) {
        new2 = *o2;
        Py_INCREF(new2);
    }
    else if ((r = pgRect_FromObject(*o2, &temp)))
        new2 = pgRect_New4(r->x, r->y, r->w, r->h);
    else
    {
        Py_DECREF(new1);
        return 1;
    }

    *o1 = new1;
    *o2 = new2;
    return 0;
}
#endif

static PyNumberMethods pg_rect_as_number =
{
    (binaryfunc)NULL,         /*add*/
    (binaryfunc)NULL,         /*subtract*/
    (binaryfunc)NULL,         /*multiply*/
#if !PY3
    (binaryfunc)NULL,         /*divide*/
#endif
    (binaryfunc)NULL,         /*remainder*/
    (binaryfunc)NULL,         /*divmod*/
    (ternaryfunc)NULL,        /*power*/
    (unaryfunc)NULL,          /*negative*/
    (unaryfunc)NULL,          /*pos*/
    (unaryfunc)NULL,          /*abs*/
    (inquiry)pg_rect_bool,    /*nonzero / bool*/
    (unaryfunc)NULL,          /*invert*/
    (binaryfunc)NULL,         /*lshift*/
    (binaryfunc)NULL,         /*rshift*/
    (binaryfunc)NULL,         /*and*/
    (binaryfunc)NULL,         /*xor*/
    (binaryfunc)NULL,         /*or*/
#if !PY3
    (coercion)pg_rect_coerce, /*coerce*/
#endif
    (unaryfunc)NULL,          /*int*/
#if !PY3
    (unaryfunc)NULL,          /*long*/
#endif
    (unaryfunc)NULL,          /*float*/
};

static PyObject *
pg_rect_repr(pgRectObject *self)
{
    char string[256];

    sprintf(string, "<rect(%d, %d, %d, %d)>", self->r.x, self->r.y,
             self->r.w, self->r.h);
    return Text_FromUTF8(string);
}

static PyObject *
pg_rect_str(pgRectObject *self)
{
    return pg_rect_repr(self);
}

static PyObject *
pg_rect_richcompare(PyObject *o1, PyObject *o2, int opid)
{
    GAME_Rect *o1rect, *o2rect, temp1, temp2;
    int cmp;

    o1rect = pgRect_FromObject(o1, &temp1);
    if (!o1rect) {
        goto Unimplemented;
    }
    o2rect = pgRect_FromObject(o2, &temp2);
    if (!o2rect) {
        goto Unimplemented;
    }

    if (o1rect->x != o2rect->x) {
        cmp = o1rect->x < o2rect->x ? -1 : 1;
    }
    else if (o1rect->y != o2rect->y) {
        cmp = o1rect->y < o2rect->y ? -1 : 1;
    }
    else if (o1rect->w != o2rect->w) {
        cmp = o1rect->w < o2rect->w ? -1 : 1;
    }
    else if (o1rect->h != o2rect->h) {
        cmp = o1rect->h < o2rect->h ? -1 : 1;
    }
    else {
        cmp = 0;
    }

    switch (opid)
    {
    case Py_LT:
        return PyBool_FromLong(cmp < 0);
    case Py_LE:
        return PyBool_FromLong(cmp <= 0);
    case Py_EQ:
        return PyBool_FromLong(cmp == 0);
    case Py_NE:
        return PyBool_FromLong(cmp != 0);
    case Py_GT:
        return PyBool_FromLong(cmp > 0);
    case Py_GE:
        return PyBool_FromLong(cmp >= 0);
    default:
        break;
    }

Unimplemented:
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}

/*width*/
static PyObject *
pg_rect_getwidth(pgRectObject *self, void *closure)
{
    return PyInt_FromLong(self->r.w);
}

static int
pg_rect_setwidth(pgRectObject *self, PyObject* value, void *closure)
{
    int val1;

    if (!pg_IntFromObj(value, &val1)) {
        return -1;
    }
    self->r.w = val1;
    return 0;
}

/*height*/
static PyObject *
pg_rect_getheight(pgRectObject *self, void *closure)
{
    return PyInt_FromLong(self->r.h);
}

static int
pg_rect_setheight(pgRectObject *self, PyObject* value, void *closure)
{
    int val1;

    if (!pg_IntFromObj(value, &val1)) {
        RAISE(PyExc_TypeError, "invalid rect assignment");
        return -1;
    }
    self->r.h = val1;
    return 0;
}

/*top*/
static PyObject *
pg_rect_gettop(pgRectObject *self, void *closure)
{
    return PyInt_FromLong(self->r.y);
}

static int
pg_rect_settop(pgRectObject *self, PyObject* value, void *closure)
{
    int val1;

    if (!pg_IntFromObj(value, &val1)) {
        RAISE(PyExc_TypeError, "invalid rect assignment");
        return -1;
    }
    self->r.y = val1;
    return 0;
}

/*left*/
static PyObject *
pg_rect_getleft(pgRectObject *self, void *closure)
{
    return PyInt_FromLong(self->r.x);
}

static int
pg_rect_setleft(pgRectObject *self, PyObject* value, void *closure)
{
    int val1;

    if (!pg_IntFromObj(value, &val1)) {
        RAISE(PyExc_TypeError, "invalid rect assignment");
        return -1;
    }
    self->r.x = val1;
    return 0;
}

/*right*/
static PyObject *
pg_rect_getright(pgRectObject *self, void *closure)
{
    return PyInt_FromLong(self->r.x + self->r.w);
}

static int
pg_rect_setright(pgRectObject *self, PyObject* value, void *closure)
{
    int val1;

    if (!pg_IntFromObj(value, &val1)) {
        RAISE(PyExc_TypeError, "invalid rect assignment");
        return -1;
    }
    self->r.x = val1-self->r.w;
    return 0;
}

/*bottom*/
static PyObject *
pg_rect_getbottom(pgRectObject *self, void *closure)
{
    return PyInt_FromLong(self->r.y + self->r.h);
}

static int
pg_rect_setbottom(pgRectObject *self, PyObject* value, void *closure)
{
    int val1;

    if (!pg_IntFromObj(value, &val1)) {
        RAISE(PyExc_TypeError, "invalid rect assignment");
        return -1;
    }
    self->r.y = val1-self->r.h;
    return 0;
}

/*centerx*/
static PyObject *
pg_rect_getcenterx(pgRectObject *self, void *closure)
{
    return PyInt_FromLong(self->r.x + (self->r.w >> 1));
}

static int
pg_rect_setcenterx(pgRectObject *self, PyObject* value, void *closure)
{
    int val1;

    if (!pg_IntFromObj(value, &val1)) {
        RAISE(PyExc_TypeError, "invalid rect assignment");
        return -1;
    }
    self->r.x = val1 - (self->r.w >> 1);
    return 0;
}

/*centery*/
static PyObject *
pg_rect_getcentery(pgRectObject *self, void *closure)
{
    return PyInt_FromLong(self->r.y + (self->r.h >> 1));
}

static int
pg_rect_setcentery(pgRectObject *self, PyObject* value, void *closure)
{
    int val1;

    if (!pg_IntFromObj(value, &val1)) {
        RAISE(PyExc_TypeError, "invalid rect assignment");
        return -1;
    }
    self->r.y = val1 - (self->r.h >> 1);
    return 0;
}

/*topleft*/
static PyObject *
pg_rect_gettopleft(pgRectObject *self, void *closure)
{
    return Py_BuildValue("(ii)", self->r.x, self->r.y);
}

static int
pg_rect_settopleft(pgRectObject *self, PyObject* value, void *closure)
{
    int val1, val2;

    if (!pg_TwoIntsFromObj(value, &val1, &val2)) {
        RAISE(PyExc_TypeError, "invalid rect assignment");
        return -1;
    }
    self->r.x = val1;
    self->r.y = val2;
    return 0;
}

/*topright*/
static PyObject *
pg_rect_gettopright(pgRectObject *self, void *closure)
{
    return Py_BuildValue("(ii)", self->r.x+self->r.w, self->r.y);
}

static int
pg_rect_settopright(pgRectObject *self, PyObject* value, void *closure)
{
    int val1, val2;

    if (!pg_TwoIntsFromObj(value, &val1, &val2)) {
        RAISE(PyExc_TypeError, "invalid rect assignment");
        return -1;
    }
    self->r.x = val1-self->r.w;
    self->r.y = val2;
    return 0;
}

/*bottomleft*/
static PyObject *
pg_rect_getbottomleft(pgRectObject *self, void *closure)
{
    return Py_BuildValue("(ii)", self->r.x, self->r.y+self->r.h);
}

static int
pg_rect_setbottomleft(pgRectObject *self, PyObject* value, void *closure)
{
    int val1, val2;

    if (!pg_TwoIntsFromObj(value, &val1, &val2)) {
        RAISE(PyExc_TypeError, "invalid rect assignment");
        return -1;
    }
    self->r.x = val1;
    self->r.y = val2-self->r.h;
    return 0;
}

/*bottomright*/
static PyObject *
pg_rect_getbottomright(pgRectObject *self, void *closure)
{
    return Py_BuildValue("(ii)", self->r.x+self->r.w, self->r.y+self->r.h);
}

static int
pg_rect_setbottomright(pgRectObject *self, PyObject* value, void *closure)
{
    int val1, val2;

    if (!pg_TwoIntsFromObj(value, &val1, &val2)) {
        RAISE(PyExc_TypeError, "invalid rect assignment");
        return -1;
    }
    self->r.x = val1-self->r.w;
    self->r.y = val2-self->r.h;
    return 0;
}

/*midtop*/
static PyObject *
pg_rect_getmidtop(pgRectObject *self, void *closure)
{
    return Py_BuildValue("(ii)", self->r.x + (self->r.w >> 1), self->r.y);
}

static int
pg_rect_setmidtop(pgRectObject *self, PyObject* value, void *closure)
{
    int val1, val2;

    if (!pg_TwoIntsFromObj(value, &val1, &val2)) {
        RAISE(PyExc_TypeError, "invalid rect assignment");
        return -1;
    }
    self->r.x += val1 - (self->r.x + (self->r.w >> 1));
    self->r.y = val2;
    return 0;
}

/*midleft*/
static PyObject *
pg_rect_getmidleft(pgRectObject *self, void *closure)
{
    return Py_BuildValue("(ii)", self->r.x, self->r.y+(self->r.h>>1));
}

static int
pg_rect_setmidleft(pgRectObject *self, PyObject* value, void *closure)
{
    int val1, val2;

    if (!pg_TwoIntsFromObj(value, &val1, &val2)) {
        RAISE(PyExc_TypeError, "invalid rect assignment");
        return -1;
    }
    self->r.x = val1;
    self->r.y += val2 - (self->r.y + (self->r.h >> 1));
    return 0;
}

/*midbottom*/
static PyObject *
pg_rect_getmidbottom(pgRectObject *self, void *closure)
{
    return Py_BuildValue("(ii)", self->r.x + (self->r.w >> 1),
                          self->r.y + self->r.h);
}

static int
pg_rect_setmidbottom(pgRectObject *self, PyObject* value, void *closure)
{
    int val1, val2;

    if (!pg_TwoIntsFromObj(value, &val1, &val2)) {
        RAISE(PyExc_TypeError, "invalid rect assignment");
        return -1;
    }
    self->r.x += val1 - (self->r.x + (self->r.w >> 1));
    self->r.y = val2 - self->r.h;
    return 0;
}

/*midright*/
static PyObject *
pg_rect_getmidright(pgRectObject *self, void *closure)
{
    return Py_BuildValue("(ii)", self->r.x + self->r.w,
                          self->r.y + (self->r.h >> 1));
}

static int
pg_rect_setmidright(pgRectObject *self, PyObject* value, void *closure)
{
    int val1, val2;

    if (!pg_TwoIntsFromObj(value, &val1, &val2)) {
        RAISE(PyExc_TypeError, "invalid rect assignment");
        return -1;
    }
    self->r.x = val1 - self->r.w;
    self->r.y += val2 -(self->r.y + (self->r.h >> 1));
    return 0;
}

/*center*/
static PyObject *
pg_rect_getcenter(pgRectObject *self, void *closure)
{
    return Py_BuildValue("(ii)", self->r.x + (self->r.w >> 1),
                          self->r.y + (self->r.h >> 1));
}

static int
pg_rect_setcenter(pgRectObject *self, PyObject* value, void *closure)
{
    int val1, val2;

    if (!pg_TwoIntsFromObj(value, &val1, &val2)) {
        RAISE(PyExc_TypeError, "invalid rect assignment");
        return -1;
    }
    self->r.x += val1 - (self->r.x + (self->r.w >> 1));
    self->r.y += val2 - (self->r.y + (self->r.h >> 1));
    return 0;
}

/*size*/
static PyObject *
pg_rect_getsize(pgRectObject *self, void *closure)
{
    return Py_BuildValue("(ii)", self->r.w, self->r.h);
}

static int
pg_rect_setsize(pgRectObject *self, PyObject* value, void *closure)
{
    int val1, val2;

    if (!pg_TwoIntsFromObj(value, &val1, &val2)) {
        RAISE(PyExc_TypeError, "invalid rect assignment");
        return -1;
    }
    self->r.w = val1;
    self->r.h = val2;
    return 0;
}

static PyObject *
pg_rect_getsafepickle(pgRectObject *self, void *closure)
{
    Py_RETURN_TRUE;
}

static PyGetSetDef pg_rect_getsets[] = {
    { "x", (getter)pg_rect_getleft, (setter)pg_rect_setleft, NULL, NULL },
    { "y", (getter)pg_rect_gettop, (setter)pg_rect_settop, NULL, NULL },
    { "w", (getter)pg_rect_getwidth, (setter)pg_rect_setwidth, NULL, NULL },
    { "h", (getter)pg_rect_getheight, (setter)pg_rect_setheight, NULL, NULL },
    { "width", (getter)pg_rect_getwidth, (setter)pg_rect_setwidth, NULL, NULL },
    { "height", (getter)pg_rect_getheight, (setter)pg_rect_setheight, NULL, NULL },
    { "top", (getter)pg_rect_gettop, (setter)pg_rect_settop, NULL, NULL },
    { "left", (getter)pg_rect_getleft, (setter)pg_rect_setleft, NULL, NULL },
    { "bottom", (getter)pg_rect_getbottom, (setter)pg_rect_setbottom, NULL, NULL },
    { "right", (getter)pg_rect_getright, (setter)pg_rect_setright, NULL, NULL },
    { "centerx", (getter)pg_rect_getcenterx, (setter)pg_rect_setcenterx, NULL, NULL },
    { "centery", (getter)pg_rect_getcentery, (setter)pg_rect_setcentery, NULL, NULL },
    { "topleft", (getter)pg_rect_gettopleft, (setter)pg_rect_settopleft, NULL, NULL },
    { "topright", (getter)pg_rect_gettopright, (setter)pg_rect_settopright, NULL,
     NULL },
    { "bottomleft", (getter)pg_rect_getbottomleft, (setter)pg_rect_setbottomleft,
      NULL, NULL },
    { "bottomright", (getter)pg_rect_getbottomright, (setter)pg_rect_setbottomright,
      NULL, NULL },
    { "midtop", (getter)pg_rect_getmidtop, (setter)pg_rect_setmidtop, NULL, NULL },
    { "midleft", (getter)pg_rect_getmidleft, (setter)pg_rect_setmidleft, NULL, NULL },
    { "midbottom", (getter)pg_rect_getmidbottom, (setter)pg_rect_setmidbottom, NULL,
      NULL },
    { "midright", (getter)pg_rect_getmidright, (setter)pg_rect_setmidright, NULL,
      NULL },
    { "size", (getter)pg_rect_getsize, (setter)pg_rect_setsize, NULL, NULL },
    { "center", (getter)pg_rect_getcenter, (setter)pg_rect_setcenter, NULL, NULL },

    { "__safe_for_unpickling__", (getter)pg_rect_getsafepickle, NULL, NULL, NULL },
    { NULL, 0, NULL, NULL, NULL }  /* Sentinel */
};

static PyTypeObject pgRect_Type =
{
    TYPE_HEAD(NULL, 0)
    "pygame.Rect",                      /*name*/
    sizeof(pgRectObject),               /*basicsize*/
    0,                                  /*itemsize*/
    /* methods */
    (destructor)pg_rect_dealloc,        /*dealloc*/
    (printfunc)NULL,                    /*print*/
    NULL,                               /*getattr*/
    NULL,                               /*setattr*/
    NULL,                               /*compare/reserved*/
    (reprfunc)pg_rect_repr,             /*repr*/
    &pg_rect_as_number,                 /*as_number*/
    &pg_rect_as_sequence,               /*as_sequence*/
    &pg_rect_as_mapping,                /*as_mapping*/
    (hashfunc)NULL,                     /*hash*/
    (ternaryfunc)NULL,                  /*call*/
    (reprfunc)pg_rect_str,              /*str*/

    /* Space for future expansion */
    0L,0L,0L,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    DOC_PYGAMERECT,                     /* Documentation string */
    NULL,                               /* tp_traverse */
    NULL,                               /* tp_clear */
    (richcmpfunc)pg_rect_richcompare,     /* tp_richcompare */
    offsetof(pgRectObject, weakreflist),  /* tp_weaklistoffset */
    NULL,                               /* tp_iter */
    NULL,                               /* tp_iternext */
    pg_rect_methods,                    /* tp_methods */
    NULL,                               /* tp_members */
    pg_rect_getsets,                    /* tp_getset */
    NULL,                               /* tp_base */
    NULL,                               /* tp_dict */
    NULL,                               /* tp_descr_get */
    NULL,                               /* tp_descr_set */
    0,                                  /* tp_dictoffset */
    (initproc)pg_rect_init,             /* tp_init */
    NULL,                               /* tp_alloc */
    pg_rect_new,                        /* tp_new */
};

static int
pg_rect_init(pgRectObject *self, PyObject *args, PyObject *kwds)
{
    GAME_Rect temp;
    GAME_Rect *argrect = pgRect_FromObject(args, &temp);

    if (argrect == NULL) {
        RAISE(PyExc_TypeError, "Argument must be rect style object");
        return -1;
    }
    self->r.x = argrect->x;
    self->r.y = argrect->y;
    self->r.w = argrect->w;
    self->r.h = argrect->h;
    return 0;
}

static PyMethodDef _pg_module_methods[] =
{
    {NULL, NULL, 0, NULL}
};

/*DOC*/ static char _pg_module_doc[] =
/*DOC*/    "Module for the rectangle object\n";

MODINIT_DEFINE(rect)
{
    PyObject *module, *dict, *apiobj;
    int ecode;
    static void* c_api[PYGAMEAPI_RECT_NUMSLOTS];

#if PY3
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "rect",
        _pg_module_doc,
        -1,
        _pg_module_methods,
        NULL, NULL, NULL, NULL
    };
#endif

    /* import needed apis; Do this first so if there is an error
       the module is not loaded.
    */
    import_pygame_base();
    if (PyErr_Occurred()) {
        MODINIT_ERROR;
    }

    /* Create the module and add the functions */
    if (PyType_Ready(&pgRect_Type) < 0) {
        MODINIT_ERROR;
    }

#if PY3
    module = PyModule_Create(&_module);
#else
    module = Py_InitModule3(MODPREFIX "rect",
                            _pg_module_methods,
                            _pg_module_doc);
#endif
    if (module == NULL) {
        MODINIT_ERROR;
    }
    dict = PyModule_GetDict(module);

    if (PyDict_SetItemString(dict, "RectType", (PyObject *)&pgRect_Type)) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    if (PyDict_SetItemString(dict, "Rect", (PyObject *)&pgRect_Type)) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }

    /* export the c api */
    c_api[0] = &pgRect_Type;
    c_api[1] = pgRect_New;
    c_api[2] = pgRect_New4;
    c_api[3] = pgRect_FromObject;
    apiobj = encapsulate_api(c_api, "rect");
    if (apiobj == NULL) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    ecode = PyDict_SetItemString(dict, PYGAMEAPI_LOCAL_ENTRY, apiobj);
    Py_DECREF(apiobj);
    if (ecode) {
        DECREF_MOD(module);
        MODINIT_ERROR;
    }
    MODINIT_RETURN(module);
}
