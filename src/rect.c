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

static PyTypeObject PyRect_Type;
#define PyRect_Check(x) ((x)->ob_type == &PyRect_Type)

static PyObject* rect_new (PyTypeObject *type, PyObject *args, PyObject *kwds);
static int rect_init (PyRectObject *self, PyObject *args, PyObject *kwds);


/* We store some rect objects which have been allocated already.
   Mostly to work around an old pypy cpyext performance issue.
*/
#ifdef PYPY_VERSION
#define PG_RECT_NUM 49152
const int PG_RECT_FREELIST_MAX = PG_RECT_NUM;
static PyRectObject *pg_rect_freelist[PG_RECT_NUM];
int pg_rect_freelist_num = -1;
#endif


PyObject*
rect_subtype_new4 (PyTypeObject *type, int x, int y, int w, int h)
{
    PyRectObject* rect;
    rect = (PyRectObject *) PyRect_Type.tp_new (type, NULL, NULL);
    if (rect)
    {
        rect->r.x = x;
        rect->r.y = y;
        rect->r.w = w;
        rect->r.h = h;
    }
    return (PyObject*)rect;
}

static PyObject*
rect_new (PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyRectObject *self;

#ifdef PYPY_VERSION
    if (pg_rect_freelist_num > -1) {
        self = pg_rect_freelist[pg_rect_freelist_num];
        Py_INCREF(self);
        /* This is so that pypy garbage collector thinks it is a new obj
           TODO: May be a hack. Is a hack.
           See https://github.com/pygame/pygame/issues/430
        */
        ((PyObject*)(self))->ob_pypy_link = 0;
        pg_rect_freelist_num--;
    } else {
        self = (PyRectObject *)type->tp_alloc (type, 0);
    }
#else
    self = (PyRectObject *)type->tp_alloc (type, 0);
#endif

    if (self)
    {
        self->r.x = self->r.y = 0;
        self->r.w = self->r.h = 0;
        self->weakreflist = NULL;
    }
    return (PyObject*)self;
}

/* object type functions */
static void
rect_dealloc (PyRectObject *self)
{
    if (self->weakreflist)
        PyObject_ClearWeakRefs ((PyObject*)self);

#ifdef PYPY_VERSION
    if (pg_rect_freelist_num < PG_RECT_FREELIST_MAX) {
        pg_rect_freelist_num++;
        pg_rect_freelist[pg_rect_freelist_num] = self;
    } else {
        Py_TYPE(self)->tp_free ((PyObject*)self);
    }
#else
    Py_TYPE(self)->tp_free ((PyObject*)self);
#endif
}

GAME_Rect*
GameRect_FromObject (PyObject* obj, GAME_Rect* temp)
{
    int val;
    int length;

    if (PyRect_Check (obj))
        return &((PyRectObject*) obj)->r;
    if (PySequence_Check (obj) && (length = PySequence_Length (obj)) > 0)
    {
        if (length == 4)
        {
            if (!IntFromObjIndex (obj, 0, &val))
                return NULL;
            temp->x = val;
            if (!IntFromObjIndex (obj, 1, &val))
                return NULL;
            temp->y = val;
            if (!IntFromObjIndex (obj, 2, &val))
                return NULL;
            temp->w = val;
            if (!IntFromObjIndex (obj, 3, &val))
                return NULL;
            temp->h = val;
            return temp;
        }
        if (length == 2)
        {
            PyObject* sub = PySequence_GetItem (obj, 0);
            if (!sub || !PySequence_Check (sub) || PySequence_Length (sub) != 2)
            {
                Py_XDECREF (sub);
                return NULL;
            }
            if (!IntFromObjIndex (sub, 0, &val))
            {
                Py_DECREF (sub);
                return NULL;
            }
            temp->x = val;
            if (!IntFromObjIndex (sub, 1, &val))
            {
                Py_DECREF (sub);
                return NULL;
            }
            temp->y = val;
            Py_DECREF (sub);

            sub = PySequence_GetItem (obj, 1);
            if (!sub || !PySequence_Check (sub) || PySequence_Length (sub) != 2)
            {
                Py_XDECREF (sub);
                return NULL;
            }
            if (!IntFromObjIndex (sub, 0, &val))
            {
                Py_DECREF (sub);
                return NULL;
            }
            temp->w = val;
            if (!IntFromObjIndex (sub, 1, &val))
            {
                Py_DECREF (sub);
                return NULL;
            }
            temp->h = val;
            Py_DECREF (sub);
            return temp;
        }
        if (PyTuple_Check (obj) && length == 1) /*looks like an arg?*/
        {
            PyObject* sub = PyTuple_GET_ITEM (obj, 0);
            if (sub)
                return GameRect_FromObject (sub, temp);
        }
    }
    if (PyObject_HasAttrString (obj, "rect"))
    {
        PyObject *rectattr;
        GAME_Rect *returnrect;
        rectattr = PyObject_GetAttrString (obj, "rect");
        if (PyCallable_Check (rectattr)) /*call if it's a method*/
        {
            PyObject *rectresult = PyObject_CallObject (rectattr, NULL);
            Py_DECREF (rectattr);
            if (!rectresult)
                return NULL;
            rectattr = rectresult;
        }
        returnrect = GameRect_FromObject (rectattr, temp);
        Py_DECREF (rectattr);
        return returnrect;
    }
    return NULL;
}

PyObject*
PyRect_New (SDL_Rect* r)
{
    return rect_subtype_new4 (&PyRect_Type, r->x, r->y, r->w, r->h);
}

PyObject*
PyRect_New4 (int x, int y, int w, int h)
{
    return rect_subtype_new4 (&PyRect_Type, x, y, w, h);
}

static int
DoRectsIntersect (GAME_Rect *A, GAME_Rect *B)
{
    //A.topleft < B.bottomright &&
    //A.bottomright > B.topleft
    return (A->x < B->x + B->w && A->y < B->y + B->h &&
            A->x + A->w > B->x && A->y + A->h > B->y);
}

static PyObject*
rect_normalize (PyObject* oself)
{
    PyRectObject* self = (PyRectObject*)oself;

    if (self->r.w < 0)
    {
        self->r.x += self->r.w;
        self->r.w = -self->r.w;
    }
    if (self->r.h < 0)
    {
        self->r.y += self->r.h;
        self->r.h = -self->r.h;
    }

    Py_RETURN_NONE;
}

static PyObject*
rect_move (PyObject* oself, PyObject* args)
{
    PyRectObject* self = (PyRectObject*)oself;
    int x, y;

    if (!TwoIntsFromObj (args, &x, &y))
        return RAISE (PyExc_TypeError, "argument must contain two numbers");

    return rect_subtype_new4 (Py_TYPE (oself),
                              self->r.x + x, self->r.y + y,
                              self->r.w, self->r.h);
}

static PyObject*
rect_move_ip (PyObject* oself, PyObject* args)
{
    PyRectObject* self = (PyRectObject*)oself;
    int x, y;

    if (!TwoIntsFromObj (args, &x, &y))
        return RAISE (PyExc_TypeError, "argument must contain two numbers");

    self->r.x += x;
    self->r.y += y;
    Py_RETURN_NONE;
}

static PyObject*
rect_inflate (PyObject* oself, PyObject* args)
{
    PyRectObject* self = (PyRectObject*)oself;
    int x, y;

    if (!TwoIntsFromObj (args, &x, &y))
        return RAISE (PyExc_TypeError, "argument must contain two numbers");

    return rect_subtype_new4 (Py_TYPE (oself),
                              self->r.x - x / 2, self->r.y - y / 2,
                              self->r.w + x, self->r.h + y);
}

static PyObject*
rect_inflate_ip (PyObject* oself, PyObject* args)
{
    PyRectObject* self = (PyRectObject*)oself;
    int x, y;

    if (!TwoIntsFromObj (args, &x, &y))
        return RAISE (PyExc_TypeError, "argument must contain two numbers");

    self->r.x -= x / 2;
    self->r.y -= y / 2;
    self->r.w += x;
    self->r.h += y;
    Py_RETURN_NONE;
}

static PyObject*
rect_union (PyObject* oself, PyObject* args)
{
    PyRectObject* self = (PyRectObject*)oself;
    GAME_Rect *argrect, temp;
    int x, y, w, h;
    if (!(argrect = GameRect_FromObject (args, &temp)))
        return RAISE (PyExc_TypeError, "Argument must be rect style object");

    x = MIN (self->r.x, argrect->x);
    y = MIN (self->r.y, argrect->y);
    w = MAX (self->r.x + self->r.w, argrect->x + argrect->w) - x;
    h = MAX (self->r.y + self->r.h, argrect->y + argrect->h) - y;
    return rect_subtype_new4 (Py_TYPE (oself), x, y, w, h);
}

static PyObject*
rect_union_ip (PyObject* oself, PyObject* args)
{
    PyRectObject* self = (PyRectObject*)oself;
    GAME_Rect *argrect, temp;
    int x, y, w, h;
    if (!(argrect = GameRect_FromObject (args, &temp)))
        return RAISE (PyExc_TypeError, "Argument must be rect style object");

    x = MIN (self->r.x, argrect->x);
    y = MIN (self->r.y, argrect->y);
    w = MAX (self->r.x + self->r.w, argrect->x + argrect->w) - x;
    h = MAX (self->r.y + self->r.h, argrect->y + argrect->h) - y;
    self->r.x = x;
    self->r.y = y;
    self->r.w = w;
    self->r.h = h;
    Py_RETURN_NONE;
}

static PyObject*
rect_unionall (PyObject* oself, PyObject* args)
{
    PyRectObject* self = (PyRectObject*)oself;
    GAME_Rect *argrect, temp;
    int loop, size;
    PyObject* list, *obj;
    int t, l, b, r;

    if (!PyArg_ParseTuple (args, "O", &list))
        return NULL;
    if (!PySequence_Check (list))
        return RAISE (PyExc_TypeError,
                      "Argument must be a sequence of rectstyle objects.");

    l = self->r.x;
    t = self->r.y;
    r = self->r.x + self->r.w;
    b = self->r.y + self->r.h;
    size = PySequence_Length (list); /*warning, size could be -1 on error?*/
    if (size < 1) {
        if (size < 0) {
            /*Error.*/
            return NULL;
        }
        /*Empty list: nothing to be done.*/
        return rect_subtype_new4 (Py_TYPE (oself), l, t, r-l, b-t);
    }

    for (loop = 0; loop < size; ++loop)
    {
        obj = PySequence_GetItem (list, loop);
        if(!obj || !(argrect = GameRect_FromObject (obj, &temp)))
        {
            RAISE (PyExc_TypeError,
                   "Argument must be a sequence of rectstyle objects.");
            Py_XDECREF (obj);
            break;
        }
        l = MIN (l, argrect->x);
        t = MIN (t, argrect->y);
        r = MAX (r, argrect->x + argrect->w);
        b = MAX (b, argrect->y + argrect->h);
        Py_DECREF (obj);
    }
    return rect_subtype_new4 (Py_TYPE (oself), l, t, r-l, b-t);
}

static PyObject*
rect_unionall_ip (PyObject* oself, PyObject* args)
{
    PyRectObject* self = (PyRectObject*)oself;
    GAME_Rect *argrect, temp;
    int loop, size;
    PyObject* list, *obj;
    int t, l, b, r;

    if (!PyArg_ParseTuple (args, "O", &list))
        return NULL;
    if (!PySequence_Check (list))
        return RAISE (PyExc_TypeError,
                      "Argument must be a sequence of rectstyle objects.");

    l = self->r.x;
    t = self->r.y;
    r = self->r.x + self->r.w;
    b = self->r.y + self->r.h;

    size = PySequence_Length (list); /*warning, size could be -1 on error?*/
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
        obj = PySequence_GetItem (list, loop);
        if (!obj || !(argrect = GameRect_FromObject (obj, &temp)))
        {
            RAISE (PyExc_TypeError,
                   "Argument must be a sequence of rectstyle objects.");
            Py_XDECREF (obj);
            break;
        }
        l = MIN (l, argrect->x);
        t = MIN (t, argrect->y);
        r = MAX (r, argrect->x + argrect->w);
        b = MAX (b, argrect->y + argrect->h);
        Py_DECREF (obj);
    }

    self->r.x = l;
    self->r.y = t;
    self->r.w = r - l;
    self->r.h = b - t;
    Py_RETURN_NONE;
}

static PyObject*
rect_collidepoint (PyObject* oself, PyObject* args)
{
    PyRectObject* self = (PyRectObject*)oself;
    int x, y;
    int inside;

    if (!TwoIntsFromObj (args, &x, &y))
        return RAISE (PyExc_TypeError, "argument must contain two numbers");

    inside = x >= self->r.x && x < self->r.x + self->r.w &&
        y >= self->r.y && y < self->r.y + self->r.h;

    return PyInt_FromLong (inside);
}

static PyObject*
rect_colliderect (PyObject* oself, PyObject* args)
{
    PyRectObject* self = (PyRectObject*)oself;
    GAME_Rect *argrect, temp;
    if (!(argrect = GameRect_FromObject (args, &temp)))
        return RAISE (PyExc_TypeError, "Argument must be rect style object");

    return PyInt_FromLong (DoRectsIntersect (&self->r, argrect));
}

static PyObject*
rect_collidelist (PyObject* oself, PyObject* args)
{
    PyRectObject* self = (PyRectObject*)oself;
    GAME_Rect *argrect, temp;
    int loop, size;
    PyObject* list, *obj;
    PyObject* ret = NULL;

    if (!PyArg_ParseTuple (args, "O", &list))
        return NULL;

    if (!PySequence_Check (list))
        return RAISE (PyExc_TypeError,
                      "Argument must be a sequence of rectstyle objects.");

    size = PySequence_Length (list); /*warning, size could be -1 on error?*/
    for (loop = 0; loop < size; ++loop)
    {
        obj = PySequence_GetItem (list, loop);
        if (!obj || !(argrect = GameRect_FromObject (obj, &temp)))
        {
            RAISE (PyExc_TypeError,
                   "Argument must be a sequence of rectstyle objects.");
            Py_XDECREF (obj);
            break;
        }
        if (DoRectsIntersect (&self->r, argrect))
        {
            ret = PyInt_FromLong (loop);
            Py_DECREF (obj);
            break;
        }
        Py_DECREF (obj);
    }
    if (loop == size)
        ret = PyInt_FromLong (-1);

    return ret;
}

static PyObject*
rect_collidelistall (PyObject* oself, PyObject* args)
{
    PyRectObject* self = (PyRectObject*)oself;
    GAME_Rect *argrect, temp;
    int loop, size;
    PyObject* list, *obj;
    PyObject* ret = NULL;

    if (!PyArg_ParseTuple (args, "O", &list))
        return NULL;

    if (!PySequence_Check (list))
        return RAISE (PyExc_TypeError,
                      "Argument must be a sequence of rectstyle objects.");

    ret = PyList_New (0);
    if (!ret)
        return NULL;

    size = PySequence_Length (list); /*warning, size could be -1?*/
    for (loop = 0; loop < size; ++loop)
    {
        obj = PySequence_GetItem (list, loop);

        if(!obj || !(argrect = GameRect_FromObject (obj, &temp)))
        {
            Py_XDECREF (obj);
            Py_DECREF (ret);
            return RAISE (PyExc_TypeError,
                          "Argument must be a sequence of rectstyle objects.");
        }

        if (DoRectsIntersect (&self->r, argrect))
        {
            PyObject* num = PyInt_FromLong (loop);
            if (!num)
            {
                Py_DECREF (obj);
                return NULL;
            }
            PyList_Append (ret, num);
            Py_DECREF (num);
        }
        Py_DECREF (obj);
    }

    return ret;
}

static PyObject*
rect_collidedict (PyObject* oself, PyObject* args)
{
    PyRectObject* self = (PyRectObject*)oself;
    GAME_Rect *argrect, temp;
    Py_ssize_t loop=0;
    Py_ssize_t values=0;
    PyObject* dict, *key, *val;
    PyObject* ret = NULL;

    if (!PyArg_ParseTuple (args, "O|i", &dict, &values))
        return NULL;
    if (!PyDict_Check (dict))
        return RAISE (PyExc_TypeError,
                      "Argument must be a dict with rectstyle keys.");

    while (PyDict_Next (dict, &loop, &key, &val))
    {
        if(values) {
            if (!(argrect = GameRect_FromObject (val, &temp)))
            {
                RAISE (PyExc_TypeError,
                       "Argument must be a dict with rectstyle values.");
                break;
            }
        } else {
            if (!(argrect = GameRect_FromObject (key, &temp)))
            {
                RAISE (PyExc_TypeError,
                       "Argument must be a dict with rectstyle keys.");
                break;
            }
        }


        if (DoRectsIntersect (&self->r, argrect))
        {
            ret = Py_BuildValue ("(OO)", key, val);
            break;
        }
    }

    if (!ret)
        Py_RETURN_NONE;
    return ret;
}

static PyObject*
rect_collidedictall (PyObject* oself, PyObject* args)
{
    PyRectObject* self = (PyRectObject*)oself;
    GAME_Rect *argrect, temp;
    Py_ssize_t loop=0;
    /* should we use values or keys? */
    Py_ssize_t values=0;

    PyObject* dict, *key, *val;
    PyObject* ret = NULL;

    if (!PyArg_ParseTuple (args, "O|i", &dict, &values))
        return NULL;
    if (!PyDict_Check (dict))
        return RAISE (PyExc_TypeError,
                      "Argument must be a dict with rectstyle keys.");

    ret = PyList_New (0);
    if(!ret)
        return NULL;

    while (PyDict_Next (dict, &loop, &key, &val))
    {
        if (values) {
            if (!(argrect = GameRect_FromObject (val, &temp)))
            {
                Py_DECREF (ret);
                return RAISE (PyExc_TypeError,
                              "Argument must be a dict with rectstyle values.");
            }
        } else {
            if (!(argrect = GameRect_FromObject (key, &temp)))
            {
                Py_DECREF (ret);
                return RAISE (PyExc_TypeError,
                              "Argument must be a dict with rectstyle keys.");
            }
        }

        if (DoRectsIntersect (&self->r, argrect))
        {
            PyObject* num = Py_BuildValue ("(OO)", key, val);
            if(!num)
                return NULL;
            PyList_Append (ret, num);
            Py_DECREF (num);
        }
    }

    return ret;
}

static PyObject*
rect_clip (PyObject* self, PyObject* args)
{
    GAME_Rect *A, *B, temp;
    int x, y, w, h;

    A = &((PyRectObject*) self)->r;
    if (!(B = GameRect_FromObject (args, &temp)))
        return RAISE (PyExc_TypeError, "Argument must be rect style object");

    /* Left */
    if ((A->x >= B->x) && (A->x < (B->x + B->w)))
        x = A->x;
    else if ((B->x >= A->x) && (B->x < (A->x + A->w)))
        x = B->x;
    else
        goto nointersect;

    /* Right */
    if (((A->x + A->w) > B->x) && ((A->x + A->w) <= (B->x + B->w)))
        w = (A->x + A->w) - x;
    else if (((B->x + B->w) > A->x) && ((B->x + B->w) <= (A->x + A->w)))
        w = (B->x + B->w) - x;
    else
        goto nointersect;

    /* Top */
    if ((A->y >= B->y) && (A->y < (B->y + B->h)))
        y = A->y;
    else if ((B->y >= A->y) && (B->y < (A->y + A->h)))
        y = B->y;
    else
        goto nointersect;

    /* Bottom */
    if (((A->y + A->h) > B->y) && ((A->y + A->h) <= (B->y + B->h)))
        h = (A->y + A->h) - y;
    else if (((B->y + B->h) > A->y) && ((B->y + B->h) <= (A->y + A->h)))
        h = (B->y + B->h) - y;
    else
        goto nointersect;

    return rect_subtype_new4 (Py_TYPE (self), x, y, w, h);

nointersect:
    return rect_subtype_new4 (Py_TYPE (self), A->x, A->y, 0, 0);
}

static PyObject*
rect_contains (PyObject* oself, PyObject* args)
{
    int contained;
    PyRectObject* self = (PyRectObject*)oself;
    GAME_Rect *argrect, temp;
    if (!(argrect = GameRect_FromObject (args, &temp)))
        return RAISE (PyExc_TypeError, "Argument must be rect style object");

    contained = (self->r.x <= argrect->x) && (self->r.y <= argrect->y) &&
        (self->r.x + self->r.w >= argrect->x + argrect->w) &&
        (self->r.y + self->r.h >= argrect->y + argrect->h) &&
        (self->r.x + self->r.w > argrect->x) &&
        (self->r.y + self->r.h > argrect->y);

    return PyInt_FromLong (contained);
}

static PyObject*
rect_clamp (PyObject* oself, PyObject* args)
{
    PyRectObject* self = (PyRectObject*)oself;
    GAME_Rect *argrect, temp;
    int x, y;
    if (!(argrect = GameRect_FromObject (args, &temp)))
        return RAISE (PyExc_TypeError, "Argument must be rect style object");

    if (self->r.w >= argrect->w)
        x = argrect->x + argrect->w / 2 - self->r.w / 2;
    else if (self->r.x < argrect->x)
        x = argrect->x;
    else if (self->r.x + self->r.w > argrect->x + argrect->w)
        x = argrect->x + argrect->w - self->r.w;
    else
        x = self->r.x;

    if (self->r.h >= argrect->h)
        y = argrect->y + argrect->h / 2 - self->r.h / 2;
    else if (self->r.y < argrect->y)
        y = argrect->y;
    else if (self->r.y + self->r.h > argrect->y + argrect->h)
        y = argrect->y + argrect->h - self->r.h;
    else
        y = self->r.y;

    return rect_subtype_new4 (Py_TYPE (oself), x, y, self->r.w, self->r.h);
}

static PyObject*
rect_fit (PyObject* oself, PyObject* args)
{
    PyRectObject* self = (PyRectObject*)oself;
    GAME_Rect *argrect, temp;
    int w, h, x, y;
    float xratio, yratio, maxratio;
    if (!(argrect = GameRect_FromObject (args, &temp)))
        return RAISE (PyExc_TypeError, "Argument must be rect style object");

    xratio = (float) self->r.w / (float) argrect->w;
    yratio = (float) self->r.h / (float) argrect->h;
    maxratio = (xratio > yratio) ? xratio : yratio;

    w = (int) (self->r.w / maxratio);
    h = (int) (self->r.h / maxratio);

    x = argrect->x + (argrect->w - w)/2;
    y = argrect->y + (argrect->h - h)/2;

    return rect_subtype_new4 (Py_TYPE (oself), x, y, w, h);
}

static PyObject*
rect_clamp_ip (PyObject* oself, PyObject* args)
{
    PyRectObject* self = (PyRectObject*)oself;
    GAME_Rect *argrect, temp;
    int x, y;
    if (!(argrect = GameRect_FromObject (args, &temp)))
        return RAISE (PyExc_TypeError, "Argument must be rect style object");

    if (self->r.w >= argrect->w)
        x = argrect->x + argrect->w / 2 - self->r.w / 2;
    else if (self->r.x < argrect->x)
        x = argrect->x;
    else if (self->r.x + self->r.w > argrect->x + argrect->w)
        x = argrect->x + argrect->w - self->r.w;
    else
        x = self->r.x;

    if (self->r.h >= argrect->h)
        y = argrect->y + argrect->h / 2 - self->r.h / 2;
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
static PyObject*
rect_reduce (PyObject* oself)
{
    PyRectObject* self = (PyRectObject*)oself;
    return Py_BuildValue ("(O(iiii))", oself->ob_type,
                          (int)self->r.x, (int)self->r.y,
                          (int)self->r.w, (int)self->r.h);
}

/* for copy module */
static PyObject*
rect_copy (PyObject* oself)
{
    PyRectObject* self = (PyRectObject*)oself;
    return rect_subtype_new4 (Py_TYPE (oself),
                              self->r.x, self->r.y, self->r.w, self->r.h);
}

static struct PyMethodDef rect_methods[] =
{
    { "normalize", (PyCFunction) rect_normalize, METH_NOARGS,
      DOC_RECTNORMALIZE },
    { "clip", rect_clip, METH_VARARGS, DOC_RECTCLIP},
    { "clamp", rect_clamp, METH_VARARGS, DOC_RECTCLAMP},
    { "clamp_ip", rect_clamp_ip, METH_VARARGS, DOC_RECTCLAMPIP},
    { "copy", (PyCFunction) rect_copy, METH_NOARGS, DOC_RECTCOPY},
    { "fit", rect_fit, METH_VARARGS, DOC_RECTFIT},
    { "move", rect_move, METH_VARARGS, DOC_RECTMOVE},
    { "inflate",  rect_inflate, METH_VARARGS, DOC_RECTINFLATE},
    { "union",  rect_union, METH_VARARGS, DOC_RECTUNION},
    { "unionall",  rect_unionall, METH_VARARGS, DOC_RECTUNIONALL},
    { "move_ip",  rect_move_ip, METH_VARARGS, DOC_RECTMOVEIP},
    { "inflate_ip", rect_inflate_ip, METH_VARARGS, DOC_RECTINFLATEIP},
    { "union_ip", rect_union_ip, METH_VARARGS, DOC_RECTUNIONIP},
    { "unionall_ip", rect_unionall_ip, METH_VARARGS, DOC_RECTUNIONALLIP},
    { "collidepoint", rect_collidepoint, METH_VARARGS, DOC_RECTCOLLIDEPOINT},
    { "colliderect", rect_colliderect, METH_VARARGS, DOC_RECTCOLLIDERECT},
    { "collidelist", rect_collidelist, METH_VARARGS, DOC_RECTCOLLIDELIST},
    { "collidelistall", rect_collidelistall, METH_VARARGS,
      DOC_RECTCOLLIDELISTALL},
    { "collidedict", rect_collidedict, METH_VARARGS, DOC_RECTCOLLIDEDICT},
    { "collidedictall", rect_collidedictall, METH_VARARGS,
      DOC_RECTCOLLIDEDICTALL},
    { "contains", rect_contains, METH_VARARGS, DOC_RECTCONTAINS},
    { "__reduce__", (PyCFunction) rect_reduce, METH_NOARGS, NULL},
    { "__copy__", (PyCFunction) rect_copy, METH_NOARGS, NULL},
    { NULL, NULL, 0, NULL }
};

/* sequence functions */

static Py_ssize_t
rect_length (PyObject *_self)
{
    return 4;
}

static PyObject*
rect_item (PyRectObject *self, Py_ssize_t i)
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

    return PyInt_FromLong (data[i]);
}

static int
rect_ass_item (PyRectObject *self, Py_ssize_t i, PyObject *v)
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
    if (!IntFromObj (v, &val))
    {
        RAISE (PyExc_TypeError, "Must assign numeric values");
        return -1;
    }
    data[i] = val;
    return 0;
}

static PySequenceMethods rect_as_sequence =
{
    rect_length,                     /*length*/
    NULL,                            /*concat*/
    NULL,                            /*repeat*/
    (ssizeargfunc)rect_item,         /*item*/
    NULL,                            /*slice*/
    (ssizeobjargproc)rect_ass_item,  /*ass_item*/
    NULL,                            /*ass_slice*/
};

static PyObject *
rect_subscript(PyRectObject *self, PyObject *op)
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
        return rect_item(self, i);
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
rect_ass_subscript(PyRectObject *self, PyObject *op, PyObject *value)
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
        return rect_ass_item(self, i, value);
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

static PyMappingMethods rect_as_mapping =
{
    (lenfunc)rect_length,             /*mp_length*/
    (binaryfunc)rect_subscript,       /*mp_subscript*/
    (objobjargproc)rect_ass_subscript /*mp_ass_subscript*/
};

/* numeric functions */
static int
rect_nonzero (PyRectObject *self)
{
    return self->r.w != 0 && self->r.h != 0;
}

#if !PY3
static int
rect_coerce (PyObject** o1, PyObject** o2)
{
    PyObject* new1;
    PyObject* new2;
    GAME_Rect* r, temp;

    if (PyRect_Check (*o1))
    {
        new1 = *o1;
        Py_INCREF (new1);
    }
    else if ((r = GameRect_FromObject (*o1, &temp)))
        new1 = PyRect_New4 (r->x, r->y, r->w, r->h);
    else
        return 1;

    if (PyRect_Check (*o2))
    {
        new2 = *o2;
        Py_INCREF (new2);
    }
    else if ((r = GameRect_FromObject (*o2, &temp)))
        new2 = PyRect_New4 (r->x, r->y, r->w, r->h);
    else
    {
        Py_DECREF (new1);
        return 1;
    }

    *o1 = new1;
    *o2 = new2;
    return 0;
}
#endif

static PyNumberMethods rect_as_number =
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
    (inquiry)rect_nonzero,    /*nonzero / bool*/
    (unaryfunc)NULL,          /*invert*/
    (binaryfunc)NULL,         /*lshift*/
    (binaryfunc)NULL,         /*rshift*/
    (binaryfunc)NULL,         /*and*/
    (binaryfunc)NULL,         /*xor*/
    (binaryfunc)NULL,         /*or*/
#if !PY3
    (coercion)rect_coerce,    /*coerce*/
#endif
    (unaryfunc)NULL,          /*int*/
#if !PY3
    (unaryfunc)NULL,          /*long*/
#endif
    (unaryfunc)NULL,          /*float*/
};

static PyObject*
rect_repr (PyRectObject *self)
{
    char string[256];
    sprintf (string, "<rect(%d, %d, %d, %d)>", self->r.x, self->r.y,
             self->r.w, self->r.h);
    return Text_FromUTF8 (string);
}

static PyObject*
rect_str (PyRectObject *self)
{
    return rect_repr (self);
}

static PyObject*
rect_richcompare(PyObject *o1, PyObject *o2, int opid)
{
    GAME_Rect *o1rect, *o2rect, temp1, temp2;
    int cmp;

    o1rect = GameRect_FromObject (o1, &temp1);
    if (!o1rect) {
        goto Unimplemented;
    }
    o2rect = GameRect_FromObject (o2, &temp2);
    if (!o2rect)
    {
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
        return PyBool_FromLong (cmp < 0);
    case Py_LE:
        return PyBool_FromLong (cmp <= 0);
    case Py_EQ:
        return PyBool_FromLong (cmp == 0);
    case Py_NE:
        return PyBool_FromLong (cmp != 0);
    case Py_GT:
        return PyBool_FromLong (cmp > 0);
    case Py_GE:
        return PyBool_FromLong (cmp >= 0);
    default:
        break;
    }

Unimplemented:
    Py_INCREF (Py_NotImplemented);
    return Py_NotImplemented;
}

/*width*/
static PyObject*
rect_getwidth (PyRectObject *self, void *closure)
{
    return PyInt_FromLong (self->r.w);
}

static int
rect_setwidth (PyRectObject *self, PyObject* value, void *closure)
{
    int val1;
    if (!IntFromObj (value, &val1))
        return -1;
    self->r.w = val1;
    return 0;
}

/*height*/
static PyObject*
rect_getheight (PyRectObject *self, void *closure)
{
    return PyInt_FromLong (self->r.h);
}

static int
rect_setheight (PyRectObject *self, PyObject* value, void *closure)
{
    int val1;
    if (!IntFromObj (value, &val1))
    {
        RAISE (PyExc_TypeError, "invalid rect assignment");
        return -1;
    }
    self->r.h = val1;
    return 0;
}

/*top*/
static PyObject*
rect_gettop (PyRectObject *self, void *closure)
{
    return PyInt_FromLong (self->r.y);
}

static int
rect_settop (PyRectObject *self, PyObject* value, void *closure)
{
    int val1;
    if (!IntFromObj (value, &val1))
    {
        RAISE (PyExc_TypeError, "invalid rect assignment");
        return -1;
    }
    self->r.y = val1;
    return 0;
}

/*left*/
static PyObject*
rect_getleft (PyRectObject *self, void *closure)
{
    return PyInt_FromLong (self->r.x);
}

static int
rect_setleft (PyRectObject *self, PyObject* value, void *closure)
{
    int val1;
    if (!IntFromObj (value, &val1))
    {
        RAISE (PyExc_TypeError, "invalid rect assignment");
        return -1;
    }
    self->r.x = val1;
    return 0;
}

/*right*/
static PyObject*
rect_getright (PyRectObject *self, void *closure)
{
    return PyInt_FromLong (self->r.x + self->r.w);
}

static int
rect_setright (PyRectObject *self, PyObject* value, void *closure)
{
    int val1;
    if (!IntFromObj (value, &val1))
    {
        RAISE (PyExc_TypeError, "invalid rect assignment");
        return -1;
    }
    self->r.x = val1-self->r.w;
    return 0;
}

/*bottom*/
static PyObject*
rect_getbottom (PyRectObject *self, void *closure)
{
    return PyInt_FromLong (self->r.y + self->r.h);
}

static int
rect_setbottom (PyRectObject *self, PyObject* value, void *closure)
{
    int val1;
    if (!IntFromObj (value, &val1))
    {
        RAISE (PyExc_TypeError, "invalid rect assignment");
        return -1;
    }
    self->r.y = val1-self->r.h;
    return 0;
}

/*centerx*/
static PyObject*
rect_getcenterx (PyRectObject *self, void *closure)
{
    return PyInt_FromLong (self->r.x + (self->r.w >> 1));
}

static int
rect_setcenterx (PyRectObject *self, PyObject* value, void *closure)
{
    int val1;
    if (!IntFromObj (value, &val1))
    {
        RAISE (PyExc_TypeError, "invalid rect assignment");
        return -1;
    }
    self->r.x = val1 - (self->r.w >> 1);
    return 0;
}

/*centery*/
static PyObject*
rect_getcentery (PyRectObject *self, void *closure)
{
    return PyInt_FromLong (self->r.y + (self->r.h >> 1));
}

static int
rect_setcentery (PyRectObject *self, PyObject* value, void *closure)
{
    int val1;
    if (!IntFromObj (value, &val1))
    {
        RAISE (PyExc_TypeError, "invalid rect assignment");
        return -1;
    }
    self->r.y = val1 - (self->r.h >> 1);
    return 0;
}

/*topleft*/
static PyObject*
rect_gettopleft (PyRectObject *self, void *closure)
{
    return Py_BuildValue ("(ii)", self->r.x, self->r.y);
}

static int
rect_settopleft (PyRectObject *self, PyObject* value, void *closure)
{
    int val1, val2;
    if (!TwoIntsFromObj (value, &val1, &val2))
    {
        RAISE (PyExc_TypeError, "invalid rect assignment");
        return -1;
    }
    self->r.x = val1;
    self->r.y = val2;
    return 0;
}

/*topright*/
static PyObject*
rect_gettopright (PyRectObject *self, void *closure)
{
    return Py_BuildValue ("(ii)", self->r.x+self->r.w, self->r.y);
}

static int
rect_settopright (PyRectObject *self, PyObject* value, void *closure)
{
    int val1, val2;
    if (!TwoIntsFromObj (value, &val1, &val2))
    {
        RAISE (PyExc_TypeError, "invalid rect assignment");
        return -1;
    }
    self->r.x = val1-self->r.w;
    self->r.y = val2;
    return 0;
}

/*bottomleft*/
static PyObject*
rect_getbottomleft (PyRectObject *self, void *closure)
{
    return Py_BuildValue ("(ii)", self->r.x, self->r.y+self->r.h);
}

static int
rect_setbottomleft (PyRectObject *self, PyObject* value, void *closure)
{
    int val1, val2;
    if (!TwoIntsFromObj (value, &val1, &val2))
    {
        RAISE (PyExc_TypeError, "invalid rect assignment");
        return -1;
    }
    self->r.x = val1;
    self->r.y = val2-self->r.h;
    return 0;
}

/*bottomright*/
static PyObject*
rect_getbottomright (PyRectObject *self, void *closure)
{
    return Py_BuildValue ("(ii)", self->r.x+self->r.w, self->r.y+self->r.h);
}

static int
rect_setbottomright (PyRectObject *self, PyObject* value, void *closure)
{
    int val1, val2;
    if (!TwoIntsFromObj (value, &val1, &val2))
    {
        RAISE (PyExc_TypeError, "invalid rect assignment");
        return -1;
    }
    self->r.x = val1-self->r.w;
    self->r.y = val2-self->r.h;
    return 0;
}

/*midtop*/
static PyObject*
rect_getmidtop (PyRectObject *self, void *closure)
{
    return Py_BuildValue ("(ii)", self->r.x + (self->r.w >> 1), self->r.y);
}

static int
rect_setmidtop (PyRectObject *self, PyObject* value, void *closure)
{
    int val1, val2;
    if (!TwoIntsFromObj (value, &val1, &val2))
    {
        RAISE (PyExc_TypeError, "invalid rect assignment");
        return -1;
    }
    self->r.x += val1 - (self->r.x + (self->r.w >> 1));
    self->r.y = val2;
    return 0;
}

/*midleft*/
static PyObject*
rect_getmidleft (PyRectObject *self, void *closure)
{
    return Py_BuildValue ("(ii)", self->r.x, self->r.y+(self->r.h>>1));
}

static int
rect_setmidleft (PyRectObject *self, PyObject* value, void *closure)
{
    int val1, val2;
    if (!TwoIntsFromObj (value, &val1, &val2))
    {
        RAISE (PyExc_TypeError, "invalid rect assignment");
        return -1;
    }
    self->r.x = val1;
    self->r.y += val2 - (self->r.y + (self->r.h >> 1));
    return 0;
}

/*midbottom*/
static PyObject*
rect_getmidbottom (PyRectObject *self, void *closure)
{
    return Py_BuildValue ("(ii)", self->r.x + (self->r.w >> 1),
                          self->r.y + self->r.h);
}

static int
rect_setmidbottom (PyRectObject *self, PyObject* value, void *closure)
{
    int val1, val2;
    if (!TwoIntsFromObj (value, &val1, &val2))
    {
        RAISE (PyExc_TypeError, "invalid rect assignment");
        return -1;
    }
    self->r.x += val1 - (self->r.x + (self->r.w >> 1));
    self->r.y = val2 - self->r.h;
    return 0;
}

/*midright*/
static PyObject*
rect_getmidright (PyRectObject *self, void *closure)
{
    return Py_BuildValue ("(ii)", self->r.x + self->r.w,
                          self->r.y + (self->r.h >> 1));
}

static int
rect_setmidright (PyRectObject *self, PyObject* value, void *closure)
{
    int val1, val2;
    if (!TwoIntsFromObj (value, &val1, &val2))
    {
        RAISE (PyExc_TypeError, "invalid rect assignment");
        return -1;
    }
    self->r.x = val1 - self->r.w;
    self->r.y += val2 -(self->r.y + (self->r.h >> 1));
    return 0;
}

/*center*/
static PyObject*
rect_getcenter (PyRectObject *self, void *closure)
{
    return Py_BuildValue ("(ii)", self->r.x + (self->r.w >> 1),
                          self->r.y + (self->r.h >> 1));
}

static int
rect_setcenter (PyRectObject *self, PyObject* value, void *closure)
{
    int val1, val2;
    if (!TwoIntsFromObj (value, &val1, &val2))
    {
        RAISE (PyExc_TypeError, "invalid rect assignment");
        return -1;
    }
    self->r.x += val1 - (self->r.x + (self->r.w >> 1));
    self->r.y += val2 - (self->r.y + (self->r.h >> 1));
    return 0;
}

/*size*/
static PyObject*
rect_getsize (PyRectObject *self, void *closure)
{
    return Py_BuildValue ("(ii)", self->r.w, self->r.h);
}

static int
rect_setsize (PyRectObject *self, PyObject* value, void *closure)
{
    int val1, val2;
    if (!TwoIntsFromObj (value, &val1, &val2))
    {
        RAISE (PyExc_TypeError, "invalid rect assignment");
        return -1;
    }
    self->r.w = val1;
    self->r.h = val2;
    return 0;
}

static PyObject*
rect_getsafepickle (PyRectObject *self, void *closure)
{
    Py_RETURN_TRUE;
}

static PyGetSetDef rect_getsets[] = {
    { "x", (getter)rect_getleft, (setter)rect_setleft, NULL, NULL },
    { "y", (getter)rect_gettop, (setter)rect_settop, NULL, NULL },
    { "w", (getter)rect_getwidth, (setter)rect_setwidth, NULL, NULL },
    { "h", (getter)rect_getheight, (setter)rect_setheight, NULL, NULL },
    { "width", (getter)rect_getwidth, (setter)rect_setwidth, NULL, NULL },
    { "height", (getter)rect_getheight, (setter)rect_setheight, NULL, NULL },
    { "top", (getter)rect_gettop, (setter)rect_settop, NULL, NULL },
    { "left", (getter)rect_getleft, (setter)rect_setleft, NULL, NULL },
    { "bottom", (getter)rect_getbottom, (setter)rect_setbottom, NULL, NULL },
    { "right", (getter)rect_getright, (setter)rect_setright, NULL, NULL },
    { "centerx", (getter)rect_getcenterx, (setter)rect_setcenterx, NULL, NULL },
    { "centery", (getter)rect_getcentery, (setter)rect_setcentery, NULL, NULL },
    { "topleft", (getter)rect_gettopleft, (setter)rect_settopleft, NULL, NULL },
    { "topright", (getter)rect_gettopright, (setter)rect_settopright, NULL,
     NULL },
    { "bottomleft", (getter)rect_getbottomleft, (setter)rect_setbottomleft,
      NULL, NULL },
    { "bottomright", (getter)rect_getbottomright, (setter)rect_setbottomright,
      NULL, NULL },
    { "midtop", (getter)rect_getmidtop, (setter)rect_setmidtop, NULL, NULL },
    { "midleft", (getter)rect_getmidleft, (setter)rect_setmidleft, NULL, NULL },
    { "midbottom", (getter)rect_getmidbottom, (setter)rect_setmidbottom, NULL,
      NULL },
    { "midright", (getter)rect_getmidright, (setter)rect_setmidright, NULL,
      NULL },
    { "size", (getter)rect_getsize, (setter)rect_setsize, NULL, NULL },
    { "center", (getter)rect_getcenter, (setter)rect_setcenter, NULL, NULL },

    { "__safe_for_unpickling__", (getter)rect_getsafepickle, NULL, NULL, NULL },
    { NULL, 0, NULL, NULL, NULL }  /* Sentinel */
};

static PyTypeObject PyRect_Type =
{
    TYPE_HEAD (NULL, 0)
    "pygame.Rect",                      /*name*/
    sizeof(PyRectObject),               /*basicsize*/
    0,                                  /*itemsize*/
    /* methods */
    (destructor)rect_dealloc,           /*dealloc*/
    (printfunc)NULL,                    /*print*/
    NULL,                               /*getattr*/
    NULL,                               /*setattr*/
    NULL,                               /*compare/reserved*/
    (reprfunc)rect_repr,                /*repr*/
    &rect_as_number,                    /*as_number*/
    &rect_as_sequence,                  /*as_sequence*/
    &rect_as_mapping,                   /*as_mapping*/
    (hashfunc)NULL,                     /*hash*/
    (ternaryfunc)NULL,                  /*call*/
    (reprfunc)rect_str,                 /*str*/

    /* Space for future expansion */
    0L,0L,0L,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    DOC_PYGAMERECT,                     /* Documentation string */
    0,                                  /* tp_traverse */
    0,                                  /* tp_clear */
    (richcmpfunc)rect_richcompare,      /* tp_richcompare */
    offsetof(PyRectObject, weakreflist),  /* tp_weaklistoffset */
    0,                                  /* tp_iter */
    0,                                  /* tp_iternext */
    rect_methods,                       /* tp_methods */
    0,                                  /* tp_members */
    rect_getsets,                       /* tp_getset */
    0,                                  /* tp_base */
    0,                                  /* tp_dict */
    0,                                  /* tp_descr_get */
    0,                                  /* tp_descr_set */
    0,                                  /* tp_dictoffset */
    (initproc)rect_init,                /* tp_init */
    0,                                  /* tp_alloc */
    rect_new,                           /* tp_new */
};


static int
rect_init (PyRectObject *self, PyObject *args, PyObject *kwds)
{
    GAME_Rect *argrect, temp;
    if (!(argrect = GameRect_FromObject (args, &temp)))
    {
        RAISE (PyExc_TypeError, "Argument must be rect style object");
        return -1;
    }

    self->r.x = argrect->x;
    self->r.y = argrect->y;
    self->r.w = argrect->w;
    self->r.h = argrect->h;
    return 0;
}

static PyMethodDef _rect_methods[] =
{
    {NULL, NULL, 0, NULL}
};

/*DOC*/ static char _rectangle_doc[] =
/*DOC*/    "Module for the rectangle object\n";

MODINIT_DEFINE (rect)
{
    PyObject *module, *dict, *apiobj;
    int ecode;
    static void* c_api[PYGAMEAPI_RECT_NUMSLOTS];

#if PY3
    static struct PyModuleDef _module = {
        PyModuleDef_HEAD_INIT,
        "rect",
        _rectangle_doc,
        -1,
        _rect_methods,
        NULL, NULL, NULL, NULL
    };
#endif

    /* imported needed apis; Do this first so if there is an error
       the module is not loaded.
    */
    import_pygame_base ();
    if (PyErr_Occurred ()) {
        MODINIT_ERROR;
    }

    /* Create the module and add the functions */
    if (PyType_Ready (&PyRect_Type) < 0) {
        MODINIT_ERROR;
    }

#if PY3
    module = PyModule_Create (&_module);
#else
    module = Py_InitModule3 (MODPREFIX "rect", _rect_methods, _rectangle_doc);
#endif
    if (module == NULL) {
        MODINIT_ERROR;
    }
    dict = PyModule_GetDict (module);

    if (PyDict_SetItemString (dict, "RectType", (PyObject *)&PyRect_Type)) {
        DECREF_MOD (module);
        MODINIT_ERROR;
    }
    if (PyDict_SetItemString (dict, "Rect", (PyObject *)&PyRect_Type)) {
        DECREF_MOD (module);
        MODINIT_ERROR;
    }

    /* export the c api */
    c_api[0] = &PyRect_Type;
    c_api[1] = PyRect_New;
    c_api[2] = PyRect_New4;
    c_api[3] = GameRect_FromObject;
    apiobj = encapsulate_api (c_api, "rect");
    if (apiobj == NULL) {
        DECREF_MOD (module);
        MODINIT_ERROR;
    }
    ecode = PyDict_SetItemString (dict, PYGAMEAPI_LOCAL_ENTRY, apiobj);
    Py_DECREF (apiobj);
    if (ecode) {
        DECREF_MOD (module);
        MODINIT_ERROR;
    }
    MODINIT_RETURN (module);
}
