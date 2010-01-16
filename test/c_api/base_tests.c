#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pygame2/pgbase.h>

#define ERROR(x)                                \
    {                                           \
        fprintf(stderr, "*** %s\n", x);         \
        PyErr_Print ();                         \
        Py_Finalize ();                         \
        exit(1);                                \
    }
#define NEAR_ZERO(x) (fabs(x) <= 1e-6)

static void
test_helpers (void)
{
    PyObject *val, *seq, *string, *tmp;
    pgint32 sw = 0, sh = 0;
    double d, e;
    int i, j;
    unsigned int u;
    long v;
    unsigned long q;
    char *str;

    val = PyFloat_FromDouble (55.33f);
    if (!DoubleFromObj(val, &d))
        ERROR ("Mismatch in DoubleFromObj");
    if (d != 55.33f)
        ERROR ("Mismatch in DoubleFromObj result");
    Py_DECREF (val);
    
    val = PyLong_FromLong (-10);
    if (!IntFromObj(val, &i))
        ERROR ("Mismatch in IntFromObj");
    if (i != -10)
        ERROR ("Mismatch in IntFromObj result");
    Py_DECREF (val);

    val = PyLong_FromLong (-10);
    if (!LongFromObj(val, &v))
        ERROR ("Mismatch in LongFromObj");
    if (v != -10)
        ERROR ("Mismatch in LongFromObj result");
    Py_DECREF (val);

    val = PyLong_FromLong (10);
    if (!UintFromObj(val, &u))
        ERROR ("Mismatch in UintFromObj");
    if (u != 10)
        ERROR ("Mismatch in UintFromObj result");
    Py_DECREF (val);

    val = PyLong_FromUnsignedLong (0xff00ff00ff);
    if (!UlongFromObj(val, &q))
        ERROR ("Mismatch in UlongFromObj");
    if (u != 0xff00ff00ff)
        ERROR ("Mismatch in UlongFromObj result");
    Py_DECREF (val);

    seq = PyTuple_New (2);
    PyTuple_SET_ITEM (seq, 0, PyLong_FromLong (2));
    PyTuple_SET_ITEM (seq, 1, PyLong_FromLong (4));

    if (!DoubleFromSeqIndex (seq, 0, &d))
        ERROR ("Mismatch in DoubleFromSeqIndex (0)");
    if (d != 2.f)
        ERROR ("Mismatch in DoubleFromSeqIndex result 0");
    if (!DoubleFromSeqIndex (seq, 1, &d))
        ERROR ("Mismatch in DoubleFromSeqIndex (1)");
    if (d != 4.f)
        ERROR ("Mismatch in DoubleFromSeqIndex result 1");

    if (!IntFromSeqIndex (seq, 0, &i))
        ERROR ("Mismatch in IntFromSeqIndex (0)");
    if (i != 2)
        ERROR ("Mismatch in IntFromSeqIndex result 0");
    if (!IntFromSeqIndex (seq, 1, &i))
        ERROR ("Mismatch in IntFromSeqIndex (1)");
    if (i != 4)
        ERROR ("Mismatch in IntFromSeqIndex result 1");

    if (!UintFromSeqIndex (seq, 0, &u))
        ERROR ("Mismatch in UintFromSeqIndex (0)");
    if (u != 2)
        ERROR ("Mismatch in UintFromSeqIndex result 0");
    if (!UintFromSeqIndex (seq, 1, &u))
        ERROR ("Mismatch in UintFromSeqIndex (1)");
    if (u != 4)
        ERROR ("Mismatch in UintFromSeqIndex result 1");

    if (!PointFromObject (seq, &i, &j))
        ERROR ("Mismatch in PointFromObject");
    if (i != 2 || j != 4)
        ERROR ("Mismatch in PointFromObject result");

    if (!SizeFromObject (seq, &sw, &sh))
        ERROR ("Mismatch in SizeFromObject");
    if (sw != 2 || sh != 4)
        ERROR ("Mismatch in SizeFromObject result");

    if (!FPointFromObject (seq, &d, &e))
        ERROR ("Mismatch in FPointFromObject");
    if (d != 2.f || e != 4.f)
        ERROR ("Mismatch in FPointFromObject result");

    if (!FSizeFromObject (seq, &d, &e))
        ERROR ("Mismatch in FSizeFromObject");
    if (d != 2.f || e != 4.f)
        ERROR ("Mismatch in FSizeFromObject result");
    Py_DECREF (seq);

    string = PyString_FromString ("Hello World!");
    if (!ASCIIFromObject (string, &str, &tmp))
        ERROR ("Mismatch in ASCIIFromObject");
    if (strcmp (str, "Hello World!") != 0)
        ERROR ("Mismatch in ASCIIFromObject result");
    Py_XDECREF (tmp);

    if (!UTF8FromObject (string, &str, &tmp))
        ERROR ("Mismatch in UTF8FromObject");
    if (strcmp (str, "Hello World!") != 0)
        ERROR ("Mismatch in UTF8FromObject result");
    Py_XDECREF (tmp);
}

static void
test_colors (void)
{
    PyObject *color;
    pgbyte rgba[4] = { 255, 155, 55, 5 };
    pguint32 rgba_int = 0x05ff9b37;
    pguint32 tmp = 0;

    color = PyColor_New (rgba);
    if (!PyColor_Check (color))
        ERROR ("Color mismatch in PyColor_Check");

    if (((PyColor*)color)->r != 255 ||
        ((PyColor*)color)->g != 155 ||
        ((PyColor*)color)->b != 55 ||
        ((PyColor*)color)->a != 5)
        ERROR ("Color mismatch in PyColor_New");

    tmp = PyColor_AsNumber (color);
    if (tmp != rgba_int)
        ERROR("Color mismatch in PyColor_AsNumber");
    Py_DECREF (color);

    rgba_int = 0xFF00FF00;
    color = PyColor_NewFromNumber (rgba_int);
    tmp = PyColor_AsNumber (color);
    if (tmp != rgba_int)
        ERROR("Color mismatch in PyColor_NewFromNumber");
    Py_DECREF (color);
}

static void
test_rect (void)
{
    PyObject *rect;
    double dx = 0, dy = 0, dw = 0, dh = 0;
    pgint32 x = 0, y = 0;
    pgint32 w = 0, h = 0;

    rect = PyRect_New (1, 2, 3, 4);
    if (!PyRect_Check (rect))
        ERROR ("Rect mismatch in PyRect_Check");

    if (((PyRect*)rect)->x != 1 ||
        ((PyRect*)rect)->y != 2 ||
        ((PyRect*)rect)->w != 3 ||
        ((PyRect*)rect)->h != 4)
        ERROR ("Rect mismatch in PyRect_New");

    if (!SizeFromObject (rect, &w, &h))
        ERROR ("Mismatch in SizeFromObject for PyRect");
    if (w != 3 || h != 4)
        ERROR ("Mismatch in SizeFromObject result for PyRect");

    if (!PointFromObject (rect, (int*)&x, (int*)&y))
        ERROR ("Mismatch in PointFromObject for PyRect");
    if (x != 1 || y != 2)
        ERROR ("Mismatch in PointFromObject result for PyRect");

    if (!FSizeFromObject (rect, &dw, &dh))
        ERROR ("Mismatch in FSizeFromObject for PyRect");
    if (dw != 3 || dh != 4)
        ERROR ("Mismatch in FSizeFromObject result for PyRect");

    if (!FPointFromObject (rect, &dx, &dy))
        ERROR ("Mismatch in FPointFromObject for PyRect");
    if (dx != 1 || dy != 2)
        ERROR ("Mismatch in FPointFromObject result for PyRect");
    
    Py_DECREF (rect);
}

static void
test_frect (void)
{
    PyObject *rect;
    double x, y, w, h;
    pgint32 sw, sh;
    int ix, iy;

    rect = PyFRect_New (1.01f, 2.02f, 3.03f, 4.04f);
    if (!PyFRect_Check (rect))
        ERROR ("FRect mismatch in PyFRect_Check");

    if (((PyFRect*)rect)->x != 1.01f ||
        ((PyFRect*)rect)->y != 2.02f ||
        ((PyFRect*)rect)->w != 3.03f ||
        ((PyFRect*)rect)->h != 4.04f)
        ERROR ("FRect mismatch in PyFRect_New");

    if (!SizeFromObject (rect, &sw, &sh))
        ERROR ("Mismatch in SizeFromObject for PyFRect");
    if (sw != 3 || sh != 4)
        ERROR ("Mismatch in SizeFromObject result for PyFRect");

    if (!PointFromObject (rect, &ix, &iy))
        ERROR ("Mismatch in PointFromObject for PyFRect");
    if (ix != 1 || iy != 2)
        ERROR ("Mismatch in PointFromObject result for PyFRect");

    if (!FSizeFromObject (rect, &w, &h))
        ERROR ("Mismatch in FSizeFromObject for PyFRect");
    if (!NEAR_ZERO (w - 3.03) || !NEAR_ZERO (h - 4.04))
        ERROR ("Mismatch in FSizeFromObject result for PyFRect");

    if (!FPointFromObject (rect, &x, &y))
        ERROR ("Mismatch in FPointFromObject for PyFRect");
    if (!NEAR_ZERO (x - 1.01) || !NEAR_ZERO (y - 2.02))
        ERROR ("Mismatch in FPointFromObject result for PyFRect");
    
    Py_DECREF (rect);
}

static void
test_bufferproxy (void)
{
    PyObject *buf, *str;
    void *strbuf;

    str = PyString_FromString ("Hello World!");
    strbuf = PyString_AS_STRING (str);
    buf = PyBufferProxy_New (str, strbuf, PyString_GET_SIZE (str), NULL);
    if (!PyBufferProxy_Check (buf))
        ERROR ("BufferProxy mismatch in PyBufferProxy_Check");
    Py_DECREF (buf);
}

static void
test_surface (void)
{
    PyObject *surface;
    PySurface *sf;

    surface = PySurface_New ();
    if (!PySurface_Check (surface))
        ERROR ("Surface mismatch in PySurface_Check");
    sf = (PySurface*) surface;
    if (!sf->get_width || !sf->get_height || !sf->get_size ||
        !sf->get_pixels || !sf->blit || !sf->copy)
        ERROR ("Surface is not properly initialised");
    Py_DECREF (sf);
}

static void
test_font (void)
{
    PyObject *font;
    PyFont *ft;

    font = PyFont_New ();
    if (!PyFont_Check (font))
        ERROR ("Font mismatch in PyFont_Check");
    ft = (PyFont*) font;
    if (!ft->get_height || !ft->get_name || !ft->get_style ||
        !ft->set_style || !ft->get_size || !ft->render || !ft->copy)
        ERROR ("Font is not properly initialised");
    Py_DECREF (ft);
}

int
main (int argc, char *argv[])
{
    Py_Initialize ();
    if (import_pygame2_base () == -1)
        ERROR("Could not import pygame2.base");
    
    test_helpers ();
    test_colors ();
    test_rect ();
    test_frect ();
    test_bufferproxy ();
    test_surface ();
    test_font ();
    Py_Finalize ();
    return 0;
}
