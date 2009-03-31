#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pygame2/pgphysics.h>

#define ERROR(x)                                \
    {                                           \
        fprintf(stderr, "*** %s\n", x);         \
        PyErr_Print ();                         \
        Py_Finalize ();                         \
        exit(1);                                \
    }

static void
test_vector (void)
{
    PyVector2 v, r, t, u;
    
    PyVector2_Set (v, 0, 0);
    PyVector2_Set (r, 1, 1);
    PyVector2_Set (t, 2, 2);

    if (v.real != 0 || v.imag != 0)
        ERROR ("Mismatch in PyVector2_Set");

    if (PyVector2_GetLengthSquare (v) != 0)
        ERROR ("Mismatch in PyVector2_GetLengthSquare (0,0)");
    if (PyVector2_GetLengthSquare (r) != 2)
        ERROR ("Mismatch in PyVector2_GetLengthSquare (1,1)");
    if (PyVector2_GetLengthSquare (t) != 8)
        ERROR ("Mismatch in PyVector2_GetLengthSquare (2,2)");

    if (PyVector2_GetLength (v) != 0)
        ERROR ("Mismatch in PyVector2_GetLength (0,0)");
    if (!IS_NEAR_ZERO (PyVector2_GetLength (r) - sqrt (2)))
        ERROR ("Mismatch in PyVector2_GetLength (1,1)");
    if (!IS_NEAR_ZERO (PyVector2_GetLength (t) - sqrt (8)))
        ERROR ("Mismatch in PyVector2_GetLength (2,2)");
    
    if (PyVector2_Dot (v, r) != 0)
        ERROR ("Mismatch in PyVector2_Dot (0,0, 1,1)");
    if (PyVector2_Dot (r, t) != 4)
        ERROR ("Mismatch in PyVector2_Dot (1,1, 2,2)");

    if (PyVector2_Cross (v, r) != 0)
        ERROR ("Mismatch in PyVector2_Cross (0,0, 1,1)");
    if (PyVector2_Cross (r, t) != 0)
        ERROR ("Mismatch in PyVector2_Cross (1,1, 2,2)");

    u = r;
    PyVector2_Normalize (&u);
    if (!IS_NEAR_ZERO (u.real - sqrt (2) / 2) ||
        !IS_NEAR_ZERO (u.imag - sqrt (2) / 2))
        ERROR ("Mismatch in PyVector2_Normalize (1,1)");
    u = t;
    PyVector2_Normalize (&u);
    if (!IS_NEAR_ZERO (u.real - sqrt (2) / 2) ||
        !IS_NEAR_ZERO (u.imag - sqrt (2) / 2))
        ERROR ("Mismatch in PyVector2_Normalize (2,2)");

    u = v;
    PyVector2_Rotate (&u, DEG2RAD (180.f));
    if (u.real != 0 || u.imag != 0)
        ERROR ("Mismatch in PyVector2_Rotate (0,0)");
    u = r;
    PyVector2_Rotate (&u, DEG2RAD (180.f));
    if (!IS_NEAR_ZERO (u.real + 1.f) || !IS_NEAR_ZERO (u.imag + 1.f))
        ERROR ("Mismatch in PyVector2_Rotate (1,1)");
    u = t;
    PyVector2_Rotate (&u, DEG2RAD (180.f));
    if (!IS_NEAR_ZERO (u.real + 2.f) || !IS_NEAR_ZERO (u.imag + 2.f))
        ERROR ("Mismatch in PyVector2_Rotate (2,2)");

    u = t;
    if (!PyVector2_Equal (u, t))
        ERROR ("Mismatch in PyVector2_Equal (2,2, 2,2)");

    u = PyVector2_MultiplyWithReal (v, 5.f);
    if (u.real != 0.f || u.imag != 0.f)
        ERROR ("Mismatch in PyVector2_MultiplyWithReal (0,0, 5)");
    u = PyVector2_MultiplyWithReal (r, 3.f);
    if (u.real != 3.f || u.imag != 3.f)
        ERROR ("Mismatch in PyVector2_MultiplyWithReal (1,1, 3)");
    u = PyVector2_MultiplyWithReal (t, 2.5f);
    if (u.real != 5.f || u.imag != 5.f)
        ERROR ("Mismatch in PyVector2_MultiplyWithReal (2,2, 2.5)");

    u = PyVector2_DivideWithReal (v, 2.f);
    if (u.real != 0.f || u.imag != 0.f)
        ERROR ("Mismatch in PyVector2_DivideWithReal (0,0, 2)");
    u = PyVector2_DivideWithReal (r, 2.f);
    if (u.real != 0.5f || u.imag != 0.5f)
        ERROR ("Mismatch in PyVector2_DivideWithReal (1,1, 2)");
    u = PyVector2_DivideWithReal (t, .5f);
    if (u.real != 4.f || u.imag != 4.f)
        ERROR ("Mismatch in PyVector2_DivideWithReal (2,2, 0.5)");
    
    /*
      TODO:

      PyVector2_fCross ()
      PyVector2_Crossf ()
      PyVector2_Project ()
      PyVector2_AsTuple ()
      PyVector2_FromSequence ()
      PyVector2_Transform ()
      PyVector2_TransformMultiple ()
    */
}

static void
test_math (void)
{
    double a, a2, b, c;
    
    a = 1.f;
    a2 = 1.000001f;
    b = 1.1f;
    c = .9f;
    
    if (PyMath_IsNearEqual (a, b))
        ERROR ("Mismatch in PyMath_IsNearEqual (1,1.1)");
    if (PyMath_IsNearEqual (a, c))
        ERROR ("Mismatch in PyMath_IsNearEqual (1,.9)");
    if (!PyMath_IsNearEqual (a, a2))
        ERROR ("Mismatch in PyMath_IsNearEqual (1,1.000001)");

    if (!PyMath_LessEqual (a, b))
        ERROR ("Mismatch in PyMath_LessEqual (1,1.1)");
    if (PyMath_LessEqual (a, c))
        ERROR ("Mismatch in PyMath_LessEqual (1,.9)");
    if (!PyMath_LessEqual (a, a2))
        ERROR ("Mismatch in PyMath_LessEqual (1,1.000001)");

    if (PyMath_MoreEqual (a, b))
        ERROR ("Mismatch in PyMath_MoreEqual (1,1.1)");
    if (!PyMath_MoreEqual (a, c))
        ERROR ("Mismatch in PyMath_MoreEqual (1,.9)");
    if (!PyMath_MoreEqual (a, a2))
        ERROR ("Mismatch in PyMath_MoreEqual (1,1.000001)");
}

static void
test_aabbox (void)
{
    AABBox a, b, c;
    PyVector2 v;
    PyObject *rect, *seq;

    a = AABBox_New (1, 2, 3, 4);
    b = AABBox_New (2, 4, 6, 8);

    if (a.left != 1 || a.right != 2 || a.bottom != 3 || a.top != 4 ||
        b.left != 2 || b.right != 4 || b.bottom != 6 || b.top != 8)
        ERROR ("Mismatch in AABBox_New");

    AABBox_Reset (&b);
    if (b.left != DBL_MAX || b.right != -DBL_MAX || 
        b.bottom != -DBL_MAX || b.top != DBL_MAX)
        ERROR ("Mismatch in AABBox_Reset");

    PyVector2_Set (v, 10, 10);
    AABBox_ExpandTo (&a, &v);
    if (a.left != 1 || a.right != 10 || a.bottom != 10 || a.top != 4)
        ERROR ("Mismatch in AABBox_ExpandTo");

    b = AABBox_New (2, 4, 6, 8);
    if (!AABBox_Overlaps (&a, &b, 0.f))
        ERROR ("Mismatch in AABBox_Overlaps");
    b = AABBox_New (1, 4, 6, 8);
    if (!AABBox_Overlaps (&a, &b, .2f))
        ERROR ("Mismatch in AABBox_Overlaps");

    PyVector2_Set (v, 1.2, 8);
    if (!AABBox_Contains (&a, &v, 0.f))
        ERROR ("Mismatch in AABBox_Contains (1.2, 8)");
    PyVector2_Set (v, .8, 8);
    if (!AABBox_Contains (&a, &v, 0.3f))
        ERROR ("Mismatch in AABBox_Contains (.8, 8)");

    rect = AABBox_AsFRect (&a);
    if (!rect || !PyFRect_Check (rect))
        ERROR ("Mismatch in AABBox_AsFRect return value");
    if (((PyFRect*)rect)->x != 1 || ((PyFRect*)rect)->y != 4 ||
        ((PyFRect*)rect)->w != 9 || ((PyFRect*)rect)->h != 6)
        ERROR ("Mismatch in AABBox_AsFRect");

    seq = PyTuple_New (4);
    PyTuple_SET_ITEM (seq, 0, PyLong_FromLong (1));
    PyTuple_SET_ITEM (seq, 1, PyLong_FromLong (2));
    PyTuple_SET_ITEM (seq, 2, PyLong_FromLong (3));
    PyTuple_SET_ITEM (seq, 3, PyLong_FromLong (4));
    if (!AABBox_FromSequence (seq, &c))
        ERROR ("Mismatch in AABBox_FromSequence creation");
    if (c.left != 1 || c.top != 2 || c.right != 4 || c.bottom != 6)
        ERROR ("Mismatch in AABBox_FromSequence");

    if (!AABBox_FromRect (rect, &c))
        ERROR ("Mismatch in AABBox_FromRect creation");
    if (!IS_NEAR_ZERO (c.top - 4.f) || !IS_NEAR_ZERO (c.left - 1.f) ||
        !IS_NEAR_ZERO (c.bottom - 10.f) || !IS_NEAR_ZERO (c.right - 10.f))
        ERROR ("Mismatch in AABBox_FromRect");
    
    Py_DECREF (rect);
}

static void
test_shape (void)
{
    PyVector2 *vertices;
    Py_ssize_t count;
    PyObject *shape1;
    PyRectShape *rshape1;
    AABBox box = AABBox_New (10, 20, 20, 10);

    shape1 = PyRectShape_New (box);
    if (!shape1 || !PyRectShape_Check (shape1))
        ERROR ("Mismatch in PyRectShape_New");
    rshape1 = (PyRectShape *)shape1;

    if (rshape1->topleft.real != 10 || rshape1->topleft.imag != 10 || 
        rshape1->topright.real != 20 || rshape1->topright.imag != 10 || 
        rshape1->bottomleft.real != 10 || rshape1->bottomleft.imag != 20 || 
        rshape1->bottomright.real != 20 || rshape1->bottomright.imag != 20)
        ERROR ("Mismatch in PyRectShape_New box assignment");

    vertices = PyShape_GetVertices (shape1, &count);
    if (!vertices || count != 4)
        ERROR ("Mismatch in PyShape_GetVertices");
    if (vertices[0].real != 10 || vertices[0].imag != 20 ||
        vertices[1].real != 20 || vertices[1].imag != 20 ||
        vertices[2].real != 20 || vertices[2].imag != 10 ||
        vertices[3].real != 10 || vertices[3].imag != 10)
        ERROR ("Mismatch in PyShape_GetVertices assignment");
    PyMem_Free (vertices);

    vertices = PyShape_GetVertices_FAST ((PyShape*)shape1, &count);
    if (!vertices || count != 4)
        ERROR ("Mismatch in PyShape_GetVertices_FAST");
    if (vertices[0].real != 10 || vertices[0].imag != 20 ||
        vertices[1].real != 20 || vertices[1].imag != 20 ||
        vertices[2].real != 20 || vertices[2].imag != 10 ||
        vertices[3].real != 10 || vertices[3].imag != 10)
        ERROR ("Mismatch in PyShape_GetVertices_FAST assignment");
    PyMem_Free (vertices);
    Py_DECREF (shape1);

}

int
main (int argc, char *argv[])
{
    Py_Initialize ();
    if (import_pygame2_base () == -1)
        ERROR("Could not import pygame2.base");
    if (import_pygame2_physics () == -1)
        ERROR("Could not import pygame2.physics");

    test_vector ();
    test_math ();
    test_aabbox ();
    test_shape ();
    Py_Finalize ();
    return 0;
}

