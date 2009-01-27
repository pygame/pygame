#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pygame2/pgmask.h>

#define ERROR(x)                                \
    {                                           \
        fprintf(stderr, "*** %s\n", x);         \
        PyErr_Print ();                         \
        Py_Finalize ();                         \
        exit(1);                                \
    }

static void
test_mask (void)
{
    PyObject *mask;
    
    mask = PyMask_New (10, 10);
    if (!PyMask_Check (mask))
        ERROR ("Mask mismatch in PyMask_Check");
    Py_DECREF (mask);
}

int
main (int argc, char *argv[])
{
    Py_Initialize ();
    if (import_pygame2_base () == -1)
        ERROR("Could not import pygame2.base");
    if (import_pygame2_mask () == -1)
        ERROR("Could not import pygame2.mask");
    
    test_mask ();
    Py_Finalize ();
    return 0;
}
