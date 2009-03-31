#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pygame2/pgsdl.h>

#define ERROR(x)                                \
    {                                           \
        fprintf(stderr, "*** %s\n", x);         \
        PyErr_Print ();                         \
        Py_Finalize ();                         \
        exit(1);                                \
    }
#define NEAR_ZERO(x) (fabs(x) <= 1e-6)

static void
test_surface (void)
{
    SDL_Surface *sdlsf;
    PyObject *sf1, *sf2;
    PyObject *r1, *r2;

    SDL_Init(SDL_INIT_VIDEO);
    SDL_EnableUNICODE (1);

    sdlsf = SDL_SetVideoMode (640, 480, 0, 0);

    sf1 = PySDLSurface_NewFromSDLSurface (sdlsf);
    if (!sf1)
        ERROR ("sf1");
    
    sf2 = PySDLSurface_New (100, 100);
    if (!sf2)
        ERROR ("sf2");

    r1 = PyRect_New (0, 0, 640 ,480);
    r2 = PyRect_New (0, 0, 100 ,100);
    PyObject_CallMethod (sf1, "blit", "OOOi", sf2, r1, r2, 0x4);

    Py_DECREF (sf1);
    Py_DECREF (sf2);
    Py_DECREF (r1);
    Py_DECREF (r2);
}

int
main (int argc, char *argv[])
{
    Py_Initialize ();
    if (import_pygame2_base () == -1)
        ERROR("Could not import pygame2.base");
    if (import_pygame2_sdl_video () == -1)
        ERROR("Could not import pygame2.video");
    
    test_surface ();
    Py_Finalize ();
    return 0;
}
