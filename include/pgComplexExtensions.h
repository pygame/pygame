#ifndef _PYGAME_COMPLEX_EXTENSIONS_
#define _PYGAME_COMPLEX_EXTENSIONS_


#include <Python.h>

double c_get_length_square(Py_complex c);
double c_get_length(Py_complex c);
Py_complex c_mul_complex_with_real(Py_complex c,double d);

#endif //_PYGAME_COMPLEX_EXTENSIONS_