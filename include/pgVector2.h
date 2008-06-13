#ifndef _PYGAME_COMPLEX_EXTENSIONS_
#define _PYGAME_COMPLEX_EXTENSIONS_


#include <Python.h>

#define ZERO_EPSILON 1e-7

typedef Py_complex	pgVector2;
#define PG_Set_Vector2(vec, x, y) {vec.real = x;vec.imag = y;}

int is_zero(double num);
int is_equal(double a, double b);

double c_get_length_square(pgVector2 c);
double c_get_length(pgVector2 c);
Py_complex c_mul_complex_with_real(pgVector2 c,double d);
Py_complex c_div_complex_with_real(pgVector2 c,double d);
void	c_normalize(pgVector2* pVec);
double c_dot(pgVector2 a,pgVector2 b);
double c_cross(pgVector2 a, pgVector2 b);
void c_rotate(pgVector2* a, double seta);

#endif //_PYGAME_COMPLEX_EXTENSIONS_
