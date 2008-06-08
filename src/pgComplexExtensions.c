#include "pgComplexExtensions.h"

double c_get_length_square(Py_complex c)
{
	double r;
	r = c.real * c.real;
	r += (c.imag * c.imag);
	return r;
}

double c_get_length(Py_complex c)
{
	double r;
	r = c.real * c.real;
	r += (c.imag * c.imag);
	return sqrt(r);
}

Py_complex c_mul_complex_with_real(Py_complex c,double d)
{
	Py_complex r;
	r.real = c.real * d;
	r.imag = c.imag * d;
	return r;
}

