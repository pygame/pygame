#include "pgVector2.h"
#include <assert.h>
#include <math.h>

int is_zero(double num)
{
	return fabs(num) < ZERO_EPSILON;
}

int is_equal(double a, double b)
{
	return is_zero(a - b);
}

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

Py_complex c_div_complex_with_real(pgVector2 c,double d)
{
	Py_complex r;
	r.real = c.real / d;
	r.imag = c.imag / d;
	return r;
}

void c_normalize(pgVector2* pVec)
{
	double l = c_get_length(*pVec);
	assert(l > 0);
	pVec->real /= l;
	pVec->imag /= l;
}

double c_dot(pgVector2 a,pgVector2 b)
{
	return a.real * b.real + a.imag * b.imag;
}

double c_cross(pgVector2 a, pgVector2 b)
{
	return a.real*b.imag - a.imag*b.real;
}

void c_rotate(pgVector2* a, double seta)
{
	double x = a->real;
	double y = a->imag;
	a->real = x*cos(seta) - y*sin(seta);
	a->imag = x*sin(seta) + y*cos(seta);
}
