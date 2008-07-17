#include "pgVector2.h"
#include <assert.h>

int is_zero(double num)
{
	return fabs(num) <= ZERO_EPSILON;
}

int is_equal(double a, double b)
{
	double rErr;
	if(is_zero(a - b)) return 1;

	if(fabs(b) > fabs(a))
		rErr = fabs((a - b) / b);
	else
		rErr = fabs((a - b) / a);

	return rErr <= RELATIVE_ZERO;
}

int less_equal(double a, double b)
{
	return a < b || is_equal(a, b);
}

int more_equal(double a, double b)
{
	return a > b || is_equal(a, b);
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
	assert(d != 0);
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

pgVector2 c_fcross(double a, pgVector2 b)
{
	pgVector2 ans;
	ans.real = -a*b.imag;
	ans.imag = a*b.real;
	return ans;
}

pgVector2 c_crossf(pgVector2 a, double b)
{
	pgVector2 ans;
	ans.real = a.imag*b;
	ans.imag = -a.real*b;
	return ans;
}

void c_rotate(pgVector2* a, double seta)
{
	double x = a->real;
	double y = a->imag;
	a->real = x*cos(seta) - y*sin(seta);
	a->imag = x*sin(seta) + y*cos(seta);
}

int c_equal(pgVector2* a, pgVector2* b)
{
	return is_equal(a->real, b->real) && is_equal(a->imag, b->imag);
}


pgVector2 c_project(pgVector2 l,pgVector2 p)
{
	double lp;
	c_normalize(&l);
	lp = c_dot(l,p);
	return c_mul_complex_with_real(l,lp);
}


