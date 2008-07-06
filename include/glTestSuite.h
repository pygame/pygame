#ifndef _PG_TEST_SUITE_H
#define _PG_TEST_SUITE_H

#include <GL/glut.h>

void watch_start();
double watch_stop();
void draw_text(int x, int y, char *str);
void glprintf(int x, int y, const char* fmt, ...);

#endif
