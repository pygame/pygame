#include "glTestSuite.h"
#include <time.h>
#include <string.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef WIN32
#pragma warning(disable:4996)
#endif

#define MAX_STR_LEN 1000

//跑表
static double s_time;

void watch_start()
{
	s_time = (double)clock();
}

//计算从跑表打开到停止共经历了多少秒
double watch_stop()
{
	return ((double)clock() - s_time)/CLOCKS_PER_SEC;
}

//渲染文字, 仅限英文，(x, y)是文字矩阵左上角坐标
//每个字符字符宽9像素，高15像素
void draw_text(int x, int y, char *str)
{
	int len, i, w, h;

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	w = glutGet(GLUT_WINDOW_WIDTH);
	h = glutGet(GLUT_WINDOW_HEIGHT);
	gluOrtho2D(0, w, h, 0);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glColor3f(0.6f, 0.8f, 0.6f);
	glRasterPos2i(x, y + 15);
	len = (int) strlen(str);
	for (i = 0; i < len; i++)
		glutBitmapCharacter(GLUT_BITMAP_9_BY_15, str[i]);

	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
}

//打印文字到屏幕，类似printf，推荐
void glprintf(int x, int y, const char* fmt, ...)
{
	static char buf[MAX_STR_LEN];
	va_list p;
	va_start(p, fmt);
	memset(buf, 0, sizeof(buf));
	vsprintf(buf, fmt, p);
	draw_text(x, y, buf);
}
