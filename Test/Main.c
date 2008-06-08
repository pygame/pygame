#include <GL/glut.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <stdarg.h>
#include <stdio.h>

#define HEIGHT 600
#define WIDTH 600
#define MAX_STR_LEN 1000

/*-------------------------------测试工具----------------------------*/

#include "pgPhysicsRenderer.h"

static pgWorldObject* s_world = NULL;
static double s_updateTime = 0.0;

//跑表
static double s_time;
static int s_first = 1;

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





/*-------------------------------测试函数----------------------------*/

//渲染函数
void do_render()
{
	//glprintf(0, 0, "Your RP value is: %d", 0);
	glColor3f(1.f, 1.f, 1.f);
	/*glBegin(GL_LINES);
	glVertex2f(-10, 0);
	glVertex2f(10, 0);
	glEnd();*/
	PGT_RenderWorld(s_world);

}



//keyboard输入响应
void keyboard (unsigned char key, int x, int y)
{
	switch(key)
	{
	case 27:
		exit(0);
		break;
	}
}





/*-------------------------------设置函数----------------------------*/


void InitGL()
{
	glShadeModel(GL_SMOOTH);	
	glPointSize(3.f);
	glEnable(GL_LINE_SMOOTH);
	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glHint(GL_LINE_SMOOTH_HINT, GL_DONT_CARE);
	glHint(GL_POINT_SMOOTH_HINT, GL_DONT_CARE);
	glLineWidth(2.5f);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-WIDTH/2, WIDTH/2, -HEIGHT/2, HEIGHT/2, -1.f, 1.f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}


void display(void)
{
	double dt;
	//watch_start();

	

	if(s_first)
	{
		watch_start();
		s_first = 0;
		return;
	}
	dt = watch_stop();
	/*s_updateTime += dt;
	if(s_updateTime >= s_world->fStepTime)
	{
		PG_Update(s_world,s_updateTime);
		s_updateTime = 0.0;

		glClear(GL_COLOR_BUFFER_BIT);
		glLoadIdentity();
		do_render();
		glutSwapBuffers();
	}*/
	PG_Update(s_world,dt);
	watch_start();
	glClear(GL_COLOR_BUFFER_BIT);
	glLoadIdentity();
	do_render();
	glutSwapBuffers();

	
	
}

//这个函数一开始就会被调用，故gluPerspective函数没必要在initGL或者display函数里调用
void reshape (int width , int height)
{
	if(height == 0)										
		height = 1;										

	glViewport(0,0,width,height);						
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-width/2, width/2, -height/2, height/2, -1.f, 1.f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();									
}

void InitWorld()
{
	pgBodyObject* body;
	s_world = PG_WorldNew();
	s_world->fStepTime = 0.03;
	body = PG_BodyNew();
	body->vecPosition.real = 0;
	body->vecPosition.imag = 0;
	body->vecLinearVelocity.real = 10;
	PG_AddBodyToWorld(s_world,body);
}


int main (int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(WIDTH, HEIGHT);
	glutCreateWindow("test physics");
	InitWorld();
	InitGL();
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutIdleFunc(display);
	glutMainLoop();
	return 0;
}

