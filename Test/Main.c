#include <GL/glut.h>

#define HEIGHT 600
#define WIDTH 600

/*-------------------------------测试工具----------------------------*/

#include "pgPhysicsRenderer.h"
#include "pgBodyObject.h"
#include "glTestSuite.h"

static pgWorldObject* s_world = NULL;

/*-------------------------------测试函数----------------------------*/

pgBodyObject* body, * body1;

//渲染函数
void do_render()
{
	glColor3f(1.f, 1.f, 1.f);
	PG_Update(s_world, 0.005);
	PGT_RenderWorld(s_world);
	//glprintf(0, 0, "Velocity of body: (%.2f, %.2f)", body->vecLinearVelocity.real, 
	//	body->vecLinearVelocity.imag);
	//glprintf(0, 20, "w of body: %d", body->fAngleVelocity);
}



//keyboard输入响应
void keyboard (unsigned char key, int x, int y)
{
	switch(key)
	{
	case 27:
		PG_WorldDestroy(s_world);
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

//============================================

void TestBasic1Init()
{
	pgBodyObject* body;
	pgJointObject* joint;
	pgVector2 a1,a2;
	PG_Set_Vector2(a1,0,0);
	PG_Set_Vector2(a2,0,100);

	s_world = PG_WorldNew();
	s_world->fStepTime = 0.03;
	body = PG_BodyNew();
	PG_Bind_RectShape(body, 20, 20, 0);
	PG_Set_Vector2(body->vecPosition,0,0)
	PG_Set_Vector2(body->vecLinearVelocity, 40, 0)
	PG_AddBodyToWorld(s_world,body);
	
	
	joint = PG_DistanceJointNew(body,NULL,0,100,a1,a2);
	PG_AddJointToWorld(s_world,joint);
}
//test collision
void TestBasic2Init()
{
	s_world = PG_WorldNew();
	s_world->fStepTime = 0.03;

	body = PG_BodyNew();
	PG_Set_Vector2(body->vecPosition, -50, 0);
	PG_Set_Vector2(body->vecLinearVelocity, 16.f, -400.f);
	body->fRotation = M_PI/4;
	body->fAngleVelocity = 5.f;
	body->fRestitution = 1.f;
	PG_Bind_RectShape(body, 30, 30, 0);
	PG_AddBodyToWorld(s_world, body);
	
	body1 = PG_BodyNew();
	PG_Set_Vector2(body1->vecPosition,0, -100);
	body1->bStatic = 1;
	body1->fRestitution = 1.f;//for test
	body1->fMass = 1e24;
	PG_Bind_RectShape(body1, 500, 20, 0);
	PG_AddBodyToWorld(s_world, body1);

}

void TestBasic3Init()
{
	pgBodyObject* body1,*body2;
	pgJointObject* joint;
	pgVector2 a1,a2;
	PG_Set_Vector2(a1,0,0);
	PG_Set_Vector2(a2,0,0);

	s_world = PG_WorldNew();
	s_world->fStepTime = 0.03;
	body1 = PG_BodyNew();
	PG_Bind_RectShape(body1, 20, 20, 0);
	PG_Set_Vector2(body1->vecPosition,0,0)
	PG_Set_Vector2(body1->vecLinearVelocity,10,0)
	PG_AddBodyToWorld(s_world,body1);

	body2 = PG_BodyNew();
	PG_Bind_RectShape(body2, 20, 20, 0);
	PG_Set_Vector2(body2->vecPosition,0,100)
	PG_Set_Vector2(body2->vecLinearVelocity,0,0)
	PG_AddBodyToWorld(s_world,body2);


	joint = PG_DistanceJointNew(body1,body2,0,100,a1,a2);
	PG_AddJointToWorld(s_world,joint);
}

void TestBasic4Init()
{
	#define BODY_NUM 3

	int i;
	pgBodyObject* body[BODY_NUM + 1];
	pgJointObject* joint[BODY_NUM];
	pgVector2 a1,a2;
	PG_Set_Vector2(a1,0,0);
	PG_Set_Vector2(a2,0,100);

	s_world = PG_WorldNew();
	s_world->fStepTime = 0.03;

	body[0] = NULL;
	for (i = 1;i < BODY_NUM + 1;i++)
	{
		body[i] = PG_BodyNew();
		PG_Bind_RectShape(body[i], 20, 20, 0);
		PG_Set_Vector2(body[i]->vecPosition,0,(-i*50 + 100))
		PG_Set_Vector2(body[i]->vecLinearVelocity,0,0)
		PG_AddBodyToWorld(s_world,body[i]);
	}

	body1 = PG_BodyNew();
	PG_Set_Vector2(body1->vecPosition,50, 0);
	body1->bStatic = 1;
	PG_Bind_RectShape(body1, 20, 300, 0);
	PG_AddBodyToWorld(s_world, body1);

	PG_Set_Vector2(body[BODY_NUM]->vecLinearVelocity,100,0)

	i = 0;
	joint[i] = PG_DistanceJointNew(body[i+1],body[i],0,50,a1,a2);
	PG_AddJointToWorld(s_world,joint[i]);
	for (i = 1;i < BODY_NUM;i++)
	{
		joint[i] = PG_DistanceJointNew(body[i],body[i+1],0,50,a1,a2);
		PG_AddJointToWorld(s_world,joint[i]);
	}


}

//===============================================

void InitWorld()
{
	TestBasic2Init();
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
