#include <stdio.h>
#ifdef WIN32
#include <windows.h>
#endif
#include <GL/glut.h>
#include <math.h>

#define HEIGHT 800
#define WIDTH 800

/*-------------------------------测试工具----------------------------*/

#include <physics/pgphysics.h>
#include "pgPhysicsRenderer.h"

static PyObject* s_world = NULL;

/*-------------------------------测试函数----------------------------*/

/*PyObject* body, * body1;*/

//渲染函数
void do_render()
{
    glColor3f(1.f, 1.f, 1.f);
    PyWorld_Update(s_world, 0.2);
    PGT_RenderWorld((PyWorldObject*)s_world);
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
        Py_DECREF(s_world);
		Py_Finalize();
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
    PyObject* body;
    PyObject* joint;
    PyVector2 a1,a2;
    PyVector2_Set(a1,0,0);
    PyVector2_Set(a2,0,100);

    s_world = PyWorld_New();
    ((PyWorldObject*)s_world)->fStepTime = 0.03;
    body = PyBody_New();
    PyBody_SetShape (body, PyRectShape_New(20, 20, 0));
    PyVector2_Set(((PyBodyObject*)body)->vecPosition,0,0);
    PyVector2_Set(((PyBodyObject*)body)->vecLinearVelocity, 40, 0);
    PyWorld_AddBody(s_world,body);
    
    joint = PyDistanceJoint_New(body,NULL,0);
    PyDistanceJoint_SetAnchors (joint,a1,a2);
    PyWorld_AddJoint(s_world,joint);
}
//test collision
void TestBasic2Init()
{
    int i = 0;
    PyObject* body, *body1;
    s_world = PyWorld_New();
    ((PyWorldObject*)s_world)->fStepTime = 0.03;
    PyVector2_Set(((PyWorldObject*)s_world)->vecGravity, 0, -2000.f);

    for(i = 0; i < 10; ++i)
    {
        body = PyBody_New();
        PyVector2_Set(((PyBodyObject*)body)->vecPosition, 0, 400 - 40*i);
        
        ((PyBodyObject*)body)->fRotation = 0.f;
        ((PyBodyObject*)body)->fAngleVelocity = 0.f;
        ((PyBodyObject*)body)->fRestitution = 0.0f;
        ((PyBodyObject*)body)->fMass = 10;
        PyBody_SetShape (body, PyRectShape_New(30, 30, 0));
        PyWorld_AddBody(s_world, body);
    }

    body1 = PyBody_New();
    PyVector2_Set(((PyBodyObject*)body1)->vecPosition,0, -100);
    ((PyBodyObject*)body1)->bStatic = 1;
    ((PyBodyObject*)body1)->fRestitution = 1.f;//for test
    ((PyBodyObject*)body1)->fMass = 1e32;
    ((PyBodyObject*)body1)->fRotation = 0.f;
    PyBody_SetShape(body1, PyRectShape_New (1000, 20, 0));
    PyWorld_AddBody(s_world, body1);
}

void TestBasic3Init()
{
    PyObject* body1,*body2;
    PyObject* joint;
    PyVector2 a1,a2;
    PyVector2_Set(a1,0,0);
    PyVector2_Set(a2,0,0);

    s_world = PyWorld_New();
    ((PyWorldObject*)s_world)->fStepTime = 0.03;
    body1 = PyBody_New();
    PyBody_SetShape(body1, PyRectShape_New (20, 20, 0));
    PyVector2_Set(((PyBodyObject*)body1)->vecPosition,0,0);
    PyVector2_Set(((PyBodyObject*)body1)->vecLinearVelocity,10,0);
    PyWorld_AddBody(s_world,body1);

    body2 = PyBody_New();
    PyBody_SetShape(body2, PyRectShape_New (20, 20, 0));
    PyVector2_Set(((PyBodyObject*)body2)->vecPosition,0,100);
    PyVector2_Set(((PyBodyObject*)body2)->vecLinearVelocity,0,0);
    PyWorld_AddBody(s_world,body2);

    joint = PyDistanceJoint_New(body1,body2,0);
    PyDistanceJoint_SetAnchors (joint,a1,a2);
    PyWorld_AddJoint(s_world,joint);
}

void TestBasic4Init()
{
#define  BODY_NUM  1

    int i;
    PyObject* body[BODY_NUM + 1];
    PyObject* joint[BODY_NUM];
    PyVector2 a1,a2;
    PyVector2_Set(a1,5,0);
    PyVector2_Set(a2,0,100);

    s_world = PyWorld_New();
    ((PyWorldObject*)s_world)->fStepTime = 0.03;
    PyVector2_Set(((PyWorldObject*)s_world)->vecGravity,0,0);

	body[0] = NULL;
    for (i = 1;i < BODY_NUM + 1;i++)
    {
        body[i] = PyBody_New();
        PyBody_SetShape (body[i], PyRectShape_New (20, 20, 0));
        PyVector2_Set(((PyBodyObject*)body[i])->vecPosition,0,(-i*60 + 100));
        //PG_Set_Vector2(body[i]->vecLinearVelocity,20,0);
        PyWorld_AddBody(s_world,body[i]);
    }

    /*body1 = PG_BodyNew();
      PyVector2_Set(body1->vecPosition,50, 0);
      body1->bStatic = 1;
      PG_Bind_RectShape(body1, 20, 300, 0);
      PG_AddBodyToWorld(s_world, body1);*/

    PyVector2_Set(((PyBodyObject*)body[BODY_NUM])->vecLinearVelocity,40,0)

	i = 0;
    joint[i] = PyDistanceJoint_New(body[i+1],body[i],0);
    PyDistanceJoint_SetAnchors (joint[i], a1, a2);
    PyWorld_AddJoint(s_world,joint[i]);
    for (i = 1;i < BODY_NUM;i++)
    {
        joint[i] = PyDistanceJoint_New(body[i+1],body[i],0);
        PyDistanceJoint_SetAnchors (joint[i], a1, a1);
        PyWorld_AddJoint(s_world,joint[i]);
    }
#undef BODY_NUM

}

void TestBasic5Init()
{
#define  BODY_NUM  4

    int i;
    PyObject* body[BODY_NUM + 1];
    PyObject* joint[BODY_NUM];
    PyVector2 a1,a2;
    PyVector2_Set(a1,0,0);
    PyVector2_Set(a2,0,0);

    s_world = PyWorld_New();
    ((PyWorldObject*)s_world)->fStepTime = 0.03;
    PyVector2_Set(((PyWorldObject*)s_world)->vecGravity,0,10);

	for (i = 0;i < BODY_NUM;i++)
	{
        body[i] = PyBody_New();
        if (i != BODY_NUM - 1)
        {
            PyBody_SetShape(body[i], PyRectShape_New (20, 20, 0));
        }
        else
        {
            PyBody_SetShape(body[i], PyRectShape_New (20, 100, 0));
        }
        //PyVector2_Set(((PyBodyObject*)body[i])->vecLinearVelocity,50,0)
        PyWorld_AddBody(s_world,body[i]);
	}
    PyVector2_Set(((PyBodyObject*)body[0])->vecPosition,200,0);
	PyVector2_Set(((PyBodyObject*)body[1])->vecPosition,200,100);
	PyVector2_Set(((PyBodyObject*)body[2])->vecPosition,300,200);
	PyVector2_Set(((PyBodyObject*)body[3])->vecPosition,300,200);
	((PyBodyObject*)body[0])->bStatic = 1;
    ((PyBodyObject*)body[3])->bStatic = 1;

    //PyVector2_Set(((PyBodyObject*)body[BODY_NUM])->vecLinearVelocity,20,60);


    for (i = 0;i < BODY_NUM - 2;i++)
    {
        joint[i] = PyDistanceJoint_New(body[i],body[i+1],0);
        PyDistanceJoint_SetAnchors(joint[i], a1, a2);
        PyWorld_AddJoint(s_world,joint[i]);
    }

#undef BODY_NUM
}

void TestBasic6Init()
{
    PyObject* body[2];
    PyObject* joint;
    PyVector2 a1,a2;
    PyVector2_Set(a1,0,0);
    PyVector2_Set(a2,0,0);
    s_world = PyWorld_New();
    ((PyWorldObject*)s_world)->fStepTime = 0.03;
    PyVector2_Set(((PyWorldObject*)s_world)->vecGravity,0,0);

	body[0] = PyBody_New();
    PyBody_SetShape (body[0], PyRectShape_New (20, 20, 0));
    PyVector2_Set(((PyBodyObject*)body[0])->vecPosition,-50,0)
	PyWorld_AddBody(s_world,body[0]);

    body[1] = PyBody_New();
    PyBody_SetShape (body[1], PyRectShape_New (20, 20, 0));
    PyVector2_Set(((PyBodyObject*)body[1])->vecPosition,50,0);
	PyWorld_AddBody(s_world,body[1]);

    PyVector2_Set(((PyBodyObject*)body[0])->vecLinearVelocity,0,0);
	PyVector2_Set(((PyBodyObject*)body[1])->vecLinearVelocity,0,0);

	joint = PyDistanceJoint_New(body[0],body[1],0);
    PyDistanceJoint_SetAnchors (joint,a1, a2);
    PyWorld_AddJoint(s_world,joint);
}

void TestBasic7Init()
{
    PyObject* body[3];
    PyObject* joint;
    
    PyVector2 a1,a2;
    PyVector2_Set(a1,0,0);
    PyVector2_Set(a2,0,0);
    
    s_world = PyWorld_New();
    ((PyWorldObject*)s_world)->fStepTime = 0.03;
    PyVector2_Set(((PyWorldObject*)s_world)->vecGravity,0,-.1);

	body[0] = PyBody_New();
    //((PyBodyObject*)body[0])->bStatic = 1;
    PyBody_SetShape (body[0], PyRectShape_New (20, 20, 0));
    PyVector2_Set(((PyBodyObject*)body[0])->vecPosition,10,200)
	PyWorld_AddBody(s_world,body[0]);

    body[1] = PyBody_New();
    ((PyBodyObject*)body[1])->bStatic = 1;
    PyBody_SetShape (body[1], PyRectShape_New (20, 20, 0));
    PyVector2_Set(((PyBodyObject*)body[1])->vecPosition,0,100);
	PyWorld_AddBody(s_world,body[1]);

    body[2] = PyBody_New();
    PyBody_SetShape (body[2], PyRectShape_New (20, 20, 0));
    PyVector2_Set(((PyBodyObject*)body[2])->vecPosition,0,0);
	PyWorld_AddBody(s_world,body[2]);

    PyVector2_Set(((PyBodyObject*)body[0])->vecLinearVelocity,0,0);
	PyVector2_Set(((PyBodyObject*)body[1])->vecLinearVelocity,0,0);
    PyVector2_Set(((PyBodyObject*)body[2])->vecLinearVelocity,0,0);

	joint = PyDistanceJoint_New(body[0],body[1],0);
    PyDistanceJoint_SetAnchors (joint,a1, a2);
    PyWorld_AddJoint(s_world,joint);
    
	joint = PyDistanceJoint_New(body[1],body[2],0);
    PyDistanceJoint_SetAnchors (joint,a1, a2);
    PyWorld_AddJoint(s_world,joint);
}


//===============================================

void InitWorld()
{
    TestBasic7Init();
}

int main (int argc, char** argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("test physics");
    
	Py_Initialize();
    if (import_physics () == -1)
	{
		// Needed for the glut threading.
		Py_Finalize();
        return 0;
	}
	InitWorld();
    InitGL();
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutIdleFunc(display);
    glutMainLoop();
    return 0;
}
