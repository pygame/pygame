#!/usr/bin/env python

"""Draw a cube on the screen. every frame we orbit
the camera around by a small amount and it appears
the object is spinning. note i've setup some simple
data structures here to represent a multicolored cube,
we then go through a semi-unopimized loop to draw
the cube points onto the screen. opengl does all the
hard work for us. :]
"""

import pygame
from pygame.locals import *

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
except ImportError:
    print ('The GLCUBE example requires PyOpenGL')
    raise SystemExit



#some simple data for a colored cube
#here we have the 3D point position and color
#for each corner. then we have a list of indices
#that describe each face, and a list of indieces
#that describes each edge


CUBE_POINTS = (
    (0.5, -0.5, -0.5),  (0.5, 0.5, -0.5),
    (-0.5, 0.5, -0.5),  (-0.5, -0.5, -0.5),
    (0.5, -0.5, 0.5),   (0.5, 0.5, 0.5),
    (-0.5, -0.5, 0.5),  (-0.5, 0.5, 0.5)
)

#colors are 0-1 floating values
CUBE_COLORS = (
    (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 0),
    (1, 0, 1), (1, 1, 1), (0, 0, 1), (0, 1, 1)
)

CUBE_QUAD_VERTS = (
    (0, 1, 2, 3), (3, 2, 7, 6), (6, 7, 5, 4),
    (4, 5, 1, 0), (1, 5, 7, 2), (4, 0, 3, 6)
)

CUBE_EDGES = (
    (0,1), (0,3), (0,4), (2,1), (2,3), (2,7),
    (6,3), (6,4), (6,7), (5,1), (5,4), (5,7),
)



def drawcube():
    "draw the cube"
    allpoints = list(zip(CUBE_POINTS, CUBE_COLORS))

    glBegin(GL_QUADS)
    for face in CUBE_QUAD_VERTS:
        for vert in face:
            pos, color = allpoints[vert]
            glColor3fv(color)
            glVertex3fv(pos)
    glEnd()

    glColor3f(1.0, 1.0, 1.0)
    glBegin(GL_LINES)
    for line in CUBE_EDGES:
        for vert in line:
            pos, color = allpoints[vert]
            glVertex3fv(pos)

    glEnd()


def main():
    "run the demo"
    #initialize pygame and setup an opengl display
    pygame.init()
    pygame.display.set_mode((640,480), OPENGL|DOUBLEBUF)
    glEnable(GL_DEPTH_TEST)        #use our zbuffer

    #setup the camera
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45.0,640/480.0,0.1,100.0)    #setup lens
    glTranslatef(0.0, 0.0, -3.0)                #move back
    glRotatef(25, 1, 0, 0)                       #orbit higher


    while 1:
        #check for quit'n events
        event = pygame.event.poll()
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            break

        #clear screen and move camera
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        #orbit camera around by 1 degree
        glRotatef(1, 0, 1, 0)                    

        drawcube()
        pygame.display.flip()
        pygame.time.wait(10)


if __name__ == '__main__': main()
