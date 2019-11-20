#!/usr/bin/env python
""" pygame.examples.glcube

Draw a cube on the screen.



Amazing.

Every frame we orbit the camera around a small amount
creating the illusion of a spinning object.

First we setup some points of a multicolored cube. Then we then go through
a semi-unoptimized loop to draw the cube points onto the screen.

OpenGL does all the hard work for us. :]


Keyboard Controls
-----------------

* ESCAPE key to quit
* f key to toggle fullscreen.

"""
import pygame as pg

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
except ImportError:
    print("The GLCUBE example requires PyOpenGL")
    raise SystemExit


# Some simple data for a colored cube here we have the 3D point position and color
# for each corner. A list of indices describes each face, and a list of
# indicies describes each edge.


CUBE_POINTS = (
    (0.5, -0.5, -0.5),
    (0.5, 0.5, -0.5),
    (-0.5, 0.5, -0.5),
    (-0.5, -0.5, -0.5),
    (0.5, -0.5, 0.5),
    (0.5, 0.5, 0.5),
    (-0.5, -0.5, 0.5),
    (-0.5, 0.5, 0.5),
)

# colors are 0-1 floating values
CUBE_COLORS = (
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0),
    (0, 0, 0),
    (1, 0, 1),
    (1, 1, 1),
    (0, 0, 1),
    (0, 1, 1),
)

CUBE_QUAD_VERTS = (
    (0, 1, 2, 3),
    (3, 2, 7, 6),
    (6, 7, 5, 4),
    (4, 5, 1, 0),
    (1, 5, 7, 2),
    (4, 0, 3, 6),
)

CUBE_EDGES = (
    (0, 1),
    (0, 3),
    (0, 4),
    (2, 1),
    (2, 3),
    (2, 7),
    (6, 3),
    (6, 4),
    (6, 7),
    (5, 1),
    (5, 4),
    (5, 7),
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


def init_gl_stuff():

    glEnable(GL_DEPTH_TEST)  # use our zbuffer

    # setup the camera
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, 640 / 480.0, 0.1, 100.0)  # setup lens
    glTranslatef(0.0, 0.0, -3.0)  # move back
    glRotatef(25, 1, 0, 0)  # orbit higher


def main():
    "run the demo"
    # initialize pygame and setup an opengl display
    pg.init()

    fullscreen = True
    pg.display.set_mode((640, 480), pg.OPENGL | pg.DOUBLEBUF | pg.FULLSCREEN)

    init_gl_stuff()

    going = True
    while going:
        # check for quit'n events
        events = pg.event.get()
        for event in events:
            if event.type == pg.QUIT or (
                event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE
            ):
                going = False

            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_f:
                    if not fullscreen:
                        print("Changing to FULLSCREEN")
                        pg.display.set_mode(
                            (640, 480), pg.OPENGL | pg.DOUBLEBUF | pg.FULLSCREEN
                        )
                    else:
                        print("Changing to windowed mode")
                        pg.display.set_mode((640, 480), pg.OPENGL | pg.DOUBLEBUF)
                    fullscreen = not fullscreen
                    init_gl_stuff()

        # clear screen and move camera
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # orbit camera around by 1 degree
        glRotatef(1, 0, 1, 0)

        drawcube()
        pg.display.flip()
        pg.time.wait(10)


if __name__ == "__main__":
    main()
