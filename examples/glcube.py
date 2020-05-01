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
    import OpenGL.GL as GL
    import OpenGL.GLU as GLU
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

    GL.glBegin(GL.GL_QUADS)
    for face in CUBE_QUAD_VERTS:
        for vert in face:
            pos, color = allpoints[vert]
            GL.glColor3fv(color)
            GL.glVertex3fv(pos)
    GL.glEnd()

    GL.glColor3f(1.0, 1.0, 1.0)
    GL.glBegin(GL.GL_LINES)
    for line in CUBE_EDGES:
        for vert in line:
            pos, color = allpoints[vert]
            GL.glVertex3fv(pos)

    GL.glEnd()


def init_gl_stuff():

    GL.glEnable(GL.GL_DEPTH_TEST)  # use our zbuffer

    # setup the camera
    GL.glMatrixMode(GL.GL_PROJECTION)
    GL.glLoadIdentity()
    GLU.gluPerspective(45.0, 640 / 480.0, 0.1, 100.0)  # setup lens
    GL.glTranslatef(0.0, 0.0, -3.0)  # move back
    GL.glRotatef(25, 1, 0, 0)  # orbit higher


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
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        # orbit camera around by 1 degree
        GL.glRotatef(1, 0, 1, 0)

        drawcube()
        pg.display.flip()
        pg.time.wait(10)


if __name__ == "__main__":
    main()
