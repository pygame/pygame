import pygame, unittest, os
from pygame.locals import *

CUBE_POINTS = (
    (0.5, -0.5, -0.5),  (0.5, 0.5, -0.5),
    (-0.5, 0.5, -0.5),  (-0.5, -0.5, -0.5),
    (0.5, -0.5, 0.5),   (0.5, 0.5, 0.5),
    (-0.5, -0.5, 0.5),  (-0.5, 0.5, 0.5)
)

CUBE_QUAD_VERTS = (
    (0, 1, 2, 3), (3, 2, 7, 6), (6, 7, 5, 4),
    (4, 5, 1, 0), (1, 5, 7, 2), (4, 0, 3, 6)
)

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *

    def drawcube():
        glBegin(GL_QUADS)
        for face in CUBE_QUAD_VERTS:
            for vert in face:
                glVertex3fv(CUBE_POINTS[vert])
        glEnd()

    class GL_ImageSave(unittest.TestCase):
        def test_image_save_works_with_opengl_surfaces(self):
            pygame.init()
            screen = pygame.display.set_mode((640,480), OPENGL|DOUBLEBUF)

            drawcube()
            pygame.display.flip()
            pygame.image.save(screen, 'opengl_save_surface_test.png')

            self.assert_(os.path.exists('opengl_save_surface_test.png'))

except:
    print 'GL test requires PyOpenGL'

if __name__ == '__main__': 
    unittest.main()