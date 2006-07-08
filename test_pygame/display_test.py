import unittest
import pygame, pygame.transform


class DisplayTest( unittest.TestCase ):
    
    def test_update( self ):
        """ see if pygame.display.update takes rects with negative values.
        """

        pygame.init()
        screen = pygame.display.set_mode((100,100))
        screen.fill((55,55,55))

        r1 = pygame.Rect(0,0,100,100)
        pygame.display.update(r1)

        r2 = pygame.Rect(-10,0,100,100)
        pygame.display.update(r2)

        r3 = pygame.Rect(-10,0,-100,-100)
        pygame.display.update(r3)

        # NOTE: if I don't call pygame.quit there is a segfault.  hrmm.
        pygame.quit()
        #  I think it's because unittest runs stuff in threads
        # here's a stack trace...
        """
#0  0x08103b7c in PyFrame_New ()
#1  0x080bd666 in PyEval_EvalCodeEx ()
#2  0x08105202 in PyFunction_SetClosure ()
#3  0x080595ae in PyObject_Call ()
#4  0x080b649f in PyEval_CallObjectWithKeywords ()
#5  0x08059585 in PyObject_CallObject ()
#6  0xb7f7aa2d in initbase () from /usr/lib/python2.4/site-packages/pygame/base.so
#7  0x080e09bd in Py_Finalize ()
#8  0x08055597 in Py_Main ()
#9  0xb7e04eb0 in __libc_start_main () from /lib/tls/libc.so.6
#10 0x08054e31 in _start ()

        """



    def test_init_quit( self ):
        """ see if initing, and quiting works.
        """

        pygame.init()
        screen = pygame.display.set_mode((100,100))
        #pygame.quit()






if __name__ == '__main__':
    unittest.main()
