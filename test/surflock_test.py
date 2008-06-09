import unittest
import pygame

class SurfaceLockTest (unittest.TestCase):

##     def test_lock (self):
##         sf = pygame.Surface ((5, 5))

##         sf.lock ()
##         self.assertEquals (sf.get_locked (), True)
##         self.assertEquals (sf.get_locks (), (sf,))

##         sf.lock ()
##         self.assertEquals (sf.get_locked (), True)
##         self.assertEquals (sf.get_locks (), (sf,sf))

##         sf.unlock ()
##         self.assertEquals (sf.get_locked (), True)
##         self.assertEquals (sf.get_locks (), (sf,))

##         sf.unlock ()
##         self.assertEquals (sf.get_locked (), False)
##         self.assertEquals (sf.get_locks (), ())

    def test_subsurface_lock (self):
        sf = pygame.Surface ((5, 5))
        subsf = sf.subsurface ((1, 1, 2, 2))
        sf2 = pygame.Surface ((5, 5))

        # Simple blits, nothing should happen here.
        sf2.blit (subsf, (0, 0))
        sf2.blit (sf, (0, 0))

        # Test blitting on self:
        self.assertRaises (pygame.error, sf.blit, subsf, (0, 0))
        #self.assertRaises (pygame.error, subsf.blit, sf, (0, 0))
        # ^ Fails although it should not in my opinion. If I cannot
        # blit the subsurface to the surface, it should not be allowed
        # the other way around as well.

        # Test additional locks.
        sf.lock ()
        sf2.blit (subsf, (0, 0))
        self.assertRaises (pygame.error, sf2.blit, sf, (0, 0))

        subsf.lock ()
        self.assertRaises (pygame.error, sf2.blit, subsf, (0, 0))
        self.assertRaises (pygame.error, sf2.blit, sf, (0, 0))

        # sf and subsf are now explicitly locked. Unlock sf, so we can
        # (assume) to blit it.
        # It will fail though as the subsurface still has a lock around,
        # which is okay and correct behaviour.
        sf.unlock ()
        self.assertRaises (pygame.error, sf2.blit, subsf, (0, 0))
        self.assertRaises (pygame.error, sf2.blit, sf, (0, 0))

        # Run a second unlock on the surface. This should ideally have
        # no effect as the subsurface is the locking reason!
        # It has though and thus is wrong.
        sf.unlock ()
        self.assertRaises (pygame.error, sf2.blit, sf, (0, 0))

        # Now the complete surface is unlocked and we can do any
        # nastiness on it, while the subsurface still keeps a (now
        # unnecessary) lock around. The same applies to other reference
        # types such as buffers.
        self.assertRaises (pygame.error, sf2.blit, subsf, (0, 0))

##         sf.lock ()
##         self.assertEquals (sf.get_locked (), True)
##         self.assertEquals (sf.get_locks (), (sf,))
##         self.assertEquals (subsf.get_locked (), False)
##         self.assertEquals (subsf.get_locks (), ())

##         subsf.lock ()
##         self.assertEquals (sf.get_locked (), True)
##         self.assertEquals (sf.get_locks (), (sf, subsf))
##         self.assertEquals (subsf.get_locked (), True)
##         self.assertEquals (subsf.get_locks (), (subsf,))

##         sf.unlock ()
##         self.assertEquals (sf.get_locked (), True)
##         self.assertEquals (sf.get_locks (), (subsf,))
##         self.assertEquals (subsf.get_locked (), True)
##         self.assertEquals (subsf.get_locks (), (subsf,))

##         subsf.unlock ()
##         self.assertEquals (sf.get_locked (), False)
##         self.assertEquals (sf.get_locks (), ())
##         self.assertEquals (subsf.get_locked (), False)
##         self.assertEquals (subsf.get_locks (), ())

##         subsf.lock ()
##         self.assertEquals (sf.get_locked (), True)
##         self.assertEquals (sf.get_locks (), (subsf,))
##         self.assertEquals (subsf.get_locked (), True)
##         self.assertEquals (subsf.get_locks (), (subsf,))

##         subsf.lock ()
##         self.assertEquals (sf.get_locked (), True)
##         self.assertEquals (sf.get_locks (), (subsf, subsf))
##         self.assertEquals (subsf.get_locked (), True)
##         self.assertEquals (subsf.get_locks (), (subsf, subsf))

if __name__ == '__main__':
    unittest.main()
