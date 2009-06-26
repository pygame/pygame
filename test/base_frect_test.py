try:
    import pygame2.test.pgunittest as unittest
except:
    import pgunittest as unittest
from pygame2.base import FRect, Rect

class FRectTest (unittest.TestCase):

    def testConstructionXYWidthHeight( self ):
        r = FRect(1.1 ,2.2 ,3.3 ,4.4 )
        self.assertEqual( 1.1, r.left )
        self.assertEqual( 2.2, r.top )
        self.assertEqual( 3.3, r.width )
        self.assertEqual( 4.4, r.height )

    def testConstructionWidthHeight( self ):
        r = FRect (3.99, 4.7)
        self.assertEqual( 0, r.left )
        self.assertEqual( 0, r.top )
        self.assertEqual( 3.99, r.width )
        self.assertEqual( 4.7, r.height )

        r2 = FRect (r.size)
        self.assertEqual( 0, r2.left )
        self.assertEqual( 0, r2.top )
        self.assertEqual( 3.99, r2.width )
        self.assertEqual( 4.7, r2.height )

        r2 = FRect ((3.009,4.1))
        self.assertEqual( 0, r2.left )
        self.assertEqual( 0, r2.top )
        self.assertEqual( 3.009, r2.width )
        self.assertEqual( 4.1, r2.height )

    def testConstructionPointSize( self ):
        r = FRect ((1.1, 2.2), (3.3, 4.4))
        self.assertEqual( 1.1, r.left )
        self.assertEqual( 2.2, r.top )
        self.assertEqual( 3.3, r.width )
        self.assertEqual( 4.4, r.height )

        r2 = FRect (r.topleft, r.size)
        self.assertEqual( 1.1, r2.left )
        self.assertEqual( 2.2, r2.top )
        self.assertEqual( 3.3, r2.width )
        self.assertEqual( 4.4, r2.height )

    def testCalculatedAttributes( self ):
        r = FRect( 1.7, 2, 3, 4.9 )
        
        self.assertEqual( r.left + r.width, r.right )
        self.assertEqual( r.top + r.height, r.bottom )
        self.assertEqual( (r.width,r.height), r.size )
        self.assertEqual( (r.left,r.top), r.topleft )
        self.assertEqual( (r.right,r.top), r.topright )
        self.assertEqual( (r.left,r.bottom), r.bottomleft )
        self.assertEqual( (r.right,r.bottom), r.bottomright )

        midx = r.left + r.width / 2
        midy = r.top + r.height / 2

        self.assertEqual( midx, r.centerx )
        self.assertEqual( midy, r.centery )
        self.assertEqual( (r.centerx,r.centery), r.center )
        self.assertEqual( (r.centerx,r.top), r.midtop )
        self.assertEqual( (r.centerx,r.bottom), r.midbottom )
        self.assertEqual( (r.left,r.centery), r.midleft )
        self.assertEqual( (r.right,r.centery), r.midright )

    def testEquals( self ):
        """ check to see how the FRect uses __eq__ 
        """
        r1 = FRect(1.65,2,3,4)
        r2 = FRect(10,20,30,40)
        r3 = (10,20,30,40)
        r4 = FRect(10,20,30,40)

        class foo (FRect):
            def __eq__(self,other):
                return id(self) == id(other);

        class foo2 (FRect):
            pass

        r5 = foo(10,20,30,40)
        r6 = foo2(10,20,30,40)

        self.assertNotEqual(r5, r2)

        # because we define equality differently for this subclass.
        self.assertEqual(r6, r2)


        rect_list = [r1,r2,r3,r4,r6]

        # see if we can remove 4 of these.
        rect_list.remove(r2)
        rect_list.remove(r2)
        rect_list.remove(r2)
        self.assertRaises(ValueError, rect_list.remove, r2)

    def test_pygame2_base_FRect_bottom(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.FRect.bottom:

        # Gets or sets the bottom edge position of the FRect.
        r = FRect( 1.99, 2, 3, 4 )
        new_bottom = r.bottom + 20.34
        expected_top = r.top + 20.34
        old_height = r.height
        
        r.bottom = new_bottom
        self.assertEqual( new_bottom, r.bottom )
        self.assertEqual( expected_top, r.top )
        self.assertEqual( old_height, r.height )

    def test_pygame2_base_FRect_bottomleft(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.FRect.bottomleft:

        # Gets or sets the bottom left corner position of the FRect.
        r = FRect( 1, 2, 3, 4 )
        new_bottomleft = (r.left+20,r.bottom+30)
        expected_topleft = (r.left+20,r.top+30)
        old_size = r.size
        
        r.bottomleft = new_bottomleft
        self.assertEqual( new_bottomleft, r.bottomleft )
        self.assertEqual( expected_topleft, r.topleft )
        self.assertEqual( old_size, r.size )

    def test_pygame2_base_FRect_bottomright(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.FRect.bottomright:

        # Gets or sets the bottom right corner position of the FRect.
        r = FRect( 1, 2, 3, 4 )
        new_bottomright = (r.right+20,r.bottom+30)
        expected_topleft = (r.left+20,r.top+30)
        old_size = r.size
        
        r.bottomright = new_bottomright
        self.assertEqual( new_bottomright, r.bottomright )
        self.assertEqual( expected_topleft, r.topleft )
        self.assertEqual( old_size, r.size )

    def test_pygame2_base_FRect_ceil(self):

        # __doc__ (as of 2008-11-04) for pygame2.base.FRect.ceil:

        # FRect.ceil () -> Rect
        #
        # Creates a Rect from the specified FRect.
        # 
        # This creates a Rect using the smallest integral values greater
        # or equal to the FRect floating point values.
        r = FRect (2.1, -2.9, 5.8, 3.01)
        self.assertEqual (r.ceil (), Rect (3, -2, 6, 4))

    def test_pygame2_base_FRect_center(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.FRect.center:

        # Gets or sets the center position of the FRect.
        r = FRect( 1, 2, 3, 4 )
        new_center = (r.centerx+20,r.centery+30)
        expected_topleft = (r.left+20,r.top+30)
        old_size = r.size
        
        r.center = new_center
        self.assertEqual( new_center, r.center )
        self.assertEqual( expected_topleft, r.topleft )
        self.assertEqual( old_size, r.size )

    def test_pygame2_base_FRect_centerx(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.FRect.centerx:

        # Gets or sets the horizontal center position of the FRect.
        r = FRect( 1, 2, 3, 4 )
        new_centerx = r.centerx + 20
        expected_left = r.left + 20
        old_width = r.width
        
        r.centerx = new_centerx
        self.assertEqual( new_centerx, r.centerx )
        self.assertEqual( expected_left, r.left )
        self.assertEqual( old_width, r.width )

    def test_pygame2_base_FRect_centery(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.FRect.centery:

        # Gets or sets the vertical center position of the FRect.
        r = FRect( 1, 2, 3, 4 )
        new_centery = r.centery + 20
        expected_top = r.top + 20
        old_height = r.height
        
        r.centery = new_centery
        self.assertEqual( new_centery, r.centery )
        self.assertEqual( expected_top, r.top )
        self.assertEqual( old_height, r.height )

    def test_pygame2_base_FRect_clamp(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.FRect.clamp:

        # FRect.clamp (FRect) -> FRect
        # 
        # Moves the rectangle inside another.
        # 
        # Returns a new rectangle that is moved to be completely inside the
        # argument FRect. If the rectangle is too large to fit inside, it is
        # centered inside the argument FRect, but its size is not changed.
        r = FRect(10, 10, 10, 10)
        c = FRect(19, 12, 5, 5).clamp(r)
        self.assertEqual(c.right, r.right)
        self.assertEqual(c.top, 12)
        
        c = FRect(1, 2, 3, 4).clamp(r)
        self.assertEqual(c.topleft, r.topleft)
        c = FRect(5, 500, 22, 30).clamp(r)
        self.assertEqual(c.center, r.center)

    def test_pygame2_base_FRect_clamp_ip(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.FRect.clamp_ip:

        # FRect.clamp_ip (FRect) -> None
        # 
        # Moves the rectangle inside another, in place.
        # 
        # Same as FRect.clamp(FRect), but operates in place.
        r = FRect(10, 10, 10, 10)
        c = FRect(19, 12, 5, 5)
        c.clamp_ip(r)
        self.assertEqual(c.right, r.right)
        self.assertEqual(c.top, 12)
        c = FRect(1, 2, 3, 4)
        c.clamp_ip(r)
        self.assertEqual(c.topleft, r.topleft)
        c = FRect(5, 500, 22, 30)
        c.clamp_ip(r)
        self.assertEqual(c.center, r.center)

    def test_pygame2_base_FRect_clip(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.FRect.clip:

        # FRect.clip (FRect) -> FRect
        # 
        # Crops a rectangle inside another.
        # 
        # Returns a new rectangle that is cropped to be completely inside the
        # argument FRect. If the two rectangles do not overlap to begin with,
        # a FRect with 0 size is returned. Thus it returns the area, in which
        # both rects overlap.
        r1 = FRect( 1, 2, 3, 4 )
        self.assertEqual( FRect( 1, 2, 2, 2 ), r1.clip( FRect(0,0,3,4) ) )
        self.assertEqual( FRect( 2, 2, 2, 4 ), r1.clip( FRect(2,2,10,20) ) )
        self.assertEqual( FRect(2,3,1,2), r1.clip( FRect(2,3,1,2) ) )
        self.assertEqual( (0,0), r1.clip(FRect (20,30,5,6)).size )
        self.assertEqual( r1, r1.clip( FRect(r1) ),
                          "r1 does not clip an identical rect to itself" )

    def test_pygame2_base_FRect_collidedict(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.FRect.collidedict:

        # FRect.collidedict (dict) -> (key, value)
        # 
        # Test if one rectangle in a dictionary intersects.
        # 
        # Returns the key and value of the first dictionary value that
        # collides with the FRect. If no collisions are found, None is
        # returned. They keys of the passed dict must be FRect objects.
        r = FRect(1, 1, 10, 10)
        r1 = FRect(1, 1, 10, 10)
        r2 = FRect(50, 50, 10, 10)
        r3 = FRect(70, 70, 10, 10)
        r4 = FRect(61, 61, 10, 10)

        d = {1: r1, 2: r2, 3: r3}

        rects_values = 1
        val = r.collidedict(d, rects_values)
        self.assertTrue(val)
        self.assertEqual(len(val), 2)
        self.assertEqual(val[0], 1)
        self.assertEqual(val[1], r1)

        none_d = {2: r2, 3: r3}
        none_val = r.collidedict(none_d, rects_values)
        self.assertFalse(none_val)

        barely_d = {1: r1, 2: r2, 3: r3}
        k3, v3 = r4.collidedict(barely_d, rects_values)
        self.assertEqual(k3, 3)
        self.assertEqual(v3, r3)

    def test_pygame2_base_FRect_collidedictall(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.FRect.collidedictall:

        # FRect.collidedictall (dict) -> [(key, value), ...]
        # 
        # Test if all rectangles in a dictionary intersect.
        # 
        # Returns a list of all the key and value pairs that intersect
        # with the FRect. If no collisions are found an empty list is
        # returned. They keys of the passed dict must be FRect objects.

        r = FRect(1, 1, 10, 10)

        r2 = FRect(1, 1, 10, 10)
        r3 = FRect(5, 5, 10, 10)
        r4 = FRect(10, 10, 10, 10)
        r5 = FRect(50, 50, 10, 10)

        rects_values = 1
        d = {2: r2}
        l = r.collidedictall(d, rects_values)
        self.assertEqual(l, [(2, r2)])

        d2 = {2: r2, 3: r3, 4: r4, 5: r5}
        l2 = r.collidedictall(d2, rects_values)
        self.assertEqual(l2, [(2, r2), (3, r3), (4, r4)])

    def test_pygame2_base_FRect_colliderect(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.FRect.colliderect:

        # FRect.colliderect (FRect) -> bool
        # 
        # Test if two rectangles overlap.
        # 
        # Returns true if any portion of either rectangle overlap (except the
        # top+bottom or left+right edges).
        r1 = FRect(1,2,3,4)
        self.failUnless( r1.colliderect( FRect(0,0,2,3) ),
                         "r1 does not collide with Rect(0,0,2,3)" )
        self.failIf( r1.colliderect( FRect(0,0,1,2) ),
                     "r1 collides with Rect(0,0,1,2)" )
        self.failIf( r1.colliderect( FRect(r1.right,r1.bottom,2,2) ),
                     "r1 collides with Rect(r1.right,r1.bottom,2,2)" )
        self.failUnless( r1.colliderect( FRect(r1.left+1,r1.top+1,
                                               r1.width-2,r1.height-2) ),
                         "r1 does not collide with Rect(r1.left+1,r1.top+1,"+
                         "r1.width-2,r1.height-2)" )
        self.failUnless( r1.colliderect( FRect(r1.left-1,r1.top-1,
                                               r1.width+2,r1.height+2) ),
                         "r1 does not collide with Rect(r1.left-1,r1.top-1,"+
                         "r1.width+2,r1.height+2)" )
        self.failUnless( r1.colliderect( FRect(r1) ),
                         "r1 does not collide with an identical rect" )
        self.failIf( r1.colliderect( FRect(r1.right,r1.bottom,0,0) ),
                     "r1 collides with Rect(r1.right,r1.bottom,0,0)" )
        self.failIf( r1.colliderect( FRect(r1.right,r1.bottom,1,1) ),
                     "r1 collides with Rect(r1.right,r1.bottom,1,1)" )

    def test_pygame2_base_FRect_collidelist(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.FRect.collidelist:

        # FRect.collidelist (list) -> index
        # 
        # Test if one rectangle in a list intersects.
        # 
        # Test whether the rectangle collides with any in a sequence of
        # rectangles. The index of the first collision found is
        # returned. If no collisions are found an index of -1 is
        # returned.

        r = FRect(1, 1, 10, 10)
        l = [FRect(50, 50, 1, 1), FRect(5, 5, 10, 10), FRect(15, 15, 1, 1)]

        self.assertEqual(r.collidelist(l), 1)

        f = [FRect(50, 50, 1, 1), FRect(100, 100, 4, 4)]
        self.assertEqual(r.collidelist(f), -1)

    def test_pygame2_base_FRect_collidelistall(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.FRect.collidelistall:

        # FRect.collidelistall (list) -> [index, ...]
        # 
        # Test if all rectangles in a list intersect.
        # 
        # Returns a list of all the indices that contain rectangles that
        # collide with the FRect. If no intersecting rectangles are
        # found, an empty list is returned.

        r = FRect(1, 1, 10, 10)

        l = [
            FRect(1, 1, 10, 10), 
            FRect(5, 5, 10, 10),
            FRect(15, 15, 1, 1),
            FRect(2, 2, 1, 1),
        ]
        self.assertEqual(r.collidelistall(l), [0, 1, 3])

        f = [FRect(50, 50, 1, 1), FRect(20, 20, 5, 5)]
        self.assertFalse(r.collidelistall(f))

    def test_pygame2_base_FRect_collidepoint(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.FRect.collidepoint:

        # FRect.collidepoint (x, y) -> bool
        # 
        # Test if a point is inside a rectangle.
        # 
        # Returns true if the given point is inside the rectangle.  A
        # point along the right or bottom edge is not considered to be
        # inside the rectangle.
        r = FRect( 1, 2, 3, 4 )
        
        self.failUnless( r.collidepoint( r.left, r.top ),
                         "r does not collide with point (left,top)" )
        self.failIf( r.collidepoint( r.left-1, r.top ),
                     "r collides with point (left-1,top)"  )
        self.failIf( r.collidepoint( r.left, r.top-1 ),
                     "r collides with point (left,top-1)"  )
        self.failIf( r.collidepoint( r.left-1,r.top-1 ),
                     "r collides with point (left-1,top-1)"  )
        
        self.failUnless( r.collidepoint( r.right-1, r.bottom-1 ),
                         "r does not collide with point (right-1,bottom-1)")
        self.failIf( r.collidepoint( r.right, r.bottom ),
                     "r collides with point (right,bottom)" )
        self.failIf( r.collidepoint( r.right-1, r.bottom ),
                     "r collides with point (right-1,bottom)" )
        self.failIf( r.collidepoint( r.right, r.bottom-1 ),
                     "r collides with point (right,bottom-1)" )

    def test_pygame2_base_FRect_contains(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.FRect.contains:

        # FRect.contains (FRect) -> bool
        # 
        # Test if one rectangle is inside another.
        # 
        # Returns true when the argument rectangle is completely
        # inside the FRect.
        r = FRect( 1, 2, 3, 4 )
        
        self.failUnless( r.contains( FRect( 2, 3, 1, 1 ) ),
                         "r does not contain Rect(2,3,1,1)" )
        self.failUnless( r.contains( FRect(r) ),
                         "r does not contain the same rect as itself" )
        self.failUnless( r.contains( FRect(2,3,0,0) ),
                         "r does not contain an empty rect within its bounds" )
        self.failIf( r.contains( FRect(0,0,1,2) ),
                     "r contains Rect(0,0,1,2)" )
        self.failIf( r.contains( FRect(4,6,1,1) ),
                     "r contains Rect(4,6,1,1)" )
        self.failIf( r.contains( FRect(4,6,0,0) ),
                     "r contains Rect(4,6,0,0)" )
    
    def test_pygame2_base_FRect_copy(self):

        # __doc__ (as of 2009-02-23) for pygame2.base.FRect.copy:

        # copy () -> FRect
        #
        # Creates a copy of the FRect.
        #
        # Returns a new FRect, that contains the same values as the
        # caller.
        r = FRect( 1.819, 2, 3, 4 )
        cp = r.copy ()
        self.failUnless (r == cp, "r (1, 2, 3, 4) is not equal to its copy")

        r = FRect( -10, 50.38, 10, 40 )
        cp = r.copy ()
        self.failUnless (r == cp,
                         "r (-10, 50, 10, 40) is not equal to its copy")
        
        r = FRect( 2, -5.5239284, 10, 40 )
        cp = r.copy ()
        self.failUnless (r == cp,
                         "r (2, -5, 10, 40) is not equal to its copy")
        
        r = FRect( -2, -5, 10.8438792849, 40 )
        cp = r.copy ()
        self.failUnless (r == cp,
                         "r (-2, -5, 10, 40) is not equal to its copy")
    
    def test_pygame2_base_FRect_fit(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.FRect.fit:

        # FRect.fit (FRect) -> FRect
        # 
        # Resize and move a rectangle with aspect ratio.
        # 
        # Returns a new rectangle that is moved and resized to fit
        # another. The aspect ratio of the original FRect is preserved, so the
        # new rectangle may be smaller than the target in either width or
        # height.

        r = FRect(10, 10, 30, 30)

        r2 = FRect(30, 30, 15, 10)

        f = r.fit(r2)
        self.assertTrue(r2.contains(f))
        
        f2 = r2.fit(r)
        self.assertTrue(r.contains(f2))

    def test_pygame2_base_FRect_floor(self):

        # __doc__ (as of 2008-11-04) for pygame2.base.FRect.floor:

        # FRect.floor () -> Rect
        #
        # Creates a Rect from the specified FRect.
        #
        # This creates a Rect using the largest integral values less than
        # or equal to the FRect floating point values.
        r = FRect (2.1, -2.9, 5.8, 3.01)
        self.assertEqual (r.floor (), Rect (2, -3, 5, 3))

    def test_pygame2_base_FRect_height(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.FRect.height:

        # Gets or sets the height of the FRect.
        r = FRect( 1, 2, 3, 4 )
        new_height = 10
        old_topleft = r.topleft
        old_width = r.width
        
        r.height = new_height
        self.assertEqual( new_height, r.height )
        self.assertEqual( old_width, r.width )
        self.assertEqual( old_topleft, r.topleft )
    
    def test_pygame2_base_FRect_h(self):

        r = FRect( 1, 2, 3, 4 )
        new_height = 10
        old_topleft = r.topleft
        old_width = r.width
        
        r.h = new_height
        self.assertEqual( new_height, r.h )
        self.assertEqual( old_width, r.width )
        self.assertEqual( old_topleft, r.topleft )
    def test_inflate__larger( self ):
        "The inflate method inflates around the center of the rectangle"
        r = FRect( 2, 4, 6, 8 )
        r2 = r.inflate( 4, 6 )

        self.assertEqual( r.center, r2.center )
        self.assertEqual( r.left-2, r2.left )
        self.assertEqual( r.top-3, r2.top )
        self.assertEqual( r.right+2, r2.right )
        self.assertEqual( r.bottom+3, r2.bottom )
        self.assertEqual( r.width+4, r2.width )
        self.assertEqual( r.height+6, r2.height )

    def test_inflate__smaller( self ):
        "The inflate method inflates around the center of the rectangle"
        r = FRect( 2, 4, 6, 8 )
        r2 = r.inflate( -4, -6 )

        self.assertEqual( r.center, r2.center )
        self.assertEqual( r.left+2, r2.left )
        self.assertEqual( r.top+3, r2.top )
        self.assertEqual( r.right-2, r2.right )
        self.assertEqual( r.bottom-3, r2.bottom )
        self.assertEqual( r.width-4, r2.width )
        self.assertEqual( r.height-6, r2.height )

    def test_inflate_ip__larger( self ):    
        "The inflate_ip method inflates around the center of the rectangle"
        r = FRect( 2, 4, 6, 8 )
        r2 = FRect( r )
        r2.inflate_ip( -4, -6 )
        
        self.assertEqual( r.center, r2.center )
        self.assertEqual( r.left+2, r2.left )
        self.assertEqual( r.top+3, r2.top )
        self.assertEqual( r.right-2, r2.right )
        self.assertEqual( r.bottom-3, r2.bottom )
        self.assertEqual( r.width-4, r2.width )
        self.assertEqual( r.height-6, r2.height )

    def test_inflate_ip__smaller( self ):
        "The inflate method inflates around the center of the rectangle"
        r = FRect( 2, 4, 6, 8 )
        r2 = FRect( r )
        r2.inflate_ip( -4, -6 )
        
        self.assertEqual( r.center, r2.center )
        self.assertEqual( r.left+2, r2.left )
        self.assertEqual( r.top+3, r2.top )
        self.assertEqual( r.right-2, r2.right )
        self.assertEqual( r.bottom-3, r2.bottom )
        self.assertEqual( r.width-4, r2.width )
        self.assertEqual( r.height-6, r2.height )

    def test_pygame2_base_FRect_inflate(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.FRect.inflate:

        # FRect.inflate (x, y) -> FRect
        # 
        # Grow or shrink the rectangle size.
        # 
        # Returns a new rectangle with the size changed by the given offset.
        # The rectangle remains centered around its current center. Negative
        # values will shrink the rectangle.
        pass

    def test_pygame2_base_FRect_inflate_ip(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.FRect.inflate_ip:

        # FRect.inflate_ip (x, y) -> None
        # 
        # Grow or shrink the rectangle size, in place.
        # 
        # Same as FRect.inflate(x, y), but operates in place.
        pass

    def test_pygame2_base_FRect_left(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.FRect.left:

        # Gets or sets the left edge position of the FRect.
        r = FRect( 1, 2, 3, 4 )
        new_left = 10
        
        r.left = new_left
        self.assertEqual( new_left, r.left )
        self.assertEqual( FRect(new_left,2,3,4), r )

    def test_pygame2_base_FRect_midbottom(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.FRect.midbottom:

        # Gets or sets the mid bottom edge position of the FRect.
        r = FRect( 1, 2, 3, 4 )
        new_midbottom = (r.centerx+20,r.bottom+30)
        expected_topleft = (r.left+20,r.top+30)
        old_size = r.size
        
        r.midbottom = new_midbottom
        self.assertEqual( new_midbottom, r.midbottom )
        self.assertEqual( expected_topleft, r.topleft )
        self.assertEqual( old_size, r.size )

    def test_pygame2_base_FRect_midleft(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.FRect.midleft:

        # Gets or sets the mid left edge position of the FRect.
        r = FRect( 1, 2, 3, 4 )
        new_midleft = (r.left+20,r.centery+30)
        expected_topleft = (r.left+20,r.top+30)
        old_size = r.size
        
        r.midleft = new_midleft
        self.assertEqual( new_midleft, r.midleft )
        self.assertEqual( expected_topleft, r.topleft )
        self.assertEqual( old_size, r.size )

    def test_pygame2_base_FRect_midright(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.FRect.midright:

        # Gets or sets the mid right edge position of the FRect.
        r = FRect( 1, 2, 3, 4 )
        new_midright= (r.right+20,r.centery+30)
        expected_topleft = (r.left+20,r.top+30)
        old_size = r.size
        
        r.midright = new_midright
        self.assertEqual( new_midright, r.midright )
        self.assertEqual( expected_topleft, r.topleft )
        self.assertEqual( old_size, r.size )

    def test_pygame2_base_FRect_midtop(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.FRect.midtop:

        # Gets or sets the mid top edge position of the FRect.
        r = FRect( 1, 2, 3, 4 )
        new_midtop= (r.centerx+20,r.top+30)
        expected_topleft = (r.left+20,r.top+30)
        old_size = r.size
        
        r.midtop = new_midtop
        self.assertEqual( new_midtop, r.midtop )
        self.assertEqual( expected_topleft, r.topleft )
        self.assertEqual( old_size, r.size )

    def test_pygame2_base_FRect_move(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.FRect.move:

        # FRect.move (x, y) -> FRect
        # 
        # Moves the rectangle.
        # 
        # Returns a new rectangle that is moved by the given offset. The
        # x and y arguments can be any integer value, positive or
        # negative.
        r = FRect( 1, 2, 3, 4 )
        move_x = 10
        move_y = 20
        r2 = r.move( move_x, move_y )
        expected_r2 = FRect(r.left+move_x,r.top+move_y,r.width,r.height)
        self.assertEqual( expected_r2, r2 )

    def test_pygame2_base_FRect_move_ip(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.FRect.move_ip:

        # FRect.move_ip (x, y) -> None
        # 
        # Moves the rectangle, in place.
        # 
        # Same as FRect.move (x, y), but operates in place.
        r = FRect( 1, 2, 3, 4 )
        r2 = FRect( r )
        move_x = 10
        move_y = 20
        r2.move_ip( move_x, move_y )
        expected_r2 = FRect(r.left+move_x,r.top+move_y,r.width,r.height)
        self.assertEqual( expected_r2, r2 )

    def test_pygame2_base_FRect_right(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.FRect.right:

        # Gets or sets the right position of the FRect.
        r = FRect( 1, 2, 3, 4 )
        new_right = r.right + 20
        expected_left = r.left + 20
        old_width = r.width
        
        r.right = new_right
        self.assertEqual( new_right, r.right )
        self.assertEqual( expected_left, r.left )
        self.assertEqual( old_width, r.width )

    def test_pygame2_base_FRect_round(self):

        # __doc__ (as of 2008-11-04) for pygame2.base.FRect.round:

        # FRect.round () -> Rect
        # 
        # Creates a Rect from the specified FRect.
        # 
        # This creates a Rect using the FRect floating point values
        # rounded to the nearest integral value.
        r = FRect (2.1, -2.9, 5.8, 3.01)
        self.assertEqual (r.round (), Rect (2, -3, 6, 3))

    def test_pygame2_base_FRect_size(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.FRect.size:

        # Gets or sets the width and height of the FRect as 2-value tuple.
        r = FRect( 1, 2, 3, 4 )
        new_size = (10,20)
        old_topleft = r.topleft
        
        r.size = new_size
        self.assertEqual( new_size, r.size )
        self.assertEqual( old_topleft, r.topleft )

    def test_pygame2_base_FRect_top(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.FRect.top:

        # Gets or sets the top edge position of the FRect.
        r = FRect( 1, 2, 3, 4 )
        new_top = 10
        
        r.top = new_top
        self.assertEqual( FRect(1,new_top,3,4), r )
        self.assertEqual( new_top, r.top )

    def test_pygame2_base_FRect_topleft(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.FRect.topleft:

        # Gets or sets the top left corner position of the FRect.
        r = FRect( 1, 2, 3, 4 )
        new_topleft = (r.left+20,r.top+30)
        old_size = r.size
        
        r.topleft = new_topleft
        self.assertEqual( new_topleft, r.topleft )
        self.assertEqual( old_size, r.size )

    def test_pygame2_base_FRect_topright(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.FRect.topright:

        # Gets or sets the top right corner position of the FRect.
        r = FRect( 1, 2, 3, 4 )
        new_topright = (r.right+20,r.top+30)
        expected_topleft = (r.left+20,r.top+30)
        old_size = r.size
        
        r.topright = new_topright
        self.assertEqual( new_topright, r.topright )
        self.assertEqual( expected_topleft, r.topleft )
        self.assertEqual( old_size, r.size )

    def test_pygame2_base_FRect_trunc(self):

        # __doc__ (as of 2008-11-04) for pygame2.base.FRect.trunc:

        # FRect.trunc () -> Rect
        #
        # Creates a Rect from the specified FRect.
        #
        # This creates a Rect using truncated integral values from the 
        # Frect floating point values.
        r = FRect (-1.57, 2.99, 8.1, 5.77)
        self.assertEqual (r.trunc (), Rect (-1, 2, 8, 5))

    def test_pygame2_base_FRect_union(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.FRect.union:

        # FRect.union (FRect) -> FRect
        # 
        # Joins two rectangles into one.
        # 
        # Returns a new rectangle that completely covers the area of the two
        # provided rectangles. There may be area inside the new FRect that is
        # not covered by the originals.
        r1 = FRect( 1, 1, 1, 2 )
        r2 = FRect( -2, -2, 1, 2 )
        self.assertEqual( FRect( -2, -2, 4, 5 ), r1.union(r2) )

    def test_union__with_identical_Rect( self ):
        r1 = FRect( 1, 2, 3, 4 )
        self.assertEqual( r1, r1.union( FRect(r1) ) )

    def test_pygame2_base_FRect_union_ip(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.FRect.union_ip:

        # FRect.union_ip (FRect) -> FRect
        # 
        # Joins two rectangles into one, in place.
        # 
        # Same as FRect.union(FRect), but operates in place.
        r1 = FRect( 1, 1, 1, 2 )
        r2 = FRect( -2, -2, 1, 2 )
        r1.union_ip(r2)
        self.assertEqual( FRect( -2, -2, 4, 5 ), r1 ) 

    def test_union__list( self ):
        r1 = FRect( 0, 0, 1, 1 )
        r2 = FRect( -2, -2, 1, 1 )
        r3 = FRect( 2, 2, 1, 1 )
        
        r4 = r1.union( [r2,r3] )
        self.assertEqual( FRect(-2, -2, 5, 5), r4 )
    
    def test_union_ip__list( self ):
        r1 = FRect( 0, 0, 1, 1 )
        r2 = FRect( -2, -2, 1, 1 )
        r3 = FRect( 2, 2, 1, 1 )
        
        r1.union_ip( [r2,r3] )
        self.assertEqual( FRect(-2, -2, 5, 5), r1 )

    def test_pygame2_base_FRect_width(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.FRect.width:

        # Gets or sets the width of the FRect.
        r = FRect( 1, 2, 3, 4 )
        new_width = 10
        old_topleft = r.topleft
        old_height = r.height
        
        r.width = new_width
        self.assertEqual( new_width, r.width )
        self.assertEqual( old_height, r.height )
        self.assertEqual( old_topleft, r.topleft )
    
    def test_pygame2_base_FRect_w(self):
        r = FRect( 1, 2, 3, 4 )
        new_width = 10
        old_topleft = r.topleft
        old_height = r.height
        
        r.w = new_width
        self.assertEqual( new_width, r.w )
        self.assertEqual( old_height, r.height )
        self.assertEqual( old_topleft, r.topleft )

    def test_pygame2_base_FRect_x(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.FRect.x:

        # Gets or sets the horizontal top left position of the FRect.
        r = FRect (1, 2, 3, 4)
        self.assertEqual (r.x, 1)
        r.topleft = 32.777, 10
        self.assertEqual (r.x, 32.777)
        r.left = -44.27458
        self.assertEqual (r.x, -44.27458)
        r.move_ip (10, 33)
        self.assertEqual (r.x, -34.27458)

    def test_pygame2_base_FRect_y(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.FRect.y:

        # Gets or sets the vertical top left position of the FRect.
        r = FRect (1, 2, 3, 4)
        self.assertEqual (r.y, 2)
        r.topleft = 32, 10.28
        self.assertEqual (r.y, 10.28)
        r.top = -44.85888
        self.assertEqual (r.y, -44.85888)
        r.move_ip (10, 33)
        self.assertEqual (r.y, -11.85888)

    def test_pygame2_base_FRect___repr__(self):
        r = FRect (10, 4, 7.12345678, 99)
        text = "FRect(10.000, 4.000, 7.123, 99.000)"
        self.assertEqual (repr (r), text)

if __name__ == "__main__":
    unittest.main ()
