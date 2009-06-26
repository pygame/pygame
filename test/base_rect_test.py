try:
    import pygame2.test.pgunittest as unittest
except:
    import pgunittest as unittest

from pygame2.base import Rect

class RectTest (unittest.TestCase):
    def testConstructionXYWidthHeight( self ):
        r = Rect(1,2,3,4)
        self.assertEqual( 1, r.left )
        self.assertEqual( 2, r.top )
        self.assertEqual( 3, r.width )
        self.assertEqual( 4, r.height )

    def testConstructionWidthHeight( self ):
        r = Rect (3, 4)
        self.assertEqual( 0, r.left )
        self.assertEqual( 0, r.top )
        self.assertEqual( 3, r.width )
        self.assertEqual( 4, r.height )

        r2 = Rect (r.size)
        self.assertEqual( 0, r2.left )
        self.assertEqual( 0, r2.top )
        self.assertEqual( 3, r2.width )
        self.assertEqual( 4, r2.height )

        r2 = Rect ((3,4))
        self.assertEqual( 0, r2.left )
        self.assertEqual( 0, r2.top )
        self.assertEqual( 3, r2.width )
        self.assertEqual( 4, r2.height )

    def testConstructionPointSize( self ):
        r = Rect ((1,2),(3,4))
        self.assertEqual( 1, r.left )
        self.assertEqual( 2, r.top )
        self.assertEqual( 3, r.width )
        self.assertEqual( 4, r.height )

        r2 = Rect (r.topleft, r.size)
        self.assertEqual( 1, r2.left )
        self.assertEqual( 2, r2.top )
        self.assertEqual( 3, r2.width )
        self.assertEqual( 4, r2.height )

    def testCalculatedAttributes( self ):
        r = Rect( 1, 2, 3, 4 )
        
        self.assertEqual( r.left+r.width, r.right )
        self.assertEqual( r.top+r.height, r.bottom )
        self.assertEqual( (r.width,r.height), r.size )
        self.assertEqual( (r.left,r.top), r.topleft )
        self.assertEqual( (r.right,r.top), r.topright )
        self.assertEqual( (r.left,r.bottom), r.bottomleft )
        self.assertEqual( (r.right,r.bottom), r.bottomright )

        midx = int (r.left + r.width / 2)
        midy = int (r.top + r.height / 2)

        self.assertEqual( midx, r.centerx )
        self.assertEqual( midy, r.centery )
        self.assertEqual( (r.centerx,r.centery), r.center )
        self.assertEqual( (r.centerx,r.top), r.midtop )
        self.assertEqual( (r.centerx,r.bottom), r.midbottom )
        self.assertEqual( (r.left,r.centery), r.midleft )
        self.assertEqual( (r.right,r.centery), r.midright )

    def testEquals( self ):
        """ check to see how the rect uses __eq__ 
        """
        r1 = Rect(1,2,3,4)
        r2 = Rect(10,20,30,40)
        r3 = (10,20,30,40)
        r4 = Rect(10,20,30,40)

        class foo (Rect):
            def __eq__(self,other):
                return id(self) == id(other);

        class foo2 (Rect):
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

    def test_pygame2_base_Rect_bottom(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.bottom:

        # Gets or sets the bottom edge position of the Rect.
        r = Rect( 1, 2, 3, 4 )
        new_bottom = r.bottom + 20
        expected_top = r.top + 20
        old_height = r.height
        
        r.bottom = new_bottom
        self.assertEqual( new_bottom, r.bottom )
        self.assertEqual( expected_top, r.top )
        self.assertEqual( old_height, r.height )

    def test_pygame2_base_Rect_bottomleft(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.bottomleft:

        # Gets or sets the bottom left corner position of the Rect.
        r = Rect( 1, 2, 3, 4 )
        new_bottomleft = (r.left+20,r.bottom+30)
        expected_topleft = (r.left+20,r.top+30)
        old_size = r.size
        
        r.bottomleft = new_bottomleft
        self.assertEqual( new_bottomleft, r.bottomleft )
        self.assertEqual( expected_topleft, r.topleft )
        self.assertEqual( old_size, r.size )

    def test_pygame2_base_Rect_bottomright(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.bottomright:

        # Gets or sets the bottom right corner position of the Rect.
        r = Rect( 1, 2, 3, 4 )
        new_bottomright = (r.right+20,r.bottom+30)
        expected_topleft = (r.left+20,r.top+30)
        old_size = r.size
        
        r.bottomright = new_bottomright
        self.assertEqual( new_bottomright, r.bottomright )
        self.assertEqual( expected_topleft, r.topleft )
        self.assertEqual( old_size, r.size )

    def test_pygame2_base_Rect_center(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.center:

        # Gets or sets the center position of the Rect.
        r = Rect( 1, 2, 3, 4 )
        new_center = (r.centerx+20,r.centery+30)
        expected_topleft = (r.left+20,r.top+30)
        old_size = r.size
        
        r.center = new_center
        self.assertEqual( new_center, r.center )
        self.assertEqual( expected_topleft, r.topleft )
        self.assertEqual( old_size, r.size )

    def test_pygame2_base_Rect_centerx(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.centerx:

        # Gets or sets the horizontal center position of the Rect.
        r = Rect( 1, 2, 3, 4 )
        new_centerx = r.centerx + 20
        expected_left = r.left + 20
        old_width = r.width
        
        r.centerx = new_centerx
        self.assertEqual( new_centerx, r.centerx )
        self.assertEqual( expected_left, r.left )
        self.assertEqual( old_width, r.width )

    def test_pygame2_base_Rect_centery(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.centery:

        # Gets or sets the vertical center position of the Rect.
        r = Rect( 1, 2, 3, 4 )
        new_centery = r.centery + 20
        expected_top = r.top + 20
        old_height = r.height
        
        r.centery = new_centery
        self.assertEqual( new_centery, r.centery )
        self.assertEqual( expected_top, r.top )
        self.assertEqual( old_height, r.height )
        
    def test_pygame2_base_Rect_clamp(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.clamp:

        # Rect.clamp (Rect) -> Rect
        # 
        # Moves the rectangle inside another.
        # 
        # Returns a new rectangle that is moved to be completely inside the
        # argument Rect. If the rectangle is too large to fit inside, it is
        # centered inside the argument Rect, but its size is not changed.
        r = Rect(10, 10, 10, 10)
        c = Rect(19, 12, 5, 5).clamp(r)
        self.assertEqual(c.right, r.right)
        self.assertEqual(c.top, 12)
        c = Rect(1, 2, 3, 4).clamp(r)
        self.assertEqual(c.topleft, r.topleft)
        c = Rect(5, 500, 22, 20).clamp(r)
        self.assertEqual(c.center, r.center)

    def test_pygame2_base_Rect_clamp_ip(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.clamp_ip:

        # Rect.clamp_ip (Rect) -> None
        # 
        # Moves the rectangle inside another, in place.
        # 
        # Same as Rect.clamp(Rect), but operates in place.
        r = Rect(10, 10, 10, 10)
        c = Rect(19, 12, 5, 5)
        c.clamp_ip(r)
        self.assertEqual(c.right, r.right)
        self.assertEqual(c.top, 12)
        c = Rect(1, 2, 3, 4)
        c.clamp_ip(r)
        self.assertEqual(c.topleft, r.topleft)
        c = Rect(5, 500, 22, 33)
        c.clamp_ip(r)
        self.assertEqual(c.center, r.center)

    def test_pygame2_base_Rect_clip(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.clip:

        # Rect.clip (Rect) -> Rect
        # 
        # Crops a rectangle inside another.
        # 
        # Returns a new rectangle that is cropped to be completely
        # inside the argument Rect. If the two rectangles do not overlap
        # to begin with, a Rect with 0 size is returned. Thus it returns
        # the area, in which both rects overlap.
        r1 = Rect( 1, 2, 3, 4 )
        self.assertEqual( Rect( 1, 2, 2, 2 ), r1.clip( Rect(0,0,3,4) ) )
        self.assertEqual( Rect( 2, 2, 2, 4 ), r1.clip( Rect(2,2,10,20) ) )
        self.assertEqual( Rect(2,3,1,2), r1.clip( Rect(2,3,1,2) ) )
        self.assertEqual( (0,0), r1.clip(Rect (20,30,5,6)).size )
        self.assertEqual( r1, r1.clip( Rect(r1) ),
                          "r1 does not clip an identical rect to itself" )

    def test_pygame2_base_Rect_collidedict(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.collidedict:

        # Rect.collidedict (dict) -> (key, value)
        # 
        # Test if one rectangle in a dictionary intersects.
        # 
        # Returns the key and value of the first dictionary value that
        # collides with the Rect. If no collisions are found, None is
        # returned. They keys of the passed dict must be Rect objects.

        r = Rect(1, 1, 10, 10)
        r1 = Rect(1, 1, 10, 10)
        r2 = Rect(50, 50, 10, 10)
        r3 = Rect(70, 70, 10, 10)
        r4 = Rect(61, 61, 10, 10)

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

    def test_pygame2_base_Rect_collidedictall(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.collidedictall:

        # Rect.collidedictall (dict) -> [(key, value), ...]
        # 
        # Test if all rectangles in a dictionary intersect.
        # 
        # Returns a list of all the key and value pairs that intersect
        # with the Rect. If no collisions are found an empty list is
        # returned. They keys of the passed dict must be Rect objects.

        r = Rect(1, 1, 10, 10)

        r2 = Rect(1, 1, 10, 10)
        r3 = Rect(5, 5, 10, 10)
        r4 = Rect(10, 10, 10, 10)
        r5 = Rect(50, 50, 10, 10)

        rects_values = 1
        d = {2: r2}
        l = r.collidedictall(d, rects_values)
        self.assertEqual(l, [(2, r2)])

        d2 = {2: r2, 3: r3, 4: r4, 5: r5}
        l2 = r.collidedictall(d2, rects_values)
        self.assertEqual(l2, [(2, r2), (3, r3), (4, r4)])

    def test_pygame2_base_Rect_collidelist(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.collidelist:

        # Rect.collidelist (list) -> index
        # 
        # Test if one rectangle in a list intersects.
        # 
        # Test whether the rectangle collides with any in a sequence of
        # rectangles. The index of the first collision found is
        # returned. If no collisions are found an index of -1 is
        # returned.

        r = Rect(1, 1, 10, 10)
        l = [Rect(50, 50, 1, 1), Rect(5, 5, 10, 10), Rect(15, 15, 1, 1)]

        self.assertEqual(r.collidelist(l), 1)
        self.assertEqual(r.collidelist(l, lambda x, y: x.top < y.top), 0)

        f = [Rect(50, 50, 1, 1), Rect(100, 100, 4, 4)]
        self.assertEqual(r.collidelist(f), -1)
        self.assertEqual(r.collidelist(l, lambda x, y: x.top > y.top), -1)

    def test_pygame2_base_Rect_collidelistall(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.collidelistall:

        # Rect.collidelistall (list) -> [index, ...]
        # 
        # Test if all rectangles in a list intersect.
        # 
        # Returns a list of all the indices that contain rectangles that
        # collide with the Rect. If no intersecting rectangles are
        # found, an empty list is returned.
        r = Rect(1, 1, 10, 10)

        l = [
            Rect(1, 1, 10, 10), 
            Rect(5, 5, 10, 10),
            Rect(15, 15, 1, 1),
            Rect(2, 2, 1, 1),
        ]
        self.assertEqual(r.collidelistall(l), [0, 1, 3])
        self.assertEqual(r.collidelistall(l, lambda x, y: x.top >= y.top), [0,])

        f = [Rect(50, 50, 1, 1), Rect(20, 20, 5, 5)]
        self.assertFalse(r.collidelistall(f))

    def test_pygame2_base_Rect_collidepoint(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.collidepoint:

        # Rect.collidepoint (x, y) -> bool
        # 
        # Test if a point is inside a rectangle.
        # 
        # Returns true if the given point is inside the rectangle.  A
        # point along the right or bottom edge is not considered to be
        # inside the rectangle.
        r = Rect( 1, 2, 3, 4 )
        
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

    def test_pygame2_base_Rect_colliderect(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.colliderect:

        # Rect.colliderect (Rect) -> bool
        # 
        # Test if two rectangles overlap.
        # 
        # Returns true if any portion of either rectangle overlap (except the
        # top+bottom or left+right edges).
        r1 = Rect(1,2,3,4)
        self.failUnless( r1.colliderect( Rect(0,0,2,3) ),
                         "r1 does not collide with Rect(0,0,2,3)" )
        self.failIf( r1.colliderect( Rect(0,0,1,2) ),
                     "r1 collides with Rect(0,0,1,2)" )
        self.failIf( r1.colliderect( Rect(r1.right,r1.bottom,2,2) ),
                     "r1 collides with Rect(r1.right,r1.bottom,2,2)" )
        self.failUnless( r1.colliderect( Rect(r1.left+1,r1.top+1,
                                              r1.width-2,r1.height-2) ),
                         "r1 does not collide with Rect(r1.left+1,r1.top+1,"+
                         "r1.width-2,r1.height-2)" )
        self.failUnless( r1.colliderect( Rect(r1.left-1,r1.top-1,
                                              r1.width+2,r1.height+2) ),
                         "r1 does not collide with Rect(r1.left-1,r1.top-1,"+
                         "r1.width+2,r1.height+2)" )
        self.failUnless( r1.colliderect( Rect(r1) ),
                         "r1 does not collide with an identical rect" )
        self.failIf( r1.colliderect( Rect(r1.right,r1.bottom,0,0) ),
                     "r1 collides with Rect(r1.right,r1.bottom,0,0)" )
        self.failIf( r1.colliderect( Rect(r1.right,r1.bottom,1,1) ),
                     "r1 collides with Rect(r1.right,r1.bottom,1,1)" )

    def test_pygame2_base_Rect_contains(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.contains:

        # Rect.contains (Rect) -> bool
        # 
        # Test if one rectangle is inside another.
        # 
        # Returns true when the argument rectangle is completely inside
        # the Rect.
        r = Rect( 1, 2, 3, 4 )
        
        self.failUnless( r.contains( Rect( 2, 3, 1, 1 ) ),
                         "r does not contain Rect(2,3,1,1)" )
        self.failUnless( r.contains( Rect(r) ),
                         "r does not contain the same rect as itself" )
        self.failUnless( r.contains( Rect(2,3,0,0) ),
                         "r does not contain an empty rect within its bounds" )
        self.failIf( r.contains( Rect(0,0,1,2) ),
                     "r contains Rect(0,0,1,2)" )
        self.failIf( r.contains( Rect(4,6,1,1) ),
                     "r contains Rect(4,6,1,1)" )
        self.failIf( r.contains( Rect(4,6,0,0) ),
                     "r contains Rect(4,6,0,0)" )
    
    def test_pygame2_base_Rect_copy(self):

        # __doc__ (as of 2009-02-23) for pygame2.base.Rect.copy:

        # copy () -> Rect
        #
        # Creates a copy of the Rect.
        #
        # Returns a new Rect, that contains the same values as the
        # caller.
        r = Rect( 1, 2, 3, 4 )
        cp = r.copy ()
        self.failUnless (r == cp, "r (1, 2, 3, 4) is not equal to its copy")

        r = Rect( -10, 50, 10, 40 )
        cp = r.copy ()
        self.failUnless (r == cp,
                         "r (-10, 50, 10, 40) is not equal to its copy")
        
        r = Rect( 2, -5, 10, 40 )
        cp = r.copy ()
        self.failUnless (r == cp,
                         "r (2, -5, 10, 40) is not equal to its copy")
        
        r = Rect( -2, -5, 10, 40 )
        cp = r.copy ()
        self.failUnless (r == cp,
                         "r (-2, -5, 10, 40) is not equal to its copy")
        
    def test_pygame2_base_Rect_fit(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.fit:

        # Rect.fit (Rect) -> Rect
        # 
        # Resize and move a rectangle with aspect ratio.
        # 
        # Returns a new rectangle that is moved and resized to fit
        # another. The aspect ratio of the original Rect is preserved,
        # so the new rectangle may be smaller than the target in either
        # width or height.

        r = Rect(10, 10, 30, 30)

        r2 = Rect(30, 30, 15, 10)

        f = r.fit(r2)
        self.assertTrue(r2.contains(f))
        
        f2 = r2.fit(r)
        self.assertTrue(r.contains(f2))

    def test_pygame2_base_Rect_height(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.height:

        # Gets or sets the height of the Rect.
        r = Rect( 1, 2, 3, 4 )
        new_height = 10
        old_topleft = r.topleft
        old_width = r.width
        
        r.height = new_height
        self.assertEqual( new_height, r.height )
        self.assertEqual( old_width, r.width )
        self.assertEqual( old_topleft, r.topleft )
    
    def test_pygame2_base_Rect_h(self):

        r = Rect( 1, 2, 3, 4 )
        new_height = 10
        old_topleft = r.topleft
        old_width = r.width
        
        r.h = new_height
        self.assertEqual( new_height, r.h )
        self.assertEqual( old_width, r.width )
        self.assertEqual( old_topleft, r.topleft )
    
    def test_pygame2_base_Rect_inflate(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.inflate:

        # Rect.inflate (x, y) -> Rect
        # 
        # Grow or shrink the rectangle size.
        # 
        # Returns a new rectangle with the size changed by the given offset.
        # The rectangle remains centered around its current center. Negative
        # values will shrink the rectangle.
        pass

    def test_inflate__larger( self ):
        "The inflate method inflates around the center of the rectangle"
        r = Rect( 2, 4, 6, 8 )
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
        r = Rect( 2, 4, 6, 8 )
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
        r = Rect( 2, 4, 6, 8 )
        r2 = Rect( r )
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
        r = Rect( 2, 4, 6, 8 )
        r2 = Rect( r )
        r2.inflate_ip( -4, -6 )
        
        self.assertEqual( r.center, r2.center )
        self.assertEqual( r.left+2, r2.left )
        self.assertEqual( r.top+3, r2.top )
        self.assertEqual( r.right-2, r2.right )
        self.assertEqual( r.bottom-3, r2.bottom )
        self.assertEqual( r.width-4, r2.width )
        self.assertEqual( r.height-6, r2.height )

    def test_pygame2_base_Rect_inflate_ip(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.inflate_ip:

        # Rect.inflate_ip (x, y) -> None
        # 
        # Grow or shrink the rectangle size, in place.
        # 
        # Same as Rect.inflate(x, y), but operates in place.
        pass

    def test_pygame2_base_Rect_left(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.left:

        # Gets or sets the left edge position of the Rect.
        r = Rect( 1, 2, 3, 4 )
        new_left = 10
        
        r.left = new_left
        self.assertEqual( new_left, r.left )
        self.assertEqual( Rect(new_left,2,3,4), r )

    def test_pygame2_base_Rect_midbottom(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.midbottom:

        # Gets or sets the mid bottom edge position of the Rect.
        r = Rect( 1, 2, 3, 4 )
        new_midbottom = (r.centerx+20,r.bottom+30)
        expected_topleft = (r.left+20,r.top+30)
        old_size = r.size
        
        r.midbottom = new_midbottom
        self.assertEqual( new_midbottom, r.midbottom )
        self.assertEqual( expected_topleft, r.topleft )
        self.assertEqual( old_size, r.size )

    def test_pygame2_base_Rect_midleft(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.midleft:

        # Gets or sets the mid left edge position of the Rect.
        r = Rect( 1, 2, 3, 4 )
        new_midleft = (r.left+20,r.centery+30)
        expected_topleft = (r.left+20,r.top+30)
        old_size = r.size
        
        r.midleft = new_midleft
        self.assertEqual( new_midleft, r.midleft )
        self.assertEqual( expected_topleft, r.topleft )
        self.assertEqual( old_size, r.size )

    def test_pygame2_base_Rect_midright(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.midright:

        # Gets or sets the mid right edge position of the Rect.
        r = Rect( 1, 2, 3, 4 )
        new_midright= (r.right+20,r.centery+30)
        expected_topleft = (r.left+20,r.top+30)
        old_size = r.size
        
        r.midright = new_midright
        self.assertEqual( new_midright, r.midright )
        self.assertEqual( expected_topleft, r.topleft )
        self.assertEqual( old_size, r.size )

    def test_pygame2_base_Rect_midtop(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.midtop:

        # Gets or sets the mid top edge position of the Rect.
        r = Rect( 1, 2, 3, 4 )
        new_midtop= (r.centerx+20,r.top+30)
        expected_topleft = (r.left+20,r.top+30)
        old_size = r.size
        
        r.midtop = new_midtop
        self.assertEqual( new_midtop, r.midtop )
        self.assertEqual( expected_topleft, r.topleft )
        self.assertEqual( old_size, r.size )

    def test_pygame2_base_Rect_move(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.move:

        # Rect.move (x, y) -> Rect
        # 
        # Moves the rectangle.
        # 
        # Returns a new rectangle that is moved by the given offset. The
        # x and y arguments can be any integer value, positive or
        # negative.
        r = Rect( 1, 2, 3, 4 )
        move_x = 10
        move_y = 20
        r2 = r.move( move_x, move_y )
        expected_r2 = Rect(r.left+move_x,r.top+move_y,r.width,r.height)
        self.assertEqual( expected_r2, r2 )

    def test_pygame2_base_Rect_move_ip(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.move_ip:

        # Rect.move_ip (x, y) -> None
        # 
        # Moves the rectangle, in place.
        # 
        # Same as Rect.move (x, y), but operates in place.
        r = Rect( 1, 2, 3, 4 )
        r2 = Rect( r )
        move_x = 10
        move_y = 20
        r2.move_ip( move_x, move_y )
        expected_r2 = Rect(r.left+move_x,r.top+move_y,r.width,r.height)
        self.assertEqual( expected_r2, r2 )
    
    def test_pygame2_base_Rect_right(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.right:

        # Gets or sets the right position of the Rect.
        r = Rect( 1, 2, 3, 4 )
        new_right = r.right + 20
        expected_left = r.left + 20
        old_width = r.width
        
        r.right = new_right
        self.assertEqual( new_right, r.right )
        self.assertEqual( expected_left, r.left )
        self.assertEqual( old_width, r.width )

    def test_pygame2_base_Rect_size(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.size:

        # Gets or sets the width and height of the Rect as 2-value tuple.
        r = Rect( 1, 2, 3, 4 )
        new_size = (10,20)
        old_topleft = r.topleft
        
        r.size = new_size
        self.assertEqual( new_size, r.size )
        self.assertEqual( old_topleft, r.topleft )

    def test_pygame2_base_Rect_top(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.top:

        # Gets or sets the top edge position of the Rect.
        r = Rect( 1, 2, 3, 4 )
        new_top = 10
        
        r.top = new_top
        self.assertEqual( Rect(1,new_top,3,4), r )
        self.assertEqual( new_top, r.top )

    def test_pygame2_base_Rect_topleft(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.topleft:

        # Gets or sets the top left corner position of the Rect.

        r = Rect( 1, 2, 3, 4 )
        new_topleft = (r.left+20,r.top+30)
        old_size = r.size
        
        r.topleft = new_topleft
        self.assertEqual( new_topleft, r.topleft )
        self.assertEqual( old_size, r.size )

    def test_pygame2_base_Rect_topright(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.topright:

        # Gets or sets the top right corner position of the Rect.
        r = Rect( 1, 2, 3, 4 )
        new_topright = (r.right+20,r.top+30)
        expected_topleft = (r.left+20,r.top+30)
        old_size = r.size
        
        r.topright = new_topright
        self.assertEqual( new_topright, r.topright )
        self.assertEqual( expected_topleft, r.topleft )
        self.assertEqual( old_size, r.size )

    def test_pygame2_base_Rect_union(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.union:

        # Rect.union (Rect) -> Rect
        # 
        # Joins two rectangles into one.
        # 
        # Returns a new rectangle that completely covers the area of the
        # two provided rectangles. There may be area inside the new Rect
        # that is not covered by the originals.
        r1 = Rect( 1, 1, 1, 2 )
        r2 = Rect( -2, -2, 1, 2 )
        self.assertEqual( Rect( -2, -2, 4, 5 ), r1.union(r2) )

    def test_union__with_identical_Rect( self ):
        r1 = Rect( 1, 2, 3, 4 )
        self.assertEqual( r1, r1.union( Rect(r1) ) )

    def test_pygame2_base_Rect_union_ip(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.union_ip:

        # Rect.union_ip (Rect) -> Rect
        # 
        # Joins two rectangles into one, in place.
        # 
        # Same as Rect.union(Rect), but operates in place.
        r1 = Rect( 1, 1, 1, 2 )
        r2 = Rect( -2, -2, 1, 2 )
        r1.union_ip(r2)
        self.assertEqual( Rect( -2, -2, 4, 5 ), r1 ) 

    def test_union__list( self ):
        r1 = Rect( 0, 0, 1, 1 )
        r2 = Rect( -2, -2, 1, 1 )
        r3 = Rect( 2, 2, 1, 1 )
        
        r4 = r1.union( [r2,r3] )
        self.assertEqual( Rect(-2, -2, 5, 5), r4 )
    
    def test_union_ip__list( self ):
        r1 = Rect( 0, 0, 1, 1 )
        r2 = Rect( -2, -2, 1, 1 )
        r3 = Rect( 2, 2, 1, 1 )
        
        r1.union_ip( [r2,r3] )
        self.assertEqual( Rect(-2, -2, 5, 5), r1 )

    def test_pygame2_base_Rect_width(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.width:

        # Gets or sets the width of the Rect.
        r = Rect( 1, 2, 3, 4 )
        new_width = 10
        old_topleft = r.topleft
        old_height = r.height
        
        r.width = new_width
        self.assertEqual( new_width, r.width )
        self.assertEqual( old_height, r.height )
        self.assertEqual( old_topleft, r.topleft )
    
    def test_pygame2_base_Rect_w(self):
        r = Rect( 1, 2, 3, 4 )
        new_width = 10
        old_topleft = r.topleft
        old_height = r.height
        
        r.w = new_width
        self.assertEqual( new_width, r.w )
        self.assertEqual( old_height, r.height )
        self.assertEqual( old_topleft, r.topleft )
    
    def test_pygame2_base_Rect_x(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.x:

        # Gets or sets the horizontal top left position of the Rect.
        r = Rect (1, 2, 3, 4)
        self.assertEqual (r.x, 1)
        r.topleft = 32, 10
        self.assertEqual (r.x, 32)
        r.left = -44
        self.assertEqual (r.x, -44)
        r.move_ip (10, 33)
        self.assertEqual (r.x, -34)

    def test_pygame2_base_Rect_y(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.y:

        # Gets or sets the vertical top left position of the Rect.
        r = Rect (1, 2, 3, 4)
        self.assertEqual (r.y, 2)
        r.topleft = 32, 10
        self.assertEqual (r.y, 10)
        r.top = -44
        self.assertEqual (r.y, -44)
        r.move_ip (10, 33)
        self.assertEqual (r.y, -11)

    def test_pygame2_base_Rect___repr__(self):
        r = Rect (10, 4, 7, 99)
        text = "Rect(10, 4, 7, 99)"
        self.assertEqual (repr (r), text)

if __name__ == "__main__":
    unittest.main ()
