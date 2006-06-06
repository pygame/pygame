


import unittest
from pygame import Rect

class RectTest( unittest.TestCase ):
    def testConstructionXYWidthHeight( self ):
        r = Rect(1,2,3,4)
        self.assertEqual( 1, r.left )
        self.assertEqual( 2, r.top )
        self.assertEqual( 3, r.width )
        self.assertEqual( 4, r.height )

    def testConstructionTopLeftSize( self ):
        r = Rect( (1,2), (3,4) )
        self.assertEqual( 1, r.left )
        self.assertEqual( 2, r.top )
        self.assertEqual( 3, r.width )
        self.assertEqual( 4, r.height )

    def testCalculatedAttributes( self ):
        r = Rect( 1, 2, 3, 4 )
        
        self.assertEqual( r.left+r.width, r.right )
        self.assertEqual( r.top+r.height, r.bottom )
        self.assertEqual( (r.width,r.height), r.size )
        self.assertEqual( (r.left,r.top), r.topleft )
        self.assertEqual( (r.right,r.top), r.topright )
        self.assertEqual( (r.left,r.bottom), r.bottomleft )
        self.assertEqual( (r.right,r.bottom), r.bottomright )

        midx = r.left + r.width/2
        midy = r.top + r.height/2

        self.assertEqual( midx, r.centerx )
        self.assertEqual( midy, r.centery )
        self.assertEqual( (r.centerx,r.centery), r.center )
        self.assertEqual( (r.centerx,r.top), r.midtop )
        self.assertEqual( (r.centerx,r.bottom), r.midbottom )
        self.assertEqual( (r.left,r.centery), r.midleft )
        self.assertEqual( (r.right,r.centery), r.midright )
    
    def testNormalize( self ):
        r = Rect( 1, 2, -3, -6 )
        r2 = Rect(r)
        r2.normalize()
        self.failUnless( r2.width >= 0 )
        self.failUnless( r2.height >= 0 )
        self.assertEqual( (abs(r.width),abs(r.height)), r2.size )
        self.assertEqual( (-2,-4), r2.topleft )

    def testSetLeft( self ):
        """Changing the left attribute moves the rect and does not change
           the rect's width
        """
        r = Rect( 1, 2, 3, 4 )
        new_left = 10
        
        r.left = new_left
        self.assertEqual( new_left, r.left )
        self.assertEqual( Rect(new_left,2,3,4), r )
    
    def testSetRight( self ):
        """Changing the right attribute moves the rect and does not change
           the rect's width
        """
        r = Rect( 1, 2, 3, 4 )
        new_right = r.right + 20
        expected_left = r.left + 20
        old_width = r.width
        
        r.right = new_right
        self.assertEqual( new_right, r.right )
        self.assertEqual( expected_left, r.left )
        self.assertEqual( old_width, r.width )
       
    def testSetTop( self ):
        """Changing the top attribute moves the rect and does not change
           the rect's width
        """
        r = Rect( 1, 2, 3, 4 )
        new_top = 10
        
        r.top = new_top
        self.assertEqual( Rect(1,new_top,3,4), r )
        self.assertEqual( new_top, r.top )
    
    def testSetBottom( self ):
        """Changing the bottom attribute moves the rect and does not change
           the rect's height
        """
        r = Rect( 1, 2, 3, 4 )
        new_bottom = r.bottom + 20
        expected_top = r.top + 20
        old_height = r.height
        
        r.bottom = new_bottom
        self.assertEqual( new_bottom, r.bottom )
        self.assertEqual( expected_top, r.top )
        self.assertEqual( old_height, r.height )
    
    def testSetCenterX( self ):
        """Changing the centerx attribute moves the rect and does not change
           the rect's width
        """
        r = Rect( 1, 2, 3, 4 )
        new_centerx = r.centerx + 20
        expected_left = r.left + 20
        old_width = r.width
        
        r.centerx = new_centerx
        self.assertEqual( new_centerx, r.centerx )
        self.assertEqual( expected_left, r.left )
        self.assertEqual( old_width, r.width )
    
    def testSetCenterY( self ):
        """Changing the centerx attribute moves the rect and does not change
           the rect's width
        """
        r = Rect( 1, 2, 3, 4 )
        new_centery = r.centery + 20
        expected_top = r.top + 20
        old_height = r.height
        
        r.centery = new_centery
        self.assertEqual( new_centery, r.centery )
        self.assertEqual( expected_top, r.top )
        self.assertEqual( old_height, r.height )
    
    def testSetTopLeft( self ):
        """Changing the topleft attribute moves the rect and does not change
           the rect's size
        """
        r = Rect( 1, 2, 3, 4 )
        new_topleft = (r.left+20,r.top+30)
        old_size = r.size
        
        r.topleft = new_topleft
        self.assertEqual( new_topleft, r.topleft )
        self.assertEqual( old_size, r.size )
        
    def testSetBottomLeft( self ):
        """Changing the bottomleft attribute moves the rect and does not change
           the rect's size
        """
        r = Rect( 1, 2, 3, 4 )
        new_bottomleft = (r.left+20,r.bottom+30)
        expected_topleft = (r.left+20,r.top+30)
        old_size = r.size
        
        r.bottomleft = new_bottomleft
        self.assertEqual( new_bottomleft, r.bottomleft )
        self.assertEqual( expected_topleft, r.topleft )
        self.assertEqual( old_size, r.size )
    
    def testSetTopRight( self ):
        """Changing the bottomleft attribute moves the rect and does not change
           the rect's size
        """
        r = Rect( 1, 2, 3, 4 )
        new_topright = (r.right+20,r.top+30)
        expected_topleft = (r.left+20,r.top+30)
        old_size = r.size
        
        r.topright = new_topright
        self.assertEqual( new_topright, r.topright )
        self.assertEqual( expected_topleft, r.topleft )
        self.assertEqual( old_size, r.size )
    
    def testSetBottomRight( self ):
        """Changing the bottomright attribute moves the rect and does not change
           the rect's size
        """
        r = Rect( 1, 2, 3, 4 )
        new_bottomright = (r.right+20,r.bottom+30)
        expected_topleft = (r.left+20,r.top+30)
        old_size = r.size
        
        r.bottomright = new_bottomright
        self.assertEqual( new_bottomright, r.bottomright )
        self.assertEqual( expected_topleft, r.topleft )
        self.assertEqual( old_size, r.size )
    
    def testSetCenter( self ):
        """Changing the center attribute moves the rect and does not change
           the rect's size
        """
        r = Rect( 1, 2, 3, 4 )
        new_center = (r.centerx+20,r.centery+30)
        expected_topleft = (r.left+20,r.top+30)
        old_size = r.size
        
        r.center = new_center
        self.assertEqual( new_center, r.center )
        self.assertEqual( expected_topleft, r.topleft )
        self.assertEqual( old_size, r.size )
    
    def testSetMidLeft( self ):
        """Changing the midleft attribute moves the rect and does not change
           the rect's size
        """
        r = Rect( 1, 2, 3, 4 )
        new_midleft = (r.left+20,r.centery+30)
        expected_topleft = (r.left+20,r.top+30)
        old_size = r.size
        
        r.midleft = new_midleft
        self.assertEqual( new_midleft, r.midleft )
        self.assertEqual( expected_topleft, r.topleft )
        self.assertEqual( old_size, r.size )
    
    def testSetMidRight( self ):
        """Changing the midright attribute moves the rect and does not change
           the rect's size
        """
        r = Rect( 1, 2, 3, 4 )
        new_midright= (r.right+20,r.centery+30)
        expected_topleft = (r.left+20,r.top+30)
        old_size = r.size
        
        r.midright = new_midright
        self.assertEqual( new_midright, r.midright )
        self.assertEqual( expected_topleft, r.topleft )
        self.assertEqual( old_size, r.size )
    
    def testSetMidTop( self ):
        """Changing the midtop attribute moves the rect and does not change
           the rect's size
        """
        r = Rect( 1, 2, 3, 4 )
        new_midtop= (r.centerx+20,r.top+30)
        expected_topleft = (r.left+20,r.top+30)
        old_size = r.size
        
        r.midtop = new_midtop
        self.assertEqual( new_midtop, r.midtop )
        self.assertEqual( expected_topleft, r.topleft )
        self.assertEqual( old_size, r.size )
    
    def testSetMidBottom( self ):
        """Changing the midbottom attribute moves the rect and does not change
           the rect's size
        """
        r = Rect( 1, 2, 3, 4 )
        new_midbottom = (r.centerx+20,r.bottom+30)
        expected_topleft = (r.left+20,r.top+30)
        old_size = r.size
        
        r.midbottom = new_midbottom
        self.assertEqual( new_midbottom, r.midbottom )
        self.assertEqual( expected_topleft, r.topleft )
        self.assertEqual( old_size, r.size )
    
    def testSetWidth( self ):
        "Changing the width resizes the rect from the top-left corner"
        r = Rect( 1, 2, 3, 4 )
        new_width = 10
        old_topleft = r.topleft
        old_height = r.height
        
        r.width = new_width
        self.assertEqual( new_width, r.width )
        self.assertEqual( old_height, r.height )
        self.assertEqual( old_topleft, r.topleft )
    
    def testSetHeight( self ):
        "Changing the height resizes the rect from the top-left corner"
        r = Rect( 1, 2, 3, 4 )
        new_height = 10
        old_topleft = r.topleft
        old_width = r.width
        
        r.height = new_height
        self.assertEqual( new_height, r.height )
        self.assertEqual( old_width, r.width )
        self.assertEqual( old_topleft, r.topleft )
    
    def testSetSize( self ):
        "Changing the size resizes the rect from the top-left corner"
        r = Rect( 1, 2, 3, 4 )
        new_size = (10,20)
        old_topleft = r.topleft
        
        r.size = new_size
        self.assertEqual( new_size, r.size )
        self.assertEqual( old_topleft, r.topleft )

    def testContains( self ):
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
    
    def testCollidePoint( self ):
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

    def testInflateLarger( self ):
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

    def testInflateSmaller( self ):
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

    def testInflateLargerIP( self ):    
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

    def testInflateSmallerIP( self ):
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

    def testClamp( self ):
        r = Rect(10, 10, 10, 10)
        c = Rect(19, 12, 5, 5).clamp(r)
        self.assertEqual(c.right, r.right)
        self.assertEqual(c.top, 12)
        c = Rect(1, 2, 3, 4).clamp(r)
        self.assertEqual(c.topleft, r.topleft)
        c = Rect(5, 500, 22, 33).clamp(r)
        self.assertEqual(c.center, r.center)

    def testClampIP( self ):
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
        
    def testClip( self ):
        r1 = Rect( 1, 2, 3, 4 )
        self.assertEqual( Rect( 1, 2, 2, 2 ), r1.clip( Rect(0,0,3,4) ) )
        self.assertEqual( Rect( 2, 2, 2, 4 ), r1.clip( Rect(2,2,10,20) ) )
        self.assertEqual( Rect(2,3,1,2), r1.clip( Rect(2,3,1,2) ) )
        self.assertEqual( (0,0), r1.clip(20,30,5,6).size )
        self.assertEqual( r1, r1.clip( Rect(r1) ),
                          "r1 does not clip an identical rect to itself" )
        
    def testMove( self ):
        r = Rect( 1, 2, 3, 4 )
        move_x = 10
        move_y = 20
        r2 = r.move( move_x, move_y )
        expected_r2 = Rect(r.left+move_x,r.top+move_y,r.width,r.height)
        self.assertEqual( expected_r2, r2 )
    
    def testMoveIP( self ):    
        r = Rect( 1, 2, 3, 4 )
        r2 = Rect( r )
        move_x = 10
        move_y = 20
        r2.move_ip( move_x, move_y )
        expected_r2 = Rect(r.left+move_x,r.top+move_y,r.width,r.height)
        self.assertEqual( expected_r2, r2 )
    
    def testUnion( self ):
        r1 = Rect( 1, 1, 1, 2 )
        r2 = Rect( -2, -2, 1, 2 )
        self.assertEqual( Rect( -2, -2, 4, 5 ), r1.union(r2) )
    
    def testUnionWithIdenticalRect( self ):
        r1 = Rect( 1, 2, 3, 4 )
        self.assertEqual( r1, r1.union( Rect(r1) ) )
    
    def testUnionIP( self ):
        r1 = Rect( 1, 1, 1, 2 )
        r2 = Rect( -2, -2, 1, 2 )
        r1.union_ip(r2)
        self.assertEqual( Rect( -2, -2, 4, 5 ), r1 )
    
    def testUnionAll( self ):
        r1 = Rect( 0, 0, 1, 1 )
        r2 = Rect( -2, -2, 1, 1 )
        r3 = Rect( 2, 2, 1, 1 )
        
        r4 = r1.unionall( [r2,r3] )
        self.assertEqual( Rect(-2, -2, 5, 5), r4 )
    
    def testUnionAllIP( self ):
        r1 = Rect( 0, 0, 1, 1 )
        r2 = Rect( -2, -2, 1, 1 )
        r3 = Rect( 2, 2, 1, 1 )
        
        r1.unionall_ip( [r2,r3] )
        self.assertEqual( Rect(-2, -2, 5, 5), r1 )



    def testCollideRect( self ):
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
        rect_list.remove(r2)
        self.assertRaises(ValueError, rect_list.remove, r2)





if __name__ == '__main__':
    unittest.main()
