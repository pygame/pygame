import unittest
from pygame import Rect

class RectTypeTest(unittest.TestCase):
    def testConstructionXYWidthHeight(self):
        r = Rect(1, 2, 3, 4)
        self.assertEqual(1, r.left)
        self.assertEqual(2, r.top)
        self.assertEqual(3, r.width)
        self.assertEqual(4, r.height)

    def testConstructionTopLeftSize(self):
        r = Rect((1, 2), (3, 4))
        self.assertEqual(1, r.left)
        self.assertEqual(2, r.top)
        self.assertEqual(3, r.width)
        self.assertEqual(4, r.height)

    def testCalculatedAttributes(self):
        r = Rect(1, 2, 3, 4)

        self.assertEqual(r.left + r.width, r.right)
        self.assertEqual(r.top + r.height, r.bottom)
        self.assertEqual((r.width, r.height), r.size)
        self.assertEqual((r.left, r.top), r.topleft)
        self.assertEqual((r.right, r.top), r.topright)
        self.assertEqual((r.left, r.bottom), r.bottomleft)
        self.assertEqual((r.right, r.bottom), r.bottomright)

        midx = r.left + r.width // 2
        midy = r.top + r.height // 2

        self.assertEqual(midx, r.centerx)
        self.assertEqual(midy, r.centery)
        self.assertEqual((r.centerx, r.centery), r.center)
        self.assertEqual((r.centerx, r.top), r.midtop)
        self.assertEqual((r.centerx, r.bottom), r.midbottom)
        self.assertEqual((r.left, r.centery), r.midleft)
        self.assertEqual((r.right, r.centery), r.midright)

    def test_normalize(self):
        r = Rect(1, 2, -3, -6)
        r2 = Rect(r)
        r2.normalize()
        self.assertTrue(r2.width >= 0)
        self.assertTrue(r2.height >= 0)
        self.assertEqual((abs(r.width), abs(r.height)), r2.size)
        self.assertEqual((-2, -4), r2.topleft)

    def test_left(self):
        """Changing the left attribute moves the rect and does not change
           the rect's width
        """
        r = Rect(1, 2, 3, 4)
        new_left = 10

        r.left = new_left
        self.assertEqual(new_left, r.left)
        self.assertEqual(Rect(new_left, 2, 3, 4), r)

    def test_right(self):
        """Changing the right attribute moves the rect and does not change
           the rect's width
        """
        r = Rect(1, 2, 3, 4)
        new_right = r.right + 20
        expected_left = r.left + 20
        old_width = r.width

        r.right = new_right
        self.assertEqual(new_right, r.right)
        self.assertEqual(expected_left, r.left)
        self.assertEqual(old_width, r.width)

    def test_top(self):
        """Changing the top attribute moves the rect and does not change
           the rect's width
        """
        r = Rect(1, 2, 3, 4)
        new_top = 10

        r.top = new_top
        self.assertEqual(Rect(1, new_top, 3, 4), r)
        self.assertEqual(new_top, r.top)

    def test_bottom(self):
        """Changing the bottom attribute moves the rect and does not change
           the rect's height
        """
        r = Rect(1, 2, 3, 4)
        new_bottom = r.bottom + 20
        expected_top = r.top + 20
        old_height = r.height

        r.bottom = new_bottom
        self.assertEqual(new_bottom, r.bottom)
        self.assertEqual(expected_top, r.top)
        self.assertEqual(old_height, r.height)

    def test_centerx(self):
        """Changing the centerx attribute moves the rect and does not change
           the rect's width
        """
        r = Rect(1, 2, 3, 4)
        new_centerx = r.centerx + 20
        expected_left = r.left + 20
        old_width = r.width

        r.centerx = new_centerx
        self.assertEqual(new_centerx, r.centerx)
        self.assertEqual(expected_left, r.left)
        self.assertEqual(old_width, r.width)

    def test_centery(self):
        """Changing the centerx attribute moves the rect and does not change
           the rect's width
        """
        r = Rect(1, 2, 3, 4)
        new_centery = r.centery + 20
        expected_top = r.top + 20
        old_height = r.height

        r.centery = new_centery
        self.assertEqual(new_centery, r.centery)
        self.assertEqual(expected_top, r.top)
        self.assertEqual(old_height, r.height)

    def test_topleft(self):
        """Changing the topleft attribute moves the rect and does not change
           the rect's size
        """
        r = Rect(1, 2, 3, 4)
        new_topleft = (r.left + 20, r.top + 30)
        old_size = r.size

        r.topleft = new_topleft
        self.assertEqual(new_topleft, r.topleft)
        self.assertEqual(old_size, r.size)

    def test_bottomleft(self):
        """Changing the bottomleft attribute moves the rect and does not change
           the rect's size
        """
        r = Rect(1, 2, 3, 4)
        new_bottomleft = (r.left + 20, r.bottom + 30)
        expected_topleft = (r.left + 20, r.top + 30)
        old_size = r.size

        r.bottomleft = new_bottomleft
        self.assertEqual(new_bottomleft, r.bottomleft)
        self.assertEqual(expected_topleft, r.topleft)
        self.assertEqual(old_size, r.size)

    def test_topright(self):
        """Changing the bottomleft attribute moves the rect and does not change
           the rect's size
        """
        r = Rect(1, 2, 3, 4)
        new_topright = (r.right + 20, r.top + 30)
        expected_topleft = (r.left + 20, r.top + 30)
        old_size = r.size

        r.topright = new_topright
        self.assertEqual(new_topright, r.topright)
        self.assertEqual(expected_topleft, r.topleft)
        self.assertEqual(old_size, r.size)

    def test_bottomright(self):
        """Changing the bottomright attribute moves the rect and does not change
           the rect's size
        """
        r = Rect(1, 2, 3, 4)
        new_bottomright = (r.right + 20, r.bottom + 30)
        expected_topleft = (r.left + 20, r.top + 30)
        old_size = r.size

        r.bottomright = new_bottomright
        self.assertEqual(new_bottomright, r.bottomright)
        self.assertEqual(expected_topleft, r.topleft)
        self.assertEqual(old_size, r.size)

    def test_center(self):
        """Changing the center attribute moves the rect and does not change
           the rect's size
        """
        r = Rect(1, 2, 3, 4)
        new_center = (r.centerx + 20, r.centery + 30)
        expected_topleft = (r.left + 20, r.top + 30)
        old_size = r.size

        r.center = new_center
        self.assertEqual(new_center, r.center)
        self.assertEqual(expected_topleft, r.topleft)
        self.assertEqual(old_size, r.size)

    def test_midleft(self):
        """Changing the midleft attribute moves the rect and does not change
           the rect's size
        """
        r = Rect(1, 2, 3, 4)
        new_midleft = (r.left + 20, r.centery + 30)
        expected_topleft = (r.left + 20, r.top + 30)
        old_size = r.size

        r.midleft = new_midleft
        self.assertEqual(new_midleft, r.midleft)
        self.assertEqual(expected_topleft, r.topleft)
        self.assertEqual(old_size, r.size)

    def test_midright(self):
        """Changing the midright attribute moves the rect and does not change
           the rect's size
        """
        r = Rect(1, 2, 3, 4)
        new_midright= (r.right + 20, r.centery + 30)
        expected_topleft = (r.left + 20, r.top + 30)
        old_size = r.size

        r.midright = new_midright
        self.assertEqual(new_midright, r.midright)
        self.assertEqual(expected_topleft, r.topleft)
        self.assertEqual(old_size, r.size)

    def test_midtop(self):
        """Changing the midtop attribute moves the rect and does not change
           the rect's size
        """
        r = Rect(1, 2, 3, 4)
        new_midtop= (r.centerx + 20, r.top + 30)
        expected_topleft = (r.left + 20, r.top + 30)
        old_size = r.size

        r.midtop = new_midtop
        self.assertEqual(new_midtop, r.midtop)
        self.assertEqual(expected_topleft, r.topleft)
        self.assertEqual(old_size, r.size)

    def test_midbottom(self):
        """Changing the midbottom attribute moves the rect and does not change
           the rect's size
        """
        r = Rect(1, 2, 3, 4)
        new_midbottom = (r.centerx + 20, r.bottom + 30)
        expected_topleft = (r.left + 20, r.top + 30)
        old_size = r.size

        r.midbottom = new_midbottom
        self.assertEqual(new_midbottom, r.midbottom)
        self.assertEqual(expected_topleft, r.topleft)
        self.assertEqual(old_size, r.size)

    def test_width(self):
        """Changing the width resizes the rect from the top-left corner
        """
        r = Rect(1, 2, 3, 4)
        new_width = 10
        old_topleft = r.topleft
        old_height = r.height

        r.width = new_width
        self.assertEqual(new_width, r.width)
        self.assertEqual(old_height, r.height)
        self.assertEqual(old_topleft, r.topleft)

    def test_height(self):
        """Changing the height resizes the rect from the top-left corner
        """
        r = Rect(1, 2, 3, 4)
        new_height = 10
        old_topleft = r.topleft
        old_width = r.width

        r.height = new_height
        self.assertEqual(new_height, r.height)
        self.assertEqual(old_width, r.width)
        self.assertEqual(old_topleft, r.topleft)

    def test_size(self):
        """Changing the size resizes the rect from the top-left corner
        """
        r = Rect(1, 2, 3, 4)
        new_size = (10, 20)
        old_topleft = r.topleft

        r.size = new_size
        self.assertEqual(new_size, r.size)
        self.assertEqual(old_topleft, r.topleft)

    def test_contains(self):
        r = Rect(1, 2, 3, 4)

        self.assertTrue(r.contains(Rect(2, 3, 1, 1)),
                        "r does not contain Rect(2, 3, 1, 1)")
        self.assertTrue(r.contains(Rect(r)),
                        "r does not contain the same rect as itself")
        self.assertTrue(r.contains(Rect(2, 3, 0, 0)),
                        "r does not contain an empty rect within its bounds")
        self.assertFalse(r.contains(Rect(0, 0, 1, 2)),
                         "r contains Rect(0, 0, 1, 2)")
        self.assertFalse(r.contains(Rect(4, 6, 1, 1)),
                         "r contains Rect(4, 6, 1, 1)")
        self.assertFalse(r.contains(Rect(4, 6, 0, 0)),
                         "r contains Rect(4, 6, 0, 0)")

    def test_collidepoint(self):
        r = Rect(1, 2, 3, 4)

        self.assertTrue(r.collidepoint(r.left, r.top),
                        "r does not collide with point (left, top)")
        self.assertFalse(r.collidepoint(r.left - 1, r.top),
                         "r collides with point (left - 1, top)")
        self.assertFalse(r.collidepoint(r.left, r.top - 1),
                         "r collides with point (left, top - 1)")
        self.assertFalse(r.collidepoint(r.left - 1, r.top - 1),
                         "r collides with point (left - 1, top - 1)")

        self.assertTrue(r.collidepoint(r.right - 1, r.bottom - 1),
                        "r does not collide with point (right - 1, bottom - 1)")
        self.assertFalse(r.collidepoint(r.right, r.bottom),
                         "r collides with point (right, bottom)")
        self.assertFalse(r.collidepoint(r.right - 1, r.bottom),
                         "r collides with point (right - 1, bottom)")
        self.assertFalse(r.collidepoint(r.right, r.bottom - 1),
                         "r collides with point (right, bottom - 1)")

    def test_inflate__larger(self):
        """The inflate method inflates around the center of the rectangle
        """
        r = Rect(2, 4, 6, 8)
        r2 = r.inflate(4, 6)

        self.assertEqual(r.center, r2.center)
        self.assertEqual(r.left - 2, r2.left)
        self.assertEqual(r.top - 3, r2.top)
        self.assertEqual(r.right + 2, r2.right)
        self.assertEqual(r.bottom + 3, r2.bottom)
        self.assertEqual(r.width + 4, r2.width)
        self.assertEqual(r.height + 6, r2.height)

    def test_inflate__smaller(self):
        """The inflate method inflates around the center of the rectangle
        """
        r = Rect(2, 4, 6, 8)
        r2 = r.inflate(-4, -6)

        self.assertEqual(r.center, r2.center)
        self.assertEqual(r.left + 2, r2.left)
        self.assertEqual(r.top + 3, r2.top)
        self.assertEqual(r.right - 2, r2.right)
        self.assertEqual(r.bottom - 3, r2.bottom)
        self.assertEqual(r.width - 4, r2.width)
        self.assertEqual(r.height - 6, r2.height)

    def test_inflate_ip__larger(self):
        """The inflate_ip method inflates around the center of the rectangle
        """
        r = Rect(2, 4, 6, 8)
        r2 = Rect(r)
        r2.inflate_ip(-4, -6)

        self.assertEqual(r.center, r2.center)
        self.assertEqual(r.left + 2, r2.left)
        self.assertEqual(r.top + 3, r2.top)
        self.assertEqual(r.right - 2, r2.right)
        self.assertEqual(r.bottom - 3, r2.bottom)
        self.assertEqual(r.width - 4, r2.width)
        self.assertEqual(r.height - 6, r2.height)

    def test_inflate_ip__smaller(self):
        """The inflate method inflates around the center of the rectangle
        """
        r = Rect(2, 4, 6, 8)
        r2 = Rect(r)
        r2.inflate_ip(-4, -6)

        self.assertEqual(r.center, r2.center)
        self.assertEqual(r.left + 2, r2.left)
        self.assertEqual(r.top + 3, r2.top)
        self.assertEqual(r.right - 2, r2.right)
        self.assertEqual(r.bottom - 3, r2.bottom)
        self.assertEqual(r.width - 4, r2.width)
        self.assertEqual(r.height - 6, r2.height)

    def test_clamp(self):
        r = Rect(10, 10, 10, 10)
        c = Rect(19, 12, 5, 5).clamp(r)
        self.assertEqual(c.right, r.right)
        self.assertEqual(c.top, 12)
        c = Rect(1, 2, 3, 4).clamp(r)
        self.assertEqual(c.topleft, r.topleft)
        c = Rect(5, 500, 22, 33).clamp(r)
        self.assertEqual(c.center, r.center)

    def test_clamp_ip(self):
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

    def test_clip(self):
        r1 = Rect(1, 2, 3, 4)
        self.assertEqual(Rect(1, 2, 2, 2), r1.clip( Rect(0, 0, 3, 4)))
        self.assertEqual(Rect(2, 2, 2, 4), r1.clip( Rect(2, 2, 10, 20)))
        self.assertEqual(Rect(2, 3, 1, 2), r1.clip(Rect(2, 3, 1, 2)))
        self.assertEqual((0, 0), r1.clip(20, 30, 5, 6).size)
        self.assertEqual(r1, r1.clip(Rect(r1)),
                         "r1 does not clip an identical rect to itself")

    def test_move(self):
        r = Rect(1, 2, 3, 4)
        move_x = 10
        move_y = 20
        r2 = r.move(move_x, move_y)
        expected_r2 = Rect(r.left + move_x, r.top + move_y, r.width, r.height)
        self.assertEqual(expected_r2, r2)

    def test_move_ip(self):
        r = Rect(1, 2, 3, 4)
        r2 = Rect(r)
        move_x = 10
        move_y = 20
        r2.move_ip(move_x, move_y)
        expected_r2 = Rect(r.left + move_x, r.top + move_y, r.width, r.height)
        self.assertEqual(expected_r2, r2)

    def test_union(self):
        r1 = Rect(1, 1, 1, 2)
        r2 = Rect(-2, -2, 1, 2)
        self.assertEqual(Rect(-2, -2, 4, 5), r1.union(r2))

    def test_union__with_identical_Rect(self):
        r1 = Rect(1, 2, 3, 4)
        self.assertEqual(r1, r1.union(Rect(r1)))

    def test_union_ip(self):
        r1 = Rect(1, 1, 1, 2)
        r2 = Rect(-2, -2, 1, 2)
        r1.union_ip(r2)
        self.assertEqual(Rect(-2, -2, 4, 5), r1)

    def test_unionall(self):
        r1 = Rect(0, 0, 1, 1)
        r2 = Rect(-2, -2, 1, 1)
        r3 = Rect(2, 2, 1, 1)

        r4 = r1.unionall([r2, r3])
        self.assertEqual(Rect(-2, -2, 5, 5), r4)

    def test_unionall_ip(self):
        r1 = Rect(0, 0, 1, 1)
        r2 = Rect(-2, -2, 1, 1)
        r3 = Rect(2, 2, 1, 1)

        r1.unionall_ip([r2, r3])
        self.assertEqual(Rect(-2, -2, 5, 5), r1)

        # Bug for an empty list. Would return a Rect instead of None.
        self.assertTrue(r1.unionall_ip([]) is None)

    def test_colliderect(self):
        r1 = Rect(1, 2, 3, 4)
        self.assertTrue(r1.colliderect(Rect(0, 0, 2, 3)),
                        "r1 does not collide with Rect(0, 0, 2, 3)")
        self.assertFalse(r1.colliderect(Rect(0, 0, 1, 2)),
                         "r1 collides with Rect(0, 0, 1, 2)")
        self.assertFalse(r1.colliderect(Rect(r1.right, r1.bottom, 2, 2)),
                         "r1 collides with Rect(r1.right, r1.bottom, 2, 2)")
        self.assertTrue(r1.colliderect(Rect(r1.left + 1, r1.top + 1,
                                            r1.width - 2, r1.height - 2)),
                        "r1 does not collide with Rect(r1.left + 1, r1.top + 1, "+
                        "r1.width - 2, r1.height - 2)")
        self.assertTrue(r1.colliderect(Rect(r1.left - 1, r1.top - 1,
                                            r1.width + 2, r1.height + 2)),
                        "r1 does not collide with Rect(r1.left - 1, r1.top - 1, "+
                        "r1.width + 2, r1.height + 2)")
        self.assertTrue(r1.colliderect(Rect(r1)),
                        "r1 does not collide with an identical rect")
        self.assertFalse(r1.colliderect(Rect(r1.right, r1.bottom, 0, 0)),
                         "r1 collides with Rect(r1.right, r1.bottom, 0, 0)")
        self.assertFalse(r1.colliderect(Rect(r1.right, r1.bottom, 1, 1)),
                         "r1 collides with Rect(r1.right, r1.bottom, 1, 1)")

    def testEquals(self):
        """ check to see how the rect uses __eq__
        """
        r1 = Rect(1, 2, 3, 4)
        r2 = Rect(10, 20, 30, 40)
        r3 = (10, 20, 30, 40)
        r4 = Rect(10, 20, 30, 40)

        class foo (Rect):
            def __eq__(self, other):
                return id(self) == id(other)
            def __ne__(self, other):
                return id(self) != id(other)

        class foo2 (Rect):
            pass

        r5 = foo(10, 20, 30, 40)
        r6 = foo2(10, 20, 30, 40)

        self.assertNotEqual(r5, r2)

        # because we define equality differently for this subclass.
        self.assertEqual(r6, r2)


        rect_list = [r1, r2, r3, r4, r6]

        # see if we can remove 4 of these.
        rect_list.remove(r2)
        rect_list.remove(r2)
        rect_list.remove(r2)
        rect_list.remove(r2)
        self.assertRaises(ValueError, rect_list.remove, r2)

    def test_collidedict(self):

        # __doc__ (as of 2008-08-02) for pygame.rect.Rect.collidedict:

          # Rect.collidedict(dict): return (key, value)
          # test if one rectangle in a dictionary intersects
          #
          # Returns the key and value of the first dictionary value that
          # collides with the Rect. If no collisions are found, None is
          # returned.
          #
          # Rect objects are not hashable and cannot be used as keys in a
          # dictionary, only as values.

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


    def test_collidedictall(self):

        # __doc__ (as of 2008-08-02) for pygame.rect.Rect.collidedictall:

          # Rect.collidedictall(dict): return [(key, value), ...]
          # test if all rectangles in a dictionary intersect
          #
          # Returns a list of all the key and value pairs that intersect with
          # the Rect. If no collisions are found an empty dictionary is
          # returned.
          #
          # Rect objects are not hashable and cannot be used as keys in a
          # dictionary, only as values.

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

    def test_collidelist(self):

        # __doc__ (as of 2008-08-02) for pygame.rect.Rect.collidelist:

          # Rect.collidelist(list): return index
          # test if one rectangle in a list intersects
          #
          # Test whether the rectangle collides with any in a sequence of
          # rectangles. The index of the first collision found is returned. If
          # no collisions are found an index of -1 is returned.

        r = Rect(1, 1, 10, 10)
        l = [Rect(50, 50, 1, 1), Rect(5, 5, 10, 10), Rect(15, 15, 1, 1)]

        self.assertEqual(r.collidelist(l), 1)

        f = [Rect(50, 50, 1, 1), (100, 100, 4, 4)]
        self.assertEqual(r.collidelist(f), -1)


    def test_collidelistall(self):

        # __doc__ (as of 2008-08-02) for pygame.rect.Rect.collidelistall:

          # Rect.collidelistall(list): return indices
          # test if all rectangles in a list intersect
          #
          # Returns a list of all the indices that contain rectangles that
          # collide with the Rect. If no intersecting rectangles are found, an
          # empty list is returned.

        r = Rect(1, 1, 10, 10)

        l = [
            Rect(1, 1, 10, 10),
            Rect(5, 5, 10, 10),
            Rect(15, 15, 1, 1),
            Rect(2, 2, 1, 1),
        ]
        self.assertEqual(r.collidelistall(l), [0, 1, 3])

        f = [Rect(50, 50, 1, 1), Rect(20, 20, 5, 5)]
        self.assertFalse(r.collidelistall(f))


    def test_fit(self):

        # __doc__ (as of 2008-08-02) for pygame.rect.Rect.fit:

          # Rect.fit(Rect): return Rect
          # resize and move a rectangle with aspect ratio
          #
          # Returns a new rectangle that is moved and resized to fit another.
          # The aspect ratio of the original Rect is preserved, so the new
          # rectangle may be smaller than the target in either width or height.

        r = Rect(10, 10, 30, 30)

        r2 = Rect(30, 30, 15, 10)

        f = r.fit(r2)
        self.assertTrue(r2.contains(f))

        f2 = r2.fit(r)
        self.assertTrue(r.contains(f2))



    def test_copy(self):
        r = Rect(1, 2, 10, 20)
        c = r.copy()
        self.assertEqual(c, r)


    def test_subscript(self):
        r = Rect(1, 2, 3, 4)
        self.assertEqual(r[0], 1)
        self.assertEqual(r[1], 2)
        self.assertEqual(r[2], 3)
        self.assertEqual(r[3], 4)
        self.assertEqual(r[-1], 4)
        self.assertEqual(r[-2], 3)
        self.assertEqual(r[-4], 1)
        self.assertRaises(IndexError, r.__getitem__, 5)
        self.assertRaises(IndexError, r.__getitem__, -5)
        self.assertEqual(r[0:2], [1, 2])
        self.assertEqual(r[0:4], [1, 2, 3, 4])
        self.assertEqual(r[0:-1], [1, 2, 3])
        self.assertEqual(r[:], [1, 2, 3, 4])
        self.assertEqual(r[...], [1, 2, 3, 4])
        self.assertEqual(r[0:4:2], [1, 3])
        self.assertEqual(r[0:4:3], [1, 4])
        self.assertEqual(r[3::-1], [4, 3, 2, 1])
        self.assertRaises(TypeError, r.__getitem__, None)

    def test_ass_subscript(self):
        r = Rect(0, 0, 0, 0)
        r[...] = 1, 2, 3, 4
        self.assertEqual(r, [1, 2, 3, 4])
        self.assertRaises(TypeError, r.__setitem__, None, 0)
        self.assertEqual(r, [1, 2, 3, 4])
        self.assertRaises(TypeError, r.__setitem__, 0, '')
        self.assertEqual(r, [1, 2, 3, 4])
        self.assertRaises(IndexError, r.__setitem__, 4, 0)
        self.assertEqual(r, [1, 2, 3, 4])
        self.assertRaises(IndexError, r.__setitem__, -5, 0)
        self.assertEqual(r, [1, 2, 3, 4])
        r[0] = 10
        self.assertEqual(r, [10, 2, 3, 4])
        r[3] = 40
        self.assertEqual(r, [10, 2, 3, 40])
        r[-1] = 400
        self.assertEqual(r, [10, 2, 3, 400])
        r[-4] = 100
        self.assertEqual(r, [100, 2, 3, 400])
        r[1:3] = 0
        self.assertEqual(r, [100, 0, 0, 400])
        r[...] = 0
        self.assertEqual(r, [0, 0, 0, 0])
        r[:] = 9
        self.assertEqual(r, [9, 9, 9, 9])
        r[:] = 11, 12, 13, 14
        self.assertEqual(r, [11, 12, 13, 14])
        r[::-1] = r
        self.assertEqual(r, [14, 13, 12, 11])


class SubclassTest(unittest.TestCase):
    class MyRect(Rect):
        def __init__(self, *args, **kwds):
            super(SubclassTest.MyRect, self).__init__(*args, **kwds)
            self.an_attribute = True

    def test_copy(self):
        mr1 = self.MyRect(1, 2, 10, 20)
        self.assertTrue(mr1.an_attribute)
        mr2 = mr1.copy()
        self.assertTrue(isinstance(mr2, self.MyRect))
        self.assertRaises(AttributeError, getattr, mr2, "an_attribute")

    def test_move(self):
        mr1 = self.MyRect(1, 2, 10, 20)
        self.assertTrue(mr1.an_attribute)
        mr2 = mr1.move(1, 2)
        self.assertTrue(isinstance(mr2, self.MyRect))
        self.assertRaises(AttributeError, getattr, mr2, "an_attribute")

    def test_inflate(self):
        mr1 = self.MyRect(1, 2, 10, 20)
        self.assertTrue(mr1.an_attribute)
        mr2 = mr1.inflate(2, 4)
        self.assertTrue(isinstance(mr2, self.MyRect))
        self.assertRaises(AttributeError, getattr, mr2, "an_attribute")

    def test_clamp(self):
        mr1 = self.MyRect(19, 12, 5, 5)
        self.assertTrue(mr1.an_attribute)
        mr2 = mr1.clamp(Rect(10, 10, 10, 10))
        self.assertTrue(isinstance(mr2, self.MyRect))
        self.assertRaises(AttributeError, getattr, mr2, "an_attribute")

    def test_clip(self):
        mr1 = self.MyRect(1, 2, 3, 4)
        self.assertTrue(mr1.an_attribute)
        mr2 = mr1.clip(Rect(0, 0, 3, 4))
        self.assertTrue(isinstance(mr2, self.MyRect))
        self.assertRaises(AttributeError, getattr, mr2, "an_attribute")

    def test_union(self):
        mr1 = self.MyRect(1, 1, 1, 2)
        self.assertTrue(mr1.an_attribute)
        mr2 = mr1.union(Rect(-2, -2, 1, 2))
        self.assertTrue(isinstance(mr2, self.MyRect))
        self.assertRaises(AttributeError, getattr, mr2, "an_attribute")

    def test_unionall(self):
        mr1 = self.MyRect(0, 0, 1, 1)
        self.assertTrue(mr1.an_attribute)
        mr2 = mr1.unionall([Rect(-2, -2, 1, 1), Rect(2, 2, 1, 1)])
        self.assertTrue(isinstance(mr2, self.MyRect))
        self.assertRaises(AttributeError, getattr, mr2, "an_attribute")

    def test_fit(self):
        mr1 = self.MyRect(10, 10, 30, 30)
        self.assertTrue(mr1.an_attribute)
        mr2 = mr1.fit(Rect(30, 30, 15, 10))
        self.assertTrue(isinstance(mr2, self.MyRect))
        self.assertRaises(AttributeError, getattr, mr2, "an_attribute")

if __name__ == '__main__':
    unittest.main()
