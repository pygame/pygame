import unittest
from pygame2.base import Rect

class RectTest (unittest.TestCase):
    def todo_test_pygame2_base_Rect_bottom(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.bottom:

        # Gets or sets the bottom edge position of the Rect.

        self.fail() 

    def todo_test_pygame2_base_Rect_bottomleft(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.bottomleft:

        # Gets or sets the bottom left corner position of the Rect.

        self.fail() 

    def todo_test_pygame2_base_Rect_bottomright(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.bottomright:

        # Gets or sets the bottom right corner position of the Rect.

        self.fail() 

    def todo_test_pygame2_base_Rect_center(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.center:

        # Gets or sets the center position of the Rect.

        self.fail() 

    def todo_test_pygame2_base_Rect_centerx(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.centerx:

        # Gets or sets the horizontal center position of the Rect.

        self.fail() 

    def todo_test_pygame2_base_Rect_centery(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.centery:

        # Gets or sets the vertical center position of the Rect.

        self.fail() 

    def todo_test_pygame2_base_Rect_clamp(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.clamp:

        # Rect.clamp (Rect) -> Rect
        # 
        # Moves the rectangle inside another.
        # 
        # Rect.clamp (Rect) -> Rect  Moves the rectangle inside another.
        # Returns a new rectangle that is moved to be completely inside the
        # argument Rect. If the rectangle is too large to fit inside, it is
        # centered inside the argument Rect, but its size is not changed.

        self.fail() 

    def todo_test_pygame2_base_Rect_clamp_ip(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.clamp_ip:

        # Rect.clamp_ip (Rect) -> None
        # 
        # Moves the rectangle inside another, in place.
        # 
        # Same as Rect.clamp(Rect), but operates in place.

        self.fail() 

    def todo_test_pygame2_base_Rect_clip(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.clip:

        # Rect.clip (Rect) -> Rect
        # 
        # Crops a rectangle inside another.
        # 
        # Rect.clip (Rect) -> Rect  Crops a rectangle inside another.  Returns
        # a new rectangle that is cropped to be completely inside the argument
        # Rect. If the two rectangles do not overlap to begin with, a Rect
        # with 0 size is returned. Thus it returns the area, in which both
        # rects overlap.

        self.fail() 

    def todo_test_pygame2_base_Rect_collidedict(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.collidedict:

        # Rect.collidedict (dict) -> (key, value)
        # 
        # Test if one rectangle in a dictionary intersects.
        # 
        # Rect.collidedict (dict) -> (key, value)  Test if one rectangle in a
        # dictionary intersects.  Returns the key and value of the first
        # dictionary value that collides with the Rect. If no collisions are
        # found, None is returned. They keys of the passed dict must be Rect
        # objects.

        self.fail() 

    def todo_test_pygame2_base_Rect_collidedictall(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.collidedictall:

        # Rect.collidedictall (dict) -> [(key, value), ...]
        # 
        # Test if all rectangles in a dictionary intersect.
        # 
        # Rect.collidedictall (dict) -> [(key, value), ...]  Test if all
        # rectangles in a dictionary intersect.  Returns a list of all the key
        # and value pairs that intersect with the Rect. If no collisions are
        # found an empty list is returned. They keys of the passed dict must
        # be Rect objects.

        self.fail() 

    def todo_test_pygame2_base_Rect_collidelist(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.collidelist:

        # Rect.collidelist (list) -> index
        # 
        # Test if one rectangle in a list intersects.
        # 
        # Rect.collidelist (list) -> index  Test if one rectangle in a list
        # intersects.  Test whether the rectangle collides with any in a
        # sequence of rectangles. The index of the first collision found is
        # returned. If no collisions are found an index of -1 is returned.

        self.fail() 

    def todo_test_pygame2_base_Rect_collidelistall(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.collidelistall:

        # Rect.collidelistall (list) -> [index, ...]
        # 
        # Test if all rectangles in a list intersect.
        # 
        # Rect.collidelistall (list) -> [index, ...]  Test if all rectangles
        # in a list intersect.  Returns a list of all the indices that contain
        # rectangles that collide with the Rect. If no intersecting rectangles
        # are found, an empty list is returned.

        self.fail() 

    def todo_test_pygame2_base_Rect_collidepoint(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.collidepoint:

        # Rect.collidepoint (x, y) -> bool
        # 
        # Test if a point is inside a rectangle.
        # 
        # Rect.collidepoint (x, y) -> bool  Test if a point is inside a
        # rectangle.  Returns true if the given point is inside the rectangle.
        # A point along the right or bottom edge is not considered to be
        # inside the rectangle.

        self.fail() 

    def todo_test_pygame2_base_Rect_colliderect(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.colliderect:

        # Rect.colliderect (Rect) -> bool
        # 
        # Test if two rectangles overlap.
        # 
        # Rect.colliderect (Rect) -> bool  Test if two rectangles overlap.
        # Returns true if any portion of either rectangle overlap (except the
        # top+bottom or left+right edges).

        self.fail() 

    def todo_test_pygame2_base_Rect_contains(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.contains:

        # Rect.contains (Rect) -> bool
        # 
        # Test if one rectangle is inside another.
        # 
        # Rect.contains (Rect) -> bool  Test if one rectangle is inside
        # another.  Returns true when the argument rectangle is completely
        # inside the Rect.

        self.fail() 

    def todo_test_pygame2_base_Rect_fit(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.fit:

        # Rect.fit (Rect) -> Rect
        # 
        # Resize and move a rectangle with aspect ratio.
        # 
        # Rect.fit (Rect) -> Rect  Resize and move a rectangle with aspect
        # ratio.  Returns a new rectangle that is moved and resized to fit
        # another. The aspect ratio of the original Rect is preserved, so the
        # new rectangle may be smaller than the target in either width or
        # height.

        self.fail() 

    def todo_test_pygame2_base_Rect_height(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.height:

        # Gets or sets the height of the Rect.

        self.fail() 

    def todo_test_pygame2_base_Rect_inflate(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.inflate:

        # Rect.inflate (x, y) -> Rect
        # 
        # Grow or shrink the rectangle size.
        # 
        # Rect.inflate (x, y) -> Rect  Grow or shrink the rectangle size.
        # Returns a new rectangle with the size changed by the given offset.
        # The rectangle remains centered around its current center. Negative
        # values will shrink the rectangle.

        self.fail() 

    def todo_test_pygame2_base_Rect_inflate_ip(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.inflate_ip:

        # Rect.inflate_ip (x, y) -> None
        # 
        # Grow or shrink the rectangle size, in place.
        # 
        # Same as Rect.inflate(x, y), but operates in place.

        self.fail() 

    def todo_test_pygame2_base_Rect_left(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.left:

        # Gets or sets the left edge position of the Rect.

        self.fail() 

    def todo_test_pygame2_base_Rect_midbottom(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.midbottom:

        # Gets or sets the mid bottom edge position of the Rect.

        self.fail() 

    def todo_test_pygame2_base_Rect_midleft(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.midleft:

        # Gets or sets the mid left edge position of the Rect.

        self.fail() 

    def todo_test_pygame2_base_Rect_midright(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.midright:

        # Gets or sets the mid right edge position of the Rect.

        self.fail() 

    def todo_test_pygame2_base_Rect_midtop(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.midtop:

        # Gets or sets the mid top edge position of the Rect.

        self.fail() 

    def todo_test_pygame2_base_Rect_move(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.move:

        # Rect.move (x, y) -> Rect
        # 
        # Moves the rectangle.
        # 
        # Rect.move (x, y) -> Rect  Moves the rectangle.  Returns a new
        # rectangle that is moved by the given offset. The x and y arguments
        # can be any integer value, positive or negative.

        self.fail() 

    def todo_test_pygame2_base_Rect_move_ip(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.move_ip:

        # Rect.move_ip (x, y) -> None
        # 
        # Moves the rectangle, in place.
        # 
        # Same as Rect.move (x, y), but operates in place.

        self.fail() 

    def todo_test_pygame2_base_Rect_right(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.right:

        # Gets or sets the right position of the Rect.

        self.fail() 

    def todo_test_pygame2_base_Rect_size(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.size:

        # Gets or sets the width and height of the Rect as 2-value tuple.

        self.fail() 

    def todo_test_pygame2_base_Rect_top(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.top:

        # Gets or sets the top edge position of the Rect.

        self.fail() 

    def todo_test_pygame2_base_Rect_topleft(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.topleft:

        # Gets or sets the top left corner position of the Rect.

        self.fail() 

    def todo_test_pygame2_base_Rect_topright(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.topright:

        # Gets or sets the top right corner position of the Rect.

        self.fail() 

    def todo_test_pygame2_base_Rect_union(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.union:

        # Rect.union (Rect) -> Rect
        # 
        # Joins two rectangles into one.
        # 
        # Rect.union (Rect) -> Rect  Joins two rectangles into one.  Returns a
        # new rectangle that completely covers the area of the two provided
        # rectangles. There may be area inside the new Rect that is not
        # covered by the originals.

        self.fail() 

    def todo_test_pygame2_base_Rect_union_ip(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.union_ip:

        # Rect.union_ip (Rect) -> Rect
        # 
        # Joins two rectangles into one, in place.
        # 
        # Same as Rect.union(Rect), but operates in place.

        self.fail() 

    def todo_test_pygame2_base_Rect_width(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.width:

        # Gets or sets the width of the Rect.

        self.fail() 

    def todo_test_pygame2_base_Rect_x(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.x:

        # Gets or sets the horizontal top left position of the Rect.

        self.fail() 

    def todo_test_pygame2_base_Rect_y(self):

        # __doc__ (as of 2008-10-17) for pygame2.base.Rect.y:

        # Gets or sets the vertical top left position of the Rect.

        self.fail() 
