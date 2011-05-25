.. include:: common.txt

:mod:`pygame.Rect`
==================

.. currentmodule:: pygame

.. class:: Rect

   | :sl:`pygame object for storing rectangular coordinates`
   | :sg:`Rect(left, top, width, height) -> Rect`
   | :sg:`Rect((left, top), (width, height)) -> Rect`
   | :sg:`Rect(object) -> Rect`

   Pygame uses Rect objects to store and manipulate rectangular areas. A Rect
   can be created from a combination of left, top, width, and height values.
   Rects can also be created from python objects that are already a Rect or
   have an attribute named "rect".

   Any Pygame function that requires a Rect argument also accepts any of these
   values to construct a Rect. This makes it easier to create Rects on the fly
   as arguments to functions.

   The Rect functions that change the position or size of a Rect return a new
   copy of the Rect with the affected changes. The original Rect is not
   modified. Some methods have an alternate "in-place" version that returns
   None but effects the original Rect. These "in-place" methods are denoted
   with the "ip" suffix.

   The Rect object has several virtual attributes which can be used to move and
   align the Rect:

   ::

       top, left, bottom, right
       topleft, bottomleft, topright, bottomright
       midtop, midleft, midbottom, midright
       center, centerx, centery
       size, width, height
       w,h

   All of these attributes can be assigned to:

   ::

       rect1.right = 10
       rect2.center = (20,30)

   Assigning to size, width or height changes the dimensions of the rectangle;
   all other assignments move the rectangle without resizing it. Notice that
   some attributes are integers and others are pairs of integers.

   If a Rect has a nonzero width or height, it will return True for a nonzero
   test. Some methods return a Rect with 0 size to represent an invalid
   rectangle.

   The coordinates for Rect objects are all integers. The size values can be
   programmed to have negative values, but these are considered illegal Rects
   for most operations.

   There are several collision tests between other rectangles. Most python
   containers can be searched for collisions against a single Rect.

   The area covered by a Rect does not include the right- and bottom-most edge
   of pixels. If one Rect's bottom border is another Rect's top border (i.e.,
   rect1.bottom=rect2.top), the two meet exactly on the screen but do not
   overlap, and ``rect1.colliderect(rect2)`` returns false.

   Though Rect can be subclassed, methods that return new rectangles are not
   subclass aware. That is, move or copy return a new :mod:`pygame.Rect`
   instance, not an instance of the subclass. This may change. To make subclass
   awareness work though, subclasses may have to maintain the same constructor
   signature as Rect.

   .. method:: copy

      | :sl:`copy the rectangle`
      | :sg:`copy() -> Rect`

      Returns a new rectangle having the same position and size as the orginal.

      New in pygame 1.9

      .. ## Rect.copy ##

   .. method:: move

      | :sl:`moves the rectangle`
      | :sg:`move(x, y) -> Rect`

      Returns a new rectangle that is moved by the given offset. The x and y
      arguments can be any integer value, positive or negative.

      .. ## Rect.move ##

   .. method:: move_ip

      | :sl:`moves the rectangle, in place`
      | :sg:`move_ip(x, y) -> None`

      Same as the ``Rect.move()`` method, but operates in place.

      .. ## Rect.move_ip ##

   .. method:: inflate

      | :sl:`grow or shrink the rectangle size`
      | :sg:`inflate(x, y) -> Rect`

      Returns a new rectangle with the size changed by the given offset. The
      rectangle remains centered around its current center. Negative values
      will shrink the rectangle.

      .. ## Rect.inflate ##

   .. method:: inflate_ip

      | :sl:`grow or shrink the rectangle size, in place`
      | :sg:`inflate_ip(x, y) -> None`

      Same as the ``Rect.inflate()`` method, but operates in place.

      .. ## Rect.inflate_ip ##

   .. method:: clamp

      | :sl:`moves the rectangle inside another`
      | :sg:`clamp(Rect) -> Rect`

      Returns a new rectangle that is moved to be completely inside the
      argument Rect. If the rectangle is too large to fit inside, it is
      centered inside the argument Rect, but its size is not changed.

      .. ## Rect.clamp ##

   .. method:: clamp_ip

      | :sl:`moves the rectangle inside another, in place`
      | :sg:`clamp_ip(Rect) -> None`

      Same as the ``Rect.clamp()`` method, but operates in place.

      .. ## Rect.clamp_ip ##

   .. method:: clip

      | :sl:`crops a rectangle inside another`
      | :sg:`clip(Rect) -> Rect`

      Returns a new rectangle that is cropped to be completely inside the
      argument Rect. If the two rectangles do not overlap to begin with, a Rect
      with 0 size is returned.

      .. ## Rect.clip ##

   .. method:: union

      | :sl:`joins two rectangles into one`
      | :sg:`union(Rect) -> Rect`

      Returns a new rectangle that completely covers the area of the two
      provided rectangles. There may be area inside the new Rect that is not
      covered by the originals.

      .. ## Rect.union ##

   .. method:: union_ip

      | :sl:`joins two rectangles into one, in place`
      | :sg:`union_ip(Rect) -> None`

      Same as the ``Rect.union()`` method, but operates in place.

      .. ## Rect.union_ip ##

   .. method:: unionall

      | :sl:`the union of many rectangles`
      | :sg:`unionall(Rect_sequence) -> Rect`

      Returns the union of one rectangle with a sequence of many rectangles.

      .. ## Rect.unionall ##

   .. method:: unionall_ip

      | :sl:`the union of many rectangles, in place`
      | :sg:`unionall_ip(Rect_sequence) -> None`

      The same as the ``Rect.unionall()`` method, but operates in place.

      .. ## Rect.unionall_ip ##

   .. method:: fit

      | :sl:`resize and move a rectangle with aspect ratio`
      | :sg:`fit(Rect) -> Rect`

      Returns a new rectangle that is moved and resized to fit another. The
      aspect ratio of the original Rect is preserved, so the new rectangle may
      be smaller than the target in either width or height.

      .. ## Rect.fit ##

   .. method:: normalize

      | :sl:`correct negative sizes`
      | :sg:`normalize() -> None`

      This will flip the width or height of a rectangle if it has a negative
      size. The rectangle will remain in the same place, with only the sides
      swapped.

      .. ## Rect.normalize ##

   .. method:: contains

      | :sl:`test if one rectangle is inside another`
      | :sg:`contains(Rect) -> bool`

      Returns true when the argument is completely inside the Rect.

      .. ## Rect.contains ##

   .. method:: collidepoint

      | :sl:`test if a point is inside a rectangle`
      | :sg:`collidepoint(x, y) -> bool`
      | :sg:`collidepoint((x,y)) -> bool`

      Returns true if the given point is inside the rectangle. A point along
      the right or bottom edge is not considered to be inside the rectangle.

      .. ## Rect.collidepoint ##

   .. method:: colliderect

      | :sl:`test if two rectangles overlap`
      | :sg:`colliderect(Rect) -> bool`

      Returns true if any portion of either rectangle overlap (except the
      top+bottom or left+right edges).

      .. ## Rect.colliderect ##

   .. method:: collidelist

      | :sl:`test if one rectangle in a list intersects`
      | :sg:`collidelist(list) -> index`

      Test whether the rectangle collides with any in a sequence of rectangles.
      The index of the first collision found is returned. If no collisions are
      found an index of -1 is returned.

      .. ## Rect.collidelist ##

   .. method:: collidelistall

      | :sl:`test if all rectangles in a list intersect`
      | :sg:`collidelistall(list) -> indices`

      Returns a list of all the indices that contain rectangles that collide
      with the Rect. If no intersecting rectangles are found, an empty list is
      returned.

      .. ## Rect.collidelistall ##

   .. method:: collidedict

      | :sl:`test if one rectangle in a dictionary intersects`
      | :sg:`collidedict(dict) -> (key, value)`

      Returns the key and value of the first dictionary value that collides
      with the Rect. If no collisions are found, None is returned.

      Rect objects are not hashable and cannot be used as keys in a dictionary,
      only as values.

      .. ## Rect.collidedict ##

   .. method:: collidedictall

      | :sl:`test if all rectangles in a dictionary intersect`
      | :sg:`collidedictall(dict) -> [(key, value), ...]`

      Returns a list of all the key and value pairs that intersect with the
      Rect. If no collisions are found an empty dictionary is returned.

      Rect objects are not hashable and cannot be used as keys in a dictionary,
      only as values.

      .. ## Rect.collidedictall ##

   .. ## pygame.Rect ##
