.. include:: common.txt

:mod:`pygame.gfxdraw`
=====================

.. module:: pygame.gfxdraw
   :synopsis: pygame module for drawing shapes

| :sl:`pygame module for drawing shapes`

EXPERIMENTAL!: meaning this API may change, or disappear in later pygame
releases. If you use this, your code will break with the next pygame release.

Draw several shapes to a surface.

Most of the functions accept a color argument that is an ``RGB`` triplet. These
can also accept an ``RGBA`` quadruplet. The color argument can also be an
integer pixel value that is already mapped to the Surface's pixel format.

For all functions the arguments are strictly positional. Only integers are
accepted for coordinates and radii.

For functions like rectangle that accept a rect argument any (x, y, w, h)
sequence is accepted, though :mod:`pygame.Rect` instances are preferred. Note
that for a :mod:`pygame.Rect` the drawing will not include
``Rect.bottomright``. The right and bottom attributes of a Rect lie one pixel
outside of the Rect's boarder.

To draw an anti aliased and filled shape, first use the aa* version of 
the function, and then use the filled version.  For example ::

   col = (255, 0, 0)
   surf.fill((255, 255, 255))
   pygame.gfxdraw.aacircle(surf, x, y, 30, col)
   pygame.gfxdraw.filled_circle(surf, x, y, 30, col)

Note that pygame does not automatically import pygame.gfxdraw, so you need to
import pygame.gfxdraw before using it.

Threading note: each of the functions releases the GIL during the C part of the call.

The pygame.gfxdraw module differs from the draw module in the API it uses, and
also the different functions available to draw.  It also wraps the primitives 
from the library called SDL_gfx, rather than using modified versions.


New in pygame 1.9.0.

.. function:: pixel

   | :sl:`place a pixel`
   | :sg:`pixel(surface, x, y, color) -> None`

   Draws a single pixel onto a surface.

   .. ## pygame.gfxdraw.pixel ##

.. function:: hline

   | :sl:`draw a horizontal line`
   | :sg:`hline(surface, x1, x2, y, color) -> None`

   Draws a straight horizontal line on a Surface from x1 to x2 for
   the given y coordinate.

   .. ## pygame.gfxdraw.hline ##

.. function:: vline

   | :sl:`draw a vertical line`
   | :sg:`vline(surface, x, y1, y2, color) -> None`

   Draws a straight vertical line on a Surface from y1 to y2 on
   the given x coordinate.

   .. ## pygame.gfxdraw.vline ##

.. function:: rectangle

   | :sl:`draw a rectangle`
   | :sg:`rectangle(surface, rect, color) -> None`

   Draws the rectangle edges onto the surface. The given Rect is the area of the
   rectangle.

   Keep in mind the ``Surface.fill()`` method works just as well for drawing
   filled rectangles. In fact the ``Surface.fill()`` can be hardware
   accelerated on some platforms with both software and hardware display modes.

   .. ## pygame.gfxdraw.rectangle ##

.. function:: box

   | :sl:`draw a box`
   | :sg:`box(surface, rect, color) -> None`

   Draws a box (a rect) onto a surface.

   .. ## pygame.gfxdraw.box ##

.. function:: line

   | :sl:`draw a line`
   | :sg:`line(surface, x1, y1, x2, y2, color) -> None`

   Draws a straight line on a Surface. There are no endcaps.

   .. ## pygame.gfxdraw.line ##

.. function:: circle

   | :sl:`draw a circle`
   | :sg:`circle(surface, x, y, r, color) -> None`

   Draws the edges of a circular shape on the Surface. The pos argument is 
   the center of the circle, and radius is the size.  The circle is not 
   filled with color.

   .. ## pygame.gfxdraw.circle ##

.. function:: arc

   | :sl:`draw an arc`
   | :sg:`arc(surface, x, y, r, start, end, color) -> None`

   Draws an arc onto a surface.

   .. ## pygame.gfxdraw.arc ##

.. function:: aacircle

   | :sl:`draw an anti-aliased circle`
   | :sg:`aacircle(surface, x, y, r, color) -> None`

   Draws the edges of an anti aliased circle onto a surface.

   .. ## pygame.gfxdraw.aacircle ##

.. function:: filled_circle

   | :sl:`draw a filled circle`
   | :sg:`filled_circle(surface, x, y, r, color) -> None`

   Draws a filled circle onto a surface.  So the inside of the circle will 
   be filled with the given color.

   .. ## pygame.gfxdraw.filled_circle ##

.. function:: ellipse

   | :sl:`draw an ellipse`
   | :sg:`ellipse(surface, x, y, rx, ry, color) -> None`

   Draws the edges of an ellipse onto a surface.

   .. ## pygame.gfxdraw.ellipse ##

.. function:: aaellipse

   | :sl:`draw an anti-aliased ellipse`
   | :sg:`aaellipse(surface, x, y, rx, ry, color) -> None`

   Draws anti aliased edges of an ellipse onto a surface.

   .. ## pygame.gfxdraw.aaellipse ##

.. function:: filled_ellipse

   | :sl:`draw a filled ellipse`
   | :sg:`filled_ellipse(surface, x, y, rx, ry, color) -> None`

   Draws a filled ellipse onto a surface.  So the inside of the ellipse will 
   be filled with the given color.

   .. ## pygame.gfxdraw.filled_ellipse ##

.. function:: pie

   | :sl:`draw a pie`
   | :sg:`pie(surface, x, y, r, start, end, color) -> None`

   Draws a pie onto the surface.

   .. ## pygame.gfxdraw.pie ##

.. function:: trigon

   | :sl:`draw a triangle`
   | :sg:`trigon(surface, x1, y1, x2, y2, x3, y3, color) -> None`

   Draws the edges of a trigon onto a surface.  A trigon is a triangle.

   .. ## pygame.gfxdraw.trigon ##

.. function:: aatrigon

   | :sl:`draw an anti-aliased triangle`
   | :sg:`aatrigon(surface, x1, y1, x2, y2, x3, y3, color) -> None`

   Draws the anti aliased edges of a trigon onto a surface.  A trigon is a triangle.

   .. ## pygame.gfxdraw.aatrigon ##

.. function:: filled_trigon

   | :sl:`draw a filled trigon`
   | :sg:`filled_trigon(surface, x1, y1, x2, y2, x3, y3, color) -> None`

   Draws a filled trigon onto a surface.  So the inside of the trigon will 
   be filled with the given color.

   .. ## pygame.gfxdraw.filled_trigon ##

.. function:: polygon

   | :sl:`draw a polygon`
   | :sg:`polygon(surface, points, color) -> None`

   Draws the edges of a polygon onto a surface.

   .. ## pygame.gfxdraw.polygon ##

.. function:: aapolygon

   | :sl:`draw an anti-aliased polygon`
   | :sg:`aapolygon(surface, points, color) -> None`

   Draws the anti aliased edges of a polygon onto a surface.

   .. ## pygame.gfxdraw.aapolygon ##

.. function:: filled_polygon

   | :sl:`draw a filled polygon`
   | :sg:`filled_polygon(surface, points, color) -> None`

   Draws a filled polygon onto a surface.  So the inside of the polygon will 
   be filled with the given color.

   .. ## pygame.gfxdraw.filled_polygon ##

.. function:: textured_polygon

   | :sl:`draw a textured polygon`
   | :sg:`textured_polygon(surface, points, texture, tx, ty) -> None`

   Draws a textured polygon onto a surface.

   A per-pixel alpha texture blit to a per-pixel alpha surface will differ from
   a ``Surface.blit()`` blit. Also, a per-pixel alpha texture cannot be used
   with an 8-bit per pixel destination.

   .. ## pygame.gfxdraw.textured_polygon ##

.. function:: bezier

   | :sl:`draw a Bézier curve`
   | :sg:`bezier(surface, points, steps, color) -> None`

   Draws a Bézier curve onto a surface.

   .. ## pygame.gfxdraw.bezier ##

.. ## pygame.gfxdraw ##
