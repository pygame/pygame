.. include:: common.txt

:mod:`pygame.gfxdraw`
=====================

.. module:: pygame.gfxdraw
   :synopsis: pygame module for drawing shapes

| :sl:`pygame module for drawing shapes`

Wraps SDL_gfx primatives.

EXPERIMENTAL!: meaning this api may change, or dissapear in later pygame
releases. If you use this, your code will break with the next pygame release.

Most of the functions accept a color argument that is an ``RGB`` triplet. These
can also accept an ``RGBA`` quadruplet. The color argument can also be an
integer pixel value that is already mapped to the Surface's pixel format.

For all functions the arguments are strictly positional. Only integers are
accepted for coordinates and radii.

For functions like rectangle that accept a rect argument any (x, y, w, h)
sequence is accepted, though :mod:`pygame.Rect` instances are prefered. Note
that for a :mod:`pygame.Rect` the drawing will not include
``Rect.bottomright``. The right and bottom attributes of a Rect lie one pixel
outside of the Rect's boarder.

New in pygame 1.9.0.

.. function:: pixel

   | :sl:`place a pixel`
   | :sg:`pixel(surface, x, y, color) -> None`

   .. ## pygame.gfxdraw.pixel ##

.. function:: hline

   | :sl:`draw a horizontal line`
   | :sg:`hline(surface, x1, x2, y, color) -> None`

   .. ## pygame.gfxdraw.hline ##

.. function:: vline

   | :sl:`draw a vertical line`
   | :sg:`vline(surface, x, y1, y2, color) -> None`

   .. ## pygame.gfxdraw.vline ##

.. function:: rectangle

   | :sl:`draw a rectangle`
   | :sg:`rectangle(surface, rect, color) -> None`

   .. ## pygame.gfxdraw.rectangle ##

.. function:: box

   | :sl:`draw a box`
   | :sg:`box(surface, rect, color) -> None`

   .. ## pygame.gfxdraw.box ##

.. function:: line

   | :sl:`draw a line`
   | :sg:`line(surface, x1, y1, x2, y2, color) -> None`

   .. ## pygame.gfxdraw.line ##

.. function:: circle

   | :sl:`draw a circle`
   | :sg:`circle(surface, x, y, r, color) -> None`

   .. ## pygame.gfxdraw.circle ##

.. function:: arc

   | :sl:`draw an arc`
   | :sg:`arc(surface, x, y, r, start, end, color) -> None`

   .. ## pygame.gfxdraw.arc ##

.. function:: aacircle

   | :sl:`draw an anti-aliased circle`
   | :sg:`aacircle(surface, x, y, r, color) -> None`

   .. ## pygame.gfxdraw.aacircle ##

.. function:: filled_circle

   | :sl:`draw a filled circle`
   | :sg:`filled_circle(surface, x, y, r, color) -> None`

   .. ## pygame.gfxdraw.filled_circle ##

.. function:: ellipse

   | :sl:`draw an ellipse`
   | :sg:`ellipse(surface, x, y, rx, ry, color) -> None`

   .. ## pygame.gfxdraw.ellipse ##

.. function:: aaellipse

   | :sl:`draw an anti-aliased ellipse`
   | :sg:`aaellipse(surface, x, y, rx, ry, color) -> None`

   .. ## pygame.gfxdraw.aaellipse ##

.. function:: filled_ellipse

   | :sl:`draw a filled ellipse`
   | :sg:`filled_ellipse(surface, x, y, rx, ry, color) -> None`

   .. ## pygame.gfxdraw.filled_ellipse ##

.. function:: pie

   | :sl:`draw a pie`
   | :sg:`pie(surface, x, y, r, start, end, color) -> None`

   .. ## pygame.gfxdraw.pie ##

.. function:: trigon

   | :sl:`draw a triangle`
   | :sg:`trigon(surface, x1, y1, x2, y2, x3, y3, color) -> None`

   .. ## pygame.gfxdraw.trigon ##

.. function:: aatrigon

   | :sl:`draw an anti-aliased triangle`
   | :sg:`aatrigon(surface, x1, y1, x2, y2, x3, y3, color) -> None`

   .. ## pygame.gfxdraw.aatrigon ##

.. function:: filled_trigon

   | :sl:`draw a filled trigon`
   | :sg:`filled_trigon(surface, x1, y1, x3, y2, x3, y3, color) -> None`

   .. ## pygame.gfxdraw.filled_trigon ##

.. function:: polygon

   | :sl:`draw a polygon`
   | :sg:`polygon(surface, points, color) -> None`

   .. ## pygame.gfxdraw.polygon ##

.. function:: aapolygon

   | :sl:`draw an anti-aliased polygon`
   | :sg:`aapolygon(surface, points, color) -> None`

   .. ## pygame.gfxdraw.aapolygon ##

.. function:: filled_polygon

   | :sl:`draw a filled polygon`
   | :sg:`filled_polygon(surface, points, color) -> None`

   .. ## pygame.gfxdraw.filled_polygon ##

.. function:: textured_polygon

   | :sl:`draw a textured polygon`
   | :sg:`textured_polygon(surface, points, texture, tx, ty) -> None`

   A per-pixel alpha texture blit to a per-pixel alpha surface will differ from
   a ``Surface.blit()`` blit. Also, a per-pixel alpha texture cannot be used
   with an 8-bit per pixel destination.

   .. ## pygame.gfxdraw.textured_polygon ##

.. function:: bezier

   | :sl:`draw a bezier curve`
   | :sg:`bezier(surface, points, steps, color) -> None`

   .. ## pygame.gfxdraw.bezier ##

.. ## pygame.gfxdraw ##
