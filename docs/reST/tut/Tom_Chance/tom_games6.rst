.. include:: common.txt

***************************
  Putting it all together
***************************

.. _makegames-6:

6. Putting it all together
============================

So far you've learnt all the basics necessary to build a simple game. You should understand how to create Pygame objects, how Pygame
displays objects, how it handles events, and how you can use physics to introduce some motion into your game. Now I'll just show how
you can take all those chunks of code and put them together into a working game. What we need first is to let the ball hit the sides
of the screen, and for the bat to be able to hit the ball, otherwise there's not going to be much game play involved. We do this
using Pygame's :meth:`collision <pygame.Rect.collidepoint>` methods.


.. _makegames-6-1:

6.1. Let the ball hit sides
---------------------------

The basics principle behind making it bounce of the sides is easy to grasp. You grab the coordinates of the four corners of the ball,
and check to see if they correspond with the x or y coordinate of the edge of the screen. So if the top right and top left corners both
have a y coordinate of zero, you know that the ball is currently on the top edge of the screen. We do all this in the ``update`` function,
after we've worked out the new position of the ball.

::

  if not self.area.contains(newpos):
  	tl = not self.area.collidepoint(newpos.topleft)
  	tr = not self.area.collidepoint(newpos.topright)
  	bl = not self.area.collidepoint(newpos.bottomleft)
  	br = not self.area.collidepoint(newpos.bottomright)
  	if tr and tl or (br and bl):
  		angle = -angle
  	if tl and bl:
  		self.offcourt(player=2)
  	if tr and br:
  		self.offcourt(player=1)

  self.vector = (angle,z)

Here we check to see if the ``area``
contains the new position of the ball (it always should, so we needn't have an ``else`` clause,
though in other circumstances you might want to consider it. We then check if the coordinates for the four corners
are *colliding* with the area's edges, and create objects for each result. If they are, the objects will have a value of 1,
or ``True``. If they don't, then the value will be ``None``, or ``False``. We then see if it has hit the top or bottom, and if it
has we change the ball's direction. Handily, using radians we can do this by simply reversing its positive/negative value.
We also check to see if the ball has gone off the sides, and if it has we call the ``offcourt`` function.
This, in my game, resets the ball, adds 1 point to the score of the player specified when calling the function, and displays the new score.

Finally, we recompile the vector based on the new angle. And that is it. The ball will now merrily bounce off the walls and go
offcourt with good grace.


.. _makegames-6-2:

6.2. Let the ball hit bats
--------------------------

Making the ball hit the bats is very similar to making it hit the sides of the screen. We still use the collide method, but this time
we check to see if the rectangles for the ball and either bat collide. In this code I've also put in some extra code to avoid various
glitches. You'll find that you'll have to put all sorts of extra code in to avoid glitches and bugs, so it's good to get used to seeing
it.

::

  else:
      # Deflate the rectangles so you can't catch a ball behind the bat
      player1.rect.inflate(-3, -3)
      player2.rect.inflate(-3, -3)

      # Do ball and bat collide?
      # Note I put in an odd rule that sets self.hit to 1 when they collide, and unsets it in the next
      # iteration. this is to stop odd ball behaviour where it finds a collision *inside* the
      # bat, the ball reverses, and is still inside the bat, so bounces around inside.
      # This way, the ball can always escape and bounce away cleanly
      if self.rect.colliderect(player1.rect) == 1 and not self.hit:
          angle = math.pi - angle
          self.hit = not self.hit
      elif self.rect.colliderect(player2.rect) == 1 and not self.hit:
          angle = math.pi - angle
          self.hit = not self.hit
      elif self.hit:
          self.hit = not self.hit
  self.vector = (angle,z)

We start this section with an ``else`` statement, because this carries on from the previous chunk of code to check if the ball
hits the sides. It makes sense that if it doesn't hit the sides, it might hit a bat, so we carry on the conditional statement. The
first glitch to fix is to shrink the players' rectangles by 3 pixels in both dimensions, to stop the bat catching a ball that goes
behind them (if you imagine you just move the bat so that as the ball travels behind it, the rectangles overlap, and so normally the
ball would then have been "hit" - this prevents that).

Next we check if the rectangles collide, with one more glitch fix. Notice that I've commented on these odd bits of code - it's always
good to explain bits of code that are abnormal, both for others who look at your code, and so you understand it when you come back to
it. The without the fix, the ball might hit a corner of the bat, change direction, and one frame later still find itself inside the
bat. Then it would again think it has been hit, and change its direction. This can happen several times, making the ball's motion
completely unrealistic. So we have a variable, ``self.hit``, which we set to ``True`` when it has been hit, and ``False`` one frame
later. When we check if the rectangles have collided, we also check if ``self.hit`` is ``True``/``False``, to stop internal bouncing.

The important code here is pretty easy to understand. All rectangles have a :meth:`colliderect <pygame.Rect.colliderect>`
function, into which you feed the rectangle of another object, which returns ``True`` if the rectangles do overlap, and ``False`` if not.
If they do, we can change the direction by subtracting the current angle from ``pi`` (again, a handy trick you can do with radians,
which will adjust the angle by 90 degrees and send it off in the right direction; you might find at this point that a thorough
understanding of radians is in order!). Just to finish the glitch checking, we switch ``self.hit`` back to ``False`` if it's the frame
after they were hit.

We also then recompile the vector. You would of course want to remove the same line in the previous chunk of code, so that you only do
this once after the ``if-else`` conditional statement. And that's it! The combined code will now allow the ball to hit sides and bats.


.. _makegames-6-3:

6.3. The Finished product
-------------------------

The final product, with all the bits of code thrown together, as well as some other bits ofcode to glue it all together, will look
like this::

  #
  # Tom's Pong
  # A simple pong game with realistic physics and AI
  # http://www.tomchance.uklinux.net/projects/pong.shtml
  #
  # Released under the GNU General Public License

  VERSION = "0.4"

  try:
      import sys
      import random
      import math
      import os
      import getopt
      import pygame
      from socket import *
      from pygame.locals import *
  except ImportError, err:
      print "couldn't load module. %s" % (err)
      sys.exit(2)

  def load_png(name):
      """ Load image and return image object"""
      fullname = os.path.join('data', name)
      try:
          image = pygame.image.load(fullname)
          if image.get_alpha is None:
              image = image.convert()
          else:
              image = image.convert_alpha()
      except pygame.error, message:
          print 'Cannot load image:', fullname
          raise SystemExit, message
      return image, image.get_rect()

  class Ball(pygame.sprite.Sprite):
      """A ball that will move across the screen
      Returns: ball object
      Functions: update, calcnewpos
      Attributes: area, vector"""

      def __init__(self, (xy), vector):
          pygame.sprite.Sprite.__init__(self)
          self.image, self.rect = load_png('ball.png')
          screen = pygame.display.get_surface()
          self.area = screen.get_rect()
          self.vector = vector
          self.hit = 0

      def update(self):
          newpos = self.calcnewpos(self.rect,self.vector)
          self.rect = newpos
          (angle,z) = self.vector

          if not self.area.contains(newpos):
              tl = not self.area.collidepoint(newpos.topleft)
              tr = not self.area.collidepoint(newpos.topright)
              bl = not self.area.collidepoint(newpos.bottomleft)
              br = not self.area.collidepoint(newpos.bottomright)
              if tr and tl or (br and bl):
                  angle = -angle
              if tl and bl:
                  #self.offcourt()
                  angle = math.pi - angle
              if tr and br:
                  angle = math.pi - angle
                  #self.offcourt()
          else:
              # Deflate the rectangles so you can't catch a ball behind the bat
              player1.rect.inflate(-3, -3)
              player2.rect.inflate(-3, -3)

              # Do ball and bat collide?
              # Note I put in an odd rule that sets self.hit to 1 when they collide, and unsets it in the next
              # iteration. this is to stop odd ball behaviour where it finds a collision *inside* the
              # bat, the ball reverses, and is still inside the bat, so bounces around inside.
              # This way, the ball can always escape and bounce away cleanly
              if self.rect.colliderect(player1.rect) == 1 and not self.hit:
                  angle = math.pi - angle
                  self.hit = not self.hit
              elif self.rect.colliderect(player2.rect) == 1 and not self.hit:
                  angle = math.pi - angle
                  self.hit = not self.hit
              elif self.hit:
                  self.hit = not self.hit
          self.vector = (angle,z)

      def calcnewpos(self,rect,vector):
          (angle,z) = vector
          (dx,dy) = (z*math.cos(angle),z*math.sin(angle))
          return rect.move(dx,dy)

  class Bat(pygame.sprite.Sprite):
      """Movable tennis 'bat' with which one hits the ball
      Returns: bat object
      Functions: reinit, update, moveup, movedown
      Attributes: which, speed"""

      def __init__(self, side):
          pygame.sprite.Sprite.__init__(self)
          self.image, self.rect = load_png('bat.png')
          screen = pygame.display.get_surface()
          self.area = screen.get_rect()
          self.side = side
          self.speed = 10
          self.state = "still"
          self.reinit()

      def reinit(self):
          self.state = "still"
          self.movepos = [0,0]
          if self.side == "left":
              self.rect.midleft = self.area.midleft
          elif self.side == "right":
              self.rect.midright = self.area.midright

      def update(self):
          newpos = self.rect.move(self.movepos)
          if self.area.contains(newpos):
              self.rect = newpos
          pygame.event.pump()

      def moveup(self):
          self.movepos[1] = self.movepos[1] - (self.speed)
          self.state = "moveup"

      def movedown(self):
          self.movepos[1] = self.movepos[1] + (self.speed)
          self.state = "movedown"


  def main():
      # Initialise screen
      pygame.init()
      screen = pygame.display.set_mode((640, 480))
      pygame.display.set_caption('Basic Pong')

      # Fill background
      background = pygame.Surface(screen.get_size())
      background = background.convert()
      background.fill((0, 0, 0))

      # Initialise players
      global player1
      global player2
      player1 = Bat("left")
      player2 = Bat("right")

      # Initialise ball
      speed = 13
      rand = ((0.1 * (random.randint(5,8))))
      ball = Ball((0,0),(0.47,speed))

      # Initialise sprites
      playersprites = pygame.sprite.RenderPlain((player1, player2))
      ballsprite = pygame.sprite.RenderPlain(ball)

      # Blit everything to the screen
      screen.blit(background, (0, 0))
      pygame.display.flip()

      # Initialise clock
      clock = pygame.time.Clock()

      # Event loop
      while 1:
          # Make sure game doesn't run at more than 60 frames per second
          clock.tick(60)

          for event in pygame.event.get():
              if event.type == QUIT:
                  return
              elif event.type == KEYDOWN:
                  if event.key == K_a:
                      player1.moveup()
                  if event.key == K_z:
                      player1.movedown()
                  if event.key == K_UP:
                      player2.moveup()
                  if event.key == K_DOWN:
                      player2.movedown()
              elif event.type == KEYUP:
                  if event.key == K_a or event.key == K_z:
                      player1.movepos = [0,0]
                      player1.state = "still"
                  if event.key == K_UP or event.key == K_DOWN:
                      player2.movepos = [0,0]
                      player2.state = "still"

          screen.blit(background, ball.rect, ball.rect)
          screen.blit(background, player1.rect, player1.rect)
          screen.blit(background, player2.rect, player2.rect)
          ballsprite.update()
          playersprites.update()
          ballsprite.draw(screen)
          playersprites.draw(screen)
          pygame.display.flip()


  if __name__ == '__main__': main()


As well as showing you the final product, I'll point you back to TomPong, upon which all of this is based. Download it, have a look
at the source code, and you'll see a full implementation of pong using all of the code you've seen in this tutorial, as well as lots of
other code I've added in various versions, such as some extra physics for spinning, and various other bug and glitch fixes.

Oh, find TomPong at http://www.tomchance.uklinux.net/projects/pong.shtml.
