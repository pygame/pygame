.. include:: ../../reST/common.txt

********************
  Juntando Todo
********************

.. _hacerjuegos-6:

1. Juntando todo
================

Hasta ahora has aprendido todo lo básico necesario para construir un juego simple. Estás en condiciones de entender cómo crear 
objetos de Pygame, cómo Pygame muestra objetos, cómo maneja eventos y cómo pódes usar física para introducir algo de 
movimiento en tu juego. Ahora simplemente te mostraré cómo podés tomar todas esas partes de código y juntarlas en un 
juego funcional. Lo que necesitamos primero es permitir que la pelota golpee los lados de la pantalla, y que la paleta pueda 
golpear la pelota, de lo contrario no habrá mucho juego involucrado. Hacemos esto utilizando los métodos de colisión de 
Pygame: :meth:`collision <pygame.Rect.collidepoint>`.


.. _hacerjuegos-6-1:

6.1. Dejá que la pelota golpee los lados
----------------------------------------

El principio básico para hacer que rebote en los lados es fácil de comprender. Se obtienen las coordenadas de las cuatro esquinas 
de la pelota y se comprueba si corresponden con la coordenada x o y del borde de la pantalla. Por lo tanto, si la esquina superior 
derecha e izquierda tienen una coordenada 'y' de cero, sabés que la pelota está actualmente en el borde superior de la 
pantalla. Se hace todo esto en la función ``update``, despue´s de haber calculado la nueva posición de la pelota.

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

Aquí comprobamos si el ``area`` contiene la nueva posición de la bola (lo que siempre debería ser así, por lo que no 
necesitamos una cláusula ``else``, aunque en otras circunstancias podríamos considerarla). Luego verificamos si las 
coordenadas de las cuatro esquinas están *colisionando* con los bordes del área, y creamos objetos para cada resultado.
Si lo están, los objetos tendrán un valor de 1 o ``True``. Si no, el valor será ``None`` o ``False``. Luego verificamos 
si ha chocado con la parte superior o inferior, y si lo ha hecho, cambiamos la dirección de la pelota. Afortunadamente, 
usando radianes podemos hacer esto simplemente invirtiendo su valor positivo/negativo. También comprobamos si la bola 
se ha salido de los lados, y si lo ha hecho, llamamos a la función ``offcourt``. En mi juego, esto reinicia la bola, 
agrega 1 punto al puntaje del jugador especificado al llamar la función y muestra el nuevo puntaje.

Finalmente, recompilamos el vector en función del nuevo ángulo. Y eso es todo. La pelota ahora rebotará felizmente en 
las paredes y saldrá de la cancha con gracia.


.. _hacerjuegos-6-2:

6.2. Let the ball hit bats
--------------------------

Hacer que la pelota golpee los bates es muy similar a hacer que golpee los lados de la pantalla. Todavía usamos el método de colisión. 
Todavía usamos el método de colisión, pero esta vez comprobamos si los rectángulos de la pelota y cualquiera de los bates colisionan.
En este código también he agregado código adicional para evitar errores. Descubrirás que tendrás que poner todo tipo de código adicional 
para evitar errores y problemas, así que es bueno acostumbrarse a verlo.

::

  else:
      # Desinflar los rectángulos para que no quede la pelota detrás del bate
      player1.rect.inflate(-3, -3)
      player2.rect.inflate(-3, -3)

      # ¿Colisionan la pelota y el bate?
      # Nota: He establecido una regla extraña que establece self.hit en 1 cuando colisionan, y lo desactiva en la siguiente
      # iteración. Esto es para evitar un comportamiento extraño de la pelota donde encuentra una colisión *dentro* del
      # bate, la pelota se invierte, y aún está dentro del bate, por lo que rebota dentro de él.
      # De esta manera, la pelota siempre puede escapar y rebotar limpiamente.
      if self.rect.colliderect(player1.rect) == 1 and not self.hit:
          angle = math.pi - angle
          self.hit = not self.hit
      elif self.rect.colliderect(player2.rect) == 1 and not self.hit:
          angle = math.pi - angle
          self.hit = not self.hit
      elif self.hit:
          self.hit = not self.hit
  self.vector = (angle,z)

Comenzamos esta sección con una declaración ``else``, porque esto continúa desde el fragmento de código anterior, para comprobar 
si la pelota golpea los lados. Tiene sentido que si no golpea los lados, podría golpear el bate, por lo que continuamos con la 
declaración condicional. El primer error a corregir es reducir el tamaño de los rectángulos de los jugadores en 3 píxeles en ambas 
dimensiones, para evitar que el bate atrape una pelota que pasa detrás de ellos (si imaginás que simplemente mueve el bate para que 
la pelota viaje detrás de él, los rectángulos se superponen, y normalmente la pelota habría sido "golepada" - esto lo evita)

A continuación, comprobamos si los rectángulos colisionan, con una corrección adicional de errores. Observá que he comentado sobre 
estas partes extrañas del código: siempre es bueno explicar las partes del código que son anormales, tanto para otros que miran tu 
código, como para que lo entiendas cuando regreses a él. Sin la corrección, la pelota podría golpear una esquina del bate, cambiar 
la dirección, y un cuadro después, aún encontrarse dentro del bate. Luego, volvería a pensar que ha sido golpeada, y cambiaría su 
dirección. Esto puede suceder varias veces, haciendo que el movimiento de la pelota sea completamente irreal. Por lo tanto, tenemos 
una variable, ``self.hit``, que la establecemos en ``True``  cuando ha sido golpeada y ``False`` un cuadro después. Cuando 
comprobamos si los rectángulos han colisionado, también verificamos si ``self.hit`` es ``True``/``False``, para evitar rebotes 
internos.

El código importante aquí es bastante fácil de entender. Todos los rectángulos tienen una función 
:meth:`colliderect <pygame.Rect.colliderect>`, en la que alimentas el rectángulo de otro objeto y devuelve ``True`` si los 
rectángulos se superpone, y ``False`` si no lo hacen. Si se superponen, podemos cambiar la dirección restando el ángulo actual 
de ``pi`` (de nuevo, un truco útil que podés hacer con radianes, que ajustará el ángulo en 90 grados y lo enviará en la dirección 
correcta; podrías encontrar en este punto que es necesario una comprensión detallada de radianes.) Solo para terminar la comprobación 
de errores, cambiamos ``self.hit`` de vuelta a ``False`` si es el cuadro por le cual fueron golpeados.

También volvemos a compilar el vector. Por supuesto, querrás eliminar la misma línea en el fragmento de código anterior, para que 
solo lo hagas una vez después de la declaración condicional ``if-else``. ¡Y eso es todo! El código combinado ahora permitirá que la 
pelota golpee los lados y los bates.


.. _hacerjuegos-6-3:

6.3. El producto final
----------------------

El producto final, con todos los fragmentos de código unidos, así como algunos otros fragmentos de código pegados juntos, 
se verá así::

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
      print(f"couldn't load module. {err}")
      sys.exit(2)

  def load_png(name):
      """ Load image and return image object"""
      fullname = os.path.join("data", name)
      try:
          image = pygame.image.load(fullname)
          if image.get_alpha is None:
              image = image.convert()
          else:
              image = image.convert_alpha()
      except FileNotFoundError:
          print(f"Cannot load image: {fullname}")
          raise SystemExit
      return image, image.get_rect()

  class Ball(pygame.sprite.Sprite):
      """A ball that will move across the screen
      Returns: ball object
      Functions: update, calcnewpos
      Attributes: area, vector"""

      def __init__(self, (xy), vector):
          pygame.sprite.Sprite.__init__(self)
          self.image, self.rect = load_png("ball.png")
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
              # Desinflar los rectángulos para que no quede la pelota detrás del bate
              player1.rect.inflate(-3, -3)
              player2.rect.inflate(-3, -3)

              # ¿Colisionan la pelota y el bate?
              # Nota: He establecido una regla extraña que establece self.hit en 1 cuando colisionan, y lo desactiva en la siguiente
              # iteración. Esto es para evitar un comportamiento extraño de la pelota donde encuentra una colisión *dentro* del
              # bate, la pelota se invierte, y aún está dentro del bate, por lo que rebota dentro de él.
              # De esta manera, la pelota siempre puede escapar y rebotar limpiamente.
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
          self.image, self.rect = load_png("bat.png")
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
      # Initializar pantalla
      pygame.init()
      screen = pygame.display.set_mode((640, 480))
      pygame.display.set_caption("Basic Pong")

      # Llenar fondo
      background = pygame.Surface(screen.get_size())
      background = background.convert()
      background.fill((0, 0, 0))

      # Initializar jugadores
      global player1
      global player2
      player1 = Bat("left")
      player2 = Bat("right")

      # Initializar pelota
      speed = 13
      rand = ((0.1 * (random.randint(5,8))))
      ball = Ball((0,0),(0.47,speed))

      # Initializar sprites
      playersprites = pygame.sprite.RenderPlain((player1, player2))
      ballsprite = pygame.sprite.RenderPlain(ball)

      # Blittear todo en la pantalla
      screen.blit(background, (0, 0))
      pygame.display.flip()

      # Initializar reloj
      clock = pygame.time.Clock()

      # Bucle de eventos
      while True:
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


  if __name__ == "__main__":
      main()


Además de mostrar el producto final, señalaré de vuelta a TomPong, en el cual se basa todo esto. Descargalo, échale un vistazo al 
código fuente y verás una implementación de pong utilizando todo el código que has visto en este tutorial, así como también 
un montón de otros códigos que he agregado en varias versiones, como física adicional para girar y varias otras correcciones 
de errores y fallas.

Ah, TomPong se encuentra en http://www.tomchance.uklinux.net/projects/pong.shtml.
