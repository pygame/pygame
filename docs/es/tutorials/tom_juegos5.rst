.. include:: ../../reST/common.txt

*************************************
Objetos controlables por los usuarios
*************************************

.. _hacerjuegos-5:

5. Objetos controlables por los usuarios
========================================

Hasta ahora puodés crear una ventana de Pygame y renderizar una pelota que volará por la pantalla. El siguiente paso es crear 
algunos bates que el usuario pueda controlar. Esto es potencialmente más simple que la pelota, porque no requiere física (a menos 
que el objeto controlado por el usuario se mueva de manera más compleja que hacia arriba y abajo, por ejemplo, un personaje de 
plataforma como Mario, en cuyo caso necesitarás más física) Los objetos controlados por el usuario son bastante fácil de crear, 
gracias al sistema de cola de Pygame, como ya verás.


.. _hacerjuegos-5-1:

5.1. Una clase simple de bate
-----------------------------

El principio detrás de la clase de bate es similar al de la clase de pelota. Necesitás una función ``__init__`` para inicializar 
el bate (para que puedas crear instancias de objeto para cada bat), una función ``update`` para realizar cambios por cuadro 
en el bate antes de que sea blitteado en la pantalla, y las funciones que definirán lo que esta clase realmente hará. Aquí tenés 
un ejemplo de código::

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

Como puedes ver, esta clase es muy similar en estructura a la clase de la pelota, pero hay diferencias en lo que hace cada función.
En primer lugar, hay una función "reinit", que es utilizada cuando una ronda finaliza y el bate debe volver a su lugar de inicio con 
cualquier atributo establecido de vuelta a sus valores necesarios. A continuación, la forma en que se mueve el bate es un poco más 
compleja que con la pelota, porque acá su movimiento es simple (arriba/abajo), pero depende de que el usuario le diga que se mueva, 
a diferencia de la pelota que simplemente sigue moviéndose en cada cuadro. Para entender cómo se mueve el bate, es útil mirar 
brevemente un diagrama para mostrar la secuencia de eventos::

.. image:: tom_event-flowchart.png

Lo que sucede aquí es que la persona controlando el bate presional la tecla que mueve el bate hacia arriba. Para cada interación 
de el bucle principal del juego (para cada cuadro), si la tecla sigue presionada, entonces el atributo ``state`` de ese objeto 
de bate se establecerá en "moviendose" y se llamará a la función ``moveup``, lo que hará que la posición 'y' de la pelota se 
reduzca por el valor de atributo ``speed`` (en este ejemplo, 10). En otras palabras, mientras la tecla se mantenga presionada, 
el bate se moverá hacia arriba de la pantalla en 10 píxeles por cuadro. El atributo ``state`` no se utiliza aquí, pero es útil 
saberlo si estás tratando con giros o si deseas obtener alguna salida de depuración útil.

Tan pronto como el jugador suelte la tecla, se invoca el segundo conjunto de cajas y el atributo ``state`` del objeto de bate 
se establecerá de nuevo en "still" y el atributo ``movepos`` volverá a establecerse en [0,0], lo que significa que cuando se 
llame a la función ``update``, ya no movera el bate. Así que cuando el jugador suelta la tecla, el bate se detiene. ¡Simple!


.. _hacerjuegos-5-1-1:

5.1.1. Digresión 3: Eventos de Pygame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Entonces, ¿cómo sabemos cuándo el jugador está presionando teclas y luego las suelta? ¡Con el sistema de cola de eventos de Pygame, 
tontuelo! Es un sistema realmente fácil de usar y entender, así que esto no debería llevar mucho tiempo :) Ya has visto la cola de 
eventos en acción en el programa básico de Pygame, donde se usó para verificar si el usuario estaba cerrando la aplicación. El código 
para mover el bate es tan simple como eso::

  for event in pygame.event.get():
      if event.type == QUIT:
          return
      elif event.type == KEYDOWN:
          if event.key == K_UP:
              player.moveup()
          if event.key == K_DOWN:
              player.movedown()
      elif event.type == KEYUP:
          if event.key == K_UP or event.key == K_DOWN:
              player.movepos = [0,0]
              player.state = "still"

Aquí se asume que ya has creado una instancia de un bate y has llamado al objeto ``player``. Podés ver el familiar diseño 
de la estrictura ``for``, que itera a través de cada evento encontrado en la cola de eventos de Pygame, que se recupera con 
la función :mod:`event.get() <pygame.event.get>`. A medida que el usuario presiona teclas, pulsa botones del ratón y mueve 
el joystick, esas acciones se bombean en la cola de eventos de Pygame, y se dejan allí hasta que se traten. Así que en cada 
iteración del bucle principal del juego, se pasa por estos eventos, comprobando si son los que se desean tratar, y luego 
tratándolos adecuadamente. La función :func:`event.pump() <pygame.event.pump>` que estaba en la función ``Bat.update`` se 
llama entonces en cada iteración para eliminar los eventos antiguos y mantener la cola actual. 

Primero verificamos si el usuario está saliendo del programa, y lo cerramos si es así. Luego verificamos si se está 
presionando alguna tecla, y si lo están, verificamos si son las teclas designadas para mover la paleta hacia arriba 
y hacia abajo. Si lo son, llamamos a la función de movimiento correspondiente y establecemos el estado del jugador 
adecuadamente (aunque los estados moveup (moverarriba) y movedown (moverabajo) se cambian en las funciones ``moveup()`` 
y ``movedown()``, lo que hace que el código sea más ordenado y no rompe la *encapsulación*, lo que significa que se 
asignan atributos al objeto en sí, sin referirse al nombre de la instancia de ese objeto). Aquí notamos que tenemos 
tres estados: still (quieto), moveup (moverarriba), movedown (moverabajo). De nuevo, estos son útiles si se quiere 
depurar o calular giros, efectos de rotación. También verificamos si alguna tecla ha sido "soltada" (es decir, que 
ya no está siendo presionada), y nuevamente, si son las teclas correctas, detenemos el movimiento del bate.
