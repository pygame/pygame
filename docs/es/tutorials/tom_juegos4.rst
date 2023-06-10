.. include:: ../../reST/common.txt

**************************
Clases de objetos de juego
**************************

.. role:: firstterm(emphasis)

.. _hacerjuegos-4:

4. Clases de objetos de juego
=============================

Una vez que hayas cargado tus módulos y escrito tus funciones de manejo de recursos, querrás pasar a escribir algunos 
objetos de juego. La forma en que esto se realiza es bastante simple, sin embargo puede parecer complejo al principio. 
Escribirás una clase para cada tipo de objeto en el juego y después crearás una instancia de esas clases para los objetos. 
Luego podés usar los métodos de esas clases para manipular los objetos, dándoles algún tipo de movimiento y capacidades 
interactivas. Entonces tu juego, en pseudo-código, se verá así::


  #!/usr/bin/python

  # [load modules here]

  # [resource handling functions here]

  class Ball:
      # [ball functions (methods) here]
      # [e.g. a function to calculate new position]
      # [and a function to check if it hits the side]

  def main:
      # [initiate game environment here]

      # [create new object as instance of ball class]
      ball = Ball()

      while True:
          # [check for user input]

          # [call ball's update function]
          ball.update()

Por supuesto, esto es un ejemplo muy simple, y tendrías que agregar todo el código en lugar de esos pequeños comentarios entre 
corchetes. Pero deberías entender la idea básica. Creás una clase, en la cual colocás todas las funciones de la pelota, incluyendo 
``__init__``,que crearía todos los atributos de la pelota, y ``update``, que movería la pelota a su nueva posición antes de 
blittearla en la pantalla en esta posición.

Luego podés crear más clases para todos tus otros objetos de juego, y luego crear instancias de los mismos para que puedas manejarlos 
fácilmente en la función ``main`` y en el bucle principal del programa. En contraste con iniciar la pelota en la función ``main``, 
y luego tener muchas funciones sin clase para manipular un objeto de pelota establecido, y espero que puedas ver por qué usar clases 
es una ventaja: te permite poner todo el código perteneciente a cada objeto en un único lugar; hace que sea más fácil usar objetos;
hace que agregar nuevos objetos y manipularlos sea más flexible. En lugar de agregar más código para cada nuevo objeto de pelota, 
podés simplemente crear instancias de la clase ``Ball`` para cada nuevo objeto de pelota. ¡Mágia!


.. _hacerjuegos-4-1:

4.1. Una clase simple de pelota
-------------------------------

Aquí hay una clase simple con el código necesario para crear un objeto pelota que se moverá a través de la pantalla, si la 
función ``update`` esa llamada en el bucleo principal::

  class Ball(pygame.sprite.Sprite):
      """A ball that will move across the screen  (Una peleota se moverá a través de la pantalla)
      Returns: ball object
      Functions: update, calcnewpos
      Attributes: area, vector"""

      def __init__(self, vector):
          pygame.sprite.Sprite.__init__(self)
          self.image, self.rect = load_png('ball.png')
          screen = pygame.display.get_surface()
          self.area = screen.get_rect()
          self.vector = vector

      def update(self):
          newpos = self.calcnewpos(self.rect,self.vector)
          self.rect = newpos

      def calcnewpos(self,rect,vector):
          (angle,z) = vector
          (dx,dy) = (z*math.cos(angle),z*math.sin(angle))
          return rect.move(dx,dy)

Aquí tenemos la clase  ``Ball`` con una función ``__init__`` que configura la pelota, una función ``update`` que cambia el 
rectángulo de la pelota para que esté en la nueva posición, y una función ``calcnewpos`` para calcular la nueva posición de 
la pelota basada en su posición actual, y el vector por el cual se está moviendo. Explicaré la física en un momento. 
Lo único más a destacar es la cadena de documentación, que es un poco más larga esta vez, y explica los conceptos básicos 
del la clase. Estas cadenas son útiles no solo para ti mismo y otros programadores que revisen el código, sino también para 
las herramientas que analicen y documenten tu código. No harán mucha diferencia en programas pequeños, pero en los grandes 
son invaluables, así que es una buena costumbre de adquirir.

.. _hacerjuegos-4-1-1:

4.1.1. Digresión 1: Sprites
~~~~~~~~~~~~~~~~~~~~~~~~~~~

La otra razón por la cual crear una clase por cada objeto son los sprites. Cada imagen que se renderiza en tu juego será un objeto, 
por lo que en principio, la clase de cada objeto debería heredar la clase :class:`Sprite <pygame.sprite.Sprite>`. Esta es una 
característica muy útil de Python: la herencia de clases.
Ahora, la clase ``Ball`` tiene todas las funciones que vienen con la clase ``Sprite``, y cualquier instancia del objeto de la clase 
``Ball`` será registrada por Pygame como un sprite. Mientras que con el texto y el fondo, que no se mueven, está bien hacer un blit 
del objeto sobre el fondo, Pygame maneja los objetos sprites de manera diferente, lo cual verás cuando miremos el código completo del 
programa.

Básicamente, creas tanto un objeto pelota y un objeto sprite para la pelota, y luego llamás a la función update de la pelota en el 
objeto de sprite, actualizando así el sprite. Los sprites también te dan formas sofisticadas de determinar si dos objetos han 
colisionado. Normalmente, podrías simplemente comprobar en el bucle principal para ver si sus rectángulos se superponen, pero eso 
implicaría mucho código, lo cual sería una pérdida de tiempo porque la clase ``Sprite`` proporciona dos funciones (``spritecollide`` 
y ``groupcollide``) para hacer esto por vos.

.. _hacerjuegos-4-1-2:


4.1.2. Digresión 2: Física de vectores
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Aparte de la estructura de la clase ``Ball``, lo notable de este código es la física de vectores utilizada para calcular el 
movimiento de la pelota. En cualquier juego que involucre movimiento angular, no se llegará muy lejos a menos que se esté 
cómodo con la trigonometría, así que simplemente introduciré los conceptos básicos que necesitás saber para entender la 
función ``calcnewpos``.

Para empezar, notarás que la pelota tiene un atributo llamado ``vector``, que está compuesto por ``angle`` y ``z``. El 
ángulo estpa medido en radianes y dará la dirección en la que la pelota se mueve. Z es la velocidad a la que se mueve la 
pelota. Entonces, usando este vector, podemos determinar la dirección y velocidad de la pelota, y por lo tanto, cuánto se 
moverá en los ejes x e y:

.. image:: ../../reST/tut/tom_radians.png

El diagrama anterior ilustra las matemáticas básicas detrás de los vectores. En el diagrama de la izquierda, se puede ver el 
movimiento proyectado de la pelota representado por una línea azul. La longitud de esa línea (z) representa su velocidad, y el 
ángulo es la dirección en la que se moverá. El ángulo para el movimiento de la pelota siempre se tomará desde el eje x a la 
derecha, y se mide en sentido horario desde esa línea, como se muestra en el diagrama.

A partir del ángulo y la velocidad de la pelota, podemos lograr calcular cuánto se ha movido a lo largo de los ejes x e y. 
Necesitamos hacer esto porque Pygame en sí no admite vectores, y solo podemos mover la pelota moviendo su rectángulo a lo largo 
de los dos ejes. Por lo tanto, necesitamos :firstterm:`resolve` (resolver) el ángulo y la velocidad en su movimiento en el eje 
x (dx) y en el eje y (dy). Esto es un asunto sencillo de trigonometría y se puede hacer con las fórmulas que se muestran en el 
diagrama.

Si has estudiado trigonometría elemental antes, nada de esto debería ser nuevo para vos. Pero en caso que seas olvidadizo, acá 
hay algunas fórmulas útiles para recordar, que te ayudarán a visualizar los ángulos (a mi me resulta más fácil visualizar los 
ángulos en grados que en radianes.

.. image:: ../../reST/tut/tom_formulae.png

