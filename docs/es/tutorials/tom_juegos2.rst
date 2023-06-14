.. include:: ../../reST/common.txt

*********************************
  Revisión: Fundamentos de Pygame
*********************************

.. role:: firstterm(emphasis)

.. _hacerjuegos-2:

2. Revisión: Fundamentos de Pygame
==================================

.. _hacerjuegos-2-1:

2.1. El juego básico de Pygame
------------------------------

Por el bien de la revisión, y para asegurarme de que estés familiarizado/a con la estrucutra de un programa básico de Pygame, voy a 
ejecutar brevemente un programa básico de Pygame, que mostrará no más que una ventana con un poco de texto en ella. Al final, debería 
verse algo así (aunque claro que la decoración de la ventana será probablemente diferente en tu sistema):

.. image:: ../../reST/tut/tom_basic.png

El código completo para este ejemplo se ve así::

  #!/usr/bin/python

  import pygame
  from pygame.locals import *

  def main():
      # Inicializar pantalla
      pygame.init()
      screen = pygame.display.set_mode((150, 50))
      pygame.display.set_caption('Basic Pygame program')

      # Llenar fondo
      background = pygame.Surface(screen.get_size())
      background = background.convert()
      background.fill((250, 250, 250))

      # Mostrar texto
      font = pygame.font.Font(None, 36)
      text = font.render("Hello There", 1, (10, 10, 10))
      textpos = text.get_rect()
      textpos.centerx = background.get_rect().centerx
      background.blit(text, textpos)

      # Blittear todo a la pantalla
      screen.blit(background, (0, 0))
      pygame.display.flip()

      # Bucle de eventos (event loop)
      while True:
          for event in pygame.event.get():
              if event.type == QUIT:
                  return

          screen.blit(background, (0, 0))
          pygame.display.flip()


  if __name__ == '__main__': main()


.. _hacerjuegos-2-2:

2.2. Objetos Pygame Básicos
---------------------------

Como pueden ver, el código consiste de tres objetos proncipales: la pantalla, el fondo y el texto. Cada uno de estos objetos está 
creado primero llamando a una instancia de objeto integrado de Pygame, y luego modificándolo para adaptarse a nuestras necesidades.
La pantalla es un caso levemente especial, porque todavía modificamos la pantalla a través de llamadas de Pygame, en lugar de llamar 
los métodos pertenecientes al objeto de pantalla. Pero para los demás objetos de Pygame, primero creamos el objeto como una 
copia de un objeto de Pygame, dándole algunos atributos, y construimos nuestro objeto a partir de ellos.

Con el fondo, primero creamos un objeto Surface de Pygame y le damos el tamaño de la pantalla. Luego realizamos la operación 
convert() para convertir la Surface a un formato de un solo píxel. Esto es obviamente necesario cuando tenemos varias imágenes y 
superficies, todas con diferentes formatos de píxeles, lo cual hace que su renderización sea bastante lenta. Al convertir todas las 
superficies (surfaces), podemos acelerar drásticamente los tiempos de renderizado. Finalmente, llenamos la superficie de fondo con 
color blanco (255, 255, 255). Estos valores son :firstterm:`RGB` (Red Green Blue), y se pueden obtener desde cualquier buen programa 
de dibujo.

Con el texto, requerimos más de un objeto. Primero, creamos un objeto de fuente (font object), que define qué fuente usar y qué 
tamaño va a tener. Luego, creamos un objeto texto (text object) usando el método ``render`` que pertenece a nuestro objeto de fuente, 
suministrando tres argumentos: el texto que se va a renderizar, si debe tener anti-aliasing (1=yes, 0=no), y el color para el texto 
(otra vez en formato RGB). A continuación, creamos un tercer objeto de texto, que obtiene un rectangulo para el texto. La forma más 
fácil de entender esto es imaginando dibujar un rectángulo que rodeará todo el texto; luego se puede usar este rectángulo para 
obtener/establecer la posición del texto en la pantalla. En este ejemplo, obtenemos el rectángulo y establecemos su atributo 
``centerx`` para que sea el atributo ``centerx`` del fondo (así el centro del texto será el mismo que el centro del fondo, es decir 
el texto estará centrado en la pantalla en el eje x). También podríamos establecer la coordenada y, pero no es diferente, así que 
dejé el texto en la parte superior de la pantalla. Como la pantalla es pequeña de todas formas, no parecía necesario.


.. _hacerjuegos-2-3:

2.3. Blitting
-------------

Ahora que hemos creado nuestros objetos de juego, necesitamos renderizarlos. Si no lo hiciéramos, y ejecutáramos el programa, 
solo veríamos una pantalla en blanco y los objetos permanecerían invisibles. El término usado para renderizar objetos es 
:firstterm:`blitting`, que es donde se copian los píxeles pertenecientes a dicho objeto en el objeto de destino. Entonecs, para 
renderizar el objeto de fondo, lo blitteamos en la pantalla. En este ejemplo, para simplificar las cosas, blitteamos el texto 
en el fondo (para que el fondo tenga una copia del texto en él) y luego blitteamos el fondo en la pantalla.

Blitting es uno de las operaciones más lentas de cualquier juego, por lo que debes tener cuidado de no blittear demasiado en la 
pantalla en cada cuadro. Si tienes una imagen de fondo y una pelota volando por la pantalla, podrías blittear el fondo y luego la 
pelota en cada cuadro, lo que cubriría la posición anterior de la pelota y renderizaría la nueva pelota, pero esto sería bastante 
lento. Una mejor solución es blittear el fondo en el área que la pelota ocupó previamente, lo que se puede encontrar en el 
rectángulo anterior de la pelota, y luego blittear la pelota, para que solo estés blitteando dos áreas pequeñas.

.. _hacerjuegos-2-4:

2.4. Evento en Bucle
--------------------

Una vez que ya hayas configurado el juego, necesitás ponerlo en un bucle para que se ejecute cotninuamente hasta que el usuario 
señale que quiere salir. Asi que comenzás un bucle abierto ``while``, y luego por cada iteración del bucle, que será cada cuadro 
del juego, actualizas el juego. Lo primero es verificar cualquier evento de Pygame, que será el usuario presionando el teclado, 
clickeando el botón del mouse, moviendo un joystick, redimensionando la ventana, o tratando de cerrarla. En este caso, simplemente 
queremos estar atentos a que el usuario intente salir del juego cerrando la ventana, en cuyo caso el juego debería ``return``, que 
terminará el bucle ``while``.
Luego, simplemente necesitamos volver a dibujar (re-blit) el fondo, y actualizar la pantalla para que todo se dibuje. Okay, como 
nada se mueve o sucede, en este ejemplo, estrictamente hablando no necesitamos volver a dibujar el fondo en cada iteración, pero lo 
incluí porque cuando las cosas se mueven en la pantalla, necesitarás hacer todo tu dibujado (blitting) aquí.

.. _hacerjuegos-2-5:

2.5. Ta-da!
-----------

¡Y eso es todo - tu más básico juego de Pygame! Todos los juegos tomarán una forma similar a esta, pero con mucho más código para las 
funciones del juego real en sí, que tienen más que ver con la programación y menos estructurados por el funcionamiento de Pygame. Esto 
es realmente de lo que trata este tutorial y seguiremos adelante con ello.
