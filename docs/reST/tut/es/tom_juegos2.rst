.. include:: common.txt

*********************************
  Revisión: Fundamentos de Pygame
*********************************

.. role:: firstterm(énfasis)

.. _makegames-2:

2. Revisión: Fundamentos de Pygame
================================

.. _makegames-2-1:

2.1. El juego básico de Pygame
--------------------------

Por el beneficio la revisión, y para asegurar que estas familiarizado con la estructura básica de un programa de Pygame, repasaré 
brevemente un programa básico de Pygame, que mostrará no más que una ventana con algo de texto, que al final debería parecerse a
esto (probablemente la decoración de la ventana será diferente en tu sistema):

.. image:: docs/reST/tut/tom_basic.png

El código completo para este ejemplo se ve así:

  #!/usr/bin/python

  import pygame
  from pygame.locals import *

  def main():
      # Inicializar pantalla
      pygame.init()
      screen = pygame.display.set_mode((150, 50))
      pygame.display.set_caption('Basic Pygame program')

      # Rellenar fondo
      background = pygame.Surface(screen.get_size())
      background = background.convert()
      background.fill((250, 250, 250))

      # Mostrar un poco de texto
      font = pygame.font.Font(None, 36)
      text = font.render("Hello There", 1, (10, 10, 10))
      textpos = text.get_rect()
      textpos.centerx = background.get_rect().centerx
      background.blit(text, textpos)

      # Blit todo a la pantalla
      screen.blit(background, (0, 0))
      pygame.display.flip()

      # Ciclo de eventos
      while True:
          for event in pygame.event.get():
              if event.type == QUIT:
                  return

          screen.blit(background, (0, 0))
          pygame.display.flip()


  if __name__ == '__main__': main()


.. _makegames-2-2:

2.2. Objetos básicos de Pygame
-------------------------

Como puedes ver, el código consiste de tres objetos principales: la pantalla, el fondo y el texto. Cada uno de estos objetos es 
creado llamando primero una instancia de un objeto incorporado de Pygame y luego modificándolo para que ajustarlo a nuestras 
necesidades. La pantalla es un caso especial, porque todavía modificatemos la pantalla a través de llamadas de Pygame, en vez de 
llamar métodos que pertenecen al objeto de pantalla. Pero para todos los demás objetos de Pygame, primero creamos el objeto como 
una copia de un objeto de Pygame, le damos algunos atributos y construimos nuestro objeto de ellos.

Con el fondo, primero crearenis un objeto Pygame Surface y lo haremos del tamaño de la pantalla. Luego realizamos la operación 
convert() para convertir la superficie a un formato de un solo píxel. Esto es más obviamente necesario cuando tenemos varias 
imágenes y superficies, con diferentes formatos de píxeles, lo que hace que el renderizado sea bastante lento. Al convertir todas
las superficies, podemos acelerar drásticamente los tiempos de renderizado. Finalmente, llenamos la superficie de fondo con blanco
(255, 255, 255). Estos valores son :firstterm:`RGB` (Rojo, Verde, Azul), y se pueden calcular con cualquier programa de dibujo.

Con el texto, requerimos más de un objeto. Primero, creamos un objeto de fuente/font, que define qué fuente usar y el tamaño de la 
fuente. Luego, creamos un objeto de texto, usando el método ``render`` que pertenece a nuestro objeto de fuente, proporcionando tres 
argumentos: el texto a representar, si debe o no suavizarse/anti-aliased (1=sí, 0=no ) y el color del texto (de nuevo en formato RGB). 
Luego creamos un tercer objeto de texto, el cual obtiene el rectángulo para el texto. La forma más sencilla de entender esto es si nos
imaginamos dibujando un rectángulo que rodeará todo el texto; luego este rectángulo se puede usar para obtener/establecer la posición 
del texto en la pantalla. En este ejemplo, obtenemos el rectángulo, configuramos su atributo ``centerx`` a el atributo ``centerx`` del
fondo (asi que el centro del texto será el mismo que el centro del fondo, es decir, el texto estará centrado en la pantalla en el eje x). 
También podríamos establecer la coordenada y, pero no es tan diferente, así que dejé el texto en la parte superior de la pantalla. 
Como la pantalla es pequeña de todos modos, no parecía necesario.


.. _makegames-2-3:

2.3. Blitting
-------------

Ahora que hemos creado nuestros objetos de juego, necesitamos renderizarlos. Si no lo hiciéramos y ejecutáramos el programa, solo 
veríamos una ventana vacia y los objetos permanecerían invisibles. El término utilizado para renderizar objetos es :firstterm:`blitting`, 
que es donde copias los píxeles pertenecientes a dicho objeto en el objeto de destino. Entonces, para renderizar el objeto de fondo, 
lo colocas/blit en la pantalla. En este ejemplo, para hacer las cosas mas simples, proyectamos el texto en el fondo (de modo que el 
fondo ahora tendrá una copia del texto) y luego proyectamos el fondo en la pantalla.

Blitting es una de las operaciones más lentas en cualquier juego, por lo que debe tener cuidado de no hacer blit demasiado en la 
pantalla en cada cuadro. Si tienes una imagen de fondo y una pelota volando alrededor de la pantalla, entonces podría haver el fondo blit
y luego la pelota en cada cuadro, lo que cubriría la posición anterior de la pelota y generaría la nueva pelota, pero esto sería bastante
lento. Una mejor solución es hacer blit en el fondo del área que ocupaba la bola anteriormente, que se puede encontrar en el rectángulo 
anterior de la bola, y luego hacer blit en la bola, de modo que solo estés haciendo blit en dos áreas pequeñas.


.. _makegames-2-4:

2.4. El ciclo de eventos
-------------------

Una vez que haya configurado el juego, debe ponerlo en un cyclo/loop para que se ejecute continuamente hasta que el usuario indique que quiere
salir. Así que comienzas un cyclo ``while`` abierto, y luego, para cada iteración del cyclo, que será cada fotograma del juego, actualizas
el juego. Lo primero es verificar si hay algún evento de Pygame, que será que el usuario presione el teclado, haga clic en un botón del 
mouse, mueva un joystick, cambie el tamaño de la ventana o intente cerrarla. En este caso, simplemente queremos estar atentos a que el 
usuario intente salir del juego cerrando la ventana, en cuyo caso el juego debería ``regresar``, lo que finalizará el cyclo ``while``. 
Luego, simplemente necesitamos volver a iluminar el fondo y voltear (actualizar) la pantalla para que todo se dibuje. De acuerdo, como en 
este ejemplo no se mueve ni sucede nada, estrictamente hablando, no necesitamos volver a borrar el fondo en cada iteración, pero lo incluyo 
porque cuando las cosas se estan moviendo por toda la pantalla, deberás de hacer todo lo tu bliting aquí.

.. _makegames-2-5:

2.5. Ta-da!
-----------

Y eso es todo - tu juego Pygame mas basico! Todos los juego tomaran una forma similar a este, pero con mucho mas codigo por las funciones del
juego, las cuales tendran más que ver con tu programacion, y menos que ver con el guiado en estructura por el funcionamiento de Pygame. De esto
se trata realmente este tutorial, y ahora continuaremos.
